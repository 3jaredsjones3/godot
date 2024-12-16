#pragma once

#include <cstdint>
#include <cassert>
#include <cstddef>
#include "core/math/math_defs.h"
#include "core/math/math_funcs.h"
#include "core/math/vector3.h"

// Architecture detection
#if defined(__SSE__) || defined(__x86_64__) || defined(_M_X64)
    #define VECTOR3SIMD_USE_SSE
    #include <xmmintrin.h>
    #include <smmintrin.h> // For _mm_dp_ps
#elif defined(__ARM_NEON) || defined(__aarch64__)
    #define VECTOR3SIMD_USE_NEON
    #include <arm_neon.h>
#else
    #define VECTOR3SIMD_USE_SCALAR
#endif

struct alignas(16) Vector3SIMD {
    #if defined(VECTOR3SIMD_USE_SSE)
    union {
        __m128 m_value;
        float f[4];
    };
    #elif defined(VECTOR3SIMD_USE_NEON)
    union {
        float32x4_t m_value;
        float f[4];
    };
    #else
    float f[4];
    #endif

    // Constructors
    inline Vector3SIMD() {
#if defined(VECTOR3SIMD_USE_SSE)
        m_value = _mm_setzero_ps();
#elif defined(VECTOR3SIMD_USE_NEON)
        m_value = vdupq_n_f32(0.0f);
#else
        f[0] = f[1] = f[2] = f[3] = 0.0f;
#endif
    }

    inline Vector3SIMD(float x, float y, float z) {
#if defined(VECTOR3SIMD_USE_SSE)
        m_value = _mm_set_ps(0.0f, z, y, x);
#elif defined(VECTOR3SIMD_USE_NEON)
        float temp[4] = {x, y, z, 0.0f};
        m_value = vld1q_f32(temp);
#else
        f[0] = x; f[1] = y; f[2] = z; f[3] = 0.0f;
#endif
    }

    // Conversion from Vector3
    inline explicit Vector3SIMD(const Vector3 &v) {
#if defined(VECTOR3SIMD_USE_SSE)
        m_value = _mm_set_ps(0.0f, v.z, v.y, v.x);
#elif defined(VECTOR3SIMD_USE_NEON)
        float temp[4] = {v.x, v.y, v.z, 0.0f};
        m_value = vld1q_f32(temp);
#else
        f[0] = v.x; f[1] = v.y; f[2] = v.z; f[3] = 0.0f;
#endif
    }

    // Conversion to Vector3
    inline operator Vector3() const {
        return Vector3(f[0], f[1], f[2]);
    }

    // Basic Arithmetic Operators
    inline Vector3SIMD operator+(const Vector3SIMD &other) const {
#if defined(VECTOR3SIMD_USE_SSE)
        return Vector3SIMD(_mm_add_ps(m_value, other.m_value));
#elif defined(VECTOR3SIMD_USE_NEON)
        return Vector3SIMD(vaddq_f32(m_value, other.m_value));
#else
        return Vector3SIMD(f[0] + other.f[0], f[1] + other.f[1], f[2] + other.f[2]);
#endif
    }

    inline Vector3SIMD operator-(const Vector3SIMD &other) const {
#if defined(VECTOR3SIMD_USE_SSE)
        return Vector3SIMD(_mm_sub_ps(m_value, other.m_value));
#elif defined(VECTOR3SIMD_USE_NEON)
        return Vector3SIMD(vsubq_f32(m_value, other.m_value));
#else
        return Vector3SIMD(f[0] - other.f[0], f[1] - other.f[1], f[2] - other.f[2]);
#endif
    }

    inline Vector3SIMD operator*(float scalar) const {
#if defined(VECTOR3SIMD_USE_SSE)
        return Vector3SIMD(_mm_mul_ps(m_value, _mm_set1_ps(scalar)));
#elif defined(VECTOR3SIMD_USE_NEON)
        return Vector3SIMD(vmulq_n_f32(m_value, scalar));
#else
        return Vector3SIMD(f[0] * scalar, f[1] * scalar, f[2] * scalar);
#endif
    }

    inline Vector3SIMD operator*(const Vector3SIMD &other) const {
#if defined(VECTOR3SIMD_USE_SSE)
        return Vector3SIMD(_mm_mul_ps(m_value, other.m_value));
#elif defined(VECTOR3SIMD_USE_NEON)
        return Vector3SIMD(vmulq_f32(m_value, other.m_value));
#else
        return Vector3SIMD(f[0] * other.f[0], f[1] * other.f[1], f[2] * other.f[2]);
#endif
    }

    inline Vector3SIMD operator/(float scalar) const {
#if defined(VECTOR3SIMD_USE_SSE)
        return Vector3SIMD(_mm_div_ps(m_value, _mm_set1_ps(scalar)));
#elif defined(VECTOR3SIMD_USE_NEON)
        return Vector3SIMD(vmulq_n_f32(m_value, 1.0f / scalar));
#else
        float inv = 1.0f / scalar;
        return Vector3SIMD(f[0] * inv, f[1] * inv, f[2] * inv);
#endif
    }

    // Add scalar multiplication operator
    inline Vector3SIMD& operator*=(float scalar) {
    #if defined(VECTOR3SIMD_USE_SSE)
        m_value = _mm_mul_ps(m_value, _mm_set1_ps(scalar));
    #elif defined(VECTOR3SIMD_USE_NEON)
        m_value = vmulq_n_f32(m_value, scalar);
    #else
        f[0] *= scalar;
        f[1] *= scalar;
        f[2] *= scalar;
        f[3] *= scalar;
    #endif
        return *this;
    }

    // Length Operations
    inline float length_squared() const {
#if defined(VECTOR3SIMD_USE_SSE)
        __m128 mul = _mm_mul_ps(m_value, m_value);
        __m128 sum = _mm_hadd_ps(mul, mul);
        sum = _mm_hadd_ps(sum, sum);
        return _mm_cvtss_f32(sum);
#elif defined(VECTOR3SIMD_USE_NEON)
        float32x4_t v = vmulq_f32(m_value, m_value);
        float32x2_t sum = vpadd_f32(vget_low_f32(v), vget_high_f32(v));
        return vget_lane_f32(vpadd_f32(sum, sum), 0);
#else
        return f[0] * f[0] + f[1] * f[1] + f[2] * f[2];
#endif
    }

    inline float length() const {
        return Math::sqrt(length_squared());
    }

    // Dot Product
    inline float dot(const Vector3SIMD &other) const {
#if defined(VECTOR3SIMD_USE_SSE)
        __m128 dp = _mm_dp_ps(m_value, other.m_value, 0x71);
        return _mm_cvtss_f32(dp);
#elif defined(VECTOR3SIMD_USE_NEON)
        float32x4_t mul = vmulq_f32(m_value, other.m_value);
        float32x2_t sum = vpadd_f32(vget_low_f32(mul), vget_high_f32(mul));
        return vget_lane_f32(vpadd_f32(sum, sum), 0);
#else
        return f[0] * other.f[0] + f[1] * other.f[1] + f[2] * other.f[2];
#endif
    }

    // Cross Product
    inline Vector3SIMD cross(const Vector3SIMD &other) const {
#if defined(VECTOR3SIMD_USE_SSE)
        __m128 a = _mm_shuffle_ps(m_value, m_value, _MM_SHUFFLE(3,0,2,1));
        __m128 b = _mm_shuffle_ps(other.m_value, other.m_value, _MM_SHUFFLE(3,1,0,2));
        __m128 c = _mm_shuffle_ps(m_value, m_value, _MM_SHUFFLE(3,1,0,2));
        __m128 d = _mm_shuffle_ps(other.m_value, other.m_value, _MM_SHUFFLE(3,0,2,1));
        return Vector3SIMD(_mm_sub_ps(_mm_mul_ps(a, b), _mm_mul_ps(c, d)));
#elif defined(VECTOR3SIMD_USE_NEON)
        float32x4x2_t ab = vzipq_f32(m_value, other.m_value);
        float32x4_t cross = vmulq_f32(
            vextq_f32(m_value, m_value, 1),
            vextq_f32(other.m_value, other.m_value, 2)
        );
        cross = vmlsq_f32(cross,
            vextq_f32(m_value, m_value, 2),
            vextq_f32(other.m_value, other.m_value, 1)
        );
        return Vector3SIMD(cross);
#else
        return Vector3SIMD(
            f[1] * other.f[2] - f[2] * other.f[1],
            f[2] * other.f[0] - f[0] * other.f[2],
            f[0] * other.f[1] - f[1] * other.f[0]
        );
#endif
    }

    // Normalize
    inline void normalize() {
        float len = length();
        if (len > 0) {
            *this *= 1.0f / len;
        } else {
            zero();
        }
    }

    inline Vector3SIMD normalized() const {
        Vector3SIMD v = *this;
        v.normalize();
        return v;
    }

    // Set to Zero
    inline void zero() {
#if defined(VECTOR3SIMD_USE_SSE)
        m_value = _mm_setzero_ps();
#elif defined(VECTOR3SIMD_USE_NEON)
        m_value = vdupq_n_f32(0.0f);
#else
        f[0] = f[1] = f[2] = f[3] = 0.0f;
#endif
    }

    // Accessors
    inline float x() const { return f[0]; }
    inline float y() const { return f[1]; }
    inline float z() const { return f[2]; }
    inline float &x() { return f[0]; }
    inline float &y() { return f[1]; }
    inline float &z() { return f[2]; }

#if defined(VECTOR3SIMD_USE_SSE) || defined(VECTOR3SIMD_USE_NEON)
    inline Vector3SIMD(decltype(m_value) val) : m_value(val) {}
#endif
};
