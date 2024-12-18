/**************************************************************************/
/*  vector3SIMD.h                                                         */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef VECTOR3SIMD_H
#define VECTOR3SIMD_H

#include <cstdint>
#include <cassert>
#include <cstddef>
#include <cmath>

#include "core/math/math_defs.h"
#include "core/math/math_funcs.h"

// Architecture detection
#if defined(__SSE__) || (defined(__x86_64__) && !defined(__EMSCRIPTEN__)) || defined(_M_X64)
    #define VECTOR3SIMD_USE_SSE
    #include <xmmintrin.h>  // SSE
    #include <emmintrin.h>  // SSE2
    #if defined(__SSE4_1__)
        #include <smmintrin.h> // SSE4.1 for _mm_dp_ps
    #endif
#elif defined(__ARM_NEON) || defined(__aarch64__)
    #define VECTOR3SIMD_USE_NEON
    #include <arm_neon.h>
#else
    #define VECTOR3SIMD_USE_SCALAR
#endif

struct Vector3; // Forward declaration

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

    // Default constructor
    inline Vector3SIMD() {
#if defined(VECTOR3SIMD_USE_SSE)
        m_value = _mm_setzero_ps();
#elif defined(VECTOR3SIMD_USE_NEON)
        m_value = vdupq_n_f32(0.0f);
#else
        f[0] = f[1] = f[2] = f[3] = 0.0f;
#endif
    }

    // Construct from components
    inline Vector3SIMD(float p_x, float p_y, float p_z) {
#if defined(VECTOR3SIMD_USE_SSE)
        m_value = _mm_set_ps(0.0f, p_z, p_y, p_x);
#elif defined(VECTOR3SIMD_USE_NEON)
        float temp[4] = {p_x, p_y, p_z, 0.0f};
        m_value = vld1q_f32(temp);
#else
        f[0] = p_x;
        f[1] = p_y;
        f[2] = p_z;
        f[3] = 0.0f;
#endif
    }

    // Construct from Vector3 (declared but not defined here)
    inline explicit Vector3SIMD(const Vector3 &p_v); 

    // Conversion to Vector3 (declared but not defined here)
    inline operator Vector3() const;

#if defined(VECTOR3SIMD_USE_SSE)
    inline Vector3SIMD(__m128 val) : m_value(val) {}
#elif defined(VECTOR3SIMD_USE_NEON)
    inline Vector3SIMD(float32x4_t val) : m_value(val) {}
#endif

    // Component access
    inline float x() const { return f[0]; }
    inline float y() const { return f[1]; }
    inline float z() const { return f[2]; }
    inline float w() const { return f[3]; }

    // Basic arithmetic operations
    inline Vector3SIMD operator+(const Vector3SIMD &p_v) const {
#if defined(VECTOR3SIMD_USE_SSE)
        return Vector3SIMD(_mm_add_ps(m_value, p_v.m_value));
#elif defined(VECTOR3SIMD_USE_NEON)
        return Vector3SIMD(vaddq_f32(m_value, p_v.m_value));
#else
        return Vector3SIMD(f[0] + p_v.f[0], f[1] + p_v.f[1], f[2] + p_v.f[2]);
#endif
    }

    inline Vector3SIMD operator-(const Vector3SIMD &p_v) const {
#if defined(VECTOR3SIMD_USE_SSE)
        return Vector3SIMD(_mm_sub_ps(m_value, p_v.m_value));
#elif defined(VECTOR3SIMD_USE_NEON)
        return Vector3SIMD(vsubq_f32(m_value, p_v.m_value));
#else
        return Vector3SIMD(f[0] - p_v.f[0], f[1] - p_v.f[1], f[2] - p_v.f[2]);
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

    inline Vector3SIMD operator/(float scalar) const {
        float inv = 1.0f / scalar;
#if defined(VECTOR3SIMD_USE_SSE)
        return Vector3SIMD(_mm_mul_ps(m_value, _mm_set1_ps(inv)));
#elif defined(VECTOR3SIMD_USE_NEON)
        return Vector3SIMD(vmulq_n_f32(m_value, inv));
#else
        return Vector3SIMD(f[0] * inv, f[1] * inv, f[2] * inv);
#endif
    }

    inline Vector3SIMD &operator+=(const Vector3SIMD &p_v) {
#if defined(VECTOR3SIMD_USE_SSE)
        m_value = _mm_add_ps(m_value, p_v.m_value);
#elif defined(VECTOR3SIMD_USE_NEON)
        m_value = vaddq_f32(m_value, p_v.m_value);
#else
        f[0] += p_v.f[0];
        f[1] += p_v.f[1];
        f[2] += p_v.f[2];
#endif
        return *this;
    }

    inline Vector3SIMD &operator-=(const Vector3SIMD &p_v) {
#if defined(VECTOR3SIMD_USE_SSE)
        m_value = _mm_sub_ps(m_value, p_v.m_value);
#elif defined(VECTOR3SIMD_USE_NEON)
        m_value = vsubq_f32(m_value, p_v.m_value);
#else
        f[0] -= p_v.f[0];
        f[1] -= p_v.f[1];
        f[2] -= p_v.f[2];
#endif
        return *this;
    }

    // Length operations
    inline float length_squared() const {
#if defined(VECTOR3SIMD_USE_SSE)
    #if defined(__SSE4_1__)
        return _mm_cvtss_f32(_mm_dp_ps(m_value, m_value, 0x7F));
    #else
        __m128 mul = _mm_mul_ps(m_value, m_value);
        __m128 sum = _mm_add_ps(mul, _mm_movehl_ps(mul, mul));
        sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
        return _mm_cvtss_f32(sum);
    #endif
#elif defined(VECTOR3SIMD_USE_NEON)
        float32x4_t v = vmulq_f32(m_value, m_value);
        float32x2_t sum = vpadd_f32(vget_low_f32(v), vget_high_f32(v));
        sum = vpadd_f32(sum, sum);
        return vget_lane_f32(sum, 0);
#else
        return f[0] * f[0] + f[1] * f[1] + f[2] * f[2];
#endif
    }

    // Min/Max operations
    inline Vector3SIMD min(const Vector3SIMD &p_v) const {
    #if defined(VECTOR3SIMD_USE_SSE)
        return Vector3SIMD(_mm_min_ps(m_value, p_v.m_value));
    #elif defined(VECTOR3SIMD_USE_NEON)
        return Vector3SIMD(vminq_f32(m_value, p_v.m_value));
    #else
        return Vector3SIMD(
            MIN(f[0], p_v.f[0]),
            MIN(f[1], p_v.f[1]),
            MIN(f[2], p_v.f[2])
        );
    #endif
    }

    inline Vector3SIMD minf(float p_scalar) const {
    #if defined(VECTOR3SIMD_USE_SSE)
        return Vector3SIMD(_mm_min_ps(m_value, _mm_set1_ps(p_scalar)));
    #elif defined(VECTOR3SIMD_USE_NEON)
        return Vector3SIMD(vminq_f32(m_value, vdupq_n_f32(p_scalar)));
    #else
        return Vector3SIMD(
            MIN(f[0], p_scalar),
            MIN(f[1], p_scalar),
            MIN(f[2], p_scalar)
        );
    #endif
    }

    inline Vector3SIMD max(const Vector3SIMD &p_v) const {
    #if defined(VECTOR3SIMD_USE_SSE)
        return Vector3SIMD(_mm_max_ps(m_value, p_v.m_value));
    #elif defined(VECTOR3SIMD_USE_NEON)
        return Vector3SIMD(vmaxq_f32(m_value, p_v.m_value));
    #else
        return Vector3SIMD(
            MAX(f[0], p_v.f[0]),
            MAX(f[1], p_v.f[1]),
            MAX(f[2], p_v.f[2])
        );
    #endif
    }

    inline Vector3SIMD maxf(float p_scalar) const {
    #if defined(VECTOR3SIMD_USE_SSE)
        return Vector3SIMD(_mm_max_ps(m_value, _mm_set1_ps(p_scalar)));
    #elif defined(VECTOR3SIMD_USE_NEON)
        return Vector3SIMD(vmaxq_f32(m_value, vdupq_n_f32(p_scalar)));
    #else
        return Vector3SIMD(
            MAX(f[0], p_scalar),
            MAX(f[1], p_scalar),
            MAX(f[2], p_scalar)
        );
    #endif
    }

    inline float length() const {
        return Math::sqrt(length_squared());
    }

    inline Vector3SIMD normalized() const {
        float len = length();
        if (len == 0) {
            return Vector3SIMD();
        }
        return *this / len;
    }

    // Dot product
    inline float dot(const Vector3SIMD &p_v) const {
#if defined(VECTOR3SIMD_USE_SSE)
    #if defined(__SSE4_1__)
        return _mm_cvtss_f32(_mm_dp_ps(m_value, p_v.m_value, 0x7F));
    #else
        __m128 mul = _mm_mul_ps(m_value, p_v.m_value);
        __m128 sum = _mm_add_ps(mul, _mm_movehl_ps(mul, mul));
        sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
        return _mm_cvtss_f32(sum);
    #endif
#elif defined(VECTOR3SIMD_USE_NEON)
        float32x4_t mul = vmulq_f32(m_value, p_v.m_value);
        float32x2_t sum = vpadd_f32(vget_low_f32(mul), vget_high_f32(mul));
        sum = vpadd_f32(sum, sum);
        return vget_lane_f32(sum, 0);
#else
        return f[0] * p_v.f[0] + f[1] * p_v.f[1] + f[2] * p_v.f[2];
#endif
    }

    // Cross product
    inline Vector3SIMD cross(const Vector3SIMD &p_v) const {
#if defined(VECTOR3SIMD_USE_SSE)
        __m128 a = m_value;
        __m128 b = p_v.m_value;
        __m128 a_yzx = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1));
        __m128 b_yzx = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1));
        __m128 c = _mm_sub_ps(
            _mm_mul_ps(a, b_yzx),
            _mm_mul_ps(a_yzx, b)
        );
        return Vector3SIMD(_mm_shuffle_ps(c, c, _MM_SHUFFLE(3, 0, 2, 1)));
#elif defined(VECTOR3SIMD_USE_NEON)
        float32x4_t a_yzx = vextq_f32(m_value, m_value, 1);
        float32x4_t b_yzx = vextq_f32(p_v.m_value, p_v.m_value, 1);
        float32x4_t c = vsubq_f32(
            vmulq_f32(m_value, b_yzx),
            vmulq_f32(a_yzx, p_v.m_value)
        );
        return Vector3SIMD(vextq_f32(c, c, 3));
#else
        return Vector3SIMD(
            f[1] * p_v.f[2] - f[2] * p_v.f[1],
            f[2] * p_v.f[0] - f[0] * p_v.f[2],
            f[0] * p_v.f[1] - f[1] * p_v.f[0]
        );
#endif
    }

    // Error checking
    inline bool has_nan() const {
        return std::isnan(f[0]) || std::isnan(f[1]) || std::isnan(f[2]);
    }

    inline bool has_inf() const {
        return std::isinf(f[0]) || std::isinf(f[1]) || std::isinf(f[2]);
    }

    inline bool has_error() const {
        return has_nan() || has_inf();
    }
};

#endif // VECTOR3SIMD_H
