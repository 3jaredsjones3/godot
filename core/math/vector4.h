#ifndef VECTOR4_H
#define VECTOR4_H

#include <type_traits>  // used to check if we are supposed to use float or double for real_t

#include "core/error/error_macros.h"
#include "core/math/math_defs.h"
#include "core/math/math_funcs.h"
#include "core/math/vector4i.h"
#include "core/string/ustring.h"
#include "core/typedefs.h"

// SIMD headers
#if defined(__SSE__) || (defined(_M_X64) && !defined(__EMSCRIPTEN__))
#define VECTOR4_USE_SSE
#include <emmintrin.h>
#include <xmmintrin.h>
#if defined(__SSE4_1__)
#include <smmintrin.h>
#endif
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
#define VECTOR4_USE_NEON
#include <arm_neon.h>
#endif

#include <cmath>

class String;
struct Vector4i;

struct [[nodiscard]] alignas(16) Vector4 {
    static const int AXIS_COUNT = 4;

    enum Axis {
        AXIS_X,
        AXIS_Y,
        AXIS_Z,
        AXIS_W,
    };

    union {
#if defined(VECTOR4_USE_SSE)
        __m128 m_value;
#elif defined(VECTOR4_USE_NEON)
        float32x4_t m_value;
#endif
        struct {
            real_t x, y, z, w;
        };
        real_t coord[4];
    };

    // Constructors
    _FORCE_INLINE_ Vector4() {
#if defined(VECTOR4_USE_SSE)
        m_value = _mm_setzero_ps();
#elif defined(VECTOR4_USE_NEON)
        m_value = vdupq_n_f32(0.0f);
#else
        x = y = z = w = 0.0f;
#endif
    }

    _FORCE_INLINE_ Vector4(__m128 p_value) {
#if defined(VECTOR4_USE_SSE)
        m_value = p_value;
#else
        float temp[4];
        _mm_storeu_ps(temp, p_value);
        x = temp[0];
        y = temp[1];
        z = temp[2];
        w = temp[3];
#endif
    }

    _FORCE_INLINE_ real_t &operator[](int p_index) {
        ERR_FAIL_INDEX_V(p_index, 4, x); // Bounds checking for safety
        return coord[p_index];
    }

    _FORCE_INLINE_ const real_t &operator[](int p_index) const {
        ERR_FAIL_INDEX_V(p_index, 4, x); // Bounds checking for safety
        return coord[p_index];
    }

    _FORCE_INLINE_ Vector4(real_t p_x, real_t p_y, real_t p_z, real_t p_w) {
#if defined(VECTOR4_USE_SSE)
        m_value = _mm_set_ps(p_w, p_z, p_y, p_x);
#elif defined(VECTOR4_USE_NEON)
        float temp[4] = { p_x, p_y, p_z, p_w };
        m_value = vld1q_f32(temp);
#else
        x = p_x;
        y = p_y;
        z = p_z;
        w = p_w;
#endif
    }

    // Basic arithmetic
    _FORCE_INLINE_ Vector4 operator+(const Vector4 &p_vec4) const {
#if defined(VECTOR4_USE_SSE)
        return Vector4(_mm_add_ps(m_value, p_vec4.m_value));
#elif defined(VECTOR4_USE_NEON)
        return Vector4(vaddq_f32(m_value, p_vec4.m_value));
#else
        return Vector4(x + p_vec4.x, y + p_vec4.y, z + p_vec4.z, w + p_vec4.w);
#endif
    }

    _FORCE_INLINE_ Vector4 operator-(const Vector4 &p_vec4) const {
#if defined(VECTOR4_USE_SSE)
        return Vector4(_mm_sub_ps(m_value, p_vec4.m_value));
#elif defined(VECTOR4_USE_NEON)
        return Vector4(vsubq_f32(m_value, p_vec4.m_value));
#else
        return Vector4(x - p_vec4.x, y - p_vec4.y, z - p_vec4.z, w - p_vec4.w);
#endif
    }

    _FORCE_INLINE_ Vector4 operator*(const Vector4 &p_vec4) const {
#if defined(VECTOR4_USE_SSE)
        return Vector4(_mm_mul_ps(m_value, p_vec4.m_value));
#elif defined(VECTOR4_USE_NEON)
        return Vector4(vmulq_f32(m_value, p_vec4.m_value));
#else
        return Vector4(x * p_vec4.x, y * p_vec4.y, z * p_vec4.z, w * p_vec4.w);
#endif
    }

    _FORCE_INLINE_ Vector4 operator/(const Vector4 &p_vec4) const {
#if defined(VECTOR4_USE_SSE)
        return Vector4(_mm_div_ps(m_value, p_vec4.m_value));
#elif defined(VECTOR4_USE_NEON)
        float32x4_t inv = vrecpeq_f32(p_vec4.m_value);
        // One Newton-Raphson iteration can refine this if needed.
        // inv = vmulq_f32(vrecpsq_f32(p_vec4.m_value, inv), inv);
        return Vector4(vmulq_f32(m_value, inv));
#else
        return Vector4(x / p_vec4.x, y / p_vec4.y, z / p_vec4.z, w / p_vec4.w);
#endif
    }

    // Scalar operations
    _FORCE_INLINE_ Vector4 operator*(real_t p_scalar) const {
#if defined(VECTOR4_USE_SSE)
        __m128 scalar = _mm_set1_ps(p_scalar);
        return Vector4(_mm_mul_ps(m_value, scalar));
#elif defined(VECTOR4_USE_NEON)
        return Vector4(vmulq_n_f32(m_value, p_scalar));
#else
        return Vector4(x * p_scalar, y * p_scalar, z * p_scalar, w * p_scalar);
#endif
    }

    _FORCE_INLINE_ Vector4 operator/(real_t p_scalar) const {
#if defined(VECTOR4_USE_SSE)
        __m128 scalar = _mm_set1_ps(p_scalar);
        return Vector4(_mm_div_ps(m_value, scalar));
#elif defined(VECTOR4_USE_NEON)
        return Vector4(vdivq_n_f32(m_value, p_scalar));
#else
        return Vector4(x / p_scalar, y / p_scalar, z / p_scalar, w / p_scalar);
#endif
    }

    _FORCE_INLINE_ Vector4 &operator+=(const Vector4 &p_vec4) {
#if defined(VECTOR4_USE_SSE)
        m_value = _mm_add_ps(m_value, p_vec4.m_value);
#elif defined(VECTOR4_USE_NEON)
        m_value = vaddq_f32(m_value, p_vec4.m_value);
#else
        x += p_vec4.x;
        y += p_vec4.y;
        z += p_vec4.z;
        w += p_vec4.w;
#endif
        return *this;
    }

    _FORCE_INLINE_ Vector4 &operator-=(const Vector4 &p_vec4) {
#if defined(VECTOR4_USE_SSE)
        m_value = _mm_sub_ps(m_value, p_vec4.m_value);
#elif defined(VECTOR4_USE_NEON)
        m_value = vsubq_f32(m_value, p_vec4.m_value);
#else
        x -= p_vec4.x;
        y -= p_vec4.y;
        z -= p_vec4.z;
        w -= p_vec4.w;
#endif
        return *this;
    }

    _FORCE_INLINE_ Vector4 &operator*=(real_t p_scalar) {
#if defined(VECTOR4_USE_SSE)
        __m128 scalar = _mm_set1_ps(p_scalar);
        m_value = _mm_mul_ps(m_value, scalar);
#elif defined(VECTOR4_USE_NEON)
        m_value = vmulq_n_f32(m_value, p_scalar);
#else
        x *= p_scalar;
        y *= p_scalar;
        z *= p_scalar;
        w *= p_scalar;
#endif
        return *this;
    }

    _FORCE_INLINE_ Vector4 &operator/=(real_t p_scalar) {
#if defined(VECTOR4_USE_SSE)
        __m128 scalar = _mm_set1_ps(p_scalar);
        m_value = _mm_div_ps(m_value, scalar);
#elif defined(VECTOR4_USE_NEON)
        m_value = vdivq_f32(m_value, vdupq_n_f32(p_scalar));
#else
        x /= p_scalar;
        y /= p_scalar;
        z /= p_scalar;
        w /= p_scalar;
#endif
        return *this;
    }

    _FORCE_INLINE_ Vector4 operator-() const {
#if defined(VECTOR4_USE_SSE)
        __m128 neg = _mm_set1_ps(-1.0f);
        return Vector4(_mm_mul_ps(m_value, neg));
#elif defined(VECTOR4_USE_NEON)
        return Vector4(vnegq_f32(m_value));
#else
        return Vector4(-x, -y, -z, -w);
#endif
    }

    _FORCE_INLINE_ Vector4 abs() const {
#if defined(VECTOR4_USE_SSE)
        __m128 sign_mask = _mm_set1_ps(-0.0f);         // Mask to clear sign bit
        return Vector4(_mm_andnot_ps(sign_mask, m_value)); // Absolute value
#elif defined(VECTOR4_USE_NEON)
        return Vector4(vabsq_f32(m_value));
#else
        return Vector4(Math::abs(x), Math::abs(y), Math::abs(z), Math::abs(w));
#endif
    }

    // Dot product
_FORCE_INLINE_ real_t dot(const Vector4 &p_vec4) const {
#if defined(VECTOR4SIMD_USE_SSE)
    __m128 dp = _mm_mul_ps(_mm_load_ps(coord), _mm_load_ps(p_vec4.coord));
    dp = _mm_hadd_ps(dp, dp); // Horizontal add
    dp = _mm_hadd_ps(dp, dp); // Final reduction
    return _mm_cvtss_f32(dp);
#elif defined(VECTOR4SIMD_USE_NEON)
    float32x4_t mul = vmulq_f32(vld1q_f32(coord), vld1q_f32(p_vec4.coord));
    return vaddvq_f32(mul); // Sum all elements
#else
    return x * p_vec4.x + y * p_vec4.y + z * p_vec4.z + w * p_vec4.w;
#endif
}

    _FORCE_INLINE_ real_t length_squared() const {
        return dot(*this);
    }

    _FORCE_INLINE_ real_t length() const {
#if defined(VECTOR4_USE_SSE)
        return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(length_squared())));
#elif defined(VECTOR4_USE_NEON)
        return sqrtf(length_squared());
#else
        return Math::sqrt(length_squared());
#endif
    }

_FORCE_INLINE_ Vector4 normalized() const {
    real_t len_sq = length_squared();
    if (len_sq == 0 || !Math::is_finite(len_sq)) {
        return Vector4(0, 0, 0, 0); // Return a zero vector for undefined normalization
    }
    real_t len = Math::sqrt(len_sq);
    return *this / len;
}

    _FORCE_INLINE_ bool is_normalized() const {
        using T = std::conditional<std::is_same<real_t, float>::value, float, double>::type;
        return Math::is_equal_approx((T)length_squared(), (T)1.0);
    }

    _FORCE_INLINE_ void normalize() {
        *this = normalized();
    }

    // Advanced Operations
    _FORCE_INLINE_ Vector4 clamped(const Vector4 &p_min, const Vector4 &p_max) const {
#if defined(VECTOR4_USE_SSE)
        return Vector4(_mm_max_ps(_mm_min_ps(m_value, p_max.m_value), p_min.m_value));
#elif defined(VECTOR4_USE_NEON)
        return Vector4(vmaxq_f32(vminq_f32(m_value, p_max.m_value), p_min.m_value));
#else
        return Vector4(CLAMP(x, p_min.x, p_max.x),
                       CLAMP(y, p_min.y, p_max.y),
                       CLAMP(z, p_min.z, p_max.z),
                       CLAMP(w, p_min.w, p_max.w));
#endif
    }

_FORCE_INLINE_ Vector4 floor() const {
#if defined(VECTOR4_USE_SSE) && defined(__SSE4_1__)
    return Vector4(_mm_floor_ps(m_value));
#elif defined(VECTOR4_USE_NEON) && defined(__aarch64__)
    return Vector4(vrndmq_f32(m_value)); // ARMv8+ supports vrndmq_f32
#else
    // Scalar fallback for platforms without SSE4.1 or ARMv8+
    return Vector4(
        Math::floor(x),
        Math::floor(y),
        Math::floor(z),
        Math::floor(w)
    );
#endif
}

    _FORCE_INLINE_ bool is_finite() const {
        return Math::is_finite(x) && Math::is_finite(y) && Math::is_finite(z) && Math::is_finite(w);
    }

    _FORCE_INLINE_ Vector4 lerp(const Vector4 &p_to, real_t p_weight) const {
        return *this + (p_to - *this) * p_weight;
    }

    _FORCE_INLINE_ Vector4 cubic_interpolate(const Vector4 &p_b, const Vector4 &p_pre_a, const Vector4 &p_post_b, real_t p_weight) const {
        real_t t2 = p_weight * p_weight;
        real_t t3 = t2 * p_weight;

#if defined(VECTOR4_USE_SSE)
        __m128 t       = _mm_set1_ps(p_weight);
        __m128 t2_vec  = _mm_mul_ps(t, t);
        __m128 t3_vec  = _mm_mul_ps(t2_vec, t);

        __m128 coeff_this   = _mm_sub_ps(_mm_mul_ps(t3_vec, _mm_set1_ps(2.0f)),
                                         _mm_mul_ps(t2_vec, _mm_set1_ps(3.0f)));
        __m128 coeff_b      = _mm_add_ps(_mm_mul_ps(t3_vec, _mm_set1_ps(-2.0f)), t2_vec);
        __m128 coeff_pre_a  = _mm_sub_ps(t3_vec, _mm_mul_ps(t2_vec, _mm_set1_ps(2.0f)));
        __m128 coeff_post_b = _mm_sub_ps(t3_vec, t2_vec);

        return Vector4(_mm_add_ps(
            _mm_add_ps(_mm_mul_ps(coeff_this, m_value), _mm_mul_ps(coeff_b, p_b.m_value)),
            _mm_add_ps(_mm_mul_ps(coeff_pre_a, p_pre_a.m_value), _mm_mul_ps(coeff_post_b, p_post_b.m_value))
        ));
#elif defined(VECTOR4_USE_NEON)
        float32x4_t t       = vdupq_n_f32(p_weight);
        float32x4_t t2_vec  = vmulq_f32(t, t);
        float32x4_t t3_vec  = vmulq_f32(t2_vec, t);

        float32x4_t coeff_this   = vsubq_f32(vmulq_f32(t3_vec, vdupq_n_f32(2.0f)),
                                             vmulq_f32(t2_vec, vdupq_n_f32(3.0f)));
        float32x4_t coeff_b      = vaddq_f32(vmulq_f32(t3_vec, vdupq_n_f32(-2.0f)), t2_vec);
        float32x4_t coeff_pre_a  = vsubq_f32(t3_vec, vmulq_f32(t2_vec, vdupq_n_f32(2.0f)));
        float32x4_t coeff_post_b = vsubq_f32(t3_vec, t2_vec);

        return Vector4(
            vaddq_f32(
                vaddq_f32(vmulq_f32(coeff_this, m_value), vmulq_f32(coeff_b, p_b.m_value)),
                vaddq_f32(vmulq_f32(coeff_pre_a, p_pre_a.m_value), vmulq_f32(coeff_post_b, p_post_b.m_value))
            )
        );
#else
        return (*this * (2 * t3 - 3 * t2 + 1)) +
               (p_b * (-2 * t3 + 3 * t2)) +
               (p_pre_a * (t3 - 2 * t2 + p_weight)) +
               (p_post_b * (t3 - t2));
#endif
    }

    _FORCE_INLINE_ Vector4 project(const Vector4 &p_to) const {
#if defined(VECTOR4_USE_SSE)
        __m128 dp        = _mm_mul_ps(m_value, p_to.m_value);
        __m128 dp_shuffle= _mm_movehdup_ps(dp);
        __m128 dp_sums   = _mm_add_ps(dp, dp_shuffle);
        dp_shuffle       = _mm_movehl_ps(dp_shuffle, dp_sums);
        dp_sums          = _mm_add_ss(dp_sums, dp_shuffle);

        __m128 len_sq    = _mm_mul_ps(p_to.m_value, p_to.m_value);
        __m128 ls_shuffle= _mm_movehdup_ps(len_sq);
        __m128 ls_sums   = _mm_add_ps(len_sq, ls_shuffle);
        ls_shuffle       = _mm_movehl_ps(ls_shuffle, ls_sums);
        ls_sums          = _mm_add_ss(ls_sums, ls_shuffle);

        __m128 result    = _mm_div_ss(dp_sums, ls_sums);
        result           = _mm_shuffle_ps(result, result, 0x00);
        return Vector4(_mm_mul_ps(result, p_to.m_value));
#elif defined(VECTOR4_USE_NEON)
        float32x4_t dp = vmulq_f32(m_value, p_to.m_value);
        float32x2_t sum_dp = vadd_f32(vget_high_f32(dp), vget_low_f32(dp));
        sum_dp = vpadd_f32(sum_dp, sum_dp);
        float dot_val = vget_lane_f32(sum_dp, 0);

        float32x4_t ls = vmulq_f32(p_to.m_value, p_to.m_value);
        float32x2_t sum_ls = vadd_f32(vget_high_f32(ls), vget_low_f32(ls));
        sum_ls = vpadd_f32(sum_ls, sum_ls);
        float len_sq = vget_lane_f32(sum_ls, 0);

        float scale = (len_sq == 0.0f ? 0.0f : (dot_val / len_sq));
        return Vector4(vmulq_n_f32(p_to.m_value, scale));
#else
        real_t len_sq = p_to.length_squared();
        return len_sq == 0 ? Vector4() : p_to * (dot(p_to) / len_sq);
#endif
    }

    _FORCE_INLINE_ Vector4 reflect(const Vector4 &p_normal) const {
#if defined(VECTOR4_USE_SSE)
        // dot(*this, p_normal)
        __m128 dp        = _mm_mul_ps(m_value, p_normal.m_value);
        __m128 dp_shuffle= _mm_movehdup_ps(dp);
        __m128 dp_sums   = _mm_add_ps(dp, dp_shuffle);
        dp_shuffle       = _mm_movehl_ps(dp_shuffle, dp_sums);
        dp_sums          = _mm_add_ss(dp_sums, dp_shuffle);

        // scale = 2 * dot
        __m128 scale     = _mm_add_ss(dp_sums, dp_sums);
        scale            = _mm_shuffle_ps(scale, scale, 0x00);
        // result = this - scale * normal
        return Vector4(_mm_sub_ps(m_value, _mm_mul_ps(scale, p_normal.m_value)));
#elif defined(VECTOR4_USE_NEON)
        float32x4_t dp = vmulq_f32(m_value, p_normal.m_value);
        float32x2_t sum_dp = vadd_f32(vget_high_f32(dp), vget_low_f32(dp));
        sum_dp = vpadd_f32(sum_dp, sum_dp);
        float dot_val = vget_lane_f32(sum_dp, 0);

        float32x4_t scale = vmulq_n_f32(p_normal.m_value, 2.0f * dot_val);
        return Vector4(vsubq_f32(m_value, scale));
#else
        return *this - p_normal * (2 * dot(p_normal));
#endif
    }

    _FORCE_INLINE_ Vector4 cross(const Vector4 &p_b, const Vector4 &p_c) const {
#if defined(VECTOR4_USE_SSE)
        // This cross variant is more generalized; real usage depends on your math approach
        __m128 temp1 = _mm_mul_ps(
            _mm_shuffle_ps(m_value, m_value, _MM_SHUFFLE(1, 2, 3, 0)),
            _mm_shuffle_ps(p_b.m_value, p_b.m_value, _MM_SHUFFLE(2, 3, 0, 1))
        );
        __m128 temp2 = _mm_mul_ps(
            _mm_shuffle_ps(p_b.m_value, p_b.m_value, _MM_SHUFFLE(1, 2, 3, 0)),
            _mm_shuffle_ps(p_c.m_value, p_c.m_value, _MM_SHUFFLE(2, 3, 0, 1))
        );
        return Vector4(_mm_sub_ps(temp1, temp2));
#elif defined(VECTOR4_USE_NEON)
        float32x4_t a_yzxw = vextq_f32(m_value, m_value, 1);   // (y,z,x,w)
        float32x4_t b_zxyw = vextq_f32(p_b.m_value, p_b.m_value, 2); // (z,x,y,w)
        float32x4_t temp1  = vmulq_f32(a_yzxw, b_zxyw);

        float32x4_t b_yzxw = vextq_f32(p_b.m_value, p_b.m_value, 1);   // (y,z,x,w)
        float32x4_t c_zxyw = vextq_f32(p_c.m_value, p_c.m_value, 2); // (z,x,y,w)
        float32x4_t temp2  = vmulq_f32(b_yzxw, c_zxyw);

        return Vector4(vsubq_f32(temp1, temp2));
#else
        return Vector4(
            y * (p_b.z * p_c.w - p_b.w * p_c.z) -
                z * (p_b.y * p_c.w - p_b.w * p_c.y) +
                w * (p_b.y * p_c.z - p_b.z * p_c.y),
            -(x * (p_b.z * p_c.w - p_b.w * p_c.z) -
                z * (p_b.x * p_c.w - p_b.w * p_c.x) +
                w * (p_b.x * p_c.z - p_b.z * p_c.x)),
            x * (p_b.y * p_c.w - p_b.w * p_c.y) -
                y * (p_b.x * p_c.w - p_b.w * p_c.x) +
                w * (p_b.x * p_c.y - p_b.y * p_c.x),
            -(x * (p_b.y * p_c.z - p_b.z * p_c.y) -
                y * (p_b.x * p_c.z - p_b.z * p_c.x) +
                z * (p_b.x * p_c.y - p_b.y * p_c.x))
        );
#endif
    }

    _FORCE_INLINE_ Vector4 snap(const Vector4 &p_step) const {
#if defined(VECTOR4_USE_SSE)
        __m128 steps   = p_step.m_value;
        __m128 divVal  = _mm_div_ps(m_value, steps);
        __m128 floorVal= _mm_floor_ps(divVal);
        __m128 snapped = _mm_mul_ps(floorVal, steps);
        return Vector4(snapped);
#elif defined(VECTOR4_USE_NEON)
        float32x4_t divVal   = vdivq_f32(m_value, p_step.m_value);
        float32x4_t floorVal = vfloorq_f32(divVal);
        return Vector4(vmulq_f32(floorVal, p_step.m_value));
#else
        return Vector4(Math::snapped(x, p_step.x),
                       Math::snapped(y, p_step.y),
                       Math::snapped(z, p_step.z),
                       Math::snapped(w, p_step.w));
#endif
    }

_FORCE_INLINE_ Vector4 snapped(const Vector4 &p_step) const {
#if defined(VECTOR4_USE_SSE)
    __m128 step = p_step.m_value;
    __m128 inv_step = _mm_div_ps(_mm_set1_ps(1.0f), step);
    __m128 vec = m_value;

    vec = _mm_mul_ps(vec, inv_step);
    vec = _mm_round_ps(vec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    vec = _mm_mul_ps(vec, step);

    alignas(16) float result_array[4];
    _mm_store_ps(result_array, vec);
    return Vector4(result_array[0], result_array[1], result_array[2], result_array[3]);
#elif defined(VECTOR4_USE_NEON)
    float32x4_t step = vld1q_f32(p_step.coord);
    float32x4_t inv_step = vrecpeq_f32(step);
    float32x4_t vec = m_value;

    vec = vmulq_f32(vec, inv_step);
    vec = vrndnq_f32(vec);
    vec = vmulq_f32(vec, step);

    alignas(16) float result_array[4];
    vst1q_f32(result_array, vec);
    return Vector4(result_array[0], result_array[1], result_array[2], result_array[3]);
#else
    return Vector4(
        Math::snapped(x, p_step.x),
        Math::snapped(y, p_step.y),
        Math::snapped(z, p_step.z),
        Math::snapped(w, p_step.w)
    );
#endif
}

    _FORCE_INLINE_ Vector4 min(const Vector4 &p_vec4) const {
#if defined(VECTOR4_USE_SSE)
        return Vector4(_mm_min_ps(m_value, p_vec4.m_value));
#elif defined(VECTOR4_USE_NEON)
        return Vector4(vminq_f32(m_value, p_vec4.m_value));
#else
        return Vector4(Math::min(x, p_vec4.x),
                       Math::min(y, p_vec4.y),
                       Math::min(z, p_vec4.z),
                       Math::min(w, p_vec4.w));
#endif
    }

    _FORCE_INLINE_ Vector4 max(const Vector4 &p_vec4) const {
#if defined(VECTOR4_USE_SSE)
        return Vector4(_mm_max_ps(m_value, p_vec4.m_value));
#elif defined(VECTOR4_USE_NEON)
        return Vector4(vmaxq_f32(m_value, p_vec4.m_value));
#else
        return Vector4(Math::max(x, p_vec4.x),
                       Math::max(y, p_vec4.y),
                       Math::max(z, p_vec4.z),
                       Math::max(w, p_vec4.w));
#endif
    }

    _FORCE_INLINE_ Vector4 inverse() const {
#if defined(VECTOR4_USE_SSE)
        return Vector4(_mm_div_ps(_mm_set1_ps(1.0f), m_value));
#elif defined(VECTOR4_USE_NEON)
        float32x4_t inv = vrecpeq_f32(m_value);
        // One Newton-Raphson iteration can refine this if needed:
        // inv = vmulq_f32(vrecpsq_f32(m_value, inv), inv);
        return Vector4(inv);
#else
        return Vector4(1.0f / x, 1.0f / y, 1.0f / z, 1.0f / w);
#endif
    }

    // Utility function for SIMD scalar loading
    static _FORCE_INLINE_ __m128 load_scalar(real_t scalar) {
#if defined(VECTOR4_USE_SSE)
        return _mm_set1_ps(scalar);
#elif defined(VECTOR4_USE_NEON)
        return vdupq_n_f32(scalar);
#else
        return scalar; // purely for conceptual completeness
#endif
    }

    // Equality operators
    _FORCE_INLINE_ bool operator==(const Vector4 &p_vec4) const {
#if defined(VECTOR4_USE_SSE)
        __m128 cmp = _mm_cmpeq_ps(m_value, p_vec4.m_value);
        return _mm_movemask_ps(cmp) == 0xF;
#elif defined(VECTOR4_USE_NEON)
        uint32x4_t cmp = vceqq_f32(m_value, p_vec4.m_value);
        // Combine all bits:
        uint64x2_t pairwise = vreinterpretq_u64_u32(cmp);
        return (vgetq_lane_u64(pairwise, 0) == 0xffffffffffffffffULL) &&
               (vgetq_lane_u64(pairwise, 1) == 0xffffffffffffffffULL);
#else
        return x == p_vec4.x && y == p_vec4.y && z == p_vec4.z && w == p_vec4.w;
#endif
    }

    _FORCE_INLINE_ bool operator!=(const Vector4 &p_vec4) const {
        return !(*this == p_vec4);
    }

_FORCE_INLINE_ bool operator<(const Vector4 &other) const {
#if defined(VECTOR4_USE_SSE)
    __m128 lhs = _mm_set_ps(w, z, y, x);
    __m128 rhs = _mm_set_ps(other.w, other.z, other.y, other.x);
    __m128 cmp = _mm_cmplt_ps(lhs, rhs);
    return _mm_movemask_ps(cmp) == 0xF; // All 4 comparisons must be true
#elif defined(VECTOR4_USE_NEON)
    float32x4_t lhs = vld1q_f32(coord);
    float32x4_t rhs = vld1q_f32(other.coord);
    uint32x4_t cmp = vcltq_f32(lhs, rhs);
    return vgetq_lane_u32(cmp, 0) && vgetq_lane_u32(cmp, 1) && 
           vgetq_lane_u32(cmp, 2) && vgetq_lane_u32(cmp, 3);
#else
    return x < other.x && y < other.y && z < other.z && w < other.w;
#endif
}

_FORCE_INLINE_ bool operator<=(const Vector4 &other) const {
#if defined(VECTOR4_USE_SSE)
    __m128 lhs = _mm_set_ps(w, z, y, x);
    __m128 rhs = _mm_set_ps(other.w, other.z, other.y, other.x);
    __m128 cmp = _mm_cmple_ps(lhs, rhs);
    return _mm_movemask_ps(cmp) == 0xF; // All 4 comparisons must be true
#elif defined(VECTOR4_USE_NEON)
    float32x4_t lhs = vld1q_f32(coord);
    float32x4_t rhs = vld1q_f32(other.coord);
    uint32x4_t cmp = vcleq_f32(lhs, rhs);
    return vgetq_lane_u32(cmp, 0) && vgetq_lane_u32(cmp, 1) && 
           vgetq_lane_u32(cmp, 2) && vgetq_lane_u32(cmp, 3);
#else
    return x <= other.x && y <= other.y && z <= other.z && w <= other.w;
#endif
}

_FORCE_INLINE_ bool operator>(const Vector4 &other) const {
#if defined(VECTOR4_USE_SSE)
    __m128 lhs = _mm_set_ps(w, z, y, x);
    __m128 rhs = _mm_set_ps(other.w, other.z, other.y, other.x);
    __m128 cmp = _mm_cmpgt_ps(lhs, rhs);
    return _mm_movemask_ps(cmp) == 0xF; // All 4 comparisons must be true
#elif defined(VECTOR4_USE_NEON)
    float32x4_t lhs = vld1q_f32(coord);
    float32x4_t rhs = vld1q_f32(other.coord);
    uint32x4_t cmp = vcgtq_f32(lhs, rhs);
    return vgetq_lane_u32(cmp, 0) && vgetq_lane_u32(cmp, 1) && 
           vgetq_lane_u32(cmp, 2) && vgetq_lane_u32(cmp, 3);
#else
    return x > other.x && y > other.y && z > other.z && w > other.w;
#endif
}

_FORCE_INLINE_ bool operator>=(const Vector4 &other) const {
#if defined(VECTOR4_USE_SSE)
    __m128 lhs = _mm_set_ps(w, z, y, x);
    __m128 rhs = _mm_set_ps(other.w, other.z, other.y, other.x);
    __m128 cmp = _mm_cmpge_ps(lhs, rhs);
    return _mm_movemask_ps(cmp) == 0xF; // All 4 comparisons must be true
#elif defined(VECTOR4_USE_NEON)
    float32x4_t lhs = vld1q_f32(coord);
    float32x4_t rhs = vld1q_f32(other.coord);
    uint32x4_t cmp = vcgeq_f32(lhs, rhs);
    return vgetq_lane_u32(cmp, 0) && vgetq_lane_u32(cmp, 1) && 
           vgetq_lane_u32(cmp, 2) && vgetq_lane_u32(cmp, 3);
#else
    return x >= other.x && y >= other.y && z >= other.z && w >= other.w;
#endif
}

_FORCE_INLINE_ Axis min_axis_index() const {
#if defined(VECTOR4_USE_SSE)
    // Use SSE to calculate the minimum index
    __m128 temp1 = _mm_shuffle_ps(m_value, m_value, _MM_SHUFFLE(1, 0, 3, 2)); // y, x, w, z
    __m128 min1 = _mm_min_ps(m_value, temp1);                                 // min(x, y), min(z, w)
    __m128 temp2 = _mm_shuffle_ps(min1, min1, _MM_SHUFFLE(2, 3, 0, 1));       // swap pairs
    __m128 min2 = _mm_min_ps(min1, temp2);                                    // min(all)

    float min_value = _mm_cvtss_f32(min2); // Extract scalar min value
    for (int i = 0; i < 4; i++) {
        if (coord[i] == min_value) {
            return static_cast<Axis>(i);
        }
    }
    return AXIS_X; // Fallback
#elif defined(VECTOR4_USE_NEON)
    float32x4_t shuffled1 = vextq_f32(m_value, m_value, 1); // y, z, w, x
    float32x4_t min1 = vminq_f32(m_value, shuffled1);       // min(x, y), min(z, w)
    float32x2_t min2 = vpmin_f32(vget_low_f32(min1), vget_high_f32(min1));
    float min_value = vget_lane_f32(vpmin_f32(min2, min2), 0);

    for (int i = 0; i < 4; i++) {
        if (coord[i] == min_value) {
            return static_cast<Axis>(i);
        }
    }
    return AXIS_X; // Fallback
#else
    // Scalar fallback
    int min_index = 0;
    real_t min_value = coord[0];
    for (int i = 1; i < AXIS_COUNT; i++) {
        if (coord[i] < min_value) {
            min_index = i;
            min_value = coord[i];
        }
    }
    return static_cast<Axis>(min_index);
#endif
}

_FORCE_INLINE_ Axis max_axis_index() const {
#if defined(VECTOR4_USE_SSE)
    // Use SSE to calculate the maximum index
    __m128 temp1 = _mm_shuffle_ps(m_value, m_value, _MM_SHUFFLE(1, 0, 3, 2)); // y, x, w, z
    __m128 max1 = _mm_max_ps(m_value, temp1);                                 // max(x, y), max(z, w)
    __m128 temp2 = _mm_shuffle_ps(max1, max1, _MM_SHUFFLE(2, 3, 0, 1));       // swap pairs
    __m128 max2 = _mm_max_ps(max1, temp2);                                    // max(all)

    float max_value = _mm_cvtss_f32(max2); // Extract scalar max value
    for (int i = 0; i < 4; i++) {
        if (coord[i] == max_value) {
            return static_cast<Axis>(i);
        }
    }
    return AXIS_X; // Fallback
#elif defined(VECTOR4_USE_NEON)
    float32x4_t shuffled1 = vextq_f32(m_value, m_value, 1); // y, z, w, x
    float32x4_t max1 = vmaxq_f32(m_value, shuffled1);       // max(x, y), max(z, w)
    float32x2_t max2 = vpmax_f32(vget_low_f32(max1), vget_high_f32(max1));
    float max_value = vget_lane_f32(vpmax_f32(max2, max2), 0);

    for (int i = 0; i < 4; i++) {
        if (coord[i] == max_value) {
            return static_cast<Axis>(i);
        }
    }
    return AXIS_X; // Fallback
#else
    // Scalar fallback
    int max_index = 0;
    real_t max_value = coord[0];
    for (int i = 1; i < AXIS_COUNT; i++) {
        if (coord[i] > max_value) {
            max_index = i;
            max_value = coord[i];
        }
    }
    return static_cast<Axis>(max_index);
#endif
}

_FORCE_INLINE_ Vector4 round() const {
#if defined(VECTOR4_USE_SSE) && defined(__SSE4_1__)
    return Vector4(_mm_round_ps(m_value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
#elif defined(VECTOR4_USE_NEON) && defined(__aarch64__)
    return Vector4(vrndnq_f32(m_value)); // ARMv8+ supports vrndnq_f32
#else
    // Scalar fallback for platforms without SSE4.1 or ARMv8+
    return Vector4(
        Math::round(x),
        Math::round(y),
        Math::round(z),
        Math::round(w)
    );
#endif
}

_FORCE_INLINE_ bool is_equal_approx(const Vector4 &p_vec4) const {
#if defined(VECTOR4_USE_SSE)
    __m128 lhs = _mm_set_ps(w, z, y, x);
    __m128 rhs = _mm_set_ps(p_vec4.w, p_vec4.z, p_vec4.y, p_vec4.x);
    __m128 diff = _mm_sub_ps(lhs, rhs);
    __m128 abs_diff = _mm_andnot_ps(_mm_set1_ps(-0.0f), diff); // Absolute value
    __m128 epsilon = _mm_set1_ps(CMP_EPSILON); // Approximation threshold
    __m128 cmp = _mm_cmple_ps(abs_diff, epsilon);
    return _mm_movemask_ps(cmp) == 0xF; // All components are approximately equal
#elif defined(VECTOR4_USE_NEON)
    float32x4_t lhs = vld1q_f32(coord);
    float32x4_t rhs = vld1q_f32(p_vec4.coord);
    float32x4_t diff = vsubq_f32(lhs, rhs);
    float32x4_t abs_diff = vabsq_f32(diff);
    float32x4_t epsilon = vdupq_n_f32(CMP_EPSILON);
    uint32x4_t cmp = vcleq_f32(abs_diff, epsilon);
    return vgetq_lane_u32(cmp, 0) && vgetq_lane_u32(cmp, 1) &&
           vgetq_lane_u32(cmp, 2) && vgetq_lane_u32(cmp, 3);
#else
    // Scalar fallback
    return Math::is_equal_approx(x, p_vec4.x) &&
           Math::is_equal_approx(y, p_vec4.y) &&
           Math::is_equal_approx(z, p_vec4.z) &&
           Math::is_equal_approx(w, p_vec4.w);
#endif
}

_FORCE_INLINE_ bool is_zero_approx() const {
#if defined(VECTOR4_USE_SSE)
    __m128 values = _mm_set_ps(w, z, y, x);
    __m128 abs_values = _mm_andnot_ps(_mm_set1_ps(-0.0f), values); // Absolute value
    __m128 epsilon = _mm_set1_ps(CMP_EPSILON); // Approximation threshold
    __m128 cmp = _mm_cmple_ps(abs_values, epsilon);
    return _mm_movemask_ps(cmp) == 0xF; // All components are approximately zero
#elif defined(VECTOR4_USE_NEON)
    float32x4_t values = vld1q_f32(coord);
    float32x4_t abs_values = vabsq_f32(values);
    float32x4_t epsilon = vdupq_n_f32(CMP_EPSILON);
    uint32x4_t cmp = vcleq_f32(abs_values, epsilon);
    return vgetq_lane_u32(cmp, 0) && vgetq_lane_u32(cmp, 1) &&
           vgetq_lane_u32(cmp, 2) && vgetq_lane_u32(cmp, 3);
#else
    // Scalar fallback
    return Math::is_zero_approx(x) &&
           Math::is_zero_approx(y) &&
           Math::is_zero_approx(z) &&
           Math::is_zero_approx(w);
#endif
}

Vector4 cubic_interpolate_in_time(
    const Vector4 &p_b,
    const Vector4 &p_pre_a,
    const Vector4 &p_post_b,
    real_t p_weight,
    real_t p_b_t,
    real_t p_pre_a_t,
    real_t p_post_b_t
) const {
#if defined(VECTOR4_USE_SSE)
    // Load 4D vectors into SSE registers
    __m128 v_this   = m_value;
    __m128 v_b      = p_b.m_value;
    __m128 v_pre_a  = p_pre_a.m_value;
    __m128 v_post_b = p_post_b.m_value;

    // Broadcast scalar values to all SIMD lanes
    __m128 w_val        = _mm_set1_ps(p_weight);
    __m128 pre_a_t_val  = _mm_set1_ps(p_pre_a_t);
    __m128 b_t_val      = _mm_set1_ps(p_b_t);
    __m128 post_b_t_val = _mm_set1_ps(p_post_b_t);

    // Calculate time deltas and ratios
    __m128 b_minus_pre_a_t = _mm_sub_ps(b_t_val, pre_a_t_val);
    __m128 post_b_minus_b_t = _mm_sub_ps(post_b_t_val, b_t_val);

    // Avoid division by zero with a small epsilon adjustment
    __m128 epsilon = _mm_set1_ps(CMP_EPSILON);
    b_minus_pre_a_t = _mm_max_ps(b_minus_pre_a_t, epsilon);
    post_b_minus_b_t = _mm_max_ps(post_b_minus_b_t, epsilon);

    // Calculate interpolation factors
    __m128 factor1 = _mm_div_ps(_mm_sub_ps(v_this, v_pre_a), b_minus_pre_a_t);
    __m128 factor2 = _mm_div_ps(_mm_sub_ps(v_post_b, v_b), post_b_minus_b_t);

    // Interpolate
    __m128 interp = _mm_add_ps(
        _mm_mul_ps(factor1, _mm_sub_ps(w_val, b_t_val)),
        _mm_mul_ps(factor2, _mm_sub_ps(post_b_t_val, w_val))
    );

    // Store the result back into a Vector4
    alignas(16) float result[4];
    _mm_store_ps(result, interp);
    return Vector4(result[0], result[1], result[2], result[3]);

#elif defined(VECTOR4_USE_NEON)
    // Load 4D vectors into NEON registers
    float32x4_t v_this   = m_value;
    float32x4_t v_b      = p_b.m_value;
    float32x4_t v_pre_a  = p_pre_a.m_value;
    float32x4_t v_post_b = p_post_b.m_value;

    // Broadcast scalar values to NEON lanes
    float32x4_t w_val        = vdupq_n_f32(p_weight);
    float32x4_t pre_a_t_val  = vdupq_n_f32(p_pre_a_t);
    float32x4_t b_t_val      = vdupq_n_f32(p_b_t);
    float32x4_t post_b_t_val = vdupq_n_f32(p_post_b_t);

    // Calculate time deltas and ratios
    float32x4_t b_minus_pre_a_t = vsubq_f32(b_t_val, pre_a_t_val);
    float32x4_t post_b_minus_b_t = vsubq_f32(post_b_t_val, b_t_val);

    // Avoid division by zero with a small epsilon adjustment
    float32x4_t epsilon = vdupq_n_f32(CMP_EPSILON);
    b_minus_pre_a_t = vmaxq_f32(b_minus_pre_a_t, epsilon);
    post_b_minus_b_t = vmaxq_f32(post_b_minus_b_t, epsilon);

    // Calculate interpolation factors
    float32x4_t factor1 = vdivq_f32(vsubq_f32(v_this, v_pre_a), b_minus_pre_a_t);
    float32x4_t factor2 = vdivq_f32(vsubq_f32(v_post_b, v_b), post_b_minus_b_t);

    // Interpolate
    float32x4_t interp = vaddq_f32(
        vmulq_f32(factor1, vsubq_f32(w_val, b_t_val)),
        vmulq_f32(factor2, vsubq_f32(post_b_t_val, w_val))
    );

    // Store the result back into a Vector4
    alignas(16) float result[4];
    vst1q_f32(result, interp);
    return Vector4(result[0], result[1], result[2], result[3]);

#else
    // Scalar fallback
    Vector4 result;
    result.x = Math::cubic_interpolate_in_time(x, p_b.x, p_pre_a.x, p_post_b.x,
                                               p_weight, p_b_t, p_pre_a_t, p_post_b_t);
    result.y = Math::cubic_interpolate_in_time(y, p_b.y, p_pre_a.y, p_post_b.y,
                                               p_weight, p_b_t, p_pre_a_t, p_post_b_t);
    result.z = Math::cubic_interpolate_in_time(z, p_b.z, p_pre_a.z, p_post_b.z,
                                               p_weight, p_b_t, p_pre_a_t, p_post_b_t);
    result.w = Math::cubic_interpolate_in_time(w, p_b.w, p_pre_a.w, p_post_b.w,
                                               p_weight, p_b_t, p_pre_a_t, p_post_b_t);
    return result;
#endif
}

void snapf(real_t p_step) {
#if defined(VECTOR4_USE_SSE)
    __m128 step = _mm_set1_ps(p_step);
    __m128 inv_step = _mm_div_ps(_mm_set1_ps(1.0f), step); // 1 / step
    __m128 vec = m_value;

    vec = _mm_mul_ps(vec, inv_step);
    vec = _mm_round_ps(vec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); // Round to nearest
    vec = _mm_mul_ps(vec, step);

    m_value = vec;

#elif defined(VECTOR4_USE_NEON)
    float32x4_t step = vdupq_n_f32(p_step);
    float32x4_t inv_step = vrecpeq_f32(step); // Approximation for 1 / step
    float32x4_t vec = vld1q_f32(coord);

    vec = vmulq_f32(vec, inv_step);
    vec = vrndnq_f32(vec); // Round to nearest
    vec = vmulq_f32(vec, step);

    vst1q_f32(coord, vec);

#else
    x = Math::snapped(x, p_step);
    y = Math::snapped(y, p_step);
    z = Math::snapped(z, p_step);
    w = Math::snapped(w, p_step);
#endif
}

Vector4 snappedf(real_t p_step) const {
#if defined(VECTOR4_USE_SSE)
    __m128 step = _mm_set1_ps(p_step);
    __m128 inv_step = _mm_div_ps(_mm_set1_ps(1.0f), step); // 1 / step
    __m128 vec = m_value;

    vec = _mm_mul_ps(vec, inv_step);
    vec = _mm_round_ps(vec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); // Round to nearest
    vec = _mm_mul_ps(vec, step);

    alignas(16) float result[4];
    _mm_store_ps(result, vec);
    return Vector4(result[0], result[1], result[2], result[3]);

#elif defined(VECTOR4_USE_NEON)
    float32x4_t step = vdupq_n_f32(p_step);
    float32x4_t inv_step = vrecpeq_f32(step); // Approximation for 1 / step
    float32x4_t vec = vld1q_f32(coord);

    vec = vmulq_f32(vec, inv_step);
    vec = vrndnq_f32(vec); // Round to nearest
    vec = vmulq_f32(vec, step);

    alignas(16) float result[4];
    vst1q_f32(result, vec);
    return Vector4(result[0], result[1], result[2], result[3]);

#else
    return Vector4(
        Math::snapped(x, p_step),
        Math::snapped(y, p_step),
        Math::snapped(z, p_step),
        Math::snapped(w, p_step)
    );
#endif
}


_FORCE_INLINE_ friend Vector4 operator*(float p_scalar, const Vector4 &p_vec) {
#if defined(VECTOR4_USE_SSE)
    __m128 scalar = _mm_set1_ps(p_scalar);
    __m128 vec = p_vec.m_value;
    __m128 result = _mm_mul_ps(scalar, vec);
    alignas(16) float result_array[4];
    _mm_store_ps(result_array, result);
    return Vector4(result_array[0], result_array[1], result_array[2], result_array[3]);
#elif defined(VECTOR4_USE_NEON)
    float32x4_t scalar = vdupq_n_f32(p_scalar);
    float32x4_t vec = vld1q_f32(p_vec.coord);
    float32x4_t result = vmulq_f32(scalar, vec);
    alignas(16) float result_array[4];
    vst1q_f32(result_array, result);
    return Vector4(result_array[0], result_array[1], result_array[2], result_array[3]);
#else
    return Vector4(
        p_scalar * p_vec.x,
        p_scalar * p_vec.y,
        p_scalar * p_vec.z,
        p_scalar * p_vec.w
    );
#endif
}

_FORCE_INLINE_ friend Vector4 operator*(double p_scalar, const Vector4 &p_vec) {
    return static_cast<float>(p_scalar) * p_vec; // Use the float implementation
}

_FORCE_INLINE_ friend Vector4 operator*(int32_t p_scalar, const Vector4 &p_vec) {
    return static_cast<float>(p_scalar) * p_vec; // Use the float implementation
}

_FORCE_INLINE_ friend Vector4 operator*(int64_t p_scalar, const Vector4 &p_vec) {
    return static_cast<float>(p_scalar) * p_vec; // Use the float implementation
}

_FORCE_INLINE_ Vector4 sign() const {
#if defined(VECTOR4SIMD_USE_SSE)
    __m128 zero = _mm_setzero_ps();
    __m128 positive = _mm_set1_ps(1.0f);
    __m128 negative = _mm_set1_ps(-1.0f);
    __m128 cmp_pos = _mm_cmpgt_ps(_mm_load_ps(coord), zero);
    __m128 cmp_neg = _mm_cmplt_ps(_mm_load_ps(coord), zero);
    __m128 sign_vec = _mm_or_ps(_mm_and_ps(cmp_pos, positive), _mm_and_ps(cmp_neg, negative));
    Vector4 result;
    _mm_store_ps(result.coord, sign_vec);
    return result;
#elif defined(VECTOR4SIMD_USE_NEON)
    float32x4_t zero = vdupq_n_f32(0.0f);
    float32x4_t positive = vdupq_n_f32(1.0f);
    float32x4_t negative = vdupq_n_f32(-1.0f);
    uint32x4_t gt_mask = vcgtq_f32(vld1q_f32(coord), zero);
    uint32x4_t lt_mask = vcltq_f32(vld1q_f32(coord), zero);
    float32x4_t sign_vec = vbslq_f32(gt_mask, positive, vbslq_f32(lt_mask, negative, zero));
    return Vector4(sign_vec);
#else
    return Vector4(SIGN(x), SIGN(y), SIGN(z), SIGN(w));
#endif
}

_FORCE_INLINE_ Vector4 ceil() const {
#if defined(VECTOR4_USE_SSE)
    __m128 result = _mm_ceil_ps(m_value);
    alignas(16) float result_array[4];
    _mm_store_ps(result_array, result);
    return Vector4(result_array[0], result_array[1], result_array[2], result_array[3]);
#elif defined(VECTOR4_USE_NEON)
    float32x4_t result = vrndpq_f32(m_value);
    alignas(16) float result_array[4];
    vst1q_f32(result_array, result);
    return Vector4(result_array[0], result_array[1], result_array[2], result_array[3]);
#else
    return Vector4(
        Math::ceil(x),
        Math::ceil(y),
        Math::ceil(z),
        Math::ceil(w)
    );
#endif
}

_FORCE_INLINE_ Vector4 posmod(real_t p_mod) const {
#if defined(VECTOR4_USE_SSE)
    __m128 mod = _mm_set1_ps(p_mod);
    __m128 result = _mm_sub_ps(
        _mm_add_ps(m_value, mod),
        _mm_mul_ps(_mm_floor_ps(_mm_div_ps(m_value, mod)), mod)
    );
    return Vector4(result);
#elif defined(VECTOR4_USE_NEON)
    float32x4_t mod = vdupq_n_f32(p_mod);
    float32x4_t div = vdivq_f32(m_value, mod);
    float32x4_t floor_div = vrndmq_f32(div); // Floor division
    float32x4_t result = vsubq_f32(vaddq_f32(m_value, mod), vmulq_f32(floor_div, mod));
    return Vector4(result);
#else
    return Vector4(
        Math::fposmod(x, p_mod),
        Math::fposmod(y, p_mod),
        Math::fposmod(z, p_mod),
        Math::fposmod(w, p_mod)
    );
#endif
}


_FORCE_INLINE_ Vector4 posmodv(const Vector4 &p_modv) const {
#if defined(VECTOR4_USE_SSE)
    __m128 mod = p_modv.m_value;
    __m128 result = _mm_sub_ps(
        _mm_add_ps(m_value, mod),
        _mm_mul_ps(_mm_floor_ps(_mm_div_ps(m_value, mod)), mod)
    );
    return Vector4(result);
#elif defined(VECTOR4_USE_NEON)
    float32x4_t mod = vld1q_f32(p_modv.coord);
    float32x4_t div = vdivq_f32(m_value, mod);
    float32x4_t floor_div = vrndmq_f32(div); // Floor division
    float32x4_t result = vsubq_f32(vaddq_f32(m_value, mod), vmulq_f32(floor_div, mod));
    return Vector4(result);
#else
    return Vector4(
        Math::fposmod(x, p_modv.x),
        Math::fposmod(y, p_modv.y),
        Math::fposmod(z, p_modv.z),
        Math::fposmod(w, p_modv.w)
    );
#endif
}


_FORCE_INLINE_ Vector4 clamp(const Vector4 &p_min, const Vector4 &p_max) const {
#if defined(VECTOR4SIMD_USE_SSE)
    __m128 min_vec = _mm_load_ps(p_min.coord);
    __m128 max_vec = _mm_load_ps(p_max.coord);
    __m128 clamped = _mm_min_ps(_mm_max_ps(_mm_load_ps(coord), min_vec), max_vec);
    Vector4 result;
    _mm_store_ps(result.coord, clamped);
    return result;
#elif defined(VECTOR4SIMD_USE_NEON)
    float32x4_t min_vec = vld1q_f32(p_min.coord);
    float32x4_t max_vec = vld1q_f32(p_max.coord);
    float32x4_t clamped = vminq_f32(vmaxq_f32(vld1q_f32(coord), min_vec), max_vec);
    return Vector4(clamped);
#else
    return Vector4(
        Math::clamp(x, p_min.x, p_max.x),
        Math::clamp(y, p_min.y, p_max.y),
        Math::clamp(z, p_min.z, p_max.z),
        Math::clamp(w, p_min.w, p_max.w)
    );
#endif
}


_FORCE_INLINE_ Vector4 clampf(real_t p_min, real_t p_max) const {
#if defined(VECTOR4_USE_SSE)
    __m128 min_val = _mm_set1_ps(p_min);
    __m128 max_val = _mm_set1_ps(p_max);
    return Vector4(_mm_min_ps(_mm_max_ps(m_value, min_val), max_val));
#elif defined(VECTOR4_USE_NEON)
    float32x4_t min_val = vdupq_n_f32(p_min);
    float32x4_t max_val = vdupq_n_f32(p_max);
    return Vector4(vminq_f32(vmaxq_f32(m_value, min_val), max_val));
#else
    // Scalar fallback for platforms without SIMD support
    return Vector4(
        Math::clamp(x, p_min, p_max),
        Math::clamp(y, p_min, p_max),
        Math::clamp(z, p_min, p_max),
        Math::clamp(w, p_min, p_max)
    );
#endif
}

_FORCE_INLINE_ Vector4 direction_to(const Vector4 &p_to) const {
    Vector4 diff = p_to - *this;
    return diff.normalized();
}

_FORCE_INLINE_ real_t distance_to(const Vector4 &p_to) const {
    return (p_to - *this).length();
}

_FORCE_INLINE_ real_t distance_squared_to(const Vector4 &p_to) const {
    return (p_to - *this).length_squared();
}

_FORCE_INLINE_ Vector4 minf(real_t p_scalar) const {
#if defined(VECTOR4_USE_SSE)
    __m128 scalar = _mm_set1_ps(p_scalar);
    __m128 result = _mm_min_ps(m_value, scalar);

    alignas(16) float result_array[4];
    _mm_store_ps(result_array, result);
    return Vector4(result_array[0], result_array[1], result_array[2], result_array[3]);
#elif defined(VECTOR4_USE_NEON)
    float32x4_t scalar = vdupq_n_f32(p_scalar);
    float32x4_t result = vminq_f32(m_value, scalar);

    alignas(16) float result_array[4];
    vst1q_f32(result_array, result);
    return Vector4(result_array[0], result_array[1], result_array[2], result_array[3]);
#else
    return Vector4(
        Math::min(x, p_scalar),
        Math::min(y, p_scalar),
        Math::min(z, p_scalar),
        Math::min(w, p_scalar)
    );
#endif
}

_FORCE_INLINE_ Vector4 maxf(real_t p_scalar) const {
#if defined(VECTOR4_USE_SSE)
    __m128 scalar = _mm_set1_ps(p_scalar);
    __m128 result = _mm_max_ps(m_value, scalar);

    alignas(16) float result_array[4];
    _mm_store_ps(result_array, result);
    return Vector4(result_array[0], result_array[1], result_array[2], result_array[3]);
#elif defined(VECTOR4_USE_NEON)
    float32x4_t scalar = vdupq_n_f32(p_scalar);
    float32x4_t result = vmaxq_f32(m_value, scalar);

    alignas(16) float result_array[4];
    vst1q_f32(result_array, result);
    return Vector4(result_array[0], result_array[1], result_array[2], result_array[3]);
#else
    return Vector4(
        Math::max(x, p_scalar),
        Math::max(y, p_scalar),
        Math::max(z, p_scalar),
        Math::max(w, p_scalar)
    );
#endif
}



    	operator String() const;
        operator Vector4i() const;
};

#endif // VECTOR4_H
