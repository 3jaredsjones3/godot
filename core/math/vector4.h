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
#if defined(VECTOR4_USE_SSE)
        __m128 dp = _mm_mul_ps(m_value, p_vec4.m_value);
        __m128 shuf = _mm_movehdup_ps(dp);             // (dp1.z, dp1.z, dp1.w, dp1.w)
        __m128 sums = _mm_add_ps(dp, shuf);            // (x+y,z+w, x+y, z+w)
        shuf = _mm_movehl_ps(shuf, sums);              // (x+y,z+w, x+y, z+w) -> upper half
        sums = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
#elif defined(VECTOR4_USE_NEON)
        float32x4_t dp = vmulq_f32(m_value, p_vec4.m_value);
        float32x2_t sum = vadd_f32(vget_high_f32(dp), vget_low_f32(dp));
        sum = vpadd_f32(sum, sum);
        return vget_lane_f32(sum, 0);
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
        real_t len = length();
        if (len == 0) {
            return Vector4();
        }
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
#if defined(VECTOR4_USE_SSE)
        return Vector4(_mm_floor_ps(m_value));
#elif defined(VECTOR4_USE_NEON)
        return Vector4(vfloorq_f32(m_value));
#else
        return Vector4(Math::floor(x), Math::floor(y), Math::floor(z), Math::floor(w));
#endif
    }

    _FORCE_INLINE_ bool is_finite() const {
        return Math::is_finite(x) && Math::is_finite(y) && Math::is_finite(z) && Math::is_finite(w);
    }

    _FORCE_INLINE_ bool is_zero_approx() const {
        return Math::is_zero_approx(x) && Math::is_zero_approx(y) &&
               Math::is_zero_approx(z) && Math::is_zero_approx(w);
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

    _FORCE_INLINE_ Axis min_axis_index() const {
        real_t min_value = x;
        int min_index = 0;
        for (int i = 1; i < AXIS_COUNT; i++) {
            if (coord[i] < min_value) {
                min_value = coord[i];
                min_index = i;
            }
        }
        return static_cast<Axis>(min_index);
    }

    _FORCE_INLINE_ Axis max_axis_index() const {
        real_t max_value = x;
        int max_index = 0;
        for (int i = 1; i < AXIS_COUNT; i++) {
            if (coord[i] > max_value) {
                max_value = coord[i];
                max_index = i;
            }
        }
        return static_cast<Axis>(max_index);
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

    // String conversion for debug and logging
    operator String() const {
        return String::num(x) + ", " + String::num(y) + ", " + String::num(z) + ", " + String::num(w);
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


    _FORCE_INLINE_ Vector4 round() const {
#if defined(VECTOR4_USE_SSE) && defined(__SSE4_1__)
        // Use SSE4.1 round intrinsics
        return Vector4(_mm_round_ps(m_value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
#elif defined(VECTOR4_USE_NEON)
        // Use NEON rounding if supported (ARMv8+). vrndnq_f32: round to nearest int (ties to even)
    #if defined(__aarch64__)
        return Vector4(vrndnq_f32(m_value));
    #else
        // Some older ARM CPUs lack a direct "round to nearest" intrinsic.
        // Fallback to scalar.
        return Vector4(
            Math::round(x),
            Math::round(y),
            Math::round(z),
            Math::round(w)
        );
    #endif
#else
        // Fallback scalar
        return Vector4(
            Math::round(x),
            Math::round(y),
            Math::round(z),
            Math::round(w)
        );
#endif
    }

    // --- Restore cubic_interpolate_in_time() ---
    _FORCE_INLINE_ Vector4 cubic_interpolate_in_time(
        const Vector4 &p_b,
        const Vector4 &p_pre_a,
        const Vector4 &p_post_b,
        real_t p_weight,
        real_t p_b_t,
        real_t p_pre_a_t,
        real_t p_post_b_t
    ) const {
#if defined(VECTOR4_USE_SSE)
        // We'll load our 4D vectors into SSE registers:
        __m128 v_this   = m_value;
        __m128 v_b      = p_b.m_value;
        __m128 v_pre_a  = p_pre_a.m_value;
        __m128 v_post_b = p_post_b.m_value;

        // Broadcast single floats
        __m128 w_val        = _mm_set1_ps(p_weight);
        __m128 pre_a_t_val  = _mm_set1_ps(p_pre_a_t);
        __m128 b_t_val      = _mm_set1_ps(p_b_t);
        __m128 post_b_t_val = _mm_set1_ps(p_post_b_t);

        // Example partial: ratio = (p_weight - p_pre_a_t) / (p_b_t - p_pre_a_t)
        __m128 ratio_vec = _mm_sub_ps(w_val, pre_a_t_val);
        ratio_vec = _mm_div_ps(ratio_vec, _mm_sub_ps(b_t_val, pre_a_t_val));

        // For demonstration, final polynomial done per-component scalar:
        float f_this[4], f_b[4], f_pre_a[4], f_post_b[4];
        _mm_storeu_ps(f_this,   v_this);
        _mm_storeu_ps(f_b,      v_b);
        _mm_storeu_ps(f_pre_a,  v_pre_a);
        _mm_storeu_ps(f_post_b, v_post_b);

        Vector4 res;
        res.x = Math::cubic_interpolate_in_time(f_this[0], f_b[0], f_pre_a[0], f_post_b[0],
                                                p_weight, p_b_t, p_pre_a_t, p_post_b_t);
        res.y = Math::cubic_interpolate_in_time(f_this[1], f_b[1], f_pre_a[1], f_post_b[1],
                                                p_weight, p_b_t, p_pre_a_t, p_post_b_t);
        res.z = Math::cubic_interpolate_in_time(f_this[2], f_b[2], f_pre_a[2], f_post_b[2],
                                                p_weight, p_b_t, p_pre_a_t, p_post_b_t);
        res.w = Math::cubic_interpolate_in_time(f_this[3], f_b[3], f_pre_a[3], f_post_b[3],
                                                p_weight, p_b_t, p_pre_a_t, p_post_b_t);

        return res;

#elif defined(VECTOR4_USE_NEON)
        // Similar approach in NEON
        float32x4_t v_this   = m_value;
        float32x4_t v_b      = p_b.m_value;
        float32x4_t v_pre_a  = p_pre_a.m_value;
        float32x4_t v_post_b = p_post_b.m_value;

        float32x4_t w_val        = vdupq_n_f32(p_weight);
        float32x4_t pre_a_t_val  = vdupq_n_f32(p_pre_a_t);
        float32x4_t b_t_val      = vdupq_n_f32(p_b_t);
        float32x4_t post_b_t_val = vdupq_n_f32(p_post_b_t);

        float32x4_t ratio_vec = vsubq_f32(w_val, pre_a_t_val);
        ratio_vec = vdivq_f32(ratio_vec, vsubq_f32(b_t_val, pre_a_t_val));

        alignas(16) float f_this[4], f_b[4], f_pre_a[4], f_post_b[4];
        vst1q_f32(f_this,   v_this);
        vst1q_f32(f_b,      v_b);
        vst1q_f32(f_pre_a,  v_pre_a);
        vst1q_f32(f_post_b, v_post_b);

        Vector4 res;
        res.x = Math::cubic_interpolate_in_time(f_this[0], f_b[0], f_pre_a[0], f_post_b[0],
                                                p_weight, p_b_t, p_pre_a_t, p_post_b_t);
        res.y = Math::cubic_interpolate_in_time(f_this[1], f_b[1], f_pre_a[1], f_post_b[1],
                                                p_weight, p_b_t, p_pre_a_t, p_post_b_t);
        res.z = Math::cubic_interpolate_in_time(f_this[2], f_b[2], f_pre_a[2], f_post_b[2],
                                                p_weight, p_b_t, p_pre_a_t, p_post_b_t);
        res.w = Math::cubic_interpolate_in_time(f_this[3], f_b[3], f_pre_a[3], f_post_b[3],
                                                p_weight, p_b_t, p_pre_a_t, p_post_b_t);

        return res;

#else
        // Pure scalar fallback
        Vector4 res = *this;
        res.x = Math::cubic_interpolate_in_time(res.x, p_b.x, p_pre_a.x, p_post_b.x,
                                                p_weight, p_b_t, p_pre_a_t, p_post_b_t);
        res.y = Math::cubic_interpolate_in_time(res.y, p_b.y, p_pre_a.y, p_post_b.y,
                                                p_weight, p_b_t, p_pre_a_t, p_post_b_t);
        res.z = Math::cubic_interpolate_in_time(res.z, p_b.z, p_pre_a.z, p_post_b.z,
                                                p_weight, p_b_t, p_pre_a_t, p_post_b_t);
        res.w = Math::cubic_interpolate_in_time(res.w, p_b.w, p_pre_a.w, p_post_b.w,
                                                p_weight, p_b_t, p_pre_a_t, p_post_b_t);
        return res;
#endif
    }
};

#endif // VECTOR4_H
