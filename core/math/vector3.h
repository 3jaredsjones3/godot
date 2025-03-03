#ifndef VECTOR3_H
#define VECTOR3_H

#include "core/error/error_macros.h"
#include "core/math/math_defs.h"
#include "core/math/math_funcs.h"
#include "core/string/ustring.h"
#include "core/typedefs.h"

#include "vector2.h"
#include "core/math/vector3i.h"

#include <type_traits>

const float M_PI = 3.14159265358979323846f;

#if (defined(__SSE__) || (defined(_M_X64) && !defined(__EMSCRIPTEN__))) && !defined(REAL_T_IS_DOUBLE)
#define VECTOR3SIMD_USE_SSE
#define VECTOR3_USE_SSE
#include <emmintrin.h>  // SSE2
#include <xmmintrin.h>  // SSE
#endif

//I'm still working through a lot of bugs for NEON but I will have to set up remote - ssh to test on my device first
#if defined(__ARM_NEON) || defined(__aarch64__) && !defined(REAL_T_IS_DOUBLE)
#define VECTOR3SIMD_USE_NEON
#define VECTOR3_USE_NEON
#include <arm_neon.h>
#endif

#if defined(VECTOR3SIMD_USE_SSE) || defined(VECTOR3_USE_SSE)
#include <immintrin.h>  // For _mm_sin_ps
#elif defined(VECTOR3SIMD_USE_NEON) || defined(VECTOR3_USE_NEON)
static inline float32x4_t sin_neon(float32x4_t x) {
    const float32x4_t c1 = vdupq_n_f32(1.0f);
    const float32x4_t c3 = vdupq_n_f32(-1.0f/6.0f);
    const float32x4_t c5 = vdupq_n_f32(1.0f/120.0f);
    const float32x4_t c7 = vdupq_n_f32(-1.0f/5040.0f);

    const float32x4_t two_pi = vdupq_n_f32(2.0f * M_PI);
    float32x4_t normalized = vsubq_f32(x, 
        vmulq_f32(vdupq_n_f32(roundf(vgetq_lane_f32(
            vmulq_f32(x, vdupq_n_f32(1.0f/(2.0f*M_PI))), 0))), two_pi));

    float32x4_t x2 = vmulq_f32(normalized, normalized);
    float32x4_t x3 = vmulq_f32(x2, normalized);
    float32x4_t x5 = vmulq_f32(x3, x2);
    float32x4_t x7 = vmulq_f32(x5, x2);

    return vaddq_f32(vaddq_f32(vmulq_f32(normalized, c1), vmulq_f32(x3, c3)),
                     vaddq_f32(vmulq_f32(x5, c5), vmulq_f32(x7, c7)));
}

static inline float32x4_t cos_neon(float32x4_t x) {
    const float32x4_t c0 = vdupq_n_f32(1.0f);
    const float32x4_t c2 = vdupq_n_f32(-1.0f/2.0f);
    const float32x4_t c4 = vdupq_n_f32(1.0f/24.0f);
    const float32x4_t c6 = vdupq_n_f32(-1.0f/720.0f);

    float32x4_t normalized = vsubq_f32(x, 
        vmulq_f32(vdupq_n_f32(roundf(vgetq_lane_f32(
            vmulq_f32(x, vdupq_n_f32(1.0f/(2.0f*M_PI))), 0))), vdupq_n_f32(2.0f * M_PI)));

    float32x4_t x2 = vmulq_f32(normalized, normalized);
    float32x4_t x4 = vmulq_f32(x2, x2);
    float32x4_t x6 = vmulq_f32(x4, x2);

    return vaddq_f32(vaddq_f32(c0, vmulq_f32(x2, c2)),
                     vaddq_f32(vmulq_f32(x4, c4), vmulq_f32(x6, c6)));
}
#endif

struct Basis;

#if defined(VECTOR3SIMD_USE_NEON) || defined(VECTOR3SIMD_USE_SSE) || defined(VECTOR3_USE_NEON) || defined(VECTOR3_USE_SSE)
struct [[nodiscard]] alignas(16) Vector3 {
#else
struct [[nodiscard]] Vector3 {
#endif
    static const int AXIS_COUNT = 3;

    enum Axis {
        AXIS_X,
        AXIS_Y,
        AXIS_Z,
    };

    // Static constants declarations
    static const Vector3 ZERO;
    static const Vector3 ONE;
    static const Vector3 LEFT;
    static const Vector3 RIGHT;
    static const Vector3 UP;
    static const Vector3 DOWN;
    static const Vector3 FORWARD;
    static const Vector3 BACK;

#if defined(VECTOR3SIMD_USE_NEON) || defined(VECTOR3SIMD_USE_SSE) || defined(VECTOR3_USE_NEON) || defined(VECTOR3_USE_SSE)
    // SIMD version
    union {
        struct {
            real_t x;
            real_t y;
            real_t z;
            real_t _pad;
        };
        real_t coord[4];
        #if defined(VECTOR3SIMD_USE_SSE) || defined(VECTOR3_USE_SSE)
        __m128 m_value;
        #elif defined(VECTOR3SIMD_USE_NEON) || defined(VECTOR3_USE_NEON)
        float32x4_t m_value;
        #endif
    };

    // SIMD constructors
    _FORCE_INLINE_ Vector3() {
        #if defined(VECTOR3SIMD_USE_SSE) || defined(VECTOR3_USE_SSE)
            m_value = _mm_setzero_ps();
        #elif defined(VECTOR3SIMD_USE_NEON) || defined(VECTOR3_USE_NEON)
            m_value = vdupq_n_f32(0.0f);
        #endif
    }

    _FORCE_INLINE_ Vector3(real_t p_x, real_t p_y, real_t p_z, real_t p_w = 0.0f) {
        DEV_ASSERT((reinterpret_cast<std::uintptr_t>(this) & 0xF) == 0);
        #if defined(VECTOR3SIMD_USE_SSE) || defined(VECTOR3_USE_SSE)
            m_value = _mm_set_ps(p_w, p_z, p_y, p_x);
        #elif defined(VECTOR3SIMD_USE_NEON) || defined(VECTOR3_USE_NEON)
            float temp[4] = {p_x, p_y, p_z, p_w};
            m_value = vld1q_f32(temp);
        #endif
    }

    #if defined(VECTOR3SIMD_USE_SSE) || defined(VECTOR3_USE_SSE)
    _FORCE_INLINE_ Vector3(__m128 p_val) {
        m_value = p_val;
    }
    #elif defined(VECTOR3SIMD_USE_NEON) || defined(VECTOR3_USE_NEON)
    _FORCE_INLINE_ Vector3(float32x4_t p_val) {
        m_value = p_val;
    }
    #endif

#else
    // Non-SIMD version
    union {
        struct {
            real_t x;
            real_t y;
            real_t z;
        };
        real_t coord[3];
    };

    // Standard constructors
    _FORCE_INLINE_ Vector3() : x(0), y(0), z(0) {}
    _FORCE_INLINE_ Vector3(real_t p_x, real_t p_y, real_t p_z) : x(p_x), y(p_y), z(p_z) {}
#endif

    // Common copy constructor
    _FORCE_INLINE_ Vector3(const Vector3& p_other) {
        #if defined(VECTOR3SIMD_USE_SSE)
            m_value = p_other.m_value;
        #elif defined(VECTOR3SIMD_USE_NEON)
            m_value = p_other.m_value;
        #else
            x = p_other.x;
            y = p_other.y;
            z = p_other.z;
        #endif
    }

    // Common assignment operator
    _FORCE_INLINE_ Vector3& operator=(const Vector3& p_other) {
        if (this == &p_other) return *this;
        #if defined(VECTOR3SIMD_USE_SSE)
            m_value = p_other.m_value;
        #elif defined(VECTOR3SIMD_USE_NEON)
            m_value = p_other.m_value;
        #else
            x = p_other.x;
            y = p_other.y;
            z = p_other.z;
        #endif
        return *this;
    }

    // Common methods
    _FORCE_INLINE_ real_t& operator[](int p_axis) {
        DEV_ASSERT((unsigned int)p_axis < 3);
        return coord[p_axis];
    }

    _FORCE_INLINE_ const real_t& operator[](int p_axis) const {
        DEV_ASSERT((unsigned int)p_axis < 3);
        return coord[p_axis];
    }

    _FORCE_INLINE_ static Vector3 get_zero_vector() {
        return Vector3();
    }

    _FORCE_INLINE_ Axis min_axis_index() const {
        return (x < y) ? (x < z ? AXIS_X : AXIS_Z) : (y < z ? AXIS_Y : AXIS_Z);
    }

    _FORCE_INLINE_ Axis max_axis_index() const {
        return (x < y) ? (y < z ? AXIS_Z : AXIS_Y) : (x < z ? AXIS_Z : AXIS_X);
    }

    void zero() {
        #if defined(VECTOR3SIMD_USE_SSE)
            m_value = _mm_setzero_ps();
        #elif defined(VECTOR3SIMD_USE_NEON)
            m_value = vdupq_n_f32(0.0f);
        #else
            x = 0;
            y = 0;
            z = 0;
        #endif
    }

//methods

	/**************************************************************************/
	/* Example: Cross & Dot as static methods                                 */
	/**************************************************************************/
	_FORCE_INLINE_ static Vector3 vec3_cross(const Vector3 &a, const Vector3 &b) {
		return a.cross(b);
	}

	_FORCE_INLINE_ static real_t vec3_dot(const Vector3 &a, const Vector3 &b) {
		return a.dot(b);
	}


    Basis outer(const Vector3& p_with) const;

    _FORCE_INLINE_ Vector3 cross(const Vector3& p_with) const {
#if defined(VECTOR3SIMD_USE_SSE) || defined(VECTOR3_USE_SSE)
          __m128 a = m_value;
          __m128 b = p_with.m_value;
          __m128 a_yzx = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1));
          __m128 b_yzx = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1));
          __m128 c = _mm_sub_ps(_mm_mul_ps(a, b_yzx), _mm_mul_ps(a_yzx, b));
          return Vector3(_mm_shuffle_ps(c, c, _MM_SHUFFLE(3, 0, 2, 1)));
#elif defined(VECTOR3SIMD_USE_NEON) || defined(VECTOR3_USE_NEON)
        float32x4_t a_yzx = vextq_f32(m_value, m_value, 1);
        float32x4_t b_yzx = vextq_f32(p_with.m_value, p_with.m_value, 1); // Fixed from p_with
        float32x4_t c = vsubq_f32(vmulq_f32(m_value, b_yzx),
                                vmulq_f32(a_yzx, p_with.m_value));
        return Vector3(vextq_f32(c, c, 3));
#else
     return Vector3((y * p_with.z) - (z * p_with.y),
                    (z * p_with.x) - (x * p_with.z),
                    (x * p_with.y) - (y * p_with.x));
#endif
    }

_FORCE_INLINE_ real_t dot(const Vector3& p_with) const {
#if defined(VECTOR3SIMD_USE_SSE)
    #if defined(__SSE4_1__)
        return _mm_cvtss_f32(_mm_dp_ps(m_value, p_with.m_value, 0x7F));
    #else
        __m128 mul = _mm_mul_ps(m_value, p_with.m_value);
        __m128 shuf = _mm_movehdup_ps(mul);
        __m128 sums = _mm_add_ps(mul, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
    #endif
#elif defined(VECTOR3SIMD_USE_NEON)
    float32x4_t mul = vmulq_f32(m_value, p_with.m_value);
    return vaddvq_f32(mul);
#else
    return x * p_with.x + y * p_with.y + z * p_with.z;
#endif
}

    _FORCE_INLINE_ real_t length() const {
#if defined(VECTOR3SIMD_USE_SSE)
    return _mm_cvtss_f32(_mm_sqrt_ss(_mm_dp_ps(m_value, m_value, 0x7F)));
#else
    return sqrtf(length_squared()); 
#endif
    }

    _FORCE_INLINE_ real_t length_squared() const {
        return dot(*this); 
    }

_FORCE_INLINE_ void normalize() {
    real_t l2 = length_squared();
    if (Math::is_zero_approx(l2)) {
        *this = Vector3();
        return;
    }
    if (!Math::is_finite(l2) || l2 < 0.0f) {
        ERR_PRINT("Vector3: Invalid length_squared value for normalization");
        *this = Vector3();
        return;
    }
    real_t l = Math::sqrt(l2);
#if defined(VECTOR3SIMD_USE_SSE) || defined(VECTOR3_USE_SSE)
    m_value = _mm_div_ps(m_value, _mm_set1_ps(l));
#elif defined(VECTOR3SIMD_USE_NEON) || defined(VECTOR3_USE_NEON)
    m_value = vmulq_n_f32(m_value, 1.0f / l);
#else
    *this /= l;
#endif
}

    _FORCE_INLINE_ Vector3 normalized() const {
        Vector3 norm = *this;
        norm.normalize();
        return norm;
    }

    _FORCE_INLINE_ bool is_normalized() const {
        return Math::is_equal_approx(length_squared(), 1.0f);
    }

    _FORCE_INLINE_ Vector3 inverse() const {
#if defined(VECTOR3SIMD_USE_SSE)
          __m128 one = _mm_set1_ps(1.0f);
          return Vector3(_mm_div_ps(one, m_value));
#elif defined(VECTOR3SIMD_USE_NEON)
          float32x4_t one = vdupq_n_f32(1.0f);
          return Vector3(vdivq_f32(one, m_value));
#else
             Vector3 result;
     if (x != 0.0f && y != 0.0f && z != 0.0f) {
          result = Vector3(1.0f / x, 1.0f / y, 1.0f / z);
     } else {
          return Vector3(0.0f, 0.0f, 0.0f);
          ERR_PRINT("Cannot divide by 0");
     }

     return result;
#endif
    }

    _FORCE_INLINE_ Vector3 abs() const {
#if defined(VECTOR3SIMD_USE_SSE)
    return Vector3(_mm_andnot_ps(_mm_set1_ps(-0.0f), m_value));
#elif defined(VECTOR3SIMD_USE_NEON)
    return Vector3(vabsq_f32(m_value));
#else
    return Vector3(Math::abs(x), Math::abs(y), Math::abs(z));
#endif
    }

    _FORCE_INLINE_ Vector3 sign() const {
#if defined(VECTOR3SIMD_USE_SSE)
          __m128 zero = _mm_setzero_ps();
          __m128 one = _mm_set1_ps(1.0f);
          __m128 minus_one = _mm_set1_ps(-1.0f);
          __m128 gt_mask = _mm_cmpgt_ps(m_value, zero);
          __m128 lt_mask = _mm_cmplt_ps(m_value, zero);
          return Vector3(_mm_or_ps(_mm_and_ps(gt_mask, one),
                                       _mm_and_ps(lt_mask, minus_one)));
#elif defined(VECTOR3SIMD_USE_NEON)
          float32x4_t zero = vdupq_n_f32(0.0f);
          float32x4_t one = vdupq_n_f32(1.0f);
          float32x4_t minus_one = vdupq_n_f32(-1.0f);
          uint32x4_t gt_mask = vcgtq_f32(m_value, zero);
          uint32x4_t lt_mask = vcltq_f32(m_value, zero);
          float32x4_t gt_result = vbslq_f32(gt_mask, one, zero);
          return Vector3(vbslq_f32(lt_mask, minus_one, gt_result));
#else
     return Vector3(SIGN(x), SIGN(y), SIGN(z));
#endif
    }

    _FORCE_INLINE_ Vector3 floor() const {
#if defined(VECTOR3SIMD_USE_SSE)      // Outer check starts at column 0
    #if defined(__SSE4_1__)           // Inner check indented
        return Vector3(_mm_floor_ps(m_value));
    #else
          // Fallback for SSE2/3
          __m128 two_pow_23 = _mm_set1_ps(8388608.0f);
          __m128 mask = _mm_cmplt_ps(m_value, _mm_setzero_ps());
          __m128 value_plus = _mm_add_ps(m_value, two_pow_23);
          __m128 value_minus = _mm_sub_ps(m_value, two_pow_23);
          __m128 result = _mm_sub_ps(value_plus, two_pow_23);
          __m128 result_minus = _mm_add_ps(value_minus, two_pow_23);
          return Vector3(_mm_or_ps(_mm_and_ps(mask, result_minus),
                                       _mm_andnot_ps(mask, result)));
    #endif
#elif defined(VECTOR3SIMD_USE_NEON)    // Back to column 0
    return Vector3(vcvtq_f32_s32(vcvtq_s32_f32(m_value)));
#else
    return Vector3(Math::floor(x), Math::floor(y), Math::floor(z));
#endif
}

    _FORCE_INLINE_ Vector3 ceil() const {
#if defined(VECTOR3SIMD_USE_SSE)      // Outer check starts at column 0
    #if defined(__SSE4_1__)           // Inner check indented
        return Vector3(_mm_ceil_ps(m_value));
    #else
            // Fallback for SSE2/3
          __m128 two_pow_23 = _mm_set1_ps(8388608.0f);
          __m128 mask = _mm_cmpgt_ps(m_value, _mm_setzero_ps());
          __m128 value_plus = _mm_add_ps(m_value, two_pow_23);
          __m128 value_minus = _mm_sub_ps(m_value, two_pow_23);
          __m128 result = _mm_sub_ps(value_plus, two_pow_23);
          __m128 result_minus = _mm_add_ps(value_minus, two_pow_23);
          return Vector3(_mm_or_ps(_mm_and_ps(mask, result),
                                       _mm_andnot_ps(mask, result_minus)));
    #endif
#elif defined(VECTOR3SIMD_USE_NEON)    // Back to column 0
            float32x4_t ceil_val = vaddq_f32(m_value, vdupq_n_f32(0.5f));
          return Vector3(vcvtq_f32_s32(vcvtq_s32_f32(ceil_val)));
#else
    return Vector3(Math::ceil(x), Math::ceil(y), Math::ceil(z));
#endif
}
    _FORCE_INLINE_ Vector3 round() const {
#if defined(VECTOR3SIMD_USE_SSE)      // Outer check starts at column 0
    #if defined(__SSE4_1__)           // Inner check indented
          return Vector3(_mm_round_ps(
              m_value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    #else
          // Fallback for SSE2/3
          __m128 sign = _mm_and_ps(m_value, _mm_set1_ps(-0.0f));
          __m128 magic = _mm_or_ps(_mm_set1_ps(8388608.0f), sign);
          __m128 result = _mm_sub_ps(_mm_add_ps(m_value, magic), magic);
          return Vector3(result);
    #endif
#elif defined(VECTOR3SIMD_USE_NEON)    // Back to column 0
          return Vector3(vcvtq_f32_s32(
              vcvtq_s32_f32(vaddq_f32(m_value, vdupq_n_f32(0.5f)))));
#else
    return Vector3(Math::round(x), Math::round(y), Math::round(z));
#endif
}

_FORCE_INLINE_ Vector3 limit_length(float p_len = 1.0f) {
#if defined(VECTOR3SIMD_USE_SSE)
    float l = length();
    if (l > 0.0f && p_len < l) {
        return (*this * (p_len / l));
    }
    return *this;
#elif defined(VECTOR3SIMD_USE_NEON)
    float l = length();
    if (l > 0.0f && p_len < l) {
        return (*this * (p_len / l));
    }
    return *this;
#else
    const real_t l = length();
    Vector3 v = *this;
    if (l > 0 && p_len < l) {
        v = v / l;
        v = v * p_len;
    }
    return v;
#endif
}

    _FORCE_INLINE_ Vector3 min(const Vector3& p_v) const {
#if defined(VECTOR3SIMD_USE_SSE)
          return Vector3(_mm_min_ps(m_value, p_v.m_value));
#elif defined(VECTOR3SIMD_USE_NEON)
          return Vector3(vminq_f32(m_value, p_v.m_value));
#else
     return Vector3(MIN(x, p_v.x), MIN(y, p_v.y), MIN(z, p_v.z));
#endif
    }

    _FORCE_INLINE_ Vector3 max(const Vector3& p_v) const {
#if defined(VECTOR3SIMD_USE_SSE)
    return Vector3(_mm_max_ps(m_value, p_v.m_value));
#elif defined(VECTOR3SIMD_USE_NEON)
    return Vector3(vmaxq_f32(m_value, p_v.m_value));
#else
     return Vector3(MAX(x, p_v.x), MAX(y, p_v.y), MAX(z, p_v.z));
#endif
    }

    _FORCE_INLINE_ Vector3 minf(real_t p_scalar) const {
#if defined(VECTOR3SIMD_USE_SSE)
          return Vector3(_mm_min_ps(m_value, _mm_set1_ps(p_scalar)));
#elif defined(VECTOR3SIMD_USE_NEON)
          return Vector3(vminq_f32(m_value, vdupq_n_f32(p_scalar)));
#else
     return Vector3(MIN(x, p_scalar), MIN(y, p_scalar), MIN(z, p_scalar));
#endif
    }

    _FORCE_INLINE_ Vector3 maxf(real_t p_scalar) const {
#if defined(VECTOR3SIMD_USE_SSE)
    return Vector3(_mm_max_ps(m_value, _mm_set1_ps(p_scalar)));
#elif defined(VECTOR3SIMD_USE_NEON)
    return Vector3(vmaxq_f32(m_value, vdupq_n_f32(p_scalar)));
#else
     return Vector3(MAX(x, p_scalar), MAX(y, p_scalar), MAX(z, p_scalar));
#endif
    }

_FORCE_INLINE_ Vector3 move_toward(const Vector3& p_to, real_t p_delta) const {
#if defined(VECTOR3SIMD_USE_SSE)
   __m128 diff = _mm_sub_ps(p_to.m_value, m_value);
   __m128 len_sq = _mm_dp_ps(diff, diff, 0x7F);
   float len = _mm_cvtss_f32(_mm_sqrt_ss(len_sq));
   if (len <= p_delta || len < CMP_EPSILON) {
       return p_to;
   }
   __m128 s = _mm_set1_ps(p_delta / len);
   return Vector3(_mm_add_ps(m_value, _mm_mul_ps(diff, s)));
#elif defined(VECTOR3SIMD_USE_NEON)
   float32x4_t diff = vsubq_f32(p_to.m_value, m_value);
   float len_sq = vaddvq_f32(vmulq_f32(diff, diff));
   float len = sqrtf(len_sq);
   if (len <= p_delta || len < CMP_EPSILON) {
       return p_to;
   }
   return Vector3(vaddq_f32(m_value, vmulq_n_f32(diff, p_delta / len)));
#else
   Vector3 v = *this;
   Vector3 vd(p_to.x - v.x, p_to.y - v.y, p_to.z - v.z);
   real_t len = vd.length();
   if (len <= p_delta || len < CMP_EPSILON) {
       return p_to;
   }
   return Vector3(
       v.x + vd.x * p_delta / len,
       v.y + vd.y * p_delta / len,
       v.z + vd.z * p_delta / len
   );
#endif
}

    _FORCE_INLINE_ real_t distance_to(const Vector3& p_to) const {
    Vector3 diff =(*this) - p_to;
    return diff.length();
    }

_FORCE_INLINE_ real_t distance_squared_to(const Vector3 &p_to) const {
    return (*this - p_to).length_squared();
}

_FORCE_INLINE_ Vector3 direction_to(const Vector3& p_to) const {
#if defined(VECTOR3SIMD_USE_SSE)
    __m128 diff = _mm_sub_ps(p_to.m_value, m_value);
    __m128 len = _mm_set1_ps(Vector3(diff).length());
    return Vector3(_mm_div_ps(diff, len));
#elif defined(VECTOR3SIMD_USE_NEON)
    float32x4_t diff = vsubq_f32(p_to.m_value, m_value);
    float len = Vector3(diff).length();
    return Vector3(vdivq_f32(diff, vdupq_n_f32(len)));
#else
    Vector3 diff = p_to - *this;
    return diff.normalized();
#endif
}

_FORCE_INLINE_ Vector3 project(const Vector3& p_to) const {
#if defined(VECTOR3SIMD_USE_SSE)
    __m128 scalar = _mm_set1_ps(this->dot(p_to) / p_to.length_squared());
    return Vector3(_mm_mul_ps(p_to.m_value, scalar));
#elif defined(VECTOR3SIMD_USE_NEON)
    float scalar = this->dot(p_to) / p_to.length_squared();
    return Vector3(vmulq_n_f32(p_to.m_value, scalar));
#else
    return p_to * (this->dot(p_to) / p_to.length_squared());
#endif
}

_FORCE_INLINE_ Vector3 slide(const Vector3& p_normal) const {
    Vector3 normal_to_use = p_normal;
    if (!p_normal.is_normalized()) {
        WARN_PRINT("Vector3::slide() normal is not normalized. Consider normalizing ahead of time for performance.");
        normal_to_use = p_normal.normalized(); // Normalize the vector if it's not already normalized
    }

#if defined(VECTOR3SIMD_USE_SSE)
    __m128 scalar = _mm_set1_ps(this->dot(normal_to_use));
    __m128 projection = _mm_mul_ps(scalar, normal_to_use.m_value);
    return Vector3(_mm_sub_ps(m_value, projection));

#elif defined(VECTOR3SIMD_USE_NEON)
    float scalar = this->dot(normal_to_use);
    float32x4_t projection = vmulq_n_f32(normal_to_use.m_value, scalar);
    return Vector3(vsubq_f32(m_value, projection));

#else
    return *this - this->dot(normal_to_use) * normal_to_use;
#endif
}


_FORCE_INLINE_ Vector3 bounce(const Vector3 &p_normal) const {
#if defined(VECTOR3SIMD_USE_SSE)
    __m128 normal_dot = _mm_set1_ps(this->dot(p_normal));         // Broadcast dot product
    __m128 scaled_normal = _mm_mul_ps(_mm_set1_ps(2.0f), _mm_mul_ps(normal_dot, p_normal.m_value)); // 2 * (dot product) * normal
    return Vector3(_mm_sub_ps(m_value, scaled_normal));          // Subtract scaled normal from vector
#elif defined(VECTOR3SIMD_USE_NEON)
    float32x4_t normal_dot = vdupq_n_f32(this->dot(p_normal));   // Broadcast dot product
    float32x4_t scaled_normal = vmulq_n_f32(normal_dot, 2.0f);   // 2 * (dot product)
    scaled_normal = vmulq_f32(scaled_normal, p_normal.m_value);  // Scale the normal vector
    return Vector3(vsubq_f32(m_value, scaled_normal));           // Subtract scaled normal from vector
#else
    return *this - 2.0f * this->dot(p_normal) * p_normal;         // Fallback scalar implementation
#endif
}

_FORCE_INLINE_ Vector3 reflect(const Vector3 &p_normal) const {
#if defined(VECTOR3SIMD_USE_SSE)
    __m128 normal_dot = _mm_set1_ps(this->dot(p_normal));       // Broadcast dot product
    __m128 scaled_normal = _mm_mul_ps(_mm_set1_ps(2.0f), _mm_mul_ps(normal_dot, p_normal.m_value)); // 2 * (dot product) * normal
    return Vector3(_mm_sub_ps(m_value, scaled_normal));        // Subtract scaled normal from vector
#elif defined(VECTOR3SIMD_USE_NEON)
    float32x4_t normal_dot = vdupq_n_f32(this->dot(p_normal));       // Broadcast dot product
    float32x4_t scaled_normal = vmulq_n_f32(normal_dot, 2.0f);       // 2 * (dot product) * normal
    scaled_normal = vmulq_f32(scaled_normal, p_normal.m_value);      // Scale the normal vector
    return Vector3(vsubq_f32(m_value, scaled_normal));              // Subtract scaled normal from vector
#else
    return *this - 2.0f * this->dot(p_normal) * p_normal;       // Fallback scalar implementation
#endif
}

_FORCE_INLINE_ Vector3 clamp(const Vector3& p_min, const Vector3& p_max) const {
#if defined(VECTOR3SIMD_USE_SSE)
   return Vector3(_mm_min_ps(_mm_max_ps(m_value, p_min.m_value), p_max.m_value));
#elif defined(VECTOR3SIMD_USE_NEON)
   return Vector3(vminq_f32(vmaxq_f32(m_value, p_min.m_value), p_max.m_value));
#else
   return Vector3(
       CLAMP(x, p_min.x, p_max.x),
       CLAMP(y, p_min.y, p_max.y),
       CLAMP(z, p_min.z, p_max.z)
   );
#endif
}

_FORCE_INLINE_ Vector2 octahedron_encode() const {
#if defined(VECTOR3SIMD_USE_SSE)
    __m128 abs_val = _mm_andnot_ps(_mm_set1_ps(-0.0f), m_value); // Absolute value
    __m128 sum = _mm_add_ps(_mm_add_ps(_mm_shuffle_ps(abs_val, abs_val, _MM_SHUFFLE(0, 0, 0, 0)),
                                       _mm_shuffle_ps(abs_val, abs_val, _MM_SHUFFLE(1, 1, 1, 1))),
                            _mm_shuffle_ps(abs_val, abs_val, _MM_SHUFFLE(2, 2, 2, 2))); // Sum of abs(x, y, z)

    __m128 normalized = _mm_div_ps(m_value, sum); // Normalize vector components

    // Handle the reflection for negative z
    __m128 mask = _mm_cmplt_ps(_mm_shuffle_ps(m_value, m_value, _MM_SHUFFLE(2, 2, 2, 2)), _mm_setzero_ps());
    __m128 reflected = _mm_sub_ps(_mm_set1_ps(1.0f),
                                   _mm_and_ps(mask, _mm_andnot_ps(_mm_set1_ps(-0.0f), normalized))); // Reflect components

    // Normalize to [0, 1] range for encoding
    __m128 result = _mm_mul_ps(_mm_add_ps(reflected, _mm_set1_ps(1.0f)), _mm_set1_ps(0.5f));
    return Vector2(_mm_cvtss_f32(result), _mm_cvtss_f32(_mm_shuffle_ps(result, result, _MM_SHUFFLE(1, 1, 1, 1))));

#elif defined(VECTOR3SIMD_USE_NEON)
    float32x4_t abs_val = vabsq_f32(m_value); // Absolute value
    float32x4_t sum = vdupq_n_f32(vaddvq_f32(abs_val)); // Sum of abs(x, y, z)

    float32x4_t normalized = vdivq_f32(m_value, sum); // Normalize vector components

    // Handle the reflection for negative z
    uint32x4_t mask = vcltq_f32(vdupq_lane_f32(vget_high_f32(m_value), 0), vdupq_n_f32(0.0f));
    float32x4_t reflected = vsubq_f32(vdupq_n_f32(1.0f), vbslq_f32(mask, vabsq_f32(normalized), vdupq_n_f32(0.0f))); // Reflect components

    // Normalize to [0, 1] range for encoding
    float32x4_t result = vmulq_n_f32(vaddq_f32(reflected, vdupq_n_f32(1.0f)), 0.5f);
    return Vector2(vgetq_lane_f32(result, 0), vgetq_lane_f32(result, 1));

#else
    Vector3 n = abs();
    real_t sum = n.x + n.y + n.z;
    Vector3 normalized = *this / sum;

    if (normalized.z < 0.0f) {
        normalized.x = 1.0f - Math::abs(normalized.y);
        normalized.y = 1.0f - Math::abs(normalized.x);
    }
    normalized.x = normalized.x * 0.5f + 0.5f;
    normalized.y = normalized.y * 0.5f + 0.5f;
    return Vector2(normalized.x, normalized.y);
#endif
}

_FORCE_INLINE_ static Vector3 octahedron_decode(const Vector2& p_oct) {
#if defined(VECTOR3SIMD_USE_SSE)
    __m128 oct = _mm_set_ps(0.0f, 0.0f, p_oct.y, p_oct.x); // Load Vector2 into SSE register

    __m128 scaled = _mm_sub_ps(_mm_mul_ps(oct, _mm_set1_ps(2.0f)), _mm_set1_ps(1.0f)); // Scale from [0, 1] to [-1, 1]

    __m128 computed_z = _mm_sub_ps(_mm_set1_ps(1.0f),
                      _mm_add_ps(_mm_shuffle_ps(scaled, scaled, _MM_SHUFFLE(0, 0, 0, 0)),
                                _mm_shuffle_ps(scaled, scaled, _MM_SHUFFLE(1, 1, 1, 1))));

    __m128 t = _mm_max_ps(_mm_setzero_ps(), _mm_sub_ps(_mm_set1_ps(0.0f), computed_z)); // t = clamp(-z, 0, 1)

    __m128 adjusted = _mm_add_ps(scaled, _mm_and_ps(t, _mm_set1_ps(1.0f))); // Adjust reflection

    return Vector3(adjusted);

#elif defined(VECTOR3SIMD_USE_NEON)
    float32x4_t oct = vsetq_lane_f32(p_oct.x, vsetq_lane_f32(p_oct.y, vdupq_n_f32(0.0f), 1), 2);

    float32x4_t scaled = vsubq_f32(vmulq_n_f32(oct, 2.0f), vdupq_n_f32(1.0f)); // Scale from [0, 1] to [-1, 1]

    float32x4_t computed_z = vsubq_f32(vdupq_n_f32(1.0f), 
                          vaddq_f32(vabsq_f32(vget_low_f32(scaled)), 
                                   vabsq_f32(vextq_f32(scaled, scaled, 1)))); // z = 1 - |x| - |y|

    float32x4_t t = vmaxq_f32(vdupq_n_f32(0.0f), vnegq_f32(computed_z)); // t = clamp(-z, 0, 1)

    float32x4_t adjusted = vaddq_f32(scaled, t); // Adjust reflection

    return Vector3(adjusted);

#else
    Vector2 f = p_oct * 2.0f - Vector2(1.0f, 1.0f);
    Vector3 n(f.x, f.y, 1.0f - Math::abs(f.x) - Math::abs(f.y));
    if (n.z < 0.0f) {
        real_t t = CLAMP(-n.z, 0.0f, 1.0f);
        n.x += (n.x >= 0) ? -t : t;
        n.y += (n.y >= 0) ? -t : t;
    }
    return n.normalized();
#endif
}

_FORCE_INLINE_ Vector3 bezier_interpolate(const Vector3& p_control_1, const Vector3& p_control_2, const Vector3& p_end, real_t p_t) const {
    real_t omt = (1.0f - p_t);
    real_t omt2 = omt * omt;
    real_t omt3 = omt2 * omt;
    real_t t2 = p_t * p_t;
    real_t t3 = t2 * p_t;

#if defined(VECTOR3SIMD_USE_SSE)
    __m128 coef1 = _mm_set1_ps(omt3);
    __m128 coef2 = _mm_set1_ps(3.0f * omt2 * p_t);
    __m128 coef3 = _mm_set1_ps(3.0f * omt * t2);
    __m128 coef4 = _mm_set1_ps(t3);

    __m128 result = _mm_add_ps(
        _mm_add_ps(_mm_mul_ps(coef1, m_value), _mm_mul_ps(coef2, p_control_1.m_value)),
        _mm_add_ps(_mm_mul_ps(coef3, p_control_2.m_value), _mm_mul_ps(coef4, p_end.m_value))
    );
    return Vector3(result);

#elif defined(VECTOR3SIMD_USE_NEON)
    float32x4_t coef1 = vdupq_n_f32(omt3);
    float32x4_t coef2 = vdupq_n_f32(3.0f * omt2 * p_t);
    float32x4_t coef3 = vdupq_n_f32(3.0f * omt * t2);
    float32x4_t coef4 = vdupq_n_f32(t3);

    float32x4_t result = vaddq_f32(
        vaddq_f32(vmulq_f32(coef1, m_value), vmulq_f32(coef2, p_control_1.m_value)),
        vaddq_f32(vmulq_f32(coef3, p_control_2.m_value), vmulq_f32(coef4, p_end.m_value))
    );
    return Vector3(result);

#else
    return Vector3(
        Math::bezier_interpolate(x, p_control_1.x, p_control_2.x, p_end.x, p_t),
        Math::bezier_interpolate(y, p_control_1.y, p_control_2.y, p_end.y, p_t),
        Math::bezier_interpolate(z, p_control_1.z, p_control_2.z, p_end.z, p_t)
    );
#endif
}

_FORCE_INLINE_ Vector3 cubic_interpolate(const Vector3& p_b, const Vector3& p_pre_a, const Vector3& p_post_b, real_t p_weight) const {
#if defined(VECTOR3SIMD_USE_SSE)
    __m128 w = _mm_set1_ps(p_weight);
    __m128 w2 = _mm_mul_ps(w, w);
    __m128 w3 = _mm_mul_ps(w2, w);

    __m128 a0 = _mm_sub_ps(p_b.m_value, m_value);
    __m128 a1 = _mm_sub_ps(p_pre_a.m_value, m_value);
    __m128 a2 = _mm_add_ps(_mm_sub_ps(a0, a1), _mm_sub_ps(p_post_b.m_value, p_b.m_value));

    __m128 c0 = m_value;
    __m128 c1 = _mm_mul_ps(a1, _mm_set1_ps(0.5f));
    __m128 c2 = _mm_mul_ps(a0, _mm_set1_ps(2.0f));
    __m128 c3 = _mm_mul_ps(a2, _mm_set1_ps(0.5f));

    return Vector3(_mm_add_ps(_mm_add_ps(_mm_add_ps(c0, _mm_mul_ps(c1, w)), _mm_mul_ps(c2, w2)), _mm_mul_ps(c3, w3)));

#elif defined(VECTOR3SIMD_USE_NEON)
    float32x4_t w = vdupq_n_f32(p_weight);
    float32x4_t w2 = vmulq_f32(w, w);
    float32x4_t w3 = vmulq_f32(w2, w);

    float32x4_t a0 = vsubq_f32(p_b.m_value, m_value);
    float32x4_t a1 = vsubq_f32(p_pre_a.m_value, m_value);
    float32x4_t a2 = vaddq_f32(vsubq_f32(a0, a1), vsubq_f32(p_post_b.m_value, p_b.m_value));

    float32x4_t c0 = m_value;
    float32x4_t c1 = vmulq_n_f32(a1, 0.5f);
    float32x4_t c2 = vmulq_n_f32(a0, 2.0f);
    float32x4_t c3 = vmulq_n_f32(a2, 0.5f);

    return Vector3(vaddq_f32(vaddq_f32(vaddq_f32(c0, vmulq_f32(c1, w)), vmulq_f32(c2, w2)), vmulq_f32(c3, w3)));

#else
    return Vector3(
        Math::cubic_interpolate(x, p_b.x, p_pre_a.x, p_post_b.x, p_weight),
        Math::cubic_interpolate(y, p_b.y, p_pre_a.y, p_post_b.y, p_weight),
        Math::cubic_interpolate(z, p_b.z, p_pre_a.z, p_post_b.z, p_weight)
    );
#endif
}



_FORCE_INLINE_ Vector3 lerp(const Vector3& p_to, real_t p_weight) const {
#if defined(VECTOR3SIMD_USE_SSE)
   __m128 w = _mm_set1_ps(p_weight);
   __m128 inv_w = _mm_sub_ps(_mm_set1_ps(1.0f), w);
   return Vector3(_mm_add_ps(_mm_mul_ps(m_value, inv_w), _mm_mul_ps(p_to.m_value, w)));
#elif defined(VECTOR3SIMD_USE_NEON)
   float32x4_t w = vdupq_n_f32(p_weight);
   float32x4_t inv_w = vsubq_f32(vdupq_n_f32(1.0f), w);
   return Vector3(vaddq_f32(vmulq_f32(m_value, inv_w), vmulq_f32(p_to.m_value, w)));
#else
   return Vector3(
       Math::lerp(x, p_to.x, p_weight),
       Math::lerp(y, p_to.y, p_weight), 
       Math::lerp(z, p_to.z, p_weight)
   );
#endif
}

_FORCE_INLINE_ Vector3 slerp(const Vector3& p_to, real_t p_weight) const {
   real_t start_length_sq = length_squared();
   real_t end_length_sq = p_to.length_squared();
   if (unlikely(start_length_sq == 0.0f || end_length_sq == 0.0f)) {
       return lerp(p_to, p_weight);
   }
   Vector3 axis = cross(p_to);
   real_t axis_length_sq = axis.length_squared();
   if (unlikely(axis_length_sq == 0.0f)) {
       return lerp(p_to, p_weight);
   }
   real_t angle = Math::acos(dot(p_to) / Math::sqrt(start_length_sq * end_length_sq));
   if (unlikely(angle == 0.0f)) {
       return *this;
   }

   Vector3 sins = sin_vec(angle, (1.0f - p_weight) * angle, p_weight * angle, angle); //we only need the first 3 components -- the last is a dummy in this case
   real_t scale1 = sins.y / sins.x;
   real_t scale2 = sins.z / sins.x;

#if defined(VECTOR3SIMD_USE_SSE)
   __m128 v1 = _mm_mul_ps(m_value, _mm_set1_ps(scale1));
   __m128 v2 = _mm_mul_ps(p_to.m_value, _mm_set1_ps(scale2));
   return Vector3(_mm_add_ps(v1, v2));
#elif defined(VECTOR3SIMD_USE_NEON)
   float32x4_t v1 = vmulq_n_f32(m_value, scale1);
   float32x4_t v2 = vmulq_n_f32(p_to.m_value, scale2);
   return Vector3(vaddq_f32(v1, v2));
#else
   return *this * scale1 + p_to * scale2;
#endif
}

_FORCE_INLINE_ Vector3 clampf(real_t p_min, real_t p_max) const {
#if defined(VECTOR3SIMD_USE_SSE)
    __m128 min_val = _mm_set1_ps(p_min);
    __m128 max_val = _mm_set1_ps(p_max);
    return Vector3(_mm_min_ps(_mm_max_ps(m_value, min_val), max_val));
#elif defined(VECTOR3SIMD_USE_NEON)
    float32x4_t min_val = vdupq_n_f32(p_min);
    float32x4_t max_val = vdupq_n_f32(p_max);
    return Vector3(vminq_f32(vmaxq_f32(m_value, min_val), max_val));
#else
    return Vector3(
        CLAMP(x, p_min, p_max),
        CLAMP(y, p_min, p_max),
        CLAMP(z, p_min, p_max)
    );
#endif
}

_FORCE_INLINE_ void snap(const Vector3& p_step) {
#if defined(VECTOR3SIMD_USE_SSE)
    __m128 step = p_step.m_value;
    __m128 div = _mm_div_ps(m_value, step);
    __m128 rounded = _mm_round_ps(div, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    m_value = _mm_mul_ps(rounded, step);
#elif defined(VECTOR3SIMD_USE_NEON)
    float32x4_t div = vdivq_f32(m_value, p_step.m_value);
    float32x4_t rounded = vcvtq_f32_s32(vcvtq_s32_f32(vaddq_f32(div, vdupq_n_f32(0.5f))));
    m_value = vmulq_f32(rounded, p_step.m_value);
#else
    x = Math::snapped(x, p_step.x);
    y = Math::snapped(y, p_step.y);
    z = Math::snapped(z, p_step.z);
#endif
}

_FORCE_INLINE_ Vector3 snapped(const Vector3& p_step) const {
    Vector3 v = *this;
    v.snap(p_step);
    return v;
}

_FORCE_INLINE_ void snapf(real_t p_step) {
#if defined(VECTOR3SIMD_USE_SSE)
   __m128 step = _mm_set1_ps(p_step);
   __m128 div = _mm_div_ps(m_value, step);
   __m128 rounded = _mm_round_ps(div, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
   m_value = _mm_mul_ps(rounded, step);
#elif defined(VECTOR3SIMD_USE_NEON)
   // Check for division by zero
   if (Math::is_zero_approx(p_step)) {
       ERR_PRINT("Division by zero in snapf");
       return;
   }
   float32x4_t step = vdupq_n_f32(p_step);
   // Use reciprocal and multiply for division since vdivq_f32 is not native on some ARM platforms
   float32x4_t recip = vrecpeq_f32(step);
   // One Newton-Raphson iteration for better precision
   recip = vmulq_f32(vrecpsq_f32(step, recip), recip);
   float32x4_t div = vmulq_f32(m_value, recip);
   float32x4_t rounded = vcvtq_f32_s32(vcvtq_s32_f32(vaddq_f32(div, vdupq_n_f32(0.5f))));
   m_value = vmulq_f32(rounded, step);
#else
   x = Math::snapped(x, p_step);
   y = Math::snapped(y, p_step);
   z = Math::snapped(z, p_step);
#endif
}

_FORCE_INLINE_ Vector3 snappedf(real_t p_step) const {
   Vector3 v = *this;
   v.snapf(p_step);
   return v;
}

_FORCE_INLINE_ real_t signed_angle_to(const Vector3& p_to, const Vector3& p_axis) const {
#if defined(VECTOR3SIMD_USE_SSE)
    __m128 cross_vec = Vector3::vec3_cross(*this, p_to).m_value;
    __m128 angle = _mm_set1_ps(angle_to(p_to));
    __m128 dot_with_axis = _mm_dp_ps(cross_vec, p_axis.m_value, 0x7F);
    __m128 sign = _mm_and_ps(dot_with_axis, _mm_set1_ps(-0.0f));
    return _mm_cvtss_f32(_mm_xor_ps(angle, sign));
#elif defined(VECTOR3SIMD_USE_NEON)
    Vector3 cross_vec = vec3_cross(*this, p_to);
    float angle = angle_to(p_to);
    float32x4_t dot = vmulq_f32(cross_vec.m_value, p_axis.m_value);
    float32x2_t dot_sum = vpadd_f32(vget_low_f32(dot), vget_high_f32(dot));
    float sign = vget_lane_f32(dot_sum, 0);
    return copysignf(angle, sign);
#else
    Vector3 cross_vec = cross(p_to);
    return copysignf(angle_to(p_to), cross_vec.dot(p_axis));
#endif
}

_FORCE_INLINE_ Vector3 posmod(real_t p_mod) const {
#if defined(VECTOR3SIMD_USE_SSE)
    __m128 mod_val = _mm_set1_ps(p_mod);
    __m128 div = _mm_div_ps(m_value, mod_val);
    __m128 floor_div = _mm_floor_ps(div);
    __m128 result = _mm_sub_ps(m_value, _mm_mul_ps(floor_div, mod_val));
    // Handle negative cases
    __m128 neg_mask = _mm_cmplt_ps(result, _mm_setzero_ps());
    result = _mm_add_ps(result, _mm_and_ps(neg_mask, mod_val));
    return Vector3(result);
#elif defined(VECTOR3SIMD_USE_NEON)
    float32x4_t mod_val = vdupq_n_f32(p_mod);
    float32x4_t div = vdivq_f32(m_value, mod_val);
    float32x4_t floor_div = vcvtq_f32_s32(vcvtq_s32_f32(div));
    float32x4_t result = vsubq_f32(m_value, vmulq_f32(floor_div, mod_val));
    // Handle negative cases
    uint32x4_t neg_mask = vcltq_f32(result, vdupq_n_f32(0.0f));
    result = vaddq_f32(result, vbslq_f32(neg_mask, mod_val, vdupq_n_f32(0.0f)));
    return Vector3(result);
#else
    return Vector3(
        Math::fposmod(x, p_mod),
        Math::fposmod(y, p_mod),
        Math::fposmod(z, p_mod)
    );
#endif
}

_FORCE_INLINE_ Vector3 posmodv(const Vector3& p_modv) const {
#if defined(VECTOR3SIMD_USE_SSE)
    __m128 div = _mm_div_ps(m_value, p_modv.m_value);
    __m128 floor_div = _mm_floor_ps(div);
    __m128 result = _mm_sub_ps(m_value, _mm_mul_ps(floor_div, p_modv.m_value));
    // Handle negative cases
    __m128 neg_mask = _mm_cmplt_ps(result, _mm_setzero_ps());
    result = _mm_add_ps(result, _mm_and_ps(neg_mask, p_modv.m_value));
    return Vector3(result);
#elif defined(VECTOR3SIMD_USE_NEON)
    float32x4_t div = vdivq_f32(m_value, p_modv.m_value);
    float32x4_t floor_div = vcvtq_f32_s32(vcvtq_s32_f32(div));
    float32x4_t result = vsubq_f32(m_value, vmulq_f32(floor_div, p_modv.m_value));
    // Handle negative cases
    uint32x4_t neg_mask = vcltq_f32(result, vdupq_n_f32(0.0f));
    result = vaddq_f32(result, vbslq_f32(neg_mask, p_modv.m_value, vdupq_n_f32(0.0f)));
    return Vector3(result);
#else
    return Vector3(
        Math::fposmod(x, p_modv.x),
        Math::fposmod(y, p_modv.y),
        Math::fposmod(z, p_modv.z)
    );
#endif
}

_FORCE_INLINE_ real_t angle_to(const Vector3& p_to) const {
#if defined(VECTOR3SIMD_USE_SSE)
    __m128 dot_prod = _mm_dp_ps(m_value, p_to.m_value, 0x7F);
    __m128 len_prod = _mm_sqrt_ps(_mm_mul_ps(
        _mm_dp_ps(m_value, m_value, 0x7F),
        _mm_dp_ps(p_to.m_value, p_to.m_value, 0x7F)
    ));
    return Math::acos(_mm_cvtss_f32(_mm_div_ps(dot_prod, len_prod)));
#elif defined(VECTOR3SIMD_USE_NEON)
    float dot_prod = vaddvq_f32(vmulq_f32(m_value, p_to.m_value));
    float len_prod = sqrtf(vaddvq_f32(vmulq_f32(m_value, m_value)) * 
                          vaddvq_f32(vmulq_f32(p_to.m_value, p_to.m_value)));
    return Math::acos(dot_prod / len_prod);
#else
    return Math::acos(dot(p_to) / (length() * p_to.length()));
#endif
}

_FORCE_INLINE_ Vector2 octahedron_tangent_encode(float p_sign) const {
    const real_t bias = 1.0f / (real_t)32767.0f;
    Vector2 res = this->octahedron_encode();
    res.y = MAX(res.y, bias);
    res.y = res.y * 0.5f + 0.5f;
    res.y = p_sign >= 0.0f ? res.y : 1.0f - res.y;
    return res;
}

_FORCE_INLINE_ static Vector3 octahedron_tangent_decode(const Vector2& p_oct, float* r_sign) {
    Vector2 oct_compressed = p_oct;
    oct_compressed.y = oct_compressed.y * 2.0f - 1.0f;
    *r_sign = oct_compressed.y >= 0.0f ? 1.0f : -1.0f;
    oct_compressed.y = Math::abs(oct_compressed.y);
    Vector3 temp;
    return temp.octahedron_decode(oct_compressed);
}


_FORCE_INLINE_ void rotate(const Vector3& p_axis, real_t p_angle) {
    // Precompute sine and cosine
    real_t s = Math::sin(p_angle);
    real_t c = Math::cos(p_angle);
    real_t k = 1.0f - c;

    // Normalize the axis vector and load it into a SIMD register
    Vector3 axis = p_axis.normalized();
#if defined(VECTOR3SIMD_USE_SSE)
    __m128 axis_sse = _mm_set_ps(0.0f, axis.z, axis.y, axis.x); // Axis (x, y, z, 0)

    // Precompute components using SIMD
    __m128 axis2 = _mm_mul_ps(axis_sse, axis_sse);             // (x*x, y*y, z*z, 0)
    __m128 xy_xz_yz = _mm_mul_ps(axis_sse, _mm_shuffle_ps(axis_sse, axis_sse, _MM_SHUFFLE(3, 0, 2, 1))); // (xy, xz, yz, 0)

    // Prepare k, s, and c for broadcasting
    __m128 k_sse = _mm_set1_ps(k);
    __m128 s_sse = _mm_set1_ps(s);
    __m128 c_sse = _mm_set1_ps(c);

    // Precompute components of the rotation matrix
    __m128 row1 = _mm_add_ps(
        _mm_mul_ps(_mm_shuffle_ps(axis2, axis2, _MM_SHUFFLE(3, 0, 0, 0)), k_sse), // xx * k
        _mm_add_ps(
            c_sse,
            _mm_mul_ps(_mm_shuffle_ps(xy_xz_yz, xy_xz_yz, _MM_SHUFFLE(3, 2, 1, 0)), s_sse) // (yz*s, xz*s, xy*s, 0)
        )
    );

    __m128 row2 = _mm_add_ps(
        _mm_mul_ps(_mm_shuffle_ps(axis2, axis2, _MM_SHUFFLE(3, 1, 1, 1)), k_sse), // yy * k
        _mm_add_ps(
            c_sse,
            _mm_mul_ps(_mm_shuffle_ps(xy_xz_yz, xy_xz_yz, _MM_SHUFFLE(3, 0, 2, 1)), s_sse) // (xz*s, xy*s, yz*s, 0)
        )
    );

    __m128 row3 = _mm_add_ps(
        _mm_mul_ps(_mm_shuffle_ps(axis2, axis2, _MM_SHUFFLE(3, 2, 2, 2)), k_sse), // zz * k
        _mm_add_ps(
            c_sse,
            _mm_mul_ps(_mm_shuffle_ps(xy_xz_yz, xy_xz_yz, _MM_SHUFFLE(3, 1, 0, 2)), s_sse) // (xy*s, yz*s, xz*s, 0)
        )
    );

    // Apply the rotation matrix to the vector
    __m128 result = _mm_add_ps(
        _mm_add_ps(
            _mm_mul_ps(_mm_shuffle_ps(m_value, m_value, _MM_SHUFFLE(3, 0, 0, 0)), row1),
            _mm_mul_ps(_mm_shuffle_ps(m_value, m_value, _MM_SHUFFLE(3, 1, 1, 1)), row2)
        ),
        _mm_mul_ps(_mm_shuffle_ps(m_value, m_value, _MM_SHUFFLE(3, 2, 2, 2)), row3)
    );

    m_value = result;

#elif defined(VECTOR3SIMD_USE_NEON)
    // Similar NEON implementation
    float32x4_t axis_neon = vld1q_f32((const float[]){axis.x, axis.y, axis.z, 0.0f});

    // Precompute squares and products
    float32x4_t axis2 = vmulq_f32(axis_neon, axis_neon); // (x*x, y*y, z*z, 0)
    float32x4_t xy_xz_yz = vmulq_f32(axis_neon, vextq_f32(axis_neon, axis_neon, 1)); // (xy, xz, yz, 0)

    float32x4_t k_neon = vdupq_n_f32(k);
    float32x4_t s_neon = vdupq_n_f32(s);
    float32x4_t c_neon = vdupq_n_f32(c);

    // Precompute rows of the rotation matrix
    float32x4_t row1 = vaddq_f32(
        vmulq_f32(vextq_f32(axis2, axis2, 0), k_neon),
        vaddq_f32(c_neon, vmulq_f32(vextq_f32(xy_xz_yz, xy_xz_yz, 2), s_neon))
    );

    float32x4_t row2 = vaddq_f32(
        vmulq_f32(vextq_f32(axis2, axis2, 1), k_neon),
        vaddq_f32(c_neon, vmulq_f32(vextq_f32(xy_xz_yz, xy_xz_yz, 0), s_neon))
    );

    float32x4_t row3 = vaddq_f32(
        vmulq_f32(vextq_f32(axis2, axis2, 2), k_neon),
        vaddq_f32(c_neon, vmulq_f32(vextq_f32(xy_xz_yz, xy_xz_yz, 1), s_neon))
    );

    // Apply rotation
    float32x4_t result = vaddq_f32(
        vaddq_f32(vmulq_f32(vdupq_lane_f32(vget_low_f32(m_value), 0), row1),
                  vmulq_f32(vdupq_lane_f32(vget_low_f32(m_value), 1), row2)),
        vmulq_f32(vdupq_lane_f32(vget_high_f32(m_value), 0), row3)
    );

    m_value = result;

#else
    real_t xx = axis.x * axis.x;
    real_t xy = axis.x * axis.y;
    real_t xz = axis.x * axis.z;
    real_t yy = axis.y * axis.y;
    real_t yz = axis.y * axis.z;
    real_t zz = axis.z * axis.z;

    // Scalar fallback remains unchanged
    real_t nx = (xx * k + c) * x + (xy * k - axis.z * s) * y + (xz * k + axis.y * s) * z;
    real_t ny = (xy * k + axis.z * s) * x + (yy * k + c) * y + (yz * k - axis.x * s) * z;
    real_t nz = (xz * k - axis.y * s) * x + (yz * k + axis.x * s) * y + (zz * k + c) * z;

    x = nx;
    y = ny;
    z = nz;
#endif
}


_FORCE_INLINE_ Vector3 rotated(const Vector3& p_axis, real_t p_angle) const {
    Vector3 v = *this;
    v.rotate(p_axis, p_angle);
    return v;
}

_FORCE_INLINE_ Vector3 rotated_local(const Vector3& p_axis, real_t p_angle) const {
    Vector3 axis = p_axis.normalized();
    real_t c = Math::cos(p_angle);
    real_t s = Math::sin(p_angle);
    real_t C = 1.0f - c;

#if defined(VECTOR3SIMD_USE_SSE)
    // Load constants
    __m128 cos_v = _mm_set1_ps(c);
    __m128 sin_v = _mm_set1_ps(s);
    __m128 C_v = _mm_set1_ps(C);
    
    // Calculate axis products
    __m128 axis_sq = _mm_mul_ps(axis.m_value, axis.m_value);                 // [xx, yy, zz, ww]
    __m128 axis_v = axis.m_value;                                            // [x, y, z, w]
    __m128 axis_yzx = _mm_shuffle_ps(axis_v, axis_v, _MM_SHUFFLE(3,0,2,1)); // [y, z, x, w]
    __m128 axis_prod = _mm_mul_ps(axis_v, axis_yzx);                        // [xy, yz, xz, w]

    // Build rotation matrix elements
    __m128 diag = _mm_add_ps(_mm_mul_ps(axis_sq, C_v), cos_v);
    __m128 cross_terms = _mm_mul_ps(axis_prod, C_v);
    __m128 sin_terms = _mm_mul_ps(axis_v, sin_v);

    // Apply rotation
    __m128 row1 = _mm_setr_ps(
        _mm_cvtss_f32(diag),
        _mm_cvtss_f32(_mm_sub_ps(cross_terms, _mm_shuffle_ps(sin_terms, sin_terms, _MM_SHUFFLE(3,2,1,0)))),
        _mm_cvtss_f32(_mm_add_ps(cross_terms, _mm_shuffle_ps(sin_terms, sin_terms, _MM_SHUFFLE(3,1,2,0)))),
        0.0f
    );
    __m128 row2 = _mm_setr_ps(
        _mm_cvtss_f32(_mm_add_ps(cross_terms, sin_terms)),
        _mm_cvtss_f32(_mm_shuffle_ps(diag, diag, _MM_SHUFFLE(3,1,1,1))),
        _mm_cvtss_f32(_mm_sub_ps(_mm_shuffle_ps(cross_terms, cross_terms, _MM_SHUFFLE(3,1,2,0)), 
                                _mm_shuffle_ps(sin_terms, sin_terms, _MM_SHUFFLE(3,0,2,1)))),
        0.0f
    );
    __m128 row3 = _mm_setr_ps(
        _mm_cvtss_f32(_mm_sub_ps(_mm_shuffle_ps(cross_terms, cross_terms, _MM_SHUFFLE(3,2,0,1)), sin_terms)),
        _mm_cvtss_f32(_mm_add_ps(_mm_shuffle_ps(cross_terms, cross_terms, _MM_SHUFFLE(3,2,1,0)), 
                                _mm_shuffle_ps(sin_terms, sin_terms, _MM_SHUFFLE(3,0,1,2)))),
        _mm_cvtss_f32(_mm_shuffle_ps(diag, diag, _MM_SHUFFLE(3,2,2,2))),
        0.0f
    );

    __m128 result = _mm_add_ps(
        _mm_add_ps(
            _mm_mul_ps(_mm_shuffle_ps(m_value, m_value, _MM_SHUFFLE(3,0,0,0)), row1),
            _mm_mul_ps(_mm_shuffle_ps(m_value, m_value, _MM_SHUFFLE(3,1,1,1)), row2)
        ),
        _mm_mul_ps(_mm_shuffle_ps(m_value, m_value, _MM_SHUFFLE(3,2,2,2)), row3)
    );

    return Vector3(result);

#elif defined(VECTOR3SIMD_USE_NEON)
    float32x4_t cos_v = vdupq_n_f32(c);
    float32x4_t sin_v = vdupq_n_f32(s);
    float32x4_t C_v = vdupq_n_f32(C);
    
    float32x4_t axis_sq = vmulq_f32(axis.m_value, axis.m_value);
    float32x4_t axis_prod = vmulq_f32(axis.m_value, vextq_f32(axis.m_value, axis.m_value, 1));
    
    float32x4_t diag = vaddq_f32(vmulq_f32(axis_sq, C_v), cos_v);
    float32x4_t cross_terms = vmulq_f32(axis_prod, C_v);
    float32x4_t sin_terms = vmulq_f32(axis.m_value, sin_v);
    
    float temp[4];
    vst1q_f32(temp, diag);
    float xx_C_c = temp[0];
    float yy_C_c = temp[1];
    float zz_C_c = temp[2];
    
    vst1q_f32(temp, cross_terms);
    float xy_C = temp[0];
    float yz_C = temp[1];
    float xz_C = temp[2];
    
    vst1q_f32(temp, sin_terms);
    float x_s = temp[0];
    float y_s = temp[1];
    float z_s = temp[2];
    
    float32x4_t row1 = vsetq_lane_f32(0.0f,
        vsetq_lane_f32(xz_C + y_s,
            vsetq_lane_f32(xy_C - z_s,
                vsetq_lane_f32(xx_C_c, vdupq_n_f32(0.0f), 0),
            1),
        2),
    3);
    
    float32x4_t row2 = vsetq_lane_f32(0.0f,
        vsetq_lane_f32(yz_C - x_s,
            vsetq_lane_f32(yy_C_c,
                vsetq_lane_f32(xy_C + z_s, vdupq_n_f32(0.0f), 0),
            1),
        2),
    3);
    
    float32x4_t row3 = vsetq_lane_f32(0.0f,
        vsetq_lane_f32(zz_C_c,
            vsetq_lane_f32(yz_C + x_s,
                vsetq_lane_f32(xz_C - y_s, vdupq_n_f32(0.0f), 0),
            1),
        2),
    3);

    return Vector3(vaddq_f32(
        vaddq_f32(
            vmulq_f32(vdupq_lane_f32(vget_low_f32(m_value), 0), row1),
            vmulq_f32(vdupq_lane_f32(vget_low_f32(m_value), 1), row2)
        ),
        vmulq_f32(vdupq_lane_f32(vget_high_f32(m_value), 0), row3)
    ));

#else
    real_t xx = axis.x * axis.x;
    real_t xy = axis.x * axis.y;
    real_t xz = axis.x * axis.z;
    real_t yy = axis.y * axis.y;
    real_t yz = axis.y * axis.z;
    real_t zz = axis.z * axis.z;

    return Vector3(
        (xx * C + c) * x + (xy * C - axis.z * s) * y + (xz * C + axis.y * s) * z,
        (xy * C + axis.z * s) * x + (yy * C + c) * y + (yz * C - axis.x * s) * z,
        (xz * C - axis.y * s) * x + (yz * C + axis.x * s) * y + (zz * C + c) * z
    );
#endif
}

_FORCE_INLINE_ Vector3 cubic_interpolate_in_time(const Vector3& p_b, const Vector3& p_pre_a, const Vector3& p_post_b, 
   real_t p_weight, real_t p_b_t, real_t p_pre_a_t, real_t p_post_b_t) const {
#if defined(VECTOR3SIMD_USE_SSE)
   __m128 t = _mm_set1_ps(p_weight);
   __m128 t2 = _mm_mul_ps(t, t);
   __m128 t3 = _mm_mul_ps(t2, t);

   __m128 pb_pa = _mm_div_ps(_mm_sub_ps(p_b.m_value, m_value), _mm_set1_ps(p_b_t));
   __m128 pc_pa = _mm_div_ps(_mm_sub_ps(p_pre_a.m_value, m_value), _mm_set1_ps(p_pre_a_t));
   __m128 pb_pc = _mm_div_ps(_mm_sub_ps(p_b.m_value, p_post_b.m_value), _mm_set1_ps(p_post_b_t));

   __m128 h1 = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(2.0f), t3), _mm_mul_ps(_mm_set1_ps(3.0f), t2));
   __m128 h2 = _mm_sub_ps(t3, _mm_mul_ps(_mm_set1_ps(2.0f), t2));
   __m128 h3 = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(3.0f), t2), _mm_mul_ps(_mm_set1_ps(2.0f), t3));
   __m128 h4 = _mm_sub_ps(t3, t2);

   return Vector3(_mm_add_ps(
       _mm_add_ps(_mm_mul_ps(m_value, h1), _mm_mul_ps(pb_pa, h2)),
       _mm_add_ps(_mm_mul_ps(p_b.m_value, h3), _mm_mul_ps(pb_pc, h4))
   ));
#elif defined(VECTOR3SIMD_USE_NEON)
   float32x4_t t = vdupq_n_f32(p_weight);
   float32x4_t t2 = vmulq_f32(t, t);
   float32x4_t t3 = vmulq_f32(t2, t); 

   float32x4_t pb_pa = vdivq_f32(vsubq_f32(p_b.m_value, m_value), vdupq_n_f32(p_b_t));
   float32x4_t pc_pa = vdivq_f32(vsubq_f32(p_pre_a.m_value, m_value), vdupq_n_f32(p_pre_a_t));
   float32x4_t pb_pc = vdivq_f32(vsubq_f32(p_b.m_value, p_post_b.m_value), vdupq_n_f32(p_post_b_t));

   float32x4_t h1 = vsubq_f32(vmulq_n_f32(t3, 2.0f), vmulq_n_f32(t2, 3.0f));
   float32x4_t h2 = vsubq_f32(t3, vmulq_n_f32(t2, 2.0f));
   float32x4_t h3 = vsubq_f32(vmulq_n_f32(t2, 3.0f), vmulq_n_f32(t3, 2.0f));
   float32x4_t h4 = vsubq_f32(t3, t2);

   return Vector3(vaddq_f32(
       vaddq_f32(vmulq_f32(m_value, h1), vmulq_f32(pb_pa, h2)),
       vaddq_f32(vmulq_f32(p_b.m_value, h3), vmulq_f32(pb_pc, h4))
   ));
#else
   return Vector3(
       Math::cubic_interpolate_in_time(x, p_b.x, p_pre_a.x, p_post_b.x, p_weight, p_b_t, p_pre_a_t, p_post_b_t),
       Math::cubic_interpolate_in_time(y, p_b.y, p_pre_a.y, p_post_b.y, p_weight, p_b_t, p_pre_a_t, p_post_b_t),
       Math::cubic_interpolate_in_time(z, p_b.z, p_pre_a.z, p_post_b.z, p_weight, p_b_t, p_pre_a_t, p_post_b_t)
   );
#endif
}

_FORCE_INLINE_ Vector3 bezier_derivative(const Vector3& p_control_1, const Vector3& p_control_2, const Vector3& p_end, real_t p_t) const {
   real_t omt = (1.0f - p_t);
   real_t omt2 = omt * omt;
   real_t t2 = p_t * p_t;

#if defined(VECTOR3SIMD_USE_SSE)
   __m128 coef1 = _mm_set1_ps(-3.0f * omt2);
   __m128 coef2 = _mm_set1_ps(3.0f * omt2 - 6.0f * p_t * omt);
   __m128 coef3 = _mm_set1_ps(6.0f * p_t * omt - 3.0f * t2);
   __m128 coef4 = _mm_set1_ps(3.0f * t2);

   return Vector3(_mm_add_ps(
       _mm_add_ps(_mm_mul_ps(coef1, m_value), _mm_mul_ps(coef2, p_control_1.m_value)),
       _mm_add_ps(_mm_mul_ps(coef3, p_control_2.m_value), _mm_mul_ps(coef4, p_end.m_value))
   ));
#elif defined(VECTOR3SIMD_USE_NEON)
   float32x4_t coef1 = vdupq_n_f32(-3.0f * omt2);
   float32x4_t coef2 = vdupq_n_f32(3.0f * omt2 - 6.0f * p_t * omt);
   float32x4_t coef3 = vdupq_n_f32(6.0f * p_t * omt - 3.0f * t2);
   float32x4_t coef4 = vdupq_n_f32(3.0f * t2);

   return Vector3(vaddq_f32(
       vaddq_f32(vmulq_f32(coef1, m_value), vmulq_f32(coef2, p_control_1.m_value)),
       vaddq_f32(vmulq_f32(coef3, p_control_2.m_value), vmulq_f32(coef4, p_end.m_value))
   ));
#else
   return Vector3(
       Math::bezier_derivative(x, p_control_1.x, p_control_2.x, p_end.x, p_t),
       Math::bezier_derivative(y, p_control_1.y, p_control_2.y, p_end.y, p_t),
       Math::bezier_derivative(z, p_control_1.z, p_control_2.z, p_end.z, p_t)
   );
#endif
}



Vector3 vector_divide_neon(Vector3 a, Vector3 b) {
    #if defined(VECTOR3SIMD_USE_NEON) || defined(VECTOR3_USE_NEON)
    uint32x4_t is_zero = vceqq_f32(b.m_value, vdupq_n_f32(0.0f));
    if (vgetq_lane_u32(is_zero, 0) || vgetq_lane_u32(is_zero, 1) || vgetq_lane_u32(is_zero, 2)) {
        ERR_PRINT("Division by zero in vector division");
        return Vector3();
    }
    float32x4_t reciprocal = vrecpeq_f32(b.m_value);
    reciprocal = vmulq_f32(vrecpsq_f32(b.m_value, reciprocal), reciprocal);
    return Vector3(vmulq_f32(a.m_value, reciprocal));
    #else
        // Fallback for scalar
        return Vector3(a.x / b.x, a.y / b.y, a.z / b.z); 
    #endif
}

_FORCE_INLINE_ static Vector3 sin_vec(real_t x1, real_t x2, real_t x3, real_t x4) {
#if defined(VECTOR3SIMD_USE_SSE)
    return Vector3(_mm_sin_ps(_mm_set_ps(x4, x3, x2, x1)));
#elif defined(VECTOR3SIMD_USE_NEON)
    float32x4_t v = {x1, x2, x3, x4};
    return Vector3(sin_neon(v)); 
#else
    return Vector3(Math::sin(x1), Math::sin(x2), Math::sin(x3), Math::sin(x4));
#endif
}

_FORCE_INLINE_ static Vector3 cos_vec(real_t x1, real_t x2, real_t x3, real_t x4) {
#if defined(VECTOR3SIMD_USE_SSE)
    return Vector3(_mm_cos_ps(_mm_set_ps(x4, x3, x2, x1)));
#elif defined(VECTOR3SIMD_USE_NEON) 
    float32x4_t v = {x1, x2, x3, x4};
    return Vector3(cos_neon(v));
#else
    return Vector3(Math::cos(x1), Math::cos(x2), Math::cos(x3), Math::cos(x4)); 
#endif
}

// Add-assign: modifies *this in place, then returns *this.
_FORCE_INLINE_ Vector3 &operator+=(const Vector3 &p_v) {
#if defined(VECTOR3SIMD_USE_SSE)
    // SSE version: add the __m128 registers and store back into *this.
    m_value = _mm_add_ps(m_value, p_v.m_value);
#elif defined(VECTOR3SIMD_USE_NEON)
    // NEON version: add float32x4_t and store back into *this.
    m_value = vaddq_f32(m_value, p_v.m_value);
#else
    x += p_v.x;
    y += p_v.y;
    z += p_v.z;
#endif
    return *this;
}

// Binary plus: returns a NEW Vector3 that is (*this + p_v), leaving *this unmodified.
_FORCE_INLINE_ Vector3 operator+(const Vector3 &p_v) const {
#if defined(VECTOR3SIMD_USE_SSE)
    return Vector3(_mm_add_ps(m_value, p_v.m_value));
#elif defined(VECTOR3SIMD_USE_NEON)
    return Vector3(vaddq_f32(m_value, p_v.m_value));
#else
    return Vector3(x + p_v.x, y + p_v.y, z + p_v.z);
#endif
}


// Subtract-assign: modifies *this in place, then returns *this.
_FORCE_INLINE_ Vector3 &operator-=(const Vector3 &p_v) {
#if defined(VECTOR3SIMD_USE_SSE)
    // SSE version: subtract the __m128 registers and store back in *this.
    m_value = _mm_sub_ps(m_value, p_v.m_value);
#elif defined(VECTOR3SIMD_USE_NEON)
    // NEON version: subtract the float32x4_t and store back in *this.
    m_value = vsubq_f32(m_value, p_v.m_value);
#else
    x -= p_v.x;
    y -= p_v.y;
    z -= p_v.z;
#endif
    return *this;
}

// Binary minus: returns a NEW Vector3 that is (*this - p_v), leaving *this unmodified.
_FORCE_INLINE_ Vector3 operator-(const Vector3 &p_v) const {
#if defined(VECTOR3SIMD_USE_SSE)
    // SSE version: return a new Vector3 from the __m128 difference.
    return Vector3(_mm_sub_ps(m_value, p_v.m_value));
#elif defined(VECTOR3SIMD_USE_NEON)
    // NEON version: return a new Vector3 from vsubq_f32.
    return Vector3(vsubq_f32(m_value, p_v.m_value));
#else
    return Vector3(x - p_v.x, y - p_v.y, z - p_v.z);
#endif
}


_FORCE_INLINE_ Vector3& operator*=(const Vector3& p_v) {
#if defined(VECTOR3SIMD_USE_SSE)
    m_value = _mm_mul_ps(m_value, p_v.m_value);
#elif defined(VECTOR3SIMD_USE_NEON)
    m_value = vmulq_f32(m_value, p_v.m_value);
#else
    x *= p_v.x;
    y *= p_v.y;
    z *= p_v.z;
#endif
    return *this;
}

_FORCE_INLINE_ Vector3 operator*(const Vector3& p_v) const {
#if defined(VECTOR3SIMD_USE_SSE)
    return Vector3(_mm_mul_ps(m_value, p_v.m_value));
#elif defined(VECTOR3SIMD_USE_NEON)
    return Vector3(vmulq_f32(m_value, p_v.m_value));
#else
    return Vector3(x * p_v.x, y * p_v.y, z * p_v.z);
#endif
}

_FORCE_INLINE_ Vector3& operator/=(const Vector3& p_v) {
#if defined(VECTOR3SIMD_USE_SSE)
    m_value = _mm_div_ps(m_value, p_v.m_value);
#elif defined(VECTOR3SIMD_USE_NEON) || defined(VECTOR3_USE_NEON)
    // Check for division by zero
    uint32x4_t is_zero = vceqq_f32(p_v.m_value, vdupq_n_f32(0.0f));
    if (vgetq_lane_u32(is_zero, 0) || vgetq_lane_u32(is_zero, 1) || vgetq_lane_u32(is_zero, 2)) {
        ERR_PRINT("Division by zero in vector division");
        *this = Vector3();
        return *this;
    }
    // neon doesn't have a divide instruction, so we need to use a reciprocal and multiply
    float32x4_t reciprocal = vrecpeq_f32(p_v.m_value);
    // One Newton-Raphson iteration for better precision
    reciprocal = vmulq_f32(vrecpsq_f32(p_v.m_value, reciprocal), reciprocal);
    m_value = vmulq_f32(m_value, reciprocal);
#else
    x /= p_v.x;
    y /= p_v.y;
    z /= p_v.z;
#endif
    return *this;
}

_FORCE_INLINE_ Vector3 operator/(const Vector3& p_v) const {
#if defined(VECTOR3SIMD_USE_SSE)
    return Vector3(_mm_div_ps(m_value, p_v.m_value));
#elif defined(VECTOR3SIMD_USE_NEON)
    return Vector3(vector_divide_neon(m_value, p_v.m_value));
#else
    return Vector3(x / p_v.x, y / p_v.y, z / p_v.z);
#endif
}

_FORCE_INLINE_ Vector3& operator*=(real_t p_scalar) {
#if defined(VECTOR3SIMD_USE_SSE)
    __m128 scalar = _mm_set1_ps(p_scalar);
    m_value = _mm_mul_ps(m_value, scalar);
#elif defined(VECTOR3SIMD_USE_NEON)
    m_value = vmulq_n_f32(m_value, p_scalar);
#else
    x *= p_scalar;
    y *= p_scalar;
    z *= p_scalar;
#endif
    return *this;
}

_FORCE_INLINE_ Vector3 operator*(real_t p_scalar) const {
#if defined(VECTOR3SIMD_USE_SSE)
    __m128 scalar = _mm_set1_ps(p_scalar);
    return Vector3(_mm_mul_ps(m_value, scalar));
#elif defined(VECTOR3SIMD_USE_NEON)
    return Vector3(vmulq_n_f32(m_value, p_scalar));
#else
    return Vector3(x * p_scalar, y * p_scalar, z * p_scalar);
#endif
}

_FORCE_INLINE_ Vector3& operator/=(real_t p_scalar) {
#if defined(VECTOR3SIMD_USE_SSE)
    __m128 scalar = _mm_set1_ps(p_scalar);
    m_value = _mm_div_ps(m_value, scalar);
#elif defined(VECTOR3SIMD_USE_NEON) || defined(VECTOR3_USE_NEON)
    // Check for division by zero
    if (Math::is_zero_approx(p_scalar)) {
        ERR_PRINT("Division by zero in vector-scalar division");
        *this = Vector3();
        return *this;
    }
    // NEON doesn't have a direct scalar divide, use multiplication by reciprocal
    m_value = vmulq_n_f32(m_value, 1.0f / p_scalar);
#else
    x /= p_scalar;
    y /= p_scalar;
    z /= p_scalar;
#endif
    return *this;
}

_FORCE_INLINE_ Vector3 operator/(real_t p_scalar) const {
#if defined(VECTOR3SIMD_USE_SSE)
    __m128 scalar = _mm_set1_ps(p_scalar);
    return Vector3(_mm_div_ps(m_value, scalar));
#elif defined(VECTOR3SIMD_USE_NEON) || defined(VECTOR3_USE_NEON)
    // Check for division by zero
    if (Math::is_zero_approx(p_scalar)) {
        ERR_PRINT("Division by zero in vector-scalar division");
        return Vector3();
    }
    // NEON doesn't have a direct scalar divide, use multiplication by reciprocal
    return Vector3(vmulq_n_f32(m_value, 1.0f / p_scalar));
#else
    return Vector3(x / p_scalar, y / p_scalar, z / p_scalar);
#endif
}

_FORCE_INLINE_ Vector3 operator-() const {
#if defined(VECTOR3SIMD_USE_SSE)
    __m128 zero = _mm_setzero_ps();
    return Vector3(_mm_sub_ps(zero, m_value));
#elif defined(VECTOR3SIMD_USE_NEON)
    float32x4_t zero = vdupq_n_f32(0.0f);
    return Vector3(vsubq_f32(zero, m_value));
#else
    return Vector3(-x, -y, -z);
#endif
}

    _FORCE_INLINE_ bool operator==(const Vector3& p_v) const {
        return x == p_v.x && y == p_v.y && z == p_v.z;
    }

    _FORCE_INLINE_ bool operator!=(const Vector3& p_v) const {
        return x != p_v.x || y != p_v.y || z != p_v.z;
    }

    _FORCE_INLINE_ bool operator<(const Vector3& p_v) const {
        if (x == p_v.x) {
            if (y == p_v.y) {
                return z < p_v.z;
            }
            return y < p_v.y;
        }
        return x < p_v.x;
    }

    _FORCE_INLINE_ bool operator<=(const Vector3& p_v) const {
        if (x == p_v.x) {
            if (y == p_v.y) {
                return z <= p_v.z;
            }
            return y < p_v.y;
        }
        return x < p_v.x;
    }

    _FORCE_INLINE_ bool operator>(const Vector3& p_v) const {
        if (x == p_v.x) {
            if (y == p_v.y) {
                return z > p_v.z;
            }
            return y > p_v.y;
        }
        return x > p_v.x;
    }

    _FORCE_INLINE_ bool operator>=(const Vector3& p_v) const {
        if (x == p_v.x) {
            if (y == p_v.y) {
                return z >= p_v.z;
            }
            return y > p_v.y;
        }
        return x > p_v.x;
    }

_FORCE_INLINE_ bool is_equal_approx(const Vector3& p_v) const {
    #if defined(VECTOR3SIMD_USE_SSE)
        __m128 epsilon = _mm_set1_ps(CMP_EPSILON);
        __m128 diff = _mm_sub_ps(m_value, p_v.m_value);
        __m128 abs_diff = _mm_andnot_ps(_mm_set1_ps(-0.0f), diff);
        __m128 cmp = _mm_cmple_ps(abs_diff, epsilon);
        return (_mm_movemask_ps(cmp) & 0x7) == 0x7;
    #elif defined(VECTOR3SIMD_USE_NEON)
        float32x4_t epsilon = vdupq_n_f32(CMP_EPSILON);
        float32x4_t diff = vsubq_f32(m_value, p_v.m_value);
        float32x4_t abs_diff = vabsq_f32(diff);
        uint32x4_t cmp = vcleq_f32(abs_diff, epsilon);
        return (vgetq_lane_u32(cmp, 0) & vgetq_lane_u32(cmp, 1) & vgetq_lane_u32(cmp, 2)) != 0;
    #else
        return Math::is_equal_approx(x, p_v.x) && Math::is_equal_approx(y, p_v.y) && Math::is_equal_approx(z, p_v.z);
    #endif
}

   _FORCE_INLINE_ bool is_zero_approx() const {
    #if defined(VECTOR3SIMD_USE_SSE)
          __m128 epsilon = _mm_set1_ps(CMP_EPSILON);
          __m128 abs_val = _mm_andnot_ps(_mm_set1_ps(-0.0f), m_value);
          __m128 cmp = _mm_cmple_ps(abs_val, epsilon);
          return (_mm_movemask_ps(cmp) & 0x7) == 0x7;
    #elif defined(VECTOR3SIMD_USE_NEON)
        float32x4_t epsilon = vdupq_n_f32(CMP_EPSILON);
        float32x4_t abs_val = vabsq_f32(m_value);
        uint32x4_t cmp = vcleq_f32(abs_val, epsilon);
        return (vgetq_lane_u32(cmp, 0) & vgetq_lane_u32(cmp, 1) & vgetq_lane_u32(cmp, 2)) != 0;
    #else
       return Math::is_zero_approx(x) && Math::is_zero_approx(y) && Math::is_zero_approx(z);
    #endif
}

   bool is_finite() const {
       return Math::is_finite(x) && Math::is_finite(y) && Math::is_finite(z);
   }

/**************************************************************************/
   /* Type conversions and utility methods */
   /**************************************************************************/
   operator String() const;
   operator Vector3i() const;

};

/*********************************************************************************/
/* Global operators */
/*********************************************************************************/
template<typename T>
struct is_valid_vector3_scalar {
    static constexpr bool value = 
        std::is_arithmetic<T>::value && 
        !std::is_same<typename std::remove_cv<typename std::remove_reference<T>::type>::type, Vector3i>::value;
};

template <typename T, 
          typename std::enable_if<is_valid_vector3_scalar<T>::value, bool>::type = true>
_FORCE_INLINE_ Vector3 operator*(T scalar, const Vector3& vec) {
    return vec * static_cast<real_t>(scalar);
}

_FORCE_INLINE_ Vector3 vec3_cross(const Vector3& a, const Vector3& b) {
    return Vector3::vec3_cross(a, b);
}

_FORCE_INLINE_ real_t vec3_dot(const Vector3& a, const Vector3& b) {
    return Vector3::vec3_dot(a, b);
}

/**********************************************************************************/
/* Static constants */
/**********************************************************************************/

#endif // VECTOR3_H