/**************************************************************************/
/*  vector3SIMD.h                                                         */
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

#ifndef VECTOR3_SIMD_H
#define VECTOR3_SIMD_H

#include <cmath>
#include <cstdint>

// Core math includes
#include "core/math/math_defs.h"
#include "core/math/math_funcs.h"

// Error handling
#include "core/error/error_macros.h"

// Architecture detection and SIMD includes
#if defined(__SSE__) || (defined(_M_X64) && !defined(__EMSCRIPTEN__))
#define VECTOR3SIMD_USE_SSE
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#if defined(__SSE3__)
#include <pmmintrin.h>  // SSE3
#endif
#if defined(__SSE4_1__)
#include <smmintrin.h>  // SSE4.1
#endif
#elif defined(__ARM_NEON) || defined(__aarch64__)
#define VECTOR3SIMD_USE_NEON
#include <arm_neon.h>
#endif

struct Vector3; // Forward declaration

struct alignas(16) Vector3SIMD {
private:
    union {
#if defined(VECTOR3SIMD_USE_SSE)
        __m128 m_value;
#elif defined(VECTOR3SIMD_USE_NEON)
        float32x4_t m_value;
#endif
        float f[4];  // Always keep as float for SIMD alignment
    };

public:
    /******************/
    /* Constructors   */
    /******************/
    
    _FORCE_INLINE_ Vector3SIMD() {
#if defined(VECTOR3SIMD_USE_SSE)
        m_value = _mm_setzero_ps();
#elif defined(VECTOR3SIMD_USE_NEON)
        m_value = vdupq_n_f32(0.0f);
#else
        f[0] = f[1] = f[2] = f[3] = 0.0f;
#endif
    }

    _FORCE_INLINE_ Vector3SIMD(float p_x, float p_y, float p_z) {
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

    // Array constructor
    _FORCE_INLINE_ explicit Vector3SIMD(const float *p_array) {
#if defined(VECTOR3SIMD_USE_SSE)
        m_value = _mm_loadu_ps(p_array);  // Using unaligned load for safety
        m_value = _mm_insert_ps(m_value, _mm_setzero_ps(), 0x30); // Zero w component
#elif defined(VECTOR3SIMD_USE_NEON)
        m_value = vld1q_f32(p_array);
        f[3] = 0.0f;  // Zero w component
#else
        f[0] = p_array[0];
        f[1] = p_array[1];
        f[2] = p_array[2];
        f[3] = 0.0f;
#endif
    }

    // Platform-specific constructors
#if defined(VECTOR3SIMD_USE_SSE)
    _FORCE_INLINE_ Vector3SIMD(__m128 val) : m_value(val) {}
#elif defined(VECTOR3SIMD_USE_NEON)
    _FORCE_INLINE_ Vector3SIMD(float32x4_t val) : m_value(val) {}
#endif

    /**********************/
    /* Component Access   */
    /**********************/

    _FORCE_INLINE_ float x() const { return f[0]; }
    _FORCE_INLINE_ float y() const { return f[1]; }
    _FORCE_INLINE_ float z() const { return f[2]; }
    _FORCE_INLINE_ float w() const { return f[3]; }

    _FORCE_INLINE_ void set_x(float p_x) { f[0] = p_x; }
    _FORCE_INLINE_ void set_y(float p_y) { f[1] = p_y; }
    _FORCE_INLINE_ void set_z(float p_z) { f[2] = p_z; }

    /*************************/
    /* SSE Implementations   */
    /*************************/
#if defined(VECTOR3SIMD_USE_SSE)
    static _FORCE_INLINE_ __m128 load_scalar_sse(float s) {
        return _mm_set1_ps(s);
    }

    _FORCE_INLINE_ Vector3SIMD cross_sse(const Vector3SIMD &p_v) const {
        __m128 a = m_value;
        __m128 b = p_v.m_value;
        __m128 a_yzx = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1));
        __m128 b_yzx = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1));
        __m128 c = _mm_sub_ps(
            _mm_mul_ps(a, b_yzx),
            _mm_mul_ps(a_yzx, b));
        return Vector3SIMD(_mm_shuffle_ps(c, c, _MM_SHUFFLE(3, 0, 2, 1)));
    }

    _FORCE_INLINE_ float dot_sse(const Vector3SIMD &p_v) const {
#if defined(__SSE4_1__)
        return _mm_cvtss_f32(_mm_dp_ps(m_value, p_v.m_value, 0x7F));
#else
        __m128 mul = _mm_mul_ps(m_value, p_v.m_value);
        __m128 shuf = _mm_movehdup_ps(mul);
        __m128 sums = _mm_add_ps(mul, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
#endif
    }

    _FORCE_INLINE_ Vector3SIMD add_sse(const Vector3SIMD &p_v) const {
        return Vector3SIMD(_mm_add_ps(m_value, p_v.m_value));
    }

    _FORCE_INLINE_ Vector3SIMD sub_sse(const Vector3SIMD &p_v) const {
        return Vector3SIMD(_mm_sub_ps(m_value, p_v.m_value));
    }

    _FORCE_INLINE_ Vector3SIMD mul_sse(const Vector3SIMD &p_v) const {
        return Vector3SIMD(_mm_mul_ps(m_value, p_v.m_value));
    }

    _FORCE_INLINE_ Vector3SIMD div_sse(const Vector3SIMD &p_v) const {
        return Vector3SIMD(_mm_div_ps(m_value, p_v.m_value));
    }

    _FORCE_INLINE_ Vector3SIMD mul_scalar_sse(float p_scalar) const {
        return Vector3SIMD(_mm_mul_ps(m_value, load_scalar_sse(p_scalar)));
    }

    _FORCE_INLINE_ Vector3SIMD div_scalar_sse(float p_scalar) const {
        return Vector3SIMD(_mm_div_ps(m_value, load_scalar_sse(p_scalar)));
    }

    _FORCE_INLINE_ Vector3SIMD neg_sse() const {
        return Vector3SIMD(_mm_xor_ps(m_value, _mm_set1_ps(-0.0f)));
    }

    _FORCE_INLINE_ float length_squared_sse() const {
        return dot_sse(*this);
    }

    _FORCE_INLINE_ float length_sse() const {
        return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(length_squared_sse())));
    }

    _FORCE_INLINE_ Vector3SIMD normalize_sse() const {
        __m128 l = _mm_sqrt_ps(_mm_set1_ps(length_squared_sse()));
        return Vector3SIMD(_mm_div_ps(m_value, l));
    }

    _FORCE_INLINE_ Vector3SIMD min_sse(const Vector3SIMD &p_v) const {
        return Vector3SIMD(_mm_min_ps(m_value, p_v.m_value));
    }

    _FORCE_INLINE_ Vector3SIMD max_sse(const Vector3SIMD &p_v) const {
        return Vector3SIMD(_mm_max_ps(m_value, p_v.m_value));
    }

    _FORCE_INLINE_ Vector3SIMD abs_sse() const {
        return Vector3SIMD(_mm_andnot_ps(_mm_set1_ps(-0.0f), m_value));
    }

#if defined(__SSE4_1__)
    _FORCE_INLINE_ Vector3SIMD floor_sse() const {
        return Vector3SIMD(_mm_floor_ps(m_value));
    }

    _FORCE_INLINE_ Vector3SIMD ceil_sse() const {
        return Vector3SIMD(_mm_ceil_ps(m_value));
    }

    _FORCE_INLINE_ Vector3SIMD round_sse() const {
        return Vector3SIMD(_mm_round_ps(m_value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }
#endif
#endif // SSE implementations

    /*************************/
    /* NEON Implementations  */
    /*************************/
#if defined(VECTOR3SIMD_USE_NEON)
    static _FORCE_INLINE_ float32x4_t load_scalar_neon(float s) {
        return vdupq_n_f32(s);
    }

    _FORCE_INLINE_ Vector3SIMD cross_neon(const Vector3SIMD &p_v) const {
        float32x4_t a_yzx = vextq_f32(m_value, m_value, 1);
        float32x4_t b_yzx = vextq_f32(p_v.m_value, p_v.m_value, 1);
        float32x4_t temp = vsubq_f32(
            vmulq_f32(m_value, b_yzx),
            vmulq_f32(a_yzx, p_v.m_value));
        return Vector3SIMD(vextq_f32(temp, temp, 3));
    }

    _FORCE_INLINE_ float dot_neon(const Vector3SIMD &p_v) const {
        float32x4_t mul = vmulq_f32(m_value, p_v.m_value);
        float32x2_t sum = vpadd_f32(vget_low_f32(mul), vget_high_f32(mul));
        sum = vpadd_f32(sum, sum);
        return vget_lane_f32(sum, 0);
    }

    _FORCE_INLINE_ Vector3SIMD add_neon(const Vector3SIMD &p_v) const {
        return Vector3SIMD(vaddq_f32(m_value, p_v.m_value));
    }

    _FORCE_INLINE_ Vector3SIMD sub_neon(const Vector3SIMD &p_v) const {
        return Vector3SIMD(vsubq_f32(m_value, p_v.m_value));
    }

    _FORCE_INLINE_ Vector3SIMD mul_neon(const Vector3SIMD &p_v) const {
        return Vector3SIMD(vmulq_f32(m_value, p_v.m_value));
    }

    _FORCE_INLINE_ Vector3SIMD div_neon(const Vector3SIMD &p_v) const {
        return Vector3SIMD(vdivq_f32(m_value, p_v.m_value));
    }

    _FORCE_INLINE_ Vector3SIMD mul_scalar_neon(float p_scalar) const {
        return Vector3SIMD(vmulq_n_f32(m_value, p_scalar));
    }

    _FORCE_INLINE_ Vector3SIMD div_scalar_neon(float p_scalar) const {
        return Vector3SIMD(vdivq_f32(m_value, vdupq_n_f32(p_scalar)));
    }

    _FORCE_INLINE_ Vector3SIMD neg_neon() const {
        return Vector3SIMD(vnegq_f32(m_value));
    }

    _FORCE_INLINE_ float length_squared_neon() const {
        return dot_neon(*this);
    }

    _FORCE_INLINE_ float length_neon() const {
        float32x4_t v = vdupq_n_f32(length_squared_neon());
        float32x2_t s = vget_low_f32(v);
        s = vsqrt_f32(s);
        return vget_lane_f32(s, 0);
    }

    _FORCE_INLINE_ Vector3SIMD normalize_neon() const {
        float32x4_t l = vdupq_n_f32(length_neon());
        return Vector3SIMD(vdivq_f32(m_value, l));
    }

    _FORCE_INLINE_ Vector3SIMD min_neon(const Vector3SIMD &p_v) const {
        return Vector3SIMD(vminq_f32(m_value, p_v.m_value));
    }

    _FORCE_INLINE_ Vector3SIMD max_neon(const Vector3SIMD &p_v) const {
        return Vector3SIMD(vmaxq_f32(m_value, p_v.m_value));
    }

    _FORCE_INLINE_ Vector3SIMD abs_neon() const {
        return Vector3SIMD(vabsq_f32(m_value));
    }

#endif // NEON

    /***********************************/
    /* Advanced Vector Operations - SSE */
    /***********************************/
#if defined(VECTOR3SIMD_USE_SSE)
    _FORCE_INLINE_ Vector3SIMD reflect_sse(const Vector3SIMD &p_normal) const {
        __m128 dot = _mm_dp_ps(m_value, p_normal.m_value, 0x7F);
        __m128 double_proj = _mm_mul_ps(_mm_mul_ps(p_normal.m_value, dot), _mm_set1_ps(2.0f));
        return Vector3SIMD(_mm_sub_ps(m_value, double_proj));
    }

    _FORCE_INLINE_ Vector3SIMD slide_sse(const Vector3SIMD &p_normal) const {
        __m128 dot = _mm_dp_ps(m_value, p_normal.m_value, 0x7F);
        __m128 scaled = _mm_mul_ps(p_normal.m_value, dot);
        return Vector3SIMD(_mm_sub_ps(m_value, scaled));
    }

    _FORCE_INLINE_ Vector3SIMD bounce_sse(const Vector3SIMD &p_normal) const {
        __m128 dot = _mm_dp_ps(m_value, p_normal.m_value, 0x7F);
        __m128 double_proj = _mm_mul_ps(_mm_mul_ps(p_normal.m_value, dot), _mm_set1_ps(2.0f));
        return Vector3SIMD(_mm_sub_ps(double_proj, m_value));
    }

    _FORCE_INLINE_ Vector3SIMD project_sse(const Vector3SIMD &p_to) const {
        __m128 dot = _mm_dp_ps(m_value, p_to.m_value, 0x7F);
        __m128 len_sq = _mm_dp_ps(p_to.m_value, p_to.m_value, 0x7F);
        __m128 scale = _mm_div_ps(dot, len_sq);
        return Vector3SIMD(_mm_mul_ps(p_to.m_value, scale));
    }

    _FORCE_INLINE_ float distance_to_sse(const Vector3SIMD &p_to) const {
        return sub_sse(p_to).length_sse();
    }

    _FORCE_INLINE_ float distance_squared_to_sse(const Vector3SIMD &p_to) const {
        return sub_sse(p_to).length_squared_sse();
    }

    _FORCE_INLINE_ Vector3SIMD direction_to_sse(const Vector3SIMD &p_to) const {
        Vector3SIMD diff = sub_sse(p_to);
        return diff.normalize_sse();
    }

    _FORCE_INLINE_ Vector3SIMD rotated_sse(const Vector3SIMD &p_axis, float p_angle) const {
        __m128 cos_angle = _mm_set1_ps(cosf(p_angle));
        __m128 sin_angle = _mm_set1_ps(sinf(p_angle));
        
        __m128 dot = _mm_dp_ps(m_value, p_axis.m_value, 0x7F);
        __m128 cross = p_axis.cross_sse(*this).m_value;
        
        __m128 cos_term = _mm_mul_ps(m_value, cos_angle);
        __m128 sin_term = _mm_mul_ps(cross, sin_angle);
        __m128 dot_term = _mm_mul_ps(_mm_mul_ps(p_axis.m_value, dot), _mm_sub_ps(_mm_set1_ps(1.0f), cos_angle));
        
        return Vector3SIMD(_mm_add_ps(_mm_add_ps(cos_term, sin_term), dot_term));
    }

#endif

    /***********************************/
    /* Advanced Vector Operations - NEON */
    /***********************************/
#if defined(VECTOR3SIMD_USE_NEON)
    _FORCE_INLINE_ Vector3SIMD reflect_neon(const Vector3SIMD &p_normal) const {
        float d = dot_neon(p_normal);
        float32x4_t double_proj = vmulq_n_f32(p_normal.m_value, 2.0f * d);
        return Vector3SIMD(vsubq_f32(m_value, double_proj));
    }

    _FORCE_INLINE_ Vector3SIMD slide_neon(const Vector3SIMD &p_normal) const {
        float d = dot_neon(p_normal);
        float32x4_t scaled = vmulq_n_f32(p_normal.m_value, d);
        return Vector3SIMD(vsubq_f32(m_value, scaled));
    }

    _FORCE_INLINE_ Vector3SIMD bounce_neon(const Vector3SIMD &p_normal) const {
        float d = dot_neon(p_normal);
        float32x4_t double_proj = vmulq_n_f32(p_normal.m_value, 2.0f * d);
        return Vector3SIMD(vsubq_f32(double_proj, m_value));
    }

    _FORCE_INLINE_ Vector3SIMD project_neon(const Vector3SIMD &p_to) const {
        float d = dot_neon(p_to);
        float len_sq = p_to.length_squared_neon();
        float32x4_t scale = vdupq_n_f32(d / len_sq);
        return Vector3SIMD(vmulq_f32(p_to.m_value, scale));
    }

    _FORCE_INLINE_ float distance_to_neon(const Vector3SIMD &p_to) const {
        return sub_neon(p_to).length_neon();
    }

    _FORCE_INLINE_ float distance_squared_to_neon(const Vector3SIMD &p_to) const {
        return sub_neon(p_to).length_squared_neon();
    }

    _FORCE_INLINE_ Vector3SIMD direction_to_neon(const Vector3SIMD &p_to) const {
        Vector3SIMD diff = sub_neon(p_to);
        return diff.normalize_neon();
    }

    _FORCE_INLINE_ Vector3SIMD rotated_neon(const Vector3SIMD &p_axis, float p_angle) const {
        float cos_angle = cosf(p_angle);
        float sin_angle = sinf(p_angle);
        float d = dot_neon(p_axis);
        
        Vector3SIMD cross = p_axis.cross_neon(*this);
        
        float32x4_t cos_term = vmulq_n_f32(m_value, cos_angle);
        float32x4_t sin_term = vmulq_n_f32(cross.m_value, sin_angle);
        float32x4_t dot_term = vmulq_n_f32(p_axis.m_value, d * (1.0f - cos_angle));
        
        return Vector3SIMD(vaddq_f32(vaddq_f32(cos_term, sin_term), dot_term));
    }
#endif

        /***************************/
    /* Interpolation - SSE      */
    /***************************/
#if defined(VECTOR3SIMD_USE_SSE)
    _FORCE_INLINE_ Vector3SIMD lerp_sse(const Vector3SIMD& p_to, float p_weight) const {
        __m128 w = _mm_set1_ps(p_weight);
        __m128 inv_w = _mm_sub_ps(_mm_set1_ps(1.0f), w);
        return Vector3SIMD(_mm_add_ps(
            _mm_mul_ps(m_value, inv_w),
            _mm_mul_ps(p_to.m_value, w)
        ));
    }

    _FORCE_INLINE_ Vector3SIMD cubic_interpolate_sse(const Vector3SIMD& p_b, 
        const Vector3SIMD& p_pre_a, const Vector3SIMD& p_post_b, float p_weight) const {
        __m128 w = _mm_set1_ps(p_weight);
        __m128 w2 = _mm_mul_ps(w, w);
        __m128 w3 = _mm_mul_ps(w2, w);

        /* Coefficients from the Cubic Hermite spline formula:
         * p(t) = (2t³ - 3t² + 1)p₀ + (t³ - 2t² + t)m₀ + (-2t³ + 3t²)p₁ + (t³ - t²)m₁
         * where p₀ = this, p₁ = p_b, m₀ = (p_b - p_pre_a) * 0.5, m₁ = (p_post_b - this) * 0.5
         */
        __m128 coef_0 = _mm_add_ps(_mm_sub_ps(_mm_mul_ps(_mm_set1_ps(2.0f), w3), 
                                              _mm_mul_ps(_mm_set1_ps(3.0f), w2)), 
                                   _mm_set1_ps(1.0f));
        __m128 coef_1 = _mm_add_ps(_mm_sub_ps(w3, _mm_mul_ps(_mm_set1_ps(2.0f), w2)), w);
        __m128 coef_2 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-2.0f), w3), 
                                   _mm_mul_ps(_mm_set1_ps(3.0f), w2));
        __m128 coef_3 = _mm_sub_ps(w3, w2);

        __m128 m0 = _mm_mul_ps(_mm_sub_ps(p_b.m_value, p_pre_a.m_value), _mm_set1_ps(0.5f));
        __m128 m1 = _mm_mul_ps(_mm_sub_ps(p_post_b.m_value, m_value), _mm_set1_ps(0.5f));

        return Vector3SIMD(_mm_add_ps(_mm_add_ps(
            _mm_mul_ps(coef_0, m_value),
            _mm_mul_ps(coef_1, m0)),
            _mm_add_ps(
                _mm_mul_ps(coef_2, p_b.m_value),
                _mm_mul_ps(coef_3, m1)
            )));
    }

    _FORCE_INLINE_ Vector3SIMD move_toward_sse(const Vector3SIMD& p_to, float p_delta) const {
        __m128 diff = _mm_sub_ps(p_to.m_value, m_value);
        __m128 len = _mm_sqrt_ps(_mm_dp_ps(diff, diff, 0x7F));
        __m128 delta = _mm_set1_ps(p_delta);
        
        // If length <= delta, return target
        __m128 cmp = _mm_cmple_ps(len, delta);
        if(_mm_movemask_ps(cmp) & 0x1) {
            return p_to;
        }

        // Otherwise interpolate
        __m128 scale = _mm_div_ps(delta, len);
        return Vector3SIMD(_mm_add_ps(m_value, _mm_mul_ps(diff, scale)));
    }

    _FORCE_INLINE_ Vector3SIMD limit_length_sse(float p_len = 1.0f) const {
        float len = length_sse();
        if(len > 0.0f && p_len < len) {
            __m128 scale = _mm_set1_ps(p_len / len);
            return Vector3SIMD(_mm_mul_ps(m_value, scale));
        }
        return *this;
    }
#endif

    /***************************/
    /* Interpolation - NEON    */
    /***************************/
#if defined(VECTOR3SIMD_USE_NEON)
    _FORCE_INLINE_ Vector3SIMD lerp_neon(const Vector3SIMD& p_to, float p_weight) const {
        float32x4_t w = vdupq_n_f32(p_weight);
        float32x4_t inv_w = vsubq_f32(vdupq_n_f32(1.0f), w);
        return Vector3SIMD(vaddq_f32(
            vmulq_f32(m_value, inv_w),
            vmulq_f32(p_to.m_value, w)
        ));
    }

    _FORCE_INLINE_ Vector3SIMD cubic_interpolate_neon(const Vector3SIMD& p_b,
        const Vector3SIMD& p_pre_a, const Vector3SIMD& p_post_b, float p_weight) const {
        float32x4_t w = vdupq_n_f32(p_weight);
        float32x4_t w2 = vmulq_f32(w, w);
        float32x4_t w3 = vmulq_f32(w2, w);

        float32x4_t coef_0 = vaddq_f32(vsubq_f32(vmulq_n_f32(w3, 2.0f), 
                                                vmulq_n_f32(w2, 3.0f)), 
                                    vdupq_n_f32(1.0f));
        float32x4_t coef_1 = vaddq_f32(vsubq_f32(w3, vmulq_n_f32(w2, 2.0f)), w);
        float32x4_t coef_2 = vaddq_f32(vmulq_n_f32(w3, -2.0f), 
                                    vmulq_n_f32(w2, 3.0f));
        float32x4_t coef_3 = vsubq_f32(w3, w2);

        float32x4_t m0 = vmulq_n_f32(vsubq_f32(p_b.m_value, p_pre_a.m_value), 0.5f);
        float32x4_t m1 = vmulq_n_f32(vsubq_f32(p_post_b.m_value, m_value), 0.5f);

        return Vector3SIMD(vaddq_f32(vaddq_f32(
            vmulq_f32(coef_0, m_value),
            vmulq_f32(coef_1, m0)),
            vaddq_f32(
                vmulq_f32(coef_2, p_b.m_value),
                vmulq_f32(coef_3, m1)
            )));
    }

    _FORCE_INLINE_ Vector3SIMD move_toward_neon(const Vector3SIMD& p_to, float p_delta) const {
        float32x4_t diff = vsubq_f32(p_to.m_value, m_value);
        float len = sqrtf(dot_neon(Vector3SIMD(diff)));
        
        if(len <= p_delta) {
            return p_to;
        }

        float32x4_t scale = vdupq_n_f32(p_delta / len);
        return Vector3SIMD(vaddq_f32(m_value, vmulq_f32(diff, scale)));
    }

    _FORCE_INLINE_ Vector3SIMD limit_length_neon(float p_len = 1.0f) const {
        float len = length_neon();
        if(len > 0.0f && p_len < len) {
            float32x4_t scale = vdupq_n_f32(p_len / len);
            return Vector3SIMD(vmulq_f32(m_value, scale));
        }
        return *this;
    }
#endif

    /********************************/
    /* Component-wise Ops - SSE     */
    /********************************/
#if defined(VECTOR3SIMD_USE_SSE)
    _FORCE_INLINE_ Vector3SIMD posmod_sse(float p_mod) const {
        __m128 mod = _mm_set1_ps(p_mod);
        __m128 div = _mm_div_ps(m_value, mod);
        __m128 floor = _mm_floor_ps(div);
        return Vector3SIMD(_mm_sub_ps(m_value, _mm_mul_ps(floor, mod)));
    }

    _FORCE_INLINE_ Vector3SIMD posmodv_sse(const Vector3SIMD& p_modv) const {
        __m128 div = _mm_div_ps(m_value, p_modv.m_value);
        __m128 floor = _mm_floor_ps(div);
        return Vector3SIMD(_mm_sub_ps(m_value, _mm_mul_ps(floor, p_modv.m_value)));
    }

    _FORCE_INLINE_ bool is_equal_approx_sse(const Vector3SIMD& p_v) const {
        __m128 epsilon = _mm_set1_ps((real_t)CMP_EPSILON);
        __m128 diff = _mm_sub_ps(m_value, p_v.m_value);
        __m128 abs_diff = _mm_andnot_ps(_mm_set1_ps(-0.0f), diff);
        __m128 cmp = _mm_cmple_ps(abs_diff, epsilon);
        return (_mm_movemask_ps(cmp) & 0x7) == 0x7;
    }

    _FORCE_INLINE_ bool is_zero_approx_sse() const {
        __m128 epsilon = _mm_set1_ps((real_t)CMP_EPSILON);
        __m128 abs_val = _mm_andnot_ps(_mm_set1_ps(-0.0f), m_value);
        __m128 cmp = _mm_cmple_ps(abs_val, epsilon);
        return (_mm_movemask_ps(cmp) & 0x7) == 0x7;
    }

_FORCE_INLINE_ bool is_normalized_sse() const {
        __m128 len_sq = _mm_dp_ps(m_value, m_value, 0x7F);
        __m128 one = _mm_set1_ps(1.0f);
        __m128 epsilon = _mm_set1_ps((real_t)UNIT_EPSILON);
        __m128 diff = _mm_sub_ps(len_sq, one);
        __m128 abs_diff = _mm_andnot_ps(_mm_set1_ps(-0.0f), diff);
        __m128 cmp = _mm_cmple_ps(abs_diff, epsilon);
        return (_mm_movemask_ps(cmp) & 0x1) == 0x1;
    }

_FORCE_INLINE_ Vector3SIMD minf_sse(real_t p_scalar) const {
    __m128 scalar = load_scalar_sse(p_scalar);
    return Vector3SIMD(_mm_min_ps(m_value, scalar));
}

_FORCE_INLINE_ Vector3SIMD maxf_sse(real_t p_scalar) const {
    __m128 scalar = load_scalar_sse(p_scalar);
    return Vector3SIMD(_mm_max_ps(m_value, scalar));
}

_FORCE_INLINE_ Vector3SIMD slide_sse(const Vector3SIMD& p_normal) const {
    __m128 dot = _mm_dp_ps(m_value, p_normal.m_value, 0x7F);
    __m128 scaled = _mm_mul_ps(p_normal.m_value, dot);
    return Vector3SIMD(_mm_sub_ps(m_value, scaled));
}

_FORCE_INLINE_ Vector3SIMD slerp_sse(const Vector3SIMD& p_to, real_t p_weight) const {
    // Get lengths of input vectors
    real_t start_length_sq = length_squared_sse();
    real_t end_length_sq = p_to.length_squared_sse();

    if (unlikely(start_length_sq == 0.0f || end_length_sq == 0.0f)) {
        // Zero length vectors have no angle, so lerp instead
        return Vector3SIMD(_mm_add_ps(
            _mm_mul_ps(m_value, _mm_set1_ps(1.0f - p_weight)),
            _mm_mul_ps(p_to.m_value, _mm_set1_ps(p_weight))
        ));
    }

    // Get axis and angle
    __m128 cross = cross_sse(p_to).m_value;
    real_t axis_length_sq = _mm_cvtss_f32(_mm_dp_ps(cross, cross, 0x7F));

    if (unlikely(axis_length_sq == 0.0f)) {
        // Vectors are collinear, so lerp
        return Vector3SIMD(_mm_add_ps(
            _mm_mul_ps(m_value, _mm_set1_ps(1.0f - p_weight)),
            _mm_mul_ps(p_to.m_value, _mm_set1_ps(p_weight))
        ));
    }

    // Get angle between vectors using dot product
    real_t angle = angle_to_sse(p_to);
    real_t sin_angle = Math::sin(angle);

    // Calculate spherical interpolation coefficients
    real_t scale1 = Math::sin((1.0f - p_weight) * angle) / sin_angle;
    real_t scale2 = Math::sin(p_weight * angle) / sin_angle;

    __m128 scale1_ps = _mm_set1_ps(scale1);
    __m128 scale2_ps = _mm_set1_ps(scale2);

    return Vector3SIMD(_mm_add_ps(
        _mm_mul_ps(m_value, scale1_ps),
        _mm_mul_ps(p_to.m_value, scale2_ps)
    ));
}

_FORCE_INLINE_ real_t angle_to_sse(const Vector3SIMD& p_to) const {
    __m128 dot = _mm_dp_ps(m_value, p_to.m_value, 0x7F);
    __m128 length_prod = _mm_mul_ps(
        _mm_sqrt_ps(_mm_dp_ps(m_value, m_value, 0x7F)),
        _mm_sqrt_ps(_mm_dp_ps(p_to.m_value, p_to.m_value, 0x7F))
    );
    // Ensure the argument to acos is clamped between -1 and 1
    __m128 ratio = _mm_div_ps(dot, length_prod);
    ratio = _mm_max_ps(_mm_min_ps(ratio, _mm_set1_ps(1.0f)), _mm_set1_ps(-1.0f));
    return Math::acos(_mm_cvtss_f32(ratio));
}

_FORCE_INLINE_ real_t signed_angle_to_sse(const Vector3SIMD& p_to, const Vector3SIMD& p_axis) const {
    Vector3SIMD cross_product = cross_sse(p_to);
    real_t unsigned_angle = angle_to_sse(p_to);
    real_t sign = cross_product.dot_sse(p_axis);
    return (sign < 0) ? -unsigned_angle : unsigned_angle;
}

#endif // SSE implementations

    /********************************/
    /* Component-wise Ops - NEON    */
    /********************************/
#if defined(VECTOR3SIMD_USE_NEON)
    _FORCE_INLINE_ Vector3SIMD posmod_neon(float p_mod) const {
        float32x4_t mod = vdupq_n_f32(p_mod);
        float32x4_t div = vdivq_f32(m_value, mod);
        float32x4_t floor = vcvtq_f32_s32(vcvtq_s32_f32(div));
        return Vector3SIMD(vsubq_f32(m_value, vmulq_f32(floor, mod)));
    }

    _FORCE_INLINE_ Vector3SIMD posmodv_neon(const Vector3SIMD& p_modv) const {
        float32x4_t div = vdivq_f32(m_value, p_modv.m_value);
        float32x4_t floor = vcvtq_f32_s32(vcvtq_s32_f32(div));
        return Vector3SIMD(vsubq_f32(m_value, vmulq_f32(floor, p_modv.m_value)));
    }

    _FORCE_INLINE_ bool is_equal_approx_neon(const Vector3SIMD& p_v) const {
        float32x4_t epsilon = vdupq_n_f32((real_t)CMP_EPSILON);
        float32x4_t diff = vsubq_f32(m_value, p_v.m_value);
        float32x4_t abs_diff = vabsq_f32(diff);
        uint32x4_t cmp = vcleq_f32(abs_diff, epsilon);
        return (vgetq_lane_u32(cmp, 0) & vgetq_lane_u32(cmp, 1) & vgetq_lane_u32(cmp, 2)) != 0;
    }

    _FORCE_INLINE_ bool is_zero_approx_neon() const {
        float32x4_t epsilon = vdupq_n_f32((real_t)CMP_EPSILON);
        float32x4_t abs_val = vabsq_f32(m_value);
        uint32x4_t cmp = vcleq_f32(abs_val, epsilon);
        return (vgetq_lane_u32(cmp, 0) & vgetq_lane_u32(cmp, 1) & vgetq_lane_u32(cmp, 2)) != 0;
    }
#endif // NEON implementations

public:
    // Defined in vector3SIMD.cpp:
    Vector3SIMD(const Vector3 &p_v);
    operator Vector3() const;
}; // End of Vector3SIMD class

#endif // VECTOR3_SIMD_H