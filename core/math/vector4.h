/**************************************************************************/
/*  vector4.h                                                             */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2007-2023 Godot Engine contributors */
/* See AUTHORS.md for details.                                            */
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

#ifndef VECTOR4_H
#define VECTOR4_H

#include "core/error/error_macros.h"
#include "core/math/math_funcs.h"
#include "core/math/math_defs.h"
#include "core/typedefs.h"
#include "core/string/ustring.h"
#include "core/math/vector4i.h"
#include <xmmintrin.h>
#include <smmintrin.h> // SSE4.1 for floor, ceil, round
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
		__m128 m_value;
		struct {
			real_t x, y, z, w;
		};
		real_t coord[4];
	};

	_FORCE_INLINE_ Vector4() : m_value(_mm_setzero_ps()) {}
	_FORCE_INLINE_ Vector4(real_t p_x, real_t p_y, real_t p_z, real_t p_w) {
		m_value = _mm_set_ps(p_w, p_z, p_y, p_x);
	}

	_FORCE_INLINE_ real_t &operator[](int p_axis) {
		DEV_ASSERT((unsigned int)p_axis < 4);
		return coord[p_axis];
	}
	_FORCE_INLINE_ const real_t &operator[](int p_axis) const {
		DEV_ASSERT((unsigned int)p_axis < 4);
		return coord[p_axis];
	}

	// Utility intrinsics
	static _FORCE_INLINE_ __m128 load_scalar(real_t s) {
		return _mm_set1_ps(s);
	}

	// Basic arithmetic
	_FORCE_INLINE_ Vector4 operator+(const Vector4 &p_vec4) const {
		return Vector4(_mm_add_ps(m_value, p_vec4.m_value));
	}
	_FORCE_INLINE_ Vector4 operator-(const Vector4 &p_vec4) const {
		return Vector4(_mm_sub_ps(m_value, p_vec4.m_value));
	}
	_FORCE_INLINE_ Vector4 operator*(const Vector4 &p_vec4) const {
		return Vector4(_mm_mul_ps(m_value, p_vec4.m_value));
	}
	_FORCE_INLINE_ Vector4 operator/(const Vector4 &p_vec4) const {
		return Vector4(_mm_div_ps(m_value, p_vec4.m_value));
	}

	_FORCE_INLINE_ void operator+=(const Vector4 &p_vec4) {
		m_value = _mm_add_ps(m_value, p_vec4.m_value);
	}
	_FORCE_INLINE_ void operator-=(const Vector4 &p_vec4) {
		m_value = _mm_sub_ps(m_value, p_vec4.m_value);
	}
	_FORCE_INLINE_ void operator*=(const Vector4 &p_vec4) {
		m_value = _mm_mul_ps(m_value, p_vec4.m_value);
	}
	_FORCE_INLINE_ void operator/=(const Vector4 &p_vec4) {
		m_value = _mm_div_ps(m_value, p_vec4.m_value);
	}

	_FORCE_INLINE_ Vector4 operator*(real_t p_s) const {
		__m128 s = load_scalar(p_s);
		return Vector4(_mm_mul_ps(m_value, s));
	}
	_FORCE_INLINE_ Vector4 operator/(real_t p_s) const {
		__m128 s = load_scalar(p_s);
		return Vector4(_mm_div_ps(m_value, s));
	}
	_FORCE_INLINE_ void operator*=(real_t p_s) {
		__m128 s = load_scalar(p_s);
		m_value = _mm_mul_ps(m_value, s);
	}
	_FORCE_INLINE_ void operator/=(real_t p_s) {
		__m128 s = load_scalar(p_s);
		m_value = _mm_div_ps(m_value, s);
	}

	_FORCE_INLINE_ Vector4 operator-() const {
		const __m128 sign_mask = _mm_set1_ps(-0.0f);
		return Vector4(_mm_xor_ps(m_value, sign_mask));
	}

	_FORCE_INLINE_ bool operator==(const Vector4 &p_vec4) const {
		__m128 cmp = _mm_cmpeq_ps(m_value, p_vec4.m_value);
		return (_mm_movemask_ps(cmp) == 0xF);
	}
	_FORCE_INLINE_ bool operator!=(const Vector4 &p_vec4) const {
		return !(*this == p_vec4);
	}

	// Lexicographic comparisons
	_FORCE_INLINE_ bool operator<(const Vector4 &p_v) const {
		if (x == p_v.x) {
			if (y == p_v.y) {
				if (z == p_v.z) {
					return w < p_v.w;
				}
				return z < p_v.z;
			}
			return y < p_v.y;
		}
		return x < p_v.x;
	}

	_FORCE_INLINE_ bool operator>(const Vector4 &p_v) const {
		if (x == p_v.x) {
			if (y == p_v.y) {
				if (z == p_v.z) {
					return w > p_v.w;
				}
				return z > p_v.z;
			}
			return y > p_v.y;
		}
		return x > p_v.x;
	}

	_FORCE_INLINE_ bool operator<=(const Vector4 &p_v) const {
		return (*this < p_v) || (*this == p_v);
	}

	_FORCE_INLINE_ bool operator>=(const Vector4 &p_v) const {
		return (*this > p_v) || (*this == p_v);
	}

	// Dot product
	_FORCE_INLINE_ real_t dot(const Vector4 &p_vec4) const {
		__m128 dp = _mm_mul_ps(m_value, p_vec4.m_value);
		__m128 shuf = _mm_movehdup_ps(dp);
		__m128 sums = _mm_add_ps(dp, shuf);
		shuf = _mm_movehl_ps(shuf, sums);
		sums = _mm_add_ss(sums, shuf);
		return _mm_cvtss_f32(sums);
	}

	_FORCE_INLINE_ real_t length_squared() const {
		return dot(*this);
	}

	_FORCE_INLINE_ Axis min_axis_index() const {
		uint32_t min_index = 0;
		real_t min_value = x;
		for (uint32_t i = 1; i < 4; i++) {
			if (operator[](i) <= min_value) {
				min_index = i;
				min_value = operator[](i);
			}
		}
		return Axis(min_index);
	}

	_FORCE_INLINE_ Axis max_axis_index() const {
		uint32_t max_index = 0;
		real_t max_value = x;
		for (uint32_t i = 1; i < 4; i++) {
			if (operator[](i) > max_value) {
				max_index = i;
				max_value = operator[](i);
			}
		}
		return Axis(max_index);
	}

	// Additions start here
	_FORCE_INLINE_ Vector4 min(const Vector4 &p_vec4) const {
		return Vector4(_mm_min_ps(m_value, p_vec4.m_value));
	}

	_FORCE_INLINE_ Vector4 minf(real_t p_val) const {
		__m128 scalar = load_scalar(p_val);
		return Vector4(_mm_min_ps(m_value, scalar));
	}

	_FORCE_INLINE_ Vector4 max(const Vector4 &p_vec4) const {
		return Vector4(_mm_max_ps(m_value, p_vec4.m_value));
	}

	_FORCE_INLINE_ Vector4 maxf(real_t p_val) const {
		__m128 scalar = load_scalar(p_val);
		return Vector4(_mm_max_ps(m_value, scalar));
	}

	_FORCE_INLINE_ bool is_equal_approx(const Vector4 &p_vec4) const {
		return Math::is_equal_approx(x, p_vec4.x) &&
			   Math::is_equal_approx(y, p_vec4.y) &&
			   Math::is_equal_approx(z, p_vec4.z) &&
			   Math::is_equal_approx(w, p_vec4.w);
	}

	_FORCE_INLINE_ bool is_zero_approx() const {
		return Math::is_zero_approx(x) &&
			   Math::is_zero_approx(y) &&
			   Math::is_zero_approx(z) &&
			   Math::is_zero_approx(w);
	}

	_FORCE_INLINE_ bool is_finite() const {
		return Math::is_finite(x) && Math::is_finite(y) &&
			   Math::is_finite(z) && Math::is_finite(w);
	}

	_FORCE_INLINE_ real_t length() const {
		return Math::sqrt(length_squared());
	}

	_FORCE_INLINE_ void normalize() {
		real_t lengthsq = length_squared();
		if (lengthsq == 0) {
			x = y = z = w = 0;
		} else {
			real_t length = Math::sqrt(lengthsq);
			*this /= length;
		}
	}

	_FORCE_INLINE_ Vector4 normalized() const {
		Vector4 v = *this;
		v.normalize();
		return v;
	}

	_FORCE_INLINE_ bool is_normalized() const {
		return Math::is_equal_approx(length_squared(), (real_t)1, (real_t)UNIT_EPSILON);
	}

	_FORCE_INLINE_ real_t distance_to(const Vector4 &p_to) const {
		return (p_to - *this).length();
	}

	_FORCE_INLINE_ real_t distance_squared_to(const Vector4 &p_to) const {
		return (p_to - *this).length_squared();
	}

	_FORCE_INLINE_ Vector4 direction_to(const Vector4 &p_to) const {
		Vector4 ret(p_to.x - x, p_to.y - y, p_to.z - z, p_to.w - w);
		ret.normalize();
		return ret;
	}

	_FORCE_INLINE_ Vector4 abs() const {
		const __m128 sign_mask = _mm_set1_ps(-0.0f);
		return Vector4(_mm_andnot_ps(sign_mask, m_value));
	}

	_FORCE_INLINE_ Vector4 sign() const {
		__m128 zero = _mm_setzero_ps();
		__m128 cmp_gt = _mm_cmpgt_ps(m_value, zero);
		__m128 cmp_lt = _mm_cmplt_ps(m_value, zero);
		__m128 pos_ones = _mm_and_ps(cmp_gt, _mm_set1_ps(1.0f));
		__m128 neg_ones = _mm_and_ps(cmp_lt, _mm_set1_ps(-1.0f));
		return Vector4(_mm_or_ps(pos_ones, neg_ones));
	}

	_FORCE_INLINE_ Vector4 floor() const {
		return Vector4(_mm_floor_ps(m_value));
	}

	_FORCE_INLINE_ Vector4 ceil() const {
		return Vector4(_mm_ceil_ps(m_value));
	}

	_FORCE_INLINE_ Vector4 round() const {
		return Vector4(_mm_round_ps(m_value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
	}

	_FORCE_INLINE_ Vector4 lerp(const Vector4 &p_to, real_t p_weight) const {
		return Vector4(Math::lerp(x, p_to.x, p_weight),
					   Math::lerp(y, p_to.y, p_weight),
					   Math::lerp(z, p_to.z, p_weight),
					   Math::lerp(w, p_to.w, p_weight));
	}

	_FORCE_INLINE_ Vector4 cubic_interpolate(const Vector4 &p_b, const Vector4 &p_pre_a, const Vector4 &p_post_b, real_t p_weight) const {
		return Vector4(
			Math::cubic_interpolate(x, p_b.x, p_pre_a.x, p_post_b.x, p_weight),
			Math::cubic_interpolate(y, p_b.y, p_pre_a.y, p_post_b.y, p_weight),
			Math::cubic_interpolate(z, p_b.z, p_pre_a.z, p_post_b.z, p_weight),
			Math::cubic_interpolate(w, p_b.w, p_pre_a.w, p_post_b.w, p_weight)
		);
	}

	_FORCE_INLINE_ Vector4 cubic_interpolate_in_time(const Vector4 &p_b, const Vector4 &p_pre_a, const Vector4 &p_post_b, real_t p_weight, real_t p_b_t, real_t p_pre_a_t, real_t p_post_b_t) const {
		return Vector4(
			Math::cubic_interpolate_in_time(x, p_b.x, p_pre_a.x, p_post_b.x, p_weight, p_b_t, p_pre_a_t, p_post_b_t),
			Math::cubic_interpolate_in_time(y, p_b.y, p_pre_a.y, p_post_b.y, p_weight, p_b_t, p_pre_a_t, p_post_b_t),
			Math::cubic_interpolate_in_time(z, p_b.z, p_pre_a.z, p_post_b.z, p_weight, p_b_t, p_pre_a_t, p_post_b_t),
			Math::cubic_interpolate_in_time(w, p_b.w, p_pre_a.w, p_post_b.w, p_weight, p_b_t, p_pre_a_t, p_post_b_t)
		);
	}

	_FORCE_INLINE_ Vector4 posmod(real_t p_mod) const {
		return Vector4(Math::fposmod(x, p_mod),
					   Math::fposmod(y, p_mod),
					   Math::fposmod(z, p_mod),
					   Math::fposmod(w, p_mod));
	}

	_FORCE_INLINE_ Vector4 posmodv(const Vector4 &p_modv) const {
		return Vector4(Math::fposmod(x, p_modv.x),
					   Math::fposmod(y, p_modv.y),
					   Math::fposmod(z, p_modv.z),
					   Math::fposmod(w, p_modv.w));
	}

	_FORCE_INLINE_ void snap(const Vector4 &p_step) {
		x = Math::snapped(x, p_step.x);
		y = Math::snapped(y, p_step.y);
		z = Math::snapped(z, p_step.z);
		w = Math::snapped(w, p_step.w);
	}

	_FORCE_INLINE_ void snapf(real_t p_step) {
		x = Math::snapped(x, p_step);
		y = Math::snapped(y, p_step);
		z = Math::snapped(z, p_step);
		w = Math::snapped(w, p_step);
	}

	_FORCE_INLINE_ Vector4 snapped(const Vector4 &p_step) const {
		Vector4 v = *this;
		v.snap(p_step);
		return v;
	}

	_FORCE_INLINE_ Vector4 snappedf(real_t p_step) const {
		Vector4 v = *this;
		v.snapf(p_step);
		return v;
	}

	_FORCE_INLINE_ Vector4 inverse() const {
		__m128 ones = _mm_set1_ps(1.0f);
		return Vector4(_mm_div_ps(ones, m_value));
	}

	_FORCE_INLINE_ Vector4 clamp(const Vector4 &p_min, const Vector4 &p_max) const {
    	return Vector4(_mm_min_ps(_mm_max_ps(m_value, p_min.m_value), p_max.m_value));
	}

	_FORCE_INLINE_ Vector4 clampf(real_t p_min, real_t p_max) const {
    	__m128 min_val = _mm_set1_ps(p_min);
    	__m128 max_val = _mm_set1_ps(p_max);
    	return Vector4(_mm_min_ps(_mm_max_ps(m_value, min_val), max_val));
	}

	operator String() const {
		return "(" + String::num_real(x, true) + ", " +
			   String::num_real(y, true) + ", " +
			   String::num_real(z, true) + ", " +
			   String::num_real(w, true) + ")";
	}

	operator Vector4i() const {
		return Vector4i(x, y, z, w);
	}

	_FORCE_INLINE_ Vector4(__m128 val) {
		m_value = val;
	}
};

_FORCE_INLINE_ Vector4 operator*(float p_scalar, const Vector4 &p_vec) {
	return p_vec * p_scalar;
}

_FORCE_INLINE_ Vector4 operator*(double p_scalar, const Vector4 &p_vec) {
	return p_vec * (real_t)p_scalar;
}

_FORCE_INLINE_ Vector4 operator*(int32_t p_scalar, const Vector4 &p_vec) {
	return p_vec * (real_t)p_scalar;
}

_FORCE_INLINE_ Vector4 operator*(int64_t p_scalar, const Vector4 &p_vec) {
	return p_vec * (real_t)p_scalar;
}

static_assert(sizeof(Vector4) == 4 * sizeof(real_t));

#endif // VECTOR4_H
