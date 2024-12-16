/**************************************************************************/
/*  vector3.h                                                             */
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

#ifndef VECTOR3_H
#define VECTOR3_H

#include "core/error/error_macros.h"
#include "core/math/math_funcs.h"
#include "core/string/ustring.h"
#include "vector3simd.h"

struct Basis;
struct Vector2;
struct Vector3i;

struct [[nodiscard]] Vector3 {
	static const int AXIS_COUNT = 3;

	enum Axis {
		AXIS_X,
		AXIS_Y,
		AXIS_Z,
	};

	union {
		struct {
			real_t x;
			real_t y;
			real_t z;
		};

		real_t coord[3] = { 0 };
	};

	_FORCE_INLINE_ const real_t &operator[](int p_axis) const {
		DEV_ASSERT((unsigned int)p_axis < 3);
		return coord[p_axis];
	}

	_FORCE_INLINE_ real_t &operator[](int p_axis) {
		DEV_ASSERT((unsigned int)p_axis < 3);
		return coord[p_axis];
	}

	_FORCE_INLINE_ Vector3::Axis min_axis_index() const {
		return x < y ? (x < z ? Vector3::AXIS_X : Vector3::AXIS_Z) : (y < z ? Vector3::AXIS_Y : Vector3::AXIS_Z);
	}

	_FORCE_INLINE_ Vector3::Axis max_axis_index() const {
		return x < y ? (y < z ? Vector3::AXIS_Z : Vector3::AXIS_Y) : (x < z ? Vector3::AXIS_Z : Vector3::AXIS_X);
	}

	Vector3 min(const Vector3 &p_vector3) const {
		return Vector3(MIN(x, p_vector3.x), MIN(y, p_vector3.y), MIN(z, p_vector3.z));
	}

	Vector3 minf(real_t p_scalar) const {
		return Vector3(MIN(x, p_scalar), MIN(y, p_scalar), MIN(z, p_scalar));
	}

	Vector3 max(const Vector3 &p_vector3) const {
		return Vector3(MAX(x, p_vector3.x), MAX(y, p_vector3.y), MAX(z, p_vector3.z));
	}

	Vector3 maxf(real_t p_scalar) const {
		return Vector3(MAX(x, p_scalar), MAX(y, p_scalar), MAX(z, p_scalar));
	}

	_FORCE_INLINE_ real_t length() const {
	#if defined(VECTOR3SIMD_USE_SSE) || defined(VECTOR3SIMD_USE_NEON)
		Vector3SIMD simd_this(*this);
		real_t simd_length = simd_this.length();
		if (std::isfinite(simd_length)) {
			return simd_length;
		}
	#endif
		return Math::sqrt(x * x + y * y + z * z);
	}

	_FORCE_INLINE_ real_t length_squared() const {
	#if defined(VECTOR3SIMD_USE_SSE) || defined(VECTOR3SIMD_USE_NEON)
		Vector3SIMD simd_this(*this);
		return simd_this.length_squared();
	#endif
		return x * x + y * y + z * z;
	}

	_FORCE_INLINE_ void normalize() {
	#if defined(VECTOR3SIMD_USE_SSE) || defined(VECTOR3SIMD_USE_NEON)
		Vector3SIMD simd_this(*this);
		simd_this.normalize();
		if (!simd_this.has_error()) {
			*this = simd_this;
			return;
		}
	#endif
		real_t lengthsq = x * x + y * y + z * z;
		if (lengthsq > 0) {
			real_t length = Math::sqrt(lengthsq);
			x /= length;
			y /= length;
			z /= length;
		} else {
			x = y = z = 0;
		}
	}

	_FORCE_INLINE_ Vector3 normalized() const {
		Vector3 v = *this;
		v.normalize();
		return v;
	}

	_FORCE_INLINE_ bool is_normalized() const {
		return Math::is_equal_approx((double)length_squared(), 1.0);
	}

	_FORCE_INLINE_ Vector3 inverse() const {
		return Vector3(1.0f / x, 1.0f / y, 1.0f / z);
	}

	_FORCE_INLINE_ real_t dot(const Vector3 &p_with) const {
	#if defined(VECTOR3SIMD_USE_SSE) || defined(VECTOR3SIMD_USE_NEON)
		Vector3SIMD simd_this(*this);
		Vector3SIMD simd_with(p_with);
		real_t simd_dot = simd_this.dot(simd_with);
		if (std::isfinite(simd_dot)) {
			return simd_dot;
		}
	#endif
		return x * p_with.x + y * p_with.y + z * p_with.z;
	}

	_FORCE_INLINE_ Vector3 cross(const Vector3 &p_with) const {
	#if defined(VECTOR3SIMD_USE_SSE) || defined(VECTOR3SIMD_USE_NEON)
		Vector3SIMD simd_this(*this);
		Vector3SIMD simd_with(p_with);
		Vector3SIMD simd_result = simd_this.cross(simd_with);
		if (!simd_result.has_error()) {
			return simd_result;
		}
	#endif
		return Vector3(
			(y * p_with.z) - (z * p_with.y),
			(z * p_with.x) - (x * p_with.z),
			(x * p_with.y) - (y * p_with.x)
		);
	}

	_FORCE_INLINE_ Vector3 operator+(const Vector3 &p_v) const {
	#if defined(VECTOR3SIMD_USE_SSE) || defined(VECTOR3SIMD_USE_NEON)
		Vector3SIMD simd_this(*this);
		Vector3SIMD simd_other(p_v);
		Vector3SIMD simd_result = simd_this + simd_other;
		if (!simd_result.has_error()) {
			return simd_result;
		}
	#endif
		return Vector3(x + p_v.x, y + p_v.y, z + p_v.z);
	}

	_FORCE_INLINE_ Vector3 operator-(const Vector3 &p_v) const {
	#if defined(VECTOR3SIMD_USE_SSE) || defined(VECTOR3SIMD_USE_NEON)
		Vector3SIMD simd_this(*this);
		Vector3SIMD simd_other(p_v);
		Vector3SIMD simd_result = simd_this - simd_other;
		if (!simd_result.has_error()) {
			return simd_result;
		}
	#endif
		return Vector3(x - p_v.x, y - p_v.y, z - p_v.z);
	}

	_FORCE_INLINE_ Vector3 operator*(real_t p_scalar) const {
	#if defined(VECTOR3SIMD_USE_SSE) || defined(VECTOR3SIMD_USE_NEON)
		Vector3SIMD simd_this(*this);
		Vector3SIMD simd_result = simd_this * p_scalar;
		if (!simd_result.has_error()) {
			return simd_result;
		}
	#endif
		return Vector3(x * p_scalar, y * p_scalar, z * p_scalar);
	}

	_FORCE_INLINE_ Vector3 &operator+=(const Vector3 &p_v) {
		*this = *this + p_v;
		return *this;
	}

	_FORCE_INLINE_ Vector3 &operator-=(const Vector3 &p_v) {
		*this = *this - p_v;
		return *this;
	}

	_FORCE_INLINE_ Vector3 &operator*=(real_t p_scalar) {
		*this = *this * p_scalar;
		return *this;
	}

	_FORCE_INLINE_ Vector3() {}
	_FORCE_INLINE_ Vector3(real_t p_x, real_t p_y, real_t p_z) : x(p_x), y(p_y), z(p_z) {}
	_FORCE_INLINE_ Vector3(const Vector3SIMD &simd) : x(simd.x()), y(simd.y()), z(simd.z()) {}

	operator Vector3SIMD() const {
		return Vector3SIMD(x, y, z);
	}
};

#endif // VECTOR3_H
