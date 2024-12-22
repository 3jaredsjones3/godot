/**************************************************************************/
/*  quaternion.h                                                          */
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

#ifndef QUATERNION_H
#define QUATERNION_H

#include "core/math/vector3.h"
#include "core/math/vector4.h"
#include "core/math/math_funcs.h"
#include "core/math/math_defs.h"
#include "core/string/ustring.h"

struct Vector3;
struct Basis; 

struct [[nodiscard]] Quaternion {
    union {
        struct {
            real_t x;
            real_t y;
            real_t z;
            real_t w;
        };
        Vector4 components;

    };
	
	operator String() const;
    // Unique default constructor
    _FORCE_INLINE_ Quaternion() : x(0), y(0), z(0), w(1.0f) {}

    // Constructor with individual components
    _FORCE_INLINE_ Quaternion(real_t p_x, real_t p_y, real_t p_z, real_t p_w)
        : x(p_x), y(p_y), z(p_z), w(p_w) {}

    // Constructor from Vector4
    explicit _FORCE_INLINE_ Quaternion(const Vector4 &vec)
        : x(vec.x), y(vec.y), z(vec.z), w(vec.w) {}

    // Access individual components via index
    _FORCE_INLINE_ real_t &operator[](int p_idx) {
        return components[p_idx];
    }

    _FORCE_INLINE_ const real_t &operator[](int p_idx) const {
        return components[p_idx];
    }

	_FORCE_INLINE_ real_t length_squared() const;
	bool is_equal_approx(const struct Quaternion &p_quaternion) const;
	bool is_finite() const;
	real_t length() const;
	void normalize();
	struct Quaternion normalized() const;
	bool is_normalized() const;
	struct Quaternion inverse() const;
	struct Quaternion log() const;
	struct Quaternion exp() const;
	_FORCE_INLINE_ real_t dot(const Quaternion &p_q) const;
	real_t angle_to(const struct Quaternion &p_to) const;

	Vector3 get_euler(EulerOrder p_order = EulerOrder::YXZ) const;
	static struct Quaternion from_euler(const Vector3 &p_euler);

	struct Quaternion slerp(const struct Quaternion &p_to, real_t p_weight) const;
	struct Quaternion slerpni(const struct Quaternion &p_to, real_t p_weight) const;
	struct Quaternion spherical_cubic_interpolate(const struct Quaternion &p_b, const struct Quaternion &p_pre_a, const struct Quaternion &p_post_b, real_t p_weight) const;
	struct Quaternion spherical_cubic_interpolate_in_time(const struct Quaternion &p_b, const struct  Quaternion &p_pre_a, const struct Quaternion &p_post_b, real_t p_weight, real_t p_b_t, real_t p_pre_a_t, real_t p_post_b_t) const;

	Vector3 get_axis() const;
	real_t get_angle() const;

	_FORCE_INLINE_ void get_axis_angle(Vector3 &r_axis, real_t &r_angle) const {
		r_angle = 2 * Math::acos(w);
		real_t r = ((real_t)1) / Math::sqrt(1 - w * w);
		r_axis.x = x * r;
		r_axis.y = y * r;
		r_axis.z = z * r;
	} //need to optimize to use vector3's * operator to get simd benefits

	void operator*=(const struct Quaternion &p_q);
	struct Quaternion operator*(const struct Quaternion &p_q) const;

	_FORCE_INLINE_ Vector3 xform(const Vector3 &p_v) const {
#ifdef MATH_CHECKS
		ERR_FAIL_COND_V_MSG(!is_normalized(), p_v, "The quaternion " + operator String() + " must be normalized.");
#endif
		Vector3 u(x, y, z);
		Vector3 uv = u.cross(p_v);
		return p_v + ((uv * w) + u.cross(uv)) * ((real_t)2);
	}

	_FORCE_INLINE_ Vector3 xform_inv(const Vector3 &p_v) const {
		return inverse().xform(p_v);
	}

	_FORCE_INLINE_ void operator+=(const struct Quaternion &p_q);
	_FORCE_INLINE_ void operator-=(const struct Quaternion &p_q);
	_FORCE_INLINE_ void operator*=(real_t p_s);
	_FORCE_INLINE_ void operator/=(real_t p_s);
	_FORCE_INLINE_ Quaternion operator+(const struct Quaternion &p_q2) const;
	_FORCE_INLINE_ Quaternion operator-(const struct Quaternion &p_q2) const;
	_FORCE_INLINE_ Quaternion operator-() const;
	_FORCE_INLINE_ Quaternion operator*(real_t p_s) const;
	_FORCE_INLINE_ Quaternion operator/(real_t p_s) const;

	_FORCE_INLINE_ bool operator==(const struct Quaternion &p_quaternion) const;
	_FORCE_INLINE_ bool operator!=(const struct Quaternion &p_quaternion) const;

	operator String() const;

	_FORCE_INLINE_ Quaternion() {}

	_FORCE_INLINE_ Quaternion(real_t p_x, real_t p_y, real_t p_z, real_t p_w) :
			x(p_x),
			y(p_y),
			z(p_z),
			w(p_w) {
	}

	Quaternion(const Vector3 &p_axis, real_t p_angle);

	Quaternion(const Quaternion &p_q) :
			x(p_q.x),
			y(p_q.y),
			z(p_q.z),
			w(p_q.w) {
	}

	void operator=(const Quaternion &p_q) {
		x = p_q.x;
		y = p_q.y;
		z = p_q.z;
		w = p_q.w;
	}

	Quaternion(const Vector3 &p_v0, const Vector3 &p_v1) { // Shortest arc.
		Vector3 c = p_v0.cross(p_v1);
		real_t d = p_v0.dot(p_v1);

		if (d < -1.0f + (real_t)CMP_EPSILON) {
			x = 0;
			y = 1;
			z = 0;
			w = 0;
		} else {
			real_t s = Math::sqrt((1.0f + d) * 2.0f);
			real_t rs = 1.0f / s;

			x = c.x * rs;
			y = c.y * rs;
			z = c.z * rs;
			w = s * 0.5f;
		}
	}
};

real_t Quaternion::dot(const struct Quaternion &p_q) const {
    return components.dot(p_q.components);
}

real_t Quaternion::length_squared() const {
	return dot(*this);
}

void Quaternion::operator+=(const struct Quaternion &p_q) {
    components += p_q.components;
}

void Quaternion::operator-=(const struct Quaternion &p_q) {
    components -= p_q.components;
}

void Quaternion::operator*=(real_t p_s) {
    components *= p_s;
}

void Quaternion::operator/=(real_t p_s) {
    components /= p_s;
}

struct Quaternion Quaternion::operator+(const struct Quaternion &p_q2) const {
    return Quaternion(components + p_q2.components);
}

struct Quaternion Quaternion::operator-(const struct Quaternion &p_q2) const {
    return Quaternion(components - p_q2.components);
}

struct Quaternion Quaternion::operator-() const {
    return Quaternion(-components);
}

struct Quaternion Quaternion::operator*(real_t p_s) const {
    return Quaternion(components * p_s);
}

struct Quaternion Quaternion::operator/(real_t p_s) const {
    return Quaternion(components / p_s);
}

bool Quaternion::operator==(const struct Quaternion &p_quaternion) const {
    return components == p_quaternion.components;
}

bool Quaternion::operator!=(const struct Quaternion &p_quaternion) const {
    return components != p_quaternion.components;
}

_FORCE_INLINE_ Quaternion operator*(real_t p_real, const struct Quaternion &p_quaternion) {
    return Quaternion(p_quaternion.components * p_real);
}

#endif // QUATERNION_H
