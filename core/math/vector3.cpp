/**************************************************************************/
/*  vector3.cpp                                                           */
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

#include "core/math/basis.h"
#include "core/math/math_funcs.h"
#include "core/math/vector2.h"
#include "core/math/vector3i.h"
#include "core/string/ustring.h"
#include "vector3.h"
#include "core/error/error_macros.h"

/******************************************************************************/
/* IMPORTANT NOTE ON ARITHMETIC OPERATIONS IN VECTOR3 FALLBACK IMPLEMENTATIONS
 */
/******************************************************************************/
/*                                                                            */
/* When implementing fallback methods for Vector3, we must avoid using        */
/* arithmetic operator shorthands (+, -, *, /) because:                       */
/*                                                                           */
/* 1. These operators are defined in vector3.h to attempt SIMD operations    */
/*    first before falling back to scalar operations                         */
/*                                                                           */
/* 2. If we use these operators within our fallback methods, it creates a    */
/*    circular dependency:                                                   */
/*    - Fallback method uses operator                                       */
/*    - Operator tries SIMD                                                 */
/*    - SIMD fails, calls fallback                                          */
/*    - Fallback uses operator...                                           */
/*                                                                           */
/* To avoid this, we must either:                                           */
/* a) Break vectors into their x,y,z components and operate on them         */
/*    directly (e.g., result.x = v1.x + v2.x)                              */
/* OR                                                                       */
/* b) Use the designated fallback methods:                                  */
/*    - multiply_scalar_fallback()   - For v *= scalar                     */
/*    - multiply_scalar_const_fallback() - For const v * scalar            */
/*    - multiply_vector_fallback()   - For v *= vector                     */
/*    - multiply_vector_const_fallback() - For const v * vector            */
/*    (And similarly for other operations)                                  */
/*                                                                           */
/* This ensures that fallback implementations remain independent of the      */
/* SIMD-first operator definitions in vector3.h                             */
/******************************************************************************/

/***************************************************/
/* Fallback methods for when SIMD is unavailable or */
/* fails, as referenced by vector3.h                */
/***************************************************/

real_t Vector3::dot_fallback(const Vector3 &p_with) const {
     return x * p_with.x + y * p_with.y + z * p_with.z;
}

Vector3 Vector3::cross_fallback(const Vector3 &p_with) const {
     return Vector3((y * p_with.z) - (z * p_with.y),
                    (z * p_with.x) - (x * p_with.z),
                    (x * p_with.y) - (y * p_with.x));
}

real_t Vector3::length_fallback() const {
     return Math::sqrt(x * x + y * y + z * z);
}

real_t Vector3::length_squared_fallback() const {
     return x * x + y * y + z * z;
}

Vector3 Vector3::min_fallback(const Vector3 &p_vec3) const {
     return Vector3(MIN(x, p_vec3.x), MIN(y, p_vec3.y), MIN(z, p_vec3.z));
}

Vector3 Vector3::max_fallback(const Vector3 &p_vec3) const {
     return Vector3(MAX(x, p_vec3.x), MAX(y, p_vec3.y), MAX(z, p_vec3.z));
}

Vector3 Vector3::minf_fallback(real_t p_scalar) const {
     return Vector3(MIN(x, p_scalar), MIN(y, p_scalar), MIN(z, p_scalar));
}

Vector3 Vector3::maxf_fallback(real_t p_scalar) const {
     return Vector3(MAX(x, p_scalar), MAX(y, p_scalar), MAX(z, p_scalar));
}

// Distance calculations
real_t Vector3::distance_to_fallback(const Vector3 &p_to) const {
     Vector3 diff;
     diff.x = p_to.x - x;
     diff.y = p_to.y - y;
     diff.z = p_to.z - z;
     return diff.length_fallback();
}

real_t Vector3::distance_squared_to_fallback(const Vector3 &p_to) const {
     Vector3 diff;
     diff.x = p_to.x - x;
     diff.y = p_to.y - y;
     diff.z = p_to.z - z;
     return diff.length_squared_fallback();
}

/***************************************************/
/* Original fallback (scalar-only) implementations  */
/***************************************************/

void Vector3::rotate(const Vector3 &p_axis, real_t p_angle) {
     *this = Basis(p_axis, p_angle).xform(*this);
}

Vector3 Vector3::rotated(const Vector3 &p_axis, real_t p_angle) const {
     Vector3 r = *this;
     r.rotate(p_axis, p_angle);
     return r;
}

Vector3 Vector3::clamp(const Vector3 &p_min, const Vector3 &p_max) const {
     return Vector3(CLAMP(x, p_min.x, p_max.x), CLAMP(y, p_min.y, p_max.y),
                    CLAMP(z, p_min.z, p_max.z));
}

Vector3 Vector3::clampf(real_t p_min, real_t p_max) const {
     return Vector3(CLAMP(x, p_min, p_max), CLAMP(y, p_min, p_max),
                    CLAMP(z, p_min, p_max));
}

void Vector3::snap(const Vector3 &p_step) {
     x = Math::snapped(x, p_step.x);
     y = Math::snapped(y, p_step.y);
     z = Math::snapped(z, p_step.z);
}

Vector3 Vector3::snapped(const Vector3 &p_step) const {
     Vector3 v = *this;
     v.snap(p_step);
     return v;
}

void Vector3::snapf(real_t p_step) {
     x = Math::snapped(x, p_step);
     y = Math::snapped(y, p_step);
     z = Math::snapped(z, p_step);
}

Vector3 Vector3::snappedf(real_t p_step) const {
     Vector3 v = *this;
     v.snapf(p_step);
     return v;
}

Vector3 Vector3::limit_length(real_t p_len) const {
     const real_t l = length_fallback();
     Vector3 v = *this;
     if (l > 0 && p_len < l) {
          v = v.divide_scalar_fallback(l);
          v = v.multiply_scalar_fallback(p_len);
     }
     return v;
}

Vector3 Vector3::move_toward(const Vector3 &p_to, real_t p_delta) const {
     Vector3 v = *this;
     Vector3 vd;
     vd.x = p_to.x - v.x;
     vd.y = p_to.y - v.y;
     vd.z = p_to.z - v.z;
     real_t len = vd.length_fallback();
     Vector3 scaled =
         vd.divide_scalar_fallback(len).multiply_scalar_fallback(p_delta);
     Vector3 result;
     result.x = v.x + scaled.x;
     result.y = v.y + scaled.y;
     result.z = v.z + scaled.z;
     return result;
}

Vector2 Vector3::octahedron_encode() const {
     Vector3 n = *this;
     real_t sum = Math::abs(n.x) + Math::abs(n.y) + Math::abs(n.z);
     n = n.divide_scalar_fallback(sum);
     Vector2 o;
     if (n.z >= 0.0f) {
          o.x = n.x;
          o.y = n.y;
     } else {
          o.x = (1.0f - Math::abs(n.y)) * (n.x >= 0.0f ? 1.0f : -1.0f);
          o.y = (1.0f - Math::abs(n.x)) * (n.y >= 0.0f ? 1.0f : -1.0f);
     }
     o.x = o.x * 0.5f + 0.5f;
     o.y = o.y * 0.5f + 0.5f;
     return o;
}

Vector3 Vector3::octahedron_decode(const Vector2 &p_oct) {
     Vector2 f(p_oct.x * 2.0f - 1.0f, p_oct.y * 2.0f - 1.0f);
     Vector3 n(f.x, f.y, 1.0f - Math::abs(f.x) - Math::abs(f.y));
     const real_t t = CLAMP(-n.z, 0.0f, 1.0f);
     n.x += n.x >= 0 ? -t : t;
     n.y += n.y >= 0 ? -t : t;
     return n.normalized();
}

Vector2 Vector3::octahedron_tangent_encode(float p_sign) const {
     const real_t bias = 1.0f / (real_t)32767.0f;
     Vector2 res = octahedron_encode();
     res.y = MAX(res.y, bias);
     res.y = res.y * 0.5f + 0.5f;
     res.y = p_sign >= 0.0f ? res.y : 1 - res.y;
     return res;
}

Vector3 Vector3::octahedron_tangent_decode(const Vector2 &p_oct,
                                           float *r_sign) {
     Vector2 oct_compressed = p_oct;
     oct_compressed.y = oct_compressed.y * 2 - 1;
     *r_sign = oct_compressed.y >= 0.0f ? 1.0f : -1.0f;
     oct_compressed.y = Math::abs(oct_compressed.y);
     Vector3 res = Vector3::octahedron_decode(oct_compressed);
     return res;
}

Basis Vector3::outer(const Vector3 &p_with) const {
     Basis basis;
     basis.rows[0] = Vector3(x * p_with.x, x * p_with.y, x * p_with.z);
     basis.rows[1] = Vector3(y * p_with.x, y * p_with.y, y * p_with.z);
     basis.rows[2] = Vector3(z * p_with.x, z * p_with.y, z * p_with.z);
     return basis;
}

Vector3 Vector3::posmod(real_t p_mod) const {
     return Vector3(Math::fposmod(x, p_mod), Math::fposmod(y, p_mod),
                    Math::fposmod(z, p_mod));
}

Vector3 Vector3::posmodv(const Vector3 &p_modv) const {
     return Vector3(Math::fposmod(x, p_modv.x), Math::fposmod(y, p_modv.y),
                    Math::fposmod(z, p_modv.z));
}

Vector3 &Vector3::operator*=(const Vector3 &p_v) {
     x *= p_v.x;
     y *= p_v.y;
     z *= p_v.z;
     return *this;
}

Vector3 Vector3::operator*(const Vector3 &p_v) const {
     Vector3 res = *this;
     res *= p_v;
     return res;
}

Vector3 &Vector3::operator*=(real_t p_scalar) {
     x *= p_scalar;
     y *= p_scalar;
     z *= p_scalar;
     return *this;
}

Vector3 Vector3::operator*(real_t p_scalar) const {
     return Vector3(x * p_scalar, y * p_scalar, z * p_scalar);
}

Vector3 &Vector3::operator/=(real_t p_scalar) {
     x /= p_scalar;
     y /= p_scalar;
     z /= p_scalar;
     return *this;
}

Vector3 Vector3::operator/(real_t p_scalar) const {
     Vector3 result;
     if (p_scalar == 0) {
          result = Vector3(0.0f, 0.0f, 0.0f);
          ERR_PRINT("Cannot divide by 0");
     } else {
          result = Vector3(x / p_scalar, y / p_scalar, z / p_scalar);
     }
     return result;
}

Vector3 Vector3::cross(const Vector3 &p_with) const {
     return Vector3((y * p_with.z) - (z * p_with.y),
                    (z * p_with.x) - (x * p_with.z),
                    (x * p_with.y) - (y * p_with.x));
}

void Vector3::normalize() {
     real_t lengthsq = length_squared_fallback();
     if (lengthsq == 0) {
          x = y = z = 0;
     } else {
          real_t length = Math::sqrt(lengthsq);
          x /= length;
          y /= length;
          z /= length;
     }
}

Vector3 Vector3::normalized() const {
     Vector3 v = *this;
     v.normalize();
     return v;
}

bool Vector3::is_normalized() const {
     return Math::is_equal_approx(length_squared_fallback(), (real_t)1.0);
}

Vector3 Vector3::abs() const {
     return Vector3(Math::abs(x), Math::abs(y), Math::abs(z));
}

Vector3 Vector3::sign() const { return Vector3(SIGN(x), SIGN(y), SIGN(z)); }

Vector3 Vector3::floor() const {
     return Vector3(Math::floor(x), Math::floor(y), Math::floor(z));
}

Vector3 Vector3::ceil() const {
     return Vector3(Math::ceil(x), Math::ceil(y), Math::ceil(z));
}

Vector3 Vector3::round() const {
     return Vector3(Math::round(x), Math::round(y), Math::round(z));
}

Vector3 Vector3::inverse() const {
     Vector3 result;
     if (x != 0.0f && y != 0.0f && z != 0.0f) {
          result = Vector3(1.0f / x, 1.0f / y, 1.0f / z);
     } else {
          return Vector3(0.0f, 0.0f, 0.0f);
          ERR_PRINT("Cannot divide by 0");
     }

     return result;
}

Vector3 Vector3::project_fallback(const Vector3 &p_to) const {
     Vector3 result;
     if (p_to.length_squared_fallback() < CMP_EPSILON) {
          result = Vector3(0.0f, 0.0f, 0.0f);
     } else {
          real_t scalar = dot_fallback(p_to) / p_to.length_squared_fallback();
          result = p_to.multiply_scalar_const_fallback(scalar);
     }

     return result;
}

real_t Vector3::angle_to_fallback(const Vector3 &p_to) const {
     Vector3 c = cross(p_to);
     return Math::atan2(c.length(), dot(p_to));
}

real_t Vector3::signed_angle_to_fallback(const Vector3 &p_to,
                                         const Vector3 &p_axis) const {
     Vector3 cross_to = cross(p_to);
     real_t unsigned_angle = Math::atan2(cross_to.length(), dot(p_to));
     real_t sign = cross_to.dot(p_axis);
     return (sign < 0) ? -unsigned_angle : unsigned_angle;
}

Vector3 Vector3::direction_to_fallback(const Vector3 &p_to) const {
     Vector3 ret(p_to.x - x, p_to.y - y, p_to.z - z);
     Vector3 result = ret;

     if (result.length_squared_fallback() < CMP_EPSILON) {
          return Vector3(0.0f, 0.0f, 0.0f);
     } else {
          result.normalize();
     }

     return result;
}

// slide returns the component of the vector along the given plane, specified by
// its normal vector.
Vector3 Vector3::slide_fallback(const Vector3 &p_normal) const {
#ifdef MATH_CHECKS
     ERR_FAIL_COND_V_MSG(!p_normal.is_normalized(), Vector3(),
                         "The normal Vector3 " + p_normal.operator String() +
                             " must be normalized.");
     // perhaps we could just normalize the vector for them if they do not have
     // it normalized already we could just output a message about how the
     // operation would proceed faster if they already had it normalized
     // (important if the operation is called many times with the same normal)
#endif
     Vector3 scaled =
         p_normal.multiply_scalar_const_fallback(dot_fallback(p_normal));
     Vector3 result;
     result.x = x - scaled.x;
     result.y = y - scaled.y;
     result.z = z - scaled.z;
     return result;
}

Vector3 Vector3::bounce(const Vector3 &p_normal) const {
     Vector3 reflected = reflect_fallback(p_normal);
     return Vector3(-reflected.x, -reflected.y, -reflected.z);
}

Vector3 Vector3::reflect_fallback(const Vector3 &p_normal) const {
#ifdef MATH_CHECKS
     ERR_FAIL_COND_V_MSG(!p_normal.is_normalized(), Vector3(),
                         "The normal Vector3 " + p_normal.operator String() +
                             " must be normalized.");
     // perhaps we could just normalize the vector for them if they do not have
     // it normalized already we could just output a message about how the
     // operation would proceed faster if they already had it normalized
     // (important if the operation is called many times with the same normal)
#endif
     real_t dot_result = dot_fallback(p_normal);
     real_t scaled_dot =
         dot_result + dot_result;  // Avoid using * to prevent operator overload
     Vector3 scaled = p_normal.multiply_scalar_const_fallback(scaled_dot);
     Vector3 result;
     result.x = scaled.x - x;
     result.y = scaled.y - y;
     result.z = scaled.z - z;
     return result;
}

Vector3 Vector3::lerp(const Vector3 &p_to, real_t p_weight) const {
     Vector3 res = *this;
     res.x = Math::lerp(res.x, p_to.x, p_weight);
     res.y = Math::lerp(res.y, p_to.y, p_weight);
     res.z = Math::lerp(res.z, p_to.z, p_weight);
     return res;
}  // I'm not sure if SIMD could be used here... but maybe?

Vector3 Vector3::slerp_fallback(const Vector3 &p_to, real_t p_weight) const {
     // This method seems more complicated than it really is, since we write out
     // the internals of some methods for efficiency (mainly, checking length).
     real_t start_length_sq = length_squared_fallback();
     real_t end_length_sq = p_to.length_squared_fallback();
     if (unlikely(start_length_sq == 0.0f || end_length_sq == 0.0f)) {
          // Zero length vectors have no angle, so the best we can do is either
          // lerp or throw an error.
          return lerp(p_to, p_weight);
     }
     Vector3 axis = cross(p_to);
     real_t axis_length_sq = axis.length_squared_fallback();
     if (unlikely(axis_length_sq == 0.0f)) {
          // Colinear vectors have no rotation axis or angle between them, so
          // the best we can do is lerp.
          return lerp(p_to, p_weight);
     }
     axis /= Math::sqrt(axis_length_sq);
     real_t start_length = Math::sqrt(start_length_sq);
     real_t result_length =
         Math::lerp(start_length, Math::sqrt(end_length_sq), p_weight);
     real_t angle = angle_to(p_to);
     return rotated(axis, angle * p_weight) * (result_length / start_length);
}  // maybe this could use SIMD and be put in vector3SIMD.h

Vector3 Vector3::cubic_interpolate(const Vector3 &p_b, const Vector3 &p_pre_a,
                                   const Vector3 &p_post_b,
                                   real_t p_weight) const {
     Vector3 res = *this;
     res.x =
         Math::cubic_interpolate(res.x, p_b.x, p_pre_a.x, p_post_b.x, p_weight);
     res.y =
         Math::cubic_interpolate(res.y, p_b.y, p_pre_a.y, p_post_b.y, p_weight);
     res.z =
         Math::cubic_interpolate(res.z, p_b.z, p_pre_a.z, p_post_b.z, p_weight);
     return res;
}

Vector3 Vector3::cubic_interpolate_in_time(
    const Vector3 &p_b, const Vector3 &p_pre_a, const Vector3 &p_post_b,
    real_t p_weight, real_t p_b_t, real_t p_pre_a_t, real_t p_post_b_t) const {
     Vector3 res = *this;
     res.x = Math::cubic_interpolate_in_time(res.x, p_b.x, p_pre_a.x,
                                             p_post_b.x, p_weight, p_b_t,
                                             p_pre_a_t, p_post_b_t);
     res.y = Math::cubic_interpolate_in_time(res.y, p_b.y, p_pre_a.y,
                                             p_post_b.y, p_weight, p_b_t,
                                             p_pre_a_t, p_post_b_t);
     res.z = Math::cubic_interpolate_in_time(res.z, p_b.z, p_pre_a.z,
                                             p_post_b.z, p_weight, p_b_t,
                                             p_pre_a_t, p_post_b_t);
     return res;
}

Vector3 Vector3::bezier_interpolate(const Vector3 &p_control_1,
                                    const Vector3 &p_control_2,
                                    const Vector3 &p_end, real_t p_t) const {
     Vector3 res = *this;
     res.x = Math::bezier_interpolate(res.x, p_control_1.x, p_control_2.x,
                                      p_end.x, p_t);
     res.y = Math::bezier_interpolate(res.y, p_control_1.y, p_control_2.y,
                                      p_end.y, p_t);
     res.z = Math::bezier_interpolate(res.z, p_control_1.z, p_control_2.z,
                                      p_end.z, p_t);
     return res;
}

Vector3 Vector3::bezier_derivative(const Vector3 &p_control_1,
                                   const Vector3 &p_control_2,
                                   const Vector3 &p_end, real_t p_t) const {
     Vector3 res = *this;
     res.x = Math::bezier_derivative(res.x, p_control_1.x, p_control_2.x,
                                     p_end.x, p_t);
     res.y = Math::bezier_derivative(res.y, p_control_1.y, p_control_2.y,
                                     p_end.y, p_t);
     res.z = Math::bezier_derivative(res.z, p_control_1.z, p_control_2.z,
                                     p_end.z, p_t);
     return res;
}

bool Vector3::operator==(const Vector3 &p_v) const {
     return x == p_v.x && y == p_v.y && z == p_v.z;
}

bool Vector3::operator!=(const Vector3 &p_v) const { return !(*this == p_v); }

bool Vector3::operator<(const Vector3 &p_v) const {
     if (x == p_v.x) {
          if (y == p_v.y) {
               return z < p_v.z;
          }
          return y < p_v.y;
     }
     return x < p_v.x;
}

bool Vector3::operator<=(const Vector3 &p_v) const {
     return *this < p_v || *this == p_v;
}

bool Vector3::operator>(const Vector3 &p_v) const { return !(*this <= p_v); }

bool Vector3::operator>=(const Vector3 &p_v) const {
     return *this > p_v || *this == p_v;
}

bool Vector3::is_equal_approx(const Vector3 &p_v) const {
     return Math::is_equal_approx(x, p_v.x) &&
            Math::is_equal_approx(y, p_v.y) && Math::is_equal_approx(z, p_v.z);
}

bool Vector3::is_zero_approx() const {
     return Math::is_zero_approx(x) && Math::is_zero_approx(y) &&
            Math::is_zero_approx(z);
}

bool Vector3::is_finite() const {
     return Math::is_finite(x) && Math::is_finite(y) && Math::is_finite(z);
}

Vector3 &Vector3::multiply_vector_fallback(const Vector3 &p_v) {
     x *= p_v.x;
     y *= p_v.y;
     z *= p_v.z;
     return *this;
}

Vector3 &Vector3::multiply_scalar_fallback(real_t p_scalar) {
     x *= p_scalar;
     y *= p_scalar;
     z *= p_scalar;
     return *this;
}

Vector3 &Vector3::divide_vector_fallback(const Vector3 &p_v) {

     if (Math::is_zero_approx(p_v.x) || Math::is_zero_approx(p_v.y) ||
         Math::is_zero_approx(p_v.z)) {
          ERR_PRINT("Divide by zero error.");
          return *this;
     } //Only continue if none of the components of the divisor are zero
     
     x /= p_v.x;
     y /= p_v.y;
     z /= p_v.z;
     return *this;
}

Vector3 &Vector3::divide_scalar_fallback(real_t p_scalar) {
     x /= p_scalar;
     y /= p_scalar;
     z /= p_scalar;
     return *this;
}

Vector3 Vector3::multiply_vector_const_fallback(const Vector3 &p_v) const {
     return Vector3(x * p_v.x, y * p_v.y, z * p_v.z);
}

Vector3 Vector3::multiply_scalar_const_fallback(real_t p_scalar) const {
     return Vector3(x * p_scalar, y * p_scalar, z * p_scalar);
}

Vector3::operator String() const {
     return "(" + String::num_real(x, true) + ", " + String::num_real(y, true) +
            ", " + String::num_real(z, true) + ")";
}

Vector3::operator Vector3i() const { return Vector3i(x, y, z); }