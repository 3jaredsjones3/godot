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

     // Member data
     union {
          struct {
               real_t x;
               real_t y;
               real_t z;
          };
          real_t coord[3] = {0};
     };

     // Constructors
     _FORCE_INLINE_ Vector3() {}
     _FORCE_INLINE_ Vector3(real_t p_x, real_t p_y, real_t p_z)
         : x(p_x), y(p_y), z(p_z) {}

     // SIMD conversion constructors and operators
     _FORCE_INLINE_ Vector3(const Vector3SIMD& p_simd) {
          x = p_simd.x();
          y = p_simd.y();
          z = p_simd.z();
     }

     _FORCE_INLINE_ operator Vector3SIMD() const {
          return Vector3SIMD(x, y, z);
     }

     // Static methods
     static const Vector3& get_zero_vector() {
          static const Vector3 zero_vector(0.0f, 0.0f, 0.0f);
          return zero_vector;
     }

     // Basic operations
     _FORCE_INLINE_ void zero() {
          x = 0.0f;
          y = 0.0f;
          z = 0.0f;
     }

     // Array access
     _FORCE_INLINE_ const real_t& operator[](int p_axis) const {
          DEV_ASSERT((unsigned int)p_axis < 3);
          return coord[p_axis];
     }

     _FORCE_INLINE_ real_t& operator[](int p_axis) {
          DEV_ASSERT((unsigned int)p_axis < 3);
          return coord[p_axis];
     }

     // Axis methods
     _FORCE_INLINE_ Axis min_axis_index() const {
          return x < y ? (x < z ? AXIS_X : AXIS_Z) : (y < z ? AXIS_Y : AXIS_Z);
     }

     _FORCE_INLINE_ Axis max_axis_index() const {
          return x < y ? (y < z ? AXIS_Z : AXIS_Y) : (x < z ? AXIS_Z : AXIS_X);
     }
     // SIMD-enabled methods with fallback

     _FORCE_INLINE_ Vector3& operator*=(const Vector3& p_v) {
#if defined(VECTOR3SIMD_USE_SSE)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_with(p_v);
          Vector3SIMD simd_result = simd_this.mul_sse(simd_with);
          *this = Vector3(simd_result);
          return *this;
#elif defined(VECTOR3SIMD_USE_NEON)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_with(p_v);
          Vector3SIMD simd_result = simd_this.mul_neon(simd_with);
          *this = Vector3(simd_result);
          return *this;
#else
          return multiply_vector_fallback(p_v);
#endif
     }

     _FORCE_INLINE_ Vector3& operator*=(real_t p_scalar) {
#if defined(VECTOR3SIMD_USE_SSE)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_result = simd_this.mul_scalar_sse(p_scalar);
          *this = Vector3(simd_result);
          return *this;
#elif defined(VECTOR3SIMD_USE_NEON)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_result = simd_this.mul_scalar_neon(p_scalar);
          *this = Vector3(simd_result);
          return *this;
#else
          return multiply_scalar_fallback(p_scalar);
#endif
     }

     _FORCE_INLINE_ Vector3& operator/=(const Vector3& p_v) {
#if defined(VECTOR3SIMD_USE_SSE)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_with(p_v);
          Vector3SIMD simd_result = simd_this.div_sse(simd_with);
          *this = Vector3(simd_result);
          return *this;
#elif defined(VECTOR3SIMD_USE_NEON)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_with(p_v);
          Vector3SIMD simd_result = simd_this.div_neon(simd_with);
          *this = Vector3(simd_result);
          return *this;
#else
          return divide_vector_fallback(p_v);
#endif
     }

     _FORCE_INLINE_ Vector3& operator/=(real_t p_scalar) {
#if defined(VECTOR3SIMD_USE_SSE)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_result = simd_this.div_scalar_sse(p_scalar);
          *this = Vector3(simd_result);
          return *this;
#elif defined(VECTOR3SIMD_USE_NEON)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_result = simd_this.div_scalar_neon(p_scalar);
          *this = Vector3(simd_result);
          return *this;
#else
          return divide_scalar_fallback(p_scalar);
#endif
     }

     // Non-modifying operators
     _FORCE_INLINE_ Vector3 operator*(const Vector3& p_v) const {
#if defined(VECTOR3SIMD_USE_SSE)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_with(p_v);
          Vector3SIMD simd_result = simd_this.mul_sse(simd_with);
          return Vector3(simd_result);
#elif defined(VECTOR3SIMD_USE_NEON)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_with(p_v);
          Vector3SIMD simd_result = simd_this.mul_neon(simd_with);
          return Vector3(simd_result);
#else
          return multiply_vector_const_fallback(p_v);
#endif
     }

     _FORCE_INLINE_ Vector3 operator*(real_t p_scalar) const {
#if defined(VECTOR3SIMD_USE_SSE)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_result = simd_this.mul_scalar_sse(p_scalar);
          return Vector3(simd_result);
#elif defined(VECTOR3SIMD_USE_NEON)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_result = simd_this.mul_scalar_neon(p_scalar);
          return Vector3(simd_result);
#else
          return multiply_scalar_const_fallback(p_scalar);
#endif
     }

     _FORCE_INLINE_ Vector3 min(const Vector3& p_vector3) const {
#if defined(VECTOR3SIMD_USE_SSE)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_with(p_vector3);
          Vector3SIMD simd_result = simd_this.min_sse(simd_with);
          return Vector3(simd_result);
#elif defined(VECTOR3SIMD_USE_NEON)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_with(p_vector3);
          Vector3SIMD simd_result = simd_this.min_neon(simd_with);
          return Vector3(simd_result);
#else
          return Vector3(MIN(x, p_vector3.x), MIN(y, p_vector3.y),
                         MIN(z, p_vector3.z));
#endif
     }

     _FORCE_INLINE_ Vector3 max(const Vector3& p_vector3) const {
#if defined(VECTOR3SIMD_USE_SSE)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_with(p_vector3);
          Vector3SIMD simd_result = simd_this.max_sse(simd_with);
          return Vector3(simd_result);
#elif defined(VECTOR3SIMD_USE_NEON)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_with(p_vector3);
          Vector3SIMD simd_result = simd_this.max_neon(simd_with);
          return Vector3(simd_result);
#else
          return Vector3(MAX(x, p_vector3.x), MAX(y, p_vector3.y),
                         MAX(z, p_vector3.z));
#endif
     }

     _FORCE_INLINE_ Vector3 minf(real_t p_scalar) const {
#if defined(VECTOR3SIMD_USE_SSE)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_result = simd_this.minf_sse(p_scalar);
          return Vector3(simd_result);
#elif defined(VECTOR3SIMD_USE_NEON)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_result = simd_this.minf_neon(p_scalar);
          return Vector3(simd_result);
#else
          return Vector3(MIN(x, p_scalar), MIN(y, p_scalar), MIN(z, p_scalar));
#endif
     }

     _FORCE_INLINE_ Vector3 maxf(real_t p_scalar) const {
#if defined(VECTOR3SIMD_USE_SSE)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_result = simd_this.maxf_sse(p_scalar);
          return Vector3(simd_result);
#elif defined(VECTOR3SIMD_USE_NEON)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_result = simd_this.maxf_neon(p_scalar);
          return Vector3(simd_result);
#else
          return Vector3(MAX(x, p_scalar), MAX(y, p_scalar), MAX(z, p_scalar));
#endif
     }

     _FORCE_INLINE_ real_t dot(const Vector3& p_with) const {
#if defined(VECTOR3SIMD_USE_SSE)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_with(p_with);
          return simd_this.dot_sse(simd_with);
#elif defined(VECTOR3SIMD_USE_NEON)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_with(p_with);
          return simd_this.dot_neon(simd_with);
#else
          return dot_fallback(p_with);
#endif
     }

     _FORCE_INLINE_ Vector3 cross(const Vector3& p_with) const {
#if defined(VECTOR3SIMD_USE_SSE)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_with(p_with);
          Vector3SIMD simd_result = simd_this.cross_sse(simd_with);
          return Vector3(simd_result);
#elif defined(VECTOR3SIMD_USE_NEON)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_with(p_with);
          Vector3SIMD simd_result = simd_this.cross_neon(simd_with);
          return Vector3(simd_result);
#else
          return cross_fallback(p_with);
#endif
     }

     _FORCE_INLINE_ real_t length() const {
#if defined(VECTOR3SIMD_USE_SSE)
          Vector3SIMD simd_this(*this);
          return simd_this.length_sse();
#elif defined(VECTOR3SIMD_USE_NEON)
          Vector3SIMD simd_this(*this);
          return simd_this.length_neon();
#else
          return length_fallback();
#endif
     }

     _FORCE_INLINE_ real_t length_squared() const {
#if defined(VECTOR3SIMD_USE_SSE)
          Vector3SIMD simd_this(*this);
          return simd_this.length_squared_sse();
#elif defined(VECTOR3SIMD_USE_NEON)
          Vector3SIMD simd_this(*this);
          return simd_this.length_squared_neon();
#else
          return length_squared_fallback();
#endif
     }

     _FORCE_INLINE_ void normalize() {
#if defined(VECTOR3SIMD_USE_SSE)
          Vector3SIMD simd_this(*this);
          *this = Vector3(simd_this.normalize_sse());
#elif defined(VECTOR3SIMD_USE_NEON)
          Vector3SIMD simd_this(*this);
          *this = Vector3(simd_this.normalize_neon());
#else
          real_t l = length();
          if (l == 0) {
               x = y = z = 0;
          } else {
               x /= l;
               y /= l;
               z /= l;
          }
#endif
     }

     _FORCE_INLINE_ Vector3 normalized() const {
#if defined(VECTOR3SIMD_USE_SSE)
          Vector3SIMD simd_this(*this);
          return Vector3(simd_this.normalize_sse());
#elif defined(VECTOR3SIMD_USE_NEON)
          Vector3SIMD simd_this(*this);
          return Vector3(simd_this.normalize_neon());
#else
          Vector3 v = *this;
          v.normalize();
          return v;
#endif
     }

     _FORCE_INLINE_ bool is_normalized() const {
#if defined(VECTOR3SIMD_USE_SSE)
          Vector3SIMD simd_this(*this);
          return simd_this.is_normalized_sse();
#elif defined(VECTOR3SIMD_USE_NEON)
          Vector3SIMD simd_this(*this);
          return simd_this.is_normalized_neon();
#else
          return Math::is_equal_approx(length_squared(), 1,
                                       (real_t)UNIT_EPSILON);
#endif
     }

     _FORCE_INLINE_ Vector3 slerp(const Vector3& p_to, real_t p_weight) const {
#if defined(VECTOR3SIMD_USE_SSE)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_to(p_to);
          return Vector3(simd_this.slerp_sse(simd_to, p_weight));
#elif defined(VECTOR3SIMD_USE_NEON)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_to(p_to);
          return Vector3(simd_this.slerp_neon(simd_to, p_weight));
#else
          return slerp_fallback(p_to, p_weight);
#endif
     }

     _FORCE_INLINE_ Vector3 reflect(const Vector3& p_normal) const {
#ifdef MATH_CHECKS
          ERR_FAIL_COND_V_MSG(!p_normal.is_normalized(), Vector3(),
                              "The normal Vector3 must be normalized.");
#endif
#if defined(VECTOR3SIMD_USE_SSE)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_normal(p_normal);
          return Vector3(simd_this.reflect_sse(simd_normal));
#elif defined(VECTOR3SIMD_USE_NEON)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_normal(p_normal);
          return Vector3(simd_this.reflect_neon(simd_normal));
#else
          return reflect_fallback(p_normal);
#endif
     }

     _FORCE_INLINE_ Vector3 slide(const Vector3& p_normal) const {
#ifdef MATH_CHECKS
          ERR_FAIL_COND_V_MSG(!p_normal.is_normalized(), Vector3(),
                              "The normal Vector3 must be normalized.");
#endif
#if defined(VECTOR3SIMD_USE_SSE)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_normal(p_normal);
          return Vector3(simd_this.slide_sse(simd_normal));
#elif defined(VECTOR3SIMD_USE_NEON)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_normal(p_normal);
          return Vector3(simd_this.slide_neon(simd_normal));
#else
          return slide_fallback(p_normal);
#endif
     }

     _FORCE_INLINE_ Vector3 direction_to(const Vector3& p_to) const {
#if defined(VECTOR3SIMD_USE_SSE)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_to(p_to);
          return Vector3(simd_this.direction_to_sse(simd_to));
#elif defined(VECTOR3SIMD_USE_NEON)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_to(p_to);
          return Vector3(simd_this.direction_to_neon(simd_to));
#else
          return direction_to_fallback(p_to);
#endif
     }

     _FORCE_INLINE_ real_t signed_angle_to(const Vector3& p_to,
                                           const Vector3& p_axis) const {
#if defined(VECTOR3SIMD_USE_SSE)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_to(p_to);
          Vector3SIMD simd_axis(p_axis);
          return simd_this.signed_angle_to_sse(simd_to, simd_axis);
#elif defined(VECTOR3SIMD_USE_NEON)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_to(p_to);
          Vector3SIMD simd_axis(p_axis);
          return simd_this.signed_angle_to_neon(simd_to, simd_axis);
#else
          return signed_angle_to_fallback(p_to, p_axis);
#endif
     }

     _FORCE_INLINE_ real_t angle_to(const Vector3& p_to) const {
#if defined(VECTOR3SIMD_USE_SSE)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_to(p_to);
          return simd_this.angle_to_sse(simd_to);
#elif defined(VECTOR3SIMD_USE_NEON)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_to(p_to);
          return simd_this.angle_to_neon(simd_to);
#else
          return angle_to_fallback(p_to);
#endif
     }

     _FORCE_INLINE_ Vector3 project(const Vector3& p_to) const {
#if defined(VECTOR3SIMD_USE_SSE)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_to(p_to);
          return Vector3(simd_this.project_sse(simd_to));
#elif defined(VECTOR3SIMD_USE_NEON)
          Vector3SIMD simd_this(*this);
          Vector3SIMD simd_to(p_to);
          return Vector3(simd_this.project_neon(simd_to));
#else
          return project_fallback(p_to);
#endif
     }

     // Additional operations with fallback
     Vector3 inverse() const { return Vector3(1.0f / x, 1.0f / y, 1.0f / z); }
     void rotate(const Vector3& p_axis, real_t p_angle);
     Vector3 rotated(const Vector3& p_axis, real_t p_angle) const;
     Vector3 limit_length(real_t p_len = 1.0) const;
     Vector3 move_toward(const Vector3& p_to, real_t p_delta) const;
     Basis outer(const Vector3& p_with) const;
     Vector3 clamp(const Vector3& p_min, const Vector3& p_max) const;
     Vector3 clampf(real_t p_min, real_t p_max) const;
     void snap(const Vector3& p_step);
     Vector3 snapped(const Vector3& p_step) const;
     void snapf(real_t p_step);
     Vector3 snappedf(real_t p_step) const;

     // Interpolation methods
     Vector3 lerp(const Vector3& p_to, real_t p_weight) const;
     Vector3 slerp(const Vector3& p_to, real_t p_weight) const;
     Vector3 cubic_interpolate(const Vector3& p_b, const Vector3& p_pre_a,
                               const Vector3& p_post_b, real_t p_weight) const;
     Vector3 cubic_interpolate_in_time(const Vector3& p_b,
                                       const Vector3& p_pre_a,
                                       const Vector3& p_post_b, real_t p_weight,
                                       real_t p_b_t, real_t p_pre_a_t,
                                       real_t p_post_b_t) const;
     Vector3 bezier_interpolate(const Vector3& p_control_1,
                                const Vector3& p_control_2,
                                const Vector3& p_end, real_t p_t) const;
     Vector3 bezier_derivative(const Vector3& p_control_1,
                               const Vector3& p_control_2, const Vector3& p_end,
                               real_t p_t) const;

     // Distance methods
     real_t distance_to_fallback(const Vector3& p_to) const;
     real_t distance_squared_to_fallback(const Vector3& p_to) const;

     // Modulo operations
     Vector3 posmod(real_t p_mod) const;
     Vector3 posmodv(const Vector3& p_modv) const;
     Vector3 project(const Vector3& p_to) const;

     // Angle calculations
     real_t angle_to(const Vector3& p_to) const;
     real_t signed_angle_to(const Vector3& p_to, const Vector3& p_axis) const;
     Vector3 direction_to(const Vector3& p_to) const;

     // Reflection operations
     Vector3 slide(const Vector3& p_normal) const;
     Vector3 bounce(const Vector3& p_normal) const;
     Vector3 reflect(const Vector3& p_normal) const;

     // Operators
     Vector3& operator*=(const Vector3& p_v);
     Vector3 operator*(const Vector3& p_v) const;
     Vector3& operator/=(real_t p_scalar);
     Vector3 operator/(real_t p_scalar) const;

     // Component-wise operations
     Vector3 abs() const;
     Vector3 sign() const;
     Vector3 floor() const;
     Vector3 ceil() const;
     Vector3 round() const;

     // State checks
     bool is_equal_approx(const Vector3& p_v) const;
     bool is_zero_approx() const;
     bool is_finite() const;

    public:
     // Octahedron encoding/decoding methods
     Vector2 octahedron_encode() const;
     static Vector3 octahedron_decode(const Vector2& p_oct);
     Vector2 octahedron_tangent_encode(float p_sign) const;
     static Vector3 octahedron_tangent_decode(const Vector2& p_oct,
                                              float* r_sign);

     // Comparison operators
     inline bool operator==(const Vector3& p_v) const {
          return x == p_v.x && y == p_v.y && z == p_v.z;
     }

     inline bool operator!=(const Vector3& p_v) const {
          return x != p_v.x || y != p_v.y || z != p_v.z;
     }

     inline bool operator<(const Vector3& p_v) const {
          if (x == p_v.x) {
               if (y == p_v.y) {
                    return z < p_v.z;
               }
               return y < p_v.y;
          }
          return x < p_v.x;
     }

     inline bool operator<=(const Vector3& p_v) const {
          if (x == p_v.x) {
               if (y == p_v.y) {
                    return z <= p_v.z;
               }
               return y < p_v.y;
          }
          return x < p_v.x;
     }

     inline bool operator>(const Vector3& p_v) const {
          if (x == p_v.x) {
               if (y == p_v.y) {
                    return z > p_v.z;
               }
               return y > p_v.y;
          }
          return x > p_v.x;
     }

     inline bool operator>=(const Vector3& p_v) const {
          if (x == p_v.x) {
               if (y == p_v.y) {
                    return z >= p_v.z;
               }
               return y > p_v.y;
          }
          return x > p_v.x;
     }

     // Vector * Vector multiplication
     Vector3& operator*=(const Vector3& p_v);
     Vector3 operator*(const Vector3& p_v) const;

     // Vector * scalar multiplication
     Vector3& operator*=(real_t p_scalar);
     Vector3 operator*(real_t p_scalar) const;

     // Conversion operators
     operator String() const;    // Declares conversion to String
     operator Vector3i() const;  // Declares conversion to Vector3i

     // Binary subtraction (Vector3 - Vector3)
     _FORCE_INLINE_ Vector3 operator-(const Vector3& p_v) const {
          return Vector3(x - p_v.x, y - p_v.y, z - p_v.z);
     }

     _FORCE_INLINE_ Vector3& operator-=(const Vector3& p_v) {
          x -= p_v.x;
          y -= p_v.y;
          z -= p_v.z;
          return *this;
     }

     // Unary negation (-Vector3)
     _FORCE_INLINE_ Vector3 operator-() const { return Vector3(-x, -y, -z); }

     // Binary Addition (Vector3 - Vector3)
     _FORCE_INLINE_ Vector3 operator+(const Vector3& p_v) const {
          return Vector3(x + p_v.x, y + p_v.y, z + p_v.z);
     }

     _FORCE_INLINE_ Vector3& operator+=(const Vector3& p_v) {
          x += p_v.x;
          y += p_v.y;
          z += p_v.z;
          return *this;
     }

    private:
     // Fallback declarations (to be implemented in vector3.cpp)
     Vector3 slerp_fallback(const Vector3& p_to, real_t p_weight) const;
     Vector3 reflect_fallback(const Vector3& p_normal) const;
     Vector3 slide_fallback(const Vector3& p_normal) const;
     Vector3 direction_to_fallback(const Vector3& p_to) const;
     real_t signed_angle_to_fallback(const Vector3& p_to,
                                     const Vector3& p_axis) const;
     real_t angle_to_fallback(const Vector3& p_to) const;
     Vector3 project_fallback(const Vector3& p_to) const;

     Vector3 min_fallback(const Vector3& p_vec3) const;
     Vector3 max_fallback(const Vector3& p_vec3) const;
     Vector3 minf_fallback(real_t p_scalar) const;
     Vector3 maxf_fallback(real_t p_scalar) const;

     real_t dot_fallback(const Vector3& p_with) const;
     Vector3 cross_fallback(const Vector3& p_with) const;
     real_t length_fallback() const;
     real_t length_squared_fallback() const;

     Vector3& multiply_vector_fallback(const Vector3& p_v);
     Vector3& multiply_scalar_fallback(real_t p_scalar);
     Vector3& divide_vector_fallback(const Vector3& p_v);
     Vector3& divide_scalar_fallback(real_t p_scalar);
     Vector3 multiply_vector_const_fallback(const Vector3& p_v) const;
     Vector3 multiply_scalar_const_fallback(real_t p_scalar) const;
};

#endif  // VECTOR3_H
