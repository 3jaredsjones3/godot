#ifndef VECTOR3_H
#define VECTOR3_H

#include "core/error/error_macros.h"
#include "core/math/math_defs.h"
#include "core/math/math_funcs.h"
#include "core/string/ustring.h"
#include "core/typedefs.h"

// Forward declarations
struct Basis;
struct Vector2;
struct Vector3i;
struct Vector3SIMD;

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
        real_t coord[3];
    };

    // Static constants
    static const Vector3 ZERO;
    static const Vector3 ONE;
    static const Vector3 LEFT;
    static const Vector3 RIGHT;
    static const Vector3 UP;
    static const Vector3 DOWN;
    static const Vector3 FORWARD;
    static const Vector3 BACK;

    /**************************************************************************/
    /* Constructors and basic methods */
    /**************************************************************************/
    _FORCE_INLINE_ Vector3() : x(0), y(0), z(0) {}
    _FORCE_INLINE_ Vector3(real_t p_x, real_t p_y, real_t p_z) : x(p_x), y(p_y), z(p_z) {}

    _FORCE_INLINE_ static Vector3 get_zero_vector() {
        return Vector3(0, 0, 0);
    }

    /**************************************************************************/
    /* Array access */
    /**************************************************************************/
    _FORCE_INLINE_ const real_t& operator[](int p_axis) const {
        DEV_ASSERT((unsigned int)p_axis < 3);
        return coord[p_axis];
    }

    _FORCE_INLINE_ real_t& operator[](int p_axis) {
        DEV_ASSERT((unsigned int)p_axis < 3);
        return coord[p_axis];
    }

    /**************************************************************************/
    /* Axis methods */
    /**************************************************************************/
    _FORCE_INLINE_ Vector3::Axis min_axis_index() const {
        return x < y ? (x < z ? AXIS_X : AXIS_Z) : (y < z ? AXIS_Y : AXIS_Z);
    }

    _FORCE_INLINE_ Vector3::Axis max_axis_index() const {
        return x < y ? (y < z ? AXIS_Z : AXIS_Y) : (x < z ? AXIS_Z : AXIS_X);
    }

    void zero() { x = y = z = 0; }

    /**************************************************************************/
    /* Core vector operations */
    /**************************************************************************/
    
    _FORCE_INLINE_ static Vector3 vec3_cross(const Vector3& a, const Vector3& b) {
        return a.cross(b);
    }

    _FORCE_INLINE_ Vector3 cross(const Vector3& p_with) const {
#if defined(VECTOR3SIMD_USE_SSE)
        return Vector3(Vector3SIMD(*this).cross_sse(Vector3SIMD(p_with)));
#elif defined(VECTOR3SIMD_USE_NEON)
        return Vector3(Vector3SIMD(*this).cross_neon(Vector3SIMD(p_with)));
#else
        return cross_fallback(p_with);
#endif
    }

    _FORCE_INLINE_ real_t dot(const Vector3& p_with) const {
#if defined(VECTOR3SIMD_USE_SSE)
        return Vector3SIMD(*this).dot_sse(Vector3SIMD(p_with));
#elif defined(VECTOR3SIMD_USE_NEON)
        return Vector3SIMD(*this).dot_neon(Vector3SIMD(p_with));
#else
        return dot_fallback(p_with);
#endif
    }

    _FORCE_INLINE_ real_t length() const {
#if defined(VECTOR3SIMD_USE_SSE)
        return Vector3SIMD(*this).length_sse();
#elif defined(VECTOR3SIMD_USE_NEON)
        return Vector3SIMD(*this).length_neon();
#else
        return length_fallback();
#endif
    }

    _FORCE_INLINE_ real_t length_squared() const {
#if defined(VECTOR3SIMD_USE_SSE)
        return Vector3SIMD(*this).length_squared_sse();
#elif defined(VECTOR3SIMD_USE_NEON)
        return Vector3SIMD(*this).length_squared_neon();
#else
        return length_squared_fallback();
#endif
    }

    _FORCE_INLINE_ void normalize() {
#if defined(VECTOR3SIMD_USE_SSE)
        *this = Vector3(Vector3SIMD(*this).normalize_sse());
#elif defined(VECTOR3SIMD_USE_NEON)
        *this = Vector3(Vector3SIMD(*this).normalize_neon());
#else
        normalize_fallback();
#endif
    }

    _FORCE_INLINE_ Vector3 normalized() const {
#if defined(VECTOR3SIMD_USE_SSE)
        return Vector3(Vector3SIMD(*this).normalize_sse());
#elif defined(VECTOR3SIMD_USE_NEON)
        return Vector3(Vector3SIMD(*this).normalize_neon());
#else
        return normalized_fallback();
#endif
    }

    _FORCE_INLINE_ bool is_normalized() const {
#if defined(VECTOR3SIMD_USE_SSE)
        return Vector3SIMD(*this).is_normalized_sse();
#elif defined(VECTOR3SIMD_USE_NEON)
        return Vector3SIMD(*this).is_normalized_neon();
#else
        return is_normalized_fallback();
#endif
    }

    _FORCE_INLINE_ Vector3 inverse() const {
#if defined(VECTOR3SIMD_USE_SSE)
        return Vector3(Vector3SIMD(*this).inverse_sse());
#elif defined(VECTOR3SIMD_USE_NEON)
        return Vector3(Vector3SIMD(*this).inverse_neon());
#else
        return inverse_fallback();
#endif
    }

/**************************************************************************/
   /* Geometric operations */
   /**************************************************************************/
   _FORCE_INLINE_ Vector3 abs() const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).abs_sse());
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).abs_neon());
#else
       return abs_fallback();
#endif
   }

   _FORCE_INLINE_ Vector3 sign() const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).sign_sse());
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).sign_neon());
#else
       return sign_fallback();
#endif
   }

   _FORCE_INLINE_ Vector3 floor() const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).floor_sse());
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).floor_neon());
#else
       return floor_fallback();
#endif
   }

   _FORCE_INLINE_ Vector3 ceil() const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).ceil_sse());
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).ceil_neon());
#else
       return ceil_fallback();
#endif
   }

   _FORCE_INLINE_ Vector3 round() const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).round_sse());
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).round_neon());
#else
       return round_fallback();
#endif
   }

   /**************************************************************************/
   /* Min/Max operations */
   /**************************************************************************/
   Vector3 min(const Vector3& p_vector3) const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).min_sse(Vector3SIMD(p_vector3)));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).min_neon(Vector3SIMD(p_vector3)));
#else
       return min_fallback(p_vector3);
#endif
   }

   Vector3 max(const Vector3& p_vector3) const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).max_sse(Vector3SIMD(p_vector3)));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).max_neon(Vector3SIMD(p_vector3)));
#else
       return max_fallback(p_vector3);
#endif
   }

   Vector3 minf(real_t p_scalar) const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).minf_sse(p_scalar));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).minf_neon(p_scalar));
#else
       return minf_fallback(p_scalar);
#endif
   }

   Vector3 maxf(real_t p_scalar) const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).maxf_sse(p_scalar));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).maxf_neon(p_scalar));
#else
       return maxf_fallback(p_scalar);
#endif
   }

   /**************************************************************************/
   /* Distance calculations */
   /**************************************************************************/
   _FORCE_INLINE_ real_t distance_to(const Vector3& p_to) const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3SIMD(*this).distance_to_sse(Vector3SIMD(p_to));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3SIMD(*this).distance_to_neon(Vector3SIMD(p_to));
#else
       return distance_to_fallback(p_to);
#endif
   }

   _FORCE_INLINE_ real_t distance_squared_to(const Vector3& p_to) const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3SIMD(*this).distance_squared_to_sse(Vector3SIMD(p_to));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3SIMD(*this).distance_squared_to_neon(Vector3SIMD(p_to));
#else
       return distance_squared_to_fallback(p_to);
#endif
   }

   _FORCE_INLINE_ Vector3 direction_to(const Vector3& p_to) const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).direction_to_sse(Vector3SIMD(p_to)));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).direction_to_neon(Vector3SIMD(p_to)));
#else
       return direction_to_fallback(p_to);
#endif
   }

/**************************************************************************/
   /* Angle calculations */
   /**************************************************************************/
   _FORCE_INLINE_ real_t angle_to(const Vector3& p_to) const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3SIMD(*this).angle_to_sse(Vector3SIMD(p_to));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3SIMD(*this).angle_to_neon(Vector3SIMD(p_to));
#else
       return angle_to_fallback(p_to);
#endif
   }

   _FORCE_INLINE_ real_t signed_angle_to(const Vector3& p_to, const Vector3& p_axis) const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3SIMD(*this).signed_angle_to_sse(Vector3SIMD(p_to), Vector3SIMD(p_axis));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3SIMD(*this).signed_angle_to_neon(Vector3SIMD(p_to), Vector3SIMD(p_axis));
#else
       return signed_angle_to_fallback(p_to, p_axis);
#endif
   }

   /**************************************************************************/
   /* Projection and reflection */
   /**************************************************************************/
   _FORCE_INLINE_ Vector3 project(const Vector3& p_to) const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).project_sse(Vector3SIMD(p_to)));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).project_neon(Vector3SIMD(p_to)));
#else
       return project_fallback(p_to);
#endif
   }

   _FORCE_INLINE_ Vector3 reflect(const Vector3& p_normal) const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).reflect_sse(Vector3SIMD(p_normal)));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).reflect_neon(Vector3SIMD(p_normal)));
#else
       return reflect_fallback(p_normal);
#endif
   }

   _FORCE_INLINE_ Vector3 bounce(const Vector3& p_normal) const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).bounce_sse(Vector3SIMD(p_normal)));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).bounce_neon(Vector3SIMD(p_normal)));
#else
       return bounce_fallback(p_normal);
#endif
   }

   _FORCE_INLINE_ Vector3 slide(const Vector3& p_normal) const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).slide_sse(Vector3SIMD(p_normal)));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).slide_neon(Vector3SIMD(p_normal)));
#else
       return slide_fallback(p_normal);
#endif
   }

   _FORCE_INLINE_ Vector3 move_toward(const Vector3& p_to, real_t p_delta) const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).move_toward_sse(Vector3SIMD(p_to), p_delta));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).move_toward_neon(Vector3SIMD(p_to), p_delta));
#else
       return move_toward_fallback(p_to, p_delta);
#endif
   }

   /**************************************************************************/
   /* Modulo operations */
   /**************************************************************************/
   _FORCE_INLINE_ Vector3 posmod(real_t p_mod) const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).posmod_sse(p_mod));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).posmod_neon(p_mod));
#else
       return Vector3(Math::fposmod(x, p_mod), Math::fposmod(y, p_mod), Math::fposmod(z, p_mod));
#endif
   }

   _FORCE_INLINE_ Vector3 posmodv(const Vector3& p_modv) const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).posmodv_sse(Vector3SIMD(p_modv)));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).posmodv_neon(Vector3SIMD(p_modv)));
#else
       return Vector3(Math::fposmod(x, p_modv.x), Math::fposmod(y, p_modv.y), Math::fposmod(z, p_modv.z));
#endif
   }

/**************************************************************************/
   /* Rotation operations */
   /**************************************************************************/
   void rotate(const Vector3& p_axis, real_t p_angle) {
       *this = Basis(p_axis, p_angle).xform(*this);
   }

   Vector3 rotated(const Vector3& p_axis, real_t p_angle) const {
       Vector3 r = *this;
       r.rotate(p_axis, p_angle);
       return r;
   }

   /**************************************************************************/
   /* Snapping operations */
   /**************************************************************************/
   void snap(const Vector3& p_step) {
#if defined(VECTOR3SIMD_USE_SSE)
       *this = Vector3(Vector3SIMD(*this).snap_sse(Vector3SIMD(p_step)));
#elif defined(VECTOR3SIMD_USE_NEON)
       *this = Vector3(Vector3SIMD(*this).snap_neon(Vector3SIMD(p_step)));
#else
       snap_fallback(p_step);
#endif
   }

   Vector3 snapped(const Vector3& p_step) const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).snapped_sse(Vector3SIMD(p_step)));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).snapped_neon(Vector3SIMD(p_step)));
#else
       return snapped_fallback(p_step);
#endif
   }

   void snapf(real_t p_step) {
#if defined(VECTOR3SIMD_USE_SSE)
       *this = Vector3(Vector3SIMD(*this).snapf_sse(p_step));
#elif defined(VECTOR3SIMD_USE_NEON)
       *this = Vector3(Vector3SIMD(*this).snapf_neon(p_step));
#else
       snapf_fallback(p_step);
#endif
   }

   Vector3 snappedf(real_t p_step) const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).snappedf_sse(p_step));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).snappedf_neon(p_step));
#else
       return snappedf_fallback(p_step);
#endif
   }

   /**************************************************************************/
   /* Clamping operations */
   /**************************************************************************/
   Vector3 clamp(const Vector3& p_min, const Vector3& p_max) const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).clamp_sse(Vector3SIMD(p_min), Vector3SIMD(p_max)));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).clamp_neon(Vector3SIMD(p_min), Vector3SIMD(p_max)));
#else
       return clamp_fallback(p_min, p_max);
#endif
   }

   Vector3 clampf(real_t p_min, real_t p_max) const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).clampf_sse(p_min, p_max));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).clampf_neon(p_min, p_max));
#else
       return clampf_fallback(p_min, p_max);
#endif
   }

   Vector3 limit_length(real_t p_len = 1.0) const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).limit_length_sse(p_len));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).limit_length_neon(p_len));
#else
       return limit_length_fallback(p_len);
#endif
   }

/**************************************************************************/
   /* Interpolation operations */
   /**************************************************************************/
   _FORCE_INLINE_ Vector3 lerp(const Vector3& p_to, real_t p_weight) const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).lerp_sse(Vector3SIMD(p_to), p_weight));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).lerp_neon(Vector3SIMD(p_to), p_weight));
#else
       return lerp_fallback(p_to, p_weight);
#endif
   }

   _FORCE_INLINE_ Vector3 slerp(const Vector3& p_to, real_t p_weight) const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).slerp_sse(Vector3SIMD(p_to), p_weight));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).slerp_neon(Vector3SIMD(p_to), p_weight));
#else
       return slerp_fallback(p_to, p_weight);
#endif
   }

   _FORCE_INLINE_ Vector3 cubic_interpolate(const Vector3& p_b, const Vector3& p_pre_a,
                                         const Vector3& p_post_b, real_t p_weight) const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).cubic_interpolate_sse(Vector3SIMD(p_b),
                     Vector3SIMD(p_pre_a), Vector3SIMD(p_post_b), p_weight));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).cubic_interpolate_neon(Vector3SIMD(p_b),
                     Vector3SIMD(p_pre_a), Vector3SIMD(p_post_b), p_weight));
#else
       return cubic_interpolate_fallback(p_b, p_pre_a, p_post_b, p_weight);
#endif
   }

   _FORCE_INLINE_ Vector3 cubic_interpolate_in_time(const Vector3& p_b, const Vector3& p_pre_a,
                                                 const Vector3& p_post_b, real_t p_weight,
                                                 real_t p_b_t, real_t p_pre_a_t,
                                                 real_t p_post_b_t) const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).cubic_interpolate_in_time_sse(Vector3SIMD(p_b),
                     Vector3SIMD(p_pre_a), Vector3SIMD(p_post_b), p_weight,
                     p_b_t, p_pre_a_t, p_post_b_t));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).cubic_interpolate_in_time_neon(Vector3SIMD(p_b),
                     Vector3SIMD(p_pre_a), Vector3SIMD(p_post_b), p_weight,
                     p_b_t, p_pre_a_t, p_post_b_t));
#else
       return cubic_interpolate_in_time_fallback(p_b, p_pre_a, p_post_b,
                                              p_weight, p_b_t, p_pre_a_t, p_post_b_t);
#endif
   }

   _FORCE_INLINE_ Vector3 bezier_interpolate(const Vector3& p_control_1,
                                          const Vector3& p_control_2,
                                          const Vector3& p_end, real_t p_t) const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).bezier_interpolate_sse(Vector3SIMD(p_control_1),
                     Vector3SIMD(p_control_2), Vector3SIMD(p_end), p_t));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).bezier_interpolate_neon(Vector3SIMD(p_control_1),
                     Vector3SIMD(p_control_2), Vector3SIMD(p_end), p_t));
#else
       return bezier_interpolate_fallback(p_control_1, p_control_2, p_end, p_t);
#endif
   }

   _FORCE_INLINE_ Vector3 bezier_derivative(const Vector3& p_control_1,
                                         const Vector3& p_control_2,
                                         const Vector3& p_end, real_t p_t) const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).bezier_derivative_sse(Vector3SIMD(p_control_1),
                     Vector3SIMD(p_control_2), Vector3SIMD(p_end), p_t));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).bezier_derivative_neon(Vector3SIMD(p_control_1),
                     Vector3SIMD(p_control_2), Vector3SIMD(p_end), p_t));
#else
       return bezier_derivative_fallback(p_control_1, p_control_2, p_end, p_t);
#endif
   }

   /**************************************************************************/
   /* Special methods */
   /**************************************************************************/
   _FORCE_INLINE_ Vector2 octahedron_encode() const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3SIMD(*this).octahedron_encode_sse();
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3SIMD(*this).octahedron_encode_neon();
#else
       return octahedron_encode_fallback();
#endif
   }

   _FORCE_INLINE_ static Vector3 octahedron_decode(const Vector2& p_oct) {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD::octahedron_decode_sse(p_oct));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD::octahedron_decode_neon(p_oct));
#else
       return octahedron_decode_fallback(p_oct);
#endif
   }

/**************************************************************************/
   /* Operators */
   /**************************************************************************/
   _FORCE_INLINE_ Vector3& operator+=(const Vector3& p_v) {
#if defined(VECTOR3SIMD_USE_SSE)
       *this = Vector3(Vector3SIMD(*this).add_sse(Vector3SIMD(p_v)));
#elif defined(VECTOR3SIMD_USE_NEON)
       *this = Vector3(Vector3SIMD(*this).add_neon(Vector3SIMD(p_v)));
#else
       x += p_v.x;
       y += p_v.y;
       z += p_v.z;
#endif
       return *this;
   }

   _FORCE_INLINE_ Vector3 operator+(const Vector3& p_v) const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).add_sse(Vector3SIMD(p_v)));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).add_neon(Vector3SIMD(p_v)));
#else
       return Vector3(x + p_v.x, y + p_v.y, z + p_v.z);
#endif
   }

   _FORCE_INLINE_ Vector3& operator-=(const Vector3& p_v) {
#if defined(VECTOR3SIMD_USE_SSE)
       *this = Vector3(Vector3SIMD(*this).sub_sse(Vector3SIMD(p_v)));
#elif defined(VECTOR3SIMD_USE_NEON)
       *this = Vector3(Vector3SIMD(*this).sub_neon(Vector3SIMD(p_v)));
#else
       x -= p_v.x;
       y -= p_v.y;
       z -= p_v.z;
#endif
       return *this;
   }

   _FORCE_INLINE_ Vector3 operator-(const Vector3& p_v) const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).sub_sse(Vector3SIMD(p_v)));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).sub_neon(Vector3SIMD(p_v)));
#else
       return Vector3(x - p_v.x, y - p_v.y, z - p_v.z);
#endif
   }

   _FORCE_INLINE_ Vector3& operator*=(const Vector3& p_v) {
#if defined(VECTOR3SIMD_USE_SSE)
       *this = Vector3(Vector3SIMD(*this).mul_sse(Vector3SIMD(p_v)));
#elif defined(VECTOR3SIMD_USE_NEON)
       *this = Vector3(Vector3SIMD(*this).mul_neon(Vector3SIMD(p_v)));
#else
       return multiply_vector_fallback(p_v);
#endif
       return *this;
   }

   _FORCE_INLINE_ Vector3 operator*(const Vector3& p_v) const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).mul_sse(Vector3SIMD(p_v)));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).mul_neon(Vector3SIMD(p_v)));
#else
       return multiply_vector_const_fallback(p_v);
#endif
   }

   _FORCE_INLINE_ Vector3& operator*=(real_t p_scalar) {
#if defined(VECTOR3SIMD_USE_SSE)
       *this = Vector3(Vector3SIMD(*this).mul_scalar_sse(p_scalar));
#elif defined(VECTOR3SIMD_USE_NEON)
       *this = Vector3(Vector3SIMD(*this).mul_scalar_neon(p_scalar));
#else
       return multiply_scalar_fallback(p_scalar);
#endif
       return *this;
   }

   _FORCE_INLINE_ Vector3 operator*(real_t p_scalar) const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).mul_scalar_sse(p_scalar));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).mul_scalar_neon(p_scalar));
#else
       return multiply_scalar_const_fallback(p_scalar);
#endif
   }

   _FORCE_INLINE_ Vector3& operator/=(const Vector3& p_v) {
#if defined(VECTOR3SIMD_USE_SSE)
       *this = Vector3(Vector3SIMD(*this).div_sse(Vector3SIMD(p_v)));
#elif defined(VECTOR3SIMD_USE_NEON)
       *this = Vector3(Vector3SIMD(*this).div_neon(Vector3SIMD(p_v)));
#else
       return divide_vector_fallback(p_v);
#endif
       return *this;
   }

   _FORCE_INLINE_ Vector3 operator/(const Vector3& p_v) const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).div_sse(Vector3SIMD(p_v)));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).div_neon(Vector3SIMD(p_v)));
#else
       return Vector3(x / p_v.x, y / p_v.y, z / p_v.z);
#endif
   }

_FORCE_INLINE_ Vector3& operator/=(real_t p_scalar) {
#if defined(VECTOR3SIMD_USE_SSE)
       *this = Vector3(Vector3SIMD(*this).div_scalar_sse(p_scalar));
#elif defined(VECTOR3SIMD_USE_NEON)
       *this = Vector3(Vector3SIMD(*this).div_scalar_neon(p_scalar));
#else
       return divide_scalar_fallback(p_scalar);
#endif
       return *this;
   }

   _FORCE_INLINE_ Vector3 operator/(real_t p_scalar) const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).div_scalar_sse(p_scalar));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).div_scalar_neon(p_scalar));
#else
       return Vector3(x / p_scalar, y / p_scalar, z / p_scalar);
#endif
   }

   _FORCE_INLINE_ Vector3 operator-() const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3(Vector3SIMD(*this).neg_sse());
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3(Vector3SIMD(*this).neg_neon());
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
       return Vector3SIMD(*this).is_equal_approx_sse(Vector3SIMD(p_v));
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3SIMD(*this).is_equal_approx_neon(Vector3SIMD(p_v));
#else
       return is_equal_approx_fallback(p_v);
#endif
   }

   _FORCE_INLINE_ bool is_zero_approx() const {
#if defined(VECTOR3SIMD_USE_SSE)
       return Vector3SIMD(*this).is_zero_approx_sse();
#elif defined(VECTOR3SIMD_USE_NEON)
       return Vector3SIMD(*this).is_zero_approx_neon();
#else
       return is_zero_approx_fallback();
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

private:
   /**************************************************************************/
   /* Private fallback implementations */
   /**************************************************************************/

   // Core math fallbacks
   Vector3 cross_fallback(const Vector3& p_with) const;
   real_t dot_fallback(const Vector3& p_with) const;
   real_t length_fallback() const;
   real_t length_squared_fallback() const;
   void normalize_fallback();
   Vector3 normalized_fallback() const;
   bool is_normalized_fallback() const;
   Vector3 inverse_fallback() const;

   // Min/Max fallbacks
   Vector3 min_fallback(const Vector3& p_vector3) const;
   Vector3 max_fallback(const Vector3& p_vector3) const;
   Vector3 minf_fallback(real_t p_scalar) const;
   Vector3 maxf_fallback(real_t p_scalar) const;

   // Distance and projection fallbacks
   real_t distance_to_fallback(const Vector3& p_to) const;
   real_t distance_squared_to_fallback(const Vector3& p_to) const;
   Vector3 project_fallback(const Vector3& p_to) const;

   // Angle calculation fallbacks
   real_t angle_to_fallback(const Vector3& p_to) const;
   real_t signed_angle_to_fallback(const Vector3& p_to, const Vector3& p_axis) const;
   Vector3 direction_to_fallback(const Vector3& p_to) const;

   // Reflection and bounce fallbacks
   Vector3 reflect_fallback(const Vector3& p_normal) const;
   Vector3 slide_fallback(const Vector3& p_normal) const;
   Vector3 bounce_fallback(const Vector3& p_normal) const;

   // Interpolation fallbacks
   Vector3 lerp_fallback(const Vector3& p_to, real_t p_weight) const;
   Vector3 slerp_fallback(const Vector3& p_to, real_t p_weight) const;
   Vector3 cubic_interpolate_fallback(const Vector3& p_b, const Vector3& p_pre_a,
                                    const Vector3& p_post_b, real_t p_weight) const;
   Vector3 cubic_interpolate_in_time_fallback(const Vector3& p_b, const Vector3& p_pre_a,
                                            const Vector3& p_post_b, real_t p_weight,
                                            real_t p_b_t, real_t p_pre_a_t,
                                            real_t p_post_b_t) const;
   Vector3 bezier_interpolate_fallback(const Vector3& p_control_1,
                                     const Vector3& p_control_2,
                                     const Vector3& p_end, real_t p_t) const;
   Vector3 bezier_derivative_fallback(const Vector3& p_control_1,
                                    const Vector3& p_control_2,
                                    const Vector3& p_end, real_t p_t) const;
   Vector3 move_toward_fallback(const Vector3& p_to, real_t p_delta) const;

   // Arithmetic operation fallbacks
   Vector3& multiply_vector_fallback(const Vector3& p_v);
   Vector3& multiply_scalar_fallback(real_t p_scalar);
   Vector3& divide_vector_fallback(const Vector3& p_v);
   Vector3& divide_scalar_fallback(real_t p_scalar);
   Vector3 multiply_vector_const_fallback(const Vector3& p_v) const;
   Vector3 multiply_scalar_const_fallback(real_t p_scalar) const;

   // Transform fallbacks
   Vector3 abs_fallback() const;
   Vector3 sign_fallback() const;
   Vector3 floor_fallback() const;
   Vector3 ceil_fallback() const;
   Vector3 round_fallback() const;
   Vector3 rotated_fallback(const Vector3& p_axis, real_t p_angle) const;

// Clamping and snapping fallbacks
   Vector3 clamp_fallback(const Vector3& p_min, const Vector3& p_max) const;
   Vector3 clampf_fallback(real_t p_min, real_t p_max) const;
   void snap_fallback(const Vector3& p_step);
   Vector3 snapped_fallback(const Vector3& p_step) const;
   void snapf_fallback(real_t p_step);
   Vector3 snappedf_fallback(real_t p_step) const;
   Vector3 limit_length_fallback(real_t p_len) const;

   // Approximate equality fallbacks
   bool is_equal_approx_fallback(const Vector3& p_v) const;
   bool is_zero_approx_fallback() const;

   // Octahedron encoding/decoding fallbacks
   Vector2 octahedron_encode_fallback() const;
   static Vector3 octahedron_decode_fallback(const Vector2& p_oct);

   Vector2 octahedron_tangent_encode(float p_sign) const;
   static Vector3 octahedron_tangent_decode(const Vector2& p_oct, float* r_sign);
};

/*********************************************************************************/
/* Global operators */
/*********************************************************************************/
_FORCE_INLINE_ Vector3 operator*(float p_scalar, const Vector3& p_vec) {
   return p_vec * p_scalar;
}

_FORCE_INLINE_ Vector3 operator*(double p_scalar, const Vector3& p_vec) {
   return p_vec * p_scalar;
}

_FORCE_INLINE_ Vector3 operator*(int32_t p_scalar, const Vector3& p_vec) {
   return p_vec * p_scalar;
}

_FORCE_INLINE_ Vector3 operator*(int64_t p_scalar, const Vector3& p_vec) {
   return p_vec * p_scalar;
}

/**********************************************************************************/
/* Now that Vector3 is fully declared, we can include Vector3SIMD */
/**********************************************************************************/
#include "vector3simd.h"

/**********************************************************************************/
/* Static constants */
/**********************************************************************************/
// Note: These were causing compilation errors in the class, so defining them here
inline const Vector3 Vector3::ZERO = Vector3(0, 0, 0);
inline const Vector3 Vector3::ONE = Vector3(1, 1, 1);
inline const Vector3 Vector3::LEFT = Vector3(-1, 0, 0);
inline const Vector3 Vector3::RIGHT = Vector3(1, 0, 0);
inline const Vector3 Vector3::UP = Vector3(0, 1, 0);
inline const Vector3 Vector3::DOWN = Vector3(0, -1, 0);
inline const Vector3 Vector3::FORWARD = Vector3(0, 0, 1);
inline const Vector3 Vector3::BACK = Vector3(0, 0, -1);

#endif // VECTOR3_H