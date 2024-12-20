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
        x = y = z = 0.0f;
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

    // Vector operations with SIMD dispatch
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

    _FORCE_INLINE_ real_t distance_to(const Vector3& p_to) const {
#if defined(VECTOR3SIMD_USE_SSE)
        Vector3SIMD simd_this(*this);
        Vector3SIMD simd_to(p_to);
        return simd_this.distance_to_sse(simd_to);
#elif defined(VECTOR3SIMD_USE_NEON)
        Vector3SIMD simd_this(*this);
        Vector3SIMD simd_to(p_to);
        return simd_this.distance_to_neon(simd_to);
#else
        return distance_to_fallback(p_to);
#endif
    }

    _FORCE_INLINE_ real_t distance_squared_to(const Vector3& p_to) const {
#if defined(VECTOR3SIMD_USE_SSE)
        Vector3SIMD simd_this(*this);
        Vector3SIMD simd_to(p_to);
        return simd_this.distance_squared_to_sse(simd_to);
#elif defined(VECTOR3SIMD_USE_NEON)
        Vector3SIMD simd_this(*this);
        Vector3SIMD simd_to(p_to);
        return simd_this.distance_squared_to_neon(simd_to);
#else
        return distance_squared_to_fallback(p_to);
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
        return Math::is_equal_approx(length_squared(), 1.0f);
#endif
    }

    _FORCE_INLINE_ Vector3 inverse() const {
#if defined(VECTOR3SIMD_USE_SSE)
        Vector3SIMD simd_this(*this);
        return Vector3(simd_this.inverse_sse());
#elif defined(VECTOR3SIMD_USE_NEON)
        Vector3SIMD simd_this(*this);
        return Vector3(simd_this.inverse_neon());
#else
        return inverse_fallback();
#endif
    }

    _FORCE_INLINE_ Vector3 clamp(const Vector3& p_min, const Vector3& p_max) const {
        return Vector3(
            CLAMP(x, p_min.x, p_max.x),
            CLAMP(y, p_min.y, p_max.y),
            CLAMP(z, p_min.z, p_max.z)
        );
    }

    _FORCE_INLINE_ Vector3 clampf(real_t p_min, real_t p_max) const {
        return Vector3(
            CLAMP(x, p_min, p_max),
            CLAMP(y, p_min, p_max),
            CLAMP(z, p_min, p_max)
        );
    }

    // Snap operations
    void snap(const Vector3& p_step);
    Vector3 snapped(const Vector3& p_step) const;
    void snapf(real_t p_step);
    Vector3 snappedf(real_t p_step) const;

    // Directional operations with SIMD dispatch
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
        return Vector3(simd_this.cross_sse(simd_with));
#elif defined(VECTOR3SIMD_USE_NEON)
        Vector3SIMD simd_this(*this);
        Vector3SIMD simd_with(p_with);
        return Vector3(simd_this.cross_neon(simd_with));
#else
        return cross_fallback(p_with);
#endif
    }

    _FORCE_INLINE_ Vector3 min(const Vector3& p_vector3) const {
#if defined(VECTOR3SIMD_USE_SSE)
        Vector3SIMD simd_this(*this);
        Vector3SIMD simd_with(p_vector3);
        return Vector3(simd_this.min_sse(simd_with));
#elif defined(VECTOR3SIMD_USE_NEON)
        Vector3SIMD simd_this(*this);
        Vector3SIMD simd_with(p_vector3);
        return Vector3(simd_this.min_neon(simd_with));
#else
        return min_fallback(p_vector3);
#endif
    }

    _FORCE_INLINE_ Vector3 max(const Vector3& p_vector3) const {
#if defined(VECTOR3SIMD_USE_SSE)
        Vector3SIMD simd_this(*this);
        Vector3SIMD simd_with(p_vector3);
        return Vector3(simd_this.max_sse(simd_with));
#elif defined(VECTOR3SIMD_USE_NEON)
        Vector3SIMD simd_this(*this);
        Vector3SIMD simd_with(p_vector3);
        return Vector3(simd_this.max_neon(simd_with));
#else
        return max_fallback(p_vector3);
#endif
    }

    // Operators with SIMD dispatch
    _FORCE_INLINE_ Vector3 operator+(const Vector3& p_v) const {
        return Vector3(x + p_v.x, y + p_v.y, z + p_v.z);
    }

    _FORCE_INLINE_ void operator+=(const Vector3& p_v) {
        x += p_v.x;
        y += p_v.y;
        z += p_v.z;
    }

    _FORCE_INLINE_ Vector3 operator-(const Vector3& p_v) const {
        return Vector3(x - p_v.x, y - p_v.y, z - p_v.z);
    }

    _FORCE_INLINE_ void operator-=(const Vector3& p_v) {
        x -= p_v.x;
        y -= p_v.y;
        z -= p_v.z;
    }

    _FORCE_INLINE_ Vector3 operator*(const Vector3& p_v) const {
#if defined(VECTOR3SIMD_USE_SSE)
        Vector3SIMD simd_this(*this);
        Vector3SIMD simd_with(p_v);
        return Vector3(simd_this.mul_sse(simd_with));
#elif defined(VECTOR3SIMD_USE_NEON)
        Vector3SIMD simd_this(*this);
        Vector3SIMD simd_with(p_v);
        return Vector3(simd_this.mul_neon(simd_with));
#else
        return multiply_vector_const_fallback(p_v);
#endif
    }

    _FORCE_INLINE_ void operator*=(const Vector3& p_v) {
#if defined(VECTOR3SIMD_USE_SSE)
        Vector3SIMD simd_this(*this);
        Vector3SIMD simd_with(p_v);
        *this = Vector3(simd_this.mul_sse(simd_with));
#elif defined(VECTOR3SIMD_USE_NEON)
        Vector3SIMD simd_this(*this);
        Vector3SIMD simd_with(p_v);
        *this = Vector3(simd_this.mul_neon(simd_with));
#else
        multiply_vector_fallback(p_v);
#endif
    }

    _FORCE_INLINE_ Vector3 operator*(real_t p_scalar) const {
#if defined(VECTOR3SIMD_USE_SSE)
        Vector3SIMD simd_this(*this);
        return Vector3(simd_this.mul_scalar_sse(p_scalar));
#elif defined(VECTOR3SIMD_USE_NEON)
        Vector3SIMD simd_this(*this);
        return Vector3(simd_this.mul_scalar_neon(p_scalar));
#else
        return multiply_scalar_const_fallback(p_scalar);
#endif
    }

    _FORCE_INLINE_ void operator*=(real_t p_scalar) {
#if defined(VECTOR3SIMD_USE_SSE)
        Vector3SIMD simd_this(*this);
        *this = Vector3(simd_this.mul_scalar_sse(p_scalar));
#elif defined(VECTOR3SIMD_USE_NEON)
        Vector3SIMD simd_this(*this);
        *this = Vector3(simd_this.mul_scalar_neon(p_scalar));
#else
        multiply_scalar_fallback(p_scalar);
#endif
    }

    _FORCE_INLINE_ Vector3 operator/(const Vector3& p_v) const {
#if defined(VECTOR3SIMD_USE_SSE)
        Vector3SIMD simd_this(*this);
        Vector3SIMD simd_with(p_v);
        return Vector3(simd_this.div_sse(simd_with));
#elif defined(VECTOR3SIMD_USE_NEON)
        Vector3SIMD simd_this(*this);
        Vector3SIMD simd_with(p_v);
        return Vector3(simd_this.div_neon(simd_with));
#else
        Vector3 ret;
        ret.x = x / p_v.x;
        ret.y = y / p_v.y;
        ret.z = z / p_v.z;
        return ret;
#endif
    }

    _FORCE_INLINE_ void operator/=(const Vector3& p_v) {
#if defined(VECTOR3SIMD_USE_SSE)
        Vector3SIMD simd_this(*this);
        Vector3SIMD simd_with(p_v);
        *this = Vector3(simd_this.div_sse(simd_with));
#elif defined(VECTOR3SIMD_USE_NEON)
        Vector3SIMD simd_this(*this);
        Vector3SIMD simd_with(p_v);
        *this = Vector3(simd_this.div_neon(simd_with));
#else
        divide_vector_fallback(p_v);
#endif
    }

    _FORCE_INLINE_ Vector3 operator/(real_t p_scalar) const {
#if defined(VECTOR3SIMD_USE_SSE)
        Vector3SIMD simd_this(*this);
        return Vector3(simd_this.div_scalar_sse(p_scalar));
#elif defined(VECTOR3SIMD_USE_NEON)
        Vector3SIMD simd_this(*this);
        return Vector3(simd_this.div_scalar_neon(p_scalar));
#else
        Vector3 ret;
        ret.x = x / p_scalar;
        ret.y = y / p_scalar;
        ret.z = z / p_scalar;
        return ret;
#endif
    }

    _FORCE_INLINE_ void operator/=(real_t p_scalar) {
#if defined(VECTOR3SIMD_USE_SSE)
        Vector3SIMD simd_this(*this);
        *this = Vector3(simd_this.div_scalar_sse(p_scalar));
#elif defined(VECTOR3SIMD_USE_NEON)
        Vector3SIMD simd_this(*this);
        *this = Vector3(simd_this.div_scalar_neon(p_scalar));
#else
        divide_scalar_fallback(p_scalar);
#endif
    }

    _FORCE_INLINE_ Vector3 operator-() const {
        return Vector3(-x, -y, -z);
    }

    // Comparison operators
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

    // Advanced vector operations with SIMD dispatch
    _FORCE_INLINE_ Vector3 abs() const {
#if defined(VECTOR3SIMD_USE_SSE)
        Vector3SIMD simd_this(*this);
        return Vector3(simd_this.abs_sse());
#elif defined(VECTOR3SIMD_USE_NEON)
        Vector3SIMD simd_this(*this);
        return Vector3(simd_this.abs_neon());
#else
        return Vector3(Math::abs(x), Math::abs(y), Math::abs(z));
#endif
    }

    _FORCE_INLINE_ Vector3 sign() const {
        return Vector3(SIGN(x), SIGN(y), SIGN(z));
    }

    _FORCE_INLINE_ Vector3 floor() const {
        return Vector3(Math::floor(x), Math::floor(y), Math::floor(z));
    }

    _FORCE_INLINE_ Vector3 ceil() const {
        return Vector3(Math::ceil(x), Math::ceil(y), Math::ceil(z));
    }

    _FORCE_INLINE_ Vector3 round() const {
        return Vector3(Math::round(x), Math::round(y), Math::round(z));
    }

    _FORCE_INLINE_ Vector3 lerp(const Vector3& p_to, real_t p_weight) const {
#if defined(VECTOR3SIMD_USE_SSE)
        Vector3SIMD simd_this(*this);
        Vector3SIMD simd_to(p_to);
        return Vector3(simd_this.lerp_sse(simd_to, p_weight));
#elif defined(VECTOR3SIMD_USE_NEON)
        Vector3SIMD simd_this(*this);
        Vector3SIMD simd_to(p_to);
        return Vector3(simd_this.lerp_neon(simd_to, p_weight));
#else
        return lerp_fallback(p_to, p_weight);
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

    _FORCE_INLINE_ Vector3 cubic_interpolate(const Vector3& p_b, const Vector3& p_pre_a,
                                          const Vector3& p_post_b, real_t p_weight) const {
#if defined(VECTOR3SIMD_USE_SSE)
        Vector3SIMD simd_this(*this);
        Vector3SIMD simd_b(p_b);
        Vector3SIMD simd_pre_a(p_pre_a);
        Vector3SIMD simd_post_b(p_post_b);
        return Vector3(simd_this.cubic_interpolate_sse(simd_b, simd_pre_a, simd_post_b, p_weight));
#elif defined(VECTOR3SIMD_USE_NEON)
        Vector3SIMD simd_this(*this);
        Vector3SIMD simd_b(p_b);
        Vector3SIMD simd_pre_a(p_pre_a);
        Vector3SIMD simd_post_b(p_post_b);
        return Vector3(simd_this.cubic_interpolate_neon(simd_b, simd_pre_a, simd_post_b, p_weight));
#else
        Vector3 p0 = *this;
        Vector3 p1 = p_b;
        Vector3 v0 = (p1 - p_pre_a) * 0.5;
        Vector3 v1 = (p_post_b - p0) * 0.5;

        real_t t = p_weight;
        real_t t2 = t * t;
        real_t t3 = t2 * t;

        Vector3 ret;
        ret = p0 * (2.0 * t3 - 3.0 * t2 + 1.0);
        ret += v0 * (t3 - 2.0 * t2 + t);
        ret += p1 * (-2.0 * t3 + 3.0 * t2);
        ret += v1 * (t3 - t2);

        return ret;
#endif
    }

    _FORCE_INLINE_ Vector3 move_toward(const Vector3& p_to, real_t p_delta) const {
#if defined(VECTOR3SIMD_USE_SSE)
        Vector3SIMD simd_this(*this);
        Vector3SIMD simd_to(p_to);
        return Vector3(simd_this.move_toward_sse(simd_to, p_delta));
#elif defined(VECTOR3SIMD_USE_NEON)
        Vector3SIMD simd_this(*this);
        Vector3SIMD simd_to(p_to);
        return Vector3(simd_this.move_toward_neon(simd_to, p_delta));
#else
        return move_toward_fallback(p_to, p_delta);
#endif
    }

    // Rotation methods
    void rotate(const Vector3& p_axis, real_t p_angle);
    Vector3 rotated(const Vector3& p_axis, real_t p_angle) const;

    // Project methods
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

    // Slide and bounce methods
    _FORCE_INLINE_ Vector3 slide(const Vector3& p_normal) const {
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

    _FORCE_INLINE_ Vector3 bounce(const Vector3& p_normal) const {
#if defined(VECTOR3SIMD_USE_SSE)
        Vector3SIMD simd_this(*this);
        Vector3SIMD simd_normal(p_normal);
        return Vector3(simd_this.bounce_sse(simd_normal));
#elif defined(VECTOR3SIMD_USE_NEON)
        Vector3SIMD simd_this(*this);
        Vector3SIMD simd_normal(p_normal);
        return Vector3(simd_this.bounce_neon(simd_normal));
#else
        return bounce_fallback(p_normal);
#endif
    }

    _FORCE_INLINE_ Vector3 reflect(const Vector3& p_normal) const {
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

    // Basis/matrix operations
    Basis outer(const Vector3& p_b) const;

    // Octahedron encoding/decoding
    Vector2 octahedron_encode() const;
    static Vector3 octahedron_decode(const Vector2& p_oct);
    Vector2 octahedron_tangent_encode(float p_sign) const;
    static Vector3 octahedron_tangent_decode(const Vector2& p_oct, float* r_sign);

     // Normalization methods
    void normalize();
    bool is_normalized() const;

    // Type conversion
    operator String() const;
    operator Vector3i() const;

    // Modulo operations
    Vector3 posmod(real_t p_mod) const;
    Vector3 posmodv(const Vector3& p_modv) const;

private:
    // Fallback implementations for when SIMD is unavailable
    real_t length_fallback() const;
    real_t length_squared_fallback() const;
    real_t distance_to_fallback(const Vector3& p_to) const;
    real_t distance_squared_to_fallback(const Vector3& p_to) const;
    Vector3 direction_to_fallback(const Vector3& p_to) const;
    real_t angle_to_fallback(const Vector3& p_to) const;
    real_t signed_angle_to_fallback(const Vector3& p_to, const Vector3& p_axis) const;
    real_t dot_fallback(const Vector3& p_with) const;
    Vector3 cross_fallback(const Vector3& p_with) const;
    Vector3 project_fallback(const Vector3& p_to) const;
    Vector3 slide_fallback(const Vector3& p_normal) const;
    Vector3 bounce_fallback(const Vector3& p_normal) const;
    Vector3 reflect_fallback(const Vector3& p_normal) const;
    Vector3 min_fallback(const Vector3& p_vector3) const;
    Vector3 max_fallback(const Vector3& p_vector3) const;
    Vector3 inverse_fallback() const;
    Vector3 lerp_fallback(const Vector3& p_to, real_t p_weight) const;
    Vector3 slerp_fallback(const Vector3& p_to, real_t p_weight) const;
    Vector3 move_toward_fallback(const Vector3& p_to, real_t p_delta) const;
    Vector3& multiply_vector_fallback(const Vector3& p_v);
    Vector3& multiply_scalar_fallback(real_t p_scalar);
    Vector3& divide_vector_fallback(const Vector3& p_v);
    Vector3& divide_scalar_fallback(real_t p_scalar);
    Vector3 multiply_vector_const_fallback(const Vector3& p_v) const;
    Vector3 multiply_scalar_const_fallback(real_t p_scalar) const;

    // Component-wise tests
    bool is_equal_approx(const Vector3& p_v) const;
    bool is_zero_approx() const;
    bool is_finite() const;

    // Length and limitation methods
    Vector3 limit_length(real_t p_len = 1.0f) const;
    Vector3 minf_fallback(real_t p_scalar) const;
    Vector3 maxf_fallback(real_t p_scalar) const;

    // Interpolation methods
    Vector3 cubic_interpolate_in_time(const Vector3& p_b, 
                                    const Vector3& p_pre_a,
                                    const Vector3& p_post_b,
                                    real_t p_weight,
                                    real_t p_b_t,
                                    real_t p_pre_a_t,
                                    real_t p_post_b_t) const;
    Vector3 bezier_interpolate(const Vector3& p_control_1,
                             const Vector3& p_control_2,
                             const Vector3& p_end,
                             real_t p_t) const;
    Vector3 bezier_derivative(const Vector3& p_control_1,
                            const Vector3& p_control_2,
                            const Vector3& p_end,
                            real_t p_t) const;
};

_FORCE_INLINE_ Vector3 operator*(real_t p_scalar, const Vector3& p_vec) {
    return p_vec * p_scalar;
}

#endif // VECTOR3_H