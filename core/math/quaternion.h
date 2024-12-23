#ifndef QUATERNION_H
#define QUATERNION_H

#include "core/math/vector4.h"
#include "core/math/math_funcs.h"
#include "core/math/math_defs.h"
#include "core/string/ustring.h"

// Forward declarations
struct Vector3;

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

    // Constructors
    _FORCE_INLINE_ Quaternion() : x(0), y(0), z(0), w(1.0f) {}
    
    _FORCE_INLINE_ Quaternion(real_t p_x, real_t p_y, real_t p_z, real_t p_w) :
        x(p_x), y(p_y), z(p_z), w(p_w) {}

    _FORCE_INLINE_ Quaternion(const Quaternion& p_q) :
        x(p_q.x), y(p_q.y), z(p_q.z), w(p_q.w) {}

    explicit _FORCE_INLINE_ Quaternion(const Vector4& vec) :
        x(vec.x), y(vec.y), z(vec.z), w(vec.w) {}

    Quaternion(const Vector3& p_axis, real_t p_angle);
    Quaternion(const Vector3& p_v0, const Vector3& p_v1); // Shortest arc

    // Assignment
    _FORCE_INLINE_ Quaternion& operator=(const Quaternion& p_q) {
        x = p_q.x;
        y = p_q.y;
        z = p_q.z;
        w = p_q.w;
        return *this;
    }

    // Array access
    _FORCE_INLINE_ real_t& operator[](int p_idx) {
        return components[p_idx];
    }

    _FORCE_INLINE_ const real_t& operator[](int p_idx) const {
        return components[p_idx];
    }

private:
#if defined(VECTOR4_USE_SSE)
    _FORCE_INLINE_ Quaternion quaternion_mul_sse(const Quaternion& p_q) const;
#endif
#if defined(VECTOR4_USE_NEON)
    _FORCE_INLINE_ Quaternion quaternion_mul_neon(const Quaternion& p_q) const;
#endif
    _FORCE_INLINE_ Quaternion quaternion_mul_fallback(const Quaternion& p_q) const;

public:
    // Core methods
    _FORCE_INLINE_ real_t length_squared() const { return dot(*this); }
    _FORCE_INLINE_ real_t dot(const Quaternion& p_q) const { return components.dot(p_q.components); }
    bool is_equal_approx(const Quaternion& p_quaternion) const;
    bool is_finite() const;
    real_t length() const;
    void normalize();
    Quaternion normalized() const;
    bool is_normalized() const;
    Quaternion inverse() const;
    Quaternion log() const;
    Quaternion exp() const;
    real_t angle_to(const Quaternion& p_to) const;

    // Euler conversions
    Vector3 get_euler(EulerOrder p_order = EulerOrder::YXZ) const;
    static Quaternion from_euler(const Vector3& p_euler);

    // Interpolation
    Quaternion slerp(const Quaternion& p_to, real_t p_weight) const;
    Quaternion slerpni(const Quaternion& p_to, real_t p_weight) const;
    Quaternion spherical_cubic_interpolate(const Quaternion& p_b, const Quaternion& p_pre_a, 
        const Quaternion& p_post_b, real_t p_weight) const;
    Quaternion spherical_cubic_interpolate_in_time(const Quaternion& p_b, const Quaternion& p_pre_a,
        const Quaternion& p_post_b, real_t p_weight, real_t p_b_t, real_t p_pre_a_t, real_t p_post_b_t) const;

    // Axis/angle methods
    Vector3 get_axis() const;
    real_t get_angle() const;
    _FORCE_INLINE_ void get_axis_angle(Vector3& r_axis, real_t& r_angle) const {
        r_angle = 2 * Math::acos(w);
        real_t r = ((real_t)1) / Math::sqrt(1 - w * w);
        r_axis.x = x * r;
        r_axis.y = y * r;
        r_axis.z = z * r;
    }

    // Vector transformation
    _FORCE_INLINE_ Vector3 xform(const Vector3& p_v) const {
#ifdef MATH_CHECKS
        ERR_FAIL_COND_V_MSG(!is_normalized(), p_v, "The quaternion " + operator String() + " must be normalized.");
#endif
        Vector3 u(x, y, z);
        Vector3 uv = u.cross(p_v);
        return p_v + ((uv * w) + u.cross(uv)) * ((real_t)2);
    }

    _FORCE_INLINE_ Vector3 xform_inv(const Vector3& p_v) const {
        return inverse().xform(p_v);
    }

    // Operators
    _FORCE_INLINE_ Quaternion operator*(const Quaternion& p_q) const {
#if defined(VECTOR4_USE_SSE)
        return quaternion_mul_sse(p_q);
#elif defined(VECTOR4_USE_NEON)
        return quaternion_mul_neon(p_q);
#else
        return quaternion_mul_fallback(p_q);
#endif
    }

    _FORCE_INLINE_ void operator*=(const Quaternion& p_q) {
        *this = *this * p_q;
    }

    _FORCE_INLINE_ void operator+=(const Quaternion& p_q) {
        components += p_q.components;
    }

    _FORCE_INLINE_ void operator-=(const Quaternion& p_q) {
        components -= p_q.components;
    }

    _FORCE_INLINE_ void operator*=(real_t p_s) {
        components *= p_s;
    }

    _FORCE_INLINE_ void operator/=(real_t p_s) {
        components /= p_s;
    }

    _FORCE_INLINE_ Quaternion operator+(const Quaternion& p_q2) const {
        return Quaternion(components + p_q2.components);
    }

    _FORCE_INLINE_ Quaternion operator-(const Quaternion& p_q2) const {
        return Quaternion(components - p_q2.components);
    }

    _FORCE_INLINE_ Quaternion operator-() const {
        return Quaternion(-components);
    }

    _FORCE_INLINE_ Quaternion operator*(real_t p_s) const {
        return Quaternion(components * p_s);
    }

    _FORCE_INLINE_ Quaternion operator/(real_t p_s) const {
        return Quaternion(components / p_s);
    }

    _FORCE_INLINE_ bool operator==(const Quaternion& p_quaternion) const {
        return components == p_quaternion.components;
    }

    _FORCE_INLINE_ bool operator!=(const Quaternion& p_quaternion) const {
        return components != p_quaternion.components;
    }

    operator String() const;
};

_FORCE_INLINE_ Quaternion operator*(real_t p_real, const Quaternion& p_quaternion) {
    return Quaternion(p_quaternion.components * p_real);
}

#endif // QUATERNION_H