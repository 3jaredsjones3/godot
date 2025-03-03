#ifndef VECTOR4_H
#define VECTOR4_H

#include "core/typedefs.h"
#include "core/math/math_funcs.h"

class String;
struct Vector4i;

// Simple non-SIMD implementation of Vector4
struct Vector4 {
    static const int AXIS_COUNT = 4;

    enum Axis {
        AXIS_X,
        AXIS_Y,
        AXIS_Z,
        AXIS_W,
    };

    union {
        struct {
            real_t x, y, z, w;
        };
        real_t coord[4];
    };

    // Constructors
    _FORCE_INLINE_ Vector4() {
        x = y = z = w = 0.0f;
    }

    _FORCE_INLINE_ Vector4(const Vector4& p_v) {
        x = p_v.x;
        y = p_v.y;
        z = p_v.z;
        w = p_v.w;
    }

    _FORCE_INLINE_ Vector4(real_t p_x, real_t p_y, real_t p_z, real_t p_w) {
        x = p_x;
        y = p_y;
        z = p_z;
        w = p_w;
    }

    // Array access
    _FORCE_INLINE_ real_t &operator[](int p_index) {
        return coord[p_index];
    }

    _FORCE_INLINE_ const real_t &operator[](int p_index) const {
        return coord[p_index];
    }

    // Basic arithmetic
    _FORCE_INLINE_ Vector4 operator+(const Vector4 &p_v) const {
        return Vector4(x + p_v.x, y + p_v.y, z + p_v.z, w + p_v.w);
    }

    _FORCE_INLINE_ Vector4 operator-(const Vector4 &p_v) const {
        return Vector4(x - p_v.x, y - p_v.y, z - p_v.z, w - p_v.w);
    }

    _FORCE_INLINE_ Vector4 operator*(const Vector4 &p_v) const {
        return Vector4(x * p_v.x, y * p_v.y, z * p_v.z, w * p_v.w);
    }

    _FORCE_INLINE_ Vector4 operator/(const Vector4 &p_v) const {
        return Vector4(x / p_v.x, y / p_v.y, z / p_v.z, w / p_v.w);
    }

    // Scalar operations
    _FORCE_INLINE_ Vector4 operator*(real_t p_scalar) const {
        return Vector4(x * p_scalar, y * p_scalar, z * p_scalar, w * p_scalar);
    }

    _FORCE_INLINE_ Vector4 operator/(real_t p_scalar) const {
        return Vector4(x / p_scalar, y / p_scalar, z / p_scalar, w / p_scalar);
    }

    _FORCE_INLINE_ Vector4 operator-() const {
        return Vector4(-x, -y, -z, -w);
    }

    // Compound assignment
    _FORCE_INLINE_ Vector4 &operator+=(const Vector4 &p_v) {
        x += p_v.x;
        y += p_v.y;
        z += p_v.z;
        w += p_v.w;
        return *this;
    }

    _FORCE_INLINE_ Vector4 &operator-=(const Vector4 &p_v) {
        x -= p_v.x;
        y -= p_v.y;
        z -= p_v.z;
        w -= p_v.w;
        return *this;
    }

    _FORCE_INLINE_ Vector4 &operator*=(const Vector4 &p_v) {
        x *= p_v.x;
        y *= p_v.y;
        z *= p_v.z;
        w *= p_v.w;
        return *this;
    }

    _FORCE_INLINE_ Vector4 &operator/=(const Vector4 &p_v) {
        x /= p_v.x;
        y /= p_v.y;
        z /= p_v.z;
        w /= p_v.w;
        return *this;
    }

    _FORCE_INLINE_ Vector4 &operator*=(real_t p_scalar) {
        x *= p_scalar;
        y *= p_scalar;
        z *= p_scalar;
        w *= p_scalar;
        return *this;
    }

    _FORCE_INLINE_ Vector4 &operator/=(real_t p_scalar) {
        x /= p_scalar;
        y /= p_scalar;
        z /= p_scalar;
        w /= p_scalar;
        return *this;
    }

    // Comparison
    _FORCE_INLINE_ bool operator==(const Vector4 &p_v) const {
        return (x == p_v.x && y == p_v.y && z == p_v.z && w == p_v.w);
    }

    _FORCE_INLINE_ bool operator!=(const Vector4 &p_v) const {
        return (x != p_v.x || y != p_v.y || z != p_v.z || w != p_v.w);
    }
    
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
        if (x == p_v.x) {
            if (y == p_v.y) {
                if (z == p_v.z) {
                    return w <= p_v.w;
                }
                return z < p_v.z;
            }
            return y < p_v.y;
        }
        return x < p_v.x;
    }
    
    _FORCE_INLINE_ bool operator>=(const Vector4 &p_v) const {
        if (x == p_v.x) {
            if (y == p_v.y) {
                if (z == p_v.z) {
                    return w >= p_v.w;
                }
                return z > p_v.z;
            }
            return y > p_v.y;
        }
        return x > p_v.x;
    }

    // Length methods
    _FORCE_INLINE_ real_t length_squared() const {
        return x * x + y * y + z * z + w * w;
    }

    real_t length() const {
        return Math::sqrt(length_squared());
    }

    // Normalization
    _FORCE_INLINE_ Vector4 normalized() const {
        real_t l = length();
        if (l == 0) {
            return Vector4(0, 0, 0, 0);
        }
        return Vector4(x / l, y / l, z / l, w / l);
    }

    _FORCE_INLINE_ bool is_normalized() const {
        return Math::is_equal_approx(length_squared(), (real_t)1.0, (real_t)UNIT_EPSILON);
    }
    
    _FORCE_INLINE_ bool is_zero_approx() const {
        return Math::is_zero_approx(x) && Math::is_zero_approx(y) && Math::is_zero_approx(z) && Math::is_zero_approx(w);
    }

    _FORCE_INLINE_ void normalize() {
        real_t l = length();
        if (l == 0) {
            x = y = z = w = 0;
        } else {
            x /= l;
            y /= l;
            z /= l;
            w /= l;
        }
    }

    // Dot product
    _FORCE_INLINE_ real_t dot(const Vector4 &p_v) const {
        return x * p_v.x + y * p_v.y + z * p_v.z + w * p_v.w;
    }

    // Linear interpolation
    _FORCE_INLINE_ Vector4 lerp(const Vector4 &p_to, real_t p_weight) const {
        return Vector4(
            x + (p_to.x - x) * p_weight,
            y + (p_to.y - y) * p_weight,
            z + (p_to.z - z) * p_weight,
            w + (p_to.w - w) * p_weight
        );
    }

    // Cubic interpolation
    _FORCE_INLINE_ Vector4 cubic_interpolate(const Vector4 &p_b, const Vector4 &p_pre_a, const Vector4 &p_post_b, real_t p_weight) const {
        real_t p_weight2 = p_weight * p_weight;
        real_t p_weight3 = p_weight2 * p_weight;
        
        return Vector4(
            0.5f * ((x * 2.0f) +
                    (-p_pre_a.x + p_b.x) * p_weight +
                    (2.0f * p_pre_a.x - 5.0f * x + 4.0f * p_b.x - p_post_b.x) * p_weight2 +
                    (-p_pre_a.x + 3.0f * x - 3.0f * p_b.x + p_post_b.x) * p_weight3),
            0.5f * ((y * 2.0f) +
                    (-p_pre_a.y + p_b.y) * p_weight +
                    (2.0f * p_pre_a.y - 5.0f * y + 4.0f * p_b.y - p_post_b.y) * p_weight2 +
                    (-p_pre_a.y + 3.0f * y - 3.0f * p_b.y + p_post_b.y) * p_weight3),
            0.5f * ((z * 2.0f) +
                    (-p_pre_a.z + p_b.z) * p_weight +
                    (2.0f * p_pre_a.z - 5.0f * z + 4.0f * p_b.z - p_post_b.z) * p_weight2 +
                    (-p_pre_a.z + 3.0f * z - 3.0f * p_b.z + p_post_b.z) * p_weight3),
            0.5f * ((w * 2.0f) +
                    (-p_pre_a.w + p_b.w) * p_weight +
                    (2.0f * p_pre_a.w - 5.0f * w + 4.0f * p_b.w - p_post_b.w) * p_weight2 +
                    (-p_pre_a.w + 3.0f * w - 3.0f * p_b.w + p_post_b.w) * p_weight3)
        );
    }
    
    // Time-based cubic interpolation
    _FORCE_INLINE_ Vector4 cubic_interpolate_in_time(const Vector4 &p_b, const Vector4 &p_pre_a, const Vector4 &p_post_b, 
                                                    real_t p_weight, real_t p_b_t, real_t p_pre_a_t, real_t p_post_b_t) const {
        // Hermite based interpolation with respect to different time points
        real_t t = p_weight;
        real_t t2 = t * t;
        real_t t3 = t2 * t;
        
        // Calculate tangents
        Vector4 from_tangent, to_tangent;
        
        if (Math::is_zero_approx(p_pre_a_t - p_b_t)) {
            from_tangent = (*this - p_pre_a) / (real_t)0.0001;
        } else {
            from_tangent = (*this - p_pre_a) / (p_b_t - p_pre_a_t);
        }
        
        if (Math::is_zero_approx(p_post_b_t - p_b_t)) {
            to_tangent = (p_post_b - p_b) / (real_t)0.0001;
        } else {
            to_tangent = (p_post_b - p_b) / (p_post_b_t - p_b_t);
        }
        
        // Hermite polynomial coefficients
        real_t h1 = 2.0f * t3 - 3.0f * t2 + 1.0f;
        real_t h2 = -2.0f * t3 + 3.0f * t2;
        real_t h3 = t3 - 2.0f * t2 + t;
        real_t h4 = t3 - t2;
        
        // Interpolate
        return Vector4(
            h1 * x + h2 * p_b.x + h3 * from_tangent.x + h4 * to_tangent.x,
            h1 * y + h2 * p_b.y + h3 * from_tangent.y + h4 * to_tangent.y,
            h1 * z + h2 * p_b.z + h3 * from_tangent.z + h4 * to_tangent.z,
            h1 * w + h2 * p_b.w + h3 * from_tangent.w + h4 * to_tangent.w
        );
    }

    // Axis methods
    _FORCE_INLINE_ int min_axis() const {
        return (x < y) ? (x < z ? (x < w ? 0 : 3) : (z < w ? 2 : 3)) : (y < z ? (y < w ? 1 : 3) : (z < w ? 2 : 3));
    }

    _FORCE_INLINE_ int max_axis() const {
        return (x < y) ? (y < z ? (z < w ? 3 : 2) : (y < w ? 3 : 1)) : (x < z ? (z < w ? 3 : 2) : (x < w ? 3 : 0));
    }
    
    _FORCE_INLINE_ Axis min_axis_index() const {
        return (x < y) ? (x < z ? (x < w ? AXIS_X : AXIS_W) : (z < w ? AXIS_Z : AXIS_W)) : (y < z ? (y < w ? AXIS_Y : AXIS_W) : (z < w ? AXIS_Z : AXIS_W));
    }

    _FORCE_INLINE_ Axis max_axis_index() const {
        return (x < y) ? (y < z ? (z < w ? AXIS_W : AXIS_Z) : (y < w ? AXIS_W : AXIS_Y)) : (x < z ? (z < w ? AXIS_W : AXIS_Z) : (x < w ? AXIS_W : AXIS_X));
    }

    // MinMax component-wise operations
    _FORCE_INLINE_ Vector4 min(const Vector4 &p_v) const {
        return Vector4(
            (x < p_v.x) ? x : p_v.x,
            (y < p_v.y) ? y : p_v.y,
            (z < p_v.z) ? z : p_v.z,
            (w < p_v.w) ? w : p_v.w
        );
    }

    _FORCE_INLINE_ Vector4 max(const Vector4 &p_v) const {
        return Vector4(
            (x > p_v.x) ? x : p_v.x,
            (y > p_v.y) ? y : p_v.y,
            (z > p_v.z) ? z : p_v.z,
            (w > p_v.w) ? w : p_v.w
        );
    }

    _FORCE_INLINE_ Vector4 abs() const {
        return Vector4(Math::abs(x), Math::abs(y), Math::abs(z), Math::abs(w));
    }

    _FORCE_INLINE_ Vector4 sign() const {
        return Vector4(
            (x > 0) ? (real_t)1.0 : ((x < 0) ? (real_t)-1.0 : (real_t)0.0),
            (y > 0) ? (real_t)1.0 : ((y < 0) ? (real_t)-1.0 : (real_t)0.0),
            (z > 0) ? (real_t)1.0 : ((z < 0) ? (real_t)-1.0 : (real_t)0.0),
            (w > 0) ? (real_t)1.0 : ((w < 0) ? (real_t)-1.0 : (real_t)0.0)
        );
    }

    _FORCE_INLINE_ Vector4 floor() const {
        return Vector4(Math::floor(x), Math::floor(y), Math::floor(z), Math::floor(w));
    }

    _FORCE_INLINE_ Vector4 ceil() const {
        return Vector4(Math::ceil(x), Math::ceil(y), Math::ceil(z), Math::ceil(w));
    }

    _FORCE_INLINE_ Vector4 round() const {
        return Vector4(Math::round(x), Math::round(y), Math::round(z), Math::round(w));
    }

    _FORCE_INLINE_ Vector4 inverse() const {
        return Vector4(1.0f / x, 1.0f / y, 1.0f / z, 1.0f / w);
    }

    _FORCE_INLINE_ real_t distance_to(const Vector4 &p_to) const {
        return (p_to - *this).length();
    }

    _FORCE_INLINE_ real_t distance_squared_to(const Vector4 &p_to) const {
        return (p_to - *this).length_squared();
    }
    
    _FORCE_INLINE_ Vector4 direction_to(const Vector4 &p_to) const {
        Vector4 ret = p_to - *this;
        ret.normalize();
        return ret;
    }

    // Scalar min/max operations
    _FORCE_INLINE_ Vector4 minf(real_t p_scalar) const {
        return Vector4(
            (x < p_scalar) ? x : p_scalar,
            (y < p_scalar) ? y : p_scalar,
            (z < p_scalar) ? z : p_scalar,
            (w < p_scalar) ? w : p_scalar
        );
    }

    _FORCE_INLINE_ Vector4 maxf(real_t p_scalar) const {
        return Vector4(
            (x > p_scalar) ? x : p_scalar,
            (y > p_scalar) ? y : p_scalar,
            (z > p_scalar) ? z : p_scalar,
            (w > p_scalar) ? w : p_scalar
        );
    }

    operator String() const;
    operator Vector4i() const;
    
    // Additional utility methods
    _FORCE_INLINE_ Vector4 clamp(real_t p_min, real_t p_max) const {
        return Vector4(
            Math::clamp(x, p_min, p_max),
            Math::clamp(y, p_min, p_max),
            Math::clamp(z, p_min, p_max),
            Math::clamp(w, p_min, p_max)
        );
    }
    
    // Alias to match Vector2/3 naming convention
    _FORCE_INLINE_ Vector4 clampf(real_t p_min, real_t p_max) const {
        return clamp(p_min, p_max);
    }
    
    _FORCE_INLINE_ Vector4 clamp(const Vector4 &p_min, const Vector4 &p_max) const {
        return Vector4(
            Math::clamp(x, p_min.x, p_max.x),
            Math::clamp(y, p_min.y, p_max.y),
            Math::clamp(z, p_min.z, p_max.z),
            Math::clamp(w, p_min.w, p_max.w)
        );
    }
    
    _FORCE_INLINE_ Vector4 snapped(const Vector4 &p_step) const {
        return Vector4(
            p_step.x != 0 ? Math::floor(x / p_step.x + 0.5) * p_step.x : x,
            p_step.y != 0 ? Math::floor(y / p_step.y + 0.5) * p_step.y : y,
            p_step.z != 0 ? Math::floor(z / p_step.z + 0.5) * p_step.z : z,
            p_step.w != 0 ? Math::floor(w / p_step.w + 0.5) * p_step.w : w
        );
    }
    
    _FORCE_INLINE_ Vector4 snapped(real_t p_step) const {
        if (p_step == 0) {
            return *this;
        }
        return Vector4(
            Math::floor(x / p_step + 0.5) * p_step,
            Math::floor(y / p_step + 0.5) * p_step,
            Math::floor(z / p_step + 0.5) * p_step,
            Math::floor(w / p_step + 0.5) * p_step
        );
    }
    
    // Alias to match Vector2/3 naming convention
    _FORCE_INLINE_ Vector4 snappedf(real_t p_step) const {
        return snapped(p_step);
    }
    
    _FORCE_INLINE_ Vector4 posmod(real_t p_mod) const {
        return Vector4(
            Math::fposmod(x, p_mod),
            Math::fposmod(y, p_mod),
            Math::fposmod(z, p_mod),
            Math::fposmod(w, p_mod)
        );
    }
    
    _FORCE_INLINE_ Vector4 posmod(const Vector4 &p_mod) const {
        return Vector4(
            Math::fposmod(x, p_mod.x),
            Math::fposmod(y, p_mod.y),
            Math::fposmod(z, p_mod.z),
            Math::fposmod(w, p_mod.w)
        );
    }
    
    // Alias to match Vector2/3 naming convention
    _FORCE_INLINE_ Vector4 posmodv(const Vector4 &p_mod) const {
        return posmod(p_mod);
    }
    
    _FORCE_INLINE_ Vector4 project(const Vector4 &p_to) const {
        real_t dot_product = dot(p_to);
        real_t length_sq = p_to.length_squared();
        
        if (length_sq < CMP_EPSILON) {
            return Vector4(0, 0, 0, 0);
        }
        
        return p_to * (dot_product / length_sq);
    }
    
    _FORCE_INLINE_ bool is_equal_approx(const Vector4 &p_v) const {
        return Math::is_equal_approx(x, p_v.x) && 
               Math::is_equal_approx(y, p_v.y) && 
               Math::is_equal_approx(z, p_v.z) && 
               Math::is_equal_approx(w, p_v.w);
    }
    
    _FORCE_INLINE_ bool is_finite() const {
        return Math::is_finite(x) && Math::is_finite(y) && Math::is_finite(z) && Math::is_finite(w);
    }
};

// Global operators
#if defined(REAL_T_IS_DOUBLE)
_FORCE_INLINE_ Vector4 operator*(double p_scalar, const Vector4 &p_vec) {
    return p_vec * p_scalar;
}

_FORCE_INLINE_ Vector4 operator*(float p_scalar, const Vector4 &p_vec) {
    return p_vec * (real_t)p_scalar;
}
#else
_FORCE_INLINE_ Vector4 operator*(float p_scalar, const Vector4 &p_vec) {
    return p_vec * p_scalar;
}

_FORCE_INLINE_ Vector4 operator*(double p_scalar, const Vector4 &p_vec) {
    return p_vec * (real_t)p_scalar;
}
#endif

// Integer type scalar operators to avoid ambiguity
_FORCE_INLINE_ Vector4 operator*(int32_t p_scalar, const Vector4 &p_vec) {
    return p_vec * (real_t)p_scalar;
}

_FORCE_INLINE_ Vector4 operator*(int64_t p_scalar, const Vector4 &p_vec) {
    return p_vec * (real_t)p_scalar;
}

#endif // VECTOR4_H