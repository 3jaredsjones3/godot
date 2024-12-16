#include "vector3.h"
#include "core/math/math_funcs.h"
#include "core/error/error_macros.h"
#include "typedefs.h"

// Constructor Definitions
Vector3::Vector3() : x(0), y(0), z(0) {}
Vector3::Vector3(real_t p_x, real_t p_y, real_t p_z) : x(p_x), y(p_y), z(p_z) {}

// Length and Normalization
real_t Vector3::length() const {
    return Math::sqrt(x * x + y * y + z * z);
}

real_t Vector3::length_squared() const {
    return x * x + y * y + z * z;
}

void Vector3::normalize() {
    real_t lengthsq = length_squared();
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
    return Math::is_equal_approx(length_squared(), 1, (real_t)UNIT_EPSILON);
}

// Vector Arithmetic
Vector3 Vector3::operator+(const Vector3 &p_v) const {
    return Vector3(x + p_v.x, y + p_v.y, z + p_v.z);
}

Vector3 &Vector3::operator+=(const Vector3 &p_v) {
    x += p_v.x;
    y += p_v.y;
    z += p_v.z;
    return *this;
}

Vector3 Vector3::operator-(const Vector3 &p_v) const {
    return Vector3(x - p_v.x, y - p_v.y, z - p_v.z);
}

Vector3 &Vector3::operator-=(const Vector3 &p_v) {
    x -= p_v.x;
    y -= p_v.y;
    z -= p_v.z;
    return *this;
}

Vector3 Vector3::operator*(real_t p_scalar) const {
    return Vector3(x * p_scalar, y * p_scalar, z * p_scalar);
}

Vector3 &Vector3::operator*=(real_t p_scalar) {
    x *= p_scalar;
    y *= p_scalar;
    z *= p_scalar;
    return *this;
}

Vector3 Vector3::operator/(real_t p_scalar) const {
    return Vector3(x / p_scalar, y / p_scalar, z / p_scalar);
}

Vector3 &Vector3::operator/=(real_t p_scalar) {
    x /= p_scalar;
    y /= p_scalar;
    z /= p_scalar;
    return *this;
}

// Dot and Cross Product
real_t Vector3::dot(const Vector3 &p_with) const {
    return x * p_with.x + y * p_with.y + z * p_with.z;
}

Vector3 Vector3::cross(const Vector3 &p_with) const {
    return Vector3(
        (y * p_with.z) - (z * p_with.y),
        (z * p_with.x) - (x * p_with.z),
        (x * p_with.y) - (y * p_with.x)
    );
}

// Additional Vector Operations
Vector3 Vector3::abs() const {
    return Vector3(Math::abs(x), Math::abs(y), Math::abs(z));
}

Vector3 Vector3::floor() const {
    return Vector3(Math::floor(x), Math::floor(y), Math::floor(z));
}

Vector3 Vector3::ceil() const {
    return Vector3(Math::ceil(x), Math::ceil(y), Math::ceil(z));
}

Vector3 Vector3::clamp(const Vector3 &p_min, const Vector3 &p_max) const {
    return Vector3(
        CLAMP(x, p_min.x, p_max.x),
        CLAMP(y, p_min.y, p_max.y),
        CLAMP(z, p_min.z, p_max.z)
    );
}

real_t Vector3::distance_to(const Vector3 &p_to) const {
    return (p_to - *this).length();
}

real_t Vector3::distance_squared_to(const Vector3 &p_to) const {
    return (p_to - *this).length_squared();
}

// Zero and Approximation
void Vector3::zero() {
    x = y = z = 0;
}

bool Vector3::is_zero_approx() const {
    return Math::is_zero_approx(x) && Math::is_zero_approx(y) && Math::is_zero_approx(z);
}

// Reflection and Projection
Vector3 Vector3::reflect(const Vector3 &p_normal) const {
    return 2.0f * p_normal * dot(p_normal) - *this;
}

Vector3 Vector3::project(const Vector3 &p_to) const {
    return p_to * (dot(p_to) / p_to.length_squared());
}
