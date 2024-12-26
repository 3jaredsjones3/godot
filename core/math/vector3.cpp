#include "core/math/vector3.h"
#include "core/math/basis.h"
#include <cstdio> // for snprintf

Vector3::operator Vector3i() const { 
    return Vector3i(x, y, z); 
}

Vector3::operator String() const {
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "(%f, %f, %f)", x, y, z);
    return String(buffer);
}

Basis Vector3::outer(const Vector3& p_with) const {
#if defined(VECTOR3SIMD_USE_SSE)
    __m128 this_x = _mm_set1_ps(x);
    __m128 this_y = _mm_set1_ps(y);
    __m128 this_z = _mm_set1_ps(z);

    __m128 with_vec = p_with.m_value;
    Vector3 row1(_mm_mul_ps(this_x, with_vec));
    Vector3 row2(_mm_mul_ps(this_y, with_vec));
    Vector3 row3(_mm_mul_ps(this_z, with_vec));
    return Basis(row1, row2, row3);

#elif defined(VECTOR3SIMD_USE_NEON)
    float32x4_t this_x = vdupq_n_f32(x);
    float32x4_t this_y = vdupq_n_f32(y);
    float32x4_t this_z = vdupq_n_f32(z);

    float32x4_t with_vec = p_with.m_value;
    Vector3 row1(vmulq_f32(this_x, with_vec));
    Vector3 row2(vmulq_f32(this_y, with_vec));
    Vector3 row3(vmulq_f32(this_z, with_vec));
    return Basis(row1, row2, row3);

#else
    return Basis(
        Vector3(x * p_with.x, x * p_with.y, x * p_with.z),
        Vector3(y * p_with.x, y * p_with.y, y * p_with.z),
        Vector3(z * p_with.x, z * p_with.y, z * p_with.z)
    );
#endif
}

const Vector3 Vector3::ZERO = Vector3(0.0f, 0.0f, 0.0f);
const Vector3 Vector3::ONE = Vector3(1.0f, 1.0f, 1.0f);
const Vector3 Vector3::LEFT = Vector3(-1.0f, 0.0f, 0.0f);
const Vector3 Vector3::RIGHT = Vector3(1.0f, 0.0f, 0.0f);
const Vector3 Vector3::UP = Vector3(0.0f, 1.0f, 0.0f);
const Vector3 Vector3::DOWN = Vector3(0.0f, -1.0f, 0.0f);
const Vector3 Vector3::FORWARD = Vector3(0.0f, 0.0f, 1.0f);
const Vector3 Vector3::BACK = Vector3(0.0f, 0.0f, -1.0f);
