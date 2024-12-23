#include "vector3SIMD.h"
#include "vector3.h"

Vector3SIMD::Vector3SIMD(const Vector3& p_v) {
#if defined(VECTOR3SIMD_USE_SSE)
    m_value = _mm_set_ps(0.0f, p_v.z, p_v.y, p_v.x);
#elif defined(VECTOR3SIMD_USE_NEON)
    float temp[4] = {p_v.x, p_v.y, p_v.z, 0.0f};
    m_value = vld1q_f32(temp);
#else
    f[0] = p_v.x;
    f[1] = p_v.y;
    f[2] = p_v.z;
    f[3] = 0.0f;
#endif
}

Vector3SIMD::operator Vector3() const {
    return Vector3(f[0], f[1], f[2]);
}