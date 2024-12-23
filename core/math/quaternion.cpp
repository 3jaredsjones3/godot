#include "core/math/basis.h"
#include "core/math/quaternion.h"
#include "core/string/ustring.h"

real_t Quaternion::angle_to(const Quaternion &p_to) const {
    real_t d = Math::clamp(dot(p_to), -1.0f, 1.0f);
    return Math::acos(2.0f * d * d - 1.0f);
}

Vector3 Quaternion::get_euler(EulerOrder p_order) const {
#ifdef MATH_CHECKS
    ERR_FAIL_COND_V_MSG(!is_normalized(), Vector3(0, 0, 0), "The quaternion " + operator String() + " must be normalized.");
#endif
    return Basis(*this).get_euler(p_order);
}

#if defined(VECTOR4_USE_SSE)
Quaternion Quaternion::quaternion_mul_sse(const Quaternion &p_q) const {
    // Fast Quaternion * Quaternion using SSE vector instructions.
    // 1) Broadcast 'w' across four lanes (a_wwww) to multiply it by (x,y,z,w) of p_q.
    // 2) Shuffle other components to handle cross-product parts in parallel.
    // 3) Add or subtract them appropriately to get x, y, z, w.
    // This approach uses fewer instructions than a naive scalar loop.
    __m128 a = components.m_value;
    __m128 b = p_q.components.m_value;

    __m128 a_wwww = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3,3,3,3));
    __m128 a_vzxy = _mm_shuffle_ps(a, a, _MM_SHUFFLE(1,2,0,1));
    __m128 a_vwyz = _mm_shuffle_ps(a, a, _MM_SHUFFLE(2,1,3,2));
    
    __m128 result = _mm_mul_ps(a_wwww, b);
    
    __m128 b_yzxw = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3,0,2,1));
    __m128 b_zxyw = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3,1,0,2));
    
    __m128 part2 = _mm_mul_ps(a_vzxy, b_yzxw);
    __m128 part3 = _mm_mul_ps(a_vwyz, b_zxyw);
    
    result = _mm_add_ps(result, part2);
    result = _mm_sub_ps(result, part3);

    return Quaternion(result);
}
#endif

#if defined(VECTOR4_USE_NEON)
Quaternion Quaternion::quaternion_mul_neon(const Quaternion &p_q) const {
	// NEON-specific version of Quaternion * Quaternion, analogous to SSE code.
    // vdupq_laneq_f32() and vextq_f32() help reorder lanes for cross products.
    float32x4_t a = components.m_value;
    float32x4_t b = p_q.components.m_value;

    float32x4_t a_wwww = vdupq_laneq_f32(a, 3);
    float32x4_t a_vzxy = vextq_f32(a, a, 1);
    float32x4_t a_vwyz = vextq_f32(a, a, 2);
    
    float32x4_t result = vmulq_f32(a_wwww, b);
    
    float32x4_t b_yzxw = vextq_f32(b, b, 1);
    float32x4_t b_zxyw = vextq_f32(b, b, 2);
    
    float32x4_t part2 = vmulq_f32(a_vzxy, b_yzxw);
    float32x4_t part3 = vmulq_f32(a_vwyz, b_zxyw);
    
    result = vaddq_f32(result, part2);
    result = vsubq_f32(result, part3);

    return Quaternion(result);
}
#endif

Quaternion Quaternion::quaternion_mul_fallback(const Quaternion &p_q) const {
    return Quaternion(
		// Scalar fallback for Quaternion * Quaternion.
    	// This uses four dot products with carefully arranged vectors
    	// to compute (x, y, z, w) components without SSE/NEON instructions.
        components.dot(Vector4(p_q.x, p_q.w, p_q.z, -p_q.y)),
        components.dot(Vector4(p_q.y, p_q.x, p_q.w, -p_q.z)),
        components.dot(Vector4(p_q.z, p_q.y, p_q.w, -p_q.x)),
        components.dot(Vector4(p_q.w, -p_q.x, -p_q.y, -p_q.z))
    );
}

bool Quaternion::is_equal_approx(const Quaternion &p_quaternion) const {
    return Math::is_equal_approx(x, p_quaternion.x) && 
           Math::is_equal_approx(y, p_quaternion.y) && 
           Math::is_equal_approx(z, p_quaternion.z) && 
           Math::is_equal_approx(w, p_quaternion.w);
}

bool Quaternion::is_finite() const {
    return Math::is_finite(x) && Math::is_finite(y) && Math::is_finite(z) && Math::is_finite(w);
}

real_t Quaternion::length() const {
    return Math::sqrt(length_squared());
}

void Quaternion::normalize() {
    *this /= length();
}

Quaternion Quaternion::normalized() const {
    return *this / length();
}

bool Quaternion::is_normalized() const {
    return Math::is_equal_approx(length_squared(), (real_t)1.0, (real_t)UNIT_EPSILON);
}

Quaternion Quaternion::inverse() const {
#ifdef MATH_CHECKS
    ERR_FAIL_COND_V_MSG(!is_normalized(), Quaternion(), "The quaternion " + operator String() + " must be normalized.");
#endif
    return Quaternion(-x, -y, -z, w);
}

Quaternion Quaternion::log() const {
    // log(q) for a quaternion q = exp(v), is basically v = axis * angle,
    // where axis is the unit vector and angle is the rotation magnitude.
    // We store that in (x, y, z) and set w to 0.
    Quaternion src = *this;
    Vector3 src_v = src.get_axis() * src.get_angle();
    return Quaternion(src_v.x, src_v.y, src_v.z, 0);
}

Quaternion Quaternion::exp() const {
    // exp(q) interprets the (x, y, z) as an axis scaled by the angle.
    // If angle is near zero, return identity. Otherwise, reconstruct the axis-angle.
    Vector3 src_v = Vector3(x, y, z);
    real_t theta = src_v.length();
    src_v = src_v.normalized();
    if (theta < CMP_EPSILON || !src_v.is_normalized()) {
        return Quaternion(0, 0, 0, 1);
    }
    return Quaternion(src_v, theta);
}

Quaternion Quaternion::slerp(const Quaternion &p_to, real_t p_weight) const {
#ifdef MATH_CHECKS
    ERR_FAIL_COND_V_MSG(!is_normalized(), Quaternion(), "The start quaternion " + operator String() + " must be normalized.");
    ERR_FAIL_COND_V_MSG(!p_to.is_normalized(), Quaternion(), "The end quaternion " + p_to.operator String() + " must be normalized.");
#endif
    Quaternion to1;
    real_t omega, cosom, sinom, scale0, scale1;

    cosom = dot(p_to);
    if (cosom < 0.0f) {
        cosom = -cosom;
        to1 = -p_to;
    } else {
        to1 = p_to;
    }

    if ((1.0f - cosom) > CMP_EPSILON) {
        omega = Math::acos(cosom);
        sinom = Math::sin(omega);
        scale0 = Math::sin((1.0f - p_weight) * omega) / sinom;
        scale1 = Math::sin(p_weight * omega) / sinom;
    } else {
        scale0 = 1.0f - p_weight;
        scale1 = p_weight;
    }

    return Quaternion(
        scale0 * x + scale1 * to1.x,
        scale0 * y + scale1 * to1.y,
        scale0 * z + scale1 * to1.z,
        scale0 * w + scale1 * to1.w);
}

Quaternion Quaternion::slerpni(const Quaternion &p_to, real_t p_weight) const {
#ifdef MATH_CHECKS
    ERR_FAIL_COND_V_MSG(!is_normalized(), Quaternion(), "The start quaternion " + operator String() + " must be normalized.");
    ERR_FAIL_COND_V_MSG(!p_to.is_normalized(), Quaternion(), "The end quaternion " + p_to.operator String() + " must be normalized.");
#endif
    const Quaternion &from = *this;
    real_t dot = from.dot(p_to);

    if (Math::absf(dot) > 0.9999f) {
        return from;
    }

    real_t theta = Math::acos(dot);
    real_t sinT = 1.0f / Math::sin(theta);
    real_t newFactor = Math::sin(p_weight * theta) * sinT;
    real_t invFactor = Math::sin((1.0f - p_weight) * theta) * sinT;

    return Quaternion(
        invFactor * from.x + newFactor * p_to.x,
        invFactor * from.y + newFactor * p_to.y,
        invFactor * from.z + newFactor * p_to.z,
        invFactor * from.w + newFactor * p_to.w);
}

Quaternion Quaternion::spherical_cubic_interpolate(const Quaternion &p_b, 
    const Quaternion &p_pre_a, const Quaternion &p_post_b, real_t p_weight) const {
#ifdef MATH_CHECKS
	// The quaternions must be normalized for a correct spherical interpolation.
    ERR_FAIL_COND_V_MSG(!is_normalized(), Quaternion(), "The start quaternion " + operator String() + " must be normalized.");
    ERR_FAIL_COND_V_MSG(!p_b.is_normalized(), Quaternion(), "The end quaternion " + p_b.operator String() + " must be normalized.");
#endif
    // 1) Convert from_q, pre_q, to_q, post_q to a consistent orientation using Basis::get_rotation_quaternion().
    // 2) Check dot products to see if flipping is needed (signbit) to ensure we follow the shortest path.
    // 3) Logarithms (log) and exponentials (exp) of quaternions are used to do the 'cubic' part of the interpolation
    //    in an exponential map, then we blend them and finally slerp between them to reduce ambiguity.
    Quaternion from_q = *this;
    Quaternion pre_q = p_pre_a;
    Quaternion to_q = p_b;
    Quaternion post_q = p_post_b;

    // Align flip phases.
    from_q = Basis(from_q).get_rotation_quaternion();
    pre_q = Basis(pre_q).get_rotation_quaternion();
    to_q = Basis(to_q).get_rotation_quaternion();
    post_q = Basis(post_q).get_rotation_quaternion();

    // Flip quaternions to shortest path if necessary.
    bool flip1 = signbit(from_q.dot(pre_q));
    pre_q = flip1 ? -pre_q : pre_q;
    bool flip2 = signbit(from_q.dot(to_q));
    to_q = flip2 ? -to_q : to_q;
    bool flip3 = flip2 ? to_q.dot(post_q) <= 0 : signbit(to_q.dot(post_q));
    post_q = flip3 ? -post_q : post_q;

    // Calc by Expmap in from_q space.
    Quaternion ln_from = Quaternion(0, 0, 0, 0);
    Quaternion ln_to = (from_q.inverse() * to_q).log();
    Quaternion ln_pre = (from_q.inverse() * pre_q).log();
    Quaternion ln_post = (from_q.inverse() * post_q).log();
    Quaternion ln = Quaternion(0, 0, 0, 0);
    ln.x = Math::cubic_interpolate(ln_from.x, ln_to.x, ln_pre.x, ln_post.x, p_weight);
    ln.y = Math::cubic_interpolate(ln_from.y, ln_to.y, ln_pre.y, ln_post.y, p_weight);
    ln.z = Math::cubic_interpolate(ln_from.z, ln_to.z, ln_pre.z, ln_post.z, p_weight);
    Quaternion q1 = from_q * ln.exp();

    // Calc by Expmap in to_q space.
    ln_from = (to_q.inverse() * from_q).log();
    ln_to = Quaternion(0, 0, 0, 0);
    ln_pre = (to_q.inverse() * pre_q).log();
    ln_post = (to_q.inverse() * post_q).log();
    ln = Quaternion(0, 0, 0, 0);
    ln.x = Math::cubic_interpolate(ln_from.x, ln_to.x, ln_pre.x, ln_post.x, p_weight);
    ln.y = Math::cubic_interpolate(ln_from.y, ln_to.y, ln_pre.y, ln_post.y, p_weight);
    ln.z = Math::cubic_interpolate(ln_from.z, ln_to.z, ln_pre.z, ln_post.z, p_weight);
    Quaternion q2 = to_q * ln.exp();

    return q1.slerp(q2, p_weight);
}

Quaternion Quaternion::spherical_cubic_interpolate_in_time(const Quaternion &p_b, 
    const Quaternion &p_pre_a, const Quaternion &p_post_b, real_t p_weight,
    real_t p_b_t, real_t p_pre_a_t, real_t p_post_b_t) const {
#ifdef MATH_CHECKS
    ERR_FAIL_COND_V_MSG(!is_normalized(), Quaternion(), "The start quaternion " + operator String() + " must be normalized.");
    ERR_FAIL_COND_V_MSG(!p_b.is_normalized(), Quaternion(), "The end quaternion " + p_b.operator String() + " must be normalized.");
#endif
    Quaternion from_q = *this;
    Quaternion pre_q = p_pre_a;
    Quaternion to_q = p_b;
    Quaternion post_q = p_post_b;

    // Align flip phases.
    from_q = Basis(from_q).get_rotation_quaternion();
    pre_q = Basis(pre_q).get_rotation_quaternion();
    to_q = Basis(to_q).get_rotation_quaternion();
    post_q = Basis(post_q).get_rotation_quaternion();

    // Flip quaternions to shortest path if necessary.
    bool flip1 = signbit(from_q.dot(pre_q));
    pre_q = flip1 ? -pre_q : pre_q;
    bool flip2 = signbit(from_q.dot(to_q));
    to_q = flip2 ? -to_q : to_q;
    bool flip3 = flip2 ? to_q.dot(post_q) <= 0 : signbit(to_q.dot(post_q));
    post_q = flip3 ? -post_q : post_q;

    // Calc by Expmap in from_q space.
    Quaternion ln_from = Quaternion(0, 0, 0, 0);
    Quaternion ln_to = (from_q.inverse() * to_q).log();
    Quaternion ln_pre = (from_q.inverse() * pre_q).log();
    Quaternion ln_post = (from_q.inverse() * post_q).log();
    Quaternion ln = Quaternion(0, 0, 0, 0);
ln.x = Math::cubic_interpolate_in_time(ln_from.x, ln_to.x, ln_pre.x, ln_post.x, 
        p_weight, p_b_t, p_pre_a_t, p_post_b_t);
    ln.y = Math::cubic_interpolate_in_time(ln_from.y, ln_to.y, ln_pre.y, ln_post.y, 
        p_weight, p_b_t, p_pre_a_t, p_post_b_t);
    ln.z = Math::cubic_interpolate_in_time(ln_from.z, ln_to.z, ln_pre.z, ln_post.z, 
        p_weight, p_b_t, p_pre_a_t, p_post_b_t);
    Quaternion q1 = from_q * ln.exp();

    // Calc by Expmap in to_q space.
    ln_from = (to_q.inverse() * from_q).log();
    ln_to = Quaternion(0, 0, 0, 0);
    ln_pre = (to_q.inverse() * pre_q).log();
    ln_post = (to_q.inverse() * post_q).log();
    ln = Quaternion(0, 0, 0, 0);
    ln.x = Math::cubic_interpolate_in_time(ln_from.x, ln_to.x, ln_pre.x, ln_post.x, 
        p_weight, p_b_t, p_pre_a_t, p_post_b_t);
    ln.y = Math::cubic_interpolate_in_time(ln_from.y, ln_to.y, ln_pre.y, ln_post.y, 
        p_weight, p_b_t, p_pre_a_t, p_post_b_t);
    ln.z = Math::cubic_interpolate_in_time(ln_from.z, ln_to.z, ln_pre.z, ln_post.z, 
        p_weight, p_b_t, p_pre_a_t, p_post_b_t);
    Quaternion q2 = to_q * ln.exp();

    return q1.slerp(q2, p_weight);
}

Quaternion::operator String() const {
    return String("{ x: ") + String::num(x) +
           String(", y: ") + String::num(y) +
           String(", z: ") + String::num(z) +
           String(", w: ") + String::num(w) + String(" }");
}

Vector3 Quaternion::get_axis() const {
    if (Math::abs(w) > 1 - CMP_EPSILON) {
        return Vector3(x, y, z);
    }
    real_t r = ((real_t)1) / Math::sqrt(1 - w * w);
    return Vector3(x * r, y * r, z * r);
}

real_t Quaternion::get_angle() const {
    return 2 * Math::acos(w);
}

Quaternion::Quaternion(const Vector3 &p_axis, real_t p_angle) {
#ifdef MATH_CHECKS
    ERR_FAIL_COND_MSG(!p_axis.is_normalized(), "The axis Vector3 " + p_axis.operator String() + " must be normalized.");
#endif
    real_t d = p_axis.length();
    if (d == 0) {
        x = 0;
        y = 0;
        z = 0;
        w = 0;
    } else {
        real_t sin_angle = Math::sin(p_angle * 0.5f);
        real_t cos_angle = Math::cos(p_angle * 0.5f);
        real_t s = sin_angle / d;
        x = p_axis.x * s;
        y = p_axis.y * s;
        z = p_axis.z * s;
        w = cos_angle;
    }
}

Quaternion Quaternion::from_euler(const Vector3 &p_euler) {
    real_t half_a1 = p_euler.y * 0.5f;
    real_t half_a2 = p_euler.x * 0.5f;
    real_t half_a3 = p_euler.z * 0.5f;

    real_t cos_a1 = Math::cos(half_a1);
    real_t sin_a1 = Math::sin(half_a1);
    real_t cos_a2 = Math::cos(half_a2);
    real_t sin_a2 = Math::sin(half_a2);
    real_t cos_a3 = Math::cos(half_a3);
    real_t sin_a3 = Math::sin(half_a3);

    return Quaternion(
        sin_a1 * cos_a2 * sin_a3 + cos_a1 * sin_a2 * cos_a3,
        sin_a1 * cos_a2 * cos_a3 - cos_a1 * sin_a2 * sin_a3,
        -sin_a1 * sin_a2 * cos_a3 + cos_a1 * cos_a2 * sin_a3,
        sin_a1 * sin_a2 * sin_a3 + cos_a1 * cos_a2 * cos_a3);
}