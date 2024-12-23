#ifndef VECTOR3_SIMD_H
#define VECTOR3_SIMD_H

#include <cmath>
#include <cstddef>  // For alignas
#include <cstdint>

#include "core/math/math_defs.h"
#include "core/math/math_funcs.h"
#include "core/typedefs.h"
#include "vector2.h"

// Forward declarations
struct Vector3;
struct Vector2;
struct Quaternion;

// SSE headers
#if defined(__SSE__) || (defined(_M_X64) && !defined(__EMSCRIPTEN__))
#define VECTOR3SIMD_USE_SSE
#include <emmintrin.h>  // SSE2
#include <xmmintrin.h>  // SSE
// ... rest of SSE includes
#endif

// NEON headers
#if defined(__ARM_NEON) || defined(__aarch64__)
#define VECTOR3SIMD_USE_NEON
#include <arm_neon.h>
#endif

struct Vector3;  // Forward declaration for conversions

struct alignas(16) Vector3SIMD {
     union {
#if defined(VECTOR3SIMD_USE_SSE)
          __m128 m_value;
#elif defined(VECTOR3SIMD_USE_NEON)
          float32x4_t m_value;
#endif
          float f[4];
     };

// SSE Helper
#if defined(VECTOR3SIMD_USE_SSE)
     static _FORCE_INLINE_ __m128 load_scalar(float s) {
          return _mm_set1_ps(s);
     }
#endif

// NEON Helper
#if defined(VECTOR3SIMD_USE_NEON)
     static _FORCE_INLINE_ float32x4_t load_scalar_neon(float s) {
          return vdupq_n_f32(s);
     }
#endif

    public:
     // Basic constructors
     _FORCE_INLINE_ Vector3SIMD() {
#if defined(VECTOR3SIMD_USE_SSE)
          m_value = _mm_setzero_ps();
#elif defined(VECTOR3SIMD_USE_NEON)
          m_value = vdupq_n_f32(0.0f);
#else
          f[0] = f[1] = f[2] = f[3] = 0.0f;
#endif
     }

     // Conversion to/from Godot's Vector3
     _FORCE_INLINE_ Vector3SIMD(const Vector3 &p_v) {
#if defined(VECTOR3SIMD_USE_SSE)
          m_value =
              _mm_set_ps(0.0f, p_v.z, p_v.y, p_v.x);  // Corrected arguments
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

     _FORCE_INLINE_ operator Vector3() const {
          return Vector3(f[0], f[1], f[2]);
     }

     _FORCE_INLINE_ Vector3SIMD(real_t p_x, real_t p_y, real_t p_z) {
#if defined(VECTOR3SIMD_USE_SSE)
          m_value = _mm_set_ps(0.0f, float(p_z), float(p_y), float(p_x));
#elif defined(VECTOR3SIMD_USE_NEON)
          m_value = (float32x4_t){float(p_x), float(p_y), float(p_z), 0.0f};
#else
          f[0] = float(p_x);
          f[1] = float(p_y);
          f[2] = float(p_z);
          f[3] = 0.0f;
#endif
     }

     _FORCE_INLINE_ explicit Vector3SIMD(const real_t *p_array) {
#if defined(VECTOR3SIMD_USE_SSE)
          float temp[4] = {float(p_array[0]), float(p_array[1]),
                           float(p_array[2]), 0.0f};
          m_value = _mm_loadu_ps(temp);
#elif defined(VECTOR3SIMD_USE_NEON)
          m_value = (float32x4_t){float(p_array[0]), float(p_array[1]),
                                  float(p_array[2]), 0.0f};
#else
          f[0] = float(p_array[0]);
          f[1] = float(p_array[1]);
          f[2] = float(p_array[2]);
          f[3] = 0.0f;
#endif
     }

#if defined(VECTOR3SIMD_USE_SSE)
     _FORCE_INLINE_ Vector3SIMD(__m128 val) : m_value(val) {}
#endif

#if defined(VECTOR3SIMD_USE_NEON)
     _FORCE_INLINE_ Vector3SIMD(float32x4_t val) : m_value(val) {}
#endif

     // Copy constructor and assignment
     _FORCE_INLINE_ Vector3SIMD(const Vector3SIMD &p_other) {
          m_value = p_other.m_value;
     }

     _FORCE_INLINE_ Vector3SIMD &operator=(const Vector3SIMD &p_other) {
          m_value = p_other.m_value;
          return *this;
     }

     // Component access
     _FORCE_INLINE_ real_t x() const { return real_t(f[0]); }
     _FORCE_INLINE_ real_t y() const { return real_t(f[1]); }
     _FORCE_INLINE_ real_t z() const { return real_t(f[2]); }
     _FORCE_INLINE_ real_t w() const { return real_t(f[3]); }

     _FORCE_INLINE_ void set_x(real_t p_x) { f[0] = float(p_x); }
     _FORCE_INLINE_ void set_y(real_t p_y) { f[1] = float(p_y); }
     _FORCE_INLINE_ void set_z(real_t p_z) { f[2] = float(p_z); }

// SSE Operations
#if defined(VECTOR3SIMD_USE_SSE)
     _FORCE_INLINE_ Vector3SIMD add_sse(const Vector3SIMD &p_v) const {
          return Vector3SIMD(_mm_add_ps(m_value, p_v.m_value));
     }

     _FORCE_INLINE_ Vector3SIMD sub_sse(const Vector3SIMD &p_v) const {
          return Vector3SIMD(_mm_sub_ps(m_value, p_v.m_value));
     }

     _FORCE_INLINE_ float dot_sse(const Vector3SIMD &p_v) const {
#if defined(__SSE4_1__)
          return _mm_cvtss_f32(_mm_dp_ps(m_value, p_v.m_value, 0x7F));
#else
          __m128 mul = _mm_mul_ps(m_value, p_v.m_value);
          __m128 shuf = _mm_movehdup_ps(mul);
          __m128 sums = _mm_add_ps(mul, shuf);
          shuf = _mm_movehl_ps(shuf, sums);
          sums = _mm_add_ss(sums, shuf);
          return _mm_cvtss_f32(sums);
#endif
     }

     _FORCE_INLINE_ Vector3SIMD cross_sse(const Vector3SIMD &p_v) const {
          __m128 a = m_value;
          __m128 b = p_v.m_value;
          __m128 a_yzx = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1));
          __m128 b_yzx = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1));
          __m128 c = _mm_sub_ps(_mm_mul_ps(a, b_yzx), _mm_mul_ps(a_yzx, b));
          return Vector3SIMD(_mm_shuffle_ps(c, c, _MM_SHUFFLE(3, 0, 2, 1)));
     }

     _FORCE_INLINE_ float length_squared_sse() const { return dot_sse(*this); }

     _FORCE_INLINE_ float length_sse() const {
          return _mm_cvtss_f32(_mm_sqrt_ss(_mm_dp_ps(m_value, m_value, 0x7F)));
     }

     _FORCE_INLINE_ Vector3SIMD normalize_sse() const {
          float l = length_sse();
          if (l == 0) {
               return Vector3SIMD();
          }
          return Vector3SIMD(_mm_div_ps(m_value, _mm_set1_ps(l)));
     }

     _FORCE_INLINE_ bool is_normalized_sse() const {
          // Calculate length squared using dot product with self
          __m128 len_sq = _mm_dp_ps(m_value, m_value, 0x7F);

          // Compare with 1 within epsilon
          __m128 one = _mm_set1_ps(1.0f);
          __m128 epsilon = _mm_set1_ps(UNIT_EPSILON);
          __m128 diff = _mm_sub_ps(len_sq, one);
          __m128 abs_diff =
              _mm_andnot_ps(_mm_set1_ps(-0.0f), diff);  // absolute value
          __m128 cmp = _mm_cmple_ps(abs_diff, epsilon);

          // Check if difference is within epsilon
          return (_mm_movemask_ps(cmp) & 0x1) == 0x1;
     }

     _FORCE_INLINE_ Vector3SIMD mul_sse(const Vector3SIMD &p_v) const {
          return Vector3SIMD(_mm_mul_ps(m_value, p_v.m_value));
     }

     _FORCE_INLINE_ Vector3SIMD div_sse(const Vector3SIMD &p_v) const {
          return Vector3SIMD(_mm_div_ps(m_value, p_v.m_value));
     }

     _FORCE_INLINE_ Vector3SIMD mul_scalar_sse(float p_scalar) const {
          return Vector3SIMD(_mm_mul_ps(m_value, load_scalar(p_scalar)));
     }

     _FORCE_INLINE_ Vector3SIMD div_scalar_sse(float p_scalar) const {
          return Vector3SIMD(_mm_div_ps(m_value, load_scalar(p_scalar)));
     }

     _FORCE_INLINE_ Vector3SIMD abs_sse() const {
          return Vector3SIMD(_mm_andnot_ps(_mm_set1_ps(-0.0f), m_value));
     }

     _FORCE_INLINE_ Vector3SIMD neg_sse() const {
          return Vector3SIMD(_mm_xor_ps(m_value, _mm_set1_ps(-0.0f)));
     }

     _FORCE_INLINE_ Vector3SIMD min_sse(const Vector3SIMD &p_v) const {
          return Vector3SIMD(_mm_min_ps(m_value, p_v.m_value));
     }

     _FORCE_INLINE_ Vector3SIMD max_sse(const Vector3SIMD &p_v) const {
          return Vector3SIMD(_mm_max_ps(m_value, p_v.m_value));
     }

     _FORCE_INLINE_ Vector3SIMD snap_sse(const Vector3SIMD &p_step) {
          // For each component: result = Math::snapped(value, step)
          // snapped = round(value/step) * step
          Vector3SIMD result = *this;
          __m128 step = p_step.m_value;
          __m128 div = _mm_div_ps(m_value, step);
          __m128 rounded =
              _mm_round_ps(div, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
          result.m_value = _mm_mul_ps(rounded, step);
          return result;
     }

     _FORCE_INLINE_ Vector3SIMD snapped_sse(const Vector3SIMD &p_step) const {
          Vector3SIMD v = *this;
          v.snap_sse(p_step);
          return v;
     }

     _FORCE_INLINE_ Vector3SIMD snapf_sse(real_t p_step) {
          // Create vector with all components equal to p_step
          Vector3SIMD result = *this;
          __m128 step = _mm_set1_ps(p_step);
          __m128 div = _mm_div_ps(m_value, step);
          __m128 rounded =
              _mm_round_ps(div, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
          result.m_value = _mm_mul_ps(rounded, step);
          return result;
     }

     _FORCE_INLINE_ Vector3SIMD snappedf_sse(real_t p_step) const {
          Vector3SIMD v = *this;
          v.snapf_sse(p_step);
          return v;
     }

     _FORCE_INLINE_ Vector3SIMD reflect_sse(const Vector3SIMD &p_normal) const {
          __m128 dot = _mm_dp_ps(m_value, p_normal.m_value, 0x7F);
          __m128 scale = _mm_mul_ps(dot, _mm_set1_ps(2.0f));
          __m128 proj = _mm_mul_ps(p_normal.m_value, scale);
          return Vector3SIMD(_mm_sub_ps(m_value, proj));
     }

     _FORCE_INLINE_ Vector3SIMD project_sse(const Vector3SIMD &p_to) const {
          __m128 dot = _mm_dp_ps(m_value, p_to.m_value, 0x7F);
          __m128 len_sq = _mm_dp_ps(p_to.m_value, p_to.m_value, 0x7F);
          __m128 scale = _mm_div_ps(dot, len_sq);
          return Vector3SIMD(_mm_mul_ps(p_to.m_value, scale));
     }

     _FORCE_INLINE_ Vector3SIMD slide_sse(const Vector3SIMD &p_normal) const {
          __m128 dot = _mm_dp_ps(m_value, p_normal.m_value, 0x7F);
          __m128 proj = _mm_mul_ps(p_normal.m_value, dot);
          return Vector3SIMD(_mm_sub_ps(m_value, proj));
     }

     _FORCE_INLINE_ Vector3SIMD bounce_sse(const Vector3SIMD &p_normal) {
          __m128 dot = _mm_dp_ps(m_value, p_normal.m_value, 0x7F);
          __m128 scale = _mm_mul_ps(dot, _mm_set1_ps(2.0f));
          __m128 proj = _mm_mul_ps(p_normal.m_value, scale);
          return Vector3SIMD(_mm_sub_ps(proj, m_value));
     }

     _FORCE_INLINE_ void rotate_sse(const Vector3SIMD &p_axis, real_t p_angle) {
          // Ensure axis is normalized
          Vector3SIMD axis = p_axis;
          if (!axis.is_normalized_sse()) {
               axis = axis.normalize_sse();
          }

          float s = Math::sin(p_angle);
          float c = Math::cos(p_angle);
          float k = 1.0f - c;

          float x = axis.x();
          float y = axis.y();
          float z = axis.z();

          // Build rotation matrix components
          __m128 row1 = _mm_setr_ps(x * x * k + c, x * y * k - z * s,
                                    x * z * k + y * s, 0.0f);

          __m128 row2 = _mm_setr_ps(y * x * k + z * s, y * y * k + c,
                                    y * z * k - x * s, 0.0f);

          __m128 row3 = _mm_setr_ps(z * x * k - y * s, z * y * k + x * s,
                                    z * z * k + c, 0.0f);

          // Transform the vector
          __m128 result = _mm_add_ps(
              _mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(m_value, m_value,
                                                   _MM_SHUFFLE(0, 0, 0, 0)),
                                    row1),
                         _mm_mul_ps(_mm_shuffle_ps(m_value, m_value,
                                                   _MM_SHUFFLE(1, 1, 1, 1)),
                                    row2)),
              _mm_mul_ps(
                  _mm_shuffle_ps(m_value, m_value, _MM_SHUFFLE(2, 2, 2, 2)),
                  row3));

          m_value = result;
     }

     _FORCE_INLINE_ Vector3SIMD rotated_sse(const Vector3SIMD &p_axis,
                                            real_t p_angle) const {
          Vector3SIMD result = *this;
          result.rotate_sse(p_axis, p_angle);
          return result;
     }

     _FORCE_INLINE_ float distance_to_sse(const Vector3SIMD &p_to) const {
          Vector3SIMD diff = sub_sse(p_to);
          return diff.length_sse();
     }

     _FORCE_INLINE_ float distance_squared_to_sse(
         const Vector3SIMD &p_to) const {
          Vector3SIMD diff = sub_sse(p_to);
          return diff.length_squared_sse();
     }

     _FORCE_INLINE_ Vector3SIMD move_toward_sse(const Vector3SIMD &p_to,
                                                real_t p_delta) const {
          // Calculate vector difference
          __m128 diff = _mm_sub_ps(p_to.m_value, m_value);

          // Calculate length of diff vector using dot product with itself
          __m128 len_sq = _mm_dp_ps(diff, diff, 0x7F);
          __m128 len = _mm_sqrt_ps(len_sq);

          // Check if length <= delta
          __m128 delta = _mm_set1_ps(p_delta);
          __m128 cmp = _mm_cmple_ps(len, delta);

          // If length <= delta, return target
          if (_mm_movemask_ps(cmp) & 0x1) {
               return p_to;
          }

          // Calculate scale factor (delta / length)
          __m128 scale = _mm_div_ps(delta, len);

          // Scale the difference vector and add to original position
          __m128 scaled_diff = _mm_mul_ps(diff, scale);
          return Vector3SIMD(_mm_add_ps(m_value, scaled_diff));
     }

     _FORCE_INLINE_ Vector3SIMD lerp_sse(const Vector3SIMD &p_to,
                                         float p_weight) const {
          __m128 w = _mm_set1_ps(p_weight);
          __m128 inv_w = _mm_sub_ps(_mm_set1_ps(1.0f), w);
          return Vector3SIMD(_mm_add_ps(_mm_mul_ps(m_value, inv_w),
                                        _mm_mul_ps(p_to.m_value, w)));
     }

     _FORCE_INLINE_ Vector3SIMD
     cubic_interpolate_sse(const Vector3SIMD &p_b, const Vector3SIMD &p_pre_a,
                           const Vector3SIMD &p_post_b, float p_weight) const {
          __m128 w = _mm_set1_ps(p_weight);
          __m128 w2 = _mm_mul_ps(w, w);
          __m128 w3 = _mm_mul_ps(w2, w);

          __m128 a0 = _mm_sub_ps(p_b.m_value, m_value);
          __m128 a1 = _mm_sub_ps(p_pre_a.m_value, m_value);
          __m128 a2 = _mm_add_ps(_mm_sub_ps(a0, a1),
                                 _mm_sub_ps(p_post_b.m_value, p_b.m_value));

          __m128 c0 = m_value;
          __m128 c1 = _mm_mul_ps(a1, _mm_set1_ps(0.5f));
          __m128 c2 = _mm_mul_ps(a0, _mm_set1_ps(2.0f));
          __m128 c3 = _mm_mul_ps(a2, _mm_set1_ps(0.5f));

          return Vector3SIMD(_mm_add_ps(
              _mm_add_ps(_mm_add_ps(c0, _mm_mul_ps(c1, w)), _mm_mul_ps(c2, w2)),
              _mm_mul_ps(c3, w3)));
     }

     _FORCE_INLINE_ Vector3SIMD slerp_sse(const Vector3SIMD &p_to,
                                          float p_weight) const {
          float start_length_sq = length_squared_sse();
          float end_length_sq = p_to.length_squared_sse();

          if (start_length_sq == 0.0f || end_length_sq == 0.0f) {
               return lerp_sse(p_to, p_weight);
          }

          Vector3SIMD axis = cross_sse(p_to);
          float axis_length_sq = axis.length_squared_sse();

          if (axis_length_sq == 0.0f) {
               return lerp_sse(p_to, p_weight);
          }

          float angle =
              acos(dot_sse(p_to) / (sqrt(start_length_sq * end_length_sq)));
          if (angle == 0.0f) {
               return *this;
          }

          float sin_angle = sin(angle);
          float scale1 = sin((1.0f - p_weight) * angle) / sin_angle;
          float scale2 = sin(p_weight * angle) / sin_angle;

          __m128 v1 = _mm_mul_ps(m_value, _mm_set1_ps(scale1));
          __m128 v2 = _mm_mul_ps(p_to.m_value, _mm_set1_ps(scale2));
          return Vector3SIMD(_mm_add_ps(v1, v2));
     }

     _FORCE_INLINE_ bool is_zero_approx_sse() const {
          __m128 epsilon = _mm_set1_ps(CMP_EPSILON);
          __m128 abs_val = _mm_andnot_ps(_mm_set1_ps(-0.0f), m_value);
          __m128 cmp = _mm_cmple_ps(abs_val, epsilon);
          return (_mm_movemask_ps(cmp) & 0x7) == 0x7;
     }
     _FORCE_INLINE_ bool is_equal_approx_sse(const Vector3SIMD &p_v) {
          __m128 epsilon = _mm_set1_ps(CMP_EPSILON);
          __m128 diff = _mm_sub_ps(m_value, p_v.m_value);
          __m128 abs_diff = _mm_andnot_ps(_mm_set1_ps(-0.0f), diff);
          __m128 cmp = _mm_cmple_ps(abs_diff, epsilon);
          return (_mm_movemask_ps(cmp) & 0x7) == 0x7;
     }

     _FORCE_INLINE_ bool lesser_sse(const Vector3SIMD &p_v) const {
          float x1 = _mm_cvtss_f32(m_value);
          float x2 = _mm_cvtss_f32(p_v.m_value);
          if (x1 == x2) {
               float y1 = _mm_cvtss_f32(
                   _mm_shuffle_ps(m_value, m_value, _MM_SHUFFLE(1, 1, 1, 1)));
               float y2 = _mm_cvtss_f32(_mm_shuffle_ps(
                   p_v.m_value, p_v.m_value, _MM_SHUFFLE(1, 1, 1, 1)));
               if (y1 == y2) {
                    return _mm_cvtss_f32(_mm_shuffle_ps(
                               m_value, m_value, _MM_SHUFFLE(2, 2, 2, 2))) <
                           _mm_cvtss_f32(
                               _mm_shuffle_ps(p_v.m_value, p_v.m_value,
                                              _MM_SHUFFLE(2, 2, 2, 2)));
               }
               return y1 < y2;
          }
          return x1 < x2;
     }

     _FORCE_INLINE_ bool greater_sse(const Vector3SIMD &p_v) {
          return !lesser_eq_sse(p_v);
     }

     _FORCE_INLINE_ bool greater_eq_sse(const Vector3SIMD &p_v) {
          return !lesser_sse(p_v);
     }

     _FORCE_INLINE_ bool not_eq_sse(const Vector3SIMD &p_v) {
          return !is_equal_approx_sse(p_v);
     }

     _FORCE_INLINE_ bool lesser_eq_sse(const Vector3SIMD &p_v) {
          return lesser_sse(p_v) || is_equal_approx_sse(p_v);
     }

     // Outer product implementation
     _FORCE_INLINE_ void outer_sse(const Vector3SIMD &p_with,
                                   float *r_ptr) const {
          __m128 row = m_value;
          __m128 col = p_with.m_value;

          // Broadcast each component and multiply
          __m128 row_x = _mm_shuffle_ps(row, row, _MM_SHUFFLE(0, 0, 0, 0));
          __m128 row_y = _mm_shuffle_ps(row, row, _MM_SHUFFLE(1, 1, 1, 1));
          __m128 row_z = _mm_shuffle_ps(row, row, _MM_SHUFFLE(2, 2, 2, 2));

          __m128 result_row1 = _mm_mul_ps(row_x, col);
          __m128 result_row2 = _mm_mul_ps(row_y, col);
          __m128 result_row3 = _mm_mul_ps(row_z, col);

          _mm_store_ps(r_ptr, result_row1);
          _mm_store_ps(r_ptr + 4, result_row2);
          _mm_store_ps(r_ptr + 8, result_row3);
     }

     _FORCE_INLINE_ Vector3SIMD limit_length_sse(float p_len = 1.0f) {
          float l = length_sse();
          if (l > 0.0f && p_len < l) {
               return Vector3SIMD(_mm_mul_ps(m_value, _mm_set1_ps(p_len / l)));
          }
          return *this;
     }

     _FORCE_INLINE_ Vector3SIMD clamp_sse(const Vector3SIMD &p_min,
                                          const Vector3SIMD &p_max) {
          return Vector3SIMD(
              _mm_min_ps(_mm_max_ps(m_value, p_min.m_value), p_max.m_value));
     }

     _FORCE_INLINE_ Vector3SIMD clampf_sse(float p_min, float p_max) {
          __m128 min_val = _mm_set1_ps(p_min);
          __m128 max_val = _mm_set1_ps(p_max);
          return Vector3SIMD(_mm_min_ps(_mm_max_ps(m_value, min_val), max_val));
     }

     _FORCE_INLINE_ Vector3SIMD minf_sse(float p_scalar) {
          return Vector3SIMD(_mm_min_ps(m_value, _mm_set1_ps(p_scalar)));
     }

     _FORCE_INLINE_ Vector3SIMD maxf_sse(float p_scalar) {
          return Vector3SIMD(_mm_max_ps(m_value, _mm_set1_ps(p_scalar)));
     }

     _FORCE_INLINE_ Vector3SIMD inverse_sse() {
          __m128 one = _mm_set1_ps(1.0f);
          return Vector3SIMD(_mm_div_ps(one, m_value));
     }

     _FORCE_INLINE_ Vector3SIMD direction_to_sse(const Vector3SIMD &p_to) {
          __m128 diff = _mm_sub_ps(p_to.m_value, m_value);
          __m128 len_sq = _mm_dp_ps(diff, diff, 0x7F);
          float len = _mm_cvtss_f32(_mm_sqrt_ss(len_sq));
          if (len <= CMP_EPSILON) {
               return Vector3SIMD();
          }
          return Vector3SIMD(_mm_div_ps(diff, _mm_set1_ps(len)));
     }

     _FORCE_INLINE_ float angle_to_sse(const Vector3SIMD &p_to) {
          __m128 len_prod = _mm_sqrt_ps(
              _mm_mul_ps(_mm_dp_ps(m_value, m_value, 0x7F),
                         _mm_dp_ps(p_to.m_value, p_to.m_value, 0x7F)));
          float cos_angle = dot_sse(p_to) / _mm_cvtss_f32(len_prod);
          cos_angle =
              (cos_angle < -1.0f ? -1.0f
                                 : (cos_angle > 1.0f ? 1.0f : cos_angle));
          return acos(cos_angle);
     }

     _FORCE_INLINE_ float signed_angle_to_sse(const Vector3SIMD &p_to,
                                              const Vector3SIMD &p_axis) {
          Vector3SIMD cross_vec = cross_sse(p_to);
          float unsigned_angle = angle_to_sse(p_to);
          float sign = cross_vec.dot_sse(p_axis);
          return (sign < 0) ? -unsigned_angle : unsigned_angle;
     }

     _FORCE_INLINE_ Vector3SIMD posmod_sse(float p_mod) {
          __m128 mod = _mm_set1_ps(p_mod);
          __m128 div = _mm_div_ps(m_value, mod);
          __m128 floor = _mm_floor_ps(div);
          return Vector3SIMD(_mm_sub_ps(m_value, _mm_mul_ps(floor, mod)));
     }

     _FORCE_INLINE_ Vector3SIMD posmodv_sse(const Vector3SIMD &p_modv) {
          __m128 div = _mm_div_ps(m_value, p_modv.m_value);
          __m128 floor = _mm_floor_ps(div);
          return Vector3SIMD(
              _mm_sub_ps(m_value, _mm_mul_ps(floor, p_modv.m_value)));
     }

     _FORCE_INLINE_ Vector3SIMD snapped_sse(const Vector3SIMD &p_step) {
          __m128 div = _mm_div_ps(m_value, p_step.m_value);
          __m128 rounded =
              _mm_round_ps(div, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
          return Vector3SIMD(_mm_mul_ps(rounded, p_step.m_value));
     }

     _FORCE_INLINE_ Vector3SIMD bezier_interpolate_sse(
         const Vector3SIMD &p_control_1, const Vector3SIMD &p_control_2,
         const Vector3SIMD &p_end, float p_t) {
          float omt = (1.0f - p_t);
          float omt2 = omt * omt;
          float omt3 = omt2 * omt;
          float t2 = p_t * p_t;
          float t3 = t2 * p_t;

          __m128 coef1 = _mm_set1_ps(omt3);
          __m128 coef2 = _mm_set1_ps(3.0f * omt2 * p_t);
          __m128 coef3 = _mm_set1_ps(3.0f * omt * t2);
          __m128 coef4 = _mm_set1_ps(t3);

          __m128 result =
              _mm_add_ps(_mm_add_ps(_mm_mul_ps(coef1, m_value),
                                    _mm_mul_ps(coef2, p_control_1.m_value)),
                         _mm_add_ps(_mm_mul_ps(coef3, p_control_2.m_value),
                                    _mm_mul_ps(coef4, p_end.m_value)));

          return Vector3SIMD(result);
     }

     _FORCE_INLINE_ Vector3SIMD floor_sse() const {
#if defined(__SSE4_1__)
          return Vector3SIMD(_mm_floor_ps(m_value));
#else
          // Fallback for SSE2/3
          __m128 two_pow_23 = _mm_set1_ps(8388608.0f);
          __m128 mask = _mm_cmplt_ps(m_value, _mm_setzero_ps());
          __m128 value_plus = _mm_add_ps(m_value, two_pow_23);
          __m128 value_minus = _mm_sub_ps(m_value, two_pow_23);
          __m128 result = _mm_sub_ps(value_plus, two_pow_23);
          __m128 result_minus = _mm_add_ps(value_minus, two_pow_23);
          return Vector3SIMD(_mm_or_ps(_mm_and_ps(mask, result_minus),
                                       _mm_andnot_ps(mask, result)));
#endif
     }

     _FORCE_INLINE_ Vector3SIMD ceil_sse() const {
#if defined(__SSE4_1__)
          return Vector3SIMD(_mm_ceil_ps(m_value));
#else
          // Fallback for SSE2/3
          __m128 two_pow_23 = _mm_set1_ps(8388608.0f);
          __m128 mask = _mm_cmpgt_ps(m_value, _mm_setzero_ps());
          __m128 value_plus = _mm_add_ps(m_value, two_pow_23);
          __m128 value_minus = _mm_sub_ps(m_value, two_pow_23);
          __m128 result = _mm_sub_ps(value_plus, two_pow_23);
          __m128 result_minus = _mm_add_ps(value_minus, two_pow_23);
          return Vector3SIMD(_mm_or_ps(_mm_and_ps(mask, result),
                                       _mm_andnot_ps(mask, result_minus)));
#endif
     }

     _FORCE_INLINE_ Vector3SIMD round_sse() const {
#if defined(__SSE4_1__)
          return Vector3SIMD(_mm_round_ps(
              m_value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
#else
          // Fallback for SSE2/3
          __m128 sign = _mm_and_ps(m_value, _mm_set1_ps(-0.0f));
          __m128 magic = _mm_or_ps(_mm_set1_ps(8388608.0f), sign);
          __m128 result = _mm_sub_ps(_mm_add_ps(m_value, magic), magic);
          return Vector3SIMD(result);
#endif
     }

     _FORCE_INLINE_ Vector3SIMD sign_sse() const {
          __m128 zero = _mm_setzero_ps();
          __m128 one = _mm_set1_ps(1.0f);
          __m128 minus_one = _mm_set1_ps(-1.0f);
          __m128 gt_mask = _mm_cmpgt_ps(m_value, zero);
          __m128 lt_mask = _mm_cmplt_ps(m_value, zero);
          return Vector3SIMD(_mm_or_ps(_mm_and_ps(gt_mask, one),
                                       _mm_and_ps(lt_mask, minus_one)));
     }

     _FORCE_INLINE_ Vector3SIMD cubic_interpolate_in_time_sse(
         const Vector3SIMD &p_b, const Vector3SIMD &p_pre_a,
         const Vector3SIMD &p_post_b, float p_weight, float p_b_t,
         float p_pre_a_t, float p_post_b_t) const {
          __m128 t = _mm_set1_ps(p_weight);
          __m128 t2 = _mm_mul_ps(t, t);
          __m128 t3 = _mm_mul_ps(t2, t);

          // Calculate tangents
          __m128 pb_pa =
              _mm_div_ps(_mm_sub_ps(p_b.m_value, m_value), _mm_set1_ps(p_b_t));
          __m128 pc_pa = _mm_div_ps(_mm_sub_ps(p_pre_a.m_value, m_value),
                                    _mm_set1_ps(p_pre_a_t));
          __m128 pb_pc = _mm_div_ps(_mm_sub_ps(p_b.m_value, p_post_b.m_value),
                                    _mm_set1_ps(p_post_b_t));

          // Hermite basis functions
          __m128 h1 = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(2.0f), t3),
                                 _mm_mul_ps(_mm_set1_ps(3.0f), t2));
          __m128 h2 = _mm_sub_ps(t3, _mm_mul_ps(_mm_set1_ps(2.0f), t2));
          __m128 h3 = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(3.0f), t2),
                                 _mm_mul_ps(_mm_set1_ps(2.0f), t3));
          __m128 h4 = _mm_sub_ps(t3, t2);

          return Vector3SIMD(_mm_add_ps(
              _mm_add_ps(_mm_mul_ps(m_value, h1), _mm_mul_ps(pb_pa, h2)),
              _mm_add_ps(_mm_mul_ps(p_b.m_value, h3), _mm_mul_ps(pb_pc, h4))));
     }

     _FORCE_INLINE_ Vector3SIMD bezier_derivative_sse(
         const Vector3SIMD &p_control_1, const Vector3SIMD &p_control_2,
         const Vector3SIMD &p_end, float p_t) const {
          float omt = (1.0f - p_t);
          float omt2 = omt * omt;
          float t2 = p_t * p_t;

          __m128 coef1 = _mm_set1_ps(-3.0f * omt2);
          __m128 coef2 = _mm_set1_ps(3.0f * omt2 - 6.0f * p_t * omt);
          __m128 coef3 = _mm_set1_ps(6.0f * p_t * omt - 3.0f * t2);
          __m128 coef4 = _mm_set1_ps(3.0f * t2);

          return Vector3SIMD(
              _mm_add_ps(_mm_add_ps(_mm_mul_ps(coef1, m_value),
                                    _mm_mul_ps(coef2, p_control_1.m_value)),
                         _mm_add_ps(_mm_mul_ps(coef3, p_control_2.m_value),
                                    _mm_mul_ps(coef4, p_end.m_value))));
     }

     _FORCE_INLINE_ Vector2 octahedron_encode_sse() const {
          // n = v / (|x| + |y| + |z|)
          __m128 abs_v = _mm_andnot_ps(_mm_set1_ps(-0.0f), m_value);
          __m128 sum = _mm_add_ps(
              _mm_add_ps(_mm_shuffle_ps(abs_v, abs_v, _MM_SHUFFLE(0, 0, 0, 0)),
                         _mm_shuffle_ps(abs_v, abs_v, _MM_SHUFFLE(1, 1, 1, 1))),
              _mm_shuffle_ps(abs_v, abs_v, _MM_SHUFFLE(2, 2, 2, 2)));
          __m128 n = _mm_div_ps(m_value, sum);

          // Get components
          float x = _mm_cvtss_f32(n);
          float y =
              _mm_cvtss_f32(_mm_shuffle_ps(n, n, _MM_SHUFFLE(1, 1, 1, 1)));
          float z =
              _mm_cvtss_f32(_mm_shuffle_ps(n, n, _MM_SHUFFLE(2, 2, 2, 2)));

          Vector2 o;
          if (z >= 0.0f) {
               o.x = x;
               o.y = y;
          } else {
               o.x = (1.0f - Math::abs(y)) * (x >= 0.0f ? 1.0f : -1.0f);
               o.y = (1.0f - Math::abs(x)) * (y >= 0.0f ? 1.0f : -1.0f);
          }

          // Map to [0,1] range
          o.x = o.x * 0.5f + 0.5f;
          o.y = o.y * 0.5f + 0.5f;
          return o;
     }

     _FORCE_INLINE_ static Vector3SIMD octahedron_decode_sse(
         const Vector2 &p_oct) {
          // Map input from [0,1] to [-1,1]
          __m128 f = _mm_setr_ps(p_oct.x * 2.0f - 1.0f, p_oct.y * 2.0f - 1.0f,
                                 0.0f, 0.0f);

          // Calculate z component
          __m128 abs_x =
              _mm_andnot_ps(_mm_set1_ps(-0.0f),
                            _mm_shuffle_ps(f, f, _MM_SHUFFLE(0, 0, 0, 0)));
          __m128 abs_y =
              _mm_andnot_ps(_mm_set1_ps(-0.0f),
                            _mm_shuffle_ps(f, f, _MM_SHUFFLE(1, 1, 1, 1)));
          __m128 z = _mm_sub_ps(_mm_set1_ps(1.0f), _mm_add_ps(abs_x, abs_y));

          // Construct full vector
          __m128 n = _mm_shuffle_ps(f, z, _MM_SHUFFLE(0, 0, 1, 0));

          // Handle z correction
          float z_val = _mm_cvtss_f32(z);
          float t = std::max(
              -z_val,
              0.0f);  // Math:: apparently did not contain max definition

          // Apply correction
          __m128 x_sign =
              _mm_and_ps(_mm_cmpge_ps(n, _mm_setzero_ps()), _mm_set1_ps(1.0f));
          __m128 correction = _mm_mul_ps(_mm_set1_ps(t), x_sign);
          n = _mm_add_ps(n, correction);

          // Normalize
          __m128 length = _mm_sqrt_ps(_mm_dp_ps(n, n, 0x7F));
          n = _mm_div_ps(n, length);

          return Vector3SIMD(n);
     }

#endif  // endif for VECTOR3SIMD_USE_SSE

// Now adding the NEON implementations of these operations
#if defined(VECTOR3SIMD_USE_NEON)
     _FORCE_INLINE_ Vector3SIMD add_neon(const Vector3SIMD &p_v) {
          return Vector3SIMD(vaddq_f32(m_value, p_v.m_value));
     }

     _FORCE_INLINE_ Vector3SIMD sub_neon(const Vector3SIMD &p_v) {
          return Vector3SIMD(vsubq_f32(m_value, p_v.m_value));
     }

     _FORCE_INLINE_ float dot_neon(const Vector3SIMD &p_v) {
          float32x4_t mul = vmulq_f32(m_value, p_v.m_value);
          float32x2_t v = vpadd_f32(vget_low_f32(mul), vget_high_f32(mul));
          v = vpadd_f32(v, v);
          return vget_lane_f32(v, 0);
     }

     _FORCE_INLINE_ Vector3SIMD cross_neon(const Vector3SIMD &p_v) {
          float32x4_t a_yzx = vextq_f32(m_value, m_value, 1);
          float32x4_t b_yzx = vextq_f32(p_v.m_value, p_v.m_value, 1);
          float32x4_t c = vsubq_f32(vmulq_f32(m_value, b_yzx),
                                    vmulq_f32(a_yzx, p_v.m_value));
          return Vector3SIMD(vextq_f32(c, c, 3));
     }

     _FORCE_INLINE_ bool is_zero_approx_neon() const {
          float32x4_t epsilon = vdupq_n_f32(CMP_EPSILON);
          float32x4_t abs_val = vabsq_f32(m_value);
          uint32x4_t cmp = vcleq_f32(abs_val, epsilon);
          // We only care about x,y,z components (first 3), hence 0x7
          uint32x2_t cmp_fold = vand_u32(vget_low_u32(cmp), vget_high_u32(cmp));
          return (vget_lane_u32(cmp_fold, 0) & 0x7) == 0x7;
     }

     _FORCE_INLINE_ float length_squared_neon() { return dot_neon(*this); }

     _FORCE_INLINE_ float length_neon() { return sqrtf(length_squared_neon()); }

     _FORCE_INLINE_ Vector3SIMD normalize_neon() {
          float l = length_neon();
          if (l == 0) {
               return Vector3SIMD();
          }
          return Vector3SIMD(vmulq_n_f32(m_value, 1.0f / l));
     }

     _FORCE_INLINE_ bool is_normalized_neon() const {
          // Calculate length squared using dot product with self
          float len_sq = dot_neon(*this);

          // Compare with 1 within epsilon
          return Math::is_equal_approx(len_sq, 1.0f, UNIT_EPSILON);
     }

     _FORCE_INLINE_ Vector3SIMD mul_neon(const Vector3SIMD &p_v) {
          return Vector3SIMD(vmulq_f32(m_value, p_v.m_value));
     }

     _FORCE_INLINE_ Vector3SIMD div_neon(const Vector3SIMD &p_v) {
          return Vector3SIMD(vdivq_f32(m_value, p_v.m_value));
     }

     _FORCE_INLINE_ Vector3SIMD mul_scalar_neon(float p_scalar) {
          return Vector3SIMD(vmulq_n_f32(m_value, p_scalar));
     }

     _FORCE_INLINE_ Vector3SIMD div_scalar_neon(float p_scalar) {
          return Vector3SIMD(vmulq_n_f32(m_value, 1.0f / p_scalar));
     }

     _FORCE_INLINE_ Vector3SIMD abs_neon() {
          return Vector3SIMD(vabsq_f32(m_value));
     }

     _FORCE_INLINE_ Vector3SIMD neg_neon() {
          return Vector3SIMD(vnegq_f32(m_value));
     }

     _FORCE_INLINE_ Vector3SIMD min_neon(const Vector3SIMD &p_v) {
          return Vector3SIMD(vminq_f32(m_value, p_v.m_value));
     }

     _FORCE_INLINE_ Vector3SIMD max_neon(const Vector3SIMD &p_v) {
          return Vector3SIMD(vmaxq_f32(m_value, p_v.m_value));
     }

     _FORCE_INLINE_ Vector3SIMD snap_neon(const Vector3SIMD &p_step) {
          Vector3SIMD result = *this;
          float32x4_t div = vdivq_f32(m_value, p_step.m_value);

          // NEON doesn't have a direct round instruction, so we implement
          // rounding by adding 0.5 for positive numbers and subtracting 0.5 for
          // negative
          float32x4_t sign_mask = vcltq_f32(div, vdupq_n_f32(0.0f));
          float32x4_t add_value =
              vbslq_f32(sign_mask, vdupq_n_f32(-0.5f), vdupq_n_f32(0.5f));
          float32x4_t added = vaddq_f32(div, add_value);

          // Convert to int and back to float to truncate
          int32x4_t as_int = vcvtq_s32_f32(added);
          float32x4_t rounded = vcvtq_f32_s32(as_int);

          result.m_value = vmulq_f32(rounded, p_step.m_value);
          return result;
     }

     _FORCE_INLINE_ Vector3SIMD snapped_neon(const Vector3SIMD &p_step) const {
          Vector3SIMD v = *this;
          v.snap_neon(p_step);
          return v;
     }

     _FORCE_INLINE_ void snapf_neon(real_t p_step) {
          float32x4_t step = vdupq_n_f32(p_step);
          float32x4_t div = vdivq_f32(m_value, step);

          // Implement rounding as above
          float32x4_t sign_mask = vcltq_f32(div, vdupq_n_f32(0.0f));
          float32x4_t add_value =
              vbslq_f32(sign_mask, vdupq_n_f32(-0.5f), vdupq_n_f32(0.5f));
          float32x4_t added = vaddq_f32(div, add_value);

          int32x4_t as_int = vcvtq_s32_f32(added);
          float32x4_t rounded = vcvtq_f32_s32(as_int);

          m_value = vmulq_f32(rounded, step);
     }

     _FORCE_INLINE_ Vector3SIMD snappedf_neon(real_t p_step) const {
          Vector3SIMD v = *this;
          v.snapf_neon(p_step);
          return v;
     }
     _FORCE_INLINE_ Vector3SIMD reflect_neon(const Vector3SIMD &p_normal) {
          float d = dot_neon(p_normal);
          float32x4_t scale = vdupq_n_f32(2.0f * d);
          float32x4_t proj = vmulq_f32(p_normal.m_value, scale);
          return Vector3SIMD(vsubq_f32(m_value, proj));
     }

     _FORCE_INLINE_ Vector3SIMD project_neon(const Vector3SIMD &p_to) {
          float d = dot_neon(p_to);
          float len_sq = p_to.length_squared_neon();
          if (len_sq == 0.0f) {
               return Vector3SIMD();
          }
          float32x4_t scale = vdupq_n_f32(d / len_sq);
          return Vector3SIMD(vmulq_f32(p_to.m_value, scale));
     }

     _FORCE_INLINE_ Vector3SIMD slide_neon(const Vector3SIMD &p_normal) {
          float d = dot_neon(p_normal);
          float32x4_t proj = vmulq_n_f32(p_normal.m_value, d);
          return Vector3SIMD(vsubq_f32(m_value, proj));
     }
     _FORCE_INLINE_ Vector3SIMD bounce_neon(const Vector3SIMD &p_normal) {
          float d = dot_neon(p_normal);
          float32x4_t scale = vdupq_n_f32(2.0f * d);
          float32x4_t proj = vmulq_f32(p_normal.m_value, scale);
          return Vector3SIMD(vsubq_f32(proj, m_value));
     }

     _FORCE_INLINE_ void rotate_neon(const Vector3SIMD &p_axis,
                                     real_t p_angle) {
          // Ensure axis is normalized
          Vector3SIMD axis = p_axis;
          if (!axis.is_normalized_neon()) {
               axis = axis.normalize_neon();
          }

          float s = Math::sin(p_angle);
          float c = Math::cos(p_angle);
          float k = 1.0f - c;

          float x = axis.x();
          float y = axis.y();
          float z = axis.z();

          // Build rotation matrix components using NEON
          float32x4_t row1 = {x * x * k + c, x * y * k - z * s,
                              x * z * k + y * s, 0.0f};

          float32x4_t row2 = {y * x * k + z * s, y * y * k + c,
                              y * z * k - x * s, 0.0f};

          float32x4_t row3 = {z * x * k - y * s, z * y * k + x * s,
                              z * z * k + c, 0.0f};

          // Transform the vector using NEON
          float32x4_t vX = vdupq_n_f32(vgetq_lane_f32(m_value, 0));
          float32x4_t vY = vdupq_n_f32(vgetq_lane_f32(m_value, 1));
          float32x4_t vZ = vdupq_n_f32(vgetq_lane_f32(m_value, 2));

          float32x4_t result =
              vaddq_f32(vaddq_f32(vmulq_f32(vX, row1), vmulq_f32(vY, row2)),
                        vmulq_f32(vZ, row3));

          m_value = result;
     }

     _FORCE_INLINE_ Vector3SIMD rotated_neon(const Vector3SIMD &p_axis,
                                             real_t p_angle) const {
          Vector3SIMD result = *this;
          result.rotate_neon(p_axis, p_angle);
          return result;
     }

     _FORCE_INLINE_ float distance_to_neon(const Vector3SIMD &p_to) {
          Vector3SIMD diff = sub_neon(p_to);
          return diff.length_neon();
     }

     _FORCE_INLINE_ float distance_squared_to_neon(const Vector3SIMD &p_to) {
          Vector3SIMD diff = sub_neon(p_to);
          return diff.length_squared_neon();
     }

     _FORCE_INLINE_ Vector3SIMD lerp_neon(const Vector3SIMD &p_to,
                                          float p_weight) {
          float32x4_t w = vdupq_n_f32(p_weight);
          float32x4_t inv_w = vsubq_f32(vdupq_n_f32(1.0f), w);
          return Vector3SIMD(
              vaddq_f32(vmulq_f32(m_value, inv_w), vmulq_f32(p_to.m_value, w)));
     }

     _FORCE_INLINE_ Vector3SIMD move_toward_neon(const Vector3SIMD &p_to,
                                                 float p_delta) {
          Vector3SIMD diff = sub_neon(p_to);
          float len = diff.length_neon();

          if (len <= p_delta || len < CMP_EPSILON) {
               return p_to;
          }

          return Vector3SIMD(
              vaddq_f32(m_value, vmulq_n_f32(diff.m_value, p_delta / len)));
     }

     _FORCE_INLINE_ Vector3SIMD
     cubic_interpolate_neon(const Vector3SIMD &p_b, const Vector3SIMD &p_pre_a,
                            const Vector3SIMD &p_post_b, float p_weight) {
          float32x4_t w = vdupq_n_f32(p_weight);
          float32x4_t w2 = vmulq_f32(w, w);
          float32x4_t w3 = vmulq_f32(w2, w);

          float32x4_t a0 = vsubq_f32(p_b.m_value, m_value);
          float32x4_t a1 = vsubq_f32(p_pre_a.m_value, m_value);
          float32x4_t a2 = vaddq_f32(vsubq_f32(a0, a1),
                                     vsubq_f32(p_post_b.m_value, p_b.m_value));

          float32x4_t c0 = m_value;
          float32x4_t c1 = vmulq_n_f32(a1, 0.5f);
          float32x4_t c2 = vmulq_n_f32(a0, 2.0f);
          float32x4_t c3 = vmulq_n_f32(a2, 0.5f);

          return Vector3SIMD(vaddq_f32(
              vaddq_f32(vaddq_f32(c0, vmulq_f32(c1, w)), vmulq_f32(c2, w2)),
              vmulq_f32(c3, w3)));
     }

     _FORCE_INLINE_ Vector3SIMD slerp_neon(const Vector3SIMD &p_to,
                                           float p_weight) {
          float start_length_sq = length_squared_neon();
          float end_length_sq = p_to.length_squared_neon();

          if (start_length_sq == 0.0f || end_length_sq == 0.0f) {
               return lerp_neon(p_to, p_weight);
          }

          Vector3SIMD axis = cross_neon(p_to);
          float axis_length_sq = axis.length_squared_neon();

          if (axis_length_sq == 0.0f) {
               return lerp_neon(p_to, p_weight);
          }

          float angle =
              acos(dot_neon(p_to) / (sqrt(start_length_sq * end_length_sq)));
          if (angle == 0.0f) {
               return *this;
          }

          float sin_angle = sin(angle);
          float scale1 = sin((1.0f - p_weight) * angle) / sin_angle;
          float scale2 = sin(p_weight * angle) / sin_angle;

          float32x4_t v1 = vmulq_n_f32(m_value, scale1);
          float32x4_t v2 = vmulq_n_f32(p_to.m_value, scale2);
          return Vector3SIMD(vaddq_f32(v1, v2));
     }

     _FORCE_INLINE_ bool is_equal_approx_neon(const Vector3SIMD &p_v) {
          float32x4_t diff = vsubq_f32(m_value, p_v.m_value);
          float32x4_t abs_diff = vabsq_f32(diff);
          uint32x4_t cmp = vcleq_f32(abs_diff, vdupq_n_f32(CMP_EPSILON));
          uint32x2_t cmp_fold = vand_u32(vget_low_u32(cmp), vget_high_u32(cmp));
          return vget_lane_u32(cmp_fold, 0) & vget_lane_u32(cmp_fold, 1);
     }

     _FORCE_INLINE_ Vector3SIMD limit_length_neon(float p_len = 1.0f) {
          float l = length_neon();
          if (l > 0.0f && p_len < l) {
               return Vector3SIMD(vmulq_n_f32(m_value, p_len / l));
          }
          return *this;
     }

     _FORCE_INLINE_ Vector3SIMD clamp_neon(const Vector3SIMD &p_min,
                                           const Vector3SIMD &p_max) {
          return Vector3SIMD(
              vminq_f32(vmaxq_f32(m_value, p_min.m_value), p_max.m_value));
     }

     _FORCE_INLINE_ Vector3SIMD clampf_neon(float p_min, float p_max) {
          float32x4_t min_val = vdupq_n_f32(p_min);
          float32x4_t max_val = vdupq_n_f32(p_max);
          return Vector3SIMD(vminq_f32(vmaxq_f32(m_value, min_val), max_val));
     }

     _FORCE_INLINE_ Vector3SIMD minf_neon(float p_scalar) {
          return Vector3SIMD(vminq_f32(m_value, vdupq_n_f32(p_scalar)));
     }

     _FORCE_INLINE_ Vector3SIMD maxf_neon(float p_scalar) {
          return Vector3SIMD(vmaxq_f32(m_value, vdupq_n_f32(p_scalar)));
     }

     _FORCE_INLINE_ Vector3SIMD inverse_neon() {
          float32x4_t one = vdupq_n_f32(1.0f);
          return Vector3SIMD(vdivq_f32(one, m_value));
     }

     _FORCE_INLINE_ Vector3SIMD direction_to_neon(const Vector3SIMD &p_to) {
          float32x4_t diff = vsubq_f32(p_to.m_value, m_value);
          float len_sq = vaddvq_f32(vmulq_f32(diff, diff));
          float len = sqrtf(len_sq);
          if (len <= CMP_EPSILON) {
               return Vector3SIMD();
          }
          return Vector3SIMD(vmulq_n_f32(diff, 1.0f / len));
     }

     _FORCE_INLINE_ float angle_to_neon(const Vector3SIMD &p_to) {
          float len_prod =
              sqrtf(length_squared_neon() * p_to.length_squared_neon());
          if (len_prod == 0.0f) {
               return 0.0f;
          }
          float cos_angle = dot_neon(p_to) / len_prod;
          cos_angle =
              (cos_angle < -1.0f ? -1.0f
                                 : (cos_angle > 1.0f ? 1.0f : cos_angle));
          return acos(cos_angle);
     }

     _FORCE_INLINE_ float signed_angle_to_neon(
         const Vector3SIMD &p_to, const Vector3SIMD &p_axis) const {
          Vector3SIMD cross_vec = cross_neon(p_to);
          float unsigned_angle = angle_to_neon(p_to);
          float sign = cross_vec.dot_neon(p_axis);
          return (sign < 0) ? -unsigned_angle : unsigned_angle;
     }

     _FORCE_INLINE_ Vector3SIMD posmod_neon(float p_mod) const {
          float32x4_t mod = vdupq_n_f32(p_mod);
          float32x4_t div = vdivq_f32(m_value, mod);
          float32x4_t floor = vcvtq_f32_s32(vcvtq_s32_f32(div));
          return Vector3SIMD(vsubq_f32(m_value, vmulq_f32(floor, mod)));
     }

     _FORCE_INLINE_ Vector3SIMD posmodv_neon(const Vector3SIMD &p_modv) const {
          float32x4_t div = vdivq_f32(m_value, p_modv.m_value);
          float32x4_t floor = vcvtq_f32_s32(vcvtq_s32_f32(div));
          return Vector3SIMD(
              vsubq_f32(m_value, vmulq_f32(floor, p_modv.m_value)));
     }

     _FORCE_INLINE_ Vector3SIMD snapped_neon(const Vector3SIMD &p_step) const {
          float32x4_t div = vdivq_f32(m_value, p_step.m_value);
          float32x4_t rounded =
              vcvtq_f32_s32(vcvtq_s32_f32(vaddq_f32(div, vdupq_n_f32(0.5f))));
          return Vector3SIMD(vmulq_f32(rounded, p_step.m_value));
     }

     _FORCE_INLINE_ Vector3SIMD bezier_interpolate_neon(
         const Vector3SIMD &p_control_1, const Vector3SIMD &p_control_2,
         const Vector3SIMD &p_end, float p_t) const {
          float omt = (1.0f - p_t);
          float omt2 = omt * omt;
          float omt3 = omt2 * omt;
          float t2 = p_t * p_t;
          float t3 = t2 * p_t;

          float32x4_t coef1 = vdupq_n_f32(omt3);
          float32x4_t coef2 = vdupq_n_f32(3.0f * omt2 * p_t);
          float32x4_t coef3 = vdupq_n_f32(3.0f * omt * t2);
          float32x4_t coef4 = vdupq_n_f32(t3);

          float32x4_t result =
              vaddq_f32(vaddq_f32(vmulq_f32(coef1, m_value),
                                  vmulq_f32(coef2, p_control_1.m_value)),
                        vaddq_f32(vmulq_f32(coef3, p_control_2.m_value),
                                  vmulq_f32(coef4, p_end.m_value)));

          return Vector3SIMD(result);
     }

     _FORCE_INLINE_ Vector3SIMD floor_neon() const {
          return Vector3SIMD(vcvtq_f32_s32(vcvtq_s32_f32(m_value)));
     }

     _FORCE_INLINE_ Vector3SIMD ceil_neon() const {
          float32x4_t ceil_val = vaddq_f32(m_value, vdupq_n_f32(0.5f));
          return Vector3SIMD(vcvtq_f32_s32(vcvtq_s32_f32(ceil_val)));
     }

     _FORCE_INLINE_ Vector3SIMD round_neon() const {
          return Vector3SIMD(vcvtq_f32_s32(
              vcvtq_s32_f32(vaddq_f32(m_value, vdupq_n_f32(0.5f)))));
     }

     _FORCE_INLINE_ Vector3SIMD sign_neon() const {
          float32x4_t zero = vdupq_n_f32(0.0f);
          float32x4_t one = vdupq_n_f32(1.0f);
          float32x4_t minus_one = vdupq_n_f32(-1.0f);
          uint32x4_t gt_mask = vcgtq_f32(m_value, zero);
          uint32x4_t lt_mask = vcltq_f32(m_value, zero);
          float32x4_t gt_result = vbslq_f32(gt_mask, one, zero);
          return Vector3SIMD(vbslq_f32(lt_mask, minus_one, gt_result));
     }

     _FORCE_INLINE_ Vector3SIMD cubic_interpolate_in_time_neon(
         const Vector3SIMD &p_b, const Vector3SIMD &p_pre_a,
         const Vector3SIMD &p_post_b, float p_weight, float p_b_t,
         float p_pre_a_t, float p_post_b_t) const {
          float32x4_t t = vdupq_n_f32(p_weight);
          float32x4_t t2 = vmulq_f32(t, t);
          float32x4_t t3 = vmulq_f32(t2, t);

          // Calculate tangents
          float32x4_t pb_pa =
              vdivq_f32(vsubq_f32(p_b.m_value, m_value), vdupq_n_f32(p_b_t));
          float32x4_t pc_pa = vdivq_f32(vsubq_f32(p_pre_a.m_value, m_value),
                                        vdupq_n_f32(p_pre_a_t));
          float32x4_t pb_pc =
              vdivq_f32(vsubq_f32(p_b.m_value, p_post_b.m_value),
                        vdupq_n_f32(p_post_b_t));

          // Hermite basis functions
          float32x4_t h1 =
              vsubq_f32(vmulq_n_f32(t3, 2.0f), vmulq_n_f32(t2, 3.0f));
          float32x4_t h2 = vsubq_f32(t3, vmulq_n_f32(t2, 2.0f));
          float32x4_t h3 =
              vsubq_f32(vmulq_n_f32(t2, 3.0f), vmulq_n_f32(t3, 2.0f));
          float32x4_t h4 = vsubq_f32(t3, t2);

          return Vector3SIMD(vaddq_f32(
              vaddq_f32(vmulq_f32(m_value, h1), vmulq_f32(pb_pa, h2)),
              vaddq_f32(vmulq_f32(p_b.m_value, h3), vmulq_f32(pb_pc, h4))));
     }

     _FORCE_INLINE_ Vector3SIMD bezier_derivative_neon(
         const Vector3SIMD &p_control_1, const Vector3SIMD &p_control_2,
         const Vector3SIMD &p_end, float p_t) const {
          float omt = (1.0f - p_t);
          float omt2 = omt * omt;
          float t2 = p_t * p_t;

          float32x4_t coef1 = vdupq_n_f32(-3.0f * omt2);
          float32x4_t coef2 = vdupq_n_f32(3.0f * omt2 - 6.0f * p_t * omt);
          float32x4_t coef3 = vdupq_n_f32(6.0f * p_t * omt - 3.0f * t2);
          float32x4_t coef4 = vdupq_n_f32(3.0f * t2);

          return Vector3SIMD(
              vaddq_f32(vaddq_f32(vmulq_f32(coef1, m_value),
                                  vmulq_f32(coef2, p_control_1.m_value)),
                        vaddq_f32(vmulq_f32(coef3, p_control_2.m_value),
                                  vmulq_f32(coef4, p_end.m_value))));
     }
     float32x4_t abs_val = vabsq_f32(m_value);
     uint32x4_t cmp = vcleq_f32(abs_val, vdupq_n_f32(CMP_EPSILON));
     uint32x2_t cmp_fold = vand_u32(vget_low_u32(cmp), vget_high_u32(cmp));
     return vget_lane_u32(cmp_fold, 0) & vget_lane_u32(cmp_fold, 1);

     _FORCE_INLINE_ bool is_equal_approx_neon(const Vector3SIMD &p_v) const {
          float32x4_t diff = vsubq_f32(m_value, p_v.m_value);
          float32x4_t abs_diff = vabsq_f32(diff);
          uint32x4_t cmp = vcleq_f32(abs_diff, vdupq_n_f32(CMP_EPSILON));
          uint32x2_t cmp_fold = vand_u32(vget_low_u32(cmp), vget_high_u32(cmp));
          return vget_lane_u32(cmp_fold, 0) & vget_lane_u32(cmp_fold, 1);
     }

     _FORCE_INLINE_ bool not_eq_neon(const Vector3SIMD &p_v) const {
          return !is_equal_approx_neon(p_v);
     }

     _FORCE_INLINE_ bool lesser_neon(const Vector3SIMD &p_v) const {
          float x1 = vgetq_lane_f32(m_value, 0);
          float x2 = vgetq_lane_f32(p_v.m_value, 0);
          if (x1 == x2) {
               float y1 = vgetq_lane_f32(m_value, 1);
               float y2 = vgetq_lane_f32(p_v.m_value, 1);
               if (y1 == y2) {
                    return vgetq_lane_f32(m_value, 2) <
                           vgetq_lane_f32(p_v.m_value, 2);
               }
               return y1 < y2;
          }
          return x1 < x2;
     }

     _FORCE_INLINE_ bool lesser_eq_neon(const Vector3SIMD &p_v) const {
          return lesser_neon(p_v) || is_equal_approx_neon(p_v);
     }

     _FORCE_INLINE_ bool greater_neon(const Vector3SIMD &p_v) const {
          return !lesser_eq_neon(p_v);
     }

     _FORCE_INLINE_ bool greater_eq_neon(const Vector3SIMD &p_v) const {
          return !lesser_neon(p_v);
     }

     // Outer product implementation for NEON
     _FORCE_INLINE_ void outer_neon(const Vector3SIMD &p_with,
                                    float *r_ptr) const {
          float32x4_t row = m_value;
          float32x4_t col = p_with.m_value;

          float32x4_t row_x = vdupq_lane_f32(vget_low_f32(row), 0);
          float32x4_t row_y = vdupq_lane_f32(vget_low_f32(row), 1);
          float32x4_t row_z = vdupq_lane_f32(vget_high_f32(row), 0);

          float32x4_t result_row1 = vmulq_f32(row_x, col);
          float32x4_t result_row2 = vmulq_f32(row_y, col);
          float32x4_t result_row3 = vmulq_f32(row_z, col);

          vst1q_f32(r_ptr, result_row1);
          vst1q_f32(r_ptr + 4, result_row2);
          vst1q_f32(r_ptr + 8, result_row3);
     }

     _FORCE_INLINE_ Vector2 octahedron_encode_neon() const {
          // n = v / (|x| + |y| + |z|)
          float32x4_t abs_v = vabsq_f32(m_value);
          float sum = vaddvq_f32(abs_v);  // NEON horizontal add
          float32x4_t n = vdivq_f32(m_value, vdupq_n_f32(sum));

          // Get components
          float x = vgetq_lane_f32(n, 0);
          float y = vgetq_lane_f32(n, 1);
          float z = vgetq_lane_f32(n, 2);

          Vector2 o;
          if (z >= 0.0f) {
               o.x = x;
               o.y = y;
          } else {
               o.x = (1.0f - Math::abs(y)) * (x >= 0.0f ? 1.0f : -1.0f);
               o.y = (1.0f - Math::abs(x)) * (y >= 0.0f ? 1.0f : -1.0f);
          }

          // Map to [0,1] range
          o.x = o.x * 0.5f + 0.5f;
          o.y = o.y * 0.5f + 0.5f;
          return o;
     }

     _FORCE_INLINE_ static Vector3SIMD octahedron_decode_neon(
         const Vector2 &p_oct) {
          // Map input from [0,1] to [-1,1]
          float32x4_t f = {p_oct.x * 2.0f - 1.0f, p_oct.y * 2.0f - 1.0f, 0.0f,
                           0.0f};

          // Calculate z component
          float32x2_t abs_xy = vabs_f32(vget_low_f32(f));
          float z = 1.0f - vget_lane_f32(abs_xy, 0) - vget_lane_f32(abs_xy, 1);

          // Construct full vector
          float32x4_t n = {vgetq_lane_f32(f, 0), vgetq_lane_f32(f, 1), z, 0.0f};

          // Handle z correction
          float t = Math::max(-z, 0.0f);

          // Apply correction using NEON select
          uint32x4_t x_sign = vcgeq_f32(n, vdupq_n_f32(0.0f));
          float32x4_t correction =
              vmulq_n_f32(vreinterpretq_f32_u32(x_sign), t);
          n = vaddq_f32(n, correction);

          // Normalize
          float32x4_t squared = vmulq_f32(n, n);
          float length = sqrtf(vaddvq_f32(squared));  // Horizontal add and sqrt
          n = vdivq_f32(n, vdupq_n_f32(length));

          return Vector3SIMD(n);
     }

#endif

     // Could add static Constants and static constant definitions
     // To match vector3.h, but there is no real reason to at the moment
};

#endif  // VECTOR3_SIMD_H