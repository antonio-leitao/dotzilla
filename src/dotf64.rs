#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
use std::arch::is_aarch64_feature_detected;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

/// Computes the dot product of two f64 slices
/// # Panics
/// Panics if the slices have different lengths
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Slices must have equal length");

    // Architecture-specific dispatch
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_product_avx2_fma(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
            return unsafe { dot_product_neon(a, b) };
        }
    }

    // Fast scalar fallback
    dot_product_scalar(a, b)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_avx2_fma(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len();
    let mut accum0 = _mm256_setzero_pd();
    let mut accum1 = _mm256_setzero_pd();
    let mut accum2 = _mm256_setzero_pd();
    let mut accum3 = _mm256_setzero_pd();
    let mut accum4 = _mm256_setzero_pd();
    let mut accum5 = _mm256_setzero_pd();
    let mut accum6 = _mm256_setzero_pd();
    let mut accum7 = _mm256_setzero_pd();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let mut i = 0;

    // Process 32 elements per iteration (8 vectors × 4 elements)
    let chunk_size = 32;
    let iterations = len / chunk_size;
    let remainder_start = iterations * chunk_size;

    for _ in 0..iterations {
        let offset = i;

        // Load 8 vectors from a and b
        let a0 = _mm256_loadu_pd(a_ptr.add(offset));
        let b0 = _mm256_loadu_pd(b_ptr.add(offset));
        accum0 = _mm256_fmadd_pd(a0, b0, accum0);

        let a1 = _mm256_loadu_pd(a_ptr.add(offset + 4));
        let b1 = _mm256_loadu_pd(b_ptr.add(offset + 4));
        accum1 = _mm256_fmadd_pd(a1, b1, accum1);

        let a2 = _mm256_loadu_pd(a_ptr.add(offset + 8));
        let b2 = _mm256_loadu_pd(b_ptr.add(offset + 8));
        accum2 = _mm256_fmadd_pd(a2, b2, accum2);

        let a3 = _mm256_loadu_pd(a_ptr.add(offset + 12));
        let b3 = _mm256_loadu_pd(b_ptr.add(offset + 12));
        accum3 = _mm256_fmadd_pd(a3, b3, accum3);

        let a4 = _mm256_loadu_pd(a_ptr.add(offset + 16));
        let b4 = _mm256_loadu_pd(b_ptr.add(offset + 16));
        accum4 = _mm256_fmadd_pd(a4, b4, accum4);

        let a5 = _mm256_loadu_pd(a_ptr.add(offset + 20));
        let b5 = _mm256_loadu_pd(b_ptr.add(offset + 20));
        accum5 = _mm256_fmadd_pd(a5, b5, accum5);

        let a6 = _mm256_loadu_pd(a_ptr.add(offset + 24));
        let b6 = _mm256_loadu_pd(b_ptr.add(offset + 24));
        accum6 = _mm256_fmadd_pd(a6, b6, accum6);

        let a7 = _mm256_loadu_pd(a_ptr.add(offset + 28));
        let b7 = _mm256_loadu_pd(b_ptr.add(offset + 28));
        accum7 = _mm256_fmadd_pd(a7, b7, accum7);

        i += chunk_size;
    }

    // Combine accumulators
    accum0 = _mm256_add_pd(accum0, accum1);
    accum2 = _mm256_add_pd(accum2, accum3);
    accum4 = _mm256_add_pd(accum4, accum5);
    accum6 = _mm256_add_pd(accum6, accum7);

    accum0 = _mm256_add_pd(accum0, accum2);
    accum4 = _mm256_add_pd(accum4, accum6);

    let total = _mm256_add_pd(accum0, accum4);

    // Horizontal sum
    let sum = hsum_avx(total);

    // Process remaining elements with optimized scalar
    sum + dot_product_scalar(&a[remainder_start..], &b[remainder_start..])
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_product_neon(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len();
    let mut accum0 = vdupq_n_f64(0.0);
    let mut accum1 = vdupq_n_f64(0.0);
    let mut accum2 = vdupq_n_f64(0.0);
    let mut accum3 = vdupq_n_f64(0.0);
    let mut accum4 = vdupq_n_f64(0.0);
    let mut accum5 = vdupq_n_f64(0.0);
    let mut accum6 = vdupq_n_f64(0.0);
    let mut accum7 = vdupq_n_f64(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let mut i = 0;

    // Process 16 elements per iteration (8 vectors × 2 elements)
    let chunk_size = 16;
    let iterations = len / chunk_size;
    let remainder_start = iterations * chunk_size;

    for _ in 0..iterations {
        let offset = i;

        let a0 = vld1q_f64(a_ptr.add(offset));
        let b0 = vld1q_f64(b_ptr.add(offset));
        accum0 = vfmaq_f64(accum0, a0, b0);

        let a1 = vld1q_f64(a_ptr.add(offset + 2));
        let b1 = vld1q_f64(b_ptr.add(offset + 2));
        accum1 = vfmaq_f64(accum1, a1, b1);

        let a2 = vld1q_f64(a_ptr.add(offset + 4));
        let b2 = vld1q_f64(b_ptr.add(offset + 4));
        accum2 = vfmaq_f64(accum2, a2, b2);

        let a3 = vld1q_f64(a_ptr.add(offset + 6));
        let b3 = vld1q_f64(b_ptr.add(offset + 6));
        accum3 = vfmaq_f64(accum3, a3, b3);

        let a4 = vld1q_f64(a_ptr.add(offset + 8));
        let b4 = vld1q_f64(b_ptr.add(offset + 8));
        accum4 = vfmaq_f64(accum4, a4, b4);

        let a5 = vld1q_f64(a_ptr.add(offset + 10));
        let b5 = vld1q_f64(b_ptr.add(offset + 10));
        accum5 = vfmaq_f64(accum5, a5, b5);

        let a6 = vld1q_f64(a_ptr.add(offset + 12));
        let b6 = vld1q_f64(b_ptr.add(offset + 12));
        accum6 = vfmaq_f64(accum6, a6, b6);

        let a7 = vld1q_f64(a_ptr.add(offset + 14));
        let b7 = vld1q_f64(b_ptr.add(offset + 14));
        accum7 = vfmaq_f64(accum7, a7, b7);

        i += chunk_size;
    }

    // Combine accumulators
    accum0 = vaddq_f64(accum0, accum1);
    accum2 = vaddq_f64(accum2, accum3);
    accum4 = vaddq_f64(accum4, accum5);
    accum6 = vaddq_f64(accum6, accum7);

    accum0 = vaddq_f64(accum0, accum2);
    accum4 = vaddq_f64(accum4, accum6);

    let total = vaddq_f64(accum0, accum4);

    // Horizontal sum
    let sum = hsum_neon(total);

    // Process remaining elements
    sum + dot_product_scalar(&a[remainder_start..], &b[remainder_start..])
}

/// Optimized scalar fallback with manual unrolling
#[inline(always)]
fn dot_product_scalar(a: &[f64], b: &[f64]) -> f64 {
    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;

    let mut i = 0;
    let len = a.len();
    let upper = len - (len % 4);

    while i < upper {
        sum0 += a[i] * b[i];
        sum1 += a[i + 1] * b[i + 1];
        sum2 += a[i + 2] * b[i + 2];
        sum3 += a[i + 3] * b[i + 3];
        i += 4;
    }

    let mut total = sum0 + sum1 + sum2 + sum3;

    // Process remaining elements
    while i < len {
        total += a[i] * b[i];
        i += 1;
    }

    total
}

// --- Euclidean Distance Squared (and then Euclidean Distance) ---

/// Computes the squared Euclidean distance: sum((a[i] - b[i])^2)
/// This is the core computation before the final sqrt.
fn euclidean_distance_squared(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "Slices must have equal length for euclidean_distance_squared"
    );
    if a.is_empty() {
        return 0.0;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // AVX2 is sufficient for (a-b)^2. FMA is not directly helpful here.
        if is_x86_feature_detected!("avx2") {
            return unsafe { euclidean_distance_squared_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
            return unsafe { euclidean_distance_squared_neon(a, b) };
        }
    }
    euclidean_distance_squared_scalar(a, b)
}
/// Computes the Euclidean distance between two f64 slices
/// # Panics
/// Panics if the slices have different lengths
pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    let dist_sq = euclidean_distance_squared(a, b);
    dist_sq.sqrt()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")] // FMA not needed for (a-b)^2
unsafe fn euclidean_distance_squared_avx2(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len();
    let mut accum0 = _mm256_setzero_pd();
    let mut accum1 = _mm256_setzero_pd();
    let mut accum2 = _mm256_setzero_pd();
    let mut accum3 = _mm256_setzero_pd();
    let mut accum4 = _mm256_setzero_pd();
    let mut accum5 = _mm256_setzero_pd();
    let mut accum6 = _mm256_setzero_pd();
    let mut accum7 = _mm256_setzero_pd();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let mut i = 0;

    let chunk_size = 32; // 8 vectors * 4 f64
    let iterations = len / chunk_size;
    let remainder_start = iterations * chunk_size;

    for _ in 0..iterations {
        let offset = i;

        let a0 = _mm256_loadu_pd(a_ptr.add(offset));
        let b0 = _mm256_loadu_pd(b_ptr.add(offset));
        let diff0 = _mm256_sub_pd(a0, b0);
        accum0 = _mm256_add_pd(accum0, _mm256_mul_pd(diff0, diff0)); // (a-b)^2, could use FMA as diff0*diff0 + accum0 but add is fine

        let a1 = _mm256_loadu_pd(a_ptr.add(offset + 4));
        let b1 = _mm256_loadu_pd(b_ptr.add(offset + 4));
        let diff1 = _mm256_sub_pd(a1, b1);
        accum1 = _mm256_add_pd(accum1, _mm256_mul_pd(diff1, diff1));

        let a2 = _mm256_loadu_pd(a_ptr.add(offset + 8));
        let b2 = _mm256_loadu_pd(b_ptr.add(offset + 8));
        let diff2 = _mm256_sub_pd(a2, b2);
        accum2 = _mm256_add_pd(accum2, _mm256_mul_pd(diff2, diff2));

        let a3 = _mm256_loadu_pd(a_ptr.add(offset + 12));
        let b3 = _mm256_loadu_pd(b_ptr.add(offset + 12));
        let diff3 = _mm256_sub_pd(a3, b3);
        accum3 = _mm256_add_pd(accum3, _mm256_mul_pd(diff3, diff3));

        let a4 = _mm256_loadu_pd(a_ptr.add(offset + 16));
        let b4 = _mm256_loadu_pd(b_ptr.add(offset + 16));
        let diff4 = _mm256_sub_pd(a4, b4);
        accum4 = _mm256_add_pd(accum4, _mm256_mul_pd(diff4, diff4));

        let a5 = _mm256_loadu_pd(a_ptr.add(offset + 20));
        let b5 = _mm256_loadu_pd(b_ptr.add(offset + 20));
        let diff5 = _mm256_sub_pd(a5, b5);
        accum5 = _mm256_add_pd(accum5, _mm256_mul_pd(diff5, diff5));

        let a6 = _mm256_loadu_pd(a_ptr.add(offset + 24));
        let b6 = _mm256_loadu_pd(b_ptr.add(offset + 24));
        let diff6 = _mm256_sub_pd(a6, b6);
        accum6 = _mm256_add_pd(accum6, _mm256_mul_pd(diff6, diff6));

        let a7 = _mm256_loadu_pd(a_ptr.add(offset + 28));
        let b7 = _mm256_loadu_pd(b_ptr.add(offset + 28));
        let diff7 = _mm256_sub_pd(a7, b7);
        accum7 = _mm256_add_pd(accum7, _mm256_mul_pd(diff7, diff7));

        i += chunk_size;
    }

    accum0 = _mm256_add_pd(accum0, accum1);
    accum2 = _mm256_add_pd(accum2, accum3);
    accum4 = _mm256_add_pd(accum4, accum5);
    accum6 = _mm256_add_pd(accum6, accum7);

    accum0 = _mm256_add_pd(accum0, accum2);
    accum4 = _mm256_add_pd(accum4, accum6);

    let total_vec = _mm256_add_pd(accum0, accum4);
    let sum = hsum_avx(total_vec);

    sum + euclidean_distance_squared_scalar(&a[remainder_start..], &b[remainder_start..])
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn euclidean_distance_squared_neon(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len();
    let mut accum0 = vdupq_n_f64(0.0);
    let mut accum1 = vdupq_n_f64(0.0);
    let mut accum2 = vdupq_n_f64(0.0);
    let mut accum3 = vdupq_n_f64(0.0);
    let mut accum4 = vdupq_n_f64(0.0);
    let mut accum5 = vdupq_n_f64(0.0);
    let mut accum6 = vdupq_n_f64(0.0);
    let mut accum7 = vdupq_n_f64(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let mut i = 0;

    let chunk_size = 16; // 8 vectors * 2 f64
    let iterations = len / chunk_size;
    let remainder_start = iterations * chunk_size;

    for _ in 0..iterations {
        let offset = i;

        let va0 = vld1q_f64(a_ptr.add(offset));
        let vb0 = vld1q_f64(b_ptr.add(offset));
        let diff0 = vsubq_f64(va0, vb0);
        accum0 = vfmaq_f64(accum0, diff0, diff0); // Neon FMA: accum + diff * diff

        let va1 = vld1q_f64(a_ptr.add(offset + 2));
        let vb1 = vld1q_f64(b_ptr.add(offset + 2));
        let diff1 = vsubq_f64(va1, vb1);
        accum1 = vfmaq_f64(accum1, diff1, diff1);

        let va2 = vld1q_f64(a_ptr.add(offset + 4));
        let vb2 = vld1q_f64(b_ptr.add(offset + 4));
        let diff2 = vsubq_f64(va2, vb2);
        accum2 = vfmaq_f64(accum2, diff2, diff2);

        let va3 = vld1q_f64(a_ptr.add(offset + 6));
        let vb3 = vld1q_f64(b_ptr.add(offset + 6));
        let diff3 = vsubq_f64(va3, vb3);
        accum3 = vfmaq_f64(accum3, diff3, diff3);

        let va4 = vld1q_f64(a_ptr.add(offset + 8));
        let vb4 = vld1q_f64(b_ptr.add(offset + 8));
        let diff4 = vsubq_f64(va4, vb4);
        accum4 = vfmaq_f64(accum4, diff4, diff4);

        let va5 = vld1q_f64(a_ptr.add(offset + 10));
        let vb5 = vld1q_f64(b_ptr.add(offset + 10));
        let diff5 = vsubq_f64(va5, vb5);
        accum5 = vfmaq_f64(accum5, diff5, diff5);

        let va6 = vld1q_f64(a_ptr.add(offset + 12));
        let vb6 = vld1q_f64(b_ptr.add(offset + 12));
        let diff6 = vsubq_f64(va6, vb6);
        accum6 = vfmaq_f64(accum6, diff6, diff6);

        let va7 = vld1q_f64(a_ptr.add(offset + 14));
        let vb7 = vld1q_f64(b_ptr.add(offset + 14));
        let diff7 = vsubq_f64(va7, vb7);
        accum7 = vfmaq_f64(accum7, diff7, diff7);

        i += chunk_size;
    }

    accum0 = vaddq_f64(accum0, accum1);
    accum2 = vaddq_f64(accum2, accum3);
    accum4 = vaddq_f64(accum4, accum5);
    accum6 = vaddq_f64(accum6, accum7);

    accum0 = vaddq_f64(accum0, accum2);
    accum4 = vaddq_f64(accum4, accum6);

    let total_vec = vaddq_f64(accum0, accum4);
    let sum = hsum_neon(total_vec);

    sum + euclidean_distance_squared_scalar(&a[remainder_start..], &b[remainder_start..])
}

#[inline(always)]
fn euclidean_distance_squared_scalar(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len();
    if len == 0 {
        return 0.0;
    }

    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;

    let mut i = 0;
    let upper = len - (len % 4);

    while i < upper {
        let d0 = a[i] - b[i];
        sum0 += d0 * d0;
        let d1 = a[i + 1] - b[i + 1];
        sum1 += d1 * d1;
        let d2 = a[i + 2] - b[i + 2];
        sum2 += d2 * d2;
        let d3 = a[i + 3] - b[i + 3];
        sum3 += d3 * d3;
        i += 4;
    }

    let mut total_sum = sum0 + sum1 + sum2 + sum3;

    while i < len {
        let d = a[i] - b[i];
        total_sum += d * d;
        i += 1;
    }
    total_sum
}

// Horizontal sum utilities
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn hsum_avx(v: __m256d) -> f64 {
    let hi = _mm256_extractf128_pd(v, 1);
    let lo = _mm256_extractf128_pd(v, 0);
    let sum = _mm_add_pd(hi, lo);
    let hi64 = _mm_unpackhi_pd(sum, sum);
    _mm_cvtsd_f64(_mm_add_sd(sum, hi64))
}

#[cfg(target_arch = "aarch64")]
unsafe fn hsum_neon(v: float64x2_t) -> f64 {
    vgetq_lane_f64(v, 0) + vgetq_lane_f64(v, 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use rand::{thread_rng, Rng};

    fn random_f64_vec(len: usize) -> Vec<f64> {
        let mut rng = thread_rng();
        (0..len).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    #[test]
    fn test_dot_product_consistency_small() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert_abs_diff_eq!(dot_product(&a, &b), 32.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dot_specialized_vs_general() {
        let a = random_f64_vec(1024);
        let b = random_f64_vec(1024);

        let simd_result = dot_product(&a, &b);
        let scalar_result = dot_product_scalar(&a, &b);

        assert_abs_diff_eq!(simd_result, scalar_result, epsilon = 1e-10);
    }

    #[test]
    fn test_dot_edge_cases() {
        // Zeros
        let zeros = vec![0.0; 100];
        assert_abs_diff_eq!(dot_product(&zeros, &zeros), 0.0, epsilon = 1e-10);

        // Denormals
        let denormals = vec![f64::MIN_POSITIVE; 100];
        let expected = (f64::MIN_POSITIVE * f64::MIN_POSITIVE) * 100.0;
        assert_abs_diff_eq!(
            dot_product(&denormals, &denormals),
            expected,
            epsilon = 1e-10
        );

        // Mixed signs
        let a = vec![1.0, -1.0, 2.0, -2.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        assert_abs_diff_eq!(dot_product(&a, &b), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dot_various_lengths() {
        for len in 0..50 {
            let a = random_f64_vec(len);
            let b = random_f64_vec(len);

            let simd_result = dot_product(&a, &b);
            let scalar_result = dot_product_scalar(&a, &b);

            assert_abs_diff_eq!(simd_result, scalar_result, epsilon = 1e-10,);
        }
    }

    // --- Euclidean Distance Squared (and then Euclidean Distance) ---
    #[test]
    fn test_euclidean_distance_consistency_small() {
        let a = vec![1.0, 2.0, 3.0]; // (1-4)^2 + (2-5)^2 + (3-6)^2 = (-3)^2 + (-3)^2 + (-3)^2 = 9+9+9 = 27
        let b = vec![4.0, 5.0, 6.0]; // sqrt(27) approx 5.1961524227
        assert_abs_diff_eq!(euclidean_distance(&a, &b), 27.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_euclidean_distance_zero() {
        let a = vec![1.0, 2.0, 3.0];
        assert_abs_diff_eq!(euclidean_distance(&a, &a), 0.0, epsilon = 1e-10);
        let empty: Vec<f64> = vec![];
        assert_abs_diff_eq!(euclidean_distance(&empty, &empty), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_euclidean_spcialized_vs_general() {
        for len in [0, 1, 3, 4, 7, 15, 16, 31, 32, 33, 63, 64, 65, 100, 1024] {
            let a = random_f64_vec(len);
            let b = random_f64_vec(len);

            let simd_result = euclidean_distance(&a, &b);
            // Calculate scalar result carefully
            let scalar_sq_sum = euclidean_distance_squared_scalar(&a, &b);
            let scalar_result = scalar_sq_sum.sqrt();

            assert_abs_diff_eq!(simd_result, scalar_result, epsilon = 1e-9);
        }
    }
    #[test]
    fn test_euclidean_vs_dot() {
        // This test checks if the two methods for Euclidean distance produce similar results
        // High precision might not be achievable due to floating point arithmetic differences.
        for len in [10, 33, 128, 513] {
            // Larger lengths might show more deviation
            let a = random_f64_vec(len);
            let b = random_f64_vec(len);

            let direct_dist = euclidean_distance(&a, &b);

            let a_dot_a = dot_product(&a, &a);
            let b_dot_b = dot_product(&b, &b);
            let a_dot_b = dot_product(&a, &b);
            let term = a_dot_a + b_dot_b - 2.0 * a_dot_b;
            let dot_based_dist = if term < 0.0 { 0.0 } else { term.sqrt() };

            // Use a slightly larger epsilon for this comparison due to different computation paths
            assert_abs_diff_eq!(direct_dist, dot_based_dist, epsilon = 1e-7);
        }
    }
}
