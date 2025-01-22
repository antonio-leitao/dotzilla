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

// AVX2+FMA implementation (x86/x86_64)
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_avx2_fma(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len();
    let mut accum0 = _mm256_setzero_pd();
    let mut accum1 = _mm256_setzero_pd();
    let mut accum2 = _mm256_setzero_pd();
    let mut accum3 = _mm256_setzero_pd();

    let mut i = 0;
    while i + 32 <= len {
        let a_ptr = a.as_ptr().add(i);
        let b_ptr = b.as_ptr().add(i);

        // Process 32 elements per iteration (4 vectors × 8 elements)
        let a0 = _mm256_loadu_pd(a_ptr);
        let b0 = _mm256_loadu_pd(b_ptr);
        accum0 = _mm256_fmadd_pd(a0, b0, accum0);

        let a1 = _mm256_loadu_pd(a_ptr.add(4));
        let b1 = _mm256_loadu_pd(b_ptr.add(4));
        accum1 = _mm256_fmadd_pd(a1, b1, accum1);

        let a2 = _mm256_loadu_pd(a_ptr.add(8));
        let b2 = _mm256_loadu_pd(b_ptr.add(8));
        accum2 = _mm256_fmadd_pd(a2, b2, accum2);

        let a3 = _mm256_loadu_pd(a_ptr.add(12));
        let b3 = _mm256_loadu_pd(b_ptr.add(12));
        accum3 = _mm256_fmadd_pd(a3, b3, accum3);

        i += 16;
    }

    // Horizontal sum of accumulators
    let sum = hsum_avx(accum0) + hsum_avx(accum1) + hsum_avx(accum2) + hsum_avx(accum3);

    // Process remaining elements with scalar
    sum + dot_product_scalar(&a[i..], &b[i..])
}

// NEON implementation (aarch64)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_product_neon(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len();
    let mut accum0 = vdupq_n_f64(0.0);
    let mut accum1 = vdupq_n_f64(0.0);
    let mut accum2 = vdupq_n_f64(0.0);
    let mut accum3 = vdupq_n_f64(0.0);

    let mut i = 0;
    while i + 16 <= len {
        let a_ptr = a.as_ptr().add(i);
        let b_ptr = b.as_ptr().add(i);

        // Process 16 elements per iteration (4 vectors × 4 elements)
        let a0 = vld1q_f64(a_ptr);
        let b0 = vld1q_f64(b_ptr);
        accum0 = vfmaq_f64(accum0, a0, b0);

        let a1 = vld1q_f64(a_ptr.add(2));
        let b1 = vld1q_f64(b_ptr.add(2));
        accum1 = vfmaq_f64(accum1, a1, b1);

        let a2 = vld1q_f64(a_ptr.add(4));
        let b2 = vld1q_f64(b_ptr.add(4));
        accum2 = vfmaq_f64(accum2, a2, b2);

        let a3 = vld1q_f64(a_ptr.add(6));
        let b3 = vld1q_f64(b_ptr.add(6));
        accum3 = vfmaq_f64(accum3, a3, b3);

        i += 8;
    }

    // Combine accumulators
    let sum = hsum_neon(accum0) + hsum_neon(accum1) + hsum_neon(accum2) + hsum_neon(accum3);
    sum + dot_product_scalar(&a[i..], &b[i..])
}

/// Optimized scalar fallback with manual unrolling
#[inline(always)]
fn dot_product_scalar(a: &[f64], b: &[f64]) -> f64 {
    let mut a_chunks = a.chunks_exact(8);
    let mut b_chunks = b.chunks_exact(8);
    let mut sum = 0.0;

    // Process chunks
    for (a_chunk, b_chunk) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        sum += a_chunk[0] * b_chunk[0]
            + a_chunk[1] * b_chunk[1]
            + a_chunk[2] * b_chunk[2]
            + a_chunk[3] * b_chunk[3]
            + a_chunk[4] * b_chunk[4]
            + a_chunk[5] * b_chunk[5]
            + a_chunk[6] * b_chunk[6]
            + a_chunk[7] * b_chunk[7];
    }

    // Process remainders
    let a_rem = a_chunks.remainder();
    let b_rem = b_chunks.remainder();
    sum + a_rem
        .iter()
        .zip(b_rem)
        .fold(0.0, |acc, (&a, &b)| acc + a * b)
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
    fn test_consistency_small() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert_abs_diff_eq!(dot_product(&a, &b), 32.0, epsilon = 1e-10);
    }

    #[test]
    fn test_against_scalar_random() {
        let a = random_f64_vec(1024);
        let b = random_f64_vec(1024);

        let simd_result = dot_product(&a, &b);
        let scalar_result = dot_product_scalar(&a, &b);

        assert_abs_diff_eq!(simd_result, scalar_result, epsilon = 1e-10);
    }

    #[test]
    fn test_edge_cases() {
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
    fn test_various_lengths() {
        for len in 0..50 {
            let a = random_f64_vec(len);
            let b = random_f64_vec(len);

            let simd_result = dot_product(&a, &b);
            let scalar_result = dot_product_scalar(&a, &b);

            assert_abs_diff_eq!(simd_result, scalar_result, epsilon = 1e-10,);
        }
    }
}
