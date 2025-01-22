#![allow(non_snake_case, clippy::missing_safety_doc)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
use std::arch::is_aarch64_feature_detected;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

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

    dot_product_scalar(a, b)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,avx,fma")]
unsafe fn dot_product_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut accum0 = _mm256_setzero_ps();
    let mut accum1 = _mm256_setzero_ps();
    let mut accum2 = _mm256_setzero_ps();
    let mut accum3 = _mm256_setzero_ps();
    let mut accum4 = _mm256_setzero_ps();
    let mut accum5 = _mm256_setzero_ps();
    let mut accum6 = _mm256_setzero_ps();
    let mut accum7 = _mm256_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let mut i = 0;

    // Process 64 elements per iteration (8 AVX registers * 8 floats)
    let unroll = 8;
    let chunk_size = 8 * unroll;
    let iterations = len / chunk_size;
    let remainder = len % chunk_size;

    for _ in 0..iterations {
        let offset = i;

        let a0 = _mm256_loadu_ps(a_ptr.add(offset));
        let b0 = _mm256_loadu_ps(b_ptr.add(offset));
        accum0 = _mm256_fmadd_ps(a0, b0, accum0);

        let a1 = _mm256_loadu_ps(a_ptr.add(offset + 8));
        let b1 = _mm256_loadu_ps(b_ptr.add(offset + 8));
        accum1 = _mm256_fmadd_ps(a1, b1, accum1);

        let a2 = _mm256_loadu_ps(a_ptr.add(offset + 16));
        let b2 = _mm256_loadu_ps(b_ptr.add(offset + 16));
        accum2 = _mm256_fmadd_ps(a2, b2, accum2);

        let a3 = _mm256_loadu_ps(a_ptr.add(offset + 24));
        let b3 = _mm256_loadu_ps(b_ptr.add(offset + 24));
        accum3 = _mm256_fmadd_ps(a3, b3, accum3);

        let a4 = _mm256_loadu_ps(a_ptr.add(offset + 32));
        let b4 = _mm256_loadu_ps(b_ptr.add(offset + 32));
        accum4 = _mm256_fmadd_ps(a4, b4, accum4);

        let a5 = _mm256_loadu_ps(a_ptr.add(offset + 40));
        let b5 = _mm256_loadu_ps(b_ptr.add(offset + 40));
        accum5 = _mm256_fmadd_ps(a5, b5, accum5);

        let a6 = _mm256_loadu_ps(a_ptr.add(offset + 48));
        let b6 = _mm256_loadu_ps(b_ptr.add(offset + 48));
        accum6 = _mm256_fmadd_ps(a6, b6, accum6);

        let a7 = _mm256_loadu_ps(a_ptr.add(offset + 56));
        let b7 = _mm256_loadu_ps(b_ptr.add(offset + 56));
        accum7 = _mm256_fmadd_ps(a7, b7, accum7);

        i += chunk_size;
    }

    // Combine accumulators
    accum0 = _mm256_add_ps(accum0, accum1);
    accum2 = _mm256_add_ps(accum2, accum3);
    accum4 = _mm256_add_ps(accum4, accum5);
    accum6 = _mm256_add_ps(accum6, accum7);

    accum0 = _mm256_add_ps(accum0, accum2);
    accum4 = _mm256_add_ps(accum4, accum6);

    let total = _mm256_add_ps(accum0, accum4);

    // Horizontal sum
    let sum = hsum_avx(total);

    // Process remaining elements
    let mut rem_sum = 0.0f32;
    for j in (iterations * chunk_size)..len {
        rem_sum += a[j] * b[j];
    }

    sum + rem_sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
unsafe fn hsum_avx(v: __m256) -> f32 {
    let vlow = _mm256_extractf128_ps(v, 0);
    let vhigh = _mm256_extractf128_ps(v, 1);
    let sum128 = _mm_add_ps(vlow, vhigh);
    let sum64 = _mm_hadd_ps(sum128, sum128);
    let sum32 = _mm_hadd_ps(sum64, sum64);
    _mm_cvtss_f32(sum32)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut accum0 = vdupq_n_f32(0.0);
    let mut accum1 = vdupq_n_f32(0.0);
    let mut accum2 = vdupq_n_f32(0.0);
    let mut accum3 = vdupq_n_f32(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let mut i = 0;

    // Process 16 elements per iteration (4 NEON registers * 4 floats)
    let chunk_size = 16;
    let iterations = len / chunk_size;

    for _ in 0..iterations {
        let offset = i;

        let a0 = vld1q_f32(a_ptr.add(offset));
        let b0 = vld1q_f32(b_ptr.add(offset));
        accum0 = vfmaq_f32(accum0, a0, b0);

        let a1 = vld1q_f32(a_ptr.add(offset + 4));
        let b1 = vld1q_f32(b_ptr.add(offset + 4));
        accum1 = vfmaq_f32(accum1, a1, b1);

        let a2 = vld1q_f32(a_ptr.add(offset + 8));
        let b2 = vld1q_f32(b_ptr.add(offset + 8));
        accum2 = vfmaq_f32(accum2, a2, b2);

        let a3 = vld1q_f32(a_ptr.add(offset + 12));
        let b3 = vld1q_f32(b_ptr.add(offset + 12));
        accum3 = vfmaq_f32(accum3, a3, b3);

        i += chunk_size;
    }

    // Combine accumulators
    accum0 = vaddq_f32(accum0, accum1);
    accum2 = vaddq_f32(accum2, accum3);
    let total = vaddq_f32(accum0, accum2);

    // Horizontal sum
    let sum = hsum_neon(total);

    // Process remaining elements
    let mut rem_sum = 0.0f32;
    for j in (iterations * chunk_size)..len {
        rem_sum += a[j] * b[j];
    }

    sum + rem_sum
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn hsum_neon(v: float32x4_t) -> f32 {
    let sum_pair = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    let sum = vpadd_f32(sum_pair, sum_pair);
    vget_lane_f32(sum, 0)
}

#[inline(always)]
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut a_chunks = a.chunks_exact(8);
    let mut b_chunks = b.chunks_exact(8);
    let mut sum = 0.0;

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

    let a_rem = a_chunks.remainder();
    let b_rem = b_chunks.remainder();
    sum + a_rem
        .iter()
        .zip(b_rem)
        .fold(0.0, |acc, (&a, &b)| acc + a * b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use rand::{thread_rng, Rng};

    fn random_f32_vec(len: usize) -> Vec<f32> {
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
        let a = random_f32_vec(1024);
        let b = random_f32_vec(1024);

        let simd_result = dot_product(&a, &b);
        let scalar_result = dot_product_scalar(&a, &b);

        assert_abs_diff_eq!(simd_result, scalar_result, epsilon = 1e-5);
    }

    #[test]
    fn test_edge_cases() {
        // Zeros
        let zeros = vec![0.0; 100];
        assert_abs_diff_eq!(dot_product(&zeros, &zeros), 0.0, epsilon = 1e-10);

        // Denormals
        let denormals = vec![f32::MIN_POSITIVE; 100];
        let expected = (f32::MIN_POSITIVE * f32::MIN_POSITIVE) * 100.0;
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
            let a = random_f32_vec(len);
            let b = random_f32_vec(len);

            let simd_result = dot_product(&a, &b);
            let scalar_result = dot_product_scalar(&a, &b);

            assert_abs_diff_eq!(simd_result, scalar_result, epsilon = 1e-6);
        }
    }
}
