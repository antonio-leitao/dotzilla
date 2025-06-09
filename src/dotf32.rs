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

    // Process 128 elements per iteration (16 AVX registers × 8 floats)
    let chunk_size = 128;
    let iterations = len / chunk_size;
    let remainder_start = iterations * chunk_size;

    for _ in 0..iterations {
        let offset = i;

        // First 64 elements
        accum0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a_ptr.add(offset)),
            _mm256_loadu_ps(b_ptr.add(offset)),
            accum0,
        );
        accum1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 8)),
            _mm256_loadu_ps(b_ptr.add(offset + 8)),
            accum1,
        );
        accum2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 16)),
            _mm256_loadu_ps(b_ptr.add(offset + 16)),
            accum2,
        );
        accum3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 24)),
            _mm256_loadu_ps(b_ptr.add(offset + 24)),
            accum3,
        );
        accum4 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 32)),
            _mm256_loadu_ps(b_ptr.add(offset + 32)),
            accum4,
        );
        accum5 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 40)),
            _mm256_loadu_ps(b_ptr.add(offset + 40)),
            accum5,
        );
        accum6 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 48)),
            _mm256_loadu_ps(b_ptr.add(offset + 48)),
            accum6,
        );
        accum7 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 56)),
            _mm256_loadu_ps(b_ptr.add(offset + 56)),
            accum7,
        );

        // Second 64 elements
        accum0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 64)),
            _mm256_loadu_ps(b_ptr.add(offset + 64)),
            accum0,
        );
        accum1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 72)),
            _mm256_loadu_ps(b_ptr.add(offset + 72)),
            accum1,
        );
        accum2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 80)),
            _mm256_loadu_ps(b_ptr.add(offset + 80)),
            accum2,
        );
        accum3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 88)),
            _mm256_loadu_ps(b_ptr.add(offset + 88)),
            accum3,
        );
        accum4 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 96)),
            _mm256_loadu_ps(b_ptr.add(offset + 96)),
            accum4,
        );
        accum5 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 104)),
            _mm256_loadu_ps(b_ptr.add(offset + 104)),
            accum5,
        );
        accum6 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 112)),
            _mm256_loadu_ps(b_ptr.add(offset + 112)),
            accum6,
        );
        accum7 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 120)),
            _mm256_loadu_ps(b_ptr.add(offset + 120)),
            accum7,
        );

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

    // Process remaining elements with optimized scalar
    sum + dot_product_scalar(&a[remainder_start..], &b[remainder_start..])
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut accum0 = vdupq_n_f32(0.0);
    let mut accum1 = vdupq_n_f32(0.0);
    let mut accum2 = vdupq_n_f32(0.0);
    let mut accum3 = vdupq_n_f32(0.0);
    let mut accum4 = vdupq_n_f32(0.0);
    let mut accum5 = vdupq_n_f32(0.0);
    let mut accum6 = vdupq_n_f32(0.0);
    let mut accum7 = vdupq_n_f32(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let mut i = 0;

    // Process 64 elements per iteration (8 NEON registers × 8 floats)
    let chunk_size = 64;
    let iterations = len / chunk_size;
    let remainder_start = iterations * chunk_size;

    for _ in 0..iterations {
        let offset = i;

        // Load and process 8 vectors
        accum0 = vfmaq_f32(
            accum0,
            vld1q_f32(a_ptr.add(offset)),
            vld1q_f32(b_ptr.add(offset)),
        );
        accum1 = vfmaq_f32(
            accum1,
            vld1q_f32(a_ptr.add(offset + 4)),
            vld1q_f32(b_ptr.add(offset + 4)),
        );
        accum2 = vfmaq_f32(
            accum2,
            vld1q_f32(a_ptr.add(offset + 8)),
            vld1q_f32(b_ptr.add(offset + 8)),
        );
        accum3 = vfmaq_f32(
            accum3,
            vld1q_f32(a_ptr.add(offset + 12)),
            vld1q_f32(b_ptr.add(offset + 12)),
        );
        accum4 = vfmaq_f32(
            accum4,
            vld1q_f32(a_ptr.add(offset + 16)),
            vld1q_f32(b_ptr.add(offset + 16)),
        );
        accum5 = vfmaq_f32(
            accum5,
            vld1q_f32(a_ptr.add(offset + 20)),
            vld1q_f32(b_ptr.add(offset + 20)),
        );
        accum6 = vfmaq_f32(
            accum6,
            vld1q_f32(a_ptr.add(offset + 24)),
            vld1q_f32(b_ptr.add(offset + 24)),
        );
        accum7 = vfmaq_f32(
            accum7,
            vld1q_f32(a_ptr.add(offset + 28)),
            vld1q_f32(b_ptr.add(offset + 28)),
        );

        // Second set of 32 elements
        accum0 = vfmaq_f32(
            accum0,
            vld1q_f32(a_ptr.add(offset + 32)),
            vld1q_f32(b_ptr.add(offset + 32)),
        );
        accum1 = vfmaq_f32(
            accum1,
            vld1q_f32(a_ptr.add(offset + 36)),
            vld1q_f32(b_ptr.add(offset + 36)),
        );
        accum2 = vfmaq_f32(
            accum2,
            vld1q_f32(a_ptr.add(offset + 40)),
            vld1q_f32(b_ptr.add(offset + 40)),
        );
        accum3 = vfmaq_f32(
            accum3,
            vld1q_f32(a_ptr.add(offset + 44)),
            vld1q_f32(b_ptr.add(offset + 44)),
        );
        accum4 = vfmaq_f32(
            accum4,
            vld1q_f32(a_ptr.add(offset + 48)),
            vld1q_f32(b_ptr.add(offset + 48)),
        );
        accum5 = vfmaq_f32(
            accum5,
            vld1q_f32(a_ptr.add(offset + 52)),
            vld1q_f32(b_ptr.add(offset + 52)),
        );
        accum6 = vfmaq_f32(
            accum6,
            vld1q_f32(a_ptr.add(offset + 56)),
            vld1q_f32(b_ptr.add(offset + 56)),
        );
        accum7 = vfmaq_f32(
            accum7,
            vld1q_f32(a_ptr.add(offset + 60)),
            vld1q_f32(b_ptr.add(offset + 60)),
        );

        i += chunk_size;
    }

    // Combine accumulators
    accum0 = vaddq_f32(accum0, accum1);
    accum2 = vaddq_f32(accum2, accum3);
    accum4 = vaddq_f32(accum4, accum5);
    accum6 = vaddq_f32(accum6, accum7);

    accum0 = vaddq_f32(accum0, accum2);
    accum4 = vaddq_f32(accum4, accum6);

    let total = vaddq_f32(accum0, accum4);

    // Horizontal sum
    let sum = hsum_neon(total);

    // Process remaining elements
    sum + dot_product_scalar(&a[remainder_start..], &b[remainder_start..])
}

#[inline(always)]
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;
    let mut sum4 = 0.0;
    let mut sum5 = 0.0;
    let mut sum6 = 0.0;
    let mut sum7 = 0.0;

    let mut i = 0;
    let len = a.len();
    let upper = len - (len % 8);

    while i < upper {
        sum0 += a[i] * b[i];
        sum1 += a[i + 1] * b[i + 1];
        sum2 += a[i + 2] * b[i + 2];
        sum3 += a[i + 3] * b[i + 3];
        sum4 += a[i + 4] * b[i + 4];
        sum5 += a[i + 5] * b[i + 5];
        sum6 += a[i + 6] * b[i + 6];
        sum7 += a[i + 7] * b[i + 7];
        i += 8;
    }

    let mut total = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7;

    // Process remaining elements
    while i < len {
        total += a[i] * b[i];
        i += 1;
    }

    total
}
// --- Euclidean Distance for f32 ---

/// Computes the Euclidean distance between two f32 slices: sqrt(sum((a[i] - b[i])^2))
/// # Panics
/// Panics if the slices have different lengths
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    let dist_sq = euclidean_distance_squared(a, b);
    dist_sq.sqrt()
}

/// Computes the squared Euclidean distance for f32 slices: sum((a[i] - b[i])^2)
fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "Slices must have equal length for euclidean_distance_squared_f32"
    );
    if a.is_empty() {
        return 0.0;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { euclidean_distance_squared_avx2_fma(a, b) };
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn euclidean_distance_squared_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
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

    let chunk_size = 128; // 8 accums * 2 vectors/accum * 8 floats/vector
    let iterations = len / chunk_size;
    let remainder_start = iterations * chunk_size;

    for _ in 0..iterations {
        let offset = i;
        let mut diff: __m256;

        diff = _mm256_sub_ps(
            _mm256_loadu_ps(a_ptr.add(offset)),
            _mm256_loadu_ps(b_ptr.add(offset)),
        );
        accum0 = _mm256_fmadd_ps(diff, diff, accum0);
        diff = _mm256_sub_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 8)),
            _mm256_loadu_ps(b_ptr.add(offset + 8)),
        );
        accum1 = _mm256_fmadd_ps(diff, diff, accum1);
        diff = _mm256_sub_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 16)),
            _mm256_loadu_ps(b_ptr.add(offset + 16)),
        );
        accum2 = _mm256_fmadd_ps(diff, diff, accum2);
        diff = _mm256_sub_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 24)),
            _mm256_loadu_ps(b_ptr.add(offset + 24)),
        );
        accum3 = _mm256_fmadd_ps(diff, diff, accum3);
        diff = _mm256_sub_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 32)),
            _mm256_loadu_ps(b_ptr.add(offset + 32)),
        );
        accum4 = _mm256_fmadd_ps(diff, diff, accum4);
        diff = _mm256_sub_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 40)),
            _mm256_loadu_ps(b_ptr.add(offset + 40)),
        );
        accum5 = _mm256_fmadd_ps(diff, diff, accum5);
        diff = _mm256_sub_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 48)),
            _mm256_loadu_ps(b_ptr.add(offset + 48)),
        );
        accum6 = _mm256_fmadd_ps(diff, diff, accum6);
        diff = _mm256_sub_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 56)),
            _mm256_loadu_ps(b_ptr.add(offset + 56)),
        );
        accum7 = _mm256_fmadd_ps(diff, diff, accum7);

        diff = _mm256_sub_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 64)),
            _mm256_loadu_ps(b_ptr.add(offset + 64)),
        );
        accum0 = _mm256_fmadd_ps(diff, diff, accum0);
        diff = _mm256_sub_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 72)),
            _mm256_loadu_ps(b_ptr.add(offset + 72)),
        );
        accum1 = _mm256_fmadd_ps(diff, diff, accum1);
        diff = _mm256_sub_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 80)),
            _mm256_loadu_ps(b_ptr.add(offset + 80)),
        );
        accum2 = _mm256_fmadd_ps(diff, diff, accum2);
        diff = _mm256_sub_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 88)),
            _mm256_loadu_ps(b_ptr.add(offset + 88)),
        );
        accum3 = _mm256_fmadd_ps(diff, diff, accum3);
        diff = _mm256_sub_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 96)),
            _mm256_loadu_ps(b_ptr.add(offset + 96)),
        );
        accum4 = _mm256_fmadd_ps(diff, diff, accum4);
        diff = _mm256_sub_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 104)),
            _mm256_loadu_ps(b_ptr.add(offset + 104)),
        );
        accum5 = _mm256_fmadd_ps(diff, diff, accum5);
        diff = _mm256_sub_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 112)),
            _mm256_loadu_ps(b_ptr.add(offset + 112)),
        );
        accum6 = _mm256_fmadd_ps(diff, diff, accum6);
        diff = _mm256_sub_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 120)),
            _mm256_loadu_ps(b_ptr.add(offset + 120)),
        );
        accum7 = _mm256_fmadd_ps(diff, diff, accum7);
        i += chunk_size;
    }

    accum0 = _mm256_add_ps(accum0, accum1);
    accum2 = _mm256_add_ps(accum2, accum3);
    accum4 = _mm256_add_ps(accum4, accum5);
    accum6 = _mm256_add_ps(accum6, accum7);
    accum0 = _mm256_add_ps(accum0, accum2);
    accum4 = _mm256_add_ps(accum4, accum6);
    let total_vec = _mm256_add_ps(accum0, accum4);
    let sum = hsum_avx_ps(total_vec);

    sum + euclidean_distance_squared_scalar_f32(&a[remainder_start..], &b[remainder_start..])
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn euclidean_distance_squared_neon(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut accum0 = vdupq_n_f32(0.0);
    let mut accum1 = vdupq_n_f32(0.0);
    let mut accum2 = vdupq_n_f32(0.0);
    let mut accum3 = vdupq_n_f32(0.0);
    let mut accum4 = vdupq_n_f32(0.0);
    let mut accum5 = vdupq_n_f32(0.0);
    let mut accum6 = vdupq_n_f32(0.0);
    let mut accum7 = vdupq_n_f32(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let mut i = 0;

    let chunk_size = 64; // 8 accums * 2 vectors/accum * 4 floats/vector
    let iterations = len / chunk_size;
    let remainder_start = iterations * chunk_size;

    for _ in 0..iterations {
        let offset = i;
        let mut diff: float32x4_t;

        diff = vsubq_f32(vld1q_f32(a_ptr.add(offset)), vld1q_f32(b_ptr.add(offset)));
        accum0 = vfmaq_f32(accum0, diff, diff);
        diff = vsubq_f32(
            vld1q_f32(a_ptr.add(offset + 4)),
            vld1q_f32(b_ptr.add(offset + 4)),
        );
        accum1 = vfmaq_f32(accum1, diff, diff);
        diff = vsubq_f32(
            vld1q_f32(a_ptr.add(offset + 8)),
            vld1q_f32(b_ptr.add(offset + 8)),
        );
        accum2 = vfmaq_f32(accum2, diff, diff);
        diff = vsubq_f32(
            vld1q_f32(a_ptr.add(offset + 12)),
            vld1q_f32(b_ptr.add(offset + 12)),
        );
        accum3 = vfmaq_f32(accum3, diff, diff);
        diff = vsubq_f32(
            vld1q_f32(a_ptr.add(offset + 16)),
            vld1q_f32(b_ptr.add(offset + 16)),
        );
        accum4 = vfmaq_f32(accum4, diff, diff);
        diff = vsubq_f32(
            vld1q_f32(a_ptr.add(offset + 20)),
            vld1q_f32(b_ptr.add(offset + 20)),
        );
        accum5 = vfmaq_f32(accum5, diff, diff);
        diff = vsubq_f32(
            vld1q_f32(a_ptr.add(offset + 24)),
            vld1q_f32(b_ptr.add(offset + 24)),
        );
        accum6 = vfmaq_f32(accum6, diff, diff);
        diff = vsubq_f32(
            vld1q_f32(a_ptr.add(offset + 28)),
            vld1q_f32(b_ptr.add(offset + 28)),
        );
        accum7 = vfmaq_f32(accum7, diff, diff);

        diff = vsubq_f32(
            vld1q_f32(a_ptr.add(offset + 32)),
            vld1q_f32(b_ptr.add(offset + 32)),
        );
        accum0 = vfmaq_f32(accum0, diff, diff);
        diff = vsubq_f32(
            vld1q_f32(a_ptr.add(offset + 36)),
            vld1q_f32(b_ptr.add(offset + 36)),
        );
        accum1 = vfmaq_f32(accum1, diff, diff);
        diff = vsubq_f32(
            vld1q_f32(a_ptr.add(offset + 40)),
            vld1q_f32(b_ptr.add(offset + 40)),
        );
        accum2 = vfmaq_f32(accum2, diff, diff);
        diff = vsubq_f32(
            vld1q_f32(a_ptr.add(offset + 44)),
            vld1q_f32(b_ptr.add(offset + 44)),
        );
        accum3 = vfmaq_f32(accum3, diff, diff);
        diff = vsubq_f32(
            vld1q_f32(a_ptr.add(offset + 48)),
            vld1q_f32(b_ptr.add(offset + 48)),
        );
        accum4 = vfmaq_f32(accum4, diff, diff);
        diff = vsubq_f32(
            vld1q_f32(a_ptr.add(offset + 52)),
            vld1q_f32(b_ptr.add(offset + 52)),
        );
        accum5 = vfmaq_f32(accum5, diff, diff);
        diff = vsubq_f32(
            vld1q_f32(a_ptr.add(offset + 56)),
            vld1q_f32(b_ptr.add(offset + 56)),
        );
        accum6 = vfmaq_f32(accum6, diff, diff);
        diff = vsubq_f32(
            vld1q_f32(a_ptr.add(offset + 60)),
            vld1q_f32(b_ptr.add(offset + 60)),
        );
        accum7 = vfmaq_f32(accum7, diff, diff);
        i += chunk_size;
    }

    accum0 = vaddq_f32(accum0, accum1);
    accum2 = vaddq_f32(accum2, accum3);
    accum4 = vaddq_f32(accum4, accum5);
    accum6 = vaddq_f32(accum6, accum7);
    accum0 = vaddq_f32(accum0, accum2);
    accum4 = vaddq_f32(accum4, accum6);
    let total_vec = vaddq_f32(accum0, accum4);
    let sum = hsum_neon(total_vec);

    sum + euclidean_distance_squared_scalar(&a[remainder_start..], &b[remainder_start..])
}

#[inline(always)]
fn euclidean_distance_squared_scalar(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    if len == 0 {
        return 0.0;
    }
    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;
    let mut sum4 = 0.0;
    let mut sum5 = 0.0;
    let mut sum6 = 0.0;
    let mut sum7 = 0.0;
    let mut i = 0;
    let upper = len - (len % 8); // Unroll 8 times
    while i < upper {
        let d0 = a[i] - b[i];
        sum0 += d0 * d0;
        let d1 = a[i + 1] - b[i + 1];
        sum1 += d1 * d1;
        let d2 = a[i + 2] - b[i + 2];
        sum2 += d2 * d2;
        let d3 = a[i + 3] - b[i + 3];
        sum3 += d3 * d3;
        let d4 = a[i + 4] - b[i + 4];
        sum4 += d4 * d4;
        let d5 = a[i + 5] - b[i + 5];
        sum5 += d5 * d5;
        let d6 = a[i + 6] - b[i + 6];
        sum6 += d6 * d6;
        let d7 = a[i + 7] - b[i + 7];
        sum7 += d7 * d7;
        i += 8;
    }
    let mut total_sum = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7;
    while i < len {
        let d = a[i] - b[i];
        total_sum += d * d;
        i += 1;
    }
    total_sum
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
#[inline]
unsafe fn hsum_neon(v: float32x4_t) -> f32 {
    let sum_pair = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    let sum = vpadd_f32(sum_pair, sum_pair);
    vget_lane_f32(sum, 0)
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
    fn test_dot_consistency_small() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert_abs_diff_eq!(dot_product(&a, &b), 32.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dot_spcialized_agains_general() {
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
    // New tests for Euclidean Distance f32
    #[test]
    fn test_euclidean_consistency_small() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        // (1-4)^2 + (2-5)^2 + (3-6)^2 = (-3)^2 + (-3)^2 + (-3)^2 = 9+9+9 = 27
        // sqrt(27) approx 5.1961524227
        let expected = (27.0f32).sqrt();
        assert_abs_diff_eq!(euclidean_distance(&a, &b), expected, epsilon = 1e-6);
    }

    #[test]
    fn test_euclidean_zero_distance() {
        let a = vec![1.0f32, -2.5, 3.14, 0.0];
        assert_abs_diff_eq!(euclidean_distance(&a, &a), 0.0, epsilon = 1e-7);
        let empty: Vec<f32> = vec![];
        assert_abs_diff_eq!(euclidean_distance(&empty, &empty), 0.0, epsilon = 1e-7);
    }

    #[test]
    fn test_euclidean_specialized_vs_general() {
        for len in [
            0, 1, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 129, 1000, 1024,
        ] {
            let a = random_f32_vec(len);
            let b = random_f32_vec(len);

            let simd_result = euclidean_distance(&a, &b);

            let scalar_sq_sum = euclidean_distance_squared_scalar(&a, &b);
            let scalar_result = scalar_sq_sum.sqrt();

            assert_abs_diff_eq!(simd_result, scalar_result, epsilon = 2e-5); // Increased epsilon for f32 accumulation
        }
    }
    #[test]
    fn test_euclidean_vs_dot() {
        // This test checks if the two methods for Euclidean distance produce similar results
        // High precision might not be achievable due to floating point arithmetic differences.
        for len in [10, 33, 128, 513, 1025] {
            let a = random_f32_vec(len);
            let b = random_f32_vec(len);

            let direct_dist = euclidean_distance(&a, &b);

            let a_dot_a = dot_product(&a, &a);
            let b_dot_b = dot_product(&b, &b);
            let a_dot_b = dot_product(&a, &b);
            let term = a_dot_a + b_dot_b - 2.0 * a_dot_b;
            let dot_based_dist = if term < 0.0 { 0.0 } else { term.sqrt() };

            // Use a slightly larger epsilon for this comparison due to different computation paths and f32 precision
            assert_abs_diff_eq!(direct_dist, dot_based_dist, epsilon = 5e-5,);
        }
    }
}
