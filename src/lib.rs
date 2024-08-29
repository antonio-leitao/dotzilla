pub mod dot_avx;
pub mod dot_neon;
pub mod dot_sse;
pub mod dot_std;

#[cfg(all(target_feature = "avx", target_feature = "fma"))]
use dot_avx::*;

#[cfg(target_feature = "neon")]
use dot_neon::*;

#[cfg(target_feature = "sse")]
use dot_sse::*;

// Fallback implementation: if none of the above features are enabled, use the standard implementation
#[cfg(not(any(
    all(target_feature = "avx", target_feature = "fma"),
    target_feature = "neon",
    target_feature = "sse",
)))]
use dot_std::*;

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    return unsafe { inner_product(a, b) };
}

pub fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    return unsafe { euclidean(a, b) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inner_product() {
        let a = vec![1.0f32; 1024];
        let b = vec![2.0f32; 1024];
        assert_eq!(dot(&a, &b), dot_std::inner_product(&a, &b));
    }
    #[test]
    fn test_euclidean() {
        let a = vec![1.0f32; 1024];
        let b = vec![2.0f32; 1024];
        assert_eq!(l2sq(&a, &b), dot_std::euclidean(&a, &b));
    }
}
