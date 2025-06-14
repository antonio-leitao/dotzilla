// src/lib.rs
//! High-performance dot_product and euclidean_distance with automatic SIMD dispatch

pub mod dotf32;
pub mod dotf64;
pub mod quantized;

/// Trait providing dot product implementation for floating-point slices
pub trait DotProduct {
    /// Result type matching the input precision
    type Output;
    /// Compute the dot product of two vectors
    ///
    /// # Panics
    /// Panics if vectors have different lengths
    fn dot_product(&self, other: &Self) -> Self::Output;
}

/// Trait providing euclidean distance implementation for floating-point slices
pub trait EuclideanDistance {
    /// Result type matching the input precision
    type Output;
    /// Compute the euclidean distance between two vectors
    ///
    /// # Panics
    /// Panics if vectors have different lengths
    fn euclidean_distance(&self, other: &Self) -> Self::Output;
}

impl DotProduct for [f32] {
    type Output = f32;
    #[inline]
    fn dot_product(&self, other: &Self) -> Self::Output {
        dotf32::dot_product(self, other)
    }
}

impl DotProduct for [f64] {
    type Output = f64;
    #[inline]
    fn dot_product(&self, other: &Self) -> Self::Output {
        dotf64::dot_product(self, other)
    }
}

impl EuclideanDistance for [f32] {
    type Output = f32;
    #[inline]
    fn euclidean_distance(&self, other: &Self) -> Self::Output {
        dotf32::euclidean_distance(self, other)
    }
}

impl EuclideanDistance for [f64] {
    type Output = f64;
    #[inline]
    fn euclidean_distance(&self, other: &Self) -> Self::Output {
        dotf64::euclidean_distance(self, other)
    }
}

/// Top-level dot product function for ergonomic usage
///
/// # Examples
/// ```
/// use dotzilla::dot_product;
///
/// let a = vec![1.0f32, 2.0, 3.0];
/// let b = vec![4.0f32, 5.0, 6.0];
/// assert_eq!(dot_product(&a, &b), 32.0);
/// ```
#[inline]
pub fn dot_product<T>(a: &[T], b: &[T]) -> <[T] as DotProduct>::Output
where
    [T]: DotProduct,
{
    a.dot_product(b)
}

/// Top-level euclidean distance function for ergonomic usage
///
/// # Examples
/// ```
/// use dotzilla::euclidean_distance;
///
/// let a = vec![1.0f32, 2.0, 3.0];
/// let b = vec![4.0f32, 5.0, 6.0];
/// assert_eq!(euclidean_distance(&a, &b), 5.196152);
/// ```
#[inline]
pub fn euclidean_distance<T>(a: &[T], b: &[T]) -> <[T] as EuclideanDistance>::Output
where
    [T]: EuclideanDistance,
{
    a.euclidean_distance(b)
}
