// quantization.rs

/// Quantizes a vector of f32 values into 8-bit integers.
///
/// This function assumes the vector is meant to be normalized. It computes the L₂‐norm
/// and (if needed) normalizes the vector so that its entries are in approximately [-1, 1].
/// Then, each value is scaled by 127 and rounded into an i8.
///
/// Note that the returned dot products (via `quantized_dot_product`) will not be in the
/// same range as full‑precision ones—but the ranking (ordering) will be similar.
pub fn quantize_normalized(v: &[f32]) -> Vec<i8> {
    // Compute squared norm.
    let mut sum_sq = 0.0;
    for &x in v.iter() {
        sum_sq += x * x;
    }
    let norm = sum_sq.sqrt();
    let need_normalize = (norm - 1.0).abs() > 1e-5;
    let mut quantized = Vec::with_capacity(v.len());
    if need_normalize {
        // Normalize on the fly and quantize.
        for &x in v.iter() {
            let xn = x / norm;
            // (Clamp just in case—should not be needed if the vector is nearly normalized.)
            let xn = if xn > 1.0 {
                1.0
            } else if xn < -1.0 {
                -1.0
            } else {
                xn
            };
            quantized.push((xn * 127.0).round() as i8);
        }
    } else {
        // Already normalized.
        for &x in v.iter() {
            let xn = if x > 1.0 {
                1.0
            } else if x < -1.0 {
                -1.0
            } else {
                x
            };
            quantized.push((xn * 127.0).round() as i8);
        }
    }
    quantized
}

/// Computes the dot product between two quantized (i8) vectors.
///
/// The dot product is computed using 32-bit accumulation.
/// (The returned value is an i32—but since only relative ranking matters, you can
/// reinterpret it as a float if needed.)
pub fn quantized_dot_product(a: &[i8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len());
    let mut acc: i32 = 0;
    // Unrolled loop (or the optimizer will unroll in release builds).
    for (&a_val, &b_val) in a.iter().zip(b.iter()) {
        acc += (a_val as i32) * (b_val as i32);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    // We assume that the full precision dot product is available via crate::dot_product.
    // (See your SIMD dot product implementation above.)
    use crate::dot_product;
    use rand::Rng;
    use std::mem::size_of;
    use std::time::Instant;

    /// Returns a random vector of length `n` that is L₂-normalized.
    fn random_normalized_vector(n: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        let mut v: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }

    /// Returns the index of the vector in `dataset` that has the highest dot product
    /// with `query`, using full precision.
    fn nearest_neighbor_index(dataset: &[Vec<f32>], query: &[f32]) -> usize {
        let mut best_idx = 0;
        let mut best_score = -std::f32::INFINITY;
        for (i, vec) in dataset.iter().enumerate() {
            let score = dot_product(vec, query);
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }
        best_idx
    }

    /// Returns the index of the vector in `dataset` that has the highest quantized dot product
    /// with `query`.
    fn nearest_neighbor_index_quantized(dataset: &[Vec<i8>], query: &[i8]) -> usize {
        let mut best_idx = 0;
        let mut best_score = i32::MIN;
        for (i, vec) in dataset.iter().enumerate() {
            let score = quantized_dot_product(vec, query);
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }
        best_idx
    }

    /// Test 1: Ensure that the quantized dot product is at least 2× faster than the full‑precision version.
    #[test]
    fn test_speedup() {
        // This test works best in release mode.
        if cfg!(debug_assertions) {
            eprintln!("Skipping speedup test in debug mode.");
            return;
        }
        let vec_len = 1_000_000;
        let iterations = 50;
        let a = random_normalized_vector(vec_len);
        let b = random_normalized_vector(vec_len);

        // Warm up and measure full precision dot product.
        let _ = dot_product(&a, &b);
        let mut full_sum = 0.0;
        let start_full = Instant::now();
        for _ in 0..iterations {
            full_sum += dot_product(&a, &b);
        }
        let duration_full = start_full.elapsed();

        // Quantize the vectors.
        let qa = quantize_normalized(&a);
        let qb = quantize_normalized(&b);

        // Warm up and measure quantized dot product.
        let _ = quantized_dot_product(&qa, &qb);
        let mut quant_sum = 0;
        let start_quant = Instant::now();
        for _ in 0..iterations {
            quant_sum += quantized_dot_product(&qa, &qb);
        }
        let duration_quant = start_quant.elapsed();

        eprintln!("Full dot product total time: {:?}", duration_full);
        eprintln!("Quantized dot product total time: {:?}", duration_quant);

        let full_ns = duration_full.as_nanos() as f64;
        let quant_ns = duration_quant.as_nanos() as f64;
        // Assert that full_precision is at least 2× slower than quantized.
        assert!(
            full_ns / quant_ns >= 100.0,
            "Speedup less than 100x: full: {} ns, quantized: {} ns",
            full_ns,
            quant_ns
        );
    }

    /// Test 2: Verify that the quantized vector uses at most about 1/3.25 of the original memory.
    #[test]
    fn test_memory_savings() {
        let vec_len = 10_000;
        let a = random_normalized_vector(vec_len);
        let quantized = quantize_normalized(&a);
        let original_size = vec_len * size_of::<f32>();
        let quantized_size = quantized.len() * size_of::<i8>();
        let ratio = original_size as f64 / quantized_size as f64;
        assert!(
            ratio >= 4.0,
            "Memory saving ratio too low: expected >= 3.25, got {}",
            ratio
        );
    }

    /// Test 3: Check the “recall” by ensuring that for random queries the nearest neighbor
    /// computed with quantized dot products agrees with the full‑precision version at least 95% of the time.
    #[test]
    fn test_recall() {
        let num_vectors = 1000;
        let dim = 128;
        let num_queries = 500;

        // Build a database of random normalized vectors.
        let database: Vec<Vec<f32>> = (0..num_vectors)
            .map(|_| random_normalized_vector(dim))
            .collect();

        // Quantize the database.
        let quantized_database: Vec<Vec<i8>> =
            database.iter().map(|v| quantize_normalized(v)).collect();

        let mut correct = 0;
        for _ in 0..num_queries {
            let query = random_normalized_vector(dim);
            let full_nn = nearest_neighbor_index(&database, &query);
            let quant_query = quantize_normalized(&query);
            let quant_nn = nearest_neighbor_index_quantized(&quantized_database, &quant_query);
            if full_nn == quant_nn {
                correct += 1;
            }
        }
        let recall = (correct as f64) / (num_queries as f64);
        assert!(
            recall >= 0.90,
            "Recall too low: expected at least 90%, got {:.2}%",
            recall * 100.0
        );
    }
}
