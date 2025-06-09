// benches/benchmark.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dotzilla::quantized::{quantize_normalized, quantized_dot_product};
use dotzilla::{dot_product, euclidean_distance};

fn bench_dot_product(c: &mut Criterion) {
    let len = 1024 * 1024;
    // f64 benchmark
    let a_f64: Vec<f64> = (0..len).map(|x| x as f64).collect();
    let b_f64: Vec<f64> = (0..len).map(|x| (x as f64).sin()).collect();
    // f32 benchmark
    let a_f32: Vec<f32> = (0..len).map(|x| x as f32).collect();
    let b_f32: Vec<f32> = (0..len).map(|x| (x as f32).sin()).collect();
    // Quantized
    let a_q = quantize_normalized(&a_f32);
    let b_q = quantize_normalized(&b_f32);

    c.bench_function("dot_product_f64", |bench| {
        bench.iter(|| dot_product(black_box(&a_f64), black_box(&b_f64)))
    });
    c.bench_function("dot_product_f32", |bench| {
        bench.iter(|| dot_product(black_box(&a_f32), black_box(&b_f32)))
    });
    c.bench_function("dot_product_quantized", |bench| {
        bench.iter(|| quantized_dot_product(black_box(&a_q), black_box(&b_q)))
    });
}

fn bench_euclidean_distance(c: &mut Criterion) {
    let len = 1024 * 1024;
    // f64 benchmark
    let a_f64: Vec<f64> = (0..len).map(|x| x as f64).collect();
    let b_f64: Vec<f64> = (0..len).map(|x| (x as f64).sin()).collect();
    // f32 benchmark
    let a_f32: Vec<f32> = (0..len).map(|x| x as f32).collect();
    let b_f32: Vec<f32> = (0..len).map(|x| (x as f32).sin()).collect();

    // Dedicated euclidean distance implementations
    c.bench_function("euclidean_distance_f64_dedicated", |bench| {
        bench.iter(|| euclidean_distance(black_box(&a_f64), black_box(&b_f64)))
    });

    c.bench_function("euclidean_distance_f32_dedicated", |bench| {
        bench.iter(|| euclidean_distance(black_box(&a_f32), black_box(&b_f32)))
    });

    // Euclidean distance via dot products (f64)
    c.bench_function("euclidean_distance_f64_via_dots", |bench| {
        bench.iter(|| {
            let a = black_box(&a_f64);
            let b = black_box(&b_f64);
            let a_dot_a = dot_product(a, a);
            let b_dot_b = dot_product(b, b);
            let a_dot_b = dot_product(a, b);
            (a_dot_a + b_dot_b - 2.0 * a_dot_b).sqrt()
        })
    });

    // Euclidean distance via dot products (f32)
    c.bench_function("euclidean_distance_f32_via_dots", |bench| {
        bench.iter(|| {
            let a = black_box(&a_f32);
            let b = black_box(&b_f32);
            let a_dot_a = dot_product(a, a);
            let b_dot_b = dot_product(b, b);
            let a_dot_b = dot_product(a, b);
            (a_dot_a + b_dot_b - 2.0 * a_dot_b).sqrt()
        })
    });
}

fn bench_euclidean_distance_small(c: &mut Criterion) {
    // Test with smaller vectors to see if the pattern holds
    let len = 512;
    let a_f64: Vec<f64> = (0..len).map(|x| x as f64).collect();
    let b_f64: Vec<f64> = (0..len).map(|x| (x as f64).sin()).collect();
    let a_f32: Vec<f32> = (0..len).map(|x| x as f32).collect();
    let b_f32: Vec<f32> = (0..len).map(|x| (x as f32).sin()).collect();

    c.bench_function("euclidean_distance_f64_dedicated_small", |bench| {
        bench.iter(|| euclidean_distance(black_box(&a_f64), black_box(&b_f64)))
    });

    c.bench_function("euclidean_distance_f32_dedicated_small", |bench| {
        bench.iter(|| euclidean_distance(black_box(&a_f32), black_box(&b_f32)))
    });

    c.bench_function("euclidean_distance_f64_via_dots_small", |bench| {
        bench.iter(|| {
            let a = black_box(&a_f64);
            let b = black_box(&b_f64);
            let a_dot_a = dot_product(a, a);
            let b_dot_b = dot_product(b, b);
            let a_dot_b = dot_product(a, b);
            (a_dot_a + b_dot_b - 2.0 * a_dot_b).sqrt()
        })
    });

    c.bench_function("euclidean_distance_f32_via_dots_small", |bench| {
        bench.iter(|| {
            let a = black_box(&a_f32);
            let b = black_box(&b_f32);
            let a_dot_a = dot_product(a, a);
            let b_dot_b = dot_product(b, b);
            let a_dot_b = dot_product(a, b);
            (a_dot_a + b_dot_b - 2.0 * a_dot_b).sqrt()
        })
    });
}

criterion_group!(
    benches,
    bench_dot_product,
    bench_euclidean_distance,
    bench_euclidean_distance_small
);
criterion_main!(benches);
