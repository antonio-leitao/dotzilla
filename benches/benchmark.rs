// benches/benchmark.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dotzilla::dot_product;
use dotzilla::quantized::{quantize_normalized, quantized_dot_product};

// Update benchmark file to include f32 tests
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

criterion_group!(benches, bench_dot_product);
criterion_main!(benches);
