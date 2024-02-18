use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::prelude::*;
use std::cmp;

const ITERATIONS: usize = 1_000_000;

fn generate_random_vector(size: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen()).collect()
}

fn dotzilla(x: &[f64], y: &[f64]) -> f64 {
    let n = cmp::min(x.len(), y.len());
    let (mut x, mut y) = (&x[..n], &y[..n]);

    let mut sum = 0.0;
    while x.len() >= 8 {
        sum += x[0] * y[0]
            + x[1] * y[1]
            + x[2] * y[2]
            + x[3] * y[3]
            + x[4] * y[4]
            + x[5] * y[5]
            + x[6] * y[6]
            + x[7] * y[7];
        x = &x[8..];
        y = &y[8..];
    }

    // Take care of any left over elements (if len is not divisible by 8).
    x.iter()
        .zip(y.iter())
        .fold(sum, |sum, (&ex, &ey)| sum + (ex * ey))
}

fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");

    for size in [16, 32, 64, 128, 256, 512, 1024, 2048].iter() {
        let x = generate_random_vector(*size);
        let y = generate_random_vector(*size);
        group.bench_function(format!("size_{}", size), |b| {
            b.iter(|| {
                for _ in 0..ITERATIONS {
                    black_box(dotzilla(&x, &y));
                }
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_dot_product);
criterion_main!(benches);
