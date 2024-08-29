use criterion::{criterion_group, criterion_main, Criterion};
use dotzilla::{dot, l2sq};

fn benchmark_inner_product(c: &mut Criterion) {
    // Example data
    let a = vec![1.0f32; 1024];
    let b = vec![2.0f32; 1024];

    c.bench_function("inner_product", |bencher| bencher.iter(|| dot(&a, &b)));
    c.bench_function("baseline_inner_product", |bencher| {
        bencher.iter(|| dotzilla::dot_std::inner_product(&a, &b))
    });
}

fn benchmark_euclidean(c: &mut Criterion) {
    // Example data
    let a = vec![1.0f32; 1024];
    let b = vec![2.0f32; 1024];

    c.bench_function("euclidean", |bencher| bencher.iter(|| l2sq(&a, &b)));
    c.bench_function("baseline_euclidean", |bencher| {
        bencher.iter(|| dotzilla::dot_std::euclidean(&a, &b))
    });
}

criterion_group!(benches, benchmark_inner_product, benchmark_euclidean,);
criterion_main!(benches);
