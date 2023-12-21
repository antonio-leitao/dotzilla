extern crate rblas;
// use anyhow::Result;
use faer::Mat;
use human_repr::HumanDuration;
use ndarray::Array1;
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use rand::Rng;
use rblas::Dot;
use std::time::{Duration, Instant};

const ITERATIONS: u32 = 1_000_000;
const DIMENSIONS: [usize; 9] = [8, 16, 32, 64, 128, 256, 512, 1024, 2048];
// Function to generate a random vector in n dimensions
fn generate_random_vector(n: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

fn time(mut f: impl FnMut()) -> Duration {
    let instant = Instant::now();
    f();
    instant.elapsed()
}
//NATIVE
fn native_dot_product(v1: &[f64], v2: &[f64]) -> f64 {
    v1.iter().zip(v2.iter()).map(|(&x, &y)| x * y).sum()
}
fn base_native(x: &[f64], y: &[f64]) {
    for _ in 0..ITERATIONS {
        let _result = native_dot_product(x, y);
    }
}
fn native_vec_vec() -> Vec<Duration> {
    let mut times = Vec::new();
    for n in DIMENSIONS {
        let x = generate_random_vector(n);
        let y = generate_random_vector(n);
        times.push(time(|| base_native(&x, &y)));
    }
    times
}

// RBLAS
fn rblas_base(x: &[f64], y: &[f64]) {
    for _ in 0..ITERATIONS {
        let _result = Dot::dot(x, y);
    }
}
fn rblas_vec_vec() -> Vec<Duration> {
    let mut times = Vec::new();
    for n in DIMENSIONS {
        let x = generate_random_vector(n);
        let y = generate_random_vector(n);
        times.push(time(|| rblas_base(&x, &y)));
    }
    times
}

// NDARRAY
fn ndarray_base(x: &Array1<f64>, y: &Array1<f64>) {
    for _ in 0..ITERATIONS {
        let _result = x.dot(y);
    }
}
fn ndarray_vec_vec() -> Vec<Duration> {
    let mut times = Vec::new();
    for n in DIMENSIONS {
        let x = Array1::random(n, Uniform::<f64>::new(0., 1.));
        let y = Array1::random(n, Uniform::<f64>::new(0., 1.));
        times.push(time(|| ndarray_base(&x, &y)));
    }
    times
}

//FAER
fn faer_base(x: &Mat<f64>, y: &Mat<f64>) {
    for _ in 0..ITERATIONS {
        let _result = x * y.transpose();
    }
}

fn faer_vec_vec() -> Vec<Duration> {
    let mut times = Vec::new();
    for n in DIMENSIONS {
        let x = Mat::<f64>::from_fn(1, n, |_, _| rand::random());
        let y = Mat::<f64>::from_fn(1, n, |_, _| rand::random());
        times.push(time(|| faer_base(&x, &y)));
    }
    times
}

fn print_results(
    input_sizes: &[usize],
    native: &[Duration],
    rblas: &[Duration],
    ndarray: &[Duration],
    faer: &[Duration],
) {
    println!(
        "{:>5} {:>10} {:>10} {:>10} {:>10}",
        "n", "native", "rblas", "ndarray", "faer"
    );
    for (i, n) in input_sizes.iter().copied().enumerate() {
        println!(
            "{:5} {:>10} {:>10} {:>10} {:>10}",
            n,
            fmt(native[i]),
            fmt(rblas[i]),
            fmt(ndarray[i]),
            fmt(faer[i]),
        );
    }
}

fn fmt(duration: Duration) -> String {
    duration.human_duration().to_string()
}

fn main() {
    let native = native_vec_vec();
    let rblas = rblas_vec_vec();
    let ndarray = ndarray_vec_vec();
    let faer = faer_vec_vec();
    print_results(&DIMENSIONS, &native, &rblas, &ndarray, &faer);
}
