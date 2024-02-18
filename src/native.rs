use crate::utils;
use crate::{DIMENSIONS, ITERATIONS};
use std::time::Duration;

//NATIVE
fn native_dot_product(v1: &[f64], v2: &[f64]) -> f64 {
    v1.iter().zip(v2.iter()).map(|(&x, &y)| x * y).sum()
}
fn native_dot_base(x: &[f64], y: &[f64]) {
    for _ in 0..ITERATIONS {
        let _result = native_dot_product(x, y);
    }
}
pub fn native_dot() -> Vec<Duration> {
    let mut times = Vec::new();
    for n in DIMENSIONS {
        let x = utils::generate_random_vector(n);
        let y = utils::generate_random_vector(n);
        times.push(utils::time(|| native_dot_base(&x, &y)));
    }
    times
}
//EUCLIDEAN
fn native_euc_distance(from: &[f64], to: &[f64]) -> f64 {
    from.iter()
        .zip(to.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}
fn native_euc_base(x: &[f64], y: &[f64]) {
    for _ in 0..ITERATIONS {
        let _result = native_euc_distance(x, y);
    }
}
pub fn native_euc() -> Vec<Duration> {
    let mut times = Vec::new();
    for n in DIMENSIONS {
        let x = utils::generate_random_vector(n);
        let y = utils::generate_random_vector(n);
        times.push(utils::time(|| native_euc_base(&x, &y)));
    }
    times
}
