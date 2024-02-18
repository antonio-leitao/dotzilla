use crate::utils;
use crate::{DIMENSIONS, ITERATIONS};
use rblas::Dot;
use std::time::Duration;
// RBLAS
fn rblas_dot_base(x: &[f64], y: &[f64]) {
    for _ in 0..ITERATIONS {
        let _result = Dot::dot(x, y);
    }
}
pub fn rblas_dot() -> Vec<Duration> {
    let mut times = Vec::new();
    for n in DIMENSIONS {
        let x = utils::generate_random_vector(n);
        let y = utils::generate_random_vector(n);
        times.push(utils::time(|| rblas_dot_base(&x, &y)));
    }
    times
}

//assuming they are normed
fn rblas_euc_distance(x: &[f64], y: &[f64]) -> f64 {
    2.0 - 2.0 * Dot::dot(x, y)
}
// RBLAS
fn rblas_euc_base(x: &[f64], y: &[f64]) {
    for _ in 0..ITERATIONS {
        let _result = rblas_euc_distance(x, y);
    }
}
pub fn rblas_euc() -> Vec<Duration> {
    let mut times = Vec::new();
    for n in DIMENSIONS {
        let x = utils::generate_random_vector(n);
        let y = utils::generate_random_vector(n);
        times.push(utils::time(|| rblas_euc_base(&x, &y)));
    }
    times
}
