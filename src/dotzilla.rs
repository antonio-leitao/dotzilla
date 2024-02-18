use crate::utils;
use crate::{DIMENSIONS, ITERATIONS};
use std::cmp;
use std::time::Duration;

// Shift slices in place and add 8 elements at a time.
fn dotzilla_dot_product(x: &[f64], y: &[f64]) -> f64 {
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

fn dotzilla_dot_base(x: &[f64], y: &[f64]) {
    for _ in 0..ITERATIONS {
        let _result = dotzilla_dot_product(x, y);
    }
}

pub fn dotzilla_dot() -> Vec<Duration> {
    let mut times = Vec::new();
    for n in DIMENSIONS {
        let x = utils::generate_random_vector(n);
        let y = utils::generate_random_vector(n);
        times.push(utils::time(|| dotzilla_dot_base(&x, &y)));
    }
    times
}
//EUCLIDEAN
//assuming they are normed
fn dotzilla_euc_distance(x: &[f64], y: &[f64]) -> f64 {
    2.0 - 2.0 * dotzilla_dot_product(x, y)
}
// RBLAS
fn dotzilla_euc_base(x: &[f64], y: &[f64]) {
    for _ in 0..ITERATIONS {
        let _result = dotzilla_euc_distance(x, y);
    }
}
pub fn dotzilla_euc() -> Vec<Duration> {
    let mut times = Vec::new();
    for n in DIMENSIONS {
        let x = utils::generate_random_vector(n);
        let y = utils::generate_random_vector(n);
        times.push(utils::time(|| dotzilla_euc_base(&x, &y)));
    }
    times
}
