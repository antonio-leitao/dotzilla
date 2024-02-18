use crate::utils;
use crate::{DIMENSIONS, ITERATIONS};
use ndarray::Array1;
use ndarray_linalg::Norm;
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use std::time::Duration;
// DOT
fn ndarray_dot_base(x: &Array1<f64>, y: &Array1<f64>) {
    for _ in 0..ITERATIONS {
        let _result = x.dot(y);
    }
}
pub fn ndarray_dot() -> Vec<Duration> {
    let mut times = Vec::new();
    for n in DIMENSIONS {
        let x = Array1::random(n, Uniform::<f64>::new(0., 1.));
        let y = Array1::random(n, Uniform::<f64>::new(0., 1.));
        times.push(utils::time(|| ndarray_dot_base(&x, &y)));
    }
    times
}
//EUCLIDEAN DISTANCE
fn ndarray_euc_base(x: &Array1<f64>, y: &Array1<f64>) {
    for _ in 0..ITERATIONS {
        let diff = x - y;
        let _l2_norm = diff.norm_l2();
    }
}
pub fn ndarray_euc() -> Vec<Duration> {
    let mut times = Vec::new();
    for n in DIMENSIONS {
        let x = Array1::random(n, Uniform::<f64>::new(0., 1.));
        let y = Array1::random(n, Uniform::<f64>::new(0., 1.));
        times.push(utils::time(|| ndarray_euc_base(&x, &y)));
    }
    times
}
