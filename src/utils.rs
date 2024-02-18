use rand::Rng;
use std::time::{Duration, Instant};

pub fn generate_random_vector(n: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

pub fn time(mut f: impl FnMut()) -> Duration {
    let instant = Instant::now();
    f();
    instant.elapsed()
}
