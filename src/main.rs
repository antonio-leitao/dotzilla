extern crate rblas;
// use anyhow::Result;
use human_repr::HumanDuration;
use std::time::Duration;
mod blas;
mod dotzilla;
mod native;
mod ndarray;
mod utils;

const ITERATIONS: u32 = 1_000_000;
const DIMENSIONS: [usize; 9] = [8, 16, 32, 64, 128, 256, 512, 1024, 2048];
// Function to generate a random vector in n dimensions

fn print_results(
    input_sizes: &[usize],
    native: &[Duration],
    rblas: &[Duration],
    dotzilla: &[Duration],
    ndarray: &[Duration],
) {
    println!(
        "{:>5} {:>10} {:>10} {:>10} {:>10}",
        "n", "native", "rblas", "dotzilla", "ndarray",
    );
    for (i, n) in input_sizes.iter().copied().enumerate() {
        println!(
            "{:5} {:>10} {:>10} {:>10} {:>10}",
            n,
            fmt(native[i]),
            fmt(rblas[i]),
            fmt(dotzilla[i]),
            fmt(ndarray[i]),
        );
    }
}

fn fmt(duration: Duration) -> String {
    duration.human_duration().to_string()
}

fn main() {
    println!("DOT PRODUCTS:");
    let native = native::native_dot();
    let rblas = blas::rblas_dot();
    let dotzil = dotzilla::dotzilla_dot();
    let ndarray = ndarray::ndarray_dot();
    print_results(&DIMENSIONS, &native, &rblas, &dotzil, &ndarray);
    println!("EUCLIDEAN DISTANCE:");
    let native = native::native_euc();
    let rblas = blas::rblas_euc();
    let dotzil = dotzilla::dotzilla_euc();
    let ndarray = ndarray::ndarray_euc();
    print_results(&DIMENSIONS, &native, &rblas, &dotzil, &ndarray);
}
