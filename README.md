# Dot Product Benchmark

Do one thing, do it well. Performance comparison for a one-threaded non-parallel dot product of vectors and matrices in Rust.

### Libraries
- `native`
- [`rblas`](https://mikkyang.github.io/rust-blas/doc/rblas/index.html)
- [`ndarray`](https://docs.rs/ndarray/latest/ndarray/)
- [`faer`](https://github.com/sarah-ek/faer-rs/tree/main)

All benchmarks where run on an `16Gb` M1 Pro Macbook using `openblas` with no parallelisation.

### Level 1: vector-vector
Time it takes to do 1,000,000 (Million) vector multiplications of increasing size `n`
```text
    n     native      rblas    ndarray       faer
    8    178.3ms     23.6ms    176.5ms      2.61s
   16    284.7ms     24.5ms    210.1ms      3.58s
   32      534ms     27.2ms    134.3ms      5.46s
   64      1.02s     33.3ms    139.3ms      9.15s
  128      2.02s     42.8ms    148.1ms     17.58s
  256      3.95s     64.8ms    171.5ms     31.86s
  512      7.91s    103.4ms      209ms     1:01.5
 1024     15.54s    180.6ms    287.3ms     1:59.8
 2048     31.48s    339.7ms    449.6ms     3:57.7
```
