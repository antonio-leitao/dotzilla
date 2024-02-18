# Dot Product Benchmark

Do one thing, do it well. Performance comparison for a one-threaded non-parallel dot product of vectors and matrices in Rust.

### Libraries
- `native`
- [`rblas`](https://mikkyang.github.io/rust-blas/doc/rblas/index.html)
- [`ndarray`](https://docs.rs/ndarray/latest/ndarray/)

All benchmarks where run on an `16Gb` M1 Pro Macbook using `openblas` with no parallelisation.

### Dot Product
Time it takes to do 1,000,000 (Million) vector multiplications of increasing size `n`
```text
DOT PRODUCTS:
    n     native      rblas    ndarray
    8    176.9ms     23.9ms    178.7ms
   16      294ms     24.9ms    213.8ms
   32    625.7ms     27.4ms    136.4ms
   64      1.05s     32.3ms    141.7ms
  128      2.02s     42.5ms      151ms
  256      4.01s     62.7ms    171.1ms
  512      7.92s      103ms      211ms
 1024     15.72s    184.5ms    291.6ms
 2048     31.33s    344.5ms    455.9ms
```

### Euclidean Distance
```text
EUCLIDEAN DISTANCE:
    n     native      rblas    ndarray
    8    217.9ms     49.6ms      2.44s
   16    382.5ms     53.3ms       2.8s
   32    709.4ms     65.5ms      3.45s
   64      1.36s     87.4ms      4.74s
  128      2.66s    129.6ms      7.25s
  256      5.35s    220.5ms     12.32s
  512     10.36s    402.9ms     22.49s
 1024     20.63s    797.2ms     42.72s
 2048     41.19s       1.6s     1:25.4
```
