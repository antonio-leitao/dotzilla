# Dotzilla

Efficient Rust implementation of basic linear algebra routines.

### Installation

Run the following command in your project directory:

```bash
cargo add dotzilla
```

Or add the following line to your Cargo.toml:

```toml
[dependencies]
dotzilla = "0.1.0"
```

### Usage

```Rust
use dotzilla::{dot, l2sq};

fn main() {
    // Example data
    let a = vec![1.0f32; 1024];
    let b = vec![2.0f32; 1024];
    let inner_produce = dot(&a, &b);
    let euclidean_distance_squared = l2sq(&a, &b);
}
```

### Roadmap:
- [x] `dot` inner product implementation for `&[f32]`.
- [x] `l2sq` square euclidean distance for `&[f32]` 
- [ ] `dot_f64` inner product implementation for`&[f64]`.
- [ ] `l2sq_f64` square euclidean distance for `&[f64]` 
