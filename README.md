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
dotzilla = "0.2.0"
```

### Usage

```Rust
/// Example
use dotzilla::dot_product;

let a = vec![1.0f32, 2.0, 3.0];
let b = vec![4.0f32, 5.0, 6.0];
assert_eq!(dot_product(&a, &b), 32.0);
```
