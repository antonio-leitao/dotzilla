// Shift slices in place and add 8 elements at a time.
pub fn inner_product(x: &[f32], y: &[f32]) -> f32 {
    let n = std::cmp::min(x.len(), y.len());
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

pub fn euclidean(x: &[f32], y: &[f32]) -> f32 {
    //euclidean squared
    let n = std::cmp::min(x.len(), y.len());
    let (mut x, mut y) = (&x[..n], &y[..n]);

    let mut sum = 0.0;
    while x.len() >= 8 {
        sum += (x[0] - y[0]).powi(2)
            + (x[1] - y[1]).powi(2)
            + (x[2] - y[2]).powi(2)
            + (x[3] - y[3]).powi(2)
            + (x[4] - y[4]).powi(2)
            + (x[5] - y[5]).powi(2)
            + (x[6] - y[6]).powi(2)
            + (x[7] - y[7]).powi(2);
        x = &x[8..];
        y = &y[8..];
    }
    // Take care of any left over elements (if len is not divisible by 8).
    sum += x
        .iter()
        .zip(y.iter())
        .fold(0.0, |acc, (&ex, &ey)| acc + (ex - ey).powi(2));
    sum
}
