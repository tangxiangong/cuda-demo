use rayon::prelude::*;

pub fn matmul(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0; n * n];
    c.par_chunks_mut(n).enumerate().for_each(|(i, row_slice)| {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[i * n + k] * b[k * n + j];
            }
            row_slice[j] = sum;
        }
    });
    c
}
