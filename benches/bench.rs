use criterion::{Criterion, criterion_group, criterion_main};
use cuda_demo::{cuda, par, seq};
use diffusionx::random::normal;
use std::hint::black_box;

fn criterion_benchmark(c: &mut Criterion) {
    let n = 512;
    let len = n * n;
    let a = normal::standard_rands::<f32>(len);
    let b = normal::standard_rands::<f32>(len);

    c.bench_function("seq matmul", |bencher| {
        bencher.iter(|| {
            let _ = seq::matmul(black_box(&a), black_box(&b), n);
        })
    });

    c.bench_function("par matmul", |bencher| {
        bencher.iter(|| {
            let _ = par::matmul(black_box(&a), black_box(&b), n);
        })
    });

    c.bench_function("cuda matmul", |bencher| {
        bencher.iter(|| {
            let _ = cuda::matmul(black_box(&a), black_box(&b), n).unwrap();
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
