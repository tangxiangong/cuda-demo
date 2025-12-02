use criterion::{Criterion, criterion_group, criterion_main};
use cuda_demo::bm::*;
use diffusionx::simulation::{continuous::Bm, prelude::*};
use std::hint::black_box;

const N: usize = 10_000;

fn criterion_benchmark(c: &mut Criterion) {
    let bm = Bm::default();
    c.bench_function("BM MEAN CPU f64", |bencher| {
        bencher.iter(|| {
            let _ = bm
                .mean(black_box(100.0), black_box(N), black_box(0.01))
                .unwrap();
        })
    });

    c.bench_function("BM MEAN CUDA f32", |bencher| {
        bencher.iter(|| {
            let _ = bm_mean(
                black_box(0.0),
                black_box(0.5),
                black_box(100.0),
                black_box(0.01),
                black_box(N),
            )
            .unwrap();
        })
    });

    c.bench_function("BM MSD CPU f64", |bencher| {
        bencher.iter(|| {
            let _ = bm
                .msd(black_box(100.0), black_box(N), black_box(0.01))
                .unwrap();
        })
    });

    c.bench_function("BM MSD CUDA f32", |bencher| {
        bencher.iter(|| {
            let _ = bm_msd(
                black_box(0.5),
                black_box(100.0),
                black_box(0.01),
                black_box(N),
            )
            .unwrap();
        })
    });

    c.bench_function("BM 2nd Central Moment CPU f64", |bencher| {
        bencher.iter(|| {
            let _ = bm
                .central_moment(
                    black_box(100.0),
                    black_box(2),
                    black_box(N),
                    black_box(0.01),
                )
                .unwrap();
        })
    });

    c.bench_function("BM 2nd Central Moment CUDA f32", |bencher| {
        bencher.iter(|| {
            let _ = bm_moment(
                black_box(0.0),
                black_box(0.5),
                black_box(2),
                black_box(true),
                black_box(100.0),
                black_box(0.01),
                black_box(N),
            )
            .unwrap();
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
