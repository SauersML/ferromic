use std::{env, time::Duration};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ferromic::pca::{bench_efficient_exact_pca, bench_fast_exact_pca};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const PLOIDY: usize = 2;
const COMPONENTS: usize = 8;

fn generate_dataset(variants: usize, samples: usize, seed: u64) -> Vec<i16> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = vec![0i16; variants * samples * PLOIDY];

    for variant_idx in 0..variants {
        for sample_idx in 0..samples {
            let base = (variant_idx * samples + sample_idx) * PLOIDY;
            data[base] = rng.gen_range(0..=1);
            data[base + 1] = rng.gen_range(0..=1);
        }
    }

    data
}

fn genotype_matrix(data: &[i16], variants: usize, samples: usize) -> (Vec<f64>, usize, usize) {
    let rows = samples * PLOIDY;
    let cols = variants;
    let mut matrix = vec![0.0f64; rows * cols];
    for variant_idx in 0..variants {
        for sample_idx in 0..samples {
            let base = (variant_idx * samples + sample_idx) * PLOIDY;
            let row_left = sample_idx * PLOIDY;
            let left_idx = row_left * cols + variant_idx;
            matrix[left_idx] = data[base] as f64;
            matrix[left_idx + cols] = data[base + 1] as f64;
        }
    }
    (matrix, rows, cols)
}

fn bench_exact_pca_dense(c: &mut Criterion) {
    let mut group = c.benchmark_group("exact_pca_dense");
    env::set_var("FERROMIC_PROGRESS", "0");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(8));
    group.warm_up_time(Duration::from_secs(2));
    let sizes = [
        (200usize, 64usize),
        (800usize, 128usize),
        (2000usize, 256usize),
    ];

    for (variants, samples) in sizes {
        let seed = 0x5A5A_0000 ^ ((variants as u64) << 16) ^ samples as u64;
        let data = generate_dataset(variants, samples, seed);

        group.throughput(Throughput::Elements((variants * samples) as u64));
        group.bench_with_input(
            BenchmarkId::new("fast", format!("v{}_s{}", variants, samples)),
            &data,
            |b, data| {
                b.iter(|| {
                    let (matrix, rows, cols) = genotype_matrix(data, variants, samples);
                    let _result = bench_fast_exact_pca(matrix, rows, cols, COMPONENTS)
                        .expect("fast exact PCA should succeed");
                    criterion::black_box(&_result);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("fallback", format!("v{}_s{}", variants, samples)),
            &data,
            |b, data| {
                b.iter(|| {
                    let (matrix, rows, cols) = genotype_matrix(data, variants, samples);
                    let _result = bench_efficient_exact_pca(matrix, rows, cols, COMPONENTS)
                        .expect("fallback exact PCA should succeed");
                    criterion::black_box(&_result);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_exact_pca_dense);
criterion_main!(benches);
