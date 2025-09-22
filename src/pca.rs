use efficient_pca::PCA;
use ndarray::s;
use ndarray::Array2;
use numpy::ndarray::ArrayView3;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::process::VcfError;
use crate::progress::{
    create_spinner, display_status_box, log, set_stage, LogLevel, ProcessingStage, StatusBox,
};
use crate::Variant;

/// Structure to hold PCA results per chromosome
pub struct PcaResult {
    pub haplotype_labels: Vec<String>,
    pub pca_coordinates: Array2<f64>,
    pub positions: Vec<i64>,
}

/// Computes PCA for a single chromosome keeping haplotypes separate
///
/// # Arguments
/// * `variants` - Slice of variants for a single chromosome
/// * `sample_names` - Names of the samples
/// * `n_components` - Number of principal components to compute
///
/// # Returns
/// PCA results containing haplotype labels and their coordinates in PC space
pub fn compute_chromosome_pca(
    variants: &[Variant],
    sample_names: &[String],
    n_components: usize,
) -> Result<PcaResult, VcfError> {
    set_stage(ProcessingStage::PcaAnalysis);
    let spinner = create_spinner(&format!(
        "Preparing PCA matrix for {} variants",
        variants.len()
    ));

    if variants.is_empty() {
        return Err(VcfError::Parse("No variants provided for PCA".to_string()));
    }

    let sample_count = sample_names.len();
    let n_haplotypes = sample_count * 2;

    let mut complete_count = 0usize;
    let mut maf_filtered_indices = Vec::with_capacity(variants.len());
    let mut filtered_positions = Vec::with_capacity(variants.len());

    for (variant_idx, variant) in variants.iter().enumerate() {
        if variant.genotypes.len() != sample_count {
            spinner.finish_and_clear();
            return Err(VcfError::Parse(format!(
                "variant {} contains {} samples but {} names were provided",
                variant_idx,
                variant.genotypes.len(),
                sample_count
            )));
        }

        let mut allele_sum = 0usize;
        let mut complete = true;
        for genotype in &variant.genotypes {
            let Some(alleles) = genotype else {
                complete = false;
                break;
            };
            if alleles.len() < 2 {
                complete = false;
                break;
            }
            allele_sum += (alleles[0] + alleles[1]) as usize;
        }

        if !complete {
            continue;
        }

        complete_count += 1;

        let allele_freq = allele_sum as f64 / n_haplotypes as f64;
        let maf = allele_freq.min(1.0 - allele_freq);
        if maf >= 0.05 {
            maf_filtered_indices.push(variant_idx);
            filtered_positions.push(variant.position);
        }
    }

    log(
        LogLevel::Info,
        &format!(
            "Found {} variants with complete data out of {} total variants",
            complete_count,
            variants.len()
        ),
    );

    let max_components = std::cmp::min(complete_count, n_haplotypes);
    let n_components = std::cmp::min(n_components, max_components);

    display_status_box(StatusBox {
        title: "Chromosome PCA Data Preparation".to_string(),
        stats: vec![
            ("Total variants".to_string(), variants.len().to_string()),
            (
                "Variants with complete data".to_string(),
                complete_count.to_string(),
            ),
            ("Samples".to_string(), sample_count.to_string()),
            (
                "Haplotypes (2 per sample)".to_string(),
                n_haplotypes.to_string(),
            ),
            ("Requested PCs".to_string(), n_components.to_string()),
        ],
    });

    if maf_filtered_indices.is_empty() {
        spinner.finish_and_clear();
        return Err(VcfError::Parse(
            "No variants with MAF >= 5% found for PCA".to_string(),
        ));
    }

    log(
        LogLevel::Info,
        &format!(
            "Keeping {}/{} variants with MAF >= 5% for PCA",
            maf_filtered_indices.len(),
            complete_count
        ),
    );

    let mut data_matrix = Array2::<f64>::zeros((n_haplotypes, maf_filtered_indices.len()));
    if let Some(matrix_slice) = data_matrix.as_slice_mut() {
        let cols = maf_filtered_indices.len();
        for (column_idx, &variant_idx) in maf_filtered_indices.iter().enumerate() {
            let variant = &variants[variant_idx];
            for (sample_idx, genotype) in variant.genotypes.iter().enumerate() {
                let alleles = genotype
                    .as_ref()
                    .expect("filtered variants lack missing data");
                let left_row = sample_idx * 2;
                let right_row = left_row + 1;
                matrix_slice[left_row * cols + column_idx] = alleles[0] as f64;
                matrix_slice[right_row * cols + column_idx] = alleles[1] as f64;
            }
        }
    } else {
        for (column_idx, &variant_idx) in maf_filtered_indices.iter().enumerate() {
            let variant = &variants[variant_idx];
            for (sample_idx, genotype) in variant.genotypes.iter().enumerate() {
                let alleles = genotype
                    .as_ref()
                    .expect("filtered variants lack missing data");
                data_matrix[[sample_idx * 2, column_idx]] = alleles[0] as f64;
                data_matrix[[sample_idx * 2 + 1, column_idx]] = alleles[1] as f64;
            }
        }
    }

    spinner.finish_and_clear();

    run_pca_analysis(data_matrix, sample_names, n_components, filtered_positions)
}

pub fn compute_chromosome_pca_from_dense(
    genotypes: ArrayView3<'_, i16>,
    positions: &[i64],
    sample_names: &[String],
    n_components: usize,
) -> Result<PcaResult, VcfError> {
    set_stage(ProcessingStage::PcaAnalysis);

    let shape = genotypes.shape();
    let variant_count = shape[0];
    let sample_count = shape[1];
    let ploidy = shape[2];

    if sample_count != sample_names.len() {
        return Err(VcfError::Parse(format!(
            "genotype sample dimension {} does not match sample_names length {}",
            sample_count,
            sample_names.len()
        )));
    }
    if sample_count == 0 {
        return Err(VcfError::Parse(
            "dense PCA requires at least one sample".to_string(),
        ));
    }
    if ploidy != 2 {
        return Err(VcfError::Parse(format!(
            "expected diploid genotypes (ploidy=2) but received ploidy {}",
            ploidy
        )));
    }
    if positions.len() != variant_count {
        return Err(VcfError::Parse(format!(
            "positions length {} does not match variant dimension {}",
            positions.len(),
            variant_count
        )));
    }

    let spinner = create_spinner(&format!(
        "Preparing PCA matrix for {} dense variants",
        variant_count
    ));

    let n_haplotypes = sample_count * 2;
    let mut complete_count = 0usize;
    let mut maf_filtered_indices = Vec::with_capacity(variant_count);
    let mut filtered_positions = Vec::with_capacity(variant_count);
    let storage = genotypes.as_slice_memory_order();
    let stride = sample_count * ploidy;

    if let Some(values) = storage {
        for (variant_idx, chunk) in values.chunks_exact(stride).enumerate() {
            let mut allele_sum = 0usize;
            let mut complete = true;
            for &value in chunk {
                if value < 0 || value > 1 {
                    complete = false;
                    break;
                }
                allele_sum += value as usize;
            }
            if !complete {
                continue;
            }

            complete_count += 1;
            let allele_freq = allele_sum as f64 / n_haplotypes as f64;
            let maf = allele_freq.min(1.0 - allele_freq);
            if maf >= 0.05 {
                maf_filtered_indices.push(variant_idx);
                filtered_positions.push(positions[variant_idx]);
            }
        }
    } else {
        for variant_idx in 0..variant_count {
            let mut allele_sum = 0usize;
            let mut complete = true;
            for sample_idx in 0..sample_count {
                let left = genotypes[(variant_idx, sample_idx, 0)];
                let right = genotypes[(variant_idx, sample_idx, 1)];
                if left < 0 || right < 0 || left > 1 || right > 1 {
                    complete = false;
                    break;
                }
                allele_sum += (left + right) as usize;
            }
            if !complete {
                continue;
            }

            complete_count += 1;
            let allele_freq = allele_sum as f64 / n_haplotypes as f64;
            let maf = allele_freq.min(1.0 - allele_freq);
            if maf >= 0.05 {
                maf_filtered_indices.push(variant_idx);
                filtered_positions.push(positions[variant_idx]);
            }
        }
    }

    log(
        LogLevel::Info,
        &format!(
            "Dense PCA input: {} variants with complete data out of {}",
            complete_count, variant_count
        ),
    );

    let max_components = std::cmp::min(complete_count, n_haplotypes);
    let n_components = std::cmp::min(n_components, max_components);

    display_status_box(StatusBox {
        title: "Chromosome PCA Data Preparation".to_string(),
        stats: vec![
            ("Total variants".to_string(), variant_count.to_string()),
            (
                "Variants with complete data".to_string(),
                complete_count.to_string(),
            ),
            ("Samples".to_string(), sample_count.to_string()),
            (
                "Haplotypes (2 per sample)".to_string(),
                n_haplotypes.to_string(),
            ),
            ("Requested PCs".to_string(), n_components.to_string()),
        ],
    });

    if maf_filtered_indices.is_empty() {
        spinner.finish_and_clear();
        return Err(VcfError::Parse(
            "No variants with MAF >= 5% found for PCA".to_string(),
        ));
    }

    log(
        LogLevel::Info,
        &format!(
            "Keeping {}/{} dense variants with MAF >= 5%",
            maf_filtered_indices.len(),
            complete_count
        ),
    );

    let mut data_matrix = Array2::<f64>::zeros((n_haplotypes, maf_filtered_indices.len()));
    if let Some(matrix_slice) = data_matrix.as_slice_mut() {
        let cols = maf_filtered_indices.len();
        if let Some(values) = storage {
            let src_ptr = values.as_ptr();
            unsafe {
                for (column_idx, &variant_idx) in maf_filtered_indices.iter().enumerate() {
                    let variant_ptr = src_ptr.add(variant_idx * stride);
                    for sample_idx in 0..sample_count {
                        let row_left = sample_idx * 2;
                        let dest_idx = row_left * cols + column_idx;
                        let allele_left = *variant_ptr.add(sample_idx * ploidy) as f64;
                        let allele_right = *variant_ptr.add(sample_idx * ploidy + 1) as f64;
                        *matrix_slice.get_unchecked_mut(dest_idx) = allele_left;
                        *matrix_slice.get_unchecked_mut(dest_idx + cols) = allele_right;
                    }
                }
            }
        } else {
            for (column_idx, &variant_idx) in maf_filtered_indices.iter().enumerate() {
                for sample_idx in 0..sample_count {
                    let left = genotypes[(variant_idx, sample_idx, 0)] as f64;
                    let right = genotypes[(variant_idx, sample_idx, 1)] as f64;
                    let left_row = sample_idx * 2;
                    let right_row = left_row + 1;
                    matrix_slice[left_row * cols + column_idx] = left;
                    matrix_slice[right_row * cols + column_idx] = right;
                }
            }
        }
    } else {
        for (column_idx, &variant_idx) in maf_filtered_indices.iter().enumerate() {
            for sample_idx in 0..sample_count {
                let left = genotypes[(variant_idx, sample_idx, 0)] as f64;
                let right = genotypes[(variant_idx, sample_idx, 1)] as f64;
                data_matrix[[sample_idx * 2, column_idx]] = left;
                data_matrix[[sample_idx * 2 + 1, column_idx]] = right;
            }
        }
    }

    spinner.finish_and_clear();

    run_pca_analysis(data_matrix, sample_names, n_components, filtered_positions)
}

fn run_pca_analysis(
    data_matrix: Array2<f64>,
    sample_names: &[String],
    n_components: usize,
    positions: Vec<i64>,
) -> Result<PcaResult, VcfError> {
    if data_matrix.ncols() == 0 {
        return Err(VcfError::Parse(
            "No informative variants available for PCA".to_string(),
        ));
    }

    let spinner = create_spinner("Computing PCA");
    let mut pca = PCA::new();
    // Run an exact PCA first to maximise numerical agreement with scikit-allel.
    // Randomised SVD is used only as a fallback when the exact solve fails.
    let transformed = match pca.fit(data_matrix.clone(), None) {
        Ok(()) => {
            let fallback_matrix = data_matrix.clone();
            match pca.transform(data_matrix) {
                Ok(t) => {
                    drop(fallback_matrix);
                    t
                }
                Err(exact_transform_error) => {
                    log(
                        LogLevel::Warning,
                        "Exact PCA transform failed; retrying with randomized solver",
                    );
                    let mut randomized_pca = PCA::new();
                    match randomized_pca.rfit(fallback_matrix, n_components, 4, Some(42), None) {
                        Ok(t) => t,
                        Err(randomized_error) => {
                            spinner.finish_and_clear();
                            return Err(VcfError::Parse(format!(
                                "PCA computation failed (exact transform: {}; randomized: {})",
                                exact_transform_error, randomized_error
                            )));
                        }
                    }
                }
            }
        }
        Err(exact_fit_error) => {
            log(
                LogLevel::Warning,
                "Exact PCA fit failed; retrying with randomized solver",
            );
            let mut randomized_pca = PCA::new();
            match randomized_pca.rfit(data_matrix, n_components, 4, Some(42), None) {
                Ok(t) => t,
                Err(randomized_error) => {
                    spinner.finish_and_clear();
                    return Err(VcfError::Parse(format!(
                        "PCA computation failed (exact fit: {}; randomized: {})",
                        exact_fit_error, randomized_error
                    )));
                }
            }
        }
    };

    let available_components = transformed.ncols();
    let kept_components = std::cmp::min(n_components, available_components);
    let transformed = transformed.slice(s![.., 0..kept_components]).to_owned();

    spinner.finish_and_clear();

    let mut haplotype_labels = Vec::with_capacity(sample_names.len() * 2);
    for sample_name in sample_names {
        haplotype_labels.push(format!("{}_L", sample_name));
        haplotype_labels.push(format!("{}_R", sample_name));
    }

    log(
        LogLevel::Info,
        &format!(
            "PCA computation complete: generated {} components for {} haplotypes",
            kept_components,
            haplotype_labels.len()
        ),
    );

    Ok(PcaResult {
        haplotype_labels,
        pca_coordinates: transformed,
        positions,
    })
}

/// Writes PCA results for a single chromosome to a TSV file
pub fn write_chromosome_pca_to_file(
    result: &PcaResult,
    chromosome: &str,
    output_dir: &Path,
) -> Result<(), VcfError> {
    let file_name = format!("pca_chr_{}.tsv", chromosome);
    let output_file = output_dir.join(file_name);

    let spinner = create_spinner(&format!("Writing PCA results to {}", output_file.display()));

    let file = File::create(&output_file).map_err(|e| VcfError::Io(e))?;
    let mut writer = BufWriter::new(file);

    // Write header
    write!(writer, "Haplotype").map_err(|e| VcfError::Io(e))?;
    for i in 0..result.pca_coordinates.shape()[1] {
        write!(writer, "\tPC{}", i + 1).map_err(|e| VcfError::Io(e))?;
    }
    writeln!(writer).map_err(|e| VcfError::Io(e))?;

    // Write rows - ensure haplotype count matches coordinates
    let actual_rows = std::cmp::min(
        result.haplotype_labels.len(),
        result.pca_coordinates.shape()[0],
    );

    for idx in 0..actual_rows {
        write!(writer, "{}", result.haplotype_labels[idx]).map_err(|e| VcfError::Io(e))?;

        for j in 0..result.pca_coordinates.shape()[1] {
            write!(writer, "\t{:.6}", result.pca_coordinates[[idx, j]])
                .map_err(|e| VcfError::Io(e))?;
        }
        writeln!(writer).map_err(|e| VcfError::Io(e))?;
    }

    spinner.finish_and_clear();
    log(
        LogLevel::Info,
        &format!(
            "PCA results for chromosome {} written to {}",
            chromosome,
            output_file.display()
        ),
    );

    Ok(())
}

/// Run PCA analysis on each chromosome separately
pub fn run_chromosome_pca_analysis(
    variants_by_chr: &HashMap<String, Vec<Variant>>,
    sample_names: &[String],
    output_dir: &Path,
    n_components: usize,
) -> Result<(), VcfError> {
    set_stage(ProcessingStage::PcaAnalysis);
    log(LogLevel::Info, "Starting per-chromosome PCA analysis");

    // Create output directory if it doesn't exist
    if !output_dir.exists() {
        std::fs::create_dir_all(output_dir).map_err(|e| VcfError::Io(e))?;
    }

    // Process each chromosome separately
    let total_chr = variants_by_chr.len();
    let mut processed = 0;
    let mut successful = 0;

    for (chr, variants) in variants_by_chr {
        processed += 1;
        log(
            LogLevel::Info,
            &format!(
                "Processing chromosome {} ({}/{}) with {} variants",
                chr,
                processed,
                total_chr,
                variants.len()
            ),
        );

        // Skip chromosomes with too few variants
        if variants.len() < 2 {
            log(
                LogLevel::Warning,
                &format!(
                    "Skipping chromosome {} - too few variants ({})",
                    chr,
                    variants.len()
                ),
            );
            continue;
        }

        // Compute PCA for this chromosome
        match compute_chromosome_pca(variants, sample_names, n_components) {
            Ok(result) => {
                // Write results to file
                if let Err(e) = write_chromosome_pca_to_file(&result, chr, output_dir) {
                    log(
                        LogLevel::Warning,
                        &format!("Failed to write PCA results for chromosome {}: {}", chr, e),
                    );
                } else {
                    successful += 1;
                }
            }
            Err(e) => {
                log(
                    LogLevel::Warning,
                    &format!("Failed to compute PCA for chromosome {}: {}", chr, e),
                );
                // Continue with other chromosomes
            }
        }
    }

    if successful == 0 {
        return Err(VcfError::Parse(
            "Failed to compute PCA for any chromosome".to_string(),
        ));
    }

    log(LogLevel::Info, &format!(
        "Chromosome-specific PCA analysis completed successfully. Processed {}/{} chromosomes. Results saved to {}",
        successful, total_chr, output_dir.display()
    ));

    Ok(())
}

/// Combine PCA results from multiple chromosomes into a single file
/// with haplotype information preserved
pub fn combine_chromosome_pca_results(
    results_dir: &Path,
    output_file: &Path,
) -> Result<(), VcfError> {
    let spinner = create_spinner("Combining PCA results from all chromosomes");

    // Find all PCA result files
    let mut result_files = vec![];
    match std::fs::read_dir(results_dir) {
        Ok(entries) => {
            for entry_result in entries {
                match entry_result {
                    Ok(entry) => {
                        let path = entry.path();
                        if path.is_file() && path.to_string_lossy().contains("pca_chr_") {
                            result_files.push(path);
                        }
                    }
                    Err(e) => {
                        log(
                            LogLevel::Warning,
                            &format!("Error reading directory entry: {}", e),
                        );
                    }
                }
            }
        }
        Err(e) => {
            return Err(VcfError::Io(e));
        }
    }

    if result_files.is_empty() {
        return Err(VcfError::Parse(
            "No chromosome PCA result files found".to_string(),
        ));
    }

    // Sort files by chromosome name for consistent ordering
    result_files.sort_by(|a, b| {
        let a_name = a.file_name().unwrap_or_default().to_string_lossy();
        let b_name = b.file_name().unwrap_or_default().to_string_lossy();
        a_name.cmp(&b_name)
    });

    // Read the first file to get haplotype names and component count
    let first_file = std::fs::read_to_string(&result_files[0]).map_err(|e| VcfError::Io(e))?;

    let mut lines = first_file.lines();
    let header = match lines.next() {
        Some(h) => h,
        None => return Err(VcfError::Parse("Empty PCA result file".to_string())),
    };

    let n_components = header.split('\t').count() - 1; // Subtract 1 for the 'Haplotype' column

    // Create a combined output file
    let file = File::create(output_file).map_err(|e| VcfError::Io(e))?;
    let mut writer = BufWriter::new(file);

    // Write header
    write!(writer, "Haplotype\tChromosome").map_err(|e| VcfError::Io(e))?;

    for i in 0..n_components {
        write!(writer, "\tPC{}", i + 1).map_err(|e| VcfError::Io(e))?;
    }

    writeln!(writer).map_err(|e| VcfError::Io(e))?;

    // Write results for each chromosome
    for file_path in &result_files {
        // Extract chromosome name from filename
        let chr_name = file_path.file_name().unwrap_or_default().to_string_lossy();

        let chr = chr_name
            .strip_prefix("pca_chr_")
            .and_then(|s| s.strip_suffix(".tsv"))
            .unwrap_or(&chr_name);

        // Read chromosome file
        let file_content = match std::fs::read_to_string(file_path) {
            Ok(content) => content,
            Err(e) => {
                log(
                    LogLevel::Warning,
                    &format!("Failed to read file {}: {}", file_path.display(), e),
                );
                continue;
            }
        };

        let mut lines = file_content.lines();
        let _header = lines.next(); // Skip header

        for line in lines {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() < 2 {
                continue;
            }

            let haplotype = parts[0];
            write!(writer, "{}\t{}", haplotype, chr).map_err(|e| VcfError::Io(e))?;

            for i in 1..parts.len() {
                write!(writer, "\t{}", parts[i]).map_err(|e| VcfError::Io(e))?;
            }

            writeln!(writer).map_err(|e| VcfError::Io(e))?;
        }
    }

    spinner.finish_and_clear();
    log(
        LogLevel::Info,
        &format!("Combined PCA results written to {}", output_file.display()),
    );

    Ok(())
}

/// Run memory-efficient PCA analysis on a per-chromosome basis
/// keeping haplotypes separate
pub fn run_global_pca_analysis(
    variants_by_chr: &HashMap<String, Vec<Variant>>,
    sample_names: &[String],
    output_dir: &Path,
    n_components: usize,
) -> Result<(), VcfError> {
    // Create directory for chromosome-specific results
    let chr_results_dir = output_dir.join("chr_pca");
    if !chr_results_dir.exists() {
        std::fs::create_dir_all(&chr_results_dir).map_err(|e| VcfError::Io(e))?;
    }

    // Run PCA for each chromosome separately
    run_chromosome_pca_analysis(
        variants_by_chr,
        sample_names,
        &chr_results_dir,
        n_components,
    )?;

    // Combine results into a single file
    let combined_output = output_dir.join("combined_chromosome_pca.tsv");
    combine_chromosome_pca_results(&chr_results_dir, &combined_output)?;

    log(
        LogLevel::Info,
        &format!(
        "Memory-efficient per-chromosome PCA analysis completed successfully. Results saved to {}",
        output_dir.display()
    ),
    );

    Ok(())
}
