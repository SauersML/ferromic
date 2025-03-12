use ndarray::Array2;
use std::path::Path;
use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::fs::File;

use crate::Variant;
use crate::process::VcfError;
use crate::progress::{log, LogLevel, create_spinner, ProcessingStage, set_stage, display_status_box, StatusBox};

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
    let spinner = create_spinner(&format!("Computing PCA for {} variants", variants.len()));
    
    // Count valid variants without materializing them
    let valid_count = variants.iter()
        .filter(|v| v.genotypes.iter().all(|g| match g {
            None => false,
            Some(alleles) => alleles.len() >= 2
        }))
        .count();
    
    log(LogLevel::Info, &format!(
        "Found {} variants with complete data out of {} total variants", 
        valid_count, variants.len()
    ));
    
    if valid_count == 0 {
        return Err(VcfError::Parse(
            "No variants without missing data found".to_string()
        ));
    }
    
    // Number of haplotypes (2 per sample)
    let n_haplotypes = sample_names.len() * 2;
    
    // Calculate maximum valid number of components
    let max_components = std::cmp::min(valid_count, n_haplotypes);
    let n_components = std::cmp::min(n_components, max_components);
    
    // Display statistics
    display_status_box(StatusBox {
        title: "Chromosome PCA Data Preparation".to_string(),
        stats: vec![
            ("Total variants".to_string(), variants.len().to_string()),
            ("Variants with complete data".to_string(), valid_count.to_string()),
            ("Samples".to_string(), sample_names.len().to_string()),
            ("Haplotypes (2 per sample)".to_string(), n_haplotypes.to_string()),
            ("Requested PCs".to_string(), n_components.to_string()),
        ],
    });
    
    // Create data matrix efficiently
    let mut data_matrix = Array2::<f64>::zeros((n_haplotypes, valid_count));
    let mut positions = Vec::with_capacity(valid_count);
    
    // Fill data matrix without collecting filtered variants first
    let mut valid_idx = 0;
    for variant in variants {
        // Check if this variant has complete data
        if variant.genotypes.iter().all(|g| match g {
            None => false,
            Some(alleles) => alleles.len() >= 2
        }) {
            positions.push(variant.position);
            
            // Add each haplotype's data
            for (sample_idx, genotypes_opt) in variant.genotypes.iter().enumerate() {
                if let Some(genotypes) = genotypes_opt {
                    if genotypes.len() >= 2 {
                        // Left haplotype
                        let left_idx = sample_idx * 2;
                        data_matrix[[left_idx, valid_idx]] = genotypes[0] as f64;
                        
                        // Right haplotype
                        let right_idx = sample_idx * 2 + 1;
                        data_matrix[[right_idx, valid_idx]] = genotypes[1] as f64;
                    }
                }
            }
            valid_idx += 1;
        }
    }
    
    // Sanity check the matrix dimensions
    if valid_idx != valid_count {
        log(LogLevel::Warning, &format!(
            "Matrix inconsistency: Expected {} columns but found {}",
            valid_count, valid_idx
        ));
        
        // Resize the matrix if needed
        if valid_idx < valid_count {
            data_matrix = data_matrix.slice(s![.., 0..valid_idx]).to_owned();
            positions.truncate(valid_idx);
        }
    }
    
    spinner.finish_and_clear();
    
    // Apply PCA using the library
    let spinner = create_spinner("Computing PCA");
    
    let mut pca = pca::PCA::new();
    
    // Use randomized SVD without cloning the data matrix
    if let Err(e) = pca.rfit(
        data_matrix.clone(), // clone here due to library API
        n_components,
        5, // oversampling parameter
        Some(42), // random seed
        None, // no variance tolerance filter
    ) {
        return Err(VcfError::Parse(format!("PCA computation failed: {}", e)));
    }
    
    // Transform to get PC coordinates
    let transformed = match pca.transform(data_matrix) {
        Ok(t) => t,
        Err(e) => return Err(VcfError::Parse(format!("PCA transformation failed: {}", e))),
    };
    
    spinner.finish_and_clear();
    
    // Create haplotype labels
    let mut haplotype_labels = Vec::with_capacity(n_haplotypes);
    for sample_name in sample_names {
        haplotype_labels.push(format!("{}_L", sample_name)); // Left haplotype
        haplotype_labels.push(format!("{}_R", sample_name)); // Right haplotype
    }
    
    log(LogLevel::Info, &format!(
        "PCA computation complete: generated {} components for {} haplotypes",
        n_components, haplotype_labels.len()
    ));
    
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
    
    let file = File::create(&output_file)
        .map_err(|e| VcfError::Io(e))?;
    let mut writer = BufWriter::new(file);
    
    // Write header
    write!(writer, "Haplotype").map_err(|e| VcfError::Io(e))?;
    for i in 0..result.pca_coordinates.shape()[1] {
        write!(writer, "\tPC{}", i+1).map_err(|e| VcfError::Io(e))?;
    }
    writeln!(writer).map_err(|e| VcfError::Io(e))?;
    
    // Write rows - ensure haplotype count matches coordinates
    let actual_rows = std::cmp::min(
        result.haplotype_labels.len(),
        result.pca_coordinates.shape()[0]
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
    log(LogLevel::Info, &format!("PCA results for chromosome {} written to {}", 
                                chromosome, output_file.display()));
    
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
        std::fs::create_dir_all(output_dir)
            .map_err(|e| VcfError::Io(e))?;
    }
    
    // Process each chromosome separately
    let total_chr = variants_by_chr.len();
    let mut processed = 0;
    let mut successful = 0;
    
    for (chr, variants) in variants_by_chr {
        processed += 1;
        log(LogLevel::Info, &format!("Processing chromosome {} ({}/{}) with {} variants", 
                                  chr, processed, total_chr, variants.len()));
        
        // Skip chromosomes with too few variants
        if variants.len() < 2 {
            log(LogLevel::Warning, &format!("Skipping chromosome {} - too few variants ({})", 
                                         chr, variants.len()));
            continue;
        }
        
        // Compute PCA for this chromosome
        match compute_chromosome_pca(variants, sample_names, n_components) {
            Ok(result) => {
                // Write results to file
                if let Err(e) = write_chromosome_pca_to_file(&result, chr, output_dir) {
                    log(LogLevel::Warning, &format!("Failed to write PCA results for chromosome {}: {}", 
                                                 chr, e));
                } else {
                    successful += 1;
                }
            },
            Err(e) => {
                log(LogLevel::Warning, &format!("Failed to compute PCA for chromosome {}: {}", chr, e));
                // Continue with other chromosomes
            }
        }
    }
    
    if successful == 0 {
        return Err(VcfError::Parse("Failed to compute PCA for any chromosome".to_string()));
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
                    },
                    Err(e) => {
                        log(LogLevel::Warning, &format!("Error reading directory entry: {}", e));
                    }
                }
            }
        },
        Err(e) => {
            return Err(VcfError::Io(e));
        }
    }
    
    if result_files.is_empty() {
        return Err(VcfError::Parse("No chromosome PCA result files found".to_string()));
    }
    
    // Sort files by chromosome name for consistent ordering
    result_files.sort_by(|a, b| {
        let a_name = a.file_name().unwrap_or_default().to_string_lossy();
        let b_name = b.file_name().unwrap_or_default().to_string_lossy();
        a_name.cmp(&b_name)
    });
    
    // Read the first file to get haplotype names and component count
    let first_file = std::fs::read_to_string(&result_files[0])
        .map_err(|e| VcfError::Io(e))?;
    
    let mut lines = first_file.lines();
    let header = match lines.next() {
        Some(h) => h,
        None => return Err(VcfError::Parse("Empty PCA result file".to_string())),
    };
    
    let n_components = header.split('\t').count() - 1; // Subtract 1 for the 'Haplotype' column
    
    // Create a combined output file
    let file = File::create(output_file)
        .map_err(|e| VcfError::Io(e))?;
    let mut writer = BufWriter::new(file);
    
    // Write header
    write!(writer, "Haplotype\tChromosome").map_err(|e| VcfError::Io(e))?;
    
    for i in 0..n_components {
        write!(writer, "\tPC{}", i+1).map_err(|e| VcfError::Io(e))?;
    }
    
    writeln!(writer).map_err(|e| VcfError::Io(e))?;
    
    // Write results for each chromosome
    for file_path in &result_files {
        // Extract chromosome name from filename
        let chr_name = file_path.file_name()
            .unwrap_or_default()
            .to_string_lossy();
        
        let chr = chr_name
            .strip_prefix("pca_chr_")
            .and_then(|s| s.strip_suffix(".tsv"))
            .unwrap_or(&chr_name);
        
        // Read chromosome file
        let file_content = match std::fs::read_to_string(file_path) {
            Ok(content) => content,
            Err(e) => {
                log(LogLevel::Warning, &format!("Failed to read file {}: {}", file_path.display(), e));
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
    log(LogLevel::Info, &format!("Combined PCA results written to {}", output_file.display()));
    
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
        std::fs::create_dir_all(&chr_results_dir)
            .map_err(|e| VcfError::Io(e))?;
    }
    
    // Run PCA for each chromosome separately
    run_chromosome_pca_analysis(variants_by_chr, sample_names, &chr_results_dir, n_components)?;
    
    // Combine results into a single file
    let combined_output = output_dir.join("combined_chromosome_pca.tsv");
    combine_chromosome_pca_results(&chr_results_dir, &combined_output)?;
    
    log(LogLevel::Info, &format!(
        "Memory-efficient per-chromosome PCA analysis completed successfully. Results saved to {}",
        output_dir.display()
    ));
    
    Ok(())
}
