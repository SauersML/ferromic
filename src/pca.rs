// pca.rs

use ndarray::Array2;
use std::path::Path;
use std::collections::HashMap;

use crate::Variant;
use crate::VcfError;
use crate::progress::{log, LogLevel, create_spinner, ProcessingStage, set_stage, display_status_box, StatusBox};

/// Structure to hold PCA results
pub struct PcaResult {
    pub haplotype_labels: Vec<String>,
    pub pca_coordinates: Array2<f64>,
    pub positions: Vec<i64>,
}

/// Computes PCA on all haplotypes in the filtered variant dataset.
/// 
/// Requires complete data (no missing genotypes) for each variant.
/// 
/// # Arguments
/// * `filtered_variants` - Vector of filtered variants with genotype data
/// * `sample_names` - Names of the samples
/// * `n_components` - Number of principal components to compute (default: 10)
/// 
/// # Returns
/// PCA results containing haplotype labels and their coordinates in PC space
pub fn compute_global_pca(
    filtered_variants: &[Variant],
    sample_names: &[String],
    n_components: usize,
) -> Result<PcaResult, VcfError> {
    set_stage(ProcessingStage::PcaAnalysis);
    log(LogLevel::Info, &format!(
        "Starting PCA computation on {} variants across {} samples ({} haplotypes)",
        filtered_variants.len(), sample_names.len(), sample_names.len() * 2
    ));
    
    // Step 1: Filter out variants with any missing data
    let spinner = create_spinner("Filtering variants with missing data");
    
    let valid_variants: Vec<&Variant> = filtered_variants
        .iter()
        .filter(|v| {
            // A variant is valid if none of its genotypes are None and all have at least 2 alleles
            v.genotypes.iter().all(|g| match g {
                None => false, // Missing genotype
                Some(alleles) => alleles.len() >= 2 // At least 2 alleles for diploid
            })
        })
        .collect();
    
    spinner.finish_and_clear();
    
    log(LogLevel::Info, &format!(
        "Found {} variants with complete data out of {} total filtered variants", 
        valid_variants.len(), filtered_variants.len()
    ));
    
    if valid_variants.is_empty() {
        return Err(VcfError::Parse(
            "No variants without missing data found".to_string()
        ));
    }
    
    // Display statistics
    display_status_box(StatusBox {
        title: "PCA Data Preparation".to_string(),
        stats: vec![
            ("Total filtered variants".to_string(), filtered_variants.len().to_string()),
            ("Variants with complete data".to_string(), valid_variants.len().to_string()),
            ("Samples".to_string(), sample_names.len().to_string()),
            ("Haplotypes (2 per sample)".to_string(), (sample_names.len() * 2).to_string()),
            ("Requested PCs".to_string(), n_components.to_string()),
        ],
    });
    
    // Step 2: Create data matrix for PCA
    let spinner = create_spinner("Creating PCA data matrix");
    
    let n_haplotypes = sample_names.len() * 2;
    let n_variants = valid_variants.len();
    
    let mut data_matrix = Array2::<f64>::zeros((n_haplotypes, n_variants));
    
    // Store positions for reference
    let mut positions = Vec::with_capacity(n_variants);
    
    // Fill the data matrix
    for (variant_idx, variant) in valid_variants.iter().enumerate() {
        positions.push(variant.position);
        
        for (sample_idx, genotypes_opt) in variant.genotypes.iter().enumerate() {
            if let Some(genotypes) = genotypes_opt {
                if genotypes.len() >= 2 {
                    // Left haplotype (index 0 in genotypes)
                    let left_idx = sample_idx * 2;
                    data_matrix[[left_idx, variant_idx]] = genotypes[0] as f64;
                    
                    // Right haplotype (index 1 in genotypes)
                    let right_idx = sample_idx * 2 + 1;
                    data_matrix[[right_idx, variant_idx]] = genotypes[1] as f64;
                }
            }
        }
    }
    
    spinner.finish_and_clear();
    
    // Step 3: Apply PCA
    let spinner = create_spinner("Computing PCA");
    
    let mut pca = pca::PCA::new();
    
    // Use randomized SVD for faster computation
    pca.rfit(
        data_matrix.clone(), 
        n_components,
        5, // oversampling parameter
        Some(42), // random seed
        None, // no variance tolerance filter
    ).map_err(|e| VcfError::Parse(format!("PCA computation failed: {}", e)))?;
    
    // Transform data to get PC coordinates
    let transformed = pca.transform(data_matrix.clone())
        .map_err(|e| VcfError::Parse(format!("PCA transformation failed: {}", e)))?;
    
    spinner.finish_and_clear();
    
    // Step 4: Create haplotype labels
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

/// Writes PCA results to a TSV file
pub fn write_pca_results_to_file(
    result: &PcaResult,
    output_file: &Path,
) -> Result<(), VcfError> {
    let spinner = create_spinner(&format!("Writing PCA results to {}", output_file.display()));
    
    let file = std::fs::File::create(output_file)
        .map_err(|e| VcfError::Io(e))?;
    let mut writer = std::io::BufWriter::new(file);
    
    use std::io::Write;
    
    // Write header
    write!(writer, "Haplotype").map_err(|e| VcfError::Io(e))?;
    for i in 0..result.pca_coordinates.shape()[1] {
        write!(writer, "\tPC{}", i+1).map_err(|e| VcfError::Io(e))?;
    }
    writeln!(writer).map_err(|e| VcfError::Io(e))?;
    
    // Write rows
    for (idx, label) in result.haplotype_labels.iter().enumerate() {
        write!(writer, "{}", label).map_err(|e| VcfError::Io(e))?;
        for j in 0..result.pca_coordinates.shape()[1] {
            write!(writer, "\t{:.6}", result.pca_coordinates[[idx, j]])
                .map_err(|e| VcfError::Io(e))?;
        }
        writeln!(writer).map_err(|e| VcfError::Io(e))?;
    }
    
    spinner.finish_and_clear();
    log(LogLevel::Info, &format!("PCA results written to {}", output_file.display()));
    
    Ok(())
}

/// Function to run the complete PCA analysis on filtered variants from multiple chromosomes
pub fn run_global_pca_analysis(
    variants_by_chr: &HashMap<String, Vec<Variant>>,
    sample_names: &[String],
    output_path: &Path,
    n_components: usize,
) -> Result<(), VcfError> {
    log(LogLevel::Info, "Starting global PCA analysis");
    
    // Combine variants from all chromosomes into a single vector
    let spinner = create_spinner("Combining variants from all chromosomes");
    
    let total_variants: usize = variants_by_chr.values().map(|v| v.len()).sum();
    let mut all_variants = Vec::with_capacity(total_variants);
    
    for (chr, variants) in variants_by_chr {
        log(LogLevel::Info, &format!("Adding {} variants from chromosome {}", variants.len(), chr));
        all_variants.extend(variants.clone());
    }
    
    spinner.finish_and_clear();
    log(LogLevel::Info, &format!("Combined {} variants from {} chromosomes", 
        all_variants.len(), variants_by_chr.len()));
    
    // Compute PCA on combined variants
    let pca_result = compute_global_pca(&all_variants, sample_names, n_components)?;
    
    // Write to TSV file
    write_pca_results_to_file(&pca_result, output_path)?;
    
    // Display completion message
    log(LogLevel::Info, &format!(
        "Global PCA analysis completed successfully. Results saved to {}",
        output_path.display()
    ));
    
    Ok(())
}
