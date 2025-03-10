use pyo3::prelude::*;
use crate::stats::{calculate_pi, count_segregating_sites, calculate_watterson_theta};
use crate::process::{Variant, HaplotypeSide};

// Module declarations
pub mod parse;
pub mod process;
pub mod stats;
pub mod transcripts;
pub mod progress;

/// PyO3 wrapper for count_segregating_sites
///
/// Counts the number of segregating sites (polymorphic positions) in a collection of variants.
///
/// # Arguments
/// * `variants_obj` - A Python list of variant objects with position and genotypes
///
/// # Returns
/// * Number of segregating sites as usize
#[pyfunction]
fn count_segregating_sites_py(_py: Python, variants_obj: &PyAny) -> PyResult<usize> {
    // Convert Python variant objects to Rust Variant structs
    let rust_variants = extract_variants_from_python(variants_obj)?;
    
    // Call the Rust implementation
    Ok(count_segregating_sites(&rust_variants))
}

/// PyO3 wrapper for calculate_pi (nucleotide diversity)
///
/// Calculates π, the average number of nucleotide differences per site between sequences.
///
/// # Arguments
/// * `variants_obj` - A Python list of variant objects with position and genotypes
/// * `haplotypes_obj` - A Python list of tuples (sample_index, haplotype_side)
/// * `seq_length` - The sequence length (number of sites) to normalize by
///
/// # Returns
/// * Nucleotide diversity (π) as f64
#[pyfunction]
fn calculate_pi_py(_py: Python, variants_obj: &PyAny, haplotypes_obj: &PyAny, seq_length: i64) -> PyResult<f64> {
    // Convert Python variant objects to Rust Variant structs
    let rust_variants = extract_variants_from_python(variants_obj)?;
    
    // Convert Python haplotype objects to Rust (usize, HaplotypeSide) tuples
    let rust_haplotypes = extract_haplotypes_from_python(haplotypes_obj)?;
    
    // Validate the sequence length
    if seq_length <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Sequence length must be positive"
        ));
    }
    
    // Call the Rust implementation
    Ok(calculate_pi(&rust_variants, &rust_haplotypes, seq_length))
}

/// PyO3 wrapper for calculate_watterson_theta
///
/// Calculates Watterson's estimator of population mutation rate (θ).
///
/// # Arguments
/// * `seg_sites` - Number of segregating sites
/// * `n` - Number of sequences/haplotypes
/// * `seq_length` - The sequence length (number of sites) to normalize by
///
/// # Returns
/// * Watterson's θ as f64
#[pyfunction]
fn calculate_watterson_theta_py(
    seg_sites: usize,
    n: usize,
    seq_length: i64
) -> PyResult<f64> {
    // Validate inputs
    if n <= 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Number of sequences (n) must be greater than 1"
        ));
    }
    
    if seq_length <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Sequence length must be positive"
        ));
    }
    
    // Call the Rust implementation
    Ok(calculate_watterson_theta(seg_sites, n, seq_length))
}

/// Helper function to extract Variant structs from Python variant objects
fn extract_variants_from_python(variants_obj: &PyAny) -> PyResult<Vec<Variant>> {
    let mut rust_variants = Vec::new();
    
    // Iterate through Python list of variants
    for py_variant in variants_obj.iter()? {
        let py_variant = py_variant?;
        
        // Extract position
        let position = py_variant.getattr("position")?.extract::<i64>()?;
        
        // Extract genotypes
        let py_genotypes = py_variant.getattr("genotypes")?;
        let mut genotypes = Vec::new();
        
        // Process each genotype
        for py_gt in py_genotypes.iter()? {
            let py_gt = py_gt?;
            
            // Handle None genotypes
            if py_gt.is_none() {
                genotypes.push(None);
                continue;
            }
            
            // Extract alleles for this genotype
            let mut alleles = Vec::new();
            for py_allele in py_gt.iter()? {
                let py_allele = py_allele?;
                let allele = py_allele.extract::<u8>()?;
                alleles.push(allele);
            }
            
            genotypes.push(Some(alleles));
        }
        
        // Create and add the Rust Variant
        rust_variants.push(Variant {
            position,
            genotypes,
        });
    }
    
    Ok(rust_variants)
}

/// Helper function to extract haplotype information from Python objects
fn extract_haplotypes_from_python(haplotypes_obj: &PyAny) -> PyResult<Vec<(usize, HaplotypeSide)>> {
    let mut rust_haplotypes = Vec::new();
    
    // Iterate through Python list of haplotypes
    for py_hap in haplotypes_obj.iter()? {
        let py_hap = py_hap?;
        
        // Extract sample index
        let index = py_hap.get_item(0)?.extract::<usize>()?;
        
        // Extract and convert haplotype side
        let side_int = py_hap.get_item(1)?.extract::<u8>()?;
        
        let side = match side_int {
            0 => HaplotypeSide::Left,
            1 => HaplotypeSide::Right,
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                "Side must be 0 (Left) or 1 (Right)"
            )),
        };
        
        rust_haplotypes.push((index, side));
    }
    
    Ok(rust_haplotypes)
}

/// PyO3 module definition
#[pymodule]
fn ferromic(_py: Python, m: &PyModule) -> PyResult<()> {
    // Register Python functions
    m.add_function(wrap_pyfunction!(count_segregating_sites_py, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_pi_py, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_watterson_theta_py, m)?)?;
    
    Ok(())
}
