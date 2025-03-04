use pyo3::prelude::*;
use crate::stats::{calculate_pi, count_segregating_sites, calculate_watterson_theta};
use crate::process::{Variant, HaplotypeSide};

pub mod parse;
pub mod process;
pub mod stats;
pub mod transcripts;

/// PyO3 wrapper for count_segregating_sites
#[pyfunction]
fn count_segregating_sites_py(_py: Python, variants_obj: &PyAny) -> PyResult<usize> {
    // Extract just the needed fields from Python objects
    let mut rust_variants = Vec::new();
    
    for py_variant in variants_obj.iter()? {
        let py_variant = py_variant?;
        let position = py_variant.getattr("position")?.extract::<i64>()?;
        let py_genotypes = py_variant.getattr("genotypes")?;
        
        let mut genotypes = Vec::new();
        for py_gt in py_genotypes.iter()? {
            let py_gt = py_gt?;
            if py_gt.is_none() {
                genotypes.push(None);
                continue;
            }
            
            let mut alleles = Vec::new();
            for py_allele in py_gt.iter()? {
                let py_allele = py_allele?;
                let allele = py_allele.extract::<u8>()?;
                alleles.push(allele);
            }
            genotypes.push(Some(alleles));
        }
        
        rust_variants.push(Variant {
            position,
            genotypes,
        });
    }
    
    Ok(count_segregating_sites(&rust_variants))
}

/// PyO3 wrapper for calculate_pi
#[pyfunction]
fn calculate_pi_py(_py: Python, variants_obj: &PyAny, haplotypes_obj: &PyAny) -> PyResult<f64> {
    // Extract variants
    let mut rust_variants = Vec::new();
    for py_variant in variants_obj.iter()? {
        let py_variant = py_variant?;
        let position = py_variant.getattr("position")?.extract::<i64>()?;
        let py_genotypes = py_variant.getattr("genotypes")?;
        
        let mut genotypes = Vec::new();
        for py_gt in py_genotypes.iter()? {
            let py_gt = py_gt?;
            if py_gt.is_none() {
                genotypes.push(None);
                continue;
            }
            
            let mut alleles = Vec::new();
            for py_allele in py_gt.iter()? {
                let py_allele = py_allele?;
                let allele = py_allele.extract::<u8>()?;
                alleles.push(allele);
            }
            genotypes.push(Some(alleles));
        }
        
        rust_variants.push(Variant {
            position,
            genotypes,
        });
    }
    
    // Extract haplotypes
    let mut rust_haplotypes = Vec::new();
    for py_hap in haplotypes_obj.iter()? {
        let py_hap = py_hap?;
        let index = py_hap.get_item(0)?.extract::<usize>()?;
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
    
    Ok(calculate_pi(&rust_variants, &rust_haplotypes))
}

/// PyO3 wrapper for calculate_watterson_theta
#[pyfunction]
fn calculate_watterson_theta_py(
    seg_sites: usize,
    n: usize,
    seq_length: i64
) -> PyResult<f64> {
    Ok(calculate_watterson_theta(seg_sites, n, seq_length))
}

/// PyO3 module definition
#[pymodule]
fn ferromic(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(count_segregating_sites_py, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_pi_py, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_watterson_theta_py, m)?)?;
    Ok(())
}
