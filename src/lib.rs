use crate::process::Variant;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::wrap_pyfunction;

pub mod parse;
pub mod process;
pub mod stats;

#[pyfunction]
fn get_theta(seg_sites: usize, n: usize, seq_length: i64) -> f64 {
    calculate_watterson_theta(seg_sites, n, seq_length)
}

#[pyfunction]
fn count_segregating_sites(variants: Vec<Variant>) -> usize {
    count_segregating_sites(&variants)
}

#[pyfunction]
fn calculate_pi(variants: Vec<Variant>, haplotypes: Vec<(usize, HaplotypeSide)>) -> f64 {
    calculate_pi(&variants, &haplotypes)
}

#[pyfunction]
fn adjusted_sequence_length(region_start: i64, region_end: i64, allow_regions: Option<Vec<(i64, i64)>>, mask_regions: Option<Vec<(i64, i64)>>) -> i64 {
    calculate_adjusted_sequence_length(region_start, region_end, allow_regions.as_ref(), mask_regions.as_ref())
}

#[pymodule]
fn ferromic(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_theta, m)?)?;
    m.add_function(wrap_pyfunction!(count_segregating_sites, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_pi, m)?)?;
    m.add_function(wrap_pyfunction!(adjusted_sequence_length, m)?)?;
    Ok(())
}
