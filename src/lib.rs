use pyo3::prelude::*;
use crate::stats::{calculate_pi, count_segregating_sites, calculate_watterson_theta};
use crate::process::{Variant, HaplotypeSide};

pub mod parse;
pub mod process;
pub mod stats;

/// PyO3 wrapper for count_segregating_sites
#[pyfunction]
fn count_segregating_sites_py(variants: Vec<Variant>) -> PyResult<usize> {
    Ok(count_segregating_sites(&variants))
}

/// PyO3 wrapper for calculate_pi
#[pyfunction]
fn calculate_pi_py(
    variants: Vec<Variant>,
    haplotypes: Vec<(usize, HaplotypeSide)>
) -> PyResult<f64> {
    Ok(calculate_pi(&variants, &haplotypes))
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
fn ferromic(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(count_segregating_sites_py, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_pi_py, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_watterson_theta_py, m)?)?;
    Ok(())
}
