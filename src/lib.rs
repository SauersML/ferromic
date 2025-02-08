use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pub mod stats;
pub mod process;
pub mod parse;


// Wrapper for `calculate_watterson_theta`
#[pyfunction]
fn calculate_watterson_theta_wrapper(seg_sites: usize, n: usize, seq_length: i64) -> f64 {
    stats::calculate_watterson_theta(seg_sites, n, seq_length)
}

// Wrapper for `calculate_pi`
#[pyfunction]
fn calculate_pi_wrapper(tot_pair_diff: usize, n: usize, seq_length: i64) -> f64 {
    stats::calculate_pi(tot_pair_diff, n, seq_length)
}

#[pymodule]
fn ferromic(py: Python, m: &PyModule) -> PyResult<()> {
    // Expose the functions to Python
    m.add_function(wrap_pyfunction!(calculate_watterson_theta_wrapper, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_pi_wrapper, m)?)?;

    Ok(())
}
