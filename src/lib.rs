use crate::process::Variant;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::wrap_pyfunction;

pub mod parse;
pub mod process;
pub mod stats;

// Wrapper for `calculate_watterson_theta`
#[pyfunction]
fn get_theta(py: Python, seg_sites: usize, n: usize, seq_length: i64) -> PyResult<f64> {
    py.allow_threads(|| Ok(stats::calculate_watterson_theta(seg_sites, n, seq_length)))
}

// Wrapper for `calculate_pi`
// Expects variants as a list of [position: i64, genotypes: list of optional lists of u8]
#[pyfunction]
fn get_pi(py: Python, variants: &PyList, n: usize) -> PyResult<f64> {
    let rust_variants: Vec<Variant> = variants
        .iter()
        .map(|item| {
            let tuple = item.extract::<(i64, Vec<Option<Vec<u8>>>)>()?;
            Ok(Variant {
                position: tuple.0,
                genotypes: tuple.1,
            })
        })
        .collect::<PyResult<Vec<Variant>>>()?;

    py.allow_threads(|| {
        let hap_group: Vec<(usize, u8)> = (0..n).map(|i| (i, 0)).collect();
        Ok(stats::calculate_pi(&rust_variants, &hap_group))
    })
}

#[pymodule]
fn ferromic(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_theta, m)?)?;
    m.add_function(wrap_pyfunction!(get_pi, m)?)?;

    // Use `py` to check Python version, ensuring it's utilized
    let version = py.version();
    println!("Initialized ferromic with Python version: {}", version);

    Ok(())
}
