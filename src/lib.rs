//! Python bindings for Ferromic's population genetics statistics.
//!
//! The functions exported in this module focus on providing a lightweight bridge between
//! Python data structures and the underlying Rust implementations. The interface mirrors the
//! internal APIs: callers supply iterables describing variants and haplotypes, and Ferromic
//! returns the same statistics that power the command-line application.

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::process::{HaplotypeSide, QueryRegion, Variant, VcfError};
use crate::stats::{
    calculate_adjusted_sequence_length, calculate_d_xy_hudson, calculate_fst_wc_haplotype_groups,
    calculate_hudson_fst_for_pair, calculate_hudson_fst_for_pair_with_sites,
    calculate_hudson_fst_per_site, calculate_pairwise_differences, calculate_per_site_diversity,
    calculate_pi, calculate_watterson_theta, count_segregating_sites, DxyHudsonResult, FstEstimate,
    FstWcResults, HudsonFSTOutcome, PopulationContext, PopulationId, SiteDiversity, SiteFstHudson,
    SiteFstWc,
};

// Module declarations
pub mod parse;
pub mod pca;
pub mod process;
pub mod progress;
pub mod stats;
pub mod transcripts;

#[cfg(test)]
mod tests {
    mod filter_tests;
    mod hudson_fst_tests;
    mod interval_tests;
}

/// PyO3 wrapper for count_segregating_sites
///
/// Counts the number of segregating sites (polymorphic positions) in a collection of variants.
///
/// # Arguments
/// * `variants_obj` - An iterable of Python objects describing variants.
///   Each object must expose a `.position` attribute (integer) and a
///   `.genotypes` iterable. Elements inside `.genotypes` are either
///   `None` (missing data) or iterables of allele indices (`0` for reference,
///   `1` for the first alternate, etc.).
///
/// # Returns
/// * Number of segregating sites as usize
///
/// ## Python example
/// ```python
/// import ferromic
///
///
/// class Variant:
///     def __init__(self, position, genotypes):
///         self.position = position
///         self.genotypes = genotypes
///
/// variants = [
///     Variant(1000, [(0, 0), (0, 1), None]),
/// ]
///
/// count = ferromic.count_segregating_sites_py(variants)
/// ```
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
/// * `variants_obj` - Same structure as `count_segregating_sites_py`.
/// * `haplotypes_obj` - Iterable of two-item sequences `(sample_index, side)`.
///   `sample_index` must be a `usize` and `side` should be `0` (left) or `1`
///   (right), matching [`HaplotypeSide`].
/// * `seq_length` - The sequence length (number of sites) to normalize by.
///
/// # Returns
/// * Nucleotide diversity (π) as f64
///
/// ## Python example
/// ```python
/// import ferromic
///
/// haplotypes = [(0, 0), (0, 1), (1, 0)]
/// pi = ferromic.calculate_pi_py(variants, haplotypes, seq_length=500)
/// ```
#[pyfunction]
fn calculate_pi_py(
    _py: Python,
    variants_obj: &PyAny,
    haplotypes_obj: &PyAny,
    seq_length: i64,
) -> PyResult<f64> {
    // Convert Python variant objects to Rust Variant structs
    let rust_variants = extract_variants_from_python(variants_obj)?;

    // Convert Python haplotype objects to Rust (usize, HaplotypeSide) tuples
    let rust_haplotypes = extract_haplotypes_from_python(haplotypes_obj)?;

    // Validate the sequence length
    if seq_length <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Sequence length must be positive",
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
/// * `seg_sites` - Number of segregating sites (e.g., from `count_segregating_sites_py`).
/// * `n` - Number of sequences/haplotypes included in the analysis.
/// * `seq_length` - The sequence length (number of sites) to normalize by.
///
/// # Returns
/// * Watterson's θ as f64
///
/// ## Python example
/// ```python
/// import ferromic
///
/// theta = ferromic.calculate_watterson_theta_py(seg_sites, n=len(haplotypes), seq_length=500)
/// ```
#[pyfunction]
fn calculate_watterson_theta_py(seg_sites: usize, n: usize, seq_length: i64) -> PyResult<f64> {
    // Validate inputs
    if n <= 1 {
        return Err(PyValueError::new_err(
            "Number of sequences (n) must be greater than 1",
        ));
    }

    if seq_length <= 0 {
        return Err(PyValueError::new_err("Sequence length must be positive"));
    }

    // Call the Rust implementation
    Ok(calculate_watterson_theta(seg_sites, n, seq_length))
}

/// PyO3 wrapper for `calculate_pairwise_differences`.
///
/// Returns a list of tuples `(sample_i, sample_j, differences, comparable_sites)`.
#[pyfunction]
fn calculate_pairwise_differences_py(
    _py: Python,
    variants_obj: &PyAny,
    number_of_samples: usize,
) -> PyResult<Vec<(usize, usize, usize, usize)>> {
    let rust_variants = extract_variants_from_python(variants_obj)?;

    let differences = calculate_pairwise_differences(&rust_variants, number_of_samples)
        .into_iter()
        .map(|((i, j), diff, comparable)| (i, j, diff, comparable))
        .collect();

    Ok(differences)
}

/// PyO3 wrapper for `calculate_per_site_diversity`.
///
/// Returns a list of dictionaries with keys: `position`, `pi`, `watterson_theta`.
#[pyfunction]
fn calculate_per_site_diversity_py(
    py: Python,
    variants_obj: &PyAny,
    haplotypes_obj: &PyAny,
    region_start: i64,
    region_end: i64,
) -> PyResult<PyObject> {
    if region_end < region_start {
        return Err(PyValueError::new_err(
            "region_end must be greater than or equal to region_start",
        ));
    }

    let rust_variants = extract_variants_from_python(variants_obj)?;
    let rust_haplotypes = extract_haplotypes_from_python(haplotypes_obj)?;
    let region = QueryRegion {
        start: region_start,
        end: region_end,
    };

    let sites: Vec<SiteDiversity> =
        calculate_per_site_diversity(&rust_variants, &rust_haplotypes, region);

    let py_list = PyList::empty(py);
    for site in &sites {
        py_list.append(site_diversity_to_py(py, site)?)?;
    }

    Ok(py_list.into())
}

/// PyO3 wrapper for `calculate_d_xy_hudson`.
#[pyfunction]
fn calculate_dxy_hudson_py(py: Python, pop1_obj: &PyAny, pop2_obj: &PyAny) -> PyResult<PyObject> {
    let pop1_owned = parse_population_context(pop1_obj)?;
    let pop2_owned = parse_population_context(pop2_obj)?;

    let pop1_ctx = pop1_owned.as_population_context();
    let pop2_ctx = pop2_owned.as_population_context();

    let result = calculate_d_xy_hudson(&pop1_ctx, &pop2_ctx).map_err(vcf_error_to_pyerr)?;

    Ok(dxy_result_to_py(py, &result)?)
}

/// PyO3 wrapper for `calculate_hudson_fst_per_site`.
#[pyfunction]
fn calculate_hudson_fst_per_site_py(
    py: Python,
    pop1_obj: &PyAny,
    pop2_obj: &PyAny,
    region_start: i64,
    region_end: i64,
) -> PyResult<PyObject> {
    if region_end < region_start {
        return Err(PyValueError::new_err(
            "region_end must be greater than or equal to region_start",
        ));
    }

    let pop1_owned = parse_population_context(pop1_obj)?;
    let pop2_owned = parse_population_context(pop2_obj)?;
    let pop1_ctx = pop1_owned.as_population_context();
    let pop2_ctx = pop2_owned.as_population_context();

    let region = QueryRegion {
        start: region_start,
        end: region_end,
    };

    let sites = calculate_hudson_fst_per_site(&pop1_ctx, &pop2_ctx, region);

    let py_list = PyList::empty(py);
    for site in &sites {
        py_list.append(site_fst_hudson_to_py(py, site)?)?;
    }

    Ok(py_list.into())
}

/// PyO3 wrapper for `calculate_hudson_fst_for_pair`.
#[pyfunction]
fn calculate_hudson_fst_for_pair_py(
    py: Python,
    pop1_obj: &PyAny,
    pop2_obj: &PyAny,
) -> PyResult<PyObject> {
    let pop1_owned = parse_population_context(pop1_obj)?;
    let pop2_owned = parse_population_context(pop2_obj)?;
    let pop1_ctx = pop1_owned.as_population_context();
    let pop2_ctx = pop2_owned.as_population_context();

    let outcome =
        calculate_hudson_fst_for_pair(&pop1_ctx, &pop2_ctx).map_err(vcf_error_to_pyerr)?;

    Ok(hudson_outcome_to_py(py, &outcome)?)
}

/// PyO3 wrapper for `calculate_hudson_fst_for_pair_with_sites`.
#[pyfunction]
fn calculate_hudson_fst_for_pair_with_sites_py(
    py: Python,
    pop1_obj: &PyAny,
    pop2_obj: &PyAny,
    region_start: i64,
    region_end: i64,
) -> PyResult<PyObject> {
    if region_end < region_start {
        return Err(PyValueError::new_err(
            "region_end must be greater than or equal to region_start",
        ));
    }

    let pop1_owned = parse_population_context(pop1_obj)?;
    let pop2_owned = parse_population_context(pop2_obj)?;
    let pop1_ctx = pop1_owned.as_population_context();
    let pop2_ctx = pop2_owned.as_population_context();

    let region = QueryRegion {
        start: region_start,
        end: region_end,
    };

    let (outcome, sites) = calculate_hudson_fst_for_pair_with_sites(&pop1_ctx, &pop2_ctx, region)
        .map_err(vcf_error_to_pyerr)?;

    let output = PyDict::new(py);
    output.set_item("outcome", hudson_outcome_to_py(py, &outcome)?)?;

    let py_sites = PyList::empty(py);
    for site in &sites {
        py_sites.append(site_fst_hudson_to_py(py, site)?)?;
    }
    output.set_item("sites", py_sites)?;

    Ok(output.into())
}

/// PyO3 wrapper for `calculate_fst_wc_haplotype_groups`.
#[pyfunction]
fn calculate_fst_wc_haplotype_groups_py(
    py: Python,
    variants_obj: &PyAny,
    sample_names_obj: &PyAny,
    sample_to_group_obj: &PyAny,
    region_start: i64,
    region_end: i64,
) -> PyResult<PyObject> {
    if region_end < region_start {
        return Err(PyValueError::new_err(
            "region_end must be greater than or equal to region_start",
        ));
    }

    let variants = extract_variants_from_python(variants_obj)?;
    let sample_names = extract_sample_names_from_python(sample_names_obj)?;
    let sample_to_group = extract_sample_group_map(sample_to_group_obj)?;
    let region = QueryRegion {
        start: region_start,
        end: region_end,
    };

    let results =
        calculate_fst_wc_haplotype_groups(&variants, &sample_names, &sample_to_group, region);

    Ok(fst_wc_results_to_py(py, &results)?)
}

/// PyO3 wrapper for `calculate_adjusted_sequence_length`.
#[pyfunction]
fn calculate_adjusted_sequence_length_py(
    region_start: i64,
    region_end: i64,
    allow_regions: Option<&PyAny>,
    mask_regions: Option<&PyAny>,
) -> PyResult<i64> {
    if region_end < region_start {
        return Err(PyValueError::new_err(
            "region_end must be greater than or equal to region_start",
        ));
    }

    let allow_vec = extract_interval_list(allow_regions)?;
    let mask_vec = extract_interval_list(mask_regions)?;

    Ok(calculate_adjusted_sequence_length(
        region_start,
        region_end,
        allow_vec.as_ref(),
        mask_vec.as_ref(),
    ))
}

/// Helper function to extract [`Variant`] structs from Python variant objects.
///
/// The conversion layer accepts any Python object that exposes a `position`
/// attribute and an iterable `genotypes` attribute. Genotypes may be `None`
/// to represent missing calls or any iterable of integers describing the
/// called alleles for a sample.
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

/// Helper function to extract haplotype information from Python objects.
///
/// Each element in `haplotypes_obj` must be indexable with `[0]` for the
/// sample index and `[1]` for the side flag (`0` = left, `1` = right).
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
            _ => return Err(PyValueError::new_err("Side must be 0 (Left) or 1 (Right)")),
        };

        rust_haplotypes.push((index, side));
    }

    Ok(rust_haplotypes)
}

struct OwnedPopulationContext {
    id: PopulationId,
    haplotypes: Vec<(usize, HaplotypeSide)>,
    variants: Vec<Variant>,
    sample_names: Vec<String>,
    sequence_length: i64,
}

impl OwnedPopulationContext {
    fn as_population_context(&self) -> PopulationContext<'_> {
        PopulationContext {
            id: self.id.clone(),
            haplotypes: self.haplotypes.clone(),
            variants: &self.variants,
            sample_names: &self.sample_names,
            sequence_length: self.sequence_length,
        }
    }
}

fn parse_population_context(obj: &PyAny) -> PyResult<OwnedPopulationContext> {
    let id_obj = get_required_field(obj, "id")?;
    let haplotypes_obj = get_required_field(obj, "haplotypes")?;
    let variants_obj = get_required_field(obj, "variants")?;
    let sequence_length_obj = get_required_field(obj, "sequence_length")?;

    let id = parse_population_id(id_obj)?;
    let haplotypes = extract_haplotypes_from_python(haplotypes_obj)?;
    let variants = extract_variants_from_python(variants_obj)?;
    let sequence_length = sequence_length_obj.extract::<i64>()?;

    let sample_names = match get_field(obj, "sample_names")? {
        Some(value) => extract_sample_names_from_python(value)?,
        None => Vec::new(),
    };

    Ok(OwnedPopulationContext {
        id,
        haplotypes,
        variants,
        sample_names,
        sequence_length,
    })
}

fn get_field<'a>(obj: &'a PyAny, name: &str) -> PyResult<Option<&'a PyAny>> {
    if let Ok(value) = obj.get_item(name) {
        return Ok(Some(value));
    }

    if let Ok(value) = obj.getattr(name) {
        return Ok(Some(value));
    }

    Ok(None)
}

fn get_required_field<'a>(obj: &'a PyAny, name: &str) -> PyResult<&'a PyAny> {
    get_field(obj, name)?.ok_or_else(|| {
        PyValueError::new_err(format!(
            "Population context missing required field '{name}'"
        ))
    })
}

fn parse_population_id(obj: &PyAny) -> PyResult<PopulationId> {
    if let Ok(dict) = obj.downcast::<PyDict>() {
        if let Some(value) = dict.get_item("haplotype_group") {
            return Ok(PopulationId::HaplotypeGroup(value.extract::<u8>()?));
        }

        if let Some(value) = dict.get_item("named") {
            return Ok(PopulationId::Named(value.extract::<String>()?));
        }

        return Err(PyValueError::new_err(
            "Population id dict must contain 'haplotype_group' or 'named'",
        ));
    }

    if let Ok(group) = obj.extract::<u8>() {
        return Ok(PopulationId::HaplotypeGroup(group));
    }

    if let Ok(name) = obj.extract::<String>() {
        return Ok(PopulationId::Named(name));
    }

    Err(PyValueError::new_err(
        "Unable to interpret population id; expected int, string, or mapping",
    ))
}

fn extract_sample_names_from_python(sample_names_obj: &PyAny) -> PyResult<Vec<String>> {
    let mut names = Vec::new();
    for item in sample_names_obj.iter()? {
        names.push(item?.extract::<String>()?);
    }
    Ok(names)
}

fn extract_sample_group_map(sample_to_group_obj: &PyAny) -> PyResult<HashMap<String, (u8, u8)>> {
    let mapping: &PyDict = sample_to_group_obj
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("sample_to_group_map must be a dict"))?;

    let mut result = HashMap::with_capacity(mapping.len());
    for (key, value) in mapping.iter() {
        let sample = key.extract::<String>()?;
        let left = value.get_item(0)?.extract::<u8>()?;
        let right = value.get_item(1)?.extract::<u8>()?;
        result.insert(sample, (left, right));
    }

    Ok(result)
}

fn extract_interval_list(obj: Option<&PyAny>) -> PyResult<Option<Vec<(i64, i64)>>> {
    if let Some(seq_obj) = obj {
        let mut intervals = Vec::new();
        for entry in seq_obj.iter()? {
            let entry = entry?;
            let start = entry.get_item(0)?.extract::<i64>()?;
            let end = entry.get_item(1)?.extract::<i64>()?;
            intervals.push((start, end));
        }
        Ok(Some(intervals))
    } else {
        Ok(None)
    }
}

fn vcf_error_to_pyerr(err: VcfError) -> PyErr {
    PyValueError::new_err(format!("VCF error: {:?}", err))
}

fn site_diversity_to_py(py: Python, site: &SiteDiversity) -> PyResult<PyObject> {
    let entry = PyDict::new(py);
    entry.set_item("position", site.position)?;
    entry.set_item("pi", site.pi)?;
    entry.set_item("watterson_theta", site.watterson_theta)?;
    Ok(entry.into())
}

fn dxy_result_to_py(py: Python, result: &DxyHudsonResult) -> PyResult<PyObject> {
    let entry = PyDict::new(py);
    entry.set_item("d_xy", result.d_xy)?;
    Ok(entry.into())
}

fn population_id_to_py(py: Python, id: &PopulationId) -> PyResult<PyObject> {
    let entry = PyDict::new(py);
    match id {
        PopulationId::HaplotypeGroup(group) => entry.set_item("haplotype_group", *group)?,
        PopulationId::Named(name) => entry.set_item("named", name.clone())?,
    }
    Ok(entry.into())
}

fn hudson_outcome_to_py(py: Python, outcome: &HudsonFSTOutcome) -> PyResult<PyObject> {
    let entry = PyDict::new(py);
    let pop1 = match &outcome.pop1_id {
        Some(id) => population_id_to_py(py, id)?,
        None => py.None(),
    };
    let pop2 = match &outcome.pop2_id {
        Some(id) => population_id_to_py(py, id)?,
        None => py.None(),
    };
    entry.set_item("pop1_id", pop1)?;
    entry.set_item("pop2_id", pop2)?;
    entry.set_item("fst", outcome.fst)?;
    entry.set_item("d_xy", outcome.d_xy)?;
    entry.set_item("pi_pop1", outcome.pi_pop1)?;
    entry.set_item("pi_pop2", outcome.pi_pop2)?;
    entry.set_item("pi_xy_avg", outcome.pi_xy_avg)?;
    Ok(entry.into())
}

fn site_fst_hudson_to_py(py: Python, site: &SiteFstHudson) -> PyResult<PyObject> {
    let entry = PyDict::new(py);
    entry.set_item("position", site.position)?;
    entry.set_item("fst", site.fst)?;
    entry.set_item("d_xy", site.d_xy)?;
    entry.set_item("pi_pop1", site.pi_pop1)?;
    entry.set_item("pi_pop2", site.pi_pop2)?;
    entry.set_item("n1_called", site.n1_called)?;
    entry.set_item("n2_called", site.n2_called)?;
    entry.set_item("num_component", site.num_component)?;
    entry.set_item("den_component", site.den_component)?;
    Ok(entry.into())
}

fn fst_estimate_to_py(py: Python, estimate: &FstEstimate) -> PyResult<PyObject> {
    let entry = PyDict::new(py);
    match estimate {
        FstEstimate::Calculable {
            value,
            sum_a,
            sum_b,
            num_informative_sites,
        } => {
            entry.set_item("state", "calculable")?;
            entry.set_item("value", *value)?;
            entry.set_item("sum_a", *sum_a)?;
            entry.set_item("sum_b", *sum_b)?;
            entry.set_item("sites", *num_informative_sites)?;
        }
        FstEstimate::ComponentsYieldIndeterminateRatio {
            sum_a,
            sum_b,
            num_informative_sites,
        } => {
            entry.set_item("state", "components_yield_indeterminate_ratio")?;
            entry.set_item("value", Option::<f64>::None)?;
            entry.set_item("sum_a", *sum_a)?;
            entry.set_item("sum_b", *sum_b)?;
            entry.set_item("sites", *num_informative_sites)?;
        }
        FstEstimate::NoInterPopulationVariance {
            sum_a,
            sum_b,
            sites_evaluated,
        } => {
            entry.set_item("state", "no_inter_population_variance")?;
            entry.set_item("value", Option::<f64>::None)?;
            entry.set_item("sum_a", *sum_a)?;
            entry.set_item("sum_b", *sum_b)?;
            entry.set_item("sites", *sites_evaluated)?;
        }
        FstEstimate::InsufficientDataForEstimation {
            sum_a,
            sum_b,
            sites_attempted,
        } => {
            entry.set_item("state", "insufficient_data_for_estimation")?;
            entry.set_item("value", Option::<f64>::None)?;
            entry.set_item("sum_a", *sum_a)?;
            entry.set_item("sum_b", *sum_b)?;
            entry.set_item("sites", *sites_attempted)?;
        }
    }
    Ok(entry.into())
}

fn site_fst_wc_to_py(py: Python, site: &SiteFstWc) -> PyResult<PyObject> {
    let entry = PyDict::new(py);
    entry.set_item("position", site.position)?;
    entry.set_item("overall_fst", fst_estimate_to_py(py, &site.overall_fst)?)?;

    let pairwise = PyDict::new(py);
    for (key, value) in &site.pairwise_fst {
        pairwise.set_item(key, fst_estimate_to_py(py, value)?)?;
    }
    entry.set_item("pairwise_fst", pairwise)?;

    entry.set_item("variance_components", site.variance_components)?;

    let pop_sizes = PyDict::new(py);
    for (key, value) in &site.population_sizes {
        pop_sizes.set_item(key, *value)?;
    }
    entry.set_item("population_sizes", pop_sizes)?;

    let pairwise_var = PyDict::new(py);
    for (key, value) in &site.pairwise_variance_components {
        pairwise_var.set_item(key, value)?;
    }
    entry.set_item("pairwise_variance_components", pairwise_var)?;

    Ok(entry.into())
}

fn fst_wc_results_to_py(py: Python, results: &FstWcResults) -> PyResult<PyObject> {
    let entry = PyDict::new(py);
    entry.set_item("overall_fst", fst_estimate_to_py(py, &results.overall_fst)?)?;

    let pairwise = PyDict::new(py);
    for (key, value) in &results.pairwise_fst {
        pairwise.set_item(key, fst_estimate_to_py(py, value)?)?;
    }
    entry.set_item("pairwise_fst", pairwise)?;

    let pairwise_var = PyDict::new(py);
    for (key, value) in &results.pairwise_variance_components {
        pairwise_var.set_item(key, value)?;
    }
    entry.set_item("pairwise_variance_components", pairwise_var)?;

    let site_list = PyList::empty(py);
    for site in &results.site_fst {
        site_list.append(site_fst_wc_to_py(py, site)?)?;
    }
    entry.set_item("site_fst", site_list)?;
    entry.set_item("fst_type", results.fst_type.clone())?;

    Ok(entry.into())
}

/// PyO3 module definition.
#[pymodule]
fn ferromic(_py: Python, m: &PyModule) -> PyResult<()> {
    // Register Python functions
    m.add_function(wrap_pyfunction!(count_segregating_sites_py, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_pi_py, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_watterson_theta_py, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_pairwise_differences_py, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_per_site_diversity_py, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_dxy_hudson_py, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_hudson_fst_per_site_py, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_hudson_fst_for_pair_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        calculate_hudson_fst_for_pair_with_sites_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(calculate_fst_wc_haplotype_groups_py, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_adjusted_sequence_length_py, m)?)?;

    Ok(())
}
