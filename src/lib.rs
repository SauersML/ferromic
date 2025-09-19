#![feature(portable_simd)]
//! Ergonomic Python bindings for Ferromic's population genetics toolkit.
//!
//! The bindings expose a high-level, well-documented API that mirrors the core
//! Rust statistics while embracing Python conventions. Most functions accept
//! plain Python data structures (dictionaries, lists, tuples, dataclasses), and
//! the returned values are lightweight Python classes with useful
//! introspection-friendly attributes. The goal is to make Ferromic feel like a
//! native Python library while retaining Rust's performance.

use std::collections::HashMap;
use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyIterator, PyList, PyTuple};
use pyo3::IntoPy;

use crate::pca::{
    compute_chromosome_pca, run_chromosome_pca_analysis, run_global_pca_analysis,
    write_chromosome_pca_to_file, PcaResult,
};
use crate::process::{HaplotypeSide, QueryRegion, Variant, VcfError};
use crate::stats::{
    calculate_adjusted_sequence_length, calculate_d_xy_hudson, calculate_fst_wc_haplotype_groups,
    calculate_hudson_fst_for_pair, calculate_hudson_fst_for_pair_with_sites,
    calculate_hudson_fst_per_site, calculate_inversion_allele_frequency,
    calculate_pairwise_differences, calculate_per_site_diversity, calculate_pi,
    calculate_watterson_theta, count_segregating_sites, DxyHudsonResult, FstEstimate, FstWcResults,
    HudsonFSTOutcome, PopulationContext, PopulationId, SiteDiversity, SiteFstHudson, SiteFstWc,
};

// Module declarations so that cargo exposes the command line utilities as well.
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

/// Rich description of an FST estimate, exposed as ``ferromic.FstEstimate``.
#[pyclass(module = "ferromic", name = "FstEstimate")]
#[derive(Clone)]
struct FstEstimateInfo {
    #[pyo3(get)]
    state: String,
    #[pyo3(get)]
    value: Option<f64>,
    #[pyo3(get)]
    sum_a: Option<f64>,
    #[pyo3(get)]
    sum_b: Option<f64>,
    #[pyo3(get)]
    sites: Option<usize>,
}

#[pymethods]
impl FstEstimateInfo {
    /// Return the tuple ``(value, sum_a, sum_b, sites)``.
    fn components(&self) -> (Option<f64>, Option<f64>, Option<f64>, Option<usize>) {
        (self.value, self.sum_a, self.sum_b, self.sites)
    }

    fn __repr__(&self) -> String {
        match (self.state.as_str(), self.value) {
            (state, Some(value)) => {
                format!(
                    "FstEstimate(state='{state}', value={value:.6}, sum_a={:?}, sum_b={:?}, sites={:?})",
                    self.sum_a, self.sum_b, self.sites
                )
            }
            (state, None) => format!(
                "FstEstimate(state='{state}', value=None, sum_a={:?}, sum_b={:?}, sites={:?})",
                self.sum_a, self.sum_b, self.sites
            ),
        }
    }
}

impl FstEstimateInfo {
    fn from_estimate(py: Python, estimate: &FstEstimate) -> PyResult<Py<Self>> {
        let info = match estimate {
            FstEstimate::Calculable {
                value,
                sum_a,
                sum_b,
                num_informative_sites,
            } => FstEstimateInfo {
                state: "calculable".to_string(),
                value: Some(*value),
                sum_a: Some(*sum_a),
                sum_b: Some(*sum_b),
                sites: Some(*num_informative_sites),
            },
            FstEstimate::ComponentsYieldIndeterminateRatio {
                sum_a,
                sum_b,
                num_informative_sites,
            } => FstEstimateInfo {
                state: "components_yield_indeterminate_ratio".to_string(),
                value: None,
                sum_a: Some(*sum_a),
                sum_b: Some(*sum_b),
                sites: Some(*num_informative_sites),
            },
            FstEstimate::NoInterPopulationVariance {
                sum_a,
                sum_b,
                sites_evaluated,
            } => FstEstimateInfo {
                state: "no_inter_population_variance".to_string(),
                value: None,
                sum_a: Some(*sum_a),
                sum_b: Some(*sum_b),
                sites: Some(*sites_evaluated),
            },
            FstEstimate::InsufficientDataForEstimation {
                sum_a,
                sum_b,
                sites_attempted,
            } => FstEstimateInfo {
                state: "insufficient_data_for_estimation".to_string(),
                value: None,
                sum_a: Some(*sum_a),
                sum_b: Some(*sum_b),
                sites: Some(*sites_attempted),
            },
        };
        Py::new(py, info)
    }
}

/// Difference counts between two samples.
#[pyclass(module = "ferromic")]
#[derive(Clone)]
struct PairwiseDifference {
    #[pyo3(get)]
    sample_i: usize,
    #[pyo3(get)]
    sample_j: usize,
    #[pyo3(get)]
    differences: usize,
    #[pyo3(get)]
    comparable_sites: usize,
}

#[pymethods]
impl PairwiseDifference {
    fn __repr__(&self) -> String {
        format!(
            "PairwiseDifference(sample_i={}, sample_j={}, differences={}, comparable_sites={})",
            self.sample_i, self.sample_j, self.differences, self.comparable_sites
        )
    }
}

/// Principal component coordinates for a single chromosome.
#[pyclass(module = "ferromic")]
#[derive(Clone)]
struct ChromosomePcaResult {
    #[pyo3(get)]
    haplotype_labels: Vec<String>,
    #[pyo3(get)]
    coordinates: Vec<Vec<f64>>,
    #[pyo3(get)]
    positions: Vec<i64>,
}

#[pymethods]
impl ChromosomePcaResult {
    fn __repr__(&self) -> String {
        format!(
            "ChromosomePcaResult(haplotypes={}, components={}, variants={})",
            self.haplotype_labels.len(),
            self.coordinates.first().map(|row| row.len()).unwrap_or(0),
            self.positions.len()
        )
    }
}

impl ChromosomePcaResult {
    fn from_result(py: Python, result: &PcaResult) -> PyResult<Py<Self>> {
        let coordinates = result
            .pca_coordinates
            .outer_iter()
            .map(|row| row.to_vec())
            .collect();

        Py::new(
            py,
            ChromosomePcaResult {
                haplotype_labels: result.haplotype_labels.clone(),
                coordinates,
                positions: result.positions.clone(),
            },
        )
    }
}

/// Per-site diversity summary containing π and Watterson's θ.
#[pyclass(module = "ferromic")]
#[derive(Clone)]
struct DiversitySite {
    #[pyo3(get)]
    position: i64,
    #[pyo3(get)]
    pi: f64,
    #[pyo3(get)]
    watterson_theta: f64,
}

#[pymethods]
impl DiversitySite {
    fn __repr__(&self) -> String {
        format!(
            "DiversitySite(position={}, pi={:.6}, watterson_theta={:.6})",
            self.position, self.pi, self.watterson_theta
        )
    }
}

/// Result of Hudson's Dxy statistic, exposed as ``ferromic.HudsonDxyResult``.
#[pyclass(module = "ferromic", name = "HudsonDxyResult")]
#[derive(Clone)]
struct HudsonDxyResultPy {
    #[pyo3(get)]
    d_xy: Option<f64>,
}

#[pymethods]
impl HudsonDxyResultPy {
    fn __repr__(&self) -> String {
        match self.d_xy {
            Some(value) => format!("HudsonDxyResult(d_xy={value:.6})"),
            None => "HudsonDxyResult(d_xy=None)".to_string(),
        }
    }
}

impl HudsonDxyResultPy {
    fn from_result(py: Python, result: &DxyHudsonResult) -> PyResult<Py<Self>> {
        Py::new(py, HudsonDxyResultPy { d_xy: result.d_xy })
    }
}

/// Per-site Hudson FST values and supporting components, exposed as ``ferromic.HudsonFstSite``.
#[pyclass(module = "ferromic", name = "HudsonFstSite")]
#[derive(Clone)]
struct HudsonFstSitePy {
    #[pyo3(get)]
    position: i64,
    #[pyo3(get)]
    fst: Option<f64>,
    #[pyo3(get)]
    d_xy: Option<f64>,
    #[pyo3(get)]
    pi_pop1: Option<f64>,
    #[pyo3(get)]
    pi_pop2: Option<f64>,
    #[pyo3(get)]
    n1_called: usize,
    #[pyo3(get)]
    n2_called: usize,
    #[pyo3(get)]
    numerator_component: Option<f64>,
    #[pyo3(get)]
    denominator_component: Option<f64>,
}

#[pymethods]
impl HudsonFstSitePy {
    fn __repr__(&self) -> String {
        format!(
            "HudsonFstSite(position={}, fst={}, d_xy={}, pi_pop1={}, pi_pop2={}, n1_called={}, n2_called={})",
            self.position,
            optional_float_display(self.fst),
            optional_float_display(self.d_xy),
            optional_float_display(self.pi_pop1),
            optional_float_display(self.pi_pop2),
            self.n1_called,
            self.n2_called,
        )
    }
}

impl HudsonFstSitePy {
    fn from_site(py: Python, site: &SiteFstHudson) -> PyResult<Py<Self>> {
        Py::new(
            py,
            HudsonFstSitePy {
                position: site.position,
                fst: site.fst,
                d_xy: site.d_xy,
                pi_pop1: site.pi_pop1,
                pi_pop2: site.pi_pop2,
                n1_called: site.n1_called,
                n2_called: site.n2_called,
                numerator_component: site.num_component,
                denominator_component: site.den_component,
            },
        )
    }
}

/// Aggregated Hudson FST result with friendly labels, exposed as ``ferromic.HudsonFstResult``.
#[pyclass(module = "ferromic", name = "HudsonFstResult")]
#[derive(Clone)]
struct HudsonFstResultPy {
    #[pyo3(get)]
    fst: Option<f64>,
    #[pyo3(get)]
    d_xy: Option<f64>,
    #[pyo3(get)]
    pi_pop1: Option<f64>,
    #[pyo3(get)]
    pi_pop2: Option<f64>,
    #[pyo3(get)]
    pi_xy_avg: Option<f64>,
    #[pyo3(get)]
    population1_label: Option<String>,
    #[pyo3(get)]
    population1_haplotype_group: Option<u8>,
    #[pyo3(get)]
    population2_label: Option<String>,
    #[pyo3(get)]
    population2_haplotype_group: Option<u8>,
}

#[pymethods]
impl HudsonFstResultPy {
    fn __repr__(&self) -> String {
        format!(
            "HudsonFstResult(fst={}, d_xy={}, pi_pop1={}, pi_pop2={}, pi_xy_avg={}, pop1={}, pop2={})",
            optional_float_display(self.fst),
            optional_float_display(self.d_xy),
            optional_float_display(self.pi_pop1),
            optional_float_display(self.pi_pop2),
            optional_float_display(self.pi_xy_avg),
            self.population1_label
                .clone()
                .unwrap_or_else(|| "<unknown>".to_string()),
            self.population2_label
                .clone()
                .unwrap_or_else(|| "<unknown>".to_string())
        )
    }
}

impl HudsonFstResultPy {
    fn from_outcome(py: Python, outcome: &HudsonFSTOutcome) -> PyResult<Py<Self>> {
        let (pop1_label, pop1_group) = outcome
            .pop1_id
            .as_ref()
            .map(population_label)
            .unwrap_or((None, None));
        let (pop2_label, pop2_group) = outcome
            .pop2_id
            .as_ref()
            .map(population_label)
            .unwrap_or((None, None));

        Py::new(
            py,
            HudsonFstResultPy {
                fst: outcome.fst,
                d_xy: outcome.d_xy,
                pi_pop1: outcome.pi_pop1,
                pi_pop2: outcome.pi_pop2,
                pi_xy_avg: outcome.pi_xy_avg,
                population1_label: pop1_label,
                population1_haplotype_group: pop1_group,
                population2_label: pop2_label,
                population2_haplotype_group: pop2_group,
            },
        )
    }
}

/// Per-site Weir & Cockerham FST summary, exposed as ``ferromic.WcFstSite``.
#[pyclass(module = "ferromic", name = "WcFstSite")]
#[derive(Clone)]
struct WcFstSitePy {
    #[pyo3(get)]
    position: i64,
    #[pyo3(get)]
    overall_fst: Py<FstEstimateInfo>,
    #[pyo3(get)]
    pairwise_fst: HashMap<String, Py<FstEstimateInfo>>,
    #[pyo3(get)]
    variance_components_a: f64,
    #[pyo3(get)]
    variance_components_b: f64,
    #[pyo3(get)]
    population_sizes: HashMap<String, usize>,
    #[pyo3(get)]
    pairwise_variance_components: HashMap<String, (f64, f64)>,
}

#[pymethods]
impl WcFstSitePy {
    fn variance_components(&self) -> (f64, f64) {
        (self.variance_components_a, self.variance_components_b)
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        let overall = self.overall_fst.borrow(py);
        Ok(format!(
            "WcFstSite(position={}, overall_fst={})",
            self.position,
            overall.__repr__()
        ))
    }
}

impl WcFstSitePy {
    fn from_site(py: Python, site: &SiteFstWc) -> PyResult<Py<Self>> {
        let mut pairwise = HashMap::with_capacity(site.pairwise_fst.len());
        for (key, value) in &site.pairwise_fst {
            pairwise.insert(key.clone(), FstEstimateInfo::from_estimate(py, value)?);
        }
        let mut pairwise_components =
            HashMap::with_capacity(site.pairwise_variance_components.len());
        for (key, value) in &site.pairwise_variance_components {
            pairwise_components.insert(key.clone(), *value);
        }
        Py::new(
            py,
            WcFstSitePy {
                position: site.position,
                overall_fst: FstEstimateInfo::from_estimate(py, &site.overall_fst)?,
                pairwise_fst: pairwise,
                variance_components_a: site.variance_components.0,
                variance_components_b: site.variance_components.1,
                population_sizes: site.population_sizes.clone(),
                pairwise_variance_components: pairwise_components,
            },
        )
    }
}

/// Aggregated Weir & Cockerham FST result, exposed as ``ferromic.WcFstResult``.
#[pyclass(module = "ferromic", name = "WcFstResult")]
#[derive(Clone)]
struct WcFstResultPy {
    #[pyo3(get)]
    overall_fst: Py<FstEstimateInfo>,
    #[pyo3(get)]
    pairwise_fst: HashMap<String, Py<FstEstimateInfo>>,
    #[pyo3(get)]
    pairwise_variance_components: HashMap<String, (f64, f64)>,
    #[pyo3(get)]
    site_fst: Vec<Py<WcFstSitePy>>,
    #[pyo3(get)]
    fst_type: String,
}

#[pymethods]
impl WcFstResultPy {
    fn __repr__(&self, py: Python) -> PyResult<String> {
        let overall = self.overall_fst.borrow(py);
        Ok(format!("WcFstResult(overall_fst={})", overall.__repr__()))
    }
}

impl WcFstResultPy {
    fn from_results(py: Python, results: &FstWcResults) -> PyResult<Py<Self>> {
        let mut pairwise = HashMap::with_capacity(results.pairwise_fst.len());
        for (key, value) in &results.pairwise_fst {
            pairwise.insert(key.clone(), FstEstimateInfo::from_estimate(py, value)?);
        }
        let mut sites = Vec::with_capacity(results.site_fst.len());
        for site in &results.site_fst {
            sites.push(WcFstSitePy::from_site(py, site)?);
        }
        Py::new(
            py,
            WcFstResultPy {
                overall_fst: FstEstimateInfo::from_estimate(py, &results.overall_fst)?,
                pairwise_fst: pairwise,
                pairwise_variance_components: results.pairwise_variance_components.clone(),
                site_fst: sites,
                fst_type: results.fst_type.clone(),
            },
        )
    }
}

/// In-memory representation of a population to be reused across statistics calls.
#[pyclass(module = "ferromic")]
#[derive(Clone)]
struct Population {
    inner: OwnedPopulationContext,
}

#[pymethods]
impl Population {
    #[new]
    #[pyo3(signature = (id, variants, haplotypes, sequence_length, sample_names=None))]
    fn new(
        id: PopulationIdInput,
        variants: Vec<VariantInput>,
        haplotypes: Vec<HaplotypeInput>,
        sequence_length: i64,
        sample_names: Option<Vec<String>>,
    ) -> PyResult<Self> {
        if sequence_length <= 0 {
            return Err(PyValueError::new_err(
                "sequence_length must be a positive integer",
            ));
        }

        Ok(Population {
            inner: OwnedPopulationContext {
                id: id.0,
                haplotypes: haplotypes.into_iter().map(|h| h.into_pair()).collect(),
                variants: variants.into_iter().map(VariantInput::into_inner).collect(),
                sample_names: sample_names.unwrap_or_default(),
                sequence_length,
            },
        })
    }

    /// Identifier for the population. Haplotype groups are returned as integers, custom
    /// labels as strings.
    #[getter]
    fn id(&self) -> PyObject {
        Python::with_gil(|py| match &self.inner.id {
            PopulationId::HaplotypeGroup(group) => (*group).into_py(py),
            PopulationId::Named(name) => name.clone().into_py(py),
        })
    }

    /// When the identifier is a haplotype group, this returns its numeric value.
    #[getter]
    fn haplotype_group(&self) -> Option<u8> {
        match &self.inner.id {
            PopulationId::HaplotypeGroup(group) => Some(*group),
            PopulationId::Named(_) => None,
        }
    }

    /// Optional descriptive label for named populations.
    #[getter]
    fn label(&self) -> Option<String> {
        match &self.inner.id {
            PopulationId::HaplotypeGroup(_) => None,
            PopulationId::Named(name) => Some(name.clone()),
        }
    }

    /// Sequence length used when normalising statistics.
    #[getter]
    fn sequence_length(&self) -> i64 {
        self.inner.sequence_length
    }

    /// Number of stored variants.
    #[getter]
    fn variant_count(&self) -> usize {
        self.inner.variants.len()
    }

    /// Names of the samples backing this population.
    #[getter]
    fn sample_names(&self) -> Vec<String> {
        self.inner.sample_names.clone()
    }

    /// Haplotypes represented as ``(sample_index, side)`` where side is 0 for left and 1 for right.
    #[getter]
    fn haplotypes(&self) -> Vec<(usize, u8)> {
        self.inner
            .haplotypes
            .iter()
            .map(|(idx, side)| {
                (
                    *idx,
                    match side {
                        HaplotypeSide::Left => 0,
                        HaplotypeSide::Right => 1,
                    },
                )
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        let label = match &self.inner.id {
            PopulationId::HaplotypeGroup(group) => format!("haplotype_group {group}"),
            PopulationId::Named(name) => format!("named '{name}'"),
        };
        format!(
            "Population({label}, haplotypes={}, variants={}, sequence_length={})",
            self.inner.haplotypes.len(),
            self.inner.variants.len(),
            self.inner.sequence_length
        )
    }
}

/// Internal owned representation of a population used to build [`PopulationContext`].
#[derive(Clone)]
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

/// Simple helper that formats optional floats for ``__repr__`` implementations.
fn optional_float_display(value: Option<f64>) -> String {
    match value {
        Some(v) if v.is_finite() => format!("{v:.6}"),
        Some(v) => v.to_string(),
        None => "None".to_string(),
    }
}

#[derive(Clone)]
struct VariantInput(Variant);

impl VariantInput {
    fn into_inner(self) -> Variant {
        self.0
    }
}

impl<'source> FromPyObject<'source> for VariantInput {
    fn extract(obj: &'source PyAny) -> PyResult<Self> {
        if let Ok(tuple) = obj.downcast::<PyTuple>() {
            if tuple.len() != 2 {
                return Err(PyValueError::new_err(
                    "variant tuples must have length 2: (position, genotypes)",
                ));
            }
            let position = tuple.get_item(0)?.extract::<i64>()?;
            let genotypes = parse_genotypes(tuple.get_item(1)?)?;
            return Ok(VariantInput(Variant {
                position,
                genotypes,
            }));
        }

        if let Ok(mapping) = obj.downcast::<PyDict>() {
            let position =
                extract_from_mapping(mapping, &["position", "pos", "site"])?.extract::<i64>()?;
            let genotypes =
                parse_genotypes(extract_from_mapping(mapping, &["genotypes", "calls"])?)?;
            return Ok(VariantInput(Variant {
                position,
                genotypes,
            }));
        }

        let position = extract_optional_field(obj, &["position", "pos", "site"])
            .ok_or_else(|| PyValueError::new_err("variant is missing a position"))?
            .extract::<i64>()?;
        let genotypes_obj = extract_optional_field(obj, &["genotypes", "calls"])
            .ok_or_else(|| PyValueError::new_err("variant is missing genotypes"))?;
        let genotypes = parse_genotypes(genotypes_obj)?;

        Ok(VariantInput(Variant {
            position,
            genotypes,
        }))
    }
}

#[derive(Clone)]
struct HaplotypeInput {
    sample_index: usize,
    side: HaplotypeSide,
}

impl HaplotypeInput {
    fn into_pair(self) -> (usize, HaplotypeSide) {
        (self.sample_index, self.side)
    }
}

impl<'source> FromPyObject<'source> for HaplotypeInput {
    fn extract(obj: &'source PyAny) -> PyResult<Self> {
        if let Ok(tuple) = obj.downcast::<PyTuple>() {
            if tuple.len() < 2 {
                return Err(PyValueError::new_err(
                    "haplotypes must contain (sample_index, side)",
                ));
            }
            return Ok(HaplotypeInput {
                sample_index: tuple.get_item(0)?.extract::<usize>()?,
                side: parse_side(tuple.get_item(1)?)?,
            });
        }

        if let Ok(list) = obj.downcast::<PyList>() {
            if list.len() < 2 {
                return Err(PyValueError::new_err(
                    "haplotypes must contain (sample_index, side)",
                ));
            }
            return Ok(HaplotypeInput {
                sample_index: list.get_item(0)?.extract::<usize>()?,
                side: parse_side(list.get_item(1)?)?,
            });
        }

        let index_obj = extract_optional_field(obj, &["sample_index", "sample", "index"])
            .ok_or_else(|| PyValueError::new_err("haplotype missing sample index"))?;
        let side_obj = extract_optional_field(obj, &["side", "haplotype", "haplotype_side"])
            .ok_or_else(|| PyValueError::new_err("haplotype missing side"))?;

        Ok(HaplotypeInput {
            sample_index: index_obj.extract::<usize>()?,
            side: parse_side(side_obj)?,
        })
    }
}

#[derive(Clone)]
struct PopulationIdInput(PopulationId);

impl<'source> FromPyObject<'source> for PopulationIdInput {
    fn extract(obj: &'source PyAny) -> PyResult<Self> {
        if let Ok(dict) = obj.downcast::<PyDict>() {
            if let Some(value) = dict.get_item("haplotype_group") {
                return Ok(PopulationIdInput(PopulationId::HaplotypeGroup(
                    value.extract::<u8>()?,
                )));
            }
            if let Some(value) = dict.get_item("named") {
                return Ok(PopulationIdInput(PopulationId::Named(
                    value.extract::<String>()?,
                )));
            }
            return Err(PyValueError::new_err(
                "population id dictionaries must provide 'haplotype_group' or 'named'",
            ));
        }

        if let Ok(group) = obj.extract::<u8>() {
            return Ok(PopulationIdInput(PopulationId::HaplotypeGroup(group)));
        }

        if let Ok(group) = obj.extract::<usize>() {
            if group > u8::MAX as usize {
                return Err(PyValueError::new_err("haplotype_group ids must be <= 255"));
            }
            return Ok(PopulationIdInput(PopulationId::HaplotypeGroup(group as u8)));
        }

        if let Ok(name) = obj.extract::<String>() {
            return Ok(PopulationIdInput(PopulationId::Named(name)));
        }

        Err(PyValueError::new_err(
            "could not interpret population id; pass an int, string, or mapping",
        ))
    }
}

#[derive(Clone)]
struct PopulationInput {
    inner: OwnedPopulationContext,
}

impl PopulationInput {
    fn as_context(&self) -> PopulationContext<'_> {
        self.inner.as_population_context()
    }
}

impl<'source> FromPyObject<'source> for PopulationInput {
    fn extract(obj: &'source PyAny) -> PyResult<Self> {
        if let Ok(pop) = obj.extract::<PyRef<Population>>() {
            return Ok(PopulationInput {
                inner: pop.inner.clone(),
            });
        }

        if let Ok(dict) = obj.downcast::<PyDict>() {
            return parse_population_like_dict(dict);
        }

        parse_population_like_object(obj)
    }
}

fn parse_population_like_dict(mapping: &PyDict) -> PyResult<PopulationInput> {
    let id = extract_from_mapping(mapping, &["id", "population_id", "name"])?
        .extract::<PopulationIdInput>()?
        .0;
    let variants: Vec<Variant> = mapping
        .get_item("variants")
        .ok_or_else(|| PyValueError::new_err("population requires 'variants'"))?
        .extract::<Vec<VariantInput>>()?
        .into_iter()
        .map(VariantInput::into_inner)
        .collect();
    let haplotypes: Vec<(usize, HaplotypeSide)> = mapping
        .get_item("haplotypes")
        .ok_or_else(|| PyValueError::new_err("population requires 'haplotypes'"))?
        .extract::<Vec<HaplotypeInput>>()?
        .into_iter()
        .map(|h| h.into_pair())
        .collect();
    let sequence_length =
        extract_from_mapping(mapping, &["sequence_length", "length", "L"])?.extract::<i64>()?;
    if sequence_length <= 0 {
        return Err(PyValueError::new_err(
            "sequence_length must be a positive integer",
        ));
    }
    let sample_names = match mapping.get_item("sample_names") {
        Some(value) => value.extract::<Vec<String>>()?,
        None => Vec::new(),
    };

    Ok(PopulationInput {
        inner: OwnedPopulationContext {
            id,
            haplotypes,
            variants,
            sample_names,
            sequence_length,
        },
    })
}

fn parse_population_like_object(obj: &PyAny) -> PyResult<PopulationInput> {
    let id = extract_optional_field(obj, &["id", "population_id", "name"])
        .ok_or_else(|| PyValueError::new_err("population-like object missing 'id'"))?
        .extract::<PopulationIdInput>()?
        .0;
    let variants: Vec<Variant> = extract_optional_field(obj, &["variants"])
        .ok_or_else(|| PyValueError::new_err("population-like object missing 'variants'"))?
        .extract::<Vec<VariantInput>>()?
        .into_iter()
        .map(VariantInput::into_inner)
        .collect();
    let haplotypes: Vec<(usize, HaplotypeSide)> = extract_optional_field(obj, &["haplotypes"])
        .ok_or_else(|| PyValueError::new_err("population-like object missing 'haplotypes'"))?
        .extract::<Vec<HaplotypeInput>>()?
        .into_iter()
        .map(|h| h.into_pair())
        .collect();
    let sequence_length = extract_optional_field(obj, &["sequence_length", "length", "L"])
        .ok_or_else(|| PyValueError::new_err("population-like object missing 'sequence_length'"))?
        .extract::<i64>()?;
    if sequence_length <= 0 {
        return Err(PyValueError::new_err(
            "sequence_length must be a positive integer",
        ));
    }
    let sample_names = extract_optional_field(obj, &["sample_names", "samples"])
        .map(|value| value.extract::<Vec<String>>())
        .transpose()?
        .unwrap_or_default();

    Ok(PopulationInput {
        inner: OwnedPopulationContext {
            id,
            haplotypes,
            variants,
            sample_names,
            sequence_length,
        },
    })
}

fn parse_genotypes(genotypes_obj: &PyAny) -> PyResult<Vec<Option<Vec<u8>>>> {
    let mut genotypes = Vec::new();
    let iterator = PyIterator::from_object(genotypes_obj.py(), genotypes_obj)?;
    for entry in iterator {
        let entry = entry?;
        if entry.is_none() {
            genotypes.push(None);
            continue;
        }

        if let Ok(value) = entry.extract::<u8>() {
            genotypes.push(Some(vec![value]));
            continue;
        }

        if let Ok(seq) = PyIterator::from_object(entry.py(), entry) {
            let mut alleles = Vec::new();
            for allele in seq {
                alleles.push(allele?.extract::<u8>()?);
            }
            genotypes.push(Some(alleles));
            continue;
        }

        return Err(PyValueError::new_err(
            "genotypes must be sequences of allele integers or None",
        ));
    }
    Ok(genotypes)
}

fn parse_side(obj: &PyAny) -> PyResult<HaplotypeSide> {
    if let Ok(value) = obj.extract::<u8>() {
        return match value {
            0 => Ok(HaplotypeSide::Left),
            1 => Ok(HaplotypeSide::Right),
            _ => Err(PyValueError::new_err("haplotype side must be 0 or 1")),
        };
    }

    if let Ok(value) = obj.extract::<usize>() {
        return match value {
            0 => Ok(HaplotypeSide::Left),
            1 => Ok(HaplotypeSide::Right),
            _ => Err(PyValueError::new_err("haplotype side must be 0 or 1")),
        };
    }

    if let Ok(text) = obj.extract::<String>() {
        let lower = text.to_lowercase();
        return match lower.as_str() {
            "l" | "left" => Ok(HaplotypeSide::Left),
            "r" | "right" => Ok(HaplotypeSide::Right),
            "0" => Ok(HaplotypeSide::Left),
            "1" => Ok(HaplotypeSide::Right),
            _ => Err(PyValueError::new_err(
                "haplotype side must be one of 0, 1, 'L', 'R', 'left', 'right'",
            )),
        };
    }

    Err(PyValueError::new_err(
        "haplotype side must be 0/1 or a left/right string",
    ))
}

fn extract_optional_field<'a>(obj: &'a PyAny, names: &[&str]) -> Option<&'a PyAny> {
    for name in names {
        if let Ok(value) = obj.get_item(*name) {
            return Some(value);
        }
        if let Ok(value) = obj.getattr(*name) {
            return Some(value);
        }
    }
    None
}

fn extract_from_mapping<'a>(mapping: &'a PyDict, names: &[&str]) -> PyResult<&'a PyAny> {
    for name in names {
        if let Some(value) = mapping.get_item(*name) {
            return Ok(value);
        }
    }
    Err(PyValueError::new_err(format!(
        "mapping missing required field: {}",
        names.join(" / ")
    )))
}

fn population_label(id: &PopulationId) -> (Option<String>, Option<u8>) {
    match id {
        PopulationId::HaplotypeGroup(group) => {
            (Some(format!("haplotype_group_{group}")), Some(*group))
        }
        PopulationId::Named(name) => (Some(name.clone()), None),
    }
}

fn build_region(region: (i64, i64)) -> PyResult<QueryRegion> {
    let (start, end) = region;
    if end < start {
        return Err(PyValueError::new_err(
            "region end must be greater than or equal to region start",
        ));
    }
    Ok(QueryRegion { start, end })
}

fn build_optional_region(
    region: Option<(i64, i64)>,
    variants: &[Variant],
) -> PyResult<QueryRegion> {
    if let Some(region) = region {
        return build_region(region);
    }

    if variants.is_empty() {
        return Err(PyValueError::new_err(
            "region must be provided when no variants are supplied",
        ));
    }

    let mut min_pos = variants[0].position;
    let mut max_pos = variants[0].position;
    for variant in &variants[1..] {
        if variant.position < min_pos {
            min_pos = variant.position;
        }
        if variant.position > max_pos {
            max_pos = variant.position;
        }
    }

    Ok(QueryRegion {
        start: min_pos,
        end: max_pos,
    })
}

fn hudson_sites_to_py(py: Python, sites: &[SiteFstHudson]) -> PyResult<Vec<Py<HudsonFstSitePy>>> {
    let mut out = Vec::with_capacity(sites.len());
    for site in sites {
        out.push(HudsonFstSitePy::from_site(py, site)?);
    }
    Ok(out)
}

fn diversity_sites_to_py(py: Python, sites: &[SiteDiversity]) -> PyResult<Vec<Py<DiversitySite>>> {
    let mut out = Vec::with_capacity(sites.len());
    for site in sites {
        out.push(Py::new(
            py,
            DiversitySite {
                position: site.position,
                pi: site.pi,
                watterson_theta: site.watterson_theta,
            },
        )?);
    }
    Ok(out)
}

fn pairwise_differences_to_py(
    py: Python,
    diffs: Vec<((usize, usize), usize, usize)>,
) -> PyResult<Vec<Py<PairwiseDifference>>> {
    let mut out = Vec::with_capacity(diffs.len());
    for ((sample_i, sample_j), differences, comparable_sites) in diffs {
        out.push(Py::new(
            py,
            PairwiseDifference {
                sample_i,
                sample_j,
                differences,
                comparable_sites,
            },
        )?);
    }
    Ok(out)
}

fn extract_sample_group_map(obj: &PyAny) -> PyResult<HashMap<String, (u8, u8)>> {
    let dict = obj.downcast::<PyDict>().map_err(|_| {
        PyValueError::new_err("sample_to_group must be a dict mapping sample -> (left, right)")
    })?;

    let mut map = HashMap::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let sample = key.extract::<String>()?;
        let left = value
            .get_item(0)
            .map_err(|_| PyValueError::new_err("group tuples must contain two entries"))?
            .extract::<u8>()?;
        let right = value
            .get_item(1)
            .map_err(|_| PyValueError::new_err("group tuples must contain two entries"))?
            .extract::<u8>()?;
        map.insert(sample, (left, right));
    }

    Ok(map)
}

fn extract_variants_by_chromosome(obj: &PyAny) -> PyResult<HashMap<String, Vec<Variant>>> {
    let dict = obj.downcast::<PyDict>().map_err(|_| {
        PyValueError::new_err(
            "variants_by_chromosome must be a dict mapping chromosome -> sequence of variants",
        )
    })?;

    let mut map = HashMap::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let chromosome = key.extract::<String>()?;
        let variants: Vec<VariantInput> = value.extract()?;
        map.insert(
            chromosome,
            variants.into_iter().map(VariantInput::into_inner).collect(),
        );
    }

    Ok(map)
}

fn extract_interval_list(obj: Option<&PyAny>) -> PyResult<Option<Vec<(i64, i64)>>> {
    let Some(obj) = obj else { return Ok(None) };
    let mut intervals = Vec::new();
    let iterator = PyIterator::from_object(obj.py(), obj)?;
    for entry in iterator {
        let entry = entry?;
        let start = entry
            .get_item(0)
            .map_err(|_| PyValueError::new_err("intervals must be (start, end)"))?
            .extract::<i64>()?;
        let end = entry
            .get_item(1)
            .map_err(|_| PyValueError::new_err("intervals must be (start, end)"))?
            .extract::<i64>()?;
        if end < start {
            return Err(PyValueError::new_err(
                "interval end must be greater than or equal to start",
            ));
        }
        intervals.push((start, end));
    }
    Ok(Some(intervals))
}

fn vcf_error_to_pyerr(err: VcfError) -> PyErr {
    PyValueError::new_err(format!("VCF error: {err:?}"))
}

/// Count the number of segregating (polymorphic) sites.
#[pyfunction(name = "segregating_sites", text_signature = "(variants, /)")]
fn segregating_sites_py(variants: Vec<VariantInput>) -> PyResult<usize> {
    let variants: Vec<Variant> = variants.into_iter().map(VariantInput::into_inner).collect();
    Ok(count_segregating_sites(&variants))
}

/// Compute nucleotide diversity (π) for the provided haplotypes and region length.
#[pyfunction(
    name = "nucleotide_diversity",
    text_signature = "(variants, haplotypes, sequence_length, /)"
)]
fn nucleotide_diversity_py(
    variants: Vec<VariantInput>,
    haplotypes: Vec<HaplotypeInput>,
    sequence_length: i64,
) -> PyResult<f64> {
    if sequence_length <= 0 {
        return Err(PyValueError::new_err(
            "sequence_length must be a positive integer",
        ));
    }

    let variants: Vec<Variant> = variants.into_iter().map(VariantInput::into_inner).collect();
    let haplotypes: Vec<(usize, HaplotypeSide)> =
        haplotypes.into_iter().map(|h| h.into_pair()).collect();

    Ok(calculate_pi(&variants, &haplotypes, sequence_length))
}

/// Compute Watterson's θ estimator.
#[pyfunction(
    name = "watterson_theta",
    text_signature = "(segregating_sites, sample_count, sequence_length, /)"
)]
fn watterson_theta_py(
    segregating_sites: usize,
    sample_count: usize,
    sequence_length: i64,
) -> PyResult<f64> {
    if sample_count <= 1 {
        return Err(PyValueError::new_err(
            "sample_count must be greater than 1 for Watterson's theta",
        ));
    }
    if sequence_length <= 0 {
        return Err(PyValueError::new_err(
            "sequence_length must be a positive integer",
        ));
    }
    Ok(calculate_watterson_theta(
        segregating_sites,
        sample_count,
        sequence_length,
    ))
}

/// Compute pairwise nucleotide differences between samples.
#[pyfunction(
    name = "pairwise_differences",
    text_signature = "(variants, sample_count, /)"
)]
fn pairwise_differences_py(
    py: Python,
    variants: Vec<VariantInput>,
    sample_count: usize,
) -> PyResult<Vec<Py<PairwiseDifference>>> {
    let variants: Vec<Variant> = variants.into_iter().map(VariantInput::into_inner).collect();
    let diffs = calculate_pairwise_differences(&variants, sample_count);
    pairwise_differences_to_py(py, diffs)
}

/// Calculate per-site diversity statistics (π and Watterson's θ).
#[pyfunction(
    name = "per_site_diversity",
    signature = (variants, haplotypes, region=None),
    text_signature = "(variants, haplotypes, region=None, /)"
)]
fn per_site_diversity_py(
    py: Python,
    variants: Vec<VariantInput>,
    haplotypes: Vec<HaplotypeInput>,
    region: Option<(i64, i64)>,
) -> PyResult<Vec<Py<DiversitySite>>> {
    let variants: Vec<Variant> = variants.into_iter().map(VariantInput::into_inner).collect();
    let haplotypes: Vec<(usize, HaplotypeSide)> =
        haplotypes.into_iter().map(|h| h.into_pair()).collect();
    if haplotypes.len() < 2 {
        return Err(PyValueError::new_err(
            "at least two haplotypes are required for diversity calculations",
        ));
    }

    let region = build_optional_region(region, &variants)?;
    let sites = calculate_per_site_diversity(&variants, &haplotypes, region);
    diversity_sites_to_py(py, &sites)
}

/// Compute Hudson's Dxy between two populations.
#[pyfunction(name = "hudson_dxy", text_signature = "(population1, population2, /)")]
fn hudson_dxy_py(
    py: Python,
    population1: PopulationInput,
    population2: PopulationInput,
) -> PyResult<Py<HudsonDxyResultPy>> {
    let pop1_ctx = population1.as_context();
    let pop2_ctx = population2.as_context();
    let result = calculate_d_xy_hudson(&pop1_ctx, &pop2_ctx).map_err(vcf_error_to_pyerr)?;
    HudsonDxyResultPy::from_result(py, &result)
}

/// Compute Hudson's FST and its components for two populations.
#[pyfunction(name = "hudson_fst", text_signature = "(population1, population2, /)")]
fn hudson_fst_py(
    py: Python,
    population1: PopulationInput,
    population2: PopulationInput,
) -> PyResult<Py<HudsonFstResultPy>> {
    let pop1_ctx = population1.as_context();
    let pop2_ctx = population2.as_context();
    let outcome =
        calculate_hudson_fst_for_pair(&pop1_ctx, &pop2_ctx).map_err(vcf_error_to_pyerr)?;
    HudsonFstResultPy::from_outcome(py, &outcome)
}

/// Compute per-site Hudson FST values across a region.
#[pyfunction(
    name = "hudson_fst_sites",
    text_signature = "(population1, population2, region, /)"
)]
fn hudson_fst_sites_py(
    py: Python,
    population1: PopulationInput,
    population2: PopulationInput,
    region: (i64, i64),
) -> PyResult<Vec<Py<HudsonFstSitePy>>> {
    let pop1_ctx = population1.as_context();
    let pop2_ctx = population2.as_context();
    let region = build_region(region)?;
    let sites = calculate_hudson_fst_per_site(&pop1_ctx, &pop2_ctx, region);
    hudson_sites_to_py(py, &sites)
}

/// Compute Hudson's FST together with per-site contributions.
#[pyfunction(
    name = "hudson_fst_with_sites",
    text_signature = "(population1, population2, region, /)"
)]
fn hudson_fst_with_sites_py(
    py: Python,
    population1: PopulationInput,
    population2: PopulationInput,
    region: (i64, i64),
) -> PyResult<(Py<HudsonFstResultPy>, Vec<Py<HudsonFstSitePy>>)> {
    let pop1_ctx = population1.as_context();
    let pop2_ctx = population2.as_context();
    let region = build_region(region)?;
    let (outcome, sites) = calculate_hudson_fst_for_pair_with_sites(&pop1_ctx, &pop2_ctx, region)
        .map_err(vcf_error_to_pyerr)?;
    let outcome_py = HudsonFstResultPy::from_outcome(py, &outcome)?;
    let sites_py = hudson_sites_to_py(py, &sites)?;
    Ok((outcome_py, sites_py))
}

/// Compute Weir & Cockerham FST across haplotype groups.
#[pyfunction(
    name = "wc_fst",
    text_signature = "(variants, sample_names, sample_to_group, region, /)"
)]
fn wc_fst_py(
    py: Python,
    variants: Vec<VariantInput>,
    sample_names: Vec<String>,
    sample_to_group: &PyAny,
    region: (i64, i64),
) -> PyResult<Py<WcFstResultPy>> {
    if sample_names.is_empty() {
        return Err(PyValueError::new_err(
            "sample_names must contain at least one sample",
        ));
    }

    let variants: Vec<Variant> = variants.into_iter().map(VariantInput::into_inner).collect();
    let sample_group_map = extract_sample_group_map(sample_to_group)?;
    let region = build_region(region)?;
    let results =
        calculate_fst_wc_haplotype_groups(&variants, &sample_names, &sample_group_map, region);
    WcFstResultPy::from_results(py, &results)
}

/// Convenience helper to read the variance components from an ``FstEstimate`` object.
#[pyfunction(name = "wc_fst_components", text_signature = "(estimate, /)")]
fn wc_fst_components_py(
    estimate: PyRef<FstEstimateInfo>,
) -> (Option<f64>, Option<f64>, Option<f64>, Option<usize>) {
    estimate.components()
}

/// Compute principal components for variants on a single chromosome.
#[pyfunction(
    name = "chromosome_pca",
    signature = (variants, sample_names, n_components=10),
    text_signature = "(variants, sample_names, n_components=10, /)"
)]
fn chromosome_pca_py(
    py: Python,
    variants: Vec<VariantInput>,
    sample_names: Vec<String>,
    n_components: usize,
) -> PyResult<Py<ChromosomePcaResult>> {
    if sample_names.is_empty() {
        return Err(PyValueError::new_err(
            "sample_names must contain at least one sample",
        ));
    }
    if n_components == 0 {
        return Err(PyValueError::new_err(
            "n_components must be greater than or equal to 1",
        ));
    }

    let variants: Vec<Variant> = variants.into_iter().map(VariantInput::into_inner).collect();
    let result = compute_chromosome_pca(&variants, &sample_names, n_components)
        .map_err(vcf_error_to_pyerr)?;
    ChromosomePcaResult::from_result(py, &result)
}

/// Compute principal components for a chromosome and write them to a TSV file.
#[pyfunction(
    name = "chromosome_pca_to_file",
    signature = (variants, sample_names, chromosome, output_dir, n_components=10),
    text_signature = "(variants, sample_names, chromosome, output_dir, n_components=10, /)"
)]
fn chromosome_pca_to_file_py(
    variants: Vec<VariantInput>,
    sample_names: Vec<String>,
    chromosome: &str,
    output_dir: &str,
    n_components: usize,
) -> PyResult<()> {
    if sample_names.is_empty() {
        return Err(PyValueError::new_err(
            "sample_names must contain at least one sample",
        ));
    }
    if n_components == 0 {
        return Err(PyValueError::new_err(
            "n_components must be greater than or equal to 1",
        ));
    }

    let variants: Vec<Variant> = variants.into_iter().map(VariantInput::into_inner).collect();
    let result = compute_chromosome_pca(&variants, &sample_names, n_components)
        .map_err(vcf_error_to_pyerr)?;
    let output_dir = PathBuf::from(output_dir);
    write_chromosome_pca_to_file(&result, chromosome, output_dir.as_path())
        .map_err(vcf_error_to_pyerr)
}

/// Run per-chromosome PCA for a dictionary of chromosomes -> variants.
#[pyfunction(
    name = "per_chromosome_pca",
    signature = (variants_by_chromosome, sample_names, output_dir, n_components=10),
    text_signature = "(variants_by_chromosome, sample_names, output_dir, n_components=10, /)"
)]
fn per_chromosome_pca_py(
    variants_by_chromosome: &PyAny,
    sample_names: Vec<String>,
    output_dir: &str,
    n_components: usize,
) -> PyResult<()> {
    if sample_names.is_empty() {
        return Err(PyValueError::new_err(
            "sample_names must contain at least one sample",
        ));
    }
    if n_components == 0 {
        return Err(PyValueError::new_err(
            "n_components must be greater than or equal to 1",
        ));
    }

    let variants = extract_variants_by_chromosome(variants_by_chromosome)?;
    let output_dir = PathBuf::from(output_dir);
    run_chromosome_pca_analysis(&variants, &sample_names, output_dir.as_path(), n_components)
        .map_err(vcf_error_to_pyerr)
}

/// Execute the memory-efficient multi-chromosome PCA pipeline.
#[pyfunction(
    name = "global_pca",
    signature = (variants_by_chromosome, sample_names, output_dir, n_components=10),
    text_signature = "(variants_by_chromosome, sample_names, output_dir, n_components=10, /)"
)]
fn global_pca_py(
    variants_by_chromosome: &PyAny,
    sample_names: Vec<String>,
    output_dir: &str,
    n_components: usize,
) -> PyResult<()> {
    if sample_names.is_empty() {
        return Err(PyValueError::new_err(
            "sample_names must contain at least one sample",
        ));
    }
    if n_components == 0 {
        return Err(PyValueError::new_err(
            "n_components must be greater than or equal to 1",
        ));
    }

    let variants = extract_variants_by_chromosome(variants_by_chromosome)?;
    let output_dir = PathBuf::from(output_dir);
    run_global_pca_analysis(&variants, &sample_names, output_dir.as_path(), n_components)
        .map_err(vcf_error_to_pyerr)
}

/// Adjust the effective sequence length by applying allow and mask intervals.
#[pyfunction(
    name = "adjusted_sequence_length",
    signature = (start, end, allow=None, mask=None),
    text_signature = "(start, end, allow=None, mask=None, /)"
)]
fn adjusted_sequence_length_py(
    start: i64,
    end: i64,
    allow: Option<&PyAny>,
    mask: Option<&PyAny>,
) -> PyResult<i64> {
    if end < start {
        return Err(PyValueError::new_err(
            "end must be greater than or equal to start",
        ));
    }
    let allow = extract_interval_list(allow)?;
    let mask = extract_interval_list(mask)?;
    Ok(calculate_adjusted_sequence_length(
        start,
        end,
        allow.as_ref(),
        mask.as_ref(),
    ))
}

/// Calculate the frequency of allele 1 (e.g. inversion allele) across haplotypes.
#[pyfunction(
    name = "inversion_allele_frequency",
    text_signature = "(sample_map, /)"
)]
fn inversion_allele_frequency_py(sample_map: &PyAny) -> PyResult<Option<f64>> {
    let map = extract_sample_group_map(sample_map)?;
    Ok(calculate_inversion_allele_frequency(&map))
}

#[pymodule]
fn ferromic(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Population>()?;
    m.add_class::<PairwiseDifference>()?;
    m.add_class::<ChromosomePcaResult>()?;
    m.add_class::<DiversitySite>()?;
    m.add_class::<HudsonDxyResultPy>()?;
    m.add_class::<HudsonFstSitePy>()?;
    m.add_class::<HudsonFstResultPy>()?;
    m.add_class::<FstEstimateInfo>()?;
    m.add_class::<WcFstSitePy>()?;
    m.add_class::<WcFstResultPy>()?;

    m.add_function(wrap_pyfunction!(segregating_sites_py, m)?)?;
    m.add_function(wrap_pyfunction!(nucleotide_diversity_py, m)?)?;
    m.add_function(wrap_pyfunction!(watterson_theta_py, m)?)?;
    m.add_function(wrap_pyfunction!(pairwise_differences_py, m)?)?;
    m.add_function(wrap_pyfunction!(per_site_diversity_py, m)?)?;
    m.add_function(wrap_pyfunction!(hudson_dxy_py, m)?)?;
    m.add_function(wrap_pyfunction!(hudson_fst_py, m)?)?;
    m.add_function(wrap_pyfunction!(hudson_fst_sites_py, m)?)?;
    m.add_function(wrap_pyfunction!(hudson_fst_with_sites_py, m)?)?;
    m.add_function(wrap_pyfunction!(wc_fst_py, m)?)?;
    m.add_function(wrap_pyfunction!(wc_fst_components_py, m)?)?;
    m.add_function(wrap_pyfunction!(chromosome_pca_py, m)?)?;
    m.add_function(wrap_pyfunction!(chromosome_pca_to_file_py, m)?)?;
    m.add_function(wrap_pyfunction!(per_chromosome_pca_py, m)?)?;
    m.add_function(wrap_pyfunction!(global_pca_py, m)?)?;
    m.add_function(wrap_pyfunction!(adjusted_sequence_length_py, m)?)?;
    m.add_function(wrap_pyfunction!(inversion_allele_frequency_py, m)?)?;

    Ok(())
}
