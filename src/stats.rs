use crate::process::{
    HaplotypeSide, QueryRegion, Variant, VcfError, ZeroBasedHalfOpen, ZeroBasedPosition,
};
use crate::progress::{
    create_spinner, display_status_box, finish_step_progress, init_step_progress, log, set_stage,
    update_step_progress, LogLevel, ProcessingStage, StatusBox,
};
use rayon::iter::IntoParallelIterator;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

/// Epsilon threshold for numerical stability in FST calculations.
/// Used consistently across per-site and aggregation functions to handle
/// near-zero denominators and floating-point precision issues.
/// 
/// **Usage Guidelines:**
/// - FST_EPSILON (1e-12): For Hudson FST denominators and component sums
/// - 1e-9: For Weir-Cockerham calculations and general float comparisons
/// - The choice depends on the expected magnitude of values and required precision
const FST_EPSILON: f64 = 1e-12;

/// Encapsulates the result of an FST (Fixation Index) calculation for a specific genetic site or genomic region.
/// FST is a measure of population differentiation, reflecting how much of the total genetic variation
/// is structured among different populations. It is derived from variance components:
/// 'a', the estimated genetic variance among subpopulations, and 'b', the estimated genetic variance
/// among haplotypes (or individuals) within these subpopulations. For analyses based on
/// haplotype data, such as this implementation, a third component 'c' (related to heterozygosity
/// within diploid individuals) is effectively zero. The FST estimate, often denoted as θ (theta),
/// is then calculated as the ratio a / (a + b).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FstEstimate {
    /// The FST calculation yielded a numerically definable value, representing the estimated
    /// degree of genetic differentiation. This is the typical outcome when variance components
    /// `sum_a` and `sum_b` allow for a meaningful ratio.
    Calculable {
        /// The FST value itself. An FST value near 0 indicates little genetic differentiation,
        /// meaning the populations are genetically quite similar at the loci analyzed. Conversely,
        /// a value near 1 signifies substantial genetic differentiation, where populations
        /// have very different allele frequencies, potentially with different alleles nearing fixation.
        /// Due to sampling effects in real data, especially with low true differentiation or small
        /// sample sizes, this estimated `value` can sometimes be negative (FST ≈ 0)
        /// or, rarely, exceed 1. If `sum_a` is non-zero while `sum_a + sum_b` is zero (implying all
        /// quantifiable variance is between populations), this value can be `f64::INFINITY` or
        /// `f64::NEG_INFINITY`.
        value: f64,

        /// The sum of 'a' components (among-population variance) from all genetic sites
        /// that contributed to this FST estimate. For a per-site estimate, this is simply the 'a'
        /// value for that site. It reflects the magnitude of genetic divergence attributable to
        /// systematic differences in allele frequencies between the sampled populations.
        sum_a: f64,

        /// The sum of 'b' components (within-population variance) from all genetic sites
        /// that contributed to this FST estimate. For a per-site estimate, this is the 'b' value
        /// for that site. It reflects the magnitude of genetic diversity existing within
        /// the individual populations being compared.
        sum_b: f64,

        /// The number of distinct genetic sites (e.g., SNPs) that provided valid, non-missing
        /// variance components (`a_i`, `b_i`) which were subsequently summed to produce `sum_a` and `sum_b`.
        /// For a single-site FST estimate, this count will be 1 if the site was informative.
        /// A larger number of informative sites generally lends greater robustness to a regional FST estimate.
        num_informative_sites: usize,
    },

    /// The FST estimate is indeterminate because the estimated total variance (sum_a + sum_b)
    /// is negative. This makes the standard FST ratio a/(a+b) problematic for interpretation
    /// as a simple proportion of variance. Such outcomes often arise from statistical sampling
    /// effects, particularly when true population differentiation is minimal or sample sizes are
    /// limited, leading to unstable (and potentially negative) estimates of the variance components.
    /// This state is distinct from a complete absence of genetic variation (see `NoInterPopulationVariance`).
    ComponentsYieldIndeterminateRatio {
        /// The sum of the 'a' components (among-population variance). Its value can be
        /// positive or negative under these conditions.
        sum_a: f64,
        /// The sum of the 'b' components (within-population variance). Its value can also
        /// be positive or negative.
        sum_b: f64,
        /// The number of genetic sites whose summed variance components led to this
        /// indeterminate FST outcome.
        num_informative_sites: usize,
    },

    /// FST is undefined because the genetic data from the site or region shows no discernible
    /// allele frequency differences among populations that would indicate differentiation, leading to
    /// an FST calculation of 0/0. This can happen if, for instance, all populations are fixed for
    /// the same allele at all analyzed sites, or if allele frequencies are identical across all
    /// populations and there's no residual within-population variance contributing to 'b'.
    /// In such cases, both the estimated among-population variance (`sum_a`) and the estimated
    /// total variance (`sum_a + sum_b`) are effectively zero.
    NoInterPopulationVariance {
        /// The sum of the 'a' components, expected to be approximately 0.0 in this state.
        sum_a: f64,
        /// The sum of the 'b' components, also expected to be approximately 0.0 in this state.
        sum_b: f64,
        /// The number of genetic sites that were evaluated and found to have no
        /// inter-population variance (e.g., all monomorphic, or all having identical
        /// allele frequencies across populations leading to zero components).
        sites_evaluated: usize,
    },

    /// FST could not be estimated because the input data did not meet the fundamental
    /// requirements for the calculation. For example, FST quantifies differentiation among
    /// populations, so at least two populations are required. Other reasons could include
    /// a complete lack of processable variant sites in the specified genomic region or all
    /// individual sites resulting in a state that prevents component contribution.
    /// In this situation, no meaningful FST value or variance components can be reported.
    InsufficientDataForEstimation {
        /// The sum of 'a' components; this field is set to a default (e.g., 0.0) as meaningful
        /// components were not derived from the data due to the insufficiency.
        sum_a: f64,
        /// The sum of 'b' components; similarly, set to a default as components were not
        /// meaningfully derived.
        sum_b: f64,
        /// The number of genetic sites where an FST estimation was attempted but could not
        /// proceed to the calculation of variance components or their meaningful summation
        /// due to data limitations. For a single-site attempt, this value would be 1.
        sites_attempted: usize,
    },
}

impl fmt::Display for FstEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FstEstimate::Calculable { value, .. } => {
                // sum_a, sum_b, num_informative_sites are not used in this Display impl
                let val_str = if value.is_nan() {
                    "NaN".to_string()
                } else if value.is_infinite() {
                    if value.is_sign_positive() {
                        "Infinity".to_string()
                    } else {
                        "-Infinity".to_string()
                    }
                } else {
                    format!("{:.6}", value) // Common precision for FST
                };
                write!(f, "{}", val_str) // Output only the formatted FST value string
            }
            FstEstimate::ComponentsYieldIndeterminateRatio {
                sum_a,
                sum_b,
                num_informative_sites,
            } => {
                write!(
                    f,
                    "IndeterminateRatio (A: {:.4e}, B: {:.4e}, N_inf_sites: {})",
                    sum_a, sum_b, num_informative_sites
                )
            }
            FstEstimate::NoInterPopulationVariance {
                sum_a,
                sum_b,
                sites_evaluated,
            } => {
                write!(
                    f,
                    "NoInterPopVariance (A: {:.4e}, B: {:.4e}, SitesEval: {})",
                    sum_a, sum_b, sites_evaluated
                )
            }
            FstEstimate::InsufficientDataForEstimation {
                sum_a,
                sum_b,
                sites_attempted,
            } => {
                // For InsufficientData, sum_a and sum_b are not typically meaningful data-derived sums.
                write!(
                    f,
                    "InsufficientData (A: {:.3e}, B: {:.3e}, SitesAtt: {})",
                    sum_a, sum_b, sites_attempted
                )
            }
        }
    }
}

// Define a struct to hold diversity metrics for each genomic site
#[derive(Debug)]
pub struct SiteDiversity {
    pub position: i64,        // 1-based position of the site in the genome
    pub pi: f64,              // Nucleotide diversity (π) at this site
    pub watterson_theta: f64, // Watterson's theta (θ_w) at this site
}

/// FST results for a single site using the Weir & Cockerham method.
#[derive(Debug, Clone)]
pub struct SiteFstWc {
    /// Position (1-based coordinate) of the site.
    pub position: i64,

    /// Overall FST estimate across all populations for this site.
    pub overall_fst: FstEstimate,

    /// Pairwise FST estimates between populations for this site.
    /// Keys are formatted as "pop_id1_vs_pop_id2" where pop_id1 < pop_id2.
    pub pairwise_fst: HashMap<String, FstEstimate>,

    /// Variance components (a, b) from Weir & Cockerham used for `overall_fst` at this site.
    /// `a` is the among-population component, `b` is the within-population component.
    pub variance_components: (f64, f64),

    /// Number of haplotypes in each population group contributing to this site's calculations.
    pub population_sizes: HashMap<String, usize>,

    /// Pairwise variance components (a_xy, b_xy) for each subpopulation pair at this site.
    /// These are used to calculate the `pairwise_fst` values for this site.
    pub pairwise_variance_components: HashMap<String, (f64, f64)>,
}

/// Identifier for a population or group being analyzed, used across FST methods.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PopulationId {
    /// For predefined groups like 0 (e.g., reference) and 1 (e.g., inversion)
    HaplotypeGroup(u8),
    /// For populations defined by names from external files
    Named(String),
}

/// Represents a collection of haplotypes and associated data for a specific population/group
/// within a defined genomic region. This context is used for diversity and differentiation calculations.
/// The lifetime 'a is tied to the underlying variants and sample_names slices, so no
/// data is copied unnecessarily for these large collections.
#[derive(Debug, Clone)]
pub struct PopulationContext<'a> {
    /// Unique identifier for this population or group.
    ///
    pub id: PopulationId,
    /// List of haplotypes belonging to this population. Each tuple contains the
    /// VCF sample index and the specific haplotype side (Left or Right).
    pub haplotypes: Vec<(usize, HaplotypeSide)>,
    /// Slice of variants relevant to the genomic region being analyzed for this population.
    pub variants: &'a [Variant],
    /// Slice of all sample names present in the VCF, used for context or debugging.
    pub sample_names: &'a [String],
    /// The effective sequence length (L) for normalization in diversity calculations.
    /// This should account for any masking or specific intervals considered.
    pub sequence_length: i64,
}

/// Holds the result of a Dxy (between-population nucleotide diversity) calculation,
/// specifically for Hudson's FST.
#[derive(Debug, Clone, Default)]
pub struct DxyHudsonResult {
    /// The calculated Dxy value (average pairwise differences per site between two populations).
    /// `None` if calculation was not possible (e.g., no valid pairs, zero sequence length).
    pub d_xy: Option<f64>,
    // Maybe others later
}

/// Encapsulates all components and the final FST value for a pairwise Hudson's FST calculation.
#[derive(Debug, Clone, Default)]
pub struct HudsonFSTOutcome {
    /// Identifier for the first population in the comparison.
    pub pop1_id: Option<PopulationId>,
    /// Identifier for the second population in the comparison.
    pub pop2_id: Option<PopulationId>,
    /// The calculated Hudson's FST value.
    /// `None` if FST could not be determined (e.g., Dxy is zero or components are missing).
    pub fst: Option<f64>,
    /// Between-population nucleotide diversity (Dxy).
    pub d_xy: Option<f64>,
    /// Within-population nucleotide diversity for the first population (π1).
    pub pi_pop1: Option<f64>,
    /// Within-population nucleotide diversity for the second population (π2).
    pub pi_pop2: Option<f64>,
    /// Average within-population diversity: 0.5 * (π1 + π2).
    /// `None` if either pi_pop1 or pi_pop2 is `None`.
    pub pi_xy_avg: Option<f64>,
}

/// Per-site Hudson FST values and components.
#[derive(Debug, Clone, Default)]
pub struct SiteFstHudson {
    /// 1-based position of the site.
    pub position: i64,
    /// Per-site Hudson FST value.
    pub fst: Option<f64>,
    /// Between-population diversity at this site.
    pub d_xy: Option<f64>,
    /// Within-population diversity for population 1 at this site.
    pub pi_pop1: Option<f64>,
    /// Within-population diversity for population 2 at this site.
    pub pi_pop2: Option<f64>,
    /// Number of called haplotypes in population 1 at this site.
    pub n1_called: usize,
    /// Number of called haplotypes in population 2 at this site.
    pub n2_called: usize,
    /// Numerator component for regional aggregation: Dxy - 0.5*(pi1 + pi2).
    pub num_component: Option<f64>,
    /// Denominator component for regional aggregation: Dxy.
    pub den_component: Option<f64>,
}

/// Weir & Cockerham FST results for a genomic region.
#[derive(Debug, Clone)]
pub struct FstWcResults {
    /// Overall FST estimate for the entire region.
    pub overall_fst: FstEstimate,

    /// Pairwise FST estimates for each pair of populations across the region.
    /// Keys are population pair identifiers (e.g., "pop1_vs_pop2").
    pub pairwise_fst: HashMap<String, FstEstimate>,

    /// Summed pairwise variance components (sum_a_xy, sum_b_xy) for each subpopulation pair
    /// across the entire region. These are the sums used to calculate the values in `pairwise_fst`.
    /// Keys are population pair identifiers (e.g., "pop1_vs_pop2").
    pub pairwise_variance_components: HashMap<String, (f64, f64)>,

    /// Per-site FST values and components.
    pub site_fst: Vec<SiteFstWc>,

    /// Describes the type of grouping used for FST calculation (e.g., "haplotype_groups", "population_groups").
    pub fst_type: String,
}

/*
    Weir & Cockerham (1984) define F-statistics (F, Θ, f) as correlations
    of alleles at different levels: within individuals, among individuals
    within subpopulations, and among subpopulations. The parameters
    can be estimated by partitioning the total allelic variance into
    hierarchical components a, b, and c.

    In the standard diploid random-mating model, the parameter F (F_IS) measures
    correlation of genes within individuals, while Θ (F_ST) measures correlation
    of genes among subpopulations.

    The model also allows a “within-individual” term c.

    However, if we treat each haplotype independently and assume random union
    of gametes (no within-individual correlation), then effectively c=0
    and we can use simplified "haploid" forms of the Weir & Cockerham (W&C)
    variance-component estimators. In this scenario, a is the among-subpopulation
    variance component, and b is the within-subpopulation variance component.

    For a single site with subpopulations i = 1..r:
       - Let p_i be the allele frequency in subpopulation i,
       - Let n_i be the number of haplotypes sampled in subpopulation i,
       - Let p̄ = (Σ n_i p_i) / (Σ n_i) be the global (pooled) frequency,
       - Let S² = [ Σ n_i (p_i - p̄)² ] / [ (r-1)*n̄ ]  (a weighted variance)
         where n̄ = (Σ n_i) / r is the average sample size,
       - Let c² = [ Σ (n_i - n̄)² ] / [ r n̄² ] measure the squared CV of n_i.

    Conceptually, "a" is the among-subpopulations variance component, and "b"
    is the residual within-subpop variance. The Fst at that site is then

       Fst(site) = a / (a + b).

    We repeat for each site and sum the a_i and b_i across sites i to obtain an overall Fst:

       Fst(overall) = ( Σ_i a_i ) / ( Σ_i (a_i + b_i) ).

    Pairwise subpopulation Fst can be done by restricting the above to only the
    two subpops of interest (i.e., r=2).

    In our “haplotype-based” version, each diploid sample contributes two
    haplotypes (no inbreeding parameter), so we treat them as
    independent. We omit W&C’s “c” term for within-individual correlation.
*/

// Calculates Weir & Cockerham's FST (Fixation Index) across a specified genomic region,
// partitioning genetic variation between predefined haplotype groups. These groups
// might represent, for example, samples carrying different alleles of a structural variant
// (like an inversion) or other genetic markers defining distinct cohorts.
//
// This function implements the FST estimator as described by Weir & Cockerham (1984).
// A key assumption for haplotype-level data is that observed within-haplotype
// heterozygosity is zero. This simplifies the model,
// causing the variance component 'c' (variance between gametes within individuals)
// to also be zero. Consequently, FST (denoted theta) is estimated as the ratio of
// among-population variance ('a') to the total variance ('a' + 'b'), where 'b' is the
// variance among haplotypes within populations: FST = a / (a + b).
//
// Two main stages:
// 1. Per-Site Estimation: For each genetic site (e.g., SNP) in the region, variance
//    components (a_i, b_i) and an FST estimate are determined.
// 2. Regional Aggregation: These per-site components are then summed across all
//    informative sites in the region. An overall FST estimate for the entire region is
//    computed using these summed components, consistent with Weir & Cockerham equation 10:
//    sum a_i / sum (a_i + b_i). Overall pairwise FST estimates between specific haplotype
//    groups are also calculated using the same aggregation principle for the relevant pairs.
//
// Arguments:
// - `variants`: A slice of `Variant` structs, containing genotype data for all samples
//   across the relevant loci in the target genomic region.
// - `sample_names`: A slice of `String`s, representing the VCF sample identifiers.
//   These are used to map samples to their assigned haplotype groups.
// - `sample_to_group_map`: A `HashMap` that links each VCF sample name (String) to
//   its haplotype group assignments.
// - `region`: A `QueryRegion` struct that defines the genomic start and end coordinates
//   (0-based, inclusive) of the region to be analyzed.
//
// Returns:
// An `FstWcResults` struct. This struct encapsulates:
// - The overall FST estimate for the entire region.
// - A map of pairwise FST estimates between different haplotype groups across the region.
// - The summed pairwise variance components (a_xy, b_xy) for each pair.
// - A detailed list of per-site FST results (`SiteFstWc`).
// - A string indicating the type of FST analysis performed (e.g., "haplotype_groups").
pub fn calculate_fst_wc_haplotype_groups(
    variants: &[Variant],
    sample_names: &[String],
    sample_to_group_map: &HashMap<String, (u8, u8)>,
    region: QueryRegion,
) -> FstWcResults {
    let spinner = create_spinner(&format!(
        "Calculating FST between haplotype groups for region {}:{}..{} (length {})",
        // Attempt to display chromosome name from the first sample if available, otherwise "UnknownChr".
        sample_names.get(0).map_or("UnknownChr", |s_name| s_name
            .split('_')
            .next()
            .unwrap_or("UnknownChr")),
        ZeroBasedPosition(region.start).to_one_based(), // Convert 0-based start to 1-based for display
        ZeroBasedPosition(region.end).to_one_based(), // Convert 0-based end to 1-based for display
        region.len()
    ));

    log(LogLevel::Info, &format!(
        "Beginning FST calculation between haplotype groups (e.g., 0 vs 1) for region {}:{}..{}",
        sample_names.get(0).map_or("UnknownChr", |s_name| s_name.split('_').next().unwrap_or("UnknownChr")),
        ZeroBasedPosition(region.start).to_one_based(),
        ZeroBasedPosition(region.end).to_one_based()
    ));

    // Map each VCF sample's haplotypes to their assigned groups (e.g., group "0" or "1").
    let haplotype_to_group = map_samples_to_haplotype_groups(sample_names, sample_to_group_map);

    // Create a lookup map for quick access to variants by their genomic position.
    // This improves performance when iterating through positions in the region.
    let variant_map: HashMap<i64, &Variant> = variants.iter().map(|v| (v.position, v)).collect();

    let mut site_fst_values = Vec::with_capacity(region.len() as usize);
    let position_count = region.len() as usize; // Total number of base pairs in the region.

    init_step_progress(
        &format!(
            "Calculating FST at {} positions for haplotype groups",
            position_count
        ),
        position_count as u64,
    );

    // Iterate through each genomic position (0-based) within the specified region.
    for (idx, pos) in (region.start..=region.end).enumerate() {
        // Update progress bar periodically for long calculations.
        if idx % 1000 == 0 || idx == 0 || idx == position_count.saturating_sub(1) {
            update_step_progress(
                idx as u64,
                &format!(
                    "Position {}/{} ({:.1}%)",
                    idx + 1,
                    position_count,
                    ((idx + 1) as f64 / position_count as f64) * 100.0
                ),
            );
        }

        if let Some(variant) = variant_map.get(&pos) {
            // If a variant exists at this position, calculate its FST.
            let (overall_fst, pairwise_fst, var_comps, pop_sizes, pairwise_var_comps) =
                calculate_fst_wc_at_site_by_haplotype_group(variant, &haplotype_to_group);

            site_fst_values.push(SiteFstWc {
                position: ZeroBasedPosition(pos).to_one_based(), // Store position as 1-based for output consistency.
                overall_fst,
                pairwise_fst,
                variance_components: var_comps, // Store the raw (a,b) components for this site.
                population_sizes: pop_sizes,
                pairwise_variance_components: pairwise_var_comps,
            });
        } else {
            // If no variant is found at this position in the input `variants` slice,
            // the site is considered monomorphic across all samples for this analysis.
            // For such sites, there's no genetic variation to partition, so FST components 'a' and 'b' are zero.
            let mut actual_pop_sizes = HashMap::new();
            let mut group_counts: HashMap<String, usize> = HashMap::new();
            for group_id_str in haplotype_to_group.values() {
                *group_counts.entry(group_id_str.clone()).or_insert(0) += 1;
            }
            // Ensure "0" and "1" group sizes are recorded, even if zero.
            actual_pop_sizes.insert("0".to_string(), *group_counts.get("0").unwrap_or(&0));
            actual_pop_sizes.insert("1".to_string(), *group_counts.get("1").unwrap_or(&0));

            // Overall FST for a monomorphic site results in NoInterPopulationVariance.
            let site_is_monomorphic_overall_estimate = FstEstimate::NoInterPopulationVariance {
                sum_a: 0.0,
                sum_b: 0.0,
                sites_evaluated: 1, // This single site was evaluated as monomorphic.
            };

            let mut populated_pairwise_fst = HashMap::new();
            let mut populated_pairwise_var_comps = HashMap::new();

            // For haplotype group comparisons (typically "0" vs "1"), if both groups
            // have members, their pairwise FST at this monomorphic site is also NoInterPopulationVariance.
            if actual_pop_sizes.get("0").map_or(false, |&count| count > 0)
                && actual_pop_sizes.get("1").map_or(false, |&count| count > 0)
            {
                let site_is_monomorphic_pairwise_estimate =
                    FstEstimate::NoInterPopulationVariance {
                        sum_a: 0.0,
                        sum_b: 0.0,
                        sites_evaluated: 1, // This site was evaluated for this specific pair.
                    };
                populated_pairwise_fst
                    .insert("0_vs_1".to_string(), site_is_monomorphic_pairwise_estimate);
                // Pairwise variance components (a_xy, b_xy) for a monomorphic site are (0.0, 0.0).
                populated_pairwise_var_comps.insert("0_vs_1".to_string(), (0.0, 0.0));
            }

            site_fst_values.push(SiteFstWc {
                position: ZeroBasedPosition(pos).to_one_based(),
                overall_fst: site_is_monomorphic_overall_estimate,
                pairwise_fst: populated_pairwise_fst,
                variance_components: (0.0, 0.0), // Overall (a,b) for this monomorphic site are zero.
                population_sizes: actual_pop_sizes,
                pairwise_variance_components: populated_pairwise_var_comps,
            });
        }
    }
    finish_step_progress("Completed per-site FST calculations for haplotype groups");

    // Aggregate per-site FST components to get regional FST estimates.
    let (overall_fst_estimate, pairwise_fst_estimates, aggregated_pairwise_components) =
        calculate_overall_fst_wc(&site_fst_values);

    let defined_positive_fst_sites = site_fst_values.iter()
        .filter(|site| matches!(site.overall_fst, FstEstimate::Calculable { value, .. } if value.is_finite() && value > 0.0))
        .count();

    log(LogLevel::Info, &format!(
        "Haplotype group FST calculation complete: {} sites showed positive, finite FST out of {} total positions in the region.",
        defined_positive_fst_sites, site_fst_values.len()
    ));

    log(
        LogLevel::Info,
        &format!(
            "Overall FST between haplotype groups for the region: {}",
            overall_fst_estimate
        ),
    );

    for (pair_key, fst_estimate) in &pairwise_fst_estimates {
        log(
            LogLevel::Info,
            &format!("Regional pairwise FST for {}: {}", pair_key, fst_estimate),
        );
    }

    spinner.finish_and_clear();

    FstWcResults {
        overall_fst: overall_fst_estimate,
        pairwise_fst: pairwise_fst_estimates,
        pairwise_variance_components: aggregated_pairwise_components,
        site_fst: site_fst_values,
        fst_type: "haplotype_groups".to_string(), // Type of grouping used.
    }
}

/// Calculates Weir & Cockerham FST between population groups defined in a CSV file for a genomic region.
///
/// This function implements the Weir & Cockerham (1984) estimator for FST (theta).
/// It reads population assignments from a CSV file, then calculates per-site variance
/// components (a_i, b_i) and FST estimates. It aggregates these across all sites in
/// the region for overall and pairwise FST estimates.
///
/// # Arguments
/// * `variants`: A slice of `Variant` structs for all samples in the region.
/// * `sample_names`: A slice of `String`s, representing the names of all samples.
/// * `csv_path`: Path to the CSV file defining population assignments.
/// * `region`: A `QueryRegion` struct defining the genomic start and end (0-based, inclusive).
///
/// # Returns
/// A `Result` containing `FstWcResults` (with `FstEstimate` enums) or a `VcfError`.
pub fn calculate_fst_wc_csv_populations(
    variants: &[Variant],
    sample_names: &[String],
    csv_path: &Path,
    region: QueryRegion,
) -> Result<FstWcResults, VcfError> {
    let spinner =
        create_spinner(&format!(
        "Calculating FST between CSV-defined population groups for region {}:{}..{} (length {})",
        sample_names.get(0).map_or("UnknownChr", |s| s.split('_').next().unwrap_or("UnknownChr")),
        ZeroBasedPosition(region.start).to_one_based(),
        ZeroBasedPosition(region.end).to_one_based(),
        region.len()
    ));

    log(
        LogLevel::Info,
        &format!(
        "Beginning FST calculation between population groups defined in {} for region {}:{}..{}",
        csv_path.display(),
        sample_names.get(0).map_or("UnknownChr", |s| s.split('_').next().unwrap_or("UnknownChr")),
        ZeroBasedPosition(region.start).to_one_based(),
        ZeroBasedPosition(region.end).to_one_based()
    ),
    );

    let population_assignments = parse_population_csv(csv_path)?;
    let sample_to_pop = map_samples_to_populations(sample_names, &population_assignments);

    let variant_map: HashMap<i64, &Variant> = variants.iter().map(|v| (v.position, v)).collect();

    let mut site_fst_values = Vec::with_capacity(region.len() as usize);
    let position_count = region.len() as usize;

    init_step_progress(
        &format!(
            "Calculating FST at {} positions for CSV populations",
            position_count
        ),
        position_count as u64,
    );

    for (idx, pos) in (region.start..=region.end).enumerate() {
        if idx % 1000 == 0 || idx == 0 || idx == position_count.saturating_sub(1) {
            update_step_progress(
                idx as u64,
                &format!(
                    "Position {}/{} ({:.1}%)",
                    idx + 1,
                    position_count,
                    ((idx + 1) as f64 / position_count as f64) * 100.0
                ),
            );
        }

        if let Some(variant) = variant_map.get(&pos) {
            let (overall_fst, pairwise_fst, var_comps, pop_sizes, pairwise_var_comps) =
                calculate_fst_wc_at_site_by_population(variant, &sample_to_pop);

            site_fst_values.push(SiteFstWc {
                position: ZeroBasedPosition(pos).to_one_based(),
                overall_fst,
                pairwise_fst,
                variance_components: var_comps,
                population_sizes: pop_sizes,
                pairwise_variance_components: pairwise_var_comps,
            });
        } else {
            // Site is not in variant_map, thus considered globally monomorphic for this region.
            // FST is undefined because variance components a and b are zero.
            let mut pop_counts_for_site: HashMap<String, usize> = HashMap::new();
            for pop_id_str in sample_to_pop.values() {
                *pop_counts_for_site.entry(pop_id_str.clone()).or_insert(0) += 1;
            }
            // These are haplotype counts per population.

            let site_is_monomorphic_estimate = FstEstimate::NoInterPopulationVariance {
                sum_a: 0.0,
                sum_b: 0.0,
                sites_evaluated: 1, // This one site was evaluated as monomorphic
            };

            let mut site_pairwise_fst = HashMap::new();
            let mut site_pairwise_var_comps = HashMap::new();
            let pop_ids: Vec<_> = pop_counts_for_site.keys().cloned().collect();
            // Create pairwise entries if there are at least two populations defined for this site
            if pop_ids.len() >= 2 {
                for i in 0..pop_ids.len() {
                    for j in (i + 1)..pop_ids.len() {
                        if pop_counts_for_site
                            .get(&pop_ids[i])
                            .map_or(false, |&count| count > 0)
                            && pop_counts_for_site
                                .get(&pop_ids[j])
                                .map_or(false, |&count| count > 0)
                        {
                            let (key_pop1, key_pop2) = if pop_ids[i] < pop_ids[j] {
                                (&pop_ids[i], &pop_ids[j])
                            } else {
                                (&pop_ids[j], &pop_ids[i])
                            };
                            let pair_key = format!("{}_vs_{}", key_pop1, key_pop2);
                            site_pairwise_fst
                                .insert(pair_key.clone(), site_is_monomorphic_estimate);
                            site_pairwise_var_comps.insert(pair_key, (0.0, 0.0));
                        }
                    }
                }
            }

            site_fst_values.push(SiteFstWc {
                position: ZeroBasedPosition(pos).to_one_based(),
                overall_fst: site_is_monomorphic_estimate,
                pairwise_fst: site_pairwise_fst,
                variance_components: (0.0, 0.0),
                population_sizes: pop_counts_for_site,
                pairwise_variance_components: site_pairwise_var_comps,
            });
        }
    }

    finish_step_progress("Completed per-site FST calculations for CSV populations");

    let (overall_fst_estimate, pairwise_fst_estimates, aggregated_pairwise_components) =
        calculate_overall_fst_wc(&site_fst_values);

    let defined_positive_fst_sites = site_fst_values.iter()
        .filter(|site| matches!(site.overall_fst, FstEstimate::Calculable { value, .. } if value.is_finite() && value > 0.0))
        .count();

    let population_count = population_assignments.keys().count();

    log(LogLevel::Info, &format!(
        "CSV population FST calculation complete: {} sites with defined positive FST out of {} total positions, across {} populations.",
        defined_positive_fst_sites, site_fst_values.len(), population_count
    ));

    log(
        LogLevel::Info,
        &format!(
            "Overall FST across all populations: {}",
            overall_fst_estimate
        ),
    );

    for (pair, fst_e) in &pairwise_fst_estimates {
        log(
            LogLevel::Info,
            &format!("Pairwise FST for {}: {}", pair, fst_e),
        );
    }

    spinner.finish_and_clear();

    Ok(FstWcResults {
        overall_fst: overall_fst_estimate,
        pairwise_fst: pairwise_fst_estimates,
        pairwise_variance_components: aggregated_pairwise_components,
        site_fst: site_fst_values,
        fst_type: "population_groups".to_string(),
    })
}

/// Parses a CSV file containing population assignments for FST calculations.
///
/// The CSV file should have population labels in the first column,
/// and subsequent columns on the same row should list sample IDs belonging to that population.
/// Lines starting with '#' are treated as comments and skipped. Empty lines are also skipped.
/// Sample IDs and population names are trimmed of whitespace.
///
/// # Arguments
/// * `csv_path`: A reference to the `Path` of the CSV file.
///
/// # Returns
/// A `Result` containing a `HashMap` where keys are population names (String)
/// and values are `Vec<String>` of sample IDs associated with that population.
/// Returns `VcfError::Parse` if the file contains no valid population data after parsing,
/// or `VcfError::Io` if the file cannot be opened or read.
pub fn parse_population_csv(csv_path: &Path) -> Result<HashMap<String, Vec<String>>, VcfError> {
    let file = File::open(csv_path).map_err(|e| {
        VcfError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!(
                "Failed to open population CSV file {}: {}",
                csv_path.display(),
                e
            ),
        ))
    })?;

    let reader = BufReader::new(file);
    let mut population_map = HashMap::new();

    for line_result in reader.lines() {
        let line = line_result.map_err(VcfError::Io)?;
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<String> = line.split(',').map(|s| s.trim().to_string()).collect();
        if parts.is_empty() || parts[0].is_empty() {
            continue;
        }

        let population = parts[0].clone();
        let samples: Vec<String> = parts
            .iter()
            .skip(1)
            .filter(|s| !s.is_empty())
            .cloned()
            .collect();

        if !samples.is_empty() {
            population_map.insert(population, samples);
        } else {
            log(LogLevel::Warning, &format!("Population '{}' in CSV file '{}' has no associated sample IDs listed on its line.", population, csv_path.display()));
        }
    }

    if population_map.is_empty() {
        return Err(VcfError::Parse(format!(
            "Population CSV file '{}' contains no valid population data after parsing.",
            csv_path.display()
        )));
    }

    Ok(population_map)
}

/// Maps VCF samples to their predefined haplotype groups (0 or 1).
///
/// # Arguments
/// * `sample_names`: Slice of all sample names from the VCF.
/// * `sample_to_group_map`: HashMap from config, mapping sample names to (left_hap_group, right_hap_group).
///
/// # Returns
/// A HashMap mapping `(vcf_sample_index, HaplotypeSide)` tuples to group identifier strings ("0" or "1").
/// Extract core sample ID by removing _L/_R suffix if present
fn core_sample_id(name: &str) -> &str {
    if let Some(s) = name.strip_suffix("_L").or_else(|| name.strip_suffix("_R")) {
        s
    } else {
        name
    }
}

#[cfg(test)]
mod core_sample_id_tests {
    use super::*;

    #[test]
    fn test_core_sample_id() {
        assert_eq!(core_sample_id("NA12878_L"), "NA12878");
        assert_eq!(core_sample_id("NA12878_R"), "NA12878");
        assert_eq!(core_sample_id("SAMP_01_L"), "SAMP_01");
        assert_eq!(core_sample_id("SAMP_01_R"), "SAMP_01");
        assert_eq!(core_sample_id("NoSuffix"), "NoSuffix");
        assert_eq!(core_sample_id("Sample_With_Underscores_L"), "Sample_With_Underscores");
        assert_eq!(core_sample_id("Sample_With_Underscores_R"), "Sample_With_Underscores");
    }
}

fn map_samples_to_haplotype_groups(
    sample_names: &[String],
    sample_to_group_map: &HashMap<String, (u8, u8)>,
) -> HashMap<(usize, HaplotypeSide), String> {
    let mut haplotype_to_group = HashMap::new();

    let mut sample_id_to_index = HashMap::new();
    for (idx, name) in sample_names.iter().enumerate() {
        let core = core_sample_id(name);
        sample_id_to_index.insert(core.to_string(), idx);     // core id
        sample_id_to_index.insert(name.clone(), idx);         // exact id (defensive)
    }

    for (config_sample_name, &(left_group, right_group)) in sample_to_group_map {
        if let Some(&vcf_idx) = sample_id_to_index.get(config_sample_name.as_str()) {
            haplotype_to_group.insert((vcf_idx, HaplotypeSide::Left), left_group.to_string());

            haplotype_to_group.insert((vcf_idx, HaplotypeSide::Right), right_group.to_string());
        }
    }

    haplotype_to_group
}

/// Maps VCF samples to their population groups as defined in a CSV file.
///
/// # Arguments
/// * `sample_names`: Slice of all sample names from the VCF.
/// * `population_assignments`: HashMap from parsed CSV, mapping population names to lists of sample IDs.
///
/// # Returns
/// A HashMap mapping `(vcf_sample_index, HaplotypeSide)` tuples to population identifier strings.
fn map_samples_to_populations(
    sample_names: &[String],
    population_assignments: &HashMap<String, Vec<String>>,
) -> HashMap<(usize, HaplotypeSide), String> {
    let mut sample_to_pop_map_for_fst = HashMap::new();

    let mut csv_sample_id_to_pop_name = HashMap::new();
    for (pop_name, samples_in_pop) in population_assignments {
        for sample_id in samples_in_pop {
            csv_sample_id_to_pop_name.insert(sample_id.clone(), pop_name.clone());
        }
    }

    for (vcf_idx, vcf_sample_name) in sample_names.iter().enumerate() {
        // Attempt direct match
        if let Some(pop_name) = csv_sample_id_to_pop_name.get(vcf_sample_name) {
            sample_to_pop_map_for_fst.insert((vcf_idx, HaplotypeSide::Left), pop_name.clone());
            sample_to_pop_map_for_fst.insert((vcf_idx, HaplotypeSide::Right), pop_name.clone());
            continue;
        }

        // Attempt match by suffix (e.g., VCF "ID_L" or "ID_R" vs CSV "ID")
        let core_vcf_id = core_sample_id(vcf_sample_name);
        if let Some(pop_name) = csv_sample_id_to_pop_name.get(core_vcf_id) {
            sample_to_pop_map_for_fst.insert((vcf_idx, HaplotypeSide::Left), pop_name.clone());
            sample_to_pop_map_for_fst.insert((vcf_idx, HaplotypeSide::Right), pop_name.clone());
            continue;
        }

        // Optional: attempt match by prefix (e.g. VCF "POP_SUBPOP_ID" vs CSV "POP") - currently present
        let vcf_prefix = vcf_sample_name.split('_').next().unwrap_or(vcf_sample_name);
        for (csv_pop_name, _) in population_assignments {
            // Iterating to check if VCF name STARTS WITH a pop name
            if vcf_sample_name.starts_with(csv_pop_name) || vcf_prefix == csv_pop_name {
                sample_to_pop_map_for_fst
                    .insert((vcf_idx, HaplotypeSide::Left), csv_pop_name.clone());
                sample_to_pop_map_for_fst
                    .insert((vcf_idx, HaplotypeSide::Right), csv_pop_name.clone());
                break;
            }
        }
    }

    sample_to_pop_map_for_fst
}

/// Calculates Weir & Cockerham FST at a single site between predefined haplotype groups (0 vs 1).
///
/// This function delegates to `calculate_fst_wc_at_site_general`.
///
/// # Arguments
/// * `variant`: The `Variant` struct for the site.
/// * `haplotype_to_group`: A `HashMap` mapping `(vcf_sample_index, HaplotypeSide)` to group ID strings ("0" or "1").
///
/// # Returns
/// A tuple containing the overall FST estimate (`FstEstimate`), pairwise FST estimates,
/// variance components (a,b), population sizes, and pairwise variance components (a_xy, b_xy) for the site.
fn calculate_fst_wc_at_site_by_haplotype_group(
    variant: &Variant,
    haplotype_to_group: &HashMap<(usize, HaplotypeSide), String>,
) -> (
    FstEstimate,
    HashMap<String, FstEstimate>,
    (f64, f64),
    HashMap<String, usize>,
    HashMap<String, (f64, f64)>,
) {
    calculate_fst_wc_at_site_general(variant, haplotype_to_group)
}

/// Calculates Weir & Cockerham FST at a single site between population groups defined from a CSV file.
///
/// This function delegates to `calculate_fst_wc_at_site_general`.
///
/// # Arguments
/// * `variant`: The `Variant` struct for the site.
/// * `sample_to_pop`: A `HashMap` mapping `(vcf_sample_index, HaplotypeSide)` to population name strings.
///
/// # Returns
/// A tuple containing the overall FST estimate (`FstEstimate`), pairwise FST estimates,
/// variance components (a,b), population sizes, and pairwise variance components (a_xy, b_xy) for the site.
fn calculate_fst_wc_at_site_by_population(
    variant: &Variant,
    sample_to_pop: &HashMap<(usize, HaplotypeSide), String>,
) -> (
    FstEstimate,
    HashMap<String, FstEstimate>,
    (f64, f64),
    HashMap<String, usize>,
    HashMap<String, (f64, f64)>,
) {
    calculate_fst_wc_at_site_general(variant, sample_to_pop)
}

/// General function to calculate Weir & Cockerham FST components and estimates at a single site.
///
/// This function takes a variant and a mapping of haplotypes to subpopulation identifiers.
/// It calculates allele frequencies per subpopulation, then computes Weir & Cockerham's
/// variance components 'a' (among populations) and 'b' (within populations).
/// From these, it derives overall and pairwise FST estimates for the site.
///
/// **LIMITATION**: This implementation is effectively **biallelic only**. Multi-allelic sites
/// are collapsed by treating all non-reference alleles (allele_code != 0) as "alternate".
/// This distorts allele frequencies at truly multi-allelic sites.
///
/// # Arguments
/// * `variant`: The `Variant` data for the site.
/// * `map_subpop`: A `HashMap` where keys are `(vcf_sample_index, HaplotypeSide)` identifying a haplotype,
///   and values are `String` identifiers for the subpopulation that haplotype belongs to.
///
/// # Returns
/// A tuple:
///   - `overall_fst_at_site` (`FstEstimate`): The overall FST estimate for this site across all defined subpopulations.
///   - `pairwise_fst_estimate_map` (`HashMap<String, FstEstimate>`): Pairwise FST estimates between all pairs of subpopulations.
///   - `(overall_a, overall_b)` (`(f64, f64)`): The overall variance components 'a' and 'b' for the site.
///   - `pop_sizes` (`HashMap<String, usize>`): The number of haplotypes sampled per subpopulation at this site.
///   - `pairwise_variance_components_map` (`HashMap<String, (f64, f64)>`): The (a_xy, b_xy) components for each pair of subpopulations.
fn calculate_fst_wc_at_site_general(
    variant: &Variant,
    map_subpop: &HashMap<(usize, HaplotypeSide), String>,
) -> (
    FstEstimate,
    HashMap<String, FstEstimate>,
    (f64, f64),
    HashMap<String, usize>,
    HashMap<String, (f64, f64)>,
) {
    // 1. Count allele occurrences per subpopulation for the current variant.
    let mut allele_counts: HashMap<String, (usize, usize)> = HashMap::new(); // Stores (total_haplotypes_in_pop, alt_allele_count_in_pop)
    for (&(sample_idx, side), pop_id) in map_subpop {
        if let Some(Some(genotypes_vec)) = variant.genotypes.get(sample_idx) {
            if let Some(&allele_code) = genotypes_vec.get(side as usize) {
                let entry = allele_counts.entry(pop_id.clone()).or_insert((0, 0));
                entry.0 += 1; // Increment total haplotypes for this subpopulation.
                if allele_code != 0 {
                    // Assuming 0 is reference, non-zero is alternate.
                    entry.1 += 1; // Increment alternate allele count.
                }
            }
        }
    }

    // 2. Calculate allele frequencies and store population statistics.
    let mut pop_stats: HashMap<String, (usize, f64)> = HashMap::new(); // Stores (num_haplotypes_in_pop, alt_allele_frequency_in_pop)
    let mut pop_sizes: HashMap<String, usize> = HashMap::new();
    for (pop_id, (num_haplotypes, alt_allele_count)) in allele_counts {
        if num_haplotypes > 0 {
            let allele_freq = alt_allele_count as f64 / num_haplotypes as f64;
            pop_stats.insert(pop_id.clone(), (num_haplotypes, allele_freq));
            pop_sizes.insert(pop_id, num_haplotypes);
        }
    }

    // If fewer than two populations have data at this site, FST is not meaningful.
    if pop_stats.len() < 2 {
        log(LogLevel::Debug, &format!(
            "Site at pos {}: FST is InsufficientDataForEstimation (reason: found {} populations with data, need >= 2).",
            ZeroBasedPosition(variant.position).to_one_based(), pop_stats.len()
        ));
        let insufficient_data_estimate = FstEstimate::InsufficientDataForEstimation {
            sum_a: 0.0,
            sum_b: 0.0,
            sites_attempted: 1, // This one site was attempted
        };
        // Returns (0.0, 0.0) for site_a, site_b as they are not meaningfully calculated.
        return (
            insufficient_data_estimate,
            HashMap::new(),
            (0.0, 0.0),
            pop_sizes,
            HashMap::new(),
        );
    }

    // 3. Compute overall variance components (a, b) for all populations at this site.
    let total_haplotypes_overall: usize = pop_stats.values().map(|(n, _)| *n).sum();
    let mut weighted_freq_sum_overall = 0.0;
    for (_, (n, freq)) in &pop_stats {
        weighted_freq_sum_overall += (*n as f64) * freq;
    }
    let global_allele_freq = if total_haplotypes_overall > 0 {
        weighted_freq_sum_overall / (total_haplotypes_overall as f64)
    } else {
        0.0
    };

    let (site_a, site_b) = calculate_variance_components(&pop_stats, global_allele_freq);

    // Construct the FstEstimate for the overall FST at this site
    let overall_fst_at_site = {
        let denominator = site_a + site_b;
        let eps = 1e-9; // Small epsilon for float comparisons

        if denominator > eps {
            FstEstimate::Calculable {
                value: site_a / denominator,
                sum_a: site_a,
                sum_b: site_b,
                num_informative_sites: 1, // This site is informative
            }
        } else if denominator < -eps {
            FstEstimate::ComponentsYieldIndeterminateRatio {
                sum_a: site_a,
                sum_b: site_b,
                num_informative_sites: 1,
            }
        } else {
            // Denominator is effectively zero
            if site_a.abs() > eps {
                // Non-zero numerator / zero denominator
                FstEstimate::Calculable {
                    value: site_a / denominator, // Will be Inf or -Inf
                    sum_a: site_a,
                    sum_b: site_b,
                    num_informative_sites: 1,
                }
            } else {
                // Numerator is also effectively zero (0/0)
                FstEstimate::NoInterPopulationVariance {
                    sum_a: site_a,      // Should be ~0.0
                    sum_b: site_b,      // Should be ~0.0
                    sites_evaluated: 1, // This one site was evaluated
                }
            }
        }
    };

    // Logging for non-standard FST outcomes for overall_fst_at_site
    match overall_fst_at_site {
        FstEstimate::Calculable { value, .. } if !value.is_finite() => {
            log(
                LogLevel::Debug,
                &format!(
                    "Site at pos {}: Overall FST is {} (a={:.4e}, b={:.4e}, a+b={:.4e}).",
                    ZeroBasedPosition(variant.position).to_one_based(),
                    overall_fst_at_site,
                    site_a,
                    site_b,
                    site_a + site_b
                ),
            );
        }
        FstEstimate::ComponentsYieldIndeterminateRatio { .. }
        | FstEstimate::NoInterPopulationVariance { .. } => {
            log(
                LogLevel::Debug,
                &format!(
                    "Site at pos {}: Overall FST is {} (a={:.4e}, b={:.4e}, a+b={:.4e}).",
                    ZeroBasedPosition(variant.position).to_one_based(),
                    overall_fst_at_site,
                    site_a,
                    site_b,
                    site_a + site_b
                ),
            );
        }
        _ => {} // Finite Calculable cases or InsufficientData (already logged) don't need special logging here
    }

    // 4. Compute pairwise FSTs and their respective variance components (a_xy, b_xy).
    let mut pairwise_fst_estimate_map = HashMap::new();
    let mut pairwise_variance_components_map = HashMap::new();
    let pop_id_list: Vec<_> = pop_stats.keys().cloned().collect();

    for i in 0..pop_id_list.len() {
        for j in (i + 1)..pop_id_list.len() {
            let pop1_id_str = &pop_id_list[i];
            let pop2_id_str = &pop_id_list[j];

            let mut current_pair_stats = HashMap::new();
            current_pair_stats.insert(pop1_id_str.clone(), *pop_stats.get(pop1_id_str).unwrap());
            current_pair_stats.insert(pop2_id_str.clone(), *pop_stats.get(pop2_id_str).unwrap());

            let total_haplotypes_pair: usize = current_pair_stats.values().map(|(n, _)| *n).sum();
            let mut weighted_freq_sum_pair = 0.0;
            for (_, (n, freq)) in &current_pair_stats {
                weighted_freq_sum_pair += (*n as f64) * freq;
            }
            let pair_global_allele_freq = if total_haplotypes_pair > 0 {
                weighted_freq_sum_pair / (total_haplotypes_pair as f64)
            } else {
                0.0
            };

            let (pairwise_a_xy, pairwise_b_xy) =
                calculate_variance_components(&current_pair_stats, pair_global_allele_freq);

            let pairwise_fst_val = {
                let denominator_pair = pairwise_a_xy + pairwise_b_xy;
                let eps = 1e-9;

                if denominator_pair > eps {
                    FstEstimate::Calculable {
                        value: pairwise_a_xy / denominator_pair,
                        sum_a: pairwise_a_xy,
                        sum_b: pairwise_b_xy,
                        num_informative_sites: 1,
                    }
                } else if denominator_pair < -eps {
                    FstEstimate::ComponentsYieldIndeterminateRatio {
                        sum_a: pairwise_a_xy,
                        sum_b: pairwise_b_xy,
                        num_informative_sites: 1,
                    }
                } else {
                    // Denominator is effectively zero
                    if pairwise_a_xy.abs() > eps {
                        FstEstimate::Calculable {
                            value: pairwise_a_xy / denominator_pair,
                            sum_a: pairwise_a_xy,
                            sum_b: pairwise_b_xy,
                            num_informative_sites: 1,
                        }
                    } else {
                        FstEstimate::NoInterPopulationVariance {
                            sum_a: pairwise_a_xy,
                            sum_b: pairwise_b_xy,
                            sites_evaluated: 1,
                        }
                    }
                }
            };

            let (key_pop1, key_pop2) = if pop1_id_str < pop2_id_str {
                (pop1_id_str, pop2_id_str)
            } else {
                (pop2_id_str, pop1_id_str)
            };
            let pair_key = format!("{}_vs_{}", key_pop1, key_pop2);

            pairwise_fst_estimate_map.insert(pair_key.clone(), pairwise_fst_val);
            pairwise_variance_components_map
                .insert(pair_key.clone(), (pairwise_a_xy, pairwise_b_xy));

            match pairwise_fst_val {
                FstEstimate::Calculable { value, .. } if !value.is_finite() => {
                    log(LogLevel::Debug, &format!(
                        "Site at pos {}: Pairwise FST for {} is {} (a_xy={:.4e}, b_xy={:.4e}, a_xy+b_xy={:.4e}).",
                        ZeroBasedPosition(variant.position).to_one_based(), pair_key, pairwise_fst_val, pairwise_a_xy, pairwise_b_xy, pairwise_a_xy + pairwise_b_xy
                    ));
                }
                FstEstimate::ComponentsYieldIndeterminateRatio { .. }
                | FstEstimate::NoInterPopulationVariance { .. } => {
                    log(LogLevel::Debug, &format!(
                        "Site at pos {}: Pairwise FST for {} is {} (a_xy={:.4e}, b_xy={:.4e}, a_xy+b_xy={:.4e}).",
                        ZeroBasedPosition(variant.position).to_one_based(), pair_key, pairwise_fst_val, pairwise_a_xy, pairwise_b_xy, pairwise_a_xy + pairwise_b_xy
                    ));
                }
                _ => {}
            }
        }
    }
    // Return site_a and site_b (overall components for this site) for storage in SiteFstWc.
    (
        overall_fst_at_site,
        pairwise_fst_estimate_map,
        (site_a, site_b),
        pop_sizes,
        pairwise_variance_components_map,
    )
}

/// Calculates Weir & Cockerham (1984) variance components 'a' (among-population)
/// and 'b' (between effective individuals/haplotypes within populations) for a set of subpopulations.
///
/// These components are derived from Weir & Cockerham (equations 2 & 3)
/// under the specific assumption that the average observed heterozygosity `h_bar` (using W&C's notation
/// for heterozygosity of the allele under study) is zero. This assumption is appropriate for
/// haplotype-level data where each haplotype is treated as a haploid entity at each site,
/// meaning it carries a single allele and thus exhibits no heterozygosity itself.
/// This simplification results in W&C's variance component `c` (related to within-individual variance)
/// also being zero (their eq. 4).
///
/// # Arguments
/// * `pop_stats`: A `HashMap` mapping population identifiers (String) to tuples of
///  `(haplotype_sample_size_for_this_pop, alt_allele_frequency_in_this_pop)`.
/// * `global_freq`: The global (weighted average) frequency of the alternate allele across all considered subpopulations.
///
/// # Returns
/// A tuple `(a, b)` representing the estimated variance components. These components are not clamped
/// and can be negative due to sampling variance, which is consistent with W&C's estimator properties.
fn calculate_variance_components(
    pop_stats: &HashMap<String, (usize, f64)>, // (n_i, p_i)
    global_freq: f64,                          // p̄
) -> (f64, f64) {
    let r = pop_stats.len() as f64; // Number of subpopulations
    if r < 2.0 {
        // Need at least two populations to compare
        return (0.0, 0.0);
    }

    let mut n_values = Vec::with_capacity(pop_stats.len());
    let mut total_haplotypes = 0_usize;
    for (_, (size, _freq)) in pop_stats.iter() {
        n_values.push(*size as f64);
        total_haplotypes += *size;
    }

    let n_bar = (total_haplotypes as f64) / r; // Average sample size (n̄)

    // Check if n_bar - 1.0 is zero or negative, which would make subsequent calculations problematic.
    // This condition also covers n_bar <= 1.0.
    if (n_bar - 1.0) < 1e-9 {
        // Using < 1e-9 to catch n_bar very close to 1.0 or less than 1.0
        return (0.0, 0.0);
    }

    let global_p = global_freq; // p̄

    // Calculate c², the squared coefficient of variation of sample sizes (n_i).
    // c² = [ Σ (n_i - n̄)² ] / [ r * n̄² ]
    let mut sum_sq_diff_n = 0.0;
    for n_i_val in &n_values {
        let diff = *n_i_val - n_bar;
        sum_sq_diff_n += diff * diff;
    }
    let c_squared = if r > 0.0 && n_bar > 0.0 {
        // Avoid division by zero if r or n_bar is zero
        sum_sq_diff_n / (r * n_bar * n_bar)
    } else {
        0.0 // If r or n_bar is zero, c_squared is ill-defined or zero.
    };

    // Calculate S², the sample variance of allele frequencies over populations, weighted by n_i.
    // S² = [ Σ n_i (p_i - p̄)² ] / [ (r-1) * n̄ ]
    let mut numerator_s_squared = 0.0;
    for (_, (size, freq)) in pop_stats.iter() {
        let diff_p = *freq - global_p;
        numerator_s_squared += (*size as f64) * diff_p * diff_p;
    }
    let s_squared = if (r - 1.0) > 1e-9 && n_bar > 1e-9 {
        // denominators are positive
        numerator_s_squared / ((r - 1.0) * n_bar)
    } else {
        0.0 // If r=1 or n_bar=0, S² is undefined or zero.
    };

    // The implemented 'a' and 'b' components are derived from Weir & Cockerham (1984)
    // general estimators (their equations 2 and 3 respectively). For haplotype data, observed
    // heterozygosity (h_bar in W&C, their eq. 4 and related definitions) is effectively 0.
    // This leads to their variance component 'c' (W&C eq. 4) being 0,
    // and simplifies eqs. (2) and (3) to the forms implemented below.
    //
    // Let x_wc = global_p * (1.0 - global_p) - ((r - 1.0) / r) * s_squared.
    // This term, x_wc, represents the portion of p_bar * (1 - p_bar) that is not
    // explained by the among-population variance scaled by (r-1)/r.
    //
    // The formulas effectively compute:
    // a = (n_bar / n_c) * [s_squared - x_wc / (n_bar - 1.0)]
    //   (where n_bar / n_c is equivalent to 1.0 / (1.0 - (c_squared / r)) from W&C notation,
    //    and n_c is a correction factor for variance in sample sizes)
    // b = (n_bar / (n_bar - 1.0)) * x_wc

    let x_wc = global_p * (1.0 - global_p) - ((r - 1.0) / r) * s_squared;

    // Calculate component 'a' (among-population variance component)
    let a_numerator_term = s_squared - (x_wc / (n_bar - 1.0));
    // a_denominator_factor is n_c / n_bar, so dividing by it is multiplying by n_bar / n_c.
    let a_denominator_factor = 1.0 - (c_squared / r);

    // This division for 'a' is allowed to produce Infinity or NaN if a_denominator_factor is zero
    // (e.g., due to extreme sample size variance where n_c becomes 0)
    // and a_numerator_term is non-zero or zero, respectively.
    // These non-finite 'a' values will propagate to the calculation of (a+b),
    // and FstEstimate::from_ratio(a, a+b) will then correctly classify the
    // resulting FST estimate.
    let a = a_numerator_term / a_denominator_factor;

    // Calculate component 'b' (within-population variance component, effectively among haplotypes within populations)
    let b = (n_bar / (n_bar - 1.0)) * x_wc;

    (a, b) // Return raw estimated components; they can be negative.
}

/// Calculates overall and pairwise Weir & Cockerham FST estimates for a region from per-site FST results.
///
/// This function implements Equation 10 from Weir & Cockerham (1984) by summing
/// the among-population variance components (a_i) and within-population components (b_i)
/// from all relevant sites before calculating the final ratio using the new `FstEstimate` structure.
/// Relevant sites are those for which variance components could be estimated (i.e., not `InsufficientDataForEstimation` per-site).
///
/// # Arguments
/// * `site_fst_values`: A slice of `SiteFstWc` structs, each containing per-site
///   variance components and `FstEstimate` values.
///
/// # Returns
/// A tuple containing:
/// * `overall_fst_estimate` (`FstEstimate`): The Weir & Cockerham FST estimate for the entire region.
/// * `pairwise_fst_estimates` (`HashMap<String, FstEstimate>`): A map of regional pairwise FST estimates.
/// * `aggregated_pairwise_variance_components` (`HashMap<String, (f64,f64)>`): Summed (a_xy, b_xy) for each pair.
fn calculate_overall_fst_wc(
    site_fst_values: &[SiteFstWc],
) -> (
    FstEstimate,
    HashMap<String, FstEstimate>,
    HashMap<String, (f64, f64)>,
) {
    if site_fst_values.is_empty() {
        let estimate = FstEstimate::InsufficientDataForEstimation {
            sum_a: 0.0,
            sum_b: 0.0,
            sites_attempted: 0,
        };
        return (estimate, HashMap::new(), HashMap::new());
    }

    let mut num_per_site_insufficient = 0;
    // Stores (a_i, b_i) components from sites that were not per-site InsufficientDataForEstimation.
    let mut informative_site_components_overall: Vec<(f64, f64)> = Vec::new();
    // Stores (a_xy, b_xy) components for each pair from relevant sites.
    let mut informative_site_components_pairwise: HashMap<String, Vec<(f64, f64)>> = HashMap::new();
    // Keeps track of all unique pairwise keys observed across all sites.
    let mut all_observed_pair_keys = HashSet::new();

    for site in site_fst_values.iter() {
        // Aggregate components for overall FST
        // The SiteFstWc.variance_components stores the (a,b) for the overall calculation at that site.
        match site.overall_fst {
            FstEstimate::InsufficientDataForEstimation { .. } => {
                num_per_site_insufficient += 1;
            }
            // For Calculable, ComponentsYieldIndeterminateRatio, and NoInterPopulationVariance,
            // the raw a and b components from site.variance_components are summed.
            // These are the components that led to the per-site FstEstimate.
            _ => {
                // Catches Calculable, ComponentsYieldIndeterminateRatio, NoInterPopulationVariance for overall per-site
                let (site_a, site_b) = site.variance_components;
                informative_site_components_overall.push((site_a, site_b));
            }
        }

        // Aggregate components for pairwise FSTs
        // The SiteFstWc.pairwise_variance_components stores the (a_xy, b_xy) for each pair at that site.
        for (pair_key, &(site_a_xy, site_b_xy)) in &site.pairwise_variance_components {
            all_observed_pair_keys.insert(pair_key.clone());
            // We only sum components if the per-site pairwise FST for this pair was not InsufficientData.
            // If site.pairwise_fst for this pair_key indicates it was calculable or had components,
            // then site_a_xy and site_b_xy are relevant for summing.
            // The per-site FstEstimate itself is in site.pairwise_fst.get(pair_key)
            if !matches!(
                site.pairwise_fst.get(pair_key),
                Some(FstEstimate::InsufficientDataForEstimation { .. }) | None
            ) {
                informative_site_components_pairwise
                    .entry(pair_key.clone())
                    .or_default()
                    .push((site_a_xy, site_b_xy));
            }
        }
    }

    let total_sites_attempted = site_fst_values.len();
    // Number of sites that were not 'InsufficientDataForEstimation' at the per-site level (for the overall calculation).
    // These are the sites whose components (even if zero) are considered for summation.
    let sites_contributing_to_overall_sum = total_sites_attempted - num_per_site_insufficient;

    // Overall FST calculation
    let overall_fst_estimate = if sites_contributing_to_overall_sum == 0 {
        // This means all sites were individually InsufficientDataForEstimation (for overall FST context),
        // or no sites were provided (which is caught by the initial empty check).
        FstEstimate::InsufficientDataForEstimation {
            sum_a: 0.0,
            sum_b: 0.0,
            sites_attempted: total_sites_attempted, // Total sites input to this function.
        }
    } else {
        // At least one site contributed components for the overall FST calculation.
        let sum_a_total: f64 = informative_site_components_overall
            .iter()
            .map(|(a, _)| *a)
            .sum();
        let sum_b_total: f64 = informative_site_components_overall
            .iter()
            .map(|(_, b)| *b)
            .sum();
        // num_informative_sites is the count of sites that actually went into these sums.
        let num_sites_in_overall_sum = informative_site_components_overall.len();
        // Assertion: num_sites_in_overall_sum should equal sites_contributing_to_overall_sum if logic is correct.

        let denominator = sum_a_total + sum_b_total;
        let eps = 1e-9;

        if denominator > eps {
            FstEstimate::Calculable {
                value: sum_a_total / denominator,
                sum_a: sum_a_total,
                sum_b: sum_b_total,
                num_informative_sites: num_sites_in_overall_sum,
            }
        } else if denominator < -eps {
            FstEstimate::ComponentsYieldIndeterminateRatio {
                sum_a: sum_a_total,
                sum_b: sum_b_total,
                num_informative_sites: num_sites_in_overall_sum,
            }
        } else {
            // Denominator is effectively zero
            if sum_a_total.abs() > eps {
                // Non-zero numerator
                FstEstimate::Calculable {
                    value: sum_a_total / denominator, // Inf or -Inf
                    sum_a: sum_a_total,
                    sum_b: sum_b_total,
                    num_informative_sites: num_sites_in_overall_sum,
                }
            } else {
                // Numerator also effectively zero (0/0 from sum)
                // This state means that after summing all contributing sites, the net variance is zero.
                // 'sites_evaluated' here refers to the number of sites whose components were summed.
                FstEstimate::NoInterPopulationVariance {
                    sum_a: sum_a_total, // ~0.0
                    sum_b: sum_b_total, // ~0.0
                    sites_evaluated: num_sites_in_overall_sum,
                }
            }
        }
    };

    log(
        LogLevel::Info,
        &format!("Overall regional FST: {}", overall_fst_estimate),
    );

    // Pairwise FST calculation
    let mut pairwise_fst_estimates = HashMap::new();
    let mut aggregated_pairwise_variance_components = HashMap::new();

    for pair_key in all_observed_pair_keys {
        if let Some(components_vec) = informative_site_components_pairwise.get(&pair_key) {
            // This pair had at least one site contributing (non-InsufficientData) components for it.
            let sum_a_xy: f64 = components_vec.iter().map(|(a, _)| *a).sum();
            let sum_b_xy: f64 = components_vec.iter().map(|(_, b)| *b).sum();
            let num_informative_sites_for_pair = components_vec.len();

            aggregated_pairwise_variance_components.insert(pair_key.clone(), (sum_a_xy, sum_b_xy));

            let denominator_pair = sum_a_xy + sum_b_xy;
            let eps = 1e-9;

            let estimate_for_pair = if denominator_pair > eps {
                FstEstimate::Calculable {
                    value: sum_a_xy / denominator_pair,
                    sum_a: sum_a_xy,
                    sum_b: sum_b_xy,
                    num_informative_sites: num_informative_sites_for_pair,
                }
            } else if denominator_pair < -eps {
                FstEstimate::ComponentsYieldIndeterminateRatio {
                    sum_a: sum_a_xy,
                    sum_b: sum_b_xy,
                    num_informative_sites: num_informative_sites_for_pair,
                }
            } else {
                // Denominator is effectively zero
                if sum_a_xy.abs() > eps {
                    // Non-zero numerator
                    FstEstimate::Calculable {
                        value: sum_a_xy / denominator_pair, // Inf or -Inf
                        sum_a: sum_a_xy,
                        sum_b: sum_b_xy,
                        num_informative_sites: num_informative_sites_for_pair,
                    }
                } else {
                    // Numerator also effectively zero (0/0 from sum)
                    FstEstimate::NoInterPopulationVariance {
                        sum_a: sum_a_xy, // ~0.0
                        sum_b: sum_b_xy, // ~0.0
                        sites_evaluated: num_informative_sites_for_pair,
                    }
                }
            };
            pairwise_fst_estimates.insert(pair_key.clone(), estimate_for_pair);
            log(
                LogLevel::Info,
                &format!(
                    "Regional pairwise FST for {}: {}",
                    pair_key, estimate_for_pair
                ),
            );
        } else {
            // This pair_key was observed in some site's pairwise_variance_components map,
            // but none of those sites contributed actual components to informative_site_components_pairwise.
            // This implies for all sites where this pair was defined, its per-site FST was InsufficientData.
            // We count how many sites defined this pair (had an entry in pairwise_variance_components or pairwise_fst).
            let sites_attempted_for_this_pair = site_fst_values
                .iter()
                .filter(|s| {
                    s.pairwise_variance_components.contains_key(&pair_key)
                        || s.pairwise_fst.contains_key(&pair_key)
                })
                .count();
            pairwise_fst_estimates.insert(
                pair_key.clone(),
                FstEstimate::InsufficientDataForEstimation {
                    sum_a: 0.0,
                    sum_b: 0.0,
                    sites_attempted: sites_attempted_for_this_pair,
                },
            );
            aggregated_pairwise_variance_components.insert(pair_key.clone(), (0.0, 0.0)); // Store zero sums as components are not aggregated
            log(
                LogLevel::Info,
                &format!(
                    "Regional pairwise FST for {} (no informative components from sites): {}",
                    pair_key,
                    pairwise_fst_estimates.get(&pair_key).unwrap()
                ),
            );
        }
    }

    (
        overall_fst_estimate,
        pairwise_fst_estimates,
        aggregated_pairwise_variance_components,
    )
}

/// Calculates Dxy (average number of pairwise differences per site between two populations)
/// for Hudson's FST, as defined by Hudson et al. (1992) and elaborated by
/// de Jong et al. (2024). Dxy is the mean number of differences per site between sequences
/// sampled from two different populations.
///
/// The calculation sums the absolute number of differing sites between haplotype pairs
/// based only on the variants provided in the `popX_context.variants` slice. This sum
/// is then normalized by `(total_inter_population_pairs * popX_context.sequence_length)`.
/// It is crucial that the `popX_context.variants` slice accurately represents all and only
/// the variable sites within the genomic region whose total callable length is given by
/// `popX_context.sequence_length`. Monomorphic sites within this `sequence_length`
/// contribute zero to the sum of differences but are correctly accounted for by the
/// normalization factor `sequence_length`. A similar principle applies to `calculate_pi`.
///
/// This implementation iterates over all possible inter-population pairs of haplotypes,
/// sums the raw nucleotide differences for each pair across all provided variants,
/// and then normalizes by the total number of such pairs and the sequence length.
///
/// # Arguments
/// * `pop1_context` - A `PopulationContext` for the first population.
/// * `pop2_context` - A `PopulationContext` for the second population.
///
/// # Returns
/// A `Result` containing `DxyHudsonResult` which holds `Some(d_xy_value)` if successful,
/// or `None` within `DxyHudsonResult` if Dxy cannot be meaningfully calculated (e.g., no
/// haplotypes in one of the populations, or zero sequence length). Returns `Err(VcfError)`
/// for precondition violations like sequence length mismatch.
pub fn calculate_d_xy_hudson<'a>(
    pop1_context: &PopulationContext<'a>,
    pop2_context: &PopulationContext<'a>,
) -> Result<DxyHudsonResult, VcfError> {
    if pop1_context.sequence_length <= 0 {
        log(
            LogLevel::Error,
            "Cannot calculate Dxy: sequence_length must be positive.",
        );
        // This is a critical error in input setup
        return Err(VcfError::InvalidRegion(
            "Sequence length must be positive for Dxy calculation".to_string(),
        ));
    }

    if pop1_context.sequence_length != pop2_context.sequence_length {
        log(
            LogLevel::Error,
            "Sequence length mismatch between populations for Dxy calculation.",
        );
        return Err(VcfError::Parse(
            "Sequence length mismatch in Dxy calculation".to_string(),
        ));
    }

    if !variants_compatible(pop1_context.variants, pop2_context.variants) {
        return Err(VcfError::Parse(
            "Variant slices differ in positions/length for Dxy calculation".to_string(),
        ));
    }

    if pop1_context.haplotypes.is_empty() || pop2_context.haplotypes.is_empty() {
        log(LogLevel::Warning, &format!(
            "Cannot calculate Dxy for pops {:?}/{:?}: one or both have no haplotypes ({} and {} respectively).",
            pop1_context.id, pop2_context.id, pop1_context.haplotypes.len(), pop2_context.haplotypes.len()
        ));
        return Ok(DxyHudsonResult { d_xy: None });
    }

    // Use unbiased per-site aggregation approach
    let mut sum_dxy = 0.0;
    let mut variant_count = 0;

    for variant in pop1_context.variants {
        // Get allele counts for both populations at this variant
        let (n1, counts1) = freq_map_for_pop(variant, &pop1_context.haplotypes);
        let (n2, counts2) = freq_map_for_pop(variant, &pop2_context.haplotypes);

        // Calculate per-site Dxy using existing helper
        if let Some(dxy_site) = dxy_from_counts(n1, &counts1, n2, &counts2) {
            sum_dxy += dxy_site;
            variant_count += 1;
        }
        // Sites where either population has n=0 are skipped but contribute 0 to the sum
    }

    log(
        LogLevel::Debug,
        &format!(
            "Dxy calculation: processed {} variant sites with valid data",
            variant_count
        ),
    );

    // Final Dxy = sum of per-site Dxy values divided by sequence length
    // Monomorphic sites (including those not in variants list) contribute 0
    let effective_sequence_length = pop1_context.sequence_length as f64;
    let d_xy_value = if effective_sequence_length > 0.0 {
        Some(sum_dxy / effective_sequence_length)
    } else {
        log(LogLevel::Warning, &format!(
            "Invalid sequence length for Dxy calculation: {}",
            effective_sequence_length
        ));
        None
    };

    Ok(DxyHudsonResult { d_xy: d_xy_value })
}

/// Extract allele counts for a population at a specific variant site.
///
/// **Missing Data Handling Strategy:**
/// This function implements the "complete case analysis" approach for missing data:
/// - Only counts haplotypes with called genotypes at this site
/// - Missing genotypes (None) are excluded from frequency calculations
/// - Returns the number of successfully called haplotypes (n_called)
/// - Allele frequencies are computed as count/n_called using only available data
///
/// **Why This Approach:**
/// 1. **Unbiased estimation**: Using only called haplotypes gives unbiased allele frequencies
/// 2. **Site-specific sample sizes**: Each site can have different effective sample sizes
/// 3. **Robust to missing patterns**: Works regardless of missing data patterns
/// 4. **Conservative**: Sites with insufficient data will have low n_called and may be excluded
///
/// **Mathematical Impact:**
/// The resulting frequencies {p_a} are computed from n_called haplotypes, making
/// the downstream π and D_xy calculations appropriate for the actual available data.
fn freq_map_for_pop(
    variant: &Variant,
    haps: &[(usize, HaplotypeSide)],
) -> (usize, HashMap<i32, usize>) {
    let mut n_called = 0usize;
    let mut counts: HashMap<i32, usize> = HashMap::new();
    for &(s, side) in haps {
        if let Some(Some(g)) = variant.genotypes.get(s) {
            if let Some(&a) = g.get(side as usize) {
                n_called += 1;
                *counts.entry(a as i32).or_insert(0) += 1;
            }
        }
    }
    (n_called, counts)
}

/// Compute per-site nucleotide diversity (π) using the unbiased estimator.
///
/// **Mathematical Foundation:**
/// For a site with n called haplotypes and allele frequencies {p_a}, the unbiased
/// estimator of within-population diversity is:
///
///     π = (n/(n-1)) * (1 - Σ p_a²)
///
/// This corrects for finite sample size bias. The term (1 - Σ p_a²) is the
/// expected heterozygosity, and the n/(n-1) factor provides the unbiased correction
/// for haploid/haplotype data.
///
/// **Multi-allelic Support:**
/// Works correctly for any number of alleles by summing p_a² over all observed alleles.
///
/// **Missing Data Handling:**
/// Only uses called haplotypes at this site; n and {p_a} are computed from available data.
fn pi_from_counts(n: usize, counts: &HashMap<i32, usize>) -> Option<f64> {
    if n < 2 {
        return None;
    }
    let sum_p2 = counts
        .values()
        .map(|&c| {
            let p = c as f64 / n as f64;
            p * p
        })
        .sum::<f64>();
    Some((n as f64) / (n as f64 - 1.0) * (1.0 - sum_p2))
}

/// Compute between-population diversity (D_xy) from allele counts.
///
/// **Mathematical Foundation:**
/// For two populations with allele frequencies {p_1a} and {p_2a}, the between-population
/// diversity is the average pairwise difference between haplotypes from different populations:
///
///     D_xy = 1 - Σ_a (p_1a * p_2a)
///
/// This is Hudson's H_B term - the probability that two randomly chosen haplotypes
/// from different populations differ at this site.
///
/// **Multi-allelic Support:**
/// Correctly handles any number of alleles by computing the dot product of frequency vectors.
///
/// **Missing Data Handling:**
/// Uses only called haplotypes from each population at this site to compute frequencies.
fn dxy_from_counts(
    n1: usize,
    c1: &HashMap<i32, usize>,
    n2: usize,
    c2: &HashMap<i32, usize>,
) -> Option<f64> {
    if n1 == 0 || n2 == 0 {
        return None;
    }
    let mut dot = 0.0_f64;
    for (allele, &k1) in c1.iter() {
        if let Some(&k2) = c2.get(allele) {
            dot += (k1 as f64 / n1 as f64) * (k2 as f64 / n2 as f64);
        }
    }
    // Clamp to [0,1] to handle floating-point errors in multi-allelic tallies
    let dxy = 1.0 - dot;
    Some(dxy.max(0.0).min(1.0))
}

/// Compute per-site Hudson FST components from a single variant.
///
/// **Mathematical Foundation (Hudson et al. 1992):**
/// Hudson's FST at a single site is defined as:
///
///     FST_i = (D_xy,i - 0.5*(π_1,i + π_2,i)) / D_xy,i = (H_B - H_S) / H_B
///
/// Where:
/// - H_B = D_xy,i = 1 - Σ_a p_1a * p_2a = between-population diversity
/// - H_S = 0.5*(π_1,i + π_2,i) = average within-population diversity
/// - π_k,i = (n_k/(n_k-1)) * (1 - Σ_a p_ka²) = unbiased within-population estimator
///
/// **Literature Alignment:**
/// - **Hudson et al. (1992)**: Original definition using H_B and H_S
/// - **scikit-allel**: `average_hudson_fst` uses identical formula and ratio-of-sums aggregation
/// - **ANGSD**: Uses same per-site definition and weighted window estimator
/// - **Biallelic equivalence**: For 2 alleles, this equals (p₁-p₂)² minus finite-sample corrections
///   divided by D_xy = p₁(1-p₂) + p₂(1-p₁)
///
/// **Per-site Components:**
/// - Numerator: D_xy,i - 0.5*(π_1,i + π_2,i) = Hudson numerator with finite-sample corrections
/// - Denominator: D_xy,i = Hudson denominator  
/// - FST: numerator/denominator when denominator > FST_EPSILON
///
/// **Multi-allelic Support:**
/// All formulas use Σ_a notation, so they generalize correctly beyond biallelic SNPs.
/// D_xy = 1 - Σ_a p_1a * p_2a handles any number of alleles.
///
/// **Multi-allelic and Missing Data:**
/// Handles any number of alleles and computes frequencies from called haplotypes only.
fn hudson_site_from_variant(
    variant: &Variant,
    pop1_haps: &[(usize, HaplotypeSide)],
    pop2_haps: &[(usize, HaplotypeSide)],
) -> SiteFstHudson {
    let (n1, counts1) = freq_map_for_pop(variant, pop1_haps);
    let (n2, counts2) = freq_map_for_pop(variant, pop2_haps);

    let pi1 = pi_from_counts(n1, &counts1);
    let pi2 = pi_from_counts(n2, &counts2);
    let dxy = dxy_from_counts(n1, &counts1, n2, &counts2);

    let (fst, num_c, den_c) = match (dxy, pi1, pi2) {
        (Some(d), Some(p1), Some(p2)) => {
            if d > FST_EPSILON {
                let num = d - 0.5 * (p1 + p2);
                (Some(num / d), Some(num), Some(d))
            } else {
                let pi_avg = 0.5 * (p1 + p2);
                if pi_avg.abs() <= FST_EPSILON {
                    // Both D_xy and average π are effectively zero - monomorphic site
                    (Some(0.0), Some(0.0), Some(0.0))
                } else {
                    // D_xy ≈ 0 but π > 0 - undefined FST
                    (None, None, None)
                }
            }
        }
        _ => (None, None, None),
    };

    SiteFstHudson {
        position: ZeroBasedPosition(variant.position).to_one_based(),
        fst,
        d_xy: dxy,
        pi_pop1: pi1,
        pi_pop2: pi2,
        n1_called: n1,
        n2_called: n2,
        num_component: num_c,
        den_component: den_c,
    }
}

/// Calculate Hudson FST components on a per-site basis across a region.
///
/// **IMPORTANT**: This function assumes variant compatibility between populations.
/// For safe usage, prefer `calculate_hudson_fst_for_pair_with_sites` which includes
/// proper compatibility checks and error handling.
pub fn calculate_hudson_fst_per_site(
    pop1_context: &PopulationContext,
    pop2_context: &PopulationContext,
    region: QueryRegion,
) -> Vec<SiteFstHudson> {
    // Guard against basic misuse - variant compatibility check
    if !variants_compatible(pop1_context.variants, pop2_context.variants) {
        log(
            LogLevel::Error,
            "Variant slices differ between populations in calculate_hudson_fst_per_site. Use calculate_hudson_fst_for_pair_with_sites for safe usage.",
        );
        // Return empty vector rather than panicking
        return Vec::new();
    }

    if pop1_context.sequence_length != region.len() as i64 {
        log(
            LogLevel::Warning,
            &format!(
                "sequence_length ({}) != region length ({}). Ensure you used calculate_adjusted_sequence_length for L.",
                pop1_context.sequence_length,
                region.len()
            ),
        );
    }

    let variant_map: HashMap<i64, &Variant> = pop1_context
        .variants
        .iter()
        .map(|v| (v.position, v))
        .collect();
    let mut sites = Vec::with_capacity(region.len() as usize);

    for pos in region.start..=region.end {
        if let Some(variant) = variant_map.get(&pos) {
            sites.push(hudson_site_from_variant(
                variant,
                &pop1_context.haplotypes,
                &pop2_context.haplotypes,
            ));
        } else {
            sites.push(SiteFstHudson {
                position: ZeroBasedPosition(pos).to_one_based(),
                fst: None,
                d_xy: None,
                pi_pop1: None,
                pi_pop2: None,
                n1_called: 0,
                n2_called: 0,
                num_component: None,
                den_component: None,
            });
        }
    }

    sites
}

/// Aggregate per-site Hudson components into a window/regional FST using ratio of sums.
///
/// **Mathematical Foundation:**
/// The recommended window-level Hudson FST estimator is the "ratio of sums":
///
///     FST_window = Σ_i [D_xy,i - 0.5*(π_1,i + π_2,i)] / Σ_i D_xy,i
///
/// This is a **weighted average** where sites with higher D_xy contribute more weight.
///
/// **Literature Alignment:**
/// - **scikit-allel**: `windowed_hudson_fst` uses identical ratio-of-sums aggregation
/// - **ANGSD**: Uses same weighted estimator for window-level FST
/// - **PopGen consensus**: Preferred over "mean of ratios" in methodological reviews
/// - **Bhatia et al. (2013)**: Recommends keeping negative values (no truncation at 0)
///
/// **Why Ratio-of-Sums (not Mean-of-Ratios):**
/// 1. **Stability**: More robust to near-monomorphic sites with tiny denominators
/// 2. **Weighting**: Sites with higher diversity naturally get more influence
/// 3. **Statistical properties**: Better maximum likelihood properties under certain models
/// 4. **Tool compatibility**: Matches mainstream population genetics software
///
/// **Missing Data Robustness:**
/// Sites with undefined components (None, None) are excluded from both sums,
/// ensuring unbiased estimates regardless of missing data patterns.
///
/// **Monomorphic Sites:**
/// Sites with D_xy = π = 0 contribute (0, 0) to the sums, which is mathematically correct.
pub fn aggregate_hudson_from_sites(sites: &[SiteFstHudson]) -> Option<f64> {
    let mut num_sum = 0.0_f64;
    let mut den_sum = 0.0_f64;
    for s in sites {
        if let (Some(nc), Some(dc)) = (s.num_component, s.den_component) {
            num_sum += nc;
            den_sum += dc;
        }
    }
    if den_sum > FST_EPSILON {
        Some(num_sum / den_sum)
    } else if num_sum.abs() <= FST_EPSILON {
        // Both numerator and denominator sums are effectively zero
        Some(0.0)
    } else {
        // Denominator is zero but numerator is not - undefined FST
        None
    }
}

/// Computes Hudson's FST and its intermediate components (pi_xy_avg)
/// from pre-calculated within-population diversities (pi_pop1, pi_pop2)
/// and between-population diversity (Dxy).
///
/// # Arguments
/// * `pop1_id` - Identifier for the first population.
/// * `pop2_id` - Identifier for the second population.
/// * `pi_pop1` - `Option<f64>` for nucleotide diversity of population 1.
/// * `pi_pop2` - `Option<f64>` for nucleotide diversity of population 2.
/// * `d_xy_result` - Result of Dxy calculation (`DxyHudsonResult`).
///
/// # Returns
/// An `HudsonFSTOutcome` struct containing all components. Values will be `None`
/// if they cannot be robustly calculated from the inputs.
pub fn compute_hudson_fst_outcome(
    pop1_id: PopulationId,
    pop2_id: PopulationId,
    pi_pop1: Option<f64>,
    pi_pop2: Option<f64>,
    d_xy_result: &DxyHudsonResult,
) -> HudsonFSTOutcome {
    let mut outcome = HudsonFSTOutcome {
        pop1_id: Some(pop1_id),
        pop2_id: Some(pop2_id),
        pi_pop1,
        pi_pop2,
        d_xy: d_xy_result.d_xy,
        ..Default::default() // Initializes fst and pi_xy_avg to None
    };

    if let (Some(p1), Some(p2)) = (outcome.pi_pop1, outcome.pi_pop2) {
        // p1 and p2 are finite before averaging
        if p1.is_finite() && p2.is_finite() {
            outcome.pi_xy_avg = Some(0.5 * (p1 + p2));
        } else {
            log(
                LogLevel::Warning,
                "One or both Pi values are non-finite, cannot calculate Pi_xy_avg.",
            );
        }
    } else {
        log(
            LogLevel::Debug,
            "One or both Pi values are None, cannot calculate Pi_xy_avg.",
        );
    }

    if let (Some(dxy_val), Some(pi_xy_avg_val)) = (outcome.d_xy, outcome.pi_xy_avg) {
        // dxy_val and pi_xy_avg_val are finite and dxy_val is positive for division
        if dxy_val.is_finite() && pi_xy_avg_val.is_finite() {
            if dxy_val > FST_EPSILON {
                // Use FST_EPSILON to avoid division by effective zero
                outcome.fst = Some((dxy_val - pi_xy_avg_val) / dxy_val);
            } else if dxy_val >= 0.0 && (dxy_val - pi_xy_avg_val).abs() < FST_EPSILON {
                // Case: Dxy is ~0 and Pi_xy_avg is also ~0 (or Dxy approx equals Pi_xy_avg)
                // This implies no differentiation and possibly no variation. FST is 0.
                outcome.fst = Some(0.0);
            } else {
                // Dxy is effectively zero or negative, but Pi_xy_avg is substantially different,
                // or Dxy is non-finite.
                log(LogLevel::Warning, &format!(
                    "Cannot calculate Hudson FST: Dxy ({:.4e}) is too small or invalid relative to Pi_xy_avg ({:.4e}).",
                    dxy_val, pi_xy_avg_val
                ));
                outcome.fst = None;
            }
        } else {
            log(
                LogLevel::Warning,
                "Dxy or Pi_xy_avg is non-finite, cannot calculate Hudson FST.",
            );
        }
    } else {
        log(
            LogLevel::Debug,
            "Dxy or Pi_xy_avg is None, cannot calculate Hudson FST.",
        );
    }

    outcome
}

fn variants_compatible(a: &[Variant], b: &[Variant]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.position == y.position)
}

/// Core implementation for Hudson's FST calculation between two populations.
///
/// **Algorithm Overview:**
/// 1. **Per-site calculation**: For each variant site, compute Hudson FST components
///    using the unbiased estimators for π and D_xy
/// 2. **Regional aggregation**: Use "ratio of sums" to combine per-site components
///    into a single window-level FST estimate
///
/// **Mathematical Approach:**
/// - Per-site: FST_i = (D_xy_i - 0.5*(π_1i + π_2i)) / D_xy_i
/// - Regional: FST_region = Σ(numerator_i) / Σ(denominator_i)
///
/// **Why Ratio of Sums:**
/// This weighted approach is more robust than averaging per-site FST values because:
/// - Sites with higher diversity contribute more weight (appropriate for FST)
/// - Avoids instability from sites with very low diversity
/// - Matches standard implementations (ANGSD, scikit-allel)
/// - Provides better statistical properties under missing data
///
/// **Missing Data Strategy:**
/// This implementation uses a robust "complete case per site" approach:
/// 1. **Per-site analysis**: Each site uses only haplotypes with called genotypes
/// 2. **Site-specific sample sizes**: n1 and n2 can vary by site based on available data
/// 3. **Exclusion of undefined sites**: Sites with insufficient data (n < 2 in either pop)
///    contribute (None, None) components and are excluded from regional sums
/// 4. **Unbiased aggregation**: Regional FST uses only sites with valid components
///
/// **Advantages over alternatives:**
/// - More robust than listwise deletion (excluding samples with any missing data)
/// - Avoids bias from imputation methods
/// - Naturally handles different missing data patterns across sites
/// - Maintains statistical validity by using appropriate sample sizes per site
fn calculate_hudson_fst_for_pair_core<'a>(
    pop1_context: &PopulationContext<'a>,
    pop2_context: &PopulationContext<'a>,
    region: Option<QueryRegion>,
) -> Result<(HudsonFSTOutcome, Vec<SiteFstHudson>), VcfError> {
    if pop1_context.sequence_length != pop2_context.sequence_length {
        return Err(VcfError::Parse(
            "Sequence length mismatch between population contexts for Hudson FST calculation.".to_string(),
        ));
    }
    if !variants_compatible(pop1_context.variants, pop2_context.variants) {
        return Err(VcfError::Parse(
            "Variant slices differ in positions/length.".to_string(),
        ));
    }

    // Calculate per-site Hudson FST values
    let site_values = if let Some(reg) = region {
        calculate_hudson_fst_per_site(pop1_context, pop2_context, reg)
    } else {
        // For non-regional calculations, process only variant positions (performance optimization)
        if pop1_context.variants.is_empty() {
            Vec::new()
        } else {
            // Process only variant positions instead of creating a full region
            pop1_context.variants.iter().map(|variant| {
                hudson_site_from_variant(
                    variant,
                    &pop1_context.haplotypes,
                    &pop2_context.haplotypes,
                )
            }).collect()
        }
    };

    // Compute regional FST using unbiased ratio-of-sums from per-site components
    let regional_fst = aggregate_hudson_from_sites(&site_values);

    // Calculate auxiliary π and Dxy values for output (but don't use for FST)
    let pi1_raw = calculate_pi(
        pop1_context.variants,
        &pop1_context.haplotypes,
        pop1_context.sequence_length,
    );
    let pi1_opt = if pi1_raw.is_finite() {
        Some(pi1_raw)
    } else {
        None
    };

    let pi2_raw = calculate_pi(
        pop2_context.variants,
        &pop2_context.haplotypes,
        pop2_context.sequence_length,
    );
    let pi2_opt = if pi2_raw.is_finite() {
        Some(pi2_raw)
    } else {
        None
    };

    let dxy_result = calculate_d_xy_hudson(pop1_context, pop2_context)?;

    // Create outcome with unbiased FST from per-site aggregation
    let mut outcome = HudsonFSTOutcome {
        pop1_id: Some(pop1_context.id.clone()),
        pop2_id: Some(pop2_context.id.clone()),
        pi_pop1: pi1_opt,
        pi_pop2: pi2_opt,
        d_xy: dxy_result.d_xy,
        fst: regional_fst, // Use unbiased per-site aggregation as single source of truth
        ..Default::default()
    };

    // Calculate pi_xy_avg for auxiliary output
    if let (Some(p1), Some(p2)) = (outcome.pi_pop1, outcome.pi_pop2) {
        if p1.is_finite() && p2.is_finite() {
            outcome.pi_xy_avg = Some(0.5 * (p1 + p2));
        }
    }

    Ok((outcome, site_values))
}

/// Calculates Hudson's FST for a pair of populations, returning both regional outcome
/// and per-site values for the specified region.
///
/// **Primary Use Case:**
/// This function is the main entry point for per-site Hudson FST analysis. It returns
/// both the aggregated regional FST estimate and detailed per-site components that can
/// be used for:
/// - Writing per-site FST values to FALSTA output files
/// - Quality control and validation of regional estimates
/// - Fine-scale analysis of FST variation across sites
///
/// **Mathematical Guarantee:**
/// The regional FST in the returned HudsonFSTOutcome equals the ratio-of-sums
/// aggregation of the per-site components: Σ(numerator_i) / Σ(denominator_i)
///
/// **Performance Note:**
/// Computing per-site values has minimal overhead since the regional calculation
/// already processes each site individually.
pub fn calculate_hudson_fst_for_pair_with_sites<'a>(
    pop1_context: &PopulationContext<'a>,
    pop2_context: &PopulationContext<'a>,
    region: QueryRegion,
) -> Result<(HudsonFSTOutcome, Vec<SiteFstHudson>), VcfError> {
    calculate_hudson_fst_for_pair_core(pop1_context, pop2_context, Some(region))
}

/// Backwards-compatible wrapper returning only the regional HudsonFSTOutcome.
///
/// **Use Case:**
/// For analyses that only need the regional FST estimate without per-site details.
/// This is computationally equivalent to the full function but discards per-site data.
///
/// **Mathematical Equivalence:**
/// Returns the same regional FST as calculate_hudson_fst_for_pair_with_sites,
/// computed using the identical ratio-of-sums approach.
pub fn calculate_hudson_fst_for_pair<'a>(
    pop1_context: &PopulationContext<'a>,
    pop2_context: &PopulationContext<'a>,
) -> Result<HudsonFSTOutcome, VcfError> {
    calculate_hudson_fst_for_pair_core(pop1_context, pop2_context, None).map(|(o, _)| o)
}

// Calculate the effective sequence length after adjusting for allowed and masked regions
pub fn calculate_adjusted_sequence_length(
    region_start: i64, // Start of the genomic region (1-based, inclusive)
    region_end: i64,   // End of the genomic region (1-based, inclusive)
    allow_regions_chr: Option<&Vec<(i64, i64)>>, // Optional list of allowed regions as (start, end) tuples
    mask_regions_chr: Option<&Vec<(i64, i64)>>,  // Optional list of masked regions to exclude
) -> i64 {
    // Returns the adjusted length as an i64
    log(
        LogLevel::Info,
        &format!(
            "Calculating adjusted sequence length for region {}:{}-{}",
            if allow_regions_chr.is_some() {
                "with allow regions"
            } else {
                "full"
            },
            region_start,
            region_end
        ),
    );

    let spinner = create_spinner("Adjusting sequence length");

    // Convert the input region to a ZeroBasedHalfOpen interval
    let region = ZeroBasedHalfOpen::from_1based_inclusive(region_start, region_end);

    // Initialize a vector to store intervals that are allowed after intersecting with allow_regions_chr
    let mut allowed_intervals = Vec::new();
    if let Some(allow_regions) = allow_regions_chr {
        // If allowed regions are provided, intersect the input region with each allowed region
        for &(start, end) in allow_regions {
            // Convert each allowed region to ZeroBasedHalfOpen for consistent interval operations
            let allow_region = ZeroBasedHalfOpen::from_1based_inclusive(start, end);

            // Find the overlapping section between the input region and the allowed region
            if let Some(overlap) = region.intersect(&allow_region) {
                // Convert the overlap back to a 1-based inclusive tuple and store it
                allowed_intervals.push(overlap.to_1based_inclusive_tuple());
            }
        }
    } else {
        // If no allowed regions are specified, the entire input region is considered allowed
        allowed_intervals.push((region_start, region_end));
    }

    // Subtract any masked regions from the allowed intervals to get the final unmasked intervals
    let unmasked_intervals = subtract_regions(&allowed_intervals, mask_regions_chr.map(|v| v.as_slice()));

    // Calculate the total length of all unmasked intervals
    let adjusted_length: i64 = unmasked_intervals
        .iter() // Iterate over each unmasked interval
        .map(|&(start, end)| {
            // Convert the interval back to ZeroBasedHalfOpen to use its length method
            let interval = ZeroBasedHalfOpen::from_1based_inclusive(start, end);
            interval.len() as i64 // Get the length and cast to i64
        })
        .sum(); // Sum all lengths to get the total adjusted length

    // Display results and finish spinner
    spinner.finish_and_clear();
    log(
        LogLevel::Info,
        &format!(
            "Original length: {}, Adjusted length: {}",
            region_end - region_start + 1,
            adjusted_length
        ),
    );

    log(
        LogLevel::Info,
        &format!(
            "Adjusted sequence length: {} (original: {})",
            adjusted_length,
            region_end - region_start + 1
        ),
    );

    adjusted_length // Return the computed length
}

// Helper function to subtract masked regions from a set of intervals
fn subtract_regions(
    intervals: &[(i64, i64)],
    masks: Option<&[(i64, i64)]>,
) -> Vec<(i64, i64)> {
    let Some(masks) = masks else { return intervals.to_vec() };
    let mut out = Vec::new();

    for &(a_start, a_end) in intervals {
        let mut parts = vec![(a_start, a_end)];
        for &(m_start, m_end) in masks {
            let mut next = Vec::new();
            for (s, e) in parts {
                if m_end < s || m_start > e {
                    next.push((s, e));
                    continue;
                }
                if m_start > s {
                    let left_end = m_start - 1;
                    if left_end >= s {
                        next.push((s, left_end));
                    }
                }
                if m_end < e {
                    let right_start = m_end + 1;
                    if right_start <= e {
                        next.push((right_start, e));
                    }
                }
            }
            parts = next;
            if parts.is_empty() {
                break;
            }
        }
        out.extend(parts);
    }
    out
}

// Calculate the frequency of allele 1 (e.g., an inversion allele) across haplotypes
pub fn calculate_inversion_allele_frequency(
    sample_filter: &HashMap<String, (u8, u8)>, // Map of sample names to (haplotype1, haplotype2) alleles - order is arbitrary
) -> Option<f64> {
    // Returns Some(frequency) or None if no haplotypes are present
    let mut num_ones = 0; // Counter for haplotypes with allele 1
    let mut total_haplotypes = 0; // Total number of haplotypes (with allele 0 or 1)
    
    for (_sample, &(hap1, hap2)) in sample_filter.iter() {
        // Count each haplotype exactly once if it's 0 or 1
        for allele in [hap1, hap2] {
            if allele == 0 || allele == 1 {
                total_haplotypes += 1;
                if allele == 1 {
                    num_ones += 1;
                }
            }
            // Alleles other than 0 or 1 (e.g., missing or bad data) are ignored
        }
    }
    
    if total_haplotypes > 0 {
        // Calculate frequency as the proportion of allele 1 among all counted haplotypes
        Some(num_ones as f64 / total_haplotypes as f64)
    } else {
        // No valid haplotypes (all alleles might be missing or invalid); return None
        None
    }
}

// Count the number of segregating sites, where a site has more than one allele
pub fn count_segregating_sites(variants: &[Variant]) -> usize {
    // Returns the count as usize
    variants
        .par_iter() // Use parallel iteration for efficiency over the variants slice
        .filter(|v| {
            // Collect all alleles across all genotypes into a HashSet to find unique alleles
            let alleles: HashSet<_> = v.genotypes.iter().flatten().flatten().collect();
            alleles.len() > 1 // True if the site is segregating (has multiple alleles)
        })
        .count() // Count the number of segregating sites
}

// Calculate pairwise differences and comparable sites between all sample pairs
/// This function computes, for each pair of samples, the number of differences across comparable
/// haplotypes, treating each haplotype separately (per-haplotype analysis).
///
/// **Per-Haplotype Analysis:**
/// For each sample pair and each variant site, compares aligned haplotypes:
/// - Left haplotype of sample i vs left haplotype of sample j
/// - Right haplotype of sample i vs right haplotype of sample j
/// - Counts each comparable haplotype separately
///
/// # Arguments
/// * variants - A slice of Variant structs containing genotype data for all samples
/// * number_of_samples - The total number of samples to compare
///
/// # Returns
/// A vector of tuples, each containing:
/// * (sample_idx_i, sample_idx_j) - Indices of the sample pair
/// * difference_count - Number of haplotype differences across all comparable sites
/// * comparable_site_count - Number of comparable haplotypes across all sites
pub fn calculate_pairwise_differences(
    variants: &[Variant],
    number_of_samples: usize,
) -> Vec<((usize, usize), usize, usize)> {
    set_stage(ProcessingStage::StatsCalculation);

    let total_pairs = (number_of_samples * (number_of_samples - 1)) / 2;
    log(
        LogLevel::Info,
        &format!(
            "Calculating pairwise differences across {} samples ({} pairs)",
            number_of_samples, total_pairs
        ),
    );

    let spinner = create_spinner(&format!(
        "Processing pairwise differences for {} samples",
        number_of_samples
    ));

    // Wrap variants in an Arc for thread-safe sharing across parallel threads
    let variants_shared = Arc::new(variants);

    let result: Vec<((usize, usize), usize, usize)> = (0..number_of_samples)
        .into_par_iter() // Convert range into a parallel iterator
        .flat_map(|sample_idx_i| {
            // Clone the Arc for each thread to safely access the variants data
            let variants_local = Arc::clone(&variants_shared);
            // Parallel iteration over second sample indices (i+1 to n-1) to avoid duplicate pairs
            (sample_idx_i + 1..number_of_samples)
                .into_par_iter()
                .map(move |sample_idx_j| {
                    let mut difference_count = 0; // Number of sites where genotypes differ
                    let mut comparable_site_count = 0; // Number of sites with data for both samples

                    // Iterate over all variants to compare this pair's haplotypes
                    for variant in variants_local.iter() {
                        if let (Some(genotype_i), Some(genotype_j)) = (
                            &variant.genotypes[sample_idx_i],
                            &variant.genotypes[sample_idx_j],
                        ) {
                            // Compare all haplotype pairs (truly per-haplotype analysis)
                            // Each haplotype is treated as completely independent
                            for a in 0..genotype_i.len() {
                                for b in 0..genotype_j.len() {
                                    comparable_site_count += 1;
                                    if genotype_i[a] != genotype_j[b] {
                                        difference_count += 1;
                                    }
                                }
                            }
                        }
                        // If either genotype is None, skip this site (missing data)
                    }

                    // Return the pair's indices and their comparison metrics
                    (
                        (sample_idx_i, sample_idx_j),
                        difference_count,
                        comparable_site_count,
                    )
                })
                .collect::<Vec<_>>() // Collect results for this sample_idx_i
        })
        .collect(); // Collect all pair results into the final vector

    let result_count = result.len();
    spinner.finish_and_clear();
    log(
        LogLevel::Info,
        &format!("Computed {} pairwise comparisons", result_count),
    );

    result
}

// Calculate the harmonic number H_n = sum_{k=1}^n 1/k
pub fn harmonic(n: usize) -> f64 {
    // Returns the harmonic number as a float
    (1..=n) // Range from 1 to n inclusive
        .map(|i| 1.0 / i as f64) // Map each integer k to 1/k as a float
        .sum() // Sum all terms to get H_n
}

// Calculate Watterson's theta (θ_w), a measure of genetic diversity
pub fn calculate_watterson_theta(seg_sites: usize, n: usize, seq_length: i64) -> f64 {
    // Handle edge cases where computation isn't meaningful.
    // Theta_w = S / (a_n * L), where a_n = H_{n-1} (harmonic number for n-1 samples).
    if n <= 1 {
        // a_n (H_{n-1}) is undefined or zero if n=1, or problematic if n=0.
        log(
            LogLevel::Warning,
            &format!(
                "Cannot calculate Watterson's theta: {} haplotypes (n <= 1). S={}, L={}",
                n, seg_sites, seq_length
            ),
        );
        if seg_sites == 0 {
            return f64::NAN;
        }
        // Indeterminate (0/0 type situation)
        else {
            return f64::INFINITY;
        } // S/0 type situation
    }
    if seq_length <= 0 {
        // Denominator L is zero or negative.
        log(
            LogLevel::Warning,
            &format!(
                "Cannot calculate Watterson's theta: sequence length {} (L <= 0). S={}, n={}",
                seq_length, seg_sites, n
            ),
        );
        if seg_sites == 0 {
            return f64::NAN;
        }
        // Indeterminate (0/0 type situation)
        else {
            return f64::INFINITY;
        } // S/0 type situation
    }

    // Calculate the harmonic number H_{n-1}, used as the denominator factor a_n.
    // Since n > 1 at this point, n-1 >= 1, so harmonic(n-1) will be > 0.
    let harmonic_value = harmonic(n - 1);
    // The check for harmonic_value == 0.0 below should ideally not be strictly necessary now
    // if n > 1, as harmonic(k) for k>=1 is always positive.
    // However, keeping it as a safeguard for extreme float precision issues, though unlikely.
    if harmonic_value <= 1e-9 {
        // Using an epsilon for safety with float comparison
        // This case should be rare if n > 1.
        log(LogLevel::Error, &format!( // Error because this indicates an unexpected issue if n > 1
            "Harmonic value (a_n) is unexpectedly near zero ({}) for Watterson's theta calculation with n={}. S={}, L={}",
            harmonic_value, n, seg_sites, seq_length
        ));
        if seg_sites == 0 {
            return f64::NAN;
        } else {
            return f64::INFINITY;
        }
    }

    // Watterson's theta formula: θ_w = S / (a_n * L)
    // S = number of segregating sites, a_n = H_{n-1}, L = sequence length
    let theta = seg_sites as f64 / harmonic_value / seq_length as f64;

    log(
        LogLevel::Debug,
        &format!(
            "Watterson's theta: {} (from {} segregating sites, {} haplotypes, {} length)",
            theta, seg_sites, n, seq_length
        ),
    );

    theta
}

// Calculate nucleotide diversity (π) across all sites, accounting for missing data
/// Computes π as the average pairwise difference per site across all haplotype pairs,
/// handling missing data by only considering sites where both haplotypes have alleles.
///
/// # Returns
/// * f64::NAN if fewer than 2 haplotypes
/// * f64::NAN if no valid pairs exist
/// * Otherwise, the average π across all sites
pub fn calculate_pi(
    variants: &[Variant],
    haplotypes_in_group: &[(usize, HaplotypeSide)],
    seq_length: i64,
) -> f64 {
    if haplotypes_in_group.len() <= 1 {
        log(
            LogLevel::Warning,
            &format!(
                "Cannot calculate pi: insufficient haplotypes ({})",
                haplotypes_in_group.len()
            ),
        );
        return f64::NAN;
    }

    if seq_length <= 0 {
        log(
            LogLevel::Warning,
            &format!(
                "Cannot calculate pi: invalid sequence length ({})",
                seq_length
            ),
        );
        return f64::NAN;
    }

    let spinner = create_spinner(&format!(
        "Calculating π for {} haplotypes over {} bp using unbiased per-site aggregation",
        haplotypes_in_group.len(),
        seq_length
    ));

    // Use unbiased per-site aggregation approach
    let mut sum_pi = 0.0;
    let mut variant_count = 0;

    for variant in variants {
        // Get allele counts for this variant using existing helper
        let (n_called, allele_counts) = freq_map_for_pop(variant, haplotypes_in_group);

        // Calculate per-site π using existing helper
        if let Some(pi_site) = pi_from_counts(n_called, &allele_counts) {
            sum_pi += pi_site;
            variant_count += 1;
        }
        // Monomorphic sites contribute 0 implicitly (not added to sum_pi)
    }

    // Final π = sum of per-site π values divided by sequence length
    // Monomorphic sites (including those not in variants list) contribute 0
    let pi = sum_pi / seq_length as f64;

    spinner.finish_and_clear();
    log(
        LogLevel::Info,
        &format!(
            "π = {:.6} (from {} variant sites over {} bp total length)",
            pi, variant_count, seq_length
        ),
    );

    pi
}

// Calculate per-site diversity metrics (π and Watterson's θ) across a genomic region
pub fn calculate_per_site_diversity(
    variants: &[Variant],
    haplotypes_in_group: &[(usize, HaplotypeSide)],
    region: QueryRegion, // Inclusive range [start..end] in 0-based coordinates
) -> Vec<SiteDiversity> {
    // Returns a vector of SiteDiversity structs
    set_stage(ProcessingStage::StatsCalculation);

    let start_time = std::time::Instant::now();
    log(
        LogLevel::Info,
        &format!(
            "Calculating per-site diversity for region {}:{}-{} with {} haplotypes",
            region.start,
            region.end,
            region.len(),
            haplotypes_in_group.len()
        ),
    );

    let max_haps = haplotypes_in_group.len(); // Number of haplotypes in the group

    let region_length = region.len();

    // Pre-allocate with correct capacity for better memory efficiency
    let mut site_diversities = Vec::with_capacity(region_length as usize);

    if max_haps < 2 {
        // Need at least 2 haplotypes for diversity; return empty vector if not
        log(
            LogLevel::Warning,
            "Insufficient haplotypes (<2) for diversity calculation",
        );
        return site_diversities;
    }

    // Initialize progress with more informative message
    let spinner = create_spinner(&format!(
        "Preparing to analyze {} positions for {} haplotypes",
        region_length, max_haps
    ));

    // Build a map of variants by position for O(1) lookup
    let variant_map: HashMap<i64, &Variant> = variants.iter().map(|v| (v.position, v)).collect();
    spinner.finish_and_clear();
    log(
        LogLevel::Info,
        &format!(
            "Indexed {} variants for fast lookup ({}ms)",
            variant_map.len(),
            start_time.elapsed().as_millis()
        ),
    );

    // Initialize detailed progress tracking with checkpoints
    init_step_progress(
        &format!("Calculating diversity across {} positions", region_length),
        region_length as u64,
    );

    // Track statistics for progress updates and performance monitoring
    let mut variants_processed = 0;
    let mut polymorphic_sites = 0;
    let mut last_update_time = std::time::Instant::now();
    let mut positions_since_update = 0;
    let update_interval = std::cmp::min(1000, region_length as usize / 100);

    // Process in batches for more efficient update frequency
    for (idx, pos) in (region.start..=region.end).enumerate() {
        // Inclusive range
        positions_since_update += 1;

        // Update progress sometimes
        if positions_since_update >= update_interval || idx == 0 || idx as i64 == region_length - 1
        {
            let elapsed = last_update_time.elapsed();
            let positions_per_sec = if elapsed.as_millis() > 0 {
                positions_since_update as f64 * 1000.0 / elapsed.as_millis() as f64
            } else {
                0.0
            };

            let progress_pct = (idx as f64 / region_length as f64) * 100.0;
            let remaining_secs = if positions_per_sec > 0.0 {
                (region_length as f64 - idx as f64) / positions_per_sec
            } else {
                0.0
            };

            update_step_progress(
                idx as u64,
                &format!(
                    "Position {}/{} ({:.1}%) - {:.1} pos/sec - ~{:.0}s remaining",
                    idx, region_length, progress_pct, positions_per_sec, remaining_secs
                ),
            );

            positions_since_update = 0;
            last_update_time = std::time::Instant::now();
        }

        // Process the current position
        if let Some(var) = variant_map.get(&pos) {
            // Variant exists at this position; compute diversity metrics
            let mut allele_counts = HashMap::new(); // Map to count occurrences of each allele
            let mut total_called = 0; // Number of haplotypes with non-missing alleles

            for &(sample_index, side) in haplotypes_in_group {
                if let Some(gt_opt) = var.genotypes.get(sample_index) {
                    if let Some(gt) = gt_opt {
                        if let Some(&allele) = gt.get(side as usize) {
                            // Increment count for this allele and total called haplotypes
                            allele_counts
                                .entry(allele)
                                .and_modify(|count| *count += 1)
                                .or_insert(1);
                            total_called += 1;
                        }
                    }
                }
            }

            let (pi_value, watterson_value) = if total_called < 2 {
                // Fewer than 2 haplotypes with data; diversity is 0
                (0.0, 0.0)
            } else {
                // Compute sum of squared allele frequencies for π calculation
                let mut freq_sq_sum = 0.0;
                for count in allele_counts.values() {
                    let freq = *count as f64 / total_called as f64;
                    freq_sq_sum += freq * freq;
                }
                // Calculate π with the unbiased estimator: (n / (n-1)) * (1 - Σ p_i^2)
                let pi_value =
                    (total_called as f64 / (total_called as f64 - 1.0)) * (1.0 - freq_sq_sum);

                // Calculate Watterson's θ for this site
                let distinct_alleles = allele_counts.len();
                let watterson_value = if distinct_alleles > 1 {
                    // Site is polymorphic; θ_w = 1 / H_{n-1}
                    let denom = harmonic(total_called - 1);
                    if denom == 0.0 {
                        0.0
                    } else {
                        1.0 / denom
                    }
                } else {
                    0.0 // Monomorphic site; θ_w = 0
                };
                (pi_value, watterson_value)
            };

            // Add the diversity metrics for this site to the vector
            if pi_value > 0.0 || watterson_value > 0.0 {
                polymorphic_sites += 1;
            }

            site_diversities.push(SiteDiversity {
                position: ZeroBasedPosition(pos).to_one_based(), // Convert to 1-based for output
                pi: pi_value,
                watterson_theta: watterson_value,
            });

            variants_processed += 1;
        } else {
            // No variant at this position; it's monomorphic (all same allele)
            site_diversities.push(SiteDiversity {
                position: ZeroBasedPosition(pos).to_one_based(), // Convert to 1-based
                pi: 0.0,                                         // No diversity since no variation
                watterson_theta: 0.0,                            // No segregating site
            });
        }
    }

    let total_time = start_time.elapsed();
    // Finish progress and display summary statistics
    finish_step_progress(&format!(
        "Completed: {} positions, {} variants, {} polymorphic sites in {:.2}s",
        region_length,
        variants_processed,
        polymorphic_sites,
        total_time.as_secs_f64()
    ));

    log(
        LogLevel::Info,
        &format!(
            "Per-site diversity calculation complete: {} positions analyzed, {} polymorphic sites",
            region_length, polymorphic_sites
        ),
    );

    // Show detailed summary in status box with performance metrics
    display_status_box(StatusBox {
        title: "Per-Site Diversity Summary".to_string(),
        stats: vec![
            (
                String::from("Region"),
                format!(
                    "{}:{}-{}",
                    ZeroBasedPosition(region.start).to_one_based(),
                    ZeroBasedPosition(region.end).to_one_based(),
                    region_length
                ),
            ),
            (String::from("Haplotypes"), max_haps.to_string()),
            (
                String::from("Variants processed"),
                variants_processed.to_string(),
            ),
            (
                String::from("Polymorphic sites"),
                format!(
                    "{} ({:.2}%)",
                    polymorphic_sites,
                    if region_length > 0 {
                        (polymorphic_sites as f64 / region_length as f64) * 100.0
                    } else {
                        0.0
                    }
                ),
            ),
            (
                String::from("Processing time"),
                format!(
                    "{:.2}s ({:.1} pos/sec)",
                    total_time.as_secs_f64(),
                    region_length as f64 / total_time.as_secs_f64()
                ),
            ),
            (
                String::from("Memory usage"),
                format!(
                    "~{:.1} MB",
                    (site_diversities.capacity() * std::mem::size_of::<SiteDiversity>()) as f64
                        / 1_048_576.0
                ),
            ),
        ],
    });

    site_diversities // Return the vector of per-site diversity metrics
}

// Nucleotide diversity (π) at a single site is the average number of differences per site
// between two randomly chosen haplotypes. For a single site, this simplifies to the probability that
// two haplotypes differ at that site, which is equivalent to the expected heterozygosity.
//
// π at a site is sometimes expressed as:
// π = 1 - Σ p_i^2
//
// where p_i is the frequency of allele i in the population, and the sum is over all alleles at the site.
// This represents the probability that two randomly drawn alleles are different.
//
// Sample Correction: When estimating π from a sample of n haplotypes, the above formula is biased
// because it underestimates the population diversity. The unbiased estimator corrects this:
// π̂ = (n / (n - 1)) * (1 - Σ p_i^2)
//
// - n is the number of haplotypes with non-missing data at the site (total_called).
// - p_i is the sample frequency of allele i (count of allele i / total_called).
// - The factor n / (n - 1) adjusts for the fact that sample variance underestimates population variance.
//
// Implementation in Code:
// - freq_sq_sum computes Σ p_i^2 by summing the squared frequencies of each allele.
// - pi_value = (total_called / (total_called - 1)) * (1 - freq_sq_sum) applies the unbiased formula.
//
// Why the Correction?: Without the n / (n - 1) factor, π would be downwardly biased, especially
// for small n. For example, with n = 2, if one haplotype has allele A and the other T:
// - Frequencies: p_A = 0.5, p_T = 0.5
// - Σ p_i^2 = 0.5^2 + 0.5^2 = 0.5
// - Uncorrected: π = 1 - 0.5 = 0.5
// - Corrected: π = (2 / 1) * (1 - 0.5) = 2 * 0.5 = 1, which reflects that the two differ.
//
// When π = 1: Maximum diversity occurs when each of the n haplotypes has a unique allele:
// - If n = 2, alleles A and T: p_A = 0.5, p_T = 0.5, Σ p_i^2 = 0.5, π = (2 / 1) * (1 - 0.5) = 1.
// - If n = 4, alleles A, T, C, G: p_i = 0.25 each, Σ p_i^2 = 4 * (0.25)^2 = 0.25,
//   π = (4 / 3) * (1 - 0.25) = (4 / 3) * 0.75 = 1.
// - For DNA, since there are only 4 nucleotides, π = 1 is possible only when n ≤ 4 and all alleles differ.
// - For n > 4, some haplotypes must share alleles, so Σ p_i^2 > 1/n, and π < 1.
//
// π = 1 at a site means every pair of haplotypes differs, indicating maximum
// diversity for the sample size. In the code, this is correctly computed per site, adjusting for
// missing data by only counting haplotypes with alleles present.

/// Helper function to extract numerical value and components from an FstEstimate enum.
///
/// Returns a tuple:
///  - `Option<f64>`: The FST value. `Some(value)` if numerically calculable (can be
///    positive, negative, zero, or +/- Infinity). `None` if FST is undefined or
///    indeterminate (e.g., 0/0, negative denominator, insufficient data).
///  - `Option<f64>`: The sum of 'a' components (between-population variance).
///  - `Option<f64>`: The sum of 'b' components (within-population variance).
///  - `Option<usize>`: A count metric, which is `num_informative_sites` for
///    `Calculable` and `ComponentsYieldIndeterminateRatio`, `sites_evaluated`
///    for `NoInterPopulationVariance`, and `sites_attempted` for
///    `InsufficientDataForEstimation`.
pub fn extract_wc_fst_components(
    fst_estimate: &FstEstimate,
) -> (Option<f64>, Option<f64>, Option<f64>, Option<usize>) {
    match fst_estimate {
        FstEstimate::Calculable {
            value,
            sum_a,
            sum_b,
            num_informative_sites,
        } => {
            // FST value is numerically defined. This can include positive, negative,
            // zero (e.g., if sum_a is 0 but sum_a + sum_b > 0),
            // or +/- Infinity (e.g., if sum_a is non-zero and sum_a + sum_b is zero).
            (
                Some(*value),
                Some(*sum_a),
                Some(*sum_b),
                Some(*num_informative_sites),
            )
        }
        FstEstimate::ComponentsYieldIndeterminateRatio {
            sum_a,
            sum_b,
            num_informative_sites,
        } => {
            // FST ratio is indeterminate, typically because sum_a + sum_b is negative.
            // The FST value itself is considered undefined in this case.
            (
                None,
                Some(*sum_a),
                Some(*sum_b),
                Some(*num_informative_sites),
            )
        }
        FstEstimate::NoInterPopulationVariance {
            sum_a,
            sum_b,
            sites_evaluated,
        } => {
            // This state represents an FST calculation of 0/0, where sum_a is ~0 and sum_b is ~0.
            // The FST value is represented as None.
            // The variance components sum_a and sum_b are still reported (expected to be ~0).
            (None, Some(*sum_a), Some(*sum_b), Some(*sites_evaluated))
        }
        FstEstimate::InsufficientDataForEstimation {
            sum_a,
            sum_b,
            sites_attempted,
        } => {
            // FST could not be estimated due to fundamental data limitations (e.g., <2 populations).
            // The FST value is undefined.
            (None, Some(*sum_a), Some(*sum_b), Some(*sites_attempted))
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    include!("tests/stats_tests.rs");
    include!("tests/filter_tests.rs");
}
