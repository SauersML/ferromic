use crate::process::{Variant, ZeroBasedPosition, HaplotypeSide, ZeroBasedHalfOpen, QueryRegion, VcfError};
use crate::progress::{
    log, LogLevel, init_step_progress, update_step_progress,
    finish_step_progress, create_spinner, display_status_box, StatusBox, set_stage, ProcessingStage
};
use rayon::iter::IntoParallelIterator;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::fs::File;
use std::io::{BufReader, BufRead};
use std::path::Path;
use std::fmt;

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
            FstEstimate::Calculable { value, sum_a, sum_b, num_informative_sites } => {
                let val_str = if value.is_nan() {
                    "NaN".to_string()
                } else if value.is_infinite() {
                    if value.is_sign_positive() { "Infinity".to_string() } else { "-Infinity".to_string() }
                } else {
                    format!("{:.6}", value) // Common precision for FST
                };
                write!(f, "FST: {} (A: {:.4e}, B: {:.4e}, N_inf_sites: {})", val_str, sum_a, sum_b, num_informative_sites)
            }
            FstEstimate::ComponentsYieldIndeterminateRatio { sum_a, sum_b, num_informative_sites } => {
                write!(f, "IndeterminateRatio (A: {:.4e}, B: {:.4e}, N_inf_sites: {})", sum_a, sum_b, num_informative_sites)
            }
            FstEstimate::NoInterPopulationVariance { sum_a, sum_b, sites_evaluated } => {
                write!(f, "NoInterPopVariance (A: {:.4e}, B: {:.4e}, SitesEval: {})", sum_a, sum_b, sites_evaluated)
            }
            FstEstimate::InsufficientDataForEstimation { sum_a, sum_b, sites_attempted } => {
                // For InsufficientData, sum_a and sum_b are not typically meaningful data-derived sums.
                write!(f, "InsufficientData (A: {:.1}, B: {:.1}, SitesAtt: {})", sum_a, sum_b, sites_attempted)
            }
        }
    }
}


// Define a struct to hold diversity metrics for each genomic site
#[derive(Debug)]
pub struct SiteDiversity {
    pub position: i64, // 1-based position of the site in the genome
    pub pi: f64, // Nucleotide diversity (π) at this site
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


/// Calculates Weir & Cockerham FST between predefined haplotype groups (e.g., 0 vs 1) for a genomic region.
///
/// This function implements the Weir & Cockerham (1984) estimator for FST.
///
/// Observed within-haplotype heterozygosity (h_bar in W&C's notation) is zero. This results in the variance 
/// component 'c' being zero. The FST is then estimated as a / (a + b), using the
/// 'a' and 'b' components derived under this h_bar=0 condition.
///
/// The function first calculates per-site variance components (a_i, b_i) and FST estimates.
/// Then, it aggregates these components across all sites in the region to compute
/// an overall FST estimate for the region (using W&C eq. 10, by summing a_i and (a_i + b_i) separately), 
/// as well as overall pairwise FST estimates using the same aggregation principle.
///
/// # Arguments
/// * `variants`: A slice of `Variant` structs containing genotype data for all samples in the region.
/// * `sample_names`: A slice of `String`s, representing the names of all samples.
/// * `sample_to_group_map`: A `HashMap` mapping sample names to their (left, right) haplotype group assignments.
/// * `region`: A `QueryRegion` struct defining the genomic start and end (0-based, inclusive) to analyze.
///
/// # Returns
/// An `FstWcResults` struct containing overall FST, pairwise FSTs (as `FstEstimate` enums),
/// summed pairwise variance components, per-site FST details, and the FST type.
pub fn calculate_fst_wc_haplotype_groups(
    variants: &[Variant],
    sample_names: &[String],
    sample_to_group_map: &HashMap<String, (u8, u8)>,
    region: QueryRegion,
) -> FstWcResults {
    let spinner = create_spinner(&format!(
        "Calculating FST between haplotype groups for region {}:{}..{} (length {})",
        sample_names.get(0).map_or("UnknownChr", |s| s.split('_').next().unwrap_or("UnknownChr")), // Attempt to get chr from first sample if possible
        ZeroBasedPosition(region.start).to_one_based(), 
        ZeroBasedPosition(region.end).to_one_based(), 
        region.len()
    ));
    
    log(LogLevel::Info, &format!(
        "Beginning FST calculation between haplotype groups (0 vs 1) for region {}:{}..{}",
        sample_names.get(0).map_or("UnknownChr", |s| s.split('_').next().unwrap_or("UnknownChr")),
        ZeroBasedPosition(region.start).to_one_based(),
        ZeroBasedPosition(region.end).to_one_based()
    ));
    
    let haplotype_to_group = map_samples_to_haplotype_groups(sample_names, sample_to_group_map);
    
    let variant_map: HashMap<i64, &Variant> = variants.iter()
        .map(|v| (v.position, v))
        .collect();
    
    let mut site_fst_values = Vec::with_capacity(region.len() as usize);
    let position_count = region.len() as usize;
    
    init_step_progress(&format!(
        "Calculating FST at {} positions for haplotype groups", position_count
    ), position_count as u64);
    
    for (idx, pos) in (region.start..=region.end).enumerate() {
        if idx % 1000 == 0 || idx == 0 || idx == position_count.saturating_sub(1) {
            update_step_progress(idx as u64, &format!(
                "Position {}/{} ({:.1}%)",
                idx + 1, position_count, ((idx + 1) as f64 / position_count as f64) * 100.0
            ));
        }
        
        if let Some(variant) = variant_map.get(&pos) {
            let (overall_fst, pairwise_fst, var_comps, pop_sizes, pairwise_var_comps) =
                calculate_fst_wc_at_site_by_haplotype_group(variant, &haplotype_to_group);
            
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
            // For monomorphic sites, FST is NoInterPopulationVariance as components a and b are zero.
            let mut actual_pop_sizes = HashMap::new();
            let mut group_counts: HashMap<String, usize> = HashMap::new();
            for group_id_str in haplotype_to_group.values() {
                *group_counts.entry(group_id_str.clone()).or_insert(0) += 1;
            }
            actual_pop_sizes.insert("0".to_string(), *group_counts.get("0").unwrap_or(&0));
            actual_pop_sizes.insert("1".to_string(), *group_counts.get("1").unwrap_or(&0));
            
            let site_is_monomorphic_overall_estimate = FstEstimate::NoInterPopulationVariance {
                sum_a: 0.0,
                sum_b: 0.0,
                sites_evaluated: 1, // This one site was evaluated (by absence) as monomorphic
            };
            
            let mut populated_pairwise_fst = HashMap::new();
            let mut populated_pairwise_var_comps = HashMap::new();
            // For haplotype groups, the primary pair is "0_vs_1".
            // If both groups have members, their pairwise FST is also NoInterPopulationVariance.
            if actual_pop_sizes.get("0").map_or(false, |&c| c > 0) && actual_pop_sizes.get("1").map_or(false, |&c| c > 0) {
                 let site_is_monomorphic_pairwise_estimate = FstEstimate::NoInterPopulationVariance {
                    sum_a: 0.0,
                    sum_b: 0.0,
                    sites_evaluated: 1, // This one site evaluated for this pair
                 };
                 populated_pairwise_fst.insert("0_vs_1".to_string(), site_is_monomorphic_pairwise_estimate);
                 // Pairwise variance components for a monomorphic site are also (0.0, 0.0).
                 populated_pairwise_var_comps.insert("0_vs_1".to_string(), (0.0, 0.0));
            }

            site_fst_values.push(SiteFstWc {
                position: ZeroBasedPosition(pos).to_one_based(),
                overall_fst: site_is_monomorphic_overall_estimate,
                pairwise_fst: populated_pairwise_fst,
                variance_components: (0.0, 0.0), // Overall a, b for this monomorphic site are 0.
                population_sizes: actual_pop_sizes,
                pairwise_variance_components: populated_pairwise_var_comps,
            });
        }
    
    finish_step_progress("Completed per-site FST calculations for haplotype groups");
    
    let (overall_fst_estimate, pairwise_fst_estimates, aggregated_pairwise_components) =
        calculate_overall_fst_wc(&site_fst_values);
    
    let defined_positive_fst_sites = site_fst_values.iter()
        .filter(|site| matches!(site.overall_fst, FstEstimate::Calculable { value, .. } if value.is_finite() && value > 0.0))
        .count();
    
    log(LogLevel::Info, &format!(
        "Haplotype group FST calculation complete: {} sites with defined positive FST out of {} total positions.",
        defined_positive_fst_sites, site_fst_values.len()
    ));
    
    log(LogLevel::Info, &format!("Overall FST between haplotype groups: {}", overall_fst_estimate));
    
    for (pair, fst_e) in &pairwise_fst_estimates {
        log(LogLevel::Info, &format!("Pairwise FST for {}: {}", pair, fst_e));
    }
    
    spinner.finish_and_clear();
    
    FstWcResults {
        overall_fst: overall_fst_estimate,
        pairwise_fst: pairwise_fst_estimates,
        pairwise_variance_components: aggregated_pairwise_components,
        site_fst: site_fst_values,
        fst_type: "haplotype_groups".to_string(),
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
    let spinner = create_spinner(&format!(
        "Calculating FST between CSV-defined population groups for region {}:{}..{} (length {})",
        sample_names.get(0).map_or("UnknownChr", |s| s.split('_').next().unwrap_or("UnknownChr")),
        ZeroBasedPosition(region.start).to_one_based(), 
        ZeroBasedPosition(region.end).to_one_based(), 
        region.len()
    ));
    
    log(LogLevel::Info, &format!(
        "Beginning FST calculation between population groups defined in {} for region {}:{}..{}",
        csv_path.display(), 
        sample_names.get(0).map_or("UnknownChr", |s| s.split('_').next().unwrap_or("UnknownChr")),
        ZeroBasedPosition(region.start).to_one_based(),
        ZeroBasedPosition(region.end).to_one_based()
    ));
    
    let population_assignments = parse_population_csv(csv_path)?;
    let sample_to_pop = map_samples_to_populations(sample_names, &population_assignments);
    
    let variant_map: HashMap<i64, &Variant> = variants.iter()
        .map(|v| (v.position, v))
        .collect();
    
    let mut site_fst_values = Vec::with_capacity(region.len() as usize);
    let position_count = region.len() as usize;
    
    init_step_progress(&format!(
        "Calculating FST at {} positions for CSV populations", position_count
    ), position_count as u64);
    
    for (idx, pos) in (region.start..=region.end).enumerate() {
        if idx % 1000 == 0 || idx == 0 || idx == position_count.saturating_sub(1) {
            update_step_progress(idx as u64, &format!(
                "Position {}/{} ({:.1}%)",
                idx + 1, position_count, ((idx + 1) as f64 / position_count as f64) * 100.0
            ));
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
                        if pop_counts_for_site.get(&pop_ids[i]).map_or(false, |&count| count > 0) &&
                           pop_counts_for_site.get(&pop_ids[j]).map_or(false, |&count| count > 0) {
                            let (key_pop1, key_pop2) = if pop_ids[i] < pop_ids[j] {
                                (&pop_ids[i], &pop_ids[j])
                            } else {
                                (&pop_ids[j], &pop_ids[i])
                            };
                            let pair_key = format!("{}_vs_{}", key_pop1, key_pop2);
                            site_pairwise_fst.insert(pair_key.clone(), site_is_monomorphic_estimate);
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
    
    log(LogLevel::Info, &format!("Overall FST across all populations: {}", overall_fst_estimate));
    
    for (pair, fst_e) in &pairwise_fst_estimates {
        log(LogLevel::Info, &format!("Pairwise FST for {}: {}", pair, fst_e));
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
    let file = File::open(csv_path).map_err(|e| 
        VcfError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Failed to open population CSV file {}: {}", csv_path.display(), e)
        ))
    )?;
    
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
        let samples: Vec<String> = parts.iter().skip(1).filter(|s| !s.is_empty()).cloned().collect();
        
        if !samples.is_empty() {
            population_map.insert(population, samples);
        } else {
            log(LogLevel::Warning, &format!("Population '{}' in CSV file '{}' has no associated sample IDs listed on its line.", population, csv_path.display()));
        }
    }
    
    if population_map.is_empty() {
        return Err(VcfError::Parse(format!("Population CSV file '{}' contains no valid population data after parsing.", csv_path.display())));
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
fn map_samples_to_haplotype_groups(
    sample_names: &[String],
    sample_to_group_map: &HashMap<String, (u8, u8)>
) -> HashMap<(usize, HaplotypeSide), String> {
    let mut haplotype_to_group = HashMap::new();
    
    let mut sample_id_to_index = HashMap::new();
    for (idx, name) in sample_names.iter().enumerate() {
        let sample_id = name.rsplit('_').next().unwrap_or(name); // Uses a simplified ID if name has _L/_R suffix
        sample_id_to_index.insert(sample_id, idx);
    }
    
    for (config_sample_name, &(left_group, right_group)) in sample_to_group_map {
        if let Some(&vcf_idx) = sample_id_to_index.get(config_sample_name.as_str()) {
            haplotype_to_group.insert(
                (vcf_idx, HaplotypeSide::Left),
                left_group.to_string()
            );
            
            haplotype_to_group.insert(
                (vcf_idx, HaplotypeSide::Right),
                right_group.to_string()
            );
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
    population_assignments: &HashMap<String, Vec<String>>
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
        let core_vcf_id = vcf_sample_name.rsplit('_').next().unwrap_or(vcf_sample_name);
        if let Some(pop_name) = csv_sample_id_to_pop_name.get(core_vcf_id) {
            sample_to_pop_map_for_fst.insert((vcf_idx, HaplotypeSide::Left), pop_name.clone());
            sample_to_pop_map_for_fst.insert((vcf_idx, HaplotypeSide::Right), pop_name.clone());
            continue;
        }
        
        // Optional: attempt match by prefix (e.g. VCF "POP_SUBPOP_ID" vs CSV "POP") - currently present
        let vcf_prefix = vcf_sample_name.split('_').next().unwrap_or(vcf_sample_name);
        for (csv_pop_name, _) in population_assignments { // Iterating to check if VCF name STARTS WITH a pop name
            if vcf_sample_name.starts_with(csv_pop_name) || vcf_prefix == csv_pop_name {
                sample_to_pop_map_for_fst.insert((vcf_idx, HaplotypeSide::Left), csv_pop_name.clone());
                sample_to_pop_map_for_fst.insert((vcf_idx, HaplotypeSide::Right), csv_pop_name.clone());
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
    haplotype_to_group: &HashMap<(usize, HaplotypeSide), String>
) -> (FstEstimate, HashMap<String, FstEstimate>, (f64, f64), HashMap<String, usize>, HashMap<String, (f64, f64)>) {
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
    sample_to_pop: &HashMap<(usize, HaplotypeSide), String>
) -> (FstEstimate, HashMap<String, FstEstimate>, (f64, f64), HashMap<String, usize>, HashMap<String, (f64, f64)>) {
    calculate_fst_wc_at_site_general(variant, sample_to_pop)
}

/// General function to calculate Weir & Cockerham FST components and estimates at a single site.
///
/// This function takes a variant and a mapping of haplotypes to subpopulation identifiers.
/// It calculates allele frequencies per subpopulation, then computes Weir & Cockerham's
/// variance components 'a' (among populations) and 'b' (within populations).
/// From these, it derives overall and pairwise FST estimates for the site.
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
    map_subpop: &HashMap<(usize, HaplotypeSide), String>
) -> (FstEstimate, HashMap<String, FstEstimate>, (f64, f64), HashMap<String, usize>, HashMap<String, (f64, f64)>) {
    
    // 1. Count allele occurrences per subpopulation for the current variant.
    let mut allele_counts: HashMap<String, (usize, usize)> = HashMap::new(); // Stores (total_haplotypes_in_pop, alt_allele_count_in_pop)
    for (&(sample_idx, side), pop_id) in map_subpop {
        if let Some(Some(genotypes_vec)) = variant.genotypes.get(sample_idx) {
            if let Some(&allele_code) = genotypes_vec.get(side as usize) {
                let entry = allele_counts.entry(pop_id.clone()).or_insert((0, 0));
                entry.0 += 1; // Increment total haplotypes for this subpopulation.
                if allele_code != 0 { // Assuming 0 is reference, non-zero is alternate.
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
        return (insufficient_data_estimate, HashMap::new(), (0.0, 0.0), pop_sizes, HashMap::new());
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
        } else { // Denominator is effectively zero
            if site_a.abs() > eps { // Non-zero numerator / zero denominator
                FstEstimate::Calculable {
                    value: site_a / denominator, // Will be Inf or -Inf
                    sum_a: site_a,
                    sum_b: site_b,
                    num_informative_sites: 1,
                }
            } else { // Numerator is also effectively zero (0/0)
                FstEstimate::NoInterPopulationVariance {
                    sum_a: site_a, // Should be ~0.0
                    sum_b: site_b, // Should be ~0.0
                    sites_evaluated: 1, // This one site was evaluated
                }
            }
        }
    };
    
    // Logging for non-standard FST outcomes for overall_fst_at_site
    match overall_fst_at_site {
        FstEstimate::Calculable { value, .. } if !value.is_finite() => {
            log(LogLevel::Debug, &format!(
                "Site at pos {}: Overall FST is {} (a={:.4e}, b={:.4e}, a+b={:.4e}).",
                ZeroBasedPosition(variant.position).to_one_based(), overall_fst_at_site, site_a, site_b, site_a + site_b
            ));
        }
        FstEstimate::ComponentsYieldIndeterminateRatio { .. } | FstEstimate::NoInterPopulationVariance { .. } => {
             log(LogLevel::Debug, &format!(
                "Site at pos {}: Overall FST is {} (a={:.4e}, b={:.4e}, a+b={:.4e}).",
                ZeroBasedPosition(variant.position).to_one_based(), overall_fst_at_site, site_a, site_b, site_a + site_b
            ));
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
            } else { 0.0 };

            let (pairwise_a_xy, pairwise_b_xy) = calculate_variance_components(&current_pair_stats, pair_global_allele_freq);
            
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
                } else { // Denominator is effectively zero
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
            pairwise_variance_components_map.insert(pair_key.clone(), (pairwise_a_xy, pairwise_b_xy));

            match pairwise_fst_val {
                FstEstimate::Calculable { value, .. } if !value.is_finite() => {
                    log(LogLevel::Debug, &format!(
                        "Site at pos {}: Pairwise FST for {} is {} (a_xy={:.4e}, b_xy={:.4e}, a_xy+b_xy={:.4e}).",
                        ZeroBasedPosition(variant.position).to_one_based(), pair_key, pairwise_fst_val, pairwise_a_xy, pairwise_b_xy, pairwise_a_xy + pairwise_b_xy
                    ));
                }
                FstEstimate::ComponentsYieldIndeterminateRatio { .. } | FstEstimate::NoInterPopulationVariance { .. } => {
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
    (overall_fst_at_site, pairwise_fst_estimate_map, (site_a, site_b), pop_sizes, pairwise_variance_components_map)
}

/// Calculates Weir & Cockerham (1984) variance components 'a' (among-population)
/// and 'b' (between effective individuals/haplotypes within populations) for a set of subpopulations.
/// For haplotype data, observed heterozygosity (h_bar in W&C) is 0, leading to
/// their variance component 'c' being 0 (W&C eq. 4). The 'a' and 'b' components
/// implemented here are derived from W&C general equations (2) and (3) under this h_bar=0 condition.
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
    global_freq: f64 // p̄
) -> (f64, f64) {
    let r = pop_stats.len() as f64; // Number of subpopulations
    if r < 2.0 { // Need at least two populations to compare
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
    if (n_bar - 1.0) < 1e-9 { // Using < 1e-9 to catch n_bar very close to 1.0 or less than 1.0
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
    let c_squared = if r > 0.0 && n_bar > 0.0 { // Avoid division by zero if r or n_bar is zero
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
    let s_squared = if (r - 1.0) > 1e-9 && n_bar > 1e-9 { // denominators are positive
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
fn calculate_overall_fst_wc(site_fst_values: &[SiteFstWc]) -> (FstEstimate, HashMap<String, FstEstimate>, HashMap<String, (f64,f64)>) {
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
            _ => { // Catches Calculable, ComponentsYieldIndeterminateRatio, NoInterPopulationVariance for overall per-site
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
            if !matches!(site.pairwise_fst.get(pair_key), Some(FstEstimate::InsufficientDataForEstimation { .. }) | None) {
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
        let sum_a_total: f64 = informative_site_components_overall.iter().map(|(a, _)| *a).sum();
        let sum_b_total: f64 = informative_site_components_overall.iter().map(|(_, b)| *b).sum();
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
        } else { // Denominator is effectively zero
            if sum_a_total.abs() > eps { // Non-zero numerator
                FstEstimate::Calculable {
                    value: sum_a_total / denominator, // Inf or -Inf
                    sum_a: sum_a_total,
                    sum_b: sum_b_total,
                    num_informative_sites: num_sites_in_overall_sum,
                }
            } else { // Numerator also effectively zero (0/0 from sum)
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
    
    log(LogLevel::Info, &format!("Overall regional FST: {}", overall_fst_estimate));


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
            } else { // Denominator is effectively zero
                if sum_a_xy.abs() > eps { // Non-zero numerator
                    FstEstimate::Calculable {
                        value: sum_a_xy / denominator_pair, // Inf or -Inf
                        sum_a: sum_a_xy,
                        sum_b: sum_b_xy,
                        num_informative_sites: num_informative_sites_for_pair,
                    }
                } else { // Numerator also effectively zero (0/0 from sum)
                    FstEstimate::NoInterPopulationVariance {
                        sum_a: sum_a_xy, // ~0.0
                        sum_b: sum_b_xy, // ~0.0
                        sites_evaluated: num_informative_sites_for_pair,
                    }
                }
            };
            pairwise_fst_estimates.insert(pair_key.clone(), estimate_for_pair);
            log(LogLevel::Info, &format!("Regional pairwise FST for {}: {}", pair_key, estimate_for_pair));
        } else {
            // This pair_key was observed in some site's pairwise_variance_components map,
            // but none of those sites contributed actual components to informative_site_components_pairwise.
            // This implies for all sites where this pair was defined, its per-site FST was InsufficientData.
            // We count how many sites defined this pair (had an entry in pairwise_variance_components or pairwise_fst).
            let sites_attempted_for_this_pair = site_fst_values.iter()
                .filter(|s| s.pairwise_variance_components.contains_key(&pair_key) || s.pairwise_fst.contains_key(&pair_key) )
                .count();
             pairwise_fst_estimates.insert(pair_key.clone(), FstEstimate::InsufficientDataForEstimation {
                sum_a: 0.0, sum_b: 0.0, sites_attempted: sites_attempted_for_this_pair
            });
            aggregated_pairwise_variance_components.insert(pair_key.clone(), (0.0, 0.0)); // Store zero sums as components are not aggregated
            log(LogLevel::Info, &format!("Regional pairwise FST for {} (no informative components from sites): {}", pair_key, pairwise_fst_estimates.get(&pair_key).unwrap()));
        }
    }
    
    (overall_fst_estimate, pairwise_fst_estimates, aggregated_pairwise_variance_components)
}

/// Calculates Dxy (average number of pairwise differences per site between two populations)
/// for Hudson's FST, as defined by Hudson et al. (1992) and elaborated by
/// de Jong et al. (2024). Dxy is the mean number of differences between sequences
/// sampled from two different populations.
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
        log(LogLevel::Error, "Cannot calculate Dxy: sequence_length must be positive.");
        // This is a critical error in input setup
        return Err(VcfError::InvalidRegion(
            "Sequence length must be positive for Dxy calculation".to_string(),
        ));
    }

    if pop1_context.sequence_length != pop2_context.sequence_length {
        log(LogLevel::Error, "Sequence length mismatch between populations for Dxy calculation.");
        return Err(VcfError::Parse(
            "Sequence length mismatch in Dxy calculation".to_string(),
        ));
    }

    if pop1_context.haplotypes.is_empty() || pop2_context.haplotypes.is_empty() {
        log(LogLevel::Warning, &format!(
            "Cannot calculate Dxy for pops {:?}/{:?}: one or both have no haplotypes ({} and {} respectively).",
            pop1_context.id, pop2_context.id, pop1_context.haplotypes.len(), pop2_context.haplotypes.len()
        ));
        return Ok(DxyHudsonResult { d_xy: None });
    }

    let mut sum_total_differences_between_pops: f64 = 0.0;
    let num_haplotypes_pop1 = pop1_context.haplotypes.len();
    let num_haplotypes_pop2 = pop2_context.haplotypes.len();

    // total_inter_population_pairs is N1 * N2
    let total_inter_population_pairs = (num_haplotypes_pop1 * num_haplotypes_pop2) as f64;

    if total_inter_population_pairs == 0.0 { // Should be caught by is_empty above
        return Ok(DxyHudsonResult { d_xy: None });
    }

    for (sample_idx1, side1) in pop1_context.haplotypes.iter() {
        for (sample_idx2, side2) in pop2_context.haplotypes.iter() {
            let mut differences_for_this_specific_pair: u64 = 0;

            // Iterate through all variants provided in the context.
            // The `popX_context.variants` slice should contain only variants
            // within the specific region of interest being analyzed.
            for variant_site in pop1_context.variants.iter() { // pop1.variants is same as pop2.variants
                let allele1_opt = variant_site
                    .genotypes
                    .get(*sample_idx1)
                    .and_then(|gt_option| gt_option.as_ref())
                    .and_then(|gt_vec| gt_vec.get(*side1 as usize));

                let allele2_opt = variant_site
                    .genotypes
                    .get(*sample_idx2)
                    .and_then(|gt_option| gt_option.as_ref())
                    .and_then(|gt_vec| gt_vec.get(*side2 as usize));

                if let (Some(&a1_code), Some(&a2_code)) = (allele1_opt, allele2_opt) {
                    // Both alleles are present (not missing, e.g. not '.')
                    // This site is comparable for this pair.
                    if a1_code != a2_code {
                        differences_for_this_specific_pair += 1;
                    }
                }
            }
            sum_total_differences_between_pops += differences_for_this_specific_pair as f64;
        }
    }

    let effective_sequence_length = pop1_context.sequence_length as f64;
    let denominator = total_inter_population_pairs * effective_sequence_length;

    let d_xy_value = if denominator > 0.0 {
        Some(sum_total_differences_between_pops / denominator)
    } else {
        log(LogLevel::Warning, &format!(
            "Dxy denominator is zero for pops {:?}/{:?} (pairs: {}, L: {}). Setting Dxy to None.",
            pop1_context.id, pop2_context.id, total_inter_population_pairs, effective_sequence_length
        ));
        None
    };

    Ok(DxyHudsonResult { d_xy: d_xy_value })
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
            log(LogLevel::Warning, "One or both Pi values are non-finite, cannot calculate Pi_xy_avg.");
        }
    } else {
        log(LogLevel::Debug, "One or both Pi values are None, cannot calculate Pi_xy_avg.");
    }

    if let (Some(dxy_val), Some(pi_xy_avg_val)) = (outcome.d_xy, outcome.pi_xy_avg) {
        // dxy_val and pi_xy_avg_val are finite and dxy_val is positive for division
        if dxy_val.is_finite() && pi_xy_avg_val.is_finite() {
            if dxy_val > 1e-9 { // Use a small epsilon to avoid division by effective zero
                outcome.fst = Some((dxy_val - pi_xy_avg_val) / dxy_val);
            } else if dxy_val >= 0.0 && (dxy_val - pi_xy_avg_val).abs() < 1e-9 {
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
            log(LogLevel::Warning, "Dxy or Pi_xy_avg is non-finite, cannot calculate Hudson FST.");
        }
    } else {
        log(LogLevel::Debug, "Dxy or Pi_xy_avg is None, cannot calculate Hudson FST.");
    }

    outcome
}

/// Calculates Hudson's FST and its components for a given pair of populations.
/// This is the main public interface for obtaining Hudson's FST.
/// It orchestrates the calculation of within-population diversities (π)
/// and between-population diversity (Dxy), then computes the FST.
///
/// # Arguments
/// * `pop1_context` - `PopulationContext` for the first population.
/// * `pop2_context` - `PopulationContext` for the second population.
///
/// # Returns
/// A `Result` containing `HudsonFSTOutcome`. If any underlying calculation fails
/// or inputs are invalid (e.g., insufficient haplotypes for pi), the corresponding
/// fields in `HudsonFSTOutcome` will be `None`.
pub fn calculate_hudson_fst_for_pair<'a>(
    pop1_context: &PopulationContext<'a>,
    pop2_context: &PopulationContext<'a>,
) -> Result<HudsonFSTOutcome, VcfError> {
    // Basic input validation
    if pop1_context.sequence_length != pop2_context.sequence_length {
        return Err(VcfError::Parse(
            "Sequence length mismatch between population contexts for Hudson FST calculation.".to_string()
        ));
    }
    // A light check; deeper equality is complex and costly.
    if pop1_context.variants.as_ptr() != pop2_context.variants.as_ptr() {
        log(LogLevel::Debug,
            "Variant slices might differ between population contexts for Hudson FST. Ensure they refer to the same logical data."
        );
    }

    // 1. Calculate Pi for Pop1
    // `calculate_pi` returns f64::NAN if <2 haplotypes or seq_length <= 0.
    // We convert NAN to None here for consistent Option<f64> handling.
    let pi1_raw = calculate_pi(
        pop1_context.variants,
        &pop1_context.haplotypes,
        pop1_context.sequence_length,
    );
    let pi1_opt = if pi1_raw.is_finite() { Some(pi1_raw) } else { None };
    if pi1_opt.is_none() && pop1_context.haplotypes.len() >=2 && pop1_context.sequence_length > 0 {
        log(LogLevel::Warning, &format!("Pi calculation for pop {:?} resulted in non-finite value despite sufficient data.", pop1_context.id));
    }


    // 2. Calculate Pi for Pop2
    let pi2_raw = calculate_pi(
        pop2_context.variants,
        &pop2_context.haplotypes,
        pop2_context.sequence_length,
    );
    let pi2_opt = if pi2_raw.is_finite() { Some(pi2_raw) } else { None };
    if pi2_opt.is_none() && pop2_context.haplotypes.len() >=2 && pop2_context.sequence_length > 0 {
         log(LogLevel::Warning, &format!("Pi calculation for pop {:?} resulted in non-finite value despite sufficient data.", pop2_context.id));
    }

    // 3. Calculate Dxy between Pop1 and Pop2
    // This function returns Result<DxyHudsonResult, VcfError>
    let dxy_result = calculate_d_xy_hudson(pop1_context, pop2_context)?;

    // 4. Compute final Hudson FST outcome
    let outcome = compute_hudson_fst_outcome(
        pop1_context.id.clone(),
        pop2_context.id.clone(),
        pi1_opt,
        pi2_opt,
        &dxy_result,
    );

    Ok(outcome)
}

// Calculate the effective sequence length after adjusting for allowed and masked regions
pub fn calculate_adjusted_sequence_length(
    region_start: i64, // Start of the genomic region (1-based, inclusive)
    region_end: i64, // End of the genomic region (1-based, inclusive)
    allow_regions_chr: Option<&Vec<(i64, i64)>>, // Optional list of allowed regions as (start, end) tuples
    mask_regions_chr: Option<&Vec<(i64, i64)>>, // Optional list of masked regions to exclude
) -> i64 { // Returns the adjusted length as an i64
    log(LogLevel::Info, &format!(
        "Calculating adjusted sequence length for region {}:{}-{}", 
        if allow_regions_chr.is_some() { "with allow regions" } else { "full" },
        region_start, 
        region_end
    ));
    
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
    let unmasked_intervals = subtract_regions(&allowed_intervals, mask_regions_chr);
    
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
    log(LogLevel::Info, &format!(
        "Original length: {}, Adjusted length: {}",
        region_end - region_start + 1,
        adjusted_length
    ));

    
    log(LogLevel::Info, &format!(
        "Adjusted sequence length: {} (original: {})", 
        adjusted_length, 
        region_end - region_start + 1
    ));
        
    adjusted_length // Return the computed length
}

// Helper function to subtract masked regions from a set of intervals
fn subtract_regions(
    intervals: &Vec<(i64, i64)>, // Input intervals as (start, end) tuples in 1-based inclusive format
    masks: Option<&Vec<(i64, i64)>>, // Optional masked regions to subtract
) -> Vec<(i64, i64)> { // Returns the resulting intervals after subtraction
    if masks.is_none() {
        // If no masks are provided, return the original intervals unchanged
        return intervals.clone();
    }
    let masks = masks.unwrap(); // Unwrap the Option since we know it’s Some
    let mut result = Vec::new(); // Vector to store the final intervals after subtraction
    for &(start, end) in intervals {
        // Start with the current interval as the only interval to process
        let mut current_intervals = vec![(start, end)];
        for &(mask_start, mask_end) in masks {
            let mut new_intervals = Vec::new(); // Temporary vector for intervals after this mask
            for &(curr_start, curr_end) in &current_intervals {
                if mask_end < curr_start || mask_start > curr_end {
                    // No overlap between the mask and the current interval; keep it as is
                    new_intervals.push((curr_start, curr_end));
                } else {
                    // Overlap exists; split the current interval around the mask
                    if mask_start > curr_start {
                        // Add the portion before the mask starts
                        new_intervals.push((curr_start, mask_start));
                    }
                    if mask_end < curr_end {
                        // Add the portion after the mask ends
                        new_intervals.push((mask_end, curr_end));
                    }
                    // If mask fully covers the interval, new_intervals remains empty
                }
            }
            current_intervals = new_intervals; // Update current intervals for the next mask
            if current_intervals.is_empty() {
                // If the interval is completely masked, no need to process further masks
                break;
            }
        }
        // Add any remaining intervals to the result
        result.extend(current_intervals);
    }
    result // Return the list of unmasked intervals
}

// Calculate the frequency of allele 1 (e.g., an inversion allele) across haplotypes
pub fn calculate_inversion_allele_frequency(
    sample_filter: &HashMap<String, (u8, u8)>, // Map of sample names to (left, right) haplotype alleles
) -> Option<f64> { // Returns Some(frequency) or None if no haplotypes are present
    let mut num_ones = 0; // Counter for haplotypes with allele 1
    let mut total_haplotypes = 0; // Total number of haplotypes (with allele 0 or 1)
    for (_sample, &(left, right)) in sample_filter.iter() {
        // Check the left haplotype allele
        if left == 1 {
            num_ones += 1; // Increment if allele is 1
            total_haplotypes += 1; // Count this haplotype
        }
        // Check the right haplotype allele
        if right == 1 {
            num_ones += 1; // Increment if allele is 1
            total_haplotypes += 1; // Count this haplotype
        }
        if left == 0 {
            total_haplotypes += 1; // Count haplotype with allele 0
        }
        if right == 0 {
            total_haplotypes += 1; // Count haplotype with allele 0
        }
        // Alleles other than 0 or 1 (e.g., missing or bad data) are ignored
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
pub fn count_segregating_sites(variants: &[Variant]) -> usize { // Returns the count as usize
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
/// This function computes, for each pair of samples, the number of sites where their genotypes differ
/// and the number of sites where both have data, handling missing genotypes with parallelism.
///
/// # Arguments
/// * variants - A slice of Variant structs containing genotype data for all samples
/// * number_of_samples - The total number of samples to compare
///
/// # Returns
/// A vector of tuples, each containing:
/// * (sample_idx_i, sample_idx_j) - Indices of the sample pair
/// * difference_count - Number of sites where genotypes differ (d_ij)
/// * comparable_site_count - Number of sites where both samples have genotypes (l_ij)
pub fn calculate_pairwise_differences(
    variants: &[Variant],
    number_of_samples: usize,
) -> Vec<((usize, usize), usize, usize)> {
    set_stage(ProcessingStage::StatsCalculation);
    
    let total_pairs = (number_of_samples * (number_of_samples - 1)) / 2;
    log(LogLevel::Info, &format!(
        "Calculating pairwise differences across {} samples ({} pairs)", 
        number_of_samples, total_pairs
    ));
    
    let spinner = create_spinner(&format!(
        "Processing pairwise differences for {} samples", number_of_samples
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

                    // Iterate over all variants to compare this pair's genotypes
                    for variant in variants_local.iter() {
                        if let (Some(genotype_i), Some(genotype_j)) = (
                            &variant.genotypes[sample_idx_i],
                            &variant.genotypes[sample_idx_j],
                        ) {
                            // Both samples have genotype data at this site
                            comparable_site_count += 1;
                            if genotype_i != genotype_j {
                                // Genotypes differ; humans are haploid (single allele per genotype)
                                difference_count += 1;
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
    spinner.finish_and_clear();
        log(LogLevel::Info, &format!(
            "Computed {} pairwise comparisons", result_count
        ));

    result
}

// Calculate the harmonic number H_n = sum_{k=1}^n 1/k
pub fn harmonic(n: usize) -> f64 { // Returns the harmonic number as a float
    (1..=n) // Range from 1 to n inclusive
        .map(|i| 1.0 / i as f64) // Map each integer k to 1/k as a float
        .sum() // Sum all terms to get H_n
}

// Calculate Watterson's theta (θ_w), a measure of genetic diversity
pub fn calculate_watterson_theta(seg_sites: usize, n: usize, seq_length: i64) -> f64 {
    // Handle edge cases where computation isn't meaningful
    if n <= 1 || seq_length == 0 {
        // If 1 or fewer haplotypes or sequence length is zero, return infinity
        log(LogLevel::Warning, &format!(
            "Cannot calculate Watterson's theta: {} haplotypes, {} length",
            n, seq_length
        ));
        return f64::INFINITY;
    }

    // Calculate the harmonic number H_{n-1}, used as the denominator factor a_n
    let harmonic_value = harmonic(n - 1);
    if harmonic_value == 0.0 {
        // Prevent division by zero
        log(LogLevel::Warning, "Harmonic denominator is zero, cannot calculate theta");
        return f64::INFINITY;
    }
    
    // Watterson's theta formula: θ_w = S / (a_n * L)
    // S = number of segregating sites, a_n = H_{n-1}, L = sequence length
    let theta = seg_sites as f64 / harmonic_value / seq_length as f64;
    
    log(LogLevel::Debug, &format!(
        "Watterson's theta: {} (from {} segregating sites, {} haplotypes, {} length)",
        theta, seg_sites, n, seq_length
    ));
    
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
pub fn calculate_pi(variants: &[Variant], haplotypes_in_group: &[(usize, HaplotypeSide)], seq_length: i64) -> f64 {
    if haplotypes_in_group.len() <= 1 {
        // Need at least 2 haplotypes to compute diversity; return NaN if not
        log(LogLevel::Warning, &format!(
            "Cannot calculate pi: insufficient haplotypes ({})", 
            haplotypes_in_group.len()
        ));
        return f64::NAN;
    }

    // Calculate total possible pairs: n * (n-1) / 2
    let total_possible_pairs = haplotypes_in_group.len() * (haplotypes_in_group.len() - 1) / 2;
    if total_possible_pairs == 0 {
        // If no pairs can be formed (redundant check), return NaN
        log(LogLevel::Warning, "No valid pairs can be formed for pi calculation");
        return f64::NAN;
    }
    
    if seq_length <= 0 {
        log(LogLevel::Warning, &format!(
            "Cannot calculate pi: invalid sequence length ({})",
            seq_length
        ));
        return f64::NAN;
    }
    
    let spinner = create_spinner(&format!(
        "Calculating π for {} haplotypes ({} pairs) over {} bp",
        haplotypes_in_group.len(), total_possible_pairs, seq_length
    ));

    let mut total_differences = 0; // Total number of differences across all pairs
    let mut total_compared_pairs = 0; // Count of pairs with at least one comparable site

    for i in 0..haplotypes_in_group.len() {
        for j in (i + 1)..haplotypes_in_group.len() {
            // Extract sample index and haplotype side (left or right) for both haplotypes
            let (sample_i, side_i) = haplotypes_in_group[i];
            let (sample_j, side_j) = haplotypes_in_group[j];

            let mut diff_count = 0; // Number of sites where alleles differ
            let mut comparable_sites = 0; // Number of sites where both have data

            for var in variants {
                // Check if both samples have genotype data at this variant
                if let Some(gt_i) = var.genotypes.get(sample_i) {
                    if let Some(gt_j) = var.genotypes.get(sample_j) {
                        if let Some(alleles_i) = gt_i {
                            if let Some(alleles_j) = gt_j {
                                // Get the specific allele for each haplotype's side
                                if let (Some(&a_i), Some(&a_j)) = (
                                    alleles_i.get(side_i as usize),
                                    alleles_j.get(side_j as usize),
                                ) {
                                    comparable_sites += 1; // Both have data; count this site
                                    if a_i != a_j {
                                        diff_count += 1; // Alleles differ; increment difference
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if comparable_sites > 0 {
                // Track the number of differences and the number of pairs with data
                total_differences += diff_count;
                total_compared_pairs += 1;
            }
            // If no comparable sites, this pair is not counted
        }
    }

    // Compute nucleotide diversity: average number of differences per site
    // Pi = (total number of differences) / (sequence length * number of pairs)
    let pi = if total_compared_pairs > 0 {
        total_differences as f64 / (seq_length as f64 * total_compared_pairs as f64)
    } else {
        0.0
    };
    
    spinner.finish_and_clear();
    log(LogLevel::Info, &format!(
        "π = {:.6} (from {} differences across {} bp in {} haplotype pairs)",
        pi,
        total_differences,
        seq_length,
        total_compared_pairs
    ));

    log(LogLevel::Info, &format!(
        "Calculated nucleotide diversity (π): {:.6}",
        pi
    ));
    
    pi
}


// Calculate per-site diversity metrics (π and Watterson's θ) across a genomic region
pub fn calculate_per_site_diversity(
    variants: &[Variant], 
    haplotypes_in_group: &[(usize, HaplotypeSide)], 
    region: QueryRegion, // Inclusive range [start..end] in 0-based coordinates
) -> Vec<SiteDiversity> { // Returns a vector of SiteDiversity structs
    set_stage(ProcessingStage::StatsCalculation);

    let start_time = std::time::Instant::now();
    log(LogLevel::Info, &format!(
        "Calculating per-site diversity for region {}:{}-{} with {} haplotypes",
        region.start, region.end, region.len(), haplotypes_in_group.len()
    ));

    let max_haps = haplotypes_in_group.len(); // Number of haplotypes in the group

    let region_length = region.len();
    
    // Pre-allocate with correct capacity for better memory efficiency
    let mut site_diversities = Vec::with_capacity(region_length as usize);
    
    if max_haps < 2 {
        // Need at least 2 haplotypes for diversity; return empty vector if not
        log(LogLevel::Warning, "Insufficient haplotypes (<2) for diversity calculation");
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
    log(LogLevel::Info, &format!(
        "Indexed {} variants for fast lookup ({}ms)",
        variant_map.len(),
        start_time.elapsed().as_millis()
    ));

    // Initialize detailed progress tracking with checkpoints
    init_step_progress(&format!(
        "Calculating diversity across {} positions", region_length
    ), region_length as u64);

    // Track statistics for progress updates and performance monitoring
    let mut variants_processed = 0;
    let mut polymorphic_sites = 0;
    let mut last_update_time = std::time::Instant::now();
    let mut positions_since_update = 0;
    let update_interval = std::cmp::min(1000, region_length as usize / 100);

    // Process in batches for more efficient update frequency
    for (idx, pos) in (region.start..=region.end).enumerate() { // Inclusive range
        positions_since_update += 1;
        
        // Update progress sometimes
        if positions_since_update >= update_interval || idx == 0 || idx as i64 == region_length - 1 {
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
            
            update_step_progress(idx as u64, &format!(
                "Position {}/{} ({:.1}%) - {:.1} pos/sec - ~{:.0}s remaining",
                idx, region_length, progress_pct, positions_per_sec, remaining_secs
            ));
            
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
                pi: 0.0, // No diversity since no variation
                watterson_theta: 0.0, // No segregating site
            });
        }
    }

    let total_time = start_time.elapsed();
    // Finish progress and display summary statistics
    finish_step_progress(&format!(
        "Completed: {} positions, {} variants, {} polymorphic sites in {:.2}s",
        region_length, variants_processed, polymorphic_sites, total_time.as_secs_f64()
    ));

    log(LogLevel::Info, &format!(
        "Per-site diversity calculation complete: {} positions analyzed, {} polymorphic sites",
        region_length, polymorphic_sites
    ));

    // Show detailed summary in status box with performance metrics
    display_status_box(StatusBox {
        title: "Per-Site Diversity Summary".to_string(),
        stats: vec![
            (String::from("Region"), format!("{}:{}-{}", 
                ZeroBasedPosition(region.start).to_one_based(), 
                ZeroBasedPosition(region.end).to_one_based(), 
                region_length)),
            (String::from("Haplotypes"), max_haps.to_string()),
            (String::from("Variants processed"), variants_processed.to_string()),
            (String::from("Polymorphic sites"), format!("{} ({:.2}%)",
                polymorphic_sites,
                if region_length > 0 { (polymorphic_sites as f64 / region_length as f64) * 100.0 } else { 0.0 }
            )),
            (String::from("Processing time"), format!("{:.2}s ({:.1} pos/sec)", 
                total_time.as_secs_f64(),
                region_length as f64 / total_time.as_secs_f64()
            )),
            (String::from("Memory usage"), format!("~{:.1} MB", 
                (site_diversities.capacity() * std::mem::size_of::<SiteDiversity>()) as f64 / 1_048_576.0
            ))
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


#[cfg(test)]
mod tests {
    use super::*;
    include!("tests/stats_tests.rs");
    include!("tests/filter_tests.rs");
}
