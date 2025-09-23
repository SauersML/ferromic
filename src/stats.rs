use crate::process::{
    HaplotypeSide, QueryRegion, Variant, VcfError, ZeroBasedHalfOpen, ZeroBasedPosition,
};
use crate::progress::{
    create_spinner, display_status_box, finish_step_progress, init_step_progress, log, set_stage,
    update_step_progress, LogLevel, ProcessingStage, StatusBox,
};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

/// Epsilon threshold for numerical stability in FST calculations.
/// Used consistently across per-site and aggregation functions to handle
/// near-zero denominators and floating-point precision issues.
///
/// Usage Guidelines:
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
    pub dense_genotypes: Option<&'a DenseGenotypeMatrix>,
    pub dense_summary: Option<Arc<DensePopulationSummary>>,
}

#[derive(Debug, Clone)]
pub struct DenseGenotypeMatrix {
    data: Arc<[u8]>,
    missing: Option<Arc<[u64]>>,
    variant_count: usize,
    sample_count: usize,
    ploidy: usize,
    stride: usize,
    max_allele: u8,
}

impl DenseGenotypeMatrix {
    pub fn new(
        data: Vec<u8>,
        missing: Option<Vec<u64>>,
        variant_count: usize,
        sample_count: usize,
        ploidy: usize,
        max_allele: u8,
    ) -> Self {
        debug_assert_eq!(
            data.len(),
            variant_count * sample_count * ploidy,
            "dense genotype matrix requires variants * samples * ploidy entries",
        );
        let data = Arc::<[u8]>::from(data.into_boxed_slice());
        let missing = missing.map(|bits| Arc::<[u64]>::from(bits.into_boxed_slice()));
        Self {
            data,
            missing,
            variant_count,
            sample_count,
            ploidy,
            stride: sample_count * ploidy,
            max_allele,
        }
    }

    #[inline]
    pub fn variant_count(&self) -> usize {
        self.variant_count
    }

    #[inline]
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    #[inline]
    pub fn ploidy(&self) -> usize {
        self.ploidy
    }

    #[inline]
    fn stride(&self) -> usize {
        self.stride
    }

    #[inline]
    fn data(&self) -> &[u8] {
        &self.data
    }

    #[inline]
    fn missing_slice(&self) -> Option<&[u64]> {
        self.missing.as_deref()
    }

    #[inline]
    pub fn max_allele(&self) -> u8 {
        self.max_allele
    }
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

    /// Per-site FST values and components. Entries are emitted only for variant positions that
    /// overlap the queried region; monomorphic bases are omitted because they contribute zeros
    /// to the regional aggregates.
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
// The returned `site_fst` vector is *sparse*: it includes one entry per variant that falls
// inside `region`, and monomorphic base pairs are omitted because they contribute zeros to
// all regional aggregates. This keeps per-site memory proportional to the number of
// informative loci instead of the physical length of `region`.
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
// - A detailed list of per-site FST results (`SiteFstWc`) for variant positions only.
// - A string indicating the type of FST analysis performed (e.g., "haplotype_groups").
//
// Pairwise outputs are keyed using "{pop_a}_vs_{pop_b}" with population labels sorted
// lexicographically. Every defined pair is represented in both `pairwise_fst` and
// `pairwise_variance_components`; if a pair never has sufficient data at any variant site
// the entry is populated with `FstEstimate::InsufficientDataForEstimation` and zeroed
// components so downstream consumers can rely on a stable schema.
pub fn calculate_fst_wc_haplotype_groups(
    variants: &[Variant],
    sample_names: &[String],
    sample_to_group_map: &HashMap<String, (u8, u8)>,
    region: QueryRegion,
) -> FstWcResults {
    let spinner = create_spinner(&format!(
        "Calculating FST between haplotype groups for region {}:{}..{} (length {})",
        sample_names.get(0).map_or("UnknownChr", |s_name| s_name
            .split('_')
            .next()
            .unwrap_or("UnknownChr")),
        ZeroBasedPosition(region.start).to_one_based(),
        ZeroBasedPosition(region.end).to_one_based(),
        region.len()
    ));

    log(
        LogLevel::Info,
        &format!(
            "Beginning FST calculation between haplotype groups (e.g., 0 vs 1) for region {}:{}..{}",
            sample_names
                .get(0)
                .map_or("UnknownChr", |s_name| s_name.split('_').next().unwrap_or("UnknownChr")),
            ZeroBasedPosition(region.start).to_one_based(),
            ZeroBasedPosition(region.end).to_one_based()
        ),
    );

    let haplotype_to_group = map_samples_to_haplotype_groups(sample_names, sample_to_group_map);
    let membership = SubpopulationMembership::from_map(sample_names.len(), &haplotype_to_group);
    let mut workspace = WcSiteWorkspace::new(membership.group_count());

    let total_variants = variants
        .iter()
        .filter(|variant| region.contains(variant.position))
        .count();

    let mut site_fst_values = Vec::with_capacity(total_variants);

    init_step_progress(
        &format!(
            "Calculating FST at {} variant sites for haplotype groups",
            total_variants
        ),
        total_variants as u64,
    );

    for (idx, variant) in variants
        .iter()
        .filter(|variant| region.contains(variant.position))
        .enumerate()
    {
        if idx % 1000 == 0 || idx + 1 == total_variants {
            update_step_progress(
                idx as u64,
                &format!(
                    "Variant {}/{} ({:.1}%)",
                    idx + 1,
                    total_variants,
                    ((idx + 1) as f64 / total_variants.max(1) as f64) * 100.0
                ),
            );
        }

        let (overall_fst, pairwise_fst, var_comps, pop_sizes, pairwise_var_comps) =
            calculate_fst_wc_at_site_with_membership(variant, &membership, &mut workspace);

        site_fst_values.push(SiteFstWc {
            position: ZeroBasedPosition(variant.position).to_one_based(),
            overall_fst,
            pairwise_fst,
            variance_components: var_comps,
            population_sizes: pop_sizes,
            pairwise_variance_components: pairwise_var_comps,
        });
    }

    finish_step_progress("Completed per-site FST calculations for haplotype groups");

    let (overall_fst_estimate, pairwise_fst_estimates, aggregated_pairwise_components) =
        calculate_overall_fst_wc(&site_fst_values);

    let defined_positive_fst_sites = site_fst_values
        .iter()
        .filter(|site| {
            matches!(
                site.overall_fst,
                FstEstimate::Calculable { value, .. } if value.is_finite() && value > 0.0
            )
        })
        .count();

    log(
        LogLevel::Info,
        &format!(
            "Haplotype group FST calculation complete: {} sites showed positive, finite FST out of {} variant sites in the region.",
            defined_positive_fst_sites,
            site_fst_values.len()
        ),
    );

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
        fst_type: "haplotype_groups".to_string(),
    }
}

/// Calculates Weir & Cockerham FST for population assignments loaded from a CSV file.
///
/// Like `calculate_fst_wc_haplotype_groups`, the per-site output is variant-sparse: only
/// polymorphic positions that overlap `region` appear in the returned `site_fst` vector and
/// monomorphic bases are implicitly treated as zero contributors to regional sums. Pairwise
/// keys follow the same stable "{pop_a}_vs_{pop_b}" ordering and always appear in the
/// result maps; insufficient data for a pair is communicated via
/// `FstEstimate::InsufficientDataForEstimation` and zeroed variance components.
pub fn calculate_fst_wc_csv_populations(
    variants: &[Variant],
    sample_names: &[String],
    csv_path: &Path,
    region: QueryRegion,
) -> Result<FstWcResults, VcfError> {
    let spinner = create_spinner(&format!(
        "Calculating FST between CSV-defined population groups for region {}:{}..{} (length {})",
        sample_names.get(0).map_or("UnknownChr", |s| s
            .split('_')
            .next()
            .unwrap_or("UnknownChr")),
        ZeroBasedPosition(region.start).to_one_based(),
        ZeroBasedPosition(region.end).to_one_based(),
        region.len()
    ));

    log(
        LogLevel::Info,
        &format!(
            "Beginning FST calculation between population groups defined in {} for region {}:{}..{}",
            csv_path.display(),
            sample_names
                .get(0)
                .map_or("UnknownChr", |s| s.split('_').next().unwrap_or("UnknownChr")),
            ZeroBasedPosition(region.start).to_one_based(),
            ZeroBasedPosition(region.end).to_one_based()
        ),
    );

    let population_assignments = parse_population_csv(csv_path)?;
    let sample_to_pop = map_samples_to_populations(sample_names, &population_assignments);
    let membership = SubpopulationMembership::from_map(sample_names.len(), &sample_to_pop);
    let mut workspace = WcSiteWorkspace::new(membership.group_count());

    let total_variants = variants
        .iter()
        .filter(|variant| region.contains(variant.position))
        .count();

    let mut site_fst_values = Vec::with_capacity(total_variants);

    init_step_progress(
        &format!(
            "Calculating FST at {} variant sites for CSV populations",
            total_variants
        ),
        total_variants as u64,
    );

    for (idx, variant) in variants
        .iter()
        .filter(|variant| region.contains(variant.position))
        .enumerate()
    {
        if idx % 1000 == 0 || idx + 1 == total_variants {
            update_step_progress(
                idx as u64,
                &format!(
                    "Variant {}/{} ({:.1}%)",
                    idx + 1,
                    total_variants,
                    ((idx + 1) as f64 / total_variants.max(1) as f64) * 100.0
                ),
            );
        }

        let (overall_fst, pairwise_fst, var_comps, pop_sizes, pairwise_var_comps) =
            calculate_fst_wc_at_site_with_membership(variant, &membership, &mut workspace);

        site_fst_values.push(SiteFstWc {
            position: ZeroBasedPosition(variant.position).to_one_based(),
            overall_fst,
            pairwise_fst,
            variance_components: var_comps,
            population_sizes: pop_sizes,
            pairwise_variance_components: pairwise_var_comps,
        });
    }

    finish_step_progress("Completed per-site FST calculations for CSV populations");

    let (overall_fst_estimate, pairwise_fst_estimates, aggregated_pairwise_components) =
        calculate_overall_fst_wc(&site_fst_values);

    let defined_positive_fst_sites = site_fst_values
        .iter()
        .filter(|site| {
            matches!(
                site.overall_fst,
                FstEstimate::Calculable { value, .. } if value.is_finite() && value > 0.0
            )
        })
        .count();

    log(
        LogLevel::Info,
        &format!(
            "CSV population FST calculation complete: {} sites showed positive, finite FST out of {} variant sites",
            defined_positive_fst_sites,
            site_fst_values.len()
        ),
    );

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
            log(
                LogLevel::Warning,
                &format!(
                "Population '{}' in CSV file '{}' has no associated sample IDs listed on its line.",
                population,
                csv_path.display()
            ),
            );
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

/// Extracts a core sample identifier by removing optional haplotype suffixes.
pub fn core_sample_id(name: &str) -> &str {
    if let Some(s) = name.strip_suffix("_L").or_else(|| name.strip_suffix("_R")) {
        s
    } else {
        name
    }
}

#[cfg(test)]
mod core_sample_id_tests {
    use super::core_sample_id;

    #[test]
    fn test_core_sample_id() {
        assert_eq!(core_sample_id("NA12878_L"), "NA12878");
        assert_eq!(core_sample_id("NA12878_R"), "NA12878");
        assert_eq!(core_sample_id("SAMP_01_L"), "SAMP_01");
        assert_eq!(core_sample_id("SAMP_01_R"), "SAMP_01");
        assert_eq!(core_sample_id("NoSuffix"), "NoSuffix");
        assert_eq!(
            core_sample_id("Sample_With_Underscores_L"),
            "Sample_With_Underscores"
        );
        assert_eq!(
            core_sample_id("Sample_With_Underscores_R"),
            "Sample_With_Underscores"
        );
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
        sample_id_to_index.insert(core.to_string(), idx);
        sample_id_to_index.insert(name.clone(), idx);
    }

    for (config_sample_name, &(left_group, right_group)) in sample_to_group_map {
        if let Some(&vcf_idx) = sample_id_to_index.get(config_sample_name.as_str()) {
            haplotype_to_group.insert((vcf_idx, HaplotypeSide::Left), left_group.to_string());
            haplotype_to_group.insert((vcf_idx, HaplotypeSide::Right), right_group.to_string());
        }
    }

    haplotype_to_group
}

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
        if let Some(pop_name) = csv_sample_id_to_pop_name.get(vcf_sample_name) {
            sample_to_pop_map_for_fst.insert((vcf_idx, HaplotypeSide::Left), pop_name.clone());
            sample_to_pop_map_for_fst.insert((vcf_idx, HaplotypeSide::Right), pop_name.clone());
            continue;
        }

        let core_vcf_id = core_sample_id(vcf_sample_name);
        if let Some(pop_name) = csv_sample_id_to_pop_name.get(core_vcf_id) {
            sample_to_pop_map_for_fst.insert((vcf_idx, HaplotypeSide::Left), pop_name.clone());
            sample_to_pop_map_for_fst.insert((vcf_idx, HaplotypeSide::Right), pop_name.clone());
            continue;
        }

        let vcf_prefix = vcf_sample_name.split('_').next().unwrap_or(vcf_sample_name);
        for (csv_pop_name, _) in population_assignments {
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

const INVALID_GROUP: u16 = u16::MAX;

#[derive(Clone)]
/// Describes a pairwise comparison between two population indices. The `key` is constructed
/// using lexicographically sorted population labels ("{pop_a}_vs_{pop_b}") to provide a stable
/// ordering in downstream maps.
struct PairDescriptor {
    left: u16,
    right: u16,
    key: String,
}

#[derive(Clone)]
struct SubpopulationMembership {
    left: Vec<u16>,
    right: Vec<u16>,
    labels: Vec<String>,
    /// All unordered pairings of population labels expressed in lexicographic order. Every
    /// element in this list is emitted in per-site maps so consumers observe a consistent schema
    /// regardless of which comparisons had sufficient data.
    pair_keys: Vec<PairDescriptor>,
}

impl SubpopulationMembership {
    fn from_map(sample_count: usize, map_subpop: &HashMap<(usize, HaplotypeSide), String>) -> Self {
        let mut labels: Vec<String> = map_subpop.values().cloned().collect();
        labels.sort();
        labels.dedup();

        let mut label_to_index = HashMap::with_capacity(labels.len());
        for (idx, label) in labels.iter().enumerate() {
            label_to_index.insert(label.clone(), idx as u16);
        }

        let mut left = vec![INVALID_GROUP; sample_count];
        let mut right = vec![INVALID_GROUP; sample_count];

        for (&(sample_idx, side), pop_id) in map_subpop {
            if sample_idx >= sample_count {
                continue;
            }
            if let Some(&group_idx) = label_to_index.get(pop_id) {
                match side {
                    HaplotypeSide::Left => {
                        left[sample_idx] = group_idx;
                    }
                    HaplotypeSide::Right => {
                        right[sample_idx] = group_idx;
                    }
                }
            }
        }

        let mut pair_keys = Vec::new();
        for i in 0..labels.len() {
            for j in (i + 1)..labels.len() {
                pair_keys.push(PairDescriptor {
                    left: i as u16,
                    right: j as u16,
                    key: format!("{}_vs_{}", labels[i], labels[j]),
                });
            }
        }

        Self {
            left,
            right,
            labels,
            pair_keys,
        }
    }

    fn group_count(&self) -> usize {
        self.labels.len()
    }

    fn label(&self, idx: usize) -> &str {
        &self.labels[idx]
    }
}

#[derive(Default)]
struct WcSiteWorkspace {
    total_counts: Vec<usize>,
    alt_counts: Vec<usize>,
    stats: Vec<PopSiteStat>,
}

impl WcSiteWorkspace {
    fn new(group_count: usize) -> Self {
        Self {
            total_counts: vec![0; group_count],
            alt_counts: vec![0; group_count],
            stats: Vec::with_capacity(group_count),
        }
    }

    fn ensure_capacity(&mut self, group_count: usize) {
        if self.total_counts.len() < group_count {
            self.total_counts.resize(group_count, 0);
            self.alt_counts.resize(group_count, 0);
        }
        if self.stats.capacity() < group_count {
            self.stats.reserve(group_count - self.stats.capacity());
        }
    }

    fn reset(&mut self) {
        for value in &mut self.total_counts {
            *value = 0;
        }
        for value in &mut self.alt_counts {
            *value = 0;
        }
        self.stats.clear();
    }
}

#[derive(Clone, Copy)]
struct PopSiteStat {
    total: usize,
    freq: f64,
}

#[derive(Clone)]
struct HapMembership {
    left: Vec<bool>,
    right: Vec<bool>,
    total: usize,
}

impl HapMembership {
    fn build(sample_count: usize, haplotypes: &[(usize, HaplotypeSide)]) -> Self {
        let mut left = vec![false; sample_count];
        let mut right = vec![false; sample_count];
        let mut total = 0usize;

        for &(sample_idx, side) in haplotypes {
            if sample_idx >= sample_count {
                continue;
            }
            match side {
                HaplotypeSide::Left => {
                    if !left[sample_idx] {
                        left[sample_idx] = true;
                        total += 1;
                    }
                }
                HaplotypeSide::Right => {
                    if !right[sample_idx] {
                        right[sample_idx] = true;
                        total += 1;
                    }
                }
            }
        }

        Self { left, right, total }
    }
}

#[derive(Clone)]
struct DenseMembership {
    offsets: Vec<usize>,
}

impl DenseMembership {
    fn build(matrix: &DenseGenotypeMatrix, haplotypes: &[(usize, HaplotypeSide)]) -> Self {
        let sample_count = matrix.sample_count();
        let ploidy = matrix.ploidy();
        let mut left = vec![false; sample_count];
        let mut right = vec![false; sample_count];
        let mut offsets = Vec::with_capacity(haplotypes.len());

        for &(sample_idx, side) in haplotypes {
            if sample_idx >= sample_count {
                continue;
            }
            match side {
                HaplotypeSide::Left => {
                    if !left[sample_idx] {
                        left[sample_idx] = true;
                        offsets.push(sample_idx * ploidy);
                    }
                }
                HaplotypeSide::Right => {
                    if ploidy <= 1 {
                        continue;
                    }
                    if !right[sample_idx] {
                        right[sample_idx] = true;
                        offsets.push(sample_idx * ploidy + 1);
                    }
                }
            }
        }

        offsets.sort_unstable();
        Self { offsets }
    }

    #[inline]
    fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    #[inline]
    fn len(&self) -> usize {
        self.offsets.len()
    }
}

#[inline(always)]
fn dense_missing(bits: &[u64], idx: usize) -> bool {
    let word = idx >> 6;
    let bit = idx & 63;
    unsafe { ((*bits.get_unchecked(word) >> bit) & 1) == 1 }
}

#[inline(always)]
fn dense_should_parallelize(variant_count: usize, haplotypes: usize) -> bool {
    const PARALLEL_WORK_THRESHOLD: usize = 32 * 1024;
    variant_count.saturating_mul(haplotypes) >= PARALLEL_WORK_THRESHOLD
}

#[derive(Debug)]
pub struct DensePopulationSummary {
    alt_counts: Arc<[u32]>,
    called_counts: Arc<[u32]>,
    haplotype_capacity: usize,
    segregating_sites: usize,
    pi_sum: f64,
}

impl DensePopulationSummary {
    #[inline]
    pub fn alt_counts(&self) -> &[u32] {
        &self.alt_counts
    }

    #[inline]
    pub fn called_counts(&self) -> &[u32] {
        &self.called_counts
    }

    #[inline]
    pub fn haplotype_capacity(&self) -> usize {
        self.haplotype_capacity
    }

    #[inline]
    fn segregating_site_count(&self) -> usize {
        self.segregating_sites
    }

    #[inline]
    fn cached_pi_sum(&self) -> f64 {
        self.pi_sum
    }
}

const SUMMARY_PARALLEL_THRESHOLD: usize = 2048;

pub fn build_dense_population_summary(
    matrix: &DenseGenotypeMatrix,
    haplotypes: &[(usize, HaplotypeSide)],
) -> DensePopulationSummary {
    let membership = DenseMembership::build(matrix, haplotypes);
    let offsets = membership.offsets();
    let variant_count = matrix.variant_count();
    let stride = matrix.stride();
    let data = matrix.data();
    let mut alt_counts = vec![0u32; variant_count];
    let mut called_counts = vec![0u32; variant_count];
    let missing = matrix.missing_slice();

    let (segregating_sites, pi_sum) = if variant_count < SUMMARY_PARALLEL_THRESHOLD {
        if let Some(bits) = missing {
            let mut seg = 0usize;
            let mut pi_total = 0.0_f64;
            for variant_idx in 0..variant_count {
                let base = variant_idx * stride;
                let (called, alt) = dense_sum_alt_with_missing(data, base, offsets, bits);
                alt_counts[variant_idx] = alt as u32;
                called_counts[variant_idx] = called as u32;
                if called >= 2 && alt > 0 && alt < called {
                    seg += 1;
                }
                if let Some(value) = dense_pi_from_counts(called, alt) {
                    pi_total += value;
                }
            }
            (seg, pi_total)
        } else {
            let mut seg = 0usize;
            let mut pi_total = 0.0_f64;
            let total = offsets.len();
            for variant_idx in 0..variant_count {
                let base = variant_idx * stride;
                let alt = dense_sum_alt_no_missing(data, base, offsets);
                alt_counts[variant_idx] = alt as u32;
                called_counts[variant_idx] = total as u32;
                if alt > 0 && alt < total {
                    seg += 1;
                }
                if let Some(value) = dense_pi_from_counts(total, alt) {
                    pi_total += value;
                }
            }
            (seg, pi_total)
        }
    } else if let Some(bits) = missing {
        alt_counts
            .par_iter_mut()
            .zip_eq(called_counts.par_iter_mut())
            .enumerate()
            .fold(
                || (0usize, 0.0_f64),
                |(mut seg, mut pi_total), (variant_idx, (alt_slot, called_slot))| {
                    let base = variant_idx * stride;
                    let (called, alt) = dense_sum_alt_with_missing(data, base, offsets, bits);
                    *alt_slot = alt as u32;
                    *called_slot = called as u32;
                    if called >= 2 && alt > 0 && alt < called {
                        seg += 1;
                    }
                    if let Some(value) = dense_pi_from_counts(called, alt) {
                        pi_total += value;
                    }
                    (seg, pi_total)
                },
            )
            .reduce(|| (0usize, 0.0_f64), |a, b| (a.0 + b.0, a.1 + b.1))
    } else {
        let total = offsets.len();
        let total_u32 = total as u32;
        alt_counts
            .par_iter_mut()
            .zip_eq(called_counts.par_iter_mut())
            .enumerate()
            .fold(
                || (0usize, 0.0_f64),
                |(mut seg, mut pi_total), (variant_idx, (alt_slot, called_slot))| {
                    let base = variant_idx * stride;
                    let alt = dense_sum_alt_no_missing(data, base, offsets);
                    *alt_slot = alt as u32;
                    *called_slot = total_u32;
                    if alt > 0 && alt < total {
                        seg += 1;
                    }
                    if let Some(value) = dense_pi_from_counts(total, alt) {
                        pi_total += value;
                    }
                    (seg, pi_total)
                },
            )
            .reduce(|| (0usize, 0.0_f64), |a, b| (a.0 + b.0, a.1 + b.1))
    };

    DensePopulationSummary {
        alt_counts: Arc::from(alt_counts.into_boxed_slice()),
        called_counts: Arc::from(called_counts.into_boxed_slice()),
        haplotype_capacity: offsets.len(),
        segregating_sites,
        pi_sum,
    }
}

fn count_segregating_sites_from_summary(summary: &DensePopulationSummary) -> usize {
    summary.segregating_site_count()
}

fn calculate_pi_from_summary(summary: &DensePopulationSummary, seq_length: i64) -> f64 {
    calculate_pi_from_summary_with_precomputed(summary, seq_length, None)
}

fn calculate_pi_from_summary_with_precomputed(
    summary: &DensePopulationSummary,
    seq_length: i64,
    precomputed: Option<f64>,
) -> f64 {
    if summary.haplotype_capacity() <= 1 {
        log(
            LogLevel::Warning,
            &format!(
                "Cannot calculate pi: insufficient haplotypes ({})",
                summary.haplotype_capacity()
            ),
        );
        return 0.0;
    }

    if seq_length < 0 {
        log(
            LogLevel::Warning,
            &format!("Cannot calculate pi: invalid sequence length ({seq_length})"),
        );
        return 0.0;
    }

    if seq_length == 0 {
        log(
            LogLevel::Warning,
            "Cannot calculate pi: zero sequence length causes division by zero",
        );
        return f64::INFINITY;
    }

    let sum_pi = precomputed.unwrap_or_else(|| summary.cached_pi_sum());

    sum_pi / seq_length as f64
}

#[derive(Clone, Copy, Default)]
struct HudsonSummaryTotals {
    numerator_sum: f64,
    denominator_sum: f64,
    pi1_sum: f64,
    pi2_sum: f64,
    dxy_sum_all: f64,
}

fn aggregate_hudson_components_from_summaries(
    pop1: &DensePopulationSummary,
    pop2: &DensePopulationSummary,
) -> HudsonSummaryTotals {
    let len = pop1.alt_counts().len().min(pop2.alt_counts().len());
    let alt1 = pop1.alt_counts();
    let alt2 = pop2.alt_counts();
    let called1 = pop1.called_counts();
    let called2 = pop2.called_counts();
    let mut totals = HudsonSummaryTotals::default();

    for idx in 0..len {
        let n1 = called1[idx] as usize;
        let n2 = called2[idx] as usize;
        if n1 == 0 || n2 == 0 {
            continue;
        }

        let alt_count1 = alt1[idx] as usize;
        let alt_count2 = alt2[idx] as usize;
        let ref_count1 = n1 - alt_count1;
        let ref_count2 = n2 - alt_count2;

        let denom_pairs = (n1 * n2) as f64;
        if denom_pairs == 0.0 {
            continue;
        }

        let mut dxy = (alt_count1 * ref_count2 + ref_count1 * alt_count2) as f64 / denom_pairs;
        if dxy < 0.0 {
            dxy = 0.0;
        } else if dxy > 1.0 {
            dxy = 1.0;
        }
        totals.dxy_sum_all += dxy;

        if n1 < 2 || n2 < 2 {
            continue;
        }

        let denom1 = (n1 * (n1 - 1)) as f64;
        let denom2 = (n2 * (n2 - 1)) as f64;
        let pi1 = if denom1 > 0.0 {
            2.0 * (alt_count1 as f64) * (ref_count1 as f64) / denom1
        } else {
            0.0
        };
        let pi2 = if denom2 > 0.0 {
            2.0 * (alt_count2 as f64) * (ref_count2 as f64) / denom2
        } else {
            0.0
        };

        totals.pi1_sum += pi1;
        totals.pi2_sum += pi2;

        if dxy > FST_EPSILON {
            totals.numerator_sum += dxy - 0.5 * (pi1 + pi2);
            totals.denominator_sum += dxy;
        } else {
            let pi_avg = 0.5 * (pi1 + pi2);
            if pi_avg.abs() <= FST_EPSILON {
                // contributes zero to both sums
            }
        }
    }

    totals
}

fn hudson_component_sums(sites: &[SiteFstHudson]) -> (f64, f64) {
    let mut num_sum = 0.0_f64;
    let mut den_sum = 0.0_f64;
    for s in sites {
        if let (Some(nc), Some(dc)) = (s.num_component, s.den_component) {
            num_sum += nc;
            den_sum += dc;
        }
    }
    (num_sum, den_sum)
}

fn dxy_from_summaries(
    pop1: &DensePopulationSummary,
    pop2: &DensePopulationSummary,
    sequence_length: i64,
) -> Option<f64> {
    if sequence_length <= 0 {
        return None;
    }
    let len = pop1.alt_counts().len().min(pop2.alt_counts().len());
    let alt1 = pop1.alt_counts();
    let alt2 = pop2.alt_counts();
    let called1 = pop1.called_counts();
    let called2 = pop2.called_counts();
    let mut sum_dxy = 0.0_f64;

    for idx in 0..len {
        let n1 = called1[idx] as usize;
        let alt_count1 = alt1[idx] as usize;
        let n2 = called2[idx] as usize;
        let alt_count2 = alt2[idx] as usize;
        if let Some(dxy) = dense_dxy_from_biallelic_counts(n1, alt_count1, n2, alt_count2) {
            sum_dxy += dxy;
        }
    }

    Some(sum_dxy / sequence_length as f64)
}

#[inline(always)]
fn dense_sum_alt_no_missing(data: &[u8], base: usize, offsets: &[usize]) -> usize {
    let mut sum = 0usize;
    unsafe {
        let ptr = data.as_ptr().add(base);
        for &offset in offsets {
            sum += *ptr.add(offset) as usize;
        }
    }
    sum
}

#[inline(always)]
fn dense_sum_alt_with_missing(
    data: &[u8],
    base: usize,
    offsets: &[usize],
    bits: &[u64],
) -> (usize, usize) {
    let mut alt = 0usize;
    let mut total = 0usize;
    unsafe {
        let ptr = data.as_ptr().add(base);
        for &offset in offsets {
            let idx = base + offset;
            if dense_missing(bits, idx) {
                continue;
            }
            alt += *ptr.add(offset) as usize;
            total += 1;
        }
    }
    (total, alt)
}

#[inline(always)]
fn dense_pi_from_counts(total_called: usize, alt_count: usize) -> Option<f64> {
    if total_called < 2 {
        return None;
    }
    let n = total_called as f64;
    let alt = alt_count as f64;
    let ref_count = (total_called - alt_count) as f64;
    let sum_sq = ref_count * ref_count + alt * alt;
    Some(n / (n - 1.0) * (1.0 - sum_sq / (n * n)))
}

#[inline(always)]
fn dense_dxy_from_biallelic_counts(n1: usize, alt1: usize, n2: usize, alt2: usize) -> Option<f64> {
    if n1 == 0 || n2 == 0 {
        return None;
    }
    let n1_f = n1 as f64;
    let n2_f = n2 as f64;
    let alt1_f = alt1 as f64 / n1_f;
    let alt2_f = alt2 as f64 / n2_f;
    let ref1 = 1.0 - alt1_f;
    let ref2 = 1.0 - alt2_f;
    let mut dot = ref1 * ref2 + alt1_f * alt2_f;
    if dot < 0.0 {
        dot = 0.0;
    }
    let mut dxy = 1.0 - dot;
    if dxy < 0.0 {
        dxy = 0.0;
    } else if dxy > 1.0 {
        dxy = 1.0;
    }
    Some(dxy)
}

#[inline(always)]
fn dense_fst_components_from_biallelic(
    dxy: Option<f64>,
    pi1: Option<f64>,
    pi2: Option<f64>,
) -> (Option<f64>, Option<f64>, Option<f64>) {
    match (dxy, pi1, pi2) {
        (Some(d), Some(p1), Some(p2)) => {
            if d > FST_EPSILON {
                let num = d - 0.5 * (p1 + p2);
                (Some(num / d), Some(num), Some(d))
            } else {
                let pi_avg = 0.5 * (p1 + p2);
                if pi_avg.abs() <= FST_EPSILON {
                    (Some(0.0), Some(0.0), Some(0.0))
                } else {
                    (None, None, None)
                }
            }
        }
        _ => (None, None, None),
    }
}

/// General function to calculate Weir & Cockerham FST components and estimates at a single site.
///
/// This function takes a variant and a mapping of haplotypes to subpopulation identifiers.
/// It calculates allele frequencies per subpopulation, then computes Weir & Cockerham's
/// variance components 'a' (among populations) and 'b' (within populations).
/// From these, it derives overall and pairwise FST estimates for the site.
///
/// LIMITATION: This implementation is effectively biallelic only. Multi-allelic sites
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
fn fst_estimate_from_components(site_a: f64, site_b: f64) -> FstEstimate {
    let denominator = site_a + site_b;
    let eps = 1e-9;

    if denominator > eps {
        FstEstimate::Calculable {
            value: site_a / denominator,
            sum_a: site_a,
            sum_b: site_b,
            num_informative_sites: 1,
        }
    } else if denominator < -eps {
        FstEstimate::ComponentsYieldIndeterminateRatio {
            sum_a: site_a,
            sum_b: site_b,
            num_informative_sites: 1,
        }
    } else if site_a.abs() > eps {
        FstEstimate::Calculable {
            value: site_a / denominator,
            sum_a: site_a,
            sum_b: site_b,
            num_informative_sites: 1,
        }
    } else {
        FstEstimate::NoInterPopulationVariance {
            sum_a: site_a,
            sum_b: site_b,
            sites_evaluated: 1,
        }
    }
}

fn calculate_fst_wc_at_site_with_membership(
    variant: &Variant,
    membership: &SubpopulationMembership,
    workspace: &mut WcSiteWorkspace,
) -> (
    FstEstimate,
    HashMap<String, FstEstimate>,
    (f64, f64),
    HashMap<String, usize>,
    HashMap<String, (f64, f64)>,
) {
    workspace.ensure_capacity(membership.group_count());
    workspace.reset();

    for (sample_idx, genotype_opt) in variant.genotypes.iter().enumerate() {
        let Some(genotype) = genotype_opt else {
            continue;
        };

        if let Some(&allele) = genotype.get(0) {
            let group = membership
                .left
                .get(sample_idx)
                .copied()
                .unwrap_or(INVALID_GROUP);
            if group != INVALID_GROUP {
                let idx = group as usize;
                workspace.total_counts[idx] += 1;
                if allele != 0 {
                    workspace.alt_counts[idx] += 1;
                }
            }
        }

        if let Some(&allele) = genotype.get(1) {
            let group = membership
                .right
                .get(sample_idx)
                .copied()
                .unwrap_or(INVALID_GROUP);
            if group != INVALID_GROUP {
                let idx = group as usize;
                workspace.total_counts[idx] += 1;
                if allele != 0 {
                    workspace.alt_counts[idx] += 1;
                }
            }
        }
    }

    let mut pop_sizes = HashMap::new();
    let mut total_called = 0usize;
    let mut total_alt = 0usize;

    for idx in 0..membership.group_count() {
        let total = workspace.total_counts[idx];
        if total == 0 {
            continue;
        }
        let alt = workspace.alt_counts[idx];
        total_called += total;
        total_alt += alt;
        let freq = alt as f64 / total as f64;
        workspace.stats.push(PopSiteStat { total, freq });
        pop_sizes.insert(membership.label(idx).to_string(), total);
    }

    if workspace.stats.len() < 2 {
        let insufficient_data_estimate = FstEstimate::InsufficientDataForEstimation {
            sum_a: 0.0,
            sum_b: 0.0,
            sites_attempted: 1,
        };
        return (
            insufficient_data_estimate,
            HashMap::new(),
            (0.0, 0.0),
            pop_sizes,
            HashMap::new(),
        );
    }

    let global_freq = if total_called > 0 {
        total_alt as f64 / total_called as f64
    } else {
        0.0
    };

    let (site_a, site_b) = calculate_variance_components(&workspace.stats, global_freq);
    let overall_fst_at_site = fst_estimate_from_components(site_a, site_b);

    let mut pairwise_fst_estimate_map = HashMap::new();
    let mut pairwise_variance_components_map = HashMap::new();

    for descriptor in &membership.pair_keys {
        let idx_a = descriptor.left as usize;
        let idx_b = descriptor.right as usize;
        let total_a = workspace.total_counts[idx_a];
        let total_b = workspace.total_counts[idx_b];
        let key = descriptor.key.clone();

        if total_a == 0 || total_b == 0 {
            pairwise_variance_components_map.insert(key.clone(), (0.0, 0.0));
            pairwise_fst_estimate_map.insert(
                key,
                FstEstimate::InsufficientDataForEstimation {
                    sum_a: 0.0,
                    sum_b: 0.0,
                    sites_attempted: 1,
                },
            );
            continue;
        }

        let alt_a = workspace.alt_counts[idx_a];
        let alt_b = workspace.alt_counts[idx_b];
        let freq_a = alt_a as f64 / total_a as f64;
        let freq_b = alt_b as f64 / total_b as f64;
        let pair_total = total_a + total_b;
        let pair_global = if pair_total > 0 {
            (alt_a + alt_b) as f64 / pair_total as f64
        } else {
            0.0
        };

        let pair_stats = [
            PopSiteStat {
                total: total_a,
                freq: freq_a,
            },
            PopSiteStat {
                total: total_b,
                freq: freq_b,
            },
        ];

        let (pairwise_a_xy, pairwise_b_xy) =
            calculate_variance_components(&pair_stats, pair_global);
        let pairwise_fst_val = fst_estimate_from_components(pairwise_a_xy, pairwise_b_xy);
        pairwise_variance_components_map.insert(key.clone(), (pairwise_a_xy, pairwise_b_xy));
        pairwise_fst_estimate_map.insert(key, pairwise_fst_val);
    }

    workspace.stats.clear();

    (
        overall_fst_at_site,
        pairwise_fst_estimate_map,
        (site_a, site_b),
        pop_sizes,
        pairwise_variance_components_map,
    )
}

fn calculate_variance_components(
    pop_stats: &[PopSiteStat], // (n_i, p_i)
    global_freq: f64,          // p̄
) -> (f64, f64) {
    let r = pop_stats.len() as f64; // Number of subpopulations
    if r < 2.0 {
        // Need at least two populations to compare
        return (0.0, 0.0);
    }

    let mut n_values = Vec::with_capacity(pop_stats.len());
    let mut total_haplotypes = 0_usize;
    for stat in pop_stats.iter() {
        n_values.push(stat.total as f64);
        total_haplotypes += stat.total;
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
    for stat in pop_stats.iter() {
        let diff_p = stat.freq - global_p;
        numerator_s_squared += (stat.total as f64) * diff_p * diff_p;
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

    if let (Some(summary1), Some(summary2)) = (
        pop1_context.dense_summary.as_deref(),
        pop2_context.dense_summary.as_deref(),
    ) {
        let dxy_value = dxy_from_summaries(summary1, summary2, pop1_context.sequence_length);
        return Ok(DxyHudsonResult { d_xy: dxy_value });
    }

    if let (Some(matrix1), Some(matrix2)) =
        (pop1_context.dense_genotypes, pop2_context.dense_genotypes)
    {
        if std::ptr::eq(matrix1, matrix2) && matrix1.ploidy() == 2 {
            let membership1 = DenseMembership::build(matrix1, &pop1_context.haplotypes);
            let membership2 = DenseMembership::build(matrix2, &pop2_context.haplotypes);
            let d_xy_value = calculate_dxy_dense(
                matrix1,
                &membership1,
                &membership2,
                pop1_context.sequence_length,
            );
            return Ok(DxyHudsonResult { d_xy: d_xy_value });
        }
    }

    // Use unbiased per-site aggregation approach
    let mut sum_dxy = 0.0;
    let mut variant_count = 0;

    let pop1_mem = HapMembership::build(pop1_context.sample_names.len(), &pop1_context.haplotypes);
    let pop2_mem = HapMembership::build(pop2_context.sample_names.len(), &pop2_context.haplotypes);

    for variant in pop1_context.variants {
        // Get allele counts for both populations at this variant
        let counts1 = freq_summary_for_pop(variant, &pop1_mem);
        let counts2 = freq_summary_for_pop(variant, &pop2_mem);

        // Calculate per-site Dxy using existing helper
        if let Some(dxy_site) = dxy_from_counts(&counts1, &counts2) {
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
        log(
            LogLevel::Warning,
            &format!(
                "Invalid sequence length for Dxy calculation: {}",
                effective_sequence_length
            ),
        );
        None
    };

    Ok(DxyHudsonResult { d_xy: d_xy_value })
}

fn calculate_dxy_dense(
    matrix: &DenseGenotypeMatrix,
    pop1_mem: &DenseMembership,
    pop2_mem: &DenseMembership,
    sequence_length: i64,
) -> Option<f64> {
    if pop1_mem.len() == 0 || pop2_mem.len() == 0 {
        return None;
    }
    if sequence_length <= 0 {
        return None;
    }

    let mut counts1 = vec![0u32; 256];
    let mut used1 = Vec::with_capacity(8);
    let mut counts2 = vec![0u32; 256];
    let mut used2 = Vec::with_capacity(8);
    let mut sum_dxy = 0.0_f64;

    for variant_idx in 0..matrix.variant_count() {
        let (n1, _) = dense_collect_counts(matrix, pop1_mem, variant_idx, &mut counts1, &mut used1);
        let (n2, _) = dense_collect_counts(matrix, pop2_mem, variant_idx, &mut counts2, &mut used2);

        if n1 == 0 || n2 == 0 {
            dense_reset_counts(&mut counts1, &mut used1);
            dense_reset_counts(&mut counts2, &mut used2);
            continue;
        }

        let inv1 = 1.0 / n1 as f64;
        let inv2 = 1.0 / n2 as f64;
        let mut dot = 0.0_f64;
        if used1.len() <= used2.len() {
            for &allele in &used1 {
                let c1 = counts1[allele];
                if c1 == 0 {
                    continue;
                }
                let c2 = if allele < counts2.len() {
                    counts2[allele]
                } else {
                    0
                };
                if c2 != 0 {
                    dot += (c1 as f64 * inv1) * (c2 as f64 * inv2);
                }
            }
        } else {
            for &allele in &used2 {
                let c2 = counts2[allele];
                if c2 == 0 {
                    continue;
                }
                let c1 = if allele < counts1.len() {
                    counts1[allele]
                } else {
                    0
                };
                if c1 != 0 {
                    dot += (c1 as f64 * inv1) * (c2 as f64 * inv2);
                }
            }
        }
        let dxy_site = (1.0 - dot).max(0.0).min(1.0);
        sum_dxy += dxy_site;

        dense_reset_counts(&mut counts1, &mut used1);
        dense_reset_counts(&mut counts2, &mut used2);
    }

    Some(sum_dxy / sequence_length as f64)
}

/// Extract allele counts for a population at a specific variant site.
///
/// Missing Data Handling Strategy:
/// This function implements the "complete case analysis" approach for missing data:
/// - Only counts haplotypes with called genotypes at this site
/// - Missing genotypes (None) are excluded from frequency calculations
/// - Returns an [`AlleleCountSummary`] capturing the number of called haplotypes,
///   per-allele tallies, and Σ count² for downstream calculations
///
/// Why This Approach:
/// 1. Unbiased estimation: Using only called haplotypes gives unbiased allele frequencies
/// 2. Site-specific sample sizes: Each site can have different effective sample sizes
/// 3. Robust to missing patterns: Works regardless of missing data patterns
/// 4. Conservative: Sites with insufficient data will have low n_called and may be excluded
///
/// Mathematical Impact:
/// The resulting frequencies {p_a} are computed from n_called haplotypes, making
/// the downstream π and D_xy calculations appropriate for the actual available data.
#[derive(Clone, Debug)]
struct AlleleCountSummary {
    total_called: usize,
    sum_counts_sq: f64,
    counts: Vec<(i32, usize)>,
}

impl AlleleCountSummary {
    #[inline]
    fn with_capacity(capacity_hint: usize) -> Self {
        Self {
            total_called: 0,
            sum_counts_sq: 0.0,
            counts: Vec::with_capacity(capacity_hint.min(8)),
        }
    }

    #[inline]
    fn record(&mut self, allele: i32) {
        self.total_called += 1;
        match self
            .counts
            .binary_search_by_key(&allele, |&(stored, _)| stored)
        {
            Ok(idx) => {
                let entry = &mut self.counts[idx];
                self.sum_counts_sq += (2 * entry.1 + 1) as f64;
                entry.1 += 1;
            }
            Err(idx) => {
                self.counts.insert(idx, (allele, 1));
                self.sum_counts_sq += 1.0;
            }
        }
    }

    #[inline]
    fn total_called(&self) -> usize {
        self.total_called
    }

    #[inline]
    fn sum_counts_sq(&self) -> f64 {
        self.sum_counts_sq
    }

    #[inline]
    fn entries(&self) -> &[(i32, usize)] {
        &self.counts
    }
}

fn freq_summary_for_pop(variant: &Variant, membership: &HapMembership) -> AlleleCountSummary {
    let mut summary = AlleleCountSummary::with_capacity(membership.total);
    for (idx, genotype_opt) in variant.genotypes.iter().enumerate() {
        let Some(genotype) = genotype_opt else {
            continue;
        };
        if membership.left.get(idx).copied().unwrap_or(false) {
            if let Some(&allele) = genotype.get(0) {
                summary.record(allele as i32);
            }
        }
        if membership.right.get(idx).copied().unwrap_or(false) {
            if let Some(&allele) = genotype.get(1) {
                summary.record(allele as i32);
            }
        }
    }
    summary
}

/// Compute per-site nucleotide diversity (π) using the unbiased estimator.
///
/// Mathematical Foundation:
/// For a site with n called haplotypes and allele frequencies {p_a}, the unbiased
/// estimator of within-population diversity is:
///
/// ```text
/// π = (n/(n-1)) * (1 - Σ p_a²)
/// ```
///
/// This corrects for finite sample size bias. The term (1 - Σ p_a²) is the
/// expected heterozygosity, and the n/(n-1) factor provides the unbiased correction
/// for haploid/haplotype data.
///
/// Multi-allelic Support:
/// Works correctly for any number of alleles by summing p_a² over all observed alleles.
///
/// Missing Data Handling:
/// Only uses called haplotypes at this site; n and {p_a} are computed from available data.
#[inline]
fn pi_from_components(total_called: usize, sum_counts_sq: f64) -> Option<f64> {
    if total_called < 2 {
        return None;
    }

    let n = total_called as f64;
    let inv_n = 1.0 / n;
    let sum_p2 = sum_counts_sq * inv_n * inv_n;

    Some(n / (n - 1.0) * (1.0 - sum_p2))
}

#[inline]
fn pi_from_summary(summary: &AlleleCountSummary) -> Option<f64> {
    pi_from_components(summary.total_called(), summary.sum_counts_sq())
}

#[derive(Default)]
struct PiComputationState {
    counts: Vec<u32>,
    used_indices: Vec<usize>,
}

#[derive(Clone, Copy, Default)]
struct PiComputationOutcome {
    total_called: usize,
    sum_counts_sq: f64,
    distinct_alleles: usize,
}

impl PiComputationOutcome {
    #[inline]
    fn pi(self) -> Option<f64> {
        pi_from_components(self.total_called, self.sum_counts_sq)
    }
}

#[inline]
fn compute_pi_metrics_fast(
    variant: &Variant,
    membership: &HapMembership,
    state: &mut PiComputationState,
) -> PiComputationOutcome {
    let mut total_called = 0usize;

    for (sample_index, genotype_opt) in variant.genotypes.iter().enumerate() {
        let Some(genotype) = genotype_opt else {
            continue;
        };
        if membership.left.get(sample_index).copied().unwrap_or(false) {
            if let Some(&allele) = genotype.get(0) {
                let allele_value = i16::from(allele);
                if allele_value < 0 {
                    continue;
                }
                let idx = allele_value as usize;
                if idx >= state.counts.len() {
                    state.counts.resize(idx + 1, 0);
                }
                if state.counts[idx] == 0 {
                    state.used_indices.push(idx);
                }
                state.counts[idx] += 1;
                total_called += 1;
            }
        }
        if membership.right.get(sample_index).copied().unwrap_or(false) {
            if let Some(&allele) = genotype.get(1) {
                let allele_value = i16::from(allele);
                if allele_value < 0 {
                    continue;
                }
                let idx = allele_value as usize;
                if idx >= state.counts.len() {
                    state.counts.resize(idx + 1, 0);
                }
                if state.counts[idx] == 0 {
                    state.used_indices.push(idx);
                }
                state.counts[idx] += 1;
                total_called += 1;
            }
        }
    }

    let mut sum_counts_sq = 0.0_f64;
    for &idx in &state.used_indices {
        let count = state.counts[idx] as f64;
        sum_counts_sq += count * count;
        state.counts[idx] = 0;
    }
    let distinct_alleles = state.used_indices.len();
    state.used_indices.clear();

    PiComputationOutcome {
        total_called,
        sum_counts_sq,
        distinct_alleles,
    }
}

fn dense_collect_counts(
    matrix: &DenseGenotypeMatrix,
    membership: &DenseMembership,
    variant_idx: usize,
    counts: &mut Vec<u32>,
    used: &mut Vec<usize>,
) -> (usize, f64) {
    let stride = matrix.stride();
    let base = variant_idx * stride;
    let data = matrix.data();
    let offsets = membership.offsets();
    let total_called = if let Some(bits) = matrix.missing_slice() {
        let mut called = 0usize;
        unsafe {
            let ptr = data.as_ptr();
            for &offset in offsets {
                let idx = base + offset;
                if dense_missing(bits, idx) {
                    continue;
                }
                let allele = *ptr.add(idx) as usize;
                if allele >= counts.len() {
                    counts.resize(allele + 1, 0);
                }
                if counts[allele] == 0 {
                    used.push(allele);
                }
                counts[allele] += 1;
                called += 1;
            }
        }
        called
    } else {
        unsafe {
            let ptr = data.as_ptr();
            for &offset in offsets {
                let idx = base + offset;
                let allele = *ptr.add(idx) as usize;
                if allele >= counts.len() {
                    counts.resize(allele + 1, 0);
                }
                if counts[allele] == 0 {
                    used.push(allele);
                }
                counts[allele] += 1;
            }
        }
        offsets.len()
    };

    let mut sum_counts_sq = 0.0;
    for &allele in used.iter() {
        let count = counts[allele] as f64;
        sum_counts_sq += count * count;
    }

    (total_called, sum_counts_sq)
}

fn dense_reset_counts(counts: &mut Vec<u32>, used: &mut Vec<usize>) {
    for &idx in used.iter() {
        counts[idx] = 0;
    }
    used.clear();
}

/// Compute between-population diversity (D_xy) from allele counts.
///
/// Mathematical Foundation:
/// For two populations with allele frequencies {p_1a} and {p_2a}, the between-population
/// diversity is the average pairwise difference between haplotypes from different populations:
///
/// ```text
/// D_xy = 1 - Σ_a (p_1a * p_2a)
/// ```
///
/// This is Hudson's H_B term - the probability that two randomly chosen haplotypes
/// from different populations differ at this site.
///
/// Multi-allelic Support:
/// Correctly handles any number of alleles by computing the dot product of frequency vectors.
///
/// Missing Data Handling:
/// Uses only called haplotypes from each population at this site to compute frequencies.
fn dxy_from_counts(c1: &AlleleCountSummary, c2: &AlleleCountSummary) -> Option<f64> {
    let n1 = c1.total_called();
    let n2 = c2.total_called();
    if n1 == 0 || n2 == 0 {
        return None;
    }
    let mut dot = 0.0_f64;
    let entries1 = c1.entries();
    let entries2 = c2.entries();
    let mut i = 0usize;
    let mut j = 0usize;
    let inv1 = 1.0 / n1 as f64;
    let inv2 = 1.0 / n2 as f64;

    while i < entries1.len() && j < entries2.len() {
        match entries1[i].0.cmp(&entries2[j].0) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                dot += (entries1[i].1 as f64 * inv1) * (entries2[j].1 as f64 * inv2);
                i += 1;
                j += 1;
            }
        }
    }
    // Clamp to [0,1] to handle floating-point errors in multi-allelic tallies
    let dxy = 1.0 - dot;
    Some(dxy.max(0.0).min(1.0))
}

/// Compute per-site Hudson FST components from a single variant.
///
/// Mathematical Foundation (Hudson et al. 1992):
/// Hudson's FST at a single site is defined as:
///
/// ```text
/// FST_i = (D_xy,i - 0.5*(π_1,i + π_2,i)) / D_xy,i = (H_B - H_S) / H_B
/// ```
///
/// Where:
/// - H_B = D_xy,i = 1 - Σ_a p_1a * p_2a = between-population diversity
/// - H_S = 0.5*(π_1,i + π_2,i) = average within-population diversity
/// - π_k,i = (n_k/(n_k-1)) * (1 - Σ_a p_ka²) = unbiased within-population estimator
///
/// Literature Alignment:
/// - Hudson et al. (1992): Original definition using H_B and H_S
/// - scikit-allel: `average_hudson_fst` uses identical formula and ratio-of-sums aggregation
/// - ANGSD: Uses same per-site definition and weighted window estimator
/// - Biallelic equivalence: For 2 alleles, this equals (p₁-p₂)² minus finite-sample corrections
///   divided by D_xy = p₁(1-p₂) + p₂(1-p₁)
///
/// Per-site Components:
/// - Numerator: D_xy,i - 0.5*(π_1,i + π_2,i) = Hudson numerator with finite-sample corrections
/// - Denominator: D_xy,i = Hudson denominator  
/// - FST: numerator/denominator when denominator > FST_EPSILON
///
/// Multi-allelic Support:
/// All formulas use Σ_a notation, so they generalize correctly beyond biallelic SNPs.
/// D_xy = 1 - Σ_a p_1a * p_2a handles any number of alleles.
///
/// Multi-allelic and Missing Data:
/// Handles any number of alleles and computes frequencies from called haplotypes only.
fn hudson_site_from_variant(
    variant: &Variant,
    pop1_mem: &HapMembership,
    pop2_mem: &HapMembership,
) -> SiteFstHudson {
    let counts1 = freq_summary_for_pop(variant, pop1_mem);
    let counts2 = freq_summary_for_pop(variant, pop2_mem);

    let n1 = counts1.total_called();
    let n2 = counts2.total_called();

    let pi1 = pi_from_summary(&counts1);
    let pi2 = pi_from_summary(&counts2);
    let dxy = dxy_from_counts(&counts1, &counts2);

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
/// IMPORTANT: This function assumes variant compatibility between populations.
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

    let mut sites = Vec::with_capacity(region.len() as usize);
    let mut variant_iter = pop1_context.variants.iter().peekable();
    let pop1_mem = HapMembership::build(pop1_context.sample_names.len(), &pop1_context.haplotypes);
    let pop2_mem = HapMembership::build(pop2_context.sample_names.len(), &pop2_context.haplotypes);

    for pos in region.start..=region.end {
        while let Some(next_variant) = variant_iter.peek() {
            if next_variant.position < pos {
                variant_iter.next();
            } else {
                break;
            }
        }

        if let Some(next_variant) = variant_iter.peek() {
            if next_variant.position == pos {
                let variant = variant_iter.next().expect("peeked variant must exist");
                sites.push(hudson_site_from_variant(variant, &pop1_mem, &pop2_mem));
                continue;
            }
        }

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

    sites
}

fn dense_hudson_sites(
    matrix: &DenseGenotypeMatrix,
    variants: &[Variant],
    pop1_mem: &DenseMembership,
    pop2_mem: &DenseMembership,
) -> Vec<SiteFstHudson> {
    if matrix.max_allele() <= 1 {
        return dense_hudson_sites_biallelic(matrix, variants, pop1_mem, pop2_mem);
    }
    dense_hudson_sites_general(matrix, variants, pop1_mem, pop2_mem)
}

fn dense_hudson_sites_general(
    matrix: &DenseGenotypeMatrix,
    variants: &[Variant],
    pop1_mem: &DenseMembership,
    pop2_mem: &DenseMembership,
) -> Vec<SiteFstHudson> {
    let mut counts1 = vec![0u32; 256];
    let mut used1 = Vec::with_capacity(8);
    let mut counts2 = vec![0u32; 256];
    let mut used2 = Vec::with_capacity(8);
    let mut sites = Vec::with_capacity(variants.len());

    for (variant_idx, variant) in variants.iter().enumerate() {
        let (n1, sum_sq1) =
            dense_collect_counts(matrix, pop1_mem, variant_idx, &mut counts1, &mut used1);
        let (n2, sum_sq2) =
            dense_collect_counts(matrix, pop2_mem, variant_idx, &mut counts2, &mut used2);

        let pi1 = if n1 >= 2 {
            let n = n1 as f64;
            Some(n / (n - 1.0) * (1.0 - sum_sq1 / (n * n)))
        } else {
            None
        };
        let pi2 = if n2 >= 2 {
            let n = n2 as f64;
            Some(n / (n - 1.0) * (1.0 - sum_sq2 / (n * n)))
        } else {
            None
        };

        let dxy = if n1 == 0 || n2 == 0 {
            None
        } else {
            let inv1 = 1.0 / n1 as f64;
            let inv2 = 1.0 / n2 as f64;
            let mut dot = 0.0_f64;
            if used1.len() <= used2.len() {
                for &allele in &used1 {
                    let c1 = counts1[allele];
                    if c1 == 0 {
                        continue;
                    }
                    let c2 = if allele < counts2.len() {
                        counts2[allele]
                    } else {
                        0
                    };
                    if c2 != 0 {
                        dot += (c1 as f64 * inv1) * (c2 as f64 * inv2);
                    }
                }
            } else {
                for &allele in &used2 {
                    let c2 = counts2[allele];
                    if c2 == 0 {
                        continue;
                    }
                    let c1 = if allele < counts1.len() {
                        counts1[allele]
                    } else {
                        0
                    };
                    if c1 != 0 {
                        dot += (c1 as f64 * inv1) * (c2 as f64 * inv2);
                    }
                }
            }
            Some((1.0 - dot).max(0.0).min(1.0))
        };

        let (fst, num_component, den_component) = match (dxy, pi1, pi2) {
            (Some(d), Some(p1), Some(p2)) => {
                if d > FST_EPSILON {
                    let num = d - 0.5 * (p1 + p2);
                    (Some(num / d), Some(num), Some(d))
                } else {
                    let pi_avg = 0.5 * (p1 + p2);
                    if pi_avg.abs() <= FST_EPSILON {
                        (Some(0.0), Some(0.0), Some(0.0))
                    } else {
                        (None, None, None)
                    }
                }
            }
            _ => (None, None, None),
        };

        sites.push(SiteFstHudson {
            position: ZeroBasedPosition(variant.position).to_one_based(),
            fst,
            d_xy: dxy,
            pi_pop1: pi1,
            pi_pop2: pi2,
            n1_called: n1,
            n2_called: n2,
            num_component,
            den_component,
        });

        dense_reset_counts(&mut counts1, &mut used1);
        dense_reset_counts(&mut counts2, &mut used2);
    }

    sites
}

fn dense_hudson_sites_biallelic(
    matrix: &DenseGenotypeMatrix,
    variants: &[Variant],
    pop1_mem: &DenseMembership,
    pop2_mem: &DenseMembership,
) -> Vec<SiteFstHudson> {
    let offsets1 = pop1_mem.offsets();
    let offsets2 = pop2_mem.offsets();
    let stride = matrix.stride();
    let data = matrix.data();
    let mut sites = Vec::with_capacity(variants.len());

    match matrix.missing_slice() {
        Some(bits) => {
            for (variant_idx, variant) in variants.iter().enumerate() {
                let base = variant_idx * stride;
                let (n1, alt1) = dense_sum_alt_with_missing(data, base, offsets1, bits);
                let (n2, alt2) = dense_sum_alt_with_missing(data, base, offsets2, bits);

                let pi1 = dense_pi_from_counts(n1, alt1);
                let pi2 = dense_pi_from_counts(n2, alt2);
                let dxy = dense_dxy_from_biallelic_counts(n1, alt1, n2, alt2);

                let (fst, num_component, den_component) =
                    dense_fst_components_from_biallelic(dxy, pi1, pi2);

                sites.push(SiteFstHudson {
                    position: ZeroBasedPosition(variant.position).to_one_based(),
                    fst,
                    d_xy: dxy,
                    pi_pop1: pi1,
                    pi_pop2: pi2,
                    n1_called: n1,
                    n2_called: n2,
                    num_component,
                    den_component,
                });
            }
        }
        None => {
            let n1_total = offsets1.len();
            let n2_total = offsets2.len();
            let n1_f = n1_total as f64;
            let n2_f = n2_total as f64;
            let scale1 = if n1_total >= 2 {
                Some((n1_f / (n1_f - 1.0), 1.0 / (n1_f * n1_f)))
            } else {
                None
            };
            let scale2 = if n2_total >= 2 {
                Some((n2_f / (n2_f - 1.0), 1.0 / (n2_f * n2_f)))
            } else {
                None
            };

            for (variant_idx, variant) in variants.iter().enumerate() {
                let base = variant_idx * stride;
                let alt1 = dense_sum_alt_no_missing(data, base, offsets1);
                let alt2 = dense_sum_alt_no_missing(data, base, offsets2);

                let pi1 = scale1.map(|(scale, inv_n_sq)| {
                    if alt1 == 0 || alt1 == n1_total {
                        0.0
                    } else {
                        let alt_f = alt1 as f64;
                        let ref_f = (n1_total - alt1) as f64;
                        scale * (1.0 - (ref_f * ref_f + alt_f * alt_f) * inv_n_sq)
                    }
                });
                let pi2 = scale2.map(|(scale, inv_n_sq)| {
                    if alt2 == 0 || alt2 == n2_total {
                        0.0
                    } else {
                        let alt_f = alt2 as f64;
                        let ref_f = (n2_total - alt2) as f64;
                        scale * (1.0 - (ref_f * ref_f + alt_f * alt_f) * inv_n_sq)
                    }
                });

                let dxy = dense_dxy_from_biallelic_counts(n1_total, alt1, n2_total, alt2);
                let (fst, num_component, den_component) =
                    dense_fst_components_from_biallelic(dxy, pi1, pi2);

                sites.push(SiteFstHudson {
                    position: ZeroBasedPosition(variant.position).to_one_based(),
                    fst,
                    d_xy: dxy,
                    pi_pop1: pi1,
                    pi_pop2: pi2,
                    n1_called: n1_total,
                    n2_called: n2_total,
                    num_component,
                    den_component,
                });
            }
        }
    }

    sites
}

/// Aggregate per-site Hudson components into a window/regional FST using ratio of sums.
///
/// Mathematical Foundation:
/// The recommended window-level Hudson FST estimator is the "ratio of sums":
///
/// ```text
/// FST_window = Σ_i [D_xy,i - 0.5*(π_1,i + π_2,i)] / Σ_i D_xy,i
/// ```
///
/// This is a weighted average where sites with higher D_xy contribute more weight.
///
/// Literature Alignment:
/// - scikit-allel: `windowed_hudson_fst` uses identical ratio-of-sums aggregation
/// - ANGSD: Uses same weighted estimator for window-level FST
/// - PopGen consensus: Preferred over "mean of ratios" in methodological reviews
/// - Bhatia et al. (2013): Recommends keeping negative values (no truncation at 0)
///
/// Why Ratio-of-Sums (not Mean-of-Ratios):
/// 1. Stability: More robust to near-monomorphic sites with tiny denominators
/// 2. Weighting: Sites with higher diversity naturally get more influence
/// 3. Statistical properties: Better maximum likelihood properties under certain models
/// 4. Tool compatibility: Matches mainstream population genetics software
///
/// Missing Data Robustness:
/// Sites with undefined components (None, None) are excluded from both sums,
/// ensuring unbiased estimates regardless of missing data patterns.
///
/// Monomorphic Sites:
/// Sites with D_xy = π = 0 contribute (0, 0) to the sums, which is mathematically correct.
pub fn aggregate_hudson_from_sites(sites: &[SiteFstHudson]) -> Option<f64> {
    let (num_sum, den_sum) = hudson_component_sums(sites);
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
/// Algorithm Overview:
/// 1. Per-site calculation: For each variant site, compute Hudson FST components
///    using the unbiased estimators for π and D_xy
/// 2. Regional aggregation: Use "ratio of sums" to combine per-site components
///    into a single window-level FST estimate
///
/// Mathematical Approach:
/// - Per-site: FST_i = (D_xy_i - 0.5*(π_1i + π_2i)) / D_xy_i
/// - Regional: FST_region = Σ(numerator_i) / Σ(denominator_i)
///
/// Why Ratio of Sums:
/// This weighted approach is more robust than averaging per-site FST values because:
/// - Sites with higher diversity contribute more weight (appropriate for FST)
/// - Avoids instability from sites with very low diversity
/// - Matches standard implementations (ANGSD, scikit-allel)
/// - Provides better statistical properties under missing data
///
/// Missing Data Strategy:
/// This implementation uses a robust "complete case per site" approach:
/// 1. Per-site analysis: Each site uses only haplotypes with called genotypes
/// 2. Site-specific sample sizes: n1 and n2 can vary by site based on available data
/// 3. Exclusion of undefined sites: Sites with insufficient data (n < 2 in either pop)
///    contribute (None, None) components and are excluded from regional sums
/// 4. Unbiased aggregation: Regional FST uses only sites with valid components
///
/// Advantages over alternatives:
/// - More robust than listwise deletion (excluding samples with any missing data)
/// - Avoids bias from imputation methods
/// - Naturally handles different missing data patterns across sites
/// - Maintains statistical validity by using appropriate sample sizes per site
fn calculate_hudson_fst_for_pair_core<'a>(
    pop1_context: &PopulationContext<'a>,
    pop2_context: &PopulationContext<'a>,
    region: Option<QueryRegion>,
) -> Result<(HudsonFSTOutcome, Vec<SiteFstHudson>), VcfError> {
    if pop1_context.sequence_length <= 0 {
        return Err(VcfError::InvalidRegion(
            "Sequence length must be positive for Hudson FST calculation.".to_string(),
        ));
    }
    if pop1_context.sequence_length != pop2_context.sequence_length {
        return Err(VcfError::Parse(
            "Sequence length mismatch between population contexts for Hudson FST calculation."
                .to_string(),
        ));
    }
    if !variants_compatible(pop1_context.variants, pop2_context.variants) {
        return Err(VcfError::Parse(
            "Variant slices differ in positions/length.".to_string(),
        ));
    }

    let summary_pair = match (
        pop1_context.dense_summary.as_deref(),
        pop2_context.dense_summary.as_deref(),
    ) {
        (Some(a), Some(b)) => Some((a, b)),
        _ => None,
    };

    let mut summary_totals: Option<HudsonSummaryTotals> = None;
    let mut summary_refs: Option<(&DensePopulationSummary, &DensePopulationSummary)> = None;
    let dense_shared = match (pop1_context.dense_genotypes, pop2_context.dense_genotypes) {
        (Some(a), Some(b)) if std::ptr::eq(a, b) && a.ploidy() == 2 => Some(a),
        _ => None,
    };

    let mut site_values = Vec::new();
    let (num_sum, den_sum) = if let Some(reg) = region {
        site_values = calculate_hudson_fst_per_site(pop1_context, pop2_context, reg);
        hudson_component_sums(&site_values)
    } else if let Some((summary1, summary2)) = summary_pair {
        let totals = aggregate_hudson_components_from_summaries(summary1, summary2);
        summary_totals = Some(totals);
        summary_refs = Some((summary1, summary2));
        (totals.numerator_sum, totals.denominator_sum)
    } else if let Some(matrix) = dense_shared {
        if pop1_context.variants.is_empty() {
            (0.0, 0.0)
        } else {
            let pop1_mem = DenseMembership::build(matrix, &pop1_context.haplotypes);
            let pop2_mem = DenseMembership::build(matrix, &pop2_context.haplotypes);
            site_values = dense_hudson_sites(matrix, pop1_context.variants, &pop1_mem, &pop2_mem);
            hudson_component_sums(&site_values)
        }
    } else if pop1_context.variants.is_empty() {
        (0.0, 0.0)
    } else {
        let pop1_mem =
            HapMembership::build(pop1_context.sample_names.len(), &pop1_context.haplotypes);
        let pop2_mem =
            HapMembership::build(pop2_context.sample_names.len(), &pop2_context.haplotypes);
        site_values = pop1_context
            .variants
            .iter()
            .map(|variant| hudson_site_from_variant(variant, &pop1_mem, &pop2_mem))
            .collect();
        hudson_component_sums(&site_values)
    };

    let regional_fst = if den_sum > FST_EPSILON {
        Some(num_sum / den_sum)
    } else if num_sum.abs() <= FST_EPSILON {
        Some(0.0)
    } else {
        None
    };

    // Calculate auxiliary π and Dxy values for output (but don't use for FST)
    let (pi1_raw, pi2_raw, dxy_result) = if let (Some((summary1, summary2)), Some(totals)) =
        (summary_refs, summary_totals)
    {
        let pi1_raw = calculate_pi_from_summary_with_precomputed(
            summary1,
            pop1_context.sequence_length,
            Some(totals.pi1_sum),
        );
        let pi2_raw = calculate_pi_from_summary_with_precomputed(
            summary2,
            pop2_context.sequence_length,
            Some(totals.pi2_sum),
        );

        let dxy_value = if pop1_context.haplotypes.is_empty() || pop2_context.haplotypes.is_empty()
        {
            log(
                LogLevel::Warning,
                &format!(
                    "Cannot calculate Dxy for pops {:?}/{:?}: one or both have no haplotypes ({} and {} respectively).",
                    pop1_context.id,
                    pop2_context.id,
                    pop1_context.haplotypes.len(),
                    pop2_context.haplotypes.len()
                ),
            );
            None
        } else {
            Some(totals.dxy_sum_all / pop1_context.sequence_length as f64)
        };

        (pi1_raw, pi2_raw, DxyHudsonResult { d_xy: dxy_value })
    } else {
        let pi1_raw = calculate_pi_for_population(pop1_context);
        let pi2_raw = calculate_pi_for_population(pop2_context);
        let dxy_result = calculate_d_xy_hudson(pop1_context, pop2_context)?;
        (pi1_raw, pi2_raw, dxy_result)
    };

    let pi1_opt = if pi1_raw.is_finite() {
        Some(pi1_raw)
    } else {
        None
    };

    let pi2_opt = if pi2_raw.is_finite() {
        Some(pi2_raw)
    } else {
        None
    };

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
/// Primary Use Case:
/// This function is the main entry point for per-site Hudson FST analysis. It returns
/// both the aggregated regional FST estimate and detailed per-site components that can
/// be used for:
/// - Writing per-site FST values to FALSTA output files
/// - Quality control and validation of regional estimates
/// - Fine-scale analysis of FST variation across sites
///
/// Mathematical Guarantee:
/// The regional FST in the returned HudsonFSTOutcome equals the ratio-of-sums
/// aggregation of the per-site components: Σ(numerator_i) / Σ(denominator_i)
///
/// Performance Note:
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
/// Use Case:
/// For analyses that only need the regional FST estimate without per-site details.
/// This is computationally equivalent to the full function but discards per-site data.
///
/// Mathematical Equivalence:
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
    let unmasked_intervals =
        subtract_regions(&allowed_intervals, mask_regions_chr.map(|v| v.as_slice()));

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
fn subtract_regions(intervals: &[(i64, i64)], masks: Option<&[(i64, i64)]>) -> Vec<(i64, i64)> {
    let Some(masks) = masks else {
        return intervals.to_vec();
    };
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
    variants
        .par_iter()
        .filter(|variant| variant_is_segregating(variant))
        .count()
}

fn variant_is_segregating(variant: &Variant) -> bool {
    let mut first = None;
    for genotype_opt in &variant.genotypes {
        if let Some(genotype) = genotype_opt {
            for &allele in genotype {
                match first {
                    None => first = Some(allele),
                    Some(value) if value != allele => return true,
                    _ => {}
                }
            }
        }
    }
    false
}

pub fn count_segregating_sites_for_population(context: &PopulationContext<'_>) -> usize {
    if let Some(summary) = context.dense_summary.as_deref() {
        return count_segregating_sites_from_summary(summary);
    }
    if let Some(matrix) = context.dense_genotypes {
        if matrix.ploidy() == 2 {
            let membership = DenseMembership::build(matrix, &context.haplotypes);
            if membership.len() <= 1 {
                return 0;
            }
            return count_segregating_sites_dense(matrix, &membership);
        }
    }
    count_segregating_sites(context.variants)
}

fn count_segregating_sites_dense(
    matrix: &DenseGenotypeMatrix,
    membership: &DenseMembership,
) -> usize {
    if matrix.max_allele() <= 1 {
        return count_segregating_sites_dense_biallelic(matrix, membership);
    }

    let offsets = membership.offsets();
    if offsets.is_empty() {
        return 0;
    }
    let stride = matrix.stride();
    let data = matrix.data();
    let variant_count = matrix.variant_count();

    let should_parallel = dense_should_parallelize(variant_count, offsets.len());

    if !should_parallel {
        if let Some(bits) = matrix.missing_slice() {
            let mut segregating = 0usize;
            for variant_idx in 0..variant_count {
                let base = variant_idx * stride;
                let mut first = 0u8;
                let mut seen = false;
                let mut polymorphic = false;
                unsafe {
                    let ptr = data.as_ptr().add(base);
                    for &offset in offsets {
                        let idx = base + offset;
                        if dense_missing(bits, idx) {
                            continue;
                        }
                        let allele = *ptr.add(offset);
                        if seen {
                            if allele != first {
                                polymorphic = true;
                                break;
                            }
                        } else {
                            first = allele;
                            seen = true;
                        }
                    }
                }
                if polymorphic {
                    segregating += 1;
                }
            }
            segregating
        } else {
            let mut segregating = 0usize;
            for variant_idx in 0..variant_count {
                let base = variant_idx * stride;
                let mut first = 0u8;
                let mut seen = false;
                let mut polymorphic = false;
                unsafe {
                    let ptr = data.as_ptr().add(base);
                    for &offset in offsets {
                        let allele = *ptr.add(offset);
                        if seen {
                            if allele != first {
                                polymorphic = true;
                                break;
                            }
                        } else {
                            first = allele;
                            seen = true;
                        }
                    }
                }
                if polymorphic {
                    segregating += 1;
                }
            }
            segregating
        }
    } else if let Some(bits) = matrix.missing_slice() {
        (0..variant_count)
            .into_par_iter()
            .map(|variant_idx| {
                let base = variant_idx * stride;
                let mut first = 0u8;
                let mut seen = false;
                let mut polymorphic = false;
                unsafe {
                    let ptr = data.as_ptr().add(base);
                    for &offset in offsets {
                        let idx = base + offset;
                        if dense_missing(bits, idx) {
                            continue;
                        }
                        let allele = *ptr.add(offset);
                        if seen {
                            if allele != first {
                                polymorphic = true;
                                break;
                            }
                        } else {
                            first = allele;
                            seen = true;
                        }
                    }
                }
                usize::from(polymorphic)
            })
            .sum()
    } else {
        (0..variant_count)
            .into_par_iter()
            .map(|variant_idx| {
                let base = variant_idx * stride;
                let mut first = 0u8;
                let mut seen = false;
                let mut polymorphic = false;
                unsafe {
                    let ptr = data.as_ptr().add(base);
                    for &offset in offsets {
                        let allele = *ptr.add(offset);
                        if seen {
                            if allele != first {
                                polymorphic = true;
                                break;
                            }
                        } else {
                            first = allele;
                            seen = true;
                        }
                    }
                }
                usize::from(polymorphic)
            })
            .sum()
    }
}

fn count_segregating_sites_dense_biallelic(
    matrix: &DenseGenotypeMatrix,
    membership: &DenseMembership,
) -> usize {
    let offsets = membership.offsets();
    if offsets.len() < 2 {
        return 0;
    }
    let stride = matrix.stride();
    let data = matrix.data();
    let total = offsets.len();
    let variant_count = matrix.variant_count();

    let should_parallel = dense_should_parallelize(variant_count, total);

    if !should_parallel {
        if let Some(bits) = matrix.missing_slice() {
            let mut segregating = 0usize;
            for variant_idx in 0..variant_count {
                let base = variant_idx * stride;
                let (called, alt) = dense_sum_alt_with_missing(data, base, offsets, bits);
                if called >= 2 && alt > 0 && alt < called {
                    segregating += 1;
                }
            }
            segregating
        } else {
            let mut segregating = 0usize;
            for variant_idx in 0..variant_count {
                let base = variant_idx * stride;
                let alt = dense_sum_alt_no_missing(data, base, offsets);
                if alt > 0 && alt < total {
                    segregating += 1;
                }
            }
            segregating
        }
    } else if let Some(bits) = matrix.missing_slice() {
        (0..variant_count)
            .into_par_iter()
            .map(|variant_idx| {
                let base = variant_idx * stride;
                let (called, alt) = dense_sum_alt_with_missing(data, base, offsets, bits);
                usize::from(called >= 2 && alt > 0 && alt < called)
            })
            .sum()
    } else {
        (0..variant_count)
            .into_par_iter()
            .map(|variant_idx| {
                let base = variant_idx * stride;
                let alt = dense_sum_alt_no_missing(data, base, offsets);
                usize::from(alt > 0 && alt < total)
            })
            .sum()
    }
}

// Calculate pairwise differences and comparable sites between all sample pairs
/// This function computes, for each pair of samples, the number of differences across comparable
/// haplotypes, treating each haplotype separately (per-haplotype analysis).
///
/// Per-Haplotype Analysis:
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

static HARMONIC_CACHE: OnceLock<Mutex<Vec<f64>>> = OnceLock::new();

fn harmonic_cached(n: usize) -> f64 {
    let cache = HARMONIC_CACHE.get_or_init(|| Mutex::new(vec![0.0]));
    let mut values = cache.lock().expect("harmonic cache poisoned");
    if n >= values.len() {
        let mut last = *values.last().unwrap();
        for k in values.len()..=n {
            last += 1.0 / k as f64;
            values.push(last);
        }
    }
    values[n]
}

// Calculate the harmonic number H_n = sum_{k=1}^n 1/k
pub fn harmonic(n: usize) -> f64 {
    harmonic_cached(n)
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
/// * `0.0` if fewer than two haplotypes are provided or if no callable pairs exist
/// * `f64::INFINITY` when `seq_length` is zero (mirrors the legacy behaviour expected by callers)
/// * Otherwise, the average π across all sites scaled by the provided `seq_length`
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
        return 0.0;
    }

    if seq_length < 0 {
        log(
            LogLevel::Warning,
            &format!(
                "Cannot calculate pi: invalid sequence length ({})",
                seq_length
            ),
        );
        return 0.0;
    }

    if seq_length == 0 {
        log(
            LogLevel::Warning,
            "Cannot calculate pi: zero sequence length causes division by zero",
        );
        return f64::INFINITY;
    }

    let spinner = create_spinner(&format!(
        "Calculating π for {} haplotypes over {} bp using unbiased per-site aggregation",
        haplotypes_in_group.len(),
        seq_length
    ));

    // Use unbiased per-site aggregation approach with parallel processing for large variant sets
    let sample_count = variants
        .first()
        .map(|variant| variant.genotypes.len())
        .unwrap_or(0);
    let membership = HapMembership::build(sample_count, haplotypes_in_group);

    let (sum_pi, variant_count) = variants
        .par_iter()
        .fold(
            || (PiComputationState::default(), 0.0_f64, 0usize),
            |(mut state, mut partial_sum, mut partial_count), variant| {
                let metrics = compute_pi_metrics_fast(variant, &membership, &mut state);
                if let Some(pi_site) = metrics.pi() {
                    partial_sum += pi_site;
                    partial_count += 1;
                }
                (state, partial_sum, partial_count)
            },
        )
        .map(|(_, sum, count)| (sum, count))
        .reduce(
            || (0.0_f64, 0usize),
            |(sum_a, count_a), (sum_b, count_b)| (sum_a + sum_b, count_a + count_b),
        );

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

fn calculate_pi_dense_biallelic(
    matrix: &DenseGenotypeMatrix,
    membership: &DenseMembership,
    seq_length: i64,
) -> f64 {
    let offsets = membership.offsets();
    if offsets.len() <= 1 {
        return 0.0;
    }
    let stride = matrix.stride();
    let data = matrix.data();
    let mut sum_pi = 0.0_f64;
    let parallel = dense_should_parallelize(matrix.variant_count(), offsets.len());

    if let Some(bits) = matrix.missing_slice() {
        if parallel {
            sum_pi = (0..matrix.variant_count())
                .into_par_iter()
                .map(|variant_idx| {
                    let base = variant_idx * stride;
                    let (total_called, alt_count) =
                        dense_sum_alt_with_missing(data, base, offsets, bits);
                    dense_pi_from_counts(total_called, alt_count).unwrap_or(0.0)
                })
                .sum();
        } else {
            for variant_idx in 0..matrix.variant_count() {
                let base = variant_idx * stride;
                let (total_called, alt_count) =
                    dense_sum_alt_with_missing(data, base, offsets, bits);
                if let Some(pi) = dense_pi_from_counts(total_called, alt_count) {
                    sum_pi += pi;
                }
            }
        }
    } else {
        let total = offsets.len();
        if total < 2 {
            return 0.0;
        }
        let n = total as f64;
        let scale = n / (n - 1.0);
        let inv_n_sq = 1.0 / (n * n);
        if parallel {
            sum_pi = (0..matrix.variant_count())
                .into_par_iter()
                .map(|variant_idx| {
                    let base = variant_idx * stride;
                    let alt = dense_sum_alt_no_missing(data, base, offsets);
                    if alt == 0 || alt == total {
                        0.0
                    } else {
                        let alt_f = alt as f64;
                        let ref_f = (total - alt) as f64;
                        let sum_sq = ref_f * ref_f + alt_f * alt_f;
                        scale * (1.0 - sum_sq * inv_n_sq)
                    }
                })
                .sum();
        } else {
            for variant_idx in 0..matrix.variant_count() {
                let base = variant_idx * stride;
                let alt = dense_sum_alt_no_missing(data, base, offsets);
                if alt == 0 || alt == total {
                    continue;
                }
                let alt_f = alt as f64;
                let ref_f = (total - alt) as f64;
                let sum_sq = ref_f * ref_f + alt_f * alt_f;
                sum_pi += scale * (1.0 - sum_sq * inv_n_sq);
            }
        }
    }

    sum_pi / seq_length as f64
}

fn calculate_pi_dense(
    matrix: &DenseGenotypeMatrix,
    membership: &DenseMembership,
    seq_length: i64,
) -> f64 {
    if membership.len() <= 1 {
        log(
            LogLevel::Warning,
            &format!(
                "Cannot calculate pi: insufficient haplotypes ({})",
                membership.len()
            ),
        );
        return 0.0;
    }

    if seq_length < 0 {
        log(
            LogLevel::Warning,
            &format!(
                "Cannot calculate pi: invalid sequence length ({})",
                seq_length
            ),
        );
        return 0.0;
    }

    if seq_length == 0 {
        log(
            LogLevel::Warning,
            "Cannot calculate pi: zero sequence length causes division by zero",
        );
        return f64::INFINITY;
    }

    if matrix.max_allele() <= 1 {
        return calculate_pi_dense_biallelic(matrix, membership, seq_length);
    }

    let mut counts = vec![0u32; 256];
    let mut used = Vec::with_capacity(8);
    let mut sum_pi = 0.0_f64;

    for variant_idx in 0..matrix.variant_count() {
        let (total_called, sum_counts_sq) =
            dense_collect_counts(matrix, membership, variant_idx, &mut counts, &mut used);
        if total_called >= 2 {
            let n = total_called as f64;
            let sum_p2 = sum_counts_sq / (n * n);
            sum_pi += n / (n - 1.0) * (1.0 - sum_p2);
        }
        dense_reset_counts(&mut counts, &mut used);
    }

    sum_pi / seq_length as f64
}

pub fn calculate_pi_for_population(context: &PopulationContext<'_>) -> f64 {
    if let Some(summary) = context.dense_summary.as_deref() {
        return calculate_pi_from_summary(summary, context.sequence_length);
    }
    if let Some(matrix) = context.dense_genotypes {
        if matrix.ploidy() == 2 {
            let membership = DenseMembership::build(matrix, &context.haplotypes);
            return calculate_pi_dense(matrix, &membership, context.sequence_length);
        }
    }
    calculate_pi(
        context.variants,
        &context.haplotypes,
        context.sequence_length,
    )
}

/// Calculate per-site diversity metrics (π and Watterson's θ) across a genomic region.
///
/// The iterator over `variants` is filtered to those whose positions fall within `region` and
/// the returned vector contains one `SiteDiversity` entry per such variant. Monomorphic bases
/// between variant positions are skipped—consistent with the regional aggregations where these
/// sites contribute zeros—so callers should no longer expect `site_diversities.len()` to equal
/// `region.len()`. The sparsity keeps runtime and memory proportional to the number of
/// informative loci while preserving aggregate results.
pub fn calculate_per_site_diversity(
    variants: &[Variant],
    haplotypes_in_group: &[(usize, HaplotypeSide)],
    region: QueryRegion, // Inclusive range [start..end] in 0-based coordinates
) -> Vec<SiteDiversity> {
    set_stage(ProcessingStage::StatsCalculation);

    let start_time = std::time::Instant::now();
    let mut pi_state = PiComputationState::default();
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

    let sample_count = variants
        .first()
        .map(|variant| variant.genotypes.len())
        .unwrap_or(0);
    let membership = HapMembership::build(sample_count, haplotypes_in_group);

    let total_variants = variants
        .iter()
        .filter(|variant| region.contains(variant.position))
        .count();

    let mut site_diversities = Vec::with_capacity(total_variants);

    if haplotypes_in_group.len() < 2 {
        log(
            LogLevel::Warning,
            "Insufficient haplotypes (<2) for diversity calculation",
        );
        return site_diversities;
    }

    init_step_progress(
        &format!(
            "Analyzing {} variant positions for diversity",
            total_variants
        ),
        total_variants as u64,
    );

    let mut polymorphic_sites = 0usize;

    for (idx, variant) in variants
        .iter()
        .filter(|variant| region.contains(variant.position))
        .enumerate()
    {
        if idx % 1000 == 0 || idx + 1 == total_variants {
            update_step_progress(
                idx as u64,
                &format!(
                    "Variant {}/{} ({:.1}%)",
                    idx + 1,
                    total_variants,
                    ((idx + 1) as f64 / total_variants.max(1) as f64) * 100.0
                ),
            );
        }

        let metrics = compute_pi_metrics_fast(variant, &membership, &mut pi_state);
        let total_called = metrics.total_called;

        let (pi_value, watterson_value) = if total_called < 2 {
            (0.0, 0.0)
        } else {
            let distinct_alleles = metrics.distinct_alleles;
            let watterson_value = if distinct_alleles > 1 {
                let denom = harmonic_cached(total_called - 1);
                if denom > 0.0 {
                    1.0 / denom
                } else {
                    0.0
                }
            } else {
                0.0
            };
            let pi_value = metrics.pi().unwrap_or(0.0);
            (pi_value, watterson_value)
        };

        if pi_value > 0.0 || watterson_value > 0.0 {
            polymorphic_sites += 1;
        }

        site_diversities.push(SiteDiversity {
            position: ZeroBasedPosition(variant.position).to_one_based(),
            pi: pi_value,
            watterson_theta: watterson_value,
        });
    }

    finish_step_progress(&format!(
        "Completed: {} variants processed with {} polymorphic sites",
        total_variants, polymorphic_sites
    ));

    log(
        LogLevel::Info,
        &format!(
            "Per-site diversity calculation complete: {} variants analyzed, {} polymorphic sites",
            total_variants, polymorphic_sites
        ),
    );

    let total_time = start_time.elapsed();
    display_status_box(StatusBox {
        title: "Per-Site Diversity Summary".to_string(),
        stats: vec![
            (
                String::from("Region"),
                format!(
                    "{}:{}-{}",
                    ZeroBasedPosition(region.start).to_one_based(),
                    ZeroBasedPosition(region.end).to_one_based(),
                    region.len()
                ),
            ),
            (
                String::from("Haplotypes"),
                haplotypes_in_group.len().to_string(),
            ),
            (
                String::from("Variants processed"),
                total_variants.to_string(),
            ),
            (
                String::from("Polymorphic sites"),
                polymorphic_sites.to_string(),
            ),
            (
                String::from("Elapsed time"),
                format!("{:.2}s", total_time.as_secs_f64()),
            ),
            (
                String::from("Vector capacity"),
                format!(
                    "~{:.1} MB",
                    (site_diversities.capacity() * std::mem::size_of::<SiteDiversity>()) as f64
                        / 1_048_576.0
                ),
            ),
        ],
    });

    site_diversities
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
// - `AlleleCountSummary` accumulates Σ count_i² during tallying so no extra pass is needed.
// - `pi_from_summary` converts those tallies into π using the unbiased estimator.
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
    include!("tests/stats_tests.rs");
    include!("tests/filter_tests.rs");
}
