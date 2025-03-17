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

// Define a struct to hold diversity metrics for each genomic site
#[derive(Debug)]
pub struct SiteDiversity {
    pub position: i64, // 1-based position of the site in the genome
    pub pi: f64, // Nucleotide diversity (π) at this site
    pub watterson_theta: f64, // Watterson's theta (θ_w) at this site
}

/// FST results for a single site
#[derive(Debug, Clone)]
pub struct SiteFST {
    /// Position (1-based coordinate)
    pub position: i64,

    /// Overall FST value across all populations
    pub overall_fst: f64,

    /// Pairwise FST values between populations
    /// Keys are (pop_id1, pop_id2) where pop_id1 < pop_id2
    pub pairwise_fst: HashMap<String, f64>,

    /// Variance components (a, b) from Weir & Cockerham
    pub variance_components: (f64, f64),

    /// Number of samples in each population group
    pub population_sizes: HashMap<String, usize>,

    /// Pairwise variance components for each subpopulation pair (a_xy, b_xy)
    pub pairwise_variance_components: HashMap<String, (f64, f64)>,
}


/// FST results for a genomic region
#[derive(Debug, Clone)]
pub struct FSTResults {
    /// Overall FST value for the region
    pub overall_fst: f64,
    
    /// Pairwise FST values
    pub pairwise_fst: HashMap<String, f64>,

    /// Pairwise variance components for each subpopulation pair
    /// Keys match those in `pairwise_fst`, storing (a_xy, b_xy) for that pair at this site
    pub pairwise_variance_components: HashMap<String, (f64, f64)>,
    
    /// Per-site FST values
    pub site_fst: Vec<SiteFST>,
    
    /// Type of FST calculation (e.g., "haplotype_groups" or "population_groups")
    pub fst_type: String,
}

/*
    Weir & Cockerham (1984) define F-statistics (F, Θ, f) as correlations
    of alleles at different levels: within individuals, among individuals
    within subpopulations, and among subpopulations. The key parameters
    can be estimated by partitioning the total allelic variance into
    hierarchical components usually denoted a, b, c.

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

    The W&C 1984 "haploid ignoring c term" formulas for a, b (equations 5–7 in the text,
    specialized to random union of gametes) look like:

       a = [n̄ / (n̄ - 1)] * [ S² - { p̄(1-p̄) - ((r-1)/r) S² } / n̄ ],
       b = [n̄ / (n̄ - 1)] * [ p̄(1 - p̄) + (some c² adjustments) - ((r-1)/r)*S² - ...].

    Conceptually, "a" is the among-subpopulations variance component, and "b"
    is the residual within-subpop variance. The Fst at that site is then

       Fst(site) = a / (a + b).

    We repeat for each site and sum the a_i and b_i across sites i to obtain an overall Fst:

       Fst(overall) = ( Σ_i a_i ) / ( Σ_i (a_i + b_i) ).

    Pairwise subpopulation Fst can be done by restricting the above to only the
    two subpops of interest (i.e., r=2).

    In our “haplotype-based” version, each diploid sample contributes two
    haplotypes (assuming no inbreeding parameter), so we treat them as
    independent. We omit W&C’s “c” term for within-individual correlation.
*/


/// Calculate FST between haplotype groups (0 vs 1) for a region
///
/// This implements the Weir & Cockerham (1984) estimator for FST calculation.
/// It uses variance components to estimate θ (theta), which is equivalent to FST.
///
/// # Arguments
/// * `variants` - The variant data for all samples in the region
/// * `sample_names` - Names of all samples
/// * `sample_to_group_map` - Maps sample names to their (left, right) haplotype group assignments
/// * `region` - Genomic region to analyze
///
/// # Returns
/// FST results including per-site values and overall estimates
pub fn calculate_fst_between_groups(
    variants: &[Variant],
    sample_names: &[String],
    sample_to_group_map: &HashMap<String, (u8, u8)>,
    region: QueryRegion,
) -> FSTResults {
    // Create a more descriptive progress spinner
    let spinner = create_spinner(&format!(
        "Calculating FST between haplotype groups for region {}-{} of length {}",
        region.start, region.end, region.len()
    ));
    
    log(LogLevel::Info, &format!(
        "Beginning FST calculation between haplotype groups (0 vs 1) for region {}-{}",
        region.start, region.end
    ));
    
    // 1. Map samples to their respective haplotype groups
    let haplotype_to_group = map_samples_to_haplotype_groups(sample_names, sample_to_group_map);
    
    // 2. Build variant lookup map for quick position-based access
    let variant_map: HashMap<i64, &Variant> = variants.iter()
        .map(|v| (v.position, v))
        .collect();
    
    // 3. Calculate FST at each position in region
    let mut site_fst_values = Vec::with_capacity(region.len() as usize);
    let position_count = region.len() as usize;
    
    // Initialize detailed progress tracking
    init_step_progress(&format!(
        "Calculating FST at {} positions", position_count
    ), position_count as u64);
    
    for (idx, pos) in (region.start..=region.end).enumerate() {
        // Update progress periodically
        if idx % 1000 == 0 || idx == 0 || idx == position_count - 1 {
            update_step_progress(idx as u64, &format!(
                "Position {}/{} ({:.1}%)",
                idx, position_count, (idx as f64 / position_count as f64) * 100.0
            ));
        }
        
        if let Some(variant) = variant_map.get(&pos) {
            // Calculate FST at this site using haplotype groups
            let site_result = calculate_fst_at_site_by_group(variant, &haplotype_to_group);
            
            site_fst_values.push(SiteFST {
                position: ZeroBasedPosition(pos).to_one_based(),
                overall_fst: site_result.0,
                pairwise_fst: site_result.1,
                variance_components: site_result.2,
                population_sizes: site_result.3,
                pairwise_variance_components: site_result.4,
            });
        } else {
            // No variant at this position (monomorphic site): we store FST as NaN
            // "Notice that if the same allele is fixed in all samples, then (1) provides θ = a⁄a,
            // so that the estimate is undefined. We feel that this is an appropriate outcome since
            // there is no way of knowing from presently observed homozygosity whether the different
            // populations have just become homozygous or have been homozygous for some time that may
            // have extended back to the founding population..."

            let mut empty_pairwise = HashMap::new();
            empty_pairwise.insert("0_vs_1".to_string(), f64::NAN);
            
            let mut empty_sizes = HashMap::new();
            empty_sizes.insert("0".to_string(), 0);
            empty_sizes.insert("1".to_string(), 0);
            
            site_fst_values.push(SiteFST {
                position: ZeroBasedPosition(pos).to_one_based(),
                overall_fst: f64::NAN,
                pairwise_fst: empty_pairwise,
                variance_components: (0.0, 0.0),
                population_sizes: empty_sizes,
                pairwise_variance_components: HashMap::new(),
            });
        }
    }
    
    finish_step_progress("Completed per-site FST calculations");
    
    // 4. Calculate overall FST for the region
    let (overall_fst, pairwise_fst) = calculate_overall_fst(&site_fst_values);
    
    // Log summary statistics
    let polymorphic_sites = site_fst_values.iter()
        .filter(|site| site.overall_fst > 0.0)
        .count();
    
    log(LogLevel::Info, &format!(
        "FST calculation complete: {} polymorphic sites out of {} total positions",
        polymorphic_sites, site_fst_values.len()
    ));
    
    log(LogLevel::Info, &format!(
        "Overall FST between haplotype groups: {:.6}",
        overall_fst
    ));
    
    for (pair, fst) in &pairwise_fst {
        log(LogLevel::Info, &format!(
            "Pairwise FST for {}: {:.6}",
            pair, fst
        ));
    }
    
    spinner.finish_and_clear();
    
    // Return complete results
    FSTResults {
        overall_fst,
        pairwise_fst,
        pairwise_variance_components: HashMap::new(),
        site_fst: site_fst_values,
        fst_type: "haplotype_groups".to_string(),
    }
}

/// Calculate FST from a CSV file defining population groups
///
/// # Arguments
/// * `variants` - The variant data for all samples in the region
/// * `sample_names` - Names of all samples
/// * `csv_path` - Path to CSV file defining population assignments
/// * `region` - Genomic region to analyze
///
/// # Returns
/// FST results or an error if the CSV cannot be parsed
pub fn calculate_fst_from_csv(
    variants: &[Variant],
    sample_names: &[String],
    csv_path: &Path,
    region: QueryRegion,
) -> Result<FSTResults, VcfError> {
    // The code that calls calculate_fst_at_site_by_population now expects
    // a 5-tuple from that function. We handle the new pairwise (a,b) map when building SiteFST.

    let spinner = create_spinner(&format!(
        "Calculating FST between population groups for region {}:{}-{}",
        region.start, region.end, region.len()
    ));
    
    log(LogLevel::Info, &format!(
        "Beginning FST calculation between population groups defined in {} for region {}-{}",
        csv_path.display(), region.start, region.end
    ));
    
    // 1. Parse CSV file to get population assignments
    let population_assignments = parse_population_csv(csv_path)?;
    
    // 2. Map samples to their respective populations
    let sample_to_pop = map_samples_to_populations(sample_names, &population_assignments);
    
    // 3. Build variant lookup map for quick position-based access
    let variant_map: HashMap<i64, &Variant> = variants.iter()
        .map(|v| (v.position, v))
        .collect();
    
    // 4. Calculate FST at each position in region
    let mut site_fst_values = Vec::with_capacity(region.len() as usize);
    let position_count = region.len() as usize;
    
    // Initialize detailed progress tracking
    init_step_progress(&format!(
        "Calculating FST at {} positions", position_count
    ), position_count as u64);
    
    for (idx, pos) in (region.start..=region.end).enumerate() {
        // Update progress periodically
        if idx % 1000 == 0 || idx == 0 || idx == position_count - 1 {
            update_step_progress(idx as u64, &format!(
                "Position {}/{} ({:.1}%)",
                idx, position_count, (idx as f64 / position_count as f64) * 100.0
            ));
        }
        
        if let Some(variant) = variant_map.get(&pos) {
            // Calculate FST at this site using population groups
            let site_result = calculate_fst_at_site_by_population(variant, &sample_to_pop);
            
            site_fst_values.push(SiteFST {
                position: ZeroBasedPosition(pos).to_one_based(),
                overall_fst: site_result.0,
                pairwise_fst: site_result.1,
                variance_components: site_result.2,
                population_sizes: site_result.3,
                pairwise_variance_components: site_result.4,
            });
        } else {
            // No variant at this position (monomorphic site): we store FST as NaN
            site_fst_values.push(SiteFST {
                position: ZeroBasedPosition(pos).to_one_based(),
                overall_fst: f64::NAN,
                pairwise_fst: HashMap::new(),
                variance_components: (0.0, 0.0),
                population_sizes: HashMap::new(),
                pairwise_variance_components: HashMap::new(),
            });
        }
    }
    
    finish_step_progress("Completed per-site FST calculations");
    
    // 5. Calculate overall FST for the region
    let (overall_fst, pairwise_fst) = calculate_overall_fst(&site_fst_values);
    
    // Log summary statistics
    let polymorphic_sites = site_fst_values.iter()
        .filter(|site| site.overall_fst > 0.0)
        .count();
    
    let population_count = population_assignments.keys().count();
    
    log(LogLevel::Info, &format!(
        "FST calculation complete: {} polymorphic sites out of {} total positions across {} populations",
        polymorphic_sites, site_fst_values.len(), population_count
    ));
    
    log(LogLevel::Info, &format!(
        "Overall FST across all populations: {:.6}",
        overall_fst
    ));
    
    for (pair, fst) in &pairwise_fst {
        log(LogLevel::Info, &format!(
            "Pairwise FST for {}: {:.6}",
            pair, fst
        ));
    }
    
    spinner.finish_and_clear();
    
    // Return complete results
    Ok(FSTResults {
        overall_fst,
        pairwise_fst,
        pairwise_variance_components: HashMap::new(),
        site_fst: site_fst_values,
        fst_type: "population_groups".to_string(),
    })
}

/// Parse a CSV file containing population assignments
///
/// # Format
/// First column: Population labels
/// Remaining columns: Sample IDs belonging to that population
///
/// # Returns
/// HashMap mapping sample IDs to their population labels
fn parse_population_csv(csv_path: &Path) -> Result<HashMap<String, Vec<String>>, VcfError> {
    let file = File::open(csv_path).map_err(|e| 
        VcfError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Failed to open population CSV file: {}", e)
        ))
    )?;
    
    let reader = BufReader::new(file);
    let mut population_map = HashMap::new();
    
    for line in reader.lines() {
        let line = line.map_err(|e| VcfError::Parse(format!("Error reading CSV line: {}", e)))?;
        if line.trim().is_empty() || line.starts_with('#') {
            continue; // Skip empty lines and comments
        }
        
        let parts: Vec<&str> = line.split(',').collect();
        if parts.is_empty() {
            continue;
        }
        
        let population = parts[0].trim().to_string();
        let samples = parts[1..].iter().map(|s| s.trim().to_string()).collect();
        
        population_map.insert(population, samples);
    }
    
    if population_map.is_empty() {
        return Err(VcfError::Parse("Population CSV file contains no valid data".to_string()));
    }
    
    Ok(population_map)
}

/// Map samples to their haplotype groups (0 or 1)
///
/// # Returns
/// HashMap mapping (sample_index, haplotype_side) tuples to group identifiers ("0" or "1")
fn map_samples_to_haplotype_groups(
    sample_names: &[String],
    sample_to_group_map: &HashMap<String, (u8, u8)>
) -> HashMap<(usize, HaplotypeSide), String> {
    let mut haplotype_to_group = HashMap::new();
    
    // Build sample name to index mapping
    let mut sample_id_to_index = HashMap::new();
    for (idx, name) in sample_names.iter().enumerate() {
        let sample_id = name.rsplit('_').next().unwrap_or(name);
        sample_id_to_index.insert(sample_id, idx);
    }
    
    // Map each sample's haplotypes to their assigned groups
    for (sample_name, &(left_group, right_group)) in sample_to_group_map {
        if let Some(&idx) = sample_id_to_index.get(sample_name.as_str()) {
            // Convert group numbers to strings for consistent handling with population groups
            haplotype_to_group.insert(
                (idx, HaplotypeSide::Left),
                left_group.to_string()
            );
            
            haplotype_to_group.insert(
                (idx, HaplotypeSide::Right),
                right_group.to_string()
            );
        }
    }
    
    haplotype_to_group
}

/// Map samples to their population groups from CSV
///
/// # Returns
/// HashMap mapping (sample_index, haplotype_side) tuples to population identifiers
fn map_samples_to_populations(
    sample_names: &[String],
    population_assignments: &HashMap<String, Vec<String>>
) -> HashMap<(usize, HaplotypeSide), String> {
    let mut sample_to_pop = HashMap::new();
    
    // Create flat map of sample ID to population
    let mut sample_id_to_pop = HashMap::new();
    for (pop, samples) in population_assignments {
        for sample in samples {
            sample_id_to_pop.insert(sample.clone(), pop.clone());
        }
    }
    
    // Map each sample to its population
    for (idx, name) in sample_names.iter().enumerate() {
        // Try different formats for matching
        // Some VCFs have sample names like "NA12878", others like "EUR_CEU_NA12878"
        if let Some(pop) = sample_id_to_pop.get(name) {
            // Both haplotypes of this sample belong to the same population
            sample_to_pop.insert((idx, HaplotypeSide::Left), pop.clone());
            sample_to_pop.insert((idx, HaplotypeSide::Right), pop.clone());
            continue;
        }
        
        // Try to match by the sample ID suffix
        let sample_id = name.rsplit('_').next().unwrap_or(name);
        if let Some(pop) = sample_id_to_pop.get(sample_id) {
            sample_to_pop.insert((idx, HaplotypeSide::Left), pop.clone());
            sample_to_pop.insert((idx, HaplotypeSide::Right), pop.clone());
            continue;
        }
        
        // If that fails, try to match by prefix (e.g., "EUR" in "EUR_CEU_NA12878")
        // Potentially remove this...
        let prefix = name.split('_').next().unwrap_or(name);
        for (pop, _) in population_assignments {
            if name.starts_with(pop) || prefix == pop {
                sample_to_pop.insert((idx, HaplotypeSide::Left), pop.clone());
                sample_to_pop.insert((idx, HaplotypeSide::Right), pop.clone());
                break;
            }
        }
    }
    
    sample_to_pop
}

/// Calculate FST at a single site using haplotype groups (0 vs 1)
///
/// Implements the Weir & Cockerham (1984) estimator.
///
/// # Returns
/// (overall_fst, pairwise_fst, variance_components, population_sizes)
fn calculate_fst_at_site_by_group(
    variant: &Variant,
    haplotype_to_group: &HashMap<(usize, HaplotypeSide), String>
) -> (f64, HashMap<String, f64>, (f64, f64), HashMap<String, usize>, HashMap<String, (f64, f64)>) {
    // We delegate to the new general function with an is_population flag = false
    // Then we parse out the returned data
    calculate_fst_at_site_general(variant, haplotype_to_group)
}

/// Calculate FST at a single site using population assignments
///
/// Similar to calculate_fst_at_site_by_group but handles multiple population groups
fn calculate_fst_at_site_by_population(
    variant: &Variant,
    sample_to_pop: &HashMap<(usize, HaplotypeSide), String>
) -> (f64, HashMap<String, f64>, (f64, f64), HashMap<String, usize>, HashMap<String, (f64, f64)>) {
    // We delegate to the new general function with an is_population flag = true
    // Then we parse out the returned data
    calculate_fst_at_site_general(variant, sample_to_pop)
}

fn calculate_fst_at_site_general(
    variant: &Variant,
    map_subpop: &HashMap<(usize, HaplotypeSide), String>
) -> (f64, HashMap<String, f64>, (f64, f64), HashMap<String, usize>, HashMap<String, (f64, f64)>) {
    // This single function computes the at-site FST for an arbitrary subpop mapping
    // Then returns overall_fst, pairwise_fst, (a,b), population_sizes, plus the pairwise (a_xy,b_xy).

    // 1. Count alt vs ref per subpopulation
    let mut allele_counts = HashMap::new();
    for (&(sample_idx, side), pop_id) in map_subpop {
        if let Some(genotypes) = variant.genotypes.get(sample_idx) {
            if let Some(genotypes_vec) = genotypes {
                if let Some(&allele_code) = genotypes_vec.get(side as usize) {
                    let entry = allele_counts.entry(pop_id.clone()).or_insert((0_usize, 0_usize));
                    entry.0 += 1;
                    if allele_code != 0 {
                        entry.1 += 1;
                    }
                }
            }
        }
    }

    // 2. Build pop_stats (sample_size, freq)
    let mut pop_stats = HashMap::new();
    let mut pop_sizes = HashMap::new();
    for (pop_id, (sz, alt_ct)) in allele_counts {
        if sz > 0 {
            let freq = alt_ct as f64 / sz as f64;
            pop_stats.insert(pop_id.clone(), (sz, freq));
            pop_sizes.insert(pop_id, sz);
        }
    }
    if pop_stats.len() < 2 {
        return (0.0, HashMap::new(), (0.0, 0.0), pop_sizes, HashMap::new());
    }

    // 3. Check if all subpop frequencies are effectively identical.
    // We define a small threshold for difference, and if the maximum difference
    // among any pair of frequencies is below that threshold, we consider
    // the site monomorphic in all subpops and return Fst=0.0. However, this is wrong.
    // Fix later.
    let freq_values: Vec<f64> = pop_stats.values().map(|(_, f)| *f).collect();
    let mut max_diff = 0.0;
    for i in 0..freq_values.len() {
        for j in (i+1)..freq_values.len() {
            let diff = (freq_values[i] - freq_values[j]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }
    let threshold = 1e-12;
    if max_diff < threshold {
        // If subpop frequencies are (practically) identical,
        // this site does not provide a defined Fst. Mark as NaN:
        let empty_map = HashMap::new();
        return (f64::NAN, empty_map, (0.0, 0.0), pop_sizes, HashMap::new());
    }

    // 4. Compute overall (a,b), overall_fst
    let total_samples: usize = pop_stats.values().map(|(sz, _)| *sz).sum();
    let mut freq_sum = 0.0;
    for (_, (sz, fv)) in &pop_stats {
        freq_sum += (*sz as f64) * fv;
    }
    let global_freq = freq_sum / (total_samples as f64);
    let (a, b) = calculate_variance_components(&pop_stats, global_freq);
    let overall_fst = if (a + b) > 0.0 {
        a / (a + b)
    } else {
        f64::NAN  // Undefined as per Weir & Cockerham
    };

    // 5. Compute pairwise: sum up a_xy,b_xy for each pair
    let mut pairwise_fst_map = HashMap::new();
    let mut pairwise_ab_map = HashMap::new();
    let pop_list: Vec<_> = pop_stats.keys().cloned().collect();
    for i in 0..pop_list.len() {
        for j in (i+1)..pop_list.len() {
            let p1 = &pop_list[i];
            let p2 = &pop_list[j];
            let freq1 = pop_stats[p1].1;
            let freq2 = pop_stats[p2].1;
            if (freq1 - freq2).abs() < 1e-12 {
                pairwise_fst_map.insert(format!("{}_vs_{}", p1, p2), f64::NAN);
                pairwise_ab_map.insert(format!("{}_vs_{}", p1, p2), (0.0, 0.0)); // Double-check
                continue;
            }
            let mut pair_stats = HashMap::new();
            pair_stats.insert(p1.clone(), pop_stats[p1]);
            pair_stats.insert(p2.clone(), pop_stats[p2]);
            let pair_sum = pop_stats[p1].0 + pop_stats[p2].0;
            let mut freq_pair_sum = 0.0;
            for (_, (sz2, fv2)) in &pair_stats {
                freq_pair_sum += (*sz2 as f64) * fv2;
            }
            let pair_freq = freq_pair_sum / (pair_sum as f64);
            let (a_xy, b_xy) = calculate_variance_components(&pair_stats, pair_freq);
            let f_xy = if (a_xy + b_xy) > 0.0 {
                a_xy / (a_xy + b_xy)
            } else {
                0.0
            };
            pairwise_fst_map.insert(format!("{}_vs_{}", p1, p2), f_xy);
            pairwise_ab_map.insert(format!("{}_vs_{}", p1, p2), (a_xy, b_xy));
        }
    }

    (overall_fst, pairwise_fst_map, (a, b), pop_sizes, pairwise_ab_map)
}

/// Calculate variance components for the Weir & Cockerham (1984) FST estimator
///
/// # Arguments
/// * `pop_stats` - HashMap mapping population IDs to (sample_size, allele_frequency) tuples
/// * `global_freq` - Global weighted allele frequency
///
/// # Returns
/// (a, b) variance components, where:
/// * a = variance between populations
/// * b = variance within populations
fn calculate_variance_components(
    pop_stats: &HashMap<String, (usize, f64)>,
    global_freq: f64
) -> (f64, f64) {
    /*
    This function implements the Weir & Cockerham (1984) haploid-based variance-component
    calculations for random union of gametes. It treats each haplotype as an independent sample
    and omits the within-individual component c.

    1) n_bar = mean subpopulation size, counting haplotypes.
    2) global_p = global_freq is the overall allele frequency in all subpops combined.
    3) We compute c2 as the squared coefficient of variation of subpopulation sizes.
    4) S^2 is the weighted among-subpop variance in allele frequency.
    5) a and b are computed so that Fst = a / (a + b).
    6) If sample sizes are identical, c2 = 0 and the formula reduces properly. If sample sizes
       differ greatly, c2 adjusts a and b as per the W&C derivation. This matches eqns. (5)–(7)
       under a haploid model with random union of gametes.
    */

    let r = pop_stats.len() as f64;
    if r < 2.0 {
        return (0.0, 0.0);
    }

    let mut n_values = Vec::with_capacity(pop_stats.len());
    let mut total_samples = 0_usize;
    for (_pop_id, (size, _freq)) in pop_stats.iter() {
        n_values.push(*size as f64);
        total_samples += *size;
    }

    let n_bar = (total_samples as f64) / r;
    if n_bar <= 1.0 {
        return (0.0, 0.0);
    }

    let global_p = global_freq;

    let mut sum_sq_diff = 0.0;
    for n_i in &n_values {
        let diff = *n_i - n_bar;
        sum_sq_diff += diff * diff;
    }
    let c2 = sum_sq_diff / (r * n_bar * n_bar);

    let mut numerator_s2 = 0.0;
    for (_pop_id, (size, freq)) in pop_stats.iter() {
        let diff = *freq - global_p;
        numerator_s2 += (*size as f64) * diff * diff;
    }
    let s_squared = if (r - 1.0) > 0.0 && n_bar > 0.0 {
        numerator_s2 / ((r - 1.0) * n_bar)
    } else {
        0.0
    };

    let a_num = s_squared
        - (
            global_p * (1.0 - global_p)
            - ((r - 1.0) / r) * s_squared
          )
          / (n_bar - 1.0);
    
    let denom_factor = 1.0 - (c2 / r);
    
    // If either (n_bar - 1.0) <= 0 or denom_factor <= 0, set a = 0.0 as a fallback
    // (this avoids negative or undefined denominators)
    let a = if (n_bar - 1.0) > 0.0 && denom_factor > 0.0 {
        a_num / denom_factor
    } else {
        0.0
    };

    let b_num = (global_p * (1.0 - global_p) + (c2 * global_p * (1.0 - global_p)))
        - ((r - 1.0) / r) * s_squared
        - (c2 * (r - 1.0) * s_squared / r);
    let b = if (n_bar - 1.0) > 0.0 {
        (n_bar / (n_bar - 1.0)) * b_num
    } else {
        0.0
    };

    (a, b) // do not clamp negative estimates for downstream estimate
}

/// Calculate overall FST for a region from per-site values
///
/// This follows W&C's approach of summing variance components across sites.
///
/// # Returns
/// (overall_fst, pairwise_fst)
fn calculate_overall_fst(site_fst_values: &[SiteFST]) -> (f64, HashMap<String, f64>) {
    /*
    This function sums the per-site variance components (a, b) across all sites that have (a + b) > 0.
    That means we skip monomorphic or uninformative sites where (a + b) == 0. Skipping avoids an
    undefined ratio of 0/0. Under the Weir & Cockerham (1984) formula, summing a and b over
    just the informative sites is the standard approach. We then compute the global Fst as
    sum(a) / sum(a + b).

    For pairwise Fst, we gather each site's pairwise (a_xy, b_xy), sum them, and compute the ratio.
    */
    let mut informative_sites = Vec::new();
    for site in site_fst_values.iter() {
        let (a, b) = site.variance_components;
        if (a + b) > 0.0 {
            informative_sites.push(site);
        }
    }

    if informative_sites.is_empty() {
        return (0.0, HashMap::new());
    }

    let mut sum_a_total = 0.0;
    let mut sum_b_total = 0.0;
    for site in &informative_sites {
        sum_a_total += site.variance_components.0;
        sum_b_total += site.variance_components.1;
    }
    let raw_overall = if (sum_a_total + sum_b_total) > 0.0 {
        sum_a_total / (sum_a_total + sum_b_total)
    } else {
        f64::NAN  // Undefined as per Weir & Cockerham
    };
    let overall_fst = if raw_overall < 0.0 {
        0.0
    } else if raw_overall > 1.0 {
        1.0
    } else {
        raw_overall
    };

    let mut pairwise_ab_sums: HashMap<String, (f64, f64)> = HashMap::new();
    for site in &informative_sites {
        for (pair_key, &(a_xy, b_xy)) in site.pairwise_variance_components.iter() {
            let entry = pairwise_ab_sums.entry(pair_key.clone()).or_insert((0.0, 0.0));
            entry.0 += a_xy;
            entry.1 += b_xy;
        }
    }

    let mut pairwise_fst = HashMap::new();
    for (pair_key, (a_sum, b_sum)) in pairwise_ab_sums {
        let ratio = if (a_sum + b_sum) > 0.0 {
            a_sum / (a_sum + b_sum)
        } else {
            f64::NAN
        };
        let clamp_ratio = if ratio.is_nan() {
            f64::NAN
        } else if ratio < 0.0 {
            0.0
        } else if ratio > 1.0 {
            1.0
        } else {
            ratio
        };
        pairwise_fst.insert(pair_key, clamp_ratio);
    }

    (overall_fst, pairwise_fst)
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
