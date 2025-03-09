use crate::process::{Variant, ZeroBasedPosition, HaplotypeSide};
use crate::progress::{
    log, LogLevel, init_step_progress, update_step_progress, 
    finish_step_progress, create_spinner, display_status_box, StatusBox, set_stage, ProcessingStage
};
use rayon::iter::IntoParallelIterator;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

// Define a struct to hold diversity metrics for each genomic site
#[derive(Debug)]
pub struct SiteDiversity {
    pub position: i64, // 1-based position of the site in the genome
    pub pi: f64, // Nucleotide diversity (π) at this site
    pub watterson_theta: f64, // Watterson's theta (θ_w) at this site
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
    let region = crate::process::ZeroBasedHalfOpen::from_1based_inclusive(region_start, region_end);
    
    // Initialize a vector to store intervals that are allowed after intersecting with allow_regions_chr
    let mut allowed_intervals = Vec::new();
    if let Some(allow_regions) = allow_regions_chr {
        // If allowed regions are provided, intersect the input region with each allowed region
        for &(start, end) in allow_regions {
            // Convert each allowed region to ZeroBasedHalfOpen for consistent interval operations
            let allow_region = crate::process::ZeroBasedHalfOpen::from_1based_inclusive(start, end);
            
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
            let interval = crate::process::ZeroBasedHalfOpen::from_1based_inclusive(start, end);
            interval.len() as i64 // Get the length and cast to i64
        })
        .sum(); // Sum all lengths to get the total adjusted length
    
    // Display results and finish spinner
    spinner.finish_with_message(format!(
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

    // Parallel iteration over all first sample indices (0 to n-1)
    (0..number_of_samples)
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

                    // Iterate over all variants to compare this pair’s genotypes
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

                    // Return the pair’s indices and their comparison metrics
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
    spinner.finish_with_message(format!(
        "Completed pairwise analysis for {} samples ({} pairs)", 
        number_of_samples, result_count
    ));
    
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
    // Handle edge cases where computation isn’t meaningful
    if n <= 1 || seq_length == 0 {
        // If 1 or fewer haplotypes or sequence length is zero, return infinity
        return f64::INFINITY;
    }

    // Calculate the harmonic number H_{n-1}, used as the denominator factor a_n
    let harmonic_value = harmonic(n - 1);
    if harmonic_value == 0.0 {
        // Prevent division by zero
        return f64::INFINITY;
    }
    // Watterson’s theta formula: θ_w = S / (a_n * L)
    // S = number of segregating sites, a_n = H_{n-1}, L = sequence length
    seg_sites as f64 / harmonic_value / seq_length as f64
}

// Calculate nucleotide diversity (π) across all sites, accounting for missing data
/// Computes π as the average pairwise difference per site across all haplotype pairs,
/// handling missing data by only considering sites where both haplotypes have alleles.
///
/// # Returns
/// * f64::INFINITY if fewer than 2 haplotypes
/// * f64::NAN if no valid pairs exist
/// * Otherwise, the average π across all sites
pub fn calculate_pi(variants: &[Variant], haplotypes_in_group: &[(usize, HaplotypeSide)]) -> f64 {
    if haplotypes_in_group.len() <= 1 {
        // Need at least 2 haplotypes to compute diversity; return infinity if not
        return f64::INFINITY;
    }

    // Calculate total possible pairs: n * (n-1) / 2
    let total_possible_pairs = haplotypes_in_group.len() * (haplotypes_in_group.len() - 1) / 2;
    if total_possible_pairs == 0 {
        // If no pairs can be formed (redundant check, but kept for robustness), return NaN
        return f64::NAN;
    }

    let mut difference_sum = 0.0; // Sum of per-pair differences normalized by comparable sites
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
                                // Get the specific allele for each haplotype’s side
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
                // Add this pair’s contribution: differences per comparable site
                difference_sum += diff_count as f64 / comparable_sites as f64;
            }
            // If no comparable sites, contribution is 0 (implicitly handled by not adding)
        }
    }

    // Compute average nucleotide diversity over all pairs
    difference_sum / total_possible_pairs as f64
}

// Calculate per-site diversity metrics (π and Watterson’s θ) across a genomic region
pub fn calculate_per_site(
    variants: &[Variant], // Slice of variants with genotype data
    haplotypes_in_group: &[(usize, HaplotypeSide)], // List of (sample index, haplotype side) tuples
    region_start: i64, // Start of the region (0-based, half-open)
    region_end: i64, // End of the region (0-based, half-open)
) -> Vec<SiteDiversity> { // Returns a vector of SiteDiversity structs
    set_stage(ProcessingStage::StatsCalculation);
    
    log(LogLevel::Info, &format!(
        "Calculating per-site diversity for region {}:{}-{} with {} haplotypes",
        region_start, region_end, region_end - region_start, haplotypes_in_group.len()
    ));

    let max_haps = haplotypes_in_group.len(); // Number of haplotypes in the group
    
    // Create a ZeroBasedHalfOpen interval for the region to determine its length
    let region = crate::process::ZeroBasedHalfOpen::from_0based_half_open(region_start, region_end);
    let mut site_diversities = Vec::with_capacity(region.len()); // Pre-allocate
    if max_haps < 2 {
        // Need at least 2 haplotypes for diversity; return empty vector if not
        log(LogLevel::Warning, "Insufficient haplotypes (<2) for diversity calculation");
        return site_diversities;
    }
    
    // Initialize progress for this long computation
    let region_length = region_end - region_start;
    init_step_progress("Calculating per-site diversity", region_length as u64);

    // Build a map of variants by position for O(1) lookup
    let variant_map: HashMap<i64, &Variant> = variants.iter().map(|v| (v.position, v)).collect();
    for pos in region_start..region_end {
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

                // Calculate Watterson’s θ for this site
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
            site_diversities.push(SiteDiversity {
                position: ZeroBasedPosition(pos).to_one_based(), // Convert to 1-based for output
                pi: pi_value,
                watterson_theta: watterson_value,
            });
        } else {
            // No variant at this position; it’s monomorphic (all same allele)
            site_diversities.push(SiteDiversity {
                position: ZeroBasedPosition(pos).to_one_based(), // Convert to 1-based
                pi: 0.0, // No diversity since no variation
                watterson_theta: 0.0, // No segregating site
            });
        }
    }

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
