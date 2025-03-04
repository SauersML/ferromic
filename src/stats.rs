use crate::process::{Variant, ZeroBasedPosition};
use rayon::iter::IntoParallelIterator;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[derive(Debug)]
pub struct SiteDiversity {
    pub position: i64,
    pub pi: f64,
    pub watterson_theta: f64,
}

pub fn calculate_adjusted_sequence_length(
    region_start: i64,
    region_end: i64,
    allow_regions_chr: Option<&Vec<(i64, i64)>>,
    mask_regions_chr: Option<&Vec<(i64, i64)>>,
) -> i64 {
    // Convert the input region to a ZeroBasedHalfOpen
    let region = crate::process::ZeroBasedHalfOpen::from_1based_inclusive(region_start, region_end);
    
    let mut allowed_intervals = Vec::new();
    if let Some(allow_regions) = allow_regions_chr {
        // Intersect the entry region with the allow regions
        for &(start, end) in allow_regions {
            // Convert the allow region to ZeroBasedHalfOpen
            let allow_region = crate::process::ZeroBasedHalfOpen::from_1based_inclusive(start, end);
            
            // Use the intersect method to find overlap
            if let Some(overlap) = region.intersect(&allow_region) {
                allowed_intervals.push((
                    overlap.start_1based_inclusive(),
                    overlap.end_1based_inclusive()
                ));
            }
        }
    } else {
        // If no allow regions, the entire entry region is allowed
        allowed_intervals.push((region_start, region_end));
    }
    
    // Subtract the masked regions from the allowed intervals
    let unmasked_intervals = subtract_regions(&allowed_intervals, mask_regions_chr);
    
    // Calculate the total length using ZeroBasedHalfOpen
    let adjusted_length: i64 = unmasked_intervals
        .iter()
        .map(|&(start, end)| {
            let interval = crate::process::ZeroBasedHalfOpen::from_1based_inclusive(start, end);
            interval.len() as i64
        })
        .sum();
        
    adjusted_length
}

fn subtract_regions(
    intervals: &Vec<(i64, i64)>,
    masks: Option<&Vec<(i64, i64)>>,
) -> Vec<(i64, i64)> {
    if masks.is_none() {
        return intervals.clone();
    }
    let masks = masks.unwrap();
    let mut result = Vec::new();
    for &(start, end) in intervals {
        // Start with the interval (start, end)
        let mut current_intervals = vec![(start, end)];
        for &(mask_start, mask_end) in masks {
            let mut new_intervals = Vec::new();
            for &(curr_start, curr_end) in &current_intervals {
                if mask_end < curr_start || mask_start > curr_end {
                    // No overlap
                    new_intervals.push((curr_start, curr_end));
                } else {
                    // There is overlap
                    if mask_start > curr_start {
                        new_intervals.push((curr_start, mask_start));
                    }
                    if mask_end < curr_end {
                        new_intervals.push((mask_end, curr_end));
                    }
                }
            }
            current_intervals = new_intervals;
            if current_intervals.is_empty() {
                break;
            }
        }
        result.extend(current_intervals);
    }
    result
}

pub fn calculate_inversion_allele_frequency(
    sample_filter: &HashMap<String, (u8, u8)>,
) -> Option<f64> {
    let mut num_ones = 0;
    let mut total_haplotypes = 0;
    for (_sample, &(left, right)) in sample_filter.iter() {
        if left == 1 {
            num_ones += 1;
            total_haplotypes += 1;
        }
        if right == 1 {
            num_ones += 1;
            total_haplotypes += 1;
        }
        if left == 0 {
            total_haplotypes += 1;
        }
        if right == 0 {
            total_haplotypes += 1;
        }
    }
    if total_haplotypes > 0 {
        Some(num_ones as f64 / total_haplotypes as f64)
    } else {
        None
    }
}

pub fn count_segregating_sites(variants: &[Variant]) -> usize {
    variants
        .par_iter()
        .filter(|v| {
            let alleles: HashSet<_> = v.genotypes.iter().flatten().flatten().collect();
            alleles.len() > 1
        })
        .count()
}

/// Calculates pairwise differences and comparable sites between all sample pairs.
///
/// This function computes, for each pair of samples (sample_idx_i, sample_idx_j), the number of differences
/// in their genotypes and the number of sites where both have data, handling partial missingness.
///
/// # Arguments
/// * `variants` - A slice of Variant structs containing genotype data for all samples.
/// * `number_of_samples` - The total number of samples to compare.
///
/// # Returns
/// A vector of tuples, each containing:
/// * `(sample_idx_i, sample_idx_j)` - Indices of the sample pair.
/// * `difference_count` - Number of sites where genotypes differ (d_ij).
/// * `comparable_site_count` - Number of sites where both samples have genotypes (l_ij).
pub fn calculate_pairwise_differences(
    variants: &[Variant],
    number_of_samples: usize,
) -> Vec<((usize, usize), usize, usize)> {
    // Wrap variants in an Arc for thread-safe sharing across parallel tasks
    let variants_shared = Arc::new(variants);

    // Parallel iteration over all first sample indices (0 to n-1)
    (0..number_of_samples)
        .into_par_iter()
        .flat_map(|sample_idx_i| {
            // Clone the Arc for each thread to access variants
            let variants_local = Arc::clone(&variants_shared);
            // Parallel iteration over second sample indices (i+1 to n-1) to avoid duplicate pairs
            (sample_idx_i + 1..number_of_samples)
                .into_par_iter()
                .map(move |sample_idx_j| {
                    let mut difference_count = 0; // Count of genotype differences (d_ij)
                    let mut comparable_site_count = 0; // Count of sites with data for both samples (l_ij)

                    // Iterate over all variants to compare genotypes between the pair
                    for variant in variants_local.iter() {
                        if let (Some(genotype_i), Some(genotype_j)) = (
                            &variant.genotypes[sample_idx_i],
                            &variant.genotypes[sample_idx_j],
                        ) {
                            // Both samples have genotype data at this variant
                            comparable_site_count += 1;
                            if genotype_i != genotype_j {
                                // Genotypes differ; assumes haploid data (Vec<u8> with one allele)
                                difference_count += 1;
                            }
                        }
                        // Skip if either genotype is missing (None)
                    }

                    // Return the pair's indices, differences, and comparable sites
                    (
                        (sample_idx_i, sample_idx_j),
                        difference_count,
                        comparable_site_count,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

pub fn harmonic(n: usize) -> f64 {
    (1..=n).map(|i| 1.0 / i as f64).sum()
}

pub fn calculate_watterson_theta(seg_sites: usize, n: usize, seq_length: i64) -> f64 {
    // Handle edge cases
    if n <= 1 || seq_length == 0 {
        return f64::INFINITY; // Return infinity if only 1 or fewer haplotypes or if sequence length is zero
    }

    let harmonic_value = harmonic(n - 1);
    if harmonic_value == 0.0 {
        return f64::INFINITY; // Return infinity to avoid division by zero
    }
    seg_sites as f64 / harmonic_value / seq_length as f64
}

/// Calculates nucleotide diversity (Pi) from variant data, accounting for partial missing data.
///
/// This function computes Pi as the average pairwise difference per site across all sample pairs,
/// using the number of differences and comparable sites per pair to handle missing genotypes.
///
/// # Returns
/// The nucleotide diversity (Pi) as a float, or special values for edge cases:
/// * `f64::INFINITY` if too few samples (n <= 1).
/// * `f64::NAN` if no valid pair comparisons are possible.
/// * `0.0` if no comparable sites exist between any pair.
pub fn calculate_pi(variants: &[Variant], haplotypes_in_group: &[(usize, HaplotypeSide)]) -> f64 {
    if haplotypes_in_group.len() <= 1 {
        return f64::INFINITY;
    }

    let total_possible_pairs = haplotypes_in_group.len() * (haplotypes_in_group.len() - 1) / 2;
    if total_possible_pairs == 0 {
        return f64::NAN;
    }

    let mut difference_sum = 0.0;
    for i in 0..haplotypes_in_group.len() {
        for j in (i + 1)..haplotypes_in_group.len() {
            let (sample_i, side_i) = haplotypes_in_group[i];
            let (sample_j, side_j) = haplotypes_in_group[j];

            let mut diff_count = 0;
            let mut comparable_sites = 0;

            for var in variants {
                if let Some(gt_i) = var.genotypes.get(sample_i) {
                    if let Some(gt_j) = var.genotypes.get(sample_j) {
                        if let Some(alleles_i) = gt_i {
                            if let Some(alleles_j) = gt_j {
                                if let (Some(&a_i), Some(&a_j)) = (
                                    alleles_i.get(side_i as usize),
                                    alleles_j.get(side_j as usize),
                                ) {
                                    comparable_sites += 1;
                                    if a_i != a_j {
                                        diff_count += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if comparable_sites > 0 {
                difference_sum += diff_count as f64 / comparable_sites as f64;
            }
        }
    }

    difference_sum / total_possible_pairs as f64
}

pub fn calculate_per_site(
    variants: &[Variant],
    haplotypes_in_group: &[(usize, HaplotypeSide)],
    region_start: i64,
    region_end: i64,
) -> Vec<SiteDiversity> {
    let max_haps = haplotypes_in_group.len();
    
    // Create a proper ZeroBasedHalfOpen for the region
    let region = crate::process::ZeroBasedHalfOpen::from_1based_inclusive(region_start, region_end);
    let mut site_diversities = Vec::with_capacity(region.len());
    
    if max_haps < 2 {
        return site_diversities;
    }
    
    // Convert variants to a HashMap for O(1) access by position (0-based)
    let variant_map: HashMap<i64, &Variant> = variants.iter().map(|v| (v.position, v)).collect();

    // Iterate over each position in the region (inclusive 0-based start to end)
    // Using region.start_1based_inclusive() and region.end_1based_inclusive() for correct boundaries
    let start_pos = region_start;
    let end_pos = region_end;
    
    for pos in start_pos..=end_pos {
        // Check if position is within our region using the struct's contains method
        if !region.contains(ZeroBasedPosition(pos)) {
            continue;
        }
        
        if let Some(var) = variant_map.get(&pos) {
            // Variant exists at this position; calculate diversity metrics
            let mut allele_counts = HashMap::new();
            let mut total_called = 0;

            for &(sample_index, side) in haplotypes_in_group {
                if let Some(gt_opt) = var.genotypes.get(sample_index) {
                    if let Some(gt) = gt_opt {
                        if let Some(&allele) = gt.get(side as usize) {
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
                (0.0, 0.0)
            } else {
                let mut freq_sq_sum = 0.0;
                for count in allele_counts.values() {
                    let freq = *count as f64 / total_called as f64;
                    freq_sq_sum += freq * freq;
                }
                let pi_value =
                    (total_called as f64 / (total_called as f64 - 1.0)) * (1.0 - freq_sq_sum);

                let distinct_alleles = allele_counts.len();
                let watterson_value = if distinct_alleles > 1 {
                    let denom = harmonic(total_called - 1);
                    if denom == 0.0 {
                        0.0
                    } else {
                        1.0 / denom
                    }
                } else {
                    0.0
                };
                (pi_value, watterson_value)
            };

            site_diversities.push(SiteDiversity {
                position: ZeroBasedPosition(pos).to_one_based(),
                pi: pi_value,
                watterson_theta: watterson_value,
            });
        } else {
            // No variant at this position (monomorphic site)
            site_diversities.push(SiteDiversity {
                position: ZeroBasedPosition(pos).to_one_based(),
                pi: 0.0,
                watterson_theta: 0.0,
            });
        }
    }

    site_diversities
}

#[cfg(test)]
mod tests {
    use super::*;
    include!("tests/stats_tests.rs");
    include!("tests/filter_tests.rs");
}
