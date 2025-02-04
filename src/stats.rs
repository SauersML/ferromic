use rayon::prelude::*;
use process::Variant;

fn calculate_masked_length(region_start: i64, region_end: i64, mask: &[(i64, i64)]) -> i64 { // Not used
    let mut total = 0;
    for &(start, end) in mask {
        let overlap_start = std::cmp::max(region_start, start);
        let overlap_end = std::cmp::min(region_end, end);
        if overlap_start <= overlap_end {
            total += overlap_end - overlap_start;
        } else if end > region_end {
            break; // No further overlaps possible
        }
    }
    total
}

fn calculate_adjusted_sequence_length(
    region_start: i64,
    region_end: i64,
    allow_regions_chr: Option<&Vec<(i64, i64)>>,
    mask_regions_chr: Option<&Vec<(i64, i64)>>,
) -> i64 {
    let mut allowed_intervals = Vec::new();
    if let Some(allow_regions) = allow_regions_chr {
        // Intersect the entry region with the allow regions
        for &(start, end) in allow_regions {
            let overlap_start = std::cmp::max(region_start, start);
            let overlap_end = std::cmp::min(region_end, end);
            if overlap_start <= overlap_end {
                allowed_intervals.push((overlap_start, overlap_end));
            }
        }
    } else {
        // If no allow regions, the entire entry region is allowed
        allowed_intervals.push((region_start, region_end));
    }
    // Subtract the masked regions from the allowed intervals
    let unmasked_intervals = subtract_regions(&allowed_intervals, mask_regions_chr);
    // Calculate the total length of unmasked intervals
    let adjusted_length: i64 = unmasked_intervals
        .iter()
        .map(|&(start, end)| end - start)
        .sum();
    adjusted_length
}

fn subtract_regions(
    intervals: &Vec<(i64, i64)>,
    masks: Option<&Vec<(i64, i64)>>
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

fn calculate_inversion_allele_frequency(
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

fn count_segregating_sites(variants: &[Variant]) -> usize {
    variants
        .par_iter()
        .filter(|v| {
            let alleles: HashSet<_> = v.genotypes
                .iter()
                .flatten()
                .flatten()
                .collect();
            alleles.len() > 1
        })
        .count()
}

fn calculate_pairwise_differences(
    variants: &[Variant],
    n: usize,
) -> Vec<((usize, usize), usize, Vec<i64>)> {
    let variants = Arc::new(variants);
    // Iterate over all sample indices from 0 to n - 1
    (0..n).into_par_iter().flat_map(|i| {
        let variants = Arc::clone(&variants);
        // For each i, iterate over j from i + 1 to n - 1
        (i+1..n).into_par_iter().map(move |j| {
            let mut diff_count = 0;
            let mut diff_positions = Vec::new();
            // For each variant, compare genotypes of samples i and j
            for v in variants.iter() {
                if let (Some(gi), Some(gj)) = (&v.genotypes[i], &v.genotypes[j]) {
                    if gi != gj {
                        diff_count += 1;
                        diff_positions.push(v.position);
                    }
                } else {
                    // Skip if either genotype is missing
                    continue;
                }
            }
            // Return the pair of sample indices, difference count, and positions
            ((i, j), diff_count, diff_positions)
        }).collect::<Vec<_>>()
    }).collect()
}

fn harmonic(n: usize) -> f64 {
    (1..=n).map(|i| 1.0 / i as f64).sum()
}

fn calculate_watterson_theta(seg_sites: usize, n: usize, seq_length: i64) -> f64 {
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

fn calculate_pi(tot_pair_diff: usize, n: usize, seq_length: i64) -> f64 {
    // Handle edge cases
    if n <= 1 || seq_length == 0 {
        return f64::INFINITY; // Return infinity if only 1 or fewer haplotypes or if sequence length is zero
    }
    let num_comparisons = n * (n - 1) / 2;
    if num_comparisons == 0 {
        return f64::NAN; // Return NaN if there's somehow no valid pairwise comparison
    }
    tot_pair_diff as f64 / num_comparisons as f64 / seq_length as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    include!("tests/stats_tests.rs");
    include!("tests/filter_tests.rs");
}
