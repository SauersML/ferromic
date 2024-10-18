use clap::Parser;
use colored::*;
use flate2::read::MultiGzDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use parking_lot::Mutex;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::collections::{HashMap, HashSet};
use csv::{ReaderBuilder, WriterBuilder};
use crossbeam_channel::bounded;
use std::time::Duration;
use std::sync::Arc;
use std::thread;
use bio::io::fasta::{IndexedReader, FastaRead};

// Define command-line arguments using clap
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    // Folder containing VCF files
    #[arg(short, long = "vcf_folder")]
    vcf_folder: String,

    // Chromosome to process
    #[arg(short, long = "chr")]
    chr: Option<String>,

    // Region to process (start-end)
    #[arg(short, long = "region")]
    region: Option<String>,

    // Configuration file
    #[arg(long = "config_file")]
    config_file: Option<String>,

    // Output file
    #[arg(short, long = "output_file")]
    output_file: Option<String>,

    // Minimum genotype quality
    #[arg(long = "min_gq", default_value = "30")]
    min_gq: u16,

    // Mask file (regions to exclude)
    #[arg(long = "mask_file")]
    mask_file: Option<String>,

    // Allow file (regions to include)
    #[arg(long = "allow_file")]
    allow_file: Option<String>,
}

// Data structures
#[derive(Debug, Clone)]
struct ConfigEntry {
    seqname: String,
    start: i64,
    end: i64,
    samples_unfiltered: HashMap<String, (u8, u8)>,
    samples_filtered: HashMap<String, (u8, u8)>,
}

#[derive(Debug)]
struct RegionStats {
    chr: String,
    region_start: i64,
    region_end: i64,
    sequence_length: i64,
    segregating_sites: usize,
    w_theta: f64,
    pi: f64,
}

#[derive(Debug, Default)]
struct FilteringStats {
    total_variants: usize,
    _filtered_variants: usize,
    filtered_due_to_mask: usize,
    filtered_due_to_allow: usize,
    filtered_positions: HashSet<i64>,
    missing_data_variants: usize,
    low_gq_variants: usize,
    multi_allelic_variants: usize,
    filtered_examples: Vec<String>,
}

impl FilteringStats {
    // Adds an example if there are fewer than 5
    fn add_example(&mut self, example: String) {
        if self.filtered_examples.len() < 5 {
            // println!("Adding example - {}", example); // Debug
            self.filtered_examples.push(example);
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
struct Variant {
    position: i64,
    genotypes: Vec<Option<Vec<u8>>>,
}

struct AlleleInfo {
    sample_name: String, // Or could do sample_index if name isn't available
    haplotype_group: u8,        // 0 or 1 for haplotype group
    nucleotide: char,           // Nucleotide (A, C, T, G)
    chromosome: String,         // Chromosome identifier
    position: i64,              // Chromosome position
}

#[derive(Debug, Default, Clone)]
struct MissingDataInfo {
    total_data_points: usize,
    missing_data_points: usize,
    positions_with_missing: HashSet<i64>,
}

// Custom error types
#[derive(Debug)]
enum VcfError {
    Io(io::Error),
    Parse(String),
    InvalidRegion(String),
    NoVcfFiles,
    InvalidVcfFormat(String),
    ChannelSend,
    ChannelRecv,
}

impl<T> From<crossbeam_channel::SendError<T>> for VcfError {
    fn from(_: crossbeam_channel::SendError<T>) -> Self {
        VcfError::ChannelSend
    }
}

impl From<crossbeam_channel::RecvError> for VcfError {
    fn from(_: crossbeam_channel::RecvError) -> Self {
        VcfError::ChannelRecv
    }
}

impl std::fmt::Display for VcfError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            VcfError::Io(err) => write!(f, "IO error: {}", err),
            VcfError::Parse(msg) => write!(f, "Parse error: {}", msg),
            VcfError::InvalidRegion(msg) => write!(f, "Invalid region: {}", msg),
            VcfError::NoVcfFiles => write!(f, "No VCF files found"),
            VcfError::InvalidVcfFormat(msg) => write!(f, "Invalid VCF format: {}", msg),
            VcfError::ChannelSend => write!(f, "Error sending data through channel"),
            VcfError::ChannelRecv => write!(f, "Error receiving data from channel"),
        }
    }
}

impl From<io::Error> for VcfError {
    fn from(err: io::Error) -> VcfError {
        VcfError::Io(err)
    }
}

// Main function
fn main() -> Result<(), VcfError> {
    let args = Args::parse();

    // Set Rayon to use all logical CPUs
    let num_logical_cpus = num_cpus::get();
    ThreadPoolBuilder::new()
        .num_threads(num_logical_cpus)
        .build_global()
        .unwrap();

    // Parse the mask file (exclude regions)
    let mask_regions = if let Some(mask_file) = args.mask_file.as_ref() {
        println!("Mask file provided: {}", mask_file);
        Some(Arc::new(parse_regions_file(Path::new(mask_file))?))
    } else {
        None
    };

    // Parse the allow file (include regions)
    let allow_regions = if let Some(allow_file) = args.allow_file.as_ref() {
        println!("Mask file provided: {}", allow_file);
        let parsed_allow = parse_regions_file(Path::new(allow_file))?;
        println!("Parsed Allow Regions: {:?}", parsed_allow);
        Some(Arc::new(parsed_allow))
    } else {
        None
    };

    println!("{}", "Starting VCF diversity analysis...".green());

    if let Some(config_file) = args.config_file.as_ref() {
        println!("Config file provided: {}", config_file);
        let config_entries = parse_config_file(Path::new(config_file))?;
        for entry in &config_entries {
            println!("Config entry chromosome: {}", entry.seqname);
        }
        let output_file = args
            .output_file
            .as_ref()
            .map(Path::new)
            .unwrap_or_else(|| Path::new("output.csv"));
        println!("Output file: {}", output_file.display());
        process_config_entries(
            &config_entries,
            &args.vcf_folder,
            output_file,
            args.min_gq,
            mask_regions.clone(),
            allow_regions.clone(),
        )?;
    } else if let Some(chr) = args.chr.as_ref() {
        println!("Chromosome provided: {}", chr);
        let (start, end) = if let Some(region) = args.region.as_ref() {
            println!("Region provided: {}", region);
            parse_region(region)?
        } else {
            println!("No region provided, using default region covering most of the chromosome.");
            (1, i64::MAX)
        };
        let vcf_file = find_vcf_file(&args.vcf_folder, chr)?;

        println!(
            "{}",
            format!("Processing VCF file: {}", vcf_file.display()).cyan()
        );

        // Process the VCF file
        let (
            unfiltered_variants,
            _filtered_variants,
            sample_names,
            chr_length,
            missing_data_info,
            _filtering_stats,
        ) = process_vcf(
            &vcf_file,
            &chr,
            start,
            end,
            args.min_gq,
            mask_regions.clone(),
            allow_regions.clone(),
        )?;

        println!("{}", "Calculating diversity statistics...".blue());

        let seq_length = if end == i64::MAX {
            unfiltered_variants
                .last()
                .map(|v| v.position)
                .unwrap_or(0)
                .max(chr_length)
                - start
                + 1
        } else {
            end - start + 1
        };

        if end == i64::MAX
            && unfiltered_variants
                .last()
                .map(|v| v.position)
                .unwrap_or(0)
                < chr_length
        {
            println!("{}", "Warning: The sequence length may be underestimated. Consider using the --region parameter for more accurate results.".yellow());
        }

        let num_segsites = count_segregating_sites(&unfiltered_variants); // Also need filtered here? Output required in csv: 0_segregating_sites_filtered, 1_segregating_sites_filtered
        let raw_variant_count = unfiltered_variants.len();

        let n = sample_names.len();
        if n == 0 {
            return Err(VcfError::Parse(
                "No samples found after processing VCF.".to_string(),
            ));
        }

        let pairwise_diffs = calculate_pairwise_differences(&unfiltered_variants, n); // Also need filtered here?
        let tot_pair_diff: usize = pairwise_diffs.iter().map(|&(_, count, _)| count).sum();

        let w_theta = calculate_watterson_theta(num_segsites, n, seq_length);
        let pi = calculate_pi(tot_pair_diff, n, seq_length);

        println!("\n{}", "Results:".green().bold());
        println!("\nSequence Length:{}", seq_length);
        println!("Number of Segregating Sites:{}", num_segsites);
        println!("Raw Variant Count:{}", raw_variant_count);
        println!("Watterson Theta:{:.6}", w_theta);
        println!("pi:{:.6}", pi);

        if unfiltered_variants.is_empty() {
            println!(
                "{}",
                "Warning: No variants found in the specified region.".yellow()
            );
        }

        if num_segsites == 0 {
            println!("{}", "Warning: All sites are monomorphic.".yellow());
        }

        if num_segsites != raw_variant_count {
            println!(
                "{}",
                format!(
                    "Note: Number of segregating sites ({}) differs from raw variant count ({}).",
                    num_segsites, raw_variant_count
                )
                .yellow()
            );
        }

        println!("\n{}", "Filtering Statistics:".green().bold());
        println!(
            "Total variants processed: {}",
            _filtering_stats.total_variants
        );
        println!(
            "Filtered variants: {} ({:.2}%)",
            _filtering_stats._filtered_variants,
            (_filtering_stats._filtered_variants as f64 / _filtering_stats.total_variants as f64)
                * 100.0
        );
        println!("Multi-allelic variants: {}", _filtering_stats.multi_allelic_variants);
        println!("Low GQ variants: {}", _filtering_stats.low_gq_variants);
        println!(
            "Missing data variants: {}",
            _filtering_stats.missing_data_variants
        );

        let missing_data_percentage =
            (missing_data_info.missing_data_points as f64 / missing_data_info.total_data_points as f64) * 100.0;
        println!("\n{}", "Missing Data Information:".yellow().bold());
        println!(
            "Number of missing data points: {}",
            missing_data_info.missing_data_points
        );
        println!("Percentage of missing data: {:.2}%", missing_data_percentage);
        println!(
            "Number of positions with missing data: {}",
            missing_data_info.positions_with_missing.len()
        );
    } else {
        return Err(VcfError::Parse(
            "Either config file or chromosome must be specified".to_string(),
        ));
    }

    println!("{}", "Analysis complete.".green());
    Ok(())
}

// Function to parse regions file (mask or allow)
fn parse_regions_file(
    path: &Path,
) -> Result<HashMap<String, Vec<(i64, i64)>>, VcfError> {
    let is_bed_file = path.extension().and_then(|s| s.to_str()) == Some("bed");

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut regions: HashMap<String, Vec<(i64, i64)>> = HashMap::new();

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 3 {
            eprintln!(
                "{}",
                format!("Skipping invalid line {}: '{}'", line_num + 1, line).red()
            );
            continue; // Skip invalid lines
        }

        let chr = fields[0].trim_start_matches("chr").to_string(); // Normalize chromosome name
        let start: i64 = match fields[1].trim().parse() {
            Ok(val) => val,
            Err(_) => {
                eprintln!(
                    "{}",
                    format!(
                        "Invalid start position on line {}: '{}'",
                        line_num + 1,
                        fields[1]
                    )
                    .red()
                );
                continue;
            }
        };
        let end: i64 = match fields[2].trim().parse() {
            Ok(val) => val,
            Err(_) => {
                eprintln!(
                    "{}",
                    format!(
                        "Invalid end position on line {}: '{}'",
                        line_num + 1,
                        fields[2]
                    )
                    .red()
                );
                continue;
            }
        };

        // Adjust positions based on file type
        let (start, end) = if is_bed_file {
            // BED files are zero-based, half-open intervals [start, end)
            (start, end)
        } else {
            // Other files are one-based, inclusive intervals [start-1, end)
            (start - 1, end)
        };

        regions.entry(chr.clone()).or_default().push((start, end));
    }

    // Sort the intervals for each chromosome
    for intervals in regions.values_mut() {
        intervals.sort_by_key(|&(start, _)| start);
    }

    Ok(regions)
}

// Function to check if a position is within any of the regions
fn position_in_regions(pos: i64, regions: &[(i64, i64)]) -> bool {
    // pos is zero-based
    // regions are sorted by start position
    let mut left = 0;
    let mut right = regions.len();

    while left < right {
        let mid = (left + right) / 2;
        let (start, end) = regions[mid];
        if pos < start {
            right = mid;
        } else if pos >= end {
            left = mid + 1;
        } else {
            // pos in [start, end)
            return true;
        }
    }
    false
}

fn calculate_masked_length(region_start: i64, region_end: i64, mask: &[(i64, i64)]) -> i64 {
    let mut total = 0;
    for &(start, end) in mask {
        let overlap_start = std::cmp::max(region_start, start);
        let overlap_end = std::cmp::min(region_end, end);
        if overlap_start <= overlap_end {
            total += overlap_end - overlap_start + 1;
        } else if end > region_end {
            break; // No further overlaps possible
        }
    }
    total
}

fn process_variants(
    variants: &[Variant],
    sample_names: &[String],
    haplotype_group: u8,
    sample_filter: &HashMap<String, (u8, u8)>,
    region_start: i64,
    region_end: i64,
    adjusted_sequence_length: Option<i64>,
) -> Result<Option<(usize, f64, f64, usize)>, VcfError> {
    let mut vcf_sample_id_to_index: HashMap<&str, usize> = HashMap::new();
    for (i, name) in sample_names.iter().enumerate() {
        let sample_id = extract_sample_id(name);
        if vcf_sample_id_to_index.contains_key(sample_id) {
            return Err(VcfError::Parse(format!(
                "Duplicate sample ID '{}' in VCF.",
                sample_id
            )));
        }
        vcf_sample_id_to_index.insert(sample_id, i);
    }

    let mut haplotype_indices = Vec::new();

    for (sample_name, &(left_tsv, right_tsv)) in sample_filter.iter() {
        if let Some(&i) = vcf_sample_id_to_index.get(sample_name.as_str()) {
            if left_tsv == haplotype_group as u8 {
                haplotype_indices.push((i, 0)); // Include left haplotype
            }
            if right_tsv == haplotype_group as u8 {
                haplotype_indices.push((i, 1)); // Include right haplotype
            }
        } else {
            // Sample not found in VCF
        }
    }

    if haplotype_indices.is_empty() {
        println!(
            "No haplotypes found for the specified group {}.",
            haplotype_group
        );
        return Ok(None);
    }

    let mut num_segsites = 0;
    let mut tot_pair_diff = 0;
    let n = haplotype_indices.len();

    for variant in variants {
        // Skip variants outside the region (redundant if already filtered)
        if variant.position < region_start || variant.position > region_end {
            continue;
        }

        // Collect the alleles for the haplotypes
        let mut variant_alleles = Vec::new(); // Alleles of haplotypes at this variant

        for &(i, allele_idx) in &haplotype_indices {
            let allele = variant.genotypes.get(i)
                .and_then(|gt| gt.as_ref())
                .and_then(|alleles| alleles.get(allele_idx))
                .copied();
            variant_alleles.push(allele); // allele is Option<u8>
        }

        // Determine if the variant is a segregating site
        let alleles_present: Vec<u8> = variant_alleles.iter().filter_map(|&a| a).collect();
        let unique_alleles: HashSet<_> = alleles_present.iter().cloned().collect();
        if unique_alleles.len() > 1 {
            num_segsites += 1;
        }

        // Compute pairwise differences
        for i in 0..n {
            if let Some(allele_i) = variant_alleles[i] {
                for j in (i + 1)..n {
                    if let Some(allele_j) = variant_alleles[j] {
                        if allele_i != allele_j {
                            tot_pair_diff += 1;
                        }
                    }
                }
            }
        }
    }

    let seq_length = adjusted_sequence_length.unwrap_or(region_end - region_start + 1);
    let w_theta = calculate_watterson_theta(num_segsites, n, seq_length);
    let pi = calculate_pi(tot_pair_diff, n, seq_length);

    Ok(Some((num_segsites, w_theta, pi, n)))
}

fn process_config_entries(
    config_entries: &[ConfigEntry],
    vcf_folder: &str,
    output_file: &Path,
    min_gq: u16,
    mask: Option<Arc<HashMap<String, Vec<(i64, i64)>>>>,
    allow: Option<Arc<HashMap<String, Vec<(i64, i64)>>>>,
) -> Result<(), VcfError> {
    let mut writer = WriterBuilder::new()
        .has_headers(true)
        .from_path(output_file)
        .map_err(|e| VcfError::Io(e.into()))?;

    // Write headers
    writer
        .write_record(&[
            "chr",
            "region_start",
            "region_end",
            "0_sequence_length",
            "1_sequence_length",
            "0_sequence_length_adjusted",
            "1_sequence_length_adjusted",
            "0_segregating_sites",
            "1_segregating_sites",
            "0_w_theta",
            "1_w_theta",
            "0_pi",
            "1_pi",
            "0_segregating_sites_filtered",
            "1_segregating_sites_filtered",
            "0_w_theta_filtered",
            "1_w_theta_filtered",
            "0_pi_filtered",
            "1_pi_filtered",
            "0_num_hap_no_filter",
            "1_num_hap_no_filter",
            "0_num_hap_filter",
            "1_num_hap_filter",
            "inversion_freq_no_filter",
            "inversion_freq_filter",
        ])
        .map_err(|e| VcfError::Io(e.into()))?;

    // Organize regions by chromosome
    let mut regions_per_chr: HashMap<String, Vec<&ConfigEntry>> = HashMap::new();
    for entry in config_entries {
        regions_per_chr
            .entry(entry.seqname.clone())
            .or_insert_with(Vec::new)
            .push(entry);
    }

    for (chr, entries) in regions_per_chr {
        println!("Processing chromosome: {}", chr);
    
        // Determine the range to process
        let min_start = entries.iter().map(|e| e.start).min().unwrap_or(0);
        let max_end = entries.iter().map(|e| e.end).max().unwrap_or(i64::MAX);
    
        // Locate the appropriate VCF file
        let vcf_file = match find_vcf_file(vcf_folder, &chr) {
            Ok(file) => file,
            Err(e) => {
                eprintln!("Error finding VCF file for {}: {:?}", chr, e);
                continue;
            }
        };
    
        println!(
            "Processing VCF file for chromosome {} from {} to {}",
            chr, min_start, max_end
        );
    
        // Pass the mask and allow regions (clone the Arc)
        let variants_data = match process_vcf(
            &vcf_file,
            &chr,
            min_start,
            max_end,
            min_gq,
            mask.clone(),
            allow.clone(),
        ) {
            Ok(data) => data,
            Err(e) => {
                eprintln!("Error processing VCF file for {}: {}", chr, e);
                continue;
            }
        };
    
        let (
            unfiltered_variants,
            _filtered_variants,
            sample_names,
            _chr_length,
            _missing_data_info,
            _filtering_stats,
        ) = variants_data;

        println!("\n{}", "Filtering Statistics:".green().bold());
        println!("Total variants processed: {}", _filtering_stats.total_variants);
        println!(
            "Filtered variants: {} ({:.2}%)",
            _filtering_stats._filtered_variants,
            (_filtering_stats._filtered_variants as f64 / _filtering_stats.total_variants as f64)
                * 100.0
        );
        println!("Filtered due to allow: {}", _filtering_stats.filtered_due_to_allow);
        println!("Filtered due to mask: {}", _filtering_stats.filtered_due_to_mask);
        println!("Multi-allelic variants: {}", _filtering_stats.multi_allelic_variants);
        println!("Low GQ variants: {}", _filtering_stats.low_gq_variants);
        println!("Missing data variants: {}", _filtering_stats.missing_data_variants);
    
        println!("\n{}", "Example Filtered Variants:".green().bold());
        for (i, example) in _filtering_stats.filtered_examples.iter().enumerate().take(5) {
            println!("Example {}: {}", i + 1, example);
        }
        if _filtering_stats.filtered_examples.len() > 5 {
            println!(
                "... and {} more.",
                _filtering_stats.filtered_examples.len() - 5
            );
        }

        // Collect all config samples for this chromosome
        let all_config_samples: HashSet<String> = entries
            .iter()
            .flat_map(|entry| {
                entry
                    .samples_unfiltered
                    .keys()
                    .cloned()
                    .chain(entry.samples_filtered.keys().cloned())
            })
            .collect();

        // Collect VCF sample names
        let vcf_sample_set: HashSet<String> = sample_names
            .iter()
            .map(|s| extract_sample_id(s).to_string())
            .collect();

        // Find missing samples
        let missing_samples: Vec<String> = all_config_samples
            .difference(&vcf_sample_set)
            .cloned()
            .collect();

        // Print warning if there are missing samples
        if !missing_samples.is_empty() {
            eprintln!(
                "Warning: The following samples from config file are missing in VCF for chromosome {}: {:?}",
                chr, missing_samples
            );
        }

        for entry in entries {
            println!(
                "Processing entry: {}:{}-{}",
                entry.seqname, entry.start, entry.end
            );

            // Define regions
            let sequence_length = entry.end - entry.start + 1;

            // Calculate adjusted sequence length considering allow and mask regions
            let adjusted_sequence_length = calculate_adjusted_sequence_length(
                entry.start,
                entry.end,
                allow.as_ref().and_then(|a| a.get(&chr)),
                mask.as_ref().and_then(|m| m.get(&chr)),
            );

            // Process haplotype_group=0 (unfiltered)
            let (num_segsites_0, w_theta_0, pi_0, n_hap_0_no_filter) =
                match process_variants(
                    &unfiltered_variants,
                    &sample_names,
                    0,
                    &entry.samples_unfiltered,
                    entry.start,
                    entry.end,
                    None,
                )? {
                    Some(values) => values,
                    None => continue, // Skip writing this record
                };

            // Process haplotype_group=1 (unfiltered)
            let (num_segsites_1, w_theta_1, pi_1, n_hap_1_no_filter) =
                match process_variants(
                    &unfiltered_variants,
                    &sample_names,
                    1,
                    &entry.samples_unfiltered,
                    entry.start,
                    entry.end,
                    None,
                )? {
                    Some(values) => values,
                    None => continue, // Skip writing this record
                };

            // Calculate allele frequency of inversions (no filter)
            let inversion_freq_no_filter =
                calculate_inversion_allele_frequency(&entry.samples_unfiltered);

            // Process haplotype_group=0 (filtered)
            let (num_segsites_0_filt, w_theta_0_filt, pi_0_filt, n_hap_0_filt) =
                match process_variants(
                    &_filtered_variants,
                    &sample_names,
                    0,
                    &entry.samples_filtered,
                    entry.start,
                    entry.end,
                    Some(adjusted_sequence_length),
                )? {
                    Some(values) => values,
                    None => continue, // Skip writing this record
                };

            // Process haplotype_group=1 (filtered)
            let (num_segsites_1_filt, w_theta_1_filt, pi_1_filt, n_hap_1_filt) =
                match process_variants(
                    &_filtered_variants,
                    &sample_names,
                    1,
                    &entry.samples_filtered,
                    entry.start,
                    entry.end,
                    Some(adjusted_sequence_length),
                )? {
                    Some(values) => values,
                    None => continue, // Skip writing this record
                };

            // Calculate allele frequency of inversions
            let inversion_freq_filt =
                calculate_inversion_allele_frequency(&entry.samples_filtered);

            // Write the aggregated results to CSV
            writer
                .write_record(&[
                    &entry.seqname,
                    &entry.start.to_string(),
                    &entry.end.to_string(),
                    &sequence_length.to_string(),          // 0_sequence_length
                    &sequence_length.to_string(),          // 1_sequence_length
                    &adjusted_sequence_length.to_string(), // 0_sequence_length_adjusted
                    &adjusted_sequence_length.to_string(), // 1_sequence_length_adjusted
                    &num_segsites_0.to_string(),           // 0_segregating_sites
                    &num_segsites_1.to_string(),           // 1_segregating_sites
                    &format!("{:.6}", w_theta_0),          // 0_w_theta
                    &format!("{:.6}", w_theta_1),          // 1_w_theta
                    &format!("{:.6}", pi_0),               // 0_pi
                    &format!("{:.6}", pi_1),               // 1_pi
                    &num_segsites_0_filt.to_string(),      // 0_segregating_sites_filtered
                    &num_segsites_1_filt.to_string(),      // 1_segregating_sites_filtered
                    &format!("{:.6}", w_theta_0_filt),     // 0_w_theta_filtered
                    &format!("{:.6}", w_theta_1_filt),     // 1_w_theta_filtered
                    &format!("{:.6}", pi_0_filt),          // 0_pi_filtered
                    &format!("{:.6}", pi_1_filt),          // 1_pi_filtered
                    &n_hap_0_no_filter.to_string(),        // 0_num_hap_no_filter
                    &n_hap_1_no_filter.to_string(),        // 1_num_hap_no_filter
                    &n_hap_0_filt.to_string(),             // 0_num_hap_filter
                    &n_hap_1_filt.to_string(),             // 1_num_hap_filter
                    // -1.0 should never occur
                    &format!("{:.6}", inversion_freq_no_filter.unwrap_or(-1.0)), // inversion_freq_no_filter
                    &format!("{:.6}", inversion_freq_filt.unwrap_or(-1.0)),      // inversion_freq_filter
                ])
                .map_err(|e| VcfError::Io(e.into()))?;

            println!(
                "Successfully wrote record for {}:{}-{}",
                entry.seqname, entry.start, entry.end
            );
            writer.flush().map_err(|e| VcfError::Io(e.into()))?;
        }
    }

    writer.flush().map_err(|e| VcfError::Io(e.into()))?;
    println!("Processing complete. Check the output file: {:?}", output_file);
    Ok(())
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
        .map(|&(start, end)| end - start + 1)
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
                        new_intervals.push((curr_start, mask_start - 1));
                    }
                    if mask_end < curr_end {
                        new_intervals.push((mask_end + 1, curr_end));
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

fn parse_config_file(path: &Path) -> Result<Vec<ConfigEntry>, VcfError> {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(path)
        .map_err(|e| VcfError::Io(e.into()))?;

    let headers = reader.headers().map_err(|e| VcfError::Io(e.into()))?.clone();
    let sample_names: Vec<String> = headers.iter().skip(7).map(String::from).collect();

    // Check if the number of sample names is consistent
    if sample_names.is_empty() {
        eprintln!("{}", "Error: No sample names found in the configuration file header after skipping the first 7 columns. Please ensure the config file is properly formatted with tabs separating all columns, including sample names.".red());
        return Err(VcfError::Parse("No sample names found in config file header.".to_string()));
    }

    let mut entries = Vec::new();
    let mut invalid_genotypes = 0;
    let mut total_genotypes = 0;

    for (line_num, result) in reader.records().enumerate() {
        let record = result.map_err(|e| VcfError::Io(e.into()))?;

        // Check if the record has the expected number of fields
        if record.len() != headers.len() {
            eprintln!("{}", format!("Error: Record on line {} does not have the same number of fields as the header. Expected {}, found {}. Please check for missing tabs in the config file.", line_num + 2, headers.len(), record.len()).red());
            return Err(VcfError::Parse(format!("Mismatched number of fields in record on line {}", line_num + 2)));
        }

        // Normalize chromosome name by removing "chr" prefix
        let seqname = record.get(0)
            .ok_or(VcfError::Parse("Missing seqname".to_string()))?
            .trim()
            .trim_start_matches("chr")
            .to_string();
        let start: i64 = record.get(1).ok_or(VcfError::Parse("Missing start".to_string()))?.parse().map_err(|_| VcfError::Parse("Invalid start".to_string()))?;
        let end: i64 = record.get(2).ok_or(VcfError::Parse("Missing end".to_string()))?.parse().map_err(|_| VcfError::Parse("Invalid end".to_string()))?;

        let mut samples_unfiltered = HashMap::new();
        let mut samples_filtered = HashMap::new();

        for (i, field) in record.iter().enumerate().skip(7) {
            total_genotypes += 1;
            if i < sample_names.len() + 7 {
                let sample_name = &sample_names[i - 7];
                
                // For samples_unfiltered (split on '_')
                let genotype_str_unfiltered = field.split('_').next().unwrap_or("");
                
                if genotype_str_unfiltered.len() >= 3 && genotype_str_unfiltered.chars().nth(1) == Some('|') {
                    let left_char = genotype_str_unfiltered.chars().nth(0).unwrap();
                    let right_char = genotype_str_unfiltered.chars().nth(2).unwrap();
                    if let (Some(left), Some(right)) = (left_char.to_digit(10), right_char.to_digit(10)) {
                        let left = left as u8;
                        let right = right as u8;
                        if left <= 1 && right <= 1 {
                            samples_unfiltered.insert(sample_name.clone(), (left, right));
                        } else {
                            invalid_genotypes += 1;
                        }
                    } else {
                        invalid_genotypes += 1;
                    }
                } else {
                    invalid_genotypes += 1;
                }
                
                // For samples_filtered (exact matches)
                if field == "0|0" || field == "0|1" || field == "1|0" || field == "1|1" {
                    let left = field.chars().nth(0).unwrap().to_digit(10).unwrap() as u8;
                    let right = field.chars().nth(2).unwrap().to_digit(10).unwrap() as u8;
                    samples_filtered.insert(sample_name.clone(), (left, right));
                }
            } else {
                eprintln!("Warning: More genotype fields than sample names at line {}.", line_num + 2);
            }
        }

        if samples_unfiltered.is_empty() {
            println!("Warning: No valid genotypes found for region {}:{}-{}", seqname, start, end);
            continue;
        }

        entries.push(ConfigEntry {
            seqname,
            start,
            end,
            samples_unfiltered,
            samples_filtered,
        });
    }

    let invalid_percentage = (invalid_genotypes as f64 / total_genotypes as f64) * 100.0;
    println!("Number of invalid genotypes: {} ({:.2}%)", invalid_genotypes, invalid_percentage);

    Ok(entries)
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

fn parse_region(region: &str) -> Result<(i64, i64), VcfError> {
    let parts: Vec<&str> = region.split('-').collect();
    if parts.len() != 2 {
        return Err(VcfError::InvalidRegion(
            "Invalid region format. Use start-end".to_string(),
        ));
    }
    let start: i64 = parts[0]
        .parse()
        .map_err(|_| VcfError::InvalidRegion("Invalid start position".to_string()))?;
    let end: i64 = parts[1]
        .parse()
        .map_err(|_| VcfError::InvalidRegion("Invalid end position".to_string()))?;
    if start >= end {
        return Err(VcfError::InvalidRegion(
            "Start position must be less than end position".to_string(),
        ));
    }
    Ok((start, end))
}

fn find_vcf_file(folder: &str, chr: &str) -> Result<PathBuf, VcfError> {
    let path = Path::new(folder);
    let chr_specific_files: Vec<_> = fs::read_dir(path)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            let path = entry.path();
            let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            let chr_pattern = format!("chr{}", chr);
            (file_name.starts_with(&chr_pattern) || file_name.starts_with(chr)) &&
                (file_name.ends_with(".vcf") || file_name.ends_with(".vcf.gz")) &&
                file_name.chars().nth(chr_pattern.len()).map_or(false, |c| !c.is_ascii_digit())
        })
        .map(|entry| entry.path())
        .collect();

    match chr_specific_files.len() {
        0 => Err(VcfError::NoVcfFiles),
        1 => Ok(chr_specific_files[0].clone()),
        _ => {
            let exact_match = chr_specific_files.iter().find(|&file| {
                let file_name = file.file_name().and_then(|n| n.to_str()).unwrap_or("");
                let chr_pattern = format!("chr{}", chr);
                (file_name.starts_with(&chr_pattern) || file_name.starts_with(chr)) &&
                    file_name.chars().nth(chr_pattern.len()).map_or(false, |c| !c.is_ascii_digit())
            });

            if let Some(exact_file) = exact_match {
                Ok(exact_file.clone())
            } else {
                println!("{}", "Multiple VCF files found:".yellow());
                for (i, file) in chr_specific_files.iter().enumerate() {
                    println!("{}. {}", i + 1, file.display());
                }
                
                println!("Please enter the number of the file you want to use:");
                let mut input = String::new();
                io::stdin().read_line(&mut input)?;
                let choice: usize = input.trim().parse().map_err(|_| VcfError::Parse("Invalid input".to_string()))?;
                
                chr_specific_files.get(choice - 1)
                    .cloned()
                    .ok_or_else(|| VcfError::Parse("Invalid file number".to_string()))
            }
        }
    }
}

fn open_vcf_reader(path: &Path) -> Result<Box<dyn BufRead + Send>, VcfError> {
    let file = File::open(path)?;
    
    if path.extension().and_then(|s| s.to_str()) == Some("gz") {
        let decoder = MultiGzDecoder::new(file);
        Ok(Box::new(BufReader::new(decoder)))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

// Function to collect all unique chromosome names from VCF files in the folder
fn collect_vcf_chromosomes(vcf_folder: &str) -> Result<Vec<String>, VcfError> {
    let path = Path::new(vcf_folder);
    let mut chromosomes = HashSet::new();

    // Iterate over all files in the VCF folder
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let file_path = entry.path();

        // Process only .vcf and .vcf.gz files
        if let Some(ext) = file_path.extension().and_then(|s| s.to_str()) {
            if ext != "vcf" && ext != "gz" {
                continue;
            }
        } else {
            continue;
        }

        // Open the VCF file (handle gzipped files)
        let file = File::open(&file_path)?;
        let mut reader: Box<dyn BufRead> = if file_path.extension().and_then(|s| s.to_str()) == Some("gz") {
            Box::new(BufReader::new(MultiGzDecoder::new(file)))
        } else {
            Box::new(BufReader::new(file))
        };

        // Read lines until header is found
        let mut buffer = String::new();
        while reader.read_line(&mut buffer)? > 0 {
            if buffer.starts_with("#CHROM") {
                break;
            }
            buffer.clear();
        }

        // Read variant records to collect chromosome names
        for line_result in reader.lines() {
            let line = line_result?;
            if line.starts_with("#") {
                continue; // Skip header lines
            }
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() < 1 {
                continue; // Skip invalid lines
            }
            let vcf_chr = fields[0].trim_start_matches("chr").to_string();
            if !vcf_chr.is_empty() {
                chromosomes.insert(vcf_chr);
            }
        }
    }

    let chromosomes: Vec<String> = chromosomes.into_iter().collect();
    Ok(chromosomes)
}


// Function to process a VCF file
fn process_vcf(
    file: &Path,
    chr: &str,
    start: i64,
    end: i64,
    min_gq: u16,
    mask_regions: Option<Arc<HashMap<String, Vec<(i64, i64)>>>>,
    allow_regions: Option<Arc<HashMap<String, Vec<(i64, i64)>>>>,
) -> Result<(
    Vec<Variant>,        // Unfiltered variants
    Vec<Variant>,        // Filtered variants
    Vec<String>,         // Sample names
    i64,                 // Chromosome length
    MissingDataInfo,
    FilteringStats,
), VcfError> {
    let mut reader = open_vcf_reader(file)?;
    let mut sample_names = Vec::new();
    let chr_length = 0;
    
    let unfiltered_variants = Arc::new(Mutex::new(Vec::new()));
    let _filtered_variants = Arc::new(Mutex::new(Vec::new()));

    let missing_data_info = Arc::new(Mutex::new(MissingDataInfo::default()));
    let _filtering_stats = Arc::new(Mutex::new(FilteringStats::default()));

    let is_gzipped = file.extension().and_then(|s| s.to_str()) == Some("gz");
    let progress_bar = if is_gzipped {
        ProgressBar::new_spinner()
    } else {
        let file_size = fs::metadata(file)?.len();
        ProgressBar::new(file_size)
    };

    let style = if is_gzipped {
        ProgressStyle::default_spinner()
            .template("{spinner:.bold.green} 🧬 {msg} 🧬 [{elapsed_precise}]")
            .expect("Failed to create spinner template")
            .tick_strings(&[
                "░▒▓██▓▒░", "▒▓██▓▒░", "▓██▓▒░", "██▓▒░", "█▓▒░", "▓▒░", "▒░", "░", "▒░", "▓▒░", "█▓▒░", "██▓▒░", "▓██▓▒░", "▒▓██▓▒░"
            ])
    } else {
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {bytes}/{total_bytes} {msg}")
            .expect("Failed to create progress bar template")
            .progress_chars("=>-")
    };

    progress_bar.set_style(style);

    let processing_complete = Arc::new(AtomicBool::new(false));
    let processing_complete_clone = processing_complete.clone();

    // Spawn a thread to update the progress bar
    let progress_thread = thread::spawn(move || {
        while !processing_complete_clone.load(Ordering::Relaxed) {
            progress_bar.tick();
            thread::sleep(Duration::from_millis(100));
        }
        progress_bar.finish_with_message("Variant processing complete");
    });

    // Process header
    let mut buffer = String::new();
    while reader.read_line(&mut buffer)? > 0 {
        if buffer.starts_with("##") {
            // Skip meta-information lines
        } else if buffer.starts_with("#CHROM") {
            validate_vcf_header(&buffer)?;
            sample_names = buffer.split_whitespace().skip(9).map(String::from).collect();
            break;
        }
        buffer.clear();
    }
    buffer.clear();

    // Set up channels for communication between threads
    let (line_sender, line_receiver) = bounded(1000);
    let (result_sender, result_receiver) = bounded(1000);

    // Spawn producer thread
    let producer_thread = thread::spawn(move || -> Result<(), VcfError> {
        let mut _line_count = 0;
        while reader.read_line(&mut buffer)? > 0 {
            line_sender.send(buffer.clone()).map_err(|_| VcfError::ChannelSend)?;
            buffer.clear();
            _line_count += 1;
        }
        drop(line_sender);
        Ok(())
    });

    // Spawn consumer threads
    let num_threads = num_cpus::get();
    let sample_names = Arc::new(sample_names);
    let consumer_threads: Vec<_> = (0..num_threads)
        .map(|_| {
            let line_receiver = line_receiver.clone();
            let result_sender = result_sender.clone();
            let chr = chr.to_string();
            let sample_names = Arc::clone(&sample_names);
            let mask_regions = mask_regions.clone();
            thread::spawn({
                let allow_regions = allow_regions.clone();
                move || -> Result<(), VcfError> {
                    while let Ok(line) = line_receiver.recv() {
                        let mut local_missing_data_info = MissingDataInfo::default();
                        let mut local_filtering_stats = FilteringStats::default();
                        match parse_variant(
                        &line,
                        &chr,
                        start,
                        end,
                        &mut local_missing_data_info,
                        &sample_names,
                        min_gq,
                        &mut local_filtering_stats,
                        allow_regions.as_ref().map(|arc| arc.as_ref()),
                        mask_regions.as_ref().map(|arc| arc.as_ref()),
                    ) {
                        Ok(variant_option) => {
                            result_sender
                                .send(Ok((
                                    variant_option,
                                    local_missing_data_info,
                                    local_filtering_stats,
                                )))
                                .map_err(|_| VcfError::ChannelSend)?;
                        }
                        Err(e) => {
                            result_sender
                                .send(Err(e))
                                .map_err(|_| VcfError::ChannelSend)?;
                        }
                    }
                }
                Ok(())
            }
        })
    })
    .collect();

    // Collector thread
    let collector_thread = thread::spawn({
        let unfiltered_variants = unfiltered_variants.clone();
        let _filtered_variants = _filtered_variants.clone();
        let missing_data_info = missing_data_info.clone();
        let _filtering_stats = _filtering_stats.clone();
        move || -> Result<(), VcfError> {
            while let Ok(result) = result_receiver.recv() {
                match result {
                    Ok((Some((variant, passes_filters)), local_missing_data_info, local_filtering_stats)) => {
                        unfiltered_variants.lock().push(variant.clone());
                        if passes_filters {
                            _filtered_variants.lock().push(variant);
                        }
                        let mut global_missing_data_info = missing_data_info.lock();
                        global_missing_data_info.total_data_points += local_missing_data_info.total_data_points;
                        global_missing_data_info.missing_data_points += local_missing_data_info.missing_data_points;
                        global_missing_data_info.positions_with_missing.extend(local_missing_data_info.positions_with_missing);
                        
                        let mut global_filtering_stats = _filtering_stats.lock();
                        global_filtering_stats.total_variants += local_filtering_stats.total_variants;
                        global_filtering_stats._filtered_variants += local_filtering_stats._filtered_variants;
                        global_filtering_stats.filtered_positions.extend(local_filtering_stats.filtered_positions);
                        global_filtering_stats.filtered_due_to_mask += local_filtering_stats.filtered_due_to_mask;
                        global_filtering_stats.filtered_due_to_allow += local_filtering_stats.filtered_due_to_allow;
                        global_filtering_stats.missing_data_variants += local_filtering_stats.missing_data_variants;
                        global_filtering_stats.low_gq_variants += local_filtering_stats.low_gq_variants;
                        global_filtering_stats.multi_allelic_variants += local_filtering_stats.multi_allelic_variants;

                        for example in local_filtering_stats.filtered_examples.iter() {
                            global_filtering_stats.add_example(example.clone());
                        }
                    },
                    Ok((None, local_missing_data_info, local_filtering_stats)) => {
                        let mut global_missing_data_info = missing_data_info.lock();
                        global_missing_data_info.total_data_points += local_missing_data_info.total_data_points;
                        global_missing_data_info.missing_data_points += local_missing_data_info.missing_data_points;
                        global_missing_data_info.positions_with_missing.extend(local_missing_data_info.positions_with_missing);
                        
                        let mut global_filtering_stats = _filtering_stats.lock();
                        global_filtering_stats.total_variants += local_filtering_stats.total_variants;
                        global_filtering_stats._filtered_variants += local_filtering_stats._filtered_variants;
                        global_filtering_stats.filtered_positions.extend(local_filtering_stats.filtered_positions);
                        global_filtering_stats.filtered_due_to_mask += local_filtering_stats.filtered_due_to_mask;
                        global_filtering_stats.filtered_due_to_allow += local_filtering_stats.filtered_due_to_allow;
                        global_filtering_stats.missing_data_variants += local_filtering_stats.missing_data_variants;
                        global_filtering_stats.low_gq_variants += local_filtering_stats.low_gq_variants;
                        global_filtering_stats.multi_allelic_variants += local_filtering_stats.multi_allelic_variants;
                        for example in local_filtering_stats.filtered_examples.iter() {
                            global_filtering_stats.add_example(example.clone());
                        }
                    },
                    Err(e) => {
                        // Record the error but continue consuming messages
                        eprintln!("Error processing variant: {}", e);
                    },
                }
            }
            Ok(())
        }
    });

    // Wait for all threads to complete
    producer_thread.join().expect("Producer thread panicked")?;
    for thread in consumer_threads {
        thread.join().expect("Consumer thread panicked")?;
    }
    drop(result_sender); // Close the result channel
    collector_thread.join().expect("Collector thread panicked")?;

    // Signal that processing is complete
    processing_complete.store(true, Ordering::Relaxed);

    // Wait for the progress thread to finish
    progress_thread.join().expect("Couldn't join progress thread");

    let final_unfiltered_variants = Arc::try_unwrap(unfiltered_variants)
        .map_err(|_| VcfError::Parse("Unfiltered variants still have multiple owners".to_string()))?
        .into_inner();
    let final_filtered_variants = Arc::try_unwrap(_filtered_variants)
        .map_err(|_| VcfError::Parse("Filtered variants still have multiple owners".to_string()))?
        .into_inner();
            
    let final_missing_data_info = Arc::try_unwrap(missing_data_info)
        .map_err(|_| VcfError::Parse("Missing data info still have multiple owners".to_string()))?
        .into_inner();
    let final_filtering_stats = Arc::try_unwrap(_filtering_stats)
        .map_err(|_| VcfError::Parse("Filtering stats still have multiple owners".to_string()))?
        .into_inner();

    let sample_names = Arc::try_unwrap(sample_names)
        .map_err(|_| VcfError::Parse("Sample names have multiple owners".to_string()))?;

    Ok((
        final_unfiltered_variants,
        final_filtered_variants,
        sample_names,
        chr_length,
        final_missing_data_info,
        final_filtering_stats,
    ))
}


// Function to validate VCF header
fn validate_vcf_header(header: &str) -> Result<(), VcfError> {
    let fields: Vec<&str> = header.split('\t').collect();
    let required_fields = vec!["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"];

    if fields.len() < required_fields.len() || fields[..required_fields.len()] != required_fields[..] {
        return Err(VcfError::InvalidVcfFormat("Invalid VCF header format".to_string()));
    }
    Ok(())
}

// Helper function to parse GFF file and extract CDS regions
fn parse_gff_file(
    gff_path: &Path,
    chr: &str,
    region_start: i64,
    region_end: i64,
) -> Result<Vec<CdsRegion>, VcfError> {
    let file = File::open(gff_path).map_err(|e| {
        VcfError::Io(io::Error::new(
            io::ErrorKind::NotFound,
            format!("GFF file not found: {:?}", e),
        ))
    })?;
    let reader = BufReader::new(file);
    let mut cds_regions = Vec::new();

    for line_result in reader.lines() {
        let line = line_result?;
        if line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 9 {
            continue;
        }
        let seqname = fields[0].trim().trim_start_matches("chr");
        if seqname != chr.trim_start_matches("chr") {
            continue;
        }
        let feature_type = fields[2];
        if feature_type != "CDS" {
            continue;
        }
        let start: i64 = fields[3]
            .parse()
            .map_err(|_| VcfError::Parse("Invalid CDS start position".to_string()))?;
        let end: i64 = fields[4]
            .parse()
            .map_err(|_| VcfError::Parse("Invalid CDS end position".to_string()))?;
        if end < region_start || start > region_end {
            continue;
        }
        cds_regions.push(CdsRegion { start, end });
    }

    Ok(cds_regions)
}

// Struct to hold CDS region information
struct CdsRegion {
    start: i64,
    end: i64,
}

// Write sequences to PHYLIP file
fn write_phylip_file(
    output_file: &str,
    hap_sequences: &HashMap<String, Vec<char>>,
) -> Result<(), VcfError> {
    let num_samples = hap_sequences.len();
    let seq_length = hap_sequences.values().next().unwrap().len();
    let file = File::create(output_file).map_err(|e| {
        VcfError::Io(io::Error::new(
            io::ErrorKind::Other,
            format!("Failed to create PHYLIP file '{}': {:?}", output_file, e),
        ))
    })?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "{} {}", num_samples, seq_length).map_err(|e| {
        VcfError::Io(io::Error::new(
            io::ErrorKind::Other,
            format!("Failed to write to PHYLIP file '{}': {:?}", output_file, e),
        ))
    })?;

    for (sample_name, seq_chars) in hap_sequences {
        let padded_name = format!("{:<10}", sample_name);
        let sequence: String = seq_chars.iter().collect();
        writeln!(writer, "{}{}", padded_name, sequence).map_err(|e| {
            VcfError::Io(io::Error::new(
                io::ErrorKind::Other,
                format!("Failed to write to PHYLIP file '{}': {:?}", output_file, e),
            ))
        })?;
    }

    Ok(())
}


// Function to parse a variant line
fn parse_variant(
    line: &str,
    chr: &str,
    start: i64,
    end: i64,
    missing_data_info: &mut MissingDataInfo,
    sample_names: &[String],
    min_gq: u16,
    _filtering_stats: &mut FilteringStats,
    allow_regions: Option<&HashMap<String, Vec<(i64, i64)>>>,
    mask_regions: Option<&HashMap<String, Vec<(i64, i64)>>>,
    position_allele_map: &mut HashMap<i64, (char, char)>,
) -> Result<Option<(Variant, bool)>, VcfError> {
    let fields: Vec<&str> = line.split('\t').collect();

    let required_fixed_fields = 9;
    if fields.len() < required_fixed_fields + sample_names.len() {
        return Err(VcfError::Parse(format!(
            "Invalid VCF line format: expected at least {} fields, found {}",
            required_fixed_fields + sample_names.len(),
            fields.len()
        )));
    }

    let vcf_chr = fields[0].trim().trim_start_matches("chr");

    if vcf_chr != chr.trim_start_matches("chr") {
        return Ok(None);
    }

    let pos: i64 = fields[1]
        .parse()
        .map_err(|_| VcfError::Parse("Invalid position".to_string()))?;
    if pos < start || pos > end {
        return Ok(None);
    }

    _filtering_stats.total_variants += 1; // DO NOT MOVE THIS LINE ABOVE THE CHECK FOR WITHIN RANGE
    // Only variants within the range get passed the collector which increments statistics.
    // For variants outside the range, the consumer thread does not send any result to the collector.
    // If this line is moved above the early return return Ok(None) in the range check, then it would increment all variants, not just those in the regions
    // This would mean that the maximum number of variants filtered could be below the maximum number of variants, in the case that there are variants outside of the ranges (which would not even get far enough to need to be filtered, but would be included in the total).

    let adjusted_pos = pos - 1; // Adjust VCF position (one-based) to zero-based

    // Check allow regions
    if let Some(allow_regions_chr) = allow_regions.and_then(|ar| ar.get(vcf_chr)) {
        if !position_in_regions(adjusted_pos, allow_regions_chr) {
            _filtering_stats._filtered_variants += 1;
            _filtering_stats.filtered_due_to_allow += 1;
            _filtering_stats.filtered_positions.insert(pos);
            _filtering_stats.add_example(format!("{}: Filtered due to allow", line.trim()));
            return Ok(None);
        }
    } else if allow_regions.is_some() {
        // If allow_regions is provided, but there are no allowed regions for this chromosome, filter it
        _filtering_stats._filtered_variants += 1;
        _filtering_stats.filtered_due_to_allow += 1;
        _filtering_stats.filtered_positions.insert(pos);
        _filtering_stats.add_example(format!("{}: Filtered due to allow", line.trim()));
        return Ok(None);
    }

    // Check mask regions
    if let Some(mask_regions_chr) = mask_regions.and_then(|mr| mr.get(vcf_chr)) {
        if position_in_regions(adjusted_pos, mask_regions_chr) {
            _filtering_stats._filtered_variants += 1;
            _filtering_stats.filtered_due_to_mask += 1;
            _filtering_stats.filtered_positions.insert(pos);
            _filtering_stats.add_example(format!("{}: Filtered due to mask", line.trim()));
            return Ok(None);
        }
    } else if mask_regions.is_some() {
        // If mask_regions is provided but there are no mask regions for this chromosome,
        // we do not filter the variant since it's not masked.
        // This is separate from the allow file behavior, which restricts anything not explicitly allowed.
        // No action needed here; we proceed with processing.
    }

    let ref_allele = fields[3].chars().next().unwrap_or('N');
    let alt_allele = fields[4].chars().next().unwrap_or('N');
    position_allele_map.insert(pos, (ref_allele, alt_allele));

    let alt_alleles: Vec<&str> = fields[4].split(',').collect();
    let is_multiallelic = alt_alleles.len() > 1;
    if is_multiallelic {
        _filtering_stats.multi_allelic_variants += 1;
        eprintln!("{}", format!("Warning: Multi-allelic site detected at position {}, which is not fully supported.", pos).yellow());
        _filtering_stats.add_example(format!("{}: Filtered due to multi-allelic variant", line.trim()));
    }

    // Parse the FORMAT field to get the indices of the subfields
    let format_fields: Vec<&str> = fields[8].split(':').collect();

    // Find the index of GQ
    let gq_index = format_fields.iter().position(|&s| s == "GQ");

    if gq_index.is_none() {
        return Err(VcfError::Parse("GQ field not found in FORMAT".to_string()));
    }

    let gq_index = gq_index.unwrap();

    let genotypes: Vec<Option<Vec<u8>>> = fields[9..].iter()
        .map(|gt| {
            missing_data_info.total_data_points += 1;
            let alleles_str = gt.split(':').next().unwrap_or(".");
            if alleles_str == "." || alleles_str == "./." || alleles_str == ".|." {
                missing_data_info.missing_data_points += 1;
                missing_data_info.positions_with_missing.insert(pos);
                return None;
            }
            let alleles = alleles_str.split(|c| c == '|' || c == '/')
                .map(|allele| allele.parse::<u8>().ok())
                .collect::<Option<Vec<u8>>>();
            if alleles.is_none() {
                missing_data_info.missing_data_points += 1;
                missing_data_info.positions_with_missing.insert(pos);
            }
            alleles
        })
        .collect();

    let mut sample_has_low_gq = false;
    let mut _num_samples_below_gq = 0;

    for gt_field in fields[9..].iter() {
        let gt_subfields: Vec<&str> = gt_field.split(':').collect();
        
        // Check if GQ index is within the subfields
        if gq_index >= gt_subfields.len() {
            return Err(VcfError::Parse(format!(
                "GQ value missing in sample genotype field at chr{}:{}",
                chr, pos
            )));
        }
        
        let gq_str = gt_subfields[gq_index];
        
        // Attempt to parse GQ value as u16
        // Parse GQ value, treating '.' or empty string as 0
        let gq_value: u16 = match gq_str {
            "." | "" => 0,
            _ => match gq_str.parse() {
                Ok(val) => val,
                Err(_) => {
                    eprintln!("Missing GQ value '{}' at {}:{}. Treating as 0.", gq_str, chr, pos);
                    0
                },
            },
        };
    
        // Check if GQ value is below the minimum threshold
        if gq_value < min_gq {
            sample_has_low_gq = true;
            _num_samples_below_gq += 1;
        }
    }

    if sample_has_low_gq {
        // Skip this variant
        _filtering_stats.low_gq_variants += 1;
        _filtering_stats._filtered_variants += 1;
        _filtering_stats.filtered_positions.insert(pos);
        _filtering_stats.add_example(format!("{}: Filtered due to low GQ", line.trim()));
    
        let has_missing_genotypes = genotypes.iter().any(|gt| gt.is_none());
        let passes_filters = !sample_has_low_gq && !has_missing_genotypes && !is_multiallelic;

        let variant = Variant {
            position: pos,
            genotypes: genotypes.clone(),
        };

        return Ok(Some((variant, passes_filters)));
    }
    
    // Do not exclude the variant; update the missing data info
    if genotypes.iter().any(|gt| gt.is_none()) {
        _filtering_stats.missing_data_variants += 1;
        _filtering_stats.add_example(format!("{}: Filtered due to missing data", line.trim()));
        // Continue processing
    }

    let has_missing_genotypes = genotypes.iter().any(|gt| gt.is_none());
    let passes_filters = !sample_has_low_gq && !has_missing_genotypes && !is_multiallelic;
    
    // Update filtering stats if variant is filtered out
    if !passes_filters {
        _filtering_stats._filtered_variants += 1;
        _filtering_stats.filtered_positions.insert(pos);
        
        if sample_has_low_gq {
            _filtering_stats.low_gq_variants += 1;
            _filtering_stats.add_example(format!("{}: Filtered due to low GQ", line.trim()));
        }
        
        if genotypes.iter().any(|gt| gt.is_none()) {
            _filtering_stats.missing_data_variants += 1;
            _filtering_stats.add_example(format!("{}: Filtered due to missing data", line.trim()));
        }
        
        if is_multiallelic {
            _filtering_stats.multi_allelic_variants += 1;
            _filtering_stats.add_example(format!("{}: Filtered due to multi-allelic variant", line.trim()));
        }
    }

    let variant = Variant {
        position: pos,
        genotypes: genotypes.clone(),
    };
    
    // Return the parsed variant and whether it passes filters
    Ok(Some((variant, passes_filters)))
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

    (0..n).into_par_iter().flat_map(|i| {
        let variants = Arc::clone(&variants);
        (i+1..n).into_par_iter().map(move |j| {
            let mut diff_count = 0;
            let mut diff_positions = Vec::new();

            for v in variants.iter() {
                if let (Some(gi), Some(gj)) = (&v.genotypes[i], &v.genotypes[j]) {
                    if gi != gj {
                        diff_count += 1;
                        diff_positions.push(v.position);
                    }
                } else {
                    // Skip positions where one or both genotypes are missing
                    continue;
                }
            }

            // Return statement should be inside the closure
            ((i, j), diff_count, diff_positions)
        }).collect::<Vec<_>>()
    }).collect()
}

fn extract_sample_id(name: &str) -> &str {
    name.rsplit('_').next().unwrap_or(name)
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
