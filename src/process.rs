use crate::stats::{ 
    count_segregating_sites,
    calculate_pairwise_differences,
    calculate_watterson_theta,
    calculate_pi,
    calculate_adjusted_sequence_length,
    calculate_inversion_allele_frequency,
};

use crate::parse::{ 
    parse_regions_file,
    parse_config_file,
    parse_region,
    find_vcf_file,
    open_vcf_reader,
    read_reference_sequence,
    parse_gtf_file,
    validate_vcf_header,
};

use clap::Parser;
use colored::*;
use flate2::read::MultiGzDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use parking_lot::Mutex;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::collections::{HashMap, HashSet};
use csv::{WriterBuilder};
use crossbeam_channel::bounded;
use std::time::Duration;
use std::sync::Arc;
use std::thread;
use prettytable::{Table, row};

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

    #[arg(long = "reference")]
    reference_path: String,

    #[arg(long = "gtf")]
    gtf_path: String,
}

// Data structures
#[derive(Debug, Clone)]
pub struct ConfigEntry {
    pub seqname: String,
    pub start: i64,
    pub end: i64,
    pub samples_unfiltered: HashMap<String, (u8, u8)>,
    pub samples_filtered: HashMap<String, (u8, u8)>,
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
pub struct Variant {
    pub position: i64,
    pub genotypes: Vec<Option<Vec<u8>>>,
}

// IN PROGRESS
#[derive(Debug, Clone)]
struct SeqInfo {
    sample_index: usize,         // The index of the sample this allele belongs to
    haplotype_group: u8,         // 0 or 1 for haplotype group
    vcf_allele: Option<u8>,      // The VCF allele value (0 or 1) (can be None)
    nucleotide: Option<u8>,      // The allele nucleotide (A, T, C, G) in u8 form (can be None)
    chromosome: String,          // Chromosome identifier
    position: i64,               // Chromosome position
    filtered: bool,              // Was this allele filtered or not
}

#[derive(Debug, Default, Clone)]
struct MissingDataInfo {
    total_data_points: usize,
    missing_data_points: usize,
    positions_with_missing: HashSet<i64>,
}

// Custom error types
#[derive(Debug)]
pub enum VcfError {
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
            &args,
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
        
        let ref_sequence = read_reference_sequence(
            &Path::new(&args.reference_path),
            chr,
            start,
            end
        )?;
        
        let cds_regions = parse_gtf_file(
            &Path::new(&args.gtf_path),
            chr,
            start,
            end
        )?;
        

        println!(
            "{}",
            format!("Processing VCF file: {}", vcf_file.display()).cyan()
        );

        // Initialize shared SeqInfo storage
        let seqinfo_storage = Arc::new(Mutex::new(Vec::new()));

        let position_allele_map = Arc::new(Mutex::new(HashMap::<i64, (char, char)>::new()));

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
            &Path::new(&args.reference_path),
            &chr,
            start,
            end,
            args.min_gq,
            mask_regions.clone(),
            allow_regions.clone(),
            Arc::clone(&seqinfo_storage), // Pass the storage
            Arc::clone(&position_allele_map),
        )?;
        
        {
            let seqinfo = seqinfo_storage.lock();
            if !seqinfo.is_empty() {
                display_seqinfo_entries(&seqinfo, 12);
            } else {
                println!("No SeqInfo entries were stored.");
            }
        }
        println!("{}", "Calculating diversity statistics...".blue());

        let seq_length = if end == i64::MAX {
            unfiltered_variants
                .last()
                .map(|v| v.position)
                .unwrap_or(0)
                .max(chr_length)
                - (start - 1)
        } else {
            end - (start - 1)
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

fn display_seqinfo_entries(seqinfo: &[SeqInfo], limit: usize) {
    // Create a buffer for the table output
    let mut output = Vec::new();
    let mut table = Table::new();
    
    // Set headers
    table.add_row(row![
        "Index", "Sample Index", "Haplotype Group", "VCF Allele", "Nucleotide", "Chromosome", "Position", "Filtered"
    ]);
    
    // Add rows
    for (i, info) in seqinfo.iter().take(limit).enumerate() {
        table.add_row(row![
            i + 1,
            info.sample_index,
            info.haplotype_group,
            info.vcf_allele.map(|a| a.to_string()).unwrap_or("-".to_string()),
            info.nucleotide.map(|n| n as char).unwrap_or('N'),
            info.chromosome,
            info.position,
            info.filtered
        ]);
    }
    
    // Render the table to our buffer
    table.print(&mut output).expect("Failed to print table to buffer");
    
    // Now print everything atomically as a single block
    let table_string = String::from_utf8(output).expect("Failed to convert table to string");
    
    // Combine all output into a single print statement
    print!("\n{}\n{}", 
           "Sample SeqInfo Entries:".green().bold(),
           table_string);
    
    // Add the count of remaining entries if any
    if seqinfo.len() > limit {
        println!("... and {} more entries.", seqinfo.len() - limit);
    }
    
    // Everything is flushed
    std::io::stdout().flush().expect("Failed to flush stdout");
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


/*
When the code calls something like:
    let filename = format!(
    "group_{}_{}_chr_{}_start_{}_end_{}_combined.phy",
    haplotype_group,
    transcript_id,
    chromosome,
    cds_start,
    cds_end
);
it creates one .phy file per combination of haplotype_group (0 or 1), transcript_id, and chromosome. This file can contain sequences from many samples, as long as their config entries say those samples’ haplotypes belong to that group.

Inside the file, each line is written by something like:
    writeln!(writer, "{}{}", padded_name, sequence);
where padded_name = format!("{:<10}", sample_name).

Now, the final sample_name is constructed with “_L” or “_R” to distinguish the left or right haplotype. Specifically, for (sample_idx, hap_idx) in haplotype_indices, the code does something like:
    sample_name = match *hap_idx {
        0 => format!("{}_L", sample_names[*sample_idx]),
        1 => format!("{}_R", sample_names[*sample_idx]),
        _ => panic!("Unexpected hap_idx"),
    };
and hap_sequences.insert(sample_name, reference_sequence);

Here, hap_idx of 0 means the sample’s left haplotype belongs to that inversion group; 1 means its right haplotype belongs. This logic comes from comparing haplotype_group (the “0 or 1” being processed) against the config file’s HashMap<String, (u8, u8)>, which might store (left_tsv, right_tsv) as (0,1) or (1,1). If the left_tsv matches haplotype_group, you push (sample_index, 0). If the right_tsv matches, you push (sample_index, 1).

Therefore, the “_L” or “_R” in the sample name is purely about left vs. right sides in the VCF and avoids collisions in naming. Meanwhile, the config’s “0” or “1” refers to which inversion group each side belongs to, not left/right in the final file name.

If a sample’s config entry says (left_tsv=0, right_tsv=1), that sample appears in group_0’s file as SampleName_L (if the left side belongs to group 0) and in group_1’s file as SampleName_R (if the right side belongs to group 1). Any side not matching the requested group is skipped.

Keep in mind that 0 or 1 in the config is about which haplotype group (e.g., reference or inverted) each side belongs to, whereas the “0|1” in the VCF refers to ref vs. alt alleles at a position. The config tells you which side (left or right) to collect into group_0 or group_1, and the VCF tells you whether that haplotype is ref or alt at each site.

Hence the files named group_0_<transcript>_chr_<...>.phy gather all haplotypes labeled as group 0, with lines like “SampleA_L” or “SampleB_R” (whichever sides matched group 0). Meanwhile, group_1_<transcript>_chr_<...>.phy holds group 1 haplotypes, labeled “SampleA_R,” “SampleB_L,” and so on, depending on each sample’s config. If your config uses 1 to mean “inversion,” then group_1_... will contain inverted haplotypes, while group_0_... contains non-inverted.
*/


fn process_variants(
    variants: &[Variant],
    sample_names: &[String],
    haplotype_group: u8,
    sample_filter: &HashMap<String, (u8, u8)>,
    region_start: i64,
    region_end: i64,
    adjusted_sequence_length: Option<i64>,
    seqinfo_storage: Arc<Mutex<Vec<SeqInfo>>>,
    position_allele_map: Arc<Mutex<HashMap<i64, (char, char)>>>,
    chromosome: String,
    is_filtered_set: bool,
    reference_sequence: &[u8],
    cds_regions: &[CdsRegion],
) -> Result<Option<(usize, f64, f64, usize)>, VcfError> {
    // Map sample names to indices
    let mut vcf_sample_id_to_index: HashMap<&str, usize> = HashMap::new();
    for (i, name) in sample_names.iter().enumerate() {
        let sample_id = extract_sample_id(name);
        vcf_sample_id_to_index.insert(sample_id, i);
    }

    // Collect haplotype indices for the specified group
    let mut haplotype_indices = Vec::new();
    for (sample_name, &(left_tsv, right_tsv)) in sample_filter.iter() {
        if let Some(&i) = vcf_sample_id_to_index.get(sample_name.as_str()) {
            if left_tsv == haplotype_group as u8 {
                haplotype_indices.push((i, 0)); // Include left haplotype
            }
            if right_tsv == haplotype_group as u8 {
                haplotype_indices.push((i, 1)); // Include right haplotype
            }
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

    // Early return if no variants
    if variants.is_empty() {
        return Ok(Some((0, 0.0, 0.0, n))); // Return zero values but valid result
    }

    // Collect alleles and compute statistics
    for variant in variants {
        if variant.position < region_start || variant.position > region_end { // Check
            continue;
        }

        let mut variant_alleles = Vec::new();

        for &(sample_idx, allele_idx) in &haplotype_indices {
            let allele = variant.genotypes.get(sample_idx)
                .and_then(|gt| gt.as_ref())
                .and_then(|alleles| alleles.get(allele_idx))
                .copied();


                // Convert VCF allele to actual nucleotide
                let nucleotide = if let Some(allele_val) = allele {
                    let map = position_allele_map.lock();
                    if let Some(&(ref_allele, alt_allele)) = map.get(&variant.position) {
                        match allele_val {
                            0 => Some(ref_allele as u8),
                            1 => Some(alt_allele as u8),
                            _ => Some(b'N')
                        }
                    } else {
                        eprintln!("Warning: No allele mapping found for position {}", variant.position);
                        Some(b'N')
                    }
                } else {
                    None
                };
                

                // Create and store SeqInfo
                let seq_info = SeqInfo {
                    sample_index: sample_idx,
                    haplotype_group,
                    vcf_allele: allele,
                    nucleotide,
                    chromosome: chromosome.clone(),
                    position: variant.position,
                    filtered: is_filtered_set, // MUST USE ACTUAL FILTERING INFO.
                    // Perhaps since different aspects are updated in different places we can update sections of SeqInfo at a time. However, need way to ID same allele each update
                };
        
                // Store SeqInfo if we have a valid nucleotide
                if let Some(allele_val) = allele {
                    let mut storage = seqinfo_storage.lock();
                    storage.push(seq_info);
                    variant_alleles.push(allele_val);
                }
        }

        // Determine if the variant is a segregating site
        let unique_alleles: HashSet<u8> = variant_alleles.iter().cloned().collect();
        if unique_alleles.len() > 1 {
            num_segsites += 1;
        }

        // Compute pairwise differences
        if !variant_alleles.is_empty() {
            for i in 0..n {
                if let Some(&allele_i) = variant_alleles.get(i) {
                    for j in (i + 1)..n {
                        if let Some(&allele_j) = variant_alleles.get(j) {
                            if allele_i != allele_j {
                                tot_pair_diff += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    let seq_length = adjusted_sequence_length.unwrap_or(region_end - region_start + 1);
    let w_theta = calculate_watterson_theta(num_segsites, n, seq_length);
    let pi = calculate_pi(tot_pair_diff, n, seq_length);

    // Process CDS regions and generate final coding sequences per transcript
    for cds in cds_regions {
        let transcript_id = &cds.transcript_id;
        let cds_min = cds.segments.iter().map(|(s, _, _, _)| *s).min().unwrap();
        let cds_max = cds.segments.iter().map(|(_, e, _, _)| *e).max().unwrap();
    
        // Map each haplotype to a combined coding sequence
        let mut combined_sequences: HashMap<String, Vec<u8>> = HashMap::new();
        for (sample_idx, hap_idx) in &haplotype_indices {
            let sample_name = match *hap_idx { 0 => format!("{}_L", sample_names[*sample_idx]), 1 => format!("{}_R", sample_names[*sample_idx]), _ => panic!("Unexpected hap_idx (not 0 or 1)!"), };
            combined_sequences.insert(sample_name, Vec::new());
        }

            // Just use the entire CDS:
            let cds_min = cds.segments.iter().map(|(s, _, _, _)| *s).min().unwrap();
            let cds_max = cds.segments.iter().map(|(_, e, _, _)| *e).max().unwrap();
            let codon_aligned_start = cds_min;
            let codon_aligned_end   = cds_max;

            let mut segment_map = Vec::new();
            {
                let mut current_length = 0;
                for &(seg_start, seg_end, strand_char, frame_val) in &cds.segments {
                    let overlap_start = seg_start;
                    let overlap_end   = seg_end;
                    let start_offset = (overlap_start - codon_aligned_start) as usize;
                    let end_offset   = (overlap_end - codon_aligned_start + 1) as usize;
                    if end_offset > reference_sequence.len() {
                        continue;
                    }

                    // For each haplotype, copy that slice into combined_sequences
                    for (sample_idx, hap_idx) in &haplotype_indices {
                        let sample_name = match *hap_idx {
                            0 => format!("{}_L", sample_names[*sample_idx]),
                            1 => format!("{}_R", sample_names[*sample_idx]),
                            _ => panic!("Unexpected hap_idx!"),
                        };
                        let seq = combined_sequences.get_mut(&sample_name).unwrap();

                        let mut segment_ref_seq = reference_sequence[start_offset..end_offset].to_vec();

                        // If minus strand, reverse+complement
                        if strand_char == '-' {
                            segment_ref_seq.reverse();
                            for b in segment_ref_seq.iter_mut() {
                                *b = match *b {
                                    b'A' | b'a' => b'T',
                                    b'T' | b't' => b'A',
                                    b'C' | b'c' => b'G',
                                    b'G' | b'g' => b'C',
                                    _ => b'N',
                                };
                            }
                        }
                        seq.extend_from_slice(&segment_ref_seq);
                    }

                    // Keep track of where this segment lands in the final sequence
                    let segment_len = end_offset - start_offset;
                    segment_map.push((overlap_start, overlap_end, current_length));
                    current_length += segment_len;
                }
            }
    
        // Apply variants to the combined coding sequences
        for variant in variants {
            if variant.position < cds_min || variant.position > cds_max {
                continue;
            }
            let mut pos_in_cds = None;
            for &(seg_s, seg_e, offset) in &segment_map {
                if variant.position >= seg_s && variant.position <= seg_e {
                    let rel = variant.position - seg_s;
                    let idx = offset + rel as usize;
                    pos_in_cds = Some(idx);
                    break;
                }
            }
    
            if let Some(pos_in_seq) = pos_in_cds {
                for (sample_idx, hap_idx) in &haplotype_indices {
                    if let Some(Some(alleles)) = variant.genotypes.get(*sample_idx) {
                        if let Some(allele) = alleles.get(*hap_idx) {
                            let sample_name = match *hap_idx { 0 => format!("{}_L", sample_names[*sample_idx]), 1 => format!("{}_R", sample_names[*sample_idx]), _ => panic!("Unexpected hap_idx (not 0 or 1)!"), };
                            if let Some(seq) = combined_sequences.get_mut(&sample_name) {
                                if pos_in_seq < seq.len() {
                                    let map = position_allele_map.lock();
                                    if let Some(&(ref_allele, alt_allele)) = map.get(&variant.position) {
                                        seq[pos_in_seq] = if *allele == 0 {
                                            ref_allele as u8
                                        } else {
                                            alt_allele as u8
                                        };
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    
        // Make sure all sequences are the same length across haplotypes
        let full_seq_lengths: Vec<usize> = combined_sequences.values().map(|v| v.len()).collect();
        if full_seq_lengths.is_empty() {
            continue;
        }
        let final_length = full_seq_lengths[0];
        if !full_seq_lengths.iter().all(|&l| l == final_length) {
            continue;
        }
    
    // Validate each haplotype's combined CDS
    let mut valid_map = HashMap::new();
    let mut invalid_count = 0;

    for (sample_name, seq_bytes) in &combined_sequences {
        match validate_coding_sequence(seq_bytes) {
            Ok(_) => {
                valid_map.insert(sample_name.clone(), seq_bytes.clone());
            }
            Err(reason) => {
                eprintln!(
                    "Skipping haplotype {} for transcript {} (group {}) on chr {}: {}",
                    sample_name, transcript_id, haplotype_group, chromosome, reason
                );
                invalid_count += 1;
            }
        }
    }

    // If every haplotype failed, skip this transcript
    if valid_map.is_empty() {
        eprintln!(
            "Skipping entire transcript {} for group {} on chr {} because all haplotypes failed validation.",
            transcript_id, haplotype_group, chromosome
        );
        continue;
    }

    // Convert to char sequences and write out a single final PHYLIP file per transcript
    let filename = format!(
        "group_{}_{}_chr_{}_start_{}_end_{}_combined.phy",
        haplotype_group,
        transcript_id,
        chromosome,
        cds_min,
        cds_max
    );
    let char_sequences: HashMap<String, Vec<char>> = combined_sequences
        .into_iter()
        .map(|(name, seq)| (name, seq.into_iter().map(|b| b as char).collect()))
        .collect();
    write_phylip_file(&filename, &char_sequences)?;
    }

    // Print the current contents before clearing
    {
        let seqinfo = seqinfo_storage.lock();
        if !seqinfo.is_empty() {
            display_seqinfo_entries(&seqinfo, 12);
        } else {
            println!("No SeqInfo entries were stored.");
        }
    }

    // Clear storage for next group
    seqinfo_storage.lock().clear();

    if is_filtered_set {
        make_sequences(
            variants,
            sample_names,
            haplotype_group,
            sample_filter,
            region_start,
            region_end,
            reference_sequence,
            cds_regions,
            position_allele_map.clone(),
            &chromosome,
        )?;
    }
    Ok(Some((num_segsites, w_theta, pi, n)))
}

/// Validate a final coding sequence. Returns `Ok(())` if valid; otherwise an error explaining why.
pub fn validate_coding_sequence(seq: &[u8]) -> Result<(), String> {
    if seq.is_empty() {
        return Err("CDS is empty".to_string());
    }
    if seq.len() % 3 != 0 {
        return Err(format!("Length {} not divisible by 3", seq.len()));
    }
    if seq.len() < 3 {
        return Err(format!("CDS length {} is too short", seq.len()));
    }

    // Check for ATG start
    let start_codon = seq[0..3]
        .iter()
        .map(|b| b.to_ascii_uppercase())
        .collect::<Vec<u8>>();
    if start_codon != b"ATG" {
        return Err(format!(
            "Does not begin with ATG (found {:?})",
            String::from_utf8_lossy(&start_codon)
        ));
    }

    // Check for internal stop codons
    let stops = [b"TAA", b"TAG", b"TGA"];
    let seq_upper: Vec<u8> = seq.iter().map(|b| b.to_ascii_uppercase()).collect();
    for i in (0..seq_upper.len()).step_by(3) {
        if i + 2 < seq_upper.len() {
            let codon = &seq_upper[i..i + 3];
            if stops.iter().any(|stop| *stop == codon) {
                return Err(format!(
                    "Internal stop codon {} at codon index {}",
                    String::from_utf8_lossy(codon),
                    i / 3
                ));
            }
        }
    }

    // Make sure only valid nucleotides [ACGTN]
    for (i, &nt) in seq_upper.iter().enumerate() {
        if !matches!(nt, b'A' | b'C' | b'G' | b'T' | b'N') {
            return Err(format!("Invalid nucleotide '{}' at position {}", nt as char, i));
        }
    }

    Ok(())
}

fn make_sequences(
    variants: &[Variant],
    sample_names: &[String],
    haplotype_group: u8,
    sample_filter: &HashMap<String, (u8, u8)>,
    region_start: i64,
    region_end: i64,
    reference_sequence: &[u8],
    cds_regions: &[CdsRegion],
    position_allele_map: Arc<Mutex<HashMap<i64, (char, char)>>>,
    chromosome: &str,
) -> Result<(), VcfError> {
    // Map sample names to indices
    let mut vcf_sample_id_to_index: HashMap<&str, usize> = HashMap::new();
    for (i, name) in sample_names.iter().enumerate() {
        let sample_id = extract_sample_id(name);
        vcf_sample_id_to_index.insert(sample_id, i);
    }

    // Collect haplotype indices for the specified group
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
        return Ok(());
    }


    // Initialize sequences for each sample haplotype with the reference sequence
    let mut hap_sequences: HashMap<String, Vec<u8>> = HashMap::new();
    for (sample_idx, hap_idx) in &haplotype_indices {
        let sample_name = match *hap_idx { 0 => format!("{}_L", sample_names[*sample_idx]), 1 => format!("{}_R", sample_names[*sample_idx]), _ => panic!("Unexpected hap_idx (not 0 or 1)!"), };
        hap_sequences.insert(sample_name, reference_sequence.to_vec());
    }

    // Apply variants to sequences
    for variant in variants {
        if variant.position >= region_start && variant.position <= region_end {
            let pos_in_seq = (variant.position - region_start) as usize;
            for (sample_idx, hap_idx) in &haplotype_indices {
                if let Some(Some(alleles)) = variant.genotypes.get(*sample_idx) {
                    if let Some(allele) = alleles.get(*hap_idx) {
                        let sample_name = match *hap_idx { 0 => format!("{}_L", sample_names[*sample_idx]), 1 => format!("{}_R", sample_names[*sample_idx]), _ => panic!("Unexpected hap_idx (not 0 or 1)!"), };
                        if let Some(seq) = hap_sequences.get_mut(&sample_name) {
                            if pos_in_seq >= seq.len() {
                                eprintln!(
                                    "Warning: Position {} is out of bounds for sequence of length {}. Skipping variant.",
                                    pos_in_seq, seq.len()
                                );
                                continue;
                            }
                            let map = position_allele_map.lock();
                            if let Some(&(ref_allele, alt_allele)) = map.get(&variant.position) {
                                seq[pos_in_seq] = if *allele == 0 {
                                    ref_allele as u8
                                } else {
                                    alt_allele as u8
                                };
                            }
                        }
                    }
                }
            }
        }
    }

    // Print batch statistics before CDS processing
    if hap_sequences.is_empty() {
        eprintln!("No haplotype sequences generated. Cannot compute batch statistics.");
    } else {
        let total_sequences = hap_sequences.len();
        let mut stop_codon_or_too_short = 0;
        let mut skipped_sequences = 0;
        let mut not_divisible_by_three = 0;
        let mut mid_sequence_stop = 0;
        let mut length_modified = 0;

        let stop_codons = ["TAA", "TAG", "TGA"];

        // Validate all sequences once before CDS processing
        for (_sample_name, sequence) in &hap_sequences {
            let sequence_str = String::from_utf8_lossy(sequence);

            if sequence.len() < 3 || !sequence_str.starts_with("ATG") {
                stop_codon_or_too_short += 1;
                skipped_sequences += 1;
                continue;
            }

            if sequence.len() % 3 != 0 {
                not_divisible_by_three += 1;
                length_modified += 1;
            }

            // Check for mid-sequence stop codons
            for i in (0..sequence.len() - 2).step_by(3) {
                let codon = &sequence_str[i..i + 3];
                if stop_codons.contains(&codon) {
                    mid_sequence_stop += 1;
                    break;
                }
            }
        }

        println!("\nBatch Statistics:");
        println!(
            "Percentage of sequences with stop codon or too short: {:.2}%",
            (stop_codon_or_too_short as f64 / total_sequences as f64) * 100.0
        );
        println!(
            "Percentage of sequences skipped: {:.2}%",
            (skipped_sequences as f64 / total_sequences as f64) * 100.0
        );
        println!(
            "Percentage of sequences not divisible by three: {:.2}%",
            (not_divisible_by_three as f64 / total_sequences as f64) * 100.0
        );
        println!(
            "Percentage of sequences with a mid-sequence stop codon: {:.2}%",
            (mid_sequence_stop as f64 / total_sequences as f64) * 100.0
        );
        println!(
            "Percentage of sequences with modified length: {:.2}%",
            (length_modified as f64 / total_sequences as f64) * 100.0
        );
    }

    // For each CDS, extract sequences and write to PHYLIP file
    for cds in cds_regions {
        let cds_start = cds.segments.iter().map(|(s, _, _, _)| *s).min().unwrap();
        let cds_end = cds.segments.iter().map(|(_, e, _, _)| *e).max().unwrap();

        // Check overlap with region
        let mut combined_cds_sequences: HashMap<String, Vec<u8>> = HashMap::new();
        for (name, original_seq) in &hap_sequences {
            combined_cds_sequences.insert(name.clone(), Vec::new());
        }

        let cds_start = std::cmp::max(cds_start - 1, region_start - 1); // 0-based.... or just cds_start??? Double check this.
        let cds_end = std::cmp::min(cds_end, region_end);
                
        // After processing all segments, write one final .phy file
        let filename = format!(
            "group_{}_{}_chr_{}_start_{}_end_{}_combined.phy",
            haplotype_group,
            cds.transcript_id,
            chromosome,
            cds_start,
            cds_end
        );
        
        // Convert combined_cds_sequences to char sequences
        let char_sequences: HashMap<String, Vec<char>> = combined_cds_sequences
            .into_iter()
            .map(|(name, seq)| (name, seq.into_iter().map(|b| b as char).collect()))
            .collect();
        
        write_phylip_file(&filename, &char_sequences)?;

        let mut combined_sequences: HashMap<String, Vec<u8>> = HashMap::new();

        // After processing all segments into combined_sequences, compute final length:
        let full_seq_lengths: Vec<usize> = combined_sequences.values().map(|v| v.len()).collect();
        if full_seq_lengths.is_empty() { continue; }
        let final_length = full_seq_lengths[0];
        // Check all are equal length
        if !full_seq_lengths.iter().all(|&l| l == final_length) {
            eprintln!("Error: Not all sequences are the same length after concatenation. Skipping.");
            continue;
        }

        // Check if length is multiple of 3
        if final_length % 3 != 0 {
            eprintln!("Warning: Skipping because final length ({}) is not divisible by 3.", final_length);
            continue;
        }

        // For each haplotype sequence, extract CDS sequence
        let mut cds_sequences: HashMap<String, Vec<u8>> = HashMap::new();
        for (sample_name, seq) in &hap_sequences {
            let start_offset = (cds_start - region_start) as usize;
            let end_offset = (cds_end - region_start) as usize; // No +1 needed for half-open intervals

            if end_offset > seq.len() {
                eprintln!(
                    "Warning: CDS end offset {} exceeds sequence length {} for sample {}. Skipping CDS.",
                    end_offset, seq.len(), sample_name
                );
                continue;
            }

            let cds_seq = seq[start_offset..end_offset].to_vec();
            cds_sequences.insert(sample_name.clone(), cds_seq);
        }

        let cds_start = cds.segments.iter().map(|(s, _, _, _)| *s).min().unwrap();
        let cds_end = cds.segments.iter().map(|(_, e, _, _)| *e).max().unwrap();

        if cds_sequences.is_empty() {
            eprintln!(
                "No CDS sequences generated for CDS region {}-{}. Skipping PHYLIP file writing.",
                cds_start, cds_end
            );
            continue;
        }

        // Write sequences to PHYLIP file
        let filename = format!(
            "group_{}_{}_chr_{}_start_{}_end_{}.phy",
            haplotype_group,
            cds.transcript_id,
            chromosome,
            cds_start,
            cds_end
        );

        
        let filename = format!(
            "group_{}_{}_chr_{}_start_{}_end_{}_combined.phy",
            haplotype_group,
            cds.transcript_id,
            chromosome,
            cds_start,
            cds_end
        );
        
        // Convert combined_sequences to char sequences
        let char_sequences: HashMap<String, Vec<char>> = combined_sequences
            .into_iter()
            .map(|(name, seq)| (name, seq.into_iter().map(|b| b as char).collect()))
            .collect();
        
        write_phylip_file(&filename, &char_sequences)?;
    }

    Ok(())
}

fn process_config_entries(
    config_entries: &[ConfigEntry],
    vcf_folder: &str,
    output_file: &Path,
    min_gq: u16,
    mask: Option<Arc<HashMap<String, Vec<(i64, i64)>>>>,
    allow: Option<Arc<HashMap<String, Vec<(i64, i64)>>>>,
    args: &Args,
) -> Result<(), VcfError> {
    // Initialize shared SeqInfo storage
    let seqinfo_storage = Arc::new(Mutex::new(Vec::new()));
    
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

    let position_allele_map = Arc::new(Mutex::new(HashMap::<i64, (char, char)>::new()));

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

        // Read reference sequence and CDS regions once per chromosome
        let ref_sequence = read_reference_sequence(
            &Path::new(&args.reference_path),
            &chr,
            entries.iter().map(|e| e.start).min().unwrap_or(0),
            entries.iter().map(|e| e.end).max().unwrap_or(i64::MAX)
        )?;
        
        let cds_regions = parse_gtf_file(
            &Path::new(&args.gtf_path),
            &chr,
            entries.iter().map(|e| e.start).min().unwrap_or(0),
            entries.iter().map(|e| e.end).max().unwrap_or(i64::MAX)
        )?;
        
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
            &Path::new(&args.reference_path),
            &chr,
            min_start,
            max_end,
            min_gq,
            mask.clone(),
            allow.clone(),
            Arc::clone(&seqinfo_storage),
            Arc::clone(&position_allele_map),
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
            println!("Processing region {}-{} with {} variants", 
                    entry.start, entry.end, unfiltered_variants.len());
            
            let variants_in_region: Vec<_> = unfiltered_variants.iter()
                .filter(|v| v.position >= entry.start && v.position <= entry.end)
                .cloned()
                .collect();
            println!("Found {} variants in region", variants_in_region.len());

            // Add these lines before calling process_variants
            let ref_sequence = read_reference_sequence(
                &Path::new(&args.reference_path),
                &chr,
                entry.start,
                entry.end
            )?;
            
            let cds_regions = parse_gtf_file(
                &Path::new(&args.gtf_path),
                &chr,
                entry.start,
                entry.end
            )?;
            
            let (num_segsites_0, w_theta_0, pi_0, n_hap_0_no_filter) =
                match process_variants(
                    &variants_in_region,
                    &sample_names,
                    0,
                    &entry.samples_unfiltered,
                    entry.start,
                    entry.end,
                    None,
                    Arc::clone(&seqinfo_storage),
                    Arc::clone(&position_allele_map),
                    entry.seqname.clone(),
                    false,  // unfiltered variants
                    &ref_sequence,
                    &cds_regions,
                )? {
                    Some(values) => values,
                    None => continue, // Skip writing this record
                };

            let ref_sequence = read_reference_sequence(
                &Path::new(&args.reference_path),
                &chr,
                entry.start,
                entry.end
            )?;
            
            let cds_regions = parse_gtf_file(
                &Path::new(&args.gtf_path),
                &chr,
                entry.start,
                entry.end
            )?;

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
                    Arc::clone(&seqinfo_storage),
                    Arc::clone(&position_allele_map),
                    entry.seqname.clone(),
                    false,  // unfiltered variants
                    &ref_sequence,
                    &cds_regions,
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
                    Arc::clone(&seqinfo_storage),
                    Arc::clone(&position_allele_map),
                    entry.seqname.clone(),
                    true,  // filtered variants
                    &ref_sequence,
                    &cds_regions,
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
                    Arc::clone(&seqinfo_storage),
                    Arc::clone(&position_allele_map),
                    entry.seqname.clone(),
                    true,  // filtered variants
                    &ref_sequence,
                    &cds_regions,
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


// Function to process a VCF file
fn process_vcf(
    file: &Path,
    reference_path: &Path,
    chr: &str,
    start: i64,
    end: i64,
    min_gq: u16,
    mask_regions: Option<Arc<HashMap<String, Vec<(i64, i64)>>>>,
    allow_regions: Option<Arc<HashMap<String, Vec<(i64, i64)>>>>,
    seqinfo_storage: Arc<Mutex<Vec<SeqInfo>>>,
    position_allele_map: Arc<Mutex<HashMap<i64, (char, char)>>>,
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
    let chr_length = {
        let fasta_reader = bio::io::fasta::IndexedReader::from_file(&reference_path)
            .map_err(|e| VcfError::Io(io::Error::new(io::ErrorKind::Other, e.to_string())))?;
        // Create an owned copy of the sequences
        let sequences = fasta_reader.index.sequences().to_vec();
        let seq_info = sequences.iter()
            .find(|seq| seq.name == chr || seq.name == format!("chr{}", chr))
            .ok_or_else(|| VcfError::Parse(format!("Chromosome {} not found in reference", chr)))?;
        seq_info.len as i64
    };

    // Existing unfiltered and filtered variants storage
    let unfiltered_variants = Arc::new(Mutex::new(Vec::new()));
    let filtered_variants = Arc::new(Mutex::new(Vec::new()));

    // Existing missing data and filtering stats
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
            let position_allele_map = Arc::clone(&position_allele_map);
            
            thread::spawn({
                let allow_regions = allow_regions.clone();
                move || -> Result<(), VcfError> {
                    while let Ok(line) = line_receiver.recv() {
                        let mut local_missing_data_info = MissingDataInfo::default();
                        let mut local_filtering_stats = FilteringStats::default();
                        
                        match process_variant(
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
                            &position_allele_map,
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
        let filtered_variants = Arc::new(Mutex::new(Vec::new())); // Or let filtered_variants = filtered_variants.clone();?
        let missing_data_info = missing_data_info.clone();
        let _filtering_stats = _filtering_stats.clone();
        move || -> Result<(), VcfError> {
            while let Ok(result) = result_receiver.recv() {
                match result {
                    Ok((Some((variant, passes_filters)), local_missing_data_info, local_filtering_stats)) => {
                        unfiltered_variants.lock().push(variant.clone());
                        if passes_filters {
                            filtered_variants.lock().push(variant);
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
    // Signal completion before joining collector
    processing_complete.store(true, Ordering::Relaxed);
    
    // All consumers must have finished and dropped their Arc references
    drop(result_sender);
    
    // Now join collector thread
    collector_thread.join().expect("Collector thread panicked")?;

    // Wait for the progress thread to finish
    progress_thread.join().expect("Couldn't join progress thread");

    {
        let seqinfo = seqinfo_storage.lock();
        if !seqinfo.is_empty() {
            display_seqinfo_entries(&seqinfo, 12);
        } else {
            println!("No SeqInfo entries were stored.");
        }
    }
    
    let final_unfiltered_variants = Arc::try_unwrap(unfiltered_variants)
        .map_err(|_| VcfError::Parse("Unfiltered variants still have multiple owners".to_string()))?
        .into_inner();
    let final_filtered_variants = Arc::try_unwrap(filtered_variants)
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


// Struct to hold CDS region information
pub struct CdsRegion {
    pub transcript_id: String,
    // Store (start, end, strand_char, frame)
    pub segments: Vec<(i64, i64, char, i64)>,
}


// IN PROGRESS
// Write sequences to PHYLIP file
fn write_phylip_file(
    output_file: &str,
    hap_sequences: &HashMap<String, Vec<char>>,
) -> Result<(), VcfError> {
    let file = File::create(output_file).map_err(|e| {
        VcfError::Io(io::Error::new(
            io::ErrorKind::Other,
            format!("Failed to create PHYLIP file '{}': {:?}", output_file, e),
        ))
    })?;
    let mut writer = BufWriter::new(file);

    // Process and write each sequence
    for (sample_name, seq_chars) in hap_sequences {
        let padded_name = format!("{:<10}", sample_name);
        let sequence: String = seq_chars.iter().collect();
        
        // Write sequence to PHYLIP file
        writeln!(writer, "{}{}", padded_name, sequence).map_err(|e| {
            VcfError::Io(io::Error::new(
                io::ErrorKind::Other,
                format!("Failed to write to PHYLIP file '{}': {:?}", output_file, e),
            ))
        })?;
    }

    writer.flush().map_err(|e| {
        VcfError::Io(io::Error::new(
            io::ErrorKind::Other,
            format!("Failed to flush PHYLIP file '{}': {:?}", output_file, e),
        ))
    })?;

    println!("PHYLIP file '{}' written successfully.", output_file);
    Ok(())
}


// Function to parse a variant line
fn process_variant(
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
    position_allele_map: &Mutex<HashMap<i64, (char, char)>>,
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

    // Store reference and alternate alleles
    if !fields[3].is_empty() && !fields[4].is_empty() {
        let ref_allele = fields[3].chars().next().unwrap_or('N');
        let alt_allele = fields[4].chars().next().unwrap_or('N');
        position_allele_map.lock().insert(pos, (ref_allele, alt_allele));
    }

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

fn extract_sample_id(name: &str) -> &str {
    name.rsplit('_').next().unwrap_or(name)
}
