use crate::stats::{
    calculate_adjusted_sequence_length, calculate_fst_wc_csv_populations,
    calculate_fst_wc_haplotype_groups, calculate_hudson_fst_for_pair_with_sites,
    calculate_inversion_allele_frequency, calculate_per_site_diversity, calculate_pi,
    calculate_watterson_theta, HudsonFSTOutcome, PopulationContext, PopulationId, SiteDiversity,
};

use crate::parse::{
    find_vcf_file, open_vcf_reader, parse_gtf_file, read_reference_sequence, validate_vcf_header,
};

use crate::progress::{
    create_spinner, create_vcf_progress, display_status_box, finish_entry_progress,
    finish_step_progress, finish_variant_progress, init_entry_progress, init_step_progress,
    init_variant_progress, log, set_stage, update_entry_progress, update_step_progress,
    update_variant_progress, LogLevel, ProcessingStage, StatusBox,
};

use crate::transcripts::{
    filter_and_log_transcripts, make_sequences, write_phylip_file, TranscriptAnnotationCDS,
};

use clap::Parser;
use colored::*;
use crossbeam_channel::bounded;
use csv::WriterBuilder;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use prettytable::{row, Table};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use tempfile::TempDir;

pub static TEMP_DIR: Lazy<Mutex<Option<TempDir>>> = Lazy::new(|| Mutex::new(None));

pub fn create_temp_dir() -> Result<TempDir, VcfError> {
    let ramdisk_path = std::env::var("RAMDISK_PATH").unwrap_or_else(|_| "/dev/shm".to_string());
    let temp_dir = match TempDir::new_in(&ramdisk_path) {
        Ok(dir) => dir,
        Err(_) => TempDir::new().map_err(|e| VcfError::Io(e))?,
    };
    Ok(temp_dir)
}

// Define command-line arguments using clap
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Folder containing VCF files
    #[arg(short, long = "vcf_folder")]
    pub vcf_folder: String,

    /// Chromosome to process
    #[arg(short, long = "chr")]
    pub chr: Option<String>,

    /// Region to process (start-end)
    #[arg(short, long = "region")]
    pub region: Option<String>,

    /// Configuration file
    #[arg(long = "config_file")]
    pub config_file: Option<String>,

    /// Output file
    #[arg(short, long = "output_file")]
    pub output_file: Option<String>,

    /// Minimum genotype quality
    #[arg(long = "min_gq", default_value = "30")]
    pub min_gq: u16,

    /// Mask file (regions to exclude)
    #[arg(long = "mask_file")]
    pub mask_file: Option<String>,

    /// Allow file (regions to include)
    #[arg(long = "allow_file")]
    pub allow_file: Option<String>,

    /// Reference genome .fa file
    #[arg(long = "reference")]
    pub reference_path: String,

    /// GTF or GFF
    #[arg(long = "gtf")]
    pub gtf_path: String,

    /// Enable PCA analysis on all haplotypes
    #[arg(long = "pca", help = "Perform PCA analysis on all haplotypes")]
    pub enable_pca: bool,

    /// Number of principal components to compute
    #[arg(
        long = "pca_components",
        default_value = "10",
        help = "Number of principal components to compute"
    )]
    pub pca_components: usize,

    /// Output file for PCA results
    #[arg(
        long = "pca_output",
        default_value = "pca_results.tsv",
        help = "Output file for PCA results"
    )]
    pub pca_output: String,

    /// Enable FST calculation
    #[arg(long = "fst", help = "Calculate FST")]
    pub enable_fst: bool,

    /// Path to CSV file defining population groups for FST calculation
    #[arg(
        long = "fst_populations",
        help = "Path to CSV file defining population groups for FST calculation"
    )]
    pub fst_populations: Option<String>,
}

/// ZeroBasedHalfOpen represents a half-open interval [start..end).
/// This struct is also used for slicing references safely.
#[derive(Debug, Clone, Copy)]
pub struct ZeroBasedPosition(pub i64);

impl ZeroBasedPosition {
    /// Converts the zero-based coordinate into a 1-based inclusive coordinate.
    pub fn to_one_based(self) -> i64 {
        self.0 + 1
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ZeroBasedHalfOpen {
    pub start: usize,
    pub end: usize,
}

#[inline]
fn clamp_nonnegative_i64_to_usize(value: i64) -> usize {
    if value <= 0 {
        0
    } else {
        let value_u128 = value as u128;
        let max_u128 = usize::MAX as u128;
        if value_u128 > max_u128 {
            usize::MAX
        } else {
            value as usize
        }
    }
}

#[inline]
fn clamp_usize_to_i64(value: usize) -> i64 {
    let value_u128 = value as u128;
    let max_i64_u128 = i64::MAX as u128;
    if value_u128 > max_i64_u128 {
        i64::MAX
    } else {
        value as i64
    }
}

impl ZeroBasedHalfOpen {
    /// Creates a new half-open interval from 1-based inclusive coordinates.
    /// This is for data that uses 1-based inclusive.
    pub fn from_1based_inclusive(start_inclusive: i64, end_inclusive: i64) -> Self {
        let mut adjusted_start = start_inclusive;
        if adjusted_start < 1 {
            adjusted_start = 1;
        }
        let mut adjusted_end = end_inclusive;
        if adjusted_end < adjusted_start {
            adjusted_end = adjusted_start;
        }
        ZeroBasedHalfOpen {
            start: (adjusted_start - 1) as usize,
            end: adjusted_end as usize,
        }
    }

    /// Creates a new half-open interval from 0-based inclusive coordinates.
    /// This converts [start..end] inclusive into [start..end+1) half-open.
    pub fn from_0based_inclusive(start_inclusive: i64, end_inclusive: i64) -> Self {
        let adjusted_start = start_inclusive.max(0);
        let adjusted_end = if end_inclusive < adjusted_start {
            adjusted_start
        } else {
            end_inclusive.saturating_add(1).max(adjusted_start)
        };

        ZeroBasedHalfOpen {
            start: clamp_nonnegative_i64_to_usize(adjusted_start),
            end: clamp_nonnegative_i64_to_usize(adjusted_end),
        }
    }

    /// Creates a zero-length half-open interval representing a single position p (0-based).
    pub fn from_0based_point(p: i64) -> Self {
        let start = p.max(0) as usize;
        ZeroBasedHalfOpen {
            start,
            end: start + 1,
        }
    }

    /// Returns the length of this half-open interval.
    pub fn len(&self) -> usize {
        if self.end > self.start {
            self.end - self.start
        } else {
            0
        }
    }

    /// Returns a slice of `seq` corresponding to this interval.
    /// This will panic if `end` exceeds `seq.len()`.
    pub fn slice<'a>(&self, seq: &'a [u8]) -> &'a [u8] {
        &seq[self.start..self.end]
    }

    /// Returns Some(overlap) if this interval intersects with `other`, or None if no overlap.
    pub fn intersect(&self, other: &ZeroBasedHalfOpen) -> Option<ZeroBasedHalfOpen> {
        let start = self.start.max(other.start);
        let end = self.end.min(other.end);
        if start < end {
            Some(ZeroBasedHalfOpen { start, end })
        } else {
            None
        }
    }

    /// Returns true if `pos` is in [start..end).
    pub fn contains(&self, pos: ZeroBasedPosition) -> bool {
        let p = pos.0 as usize;
        p >= self.start && p < self.end
    }

    /// Returns the 1-based inclusive start coordinate of this interval.
    pub fn start_1based_inclusive(&self) -> i64 {
        self.start as i64 + 1
    }

    /// Returns the 0-based inclusive end coordinate of this 0-based half-open interval [self.start, self.end).
    /// For a non-empty interval [S_0, E_0), this returns E_0 - 1.
    /// For an empty interval where self.end <= self.start (e.g., [0,0) or [5,2) ),
    /// this will result in an end coordinate that is less than self.start,
    /// correctly defining an empty or invalid 0-based inclusive interval [self.start, result].
    pub fn get_0based_inclusive_end_coord(&self) -> i64 {
        // This calculation means that if self.end <= self.start (empty interval),
        // the resulting inclusive end will be less than self.start, making the
        // inclusive interval [self.start, (self.end - 1)] also empty/invalid.
        // E.g., for [0,0), returns -1. For [5,5), returns 4.
        (self.end as i64) - 1
    }

    /// Returns the coordinate value that serves as the 1-based inclusive end
    /// of this 0-based half-open interval [self.start, self.end).
    /// For an interval like [S_0, E_0) (0-based half-open), the corresponding
    /// 1-based inclusive interval is [S_0 + 1, E_0]. This function returns E_0.
    pub fn get_1based_inclusive_end_coord(&self) -> i64 {
        self.end as i64
    }

    /// Returns a tuple (1-based inclusive start, 1-based inclusive end)
    /// representing this 0-based half-open interval.
    /// For an interval like [S_0, E_0) (0-based half-open), this returns (S_0 + 1, E_0).
    pub fn to_1based_inclusive_tuple(&self) -> (i64, i64) {
        (
            self.start_1based_inclusive(),
            self.get_1based_inclusive_end_coord(),
        )
    }

    /// Returns 1-based position of `pos` if inside [start..end), else None.
    // HALF-OPEN. Make clear later.
    pub fn relative_position_1based(&self, pos: i64) -> Option<usize> {
        let p = pos as usize;
        if p >= self.start && p < self.end {
            Some(p - self.start + 1)
        } else {
            None
        }
    }

    /// self is the half-open interval [start..end). self.start is the inclusive lower
    /// bound and self.end is the exclusive upper bound.
    /// It treats the input pos as 1-based inclusive and converts it to 0-based.
    /// Then it checks if the converted position is in [start..end). If so, it
    /// returns the offset plus 1 as the relative 1-based position.
    pub fn relative_position_1based_inclusive(&self, pos: i64) -> Option<usize> {
        let p = (pos - 1) as usize;
        if p >= self.start && p < self.end {
            Some(p - self.start + 1)
        } else {
            None
        }
    }

    // Query region
    pub fn to_zero_based_inclusive(&self) -> ZeroBasedInclusive {
        let start_i64 = clamp_usize_to_i64(self.start);
        let end_i64 = if self.end == 0 {
            -1
        } else {
            let inclusive_end = self.end - 1;
            clamp_usize_to_i64(inclusive_end)
        };

        ZeroBasedInclusive {
            start: start_i64,
            end: end_i64,
        }
    }

    // Make the names e.g. zero vs. 0 consistent later

    /// Creates a new half-open interval from 0-based half-open coordinates.
    /// Takes a start (inclusive) and end (exclusive) as-is, assuming they are already 0-based.
    pub fn from_0based_half_open(start: i64, end: i64) -> Self {
        ZeroBasedHalfOpen {
            start: start as usize,
            end: end as usize,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ZeroBasedInclusive {
    pub start: i64,
    pub end: i64,
}

impl From<ZeroBasedInclusive> for QueryRegion {
    fn from(interval: ZeroBasedInclusive) -> Self {
        QueryRegion {
            start: interval.start,
            end: interval.end,
        }
    }
}

/// A 1-based VCF position. Guaranteed >= 1, no runtime overhead.
#[derive(Debug, Clone, Copy)]
pub struct OneBasedPosition(i64);

impl OneBasedPosition {
    /// Creates a new 1-based position, returning an error if `val < 1`.
    pub fn new(val: i64) -> Result<Self, VcfError> {
        if val < 1 {
            Err(VcfError::Parse(format!("Invalid 1-based pos: {}", val)))
        } else {
            Ok(Self(val))
        }
    }

    /// Converts to zero-based i64. This is where we do `-1`.
    pub fn zero_based(self) -> i64 {
        self.0 - 1
    }
}

/// Represents the side of a haplotype (left or right) for a sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HaplotypeSide {
    Left,  // The left haplotype of a diploid sample
    Right, // The right haplotype of a diploid sample
}

// Data structures
#[derive(Debug, Clone)]
pub struct ConfigEntry {
    pub seqname: String,
    pub interval: ZeroBasedHalfOpen,
    pub samples_unfiltered: HashMap<String, (u8, u8)>,
    pub samples_filtered: HashMap<String, (u8, u8)>,
}

#[derive(Debug, Default, Clone)]
pub struct FilteringStats {
    pub total_variants: usize,
    pub _filtered_variants: usize,
    pub filtered_due_to_mask: usize,
    pub filtered_due_to_allow: usize,
    pub filtered_positions: HashSet<i64>,
    pub missing_data_variants: usize,
    pub low_gq_variants: usize,
    pub multi_allelic_variants: usize,
    pub filtered_examples: Vec<String>,
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

#[derive(Debug, Clone)]
pub struct SeqInfo {
    pub sample_index: usize,    // The index of the sample this allele belongs to
    pub haplotype_group: u8,    // 0 or 1 for haplotype group
    pub vcf_allele: Option<u8>, // The VCF allele value (0 or 1) (can be None)
    pub nucleotide: Option<u8>, // The allele nucleotide (A, T, C, G) in u8 form (can be None)
    pub chromosome: String,     // Chromosome identifier
    pub position: i64,          // Chromosome position
    pub filtered: bool,         // Was this allele filtered or not
}

#[derive(Debug, Default, Clone)]
pub struct MissingDataInfo {
    pub total_data_points: usize,
    pub missing_data_points: usize,
    pub positions_with_missing: HashSet<i64>,
}

#[derive(Debug, Clone, Copy)]
/// The user query region for statistics.
/// This region is inclusive of [start..end] positions in 0-based coordinates.
pub struct QueryRegion {
    /// Inclusive 0-based start position
    pub start: i64,
    /// Inclusive 0-based end position
    pub end: i64,
}

impl QueryRegion {
    /// Returns true if the given position lies in [start..end].
    pub fn contains(&self, pos: i64) -> bool {
        pos >= self.start && pos <= self.end
    }

    /// Returns the number of positions in this 0-based inclusive range [start..end].
    pub fn len(&self) -> i64 {
        // self.start and self.end are 0-based inclusive.
        // To calculate length using ZeroBasedHalfOpen, convert to 0-based half-open [start, end+1).
        // The length of [S0, E0) is E0 - S0.
        // So, for [self.start, self.end+1), the length is (self.end + 1) - self.start.
        if self.start > self.end {
            // Catches invalid ranges where start is after end.
            0
        } else {
            ZeroBasedHalfOpen::from_0based_inclusive(self.start, self.end).len() as i64
        }
    }
}

/// Holds all the output columns for writing one row in the CSV.
#[derive(Debug, Clone)]
struct CsvRowData {
    seqname: String,
    region_start: i64, // 1-based inclusive start of the processed region
    region_end: i64,   // 1-based inclusive end of the processed region
    seq_len_0: i64,
    seq_len_1: i64,
    seq_len_adj_0: i64,
    seq_len_adj_1: i64,
    seg_sites_0: usize,
    seg_sites_1: usize,
    w_theta_0: f64,
    w_theta_1: f64,
    pi_0: f64,
    pi_1: f64,
    seg_sites_0_f: usize,
    seg_sites_1_f: usize,
    w_theta_0_f: f64,
    w_theta_1_f: f64,
    pi_0_f: f64,
    pi_1_f: f64,
    n_hap_0_unf: usize,
    n_hap_1_unf: usize,
    n_hap_0_f: usize,
    n_hap_1_f: usize,
    inv_freq_no_filter: f64,
    inv_freq_filter: f64,
    // Weir & Cockerham FST components for haplotype groups (e.g., 0 vs 1)
    haplotype_overall_fst_wc: Option<f64>,
    haplotype_between_pop_variance_wc: Option<f64>, // Component 'A' from W&C
    haplotype_within_pop_variance_wc: Option<f64>,  // Component 'B' from W&C
    haplotype_num_informative_sites_wc: Option<usize>,
    // Hudson FST components for haplotype groups 0 vs 1
    hudson_fst_hap_group_0v1: Option<f64>,
    hudson_dxy_hap_group_0v1: Option<f64>,
    hudson_pi_hap_group_0: Option<f64>,
    hudson_pi_hap_group_1: Option<f64>,
    hudson_pi_avg_hap_group_0v1: Option<f64>,
    // Fields for population FST from CSV have been removed as per request to only include haplotype group (0vs1) info here.
    // The full FstWcResults for CSV populations are still processed for other outputs (e.g., .falsta), but not summarized in this main CSV row.
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

impl From<csv::Error> for VcfError {
    fn from(e: csv::Error) -> Self {
        VcfError::Parse(format!("CSV error: {}", e))
    }
}

pub fn display_seqinfo_entries(seqinfo: &[SeqInfo], limit: usize) {
    // Create a buffer for the table output
    let mut output = Vec::new();
    let mut table = Table::new();

    // Set headers
    table.add_row(row![
        "Index",
        "Sample Index",
        "Haplotype Group",
        "VCF Allele",
        "Nucleotide",
        "Chromosome",
        "Position",
        "Filtered"
    ]);

    // Add rows
    for (i, info) in seqinfo.iter().take(limit).enumerate() {
        table.add_row(row![
            i + 1,
            info.sample_index,
            info.haplotype_group,
            info.vcf_allele
                .map(|a| a.to_string())
                .unwrap_or("-".to_string()),
            info.nucleotide.map(|n| n as char).unwrap_or('N'),
            info.chromosome,
            info.position,
            info.filtered
        ]);
    }

    // Render the table to our buffer
    table
        .print(&mut output)
        .expect("Failed to print table to buffer");

    // Now print everything atomically as a single block
    let table_string = String::from_utf8(output).expect("Failed to convert table to string");

    // Combine all output into a single print statement
    print!(
        "\n{}\n{}",
        "Sample SeqInfo Entries:".green().bold(),
        table_string
    );

    // Add the count of remaining entries if any
    if seqinfo.len() > limit {
        println!("... and {} more entries.", seqinfo.len() - limit);
    }

    // Everything is flushed
    std::io::stdout().flush().expect("Failed to flush stdout");
}

// Function to check if a position is within any of the regions
fn position_in_regions(pos: i64, regions: &[(i64, i64)]) -> bool {
    // `pos` is zero-based and regions are half-open [start, end)
    // Use a linear scan to avoid assumptions about sorting or overlap
    regions
        .iter()
        .any(|&(start, end)| pos >= start && pos < end)
}

/*
When the code calls something like:
        let filename = format!(
            "group_{}_{}_chr_{}_start_{}_end_{}_combined.phy",
            haplotype_group,
            cds.transcript_id,
            chromosome,
            transcript_cds_start,
            transcript_cds_end
        );
);
it creates one .phy file per combination of haplotype_group (0 or 1), transcript_id, and chromosome. This file can contain sequences from many samples, as long as their config entries say those samples’ haplotypes belong to that group.

Inside the file, each line is written by something like:
    writeln!(writer, "{}{}", padded_name, sequence);
where padded_name = format!("{:<10}", sample_name).

Now, the final sample_name is constructed with “_L” or “_R” to distinguish the left or right haplotype.

Here, hap_idx of 0 means the sample’s left haplotype belongs to that inversion group; 1 means its right haplotype belongs. This logic comes from comparing haplotype_group (the “0 or 1” being processed) against the config file’s HashMap<String, (u8, u8)>, which might store (left_tsv, right_tsv) as (0,1) or (1,1). If the left_tsv matches haplotype_group, you push (sample_index, 0). If the right_tsv matches, you push (sample_index, 1).

Therefore, the “_L” or “_R” in the sample name is purely about left vs. right sides in the VCF and avoids collisions in naming. Meanwhile, the config’s “0” or “1” refers to which inversion group each side belongs to, not left/right in the final file name.

If a sample’s config entry says (left_tsv=0, right_tsv=1), that sample appears in group_0’s file as SampleName_L (if the left side belongs to group 0) and in group_1’s file as SampleName_R (if the right side belongs to group 1). Any side not matching the requested group is skipped.

Keep in mind that 0 or 1 in the config is about which haplotype group (e.g., reference or inverted) each side belongs to, whereas the “0|1” in the VCF refers to ref vs. alt alleles at a position. The config tells you which side (left or right) to collect into group_0 or group_1, and the VCF tells you whether that haplotype is ref or alt at each site.

Hence the files named group_0_<transcript>_chr_<...>.phy gather all haplotypes labeled as group 0, with lines like “SampleA_L” or “SampleB_R” (whichever sides matched group 0). Meanwhile, group_1_<transcript>_chr_<...>.phy holds group 1 haplotypes, labeled “SampleA_R,” “SampleB_L,” and so on, depending on each sample’s config. If your config uses 1 to mean “inversion,” then group_1_... will contain inverted haplotypes, while group_0_... contains non-inverted.
*/

pub fn process_variants(
    variants: &[Variant],
    sample_names: &[String],
    haplotype_group: u8,
    sample_filter: &HashMap<String, (u8, u8)>,
    inversion_interval: ZeroBasedHalfOpen,
    extended_region: ZeroBasedHalfOpen,
    adjusted_sequence_length: Option<i64>,
    position_allele_map: Arc<Mutex<HashMap<i64, (char, char)>>>,
    chromosome: String,
    is_filtered_set: bool,
    reference_sequence: &[u8],
    cds_regions: &[TranscriptAnnotationCDS],
) -> Result<Option<(usize, f64, f64, usize, Vec<SiteDiversity>)>, VcfError> {
    set_stage(ProcessingStage::VariantAnalysis);

    let group_type = if is_filtered_set {
        "filtered"
    } else {
        "unfiltered"
    };
    log(
        LogLevel::Info,
        &format!(
            "Processing {} variants for group {} in {}:{}-{}",
            variants.len(),
            haplotype_group,
            chromosome,
            inversion_interval.start,
            inversion_interval.end
        ),
    );

    // Map sample names to indices
    init_step_progress(&format!("Mapping samples for group {}", haplotype_group), 3);

    let mut index_map = HashMap::new();
    for (sample_index, name) in sample_names.iter().enumerate() {
        let trimmed_id = name.rsplit('_').next().unwrap_or(name);
        index_map.insert(trimmed_id, sample_index);
    }

    // List of haplotype indices and their sides (left or right) for the given haplotype_group
    let mut group_haps: Vec<(usize, HaplotypeSide)> = Vec::new();
    let mut missing_samples = Vec::new();
    for (config_sample_name, &(left_side, right_side)) in sample_filter {
        match index_map.get(config_sample_name.as_str()) {
            Some(&mapped_index) => {
                if left_side == haplotype_group {
                    group_haps.push((mapped_index, HaplotypeSide::Left));
                }
                if right_side == haplotype_group {
                    group_haps.push((mapped_index, HaplotypeSide::Right));
                }
            }
            None => {
                missing_samples.push(config_sample_name.clone());
                log(
                    LogLevel::Warning,
                    &format!(
                        "Sample '{}' from config not found in VCF - skipping",
                        config_sample_name
                    ),
                );
            }
        }
    }

    if !missing_samples.is_empty() {
        log(
            LogLevel::Warning,
            &format!(
                "Missing {} samples for chromosome {}: {}",
                missing_samples.len(),
                chromosome,
                missing_samples.join(", ")
            ),
        );
    }

    if group_haps.is_empty() {
        log(
            LogLevel::Warning,
            &format!("No haplotypes found for group {}", haplotype_group),
        );
        finish_step_progress("No haplotypes found");
        return Ok(None);
    }

    update_step_progress(
        1,
        &format!(
            "Found {} haplotypes for group {}",
            group_haps.len(),
            haplotype_group
        ),
    );

    // Count segregating sites
    let mut region_segsites = 0;
    let region_hap_count = group_haps.len();

    if variants.is_empty() {
        log(
            LogLevel::Info,
            &format!(
                "No variants found for {}:{}-{} in group {}",
                chromosome, inversion_interval.start, inversion_interval.end, haplotype_group
            ),
        );
        finish_step_progress("No variants to analyze");
        return Ok(Some((0, 0.0, 0.0, region_hap_count, Vec::new())));
    }

    let variants_in_region: Vec<Variant> = variants
        .iter()
        .filter(|v| inversion_interval.contains(ZeroBasedPosition(v.position)))
        .cloned()
        .collect();

    init_variant_progress(
        &format!("Analyzing {} variants in region", variants_in_region.len()),
        variants_in_region.len() as u64,
    );

    for (i, current_variant) in variants_in_region.iter().enumerate() {
        if i % 100 == 0 {
            update_variant_progress(
                i as u64,
                &format!(
                    "Processing variant {} of {}",
                    i + 1,
                    variants_in_region.len()
                ),
            );
        }

        let mut allele_values = Vec::new();

        // Iterate over haplotype group indices, borrowing each tuple
        for (mapped_index, _) in &group_haps {
            // We don't need side in this outer loop
            // Access genotypes using the dereferenced index
            if let Some(some_genotypes) = current_variant.genotypes.get(*mapped_index) {
                for (side, genotype_vec) in some_genotypes.iter().enumerate() {
                    // Retrieve genotype and quality values for current allele
                    if let Some(&val) = genotype_vec.get(side as usize) {
                        allele_values.push(val);
                    }
                }
            }
        }
        let distinct_alleles: HashSet<u8> = allele_values.iter().copied().collect();
        if distinct_alleles.len() > 1 {
            region_segsites += 1;
        }
    }

    finish_variant_progress(&format!("Found {} segregating sites", region_segsites));

    // Calculate diversity statistics
    update_step_progress(2, "Calculating diversity statistics");

    // Define the precise QueryRegion for per-site diversity calculation,
    // matching the original entry.interval.
    let query_region_for_diversity =
        QueryRegion::from(inversion_interval.to_zero_based_inclusive());

    // The 'final_length' for overall theta/pi should also be based on the precise entry.interval
    // 'adjusted_sequence_length' is already calculated based on entry.interval and passed in.
    // If 'adjusted_sequence_length' is None, it means we use the raw length of entry.interval.
    let length_for_overall_stats =
        adjusted_sequence_length.unwrap_or(inversion_interval.len() as i64);

    let final_theta =
        calculate_watterson_theta(region_segsites, region_hap_count, length_for_overall_stats);
    let final_pi = calculate_pi(&variants_in_region, &group_haps, length_for_overall_stats);

    log(LogLevel::Info, &format!(
        "Group {} ({}): θ={:.6}, π={:.6}, with {} segregating sites ({} haplotypes, length {}bp)",
        haplotype_group, group_type, final_theta, final_pi, region_segsites, region_hap_count, length_for_overall_stats
    ));

    // Step 4: Process transcripts for this region
    if !cds_regions.is_empty() {
        update_step_progress(3, &format!("Processing {} CDS regions", cds_regions.len()));
    }

    for transcript in cds_regions {
        // Map of haplotype labels to their assembled CDS sequences
        let mut assembled: HashMap<String, Vec<u8>> = HashMap::new();
        for (mapped_index, side) in &group_haps {
            // Generate haplotype label based on side (Left -> _L, Right -> _R)
            let label = match side {
                HaplotypeSide::Left => format!("{}_L", sample_names[*mapped_index]),
                HaplotypeSide::Right => format!("{}_R", sample_names[*mapped_index]),
            };
            assembled.insert(label, Vec::new());
        }

        let mut offset_map = Vec::new();
        let mut accumulated_length = 0;
        for seg in transcript.segments.iter() {
            let seg_start = seg.start as i64;
            let seg_end = seg.end as i64;
            let seg_strand = transcript.strand;
            let mut seg_len = seg_end.saturating_sub(seg_start).saturating_add(1) as usize;

            if seg_start < 0 {
                // Log to file instead of printing to terminal
                log(
                    LogLevel::Warning,
                    &format!(
                        "Skipping negative start {} for transcript {} on {}",
                        seg_start, transcript.transcript_id, chromosome
                    ),
                );
                continue;
            }
            let base_idx = {
                let offset = seg_start as i64 - (extended_region.start as i64);
                if offset < 0 {
                    let overlap = seg_len.saturating_sub((-offset) as usize);
                    if overlap == 0 {
                        eprintln!(
                            "Skipping partial out-of-bounds {}..{} for transcript {} on {}",
                            seg_start, seg_end, transcript.transcript_id, chromosome
                        );
                        continue;
                    }
                    seg_len = overlap;
                }

                offset as usize
            };
            let end_idx = base_idx + seg_len;
            if end_idx > reference_sequence.len() {
                let overlap2 = reference_sequence.len().saturating_sub(base_idx);
                if overlap2 == 0 {
                    eprintln!(
                        "Skipping partial out-of-bounds {}..{} for transcript {} on {}",
                        seg_start, seg_end, transcript.transcript_id, chromosome
                    );
                    continue;
                }
                seg_len = overlap2;
            }

            for (mapped_index, side) in &group_haps {
                let label = match *side {
                    HaplotypeSide::Left => format!("{}_L", sample_names[*mapped_index]),
                    HaplotypeSide::Right => format!("{}_R", sample_names[*mapped_index]),
                };
                let mutable_vec = assembled
                    .get_mut(&label)
                    .expect("Missing sample in assembled map");
                let mut slice_portion = reference_sequence[base_idx..base_idx + seg_len].to_vec();
                if seg_strand == '-' {
                    slice_portion.reverse();
                    for byte_ref in slice_portion.iter_mut() {
                        *byte_ref = match *byte_ref {
                            b'A' | b'a' => b'T',
                            b'T' | b't' => b'A',
                            b'C' | b'c' => b'G',
                            b'G' | b'g' => b'C',
                            _ => b'N',
                        };
                    }
                }
                mutable_vec.extend_from_slice(&slice_portion);
            }
            offset_map.push((seg_start, seg_end, accumulated_length));
            accumulated_length += seg_len;
        }
    }

    // Generate sequence files if this is the filtered set
    if is_filtered_set {
        log(
            LogLevel::Info,
            &format!(
                "Generating sequence files for {} CDS regions, group {}",
                cds_regions.len(),
                haplotype_group
            ),
        );

        if chromosome.contains("X") || chromosome.contains("x") {
            log(
                LogLevel::Info,
                &format!(
                "DEBUG X: Processing sequence files for chrX:{}-{}, group {}, with {} CDS regions",
                inversion_interval.start, inversion_interval.end, haplotype_group, cds_regions.len()
            ),
            );
        }

        let spinner = create_spinner(&format!(
            "Creating sequences for group {} haplotypes",
            haplotype_group
        ));

        if let Err(e) = make_sequences(
            variants,
            sample_names,
            haplotype_group,
            sample_filter,
            extended_region,
            reference_sequence,
            cds_regions,
            position_allele_map.clone(),
            &chromosome,
            inversion_interval,
        ) {
            log(
                LogLevel::Warning,
                &format!(
                    "ERROR generating sequences for group {} on {}: {}",
                    haplotype_group, chromosome, e
                ),
            );
            // Continue processing - don't let sequence generation errors affect main analysis
        }

        spinner.finish_and_clear();
        log(
            LogLevel::Info,
            &format!(
                "Created sequence files for group {} haplotypes",
                haplotype_group
            ),
        );
    }

    // Calculate per-site diversity
    let spinner = create_spinner("Calculating per-site diversity");
    // query_region_for_diversity was defined earlier based on entry.interval.start and entry.interval.end
    // to correctly represent the 0-based inclusive range [entry.interval.start, entry.interval.end - 1].
    let site_diversities =
        calculate_per_site_diversity(variants, &group_haps, query_region_for_diversity);
    spinner.finish_and_clear();
    log(
        LogLevel::Info,
        &format!("Calculated diversity for {} sites", site_diversities.len()),
    );

    log(
        LogLevel::Info,
        &format!(
            "Completed analysis for {} group {}: {} segregating sites, θ={:.6}, π={:.6}",
            group_type, haplotype_group, region_segsites, final_theta, final_pi
        ),
    );

    finish_step_progress(&format!("Completed group {} analysis", haplotype_group));

    Ok(Some((
        region_segsites,
        final_theta,
        final_pi,
        region_hap_count,
        site_diversities,
    )))
}

pub fn map_sample_names_to_indices(
    sample_names: &[String],
) -> Result<HashMap<String, usize>, VcfError> {
    let mut vcf_sample_id_to_index = HashMap::new();
    for (i, name) in sample_names.iter().enumerate() {
        // Convert to String to so the HashMap owns its keys.
        let sample_id = name.rsplit('_').next().unwrap_or(name).to_string();
        vcf_sample_id_to_index.insert(sample_id, i);
    }
    Ok(vcf_sample_id_to_index)
}

/// Retrieves VCF sample indices and HaplotypeSides for samples belonging to a specific haplotype group (0 or 1),
/// based on the sample filter definitions from the configuration.
///
/// # Arguments
/// * `haplotype_group` - The target haplotype group (0 or 1).
/// * `sample_filter` - A map from sample names (String) to their (left_haplotype_group, right_haplotype_group) assignments.
/// * `vcf_sample_id_to_index` - A map from core VCF sample IDs (String) to their 0-based VCF column indices.
///
/// # Returns
/// A `Result` containing a `Vec` of `(vcf_sample_index, HaplotypeSide)` tuples.
pub fn get_haplotype_indices_for_group(
    haplotype_group: u8,
    sample_filter: &HashMap<String, (u8, u8)>,
    vcf_sample_id_to_index: &HashMap<String, usize>,
) -> Result<Vec<(usize, HaplotypeSide)>, VcfError> {
    let mut haplotype_indices = Vec::new();
    let mut missing_samples = Vec::new();

    for (sample_name, &(left_tsv, right_tsv)) in sample_filter {
        match vcf_sample_id_to_index.get(sample_name.as_str()) {
            Some(&idx) => {
                if left_tsv == haplotype_group {
                    haplotype_indices.push((idx, HaplotypeSide::Left));
                }
                if right_tsv == haplotype_group {
                    haplotype_indices.push((idx, HaplotypeSide::Right));
                }
            }
            None => {
                missing_samples.push(sample_name.clone());
                // Log warning instead of returning error
                log(
                    LogLevel::Warning,
                    &format!(
                        "Sample '{}' from config not found in VCF - skipping (haplotype indices)",
                        sample_name
                    ),
                );
            }
        }
    }

    if !missing_samples.is_empty() {
        log(
            LogLevel::Warning,
            &format!(
                "Missing {} samples when getting haplotype indices: {}",
                missing_samples.len(),
                missing_samples.join(", ")
            ),
        );
    }

    Ok(haplotype_indices)
}

pub fn process_config_entries(
    config_entries: &[ConfigEntry],
    vcf_folder: &str,
    output_file: &Path,
    min_gq: u16,
    mask: Option<Arc<HashMap<String, Vec<(i64, i64)>>>>,
    allow: Option<Arc<HashMap<String, Vec<(i64, i64)>>>>,
    args: &Args,
) -> Result<(), VcfError> {
    // Create a temp directory and save it to TEMP_DIR
    let temp_dir_path = {
        let temp_dir = create_temp_dir()?;
        let temp_path = temp_dir.path().to_path_buf();

        let mut locked_opt = TEMP_DIR.lock();
        *locked_opt = Some(temp_dir);

        temp_path
    };

    // For PCA, collect filtered variants across chromosomes
    let global_filtered_variants: Arc<Mutex<HashMap<String, Vec<Variant>>>> =
        Arc::new(Mutex::new(HashMap::new()));
    let global_sample_names: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

    // Create CSV writer and write the header once in the temporary directory
    let temp_output_file = temp_dir_path.join(output_file.file_name().unwrap());
    let mut writer = create_and_setup_csv_writer(&temp_output_file)?;
    write_csv_header(&mut writer)?;

    // Group config entries by chromosome for efficiency
    let grouped = group_config_entries_by_chr(config_entries);

    // Log the count of entries per chromosome
    log(LogLevel::Info, "STATISTICS: Input regions by chromosome:");
    for (chr, entries) in &grouped {
        log(
            LogLevel::Info,
            &format!("  - {}: {} regions", chr, entries.len()),
        );

        if chr.contains("X") || chr.contains("x") {
            for (i, entry) in entries.iter().enumerate() {
                log(LogLevel::Info, &format!(
                    "DEBUG X: chrX input region {}: {}:{}-{} ({} filtered samples, {} unfiltered samples)",
                    i + 1,
                    entry.seqname,
                    entry.interval.start,
                    entry.interval.end,
                    entry.samples_filtered.len(),
                    entry.samples_unfiltered.len()
                ));
            }
        }
    }
    let mut all_regional_hudson_outcomes: Vec<RegionalHudsonFSTOutcome> = Vec::new();

    // Parse population CSV once if --fst and --fst_populations are provided.
    // This parsed data will be passed to each chromosome's processing.
    let parsed_csv_populations_arc: Option<Arc<HashMap<String, Vec<String>>>> = if args.enable_fst {
        if let Some(csv_path_str) = &args.fst_populations {
            log(
                LogLevel::Info,
                &format!(
                    "Parsing population definition file for FST: {}",
                    csv_path_str
                ),
            );
            match crate::stats::parse_population_csv(Path::new(csv_path_str)) {
                Ok(parsed_map) => Some(Arc::new(parsed_map)),
                Err(e) => {
                    log(LogLevel::Error, &format!("Failed to parse population CSV '{}': {}. FST calculations for CSV-defined populations will be skipped.", csv_path_str, e));
                    None
                }
            }
        } else {
            None
        }
    } else {
        None
    };

    // The map operation collects results per chromosome.
    // Each result is a tuple: (main_csv_data_for_this_chr, hudson_data_for_this_chr)
    let per_chromosome_collected_results: Vec<(
        Vec<(
            CsvRowData,
            Vec<(i64, f64, f64, u8, bool)>,
            Vec<(i64, f64, f64)>,
            Vec<(i64, f64, f64, f64)>,
        )>,
        Vec<RegionalHudsonFSTOutcome>,
    )> = grouped
        .par_iter()
        .map(|(chr, chr_entries)| {
            match process_chromosome_entries(
                chr,
                chr_entries,
                vcf_folder,
                min_gq,
                &mask,
                &allow,
                args,
                if args.enable_pca {
                    Some((
                        global_filtered_variants.clone(),
                        global_sample_names.clone(),
                    ))
                } else {
                    None
                },
                parsed_csv_populations_arc.clone(), // Pass the Arc'd map
            ) {
                Ok(data_tuple_for_chr) => Some(data_tuple_for_chr),
                Err(e) => {
                    eprintln!("Error processing chromosome {}: {}", chr, e);
                    None // This chromosome processing failed
                }
            }
        })
        .filter_map(|optional_result| optional_result) // Remove None entries (failed chromosomes)
        .collect();

    // BEFORE: big aggregation into all_main_csv_data_tuples + later write
    // AFTER:
    for (mut main_data_for_chr, mut hudson_data_for_chr) in per_chromosome_collected_results {
        // write each main CSV row + per-site outputs now
        for (csv_row, per_site_diversity_vec, fst_data_wc, fst_data_hudson) in
            main_data_for_chr.drain(..)
        {
            append_csv_row(&temp_output_file, &csv_row)?; // CSV (temp) immediate

            append_diversity_falsta(
                &temp_dir_path.join("per_site_diversity_output.falsta"),
                &csv_row,
                &per_site_diversity_vec,
            )?;

            append_fst_falsta(
                &temp_dir_path.join("per_site_fst_output.falsta"),
                &csv_row,
                &fst_data_wc,
                &fst_data_hudson,
            )?;
        }

        // Hudson TSV append for this chromosome
        if args.enable_fst && !hudson_data_for_chr.is_empty() {
            let hudson_output_filename = "hudson_fst_results.tsv".to_string();
            let hudson_output_path = if let Some(main_output_parent) = output_file.parent() {
                main_output_parent.join(&hudson_output_filename)
            } else {
                std::path::Path::new(&hudson_output_filename).to_path_buf()
            };
            append_hudson_tsv(&hudson_output_path, &hudson_data_for_chr)?;
            all_regional_hudson_outcomes.append(&mut hudson_data_for_chr);
        }
        // all per-chr vectors drop here ✅
    }

    writer.flush().map_err(|e| VcfError::Io(e.into()))?;
    println!("Wrote FASTA-style per-site diversity data to per_site_diversity_output.falsta");
    println!("Wrote FASTA-style per-site FST data to per_site_fst_output.falsta");
    println!(
        "Processing complete. Check the output file: {:?}",
        output_file
    );

    // Copy files from temporary directory to final destinations
    let temp_dir_path = {
        let locked_opt = TEMP_DIR.lock();
        if let Some(dir) = locked_opt.as_ref() {
            dir.path().to_path_buf()
        } else {
            return Err(VcfError::Parse(
                "Failed to access temporary directory".to_string(),
            ));
        }
    };

    // Copy CSV file
    let temp_csv = temp_dir_path.join(output_file.file_name().unwrap());
    if let Some(parent) = output_file.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::copy(&temp_csv, output_file)?;

    // Copy FASTA files
    let temp_fasta = temp_dir_path.join("per_site_diversity_output.falsta");
    if temp_fasta.exists() {
        std::fs::copy(
            &temp_fasta,
            std::path::Path::new("per_site_diversity_output.falsta"),
        )?;
    }

    let temp_fst_fasta = temp_dir_path.join("per_site_fst_output.falsta");
    if temp_fst_fasta.exists() {
        std::fs::copy(
            &temp_fst_fasta,
            std::path::Path::new("per_site_fst_output.falsta"),
        )?;
    }

    // Copy PHYLIP files
    for entry in std::fs::read_dir(&temp_dir_path)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("phy") {
            let file_name = path.file_name().unwrap();
            std::fs::copy(&path, std::path::Path::new(".").join(file_name))?;
        }
    }

    // Copy log files
    for log_file in ["cds_validation.log", "transcript_overlap.log"] {
        let temp_log = temp_dir_path.join(log_file);
        if temp_log.exists() {
            std::fs::copy(&temp_log, std::path::Path::new(".").join(log_file))?;
        }
    }

    // Write Hudson FST results if FST calculations were enabled
    if args.enable_fst {
        let hudson_output_filename = "hudson_fst_results.tsv".to_string();

        // Place the Hudson FST output file in the same directory as the main output file.
        let hudson_output_path = if let Some(main_output_parent) = output_file.parent() {
            main_output_parent.join(&hudson_output_filename)
        } else {
            Path::new(&hudson_output_filename).to_path_buf()
        };

        log(
            LogLevel::Info,
            &format!(
                "Writing Hudson FST results to: {}",
                hudson_output_path.display()
            ),
        );

        let hudson_file = File::create(&hudson_output_path).map_err(|e| VcfError::Io(e.into()))?;
        let mut hudson_writer = WriterBuilder::new()
            .delimiter(b'\t')
            .from_writer(BufWriter::new(hudson_file));

        // Write Hudson FST header
        hudson_writer.write_record(&[
            "chr",
            "region_start_0based",
            "region_end_0based",
            "pop1_id_type",
            "pop1_id_name",
            "pop2_id_type",
            "pop2_id_name",
            "Dxy",
            "pi_pop1",
            "pi_pop2",
            "pi_xy_avg",
            "FST",
        ])?;

        // Write Hudson FST data rows
        for regional_outcome in &all_regional_hudson_outcomes {
            let (pop1_type, pop1_name) = format_population_id(&regional_outcome.outcome.pop1_id);
            let (pop2_type, pop2_name) = format_population_id(&regional_outcome.outcome.pop2_id);
            hudson_writer.write_record(&[
                regional_outcome.chr.clone(),
                regional_outcome.region_start.to_string(), // region_start is 0-based inclusive
                regional_outcome.region_end.to_string(),   // region_end is 0-based inclusive
                pop1_type,
                pop1_name,
                pop2_type,
                pop2_name,
                format_optional_float(regional_outcome.outcome.d_xy),
                format_optional_float(regional_outcome.outcome.pi_pop1),
                format_optional_float(regional_outcome.outcome.pi_pop2),
                format_optional_float(regional_outcome.outcome.pi_xy_avg),
                format_optional_float(regional_outcome.outcome.fst),
            ])?;
        }
        hudson_writer.flush()?;
        log(
            LogLevel::Info,
            &format!(
                "Successfully wrote {} Hudson FST records to {}",
                all_regional_hudson_outcomes.len(),
                hudson_output_path.display()
            ),
        );
    }

    Ok(())
}

fn create_and_setup_csv_writer(
    output_file: &Path,
) -> Result<csv::Writer<BufWriter<File>>, VcfError> {
    // Create the file, wrap in BufWriter, then build the CSV writer from that.
    let file = File::create(output_file).map_err(|e| VcfError::Io(e.into()))?;
    let buf_writer = BufWriter::new(file);
    let writer = WriterBuilder::new()
        .has_headers(false)
        .from_writer(buf_writer);
    Ok(writer)
}

/// Writes the CSV header row.
fn write_csv_header<W: Write>(writer: &mut csv::Writer<W>) -> Result<(), VcfError> {
    writer
        .write_record(&[
            "chr",
            "region_start", // 1-based inclusive
            "region_end",   // 1-based inclusive
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
            // Weir & Cockerham FST components for haplotype groups
            "haplotype_overall_fst_wc",
            "haplotype_between_pop_variance_wc",
            "haplotype_within_pop_variance_wc",
            "haplotype_num_informative_sites_wc",
            // Hudson FST components for haplotype groups 0 vs 1
            "hudson_fst_hap_group_0v1",
            "hudson_dxy_hap_group_0v1",
            "hudson_pi_hap_group_0",
            "hudson_pi_hap_group_1",
            "hudson_pi_avg_hap_group_0v1",
        ])
        .map_err(|e| VcfError::Io(e.into()))?;
    Ok(())
}

/// Writes a single row of data to the CSV.
fn write_csv_row<W: Write>(writer: &mut csv::Writer<W>, row: &CsvRowData) -> Result<(), VcfError> {
    writer
        .write_record(&[
            &row.seqname,
            &row.region_start.to_string(), // 1-based inclusive
            &row.region_end.to_string(),   // 1-based inclusive
            &row.seq_len_0.to_string(),
            &row.seq_len_1.to_string(),
            &row.seq_len_adj_0.to_string(),
            &row.seq_len_adj_1.to_string(),
            &row.seg_sites_0.to_string(),
            &row.seg_sites_1.to_string(),
            &format!("{:.6}", row.w_theta_0),
            &format!("{:.6}", row.w_theta_1),
            &format!("{:.6}", row.pi_0),
            &format!("{:.6}", row.pi_1),
            &row.seg_sites_0_f.to_string(),
            &row.seg_sites_1_f.to_string(),
            &format!("{:.6}", row.w_theta_0_f),
            &format!("{:.6}", row.w_theta_1_f),
            &format!("{:.6}", row.pi_0_f),
            &format!("{:.6}", row.pi_1_f),
            &row.n_hap_0_unf.to_string(),
            &row.n_hap_1_unf.to_string(),
            &row.n_hap_0_f.to_string(),
            &row.n_hap_1_f.to_string(),
            &format!("{:.6}", row.inv_freq_no_filter),
            &format!("{:.6}", row.inv_freq_filter),
            // Weir & Cockerham FST components for haplotype groups
            &format_optional_float(row.haplotype_overall_fst_wc),
            &format_optional_float(row.haplotype_between_pop_variance_wc),
            &format_optional_float(row.haplotype_within_pop_variance_wc),
            &format_optional_usize(row.haplotype_num_informative_sites_wc),
            // Write Hudson FST components for haplotype groups 0 vs 1
            &format_optional_float(row.hudson_fst_hap_group_0v1),
            &format_optional_float(row.hudson_dxy_hap_group_0v1),
            &format_optional_float(row.hudson_pi_hap_group_0),
            &format_optional_float(row.hudson_pi_hap_group_1),
            &format_optional_float(row.hudson_pi_avg_hap_group_0v1),
        ])
        .map_err(|e| VcfError::Io(e.into()))?;
    Ok(())
}

/// Groups `ConfigEntry` objects by chromosome name.
/// Returns a HashMap<chr_name, Vec<ConfigEntry>>.
fn group_config_entries_by_chr(
    config_entries: &[ConfigEntry],
) -> HashMap<String, Vec<ConfigEntry>> {
    let mut regions_per_chr: HashMap<String, Vec<ConfigEntry>> = HashMap::new();
    for entry in config_entries {
        regions_per_chr
            .entry(entry.seqname.clone())
            .or_insert_with(Vec::new)
            .push(entry.clone());
    }
    regions_per_chr
}

/// Loads the reference sequence, transcripts, finds the VCF, then processes
/// each config entry for that chromosome. Returns a Vec of row data for each entry.
fn process_chromosome_entries(
    chr: &str,
    entries: &[ConfigEntry],
    vcf_folder: &str,
    min_gq: u16,
    mask: &Option<Arc<HashMap<String, Vec<(i64, i64)>>>>,
    allow: &Option<Arc<HashMap<String, Vec<(i64, i64)>>>>,
    args: &Args,
    pca_storage: Option<(
        Arc<Mutex<HashMap<String, Vec<Variant>>>>,
        Arc<Mutex<Vec<String>>>,
    )>,
    // Arc containing the parsed population definitions from the CSV file, if provided.
    // Key: Population Name (String), Value: List of Sample IDs (String) in that population.
    parsed_csv_populations_arc: Option<Arc<HashMap<String, Vec<String>>>>,
) -> Result<
    (
        Vec<(
            CsvRowData,
            Vec<(i64, f64, f64, u8, bool)>,
            Vec<(i64, f64, f64)>,
            Vec<(i64, f64, f64, f64)>,
        )>,
        Vec<RegionalHudsonFSTOutcome>,
    ),
    VcfError,
> {
    set_stage(ProcessingStage::ConfigEntry);
    log(LogLevel::Info, &format!("Processing chromosome: {}", chr));

    init_step_progress(&format!("Loading resources for chr{}", chr), 3);

    // Load entire chromosome length from reference index
    update_step_progress(0, &format!("Reading reference index for chr{}", chr));
    let chr_length = {
        let fasta_reader =
            bio::io::fasta::IndexedReader::from_file(&args.reference_path).map_err(|e| {
                log(
                    LogLevel::Error,
                    &format!("Failed to open reference file: {}", e),
                );
                VcfError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    e.to_string(),
                ))
            })?;
        let sequences = fasta_reader.index.sequences();
        let seq_info = sequences
            .iter()
            .find(|seq| seq.name == chr || seq.name == format!("chr{}", chr))
            .ok_or_else(|| {
                log(
                    LogLevel::Error,
                    &format!("Chromosome {} not found in reference", chr),
                );
                VcfError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Chromosome {} not found in reference", chr),
                ))
            })?;
        seq_info.len as i64
    };

    // Read the full reference sequence for that chromosome.
    update_step_progress(1, &format!("Loading reference sequence for chr{}", chr));
    let ref_sequence = {
        let entire_chrom = ZeroBasedHalfOpen {
            start: 0,
            end: chr_length as usize,
        };
        read_reference_sequence(Path::new(&args.reference_path), chr, entire_chrom)?
    };
    log(
        LogLevel::Info,
        &format!(
            "Loaded {}bp reference sequence for chr{}",
            ref_sequence.len(),
            chr
        ),
    );

    // Parse all transcripts for that chromosome from the GTF
    update_step_progress(2, &format!("Loading transcript annotations for chr{}", chr));
    let all_transcripts = parse_gtf_file(Path::new(&args.gtf_path), chr)?;
    // We'll keep them in `cds_regions` for subsequent filtering
    let cds_regions = all_transcripts;
    log(
        LogLevel::Info,
        &format!(
            "Loaded {} transcript annotations for chr{}",
            cds_regions.len(),
            chr
        ),
    );

    // Locate the VCF file for this chromosome
    let vcf_file = match find_vcf_file(vcf_folder, chr) {
        Ok(file) => {
            log(
                LogLevel::Info,
                &format!("Found VCF file for chr{}: {}", chr, file.display()),
            );
            file
        }
        Err(e) => {
            log(
                LogLevel::Error,
                &format!("Error finding VCF file for chr{}: {:?}", chr, e),
            );
            finish_step_progress(&format!("Failed to find VCF for chr{}", chr));
            return Ok((Vec::new(), Vec::new())); // Return empty tuple for all results
        }
    };

    finish_step_progress(&format!("Loaded resources for chr{}", chr));

    // Stores tuples for the main CSV output (W&C FST, diversity stats, Hudson per-site FST)
    let mut main_csv_tuples: Vec<(
        CsvRowData,
        Vec<(i64, f64, f64, u8, bool)>,
        Vec<(i64, f64, f64)>,
        Vec<(i64, f64, f64, f64)>,
    )> = Vec::with_capacity(entries.len());
    // Stores RegionalHudsonFSTOutcome for the dedicated Hudson FST output file for this chromosome
    let mut chromosome_hudson_fst_results: Vec<RegionalHudsonFSTOutcome> = Vec::new();

    // Store filtered variants for PCA if enabled
    if let Some((_, sample_names_storage)) = &pca_storage {
        let vcf_file = match find_vcf_file(vcf_folder, chr) {
            Ok(file) => file,
            Err(e) => {
                log(
                    LogLevel::Error,
                    &format!("Error finding VCF file for chr{} for PCA: {:?}", chr, e),
                );
                return Ok((Vec::new(), Vec::new())); // Return empty tuple for all results
            }
        };

        // Open VCF reader to get sample names
        let mut reader = open_vcf_reader(&vcf_file)?;
        let mut buffer = String::new();
        let mut sample_names = Vec::new();

        // Extract sample names from VCF header
        while reader.read_line(&mut buffer)? > 0 {
            if buffer.starts_with("##") {
                // Skip metadata lines
            } else if buffer.starts_with("#CHROM") {
                validate_vcf_header(&buffer)?;
                sample_names = buffer
                    .split_whitespace()
                    .skip(9)
                    .map(String::from)
                    .collect();
                break;
            }
            buffer.clear();
        }

        // Store sample names (once)
        {
            let mut names_storage = sample_names_storage.lock();
            if names_storage.is_empty() {
                *names_storage = sample_names;
            }
        }
    }

    // Initialize progress for config entries
    init_entry_progress(
        &format!("Processing {} regions on chr{}", entries.len(), chr),
        entries.len() as u64,
    );

    // For each config entry in this chromosome, do the work
    //    (We could also parallelize here...)
    // Should this be for entry in entries instead?
    for (idx, entry) in entries.iter().enumerate() {
        let region_desc = format!("{}:{}-{}", chr, entry.interval.start, entry.interval.end);
        update_entry_progress(idx as u64, &format!("Processing region {}", region_desc));

        let result = process_single_config_entry(
            entry.clone(),
            &vcf_file,
            min_gq,
            mask,
            allow,
            &ref_sequence,
            &cds_regions,
            chr,
            args,
            pca_storage.as_ref(),
            parsed_csv_populations_arc.clone(), // Pass down the Arc<HashMap<String, Vec<String>>>
        );

        match result {
            Ok(Some((
                main_csv_tuple_content,
                per_site_diversity_data,
                per_site_wc_fst_data,
                per_site_hudson_fst_records,
                mut hudson_outcomes_for_entry,
            ))) => {
                main_csv_tuples.push((
                    main_csv_tuple_content,
                    per_site_diversity_data,
                    per_site_wc_fst_data,
                    per_site_hudson_fst_records,
                ));
                chromosome_hudson_fst_results.append(&mut hudson_outcomes_for_entry);
                log(
                    LogLevel::Info,
                    &format!("Successfully processed region {}", region_desc),
                );

                if entry.seqname.contains("X") || entry.seqname.contains("x") {
                    log(
                        LogLevel::Info,
                        &format!(
                            "ADDED chrX region {} to output rows (now {} main CSV tuples)",
                            region_desc,
                            main_csv_tuples.len()
                        ),
                    );
                }
            }
            Ok(None) => {
                log(
                    LogLevel::Warning,
                    &format!(
                        "DROPPED: Region {} was skipped (no matching haplotypes)",
                        region_desc
                    ),
                );

                if entry.seqname.contains("X") || entry.seqname.contains("x") {
                    log(
                        LogLevel::Warning,
                        &format!(
                            "DEBUG X: DROPPED chrX region {} from output (returned Ok(None))",
                            region_desc
                        ),
                    );
                }
            }
            Err(e) => {
                log(
                    LogLevel::Error,
                    &format!("DROPPED: Error processing region {}: {}", region_desc, e),
                );

                if entry.seqname.contains("X") || entry.seqname.contains("x") {
                    log(
                        LogLevel::Error,
                        &format!(
                            "DEBUG X: DROPPED chrX region {} due to error: {}",
                            region_desc, e
                        ),
                    );
                }
            }
        }

        update_entry_progress(
            (idx + 1) as u64,
            &format!("Global progress for region {}", region_desc),
        );
    }

    finish_entry_progress(
        &format!("Processed {} regions on chr{}", entries.len(), chr),
        entries.len(),
    );

    display_status_box(StatusBox {
        title: format!("Chromosome {} Statistics", chr),
        stats: vec![
            ("Total regions".to_string(), entries.len().to_string()),
            (
                "Successful regions".to_string(),
                main_csv_tuples.len().to_string(),
            ),
            (
                "Skipped/failed".to_string(),
                (entries.len() - main_csv_tuples.len()).to_string(),
            ),
        ],
    });

    if args.enable_pca {
        if let Some((filtered_variants_map, sample_names_storage)) = &pca_storage {
            let spinner_pca = create_spinner(&format!(
                "Performing single PCA after all regions for chromosome {}",
                chr
            ));
            let sample_names_for_pca = {
                let stored = sample_names_storage.lock();
                stored.clone()
            };
            let filtered_variants_for_chr = {
                let stored = filtered_variants_map.lock();
                stored.get(chr).cloned().unwrap_or_else(Vec::new)
            };
            log(
                LogLevel::Info,
                &format!(
                    "DEBUG: entire chromosome {} has {} total filtered variants",
                    chr,
                    filtered_variants_for_chr.len()
                ),
            );
            if !filtered_variants_for_chr.is_empty() {
                match crate::pca::compute_chromosome_pca(
                    &filtered_variants_for_chr,
                    &sample_names_for_pca,
                    args.pca_components,
                ) {
                    Ok(pca_result) => {
                        let out_dir = Path::new("pca_per_chr_outputs");
                        if !out_dir.exists() {
                            std::fs::create_dir_all(out_dir)?;
                        }
                        crate::pca::write_chromosome_pca_to_file(&pca_result, chr, out_dir)?;
                    }
                    Err(err) => {
                        log(
                            LogLevel::Warning,
                            &format!("Chromosome {} PCA error: {}", chr, err),
                        );
                    }
                }
            } else {
                log(
                    LogLevel::Warning,
                    &format!(
                        "No filtered variants remain for chromosome {}. Skipping PCA.",
                        chr
                    ),
                );
            }
            spinner_pca.finish_and_clear();
        } else {
            log(LogLevel::Warning, "PCA is enabled but pca_storage is None.");
        }
    }

    // This function returns a tuple:
    // 1. Data for the main CSV output file.
    // 2. Data for the dedicated Hudson FST output file, specific to this chromosome.
    Ok((main_csv_tuples, chromosome_hudson_fst_results))
}

fn generate_full_region_alignment(
    entry: &ConfigEntry,
    haplotype_group: u8,
    region_variants: &[Variant],
    sample_names: &[String],
    full_ref_sequence: &[u8],
    position_allele_map: &Arc<Mutex<HashMap<i64, (char, char)>>>,
    mask: &Option<Arc<HashMap<String, Vec<(i64, i64)>>>>,
    allow: &Option<Arc<HashMap<String, Vec<(i64, i64)>>>>,
    vcf_sample_id_to_index: &HashMap<String, usize>,
) -> Result<(), VcfError> {
    let group_haps = get_haplotype_indices_for_group(
        haplotype_group,
        &entry.samples_filtered,
        vcf_sample_id_to_index,
    )?;

    if group_haps.is_empty() {
        log(
            LogLevel::Warning,
            &format!(
                "No haplotypes for group {} to write alignment",
                haplotype_group
            ),
        );
        return Ok(());
    }

    let region_start = entry.interval.start as usize;
    let region_end = entry.interval.end as usize;
    let reference_slice = &full_ref_sequence[region_start..region_end];

    let pos_map = position_allele_map.lock();
    let mask_map = mask.as_ref().map(|m| m.as_ref());
    let allow_map = allow.as_ref().map(|a| a.as_ref());

    let mut seq_map: HashMap<String, Vec<char>> = HashMap::new();

    for (sample_idx, side) in group_haps {
        let mut seq = reference_slice.to_vec();
        for variant in region_variants {
            if let Some(genotype_opt) = variant.genotypes.get(sample_idx) {
                if let Some(genotype_vec) = genotype_opt {
                    let allele_opt = match side {
                        HaplotypeSide::Left => genotype_vec.get(0),
                        HaplotypeSide::Right => genotype_vec.get(1),
                    };
                    if let Some(&1) = allele_opt {
                        if let Some((_ref_allele, alt_allele)) = pos_map.get(&variant.position) {
                            let rel = (variant.position - entry.interval.start as i64) as usize;
                            if rel < seq.len() {
                                seq[rel] = *alt_allele as u8;
                            }
                        }
                    }
                }
            }
        }

        // Apply mask and allow regions
        for (i, base) in seq.iter_mut().enumerate() {
            let abs_pos = entry.interval.start as i64 + i as i64;
            if let Some(mmap) = mask_map.and_then(|m| m.get(&entry.seqname)) {
                if position_in_regions(abs_pos, mmap) {
                    *base = b'N';
                    continue;
                }
            }
            if let Some(amap) = allow_map.and_then(|a| a.get(&entry.seqname)) {
                if !position_in_regions(abs_pos, amap) {
                    *base = b'N';
                }
            } else if allow_map.is_some() {
                *base = b'N';
            }
        }

        let sample_name = match side {
            HaplotypeSide::Left => format!("{}_L", sample_names[sample_idx]),
            HaplotypeSide::Right => format!("{}_R", sample_names[sample_idx]),
        };
        let seq_chars = seq.iter().map(|&b| b as char).collect::<Vec<char>>();
        seq_map.insert(sample_name, seq_chars);
    }

    if seq_map.is_empty() {
        log(
            LogLevel::Warning,
            &format!("No sequences generated for group {}", haplotype_group),
        );
        return Ok(());
    }

    let start_1based = entry.interval.start_1based_inclusive();
    let end_1based = entry.interval.get_1based_inclusive_end_coord();
    let filename = format!(
        "inversion_group{}_{}_start{}_end{}.phy",
        haplotype_group, entry.seqname, start_1based, end_1based
    );

    write_phylip_file(&filename, &seq_map, &filename)?;
    Ok(())
}

/// Processes a single config entry's region and sample sets for a given chromosome.
///  - Filters transcripts to the region
///  - Calls `process_vcf` to get unfiltered vs. filtered variants
///  - Computes population-genetic stats for group 0/1 (unfiltered & filtered)
///  - Returns one `CsvRowData` if successful, or None if e.g. no haplotypes matched
fn process_single_config_entry(
    entry: ConfigEntry,
    vcf_file: &Path,
    min_gq: u16,
    mask: &Option<Arc<HashMap<String, Vec<(i64, i64)>>>>,
    allow: &Option<Arc<HashMap<String, Vec<(i64, i64)>>>>,
    ref_sequence: &[u8],
    cds_regions: &[TranscriptAnnotationCDS],
    chr: &str,
    args: &Args,
    pca_storage: Option<&(
        Arc<Mutex<HashMap<String, Vec<Variant>>>>,
        Arc<Mutex<Vec<String>>>,
    )>,
    // Arc containing the parsed population definitions from the CSV file, if provided.
    // Key: Population Name (String), Value: List of Sample IDs (String) in that population.
    parsed_csv_populations_arc: Option<Arc<HashMap<String, Vec<String>>>>,
) -> Result<
    Option<(
        CsvRowData, // Data for the main CSV output (W&C FST, diversity stats, etc.)
        Vec<(i64, f64, f64, u8, bool)>, // Per-site diversity data (pos, pi, theta, group_id, is_filtered) for falsta output
        Vec<(i64, f64, f64)>, // Per-site W&C FST data (pos, overall_wc_fst, pairwise_wc_fst_0vs1) for falsta output
        Vec<(i64, f64, f64, f64)>, // Per-site Hudson data (pos_1based, fst, numerator, denominator) for haplotype groups
        Vec<RegionalHudsonFSTOutcome>, // Hudson FST results specific to this config entry
    )>,
    VcfError,
> {
    set_stage(ProcessingStage::ConfigEntry);
    let region_desc = format!(
        "{}:{}-{}",
        entry.seqname, entry.interval.start, entry.interval.end
    );

    log(
        LogLevel::Info,
        &format!("Processing region: {}", region_desc),
    );

    init_step_progress(&format!("Filtering transcripts for {}", region_desc), 2);

    let local_cds = filter_and_log_transcripts(
        cds_regions.to_vec(),
        entry.interval.to_zero_based_inclusive().into(),
    );

    log(
        LogLevel::Info,
        &format!(
            "Found {} transcripts overlapping region {}",
            local_cds.len(),
            region_desc
        ),
    );

    update_step_progress(1, "Preparing extended region");

    if entry.seqname.contains("X") || entry.seqname.contains("x") {
        log(
            LogLevel::Info,
            &format!(
                "DEBUG X: Processing chrX region: {}:{}-{}",
                entry.seqname, entry.interval.start, entry.interval.end
            ),
        );
    }

    let chr_length = ref_sequence.len() as i64;
    let extended_region = ZeroBasedHalfOpen::from_1based_inclusive(
        (entry.interval.start as i64 - 3_000_000).max(0),
        ((entry.interval.end as i64) + 3_000_000).min(chr_length),
    );

    finish_step_progress(&format!(
        "Extended region: {}:{}-{}",
        chr,
        extended_region.start_1based_inclusive(),
        extended_region.get_1based_inclusive_end_coord()
    ));

    let position_allele_map = Arc::new(Mutex::new(HashMap::<i64, (char, char)>::new()));

    set_stage(ProcessingStage::VcfProcessing);
    init_step_progress(&format!("Processing VCF for {}", region_desc), 2);

    log(
        LogLevel::Info,
        &format!(
            "Processing VCF for {}:{}-{} (extended: {}-{})",
            chr,
            entry.interval.start,
            entry.interval.end,
            extended_region.start,
            extended_region.end
        ),
    );

    let (all_variants, filtered_idxs, sample_names, _, _, filtering_stats) = match process_vcf(
        vcf_file,
        Path::new(&args.reference_path),
        chr.to_string(),
        extended_region,
        min_gq,
        mask.clone(),
        allow.clone(),
        position_allele_map.clone(),
    ) {
        Ok(data) => data,
        Err(e) => {
            log(
                LogLevel::Error,
                &format!("Error processing VCF for {}: {}", chr, e),
            );
            finish_step_progress("VCF processing failed");
            return Ok(None);
        }
    };

    let vcf_sample_id_to_index = map_sample_names_to_indices(&sample_names)?;

    update_step_progress(1, "Analyzing variant statistics");

    // UNFILTERED (unchanged behavior)
    let region_variants_unfiltered: Vec<Variant> = all_variants
        .iter()
        .filter(|v| entry.interval.contains(ZeroBasedPosition(v.position)))
        .cloned()
        .collect();

    // FILTERED (new): project indices -> variants, then restrict to region
    let region_variants_filtered: Vec<Variant> = filtered_idxs
        .iter()
        .map(|&i| &all_variants[i])
        .filter(|v| entry.interval.contains(ZeroBasedPosition(v.position)))
        .cloned()
        .collect();

    if !region_variants_filtered.is_empty() {
        let is_sorted = region_variants_filtered
            .windows(2)
            .all(|w| w[0].position <= w[1].position);
        if !is_sorted {
            // This should not happen if process_vcf sorts them.
            // Log an error or panic, as this is a critical pre-condition for efficient and correct sub-slicing.
            log(LogLevel::Error, "CRITICAL: region_variants_filtered are not sorted by position in process_single_config_entry. Hudson FST results may be incorrect.");
        }
    }

    // Define the precise genomic region for Hudson FST analysis, corresponding to entry.interval.
    // This region's effective length is `adjusted_sequence_length`.
    let hudson_analysis_region_start_0based = entry.interval.start as i64;
    // entry.interval.end is exclusive for ZeroBasedHalfOpen.
    let hudson_analysis_region_end_0based_exclusive = entry.interval.end as i64;

    // The variants for Hudson FST should be the filtered variants within the specific region of the entry.
    let variants_for_hudson_slice: &[Variant] = &region_variants_filtered[..];

    let hudson_query_region =
        if hudson_analysis_region_end_0based_exclusive > hudson_analysis_region_start_0based {
            QueryRegion {
                start: hudson_analysis_region_start_0based,
                end: hudson_analysis_region_end_0based_exclusive - 1,
            }
        } else {
            // Empty or invalid region
            QueryRegion { start: 0, end: -1 }
        };
    let hudson_region_is_valid = hudson_query_region.start <= hudson_query_region.end;

    // Calculate FST if enabled
    let (fst_results_filtered, _) = if args.enable_fst {
        let spinner = create_spinner("Calculating FST statistics");

        // Define the FST analysis region.
        // entry.interval is ZeroBasedHalfOpen [start, end), meaning 0-based inclusive start and 0-based exclusive end.
        // QueryRegion for FST calculations requires a 0-based inclusive start and 0-based inclusive end.
        // Convert entry.interval to ZeroBasedInclusive, then to QueryRegion.
        let fst_query_region_zbi = entry.interval.to_zero_based_inclusive();
        let fst_query_region: QueryRegion = fst_query_region_zbi.into();

        // FST between haplotype groups (0 vs 1)
        log(
            LogLevel::Info,
            "Calculating FST between haplotype groups (0 vs 1)",
        );
        let haplotype_fst = calculate_fst_wc_haplotype_groups(
            &region_variants_filtered,
            &sample_names,
            &entry.samples_filtered,
            fst_query_region,
        );

        // FST between population groups if CSV is provided
        let population_fst = if let Some(pop_csv) = &args.fst_populations {
            log(
                LogLevel::Info,
                &format!("Calculating FST between population groups from {}", pop_csv),
            );
            match calculate_fst_wc_csv_populations(
                &region_variants_filtered,
                &sample_names,
                Path::new(pop_csv),
                fst_query_region,
            ) {
                Ok(results) => {
                    // Log successful population FST calculation
                    log(
                        LogLevel::Info,
                        &format!(
                        "Successfully calculated population FST with {} populations and {} sites",
                        results.pairwise_fst.len(), results.site_fst.len()
                    ),
                    );
                    Some(results)
                }
                Err(e) => {
                    log(
                        LogLevel::Error,
                        &format!("Error calculating population FST: {}", e),
                    );
                    None
                }
            }
        } else {
            None
        };

        spinner.finish_and_clear();
        log(LogLevel::Info, "FST calculations complete");

        (Some(haplotype_fst), population_fst)
    } else {
        (None, None)
    };

    // Store filtered variants for PCA if enabled and the pca_storage is provided
    if args.enable_pca {
        if let Some((variants_storage, sample_names_storage)) = &pca_storage {
            let filtered_for_chr: Vec<Variant> = filtered_idxs
                .iter()
                .map(|&i| all_variants[i].clone())
                .collect();
            variants_storage
                .lock()
                .insert(chr.to_string(), filtered_for_chr);

            // Store sample names if not already stored
            let mut names = sample_names_storage.lock();
            if names.is_empty() {
                *names = sample_names.clone();
            }
        }
    }

    // Display variant filtering statistics
    display_status_box(StatusBox {
        title: format!("Variant Statistics for {}", region_desc),
        stats: vec![
            (
                "Total variants".to_string(),
                filtering_stats.total_variants.to_string(),
            ),
            (
                "Filtered variants".to_string(),
                filtering_stats._filtered_variants.to_string(),
            ),
            (
                "% Filtered".to_string(),
                format!(
                    "{:.1}%",
                    (filtering_stats._filtered_variants as f64
                        / filtering_stats.total_variants.max(1) as f64)
                        * 100.0
                ),
            ),
            (
                "Due to mask".to_string(),
                filtering_stats.filtered_due_to_mask.to_string(),
            ),
            (
                "Due to allow".to_string(),
                filtering_stats.filtered_due_to_allow.to_string(),
            ),
            (
                "Multi-allelic".to_string(),
                filtering_stats.multi_allelic_variants.to_string(),
            ),
            (
                "Low GQ".to_string(),
                filtering_stats.low_gq_variants.to_string(),
            ),
            (
                "Missing data".to_string(),
                filtering_stats.missing_data_variants.to_string(),
            ),
        ],
    });

    log(LogLevel::Info, &format!(
        "Variant statistics: {} total, {} filtered (mask={}, allow={}, multi={}, lowGQ={}, missing={})",
        filtering_stats.total_variants,
        filtering_stats._filtered_variants,
        filtering_stats.filtered_due_to_mask,
        filtering_stats.filtered_due_to_allow,
        filtering_stats.multi_allelic_variants,
        filtering_stats.low_gq_variants,
        filtering_stats.missing_data_variants
    ));

    let sequence_length = (entry.interval.end - entry.interval.start) as i64;
    // Calculate adjusted sequence length.
    // entry.interval is ZeroBasedHalfOpen { start: S_0, end: E_0 }, representing the 0-based interval [S_0, E_0).
    // calculate_adjusted_sequence_length expects its region_start and region_end arguments to be 1-based inclusive.
    // For a 0-based half-open interval [S_0, E_0):
    //  - The 1-based inclusive start is S_0 + 1.
    //  - The 1-based inclusive end is E_0.
    // We use methods from ZeroBasedHalfOpen to get these 1-based coordinates.
    let adj_seq_len_start_1based_inclusive = entry.interval.start_1based_inclusive();
    let adj_seq_len_end_1based_inclusive = entry.interval.get_1based_inclusive_end_coord();
    let adjusted_sequence_length = calculate_adjusted_sequence_length(
        adj_seq_len_start_1based_inclusive,
        adj_seq_len_end_1based_inclusive,
        allow.as_ref().and_then(|a| a.get(&chr.to_string())),
        mask.as_ref().and_then(|m| m.get(&chr.to_string())),
    );

    log(
        LogLevel::Info,
        &format!(
            "Region length: {}bp (adjusted: {}bp after masking)",
            sequence_length, adjusted_sequence_length
        ),
    );

    log(
        LogLevel::Info,
        &format!(
            "Found {} unfiltered variants in precise region {}:{}-{}",
            region_variants_unfiltered.len(),
            chr,
            entry.interval.start,
            entry.interval.end
        ),
    );

    finish_step_progress(&format!(
        "Found {} variants in region",
        region_variants_unfiltered.len()
    ));

    #[derive(Clone)]
    struct VariantInvocation<'a> {
        group_id: u8,
        is_filtered: bool,
        variants: &'a [Variant],
        sample_filter: &'a HashMap<String, (u8, u8)>,
        maybe_adjusted_len: Option<i64>,
        position_allele_map: Arc<Mutex<HashMap<i64, (char, char)>>>,
    }

    // Set up the four analysis invocations (filtered/unfiltered × group 0/1)
    set_stage(ProcessingStage::VariantAnalysis);
    init_step_progress("Analyzing haplotype groups", 4);

    let invocations = [
        VariantInvocation {
            group_id: 0,
            is_filtered: true,
            variants: &region_variants_filtered,
            sample_filter: &entry.samples_filtered,
            maybe_adjusted_len: Some(adjusted_sequence_length),
            position_allele_map: position_allele_map.clone(),
        },
        VariantInvocation {
            group_id: 1,
            is_filtered: true,
            variants: &region_variants_filtered,
            sample_filter: &entry.samples_filtered,
            maybe_adjusted_len: Some(adjusted_sequence_length),
            position_allele_map: position_allele_map.clone(),
        },
        VariantInvocation {
            group_id: 0,
            is_filtered: false,
            variants: &region_variants_unfiltered,
            sample_filter: &entry.samples_unfiltered,
            maybe_adjusted_len: None,
            position_allele_map: position_allele_map.clone(),
        },
        VariantInvocation {
            group_id: 1,
            is_filtered: false,
            variants: &region_variants_unfiltered,
            sample_filter: &entry.samples_unfiltered,
            maybe_adjusted_len: None,
            position_allele_map: position_allele_map.clone(),
        },
    ];

    let mut results: [Option<(usize, f64, f64, usize, Vec<SiteDiversity>)>; 4] =
        [None, None, None, None];

    for (i, call) in invocations.iter().enumerate() {
        let filter_type = if call.is_filtered {
            "filtered"
        } else {
            "unfiltered"
        };
        update_step_progress(
            i as u64,
            &format!("Analyzing {} group {}", filter_type, call.group_id),
        );

        let stats_opt = process_variants(
            call.variants,
            &sample_names,
            call.group_id,
            call.sample_filter,
            entry.interval,
            extended_region,
            call.maybe_adjusted_len,
            call.position_allele_map.clone(),
            entry.seqname.clone(),
            call.is_filtered,
            ref_sequence,
            &local_cds,
        )?;

        if let Some(x) = stats_opt {
            results[i] = Some(x);
        } else {
            let label = match (call.group_id, call.is_filtered) {
                (0, true) => "filtered group 0",
                (1, true) => "filtered group 1",
                (0, false) => "unfiltered group 0",
                (1, false) => "unfiltered group 1",
                _ => "unknown scenario",
            };
            log(
                LogLevel::Warning,
                &format!(
                    "No haplotypes found for {} in region {}-{}",
                    label, entry.interval.start, entry.interval.end
                ),
            );
            // finish_step_progress("No matching haplotypes found");
            // return Ok(None);
            // Not returning here in case some of the groups succeed
        }
    }

    // After the loop, check if ANY groups had results
    if results.iter().all(|r| r.is_none()) {
        log(
            LogLevel::Warning,
            &format!(
                "No haplotypes found for any group in region {}-{}",
                entry.interval.start, entry.interval.end
            ),
        );

        // Calculate detailed statistics about why regions failed
        let mut reasons = Vec::new();
        if entry.samples_filtered.is_empty() {
            reasons.push("No filtered samples in config");
        }
        if entry.samples_unfiltered.is_empty() {
            reasons.push("No unfiltered samples in config");
        }

        if entry.seqname.contains("X") || entry.seqname.contains("x") {
            log(
                LogLevel::Warning,
                &format!(
                "DEBUG X: DROPPED chrX region {}:{}-{} - ALL groups had no haplotypes. Reasons: {}",
                entry.seqname, entry.interval.start, entry.interval.end,
                if reasons.is_empty() { "Unknown".to_string() } else { reasons.join(", ") }
            ),
            );
        }

        finish_step_progress("No matching haplotypes found");
        return Ok(None);
    }

    // Extract all results with default values for missing ones
    let (num_segsites_0_f, w_theta_0_f, pi_0_f, n_hap_0_f, site_divs_0_f) =
        results[0].take().unwrap_or((0, 0.0, 0.0, 0, Vec::new()));

    let (num_segsites_1_f, w_theta_1_f, pi_1_f, n_hap_1_f, site_divs_1_f) =
        results[1].take().unwrap_or((0, 0.0, 0.0, 0, Vec::new()));

    let (num_segsites_0_u, w_theta_0_u, pi_0_u, n_hap_0_u, site_divs_0_u) =
        results[2].take().unwrap_or((0, 0.0, 0.0, 0, Vec::new()));

    let (num_segsites_1_u, w_theta_1_u, pi_1_u, n_hap_1_u, site_divs_1_u) =
        results[3].take().unwrap_or((0, 0.0, 0.0, 0, Vec::new()));

    // Calculate inversion frequencies
    log(LogLevel::Info, "Calculating inversion frequencies");
    let inversion_freq_filt =
        calculate_inversion_allele_frequency(&entry.samples_filtered).unwrap_or(-1.0);
    let inversion_freq_no_filter =
        calculate_inversion_allele_frequency(&entry.samples_unfiltered).unwrap_or(-1.0);

    log(
        LogLevel::Info,
        &format!(
            "Inversion frequency: {:.2}% (unfiltered), {:.2}% (filtered)",
            inversion_freq_no_filter * 100.0,
            inversion_freq_filt * 100.0
        ),
    );

    // Initialize Hudson FST components for haplotype groups 0 vs 1 to None
    let mut hudson_fst_hap_group_0v1_val: Option<f64> = None;
    let mut hudson_dxy_hap_group_0v1_val: Option<f64> = None;
    let mut hudson_pi_hap_group_0_val: Option<f64> = None;
    let mut hudson_pi_hap_group_1_val: Option<f64> = None;
    let mut hudson_pi_avg_hap_group_0v1_val: Option<f64> = None;

    let mut local_regional_hudson_outcomes: Vec<RegionalHudsonFSTOutcome> = Vec::new();
    // (pos_1based, fst, hudson numerator, hudson denominator)
    let mut per_site_hudson_fst_records: Vec<(i64, f64, f64, f64)> = Vec::new();

    if args.enable_fst {
        log(
            LogLevel::Info,
            &format!(
                "Initiating Hudson FST calculations for region: {}",
                region_desc
            ),
        );

        // A. Hudson FST for Haplotype Groups (0 vs. 1) using filtered samples and variants
        let haplotypes_group_0_res =
            get_haplotype_indices_for_group(0, &entry.samples_filtered, &vcf_sample_id_to_index);
        let haplotypes_group_1_res =
            get_haplotype_indices_for_group(1, &entry.samples_filtered, &vcf_sample_id_to_index);

        if let (Ok(haplotypes_group_0), Ok(haplotypes_group_1)) =
            (haplotypes_group_0_res, haplotypes_group_1_res)
        {
            // Check if both groups have at least two haplotypes, required for `calculate_pi`.
            if haplotypes_group_0.len() >= 2 && haplotypes_group_1.len() >= 2 {
                let pop0_context = PopulationContext {
                    id: PopulationId::HaplotypeGroup(0),
                    haplotypes: haplotypes_group_0,
                    variants: variants_for_hudson_slice, // Use the correctly scoped variant slice
                    sample_names: &sample_names,
                    sequence_length: adjusted_sequence_length,
                    dense_genotypes: None,
                    dense_summary: None,
                };
                let pop1_context = PopulationContext {
                    id: PopulationId::HaplotypeGroup(1),
                    haplotypes: haplotypes_group_1,
                    variants: variants_for_hudson_slice, // Use the correctly scoped variant slice
                    sample_names: &sample_names,
                    sequence_length: adjusted_sequence_length,
                    dense_genotypes: None,
                    dense_summary: None,
                };

                if hudson_region_is_valid {
                    match calculate_hudson_fst_for_pair_with_sites(
                        &pop0_context,
                        &pop1_context,
                        hudson_query_region,
                    ) {
                        Ok((outcome, site_values)) => {
                            local_regional_hudson_outcomes.push(RegionalHudsonFSTOutcome {
                                chr: entry.seqname.clone(),
                                region_start: entry.interval.start as i64, // 0-based inclusive
                                region_end: entry.interval.get_0based_inclusive_end_coord(), // 0-based inclusive
                                outcome: outcome.clone(),
                            });
                            for site in site_values {
                                let fst_val = site.fst.unwrap_or(f64::NAN);
                                let numerator = site.num_component.unwrap_or(f64::NAN);
                                let denominator = site.den_component.unwrap_or(f64::NAN);
                                per_site_hudson_fst_records
                                    .push((site.position, fst_val, numerator, denominator));
                            }
                            hudson_fst_hap_group_0v1_val = outcome.fst;
                            hudson_dxy_hap_group_0v1_val = outcome.d_xy;
                            hudson_pi_hap_group_0_val = outcome.pi_pop1;
                            hudson_pi_hap_group_1_val = outcome.pi_pop2;
                            hudson_pi_avg_hap_group_0v1_val = outcome.pi_xy_avg;
                        }
                        Err(e) => log(LogLevel::Error, &format!("Error calculating Hudson FST for haplotype groups 0 vs 1 in region {}: {}", region_desc, e)),
                    }
                }
            } else {
                log(LogLevel::Warning, &format!("Skipping Hudson FST for haplotype groups 0 vs 1 in region {}: one or both groups have fewer than 2 haplotypes.", region_desc));
            }
        } else {
            log(LogLevel::Error, &format!("Failed to get haplotype indices for Hudson FST (haplotype groups) in region {}.", region_desc));
        }

        // B. Hudson FST for CSV-Defined Populations using filtered samples and variants
        if let Some(csv_populations_map_arc_ref) = &parsed_csv_populations_arc {
            let csv_populations_map = csv_populations_map_arc_ref.as_ref(); // Dereference Arc

            // Map population names to their actual haplotype lists (VCF index, Side) present in this VCF.
            let mut population_name_to_haplotypes_map: HashMap<
                String,
                Vec<(usize, HaplotypeSide)>,
            > = HashMap::new();
            for pop_name in csv_populations_map.keys() {
                let pop_haplotypes = get_haplotype_indices_for_csv_population(
                    pop_name,
                    csv_populations_map,
                    &vcf_sample_id_to_index,
                );
                if !pop_haplotypes.is_empty() {
                    // Only consider populations with present samples
                    population_name_to_haplotypes_map.insert(pop_name.clone(), pop_haplotypes);
                }
            }

            let mut valid_pop_names: Vec<String> =
                population_name_to_haplotypes_map.keys().cloned().collect();
            valid_pop_names.sort(); // consistent order for generating pairs

            for i in 0..valid_pop_names.len() {
                for j in (i + 1)..valid_pop_names.len() {
                    let pop_name_a = &valid_pop_names[i];
                    let pop_name_b = &valid_pop_names[j];

                    // These unwraps are safe because valid_pop_names comes from the map's keys.
                    let haplotypes_pop_a =
                        population_name_to_haplotypes_map.get(pop_name_a).unwrap();
                    let haplotypes_pop_b =
                        population_name_to_haplotypes_map.get(pop_name_b).unwrap();

                    if haplotypes_pop_a.len() >= 2 && haplotypes_pop_b.len() >= 2 {
                        let pop_a_context_csv = PopulationContext {
                            id: PopulationId::Named(pop_name_a.clone()),
                            haplotypes: haplotypes_pop_a.clone(), // Clone Vec for ownership
                            variants: variants_for_hudson_slice, // Use the correctly scoped variant slice
                            sample_names: &sample_names,
                            sequence_length: adjusted_sequence_length,
                            dense_genotypes: None,
                            dense_summary: None,
                        };
                        let pop_b_context_csv = PopulationContext {
                            id: PopulationId::Named(pop_name_b.clone()),
                            haplotypes: haplotypes_pop_b.clone(), // Clone Vec for ownership
                            variants: variants_for_hudson_slice, // Use the correctly scoped variant slice
                            sample_names: &sample_names,
                            sequence_length: adjusted_sequence_length,
                            dense_genotypes: None,
                            dense_summary: None,
                        };

                        if hudson_region_is_valid {
                            match calculate_hudson_fst_for_pair_with_sites(&pop_a_context_csv, &pop_b_context_csv, hudson_query_region) {
                                Ok((outcome, _)) => {
                                    local_regional_hudson_outcomes.push(RegionalHudsonFSTOutcome {
                                        chr: entry.seqname.clone(),
                                        region_start: entry.interval.start as i64, // entry.interval.start is 0-based inclusive start
                                        region_end: entry.interval.get_0based_inclusive_end_coord(), // get_0based_inclusive_end_coord provides the 0-based inclusive end
                                        outcome: outcome.clone(),
                                    });
                                    // No per-site Hudson TSV output for CSV-defined populations
                                }
                                Err(e) => log(LogLevel::Error, &format!("Error calculating Hudson FST for CSV populations '{}' vs '{}' in region {}: {}", pop_name_a, pop_name_b, region_desc, e)),
                            }
                        }
                    } else {
                        log(LogLevel::Warning, &format!("Skipping Hudson FST for CSV populations '{}' vs '{}' in region {}: one or both groups have fewer than 2 haplotypes.", pop_name_a, pop_name_b, region_desc));
                    }
                }
            }
        }
    }

    // Generate full inversion region alignments for each haplotype group
    log(
        LogLevel::Info,
        "Generating full inversion region alignments...",
    );
    for haplotype_group in [0u8, 1u8] {
        generate_full_region_alignment(
            &entry,
            haplotype_group,
            &region_variants_filtered,
            &sample_names,
            ref_sequence,
            &position_allele_map,
            mask,
            allow,
            &vcf_sample_id_to_index,
        )?;
    }

    // This is the FstEstimate enum for overall haplotype W&C FST
    let haplotype_overall_fst_estimate_enum = fst_results_filtered.as_ref().map_or(
        crate::stats::FstEstimate::InsufficientDataForEstimation {
            sum_a: 0.0,
            sum_b: 0.0,
            sites_attempted: 0,
        },
        |res| res.overall_fst,
    );

    // Extract components for CSV output
    let (hap_fst_val, hap_sum_a, hap_sum_b, hap_num_sites) =
        crate::stats::extract_wc_fst_components(&haplotype_overall_fst_estimate_enum);

    let row_data = CsvRowData {
        seqname: entry.seqname.clone(),
        region_start: entry.interval.start_1based_inclusive(), // Represents the 1-based inclusive start of the entry interval
        region_end: entry.interval.get_1based_inclusive_end_coord(), // Represents the 1-based inclusive end of the entry interval
        seq_len_0: sequence_length,
        seq_len_1: sequence_length,
        seq_len_adj_0: adjusted_sequence_length,
        seq_len_adj_1: adjusted_sequence_length,
        seg_sites_0: num_segsites_0_u,
        seg_sites_1: num_segsites_1_u,
        w_theta_0: w_theta_0_u,
        w_theta_1: w_theta_1_u,
        pi_0: pi_0_u,
        pi_1: pi_1_u,
        seg_sites_0_f: num_segsites_0_f,
        seg_sites_1_f: num_segsites_1_f,
        w_theta_0_f,
        w_theta_1_f,
        pi_0_f,
        pi_1_f,
        n_hap_0_unf: n_hap_0_u,
        n_hap_1_unf: n_hap_1_u,
        n_hap_0_f,
        n_hap_1_f,
        inv_freq_no_filter: inversion_freq_no_filter,
        inv_freq_filter: inversion_freq_filt,
        // Weir & Cockerham FST components for haplotype groups
        haplotype_overall_fst_wc: hap_fst_val,
        haplotype_between_pop_variance_wc: hap_sum_a,
        haplotype_within_pop_variance_wc: hap_sum_b,
        haplotype_num_informative_sites_wc: hap_num_sites,
        // Hudson FST components for haplotype groups
        hudson_fst_hap_group_0v1: hudson_fst_hap_group_0v1_val,
        hudson_dxy_hap_group_0v1: hudson_dxy_hap_group_0v1_val,
        hudson_pi_hap_group_0: hudson_pi_hap_group_0_val,
        hudson_pi_hap_group_1: hudson_pi_hap_group_1_val,
        hudson_pi_avg_hap_group_0v1: hudson_pi_avg_hap_group_0v1_val,
    };

    // Display summary of results
    display_status_box(StatusBox {
        title: format!("Results for {}", region_desc),
        stats: vec![
            (
                "Unfiltered θ Group 0".to_string(),
                format!("{:.6}", w_theta_0_u),
            ),
            (
                "Unfiltered θ Group 1".to_string(),
                format!("{:.6}", w_theta_1_u),
            ),
            ("Unfiltered π Group 0".to_string(), format!("{:.6}", pi_0_u)),
            ("Unfiltered π Group 1".to_string(), format!("{:.6}", pi_1_u)),
            (
                "Filtered θ Group 0".to_string(),
                format!("{:.6}", w_theta_0_f),
            ),
            (
                "Filtered θ Group 1".to_string(),
                format!("{:.6}", w_theta_1_f),
            ),
            ("Filtered π Group 0".to_string(), format!("{:.6}", pi_0_f)),
            ("Filtered π Group 1".to_string(), format!("{:.6}", pi_1_f)),
            (
                "Inversion Frequency".to_string(),
                format!("{:.2}%", inversion_freq_filt * 100.0),
            ),
        ],
    });

    finish_step_progress(&format!("Completed analysis for {}", region_desc));

    // Collect per-site diversity records
    log(LogLevel::Info, "Collecting per-site diversity statistics");
    let mut per_site_diversity_records = Vec::new();
    for sd in site_divs_0_u {
        per_site_diversity_records.push((sd.position, sd.pi, sd.watterson_theta, 0, false));
    }
    for sd in site_divs_1_u {
        per_site_diversity_records.push((sd.position, sd.pi, sd.watterson_theta, 1, false));
    }
    for sd in site_divs_0_f {
        per_site_diversity_records.push((sd.position, sd.pi, sd.watterson_theta, 0, true));
    }
    for sd in site_divs_1_f {
        per_site_diversity_records.push((sd.position, sd.pi, sd.watterson_theta, 1, true));
    }

    // Collect per-site FST records specifically for the haplotype group analysis (0 vs. 1)
    // for summary FALSTA output.
    // Stores (1-based position, Overall_Haplotype_FstEstimate_as_f64, Pairwise_0v1_Haplotype_FstEstimate_as_f64).
    let mut per_site_fst_records: Vec<(i64, f64, f64)> = Vec::new();

    if let Some(fst_results_hap_groups) = &fst_results_filtered {
        // fst_results_hap_groups is &FstWcResults
        log(
            LogLevel::Info,
            &format!(
                "Collecting per-site FST data for haplotype groups (0 vs 1) for region {}",
                region_desc
            ),
        );
        for site_fst_wc in &fst_results_hap_groups.site_fst {
            // site_fst_wc is &SiteFstWc
            let overall_site_hap_fst_val = match site_fst_wc.overall_fst {
                crate::stats::FstEstimate::Calculable { value, .. } => value, // Extract the value field
                _ => f64::NAN, // Use NaN for other FstEstimate variants.
            };
            let pairwise_0_vs_1_hap_fst_val = match site_fst_wc.pairwise_fst.get("0_vs_1") {
                Some(&crate::stats::FstEstimate::Calculable { value, .. }) => value, // Extract the value field
                _ => f64::NAN, // Use NaN if the pair is not found or its FstEstimate is not Calculable.
            };
            per_site_fst_records.push((
                site_fst_wc.position,
                overall_site_hap_fst_val,
                pairwise_0_vs_1_hap_fst_val,
            ));
        }
    }

    // Note: Per-site FST data from CSV-defined populations (pop_fst_results_filtered)
    // is NOT added to this specific `per_site_fst_records` vector.
    // That data is fully contained within `row_data.population_fst_wc_results`
    // and is used directly in `process_config_entries` for detailed per-population-pair FALSTA output.
    // The overall FST for CSV populations is in `row_data.population_overall_fst_wc`.

    log(LogLevel::Info, &format!(
        "Collected {} per-site diversity records and {} per-site haplotype FST summary records for {}",
        per_site_diversity_records.len(), per_site_fst_records.len(), region_desc
    ));

    Ok(Some((
        row_data,
        per_site_diversity_records,
        per_site_fst_records, // Vec<(i64, FstEstimate, FstEstimate)> containing only haplotype group FSTs
        per_site_hudson_fst_records, // Vec<(i64, fst, numerator, denominator)> for Hudson haplotype groups
        local_regional_hudson_outcomes,
    )))
}

/// Helper struct to associate a HudsonFSTOutcome with its genomic region.
/// Used internally for collecting results before writing to file.
#[derive(Debug, Clone)]
struct RegionalHudsonFSTOutcome {
    chr: String,
    region_start: i64, // 0-based inclusive, from ConfigEntry.interval.start
    region_end: i64,   // 0-based inclusive, from ConfigEntry.interval.end
    outcome: HudsonFSTOutcome,
}

/// Formats an Option<PopulationId> into type and name strings for output.
fn format_population_id(pop_id_opt: &Option<PopulationId>) -> (String, String) {
    match pop_id_opt {
        Some(PopulationId::HaplotypeGroup(g)) => ("HaplotypeGroup".to_string(), g.to_string()),
        Some(PopulationId::Named(n)) => ("NamedPopulation".to_string(), n.clone()),
        None => ("NA".to_string(), "NA".to_string()),
    }
}

/// Formats an Option<f64> to a string, representing None or NaN as "NA".
/// Floating point values are formatted to six decimal places.
fn format_optional_float(val_opt: Option<f64>) -> String {
    match val_opt {
        Some(f) => {
            if f.is_nan() {
                "NA".to_string()
            } else {
                format!("{:.6}", f)
            }
        }
        None => "NA".to_string(),
    }
}

/// Formats an Option<usize> to a string, representing None as "NA".
fn format_optional_usize(val_opt: Option<usize>) -> String {
    match val_opt {
        Some(u) => u.to_string(),
        None => "NA".to_string(),
    }
}

// ── shared append opener ──────────────────────────────────────────────────────
fn open_append(path: &std::path::Path) -> std::io::Result<BufWriter<std::fs::File>> {
    let f = OpenOptions::new().create(true).append(true).open(path)?;
    Ok(BufWriter::new(f))
}

// ── csv append: write ONE row (no header) ─────────────────────────────────────
fn append_csv_row(csv_path: &std::path::Path, row: &CsvRowData) -> Result<(), VcfError> {
    let f = open_append(csv_path).map_err(VcfError::Io)?;
    let mut w = csv::WriterBuilder::new().has_headers(false).from_writer(f);
    write_csv_row(&mut w, row)?;
    w.flush().map_err(|e| VcfError::Io(e.into()))
}

// ── FALSTA helpers (use your existing header/value formats) ───────────────────
fn build_fasta_header(prefix: &str, row: &CsvRowData, group_id: u8) -> String {
    format!(
        ">{}chr_{}_start_{}_end_{}_group_{}",
        prefix, row.seqname, row.region_start, row.region_end, group_id
    )
}

// per-site diversity: (pos_1based, pi, theta, group_id, is_filtered)
fn append_diversity_falsta<P: AsRef<std::path::Path>>(
    path: P,
    row: &CsvRowData,
    per_site: &[(i64, f64, f64, u8, bool)],
) -> Result<(), VcfError> {
    let mut w = open_append(path.as_ref()).map_err(VcfError::Io)?;
    // region in 0-based half-open for mapping
    let region = ZeroBasedHalfOpen::from_1based_inclusive(row.region_start, row.region_end);
    let region_len = region.len();

    // group ids present
    let mut gids = std::collections::BTreeSet::<u8>::new();
    for &(_, _, _, g, _) in per_site {
        gids.insert(g);
    }

    // for each group × {unfiltered,filtered} × {pi,theta}
    for &g in &gids {
        for &(is_filtered, which, prefix) in &[
            (false, "pi", "unfiltered_pi_"),
            (false, "theta", "unfiltered_theta_"),
            (true, "pi", "filtered_pi_"),
            (true, "theta", "filtered_theta_"),
        ] {
            // allocate a line of NAs
            let mut line = vec![String::from("NA"); region_len];
            let mut any = false;

            for &(pos1, pi, th, gg, filt) in per_site {
                if gg != g || filt != is_filtered {
                    continue;
                }
                if let Some(rel1) = region.relative_position_1based_inclusive(pos1) {
                    let idx = (rel1 - 1) as usize;
                    let v = if which == "pi" { pi } else { th };
                    line[idx] = if v == 0.0 {
                        "0".into()
                    } else {
                        format!("{:.6}", v)
                    };
                    any = true;
                }
            }

            if any {
                writeln!(w, "{}", build_fasta_header(prefix, row, g)).map_err(VcfError::Io)?;
                writeln!(w, "{}", line.join(",")).map_err(VcfError::Io)?;
            }
        }
    }
    w.flush().map_err(VcfError::Io)
}

// per-site WC FST: (pos_1based, overall_wc, pairwise_0v1_wc)
// and Hudson hap FST components: (pos_1based, fst, numerator, denominator)
fn append_fst_falsta<P: AsRef<std::path::Path>>(
    path: P,
    row: &CsvRowData,
    wc_sites: &[(i64, f64, f64)],
    hudson_sites: &[(i64, f64, f64, f64)],
) -> Result<(), VcfError> {
    let mut w = open_append(path.as_ref()).map_err(VcfError::Io)?;
    let region = ZeroBasedHalfOpen::from_1based_inclusive(row.region_start, row.region_end);
    let n = region.len();

    // WC overall
    if !wc_sites.is_empty() {
        writeln!(
            w,
            ">haplotype_overall_fst_summary_chr_{}_start_{}_end_{}",
            row.seqname, row.region_start, row.region_end
        )
        .map_err(VcfError::Io)?;
        let mut v = vec![String::from("NA"); n];
        for &(p1, ovl, _) in wc_sites {
            if let Some(rel1) = region.relative_position_1based_inclusive(p1) {
                let i = (rel1 - 1) as usize;
                v[i] = if ovl.is_nan() {
                    "NA".into()
                } else {
                    format!("{:.6}", ovl)
                };
            }
        }
        writeln!(w, "{}", v.join(",")).map_err(VcfError::Io)?;

        // WC pairwise 0v1
        writeln!(
            w,
            ">haplotype_0v1_pairwise_fst_summary_chr_{}_start_{}_end_{}",
            row.seqname, row.region_start, row.region_end
        )
        .map_err(VcfError::Io)?;
        let mut pv = vec![String::from("NA"); n];
        for &(p1, _, pair) in wc_sites {
            if let Some(rel1) = region.relative_position_1based_inclusive(p1) {
                let i = (rel1 - 1) as usize;
                pv[i] = if pair.is_nan() {
                    "NA".into()
                } else {
                    format!("{:.6}", pair)
                };
            }
        }
        writeln!(w, "{}", pv.join(",")).map_err(VcfError::Io)?;
    }

    // Hudson per-site hap FST 0v1
    if !hudson_sites.is_empty() {
        let format_value = |value: f64| -> String {
            if value.is_nan() {
                "NA".into()
            } else if value.is_infinite() {
                if value.is_sign_positive() {
                    "Infinity".into()
                } else {
                    "-Infinity".into()
                }
            } else if value == 0.0 {
                "0".into()
            } else {
                format!("{:.6}", value)
            }
        };

        writeln!(
            w,
            ">hudson_pairwise_fst_hap_0v1_chr_{}_start_{}_end_{}",
            row.seqname, row.region_start, row.region_end
        )
        .map_err(VcfError::Io)?;
        let mut hv = vec![String::from("NA"); n];
        for &(p1, fst, _, _) in hudson_sites {
            if let Some(rel1) = region.relative_position_1based_inclusive(p1) {
                let i = (rel1 - 1) as usize;
                hv[i] = format_value(fst);
            }
        }
        writeln!(w, "{}", hv.join(",")).map_err(VcfError::Io)?;

        writeln!(
            w,
            ">hudson_pairwise_fst_hap_0v1_numerator_chr_{}_start_{}_end_{}",
            row.seqname, row.region_start, row.region_end
        )
        .map_err(VcfError::Io)?;
        let mut numerators = vec![String::from("NA"); n];
        for &(p1, _, numerator, _) in hudson_sites {
            if let Some(rel1) = region.relative_position_1based_inclusive(p1) {
                let i = (rel1 - 1) as usize;
                numerators[i] = format_value(numerator);
            }
        }
        writeln!(w, "{}", numerators.join(",")).map_err(VcfError::Io)?;

        writeln!(
            w,
            ">hudson_pairwise_fst_hap_0v1_denominator_chr_{}_start_{}_end_{}",
            row.seqname, row.region_start, row.region_end
        )
        .map_err(VcfError::Io)?;
        let mut denominators = vec![String::from("NA"); n];
        for &(p1, _, _, denominator) in hudson_sites {
            if let Some(rel1) = region.relative_position_1based_inclusive(p1) {
                let i = (rel1 - 1) as usize;
                denominators[i] = format_value(denominator);
            }
        }
        writeln!(w, "{}", denominators.join(",")).map_err(VcfError::Io)?;
    }

    w.flush().map_err(VcfError::Io)
}

// write Hudson TSV rows (the “regional outcomes”)
fn append_hudson_tsv(
    out_path: &std::path::Path,
    rows: &[RegionalHudsonFSTOutcome],
) -> Result<(), VcfError> {
    let f = open_append(out_path).map_err(VcfError::Io)?;
    let mut w = csv::WriterBuilder::new()
        .delimiter(b'\t')
        .has_headers(false)
        .from_writer(f);

    for r in rows {
        let (p1t, p1n) = format_population_id(&r.outcome.pop1_id);
        let (p2t, p2n) = format_population_id(&r.outcome.pop2_id);
        w.write_record(&[
            &r.chr,
            &r.region_start.to_string(),
            &r.region_end.to_string(),
            &p1t,
            &p1n,
            &p2t,
            &p2n,
            &format_optional_float(r.outcome.d_xy),
            &format_optional_float(r.outcome.pi_pop1),
            &format_optional_float(r.outcome.pi_pop2),
            &format_optional_float(r.outcome.pi_xy_avg),
            &format_optional_float(r.outcome.fst),
        ])
        .map_err(|e| VcfError::Io(e.into()))?;
    }
    w.flush().map_err(|e| VcfError::Io(e.into()))
}

/// Retrieves VCF sample indices and HaplotypeSides for samples belonging to a specified population
/// as defined in a population definition CSV file.
///
/// # Arguments
/// * `pop_name` - The name of the population to retrieve haplotypes for.
/// * `parsed_csv_populations` - A map where keys are population names and values are lists of sample IDs (from CSV).
/// * `vcf_sample_id_to_index` - A map where keys are core VCF sample IDs and values are their 0-based VCF column indices.
///
/// # Returns
/// A vector of tuples, where each tuple is `(vcf_sample_index, HaplotypeSide)`.
/// Both Left and Right haplotypes are included for each matched sample.
fn get_haplotype_indices_for_csv_population(
    pop_name: &str,
    parsed_csv_populations: &HashMap<String, Vec<String>>,
    vcf_sample_id_to_index: &HashMap<String, usize>,
) -> Vec<(usize, HaplotypeSide)> {
    let mut haplotype_indices = Vec::new();

    if let Some(sample_ids_for_pop) = parsed_csv_populations.get(pop_name) {
        for csv_sample_id in sample_ids_for_pop {
            // The vcf_sample_id_to_index keys are core VCF sample IDs (e.g., "NA12878").
            // The csv_sample_id is also expected to be a core sample ID.
            if let Some(&vcf_idx) = vcf_sample_id_to_index.get(csv_sample_id) {
                haplotype_indices.push((vcf_idx, HaplotypeSide::Left));
                haplotype_indices.push((vcf_idx, HaplotypeSide::Right));
            } else {
                log(LogLevel::Warning, &format!(
                    "Sample '{}' defined for population '{}' in CSV not found in VCF sample list. This sample will be skipped for population '{}'.",
                    csv_sample_id, pop_name, pop_name
                ));
            }
        }
    } else {
        // This case should ideally not be reached if pop_name is derived from parsed_csv_populations.keys()
        log(LogLevel::Warning, &format!(
            "Population name '{}' was queried but not found in parsed CSV definitions during haplotype gathering.",
            pop_name
        ));
    }
    haplotype_indices
}

// Function to process a VCF file
pub fn process_vcf(
    file: &Path,
    reference_path: &Path,
    chr: String,
    region: ZeroBasedHalfOpen,
    min_gq: u16,
    mask_regions: Option<Arc<HashMap<String, Vec<(i64, i64)>>>>,
    allow_regions: Option<Arc<HashMap<String, Vec<(i64, i64)>>>>,
    position_allele_map: Arc<Mutex<HashMap<i64, (char, char)>>>,
) -> Result<
    (
        Vec<Variant>,
        Vec<usize>,
        Vec<String>,
        i64,
        MissingDataInfo,
        FilteringStats,
    ),
    VcfError,
> {
    set_stage(ProcessingStage::VcfProcessing);
    log(
        LogLevel::Info,
        &format!(
            "Processing VCF file {} for chr{}:{}-{}",
            file.display(),
            chr,
            region.start,
            region.end
        ),
    );

    // Initialize the VCF reader
    let mut reader = open_vcf_reader(file)?;
    let mut sample_names = Vec::new();

    // Get chromosome length from reference
    let chr_length = {
        let fasta_reader = bio::io::fasta::IndexedReader::from_file(&reference_path)
            .map_err(|e| VcfError::Parse(e.to_string()))?;
        let sequences = fasta_reader.index.sequences().to_vec();
        let seq_info = sequences
            .iter()
            .find(|seq| seq.name == chr || seq.name == format!("chr{}", chr))
            .ok_or_else(|| VcfError::Parse(format!("Chromosome {} not found in reference", chr)))?;
        seq_info.len as i64
    };
    // Log without terminal message to reduce spam
    log(
        LogLevel::Info,
        &format!("Chromosome {} length: {}bp", chr, chr_length),
    );

    // Small vectors to hold variants in batches, limiting memory usage
    let all_variants = Arc::new(Mutex::new(Vec::with_capacity(10000)));
    let filtered_idxs = Arc::new(Mutex::new(Vec::<usize>::with_capacity(10000)));

    // Shared stats
    let missing_data_info = Arc::new(Mutex::new(MissingDataInfo::default()));
    let _filtering_stats = Arc::new(Mutex::new(FilteringStats::default()));

    // Create a shared progress bar that plays nicely with multi-threaded rendering
    let is_gzipped = file.extension().and_then(|s| s.to_str()) == Some("gz");
    let total_bytes = if is_gzipped {
        None
    } else {
        Some(fs::metadata(file)?.len())
    };
    let progress_message = format!("Reading VCF for chr{}:{}-{}", chr, region.start, region.end);
    let progress_bar = Arc::new(create_vcf_progress(total_bytes, &progress_message));

    let processing_complete = Arc::new(AtomicBool::new(false));
    let processing_complete_clone = Arc::clone(&processing_complete);
    let progress_bar_clone = Arc::clone(&progress_bar); // Clone Arc for progress thread
    let finish_message = format!(
        "Finished reading VCF for chr{}:{}-{}",
        chr, region.start, region.end
    );
    let progress_thread = thread::spawn(move || {
        while !processing_complete_clone.load(Ordering::Relaxed) {
            // Less frequent updates to prevent overprinting
            thread::sleep(Duration::from_millis(200));
        }
        // Finish with a clean message once processing is complete
        progress_bar_clone.finish_with_message(finish_message);
    });

    // Parse header lines.
    let mut buffer = String::new();
    while reader.read_line(&mut buffer)? > 0 {
        if buffer.starts_with("##") {
        } else if buffer.starts_with("#CHROM") {
            validate_vcf_header(&buffer)?;
            sample_names = buffer
                .split_whitespace()
                .skip(9)
                .map(String::from)
                .collect();
            break;
        }
        buffer.clear();
    }
    buffer.clear();

    // Bounded channels for lines and results.
    let (line_sender, line_receiver) = bounded(2000);
    let (result_sender, result_receiver) = bounded(2000);

    // Producer for reading lines from VCF.
    let producer_thread = thread::spawn({
        let mut local_buffer = String::new();
        let mut local_reader = reader;
        move || -> Result<(), VcfError> {
            while local_reader.read_line(&mut local_buffer)? > 0 {
                line_sender
                    .send(local_buffer.clone())
                    .map_err(|_| VcfError::ChannelSend)?;
                if total_bytes.is_some() {
                    progress_bar.inc(local_buffer.len() as u64);
                }
                local_buffer.clear();
            }
            drop(line_sender);
            Ok(())
        }
    });

    // Consumers for variant lines.
    let num_threads = num_cpus::get();
    let arc_sample_names = Arc::new(sample_names);
    let mut consumers = Vec::with_capacity(num_threads);
    for _ in 0..num_threads {
        let line_receiver = line_receiver.clone();
        let rs = result_sender.clone();
        let arc_names = Arc::clone(&arc_sample_names);
        let arc_mask = mask_regions.clone();
        let arc_allow = allow_regions.clone();
        let chr_copy = chr.to_string();
        let pos_map = Arc::clone(&position_allele_map);
        consumers.push(thread::spawn(move || -> Result<(), VcfError> {
            while let Ok(line) = line_receiver.recv() {
                let mut single_line_miss_info = MissingDataInfo::default();
                let mut single_line_filt_stats = FilteringStats::default();

                // Construct a ZeroBasedHalfOpen region for the consumer thread
                let consumer_region = ZeroBasedHalfOpen {
                    start: region.start,
                    end: region.end,
                };

                match process_variant(
                    &line,
                    &chr_copy,
                    consumer_region,
                    &mut single_line_miss_info,
                    &arc_names,
                    min_gq,
                    &mut single_line_filt_stats,
                    arc_allow.as_ref().map(|x| x.as_ref()),
                    arc_mask.as_ref().map(|x| x.as_ref()),
                    &pos_map,
                ) {
                    Ok(variant_opt) => {
                        rs.send(Ok((
                            variant_opt,
                            single_line_miss_info.clone(),
                            single_line_filt_stats.clone(),
                        )))
                        .map_err(|_| VcfError::ChannelSend)?;
                    }
                    Err(e) => {
                        rs.send(Err(e)).map_err(|_| VcfError::ChannelSend)?;
                    }
                }
            }
            Ok(())
        }));
    }

    // Collector merges results from consumers.
    let collector_thread = thread::spawn({
        let all_variants = Arc::clone(&all_variants);
        let filtered_idxs = Arc::clone(&filtered_idxs);
        let missing_data_info = Arc::clone(&missing_data_info);
        let _filtering_stats = Arc::clone(&_filtering_stats);
        move || -> Result<(), VcfError> {
            while let Ok(msg) = result_receiver.recv() {
                match msg {
                    Ok((Some((variant, passes)), local_miss, mut local_stats)) => {
                        let mut u = all_variants.lock();
                        u.push(variant); // single owner of the heavy genotypes
                        let idx = u.len() - 1;
                        if passes {
                            filtered_idxs.lock().push(idx); // just remember the index
                        }

                        {
                            let mut global_miss = missing_data_info.lock();
                            global_miss.total_data_points += local_miss.total_data_points;
                            global_miss.missing_data_points += local_miss.missing_data_points;
                            global_miss
                                .positions_with_missing
                                .extend(local_miss.positions_with_missing);
                        }
                        {
                            let mut gs = _filtering_stats.lock();
                            gs.total_variants += local_stats.total_variants;
                            gs._filtered_variants += local_stats._filtered_variants;
                            gs.filtered_positions.extend(local_stats.filtered_positions);
                            gs.filtered_due_to_mask += local_stats.filtered_due_to_mask;
                            gs.filtered_due_to_allow += local_stats.filtered_due_to_allow;
                            gs.missing_data_variants += local_stats.missing_data_variants;
                            gs.low_gq_variants += local_stats.low_gq_variants;
                            gs.multi_allelic_variants += local_stats.multi_allelic_variants;
                            for ex in local_stats.filtered_examples.drain(..) {
                                gs.add_example(ex);
                            }
                        }
                    }

                    Ok((None, local_miss, mut local_stats)) => {
                        let mut global_miss = missing_data_info.lock();
                        global_miss.total_data_points += local_miss.total_data_points;
                        global_miss.missing_data_points += local_miss.missing_data_points;
                        global_miss
                            .positions_with_missing
                            .extend(local_miss.positions_with_missing);
                        let mut gs = _filtering_stats.lock();
                        gs.total_variants += local_stats.total_variants;
                        gs._filtered_variants += local_stats._filtered_variants;
                        gs.filtered_positions.extend(local_stats.filtered_positions);
                        gs.filtered_due_to_mask += local_stats.filtered_due_to_mask;
                        gs.filtered_due_to_allow += local_stats.filtered_due_to_allow;
                        gs.missing_data_variants += local_stats.missing_data_variants;
                        gs.low_gq_variants += local_stats.low_gq_variants;
                        gs.multi_allelic_variants += local_stats.multi_allelic_variants;
                        for ex in local_stats.filtered_examples.drain(..) {
                            gs.add_example(ex);
                        }
                    }
                    Err(e) => {
                        eprintln!("{}", e);
                    }
                }
            }
            Ok(())
        }
    });

    // Wait for producer.
    producer_thread.join().expect("Producer thread panicked")?;
    // Wait for consumers.
    drop(line_receiver);
    drop(result_sender);
    for c in consumers {
        c.join().expect("Consumer thread panicked")?;
    }
    // Signal done, wait for collector.
    processing_complete.store(true, Ordering::Relaxed);
    collector_thread
        .join()
        .expect("Collector thread panicked")?;
    progress_thread.join().expect("Progress thread panicked");

    // Extract final variant vectors.
    let mut final_all = Arc::try_unwrap(all_variants)
        .map_err(|_| VcfError::Parse("Variants still have multiple owners".to_string()))?
        .into_inner();
    let mut final_filtered_idxs = Arc::try_unwrap(filtered_idxs)
        .map_err(|_| VcfError::Parse("Filtered variants still have multiple owners".to_string()))?
        .into_inner();

    // build a set of filtered positions (single-alt enforced, so unique per pos)
    let filt_pos: std::collections::HashSet<i64> = final_filtered_idxs
        .iter()
        .map(|&i| final_all[i].position)
        .collect();

    // sort the all-variants in place by position
    final_all.sort_by_key(|v| v.position);

    // rebuild filtered indices in the new order
    final_filtered_idxs = final_all
        .iter()
        .enumerate()
        .filter_map(|(i, v)| {
            if filt_pos.contains(&v.position) {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    // Extract stats.
    let final_miss = Arc::try_unwrap(missing_data_info)
        .map_err(|_| VcfError::Parse("Missing data info still has multiple owners".to_string()))?
        .into_inner();
    let final_stats = Arc::try_unwrap(_filtering_stats)
        .map_err(|_| VcfError::Parse("Filtering stats still have multiple owners".to_string()))?
        .into_inner();
    let final_names = Arc::try_unwrap(arc_sample_names)
        .map_err(|_| VcfError::Parse("Sample names have multiple owners".to_string()))?;

    log(
        LogLevel::Info,
        &format!(
            "VCF processing complete for chr{}:{}-{}: {} variants loaded, {} filtered",
            chr,
            region.start,
            region.end,
            final_all.len(),
            final_filtered_idxs.len()
        ),
    );

    if chr.contains("X") || chr.contains("x") {
        log(
            LogLevel::Info,
            &format!(
                "DEBUG X: chrX VCF processing complete with {} unfiltered and {} filtered variants",
                final_all.len(),
                final_filtered_idxs.len()
            ),
        );
    }

    log(
        LogLevel::Info,
        &format!(
            "VCF statistics: missing data points: {}/{} ({:.2}%)",
            final_miss.missing_data_points,
            final_miss.total_data_points,
            (final_miss.missing_data_points as f64 / final_miss.total_data_points.max(1) as f64)
                * 100.0
        ),
    );

    Ok((
        final_all,
        final_filtered_idxs,
        final_names,
        chr_length,
        final_miss,
        final_stats,
    ))
}

pub fn process_variant(
    line: &str,
    chr: &str,
    region: ZeroBasedHalfOpen,
    missing_data_info: &mut MissingDataInfo,
    sample_names: &[String],
    min_gq: u16,
    filtering_stats: &mut FilteringStats,
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

    // parse a 1-based VCF position
    let one_based_vcf_position = OneBasedPosition::new(
        fields[1]
            .parse()
            .map_err(|_| VcfError::Parse("Invalid position".to_string()))?,
    )?;

    // call region.contains using zero-based
    if !region.contains(ZeroBasedPosition(one_based_vcf_position.zero_based())) {
        return Ok(None);
    }

    filtering_stats.total_variants += 1; // DO NOT MOVE THIS LINE ABOVE THE CHECK FOR WITHIN RANGE

    // Only variants within the range get passed the collector which increments statistics.
    // For variants outside the range, the consumer thread does not send any result to the collector.
    // If this line is moved above the early return return Ok(None) in the range check, then it would increment all variants, not just those in the regions
    // This would mean that the maximum number of variants filtered could be below the maximum number of variants,
    // in the case that there are variants outside of the ranges (which would not even get far enough to need to be filtered, but would be included in the total).
    //
    // Only variants within the range get counted toward totals.
    // For variants outside the range, we return early.

    let zero_based_position = one_based_vcf_position.zero_based(); // Zero-based coordinate

    // Check allow regions
    if let Some(allow_regions_chr) = allow_regions.and_then(|ar| ar.get(vcf_chr)) {
        if !position_in_regions(zero_based_position, allow_regions_chr) {
            filtering_stats._filtered_variants += 1;
            filtering_stats.filtered_due_to_allow += 1;
            filtering_stats
                .filtered_positions
                .insert(zero_based_position);
            filtering_stats.add_example(format!("{}: Filtered due to allow", line.trim()));
            return Ok(None);
        }
    } else if allow_regions.is_some() {
        filtering_stats._filtered_variants += 1;
        filtering_stats.filtered_due_to_allow += 1;
        filtering_stats
            .filtered_positions
            .insert(zero_based_position);
        filtering_stats.add_example(format!("{}: Filtered due to allow", line.trim()));
        return Ok(None);
    }

    // Check mask regions
    if let Some(mask_regions_chr) = mask_regions.and_then(|mr| mr.get(vcf_chr)) {
        // Create a ZeroBasedHalfOpen point interval for the position
        let position_interval = ZeroBasedHalfOpen {
            start: zero_based_position as usize,
            end: (zero_based_position + 1) as usize,
        };

        // Check if position is masked using the ZeroBasedHalfOpen type
        let is_masked = mask_regions_chr.iter().any(|&(start, end)| {
            let mask_interval = ZeroBasedHalfOpen {
                start: start as usize,
                end: end as usize,
            };
            position_interval.intersect(&mask_interval).is_some()
        });

        if is_masked {
            filtering_stats._filtered_variants += 1;
            filtering_stats.filtered_due_to_mask += 1;
            filtering_stats
                .filtered_positions
                .insert(zero_based_position);
            filtering_stats.add_example(format!("{}: Filtered due to mask", line.trim()));
            return Ok(None);
        }
    } else if mask_regions.is_some() {
        // Chromosome not found in mask regions, but mask was provided
        // This is a warning condition - the chromosome exists in the VCF but not in the mask
        eprintln!(
            "{}",
            format!(
                "Warning: Chromosome {} not found in mask file. No positions will be masked for this chromosome.",
                vcf_chr
            ).yellow()
        );
    }

    // Store reference and alternate alleles
    if !fields[3].is_empty() && !fields[4].is_empty() {
        let ref_allele = match fields[3].chars().next().unwrap_or('N') {
            'A' | 'a' => 'A',
            'C' | 'c' => 'C',
            'G' | 'g' => 'G',
            'T' | 't' => 'T',
            _ => 'N',
        };
        let alt_allele = match fields[4].chars().next().unwrap_or('N') {
            'A' | 'a' => 'A',
            'C' | 'c' => 'C',
            'G' | 'g' => 'G',
            'T' | 't' => 'T',
            _ => 'N',
        };
        position_allele_map
            .lock()
            .insert(zero_based_position, (ref_allele, alt_allele));
    }

    let alt_alleles: Vec<&str> = fields[4].split(',').collect();
    let is_multiallelic = alt_alleles.len() > 1;
    if is_multiallelic {
        filtering_stats.multi_allelic_variants += 1;
        eprintln!(
            "{}",
            format!(
                "Warning: Multi-allelic site detected at position {}, which is not supported. Skipping.",
                one_based_vcf_position.0
            ).yellow()
        );
        filtering_stats.add_example(format!(
            "{}: Filtered due to multi-allelic variant",
            line.trim()
        ));
        return Ok(None);
    }
    if alt_alleles.get(0).map_or(false, |s| s.len() > 1) {
        filtering_stats.multi_allelic_variants += 1;
        eprintln!(
            "{}",
            format!(
                "Warning: Multi-nucleotide ALT detected at position {}, which is not supported. Skipping.",
                one_based_vcf_position.0
            ).yellow()
        );
        filtering_stats.add_example(format!(
            "{}: Filtered due to multi-nucleotide alt allele",
            line.trim()
        ));
        return Ok(None);
    }

    let format_fields: Vec<&str> = fields[8].split(':').collect();
    let gq_index = format_fields.iter().position(|&s| s == "GQ");
    if gq_index.is_none() {
        return Err(VcfError::Parse("GQ field not found in FORMAT".to_string()));
    }
    let gq_index = gq_index.unwrap();

    let genotypes: Vec<Option<Vec<u8>>> = fields[9..]
        .iter()
        .map(|gt| {
            missing_data_info.total_data_points += 1;
            let alleles_str = gt.split(':').next().unwrap_or(".");
            if alleles_str == "." || alleles_str == "./." || alleles_str == ".|." {
                missing_data_info.missing_data_points += 1;
                missing_data_info
                    .positions_with_missing
                    .insert(zero_based_position);
                return None;
            }
            let alleles = alleles_str
                .split(|c| c == '|' || c == '/')
                .map(|allele| allele.parse::<u8>().ok())
                .collect::<Option<Vec<u8>>>();
            if alleles.is_none() {
                missing_data_info.missing_data_points += 1;
                missing_data_info
                    .positions_with_missing
                    .insert(zero_based_position);
            }
            alleles
        })
        .collect();

    let mut sample_has_low_gq = false;
    for gt_field in fields[9..].iter() {
        let gt_subfields: Vec<&str> = gt_field.split(':').collect();
        if gq_index >= gt_subfields.len() {
            return Err(VcfError::Parse(format!(
                "GQ value missing in sample genotype field at chr{}:{}",
                chr, one_based_vcf_position.0
            )));
        }
        let gq_str = gt_subfields[gq_index].trim();

        // Attempt to parse GQ value as u16
        // Parse GQ value, treating '.' or empty string as 0
        // If you have no GQ, we treat as GQ=0 → (probably) filtered out.
        let gq_value: u16 = match gq_str {
            "." | "" => 0,
            _ => match gq_str.parse() {
                Ok(val) => val,
                Err(_) => {
                    eprintln!(
                        "Missing GQ value '{}' at {}:{}. Treating as 0.",
                        gq_str, chr, one_based_vcf_position.0
                    );
                    0
                }
            },
        };
        // Check if GQ value is below the minimum threshold
        if gq_value < min_gq {
            sample_has_low_gq = true;
        }
    }

    if sample_has_low_gq {
        // Skip this variant
        filtering_stats.low_gq_variants += 1;
        filtering_stats._filtered_variants += 1;
        filtering_stats
            .filtered_positions
            .insert(zero_based_position);
        filtering_stats.add_example(format!("{}: Filtered due to low GQ", line.trim()));

        let has_missing_genotypes = genotypes.iter().any(|gt| gt.is_none());
        let passes_filters = !sample_has_low_gq && !has_missing_genotypes && !is_multiallelic;
        let variant = Variant {
            position: zero_based_position,
            genotypes: genotypes.clone(),
        };
        return Ok(Some((variant, passes_filters)));
    }

    if genotypes.iter().any(|gt| gt.is_none()) {
        filtering_stats.missing_data_variants += 1;
        filtering_stats.add_example(format!("{}: Filtered due to missing data", line.trim()));
    }

    let has_missing_genotypes = genotypes.iter().any(|gt| gt.is_none());
    let passes_filters = !sample_has_low_gq && !has_missing_genotypes && !is_multiallelic;

    // Update filtering stats if variant is filtered out
    if !passes_filters {
        filtering_stats._filtered_variants += 1;
        filtering_stats
            .filtered_positions
            .insert(zero_based_position);

        if sample_has_low_gq {
            filtering_stats.low_gq_variants += 1;
            filtering_stats.add_example(format!("{}: Filtered due to low GQ", line.trim()));
        }
        if genotypes.iter().any(|gt| gt.is_none()) {
            filtering_stats.missing_data_variants += 1;
            filtering_stats.add_example(format!("{}: Filtered due to missing data", line.trim()));
        }
        if is_multiallelic {
            filtering_stats.multi_allelic_variants += 1;
            filtering_stats.add_example(format!(
                "{}: Filtered due to multi-allelic variant",
                line.trim()
            ));
        }
    }

    let variant = Variant {
        position: zero_based_position,
        genotypes: genotypes.clone(),
    };

    // Return the parsed variant and whether it passes filters
    Ok(Some((variant, passes_filters)))
}
