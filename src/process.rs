use crate::stats::{
    calculate_adjusted_sequence_length, calculate_inversion_allele_frequency, calculate_per_site,
    calculate_pi, calculate_watterson_theta, SiteDiversity,
};

use crate::parse::{
    find_vcf_file, open_vcf_reader, parse_gtf_file, read_reference_sequence, validate_vcf_header,
};

use clap::Parser;
use colored::*;
use crossbeam_channel::bounded;
use csv::WriterBuilder;
use indicatif::{ProgressBar, ProgressStyle};
use log::{info, warn};
use parking_lot::Mutex;
use prettytable::{row, Table};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, SystemTime};
use std::collections::HashMap as Map2;

// Define command-line arguments using clap
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    // Folder containing VCF files
    #[arg(short, long = "vcf_folder")]
    pub vcf_folder: String,

    // Chromosome to process
    #[arg(short, long = "chr")]
    pub chr: Option<String>,

    // Region to process (start-end)
    #[arg(short, long = "region")]
    pub region: Option<String>,

    // Configuration file
    #[arg(long = "config_file")]
    pub config_file: Option<String>,

    // Output file
    #[arg(short, long = "output_file")]
    pub output_file: Option<String>,

    // Minimum genotype quality
    #[arg(long = "min_gq", default_value = "30")]
    pub min_gq: u16,

    // Mask file (regions to exclude)
    #[arg(long = "mask_file")]
    pub mask_file: Option<String>,

    // Allow file (regions to include)
    #[arg(long = "allow_file")]
    pub allow_file: Option<String>,

    #[arg(long = "reference")]
    pub reference_path: String,

    // GTF or GFF
    #[arg(long = "gtf")]
    pub gtf_path: String,
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
        ZeroBasedHalfOpen {
            start: start_inclusive.max(0) as usize,
            end: (end_inclusive + 1).max(start_inclusive + 1) as usize,
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

    /// Returns the 1-based inclusive end coordinate of this interval.
    pub fn end_1based_inclusive(&self) -> i64 {
        self.end as i64
    }

    /// Returns 1-based position of `pos` if inside [start..end), else None.
    pub fn relative_position_1based(&self, pos: i64) -> Option<usize> {
        let p = pos as usize;
        if p >= self.start && p < self.end {
            Some(p - self.start + 1)
        } else {
            None
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

/// A CDS sequence guaranteed to have length divisible by 3 and no internal stops.
#[derive(Debug)]
pub struct CdsSeq {
    pub data: Vec<u8>,
}

impl CdsSeq {
    /// Creates a new CdsSeq, making sure its length is a multiple of three,
    /// checks for empty input, making sure all nucleotides are valid,
    /// starts with ATG, and contains no internal stop codons.
    /// A log file named cds_validation.log is appended with the validation result.
    pub fn new(seq: Vec<u8>) -> Result<Self, String> {
        let now = SystemTime::now();
        let mut log_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("cds_validation.log")
            .map_err(|e| format!("Failed to open cds_validation.log: {}", e))?;

        if seq.is_empty() {
            writeln!(log_file, "{:?} Invalid CDS: empty sequence", now)
                .map_err(|e| format!("Failed to write log: {}", e))?;
            return Err("CDS is empty".to_string());
        }

        let mut seq_upper = Vec::with_capacity(seq.len());
        for &b in seq.iter() {
            seq_upper.push(b.to_ascii_uppercase());
        }

        if seq_upper.len() < 3 {
            writeln!(
                log_file,
                "{:?} Invalid CDS: too short, length = {}",
                now,
                seq_upper.len()
            )
            .map_err(|e| format!("Failed to write log: {}", e))?;
            return Err("CDS is too short".to_string());
        }
        if seq_upper.len() % 3 != 0 {
            writeln!(
                log_file,
                "{:?} Invalid CDS: length not divisible by 3, length = {}",
                now,
                seq_upper.len()
            )
            .map_err(|e| format!("Failed to write log: {}", e))?;
            return Err(format!("CDS length {} not divisible by 3", seq_upper.len()));
        }

        for (i, &n) in seq_upper.iter().enumerate() {
            if !matches!(n, b'A' | b'C' | b'G' | b'T' | b'N') {
                writeln!(
                    log_file,
                    "{:?} Invalid CDS: bad nucleotide '{}' at position {}",
                    now, n as char, i
                )
                .map_err(|e| format!("Failed to write log: {}", e))?;
                return Err(format!(
                    "Invalid nucleotide '{}' at position {}",
                    n as char, i
                ));
            }
        }

        let start_codon_up = &seq_upper[0..3];
        if start_codon_up != [b'A', b'T', b'G'] {
            writeln!(
                log_file,
                "{:?} Invalid CDS: does not begin with ATG, found {:?}",
                now, start_codon_up
            )
            .map_err(|e| format!("Failed to write log: {}", e))?;
            return Err(format!(
                "CDS does not begin with ATG; found {:?}",
                start_codon_up
            ));
        }

        let stops = [b"TAA", b"TAG", b"TGA"];
        for i in (0..seq_upper.len()).step_by(3) {
            if i != 0 {
                let codon = &seq_upper[i..i + 3];
                if stops.iter().any(|stop| *stop == codon) {
                    writeln!(
                        log_file,
                        "{:?} Invalid CDS: internal stop at codon index {}",
                        now,
                        i / 3
                    )
                    .map_err(|e| format!("Failed to write log: {}", e))?;
                    return Err(format!(
                        "CDS has internal stop codon at codon index {}",
                        i / 3
                    ));
                }
            }
        }

        writeln!(
            log_file,
            "{:?} Valid CDS: length = {}",
            now,
            seq_upper.len()
        )
        .map_err(|e| format!("Failed to write log: {}", e))?;
        Ok(CdsSeq { data: seq_upper })
    }
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
}

/// Represents one transcript's coding sequence. It stores all CDS segments
/// belonging to a single transcript (no introns).
#[derive(Debug, Clone)]
pub struct TranscriptCDS {
    /// The transcript identifier from the GTF
    pub transcript_id: String,
    /// The strand for the transcript
    pub strand: char,
    /// A list of frames, one per CDS segment
    pub frames: Vec<i64>,
    /// A list of CDS segments in 0-based half-open intervals
    pub segments: Vec<ZeroBasedHalfOpen>,
}

/// Holds all the output columns for writing one row in the CSV.
#[derive(Debug, Clone)]
struct CsvRowData {
    seqname: String,
    region_start: i64,
    region_end: i64,
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
    // pos is zero-based
    // regions are sorted by start position
    
    // Create a ZeroBasedHalfOpen for the position (as a point)
    let position_interval = ZeroBasedHalfOpen::from_0based_point(pos);
    
    // Binary search through regions
    let mut left = 0;
    let mut right = regions.len();

    while left < right {
        let mid = (left + right) / 2;
        let (start, end) = regions[mid];
        
        // Convert region to ZeroBasedHalfOpen
        let region_interval = ZeroBasedHalfOpen {
            start: start as usize,
            end: end as usize,
        };
        
        // Use intersect to check for overlap
        if position_interval.start >= region_interval.end {
            // Position is beyond this region, look to the right
            left = mid + 1;
        } else if position_interval.end <= region_interval.start {
            // Position is before this region, look to the left
            right = mid;
        } else {
            // Overlap exists, position is in region
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

Now, the final sample_name is constructed with “_L” or “_R” to distinguish the left or right haplotype. Specifically, for (sample_idx, hap_idx) in haplotype_indices, the code does something like:
    sample_name = match *hap_idx {
        0 => format!("{}_L", sample_names[*mapped_index]),
        1 => format!("{}_R", sample_names[*mapped_index]),
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
    extended_region: ZeroBasedHalfOpen,
    adjusted_sequence_length: Option<i64>,
    seqinfo_storage: Arc<Mutex<Vec<SeqInfo>>>,
    position_allele_map: Arc<Mutex<HashMap<i64, (char, char)>>>,
    chromosome: String,
    is_filtered_set: bool,
    reference_sequence: &[u8],
    cds_regions: &[TranscriptCDS],
) -> Result<Option<(usize, f64, f64, usize, Vec<SiteDiversity>)>, VcfError> {
    let mut index_map = HashMap::new();
    for (sample_index, name) in sample_names.iter().enumerate() {
        let trimmed_id = name.rsplit('_').next().unwrap_or(name);
        index_map.insert(trimmed_id, sample_index);
    }

    let mut group_haps = Vec::new();
    for (config_sample_name, &(left_side, right_side)) in sample_filter {
        match index_map.get(config_sample_name.as_str()) {
            Some(&mapped_index) => {
                if left_side == haplotype_group {
                    group_haps.push((mapped_index, 0));
                }
                if right_side == haplotype_group {
                    group_haps.push((mapped_index, 1));
                }
            }
            None => {
                return Err(VcfError::Parse(format!(
                    "Sample '{}' from config not found in VCF",
                    config_sample_name
                )));
            }
        }
    }
    if group_haps.is_empty() {
        println!("No haplotypes found for group {}", haplotype_group);
        return Ok(None);
    }

    let mut region_segsites = 0;
    let region_hap_count = group_haps.len();
    if variants.is_empty() {
        return Ok(Some((0, 0.0, 0.0, region_hap_count, Vec::new())));
    }
    for current_variant in variants {
        if current_variant.position < region_start || current_variant.position > region_end {
            continue;
        }
        let mut allele_values = Vec::new();
        for &(mapped_index, side) in &group_haps {
            if let Some(some_genotypes) = current_variant.genotypes.get(mapped_index) {
                if let Some(genotype_vec) = some_genotypes {
                    if let Some(&val) = genotype_vec.get(side as usize) {
                        let locked_map = position_allele_map.lock();
                        if locked_map.get(&current_variant.position).is_some() {
                            allele_values.push(val);
                        }
                    }
                }
            }
        }
        let distinct_alleles: HashSet<u8> = allele_values.iter().copied().collect();
        if distinct_alleles.len() > 1 {
            region_segsites += 1;
        }
    }
    // Define the region as a ZeroBasedHalfOpen interval for length calculation
    let region = ZeroBasedHalfOpen {
        start: region_start as usize,
        end: region_end as usize,
    };
    let final_length = adjusted_sequence_length.unwrap_or(region.len() as i64);
    let final_theta = calculate_watterson_theta(region_segsites, region_hap_count, final_length);
    let final_pi = calculate_pi(variants, &group_haps);

    for transcript in cds_regions {
        let mut assembled: HashMap<String, Vec<u8>> = HashMap::new();
        for (mapped_index, side) in &group_haps {
            let label = match *side {
                0 => format!("{}_L", sample_names[*mapped_index]),
                1 => format!("{}_R", sample_names[*mapped_index]),
                _ => panic!("Unexpected haplotype side"),
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
                eprintln!(
                    "Skipping negative start {} for transcript {} on {}",
                    seg_start, transcript.transcript_id, chromosome
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
                    0 => format!("{}_L", sample_names[*mapped_index]),
                    1 => format!("{}_R", sample_names[*mapped_index]),
                    _ => panic!("Unexpected side"),
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

    let locked_seqinfo = seqinfo_storage.lock();
    if locked_seqinfo.is_empty() {
        println!("No SeqInfo in stats pass for group {}", haplotype_group);
    } else {
        display_seqinfo_entries(&locked_seqinfo, 12);
    }
    drop(locked_seqinfo);
    seqinfo_storage.lock().clear();

    println!(
        "Finishing up the variant region... is_filtered_set = {:?}",
        is_filtered_set
    );
    if is_filtered_set {
        println!("Calling make_sequences with filtered set...");
        make_sequences(
            variants,
            sample_names,
            haplotype_group,
            sample_filter,
            extended_region,
            reference_sequence,
            cds_regions,
            position_allele_map.clone(),
            &chromosome,
        )?;
    }

    let site_diversities = calculate_per_site(variants, &group_haps, region_start, region_end);
    println!(
        "Computed per-site diversity for group {}: found {} sites in region {}",
        haplotype_group,
        site_diversities.len(),
        chromosome
    );

    Ok(Some((
        region_segsites,
        final_theta,
        final_pi,
        region_hap_count,
        site_diversities,
    )))
}

/// Generates sequences for the specified haplotype group.
/// This enforces consistent intervals and may call for final phylip output.
pub fn make_sequences(
    variants: &[Variant],
    sample_names: &[String],
    haplotype_group: u8,
    sample_filter: &HashMap<String, (u8, u8)>,
    extended_region: ZeroBasedHalfOpen,
    reference_sequence: &[u8],
    cds_regions: &[TranscriptCDS],
    position_allele_map: Arc<Mutex<HashMap<i64, (char, char)>>>,
    chromosome: &str,
) -> Result<(), VcfError> {
    let vcf_sample_id_to_index = map_sample_names_to_indices(sample_names)?;

    let haplotype_indices =
        get_haplotype_indices_for_group(haplotype_group, sample_filter, &vcf_sample_id_to_index)?;

    if haplotype_indices.is_empty() {
        warn!("No haplotypes found for group {}.", haplotype_group);
        return Ok(());
    }

    if cds_regions.is_empty() {
        warn!("No CDS regions available for transcript. Skipping sequence generation.");
        return Ok(());
    }

    let mut hap_sequences = initialize_hap_sequences(
        &haplotype_indices,
        &sample_names,
        reference_sequence,
        extended_region,
    );

    if hap_sequences.is_empty() {
        warn!(
            "No sequences initialized for group {}. Skipping variant application.",
            haplotype_group
        );
        return Ok(());
    }

    apply_variants_to_transcripts(
        variants,
        &haplotype_indices,
        extended_region,
        position_allele_map.clone(),
        &mut hap_sequences,
        &sample_names,
    )?;

    let hap_sequences_u8: HashMap<String, Vec<u8>> = hap_sequences
        .iter()
        .map(|(k, v)| (k.clone(), v.iter().copied().map(|b| b as u8).collect()))
        .collect();

    generate_batch_statistics(&hap_sequences_u8)?;

    prepare_to_write_cds(
        haplotype_group,
        cds_regions,
        &hap_sequences_u8,
        chromosome,
        extended_region,
    )?;

    Ok(())
}

fn map_sample_names_to_indices(sample_names: &[String]) -> Result<HashMap<&str, usize>, VcfError> {
    let mut vcf_sample_id_to_index = HashMap::new();
    for (i, name) in sample_names.iter().enumerate() {
        let sample_id = name.rsplit('_').next().unwrap_or(name);
        vcf_sample_id_to_index.insert(sample_id, i);
    }
    Ok(vcf_sample_id_to_index)
}

fn get_haplotype_indices_for_group(
    haplotype_group: u8,
    sample_filter: &HashMap<String, (u8, u8)>,
    vcf_sample_id_to_index: &HashMap<&str, usize>,
) -> Result<Vec<(usize, u8)>, VcfError> {
    let mut haplotype_indices = Vec::new();
    for (sample_name, &(left_tsv, right_tsv)) in sample_filter {
        match vcf_sample_id_to_index.get(sample_name.as_str()) {
            Some(&idx) => {
                if left_tsv == haplotype_group {
                    haplotype_indices.push((idx, 0));
                }
                if right_tsv == haplotype_group {
                    haplotype_indices.push((idx, 1));
                }
            }
            None => {
                return Err(VcfError::Parse(format!(
                    "Sample '{}' from config not found in VCF",
                    sample_name
                )));
            }
        }
    }
    Ok(haplotype_indices)
}

fn initialize_hap_sequences(
    haplotype_indices: &[(usize, u8)],
    sample_names: &[String],
    reference_sequence: &[u8],
    extended_region: ZeroBasedHalfOpen,
) -> HashMap<String, Vec<u8>> {
    if extended_region.end > reference_sequence.len() {
        warn!(
            "Invalid extended region: start={}, end={}, reference length={}",
            extended_region.start,
            extended_region.end,
            reference_sequence.len()
        );
        return HashMap::new();
    }
    let region_slice = extended_region.slice(reference_sequence);

    let mut hap_sequences = HashMap::new();

    // For each relevant sample/haplotype pair, initialize its sequence by copying the chosen region.
    for (sample_idx, hap_idx) in haplotype_indices {
        // We use a consistent naming format for each sample/haplotype: "SampleName_L" or "SampleName_R."
        let sample_name = format!(
            "{}_{}",
            sample_names[*sample_idx],
            if *hap_idx == 0 { "L" } else { "R" }
        );

        // Collect the bytes into a new vector to store in our map.
        let sequence_u8: Vec<u8> = region_slice.iter().copied().collect();

        // Insert the new haplotype sequence into the hash map.
        hap_sequences.insert(sample_name, sequence_u8);
    }

    hap_sequences
}

fn apply_variants_to_transcripts(
    variants: &[Variant],
    haplotype_indices: &[(usize, u8)],
    extended_region: ZeroBasedHalfOpen,
    position_allele_map: Arc<Mutex<HashMap<i64, (char, char)>>>,
    hap_sequences: &mut HashMap<String, Vec<u8>>,
    sample_names: &[String],
) -> Result<(), VcfError> {
    // The reference sequence passed to make_sequences is the entire chromosome.
    // We rely on ZeroBasedHalfOpen externally, so we do not manually slice here.
    for variant in variants {
        if !extended_region.contains(ZeroBasedPosition(variant.position)) {
            continue;
        }

        // Map the chromosome position to the zero-based offset within extended_region.
        let pos_in_seq = (variant.position as usize).saturating_sub(extended_region.start);

        // Iterate through the haplotypes for the current group
        for &(sample_idx, hap_idx) in haplotype_indices {
            let sample_name = format!(
                "{}_{}",
                &sample_names[sample_idx],
                if hap_idx == 0 { "L" } else { "R" }
            );

            // Get the mutable sequence vector for the current sample and haplotype
            if let Some(seq_vec) = hap_sequences.get_mut(&sample_name) {
                // Check if the calculated position is within the bounds of the sequence vector
                // The length of seq_vec is the length of the region of interest.
                // The length of pos_in_seq is also relative to the beginning of the extended region,
                // Because we subtracted the extended_start value from it above.
                if pos_in_seq < seq_vec.len() {
                    // Lock the position_allele_map to get the reference and alternate alleles at this position
                    let map = position_allele_map.lock();

                    // Get the reference and alternate alleles from the map, if available
                    if let Some(&(ref_allele, alt_allele)) = map.get(&variant.position) {
                        // Determine the allele to use based on the genotype
                        let allele_to_use =
                            if let Some(genotype) = variant.genotypes[sample_idx].as_ref() {
                                if genotype[hap_idx as usize] == 0 {
                                    // If the genotype is 0 (reference allele), use the reference allele
                                    ref_allele as u8
                                } else {
                                    // If the genotype is 1 (alternate allele), use the alternate allele
                                    alt_allele as u8
                                }
                            } else {
                                // If genotype is missing, use reference allele
                                ref_allele as u8
                            };

                        // Update the sequence at the calculated position with the determined allele
                        seq_vec[pos_in_seq] = allele_to_use;
                    }
                } else {
                    // If the position is out of bounds, log a warning
                    warn!(
                        "Position {} is out of bounds for sequence {}",
                        variant.position, sample_name
                    );
                }
            }
        }
    }
    Ok(())
}

fn generate_batch_statistics(hap_sequences: &HashMap<String, Vec<u8>>) -> Result<(), VcfError> {
    let total_sequences = hap_sequences.len();
    let mut stop_codon_or_too_short = 0;
    let mut skipped_sequences = 0;
    let mut not_divisible_by_three = 0;
    let mut mid_sequence_stop = 0;
    let mut length_modified = 0;

    let stop_codons = ["TAA", "TAG", "TGA"];

    for (_sample_name, sequence) in hap_sequences {
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

        for i in (0..sequence.len() - 2).step_by(3) {
            let codon = &sequence_str[i..i + 3];
            if stop_codons.contains(&codon) {
                mid_sequence_stop += 1;
                break;
            }
        }
    }

    info!("Batch statistics: {} sequences processed.", total_sequences);
    info!(
        "Percentage of sequences with stop codon or too short: {:.2}%",
        (stop_codon_or_too_short as f64 / total_sequences as f64) * 100.0
    );
    info!(
        "Percentage of sequences skipped: {:.2}%",
        (skipped_sequences as f64 / total_sequences as f64) * 100.0
    );
    info!(
        "Percentage of sequences not divisible by three: {:.2}%",
        (not_divisible_by_three as f64 / total_sequences as f64) * 100.0
    );
    info!(
        "Percentage of sequences with a mid-sequence stop codon: {:.2}%",
        (mid_sequence_stop as f64 / total_sequences as f64) * 100.0
    );
    info!(
        "Percentage of sequences with modified length: {:.2}%",
        (length_modified as f64 / total_sequences as f64) * 100.0
    );

    Ok(())
}

fn prepare_to_write_cds(
    haplotype_group: u8,
    cds_regions: &[TranscriptCDS],
    hap_sequences: &HashMap<String, Vec<u8>>,
    chromosome: &str,
    hap_region: ZeroBasedHalfOpen,
) -> Result<(), VcfError> {
    // For each transcript, we build a spliced coding sequence for every sample in hap_sequences.
    // Offsets come from intersecting each CDS with hap_region in zero-based form.
    for cds in cds_regions {
        if cds.segments.is_empty() {
            continue;
        }
        let mut final_cds_map: HashMap<String, Vec<u8>> = HashMap::new();
        for (sample_name, _) in hap_sequences {
            final_cds_map.insert(sample_name.clone(), Vec::new());
        }

        for (sample_name, seq) in hap_sequences {
            let mut spliced_sequence = Vec::new();
            // A CDS entirely outside the query region but inside the extended region does not get written out to a .phy file.
            // If a CDS partially overlaps the query region, its entire transcript sequence
            // (including segments outside the query region but within the extended region) will be written out
            for (i, seg) in cds.segments.iter().enumerate() {
                let seg_s = seg.start as i64;
                let seg_e = seg.end as i64;
                let strand = cds.strand;
                let _frame = cds.frames.get(i).copied().unwrap_or(0);
                // If the entire CDS segment is within the extended region, then the segment is included in full.
                // If only part overlaps, only that partial segment is included. This should NEVER happen in humans, because the largest gene is smaller than the extended region.
                // If the segment lies wholly beyond the extended region (or the normal region), it is dropped entirely.
                let seg_interval = ZeroBasedHalfOpen::from_1based_inclusive(seg_s, seg_e);
                if let Some(overlap) = seg_interval.intersect(&hap_region) {
                    let offset_start = overlap.start.saturating_sub(hap_region.start);
                    let offset_end = overlap.end.saturating_sub(hap_region.start);
                    let clipped_end = offset_end.min(seq.len());
                    if clipped_end > offset_start {
                        let mut piece = seq[offset_start..clipped_end].to_vec();
                        if strand == '-' {
                            piece.reverse();
                            for base in piece.iter_mut() {
                                *base = match *base {
                                    b'A' | b'a' => b'T',
                                    b'T' | b't' => b'A',
                                    b'C' | b'c' => b'G',
                                    b'G' | b'g' => b'C',
                                    _ => b'N',
                                };
                            }
                        }
                        spliced_sequence.extend_from_slice(&piece);
                    }
                }
            }
            match CdsSeq::new(spliced_sequence) {
                Ok(valid_cds) => {
                    final_cds_map.insert(sample_name.clone(), valid_cds.data);
                }
                Err(_) => {
                    final_cds_map.insert(sample_name.clone(), Vec::new());
                }
            }
        }

        let filtered_map: HashMap<String, Vec<char>> = final_cds_map
            .into_iter()
            .filter_map(|(name, seq_data)| {
                if seq_data.is_empty() {
                    None
                } else {
                    let converted = seq_data.into_iter().map(|x| x as char).collect();
                    Some((name, converted))
                }
            })
            .collect();
        if filtered_map.is_empty() {
            continue;
        }
        let filename = format!(
            "group_{}_{}_chr_{}_start_{}_end_{}_combined.phy",
            haplotype_group, cds.transcript_id, chromosome, hap_region.start, hap_region.end
        );
        write_phylip_file(&filename, &filtered_map, &cds.transcript_id)?;
    }
    Ok(())
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
    // Create CSV writer and write the header once.
    let mut writer = create_and_setup_csv_writer(output_file)?;
    write_csv_header(&mut writer)?;

    // Group config entries by chromosome for efficiency
    let grouped = group_config_entries_by_chr(config_entries);

    // We will process each chromosome in parallel (for speed),
    // then flatten the per-chromosome results in the order we get them.
    let all_pairs: Vec<(CsvRowData, Vec<(i64, f64, f64, u8, bool)>)> = grouped
        .into_par_iter()
        .flat_map(|(chr, chr_entries)| {
            match process_chromosome_entries(
                &chr,
                chr_entries,
                vcf_folder,
                min_gq,
                &mask,
                &allow,
                args,
            ) {
                Ok(list) => list,
                Err(e) => {
                    eprintln!("Error processing chromosome {}: {}", chr, e);
                    Vec::new()
                }
            }
        })
        .collect();

    let all_results: Vec<CsvRowData> = all_pairs
        .iter()
        .map(|(csv_row, _sites)| csv_row.clone())
        .collect();

    // Write all rows to the CSV file
    for row_data in all_results {
        write_csv_row(&mut writer, &row_data)?;
    }

    // Now write per-site statistics in wide format to a single CSV file.
    // Each row is a relative_position (1-based), and each set of columns corresponds to
    // (unfiltered_pi_, unfiltered_theta_, filtered_pi_, filtered_theta_) for a region+group.

    // A structure to store columns for each region+group+filter combination.
    // Each column is keyed by a unique name (e.g., "filtered_pi_chr_2_start_3403_end_9934_group_0"),
    // and for each relative_position, we store the (f64) value. We also store the maximum
    // relative_position encountered, so we know how many rows to write.
    let mut all_columns: HashMap<String, BTreeMap<usize, f64>> = HashMap::new();
    let mut max_position = 0_usize;

    // A small helper function to build the column name.
    // prefix is one of: "filtered_pi_", "filtered_theta_", "unfiltered_pi_", or "unfiltered_theta_"
    // region_tag is "chr_{chr}_start_{start}_end_{end}_group_{g}"
    fn build_col_name(prefix: &str, row: &CsvRowData, group_id: u8) -> String {
        format!(
            "{}chr_{}_start_{}_end_{}_group_{}",
            prefix, row.seqname, row.region_start, row.region_end, group_id
        )
    }

    // Populate our all_columns map with the data from each region–group combo.
    for (csv_row, per_site_vec) in &all_pairs {
        // We'll produce 4 columns for each region+group combo:
        //   unfiltered_pi_ + region_tag
        //   unfiltered_theta_ + region_tag
        //   filtered_pi_ + region_tag
        //   filtered_theta_ + region_tag

        // We want to fill these columns for each entry in per_site_vec:
        // If is_filtered=false, we fill the "unfiltered_pi_" and "unfiltered_theta_" columns.
        // If is_filtered=true, we fill "filtered_pi_" and "filtered_theta_" columns.
        // The relative position is (pos - region_start + 1).

        for &(pos, pi_val, theta_val, group_id, is_filtered) in per_site_vec {
            // Compute 1-based relative position using ZeroBasedHalfOpen
            let region = ZeroBasedHalfOpen {
                start: csv_row.region_start as usize,
                end: csv_row.region_end as usize,
            };
            if let Some(rel_pos) = region.relative_position_1based(pos) {
                if rel_pos > max_position {
                    max_position = rel_pos;
                }
                // Decide which prefix to use
                if is_filtered {
                    let pi_col = build_col_name("filtered_pi_", csv_row, group_id);
                    let theta_col = build_col_name("filtered_theta_", csv_row, group_id);

                    all_columns
                        .entry(pi_col)
                        .or_insert_with(BTreeMap::new)
                        .insert(rel_pos as usize, pi_val);
                    all_columns
                        .entry(theta_col)
                        .or_insert_with(BTreeMap::new)
                        .insert(rel_pos as usize, theta_val);
                } else {
                    let pi_col = build_col_name("unfiltered_pi_", csv_row, group_id);
                    let theta_col = build_col_name("unfiltered_theta_", csv_row, group_id);

                    all_columns
                        .entry(pi_col)
                        .or_insert_with(BTreeMap::new)
                        .insert(rel_pos as usize, pi_val);
                    all_columns
                        .entry(theta_col)
                        .or_insert_with(BTreeMap::new)
                        .insert(rel_pos as usize, theta_val);
                }
            } else {
                eprintln!(
                    "Warning: Position {} is outside region {}-{}",
                    pos, csv_row.region_start, csv_row.region_end
                );
            }
        }
    }

    // Now we generate a FASTA-style file, one record per combination of prefix, region, and group.
    // Each record has one header line beginning with '>', and then one line of comma-separated values
    // corresponding to each position in the region. If a value is zero, we store it as an integer '0'.
    // Otherwise, we format as a float with six decimals.

    let fasta_path = std::path::Path::new("per_site_output.fasta");
    let fasta_file = File::create(fasta_path)?;
    let mut fasta_writer = BufWriter::new(fasta_file);

    // We will map each distinct (prefix, row.seqname, row.region_start, row.region_end, group_id)
    // to a vector of values for positions from row.region_start..=row.region_end.
    // The positions are stored 1-based, but we have them in the per_site_vec as 1-based positions already.

    // We define a helper to build the full FASTA-style header.
    fn build_fasta_header(prefix: &str, row: &CsvRowData, group_id: u8) -> String {
        let mut header = String::new();
        header.push('>');
        header.push_str(prefix);
        header.push_str("chr_");
        header.push_str(&row.seqname);
        header.push_str("_start_");
        header.push_str(&row.region_start.to_string());
        header.push_str("_end_");
        header.push_str(&row.region_end.to_string());
        header.push_str("_group_");
        header.push_str(&group_id.to_string());
        header
    }

    // We will create a nested map:
    // outer key: a String for the FASTA header
    // inner value: a Vec<Option<f64>> with length = region_length, storing pi or theta or None if missing
    // Because each row is a single region, but we may have up to 4 combos (filtered/unfiltered × pi/theta × group).
    // The site records contain (position, pi, watterson_theta, group_id, is_filtered).
    // We do this for each region row in all_pairs.

    let all_fasta_data: Vec<(String, String)> = Vec::new();

    for (csv_row, per_site_vec) in &all_pairs {
        // For the region length we do (row.region_end - row.region_start + 1).
        let region_len = (csv_row.region_end - csv_row.region_start + 1) as usize;

        // We store 4 possible keys: "unfiltered_pi", "unfiltered_theta", "filtered_pi", "filtered_theta".
        // Each key maps to a vector of the same length as region_len, initially None.
        let mut records_map: Map2<String, Vec<Option<f64>>> = Map2::new();

        let combos = [
            "unfiltered_pi_",
            "unfiltered_theta_",
            "filtered_pi_",
            "filtered_theta_",
        ];
        for combo_key in combos.iter() {
            records_map.insert(combo_key.to_string(), vec![None; region_len]);
        }

        // Fill in data from per_site_vec.
        // The position here is 1-based within the region, so relative_position = pos - row.region_start.
        // If pos - row.region_start is zero-based, we do index = (pos - row.region_start - 1) in 0-based array.
        // But we do not do manual +1 or -1 in code, we rely on the types for clarity. The site_diversities
        // already stores position as 1-based inclusive from the final assignment. We just compute index carefully.

        for &(pos_1based, pi_val, theta_val, is_filtered) in per_site_vec {
            let offset = pos_1based - csv_row.region_start;
            if offset < 1 {
                continue;
            }
            let idx_0based = (offset - 1) as usize;
            if idx_0based >= region_len {
                continue;
            }
            let mut key_prefix = if is_filtered { "filtered_" } else { "unfiltered_" };
            // We update pi:
            {
                let combo_key = format!("{}pi_", key_prefix);
                if let Some(vec_ref) = records_map.get_mut(&combo_key) {
                    if pi_val == 0.0 {
                        vec_ref[idx_0based] = Some(0.0);
                    } else {
                        vec_ref[idx_0based] = Some(pi_val);
                    }
                }
            }
            // We update theta:
            {
                let combo_key = format!("{}theta_", key_prefix);
                if let Some(vec_ref) = records_map.get_mut(&combo_key) {
                    if theta_val == 0.0 {
                        vec_ref[idx_0based] = Some(0.0);
                    } else {
                        vec_ref[idx_0based] = Some(theta_val);
                    }
                }
            }
        }

        // We'll find all distinct group_ids in per_site_vec. Then we produce separate FASTA blocks for each group.
        let mut group_ids_found = HashSet::new();
        for &(_, _, _, grp, _) in per_site_vec {
            group_ids_found.insert(grp);
        }

        for grp in group_ids_found {
            // We produce four lines, one for unfiltered_pi_, unfiltered_theta_, filtered_pi_, filtered_theta_.
            // We see if there's at least one non-None in each vector. If so, we output it.

            let possible_keys = [
                "unfiltered_pi_",
                "unfiltered_theta_",
                "filtered_pi_",
                "filtered_theta_",
            ];

            for key_base in possible_keys.iter() {
                // We'll read from records_map. For the final string, we build the FASTA header:
                // e.g. >unfiltered_pi_chr_14_start_244000_end_248905_group_1
                // Then below it, we produce region_len comma separated values. We skip if the data is all None.

                let full_key = key_base.to_string();
                if let Some(vec_ref) = records_map.get(&full_key) {
                    let mut any_data = false;
                    for val_opt in vec_ref.iter() {
                        if val_opt.is_some() {
                            any_data = true;
                            break;
                        }
                    }
                    if !any_data {
                        continue;
                    }

                    let header = build_fasta_header(key_base, csv_row, grp);
                    writeln!(fasta_writer, "{}", header)?;

                    // Now build the single line of comma-separated numeric data.
                    // We'll store 0 as "0". 
                    // If Some(0.0), that's "0". Otherwise format with six decimals.

                    let mut line_values = Vec::with_capacity(region_len);
                    for val_opt in vec_ref.iter() {
                        match val_opt {
                            Some(x) => {
                                if *x == 0.0 {
                                    line_values.push("0".to_string());
                                } else {
                                    line_values.push(format!("{:.6}", x));
                                }
                            }
                            None => {
                                // Represent missing data as "NA". Should never happen.
                                line_values.push("NA".to_string());
                            }
                        }
                    }
                    let joined = line_values.join(",");
                    writeln!(fasta_writer, "{}", joined)?;
                }
            }
        }
    }

    fasta_writer.flush()?;

    writer.flush().map_err(|e| VcfError::Io(e.into()))?;
    println!("Wrote FASTA-style per-site data to per_site_output.fasta");
    println!(
        "Processing complete. Check the output file: {:?}",
        output_file
    );
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
    Ok(())
}

/// Writes a single row of data to the CSV.
fn write_csv_row<W: Write>(writer: &mut csv::Writer<W>, row: &CsvRowData) -> Result<(), VcfError> {
    writer
        .write_record(&[
            &row.seqname,
            &row.region_start.to_string(),
            &row.region_end.to_string(),
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
    entries: Vec<ConfigEntry>,
    vcf_folder: &str,
    min_gq: u16,
    mask: &Option<Arc<HashMap<String, Vec<(i64, i64)>>>>,
    allow: &Option<Arc<HashMap<String, Vec<(i64, i64)>>>>,
    args: &Args,
) -> Result<Vec<(CsvRowData, Vec<(i64, f64, f64, u8, bool)>)>, VcfError> {
    println!("Processing chromosome: {}", chr);

    // Load entire chromosome length from reference index
    let chr_length = {
        let fasta_reader =
            bio::io::fasta::IndexedReader::from_file(&args.reference_path).map_err(|e| {
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
                VcfError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Chromosome {} not found in reference", chr),
                ))
            })?;
        seq_info.len as i64
    };

    // Read the full reference sequence for that chromosome.
    let ref_sequence = {
        let entire_chrom = ZeroBasedHalfOpen {
            start: 0,
            end: chr_length as usize,
        };
        read_reference_sequence(Path::new(&args.reference_path), chr, entire_chrom)?
    };

    // Parse all transcripts for that chromosome from the GTF
    let all_transcripts = parse_gtf_file(Path::new(&args.gtf_path), chr)?;
    // We'll keep them in `cds_regions` for subsequent filtering
    let cds_regions = all_transcripts;

    // Locate the VCF file for this chromosome
    let vcf_file = match find_vcf_file(vcf_folder, chr) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("Error finding VCF file for {}: {:?}", chr, e);
            return Ok(Vec::new());
        }
    };

    // We'll store final rows from each entry in this vector
    let mut rows = Vec::with_capacity(entries.len());

    // For each config entry in this chromosome, do the work
    //    (We could also parallelize here...)
    for entry in entries {
        match process_single_config_entry(
            entry,
            &vcf_file,
            min_gq,
            mask,
            allow,
            &ref_sequence,
            &cds_regions,
            chr,
            args,
        ) {
            Ok(Some(row_data)) => {
                rows.push(row_data);
            }
            Ok(None) => {
                // Something was skipped or no haplotypes found
            }
            Err(e) => {
                eprintln!("Error processing entry on {}: {}", chr, e);
            }
        }
    }

    Ok(rows)
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
    cds_regions: &[TranscriptCDS],
    chr: &str,
    args: &Args,
) -> Result<Option<(CsvRowData, Vec<(i64, f64, f64, u8, bool)>)>, VcfError> {
    println!(
        "Processing entry: {}:{}-{}",
        entry.seqname, entry.interval.start, entry.interval.end
    );

    // Filter transcripts to only those overlapping [start..end].
    let local_cds = filter_and_log_transcripts(
        cds_regions.to_vec(),
        QueryRegion {
            start: entry.interval.start as i64,
            end: entry.interval.end as i64,
        },
    );

    // Calculate EXTENDED region boundaries
    let chr_length = ref_sequence.len() as i64;
    let extended_region = ZeroBasedHalfOpen::from_1based_inclusive(
        (entry.interval.start as i64 - 3_000_000).max(0),
        ((entry.interval.end as i64) + 3_000_000).min(chr_length),
    );

    let seqinfo_storage_unfiltered = Arc::new(Mutex::new(Vec::<SeqInfo>::new()));
    let position_allele_map_unfiltered = Arc::new(Mutex::new(HashMap::<i64, (char, char)>::new()));
    let seqinfo_storage_filtered = Arc::new(Mutex::new(Vec::<SeqInfo>::new()));
    let position_allele_map_filtered = Arc::new(Mutex::new(HashMap::<i64, (char, char)>::new()));

    println!(
        "Calling process_vcf for {} from {} to {} (extended: {}-{})",
        chr, entry.interval.start, entry.interval.end, extended_region.start, extended_region.end
    );

    let (
        unfiltered_variants,
        filtered_variants,
        sample_names,
        _chr_len,
        _missing_data_info,
        filtering_stats,
    ) = match process_vcf(
        vcf_file,
        Path::new(&args.reference_path),
        chr.to_string(),
        extended_region,
        min_gq,
        mask.clone(),
        allow.clone(),
        seqinfo_storage_unfiltered.clone(),
        seqinfo_storage_filtered.clone(),
        position_allele_map_unfiltered.clone(),
        position_allele_map_filtered.clone(),
    ) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error processing VCF for {}: {}", chr, e);
            return Ok(None);
        }
    };

    // Print short summary of filtering
    println!(
        "Total variants: {}, Filtered: {} (some reasons: mask={}, allow={}, multi-allelic={}, low_GQ={}, missing={})",
        filtering_stats.total_variants,
        filtering_stats._filtered_variants,
        filtering_stats.filtered_due_to_mask,
        filtering_stats.filtered_due_to_allow,
        filtering_stats.multi_allelic_variants,
        filtering_stats.low_gq_variants,
        filtering_stats.missing_data_variants,
    );

    // Basic length info: naive and adjusted
    let sequence_length = (entry.interval.end - entry.interval.start) as i64;
    let adjusted_sequence_length = calculate_adjusted_sequence_length(
        entry.interval.start as i64,
        entry.interval.end as i64,
        allow.as_ref().and_then(|a| a.get(&chr.to_string())),
        mask.as_ref().and_then(|m| m.get(&chr.to_string())),
    );

    // Stats for filtered group 0
    let (num_segsites_0_f, w_theta_0_f, pi_0_f, n_hap_0_f, site_divs_0_f) = match process_variants(
        &filtered_variants,
        &sample_names,
        0,
        &entry.samples_filtered,
        entry.interval.start as i64,
        entry.interval.end as i64,
        extended_region,
        Some(adjusted_sequence_length),
        seqinfo_storage_filtered.clone(),
        position_allele_map_filtered.clone(),
        entry.seqname.clone(),
        true,
        ref_sequence,
        &local_cds,
    )? {
        Some(vals) => vals,
        None => {
            println!(
                "No haplotypes found for group 0 (filtered) in region {}-{}",
                entry.interval.start as i64, entry.interval.end as i64
            );
            return Ok(None);
        }
    };

    // Stats for filtered group 1
    let (num_segsites_1_f, w_theta_1_f, pi_1_f, n_hap_1_f, site_divs_1_f) = match process_variants(
        &filtered_variants,
        &sample_names,
        1,
        &entry.samples_filtered,
        entry.interval.start as i64,
        entry.interval.end as i64,
        extended_region,
        Some(adjusted_sequence_length),
        seqinfo_storage_filtered.clone(),
        position_allele_map_filtered.clone(),
        entry.seqname.clone(),
        true,
        ref_sequence,
        &local_cds,
    )? {
        Some(vals) => vals,
        None => {
            println!(
                "No haplotypes found for group 1 (filtered) in region {}-{}",
                entry.interval.start as i64, entry.interval.end as i64
            );
            return Ok(None);
        }
    };

    let inversion_freq_filt =
        calculate_inversion_allele_frequency(&entry.samples_filtered).unwrap_or(-1.0);

    // Stats for unfiltered group 0
    let region_variants_unfiltered: Vec<_> = unfiltered_variants
        .iter()
        // Filter variants within the 0-based half-open interval [start, end)
        .filter(|v| entry.interval.contains(ZeroBasedPosition(v.position)))
        .cloned()
        .collect();

    let (num_segsites_0, w_theta_0, pi_0, n_hap_0_unf, site_divs_0_unf) = match process_variants(
        &region_variants_unfiltered,
        &sample_names,
        0,
        &entry.samples_unfiltered,
        entry.interval.start as i64,
        entry.interval.end as i64,
        extended_region,
        None,
        seqinfo_storage_unfiltered.clone(),
        position_allele_map_unfiltered.clone(),
        entry.seqname.clone(),
        false,
        ref_sequence,
        &local_cds,
    )? {
        Some(vals) => vals,
        None => {
            println!(
                "No haplotypes found for group 0 in region {}-{}",
                entry.interval.start as i64, entry.interval.end as i64
            );
            return Ok(None);
        }
    };

    // Stats for unfiltered group 1
    let (num_segsites_1, w_theta_1, pi_1, n_hap_1_unf, site_divs_1_unf) = match process_variants(
        &region_variants_unfiltered,
        &sample_names,
        1,
        &entry.samples_unfiltered,
        entry.interval.start as i64,
        entry.interval.end as i64,
        extended_region,
        None,
        seqinfo_storage_unfiltered.clone(),
        position_allele_map_unfiltered.clone(),
        entry.seqname.clone(),
        false,
        ref_sequence,
        &local_cds,
    )? {
        Some(vals) => vals,
        None => {
            println!(
                "No haplotypes found for group 1 in region {}-{}",
                entry.interval.start as i64, entry.interval.end as i64
            );
            return Ok(None);
        }
    };

    let inversion_freq_no_filter =
        calculate_inversion_allele_frequency(&entry.samples_unfiltered).unwrap_or(-1.0);

    // Build final row data
    let row_data = CsvRowData {
        seqname: entry.seqname,
        region_start: entry.interval.start as i64,
        region_end: entry.interval.end as i64,
        seq_len_0: sequence_length,
        seq_len_1: sequence_length,
        seq_len_adj_0: adjusted_sequence_length,
        seq_len_adj_1: adjusted_sequence_length,
        seg_sites_0: num_segsites_0,
        seg_sites_1: num_segsites_1,
        w_theta_0,
        w_theta_1,
        pi_0,
        pi_1,
        seg_sites_0_f: num_segsites_0_f,
        seg_sites_1_f: num_segsites_1_f,
        w_theta_0_f,
        w_theta_1_f,
        pi_0_f,
        pi_1_f,
        n_hap_0_unf,
        n_hap_1_unf,
        n_hap_0_f,
        n_hap_1_f,
        inv_freq_no_filter: inversion_freq_no_filter,
        inv_freq_filter: inversion_freq_filt,
    };
    println!(
        "Finished stats for region {}-{}.",
        entry.interval.start, entry.interval.end
    );

    let mut per_site_records = Vec::new();
    for sd in site_divs_0_unf {
        per_site_records.push((sd.position, sd.pi, sd.watterson_theta, 0, false));
    }
    for sd in site_divs_1_unf {
        per_site_records.push((sd.position, sd.pi, sd.watterson_theta, 1, false));
    }
    for sd in site_divs_0_f {
        per_site_records.push((sd.position, sd.pi, sd.watterson_theta, 0, true));
    }
    for sd in site_divs_1_f {
        per_site_records.push((sd.position, sd.pi, sd.watterson_theta, 1, true));
    }

    Ok(Some((row_data, per_site_records)))
}

// This function filters the TranscriptCDS by QueryRegion overlap and prints stats
fn filter_and_log_transcripts(
    transcripts: Vec<TranscriptCDS>,
    query: QueryRegion,
) -> Vec<TranscriptCDS> {
    use colored::Colorize;
    use std::fs::OpenOptions;
    use std::io::{BufWriter, Write};

    // Open or create a log file once per call
    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("transcript_overlap.log")
        .expect("Failed to open transcript_overlap.log");
    let mut log_file = BufWriter::new(log_file);

    // Create a ZeroBasedHalfOpen for the query region
        let query_interval = ZeroBasedHalfOpen::from_0based_inclusive(query.start, query.end);
    
    writeln!(log_file, "Query region: {} to {}", query.start, query.end)
        .expect("Failed to write to transcript_overlap.log");

    let mut overlapping_transcript_count = 0;

    println!(
        "\n{}",
        "Processing CDS regions by transcript...".green().bold()
    );
    writeln!(log_file, "Processing CDS regions by transcript...")
        .expect("Failed to write to transcript_overlap.log");

    let mut filtered = Vec::new();

    #[derive(Default)]
    struct TranscriptStats {
        total_transcripts: usize,
        non_divisible_by_three: usize,
        total_cds_segments: usize,
        single_cds_transcripts: usize,
        multi_cds_transcripts: usize,
        total_coding_length: i64,
        shortest_transcript_length: Option<i64>,
        longest_transcript_length: Option<i64>,
        transcripts_with_gaps: usize,
    }

    let mut stats = TranscriptStats::default();

    for mut tcds in transcripts {
        let transcript_coding_start = tcds
            .segments
            .iter()
            .map(|seg| seg.start as i64)
            .min()
            .unwrap_or(0);
        let transcript_coding_end = tcds
            .segments
            .iter()
            .map(|seg| seg.end as i64)
            .max()
            .unwrap_or(-1);

        if transcript_coding_end < 0 || tcds.segments.is_empty() {
            continue;
        }

        // Create a ZeroBasedHalfOpen for the transcript
        let transcript_interval = ZeroBasedHalfOpen::from_1based_inclusive(
            transcript_coding_start, 
            transcript_coding_end
        );
        
        // Check for overlap using the intersect method
        let overlaps_query = transcript_interval.intersect(&query_interval).is_some();

        if !overlaps_query {
            continue;
        }

        tcds.segments.sort_by_key(|seg| seg.start);

        println!("\nProcessing transcript: {}", tcds.transcript_id);
        writeln!(log_file, "\nProcessing transcript: {}", tcds.transcript_id)
            .expect("Failed to write to transcript_overlap.log");

        println!("Found {} CDS segments", tcds.segments.len());
        writeln!(log_file, "Found {} CDS segments", tcds.segments.len())
            .expect("Failed to write to transcript_overlap.log");

        stats.total_transcripts += 1;
        stats.total_cds_segments += tcds.segments.len();

        // Increment local overlap counter
        overlapping_transcript_count += 1;

        if tcds.segments.len() == 1 {
            stats.single_cds_transcripts += 1;
        } else {
            stats.multi_cds_transcripts += 1;
        }

        let has_gaps = tcds.segments.windows(2).any(|w| {
            let prev_end = w[0].end as i64;
            let next_start = w[1].start as i64;
            next_start - prev_end > 1
        });

        if has_gaps {
            stats.transcripts_with_gaps += 1;
        }

        let mut coding_segments = Vec::new();
        for (i, seg) in tcds.segments.iter().enumerate() {
            let segment_length = seg.len() as i64;
            let frame = tcds.frames.get(i).copied().unwrap_or(0);
            println!(
                "  Segment {}: {}-{} (length: {}, frame: {})",
                i + 1,
                seg.start,
                seg.end,
                segment_length,
                frame
            );
            writeln!(
                log_file,
                "  Segment {}: {}-{} (length: {}, frame: {})",
                i + 1,
                seg.start,
                seg.end,
                segment_length,
                frame
            )
            .expect("Failed to write to transcript_overlap.log");
            coding_segments.push((seg.start as i64, seg.end as i64));
        }

        let total_coding_length: i64 = tcds.segments.iter().map(|seg| seg.len() as i64).sum();
        stats.total_coding_length += total_coding_length;

        match stats.shortest_transcript_length {
            None => stats.shortest_transcript_length = Some(total_coding_length),
            Some(current) => {
                if total_coding_length < current {
                    stats.shortest_transcript_length = Some(total_coding_length)
                }
            }
        }
        match stats.longest_transcript_length {
            None => stats.longest_transcript_length = Some(total_coding_length),
            Some(current) => {
                if total_coding_length > current {
                    stats.longest_transcript_length = Some(total_coding_length)
                }
            }
        }

        if total_coding_length % 3 != 0 {
            stats.non_divisible_by_three += 1;
            println!(
                "  {} Warning: Total CDS length {} not divisible by 3",
                "!".yellow(),
                total_coding_length
            );
            writeln!(
                log_file,
                "  Warning: Total CDS length {} not divisible by 3",
                total_coding_length
            )
            .expect("Failed to write to transcript_overlap.log");

            println!(
                "    Remainder when divided by 3: {}",
                total_coding_length % 3
            );
            writeln!(
                log_file,
                "    Remainder when divided by 3: {}",
                total_coding_length % 3
            )
            .expect("Failed to write to transcript_overlap.log");

            println!(
                "    Individual segment lengths: {:?}",
                tcds.segments
                    .iter()
                    .map(|seg| seg.len())
                    .collect::<Vec<_>>()
            );
            writeln!(
                log_file,
                "    Individual segment lengths: {:?}",
                tcds.segments
                    .iter()
                    .map(|seg| seg.len())
                    .collect::<Vec<_>>()
            )
            .expect("Failed to write to transcript_overlap.log");
        }

        let min_start = tcds.segments.iter().map(|seg| seg.start as i64).min().unwrap();
        let max_end = tcds.segments.iter().map(|seg| seg.end as i64).max().unwrap();
        let transcript_span = max_end - min_start;

        println!("  CDS region: {}-{}", min_start, max_end);
        writeln!(log_file, "  CDS region: {}-{}", min_start, max_end)
            .expect("Failed to write to transcript_overlap.log");

        println!("    Genomic span: {}", transcript_span);
        writeln!(log_file, "    Genomic span: {}", transcript_span)
            .expect("Failed to write to transcript_overlap.log");

        println!("    Total coding length: {}", total_coding_length);
        writeln!(log_file, "    Total coding length: {}", total_coding_length)
            .expect("Failed to write to transcript_overlap.log");

        filtered.push(tcds);
    }

    if stats.total_transcripts > 0 {
        println!("\n{}", "CDS Processing Summary:".blue().bold());
        writeln!(log_file, "\nCDS Processing Summary:")
            .expect("Failed to write to transcript_overlap.log");

        println!("Total transcripts processed: {}", stats.total_transcripts);
        writeln!(
            log_file,
            "Total transcripts processed: {}",
            stats.total_transcripts
        )
        .expect("Failed to write to transcript_overlap.log");

        println!("Total CDS segments: {}", stats.total_cds_segments);
        writeln!(log_file, "Total CDS segments: {}", stats.total_cds_segments)
            .expect("Failed to write to transcript_overlap.log");

        println!(
            "Average segments per transcript: {:.2}",
            stats.total_cds_segments as f64 / stats.total_transcripts as f64
        );
        writeln!(
            log_file,
            "Average segments per transcript: {:.2}",
            stats.total_cds_segments as f64 / stats.total_transcripts as f64
        )
        .expect("Failed to write to transcript_overlap.log");

        println!(
            "Single-cds transcripts: {} ({:.1}%)",
            stats.single_cds_transcripts,
            100.0 * stats.single_cds_transcripts as f64 / stats.total_transcripts as f64
        );
        writeln!(
            log_file,
            "Single-cds transcripts: {} ({:.1}%)",
            stats.single_cds_transcripts,
            100.0 * stats.single_cds_transcripts as f64 / stats.total_transcripts as f64
        )
        .expect("Failed to write to transcript_overlap.log");

        println!(
            "Multi-cds transcripts: {} ({:.1}%)",
            stats.multi_cds_transcripts,
            100.0 * stats.multi_cds_transcripts as f64 / stats.total_transcripts as f64
        );
        writeln!(
            log_file,
            "Multi-cds transcripts: {} ({:.1}%)",
            stats.multi_cds_transcripts,
            100.0 * stats.multi_cds_transcripts as f64 / stats.total_transcripts as f64
        )
        .expect("Failed to write to transcript_overlap.log");

        println!(
            "Transcripts with gaps: {} ({:.1}%)",
            stats.transcripts_with_gaps,
            100.0 * stats.transcripts_with_gaps as f64 / stats.total_transcripts as f64
        );
        writeln!(
            log_file,
            "Transcripts with gaps: {} ({:.1}%)",
            stats.transcripts_with_gaps,
            100.0 * stats.transcripts_with_gaps as f64 / stats.total_transcripts as f64
        )
        .expect("Failed to write to transcript_overlap.log");

        println!(
            "Non-divisible by three: {} ({:.1}%)",
            stats.non_divisible_by_three,
            100.0 * stats.non_divisible_by_three as f64 / stats.total_transcripts as f64
        );
        writeln!(
            log_file,
            "Non-divisible by three: {} ({:.1}%)",
            stats.non_divisible_by_three,
            100.0 * stats.non_divisible_by_three as f64 / stats.total_transcripts as f64
        )
        .expect("Failed to write to transcript_overlap.log");

        println!("Total coding bases: {}", stats.total_coding_length);
        writeln!(
            log_file,
            "Total coding bases: {}",
            stats.total_coding_length
        )
        .expect("Failed to write to transcript_overlap.log");

        if let Some(shortest) = stats.shortest_transcript_length {
            println!("Shortest transcript: {} bp", shortest);
            writeln!(log_file, "Shortest transcript: {} bp", shortest)
                .expect("Failed to write to transcript_overlap.log");
        }
        if let Some(longest) = stats.longest_transcript_length {
            println!("Longest transcript: {} bp", longest);
            writeln!(log_file, "Longest transcript: {} bp", longest)
                .expect("Failed to write to transcript_overlap.log");
        }
        let avg_len = if stats.total_transcripts == 0 {
            0.0
        } else {
            stats.total_coding_length as f64 / stats.total_transcripts as f64
        };
        println!("Average transcript length: {:.1} bp", avg_len);
        writeln!(log_file, "Average transcript length: {:.1} bp", avg_len)
            .expect("Failed to write to transcript_overlap.log");
    }

    if filtered.is_empty() {
        println!("{}", "No valid CDS regions found!".red());
        writeln!(log_file, "No valid CDS regions found!")
            .expect("Failed to write to transcript_overlap.log");
    }

    writeln!(
        log_file,
        "Summary: {} transcripts overlap query region {}..{}",
        overlapping_transcript_count, query.start, query.end
    )
    .expect("Failed to write to transcript_overlap.log");

    filtered
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
    seqinfo_storage_unfiltered: Arc<Mutex<Vec<SeqInfo>>>,
    seqinfo_storage_filtered: Arc<Mutex<Vec<SeqInfo>>>,
    position_allele_map_unfiltered: Arc<Mutex<HashMap<i64, (char, char)>>>,
    position_allele_map_filtered: Arc<Mutex<HashMap<i64, (char, char)>>>,
) -> Result<
    (
        Vec<Variant>,
        Vec<Variant>,
        Vec<String>,
        i64,
        MissingDataInfo,
        FilteringStats,
    ),
    VcfError,
> {
    // Initialize the VCF reader.
    let mut reader = open_vcf_reader(file)?;
    let mut sample_names = Vec::new();
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

    // Small vectors to hold variants in batches, limiting memory usage.
    let unfiltered_variants = Arc::new(Mutex::new(Vec::with_capacity(10000)));
    let filtered_variants = Arc::new(Mutex::new(Vec::with_capacity(10000)));

    // Shared stats.
    let missing_data_info = Arc::new(Mutex::new(MissingDataInfo::default()));
    let _filtering_stats = Arc::new(Mutex::new(FilteringStats::default()));

    // Progress UI setup.
    let is_gzipped = file.extension().and_then(|s| s.to_str()) == Some("gz");
    let progress_bar = Arc::new(if is_gzipped {
        ProgressBar::new_spinner()
    } else {
        let file_size = fs::metadata(file)?.len();
        ProgressBar::new(file_size)
    });
    let style = if is_gzipped {
        ProgressStyle::default_spinner()
            .template("{spinner:.bold.green} VCF {elapsed_precise} {msg}")
            .expect("Spinner template error")
            .tick_strings(&["░░", "▒▒", "▓▓", "██", "▓▓", "▒▒"])
    } else {
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {bytes}/{total_bytes} {msg}")
            .expect("Progress bar template error")
            .progress_chars("=>-")
    };
    progress_bar.set_style(style);
    let processing_complete = Arc::new(AtomicBool::new(false));
    let processing_complete_clone = Arc::clone(&processing_complete);
    let progress_bar_clone = Arc::clone(&progress_bar); // Clone Arc for progress thread
    let progress_thread = thread::spawn(move || {
        while !processing_complete_clone.load(Ordering::Relaxed) {
            progress_bar_clone.tick();
            thread::sleep(Duration::from_millis(100));
        }
        progress_bar_clone.finish_with_message("Finished reading VCF");
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
                progress_bar.inc(local_buffer.len() as u64);
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
        let pos_map_unfiltered = Arc::clone(&position_allele_map_unfiltered);
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
                    &pos_map_unfiltered,
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
        let unfiltered_variants = Arc::clone(&unfiltered_variants);
        let filtered_variants = Arc::clone(&filtered_variants);
        let missing_data_info = Arc::clone(&missing_data_info);
        let _filtering_stats = Arc::clone(&_filtering_stats);
        move || -> Result<(), VcfError> {
            while let Ok(msg) = result_receiver.recv() {
                match msg {
                    Ok((Some((variant, passes)), local_miss, mut local_stats)) => {
                        {
                            let mut u = unfiltered_variants.lock();
                            u.push(variant.clone());
                        }
                        if passes {
                            let mut f = filtered_variants.lock();
                            f.push(variant.clone());
                            position_allele_map_filtered.lock().insert(
                                variant.position,
                                position_allele_map_unfiltered
                                    .lock()
                                    .get(&variant.position)
                                    .copied()
                                    .unwrap_or(('N', 'N')),
                            );
                        }

                        // Write SeqInfo for unfiltered
                        // ???
                        //
                        // TODO

                        // Write SeqInfo for filtered
                        // ???
                        //
                        // TODO

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

    // Display the final SeqInfo if present.
    if !seqinfo_storage_unfiltered.lock().is_empty() {
        display_seqinfo_entries(&seqinfo_storage_unfiltered.lock(), 12);
    }
    if !seqinfo_storage_filtered.lock().is_empty() {
        display_seqinfo_entries(&seqinfo_storage_filtered.lock(), 12);
    }

    // Extract final variant vectors.
    let final_unfiltered = Arc::try_unwrap(unfiltered_variants)
        .map_err(|_| VcfError::Parse("Unfiltered variants still have multiple owners".to_string()))?
        .into_inner();
    let final_filtered = Arc::try_unwrap(filtered_variants)
        .map_err(|_| VcfError::Parse("Filtered variants still have multiple owners".to_string()))?
        .into_inner();

    // Extract stats.
    let final_miss = Arc::try_unwrap(missing_data_info)
        .map_err(|_| VcfError::Parse("Missing data info still has multiple owners".to_string()))?
        .into_inner();
    let final_stats = Arc::try_unwrap(_filtering_stats)
        .map_err(|_| VcfError::Parse("Filtering stats still have multiple owners".to_string()))?
        .into_inner();
    let final_names = Arc::try_unwrap(arc_sample_names)
        .map_err(|_| VcfError::Parse("Sample names have multiple owners".to_string()))?;

    Ok((
        final_unfiltered,
        final_filtered,
        final_names,
        chr_length,
        final_miss,
        final_stats,
    ))
}

// Struct to hold CDS region information
pub struct CdsRegion {
    pub transcript_id: String,
    // Store (start, end, strand_char, frame)
    pub segments: Vec<(i64, i64, char, i64)>,
}

// Write sequences to PHYLIP file
fn write_phylip_file(
    output_file: &str,
    hap_sequences: &HashMap<String, Vec<char>>,
    transcript_id: &str,
) -> Result<(), VcfError> {
    println!("Writing {} for transcript {}", output_file, transcript_id);
    let file = File::create(output_file).map_err(|e| {
        VcfError::Io(io::Error::new(
            io::ErrorKind::Other,
            format!("Failed to create PHYLIP file '{}': {:?}", output_file, e),
        ))
    })?;
    let mut writer = BufWriter::new(file);

    let mut length = None;
    for (sample_name, seq_chars) in hap_sequences {
        let this_len = seq_chars.len();
        if let Some(expected_len) = length {
            if this_len != expected_len {
                return Err(VcfError::Parse(format!(
                    "Mismatched alignment length for {}: got {}, expected {}",
                    sample_name, this_len, expected_len
                )));
            }
        } else {
            length = Some(this_len);
        }
    }
    let n = hap_sequences.len();
    let m = length.unwrap_or(0);
    writeln!(writer, "{} {}", n, m).map_err(|e| {
        VcfError::Io(io::Error::new(
            io::ErrorKind::Other,
            format!(
                "Failed to write PHYLIP header to '{}': {:?}",
                output_file, e
            ),
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

    writer.flush().map_err(|e| {
        VcfError::Io(io::Error::new(
            io::ErrorKind::Other,
            format!("Failed to flush PHYLIP file '{}': {:?}", output_file, e),
        ))
    })?;

    Ok(())
}

fn process_variant(
    line: &str,
    chr: &str,
    region: ZeroBasedHalfOpen,
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

    _filtering_stats.total_variants += 1; // DO NOT MOVE THIS LINE ABOVE THE CHECK FOR WITHIN RANGE

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
            _filtering_stats._filtered_variants += 1;
            _filtering_stats.filtered_due_to_allow += 1;
            _filtering_stats
                .filtered_positions
                .insert(zero_based_position);
            _filtering_stats.add_example(format!("{}: Filtered due to allow", line.trim()));
            return Ok(None);
        }
    } else if allow_regions.is_some() {
        _filtering_stats._filtered_variants += 1;
        _filtering_stats.filtered_due_to_allow += 1;
        _filtering_stats
            .filtered_positions
            .insert(zero_based_position);
        _filtering_stats.add_example(format!("{}: Filtered due to allow", line.trim()));
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
            _filtering_stats._filtered_variants += 1;
            _filtering_stats.filtered_due_to_mask += 1;
            _filtering_stats
                .filtered_positions
                .insert(zero_based_position);
            _filtering_stats.add_example(format!("{}: Filtered due to mask", line.trim()));
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
        let ref_allele = fields[3].chars().next().unwrap_or('N');
        let alt_allele = fields[4].chars().next().unwrap_or('N');
        position_allele_map
            .lock()
            .insert(zero_based_position, (ref_allele, alt_allele));
    }

    let alt_alleles: Vec<&str> = fields[4].split(',').collect();
    let is_multiallelic = alt_alleles.len() > 1;
    if is_multiallelic {
        _filtering_stats.multi_allelic_variants += 1;
        eprintln!(
            "{}",
            format!(
                "Warning: Multi-allelic site detected at position {}, which is not supported. Skipping.",
                one_based_vcf_position.0
            ).yellow()
        );
        _filtering_stats.add_example(format!(
            "{}: Filtered due to multi-allelic variant",
            line.trim()
        ));
        return Ok(None);
    }
    if alt_alleles.get(0).map_or(false, |s| s.len() > 1) {
        _filtering_stats.multi_allelic_variants += 1;
        eprintln!(
            "{}",
            format!(
                "Warning: Multi-nucleotide ALT detected at position {}, which is not supported. Skipping.",
                one_based_vcf_position.0
            ).yellow()
        );
        _filtering_stats.add_example(format!(
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
    let mut _num_samples_below_gq = 0;
    for gt_field in fields[9..].iter() {
        let gt_subfields: Vec<&str> = gt_field.split(':').collect();
        if gq_index >= gt_subfields.len() {
            return Err(VcfError::Parse(format!(
                "GQ value missing in sample genotype field at chr{}:{}",
                chr, one_based_vcf_position.0
            )));
        }
        let gq_str = gt_subfields[gq_index];

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
            _num_samples_below_gq += 1;
        }
    }

    if sample_has_low_gq {
        // Skip this variant
        _filtering_stats.low_gq_variants += 1;
        _filtering_stats._filtered_variants += 1;
        _filtering_stats
            .filtered_positions
            .insert(zero_based_position);
        _filtering_stats.add_example(format!("{}: Filtered due to low GQ", line.trim()));

        let has_missing_genotypes = genotypes.iter().any(|gt| gt.is_none());
        let passes_filters = !sample_has_low_gq && !has_missing_genotypes && !is_multiallelic;
        let variant = Variant {
            position: zero_based_position,
            genotypes: genotypes.clone(),
        };
        return Ok(Some((variant, passes_filters)));
    }

    if genotypes.iter().any(|gt| gt.is_none()) {
        _filtering_stats.missing_data_variants += 1;
        _filtering_stats.add_example(format!("{}: Filtered due to missing data", line.trim()));
    }

    let has_missing_genotypes = genotypes.iter().any(|gt| gt.is_none());
    let passes_filters = !sample_has_low_gq && !has_missing_genotypes && !is_multiallelic;

    // Update filtering stats if variant is filtered out
    if !passes_filters {
        _filtering_stats._filtered_variants += 1;
        _filtering_stats
            .filtered_positions
            .insert(zero_based_position);

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
            _filtering_stats.add_example(format!(
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
