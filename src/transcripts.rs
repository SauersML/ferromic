use crate::stats::{
    calculate_adjusted_sequence_length, calculate_inversion_allele_frequency, calculate_per_site,
    calculate_pi, calculate_watterson_theta, SiteDiversity,
};

use crate::parse::{
    find_vcf_file, open_vcf_reader, parse_gtf_file, read_reference_sequence, validate_vcf_header,
};

use crate::process::{
    ZeroBasedHalfOpen, ZeroBasedPosition, VcfError, Variant, HaplotypeSide, QueryRegion, TEMP_DIR,
    create_temp_dir, map_sample_names_to_indices, get_haplotype_indices_for_group
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
use tempfile::TempDir;
use once_cell::sync::Lazy;

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
        let log_file_path = {
            let mut locked_opt = TEMP_DIR.lock();
            if locked_opt.is_none() {
                *locked_opt = Some(create_temp_dir().expect("Failed to create temporary directory"));
            }
            if let Some(dir) = locked_opt.as_ref() {
                dir.path().join("cds_validation.log")
            } else {
                return Err("Failed to access temporary directory".to_string());
            }
        };

        let mut log_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_file_path)
            .map_err(|e| format!("Failed to open {}: {}", log_file_path.display(), e))?;

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

pub fn initialize_hap_sequences(
    haplotype_indices: &[(usize, HaplotypeSide)],
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
    // Iterate over haplotype indices and sides, binding directly to tuple components
    for (sample_idx, hap_idx) in haplotype_indices {
        // We use a consistent naming format for each sample/haplotype: "SampleName_L" or "SampleName_R."
        let sample_name = format!(
            "{}_{}",
            sample_names[*sample_idx],
            match hap_idx {
                HaplotypeSide::Left => "L",
                HaplotypeSide::Right => "R",
            }
        );

        // Collect the bytes into a new vector to store in our map.
        let sequence_u8: Vec<u8> = region_slice.iter().copied().collect();

        // Insert the new haplotype sequence into the hash map.
        hap_sequences.insert(sample_name, sequence_u8);
    }

    hap_sequences
}

pub fn apply_variants_to_transcripts(
    variants: &[Variant],
    haplotype_indices: &[(usize, HaplotypeSide)],
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
                sample_names[sample_idx],
                match hap_idx {
                    HaplotypeSide::Left => "L",
                    HaplotypeSide::Right => "R",
                }
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

pub fn generate_batch_statistics(hap_sequences: &HashMap<String, Vec<u8>>) -> Result<(), VcfError> {
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

pub fn prepare_to_write_cds(
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

// This function filters the TranscriptCDS by QueryRegion overlap and prints stats
pub fn filter_and_log_transcripts(
    transcripts: Vec<TranscriptCDS>,
    query: QueryRegion,
) -> Vec<TranscriptCDS> {
    use colored::Colorize;
    use std::fs::OpenOptions;
    use std::io::{BufWriter, Write};

    // Open or create a log file once per call in the temporary directory
    let log_file_path = {
        let locked_opt = TEMP_DIR.lock();
        if let Some(dir) = locked_opt.as_ref() {
            dir.path().join("transcript_overlap.log")
        } else {
            return Vec::new(); // Return empty vector if temp dir not available
        }
    };
        
    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_file_path)
        .expect("Failed to open transcript_overlap.log in temporary directory");
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
        let transcript_interval = ZeroBasedHalfOpen {
            start: transcript_coding_start as usize,
            end: transcript_coding_end as usize,
        };
        
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

// Struct to hold CDS region information
pub struct CdsRegion {
    pub transcript_id: String,
    // Store (start, end, strand_char, frame)
    pub segments: Vec<(i64, i64, char, i64)>,
}

// Write sequences to PHYLIP file
pub fn write_phylip_file(
    output_file: &str,
    hap_sequences: &HashMap<String, Vec<char>>,
    transcript_id: &str,
) -> Result<(), VcfError> {
    // Acquire or create the TempDir in a single scope.
    let temp_output_file = {
        let mut locked_opt = TEMP_DIR.lock();
        if locked_opt.is_none() {
            *locked_opt = Some(create_temp_dir().expect("Failed to create temporary directory"));
        }
        if let Some(dir) = locked_opt.as_ref() {
            dir.path().join(output_file)
        } else {
            return Err(VcfError::Parse("Failed to access temporary directory".to_string()));
        }
    };

    println!("Writing {} for transcript {}", temp_output_file.display(), transcript_id);
    let file = File::create(&temp_output_file).map_err(|e| {
        VcfError::Io(io::Error::new(
            io::ErrorKind::Other,
            format!("Failed to create PHYLIP file '{}': {:?}", temp_output_file.display(), e),
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
