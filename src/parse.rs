use crate::process::{ConfigEntry, VcfError, ZeroBasedHalfOpen};
use crate::transcript::TranscriptCDS;

use colored::Colorize;
use flate2::read::MultiGzDecoder;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};

// Function to parse regions file (mask or allow)
pub fn parse_regions_file(
    path: &Path,
) -> Result<HashMap<String, Vec<ZeroBasedHalfOpen>>, VcfError> {
    let is_bed_file = path.extension().and_then(|s| s.to_str()) == Some("bed");

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut regions: HashMap<String, Vec<ZeroBasedHalfOpen>> = HashMap::new();

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 3 {
            eprintln!(
                "{}",
                format!("Skipping invalid line {}: '{}'", line_num + 1, line).red()
            );
            continue;
        }

        let chr = fields[0].trim_start_matches("chr").to_string();
        let raw_start: i64 = match fields[1].trim().parse() {
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
        let raw_end: i64 = match fields[2].trim().parse() {
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

        let interval = if is_bed_file {
            ZeroBasedHalfOpen {
                start: raw_start as usize,
                end: raw_end as usize,
            }
        } else {
            ZeroBasedHalfOpen::from_1based_inclusive(raw_start, raw_end)
        };

        regions.entry(chr.clone()).or_default().push(interval);
    }

    for intervals in regions.values_mut() {
        intervals.sort_by_key(|iv| iv.start);
    }

    Ok(regions)
}

// Check 0-based vs. 1 based, half-open vs. inclusive, between config file and mask/allow file
pub fn parse_config_file(path: &Path) -> Result<Vec<ConfigEntry>, VcfError> {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(path)
        .map_err(|e| VcfError::Io(e.into()))?;

    let headers = reader
        .headers()
        .map_err(|e| VcfError::Io(e.into()))?
        .clone();
    let sample_names: Vec<String> = headers.iter().skip(7).map(String::from).collect();

    // Check if the number of sample names is consistent
    if sample_names.is_empty() {
        eprintln!("{}", "Error: No sample names found in the configuration file header after skipping the first 7 columns. Tabs must separate all columns, including sample names.".red());
        return Err(VcfError::Parse(
            "No sample names found in config file header.".to_string(),
        ));
    }

    let mut entries = Vec::new();
    let mut invalid_genotypes = 0;
    let mut total_genotypes = 0;

    for (line_num, result) in reader.records().enumerate() {
        let record = result.map_err(|e| VcfError::Io(e.into()))?;

        // Check if the record has the expected number of fields
        if record.len() != headers.len() {
            eprintln!("{}", format!("Error: Record on line {} does not have the same number of fields as the header. Expected {}, found {}. Please check for missing tabs in the config file.", line_num + 2, headers.len(), record.len()).red());
            return Err(VcfError::Parse(format!(
                "Mismatched number of fields in record on line {}",
                line_num + 2
            )));
        }

        // Normalize chromosome name by removing "chr" prefix
        let seqname = record
            .get(0)
            .ok_or(VcfError::Parse("Missing seqname".to_string()))?
            .trim()
            .trim_start_matches("chr")
            .to_string();
        let start_pos: i64 = record
            .get(1)
            .ok_or(VcfError::Parse("Missing start".to_string()))?
            .parse()
            .map_err(|_| VcfError::Parse("Invalid start".to_string()))?;
        let end_pos: i64 = record
            .get(2)
            .ok_or(VcfError::Parse("Missing end".to_string()))?
            .parse()
            .map_err(|_| VcfError::Parse("Invalid end".to_string()))?;
        let interval = ZeroBasedHalfOpen::from_1based_inclusive(start_pos, end_pos);

        let mut samples_unfiltered = HashMap::new();
        let mut samples_filtered = HashMap::new();

        for (i, field) in record.iter().enumerate().skip(7) {
            total_genotypes += 1;
            if i < sample_names.len() + 7 {
                let sample_name = &sample_names[i - 7];

                // For samples_unfiltered (split on '_')
                let genotype_str_unfiltered = field.split('_').next().unwrap_or("");

                if genotype_str_unfiltered.len() >= 3
                    && genotype_str_unfiltered.chars().nth(1) == Some('|')
                {
                    let left_char = genotype_str_unfiltered.chars().nth(0).unwrap();
                    let right_char = genotype_str_unfiltered.chars().nth(2).unwrap();
                    if let (Some(left), Some(right)) =
                        (left_char.to_digit(10), right_char.to_digit(10))
                    {
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
                eprintln!(
                    "Warning: More genotype fields than sample names at line {}.",
                    line_num + 2
                );
            }
        }

        if samples_unfiltered.is_empty() {
            println!(
                "Warning: No valid genotypes found for region {}:{}-{}",
                seqname, start_pos, end_pos
            );
            continue;
        }

        entries.push(ConfigEntry {
            seqname,
            interval,
            samples_unfiltered,
            samples_filtered,
        });
    }

    let invalid_percentage = (invalid_genotypes as f64 / total_genotypes as f64) * 100.0;
    println!(
        "Number of invalid genotypes: {} ({:.2}%)",
        invalid_genotypes, invalid_percentage
    );

    Ok(entries)
}

pub fn parse_region(region: &str) -> Result<ZeroBasedHalfOpen, VcfError> {
    let parts: Vec<&str> = region.split('-').collect();
    if parts.len() != 2 {
        return Err(VcfError::InvalidRegion(
            "Invalid region format. Use start-end".to_string(),
        ));
    }
    let start_1based: i64 = parts[0]
        .parse()
        .map_err(|_| VcfError::InvalidRegion("Invalid start position".to_string()))?;
    let end_1based: i64 = parts[1]
        .parse()
        .map_err(|_| VcfError::InvalidRegion("Invalid end position".to_string()))?;
    if start_1based >= end_1based {
        return Err(VcfError::InvalidRegion(
            "Start position must be less than end position".to_string(),
        ));
    }
    let interval = ZeroBasedHalfOpen::from_1based_inclusive(start_1based, end_1based);
    Ok(interval)
}

pub fn find_vcf_file(folder: &str, chr: &str) -> Result<PathBuf, VcfError> {
    let path = Path::new(folder);
    let chr_specific_files: Vec<_> = fs::read_dir(path)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            let path = entry.path();
            let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            let chr_pattern = format!("chr{}", chr);
            (file_name.starts_with(&chr_pattern) || file_name.starts_with(chr))
                && (file_name.ends_with(".vcf") || file_name.ends_with(".vcf.gz"))
                && file_name
                    .chars()
                    .nth(chr_pattern.len())
                    .map_or(false, |c| !c.is_ascii_digit())
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
                (file_name.starts_with(&chr_pattern) || file_name.starts_with(chr))
                    && file_name
                        .chars()
                        .nth(chr_pattern.len())
                        .map_or(false, |c| !c.is_ascii_digit())
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
                let choice: usize = input
                    .trim()
                    .parse()
                    .map_err(|_| VcfError::Parse("Invalid input".to_string()))?;

                chr_specific_files
                    .get(choice - 1)
                    .cloned()
                    .ok_or_else(|| VcfError::Parse("Invalid file number".to_string()))
            }
        }
    }
}

pub fn open_vcf_reader(path: &Path) -> Result<Box<dyn BufRead + Send>, VcfError> {
    let file = File::open(path)?;

    if path.extension().and_then(|s| s.to_str()) == Some("gz") {
        let decoder = MultiGzDecoder::new(file);
        Ok(Box::new(BufReader::new(decoder)))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

// Function to validate VCF header
pub fn validate_vcf_header(header: &str) -> Result<(), VcfError> {
    let fields: Vec<&str> = header.split('\t').collect();
    let required_fields = vec![
        "#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT",
    ];

    if fields.len() < required_fields.len()
        || fields[..required_fields.len()] != required_fields[..]
    {
        return Err(VcfError::InvalidVcfFormat(
            "Invalid VCF header format".to_string(),
        ));
    }
    Ok(())
}

pub fn read_reference_sequence(
    fasta_path: &Path,
    chr: &str,
    region: ZeroBasedHalfOpen,
) -> Result<Vec<u8>, VcfError> {
    // Create reader for the FASTA file and its index
    let mut reader = bio::io::fasta::IndexedReader::from_file(&fasta_path).map_err(|e| {
        VcfError::Io(io::Error::new(
            io::ErrorKind::Other,
            format!("Failed to open FASTA file: {}", e),
        ))
    })?;

    // Try both with and without "chr" prefix
    let chr_with_prefix = if !chr.starts_with("chr") {
        format!("chr{}", chr)
    } else {
        chr.to_string()
    };

    // Find chromosome
    let sequences = reader.index.sequences();
    let seq_info = sequences
        .iter()
        .find(|seq| seq.name == chr_with_prefix || seq.name == chr)
        .ok_or_else(|| {
            VcfError::Parse(format!(
                "Chromosome {} (or {}) not found in reference",
                chr, chr_with_prefix
            ))
        })?;

    let seq_length = seq_info.len;
    let actual_chr_name = seq_info.name.as_str();

    // Clamp the requested end to the chromosome length
    let clamped_end = std::cmp::min(region.end as u64, seq_length);
    let clamped_start = std::cmp::min(region.start as u64, clamped_end);

    // The region length in zero-based coords
    let region_length = (clamped_end - clamped_start) as usize;

    let mut sequence = Vec::with_capacity(region_length);

    // Fetch region from the reference
    reader
        .fetch(actual_chr_name, clamped_start, clamped_end)
        .map_err(|_| {
            VcfError::Io(io::Error::new(
                io::ErrorKind::Other,
                format!(
                    "Failed to fetch region {}:{}-{}",
                    actual_chr_name, clamped_start, clamped_end
                ),
            ))
        })?;

    reader.read(&mut sequence).map_err(|e| {
        VcfError::Io(io::Error::new(
            io::ErrorKind::Other,
            format!(
                "Failed to read sequence for region {}:{}-{}: {}",
                actual_chr_name, clamped_start, clamped_end, e
            ),
        ))
    })?;

    if sequence.len() != region_length {
        return Err(VcfError::Parse(format!(
            "Expected sequence length {} but got {} for region {}:{}-{}",
            region_length,
            sequence.len(),
            actual_chr_name,
            clamped_start,
            clamped_end
        )));
    }

    let invalid_chars: Vec<(usize, u8)> = sequence
        .iter()
        .enumerate()
        .filter(|(_, &b)| !matches!(b.to_ascii_uppercase(), b'A' | b'C' | b'G' | b'T' | b'N'))
        .take(10)
        .map(|(i, &b)| (i, b))
        .collect();

    if !invalid_chars.is_empty() {
        println!("Found invalid characters:");
        for (pos, ch) in invalid_chars {
            println!(
                "Position {}: '{}' (ASCII: {})",
                pos,
                String::from_utf8_lossy(&[ch]),
                ch
            );
        }
        return Err(VcfError::Parse(format!(
            "Invalid nucleotides found in sequence for region {}:{}-{}",
            actual_chr_name, clamped_start, clamped_end
        )));
    }

    Ok(sequence)
}

// IN PROGRESS
// Helper function to parse GTF file and extract CDS regions
// GTF and GFF use 1-based coordinate system
pub fn parse_gtf_file(gtf_path: &Path, chr: &str) -> Result<Vec<TranscriptCDS>, VcfError> {
    // Print overall GTF parsing context.
    println!("\nParsing GTF file for chromosome: {}", chr);

    // Open the GTF file.
    let file = File::open(gtf_path).map_err(|e| {
        VcfError::Io(io::Error::new(
            io::ErrorKind::NotFound,
            format!("GTF file not found: {:?}", e),
        ))
    })?;
    let reader = BufReader::new(file);

    // Map of transcript_id -> list of CDS segments.
    let mut transcript_map: HashMap<String, Vec<(i64, i64, char, i64)>> = HashMap::new();

    let mut skipped_lines = 0;
    let mut processed_lines = 0;
    let mut transcripts_found = HashSet::new();
    let mut malformed_attributes = 0;

    println!("Reading GTF entries...");

    // Read each line, parse if CDS, and store in transcript_map.
    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        if line.starts_with('#') {
            continue;
        }

        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 9 {
            skipped_lines += 1;
            continue;
        }

        let seqname = fields[0].trim().trim_start_matches("chr");
        if seqname != chr.trim_start_matches("chr") {
            continue;
        }

        // Only process CDS features.
        if fields[2] != "CDS" {
            continue;
        }

        processed_lines += 1;
        if processed_lines % 10000 == 0 {
            println!("Processed {} CDS entries...", processed_lines);
        }

        let start: i64 = match fields[3].parse() {
            Ok(s) => s,
            Err(_) => {
                eprintln!(
                    "Warning: Invalid start position at line {}, skipping",
                    line_num + 1
                );
                skipped_lines += 1;
                continue;
            }
        };

        let end: i64 = match fields[4].parse() {
            Ok(e) => e,
            Err(_) => {
                eprintln!(
                    "Warning: Invalid end position at line {}, skipping",
                    line_num + 1
                );
                skipped_lines += 1;
                continue;
            }
        };

        let strand_char = fields[6].chars().next().unwrap_or('.');
        let frame: i64 = fields[7].parse().unwrap_or_else(|_| {
            eprintln!("Warning: Invalid frame at line {}, using 0", line_num + 1);
            0
        });

        // Parse attributes to find transcript_id (and gene_name, if present).
        let attributes = fields[8];
        let mut transcript_id = None;
        let mut gene_name = None;

        for attr in attributes.split(';') {
            let attr = attr.trim();
            let parts: Vec<&str> = if attr.contains('=') {
                attr.splitn(2, '=').collect()
            } else {
                attr.splitn(2, ' ').collect()
            };

            if parts.len() != 2 {
                continue;
            }

            let key = parts[0].trim();
            let value = parts[1].trim().trim_matches('"').trim_matches('\'');

            match key {
                "transcript_id" => transcript_id = Some(value.to_string()),
                "gene_name" => gene_name = Some(value.to_string()),
                _ => {}
            }
        }

        let transcript_id = match transcript_id {
            Some(id) => id,
            None => {
                malformed_attributes += 1;
                if malformed_attributes <= 5 {
                    eprintln!(
                        "Warning: Could not find transcript_id in attributes at line {}: {}",
                        line_num + 1,
                        attributes
                    );
                }
                continue;
            }
        };

        // Track transcripts found
        if let Some(gene) = gene_name {
            transcripts_found.insert(format!("{}:{}", gene, transcript_id));
        } else {
            transcripts_found.insert(transcript_id.clone());
        }

        // Push this CDS segment into the map for that transcript.
        transcript_map
            .entry(transcript_id)
            .or_default()
            .push((start, end, strand_char, frame));
    }

    // Print summary of how many lines, transcripts, etc.
    println!("\nFinished reading GTF.");
    println!("Total CDS entries processed: {}", processed_lines);
    println!("Skipped lines: {}", skipped_lines);
    println!("Unique transcripts found: {}", transcripts_found.len());
    if malformed_attributes > 0 {
        println!(
            "Entries with missing transcript IDs: {}",
            malformed_attributes
        );
    }

    // Now build a vector of TranscriptCDS objects.
    let mut transcripts_vec = Vec::new();

    for (tid, mut segments) in transcript_map {
        segments.sort_by_key(|&(s, _, _, _)| s);

        if !segments.is_empty() {
            let strand = segments[0].2;
            if strand == '-' {
                segments.reverse();
            }
        }

        let strand_char = if segments.is_empty() { '.' } else { segments[0].2 };
        let mut frames_vec: Vec<i64> = segments.iter().map(|&(_s, _e, _str, f)| f).collect();
        let mut seg_intervals: Vec<ZeroBasedHalfOpen> = segments
            .iter()
            .map(|&(s, e, _, _)| ZeroBasedHalfOpen::from_1based_inclusive(s, e))
            .collect();

        if strand_char == '-' {
            seg_intervals.reverse();
            frames_vec.reverse();
        }

        transcripts_vec.push(TranscriptCDS {
            transcript_id: tid,
            strand: strand_char,
            frames: frames_vec,
            segments: seg_intervals,
        });
    }

    println!(
        "\nNumber of transcripts returned: {}",
        transcripts_vec.len()
    );
    if transcripts_vec.is_empty() {
        println!("No CDS transcripts parsed for chromosome {}", chr);
    }

    // Return them all (we do not filter by user region here).
    Ok(transcripts_vec)
}
