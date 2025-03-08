use crate::process::{ConfigEntry, VcfError, ZeroBasedHalfOpen};
use crate::transcripts::TranscriptAnnotationCDS;
use crate::progress::{
    log, LogLevel, init_step_progress, update_step_progress,
    finish_step_progress, create_spinner, set_stage, ProcessingStage
};

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
    set_stage(ProcessingStage::Global);
    let is_bed_file = path.extension().and_then(|s| s.to_str()) == Some("bed");
    
    log(LogLevel::Info, &format!("Parsing regions file: {}", path.display()));
    let spinner = create_spinner(&format!("Parsing regions file: {}", path.display()));

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut regions: HashMap<String, Vec<ZeroBasedHalfOpen>> = HashMap::new();

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 3 {
            let error_msg = format!("Skipping invalid line {}: '{}'", line_num + 1, line);
            log(LogLevel::Warning, &error_msg);
            continue;
        }

        let chr = fields[0].trim_start_matches("chr").to_string();
        let raw_start: i64 = match fields[1].trim().parse() {
            Ok(val) => val,
            Err(_) => {
                let error_msg = format!(
                    "Invalid start position on line {}: '{}'",
                    line_num + 1,
                    fields[1]
                );
                log(LogLevel::Warning, &error_msg);
                continue;
            }
        };
        let raw_end: i64 = match fields[2].trim().parse() {
            Ok(val) => val,
            Err(_) => {
                let error_msg = format!(
                    "Invalid end position on line {}: '{}'",
                    line_num + 1,
                    fields[2]
                );
                log(LogLevel::Warning, &error_msg);
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
    
    let num_regions: usize = regions.values().map(|v| v.len()).sum();
    spinner.finish_with_message(format!("Parsed {} regions across {} chromosomes", num_regions, regions.len()));
    log(LogLevel::Info, &format!("Completed parsing {} regions", num_regions));

    Ok(regions)
}

// Check 0-based vs. 1 based, half-open vs. inclusive, between config file and mask/allow file
pub fn parse_config_file(path: &Path) -> Result<Vec<ConfigEntry>, VcfError> {
    set_stage(ProcessingStage::Global);
    log(LogLevel::Info, &format!("Parsing config file: {}", path.display()));
    
    let spinner = create_spinner(&format!("Reading config file: {}", path.display()));
    
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
        let error_msg = "Error: No sample names found in the configuration file header after skipping the first 7 columns. Tabs must separate all columns, including sample names.";
        log(LogLevel::Error, error_msg);
        spinner.finish_with_message("Failed to parse config file: no sample names found");
        return Err(VcfError::Parse(
            "No sample names found in config file header.".to_string(),
        ));
    }
    
    spinner.set_message(format!("Found {} sample columns in config file", sample_names.len()));

    let mut entries = Vec::new();
    let mut invalid_genotypes = 0;
    let mut total_genotypes = 0;

    for (line_num, result) in reader.records().enumerate() {
        let record = result.map_err(|e| VcfError::Io(e.into()))?;

        // Check if the record has the expected number of fields
        if record.len() != headers.len() {
            let error_msg = format!(
                "Error: Record on line {} does not have the same number of fields as the header. Expected {}, found {}. Please check for missing tabs in the config file.", 
                line_num + 2, headers.len(), record.len()
            );
            log(LogLevel::Error, &error_msg);
            spinner.finish_with_message("Failed to parse config file: mismatched field count");
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
            log(LogLevel::Warning, &format!(
                "No valid genotypes found for region {}:{}-{}",
                seqname, start_pos, end_pos
            ));
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
    spinner.finish_with_message(format!(
        "Parsed {} entries with {} samples (invalid genotypes: {:.2}%)",
        entries.len(), sample_names.len(), invalid_percentage
    ));
    
    log(LogLevel::Info, &format!(
        "Finished parsing config file. Found {} entries with {} samples. Invalid genotypes: {} ({:.2}%)",
        entries.len(), sample_names.len(), invalid_genotypes, invalid_percentage
    ));

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
    set_stage(ProcessingStage::Global);
    log(LogLevel::Info, &format!("Searching for VCF file for chromosome {} in folder: {}", chr, folder));
    
    let spinner = create_spinner(&format!("Looking for VCF file for chr{}", chr));
    
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
                spinner.finish_with_message(format!("Found VCF file: {}", exact_file.display()));
                log(LogLevel::Info, &format!("Found exact VCF file match: {}", exact_file.display()));
                Ok(exact_file.clone())
            } else {
                log(LogLevel::Warning, "Multiple VCF files found, requesting user selection");
                spinner.finish_with_message("Multiple VCF files found, please select one");
                
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

                let chosen_file = chr_specific_files
                    .get(choice - 1)
                    .cloned()
                    .ok_or_else(|| VcfError::Parse("Invalid file number".to_string()))?;
                
                log(LogLevel::Info, &format!("User selected VCF file: {}", chosen_file.display()));
                Ok(chosen_file)
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
    set_stage(ProcessingStage::Global);
    log(LogLevel::Info, &format!(
        "Reading reference sequence for chromosome {} from {}:{}-{}", 
        chr, fasta_path.display(), region.start, region.end
    ));
    
    let spinner = create_spinner(&format!(
        "Reading reference sequence for chr{}:{}-{}", 
        chr, region.start, region.end
    ));
    
    // Create reader for the FASTA file and its index
    let mut reader = bio::io::fasta::IndexedReader::from_file(&fasta_path).map_err(|e| {
        let error_msg = format!("Failed to open FASTA file: {}", e);
        log(LogLevel::Error, &error_msg);
        spinner.finish_with_message("Failed to open reference FASTA file");
        VcfError::Io(io::Error::new(
            io::ErrorKind::Other,
            error_msg,
        ))
    })?;

    // Try both with and without "chr" prefix
    let chr_with_prefix = if !chr.starts_with("chr") {
        format!("chr{}", chr)
    } else {
        chr.to_string()
    };

    // Find chromosome
    spinner.set_message("Finding chromosome in reference index");
    let sequences = reader.index.sequences();
    let seq_info = sequences
        .iter()
        .find(|seq| seq.name == chr_with_prefix || seq.name == chr)
        .ok_or_else(|| {
            let error_msg = format!(
                "Chromosome {} (or {}) not found in reference",
                chr, chr_with_prefix
            );
            log(LogLevel::Error, &error_msg);
            spinner.finish_with_message("Chromosome not found in reference");
            VcfError::Parse(error_msg)
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
        log(LogLevel::Warning, "Found invalid characters in reference sequence:");
        for (pos, ch) in invalid_chars {
            log(LogLevel::Warning, &format!(
                "Position {}: '{}' (ASCII: {})",
                pos,
                String::from_utf8_lossy(&[ch]),
                ch
            ));
        }
        spinner.finish_with_message("Found invalid nucleotides in reference sequence");
        return Err(VcfError::Parse(format!(
            "Invalid nucleotides found in sequence for region {}:{}-{}",
            actual_chr_name, clamped_start, clamped_end
        )));
    }

    spinner.finish_with_message(format!(
        "Successfully read {}bp of reference sequence for chr{}",
        sequence.len(), chr
    ));
    log(LogLevel::Info, &format!(
        "Completed reading reference sequence: {}bp for chr{}:{}-{}", 
        sequence.len(), chr, clamped_start, clamped_end
    ));

    Ok(sequence)
}

// Helper function to parse GTF file and extract best CDS regions for each gene
// GTF and GFF use 1-based coordinate system
// Returns one TranscriptAnnotationCDS per gene (the best transcript according to priority rules)
pub fn parse_gtf_file(gtf_path: &Path, chr: &str) -> Result<Vec<TranscriptAnnotationCDS>, VcfError> {
   set_stage(ProcessingStage::Global);
   log(LogLevel::Info, &format!("Parsing GTF file for chromosome: {}", chr));
   
   init_step_progress(&format!("Parsing GTF file for chr{}", chr), 3);
   
   // Open the GTF file.
   let file = File::open(gtf_path).map_err(|e| {
       let error_msg = format!("GTF file not found: {:?}", e);
       log(LogLevel::Error, &error_msg);
       VcfError::Io(io::Error::new(
           io::ErrorKind::NotFound,
           error_msg,
       ))
   })?;
   let reader = BufReader::new(file);

   // Define priority order for transcript tags
   // Lower index = higher priority
   const PRIORITY_TAGS: [&str; 7] = [
       "MANE_Select",
       "MANE_Plus_Clinical",
       "CCDS",
       "appris_principal_1",
       "GENCODE_Primary",
       "Ensembl_canonical",
       "basic",
   ];

   // Structure to hold transcript information for selection
   #[derive(Default)]
   struct TranscriptInfo {
       segments: Vec<(i64, i64, char, i64)>, // start, end, strand, frame
       priority_level: usize,                // Lower is higher priority (usize::MAX = no priority tag)
       cds_length: i64,                      // Total length of all CDS segments
       gene_id: String,                      // Gene this transcript belongs to
       gene_name: Option<String>,            // Optional gene name
   }

   // Map of transcript_id -> transcript info
   let mut transcript_info_map: HashMap<String, TranscriptInfo> = HashMap::new();
   
   // Track statistics for logging
   let mut skipped_lines = 0;
   let mut processed_lines = 0;
   let mut transcripts_found = HashSet::new();
   let mut genes_found = HashSet::new();
   let mut malformed_attributes = 0;

   update_step_progress(0, &format!("Reading GTF entries for chr{}", chr));
   log(LogLevel::Info, "Starting to read GTF entries");

   // Read each line, parse if CDS, and store in transcript_info_map
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
           update_step_progress(0, &format!("Processed {} CDS entries for chr{}", processed_lines, chr));
           log(LogLevel::Info, &format!("Processed {} CDS entries", processed_lines));
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

       // Parse attributes to find transcript_id, gene_id, gene_name, and priority tags
       let attributes = fields[8];
       let mut transcript_id = None;
       let mut gene_id = None;
       let mut gene_name = None;
       let mut found_tags = Vec::new();
       let mut gene_type = None;
       let mut transcript_type = None;

       for attr in attributes.split(';') {
           let attr = attr.trim();
           if attr.is_empty() {
               continue;
           }

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
               "gene_id" => gene_id = Some(value.to_string()),
               "gene_name" => gene_name = Some(value.to_string()),
               "gene_type" => gene_type = Some(value.to_string()),
               "transcript_type" => transcript_type = Some(value.to_string()),
               "tag" => found_tags.push(value.to_string()),
               _ => {}
           }
       }

       // Skip non-protein-coding features if we can determine type
       if let Some(ref gt) = gene_type {
           if gt != "protein_coding" {
               continue;
           }
       }
       if let Some(ref tt) = transcript_type {
           if tt != "protein_coding" {
               continue;
           }
       }

       // Get transcript ID or skip if missing
       let transcript_id = match transcript_id {
           Some(id) => id,
           None => {
               malformed_attributes += 1;
               if malformed_attributes <= 5 {
                   log(LogLevel::Warning, &format!(
                       "Could not find transcript_id in attributes at line {}: {}",
                       line_num + 1,
                       attributes
                   ));
               }
               continue;
           }
       };

       // Get gene ID or skip if missing
       let gene_id = match gene_id {
           Some(id) => id,
           None => {
               malformed_attributes += 1;
               if malformed_attributes <= 5 {
                   log(LogLevel::Warning, &format!(
                       "Could not find gene_id in attributes at line {}: {}",
                       line_num + 1,
                       attributes
                   ));
               }
               continue;
           }
       };

       // Track stats
       genes_found.insert(gene_id.clone());
       if let Some(ref gene) = gene_name {
           transcripts_found.insert(format!("{}:{}", gene, transcript_id));
       } else {
           transcripts_found.insert(transcript_id.clone());
       }

       // Calculate CDS segment length
       let segment_length = end - start + 1;
       
       // Determine priority level based on tags
       let priority_level = found_tags.iter()
           .filter_map(|tag| PRIORITY_TAGS.iter().position(|&p| p == tag))
           .min()
           .unwrap_or(usize::MAX); // Use MAX value if no priority tag found

       // Get existing transcript info or create a new one
       let transcript_info = transcript_info_map.entry(transcript_id.clone()).or_insert_with(|| {
           TranscriptInfo {
               segments: Vec::new(),
               priority_level,
               cds_length: 0,
               gene_id: gene_id.clone(),
               gene_name: gene_name.clone(),
           }
       });

       // Update the transcript info
       transcript_info.segments.push((start, end, strand_char, frame));
       transcript_info.cds_length += segment_length;
       
       // Update priority level if we found a better one
       if priority_level < transcript_info.priority_level {
           transcript_info.priority_level = priority_level;
       }

       // Make sure gene_id is consistent (should be the same for all segments of a transcript)
       if transcript_info.gene_id != gene_id {
           log(LogLevel::Warning, &format!(
               "Inconsistent gene_id for transcript {} at line {}: {} vs {}",
               transcript_id, line_num + 1, transcript_info.gene_id, gene_id
           ));
           // Keep the first gene_id we encountered
       }

       // Set gene_name if we didn't have it before
       if transcript_info.gene_name.is_none() && gene_name.is_some() {
           transcript_info.gene_name = gene_name;
       }
   }

   // Print summary of how many lines, transcripts, genes, etc.
   println!("\nFinished reading GTF.");
   println!("Total CDS entries processed: {}", processed_lines);
   println!("Skipped lines: {}", skipped_lines);
   println!("Unique transcripts found: {}", transcripts_found.len());
   println!("Unique genes found: {}", genes_found.len());
   if malformed_attributes > 0 {
       println!(
           "Entries with missing required attributes: {}",
           malformed_attributes
       );
   }

   // Group transcripts by gene_id
   let mut gene_to_transcripts: HashMap<String, Vec<String>> = HashMap::new();
   for (transcript_id, info) in &transcript_info_map {
       gene_to_transcripts
           .entry(info.gene_id.clone())
           .or_default()
           .push(transcript_id.clone());
   }

   println!("Selecting best transcript for each gene...");
   
    // For each gene, select the best transcript based on priority rules
    let mut best_transcripts = HashSet::new();
    for (gene_id, transcript_ids) in gene_to_transcripts {
        // println!("Selecting best transcript for gene: {}", gene_id);
        if transcript_ids.is_empty() {
            continue;
        }

       // Find transcript with highest priority (lowest priority_level)
       let min_priority = transcript_ids.iter()
           .filter_map(|tid| transcript_info_map.get(tid))
           .map(|info| info.priority_level)
           .min()
           .unwrap_or(usize::MAX);

       // Get all transcripts with this priority level
       let candidates: Vec<&String> = transcript_ids.iter()
           .filter(|tid| {
               transcript_info_map.get(*tid)
                   .map(|info| info.priority_level == min_priority)
                   .unwrap_or(false)
           })
           .collect();

        // When multiple transcripts have the same priority,
        // choose the one with the longest coding sequence length.
        let best_transcript = if candidates.len() == 1 {
            // If there's only one transcript in the candidates list,
            // no need to compare anything—just copy and return it.
            candidates[0].clone()
        } else {
            // For multiple candidates, find the longest CDS length among them.
            let max_length = candidates.iter()
                // Loop through each transcript ID in the candidates list.
                .filter_map(|tid| transcript_info_map.get(*tid))
                // For each ID, look it up in the transcript_info_map to get its details.
                // filter_map skips any missing entries, but we expect all IDs to exist here.
                .map(|info| info.cds_length)
                // From each transcript's info, grab the CDS length.
                .max()
                // Find the biggest CDS length in the list.
                .unwrap_or(0);
                // If something goes wrong and we get no lengths (shouldn’t happen), use 0 as a fallback.
        
            // Now, collect all transcript IDs that have this maximum CDS length.
            let longest_candidates: Vec<&&String> = candidates.iter()
                // Iterate over references to transcript IDs in candidates.
                .filter(|tid| {
                    // For each transcript ID reference
                    transcript_info_map.get(&tid[..])
                    // Look up its details in the transcript_info_map using a string slice.
                    .map(|info| info.cds_length == max_length)
                    // Check if its CDS length matches the maximum found earlier.
                    .unwrap_or(false)
                    // Return false if the transcript isn’t found in the map.
                })
                .collect();
                // Gather all matching transcript IDs into a vector.
        
            // Pick the best one from the transcripts with the longest CDS.
            // If there’s more than one with the same max length, take the first.
            longest_candidates.first()
                // Retrieve the first reference to a transcript ID from longest_candidates.
                .map(|s| (**s).clone())
                // Dereference twice to get the String and clone it for the result.
                .unwrap_or_else(|| candidates[0].clone())
                // If no longest candidate exists, use the first candidate as a fallback.
        };

       best_transcripts.insert(best_transcript);
   }

   println!("Selected {} best transcripts out of {} total transcripts",
            best_transcripts.len(), transcript_info_map.len());

   // Now build a vector of TranscriptAnnotationCDS objects, but only for the best transcripts
   let mut transcripts_vec = Vec::new();

   for (tid, info) in transcript_info_map {
       // Skip if this transcript is not the best for its gene
       if !best_transcripts.contains(&tid) {
           continue;
       }

       // Process segments
       let mut segments = info.segments;
       segments.sort_by_key(|&(s, _, _, _)| s);

       if segments.is_empty() {
           continue;
       }

       let strand = segments[0].2;
       if strand == '-' {
           segments.reverse();
       }

       let strand_char = segments[0].2;
       let frames_vec: Vec<i64> = segments.iter().map(|&(_s, _e, _str, f)| f).collect();
       let seg_intervals: Vec<ZeroBasedHalfOpen> = segments
           .iter()
           .map(|&(s, e, _, _)| ZeroBasedHalfOpen::from_1based_inclusive(s, e))
           .collect();

        transcripts_vec.push(TranscriptAnnotationCDS {
            transcript_id: tid,
            gene_id: info.gene_id,
            gene_name: info.gene_name.unwrap_or_default(),
            strand: strand_char,
            frames: frames_vec,
            segments: seg_intervals,
        });
   }

   println!(
       "\nNumber of best transcripts returned: {}",
       transcripts_vec.len()
   );
   if transcripts_vec.is_empty() {
       println!("No CDS transcripts parsed for chromosome {}", chr);
   }

   // Return only the best transcript for each gene
   Ok(transcripts_vec)
}
