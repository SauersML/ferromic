use crate::process::{VcfError, ConfigEntry, CdsRegion};

use std::fs::{self, File};
use std::io::{self, BufReader, BufRead};
use std::path::{Path, PathBuf};
use std::collections::{HashMap, HashSet};
use flate2::read::MultiGzDecoder;
use colored::Colorize;

// Function to parse regions file (mask or allow)
pub fn parse_regions_file(
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


pub fn parse_config_file(path: &Path) -> Result<Vec<ConfigEntry>, VcfError> {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(path)
        .map_err(|e| VcfError::Io(e.into()))?;

    let headers = reader.headers().map_err(|e| VcfError::Io(e.into()))?.clone();
    let sample_names: Vec<String> = headers.iter().skip(7).map(String::from).collect();

    // Check if the number of sample names is consistent
    if sample_names.is_empty() {
        eprintln!("{}", "Error: No sample names found in the configuration file header after skipping the first 7 columns. Tabs must separate all columns, including sample names.".red());
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


pub fn parse_region(region: &str) -> Result<(i64, i64), VcfError> {
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



pub fn find_vcf_file(folder: &str, chr: &str) -> Result<PathBuf, VcfError> {
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

pub fn open_vcf_reader(path: &Path) -> Result<Box<dyn BufRead + Send>, VcfError> {
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


// Function to validate VCF header
pub fn validate_vcf_header(header: &str) -> Result<(), VcfError> {
    let fields: Vec<&str> = header.split('\t').collect();
    let required_fields = vec!["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"];

    if fields.len() < required_fields.len() || fields[..required_fields.len()] != required_fields[..] {
        return Err(VcfError::InvalidVcfFormat("Invalid VCF header format".to_string()));
    }
    Ok(())
}


pub fn read_reference_sequence(
    fasta_path: &Path,
    chr: &str, 
    start: i64,
    end: i64
) -> Result<Vec<u8>, VcfError> {
    // Input validation
    if start < 0 || end < 0 {
        return Err(VcfError::Parse(format!(
            "Invalid coordinates: start ({}) and end ({}) must be non-negative",
            start, end
        )));
    }
    
    if start > end {
        return Err(VcfError::Parse(format!(
            "Invalid coordinates: start ({}) must be less than or equal to end ({})",
            start, end
        )));
    }

    // Create reader for the FASTA file and its index
    let mut reader = bio::io::fasta::IndexedReader::from_file(&fasta_path)
        .map_err(|e| VcfError::Io(io::Error::new(
            io::ErrorKind::Other, 
            format!("Failed to open FASTA file: {}", e)
        )))?;

    // Try both with and without "chr" prefix
    let chr_with_prefix = if !chr.starts_with("chr") {
        format!("chr{}", chr)
    } else {
        chr.to_string()
    };

    // Get sequences and find our chromosome
    let sequences = reader.index.sequences();
    let seq_info = sequences
        .iter()
        .find(|seq| seq.name == chr_with_prefix || seq.name == chr)
        .ok_or_else(|| VcfError::Parse(format!(
            "Chromosome {} (or {}) not found in reference", chr, chr_with_prefix
        )))?;

    let seq_length = seq_info.len;
    let actual_chr_name = seq_info.name.as_str(); // Get a reference to the name

    // Validate start position
    if start as u64 >= seq_length {
        return Err(VcfError::Parse(format!(
            "Start position {} exceeds sequence length {} for chromosome {}",
            start, seq_length, actual_chr_name
        )));
    }

    // Clamp end position to sequence length
    let adjusted_end = std::cmp::min(end as u64, seq_length - 1);
    if adjusted_end as i64 != end {
        println!("Warning: End position {} exceeds sequence length {}. Clamping to {}", 
                 end, seq_length, adjusted_end);
    }

    // Calculate region length and allocate buffer
    let region_length = (adjusted_end - (start as u64) + 1) as usize;
    let mut sequence = Vec::with_capacity(region_length);

    // Fetch and read the sequence with proper error handling
    reader.fetch(actual_chr_name, (start - 1) as u64, end as u64)
        .map_err(|e| VcfError::Io(io::Error::new(
            io::ErrorKind::Other,
            format!("Failed to fetch region {}:{}-{}: {}", actual_chr_name, start, adjusted_end, e)
        )))?;

    reader.read(&mut sequence)
        .map_err(|e| VcfError::Io(io::Error::new(
            io::ErrorKind::Other,
            format!("Failed to read sequence for region {}:{}-{}: {}", 
                    actual_chr_name, start, adjusted_end, e)
        )))?;

    // Verify sequence length
    if sequence.len() != region_length {
        return Err(VcfError::Parse(format!(
            "Expected sequence length {} but got {} for region {}:{}-{}",
            region_length, sequence.len(), actual_chr_name, start, adjusted_end
        )));
    }

    // Verify sequence content
    let invalid_chars: Vec<(usize, u8)> = sequence.iter()
        .enumerate()
        .filter(|(_, &b)| !matches!(b.to_ascii_uppercase(), b'A' | b'C' | b'G' | b'T' | b'N'))
        .take(10)  // Show first 10 invalid chars
        .map(|(i, &b)| (i, b))  // Dereference here
        .collect();
    
    if !invalid_chars.is_empty() {
        println!("Found invalid characters:");
        for (pos, ch) in invalid_chars {
            println!("Position {}: '{}' (ASCII: {})", 
                    pos, 
                    String::from_utf8_lossy(&[ch]), 
                    ch);
        }
        return Err(VcfError::Parse(format!(
            "Invalid nucleotides found in sequence for region {}:{}-{}",
            actual_chr_name, start, adjusted_end
        )));
    }

    Ok(sequence)
}



// IN PROGRESS
// Helper function to parse GTF file and extract CDS regions
// GTF and GFF use 1-based coordinate system
pub fn parse_gtf_file(
    gtf_path: &Path, 
    chr: &str,
    region_start: i64,
    region_end: i64,
) -> Result<Vec<CdsRegion>, VcfError> {
    println!("\n{}", "Parsing GTF file...".green().bold());
    println!("Chromosome: {}", chr);
    println!("Region: {}-{}", region_start, region_end);

    let file = File::open(gtf_path).map_err(|e| {
        VcfError::Io(io::Error::new(
            io::ErrorKind::NotFound,
            format!("GTF file not found: {:?}", e),
        ))
    })?;
    let reader = BufReader::new(file);
    
    // Change to use transcript ID as key since CDS belongs to transcripts
    let mut transcript_cdss: HashMap<String, Vec<(i64, i64, char, i64)>> = HashMap::new();
    let mut skipped_lines = 0;
    let mut processed_lines = 0;
    let mut transcripts_found = HashSet::new();
    let mut malformed_attributes = 0;

    println!("Reading GTF entries...");

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
                eprintln!("Warning: Invalid start position at line {}, skipping", line_num + 1);
                skipped_lines += 1;
                continue;
            }
        };

        let end: i64 = match fields[4].parse() {
            Ok(e) => e,
            Err(_) => {
                eprintln!("Warning: Invalid end position at line {}, skipping", line_num + 1);
                skipped_lines += 1;
                continue;
            }
        };

        let strand_char = fields[6].chars().next().unwrap_or('.');
        let frame: i64 = fields[7].parse().unwrap_or_else(|_| {
            eprintln!("Warning: Invalid frame at line {}, using 0", line_num + 1);
            0
        });

        // Parse attributes to get transcript_id and gene_name
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
                _ => continue,
            }
        }

        let transcript_id = match transcript_id {
            Some(id) => id,
            None => {
                malformed_attributes += 1;
                if malformed_attributes <= 5 {
                    eprintln!("Warning: Could not find transcript_id in attributes at line {}: {}", 
                             line_num + 1, attributes);
                }
                continue;
            }
        };

        // Store for later reporting
        if let Some(gene) = gene_name {
            transcripts_found.insert(format!("{}:{}", gene, transcript_id));
        } else {
            transcripts_found.insert(transcript_id.clone());
        }

        // Store CDS segment with strand + frame
        transcript_cdss.entry(transcript_id)
            .or_default()
            .push((start, end, strand_char, frame));
    }

    println!("\n{}", "GTF Parsing Statistics:".blue().bold());
    println!("Total CDS entries processed: {}", processed_lines);
    println!("Skipped lines: {}", skipped_lines);
    println!("Unique transcripts found: {}", transcripts_found.len());
    if malformed_attributes > 0 {
        println!("{}", format!("Entries with missing transcript IDs: {}", malformed_attributes).yellow());
    }

    println!("\n{}", "Processing CDS regions by transcript...".green().bold());
    let mut cds_regions = Vec::new();
    let transcripts_processed = 0;

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

    for (transcript_id, mut segments) in transcript_cdss {
        // Skip entire transcript if it has no exons overlapping the query:
        let transcript_overlaps_region = segments
            .iter()
            .any(|&(s, e, _, _)| e >= region_start && s <= region_end);
    
        if !transcript_overlaps_region {
            continue;
        }
    
        segments.sort_by_key(|&(start, _, _, _)| start);
        
        println!("\nProcessing transcript: {}", transcript_id);
        println!("Found {} CDS segments", segments.len());

        stats.total_transcripts += 1;
        stats.total_cds_segments += segments.len();

        if segments.len() == 1 {
            stats.single_cds_transcripts += 1;
        } else {
            stats.multi_cds_transcripts += 1;
        }

        // Check for gaps between segments
        let has_gaps = segments.windows(2)
            .any(|w| w[1].0 - w[0].1 > 1);
        if has_gaps {
            stats.transcripts_with_gaps += 1;
        }

        // Calculate total coding length and check individual segments
        let mut coding_segments = Vec::new();
        for (i, &(start, end, _, frame)) in segments.iter().enumerate() {
            let segment_length = end - start + 1;
            println!("  Segment {}: {}-{} (length: {}, frame: {})", 
                    i + 1, start, end, segment_length, frame);
            coding_segments.push((start, end));
        }

        if segments.is_empty() {
            println!("  {} No valid segments for transcript {}", "!".red(), transcript_id);
            continue;
        }

        let min_start = segments.iter().map(|&(s, _, _, _)| s).min().unwrap();
        let max_end = segments.iter().map(|&(_, e, _, _)| e).max().unwrap();
        let transcript_span = max_end - min_start + 1;
        
        // Calculate actual coding length (sum of CDS lengths)
        let total_coding_length: i64 = segments.iter()
            .map(|&(s, e, _, _)| e - s + 1)
            .sum();

        stats.total_coding_length += total_coding_length;

        // Update length stats
        match stats.shortest_transcript_length {
            None => stats.shortest_transcript_length = Some(total_coding_length),
            Some(current) => if total_coding_length < current {
                stats.shortest_transcript_length = Some(total_coding_length)
            }
        }
        
        match stats.longest_transcript_length {
            None => stats.longest_transcript_length = Some(total_coding_length),
            Some(current) => if total_coding_length > current {
                stats.longest_transcript_length = Some(total_coding_length)
            }
        }

        if total_coding_length % 3 != 0 {
            stats.non_divisible_by_three += 1;
            println!("  {} Warning: Total CDS length {} not divisible by 3", 
                    "!".yellow(), total_coding_length);
            println!("    Remainder when divided by 3: {}", total_coding_length % 3);
            println!("    Individual segment lengths: {:?}", 
                    segments.iter().map(|&(s, e, _, _)| e - s + 1).collect::<Vec<_>>());
        }

        let cds_region = CdsRegion {
            transcript_id: transcript_id.clone(),
            // Use the entire triple (start,end,frame) directly
            segments,
        };

        let cds_start = cds_region.segments.iter().map(|(s, _, _, _)| *s).min().unwrap();
        let cds_end = cds_region.segments.iter().map(|(_, e, _, _)| *e).max().unwrap();

        // Create a local variable holding the cloned segments
        let cloned_segments = cds_region.segments.clone();
        
        let min_start_for_print = cloned_segments.iter().map(|(s, _, _, _)| s).min().unwrap();
        let max_end_for_print = cloned_segments.iter().map(|(_, e, _, _)| e).max().unwrap();
        
        println!("  CDS region: {}-{}", min_start_for_print, max_end_for_print);

        // Print before pushing:
        println!("  CDS region: {}-{}", min_start_for_print, max_end_for_print);

        // Now push after printing, so no borrow occurs after move:
        cds_regions.push(cds_region);

        println!("  CDS region: {}-{}", min_start_for_print, max_end_for_print);
        println!("    Genomic span: {}", transcript_span);
        println!("    Total coding length: {}", total_coding_length); 
    }

    if stats.total_transcripts > 0 {
        println!("\n{}", "CDS Processing Summary:".blue().bold());
        println!("Total transcripts processed: {}", stats.total_transcripts);
        println!("Total CDS segments: {}", stats.total_cds_segments);
        println!("Average segments per transcript: {:.2}", 
                 stats.total_cds_segments as f64 / stats.total_transcripts as f64);
        println!("Single-cds transcripts: {} ({:.1}%)", 
                 stats.single_cds_transcripts,
                 100.0 * stats.single_cds_transcripts as f64 / stats.total_transcripts as f64);
        println!("Multi-cds transcripts: {} ({:.1}%)", 
                 stats.multi_cds_transcripts,
                 100.0 * stats.multi_cds_transcripts as f64 / stats.total_transcripts as f64);
        println!("Transcripts with gaps: {} ({:.1}%)",
                 stats.transcripts_with_gaps,
                 100.0 * stats.transcripts_with_gaps as f64 / stats.total_transcripts as f64);
        println!("Non-divisible by three: {} ({:.1}%)", 
                 stats.non_divisible_by_three,
                 100.0 * stats.non_divisible_by_three as f64 / stats.total_transcripts as f64);
        println!("Total coding bases: {}", stats.total_coding_length);
        
        if let Some(shortest) = stats.shortest_transcript_length {
            println!("Shortest transcript: {} bp", shortest);
        }
        if let Some(longest) = stats.longest_transcript_length {
            println!("Longest transcript: {} bp", longest);
        }
        println!("Average transcript length: {:.1} bp",
                 stats.total_coding_length as f64 / stats.total_transcripts as f64);
    }

    if cds_regions.is_empty() {
        println!("{}", "No valid CDS regions found!".red());
    }

    Ok(cds_regions)
}
