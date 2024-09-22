use clap::Parser;
use colored::*;
use flate2::read::MultiGzDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::Mmap;
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    vcf_folder: String,

    #[arg(short, long)]
    chr: String,

    #[arg(short, long)]
    region: Option<String>,
}

#[derive(Debug, Clone)]
struct Variant {
    position: i64,
    genotypes: Vec<Option<Vec<u8>>>,
}

#[derive(Debug)]
enum VcfError {
    Io(std::io::Error),
    Parse(String),
    InvalidRegion(String),
    NoVcfFiles,
    InvalidVcfFormat(String),
}

impl std::fmt::Display for VcfError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            VcfError::Io(err) => write!(f, "IO error: {}", err),
            VcfError::Parse(msg) => write!(f, "Parse error: {}", msg),
            VcfError::InvalidRegion(msg) => write!(f, "Invalid region: {}", msg),
            VcfError::NoVcfFiles => write!(f, "No VCF files found"),
            VcfError::InvalidVcfFormat(msg) => write!(f, "Invalid VCF format: {}", msg),
        }
    }
}

impl From<std::io::Error> for VcfError {
    fn from(err: std::io::Error) -> VcfError {
        VcfError::Io(err)
    }
}

fn main() -> Result<(), VcfError> {
    let args = Args::parse();

    println!("{}", "Starting VCF diversity analysis...".green());

    let (start, end) = match args.region {
        Some(ref region) => parse_region(region)?,
        None => (0, i64::MAX),
    };

    let vcf_file = find_vcf_file(&args.vcf_folder, &args.chr)?;

    println!("{}", format!("Processing VCF file: {}", vcf_file.display()).cyan());

    let (variants, sample_names, chr_length) = process_vcf(&vcf_file, &args.chr, start, end)?;

    println!("{}", "Calculating diversity statistics...".blue());

    let n = sample_names.len();
    let seq_length = if end == i64::MAX {
        variants.last().map(|v| v.position).unwrap_or(0).max(chr_length) - start + 1
    } else {
        end - start + 1
    };

    if end == i64::MAX && variants.last().map(|v| v.position).unwrap_or(0) < chr_length {
        println!("{}", "Warning: The sequence length may be underestimated. Consider using the --region parameter for more accurate results.".yellow());
    }

    let num_segsites = count_segregating_sites(&variants);
    let raw_variant_count = variants.len();

    let pairwise_diffs = calculate_pairwise_differences(&variants, n);
    let tot_pair_diff: usize = pairwise_diffs.iter().sum();

    let w_theta = calculate_watterson_theta(num_segsites, n, seq_length);
    let pi = calculate_pi(tot_pair_diff, n, seq_length);

    println!("\n{}", "Results:".green().bold());
    println!("SeqLength:{}", seq_length);
    println!("NumSegsites:{}", num_segsites);
    println!("RawVariantCount:{}", raw_variant_count);
    println!("WattersonTheta:{:.6}", w_theta);
    println!("pi:{:.6}", pi);

    // Sanity checks and warnings
    if variants.is_empty() {
        println!("{}", "Warning: No variants found in the specified region.".yellow());
    }

    if num_segsites == 0 {
        println!("{}", "Warning: All sites are monomorphic.".yellow());
    }

    if num_segsites != raw_variant_count {
        println!("{}", format!("Note: Number of segregating sites ({}) differs from raw variant count ({}).", num_segsites, raw_variant_count).yellow());
    }

    Ok(())
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
            (file_name.starts_with(&format!("chr{}", chr)) || file_name.starts_with(chr)) &&
                (file_name.ends_with(".vcf") || file_name.ends_with(".vcf.gz"))
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
                file_name.starts_with(&chr_pattern) && 
                    file_name.chars().nth(chr_pattern.len())
                        .map(|c| !c.is_ascii_digit())
                        .unwrap_or(true)
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
                std::io::stdin().read_line(&mut input)?;
                let choice: usize = input.trim().parse().map_err(|_| VcfError::Parse("Invalid input".to_string()))?;
                
                chr_specific_files.get(choice - 1)
                    .cloned()
                    .ok_or_else(|| VcfError::Parse("Invalid file number".to_string()))
            }
        }
    }
}

fn process_vcf(
    file: &Path,
    chr: &str,
    start: i64,
    end: i64,
) -> Result<(Vec<Variant>, Vec<String>, i64), VcfError> {
    let file = File::open(file)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let content = std::str::from_utf8(&mmap).map_err(|_| VcfError::Parse("Invalid UTF-8".to_string()))?;

    let mut variants = Vec::new();
    let mut sample_names = Vec::new();
    let mut chr_length = 0;

    let progress_bar = ProgressBar::new(content.lines().count() as u64);
    let style = ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
        .expect("Failed to create progress style");
    progress_bar.set_style(style);

    for line in content.lines() {
        if line.starts_with("##") {
            if line.starts_with("##contig=<ID=") && line.contains(&format!("ID={}", chr)) {
                if let Some(length_str) = line.split(',').find(|s| s.starts_with("length=")) {
                    chr_length = length_str.trim_start_matches("length=").trim_end_matches('>').parse().unwrap_or(0);
                }
            }
            continue;
        }
        if line.starts_with("#CHROM") {
            validate_vcf_header(line)?;
            sample_names = line.split_whitespace().skip(9).map(String::from).collect();
            continue;
        }

        if let Some(variant) = parse_variant(line, chr, start, end)? {
            variants.push(variant);
        }

        progress_bar.inc(1);
    }

    progress_bar.finish_with_message("Variant processing complete");

    Ok((variants, sample_names, chr_length))
}

fn validate_vcf_header(header: &str) -> Result<(), VcfError> {
    let fields: Vec<&str> = header.split_whitespace().collect();
    let required_fields = vec!["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"];
    
    if fields.len() < required_fields.len() || fields[..required_fields.len()] != required_fields {
        return Err(VcfError::InvalidVcfFormat("Invalid VCF header format".to_string()));
    }
    Ok(())
}

fn parse_variant(line: &str, chr: &str, start: i64, end: i64) -> Result<Option<Variant>, VcfError> {
    let fields: Vec<&str> = line.split_whitespace().collect();
    if fields.len() < 10 {
        return Err(VcfError::Parse("Invalid VCF line format".to_string()));
    }

    let vcf_chr = fields[0].trim_start_matches("chr");
    if vcf_chr != chr.trim_start_matches("chr") {
        return Ok(None);
    }

    let pos: i64 = fields[1].parse().map_err(|_| VcfError::Parse("Invalid position".to_string()))?;
    if pos < start || pos > end {
        return Ok(None);
    }

    let genotypes: Vec<Option<Vec<u8>>> = fields[9..].iter()
        .map(|gt| {
            gt.split(':').next().and_then(|alleles| {
                alleles.split(|c| c == '|' || c == '/')
                    .map(|allele| allele.parse().ok())
                    .collect::<Option<Vec<u8>>>()
            })
        })
        .collect();

    Ok(Some(Variant {
        position: pos,
        genotypes,
    }))
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
) -> Vec<usize> {
    let variants = Arc::new(variants);

    (0..n)
        .into_par_iter()
        .flat_map(|i| {
            let variants = Arc::clone(&variants);
            (i+1..n).into_par_iter().map(move |j| {
                variants.iter().filter(|v| {
                    if let (Some(gi), Some(gj)) = (&v.genotypes[i], &v.genotypes[j]) {
                        gi != gj
                    } else {
                        false
                    }
                }).count()
            }).collect::<Vec<_>>()
        })
        .collect()
}

fn harmonic(n: usize) -> f64 {
    (1..n).map(|i| 1.0 / i as f64).sum()
}

fn calculate_watterson_theta(seg_sites: usize, n: usize, seq_length: i64) -> f64 {
    seg_sites as f64 / harmonic(n - 1) / seq_length as f64
}

fn calculate_pi(tot_pair_diff: usize, n: usize, seq_length: i64) -> f64 {
    let num_comparisons = n * (n - 1) / 2;
    tot_pair_diff as f64 / num_comparisons as f64 / seq_length as f64
}
