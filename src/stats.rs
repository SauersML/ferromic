use clap::Parser;
use colored::*;
use flate2::read::MultiGzDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use rand::seq::SliceRandom;
use rand::thread_rng;

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
    ref_allele: String,
    alt_alleles: Vec<String>,
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

    let (variants, sample_names) = process_vcf(&vcf_file, &args.chr, start, end)?;

    println!("{}", "Calculating diversity statistics...".blue());

    let n = sample_names.len();
    let seq_length = if end == i64::MAX {
        variants.last().map(|v| v.position).unwrap_or(0) - start + 1
    } else {
        end - start + 1
    };

    let num_segsites = count_segregating_sites(&variants);
    let raw_variant_count = variants.len();

    let pairwise_diffs = calculate_pairwise_differences(&variants, &sample_names);
    let tot_pair_diff: usize = pairwise_diffs.iter().map(|&(_, count, _)| count).sum();

    let w_theta = calculate_watterson_theta(num_segsites, n, seq_length);
    let pi = calculate_pi(tot_pair_diff, n, seq_length);

    println!("\n{}", "Results:".green().bold());
    println!("#Example pairwise nucleotide substitutions from this run");
    let mut rng = thread_rng();
    for (sample_pair, count, positions) in pairwise_diffs.choose_multiple(&mut rng, 5) {
        let sample_positions: Vec<_> = positions.choose_multiple(&mut rng, 5.min(positions.len())).cloned().collect();
        println!(
            "{}\t{}\t{}\t{:?}",
            sample_pair.0, sample_pair.1, count, sample_positions
        );
    }

    println!("\nSeqLength:{}", seq_length);
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

    if chr_specific_files.len() == 1 {
        Ok(chr_specific_files[0].clone())
    } else if chr_specific_files.is_empty() {
        Err(VcfError::NoVcfFiles)
    } else {
        Err(VcfError::Parse(format!(
            "Multiple VCF files found for chromosome {}",
            chr
        )))
    }
}

fn open_vcf_reader(path: &Path) -> Result<Box<dyn BufRead>, VcfError> {
    let file = File::open(path)?;
    
    if path.extension().and_then(|s| s.to_str()) == Some("gz") {
        let decoder = MultiGzDecoder::new(file);
        Ok(Box::new(BufReader::new(decoder)))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

fn process_vcf(
    file: &Path,
    chr: &str,
    start: i64,
    end: i64,
) -> Result<(Vec<Variant>, Vec<String>), VcfError> {
    let reader = open_vcf_reader(file)?;
    let mut variants = Vec::new();
    let mut sample_names = Vec::new();

    let progress_bar = ProgressBar::new_spinner();
    let style = ProgressStyle::default_spinner()
        .template("{spinner:.green} {msg}")
        .expect("Failed to create progress style");
    progress_bar.set_style(style);
    progress_bar.set_message("Processing variants...");

    for line in reader.lines() {
        let line = line?;
        if line.starts_with("##") {
            continue;
        }
        if line.starts_with("#CHROM") {
            validate_vcf_header(&line)?;
            sample_names = line.split_whitespace().skip(9).map(String::from).collect();
            continue;
        }

        let variant = parse_variant(&line, chr, start, end)?;
        if let Some(v) = variant {
            variants.push(v);
        }

        progress_bar.inc(1);
    }

    progress_bar.finish_with_message("Variant processing complete");

    Ok((variants, sample_names))
}

fn validate_vcf_header(header: &str) -> Result<(), VcfError> {
    let fields: Vec<&str> = header.split_whitespace().collect();
    let required_fields = vec!["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"];
    
    if fields.len() < required_fields.len() || fields[..required_fields.len()] != required_fields {
        return Err(VcfError::InvalidVcfFormat("Invalid VCF header format".to_string()));
    }
    Ok(())
}

fn is_valid_allele(allele: &str) -> bool {
    allele.chars().all(|c| matches!(c, 'A' | 'C' | 'G' | 'T' | 'N'))
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

    let ref_allele = fields[3].to_string();
    let alt_alleles: Vec<String> = fields[4].split(',').map(String::from).collect();

    if !is_valid_allele(&ref_allele) || !alt_alleles.iter().all(|a| is_valid_allele(a)) {
        return Err(VcfError::Parse("Invalid REF or ALT allele".to_string()));
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
        ref_allele,
        alt_alleles,
        genotypes,
    }))
}

fn count_segregating_sites(variants: &[Variant]) -> usize {
    variants
        .iter()
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
    sample_names: &[String],
) -> Vec<((String, String), usize, Vec<i64>)> {
    let n = sample_names.len();
    let variants = Arc::new(variants.to_vec());

    (0..n)
        .combinations(2)
        .par_bridge()
        .map(|pair| {
            let variants = Arc::clone(&variants);
            let (i, j) = (pair[0], pair[1]);
            let mut diff_count = 0;
            let mut diff_positions = Vec::new();

            for v in variants.iter() {
                if let (Some(gi), Some(gj)) = (&v.genotypes[i], &v.genotypes[j]) {
                    if gi != gj {
                        diff_count += 1;
                        diff_positions.push(v.position);
                    }
                }
            }

            (
                (sample_names[i].clone(), sample_names[j].clone()),
                diff_count,
                diff_positions,
            )
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
