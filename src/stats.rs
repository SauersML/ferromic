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

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    vcf_folder: String,

    #[arg(short, long)]
    chr: String,

    #[arg(short, long)]
    region: String,
}

#[derive(Debug, Clone)]
struct Variant {
    position: i64,
    genotypes: Vec<Option<u8>>,
}

#[derive(Debug)]
enum VcfError {
    Io(std::io::Error),
    Parse(String),
    InvalidRegion(String),
    NoVcfFiles,
    InconsistentSampleCount,
}

impl std::fmt::Display for VcfError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            VcfError::Io(err) => write!(f, "IO error: {}", err),
            VcfError::Parse(msg) => write!(f, "Parse error: {}", msg),
            VcfError::InvalidRegion(msg) => write!(f, "Invalid region: {}", msg),
            VcfError::NoVcfFiles => write!(f, "No VCF files found"),
            VcfError::InconsistentSampleCount => write!(f, "Inconsistent sample count"),
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

    let (start, end) = parse_region(&args.region)?;
    let vcf_files = find_vcf_files(&args.vcf_folder, &args.chr)?;

    println!(
        "{}",
        format!("Processing {} VCF file(s)", vcf_files.len()).cyan()
    );

    let (variants, sample_names) = process_vcf_files(&vcf_files, &args.chr, start, end)?;

    println!("{}", "Calculating diversity statistics...".blue());

    let n = sample_names.len();
    let seq_length = end - start + 1;
    let num_segsites = count_segregating_sites(&variants);

    let pairwise_diffs = calculate_pairwise_differences(&variants, &sample_names);
    let tot_pair_diff: usize = pairwise_diffs.iter().map(|&(_, count, _)| count).sum();

    let w_theta = calculate_watterson_theta(num_segsites, n, seq_length);
    let pi = calculate_pi(tot_pair_diff, n, seq_length);

    println!("\n{}", "Results:".green().bold());
    println!("#Pairwise nucleotide substitutions");
    for (sample_pair, count, positions) in pairwise_diffs {
        println!(
            "{}\t{}\t{}\t{:?}",
            sample_pair.0, sample_pair.1, count, positions
        );
    }

    println!("\nSeqLength:{}", seq_length);
    println!("NumSegsites:{}", num_segsites);
    println!("WattersonTheta:{:.6}", w_theta);
    println!("pi:{:.6}", pi);

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

fn find_vcf_files(folder: &str, chr: &str) -> Result<Vec<PathBuf>, VcfError> {
    let path = Path::new(folder);

    let chr_specific_files: Vec<_> = fs::read_dir(path)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            let path = entry.path();
            let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            file_name.starts_with(&format!("chr{}", chr)) &&
                (file_name.ends_with(".vcf") || file_name.ends_with(".vcf.gz"))
        })
        .map(|entry| entry.path())
        .collect();

    if !chr_specific_files.is_empty() {
        Ok(chr_specific_files)
    } else {
        let entries: Vec<_> = fs::read_dir(path)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                let path = entry.path();
                let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("");
                extension == "vcf" || extension == "gz"
            })
            .map(|entry| entry.path())
            .collect();
        if entries.is_empty() {
            Err(VcfError::NoVcfFiles)
        } else {
            Ok(entries)
        }
    }
}

fn process_vcf_files(
    files: &[PathBuf],
    chr: &str,
    start: i64,
    end: i64,
) -> Result<(Vec<Variant>, Vec<String>), VcfError> {
    let mut all_variants = Vec::new();
    let mut sample_names = Vec::new();

    let progress_bar = ProgressBar::new(files.len() as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("##-"),
    );

    for (i, file) in files.iter().enumerate() {
        progress_bar.set_message(format!("Processing file: {}", file.display()));
        let (variants, file_sample_names) = process_vcf(file, chr, start, end)?;

        if i == 0 {
            sample_names = file_sample_names;
        } else if sample_names != file_sample_names {
            return Err(VcfError::InconsistentSampleCount);
        }

        all_variants.extend(variants);
        progress_bar.inc(1);
    }

    progress_bar.finish_with_message("VCF processing complete");
    Ok((all_variants, sample_names))
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

    for line in reader.lines() {
        let line = line?;
        if line.starts_with("##") {
            continue;
        }
        if line.starts_with("#CHROM") {
            sample_names = line.split_whitespace().skip(9).map(String::from).collect();
            continue;
        }

        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields[0] != chr {
            continue;
        }

        let pos: i64 = fields[1]
            .parse()
            .map_err(|_| VcfError::Parse(format!("Invalid position at line: {}", line)))?;
        if pos < start || pos > end {
            continue;
        }

        let genotypes: Vec<Option<u8>> = fields[9..]
            .iter()
            .map(|gt| gt.split('|').next().and_then(|allele| allele.parse().ok()))
            .collect();

        if genotypes.len() != sample_names.len() {
            eprintln!(
                "{}",
                format!("Warning: Inconsistent sample count at position {}", pos).yellow()
            );
        }

        if genotypes.iter().any(|g| g.is_none()) {
            eprintln!(
                "{}",
                format!("Warning: Missing data at position {}", pos).yellow()
            );
        }

        variants.push(Variant {
            position: pos,
            genotypes,
        });
    }

    Ok((variants, sample_names))
}

fn count_segregating_sites(variants: &[Variant]) -> usize {
    variants
        .iter()
        .filter(|v| {
            v.genotypes
                .iter()
                .filter_map(|&g| g)
                .collect::<HashSet<_>>()
                .len()
                > 1
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
                if let (Some(gi), Some(gj)) = (v.genotypes[i], v.genotypes[j]) {
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
