use clap::Parser;
use colored::*;
use itertools::Itertools;
use rust_htslib::bcf::{Read, Reader};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

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

#[derive(Debug)]
struct Variant {
    position: u32,
    genotypes: Vec<u8>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    
    println!("{}", "Starting VCF diversity analysis...".green());

    let (start, end) = parse_region(&args.region)?;
    let vcf_files = find_vcf_files(&args.vcf_folder, &args.chr)?;

    println!("{}", format!("Processing {} VCF file(s)", vcf_files.len()).cyan());

    let mut all_variants = Vec::new();
    for file in &vcf_files {
        println!("{}", format!("Reading file: {}", file.display()).yellow());
        let variants = process_vcf(file, &args.chr, start, end)?;
        all_variants.extend(variants);
    }

    println!("{}", "Calculating diversity statistics...".blue());

    let n = count_samples(&vcf_files[0])?;
    let seq_length = end - start + 1;
    let num_segsites = count_segregating_sites(&all_variants);
    
    let pairwise_diffs = calculate_pairwise_differences(&all_variants);
    let tot_pair_diff: usize = pairwise_diffs.iter().map(|&(_, count)| count).sum();

    let w_theta = calculate_watterson_theta(num_segsites, n, seq_length);
    let pi = calculate_pi(tot_pair_diff, n, seq_length);

    println!("\n{}", "Results:".green().bold());
    println!("#Pairwise nucleotide substitutions");
    for ((sample1, sample2), count) in pairwise_diffs {
        println!("{}\t{}\t{}", sample1, sample2, count);
    }

    println!("\nSeqLength:{}", seq_length);
    println!("NumSegsites:{}", num_segsites);
    println!("WattersonTheta:{:.6}", w_theta);
    println!("pi:{:.6}", pi);

    Ok(())
}

fn parse_region(region: &str) -> Result<(u32, u32), Box<dyn std::error::Error>> {
    let parts: Vec<&str> = region.split('-').collect();
    if parts.len() != 2 {
        return Err("Invalid region format. Use start-end".into());
    }
    let start: u32 = parts[0].parse()?;
    let end: u32 = parts[1].parse()?;
    Ok((start, end))
}

fn find_vcf_files(folder: &str, chr: &str) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let path = Path::new(folder);
    let chr_specific_file = path.join(format!("chr{}*.vcf", chr));
    
    if let Ok(file) = chr_specific_file.canonicalize() {
        Ok(vec![file])
    } else {
        let entries: Vec<_> = fs::read_dir(path)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().extension().and_then(|s| s.to_str()) == Some("vcf"))
            .map(|entry| entry.path())
            .collect();
        Ok(entries)
    }
}

fn process_vcf(file: &Path, chr: &str, start: u32, end: u32) -> Result<Vec<Variant>, Box<dyn std::error::Error>> {
    let mut reader = Reader::from_path(file)?;
    reader.set_threads(4)?;
    
    let region = format!("{}:{}-{}", chr, start, end);
    reader.set_region(&region)?;

    let mut variants = Vec::new();
    let mut record = reader.empty_record();
    
    while reader.read(&mut record)? {
        let pos = record.pos() + 1;  // 0-based to 1-based
        if pos >= start && pos <= end {
            let genotypes = record.genotypes()?.genotypes().iter()
                .map(|gt| gt.allele().unwrap_or(0) as u8)
                .collect();
            variants.push(Variant { position: pos, genotypes });
        }
    }

    Ok(variants)
}

fn count_samples(file: &Path) -> Result<usize, Box<dyn std::error::Error>> {
    let reader = Reader::from_path(file)?;
    Ok(reader.header().sample_count() as usize)
}

fn count_segregating_sites(variants: &[Variant]) -> usize {
    variants.iter()
        .filter(|v| v.genotypes.iter().collect::<HashSet<_>>().len() > 1)
        .count()
}

fn calculate_pairwise_differences(variants: &[Variant]) -> Vec<((usize, usize), usize)> {
    let n = variants[0].genotypes.len();
    (0..n).combinations(2)
        .map(|pair| {
            let diff_count = variants.iter()
                .filter(|v| v.genotypes[pair[0]] != v.genotypes[pair[1]])
                .count();
            ((pair[0], pair[1]), diff_count)
        })
        .collect()
}

fn harmonic(n: usize) -> f64 {
    (1..n).map(|i| 1.0 / i as f64).sum()
}

fn calculate_watterson_theta(seg_sites: usize, n: usize, seq_length: u32) -> f64 {
    seg_sites as f64 / harmonic(n - 1) / seq_length as f64
}

fn calculate_pi(tot_pair_diff: usize, n: usize, seq_length: u32) -> f64 {
    let num_comparisons = n * (n - 1) / 2;
    tot_pair_diff as f64 / num_comparisons as f64 / seq_length as f64
}
