use anyhow::{bail, Context, Result};
use clap::Parser;
use colored::*;
use flate2::read::MultiGzDecoder;
use human_bytes::human_bytes;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::runtime::Runtime;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    input: String,

    #[arg(short, long)]
    output: String,

    #[arg(short, long, default_value = "100")]
    chunk_size: usize,

    #[arg(short, long, default_value = "8")]
    threads: usize,
}

struct VcfFile {
    path: PathBuf,
    chromosome: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let chunk_size = args.chunk_size * 1024 * 1024; // Convert MB to bytes

    println!("{}", "VCF Concatenator".green().bold());
    println!("Input directory: {}", args.input.blue());
    println!("Output file: {}", args.output.blue());
    println!("Chunk size: {}", human_bytes(chunk_size as f64).yellow());
    println!("Threads: {}", args.threads.to_string().yellow());

    let vcf_files = discover_and_sort_vcf_files(&args.input)
        .context("Failed to discover and sort VCF files")?;

    if vcf_files.is_empty() {
        bail!("No VCF files found in the input directory");
    }

    println!(
        "Found {} VCF files. Starting concatenation...",
        vcf_files.len().to_string().green()
    );

    let runtime = Runtime::new().context("Failed to create Tokio runtime")?;
    runtime.block_on(async {
        concatenate_files(vcf_files, &args.output, chunk_size, args.threads).await
    })?;

    println!("{}", "Concatenation completed successfully.".green().bold());
    Ok(())
}

fn discover_and_sort_vcf_files(dir: &str) -> Result<Vec<VcfFile>> {
    println!("Discovering VCF files in: {}", dir.blue());
    let progress_bar = ProgressBar::new_spinner();
    progress_bar.set_message("Scanning directory...");

    let vcf_files: Vec<VcfFile> = fs::read_dir(dir)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.is_file() {
                let extension = path.extension()?.to_str()?;
                if extension == "vcf" || extension == "gz" {
                    progress_bar.inc(1);
                    Some(Ok(VcfFile {
                        path: path.clone(),
                        chromosome: get_chromosome(&path).ok()?,
                    }))
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect::<Result<Vec<_>>>()?;

    progress_bar.finish_with_message("File discovery completed");

    println!("Sorting VCF files by chromosome...");
    let mut sorted_files = vcf_files;
    sorted_files.par_sort_unstable_by(|a, b| custom_chromosome_sort(&a.chromosome, &b.chromosome));

    println!(
        "Total VCF files found: {}",
        sorted_files.len().to_string().green()
    );
    Ok(sorted_files)
}

fn custom_chromosome_sort(a: &str, b: &str) -> std::cmp::Ordering {
    let order = [
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16",
        "17", "18", "19", "20", "21", "22", "X", "Y", "MT",
    ];
    let a_pos = order.iter().position(|&x| x == a);
    let b_pos = order.iter().position(|&x| x == b);
    a_pos.cmp(&b_pos)
}

fn get_chromosome(path: &Path) -> Result<String> {
    let file = File::open(path)?;
    let mut reader: Box<dyn Read> = if path.extension().and_then(|ext| ext.to_str()) == Some("gz") {
        Box::new(MultiGzDecoder::new(file))
    } else {
        Box::new(file)
    };

    let mut buffer = [0; 1024];
    let mut first_data_line = String::new();

    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        let chunk = String::from_utf8_lossy(&buffer[..bytes_read]);
        for line in chunk.lines() {
            if !line.starts_with('#') {
                first_data_line = line.to_string();
                break;
            }
        }
        if !first_data_line.is_empty() {
            break;
        }
    }

    first_data_line
        .split('\t')
        .next()
        .map(|s| s.trim_start_matches("chr").to_string())
        .context("Failed to extract chromosome from VCF file")
}

async fn concatenate_files(
    vcf_files: Vec<VcfFile>,
    output_file: &str,
    chunk_size: usize,
    num_threads: usize,
) -> Result<()> {
    let (tx, mut rx) = tokio::sync::mpsc::channel(num_threads);
    let output = Arc::new(tokio::sync::Mutex::new(BufWriter::new(File::create(
        output_file,
    )?)));

    println!("Extracting header from the first file...");
    let header = extract_header(&vcf_files[0])?;
    output.lock().await.write_all(header.as_bytes())?;
    println!("Header extracted and written to output file");

    let progress = Arc::new(AtomicUsize::new(0));
    let total_files = vcf_files.len();
    let multi_progress = MultiProgress::new();

    let overall_pb = multi_progress.add(ProgressBar::new(total_files as u64));
    overall_pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
            )
            .expect("Failed to set progress bar template")
            .progress_chars("#>-"),
    );
    overall_pb.set_message("Overall progress");

    let file_pb = multi_progress.add(ProgressBar::new(100));
    file_pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {msg}")
            .expect("Failed to set progress bar template")
            .progress_chars("#>-"),
    );
    file_pb.set_message("Current file progress");

    println!("Starting parallel file processing...");
    for (index, file) in vcf_files.into_iter().enumerate() {
        let tx = tx.clone();
        let file_pb = file_pb.clone();
        tokio::spawn(async move {
            file_pb.set_message(format!("Processing file {}: {:?}", index + 1, file.path));
            let result = process_file(&file, chunk_size, file_pb);
            tx.send(result).await.unwrap();
        });
    }

    let mut total_bytes_processed = 0;
    for _ in 0..total_files {
        if let Some(result) = rx.recv().await {
            let chunks = result?;
            for chunk in chunks {
                total_bytes_processed += chunk.len();
                output.lock().await.write_all(&chunk)?;
            }
            progress.fetch_add(1, Ordering::SeqCst);
            overall_pb.inc(1);
        }
    }

    output.lock().await.flush()?;
    overall_pb.finish_with_message("Concatenation completed");
    file_pb.finish_with_message("All files processed");

    println!(
        "Total data processed: {}",
        human_bytes(total_bytes_processed as f64).green()
    );
    Ok(())
}

fn extract_header(file: &VcfFile) -> Result<String> {
    let mut header = String::new();
    let reader = create_reader(&file.path)?;
    let buf_reader = BufReader::new(reader);

    for line in buf_reader.lines() {
        let line = line?;
        if line.starts_with('#') {
            header.push_str(&line);
            header.push('\n');
        } else {
            break;
        }
    }

    Ok(header)
}

fn process_file(file: &VcfFile, chunk_size: usize, pb: ProgressBar) -> Result<Vec<Vec<u8>>> {
    let mut chunks = Vec::new();
    let mut reader = create_reader(&file.path)?;
    let mut buffer = vec![0; chunk_size];
    let mut in_header = true;
    let mut total_bytes = 0;

    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }

        total_bytes += bytes_read;
        pb.set_position((total_bytes as f64 / chunk_size as f64 * 100.0) as u64);

        let mut chunk = buffer[..bytes_read].to_vec();

        if in_header {
            if let Some(pos) = chunk.windows(2).position(|w| w == b"\n#") {
                in_header = false;
                chunk = chunk[pos + 1..].to_vec();
            } else {
                continue;
            }
        }

        chunks.push(chunk);
    }

    pb.finish_with_message("File processed");
    Ok(chunks)
}

fn create_reader(path: &Path) -> Result<Box<dyn BufRead>> {
    let file = File::open(path)?;
    let reader: Box<dyn BufRead> = if path.extension().and_then(|ext| ext.to_str()) == Some("gz") {
        Box::new(BufReader::new(MultiGzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };
    Ok(reader)
}
