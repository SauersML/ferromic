use std::path::{Path, PathBuf};
use std::fs::{self, File};
use std::io::{Read, Write, BufReader, BufWriter, BufRead};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::runtime::Runtime;
use rayon::prelude::*;
use flate2::read::MultiGzDecoder;
use anyhow::{Context, Result, bail};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
use human_bytes::human_bytes;
use std::sync::Mutex;
use sysinfo::{System, SystemExt};
use log::{info, warn, error, LevelFilter};
use env_logger::Builder;
use num_cpus;
use crossbeam_channel::{bounded, Sender, Receiver};
use memmap2::MmapOptions;

// CLI arguments
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    input: String,

    #[arg(short, long)]
    output: String,
}

// Represents a VCF file
struct VcfFile {
    path: PathBuf,
    chromosome: String,
}

// Represents a chunk of data from a VCF file
struct Chunk {
    data: Vec<u8>,
    chromosome: String,
}

fn main() -> Result<()> {
    // Initialize logging
    Builder::new().filter_level(LevelFilter::Info).init();

    let args = Args::parse();
    info!("VCF Concatenator");
    info!("Input directory: {}", args.input);
    info!("Output file: {}", args.output);

    // Create and run Tokio runtime
    let runtime = Runtime::new().context("Failed to create Tokio runtime")?;
    runtime.block_on(async {
        process_vcf_files(&args.input, &args.output).await
    })?;

    info!("Concatenation completed successfully.");
    Ok(())
}

// Main processing function
async fn process_vcf_files(input_dir: &str, output_file: &str) -> Result<()> {
    let vcf_files = discover_and_sort_vcf_files(input_dir)
        .context("Failed to discover and sort VCF files")?;

    if vcf_files.is_empty() {
        bail!("No VCF files found in the input directory");
    }

    info!("Found {} VCF files. Starting concatenation...", vcf_files.len());

    let num_threads = num_cpus::get();
    info!("Using {} threads for processing", num_threads);

    let mut sys = System::new_all();
    sys.refresh_all();

    let total_memory = sys.total_memory();
    let max_memory_usage = total_memory / 2; // Use up to 50% of total system memory
    info!("Total system memory: {}, Max allowed usage: {}", 
          human_bytes(total_memory as f64), 
          human_bytes(max_memory_usage as f64));

    let chunk_size = calculate_chunk_size(max_memory_usage, num_threads);
    info!("Calculated chunk size: {}", human_bytes(chunk_size as f64));

    let (chunk_sender, chunk_receiver) = bounded(num_threads * 2);
    let output_file = Arc::new(Mutex::new(BufWriter::new(File::create(output_file)?)));

    // Extract and write header
    let header = extract_header(&vcf_files[0])?;
    {
        let mut output = output_file.lock().unwrap();
        output.write_all(header.as_bytes())?;
        output.flush()?;
    }
    info!("Header extracted and written to output file");

    let progress = Arc::new(AtomicUsize::new(0));
    let total_files = vcf_files.len();
    let multi_progress = MultiProgress::new();

    let overall_pb = multi_progress.add(ProgressBar::new(total_files as u64));
    overall_pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")
        .expect("Failed to set progress bar template")
        .progress_chars("#>-"));
    overall_pb.set_message("Overall progress");

    let writer_handle = tokio::spawn(chunk_writer(chunk_receiver, output_file.clone()));

    let memory_usage = Arc::new(AtomicUsize::new(0));

    // Process files in parallel
    vcf_files.into_par_iter().for_each(|file| {
        let chunk_sender = chunk_sender.clone();
        let memory_usage = memory_usage.clone();
        let result = process_file(&file, chunk_size, memory_usage, max_memory_usage, chunk_sender);
        if let Err(e) = result {
            error!("Error processing file {:?}: {}", file.path, e);
        }
        progress.fetch_add(1, Ordering::SeqCst);
        overall_pb.inc(1);
    });

    drop(chunk_sender); // Close the channel
    writer_handle.await??;

    overall_pb.finish_with_message("Concatenation completed");
    
    Ok(())
}

// Discover and sort VCF files
fn discover_and_sort_vcf_files(dir: &str) -> Result<Vec<VcfFile>> {
    info!("Discovering VCF files in: {}", dir);
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

    info!("Sorting VCF files by chromosome...");
    let mut sorted_files = vcf_files;
    sorted_files.par_sort_unstable_by(|a, b| custom_chromosome_sort(&a.chromosome, &b.chromosome));

    info!("Total VCF files found: {}", sorted_files.len());
    Ok(sorted_files)
}

// Custom chromosome sorting function
fn custom_chromosome_sort(a: &str, b: &str) -> std::cmp::Ordering {
    let order = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "X", "Y", "MT"];
    let a_pos = order.iter().position(|&x| x == a);
    let b_pos = order.iter().position(|&x| x == b);
    a_pos.cmp(&b_pos)
}

// Extract chromosome from VCF file
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

    first_data_line.split('\t')
        .next()
        .map(|s| s.trim_start_matches("chr").to_string())
        .context("Failed to extract chromosome from VCF file")
}

// Extract header from VCF file
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

// Calculate chunk size based on available memory
fn calculate_chunk_size(max_memory_usage: u64, num_threads: usize) -> usize {
    let total_chunk_memory = (max_memory_usage as f64 * 0.8) as u64;
    let memory_per_thread = total_chunk_memory / num_threads as u64;
    let chunk_size = memory_per_thread / 2;

    info!("Memory allocation for chunks: {}", human_bytes(total_chunk_memory as f64));
    info!("Memory per thread: {}", human_bytes(memory_per_thread as f64));
    info!("Calculated chunk size: {}", human_bytes(chunk_size as f64));

    chunk_size as usize
}

// Process a single VCF file
fn process_file(
    file: &VcfFile, 
    chunk_size: usize,
    memory_usage: Arc<AtomicUsize>,
    max_memory_usage: u64,
    chunk_sender: Sender<Chunk>
) -> Result<()> {
    let file_handle = File::open(&file.path)?;
    let mmap = unsafe { MmapOptions::new().map(&file_handle)? };

    let mut offset = 0;
    let file_size = mmap.len();

    while offset < file_size {
        let end = std::cmp::min(offset + chunk_size, file_size);
        let chunk_data = mmap[offset..end].to_vec();

        let chunk_size = chunk_data.len();
        let current_usage = memory_usage.fetch_add(chunk_size, Ordering::SeqCst);
        info!("Processing file: {:?}, Added chunk of size {}. New memory usage: {}", 
              file.path, human_bytes(chunk_size as f64), human_bytes((current_usage + chunk_size) as f64));

        // Wait if memory usage is too high
        while memory_usage.load(Ordering::SeqCst) as u64 > max_memory_usage {
            warn!("Memory usage exceeded limit. Waiting for memory to be freed...");
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        chunk_sender.send(Chunk {
            data: chunk_data,
            chromosome: file.chromosome.clone(),
        })?;

        offset = end;
    }

    Ok(())
}

// Write chunks to output file
async fn chunk_writer(
    chunk_receiver: Receiver<Chunk>,
    output_file: Arc<Mutex<BufWriter<File>>>,
) -> Result<()> {
    let mut current_chromosome = String::new();
    let mut chunks_written = 0;
    let mut total_bytes_written = 0;

    while let Ok(chunk) = chunk_receiver.recv() {
        if chunk.chromosome != current_chromosome {
            info!("Switching to chromosome: {}", chunk.chromosome);
            current_chromosome = chunk.chromosome;
        }

        let chunk_size = chunk.data.len();
        {
            let mut output = output_file.lock().unwrap();
            output.write_all(&chunk.data)?;
            output.flush()?;
        }

        total_bytes_written += chunk_size;
        chunks_written += 1;

        info!("Wrote chunk of size {}. Total bytes written: {}", 
              human_bytes(chunk_size as f64), human_bytes(total_bytes_written as f64));

        if chunks_written % 100 == 0 {
            info!("Progress: {} chunks written", chunks_written);
        }
    }

    info!("All chunks written. Total data processed: {}", human_bytes(total_bytes_written as f64));
    Ok(())
}

// Create a BufRead for a given path
fn create_reader(path: &Path) -> Result<Box<dyn BufRead>> {
    let file = File::open(path)?;
    let reader: Box<dyn BufRead> = if path.extension().and_then(|ext| ext.to_str()) == Some("gz") {
        Box::new(BufReader::new(MultiGzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };
    Ok(reader)
}
