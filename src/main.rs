use std::path::{Path, PathBuf};
use std::fs::{self, File};
use std::io::{Read, Write, BufReader, BufWriter, BufRead};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::runtime::Runtime;
use rayon::prelude::*;
use flate2::read::MultiGzDecoder;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
use human_bytes::human_bytes;
use std::sync::Mutex;
use sysinfo::{System, SystemExt};
use log::{info, debug, error, LevelFilter};
use env_logger::Builder;
use num_cpus;
use crossbeam_channel::{bounded, Sender, Receiver, TrySendError};
use memmap2::MmapOptions;
use thiserror::Error;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    input: String,

    #[arg(short, long)]
    output: String,

    #[arg(short, long, default_value = "100")]
    chunk_size: usize,

    #[arg(short, long, default_value_t = num_cpus::get())]
    threads: usize,
}

#[derive(Debug)]
struct VcfFile {
    path: PathBuf,
    chromosome: String,
    is_compressed: bool,
}

#[derive(Debug, Clone)]
struct Chunk {
    data: Vec<u8>,
    chromosome: String,
}

#[derive(Debug, thiserror::Error)]
enum VcfError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Channel send error: {0}")]
    ChannelSend(String),
    #[error("Channel receive error: {0}")]
    ChannelRecv(String),
    #[error("Memory limit exceeded")]
    MemoryLimitExceeded,
    #[error("UTF-8 conversion error: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    Builder::new().filter_level(LevelFilter::Info).init();

    let args = Args::parse();
    info!("VCF Concatenator");
    info!("Input directory: {}", args.input);
    info!("Output file: {}", args.output);
    info!("Chunk size: {} MB", args.chunk_size);
    info!("Threads: {}", args.threads);

    let runtime = Runtime::new()?;
    runtime.block_on(async {
        process_vcf_files(&args.input, &args.output, args.chunk_size * 1024 * 1024, args.threads).await
    })?;

    info!("Concatenation completed successfully.");
    Ok(())
}



async fn process_vcf_files(input_dir: &str, output_file: &str, chunk_size: usize, num_threads: usize) -> Result<(), VcfError> {
    let vcf_files = discover_and_sort_vcf_files(input_dir)?;

    if vcf_files.is_empty() {
        return Err(VcfError::Parse("No VCF files found in the input directory".to_string()));
    }

    info!("Found {} VCF files. Starting concatenation...", vcf_files.len());

    let mut sys = System::new_all();
    sys.refresh_all();

    let total_memory = sys.total_memory();
    let max_memory_usage = total_memory / 2;
    info!("Total system memory: {}, Max allowed usage: {}",
          human_bytes(total_memory as f64),
          human_bytes(max_memory_usage as f64));

    let (chunk_sender, chunk_receiver) = bounded(num_threads * 2);
    let output_file = Arc::new(Mutex::new(BufWriter::new(File::create(output_file)?)));

    let header = extract_header(&vcf_files[0])?;
    {
        let mut output = output_file.lock().map_err(|e| VcfError::Parse(e.to_string()))?;
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

    let writer_handle = tokio::spawn(async move {
        chunk_writer(chunk_receiver, output_file).await
    });

    let memory_usage = Arc::new(AtomicUsize::new(0));

    let worker_results: Vec<Result<(), VcfError>> = vcf_files.into_par_iter().map(|file| {
        let chunk_sender = chunk_sender.clone();
        let memory_usage = memory_usage.clone();
        if file.is_compressed {
            process_compressed_file(&file, chunk_size, memory_usage, max_memory_usage, chunk_sender)
        } else {
            process_uncompressed_file(&file, chunk_size, memory_usage, max_memory_usage, chunk_sender)
        }
    }).collect();

    // Check for any errors in worker results
    for result in worker_results {
        result?;
    }

    drop(chunk_sender);
    writer_handle.await.map_err(|e| VcfError::Parse(format!("Writer task failed: {}", e)))??;

    overall_pb.finish_with_message("Concatenation completed");

    Ok(())
}

fn discover_and_sort_vcf_files(dir: &str) -> Result<Vec<VcfFile>, VcfError> {
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
                        is_compressed: extension == "gz",
                    }))
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect::<Result<Vec<_>, VcfError>>()?;

    progress_bar.finish_with_message("File discovery completed");

    info!("Sorting VCF files by chromosome...");
    let mut sorted_files = vcf_files;
    sorted_files.par_sort_unstable_by(|a, b| custom_chromosome_sort(&a.chromosome, &b.chromosome));

    info!("Total VCF files found: {}", sorted_files.len());
    Ok(sorted_files)
}

fn custom_chromosome_sort(a: &str, b: &str) -> std::cmp::Ordering {
    let order = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "X", "Y", "MT"];
    let a_pos = order.iter().position(|&x| x == a);
    let b_pos = order.iter().position(|&x| x == b);
    a_pos.cmp(&b_pos)
}

fn get_chromosome(path: &Path) -> Result<String, VcfError> {
    let file = File::open(path)?;
    let mut reader: Box<dyn BufRead> = if path.extension().and_then(|ext| ext.to_str()) == Some("gz") {
        Box::new(BufReader::new(MultiGzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };

    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Err(VcfError::Parse("No data lines found in VCF file".to_string()));
        }
        if !line.starts_with('#') {
            // This is the first data line
            break;
        }
    }

    line.split('\t')
        .next()
        .map(|s| s.trim_start_matches("chr").to_string())
        .ok_or_else(|| VcfError::Parse("Failed to extract chromosome from VCF file".to_string()))
}

fn extract_header(file: &VcfFile) -> Result<String, VcfError> {
    let mut header = String::new();
    let reader = create_reader(&file.path)?;
    let buf_reader = BufReader::new(reader);

    for line in buf_reader.lines() {
        let line = line?;
        if line.starts_with('#') {
            header.push_str(&line);
            header.push('\n');
        } else {
            // We've reached the end of the header
            break;
        }
    }

    // Ensure the header ends with the #CHROM line
    if !header.contains("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO") {
        return Err(VcfError::Parse("Invalid VCF header: missing #CHROM line".to_string()));
    }

    Ok(header)
}

fn process_uncompressed_file(
    file: &VcfFile,
    chunk_size: usize,
    memory_usage: Arc<AtomicUsize>,
    max_memory_usage: u64,
    chunk_sender: Sender<Chunk>
) -> Result<(), VcfError> {
    let file_handle = File::open(&file.path)?;
    let mmap = unsafe { MmapOptions::new().map(&file_handle)? };

    let mut offset = 0;
    let file_size = mmap.len();

    while offset < file_size {
        let mut end = std::cmp::min(offset + chunk_size, file_size);
        let mut chunk_data = mmap[offset..end].to_vec();

        // Ensure chunk ends with a complete line
        if end < file_size {
            if let Some(newline_pos) = chunk_data.iter().rposition(|&b| b == b'\n') {
                chunk_data.truncate(newline_pos + 1);
                end = offset + newline_pos + 1;
            }
        }

        let chunk_size = chunk_data.len();
        let current_usage = memory_usage.fetch_add(chunk_size, Ordering::SeqCst);
        debug!("Processing file: {:?}, Added chunk of size {}. New memory usage: {}",
              file.path, human_bytes(chunk_size as f64), human_bytes((current_usage + chunk_size) as f64));

        send_chunk(Chunk {
            data: chunk_data,
            chromosome: file.chromosome.clone(),
        }, &chunk_sender)?;

        offset = end;
    }

    Ok(())
}

fn process_compressed_file(
    file: &VcfFile,
    chunk_size: usize,
    memory_usage: Arc<AtomicUsize>,
    max_memory_usage: u64,
    chunk_sender: Sender<Chunk>
) -> Result<(), VcfError> {
    let mut reader = create_reader(&file.path)?;
    let mut buffer = vec![0; chunk_size];

    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }

        let mut chunk_data = buffer[..bytes_read].to_vec();

        // Ensure chunk ends with a complete line
        if bytes_read == chunk_size {
            if let Some(newline_pos) = chunk_data.iter().rposition(|&b| b == b'\n') {
                chunk_data.truncate(newline_pos + 1);
            }
        }

        let chunk_size = chunk_data.len();
        let current_usage = memory_usage.fetch_add(chunk_size, Ordering::SeqCst);
        debug!("Processing file: {:?}, Added chunk of size {}. New memory usage: {}",
              file.path, human_bytes(chunk_size as f64), human_bytes((current_usage + chunk_size) as f64));

        send_chunk(Chunk {
            data: chunk_data,
            chromosome: file.chromosome.clone(),
        }, &chunk_sender)?;
    }

    Ok(())
}

fn send_chunk(
    chunk: Chunk,
    chunk_sender: &Sender<Chunk>,
) -> Result<(), VcfError> {
    loop {
        match chunk_sender.try_send(chunk.clone()) {
            Ok(_) => return Ok(()),
            Err(TrySendError::Full(_)) => {
                // Channel is full, yield to allow writer to process
                std::thread::yield_now();
                // Try again with the same chunk
                continue;
            },
            Err(TrySendError::Disconnected(_)) => {
                return Err(VcfError::ChannelSend("Chunk receiver disconnected".to_string()));
            }
        }
    }
}

async fn chunk_writer(
    chunk_receiver: Receiver<Chunk>,
    output_file: Arc<Mutex<BufWriter<File>>>,
) -> Result<(), VcfError> {
    let mut current_chromosome = String::new();
    let mut chunks_written = 0;
    let mut total_bytes_written = 0;

    while let Ok(chunk) = chunk_receiver.recv() {
        if chunk.chromosome != current_chromosome {
            //info!("Switching to chromosome: {}", chunk.chromosome);
            current_chromosome = chunk.chromosome;
        }

        let chunk_size = chunk.data.len();
        {
            let mut output = output_file.lock().map_err(|e| VcfError::Parse(e.to_string()))?;
            output.write_all(&chunk.data)?;
            output.flush()?;
        }

        total_bytes_written += chunk_size;
        chunks_written += 1;

        debug!("Wrote chunk of size {}. Total bytes written: {}",
              human_bytes(chunk_size as f64), human_bytes(total_bytes_written as f64));

        if chunks_written % 1000 == 0 {
            info!("Progress: {} chunks written", chunks_written);
        }
    }

    info!("All chunks written. Total data processed: {}", human_bytes(total_bytes_written as f64));
    Ok(())
}

fn create_reader(path: &Path) -> Result<Box<dyn BufRead>, VcfError> {
    let file = File::open(path)?;
    let reader: Box<dyn BufRead> = if path.extension().and_then(|ext| ext.to_str()) == Some("gz") {
        Box::new(BufReader::new(MultiGzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };
    Ok(reader)
}
