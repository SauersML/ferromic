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
use log::{info, debug, error, LevelFilter};
use env_logger::Builder;
use num_cpus;
use crossbeam_channel::{unbounded, Sender, Receiver};
use memmap2::MmapOptions;
use sysinfo::{System, SystemExt, ProcessExt, Pid};
use std::thread;
use std::collections::HashMap;
use std::time::{Instant, Duration};


const UPDATE_FREQUENCY_BYTES: usize = 1000 * 1024 * 1024; // MB
const PRINT_INTERVAL: Duration = Duration::from_secs(5); // Print every 5 seconds


impl Default for ChromosomeProgress {
    fn default() -> Self {
        ChromosomeProgress {
            total_bytes: 0,
            processed_bytes: 0,
            last_print: Instant::now(),
        }
    }
}

struct ChromosomeProgress {
    total_bytes: u64,
    processed_bytes: u64,
    last_print: Instant,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    input: String,

    #[arg(short, long)]
    output: String,

    #[arg(short, long, default_value = "10")]
    chunk_size: usize,

    #[arg(short, long, default_value_t = num_cpus::get())]
    threads: usize,

    #[arg(short = 'g', long, help = "Memory limit in GB")]
    memory_limit: Option<u64>,
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
        process_vcf_files(&args).await
    })?;

    info!("Concatenation completed successfully.");
    Ok(())
}





async fn process_vcf_files(args: &Args) -> Result<(), VcfError> {
    let vcf_files = discover_and_sort_vcf_files(&args.input)?;

    if vcf_files.is_empty() {
        return Err(VcfError::Parse("No VCF files found in the input directory".to_string()));
    }

    info!("Found {} VCF files. Starting concatenation...", vcf_files.len());

    let mut sys = System::new_all();
    sys.refresh_all();

    let total_memory = sys.total_memory();
    let max_memory_usage = if let Some(limit) = args.memory_limit {
        limit * 1024 * 1024 * 1024 // Convert GB to bytes
    } else {
        total_memory / 2
    };
    info!("Total system memory: {}, Max allowed usage: {}",
          human_bytes(total_memory as f64),
          human_bytes(max_memory_usage as f64));

    let (chunk_sender, chunk_receiver) = crossbeam_channel::unbounded();
    let output_file = Arc::new(Mutex::new(BufWriter::new(File::create(&args.output)?)));

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

    let chromosome_progress = Arc::new(Mutex::new(HashMap::<String, ChromosomeProgress>::new()));
    let chromosome_progress_clone = chromosome_progress.clone();

    let writer_handle = thread::spawn(move || {
        chunk_writer(chunk_receiver, output_file, chromosome_progress_clone)
    });


    for file in &vcf_files {
        let mut progress = chromosome_progress.lock().unwrap();
        progress.entry(file.chromosome.clone()).or_default().total_bytes += file.path.metadata()?.len();
    }

    println!("About to start processing {} files", vcf_files.len());

    let memory_usage = Arc::new(AtomicUsize::new(0));

    let worker_results: Vec<Result<(), VcfError>> = vcf_files.into_par_iter().map(|file| {
        let chunk_sender = chunk_sender.clone();
        let memory_usage = memory_usage.clone();
        let chromosome_progress = chromosome_progress.clone();
        println!("Started processing file: {:?}", file.path);
        if file.is_compressed {
            process_compressed_file(&file, args.chunk_size * 1024 * 1024, memory_usage, max_memory_usage, chunk_sender, chromosome_progress)
        } else {
            process_uncompressed_file(&file, args.chunk_size * 1024 * 1024, memory_usage, max_memory_usage, chunk_sender, chromosome_progress)
        }
    }).collect();

    // Check for any errors in worker results
    for result in worker_results {
        result?;
    }

    drop(chunk_sender);
    writer_handle.join().map_err(|e| VcfError::Parse(format!("Writer thread panicked: {:?}", e)))??;

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
    chunk_sender: Sender<Chunk>,
    chromosome_progress: Arc<Mutex<HashMap<String, ChromosomeProgress>>>
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
        }, &chunk_sender, &chromosome_progress)?;

        offset = end;
    }

    Ok(())
}





fn process_compressed_file(
    file: &VcfFile,
    chunk_size: usize,
    memory_usage: Arc<AtomicUsize>,
    max_memory_usage: u64,
    chunk_sender: Sender<Chunk>,
    chromosome_progress: Arc<Mutex<HashMap<String, ChromosomeProgress>>>
) -> Result<(), VcfError> {
    println!("Starting to process compressed file: {:?}", file.path);
    let mut reader = create_reader(&file.path)?;
    println!("Reader created for file: {:?}", file.path);
    let mut buffer = vec![0; chunk_size];
    let mut accumulated_data = Vec::new();

    loop {
        //println!("Attempting to read from file: {:?}", file.path);
        let bytes_read = reader.read(&mut buffer)?;
        //println!("Read {} bytes from file: {:?}", bytes_read, file.path);
        
        if bytes_read == 0 && accumulated_data.is_empty() {
            break;
        }

        accumulated_data.extend_from_slice(&buffer[..bytes_read]);

        if accumulated_data.len() >= chunk_size || bytes_read == 0 {
            // Ensure chunk ends with a complete line
            if let Some(newline_pos) = accumulated_data.iter().rposition(|&b| b == b'\n') {
                let chunk_data = accumulated_data.split_off(newline_pos + 1);
                let chunk_size = chunk_data.len();
                
                let current_usage = memory_usage.fetch_add(chunk_size, Ordering::SeqCst);
                debug!("Processing file: {:?}, Added chunk of size {}. New memory usage: {}",
                      file.path, human_bytes(chunk_size as f64), human_bytes((current_usage + chunk_size) as f64));

                    send_chunk(Chunk {
                        data: chunk_data,
                        chromosome: file.chromosome.clone(),
                    }, &chunk_sender, &chromosome_progress)?;
                                    
                //println!("Chunk sent successfully for file: {:?}, size: {}", file.path, chunk_size);
            }

            // Simple backpressure
            if chunk_sender.len() > 1000 {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }

        if bytes_read == 0 {
            break;
        }
    }

    // Send any remaining data
    if !accumulated_data.is_empty() {
        let chunk_size = accumulated_data.len();
        let current_usage = memory_usage.fetch_add(chunk_size, Ordering::SeqCst);
        debug!("Processing file: {:?}, Added final chunk of size {}. New memory usage: {}",
              file.path, human_bytes(chunk_size as f64), human_bytes((current_usage + chunk_size) as f64));
        send_chunk(Chunk {
            data: accumulated_data, // Changed from chunk_data to accumulated_data
            chromosome: file.chromosome.clone(),
        }, &chunk_sender, &chromosome_progress)?;
        //println!("Final chunk sent successfully for file: {:?}, size: {}", file.path, chunk_size);
    }

    Ok(())
}


fn send_chunk(
    chunk: Chunk,
    chunk_sender: &Sender<Chunk>,
    chromosome_progress: &Arc<Mutex<HashMap<String, ChromosomeProgress>>>
) -> Result<(), VcfError> {
    let chunk_size = chunk.data.len();

    // Update progress
    let mut progress = chromosome_progress.lock().unwrap();
    let chr_progress = progress.get_mut(&chunk.chromosome).unwrap();
    chr_progress.processed_bytes += chunk_size as u64;
    
    let now = Instant::now();
    if now.duration_since(chr_progress.last_print) >= PRINT_INTERVAL {
        let percent = (chr_progress.processed_bytes as f64 / chr_progress.total_bytes as f64) * 100.0;
        println!("Chromosome {}: {:.2}% complete ({} / {})",
            chunk.chromosome,
            percent,
            human_bytes(chr_progress.processed_bytes as f64),
            human_bytes(chr_progress.total_bytes as f64)
        );
        chr_progress.last_print = now;
    }

    // Send chunk
    chunk_sender.send(chunk).map_err(|_| VcfError::ChannelSend("Chunk receiver disconnected".to_string()))
}

fn chunk_writer(
    chunk_receiver: Receiver<Chunk>,
    output_file: Arc<Mutex<BufWriter<File>>>,
    chromosome_progress: Arc<Mutex<HashMap<String, ChromosomeProgress>>>
) -> Result<(), VcfError> {
    println!("Chunk writer started and waiting for chunks");
    let mut current_chromosome = String::new();
    let mut total_bytes_written = 0;
    let mut last_update_time = Instant::now();
    let mut sys = System::new_all();

    while let Ok(chunk) = chunk_receiver.recv() {
        //println!("Received chunk for chromosome: {}, size: {} bytes", chunk.chromosome, chunk.data.len());
        
        if chunk.chromosome != current_chromosome {
            //println!("Switching to chromosome: {}", chunk.chromosome);
            current_chromosome = chunk.chromosome;
        }

        let chunk_size = chunk.data.len();
        {
            let mut output = output_file.lock().map_err(|e| VcfError::Parse(e.to_string()))?;
            output.write_all(&chunk.data)?;
            output.flush()?;
            //println!("Wrote chunk of size {} bytes to file", chunk_size);
        }

        total_bytes_written += chunk_size;

        let now = Instant::now();
        if now.duration_since(last_update_time) >= PRINT_INTERVAL {
            sys.refresh_process(Pid::from(std::process::id() as usize));
            if let Some(process) = sys.process(Pid::from(std::process::id() as usize)) {
                let used_ram = process.memory();
                println!(
                    "Overall Progress: {} processed. RAM usage: {}",
                    human_bytes(total_bytes_written as f64),
                    human_bytes(used_ram as f64)
                );

                let progress = chromosome_progress.lock().unwrap();
                for (chr, chr_progress) in progress.iter() {
                    let percent = (chr_progress.processed_bytes as f64 / chr_progress.total_bytes as f64) * 100.0;
                    println!("Chromosome {}: {:.2}% complete ({} / {})",
                        chr,
                        percent,
                        human_bytes(chr_progress.processed_bytes as f64),
                        human_bytes(chr_progress.total_bytes as f64)
                    );
                }
            } else {
                println!("Failed to get process information");
            }
            last_update_time = now;
        }
    }

    println!("Chunk writer finished. Total data processed: {}", human_bytes(total_bytes_written as f64));
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
