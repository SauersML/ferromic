use clap::Parser;
use ferromic::parse::{
    find_vcf_file, open_vcf_reader, parse_config_file, parse_region, parse_regions_file,
};
use ferromic::process::{process_config_entries, Args, ConfigEntry, VcfError, ZeroBasedHalfOpen};
use ferromic::progress::{
    display_status_box, finish_all, init_global_progress, log, update_global_progress, LogLevel,
    StatusBox,
};
use rayon::ThreadPoolBuilder;
use std::collections::HashMap;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A helper function to read sample names from the VCF header,
/// returning them in the order found after the `#CHROM POS ID REF ALT ...` columns.
fn read_sample_names_from_vcf(vcf_path: &Path) -> Result<Vec<String>, VcfError> {
    let mut reader = open_vcf_reader(vcf_path)?;
    let mut buffer = String::new();

    while reader.read_line(&mut buffer)? > 0 {
        // The line is in `buffer`. Check if it starts with "#CHROM"
        if buffer.starts_with("#CHROM") {
            // The sample names start at column index 9 in the VCF header
            // e.g. "#CHROM POS ID REF ALT QUAL FILTER INFO FORMAT Sample1 Sample2 ..."
            let split: Vec<&str> = buffer.split_whitespace().collect();
            if split.len() <= 9 {
                return Err(VcfError::Parse(
                    "VCF header found, but no sample columns".to_string(),
                ));
            }
            let sample_names: Vec<String> = split[9..].iter().map(|s| s.to_string()).collect();
            return Ok(sample_names);
        }
        buffer.clear();
    }

    Err(VcfError::Parse(
        "No #CHROM line found in VCF header".to_string(),
    ))
}

fn main() -> Result<(), VcfError> {
    let args = Args::parse();

    // Set Rayon to use all logical CPUs
    let num_logical_cpus = num_cpus::get();
    ThreadPoolBuilder::new()
        .num_threads(num_logical_cpus)
        .build_global()
        .unwrap();

    display_status_box(StatusBox {
        title: "Ferromic VCF Analysis".to_string(),
        stats: vec![
            ("Version".to_string(), env!("CARGO_PKG_VERSION").to_string()),
            ("CPU Threads".to_string(), num_logical_cpus.to_string()),
            (
                "Date".to_string(),
                chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
            ),
        ],
    });

    // Parse a mask file (regions to exclude)
    let mask_regions = if let Some(mask_file) = args.mask_file.as_ref() {
        log(
            LogLevel::Info,
            &format!("Mask file provided: {}", mask_file),
        );
        Some(Arc::new(
            parse_regions_file(Path::new(mask_file))?
                .into_iter()
                .map(|(chr, regions)| {
                    (
                        chr,
                        regions
                            .into_iter()
                            .map(|r| (r.start as i64, r.end as i64))
                            .collect(),
                    )
                })
                .collect(),
        ))
    } else {
        None
    };

    // Parse an allow file (regions to include)
    let allow_regions = if let Some(allow_file) = args.allow_file.as_ref() {
        log(
            LogLevel::Info,
            &format!("Allow file provided: {}", allow_file),
        );
        Some(Arc::new(
            parse_regions_file(Path::new(allow_file))?
                .into_iter()
                .map(|(chr, regions)| {
                    (
                        chr,
                        regions
                            .into_iter()
                            .map(|r| (r.start as i64, r.end as i64))
                            .collect(),
                    )
                })
                .collect(),
        ))
    } else {
        None
    };

    log(LogLevel::Info, "Starting VCF analysis with ferromic...");

    // ------------------------------------------------------------------------
    // CASE 1: A config file is provided
    // ------------------------------------------------------------------------
    if let Some(config_file) = args.config_file.as_ref() {
        log(
            LogLevel::Info,
            &format!("Config file provided: {}", config_file),
        );
        let config_entries = parse_config_file(Path::new(config_file))?;

        let output_file = args
            .output_file
            .as_ref()
            .map(Path::new)
            .unwrap_or_else(|| Path::new("output.csv"));

        // Initialize global progress with total entries
        init_global_progress(config_entries.len());
        log(
            LogLevel::Info,
            &format!("Starting analysis of {} regions", config_entries.len()),
        );

        // Hand off to the standard config-based pipeline
        process_config_entries(
            &config_entries,
            &args.vcf_folder,
            output_file,
            args.min_gq,
            mask_regions.clone(),
            allow_regions.clone(),
            &args,
        )?;

    // ------------------------------------------------------------------------
    // CASE 2: Single-chromosome approach (no config file)
    //         We build a single config entry with all samples in group 0.
    // ------------------------------------------------------------------------
    } else if let Some(chr) = args.chr.as_ref() {
        // Figure out region start/end from user input, or default to the entire chromosome
        let interval = if let Some(region_str) = args.region.as_ref() {
            parse_region(region_str)?
        } else {
            // If no region, use 1 to i64::MAX as a 1-based inclusive range; the pipeline will clamp to the actual chromosome length
            ZeroBasedHalfOpen::from_1based_inclusive(1, i64::MAX)
        };

        // Find a VCF for this chromosome
        let vcf_file = find_vcf_file(&args.vcf_folder, chr)?;
        // Collect sample names so we can assign them to a default group
        let sample_names = read_sample_names_from_vcf(&vcf_file)?;

        log(
            LogLevel::Info,
            &format!(
                "Processing chromosome {} with {} samples",
                chr,
                sample_names.len()
            ),
        );

        // Build a trivial "all samples => group 0" mapping
        let mut samples_unfiltered: HashMap<String, (u8, u8)> = HashMap::new();
        for sname in sample_names {
            // The tuple (0,0) means: left haplotype belongs to group 0, right haplotype belongs to group 0
            samples_unfiltered.insert(sname, (0, 0));
        }
        // Filtered groups can be the same in this scenario
        let samples_filtered = samples_unfiltered.clone();

        // Create a single ConfigEntry using the pre-constructed interval
        let config_entry = ConfigEntry {
            seqname: chr.to_string(),
            interval,
            samples_unfiltered,
            samples_filtered,
        };

        let output_file = args
            .output_file
            .as_ref()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("output.csv"));

        // Initialize global progress with just one entry
        init_global_progress(1);
        update_global_progress(0, &format!("Processing chr{}", chr));

        // Reuse the standard config-based pipeline with our single entry
        process_config_entries(
            &vec![config_entry],
            &args.vcf_folder,
            &output_file,
            args.min_gq,
            mask_regions.clone(),
            allow_regions.clone(),
            &args,
        )?;
    } else {
        // Neither a config file nor a chromosome was specified
        return Err(VcfError::Parse(
            "Either --config_file or --chr must be specified".to_string(),
        ));
    }

    finish_all();
    Ok(())
}
