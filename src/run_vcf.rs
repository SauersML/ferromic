use clap::Parser;
use colored::*;
use ferromic::parse::{
    find_vcf_file, open_vcf_reader, parse_config_file, parse_region, parse_regions_file,
};
use ferromic::process::{
    Args, ConfigEntry, VcfError, ZeroBasedHalfOpen, create_temp_dir, process_config_entries,
};
use ferromic::progress::{
    LogLevel, StatusBox, display_status_box, finish_all, force_flush_all, init_global_progress,
    log, update_global_progress,
};
use ferromic::transcripts;
use rayon::ThreadPoolBuilder;
use std::collections::{HashMap, HashSet};
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Helper function to resolve exclusion requests against available samples.
/// It checks VCF headers and (optionally) config file entries.
/// Returns the set of FULL sample names that should be excluded.

fn resolve_sample_exclusions(
    vcf_folder: &str,
    chr: &str,
    requested_exclusions: &HashSet<String>,
    config_entries: Option<&[ConfigEntry]>,
) -> Result<HashSet<String>, VcfError> {
    if requested_exclusions.is_empty() {
        return Ok(HashSet::new());
    }

    // 1. Gather all known sample names separately
    let mut vcf_sample_ids: HashSet<String> = HashSet::new();
    let mut config_sample_ids: HashSet<String> = HashSet::new();

    // From VCF
    match find_vcf_file(vcf_folder, chr) {
        Ok(vcf_file) => {
            if let Ok(vcf_samples) = read_sample_names_from_vcf(&vcf_file) {
                vcf_sample_ids.extend(vcf_samples);
            }
        }
        Err(_) => {
            // If VCF not found, we can't validate against it, but we might have config entries.
        }
    }

    // From Config
    if let Some(entries) = config_entries {
        for entry in entries {
            config_sample_ids.extend(entry.samples_unfiltered.keys().cloned());
            config_sample_ids.extend(entry.samples_filtered.keys().cloned());
        }
    }

    if vcf_sample_ids.is_empty() && config_sample_ids.is_empty() {
        // If we found no samples anywhere, we can't resolve anything.
        // Return original set as a fallback to avoid dropping anything potentially valid
        // but not found in the representative check.
        return Ok(requested_exclusions.clone());
    }

    // 2. Resolve
    let mut resolved_set = HashSet::new();
    let mut missing = Vec::new();
    let mut successful_requests = 0usize;
    let mut vcf_resolved_set = HashSet::new();
    let mut config_resolved_set = HashSet::new();

    let mut requests: Vec<&String> = requested_exclusions.iter().collect();
    requests.sort();

    for req in requests {
        let trimmed = req.trim();

        // VCF matches
        let (vcf_matches, vcf_match_type) = if vcf_sample_ids.contains(trimmed) {
            (vec![trimmed.to_string()], "Exact Match")
        } else {
            let mut substrings: Vec<String> = vcf_sample_ids
                .iter()
                .filter(|sample| sample.contains(trimmed))
                .cloned()
                .collect();
            substrings.sort();
            if substrings.is_empty() {
                (Vec::new(), "No Match")
            } else {
                (substrings, "Fuzzy Match")
            }
        };

        // Config matches
        let (config_matches, config_match_type) = if config_sample_ids.contains(trimmed) {
            (vec![trimmed.to_string()], "Exact Match")
        } else {
            let mut substrings: Vec<String> = config_sample_ids
                .iter()
                .filter(|sample| sample.contains(trimmed))
                .cloned()
                .collect();
            substrings.sort();
            if substrings.is_empty() {
                (Vec::new(), "No Match")
            } else {
                (substrings, "Fuzzy Match")
            }
        };

        let mut request_ids: HashSet<String> = HashSet::new();
        for id in &vcf_matches {
            request_ids.insert(id.clone());
            vcf_resolved_set.insert(id.clone());
        }
        for id in &config_matches {
            request_ids.insert(id.clone());
            config_resolved_set.insert(id.clone());
        }

        let action_message = if request_ids.is_empty() {
            "No IDs were added to exclusion list.".to_string()
        } else {
            successful_requests += 1;
            resolved_set.extend(request_ids.iter().cloned());
            format!(
                "Added {} specific IDs to exclusion list.",
                request_ids.len()
            )
        };

        let log_message = format!(
            "[INFO] Exclusion Request: '{}'\n       - VCF Matches ({}): {:?} ({})\n       - TSV Matches ({}): {:?} ({})\n       - Action: {}",
            trimmed,
            vcf_matches.len(),
            vcf_matches,
            vcf_match_type,
            config_matches.len(),
            config_matches,
            config_match_type,
            action_message
        );

        log(LogLevel::Info, &log_message);

        if request_ids.is_empty() {
            log(
                LogLevel::Warning,
                &format!(
                    "[WARN] Exclusion Request '{}' yielded no matches in VCF headers or Config columns. Marking as ghost.",
                    trimmed
                ),
            );
            missing.push(trimmed.to_string());
        }
    }

    // 3. Report
    log(
        LogLevel::Info,
        &format!(
            "[INFO] Exclusion Summary:
       - User Requests Processed: {}
       - User Requests Resolved: {}
       - Total IDs Banned: {}
       - Breakdown: {} from VCF headers, {} from Config columns",
            requested_exclusions.len(),
            successful_requests,
            resolved_set.len(),
            vcf_resolved_set.len(),
            config_resolved_set.len()
        ),
    );

    if !missing.is_empty() {
        let mut missing_sorted = missing.clone();
        missing_sorted.sort();
        let msg = format!(
            "WARNING: The following samples were requested for exclusion but NOT found in VCF or Config headers (tried exact, trimmed, and substring): {:?}. Check your spelling.",
            missing_sorted
        );
        eprintln!("{}", msg.yellow().bold());
    }

    Ok(resolved_set)
}
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
    // Register panic hook
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // 1. Flush logs to save critical crash info
        force_flush_all();
        // 2. Also flush the metadata writer in transcripts.rs
        let _ = transcripts::flush_metadata();
        // 3. Call the default hook to print the error to stderr
        default_hook(info);
    }));

    let args = Args::parse();
    // Initial exclusion set from args
    let initial_exclusion_set: HashSet<String> = args
        .exclude
        .clone()
        .unwrap_or_default()
        .into_iter()
        .collect();

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

    // Create temp directory
    let _temp_dir_guard = create_temp_dir()?;
    let temp_path = _temp_dir_guard.path();

    // ------------------------------------------------------------------------
    // CASE 1: A config file is provided
    // ------------------------------------------------------------------------
    if let Some(config_file) = args.config_file.as_ref() {
        log(
            LogLevel::Info,
            &format!("Config file provided: {}", config_file),
        );
        let mut config_entries = parse_config_file(Path::new(config_file))?;

        // Resolve exclusions using the first entry's chromosome + config entries
        let resolved_exclusion_set = if let Some(first_entry) = config_entries.first() {
            resolve_sample_exclusions(
                &args.vcf_folder,
                &first_entry.seqname,
                &initial_exclusion_set,
                Some(&config_entries),
            )?
        } else {
            initial_exclusion_set.clone()
        };

        let mut removed_from_config_count: HashMap<String, usize> = HashMap::new();

        for entry in config_entries.iter_mut() {
            entry.samples_unfiltered.retain(|sample, _| {
                if resolved_exclusion_set.contains(sample) {
                    *removed_from_config_count.entry(sample.clone()).or_default() += 1;
                    false
                } else {
                    true
                }
            });
            entry.samples_filtered.retain(|sample, _| {
                if resolved_exclusion_set.contains(sample) {
                    *removed_from_config_count.entry(sample.clone()).or_default() += 1;
                    false
                } else {
                    true
                }
            });
        }

        if !removed_from_config_count.is_empty() {
            for (sample, count) in removed_from_config_count {
                log(
                    LogLevel::Info,
                    &format!(
                        "Removed '{}' from {} entries in configuration file maps.",
                        sample, count
                    ),
                );
            }
        }

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
            &resolved_exclusion_set,
            temp_path,
        )?;

    // ------------------------------------------------------------------------
    // CASE 2: Single-chromosome approach (no config file)
    //         We build a single config entry with all samples in group 0.
    // ------------------------------------------------------------------------
    } else if let Some(chr) = args.chr.as_ref() {
        // Resolve exclusions using just the chromosome
        let resolved_exclusion_set =
            resolve_sample_exclusions(&args.vcf_folder, chr, &initial_exclusion_set, None)?;

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
        let sample_names: Vec<String> = read_sample_names_from_vcf(&vcf_file)?
            .into_iter()
            .filter(|name| !resolved_exclusion_set.contains(name))
            .collect();

        if sample_names.is_empty() {
            return Err(VcfError::Parse(
                "No samples remain after applying exclusions".to_string(),
            ));
        }

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
            &resolved_exclusion_set,
            temp_path,
        )?;
    } else {
        // Neither a config file nor a chromosome was specified
        return Err(VcfError::Parse(
            "Either --config_file or --chr must be specified".to_string(),
        ));
    }

    finish_all();
    // Flush the metadata writer to ensure all data is written to disk.
    if let Err(e) = transcripts::flush_metadata() {
        log(
            LogLevel::Error,
            &format!("Failed to flush metadata writer: {}", e),
        );
    }
    Ok(())
}
