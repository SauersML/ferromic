use ferromic::process::{
    display_seqinfo_entries, process_config_entries, process_vcf, Args, VcfError, ZeroBasedHalfOpen,
    HaplotypeSide,
};

use ferromic::parse::{find_vcf_file, parse_config_file, parse_region, parse_regions_file};

use ferromic::stats::{calculate_pi, calculate_watterson_theta, count_segregating_sites};

use clap::Parser;
use colored::Colorize;
use parking_lot::Mutex;
use rayon::ThreadPoolBuilder;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

fn main() -> Result<(), VcfError> {
    let args = Args::parse();

    // Set Rayon to use all logical CPUs
    let num_logical_cpus = num_cpus::get();
    ThreadPoolBuilder::new()
        .num_threads(num_logical_cpus)
        .build_global()
        .unwrap();

    // Parse the mask file (exclude regions)
    let mask_regions = if let Some(mask_file) = args.mask_file.as_ref() {
        println!("Mask file provided: {}", mask_file);
        // Convert ZeroBasedHalfOpen to (i64, i64) tuples for compatibility with process_config_entries
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

    // Parse the allow file (include regions)
    let allow_regions = if let Some(allow_file) = args.allow_file.as_ref() {
        println!("Allow file provided: {}", allow_file);
        let parsed_allow = parse_regions_file(Path::new(allow_file))?;
        println!("Parsed Allow Regions: {:?}", parsed_allow);
        // Convert ZeroBasedHalfOpen to (i64, i64) tuples for compatibility with process_config_entries
        Some(Arc::new(
            parsed_allow
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
    println!("{}", "Starting VCF diversity analysis...".green());

    if let Some(config_file) = args.config_file.as_ref() {
        println!("Config file provided: {}", config_file);
        let config_entries = parse_config_file(Path::new(config_file))?;
        for entry in &config_entries {
            println!("Config entry chromosome: {}", entry.seqname);
        }
        let output_file = args
            .output_file
            .as_ref()
            .map(Path::new)
            .unwrap_or_else(|| Path::new("output.csv"));
        println!("Output file: {}", output_file.display());
        process_config_entries(
            &config_entries,
            &args.vcf_folder,
            output_file,
            args.min_gq,
            mask_regions.clone(),
            allow_regions.clone(),
            &args,
        )?;
    } else if let Some(chr) = args.chr.as_ref() {
        println!("Chromosome provided: {}", chr);
        let (start, end) = if let Some(region) = args.region.as_ref() {
            println!("Region provided: {}", region);
            // Extract i64 start and end from ZeroBasedHalfOpen for process_vcf compatibility
            let region = parse_region(region)?;
            (region.start as i64, region.end as i64)
        } else {
            println!("No region provided, using default region covering most of the chromosome.");
            (1, i64::MAX)
        };
        let vcf_file = find_vcf_file(&args.vcf_folder, chr)?;

        println!(
            "{}",
            format!("Processing VCF file: {}", vcf_file.display()).cyan()
        );

        // Initialize shared SeqInfo storage
        let seqinfo_storage_unfiltered = Arc::new(Mutex::new(Vec::new()));
        let seqinfo_storage_filtered = Arc::new(Mutex::new(Vec::new()));

        // Initialize shared allele map storage for unfiltered and filtered data.
        let position_allele_map_unfiltered =
            Arc::new(Mutex::new(HashMap::<i64, (char, char)>::new()));
        let position_allele_map_filtered =
            Arc::new(Mutex::new(HashMap::<i64, (char, char)>::new()));

        // Process the VCF file
        let (
            unfiltered_variants,
            _filtered_variants,
            sample_names,
            chr_length,
            missing_data_info,
            _filtering_stats,
        ) = process_vcf(
            &vcf_file,
            &Path::new(&args.reference_path),
            chr.to_string(),
            ZeroBasedHalfOpen::from_1based_inclusive(start, end),
            args.min_gq,
            mask_regions.clone(),
            allow_regions.clone(),
            seqinfo_storage_unfiltered.clone(), // Pass unfiltered SeqInfo storage
            seqinfo_storage_filtered.clone(),   // Pass filtered SeqInfo storage
            position_allele_map_unfiltered.clone(), // Pass unfiltered allele map
            position_allele_map_filtered.clone(), // Pass filtered allele map
        )?;

        {
            let seqinfo = seqinfo_storage_unfiltered.lock(); // Access the unfiltered SeqInfo
            if !seqinfo.is_empty() {
                display_seqinfo_entries(&seqinfo, 12);
            } else {
                println!("No SeqInfo entries were stored.");
            }
        }
        println!("{}", "Calculating diversity statistics...".blue());

        let seq_length = if end == i64::MAX {
            // Use the last variant position if available, otherwise fall back to chr_length
            let last_pos = unfiltered_variants
                .last()
                .map(|v| v.position)
                .unwrap_or(0)
                .max(chr_length);
                
            // Create a ZeroBasedHalfOpen interval and use its len() method
            let actual_region = ZeroBasedHalfOpen::from_1based_inclusive(start, last_pos as i64);
            actual_region.len() as i64
        } else {
            // Create a ZeroBasedHalfOpen interval from the specified boundaries and use its len() method
            let specified_region = ZeroBasedHalfOpen::from_1based_inclusive(start, end);
            specified_region.len() as i64
        };

        if end == i64::MAX
            && unfiltered_variants.last().map(|v| v.position).unwrap_or(0) < chr_length
        {
            println!("{}", "Warning: The sequence length may be underestimated. Consider using the --region parameter for more accurate results.".yellow());
        }

        let num_segsites = count_segregating_sites(&unfiltered_variants); // Also need filtered here? Output required in csv: 0_segregating_sites_filtered, 1_segregating_sites_filtered
        let raw_variant_count = unfiltered_variants.len();

        let n = sample_names.len();
        if n == 0 {
            return Err(VcfError::Parse(
                "No samples found after processing VCF.".to_string(),
            ));
        }

        // Needs to work with both filtered and unfiltered
        let group_haps: Vec<(usize, HaplotypeSide)> = sample_names
            .iter()
            .enumerate()
            .flat_map(|(i, _)| vec![(i, HaplotypeSide::Left), (i, HaplotypeSide::Right)])
            .collect();
        let pi = calculate_pi(&unfiltered_variants, &group_haps);

        let w_theta = calculate_watterson_theta(num_segsites, sample_names.len() * 2, seq_length);

        println!("\n{}", "Results:".green().bold());
        println!("\nSequence Length:{}", seq_length);
        println!("Number of Segregating Sites:{}", num_segsites);
        println!("Raw Variant Count:{}", raw_variant_count);
        println!("Watterson Theta:{:.6}", w_theta);
        println!("pi:{:.6}", pi);

        if unfiltered_variants.is_empty() {
            println!(
                "{}",
                "Warning: No variants found in the specified region.".yellow()
            );
        }

        if num_segsites == 0 {
            println!("{}", "Warning: All sites are monomorphic.".yellow());
        }

        if num_segsites != raw_variant_count {
            println!(
                "{}",
                format!(
                    "Note: Number of segregating sites ({}) differs from raw variant count ({}).",
                    num_segsites, raw_variant_count
                )
                .yellow()
            );
        }

        println!("\n{}", "Filtering Statistics:".green().bold());
        println!(
            "Total variants processed: {}",
            _filtering_stats.total_variants
        );
        println!(
            "Filtered variants: {} ({:.2}%)",
            _filtering_stats._filtered_variants,
            (_filtering_stats._filtered_variants as f64 / _filtering_stats.total_variants as f64)
                * 100.0
        );
        println!(
            "Multi-allelic variants: {}",
            _filtering_stats.multi_allelic_variants
        );
        println!("Low GQ variants: {}", _filtering_stats.low_gq_variants);
        println!(
            "Missing data variants: {}",
            _filtering_stats.missing_data_variants
        );

        let missing_data_percentage = (missing_data_info.missing_data_points as f64
            / missing_data_info.total_data_points as f64)
            * 100.0;
        println!("\n{}", "Missing Data Information:".yellow().bold());
        println!(
            "Number of missing data points: {}",
            missing_data_info.missing_data_points
        );
        println!(
            "Percentage of missing data: {:.2}%",
            missing_data_percentage
        );
        println!(
            "Number of positions with missing data: {}",
            missing_data_info.positions_with_missing.len()
        );
    } else {
        return Err(VcfError::Parse(
            "Either config file or chromosome must be specified".to_string(),
        ));
    }

    println!("{}", "Analysis complete.".green());
    Ok(())
}
