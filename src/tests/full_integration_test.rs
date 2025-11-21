#[cfg(test)]
mod full_integration_test {
    use crate::process::{process_config_entries, Args, ConfigEntry, ZeroBasedHalfOpen};
    use rand::Rng;
    use std::collections::{HashMap, HashSet};
    use std::fs::File;
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use tempfile::TempDir;

    // Helper to write a dummy FASTA reference file with an index
    fn write_dummy_reference(temp_path: &Path, chromosomes: &[String], length: usize) -> PathBuf {
        let ref_path = temp_path.join("ref.fa");
        let index_path = temp_path.join("ref.fa.fai");

        // Write the FASTA file: single line sequence for simplicity
        let mut f = File::create(&ref_path).unwrap();
        let mut current_offset = 0;

        let mut fai_content = String::new();

        for chr in chromosomes {
            let header = format!(">{}\n", chr);
            f.write_all(header.as_bytes()).unwrap();
            current_offset += header.len();

            let seq_len = length;
            let seq = "A".repeat(seq_len);
            f.write_all(seq.as_bytes()).unwrap();
            f.write_all(b"\n").unwrap();

            // NAME LENGTH OFFSET LINEBASES LINEWIDTH
            // LINEBASES = seq_len, LINEWIDTH = seq_len+1 (byte per line incl newline)
            fai_content.push_str(&format!(
                "{}\t{}\t{}\t{}\t{}\n",
                chr,
                seq_len,
                current_offset,
                seq_len,
                seq_len + 1
            ));

            current_offset += seq_len + 1;
        }

        // Write the FAI index
        let mut fai = File::create(&index_path).unwrap();
        fai.write_all(fai_content.as_bytes()).unwrap();

        ref_path
    }

    // Helper to write a dummy GTF file
    fn write_dummy_gtf(temp_path: &Path, chromosomes: &[String]) -> PathBuf {
        let gtf_path = temp_path.join("annot.gtf");
        let mut f = File::create(&gtf_path).unwrap();
        // Just write one dummy transcript per chromosome covering the whole range we care about
        for chr in chromosomes {
            // chr1 source exon 1 100000 . + . transcript_id "tx1";
            writeln!(f, "{}\tferromic_test\texon\t1\t100000\t.\t+\t.\ttranscript_id \"tx_{}\"; gene_id \"gene_{}\";", chr, chr, chr).unwrap();
            // CDS
            writeln!(f, "{}\tferromic_test\tCDS\t1\t100000\t.\t+\t0\ttranscript_id \"tx_{}\"; gene_id \"gene_{}\";", chr, chr, chr).unwrap();
        }
        gtf_path
    }


    #[test]
    fn test_full_pipeline_integration() {
        // 1. Setup
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path();

        let samples: Vec<String> = (0..8).map(|i| format!("Sample{}", i)).collect();
        let chromosomes: Vec<String> = (1..=25).map(|i| format!("chr{}", i)).collect(); // more than 20 for multiple entries

        // Create Ref and GTF
        // Reference needs to be long enough for 100,000 range
        let ref_path = write_dummy_reference(temp_path, &chromosomes, 100000);
        let gtf_path = write_dummy_gtf(temp_path, &chromosomes);

        // 2. Create Config Entries and VCFs
        let mut config_entries = Vec::new();
        let mut rng = rand::thread_rng();

        // Track which chromosomes have been used to assign distinct regions
        let mut chr_usage_count: HashMap<String, usize> = HashMap::new();

        // 30 entries, distributed
        for i in 0..30 {
            let chr_idx = i % chromosomes.len();
            let chr = &chromosomes[chr_idx];
            let usage = chr_usage_count.entry(chr.clone()).or_insert(0);

            // Region assignment:
            // 1st entry: 1000..5000
            // 2nd entry: 6000..10000
            let start = 1000 + (*usage * 5000);
            let end = start + 4000; // 4000 bp length
            *usage += 1;

            let interval = ZeroBasedHalfOpen { start: start as usize, end: end as usize };

            // Assign haplotypes randomly to group 0 or 1
            let mut samples_unfiltered = HashMap::new();
            for sample in &samples {
                let left = rng.gen_range(0..=1);
                let right = rng.gen_range(0..=1);
                samples_unfiltered.insert(sample.clone(), (left, right));
            }
            // Same for filtered (user said "mix haplotype group assignments randomly, these are tsv config groupings")
            let samples_filtered = samples_unfiltered.clone();

            config_entries.push(ConfigEntry {
                seqname: chr.clone(),
                interval,
                samples_unfiltered,
                samples_filtered,
            });
        }

        // Generate VCFs for all chromosomes
        // We need to know all regions per chromosome to generate variants
        let mut regions_per_chr: HashMap<String, Vec<(i64, i64)>> = HashMap::new();
        for entry in &config_entries {
            regions_per_chr.entry(entry.seqname.clone())
                .or_default()
                .push((entry.interval.start as i64, entry.interval.end as i64));
        }

        for chr in &chromosomes {
            if let Some(regions) = regions_per_chr.get(chr) {
                let vcf_path = temp_path.join(format!("{}.vcf", chr));
                let mut f = File::create(&vcf_path).unwrap();

                writeln!(f, "##fileformat=VCFv4.2").unwrap();
                writeln!(f, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{}", samples.join("\t")).unwrap();

                // Collect all variants for this chromosome
                let mut variants: Vec<(i64, String)> = Vec::new();

                for (start, end) in regions {
                    // Generate 15 variants per region
                    for _ in 0..15 {
                        // Generate a random position within the region [start, end)
                        // Convert to 1-based for VCF
                        let pos_0 = rng.gen_range(*start..*end);
                        let pos_1 = pos_0 + 1;

                        let mut gt_strings = Vec::new();
                        for _ in &samples {
                            let a1 = rng.gen_range(0..=1);
                            let a2 = rng.gen_range(0..=1);
                            gt_strings.push(format!("{}|{}:60", a1, a2));
                        }

                        let line = format!("{}\t{}\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t{}", chr, pos_1, gt_strings.join("\t"));
                        variants.push((pos_1, line));
                    }
                }

                // Sort variants by position to ensure valid VCF
                variants.sort_by_key(|k| k.0);

                for (_, line) in variants {
                    writeln!(f, "{}", line).unwrap();
                }
            }
        }

        // 3. Prepare Args
        let output_file = temp_path.join("output.csv");
        let args = Args {
            vcf_folder: temp_path.to_str().unwrap().to_string(),
            chr: None,
            region: None,
            config_file: None,
            output_file: Some(output_file.to_str().unwrap().to_string()),
            min_gq: 30,
            mask_file: None,
            allow_file: None,
            exclude: None,
            reference_path: ref_path.to_str().unwrap().to_string(),
            gtf_path: gtf_path.to_str().unwrap().to_string(),
            enable_pca: false,
            pca_components: 10,
            pca_output: "pca_results.tsv".to_string(),
            enable_fst: false,
            fst_populations: None,
        };

        let exclusion_set = HashSet::new();

        // 4. Run Pipeline
        let result = process_config_entries(
            &config_entries,
            &args.vcf_folder,
            &output_file,
            args.min_gq,
            None, // mask
            None, // allow
            &args,
            &exclusion_set,
            temp_path,
        );

        assert!(result.is_ok(), "Pipeline execution failed: {:?}", result.err());

        // 5. Verify Output
        // Read output.csv
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(&output_file)
            .unwrap();

        let headers = rdr.headers().unwrap().clone();

        // Columns to check for non-zero/non-missing
        // pi_0, pi_1, pi_0_f, pi_1_f
        // inv_freq_no_filter

        let mut row_count = 0;

        for result in rdr.records() {
            let record = result.unwrap();
            row_count += 1;

            // Check Pi values
            let pi_0: f64 = record[headers.iter().position(|h| h == "0_pi").unwrap()].parse().unwrap();
            let pi_1: f64 = record[headers.iter().position(|h| h == "1_pi").unwrap()].parse().unwrap();
            let pi_0_f: f64 = record[headers.iter().position(|h| h == "0_pi_filtered").unwrap()].parse().unwrap();
            let pi_1_f: f64 = record[headers.iter().position(|h| h == "1_pi_filtered").unwrap()].parse().unwrap();

            assert!(pi_0 > 0.0, "Unfiltered Pi group 0 should be > 0, got {}", pi_0);
            assert!(pi_1 > 0.0, "Unfiltered Pi group 1 should be > 0, got {}", pi_1);

            // Since no filters (no mask, no low GQ, no missing), filtered should equal unfiltered
            assert!((pi_0 - pi_0_f).abs() < 1e-9, "Filtered Pi 0 should equal Unfiltered Pi 0");
            assert!((pi_1 - pi_1_f).abs() < 1e-9, "Filtered Pi 1 should equal Unfiltered Pi 1");

            // Check segregation sites
             let seg_0: usize = record[headers.iter().position(|h| h == "0_segregating_sites").unwrap()].parse().unwrap();
             assert!(seg_0 > 0, "Should have segregating sites");
        }

        assert_eq!(row_count, 30, "Should have 30 output rows");
    }
}
