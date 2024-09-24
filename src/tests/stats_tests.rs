#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::{NamedTempFile, tempdir};
    use std::fs::{self, File};
    use std::io::{self, Write};
    use std::path::PathBuf;
    use std::collections::HashMap;

    // Helper function to create a Variant for testing
    fn create_variant(position: i64, genotypes: Vec<Option<Vec<u8>>>) -> Variant {
        Variant { position, genotypes }
    }

    #[test]
    fn test_count_segregating_sites() {
        let variants = vec![
            create_variant(1, vec![Some(vec![0, 0]), Some(vec![0, 1]), Some(vec![1, 1])]),
            create_variant(2, vec![Some(vec![0, 0]), Some(vec![0, 0]), Some(vec![0, 0])]),
            create_variant(3, vec![Some(vec![0, 1]), Some(vec![0, 1]), Some(vec![0, 1])]),
            create_variant(4, vec![Some(vec![0, 0]), Some(vec![1, 1]), Some(vec![0, 1])]),
        ];

        assert_eq!(count_segregating_sites(&variants), 3);

        // Test with no variants
        assert_eq!(count_segregating_sites(&[]), 0);

        // Test with all homozygous sites
        let homozygous_variants = vec![
            create_variant(1, vec![Some(vec![0, 0]), Some(vec![0, 0]), Some(vec![0, 0])]),
            create_variant(2, vec![Some(vec![1, 1]), Some(vec![1, 1]), Some(vec![1, 1])]),
        ];
        assert_eq!(count_segregating_sites(&homozygous_variants), 0);

        // Test with missing data
        let missing_data_variants = vec![
            create_variant(1, vec![Some(vec![0, 0]), None, Some(vec![1, 1])]),
            create_variant(2, vec![Some(vec![0, 1]), Some(vec![0, 1]), None]),
        ];
        assert_eq!(count_segregating_sites(&missing_data_variants), 2);
    }

    #[test]
    fn test_extract_sample_id() {
        // Standard cases
        assert_eq!(extract_sample_id("sample_123"), "123");
        assert_eq!(extract_sample_id("sample_with_multiple_underscores_456"), "456");

        // Edge cases
        assert_eq!(extract_sample_id("singlepart"), "singlepart");
        assert_eq!(extract_sample_id(""), "");
        assert_eq!(extract_sample_id("_"), "");
        assert_eq!(extract_sample_id("sample_"), "");

        // Complex sample names
        assert_eq!(extract_sample_id("EAS_JPT_NA18939"), "NA18939");
        assert_eq!(extract_sample_id("AMR_PEL_HG02059"), "HG02059");

        // Extra cases
        assert_eq!(extract_sample_id("double__underscore"), "underscore");
        assert_eq!(extract_sample_id("triple_part_name_789"), "789");
    }

    #[test]
    fn test_harmonic() {
        assert_eq!(harmonic(1), 1.0);
        assert!((harmonic(2) - 1.5).abs() < 1e-10);
        assert!((harmonic(3) - (1.0 + 0.5 + 1.0/3.0)).abs() < 1e-10);
        assert!((harmonic(10) - 2.9289682539682538).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_pairwise_differences() {
        let variants = vec![
            create_variant(1000, vec![Some(vec![0, 0]), Some(vec![0, 1]), Some(vec![1, 1])]),
            create_variant(2000, vec![Some(vec![0, 0]), Some(vec![0, 0]), Some(vec![0, 1])]),
            create_variant(3000, vec![Some(vec![0, 1]), Some(vec![1, 1]), Some(vec![0, 0])]),
        ];
    
        let result = calculate_pairwise_differences(&variants, 3);
    
        assert_eq!(result.len(), 3);
    
        for &((i, j), count, ref positions) in &result {
            match (i, j) {
                (0, 1) => {
                    assert_eq!(count, 2);
                    assert_eq!(positions, &vec![1000, 3000]);
                },
                (0, 2) => {
                    assert_eq!(count, 3);
                    assert_eq!(positions, &vec![1000, 2000, 3000]);
                },
                (1, 2) => {
                    assert_eq!(count, 2);
                    assert_eq!(positions, &vec![2000, 3000]);
                },
                _ => panic!("Unexpected pair: ({}, {})", i, j),
            }
        }
    
        // Test with missing data
        let missing_data_variants = vec![
            create_variant(1, vec![Some(vec![0]), None, Some(vec![1])]),
            create_variant(2, vec![Some(vec![1]), Some(vec![1]), None]),
        ];
        let missing_data_result = calculate_pairwise_differences(&missing_data_variants, 3);
    
        assert_eq!(missing_data_result.len(), 3);
    
        for &((i, j), count, _) in &missing_data_result {
            match (i, j) {
                (0, 1) => assert_eq!(count, 0),
                (0, 2) => assert_eq!(count, 1),
                (1, 2) => assert_eq!(count, 0),
                _ => panic!("Unexpected pair: ({}, {})", i, j),
            }
        }
    }

    #[test]
    fn test_calculate_watterson_theta() {
        let epsilon = 1e-6;
        assert!((calculate_watterson_theta(10, 5, 1000) - 0.0048).abs() < epsilon);
        assert!((calculate_watterson_theta(5, 2, 1000) - 0.005).abs() < epsilon);
        assert!((calculate_watterson_theta(100, 10, 1_000_000) - 0.00003534).abs() < epsilon);
        let theta_n1 = calculate_watterson_theta(100, 1, 1000);
        assert!(theta_n1.is_infinite());
        let theta_n0 = calculate_watterson_theta(0, 0, 1000);
        assert!(theta_n0.is_infinite());
        let theta_seq_zero = calculate_watterson_theta(10, 5, 0);
        assert!(theta_seq_zero.is_infinite());
    
        // Helper function to compute expected pi
        fn expected_pi(tot_pair_diff: usize, n: usize, seq_length: i64) -> f64 {
            if n <= 1 || seq_length == 0 {
                return f64::INFINITY;
            }
            let num_comparisons = n * (n - 1) / 2;
            if num_comparisons == 0 {
                return f64::INFINITY;
            }
            tot_pair_diff as f64 / (num_comparisons as f64 * seq_length as f64)
        }
    
        // Test with typical values
        let pi = calculate_pi(15, 5, 1000);
        let expected = expected_pi(15, 5, 1000);
        assert!((pi - expected).abs() < 1e-10);
    
        // Test with no pairwise differences
        let pi = calculate_pi(0, 5, 1000);
        assert_eq!(pi, 0.0);
    
        // Test with minimum sample size (n = 2)
        let pi = calculate_pi(5, 2, 1000);
        let expected = expected_pi(5, 2, 1000);
        assert!((pi - expected).abs() < 1e-10);
    
        // Test with n = 1, expecting infinity
        let pi_n1 = calculate_pi(100, 1, 1000);
        assert!(pi_n1.is_infinite());
    
        // Test with seq_length = 0, expecting infinity
        let pi_seq_zero = calculate_pi(100, 10, 0);
        assert!(pi_seq_zero.is_infinite());
    
        // Test with large values
        let pi = calculate_pi(10000, 100, 10000);
        let expected = expected_pi(10000, 100, 10000);
        assert!((pi - expected).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_pi() {
        // Test with typical values
        assert!((calculate_pi(15, 5, 1000) - 0.0015).abs() < 1e-6);

        // Test with no pairwise differences
        assert_eq!(calculate_pi(0, 5, 1000), 0.0);

        // Test with large number of pairwise differences
        assert!((calculate_pi(10000, 100, 10000) - 0.00020202).abs() < 1e-6);

        // Test with minimum possible sample size (2)
        assert!((calculate_pi(5, 2, 1000) - 0.005).abs() < 1e-6);

        // Test with very large sequence length
        assert!((calculate_pi(1000, 10, 1_000_000) - 0.0000222222).abs() < 1e-9);

        // Test with n = 1, expecting infinity
        let pi_n1 = calculate_pi(100, 1, 1000);
        assert!(pi_n1.is_infinite());

        // Test with n = 0, expecting infinity
        let pi_n0 = calculate_pi(0, 0, 1000);
        assert!(pi_n0.is_infinite());
    }

    #[test]
    fn test_parse_region() {
        assert_eq!(parse_region("1-1000").unwrap(), (1, 1000));
        assert_eq!(parse_region("1000000-2000000").unwrap(), (1000000, 2000000));

        // Test invalid formats
        assert!(matches!(parse_region("1000"), Err(VcfError::InvalidRegion(_))));
        assert!(matches!(parse_region("1000-"), Err(VcfError::InvalidRegion(_))));
        assert!(matches!(parse_region("-1000"), Err(VcfError::InvalidRegion(_))));
        assert!(matches!(parse_region("a-1000"), Err(VcfError::InvalidRegion(_))));
        assert!(matches!(parse_region("1000-b"), Err(VcfError::InvalidRegion(_))));

        // Test invalid range (start >= end)
        assert!(matches!(parse_region("1000-1000"), Err(VcfError::InvalidRegion(_))));
        assert!(matches!(parse_region("2000-1000"), Err(VcfError::InvalidRegion(_))));
    }

    #[test]
    fn test_validate_vcf_header() {
        let valid_header = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1\tSAMPLE2";
        assert!(validate_vcf_header(valid_header).is_ok());

        let invalid_header = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO";
        assert!(matches!(validate_vcf_header(invalid_header), Err(VcfError::InvalidVcfFormat(_))));

        let invalid_order = "POS\t#CHROM\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT";
        assert!(matches!(validate_vcf_header(invalid_order), Err(VcfError::InvalidVcfFormat(_))));
    }

    #[test]
    fn test_parse_variant() {
        let sample_names = vec!["SAMPLE1".to_string(), "SAMPLE2".to_string(), "SAMPLE3".to_string()];
        let mut missing_data_info = MissingDataInfo::default();
        let min_gq = 30;
    
        // Test valid variant with all GQ values above threshold
        let valid_line = "chr1\t1500\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:35\t0|1:40\t1|1:45";
        let result = parse_variant(valid_line, "1", 1, 2000, &mut missing_data_info, &sample_names, min_gq);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    
        // Test variant with one GQ value below threshold
        let invalid_gq_line = "chr1\t1000\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:35\t0|1:25\t1|1:45";
        let result = parse_variant(invalid_gq_line, "1", 1, 2000, &mut missing_data_info, &sample_names, min_gq);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none()); // Expect None for low GQ variant
    
        // Test valid variant
        let valid_line = "chr1\t1000\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:35\t0|1:40\t1|1:45";
        let result = parse_variant(valid_line, "1", 1, 2000, &mut missing_data_info, &sample_names, min_gq);
        assert!(result.is_ok());
        if let Ok(Some(variant)) = result {
            assert_eq!(variant.position, 1000);
            assert_eq!(variant.genotypes, vec![Some(vec![0, 0]), Some(vec![0, 1]), Some(vec![1, 1])]);
        }
    
        // Test variant with low GQ
        let low_gq_line = "chr1\t1000\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:35\t0|1:20\t1|1:45";
        let result = parse_variant(low_gq_line, "1", 1, 2000, &mut missing_data_info, &sample_names, min_gq);
        assert!(result.unwrap().is_none());
    
        // Test variant outside region
        let out_of_range = "chr1\t3000\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:35\t0|1:40\t1|1:45";
        assert!(parse_variant(out_of_range, "1", 1, 2000, &mut missing_data_info, &sample_names, min_gq).unwrap().is_none());
    
        // Test different chromosome
        let diff_chr = "chr2\t1000\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:35\t0|1:40\t1|1:45";
        assert!(parse_variant(diff_chr, "1", 1, 2000, &mut missing_data_info, &sample_names, min_gq).unwrap().is_none());
    
        // Test missing data
        let missing_data = "chr1\t1000\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:35\t.|.:.\t1|1:45";
        let result = parse_variant(missing_data, "1", 1, 2000, &mut missing_data_info, &sample_names, min_gq);
        assert!(result.is_ok());
        if let Ok(Some(variant)) = result {
            assert_eq!(variant.genotypes, vec![Some(vec![0, 0]), None, Some(vec![1, 1])]);
        }
    
        // Test invalid format (fewer fields than required)
        let invalid_format = "chr1\t1000\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:35"; // Only 10 fields, expecting 12 for 3 samples
        assert!(parse_variant(invalid_format, "1", 1, 2000, &mut missing_data_info, &sample_names, min_gq).is_err());
    }

    #[test]
    fn test_process_variants() {
        let variants = vec![
            create_variant(1000, vec![Some(vec![0, 0]), Some(vec![0, 1]), Some(vec![1, 1])]),
            create_variant(2000, vec![Some(vec![0, 0]), Some(vec![0, 0]), Some(vec![0, 1])]),
            create_variant(3000, vec![Some(vec![0, 1]), Some(vec![1, 1]), Some(vec![0, 0])]),
        ];
        let sample_names = vec!["SAMPLE1".to_string(), "SAMPLE2".to_string(), "SAMPLE3".to_string()];
        let mut sample_filter = HashMap::new();
        sample_filter.insert("SAMPLE1".to_string(), (0, 1));
        sample_filter.insert("SAMPLE2".to_string(), (0, 1));
        sample_filter.insert("SAMPLE3".to_string(), (0, 1));

        // Test with empty variants
        let empty_result = process_variants(&[], &sample_names, 0, &sample_filter, 1000, 3000);
        assert!(empty_result.is_ok());
        if let Ok((num_segsites, w_theta, pi, num_haplotypes, allele_frequency)) = empty_result {
            assert_eq!(num_segsites, 0);
            assert_eq!(w_theta, 0.0);
            assert_eq!(pi, 0.0);
            assert_eq!(num_haplotypes, 3);
            assert!((allele_frequency - 1.0/3.0).abs() < 1e-6);
        }

        // Test with invalid haplotype group
        let invalid_group = process_variants(&variants, &sample_names, 2, &sample_filter, 1000, 3000);
        assert!(invalid_group.is_err());

        // Test with missing samples
        let mut missing_sample_filter = HashMap::new();
        missing_sample_filter.insert("SAMPLE4".to_string(), (0, 1));
        let missing_result = process_variants(&variants, &sample_names, 0, &missing_sample_filter, 1000, 3000);
        assert!(missing_result.is_ok());
        // Since SAMPLE4 is missing, no haplotypes should be processed
        let (num_segsites, w_theta, pi, num_haplotypes, allele_frequency) = missing_result.unwrap();
        assert_eq!(num_segsites, 0);
        assert_eq!(w_theta, 0.0);
        assert_eq!(pi, 0.0);
        assert_eq!(num_haplotypes, 0);
        assert_eq!(allele_frequency, 0.0);
    }

    #[test]
    fn test_parse_config_file_with_noreads() {
        use std::io::Write;
    
        let config_content = "seqnames\tstart\tend\tPOS\torig_ID\tverdict\tcateg\tSAMPLE1\tSAMPLE2\n\
                              chr1\t1000\t2000\t1500\ttest_id\tpass\tinv\t0|1_lowconf\t1|1\n\
                              chr1\t3000\t4000\t.\t.\t.\t.\t0|0\t0|1\n";
        let path = tempfile::NamedTempFile::new().unwrap();
        write!(path.as_file(), "{}", config_content).unwrap();
    
        let config_entries = parse_config_file(path.path()).unwrap();
        assert_eq!(config_entries.len(), 2);
    
        // For the first entry, SAMPLE1 has "0|1_lowconf" and should be skipped in samples_filtered
        // SAMPLE2 has "1|1" and should be included
        assert_eq!(config_entries[0].samples_filtered.len(), 1);
        assert!(config_entries[0].samples_filtered.contains_key("SAMPLE2"));
        assert!(!config_entries[0].samples_filtered.contains_key("SAMPLE1"));
    
        // For the second entry, both SAMPLE1 ("0|0") and SAMPLE2 ("0|1") are valid and should be included
        assert_eq!(config_entries[1].samples_filtered.len(), 2);
        assert!(config_entries[1].samples_filtered.contains_key("SAMPLE1"));
        assert!(config_entries[1].samples_filtered.contains_key("SAMPLE2"));
    
        assert_eq!(config_entries[0].seqname, "chr1");
        assert_eq!(config_entries[0].start, 1000);
        assert_eq!(config_entries[0].end, 2000);
    
        assert_eq!(config_entries[1].seqname, "chr1");
        assert_eq!(config_entries[1].start, 3000);
        assert_eq!(config_entries[1].end, 4000);
    }


    #[test]
    fn test_process_variants_no_segregating_sites() {
        let variants = vec![
            create_variant(1000, vec![Some(vec![0, 0]), Some(vec![0, 0]), Some(vec![0, 0])]),
            create_variant(2000, vec![Some(vec![0, 0]), Some(vec![0, 0]), Some(vec![0, 0])]),
        ];
        let sample_names = vec!["SAMPLE1".to_string(), "SAMPLE2".to_string(), "SAMPLE3".to_string()];
        let mut sample_filter = HashMap::new();
        sample_filter.insert("SAMPLE1".to_string(), (0, 0));
        sample_filter.insert("SAMPLE2".to_string(), (0, 0));
        sample_filter.insert("SAMPLE3".to_string(), (0, 0));

        let result = process_variants(&variants, &sample_names, 0, &sample_filter, 1000, 2000);
        assert!(result.is_ok());
        if let Ok((num_segsites, w_theta, pi, num_haplotypes, allele_frequency)) = result {
            assert_eq!(num_segsites, 0);
            assert_eq!(w_theta, 0.0);
            assert_eq!(pi, 0.0);
            assert_eq!(num_haplotypes, 3);
            assert_eq!(allele_frequency, 0.0);
        }
    }

    #[test]
    fn test_process_config_entries() {
        use std::fs::{self, File};
        use std::io::Write;
        
        // Create a temporary directory for testing
        let temp_dir = tempfile::tempdir().unwrap();
        let vcf_folder = temp_dir.path().join("vcf");
        fs::create_dir(&vcf_folder).unwrap();
        
        // Create a mock VCF file
        let vcf_content = "\
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1\tSAMPLE2\tSAMPLE3\tSAMPLE4
chr1\t1000\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:35\t0|1:40\t0|1:30\t1|0:45
chr1\t1500\t.\tG\tC\t.\tPASS\t.\tGT:GQ\t0|1:35\t1|1:40\t1|0:30\t0|1:45
chr1\t2000\t.\tT\tA\t.\tPASS\t.\tGT:GQ\t1|1:35\t0|1:40\t1|1:30\t0|0:45";
        let vcf_file = vcf_folder.join("chr1.vcf");
        let mut file = File::create(&vcf_file).unwrap();
        writeln!(file, "{}", vcf_content).unwrap();
        
        // Define config entries with at least two haplotypes per group
        let config_entries = vec![
            ConfigEntry {
                seqname: "chr1".to_string(),
                start: 1000,
                end: 2000,
                samples_unfiltered: {
                    let mut map = HashMap::new();
                    map.insert("SAMPLE1".to_string(), (0, 1));
                    map.insert("SAMPLE2".to_string(), (0, 1));
                    map.insert("SAMPLE3".to_string(), (0, 1));
                    map.insert("SAMPLE4".to_string(), (0, 1));
                    map
                },
                samples_filtered: {
                    let mut map = HashMap::new();
                    map.insert("SAMPLE1".to_string(), (0, 1));
                    map.insert("SAMPLE2".to_string(), (0, 1));
                    map.insert("SAMPLE3".to_string(), (0, 1));
                    map.insert("SAMPLE4".to_string(), (0, 1));
                    map
                },
            },
        ];
        
        let output_file = temp_dir.path().join("output.csv");
        let result = process_config_entries(&config_entries, vcf_folder.to_str().unwrap(), &output_file, 30);
        
        assert!(result.is_ok());
        assert!(output_file.exists());
        
        // Read and check the output file content
        let output_content = fs::read_to_string(&output_file).unwrap();
        let lines: Vec<&str> = output_content.lines().collect();
        assert_eq!(lines.len(), 2); // Header + 1 data line
        assert!(lines[0].starts_with("chr,region_start,region_end"));
        assert!(lines[1].starts_with("chr1,1000,2000"));
        
        // Further checks can be added to verify the exact content of the data line
    }

    #[test]
    fn test_find_vcf_file() {
        use std::fs::{self, File};
        use std::io::Write;
    
        // Create a temporary directory for testing
        let temp_dir = tempfile::tempdir().unwrap();
        let temp_path = temp_dir.path();
    
        // Create some test VCF files
        File::create(temp_path.join("chr1.vcf")).unwrap();
        File::create(temp_path.join("chr2.vcf.gz")).unwrap();
        File::create(temp_path.join("chr10.vcf")).unwrap();
    
        // Test finding existing VCF files
        let vcf1 = find_vcf_file(temp_path.to_str().unwrap(), "1").unwrap();
        assert!(vcf1.ends_with("chr1.vcf"));
    
        let vcf2 = find_vcf_file(temp_path.to_str().unwrap(), "2").unwrap();
        assert!(vcf2.ends_with("chr2.vcf.gz"));
    
        let vcf10 = find_vcf_file(temp_path.to_str().unwrap(), "10").unwrap();
        assert!(vcf10.ends_with("chr10.vcf"));
    
        // Test with non-existent chromosome
        assert!(matches!(
            find_vcf_file(temp_path.to_str().unwrap(), "3"),
            Err(VcfError::NoVcfFiles)
        ));
    
        // Test with non-existent directory
        assert!(find_vcf_file("/non/existent/path", "1").is_err());
    }

    #[test]
    fn test_open_vcf_reader() {
        let path = PathBuf::from("/non/existent/file.vcf");
        let result = open_vcf_reader(&path);
        assert!(result.is_err());

        // Create a temporary uncompressed VCF file
        let temp_file = tempfile::NamedTempFile::new().unwrap();
        let path = temp_file.path();
        let mut file = File::create(path).unwrap();
        writeln!(file, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1").unwrap();

        let reader = open_vcf_reader(&path);
        assert!(reader.is_ok());

        // Create a temporary gzipped VCF file
        let gzipped_file = tempfile::NamedTempFile::new().unwrap();
        let path = gzipped_file.path();
        let mut encoder = flate2::write::GzEncoder::new(File::create(path).unwrap(), flate2::Compression::default());
        writeln!(encoder, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1").unwrap();
        encoder.finish().unwrap();

        let reader = open_vcf_reader(&path);
        assert!(reader.is_ok());
    }

    #[test]
    fn test_gq_filtering() {
        let variant_line = "chr1\t1000\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:20\t0|1:40";
        let chr = "1";
        let start = 0;
        let end = 2000;
        let min_gq = 30;
        let mut missing_data_info = MissingDataInfo::default();
        let sample_names = vec!["Sample1".to_string(), "Sample2".to_string()];

        let result = parse_variant(
            variant_line,
            chr,
            start,
            end,
            &mut missing_data_info,
            &sample_names,
            min_gq,
        ).unwrap();

        // Variant should be None because one sample has GQ < min_gq
        assert!(result.is_none());

        // Variant with all GQ >= min_gq
        let valid_variant_line = "chr1\t1000\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:35\t0|1:40";
        let result = parse_variant(
            valid_variant_line,
            chr,
            start,
            end,
            &mut missing_data_info,
            &sample_names,
            min_gq,
        ).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_parse_config_file() {
        let config_content = "\
seqnames\tstart\tend\tPOS\torig_ID\tverdict\tcateg\tSample1\tSample2\n\
chr1\t1000\t2000\t.\t.\t.\t.\t0|1_lowconf\t1|1\n\
chr1\t3000\t4000\t.\t.\t.\t.\t0|0\t0|1\n";

        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file.as_file(), "{}", config_content).unwrap();

        let config_entries = parse_config_file(temp_file.path()).unwrap();

        assert_eq!(config_entries.len(), 2);

        // First entry
        let entry1 = &config_entries[0];
        assert_eq!(entry1.samples_unfiltered.len(), 2);
        assert_eq!(entry1.samples_filtered.len(), 2); // Both samples are included after filtering
        assert!(entry1.samples_unfiltered.contains_key("Sample1"));
        assert!(entry1.samples_unfiltered.contains_key("Sample2"));
        assert!(entry1.samples_filtered.contains_key("Sample1"));
        assert!(entry1.samples_filtered.contains_key("Sample2"));
        
        assert_eq!(entry1.seqname, "chr1");
        assert_eq!(entry1.start, 1000);
        assert_eq!(entry1.end, 2000);

        // Second entry
        let entry2 = &config_entries[1];
        assert_eq!(entry2.samples_unfiltered.len(), 2);
        assert_eq!(entry2.samples_filtered.len(), 2);
        assert!(entry2.samples_unfiltered.contains_key("Sample1"));
        assert!(entry2.samples_unfiltered.contains_key("Sample2"));
        assert!(entry2.samples_filtered.contains_key("Sample1"));
        assert!(entry2.samples_filtered.contains_key("Sample2"));
        
        assert_eq!(entry2.seqname, "chr1");
        assert_eq!(entry2.start, 3000);
        assert_eq!(entry2.end, 4000);
    }

    #[test]
    fn test_haplotype_processing() {
        let variants = vec![
            create_variant(1000, vec![Some(vec![0, 0]), Some(vec![0, 1]), Some(vec![1, 1])]),
            create_variant(2000, vec![Some(vec![0, 0]), Some(vec![0, 0]), Some(vec![0, 0])]),
            create_variant(3000, vec![Some(vec![0, 1]), Some(vec![1, 1]), Some(vec![0, 0])]),
        ];
        let sample_names = vec!["SAMPLE1".to_string(), "SAMPLE2".to_string(), "SAMPLE3".to_string()];
        let mut sample_filter = HashMap::new();
        sample_filter.insert("SAMPLE1".to_string(), (0, 1));
        sample_filter.insert("SAMPLE2".to_string(), (0, 1));
        sample_filter.insert("SAMPLE3".to_string(), (0, 1));
    
        // Process haplotype group 0
        let result_group0 = process_variants(&variants, &sample_names, 0, &sample_filter, 1000, 3000).unwrap();
    
        // Process haplotype group 1
        let result_group1 = process_variants(&variants, &sample_names, 1, &sample_filter, 1000, 3000).unwrap();
    
        // Assertions for group0
        // Allele frequency should be 0.0 since all haplotypes have allele 0
        assert!((result_group0.4 - 0.0).abs() < 1e-6);
    
        // Assertions for group1
        // Allele frequency should be 1.0 / 3.0 as there's one '1' allele out of three haplotypes
        assert!((result_group1.4 - (1.0 / 3.0)).abs() < 1e-6);
    
        // Additional Assertions
        assert_eq!(result_group0.3, 3); // num_haplotypes
        assert_eq!(result_group1.3, 3); // num_haplotypes
    
        // Validate segregating sites and diversity measures
        assert!(result_group0.0 > 0); // num_segsites
        assert!(result_group1.0 > 0); // num_segsites
    }
}
