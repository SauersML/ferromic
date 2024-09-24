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
    fn test_count_segregating_sites_with_variants() {
        let variants = vec![
            create_variant(1, vec![Some(vec![0, 0]), Some(vec![0, 1]), Some(vec![1, 1])]),
            create_variant(2, vec![Some(vec![0, 0]), Some(vec![0, 0]), Some(vec![0, 0])]),
            create_variant(3, vec![Some(vec![0, 1]), Some(vec![0, 1]), Some(vec![0, 1])]),
            create_variant(4, vec![Some(vec![0, 0]), Some(vec![1, 1]), Some(vec![0, 1])]),
        ];

        assert_eq!(count_segregating_sites(&variants), 3);
    }

    #[test]
    fn test_count_segregating_sites_no_variants() {
        assert_eq!(count_segregating_sites(&[]), 0);
    }

    #[test]
    fn test_count_segregating_sites_all_homozygous() {
        let homozygous_variants = vec![
            create_variant(1, vec![Some(vec![0, 0]), Some(vec![0, 0]), Some(vec![0, 0])]),
            create_variant(2, vec![Some(vec![1, 1]), Some(vec![1, 1]), Some(vec![1, 1])]),
        ];
        assert_eq!(count_segregating_sites(&homozygous_variants), 0);
    }

    #[test]
    fn test_count_segregating_sites_with_missing_data() {
        let missing_data_variants = vec![
            create_variant(1, vec![Some(vec![0, 0]), None, Some(vec![1, 1])]),
            create_variant(2, vec![Some(vec![0, 1]), Some(vec![0, 1]), None]),
        ];
        assert_eq!(count_segregating_sites(&missing_data_variants), 2);
    }

    #[test]
    fn test_extract_sample_id_standard_case() {
        assert_eq!(extract_sample_id("sample_123"), "123");
    }

    #[test]
    fn test_extract_sample_id_multiple_underscores() {
        assert_eq!(extract_sample_id("sample_with_multiple_underscores_456"), "456");
    }

    #[test]
    fn test_extract_sample_id_singlepart() {
        assert_eq!(extract_sample_id("singlepart"), "singlepart");
    }

    #[test]
    fn test_extract_sample_id_empty_string() {
        assert_eq!(extract_sample_id(""), "");
    }

    #[test]
    fn test_extract_sample_id_only_underscore() {
        assert_eq!(extract_sample_id("_"), "");
    }

    #[test]
    fn test_extract_sample_id_trailing_underscore() {
        assert_eq!(extract_sample_id("sample_"), "");
    }

    #[test]
    fn test_extract_sample_id_complex_names_eas() {
        assert_eq!(extract_sample_id("EAS_JPT_NA18939"), "NA18939");
    }

    #[test]
    fn test_extract_sample_id_complex_names_amr() {
        assert_eq!(extract_sample_id("AMR_PEL_HG02059"), "HG02059");
    }

    #[test]
    fn test_extract_sample_id_double_underscore() {
        assert_eq!(extract_sample_id("double__underscore"), "underscore");
    }

    #[test]
    fn test_extract_sample_id_triple_part_name() {
        assert_eq!(extract_sample_id("triple_part_name_789"), "789");
    }

    #[test]
    fn test_harmonic_single() {
        assert_eq!(harmonic(1), 1.0);
    }

    #[test]
    fn test_harmonic_two() {
        assert!((harmonic(2) - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_harmonic_three() {
        let expected = 1.0 + 0.5 + 1.0/3.0;
        assert!((harmonic(3) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_harmonic_ten() {
        let expected = 2.9289682539682538;
        assert!((harmonic(10) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_pairwise_differences_basic() {
        let variants = vec![
            create_variant(1000, vec![Some(vec![0, 0]), Some(vec![0, 1]), Some(vec![1, 1])]),
            create_variant(2000, vec![Some(vec![0, 0]), Some(vec![0, 0]), Some(vec![0, 1])]),
            create_variant(3000, vec![Some(vec![0, 1]), Some(vec![1, 1]), Some(vec![0, 0])]),
        ];

        let result = calculate_pairwise_differences(&variants, 3);

        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_calculate_pairwise_differences_pair_0_1() {
        let variants = vec![
            create_variant(1000, vec![Some(vec![0, 0]), Some(vec![0, 1]), Some(vec![1, 1])]),
            create_variant(2000, vec![Some(vec![0, 0]), Some(vec![0, 0]), Some(vec![0, 1])]),
            create_variant(3000, vec![Some(vec![0, 1]), Some(vec![1, 1]), Some(vec![0, 0])]),
        ];

        let result = calculate_pairwise_differences(&variants, 3);

        for &((i, j), count, ref positions) in &result {
            if (i, j) == (0, 1) {
                assert_eq!(count, 2);
                assert_eq!(positions, &vec![1000, 3000]);
            }
        }
    }

    #[test]
    fn test_calculate_pairwise_differences_pair_0_2() {
        let variants = vec![
            create_variant(1000, vec![Some(vec![0, 0]), Some(vec![0, 1]), Some(vec![1, 1])]),
            create_variant(2000, vec![Some(vec![0, 0]), Some(vec![0, 0]), Some(vec![0, 1])]),
            create_variant(3000, vec![Some(vec![0, 1]), Some(vec![1, 1]), Some(vec![0, 0])]),
        ];

        let result = calculate_pairwise_differences(&variants, 3);

        for &((i, j), count, ref positions) in &result {
            if (i, j) == (0, 2) {
                assert_eq!(count, 3);
                assert_eq!(positions, &vec![1000, 2000, 3000]);
            }
        }
    }

    #[test]
    fn test_calculate_pairwise_differences_with_missing_data() {
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
    fn test_calculate_watterson_theta_case1() {
        let epsilon = 1e-6;
        assert!((calculate_watterson_theta(10, 5, 1000) - 0.0048).abs() < epsilon);
    }

    #[test]
    fn test_calculate_watterson_theta_case2() {
        let epsilon = 1e-6;
        assert!((calculate_watterson_theta(5, 2, 1000) - 0.005).abs() < epsilon);
    }

    #[test]
    fn test_calculate_watterson_theta_large_values() {
        let epsilon = 1e-6;
        assert!((calculate_watterson_theta(100, 10, 1_000_000) - 0.00003534).abs() < epsilon);
    }

    #[test]
    fn test_calculate_watterson_theta_n1() {
        let theta_n1 = calculate_watterson_theta(100, 1, 1000);
        assert!(theta_n1.is_infinite());
    }

    #[test]
    fn test_calculate_watterson_theta_n0() {
        let theta_n0 = calculate_watterson_theta(0, 0, 1000);
        assert!(theta_n0.is_infinite());
    }

    #[test]
    fn test_calculate_watterson_theta_seq_zero() {
        let theta_seq_zero = calculate_watterson_theta(10, 5, 0);
        assert!(theta_seq_zero.is_infinite());
    }

    #[test]
    fn test_calculate_watterson_theta_pi_typical() {
        let epsilon = 1e-10;
        let pi = calculate_pi(15, 5, 1000);
        let expected = 15f64 / ((5 * (5 - 1) / 2) as f64 * 1000.0);
        assert!((pi - expected).abs() < epsilon);
    }

    #[test]
    fn test_calculate_watterson_theta_pi_no_differences() {
        let pi = calculate_pi(0, 5, 1000);
        assert_eq!(pi, 0.0);
    }

    #[test]
    fn test_calculate_watterson_theta_pi_min_sample_size() {
        let epsilon = 1e-10;
        let pi = calculate_pi(5, 2, 1000);
        let expected = 5f64 / ((2 * (2 - 1) / 2) as f64 * 1000.0);
        assert!((pi - expected).abs() < epsilon);
    }

    #[test]
    fn test_calculate_watterson_theta_pi_n1() {
        let pi_n1 = calculate_pi(100, 1, 1000);
        assert!(pi_n1.is_infinite());
    }

    #[test]
    fn test_calculate_watterson_theta_pi_seq_zero() {
        let pi_seq_zero = calculate_pi(100, 10, 0);
        assert!(pi_seq_zero.is_infinite());
    }

    #[test]
    fn test_calculate_watterson_theta_pi_large_values() {
        let epsilon = 1e-10;
        let pi = calculate_pi(10000, 100, 10000);
        let expected = 10000f64 / ((100 * (100 - 1) / 2) as f64 * 10000.0);
        assert!((pi - expected).abs() < epsilon);
    }

    #[test]
    fn test_calculate_pi_typical_values() {
        let epsilon = 1e-6;
        assert!((calculate_pi(15, 5, 1000) - 0.0015).abs() < epsilon);
    }

    #[test]
    fn test_calculate_pi_no_pairwise_differences() {
        assert_eq!(calculate_pi(0, 5, 1000), 0.0);
    }

    #[test]
    fn test_calculate_pi_large_pairwise_differences() {
        let epsilon = 1e-6;
        assert!((calculate_pi(10000, 100, 10000) - 0.00020202).abs() < epsilon);
    }

    #[test]
    fn test_calculate_pi_min_sample_size() {
        let epsilon = 1e-6;
        assert!((calculate_pi(5, 2, 1000) - 0.005).abs() < epsilon);
    }

    #[test]
    fn test_calculate_pi_very_large_sequence_length() {
        let epsilon = 1e-9;
        assert!((calculate_pi(1000, 10, 1_000_000) - 0.0000222222).abs() < epsilon);
    }

    #[test]
    fn test_calculate_pi_n1_infinite() {
        let pi_n1 = calculate_pi(100, 1, 1000);
        assert!(pi_n1.is_infinite());
    }

    #[test]
    fn test_calculate_pi_n0_infinite() {
        let pi_n0 = calculate_pi(0, 0, 1000);
        assert!(pi_n0.is_infinite());
    }

    #[test]
    fn test_parse_region_valid_small() {
        assert_eq!(parse_region("1-1000").unwrap(), (1, 1000));
    }

    #[test]
    fn test_parse_region_valid_large() {
        assert_eq!(parse_region("1000000-2000000").unwrap(), (1000000, 2000000));
    }

    #[test]
    fn test_parse_region_invalid_missing_end() {
        assert!(matches!(parse_region("1000"), Err(VcfError::InvalidRegion(_))));
    }

    #[test]
    fn test_parse_region_invalid_missing_start() {
        assert!(matches!(parse_region("1000-"), Err(VcfError::InvalidRegion(_))));
    }

    #[test]
    fn test_parse_region_invalid_negative_start() {
        assert!(matches!(parse_region("-1000"), Err(VcfError::InvalidRegion(_))));
    }

    #[test]
    fn test_parse_region_invalid_non_numeric_start() {
        assert!(matches!(parse_region("a-1000"), Err(VcfError::InvalidRegion(_))));
    }

    #[test]
    fn test_parse_region_invalid_non_numeric_end() {
        assert!(matches!(parse_region("1000-b"), Err(VcfError::InvalidRegion(_))));
    }

    #[test]
    fn test_parse_region_invalid_start_equals_end() {
        assert!(matches!(parse_region("1000-1000"), Err(VcfError::InvalidRegion(_))));
    }

    #[test]
    fn test_parse_region_invalid_start_greater_than_end() {
        assert!(matches!(parse_region("2000-1000"), Err(VcfError::InvalidRegion(_))));
    }

    #[test]
    fn test_validate_vcf_header_valid() {
        let valid_header = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1\tSAMPLE2";
        assert!(validate_vcf_header(valid_header).is_ok());
    }

    #[test]
    fn test_validate_vcf_header_invalid_missing_fields() {
        let invalid_header = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO";
        assert!(matches!(validate_vcf_header(invalid_header), Err(VcfError::InvalidVcfFormat(_))));
    }

    #[test]
    fn test_validate_vcf_header_invalid_order() {
        let invalid_order = "POS\t#CHROM\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT";
        assert!(matches!(validate_vcf_header(invalid_order), Err(VcfError::InvalidVcfFormat(_))));
    }

    #[test]
    fn test_parse_variant_valid_all_gq_above_threshold() {
        let sample_names = vec!["SAMPLE1".to_string(), "SAMPLE2".to_string(), "SAMPLE3".to_string()];
        let mut missing_data_info = MissingDataInfo::default();
        let min_gq = 30;

        let valid_line = "chr1\t1500\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:35\t0|1:40\t1|1:45";
        let result = parse_variant(valid_line, "1", 1, 2000, &mut missing_data_info, &sample_names, min_gq);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[test]
    fn test_parse_variant_one_gq_below_threshold() {
        let sample_names = vec!["SAMPLE1".to_string(), "SAMPLE2".to_string(), "SAMPLE3".to_string()];
        let mut missing_data_info = MissingDataInfo::default();
        let min_gq = 30;

        let invalid_gq_line = "chr1\t1000\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:35\t0|1:25\t1|1:45";
        let result = parse_variant(invalid_gq_line, "1", 1, 2000, &mut missing_data_info, &sample_names, min_gq);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_parse_variant_valid_variant_details() {
        let sample_names = vec!["SAMPLE1".to_string(), "SAMPLE2".to_string(), "SAMPLE3".to_string()];
        let mut missing_data_info = MissingDataInfo::default();
        let min_gq = 30;

        let valid_line = "chr1\t1000\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:35\t0|1:40\t1|1:45";
        let result = parse_variant(valid_line, "1", 1, 2000, &mut missing_data_info, &sample_names, min_gq);
        assert!(result.is_ok());
        if let Ok(Some(variant)) = result {
            assert_eq!(variant.position, 1000);
            assert_eq!(variant.genotypes, vec![Some(vec![0, 0]), Some(vec![0, 1]), Some(vec![1, 1])]);
        }
    }

    #[test]
    fn test_parse_variant_low_gq_variant() {
        let sample_names = vec!["SAMPLE1".to_string(), "SAMPLE2".to_string(), "SAMPLE3".to_string()];
        let mut missing_data_info = MissingDataInfo::default();
        let min_gq = 30;

        let low_gq_line = "chr1\t1000\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:35\t0|1:20\t1|1:45";
        let result = parse_variant(low_gq_line, "1", 1, 2000, &mut missing_data_info, &sample_names, min_gq);
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_parse_variant_out_of_range_region() {
        let sample_names = vec!["SAMPLE1".to_string(), "SAMPLE2".to_string(), "SAMPLE3".to_string()];
        let mut missing_data_info = MissingDataInfo::default();
        let min_gq = 30;

        let out_of_range = "chr1\t3000\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:35\t0|1:40\t1|1:45";
        assert!(parse_variant(out_of_range, "1", 1, 2000, &mut missing_data_info, &sample_names, min_gq).unwrap().is_none());
    }

    #[test]
    fn test_parse_variant_different_chromosome() {
        let sample_names = vec!["SAMPLE1".to_string(), "SAMPLE2".to_string(), "SAMPLE3".to_string()];
        let mut missing_data_info = MissingDataInfo::default();
        let min_gq = 30;

        let diff_chr = "chr2\t1000\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:35\t0|1:40\t1|1:45";
        assert!(parse_variant(diff_chr, "1", 1, 2000, &mut missing_data_info, &sample_names, min_gq).unwrap().is_none());
    }

    #[test]
    fn test_parse_variant_invalid_format() {
        let sample_names = vec!["SAMPLE1".to_string(), "SAMPLE2".to_string(), "SAMPLE3".to_string()];
        let mut missing_data_info = MissingDataInfo::default();
        let min_gq = 30;

        let invalid_format = "chr1\t1000\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:35"; // Only 10 fields, expecting 12 for 3 samples
        assert!(parse_variant(invalid_format, "1", 1, 2000, &mut missing_data_info, &sample_names, min_gq).is_err());
    }

    #[test]
    fn test_process_variants_with_invalid_haplotype_group() {
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

        let invalid_group = process_variants(&variants, &sample_names, 2, &sample_filter, 1000, 3000);
        assert!(invalid_group.is_err());
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
    }

    #[test]
    fn test_parse_config_file_entry2_details() {
        use std::io::Write;

        let config_content = "seqnames\tstart\tend\tPOS\torig_ID\tverdict\tcateg\tSAMPLE1\tSAMPLE2\n\
                              chr1\t1000\t2000\t1500\ttest_id\tpass\tinv\t0|1_lowconf\t1|1\n\
                              chr1\t3000\t4000\t.\t.\t.\t.\t0|0\t0|1\n";
        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file.as_file(), "{}", config_content).unwrap();

        let config_entries = parse_config_file(temp_file.path()).unwrap();
        assert_eq!(config_entries.len(), 2);

        // Second entry details
        let entry2 = &config_entries[1];
        assert_eq!(entry2.seqname, "chr1");
        assert_eq!(entry2.start, 3000);
        assert_eq!(entry2.end, 4000);
        assert_eq!(entry2.samples_unfiltered.len(), 2);
        assert_eq!(entry2.samples_filtered.len(), 2);
        assert!(entry2.samples_unfiltered.contains_key("SAMPLE1"));
        assert!(entry2.samples_unfiltered.contains_key("SAMPLE2"));
        assert!(entry2.samples_filtered.contains_key("SAMPLE1"));
        assert!(entry2.samples_filtered.contains_key("SAMPLE2"));
    }

    #[test]
    fn test_find_vcf_file_existing_vcfs() {
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
    }

    #[test]
    fn test_find_vcf_file_non_existent_chromosome() {
        use std::fs::File;

        // Create a temporary directory for testing
        let temp_dir = tempfile::tempdir().unwrap();
        let temp_path = temp_dir.path();

        // Create some test VCF files
        File::create(temp_path.join("chr1.vcf")).unwrap();
        File::create(temp_path.join("chr2.vcf.gz")).unwrap();
        File::create(temp_path.join("chr10.vcf")).unwrap();

        // Test with non-existent chromosome "3"
        let result = find_vcf_file(temp_path.to_str().unwrap(), "3");
        assert!(matches!(result, Err(VcfError::NoVcfFiles)));
    }

    #[test]
    fn test_find_vcf_file_non_existent_directory() {
        // Test with a non-existent directory path
        let result = find_vcf_file("/non/existent/path", "1");
        assert!(result.is_err());
    }

    #[test]
    fn test_open_vcf_reader_non_existent_file() {
        let path = PathBuf::from("/non/existent/file.vcf");
        let result = open_vcf_reader(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_open_vcf_reader_uncompressed_file() {
        // Create a temporary uncompressed VCF file
        let temp_file = tempfile::NamedTempFile::new().unwrap();
        let path = temp_file.path();
        let mut file = File::create(path).unwrap();
        writeln!(file, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1").unwrap();

        let reader = open_vcf_reader(&path);
        assert!(reader.is_ok());
    }

    #[test]
    fn test_open_vcf_reader_gzipped_file() {
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
    fn test_gq_filtering_low_gq_variant() {
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
    }

    #[test]
    fn test_gq_filtering_valid_variant() {
        let valid_variant_line = "chr1\t1000\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:35\t0|1:40";
        let chr = "1";
        let start = 0;
        let end = 2000;
        let min_gq = 30;
        let mut missing_data_info = MissingDataInfo::default();
        let sample_names = vec!["Sample1".to_string(), "Sample2".to_string()];

        let result = parse_variant(
            valid_variant_line,
            chr,
            start,
            end,
            &mut missing_data_info,
            &sample_names,
            min_gq,
        ).unwrap();

        // Variant should be Some because all samples have GQ >= min_gq
        assert!(result.is_some());

        if let Some(variant) = result {
            assert_eq!(variant.genotypes, vec![Some(vec![0, 0]), Some(vec![0, 1])]);
        }
    }


    /// Common setup for all group1 tests.
    fn setup_group1_test() -> (Vec<Variant>, Vec<String>, HashMap<String, (u8, u8)>) {
        let variants = vec![
            create_variant(1000, vec![Some(vec![0, 0]), Some(vec![0, 1]), Some(vec![1, 1])]),
            create_variant(2000, vec![Some(vec![0, 0]), Some(vec![0, 0]), Some(vec![0, 0])]),
            create_variant(3000, vec![Some(vec![0, 1]), Some(vec![1, 1]), Some(vec![0, 0])]),
        ];
        let sample_names = vec![
            "SAMPLE1".to_string(),
            "SAMPLE2".to_string(),
            "SAMPLE3".to_string(),
        ];
        let mut sample_filter = HashMap::new();
        sample_filter.insert("SAMPLE1".to_string(), (0, 1));
        sample_filter.insert("SAMPLE2".to_string(), (0, 1));
        sample_filter.insert("SAMPLE3".to_string(), (0, 1));

        (variants, sample_names, sample_filter)
    }

    #[test]
    fn test_group1_allele_frequency() {
        let (variants, sample_names, sample_filter) = setup_group1_test();

        let result_group1 = process_variants(
            &variants,
            &sample_names,
            1,
            &sample_filter,
            1000,
            3000,
        )
        .unwrap();

        // Allele frequency for group1 should be approximately 0.4444 (4/9)
        let expected_freq_group1 = 4.0 / 9.0;
        let allele_frequency_diff_group1 = (result_group1.4 - expected_freq_group1).abs();
        println!(
            "Allele frequency difference for Group 1: {}",
            allele_frequency_diff_group1
        );
        assert!(
            allele_frequency_diff_group1 < 1e-6,
            "Allele frequency for Group 1 is incorrect: expected {}, got {}",
            expected_freq_group1,
            result_group1.4
        );
    }

    #[test]
    fn test_group1_number_of_haplotypes() {
        let (variants, sample_names, sample_filter) = setup_group1_test();

        let result_group1 = process_variants(
            &variants,
            &sample_names,
            1,
            &sample_filter,
            1000,
            3000,
        )
        .unwrap();

        // Number of haplotypes for group1 should be 3
        let expected_num_hap_group1 = 3;
        println!(
            "Number of haplotypes for Group 1 (expected {}): {}",
            expected_num_hap_group1, result_group1.3
        );
        assert_eq!(
            result_group1.3, expected_num_hap_group1,
            "Number of haplotypes for Group 1 is incorrect: expected {}, got {}",
            expected_num_hap_group1, result_group1.3
        );
    }

    #[test]
    fn test_group1_segregating_sites() {
        let (variants, sample_names, sample_filter) = setup_group1_test();

        let result_group1 = process_variants(
            &variants,
            &sample_names,
            1,
            &sample_filter,
            1000,
            3000,
        )
        .unwrap();

        // Number of segregating sites for group1 should be 2
        // Segregating sites at positions 1000 and 3000
        let expected_segsites_group1 = 2;
        println!(
            "Number of segregating sites for Group 1 (expected {}): {}",
            expected_segsites_group1, result_group1.0
        );
        assert_eq!(
            result_group1.0, expected_segsites_group1,
            "Number of segregating sites for Group 1 is incorrect: expected {}, got {}",
            expected_segsites_group1, result_group1.0
        );
    }

    #[test]
    fn test_group1_watterson_theta() {
        let (variants, sample_names, sample_filter) = setup_group1_test();

        let result_group1 = process_variants(
            &variants,
            &sample_names,
            1,
            &sample_filter,
            1000,
            3000,
        )
        .unwrap();

        // Calculate expected Watterson's theta
        // seg_sites = 2, n = 3, seq_length = 3000 - 1000 +1 = 2001
        // theta = seg_sites / harmonic(n-1) / seq_length
        let harmonic = 1.0 / 1.0 + 1.0 / 2.0; // n-1 = 2
        let expected_w_theta = 2.0 / harmonic / 2001.0;

        let w_theta_diff = (result_group1.1 - expected_w_theta).abs();
        println!(
            "Watterson's theta difference for Group 1: {}",
            w_theta_diff
        );
        assert!(
            w_theta_diff < 1e-6,
            "Watterson's theta for Group 1 is incorrect: expected {}, got {}",
            expected_w_theta,
            result_group1.1
        );
    }

    #[test]
    fn test_group1_nucleotide_diversity_pi() {
        let (variants, sample_names, sample_filter) = setup_group1_test();

        let result_group1 = process_variants(
            &variants,
            &sample_names,
            1,
            &sample_filter,
            1000,
            3000,
        )
        .unwrap();

        // Calculate expected nucleotide diversity (pi)
        // pairwise differences: SAMPLE1 vs SAMPLE2: 1, SAMPLE1 vs SAMPLE3:1, SAMPLE2 vs SAMPLE3:2
        // total_pair_diff = 4
        // n =3, num_comparisons=3
        // pi = total_pair_diff / num_comparisons / seq_length = 4 /3 /2001 ≈ 0.000666222
        let expected_pi = 4.0 / 3.0 / 2001.0;

        let pi_diff = (result_group1.2 - expected_pi).abs();
        println!("Nucleotide diversity (pi) difference for Group 1: {}", pi_diff);
        assert!(
            pi_diff < 1e-6,
            "Nucleotide diversity (pi) for Group 1 is incorrect: expected {}, got {}",
            expected_pi,
            result_group1.2
        );
    }

    #[test]
    fn test_group1_filtered_allele_frequency() {
        let (variants, sample_names, sample_filter) = setup_group1_test();

        // Assuming filtering is based on exact genotype matches, but in this setup all genotypes are exact or have '_lowconf' which are excluded in filtered
        let result_group1 = process_variants(
            &variants,
            &sample_names,
            1,
            &sample_filter,
            1000,
            3000,
        )
        .unwrap();

        // Allele frequency after filtering should exclude "_lowconf" genotypes
        // From the test setup, variants do not have "_lowconf" in group1, so frequency remains the same
        let expected_freq_group1_filtered = 4.0 / 9.0;
        let allele_frequency_diff_group1_filtered =
            (result_group1.4 - expected_freq_group1_filtered).abs();
        println!(
            "Filtered allele frequency difference for Group 1: {}",
            allele_frequency_diff_group1_filtered
        );
        assert!(
            allele_frequency_diff_group1_filtered < 1e-6,
            "Filtered allele frequency for Group 1 is incorrect: expected {}, got {}",
            expected_freq_group1_filtered,
            result_group1.4
        );
    }

    #[test]
    fn test_group1_filtered_number_of_haplotypes() {
        let (variants, sample_names, sample_filter) = setup_group1_test();

        let result_group1 = process_variants(
            &variants,
            &sample_names,
            1,
            &sample_filter,
            1000,
            3000,
        )
        .unwrap();

        // Number of haplotypes after filtering should be same as before if no filtering applied
        let expected_num_hap_group1_filtered = 3;
        println!(
            "Filtered number of haplotypes for Group 1 (expected {}): {}",
            expected_num_hap_group1_filtered, result_group1.3
        );
        assert_eq!(
            result_group1.3, expected_num_hap_group1_filtered,
            "Filtered number of haplotypes for Group 1 is incorrect: expected {}, got {}",
            expected_num_hap_group1_filtered, result_group1.3
        );
    }

    #[test]
    fn test_group1_filtered_segregating_sites() {
        let (variants, sample_names, sample_filter) = setup_group1_test();

        let result_group1 = process_variants(
            &variants,
            &sample_names,
            1,
            &sample_filter,
            1000,
            3000,
        )
        .unwrap();

        // Number of segregating sites after filtering should be same as before if no filtering applied
        let expected_segsites_group1_filtered = 2;
        println!(
            "Filtered number of segregating sites for Group 1 (expected {}): {}",
            expected_segsites_group1_filtered, result_group1.0
        );
        assert_eq!(
            result_group1.0, expected_segsites_group1_filtered,
            "Filtered number of segregating sites for Group 1 is incorrect: expected {}, got {}",
            expected_segsites_group1_filtered, result_group1.0
        );
    }

    #[test]
    fn test_group1_filtered_watterson_theta() {
        let (variants, sample_names, sample_filter) = setup_group1_test();

        let result_group1 = process_variants(
            &variants,
            &sample_names,
            1,
            &sample_filter,
            1000,
            3000,
        )
        .unwrap();

        // Watterson's theta after filtering should be same as before if no filtering applied
        let harmonic = 1.0 / 1.0 + 1.0 / 2.0; // n-1 =2
        let expected_w_theta_filtered = 2.0 / harmonic / 2001.0;

        let w_theta_diff_filtered = (result_group1.1 - expected_w_theta_filtered).abs();
        println!(
            "Filtered Watterson's theta difference for Group 1: {}",
            w_theta_diff_filtered
        );
        assert!(
            w_theta_diff_filtered < 1e-6,
            "Filtered Watterson's theta for Group 1 is incorrect: expected {}, got {}",
            expected_w_theta_filtered,
            result_group1.1
        );
    }

    #[test]
    fn test_group1_filtered_nucleotide_diversity_pi() {
        let (variants, sample_names, sample_filter) = setup_group1_test();

        let result_group1 = process_variants(
            &variants,
            &sample_names,
            1,
            &sample_filter,
            1000,
            3000,
        )
        .unwrap();

        // Pi after filtering should be same as before if no filtering applied
        let expected_pi_filtered = 4.0 / 3.0 / 2001.0;
        let pi_diff_filtered = (result_group1.2 - expected_pi_filtered).abs();
        println!(
            "Filtered nucleotide diversity (pi) difference for Group 1: {}",
            pi_diff_filtered
        );
        assert!(
            pi_diff_filtered < 1e-6,
            "Filtered nucleotide diversity (pi) for Group 1 is incorrect: expected {}, got {}",
            expected_pi_filtered,
            result_group1.2
        );
    }

    #[test]
    fn test_group1_missing_data_handling() {
        // Define sample haplotype groupings as per TSV config
        // For haplotype group 1, assume:
        // SAMPLE1: hap1=1
        // SAMPLE2: hap1=1
        // SAMPLE3: hap1=0
        let sample_filter_unfiltered = HashMap::from([
            ("SAMPLE1".to_string(), (0, 1)),
            ("SAMPLE2".to_string(), (0, 1)),
            ("SAMPLE3".to_string(), (0, 1)),
        ]);
    
        // Define variants (for Watterson's theta and pi)
        let variants = vec![
            create_variant(1000, vec![Some(vec![0, 0]), Some(vec![0, 1]), Some(vec![1, 1])]),
            create_variant(2000, vec![Some(vec![0, 0]), None, Some(vec![0, 0])]), // Missing genotype for SAMPLE2
            create_variant(3000, vec![Some(vec![0, 1]), Some(vec![1, 1]), Some(vec![0, 0])]),
        ];
    
        let sample_names = vec![
            "SAMPLE1".to_string(),
            "SAMPLE2".to_string(),
            "SAMPLE3".to_string(),
        ];
    
        let result_group1 = process_variants(
            &variants,
            &sample_names,
            1,
            &sample_filter_unfiltered,
            1000,
            3000,
        )
        .unwrap();
    
        // Calculate allele frequency based on TSV config:
        // haplotype_group=1:
        // SAMPLE1 hap1=1
        // SAMPLE2 hap1=1
        // SAMPLE3 hap1=0
        // Number of '1's: 2
        // Total haplotypes: 3
        let expected_freq_group1 = 2.0 / 3.0;
        let allele_frequency_diff_group1 = (result_group1.4 - expected_freq_group1).abs();
        println!(
            "Allele frequency difference for Group 1 with missing data: {}",
            allele_frequency_diff_group1
        );
        assert!(
            allele_frequency_diff_group1 < 1e-6,
            "Allele frequency for Group 1 with missing data is incorrect: expected {}, got {}",
            expected_freq_group1,
            result_group1.4
        );
    
        // Additionally, verify Watterson's theta and pi
        // Expected segregating sites: 2 (positions 1000 and 3000)
        assert_eq!(result_group1.0, 2, "Number of segregating sites should be 2");
    
        // Pi calculation:
        // For haplotype_group=1:
        // Variants considered: 1000, 3000 (2000 excluded due to missing)
        // Pairwise differences:
        // (SAMPLE1 vs SAMPLE2): 1 vs 1 -> 0 differences
        // (SAMPLE1 vs SAMPLE3): 1 vs 0 -> 1 difference
        // (SAMPLE2 vs SAMPLE3): 1 vs 0 -> 1 difference
        // Total differences: 2
        // Number of comparisons: 3
        // Pi: 2 / 3 / 2001 ≈ 0.000333
        let expected_pi = 2.0 / 3.0 / 2001.0; // ≈0.000333
        let pi_diff = (result_group1.2 - expected_pi).abs();
        println!(
            "Pi difference for Group 1: expected ~{:.6}, got {:.6}",
            expected_pi, result_group1.2
        );
        assert!(
            pi_diff < 1e-6,
            "Pi for Group 1 is incorrect: expected ~{:.6}, got {:.6}",
            expected_pi,
            result_group1.2
        );
    }
    
    #[test]
    fn test_group1_zero_segregating_sites() {
        let variants = vec![
            create_variant(1000, vec![Some(vec![0, 0]), Some(vec![0, 0]), Some(vec![0, 0])]),
            create_variant(2000, vec![Some(vec![0, 0]), Some(vec![0, 0]), Some(vec![0, 0])]),
            create_variant(3000, vec![Some(vec![0, 0]), Some(vec![0, 0]), Some(vec![0, 0])]),
        ];
        let sample_names = vec![
            "SAMPLE1".to_string(),
            "SAMPLE2".to_string(),
            "SAMPLE3".to_string(),
        ];
        let mut sample_filter = HashMap::new();
        sample_filter.insert("SAMPLE1".to_string(), (0, 1));
        sample_filter.insert("SAMPLE2".to_string(), (0, 1));
        sample_filter.insert("SAMPLE3".to_string(), (0, 1));
    
        let result_group1 = process_variants(
            &variants,
            &sample_names,
            1,
            &sample_filter,
            1000,
            3000,
        )
        .unwrap();
    
        // Calculate allele frequency based on TSV config:
        // haplotype_group=1:
        // SAMPLE1 hap1=0
        // SAMPLE2 hap1=0
        // SAMPLE3 hap1=0
        // Number of '1's: 0
        // Total haplotypes: 3
        let expected_freq_group1 = 0.0;
        let allele_frequency_diff_group1 = (result_group1.4 - expected_freq_group1).abs();
        println!(
            "Allele frequency difference for Group 1 with zero '1's: {}",
            allele_frequency_diff_group1
        );
        assert!(
            allele_frequency_diff_group1 < 1e-6,
            "Allele frequency for Group 1 with zero '1's is incorrect: expected {}, got {}",
            expected_freq_group1,
            result_group1.4
        );
    
        // No segregating sites
        let expected_segsites_group1 = 0;
        println!(
            "Number of segregating sites for Group 1 with zero segsites (expected {}): {}",
            expected_segsites_group1, result_group1.0
        );
        assert_eq!(
            result_group1.0, expected_segsites_group1,
            "Number of segregating sites for Group 1 with zero segsites is incorrect: expected {}, got {}",
            expected_segsites_group1, result_group1.0
        );
    
        // Watterson's theta should be 0
        let expected_w_theta = 0.0;
        let w_theta_diff = (result_group1.1 - expected_w_theta).abs();
        println!(
            "Watterson's theta difference for Group 1 with zero segsites: {}",
            w_theta_diff
        );
        assert!(
            w_theta_diff < 1e-6,
            "Watterson's theta for Group 1 with zero segsites is incorrect: expected {}, got {}",
            expected_w_theta,
            result_group1.1
        );
    
        // Pi should be 0
        let expected_pi = 0.0;
        let pi_diff = (result_group1.2 - expected_pi).abs();
        println!(
            "Nucleotide diversity (pi) difference for Group 1 with zero segsites: {}",
            pi_diff
        );
        assert!(
            pi_diff < 1e-6,
            "Nucleotide diversity (pi) for Group 1 with zero segsites is incorrect: expected {}, got {}",
            expected_pi,
            result_group1.2
        );
    }


}
