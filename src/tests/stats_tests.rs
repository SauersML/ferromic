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

    #[test]
    fn test_haplotype_processing_group1_allele_frequency() {
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
    
        let result_group1 = process_variants(&variants, &sample_names, 1, &sample_filter, 1000, 3000).unwrap();
    
        // Print all the result values
        println!("Result for Group 1: {:?}", result_group1);
    
        // allele_frequency for group 1 should be 0.5 (3/6)
        let allele_frequency_diff = (result_group1.4 - 0.5).abs();
        println!("Allele frequency difference: {}", allele_frequency_diff);
        assert!(allele_frequency_diff < 1e-6);
    
        // Additional Assertions
        println!("Number of haplotypes (expected 3): {}", result_group1.3);
        assert_eq!(result_group1.3, 3); // num_haplotypes
    
        // Validate segregating sites and diversity measures
        println!("Number of segregating sites: {}", result_group1.0);
        assert!(result_group1.0 > 0); // num_segsites
        println!("Watterson's theta: {}", result_group1.1);
        assert!(result_group1.1 > 0.0); // w_theta
        println!("Nucleotide diversity (pi): {}", result_group1.2);
        assert!(result_group1.2 > 0.0); // pi
    }
    
    #[test]
    fn test_process_variants_with_invalid_haplotype_group_values() {
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
    
        // Group 0
        let result_group0 = process_variants(&variants, &sample_names, 0, &sample_filter, 1000, 3000).unwrap();
        println!("Result for Group 0: {:?}", result_group0);
    
        // Group 1
        let result_group1 = process_variants(&variants, &sample_names, 1, &sample_filter, 1000, 3000).unwrap();
        println!("Result for Group 1: {:?}", result_group1);
    
        let allele_frequency_diff_group0 = (result_group0.4 - 0.0).abs();
        println!("Allele frequency difference for Group 0: {}", allele_frequency_diff_group0);
        assert!(allele_frequency_diff_group0 < 1e-6); // allele_frequency for group 0
    
        let allele_frequency_diff_group1 = (result_group1.4 - 0.5).abs();
        println!("Allele frequency difference for Group 1: {}", allele_frequency_diff_group1);
        assert!(allele_frequency_diff_group1 < 1e-6); // allele_frequency for group 1 (3/6 = 0.5)
    
        // Additional Assertions
        println!("Number of haplotypes for Group 0 (expected 3): {}", result_group0.3);
        assert_eq!(result_group0.3, 3); // num_haplotypes
        println!("Number of haplotypes for Group 1 (expected 3): {}", result_group1.3);
        assert_eq!(result_group1.3, 3); // num_haplotypes
    
        // Validate segregating sites and diversity measures
        println!("Number of segregating sites for Group 0: {}", result_group0.0);
        assert!(result_group0.0 > 0); // num_segsites
        println!("Number of segregating sites for Group 1: {}", result_group1.0);
        assert!(result_group1.0 > 0); // num_segsites
    }

}
