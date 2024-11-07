#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::fs::File;
    use std::collections::HashMap;
    use std::io::Write;
    use std::sync::Arc;
    use parking_lot::Mutex;
    use std::path::PathBuf;

    // Helper function to create a Variant for testing
    fn create_variant(position: i64, genotypes: Vec<Option<Vec<u8>>>) -> Variant {
        Variant { position, genotypes }
    }

    // Helper function to create a Variant for testing with specific number of haplotypes
    fn create_variant_with_genotypes(position: i64, genotypes: Vec<Option<Vec<u8>>>) -> Variant {
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
        let mut _filtering_stats = FilteringStats::default();
        let min_gq = 30;
        let _mask: Option<&[(i64, i64)]> = None;
        let position_allele_map = Mutex::new(HashMap::new());

        let valid_line = "chr1\t1500\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:35\t0|1:40\t1|1:45";
        let allow_regions: Option<&HashMap<String, Vec<(i64, i64)>>> = None;
        let mask_regions: Option<&HashMap<String, Vec<(i64, i64)>>> = None;
        let result = parse_variant(
            valid_line,
            "1",
            1,
            2000,
            &mut missing_data_info,
            &sample_names,
            min_gq,
            &mut _filtering_stats,
            allow_regions,
            mask_regions,
            &position_allele_map,
        );

        assert!(result.is_ok());
        let some_variant = result.unwrap();
        assert!(some_variant.is_some());

        let (variant, is_valid) = some_variant.unwrap();
        assert!(is_valid);
        assert_eq!(variant.position, 1500);
        assert_eq!(variant.genotypes, vec![Some(vec![0, 0]), Some(vec![0, 1]), Some(vec![1, 1])]);
    }

    #[test]
    fn test_parse_variant_one_gq_below_threshold() {
        let sample_names = vec![
            "SAMPLE1".to_string(),
            "SAMPLE2".to_string(),
            "SAMPLE3".to_string(),
        ];
        let mut missing_data_info = MissingDataInfo::default();
        let mut filtering_stats = FilteringStats::default();
        let min_gq = 30;
        let _mask: Option<&[(i64, i64)]> = None;
        let position_allele_map = Mutex::new(HashMap::new());
    
        // Define the expected variant using the helper function
        let expected_variant = create_variant(
            1000,
            vec![
                Some(vec![0, 0]), // SAMPLE1: 0|0:35
                Some(vec![0, 1]), // SAMPLE2: 0|1:25
                Some(vec![1, 1]), // SAMPLE3: 1|1:45
            ],
        );
    
        // VCF line with one genotype below the GQ threshold
        let invalid_gq_line = "chr1\t1000\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:35\t0|1:25\t1|1:45";
        let result = parse_variant(
            invalid_gq_line,
            "1",
            1,
            2000,
            &mut missing_data_info,
            &sample_names,
            min_gq,
            &mut filtering_stats,
            None, 
            None,
            &position_allele_map,
        );
    
        // The function executed without errors
        assert!(result.is_ok());
    
        // Assert that the variant is returned but marked as invalid (filtered out)
        assert_eq!(result.unwrap(), Some((expected_variant, false)));
    }

    #[test]
    fn test_parse_variant_valid_variant_details() {
        let sample_names = vec!["SAMPLE1".to_string(), "SAMPLE2".to_string(), "SAMPLE3".to_string()];
        let mut missing_data_info = MissingDataInfo::default();  
        let mut _filtering_stats = FilteringStats::default();
        let min_gq = 30;
        let _mask: Option<&[(i64, i64)]> = None;
        let position_allele_map = Mutex::new(HashMap::new());

        let valid_line = "chr1\t1000\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:35\t0|1:40\t1|1:45";
        let allow_regions: Option<&HashMap<String, Vec<(i64, i64)>>> = None;
        let mask_regions: Option<&HashMap<String, Vec<(i64, i64)>>> = None;
        let result = parse_variant(
            valid_line,
            "1",
            1,
            2000,
            &mut missing_data_info,
            &sample_names,
            min_gq,
            &mut _filtering_stats,
            allow_regions,
            mask_regions,
            &position_allele_map,
        );

        assert!(result.is_ok());
        if let Some((variant, is_valid)) = result.unwrap() {
            assert!(is_valid);
            assert_eq!(variant.position, 1000);
            assert_eq!(variant.genotypes, vec![Some(vec![0, 0]), Some(vec![0, 1]), Some(vec![1, 1])]);
        } else {
            panic!("Expected Some variant, got None");
        }
    }

    #[test]
    fn test_parse_variant_low_gq_variant() {
        let sample_names = vec![
            "SAMPLE1".to_string(),
            "SAMPLE2".to_string(),
            "SAMPLE3".to_string(),
        ];
        let mut missing_data_info = MissingDataInfo::default();
        let mut filtering_stats = FilteringStats::default();
        let min_gq = 30;
        let _mask: Option<&[(i64, i64)]> = None;
        let position_allele_map = Mutex::new(HashMap::new());
    
        // Define the expected variant using the helper function
        let expected_variant = create_variant(
            1000,
            vec![
                Some(vec![0, 0]), // SAMPLE1: 0|0:35
                Some(vec![0, 1]), // SAMPLE2: 0|1:20 (below threshold)
                Some(vec![1, 1]), // SAMPLE3: 1|1:45
            ],
        );
    
        // VCF line with one genotype below the GQ threshold
        let low_gq_line = "chr1\t1000\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:35\t0|1:20\t1|1:45";
        let result = parse_variant(
            low_gq_line,
            "1",
            1,
            2000,
            &mut missing_data_info,
            &sample_names,
            min_gq,
            &mut filtering_stats,
            None, 
            None,
            &position_allele_map,
        );
    
        // the function executed without errors
        assert!(result.is_ok());
    
        // Assert that the variant is returned but marked as invalid (filtered out)
        assert_eq!(result.unwrap(), Some((expected_variant, false)));
    }

    #[test]
    fn test_parse_variant_out_of_range_region() {
        let sample_names = vec!["SAMPLE1".to_string(), "SAMPLE2".to_string(), "SAMPLE3".to_string()];
        let mut missing_data_info = MissingDataInfo::default();  
        let mut _filtering_stats = FilteringStats::default();
        let min_gq = 30;
        let _mask: Option<&[(i64, i64)]> = None;
        let position_allele_map = Mutex::new(HashMap::new());

        let out_of_range = "chr1\t3000\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:35\t0|1:40\t1|1:45";
        let allow_regions: Option<&HashMap<String, Vec<(i64, i64)>>> = None;
        let mask_regions: Option<&HashMap<String, Vec<(i64, i64)>>> = None;
        let result = parse_variant(
            out_of_range,
            "1",
            1,
            2000,
            &mut missing_data_info,
            &sample_names,
            min_gq,
            &mut _filtering_stats,
            allow_regions,
            mask_regions,
            &position_allele_map,
        );

        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_parse_variant_different_chromosome() {
        let sample_names = vec!["SAMPLE1".to_string(), "SAMPLE2".to_string(), "SAMPLE3".to_string()];
        let mut missing_data_info = MissingDataInfo::default();  
        let mut _filtering_stats = FilteringStats::default();
        let min_gq = 30;
        let _mask: Option<&[(i64, i64)]> = None;
        let position_allele_map = Mutex::new(HashMap::new());

        let diff_chr = "chr2\t1000\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:35\t0|1:40\t1|1:45";
        let allow_regions: Option<&HashMap<String, Vec<(i64, i64)>>> = None;
        let mask_regions: Option<&HashMap<String, Vec<(i64, i64)>>> = None;
        let result = parse_variant(
            diff_chr,
            "1",
            1,
            2000,
            &mut missing_data_info,
            &sample_names,
            min_gq,
            &mut _filtering_stats,
            allow_regions,
            mask_regions,
            &position_allele_map,
        );

        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_parse_variant_invalid_format() {
        let sample_names = vec!["SAMPLE1".to_string(), "SAMPLE2".to_string(), "SAMPLE3".to_string()];
        let mut missing_data_info = MissingDataInfo::default();  
        let mut _filtering_stats = FilteringStats::default();
        let min_gq = 30;
        let _mask: Option<&[(i64, i64)]> = None;
        let position_allele_map = Mutex::new(HashMap::new());

        let invalid_format = "chr1\t1000\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:35"; // Only 10 fields, expecting 12 for 3 samples
        assert!(parse_variant(
            invalid_format,
            "1",
            1,
            2000,
            &mut missing_data_info,
            &sample_names,
            min_gq,
            &mut _filtering_stats,
            None, 
            None,
            &position_allele_map,
        ).is_err());
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
        let adjusted_sequence_length: Option<i64> = None;
        let seqinfo_storage = Arc::new(Mutex::new(Vec::new()));
        let position_allele_map = Arc::new(Mutex::new(HashMap::new()));
        let chromosome = "1".to_string();

        // Read reference sequence
        let (fasta_file, cds_regions) = setup_test_data();
        let reference_sequence = read_reference_sequence(fasta_file.path(), "1", 1000, 3000)
            .expect("Failed to read reference sequence");

        let invalid_group = process_variants(
            &variants,
            &sample_names,
            2, // haplotype_group=2 (invalid, since only 0 and 1)
            &sample_filter,
            1000,
            3000,
            adjusted_sequence_length,
            Arc::clone(&seqinfo_storage),
            Arc::clone(&position_allele_map),
            chromosome.clone(),
            false, // is_filtered_set
            &reference_sequence,
            &cds_regions[..],
        );
        assert!(invalid_group.unwrap_or(None).is_none(), "Expected None for invalid haplotype group");
    }

    #[test]
    fn test_parse_config_file_with_noreads() {
        let config_content = "seqnames\tstart\tend\tPOS\torig_ID\tverdict\tcateg\tSAMPLE1\tSAMPLE2\n\
                              chr1\t1000\t2000\t1500\ttest_id\tpass\tinv\t0|1_lowconf\t1|1\n\
                              chr1\t3000\t4000\t.\t.\t.\t.\t0|0\t0|1\n";
        let path = NamedTempFile::new().expect("Failed to process variants");
        write!(path.as_file(), "{}", config_content).expect("Failed to process variants");

        let config_entries = parse_config_file(path.path()).expect("Failed to process variants");
        assert_eq!(config_entries.len(), 2);
    }

    #[test]
    fn test_find_vcf_file_existing_vcfs() {
        use std::fs::File;

        // Create a temporary directory for testing
        let temp_dir = tempfile::tempdir().expect("Failed to process variants");
        let temp_path = temp_dir.path();

        // Create some test VCF files
        File::create(temp_path.join("chr1.vcf")).expect("Failed to process variants");
        File::create(temp_path.join("chr2.vcf.gz")).expect("Failed to process variants");
        File::create(temp_path.join("chr10.vcf")).expect("Failed to process variants");

        // Test finding existing VCF files
        let vcf1 = find_vcf_file(temp_path.to_str().unwrap(), "1").expect("Failed to process variants");
        assert!(vcf1.ends_with("chr1.vcf"));

        let vcf2 = find_vcf_file(temp_path.to_str().unwrap(), "2").expect("Failed to process variants");
        assert!(vcf2.ends_with("chr2.vcf.gz"));

        let vcf10 = find_vcf_file(temp_path.to_str().unwrap(), "10").expect("Failed to process variants");
        assert!(vcf10.ends_with("chr10.vcf"));
    }

    #[test]
    fn test_find_vcf_file_non_existent_chromosome() {
        use std::fs::File;

        // Create a temporary directory for testing
        let temp_dir = tempfile::tempdir().expect("Failed to process variants");
        let temp_path = temp_dir.path();

        // Create some test VCF files
        File::create(temp_path.join("chr1.vcf")).expect("Failed to process variants");
        File::create(temp_path.join("chr2.vcf.gz")).expect("Failed to process variants");
        File::create(temp_path.join("chr10.vcf")).expect("Failed to process variants");

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
        let temp_file = NamedTempFile::new().expect("Failed to process variants");
        let path = temp_file.path();
        let mut file = File::create(path).expect("Failed to process variants");
        writeln!(file, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1").expect("Failed to process variants");

        let reader = open_vcf_reader(&path);
        assert!(reader.is_ok());
    }

    #[test]
    fn test_open_vcf_reader_gzipped_file() {
        // Create a temporary gzipped VCF file
        let gzipped_file = NamedTempFile::new().expect("Failed to process variants");
        let path = gzipped_file.path();
        let mut encoder = flate2::write::GzEncoder::new(File::create(path).unwrap(), flate2::Compression::default());
        writeln!(encoder, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1").expect("Failed to process variants");
        encoder.finish().expect("Failed to process variants");

        let reader = open_vcf_reader(&path);
        assert!(reader.is_ok());
    }

    #[test]
    fn test_gq_filtering_low_gq_variant() {
        let sample_names = vec![
            "SAMPLE1".to_string(),
            "SAMPLE2".to_string(),
        ];
        let mut missing_data_info = MissingDataInfo::default();
        let mut filtering_stats = FilteringStats::default();
        let min_gq = 30;
        let _mask: Option<&[(i64, i64)]> = None;
        let position_allele_map = Mutex::new(HashMap::new());
    
        // Define the expected variant using the helper function
        let expected_variant = create_variant(
            1000,
            vec![
                Some(vec![0, 0]), // SAMPLE1: 0|0:20 (below threshold)
                Some(vec![0, 1]), // SAMPLE2: 0|1:40
            ],
        );
    
        // VCF line with one genotype below the GQ threshold
        let variant_line = "chr1\t1000\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:20\t0|1:40";
        let result = parse_variant(
            variant_line,
            "1",
            1000,
            2000,
            &mut missing_data_info,
            &sample_names,
            min_gq,
            &mut filtering_stats,
            None, 
            None,
            &position_allele_map,
        );
    
        // the function executed without errors
        assert!(result.is_ok());
    
        // Assert that the variant is returned but marked as invalid (filtered out)
        assert_eq!(result.unwrap(), Some((expected_variant, false)));
    }

    #[test]
    fn test_gq_filtering_valid_variant() {
        let sample_names = vec!["Sample1".to_string(), "Sample2".to_string()];
        let mut missing_data_info = MissingDataInfo::default();  
        let mut _filtering_stats = FilteringStats::default();
        let min_gq = 30;
        let _mask: Option<&[(i64, i64)]> = None;
        let position_allele_map = Mutex::new(HashMap::new());

        let valid_variant_line = "chr1\t1000\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:35\t0|1:40";
        let result = parse_variant(
            valid_variant_line,
            "1",
            1000,
            2000,
            &mut missing_data_info,
            &sample_names,
            min_gq,
            &mut _filtering_stats,
            None, 
            None,
            &position_allele_map,
        ).expect("Failed to process variants");
    
        // Variant should be Some because all samples have GQ >= min_gq
        assert!(result.is_some());
    
        if let Some((variant, is_valid)) = result {
            assert!(is_valid);
            assert_eq!(variant.genotypes, vec![Some(vec![0, 0]), Some(vec![0, 1])]);
        } else {
            panic!("Expected Some variant, got None");
        }
    }
    
    use super::CdsRegion;
    
    fn setup_test_data() -> (NamedTempFile, Vec<CdsRegion>) {
        let mut fasta_file = NamedTempFile::new().expect("Failed to create temporary FASTA file");
        writeln!(fasta_file, ">1").expect("Failed to write FASTA header");
        let padding = "N".repeat(200);
        let main_seq = "ACGAGTACGTGACTT".repeat(1000);
        writeln!(fasta_file, "{}{}", padding, main_seq).expect("Failed to write FASTA sequence");
        fasta_file.flush().expect("Failed to flush temporary file");
    
        let cds_regions = vec![
            CdsRegion {
                start: 1000,
                end: 2000,
            },
            CdsRegion {
                start: 2500,
                end: 3000,
            },
            CdsRegion {
                start: 3200,
                end: 3300,
            },
            CdsRegion {
                start: 3400,
                end: 3450,
            },
            CdsRegion {
                start: 20,
                end: 30,
            },
            CdsRegion {
                start: 3800,
                end: 3810,
            },
        ];
    
        (fasta_file, cds_regions)
    }


    // Setup function for Group 1 tests
    fn setup_group1_test() -> (Vec<Variant>, Vec<String>, HashMap<String, (u8, u8)>) {
        // Define the sample names as they appear in the VCF.
        let sample_names = vec![
            "Sample1".to_string(),
            "Sample2".to_string(),
            "Sample3".to_string(),
        ];
    
        // Define the sample_filter for haplotype_group=1.
        // Each entry maps a sample to its (left_haplotype, right_haplotype) group assignments.
        // To achieve an allele frequency of 2/3 for Group 1, configure as follows:
        // - Sample1: left=0 (direct), right=1 (inversion)
        // - Sample2: left=1 (inversion), right=0 (direct)
        // - Sample3: left=0 (direct), right=1 (inversion)
        let sample_filter = HashMap::from([
            ("Sample1".to_string(), (0, 1)), // haplotype_group=1: 1
            ("Sample2".to_string(), (1, 0)), // haplotype_group=1: 0
            ("Sample3".to_string(), (0, 1)), // haplotype_group=1: 1
        ]);
    
        // Define the variants within the region 1000 to 3000.
        // Each variant includes:
        // - Position on the chromosome.
        // - Genotypes for each sample, represented as Option<Vec<u8>>:
        //     - `Some(vec![allele1, allele2])` for valid genotypes.
        //     - `None` for missing genotypes.
        let variants = vec![
            // Variant at position 1000
            create_variant(
                1000,
                vec![
                    Some(vec![0, 0]), // Sample1: 0|0
                    Some(vec![0, 1]), // Sample2: 0|1
                    Some(vec![1, 1]), // Sample3: 1|1
                ],
            ),
            // Variant at position 2000 (all genotypes are 0|0)
            create_variant(
                2000,
                vec![
                    Some(vec![0, 0]), // Sample1: 0|0
                    Some(vec![0, 0]), // Sample2: 0|0
                    Some(vec![0, 0]), // Sample3: 0|0
                ],
            ),
            // Variant at position 3000
            create_variant(
                3000,
                vec![
                    Some(vec![0, 1]), // Sample1: 0|1
                    Some(vec![1, 1]), // Sample2: 1|1
                    Some(vec![0, 0]), // Sample3: 0|0
                ],
            ),
        ];
    
        // Return the setup tuple containing variants, sample names, and sample_filter.
        (variants, sample_names, sample_filter)
    }

    // Setup function for global tests
    fn setup_global_test() -> (Vec<Variant>, Vec<String>, HashMap<String, (u8, u8)>) {
        // Define the sample names as they appear in the VCF.
        let sample_names = vec![
            "Sample1".to_string(),
            "Sample2".to_string(),
            "Sample3".to_string(),
        ];
    
        // Define the sample_filter for haplotype_group=1.
        // Each entry maps a sample to its (left_haplotype, right_haplotype) group assignments.
        // To achieve an allele frequency of 2/3 for Group 1, configure as follows:
        // - Sample1: left=0 (direct), right=1 (inversion)
        // - Sample2: left=1 (inversion), right=0 (direct)
        // - Sample3: left=0 (direct), right=1 (inversion)
        let sample_filter = HashMap::from([
            ("Sample1".to_string(), (0, 1)), // haplotype_group=1: 1
            ("Sample2".to_string(), (1, 0)), // haplotype_group=1: 0
            ("Sample3".to_string(), (0, 1)), // haplotype_group=1: 1
        ]);
    
        // Define the variants within the region 1000 to 3000.
        // Each variant includes:
        // - Position on the chromosome.
        // - Genotypes for each sample, represented as Option<Vec<u8>>:
        //     - `Some(vec![allele1, allele2])` for valid genotypes.
        //     - `None` for missing genotypes.
        let variants = vec![
            // Variant at position 1000
            create_variant(
                1000,
                vec![
                    Some(vec![0, 0]), // Sample1: 0|0
                    Some(vec![0, 1]), // Sample2: 0|1
                    Some(vec![1, 1]), // Sample3: 1|1
                ],
            ),
            // Variant at position 2000 (all genotypes are 0|0)
            create_variant(
                2000,
                vec![
                    Some(vec![0, 0]), // Sample1: 0|0
                    Some(vec![0, 0]), // Sample2: 0|0
                    Some(vec![0, 0]), // Sample3: 0|0
                ],
            ),
            // Variant at position 3000
            create_variant(
                3000,
                vec![
                    Some(vec![0, 1]), // Sample1: 0|1
                    Some(vec![1, 1]), // Sample2: 1|1
                    Some(vec![0, 0]), // Sample3: 0|0
                ],
            ),
        ];
    
        // Return the setup tuple containing variants, sample names, and sample_filter.
        (variants, sample_names, sample_filter)
    }

    #[test]
    fn test_allele_frequency() {
        let sample_names = vec![
            "SAMPLE1".to_string(),
            "SAMPLE2".to_string(),
            "SAMPLE3".to_string(),
        ];
        let mut sample_filter = HashMap::new();
        sample_filter.insert("SAMPLE1".to_string(), (0, 1));
        sample_filter.insert("SAMPLE2".to_string(), (0, 1));
        sample_filter.insert("SAMPLE3".to_string(), (0, 1));
        let adjusted_sequence_length: Option<i64> = Some(2001);
        let seqinfo_storage = Arc::new(Mutex::new(Vec::new()));
        let position_allele_map = Arc::new(Mutex::new(HashMap::new()));
        let chromosome = "1".to_string();
        let (fasta_file, cds_regions) = setup_test_data();
        let reference_sequence = read_reference_sequence(fasta_file.path(), "1", 1000, 3000)
            .expect("Failed to read reference sequence");
    
        // Define VCF lines as strings
        let vcf_lines = vec![
            "chr1\t1000\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:35\t0|1:40\t1|1:45",
            "chr1\t2000\t.\tA\tA\t.\tPASS\t.\tGT:GQ\t0|0:35\t0|0:40\t0|0:45",
            "chr1\t3000\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|1:35\t1|1:40\t0|0:45",
        ];
    
        // Parse each VCF line to populate `position_allele_map`
        for line in &vcf_lines {
            let mut missing_data_info = MissingDataInfo::default();
            let mut filtering_stats = FilteringStats::default();
            let result = parse_variant(
                line,
                "1",
                1000,
                3000,
                &mut missing_data_info,
                &sample_names,
                30,
                &mut filtering_stats,
                None,
                None,
                &position_allele_map,
            );
            assert!(result.is_ok());
        }
    
        // Now, process the variants
        let variants = vec![
            create_variant(1000, vec![Some(vec![0, 0]), Some(vec![0, 1]), Some(vec![1, 1])]),
            create_variant(2000, vec![Some(vec![0, 0]), Some(vec![0, 0]), Some(vec![0, 0])]),
            create_variant(3000, vec![Some(vec![0, 1]), Some(vec![1, 1]), Some(vec![0, 0])]),
        ];
    
        let result = process_variants(
            &variants,
            &sample_names,
            0, // haplotype_group is irrelevant now
            &sample_filter,
            1000,
            3000,
            adjusted_sequence_length,
            Arc::clone(&seqinfo_storage),
            Arc::clone(&position_allele_map),
            chromosome.clone(),
            false, // is_filtered_set
            &reference_sequence,
            &cds_regions[..],
        ).unwrap();
    
        // Calculate allele frequency globally
        let allele_frequency = calculate_inversion_allele_frequency(&sample_filter);
    
        // Calculate expected allele frequency
        let expected_freq = 0.5; // Based on test setup
        let allele_frequency_diff = (allele_frequency.unwrap_or(0.0) - expected_freq).abs();
        println!(
            "Allele frequency difference: {}",
            allele_frequency_diff
        );
        assert!(
            allele_frequency_diff < 1e-6,
            "Allele frequency is incorrect: expected {:.6}, got {:.6}",
            expected_freq,
            allele_frequency.unwrap_or(0.0)
        );
    
        // Verify segregating sites
        assert_eq!(result.unwrap().0, 2, "Number of segregating sites should be 2");
    }

    #[test]
    fn test_group1_number_of_haplotypes() {
        let (variants, sample_names, sample_filter) = setup_group1_test();
        let adjusted_sequence_length: Option<i64> = None;
        let seqinfo_storage = Arc::new(Mutex::new(Vec::new()));
        let position_allele_map = Arc::new(Mutex::new(HashMap::new()));
        let chromosome = "1".to_string();
        let (fasta_file, cds_regions) = setup_test_data();
        let reference_sequence = read_reference_sequence(fasta_file.path(), "1", 1000, 3000)
            .expect("Failed to read reference sequence");
    
        let _result_group1 = process_variants(
            &variants,
            &sample_names,
            1, // haplotype_group=1
            &sample_filter,
            1000,
            3000,
            adjusted_sequence_length,
            Arc::clone(&seqinfo_storage),
            Arc::clone(&position_allele_map),
            chromosome.clone(),
            false, // is_filtered_set
            &reference_sequence,
            &cds_regions[..],
        ).unwrap();
    
        let (_segsites, _w_theta, _pi, n_hap) = match _result_group1 {
            Some(data) => data,
            None => panic!("Expected Some variant data"),
        };
    
        // Number of haplotypes for group1 should be 3
        let expected_num_hap_group1 = 3;
        println!(
            "Number of haplotypes for Group 1 (expected {}): {}",
            expected_num_hap_group1, n_hap
        );
        assert_eq!(
            n_hap, expected_num_hap_group1,
            "Number of haplotypes for Group 1 is incorrect: expected {}, got {}",
            expected_num_hap_group1, n_hap
        );
    }

    #[test]
    fn test_group1_segregating_sites() {
        let (variants, sample_names, sample_filter) = setup_group1_test();
        let adjusted_sequence_length: Option<i64> = None;
        let _mask: Option<&[(i64, i64)]> = None;
        let mut _filtering_stats = FilteringStats::default();
        let seqinfo_storage = Arc::new(Mutex::new(Vec::new()));
        let position_allele_map = Arc::new(Mutex::new(HashMap::new()));
        let chromosome = "1".to_string();
        let (fasta_file, cds_regions) = setup_test_data();
        let reference_sequence = read_reference_sequence(fasta_file.path(), "1", 1000, 3000)
            .expect("Failed to read reference sequence");

        // Manually populate the position_allele_map
        {
            let mut pam = position_allele_map.lock();
            pam.insert(1000, ('A', 'T'));
            pam.insert(2000, ('A', 'A'));
            pam.insert(3000, ('A', 'T'));
        }

        let _result_group1 = process_variants(
            &variants,
            &sample_names,
            1, // haplotype_group=1
            &sample_filter,
            1000,
            3000,
            adjusted_sequence_length,
            Arc::clone(&seqinfo_storage),
            Arc::clone(&position_allele_map),
            chromosome.clone(),
            true, // is_filtered_set
            &reference_sequence,
            &cds_regions[..],
        ).unwrap();

        // Correctly unwrap the Option to access the inner tuple
        let (segsites, _w_theta, _pi, _n_hap) = match _result_group1 {
            Some(data) => data,
            None => panic!("Expected Some variant data"),
        };

        let expected_segsites_group1 = 2;
        println!(
            "Number of segregating sites for Group 1 (expected {}): {}",
            expected_segsites_group1, segsites
        );
        assert_eq!(
            segsites, expected_segsites_group1,
            "Number of segregating sites for Group 1 is incorrect: expected {}, got {}",
            expected_segsites_group1, segsites
        );
    }

    #[test]
    fn test_watterson_theta_exact_h4() {
       let variants = vec![
           create_variant(1000, vec![
               Some(vec![0, 1]), // Sample1: 0|1
               Some(vec![1, 0]), // Sample2: 1|0
           ]),
           create_variant(2000, vec![
               Some(vec![1, 1]), // Sample1: 1|1 
               Some(vec![0, 0]), // Sample2: 0|0
           ]),
       ];
       let sample_names = vec!["Sample1".to_string(), "Sample2".to_string()];
       let sample_filter = HashMap::from([
           ("Sample1".to_string(), (1, 1)), // Add both haplotypes to group 1
           ("Sample2".to_string(), (1, 1)), // Add both haplotypes to group 1
       ]);
       let position_allele_map = Arc::new(Mutex::new(HashMap::from([
           (1000, ('A', 'T')),
           (2000, ('A', 'T')),
       ])));
       let seqinfo_storage = Arc::new(Mutex::new(Vec::new()));
       let chromosome = "1".to_string();
       let (fasta_file, cds_regions) = setup_test_data();
       let reference_sequence = read_reference_sequence(fasta_file.path(), "1", 1000, 3000)
           .expect("Failed to read reference sequence");
    
       let result = process_variants(
           &variants,
           &sample_names,
           1,  // haplotype_group=1
           &sample_filter,
           1000,
           2000,
           Some(100),  // sequence_length=100 for some reason
           Arc::clone(&seqinfo_storage),
           Arc::clone(&position_allele_map),
           chromosome,
           false,
           &reference_sequence,
           &cds_regions[..],
       ).unwrap();
    
       let (segsites, w_theta, _pi, n_hap) = match result {
           Some(data) => data,
           None => panic!("Expected Some variant data"),
       };
    
       // n=4 haplotypes means we sum 1/1 + 1/2 + 1/3 = 11/6
       // theta = 2 / (11/6) / 100 = 12/11/100
       let expected_theta = 12.0/11.0/100.0;
       println!("Got {} segregating sites with {} haplotypes", segsites, n_hap);
       println!("Expected theta: {:.8}, Actual theta: {:.8}, Difference: {:.8}", 
                expected_theta, w_theta, (w_theta - expected_theta).abs());
       assert!((w_theta - expected_theta).abs() < 1e-10);
    }
    
    #[test]
    fn test_group1_watterson_theta() {
        let (variants, sample_names, sample_filter) = setup_group1_test();
        let adjusted_sequence_length: Option<i64> = None;
        let _mask: Option<&[(i64, i64)]> = None;
        let mut _filtering_stats = FilteringStats::default();
        let seqinfo_storage = Arc::new(Mutex::new(Vec::new()));
        let position_allele_map = Arc::new(Mutex::new(HashMap::new()));
        let chromosome = "1".to_string();
        let (fasta_file, cds_regions) = setup_test_data();
        let reference_sequence = read_reference_sequence(fasta_file.path(), "1", 1000, 3000)
            .expect("Failed to read reference sequence");
    
        let _result_group1 = process_variants(
            &variants,
            &sample_names,
            1, // haplotype_group=1
            &sample_filter,
            1000,
            3000,
            adjusted_sequence_length,
            Arc::clone(&seqinfo_storage),
            Arc::clone(&position_allele_map),
            chromosome.clone(),
            false, // is_filtered_set
            &reference_sequence,
            &cds_regions[..],
        ).unwrap();
    
        // Correctly unwrap the Option to access the inner tuple
        let (_segsites, w_theta, _pi, _n_hap) = match _result_group1 {
            Some(data) => data,
            None => panic!("Expected Some variant data"),
        };
    
        // Calculate expected Watterson's theta
        let harmonic_value = harmonic(2); // n-1 =2
        let expected_w_theta = 2.0 / harmonic_value / 2001.0;
    
        let w_theta_diff = (w_theta - expected_w_theta).abs();
        println!(
            "Watterson's theta difference for Group 1: {}",
            w_theta_diff
        );
        assert!(
            w_theta_diff < 1e-6,
            "Watterson's theta for Group 1 is incorrect: expected {:.6}, got {:.6}",
            expected_w_theta,
            w_theta
        );
    }

    #[test]
    fn test_global_allele_frequency_filtered() {
        let (variants, sample_names, sample_filter) = setup_global_test();
        let adjusted_sequence_length = Some(2001); // seq_length = 2001
        let seqinfo_storage = Arc::new(Mutex::new(Vec::new()));
        let position_allele_map = Arc::new(Mutex::new(HashMap::new()));
        let chromosome = "1".to_string();
        let (fasta_file, cds_regions) = setup_test_data();
        let reference_sequence = read_reference_sequence(fasta_file.path(), "1", 1000, 3000)
            .expect("Failed to read reference sequence");

        // Manually populate the position_allele_map
        {
            let mut pam = position_allele_map.lock();
            pam.insert(1000, ('A', 'T'));
            pam.insert(2000, ('A', 'A'));
            pam.insert(3000, ('A', 'T'));
        }

        let result = process_variants(
            &variants,
            &sample_names,
            0, // haplotype_group is irrelevant now
            &sample_filter,
            1000,
            3000,
            adjusted_sequence_length,
            Arc::clone(&seqinfo_storage),
            Arc::clone(&position_allele_map),
            chromosome.clone(),
            true, // is_filtered_set
            &reference_sequence,
            &cds_regions[..],
        ).unwrap();

        // Calculate global allele frequency
        let allele_frequency = calculate_inversion_allele_frequency(&sample_filter);

        // Define expected allele frequency based on test setup
        let expected_freq = 0.5; // Adjust based on actual calculation
        let allele_frequency_diff = (allele_frequency.unwrap_or(0.0) - expected_freq).abs();
        println!(
            "Filtered global allele frequency difference: {}",
            allele_frequency_diff
        );
        assert!(
            allele_frequency_diff < 1e-6,
            "Filtered global allele frequency is incorrect: expected {:.6}, got {:.6}",
            expected_freq,
            allele_frequency.unwrap_or(0.0)
        );

        // Number of segregating sites
        assert_eq!(result.unwrap().0, 2, "Number of segregating sites should be 2");
    }

    #[test]
    fn test_group1_filtered_number_of_haplotypes() {
        let (variants, sample_names, sample_filter) = setup_group1_test();
        let adjusted_sequence_length = Some(2001); // seq_length = 2001
        let _mask: Option<&[(i64, i64)]> = None;
        let seqinfo_storage = Arc::new(Mutex::new(Vec::new()));
        let position_allele_map = Arc::new(Mutex::new(HashMap::new()));
        let chromosome = "1".to_string();
        let (fasta_file, cds_regions) = setup_test_data();
        let reference_sequence = read_reference_sequence(fasta_file.path(), "1", 1000, 3000)
            .expect("Failed to read reference sequence");
    
        let _result_group1 = process_variants(
            &variants,
            &sample_names,
            1, // haplotype_group=1
            &sample_filter,
            1000,
            3000,
            adjusted_sequence_length,
            Arc::clone(&seqinfo_storage),
            Arc::clone(&position_allele_map),
            chromosome.clone(),
            true, // is_filtered_set
            &reference_sequence,
            &cds_regions[..],
        ).unwrap();
    
        let (_segsites, _w_theta, _pi, n_hap) = match _result_group1 {
            Some(data) => data,
            None => panic!("Expected Some variant data"),
        };
    
        // Number of haplotypes after filtering should be same as before if no filtering applied
        let expected_num_hap_group1_filtered = 3;
        println!(
            "Filtered number of haplotypes for Group 1 (expected {}): {}",
            expected_num_hap_group1_filtered, n_hap
        );
        assert_eq!(
            n_hap, expected_num_hap_group1_filtered,
            "Filtered number of haplotypes for Group 1 is incorrect: expected {}, got {}",
            expected_num_hap_group1_filtered, n_hap
        );
    }

    #[test]
    fn test_group1_filtered_segregating_sites() {
        let (variants, sample_names, sample_filter) = setup_group1_test();
        let adjusted_sequence_length = Some(2001); // seq_length = 2001
        let _mask: Option<&[(i64, i64)]> = None;
        let mut _filtering_stats = FilteringStats::default();
        let seqinfo_storage = Arc::new(Mutex::new(Vec::new()));
        let position_allele_map = Arc::new(Mutex::new(HashMap::new()));
        let chromosome = "1".to_string();
        let (fasta_file, cds_regions) = setup_test_data();
        let reference_sequence = read_reference_sequence(fasta_file.path(), "1", 1000, 3000)
            .expect("Failed to read reference sequence");
    
        let _result_group1 = process_variants(
            &variants,
            &sample_names,
            1, // haplotype_group=1
            &sample_filter,
            1000,
            3000,
            adjusted_sequence_length,
            Arc::clone(&seqinfo_storage),
            Arc::clone(&position_allele_map),
            chromosome.clone(),
            true, // is_filtered_set
            &reference_sequence,
            &cds_regions[..],
        ).unwrap();
    
        // Correctly unwrap the Option to access the inner tuple
        let (segsites, _w_theta, _pi, _n_hap) = match _result_group1 {
            Some(data) => data,
            None => panic!("Expected Some variant data"),
        };
    
        let expected_segsites_group1_filtered = 2;
        println!(
            "Filtered number of segregating sites for Group 1 (expected {}): {}",
            expected_segsites_group1_filtered, segsites
        );
        assert_eq!(
            segsites, expected_segsites_group1_filtered,
            "Filtered number of segregating sites for Group 1 is incorrect: expected {}, got {}",
            expected_segsites_group1_filtered, segsites
        );
    }

    #[test]
    fn test_group1_filtered_watterson_theta() {
        let (variants, sample_names, sample_filter) = setup_group1_test();
        let adjusted_sequence_length = Some(2001); // seq_length = 2001
        let _mask: Option<&[(i64, i64)]> = None;
        let mut _filtering_stats = FilteringStats::default();
        let seqinfo_storage = Arc::new(Mutex::new(Vec::new()));
        let position_allele_map = Arc::new(Mutex::new(HashMap::new()));
        let chromosome = "1".to_string();
        let (fasta_file, cds_regions) = setup_test_data();
        let reference_sequence = read_reference_sequence(fasta_file.path(), "1", 1000, 3000)
            .expect("Failed to read reference sequence");
    
        let _result_group1 = process_variants(
            &variants,
            &sample_names,
            1, // haplotype_group=1
            &sample_filter,
            1000,
            3000,
            adjusted_sequence_length,
            Arc::clone(&seqinfo_storage),
            Arc::clone(&position_allele_map),
            chromosome.clone(),
            true, // is_filtered_set
            &reference_sequence,
            &cds_regions[..],
        ).unwrap();
    
        // Correctly unwrap the Option to access the inner tuple
        let (_segsites, w_theta, _pi, _n_hap) = match _result_group1 {
            Some(data) => data,
            None => panic!("Expected Some variant data"),
        };
    
        // Calculate expected Watterson's theta after filtering
        let harmonic_value = harmonic(2); // n-1 =2
        let expected_w_theta_filtered = 2.0 / harmonic_value / 2001.0;
    
        let w_theta_diff_filtered = (w_theta - expected_w_theta_filtered).abs();
        println!(
            "Filtered Watterson's theta difference for Group 1: {}",
            w_theta_diff_filtered
        );
        assert!(
            w_theta_diff_filtered < 1e-6,
            "Filtered Watterson's theta for Group 1 is incorrect: expected {:.6}, got {:.6}",
            expected_w_theta_filtered,
            w_theta
        );
    }

    fn setup_group1_missing_data_test() -> (
        Vec<Variant>,
        Vec<String>,
        HashMap<String, (u8, u8)>,
    ) {
        // Define sample haplotype groupings as per TSV config
        // For haplotype group 1:
        // SAMPLE1: hap1=1
        // SAMPLE2: hap1=0
        // SAMPLE3: hap1=0
        let sample_filter_unfiltered = HashMap::from([
            ("Sample1".to_string(), (0, 1)),
            ("Sample2".to_string(), (0, 1)),
            ("Sample3".to_string(), (0, 0)),
        ]);
    
        // Define variants (for Watterson's theta and pi)
        let variants = vec![
            create_variant(
                1000,
                vec![
                    Some(vec![0, 0]), // Sample1
                    Some(vec![0, 1]), // Sample2
                    Some(vec![1, 1]), // Sample3
                ],
            ),
            create_variant(
                2000,
                vec![
                    Some(vec![0, 0]), // Sample1
                    None,              // Sample2 (missing genotype)
                    Some(vec![0, 0]), // Sample3
                ],
            ), // Missing genotype for Sample2
            create_variant(
                3000,
                vec![
                    Some(vec![0, 1]), // Sample1
                    Some(vec![1, 1]), // Sample2
                    Some(vec![0, 0]), // Sample3
                ],
            ),
        ];
    
        let sample_names = vec![
            "Sample1".to_string(),
            "Sample2".to_string(),
            "Sample3".to_string(),
        ];
    
        (variants, sample_names, sample_filter_unfiltered)
    }

    #[test]
    fn test_group1_missing_data_allele_frequency() {
        let (variants, sample_names, sample_filter_unfiltered) = setup_group1_missing_data_test();
        let seqinfo_storage = Arc::new(Mutex::new(Vec::new()));
        let position_allele_map = Arc::new(Mutex::new(HashMap::new()));
        let chromosome = "1".to_string();
        let (fasta_file, cds_regions) = setup_test_data();
        let reference_sequence = read_reference_sequence(fasta_file.path(), "1", 1000, 3000)
            .expect("Failed to read reference sequence");
    
        // Process variants for haplotype_group=1 (Group 1)
        let _result_group1 = process_variants(
            &variants,
            &sample_names,
            1, // haplotype_group=1
            &sample_filter_unfiltered,
            1000,
            3000,
            None,
            Arc::clone(&seqinfo_storage),
            Arc::clone(&position_allele_map),
            chromosome.clone(),
            true, // is_filtered_set
            &reference_sequence,
            &cds_regions[..],
        ).expect("Failed to process variants");
    
        // Calculate global allele frequency using the revised function (no haplotype_group parameter)
        let allele_frequency_global = calculate_inversion_allele_frequency(&sample_filter_unfiltered);
    
        // Calculate expected global allele frequency based on all haplotypes:
        // SAMPLE1: hap1=1, hap2=0
        // SAMPLE2: hap1=0, hap2=1
        // SAMPLE3: hap1=0, hap2=0
        // Total '1's: 2 (Sample1 hap1 and Sample2 hap2)
        // Total haplotypes: 3 samples * 2 haplotypes each = 6
        let expected_freq_global = 2.0 / 6.0; // 0.333333
        let allele_frequency_diff_global = (allele_frequency_global.unwrap_or(0.0) - expected_freq_global).abs();
        println!(
            "Global allele frequency difference: {}",
            allele_frequency_diff_global
        );
        assert!(
            allele_frequency_diff_global < 1e-6,
            "Global allele frequency is incorrect: expected {:.6}, got {:.6}",
            expected_freq_global,
            allele_frequency_global.unwrap_or(0.0)
        );
    }
}
