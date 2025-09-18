#[cfg(test)]
mod tests {
    use tempfile::NamedTempFile;
    use std::fs::File;
    use std::collections::HashMap;
    use std::io::Write;
    use std::sync::Arc;
    use parking_lot::Mutex;
    use std::path::PathBuf;

    use crate::transcripts::CdsRegion;
    use crate::parse::{parse_region, validate_vcf_header, read_reference_sequence, parse_config_file, find_vcf_file, open_vcf_reader};
    use crate::process::{MissingDataInfo, FilteringStats, process_config_entries, process_variants, process_variant, ZeroBasedHalfOpen, Variant, VcfError, HaplotypeSide, Args};
    use crate::stats::{count_segregating_sites, calculate_pairwise_differences, calculate_watterson_theta, calculate_pi, harmonic, calculate_inversion_allele_frequency};

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
        use crate::stats::core_sample_id;
        assert_eq!(core_sample_id("NA12878_L"), "NA12878");
        assert_eq!(core_sample_id("NA12878_R"), "NA12878");
    }

    #[test]
    fn test_extract_sample_id_multiple_underscores() {
        use crate::stats::core_sample_id;
        assert_eq!(core_sample_id("SAMPLE_01_L"), "SAMPLE_01");
        assert_eq!(core_sample_id("SAMPLE_01_R"), "SAMPLE_01");
    }

    #[test]
    fn test_extract_sample_id_singlepart() {
        use crate::stats::core_sample_id;
        assert_eq!(core_sample_id("SAMPLE"), "SAMPLE");
        assert_eq!(core_sample_id("NoSuffix"), "NoSuffix");
    }

    #[test]
    fn test_extract_sample_id_empty_string() {
        use crate::stats::core_sample_id;
        assert_eq!(core_sample_id(""), "");
    }

    #[test]
    fn test_extract_sample_id_only_underscore() {
        use crate::stats::core_sample_id;
        assert_eq!(core_sample_id("_"), "_");
        assert_eq!(core_sample_id("_L"), "");
        assert_eq!(core_sample_id("_R"), "");
    }

    #[test]
    fn test_extract_sample_id_trailing_underscore() {
        use crate::stats::core_sample_id;
        assert_eq!(core_sample_id("SAMPLE_"), "SAMPLE_");
        assert_eq!(core_sample_id("SAMPLE__L"), "SAMPLE_");
        assert_eq!(core_sample_id("SAMPLE__R"), "SAMPLE_");
    }

    #[test]
    fn test_extract_sample_id_complex_names_eas() {
        use crate::stats::core_sample_id;
        assert_eq!(core_sample_id("HG00096_EAS_L"), "HG00096_EAS");
        assert_eq!(core_sample_id("HG00096_EAS_R"), "HG00096_EAS");
    }

    #[test]
    fn test_extract_sample_id_complex_names_amr() {
        use crate::stats::core_sample_id;
        assert_eq!(core_sample_id("NA19625_AMR_L"), "NA19625_AMR");
        assert_eq!(core_sample_id("NA19625_AMR_R"), "NA19625_AMR");
    }

    #[test]
    fn test_extract_sample_id_double_underscore() {
        use crate::stats::core_sample_id;
        assert_eq!(core_sample_id("SAMPLE__L"), "SAMPLE_");
        assert_eq!(core_sample_id("SAMPLE__R"), "SAMPLE_");
    }

    #[test]
    fn test_extract_sample_id_triple_part_name() {
        use crate::stats::core_sample_id;
        assert_eq!(core_sample_id("PART_ONE_TWO_L"), "PART_ONE_TWO");
        assert_eq!(core_sample_id("PART_ONE_TWO_R"), "PART_ONE_TWO");
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

        for &((i, j), difference_count, comparable_site_count) in &result {
            if (i, j) == (0, 1) {
                assert_eq!(difference_count, 4); // Per-haplotype comparison: 2 haplotypes × 2 haplotypes = 4 comparisons
                // comparable_site_count should be > 0
                assert!(comparable_site_count > 0);
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

        for &((i, j), difference_count, comparable_site_count) in &result {
            if (i, j) == (0, 2) {
                assert_eq!(difference_count, 8); // Per-haplotype comparison across 3 variants
                // comparable_site_count should be > 0
                assert!(comparable_site_count > 0);
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
        assert!(theta_n0.is_nan()); // n=0, seg_sites=0 should return NaN
    }

    #[test]
    fn test_calculate_watterson_theta_seq_zero() {
        let theta_seq_zero = calculate_watterson_theta(10, 5, 0);
        assert!(theta_seq_zero.is_infinite());
    }

    #[test]
    fn test_calculate_watterson_theta_pi_typical() {
        let variants = vec![
            create_variant(100, vec![Some(vec![0, 1]), Some(vec![1, 0])]),
            create_variant(200, vec![Some(vec![0, 0]), Some(vec![1, 1])]),
        ];
        let haplotypes = vec![(0, HaplotypeSide::Left), (0, HaplotypeSide::Right), (1, HaplotypeSide::Left), (1, HaplotypeSide::Right)];
        let pi = calculate_pi(&variants, &haplotypes, 1000);
        assert!(pi > 0.0);
    }

    #[test]
    fn test_calculate_watterson_theta_pi_no_differences() {
        let variants = vec![
            create_variant(100, vec![Some(vec![0, 0]), Some(vec![0, 0])]),
            create_variant(200, vec![Some(vec![1, 1]), Some(vec![1, 1])]),
        ];
        let haplotypes = vec![(0, HaplotypeSide::Left), (0, HaplotypeSide::Right), (1, HaplotypeSide::Left), (1, HaplotypeSide::Right)];
        let pi = calculate_pi(&variants, &haplotypes, 1000);
        assert_eq!(pi, 0.0);
    }

    #[test]
    fn test_calculate_watterson_theta_pi_min_sample_size() {
        let variants = vec![create_variant(100, vec![Some(vec![0, 1])])];
        let haplotypes = vec![(0, HaplotypeSide::Left), (0, HaplotypeSide::Right)];
        let pi = calculate_pi(&variants, &haplotypes, 1000);
        assert!(pi >= 0.0);
    }

    #[test]
    fn test_calculate_watterson_theta_pi_n1() {
        let variants = vec![create_variant(100, vec![Some(vec![0, 1])])];
        let haplotypes = vec![(0, HaplotypeSide::Left)];
        let pi = calculate_pi(&variants, &haplotypes, 1000);
        assert_eq!(pi, 0.0); // Single haplotype has no diversity
    }

    #[test]
    fn test_calculate_watterson_theta_pi_seq_zero() {
        let variants = vec![create_variant(100, vec![Some(vec![0, 1]), Some(vec![1, 0])])];
        let haplotypes = vec![(0, HaplotypeSide::Left), (0, HaplotypeSide::Right), (1, HaplotypeSide::Left), (1, HaplotypeSide::Right)];
        let pi = calculate_pi(&variants, &haplotypes, 0);
        assert!(pi.is_infinite() || pi.is_nan()); // Division by zero sequence length
    }

    #[test]
    fn test_calculate_watterson_theta_pi_large_values() {
        let mut variants = Vec::new();
        for i in 0..100 {
            variants.push(create_variant(i * 10, vec![Some(vec![0, 1]), Some(vec![1, 0])]));
        }
        let haplotypes = vec![(0, HaplotypeSide::Left), (0, HaplotypeSide::Right), (1, HaplotypeSide::Left), (1, HaplotypeSide::Right)];
        let pi = calculate_pi(&variants, &haplotypes, 1000000);
        assert!(pi > 0.0 && pi < 1.0);
    }

    #[test]
    fn test_calculate_pi_typical_values() {
        // Create test variants with some differences
        let variants = vec![
            create_variant(100, vec![Some(vec![0, 1]), Some(vec![1, 0])]), // Different
            create_variant(200, vec![Some(vec![0, 0]), Some(vec![1, 1])]), // Different
        ];
        
        // Create haplotype group with 2 samples (4 haplotypes total)
        let haplotypes = vec![(0, HaplotypeSide::Left), (0, HaplotypeSide::Right), (1, HaplotypeSide::Left), (1, HaplotypeSide::Right)];
        
        let pi = calculate_pi(&variants, &haplotypes, 1000);
        assert!(pi > 0.0); // Should have some diversity
    }

    #[test]
    fn test_calculate_pi_no_pairwise_differences() {
        // Create test variants with no differences (all same)
        let variants = vec![
            create_variant(100, vec![Some(vec![0, 0]), Some(vec![0, 0])]), // Same
            create_variant(200, vec![Some(vec![1, 1]), Some(vec![1, 1])]), // Same
        ];
        
        // Create haplotype group
        let haplotypes = vec![(0, HaplotypeSide::Left), (0, HaplotypeSide::Right), (1, HaplotypeSide::Left), (1, HaplotypeSide::Right)];
        
        let pi = calculate_pi(&variants, &haplotypes, 1000);
        assert_eq!(pi, 0.0); // No diversity
    }

    #[test]
    fn test_calculate_pi_large_pairwise_differences() {
        let mut variants = Vec::new();
        for i in 0..50 {
            variants.push(create_variant(i * 20, vec![Some(vec![0, 1]), Some(vec![1, 0]), Some(vec![1, 1])]));
        }
        let haplotypes = vec![(0, HaplotypeSide::Left), (0, HaplotypeSide::Right), (1, HaplotypeSide::Left), (1, HaplotypeSide::Right), (2, HaplotypeSide::Left), (2, HaplotypeSide::Right)];
        let pi = calculate_pi(&variants, &haplotypes, 10000);
        assert!(pi > 0.0);
    }

    #[test]
    fn test_calculate_pi_min_sample_size() {
        let variants = vec![create_variant(100, vec![Some(vec![0, 1])])];
        let haplotypes = vec![(0, HaplotypeSide::Left), (0, HaplotypeSide::Right)];
        let pi = calculate_pi(&variants, &haplotypes, 1000);
        assert!(pi >= 0.0);
    }

    #[test]
    fn test_calculate_pi_very_large_sequence_length() {
        let variants = vec![create_variant(100, vec![Some(vec![0, 1]), Some(vec![1, 0])])];
        let haplotypes = vec![(0, HaplotypeSide::Left), (0, HaplotypeSide::Right), (1, HaplotypeSide::Left), (1, HaplotypeSide::Right)];
        let pi = calculate_pi(&variants, &haplotypes, 1_000_000_000);
        assert!(pi > 0.0 && pi < 0.001); // Very small due to large sequence length
    }

    #[test]
    fn test_calculate_pi_n1_infinite() {
        let variants = vec![create_variant(100, vec![Some(vec![0, 1])])];
        let haplotypes = vec![(0, HaplotypeSide::Left)];
        let pi = calculate_pi(&variants, &haplotypes, 1000);
        assert_eq!(pi, 0.0); // Single haplotype has no diversity
    }

    #[test]
    fn test_calculate_pi_n0_infinite() {
        let variants = vec![create_variant(100, vec![Some(vec![0, 1])])];
        let haplotypes = vec![]; // No haplotypes
        let pi = calculate_pi(&variants, &haplotypes, 1000);
        assert_eq!(pi, 0.0); // No haplotypes means no diversity
    }

    #[test]
    fn test_parse_region_valid_small() {
        let result = parse_region("1-1000").unwrap();
        assert_eq!(result.start, 0); // 1-based to 0-based
        assert_eq!(result.end, 1000);
    }

    #[test]
    fn test_parse_region_valid_large() {
        let result = parse_region("1000000-2000000").unwrap();
        assert_eq!(result.start, 999999); // 1-based to 0-based
        assert_eq!(result.end, 2000000);
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
        let position_allele_map = Arc::new(Mutex::new(HashMap::new()));
        let chromosome = "1".to_string();

        // Read reference sequence
        let (fasta_file, _cds_regions) = setup_test_data();
        let region = ZeroBasedHalfOpen { start: 999, end: 3001 };
        let reference_sequence = read_reference_sequence(fasta_file.path(), "1", region)
            .expect("Failed to read reference sequence");

        // Manually populate the position_allele_map
        {
            let mut pam = position_allele_map.lock();
            pam.insert(1000, ('A', 'T'));
            pam.insert(2000, ('C', 'G'));
            pam.insert(3000, ('G', 'A'));
        }

        let invalid_group = process_variants(
            &variants,
            &sample_names,
            2, // haplotype_group=2 (invalid, since only 0 and 1)
            &sample_filter,
            ZeroBasedHalfOpen { start: 999, end: 3001 },
            ZeroBasedHalfOpen { start: 999, end: 3001 },
            adjusted_sequence_length,
            Arc::clone(&position_allele_map),
            chromosome.clone(),
            false, // is_filtered_set
            &reference_sequence,
            &vec![], // Empty TranscriptAnnotationCDS for test
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
            999, // VCF position 1000 becomes 0-based position 999
            vec![
                Some(vec![0, 0]), // SAMPLE1: 0|0:20 (below threshold)
                Some(vec![0, 1]), // SAMPLE2: 0|1:40
            ],
        );
    
        // VCF line with one genotype below the GQ threshold
        let variant_line = "chr1\t1000\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:20\t0|1:40";
        let region = ZeroBasedHalfOpen { start: 999, end: 2000 };
        let result = process_variant(
            variant_line,
            "1",
            region,
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
        let region = ZeroBasedHalfOpen { start: 999, end: 2000 };
        let result = process_variant(
            valid_variant_line,
            "1",
            region,
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
    

    
    fn setup_test_data() -> (NamedTempFile, Vec<CdsRegion>) {
        let mut fasta_file = NamedTempFile::new().expect("Failed to create temporary fasta file");
    
        // Write a simple sequence that's long enough to test anything
        let sequence = "ACGT".repeat(10000);
        writeln!(fasta_file, ">1").expect("Failed to write FASTA header");
        writeln!(fasta_file, "{}", sequence).expect("Failed to write sequence");
        fasta_file.flush().expect("Failed to flush file");
        
        // Create FASTA index file
        let fasta_path = fasta_file.path();
        let index_path = format!("{}.fai", fasta_path.display());
        let mut index_file = std::fs::File::create(&index_path).expect("Failed to create index file");
        
        // Write index entry: NAME\tLENGTH\tOFFSET\tLINEBASES\tLINEWIDTH
        // For our simple FASTA: chromosome "1", length 40000, offset 3 (after ">1\n"), 40000 bases per line, 40001 chars per line (including newline)
        writeln!(index_file, "1\t{}\t3\t{}\t{}", sequence.len(), sequence.len(), sequence.len() + 1)
            .expect("Failed to write index");
        index_file.flush().expect("Failed to flush index file");
    
        let cds_regions = vec![
            CdsRegion { 
                transcript_id: "test1".to_string(), 
                segments: vec![(1200, 1901, '+', 0)] 
            },
            CdsRegion { 
                transcript_id: "test2".to_string(), 
                segments: vec![(1950, 2113, '+', 0)] 
            },
            CdsRegion { 
                transcript_id: "test3".to_string(), 
                segments: vec![(2600, 2679, '+', 0)] 
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
        let position_allele_map = Arc::new(Mutex::new(HashMap::new()));
        let chromosome = "1".to_string();
        let (fasta_file, _cds_regions) = setup_test_data();
        let region = ZeroBasedHalfOpen { start: 999, end: 3001 };
        let reference_sequence = read_reference_sequence(fasta_file.path(), "1", region)
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
            let region = ZeroBasedHalfOpen { start: 999, end: 3001 };
            let result = process_variant(
                line,
                "1",
                region,
                &mut missing_data_info,
                &sample_names,
                30,
                &mut filtering_stats,
                None,
                None,
                &position_allele_map,
            );
            assert!(result.is_ok(), "Failed to process variant: {}", line);
        }
    
        // Now, process the variants
        let variants = vec![
            create_variant(1000, vec![Some(vec![0, 0]), Some(vec![0, 1]), Some(vec![1, 1])]),
            create_variant(2000, vec![Some(vec![0, 0]), Some(vec![0, 0]), Some(vec![0, 0])]),
            create_variant(3000, vec![Some(vec![0, 1]), Some(vec![1, 1]), Some(vec![0, 0])]),
        ];
        
        // Manually populate the position_allele_map
        {
            let mut pam = position_allele_map.lock();
            pam.insert(1000, ('A', 'T'));
            pam.insert(2000, ('C', 'G'));
            pam.insert(3000, ('G', 'A'));
        }
    
        let result = process_variants(
            &variants,
            &sample_names,
            0, // haplotype_group is irrelevant now
            &sample_filter,
            ZeroBasedHalfOpen { start: 999, end: 3001 },
            ZeroBasedHalfOpen { start: 999, end: 3001 },
            adjusted_sequence_length,
            Arc::clone(&position_allele_map),
            chromosome.clone(),
            false, // is_filtered_set
            &reference_sequence,
            &vec![], // Empty TranscriptAnnotationCDS for test
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
        let position_allele_map = Arc::new(Mutex::new(HashMap::new()));
        let chromosome = "1".to_string();
        let (fasta_file, _cds_regions) = setup_test_data();
        let region = ZeroBasedHalfOpen { start: 999, end: 3001 };
        let reference_sequence = read_reference_sequence(fasta_file.path(), "1", region)
            .expect("Failed to read reference sequence");
        // Manually populate the position_allele_map
        {
            let mut pam = position_allele_map.lock();
            pam.insert(1000, ('A', 'T'));
            pam.insert(2000, ('C', 'G'));
            pam.insert(3000, ('G', 'A'));
        }
    
        let _result_group1 = process_variants(
            &variants,
            &sample_names,
            1, // haplotype_group=1
            &sample_filter,
            ZeroBasedHalfOpen { start: 999, end: 3001 }, // Include position 3000
            ZeroBasedHalfOpen { start: 999, end: 3001 }, // Include position 3000
            adjusted_sequence_length,
            Arc::clone(&position_allele_map),
            chromosome.clone(),
            false, // is_filtered_set
            &reference_sequence,
            &vec![], // Empty TranscriptAnnotationCDS for test
        ).unwrap();
    
        let (_segsites, _w_theta, _pi, n_hap, _site_diversity) = match _result_group1 {
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
        
        // First, test the basic count_segregating_sites function directly
        let direct_count = count_segregating_sites(&variants);
        println!("Direct count_segregating_sites result: {}", direct_count);
        println!("Variants: {:?}", variants);
        let adjusted_sequence_length: Option<i64> = None;
        let _mask: Option<&[(i64, i64)]> = None;
        let mut _filtering_stats = FilteringStats::default();
        let position_allele_map = Arc::new(Mutex::new(HashMap::new()));
        let chromosome = "1".to_string();
    
        // Adjusted CDS regions
        let (fasta_file, _cds_regions) = setup_test_data();
    
        // Read reference sequence covering the CDS regions
        let region = ZeroBasedHalfOpen { start: 999, end: 3001 };
        let reference_sequence = read_reference_sequence(fasta_file.path(), "1", region)
            .expect("Failed to read reference sequence");
    
        // Manually populate the position_allele_map
        {
            let mut pam = position_allele_map.lock();
            pam.insert(1000, ('A', 'T'));
            pam.insert(2000, ('C', 'G'));
            pam.insert(3000, ('G', 'A'));
        }
    
        let _result_group1 = process_variants(
            &variants,
            &sample_names,
            1, // haplotype_group=1
            &sample_filter,
            ZeroBasedHalfOpen { start: 999, end: 3001 }, // Include position 3000
            ZeroBasedHalfOpen { start: 999, end: 3001 }, // Include position 3000
            adjusted_sequence_length,
            Arc::clone(&position_allele_map),
            chromosome.clone(),
            true, // is_filtered_set
            &reference_sequence,
            &vec![], // Empty TranscriptAnnotationCDS for test
        ).unwrap();

        // Correctly unwrap the Option to access the inner tuple
        let (segsites, _w_theta, _pi, _n_hap, _site_diversity) = match _result_group1 {
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
       let chromosome = "1".to_string();
       let (fasta_file, _cds_regions) = setup_test_data();
       let region = ZeroBasedHalfOpen { start: 999, end: 3001 };
       let reference_sequence = read_reference_sequence(fasta_file.path(), "1", region)
           .expect("Failed to read reference sequence");

        {
            let mut pam = position_allele_map.lock();
            pam.insert(1000, ('A', 'G'));
            pam.insert(2000, ('T', 'C'));
        }
    
       let result = process_variants(
           &variants,
           &sample_names,
           1,  // haplotype_group=1
           &sample_filter,
           ZeroBasedHalfOpen { start: 999, end: 2001 }, // Include position 2000
           ZeroBasedHalfOpen { start: 999, end: 2001 }, // Include position 2000
           Some(100),  // sequence_length=100 for some reason
           Arc::clone(&position_allele_map),
           chromosome,
           false,
           &reference_sequence,
           &vec![], // Empty TranscriptAnnotationCDS for test
       ).unwrap();
    
       let (segsites, w_theta, _pi, n_hap, _site_diversity) = match result {
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
        let position_allele_map = Arc::new(Mutex::new(HashMap::new()));
        let chromosome = "1".to_string();
        let (fasta_file, _cds_regions) = setup_test_data();
        let region = ZeroBasedHalfOpen { start: 999, end: 3001 };
        let reference_sequence = read_reference_sequence(fasta_file.path(), "1", region)
            .expect("Failed to read reference sequence");
        // Manually populate the position_allele_map
        {
            let mut pam = position_allele_map.lock();
            pam.insert(1000, ('A', 'T'));
            pam.insert(2000, ('C', 'G'));
            pam.insert(3000, ('G', 'A'));
        }
    
        let _result_group1 = process_variants(
            &variants,
            &sample_names,
            1, // haplotype_group=1
            &sample_filter,
            ZeroBasedHalfOpen { start: 999, end: 3001 }, // Include position 3000
            ZeroBasedHalfOpen { start: 999, end: 3001 }, // Include position 3000
            adjusted_sequence_length,
            Arc::clone(&position_allele_map),
            chromosome.clone(),
            false, // is_filtered_set
            &reference_sequence,
            &vec![], // Empty TranscriptAnnotationCDS for test
        ).unwrap();
    
        // Correctly unwrap the Option to access the inner tuple
        let (_segsites, w_theta, _pi, _n_hap, _site_diversity) = match _result_group1 {
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
        let position_allele_map = Arc::new(Mutex::new(HashMap::new()));
        let chromosome = "1".to_string();
        let (fasta_file, _cds_regions) = setup_test_data();
        let region = ZeroBasedHalfOpen { start: 999, end: 3001 };
        let reference_sequence = read_reference_sequence(fasta_file.path(), "1", region)
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
            ZeroBasedHalfOpen { start: 999, end: 3001 },
            ZeroBasedHalfOpen { start: 999, end: 3001 },
            adjusted_sequence_length,
            Arc::clone(&position_allele_map),
            chromosome.clone(),
            true, // is_filtered_set
            &reference_sequence,
            &vec![], // Empty TranscriptAnnotationCDS for test
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
        let position_allele_map = Arc::new(Mutex::new(HashMap::new()));
        let chromosome = "1".to_string();
        let (fasta_file, _cds_regions) = setup_test_data();
        let region = ZeroBasedHalfOpen { start: 999, end: 3001 };
        let reference_sequence = read_reference_sequence(fasta_file.path(), "1", region)
            .expect("Failed to read reference sequence");

        // Manually populate the position_allele_map
        {
            let mut pam = position_allele_map.lock();
            pam.insert(1000, ('A', 'T'));
            pam.insert(2000, ('C', 'G'));
            pam.insert(3000, ('G', 'A'));
        }
    
        let _result_group1 = process_variants(
            &variants,
            &sample_names,
            1, // haplotype_group=1
            &sample_filter,
            ZeroBasedHalfOpen { start: 999, end: 3001 }, // Include position 3000
            ZeroBasedHalfOpen { start: 999, end: 3001 }, // Include position 3000
            adjusted_sequence_length,
            Arc::clone(&position_allele_map),
            chromosome.clone(),
            true, // is_filtered_set
            &reference_sequence,
            &vec![], // Empty TranscriptAnnotationCDS for test
        ).unwrap();
    
        let (_segsites, _w_theta, _pi, n_hap, _site_diversity) = match _result_group1 {
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
        let position_allele_map = Arc::new(Mutex::new(HashMap::new()));
        let chromosome = "1".to_string();
        let (fasta_file, _cds_regions) = setup_test_data();
        let region = ZeroBasedHalfOpen { start: 999, end: 3001 };
        let reference_sequence = read_reference_sequence(fasta_file.path(), "1", region)
            .expect("Failed to read reference sequence");
        
        // Manually populate the position_allele_map
        {
            let mut pam = position_allele_map.lock();
            pam.insert(1000, ('A', 'T'));
            pam.insert(2000, ('C', 'G'));
            pam.insert(3000, ('G', 'A'));
        }
    
        let _result_group1 = process_variants(
            &variants,
            &sample_names,
            1, // haplotype_group=1
            &sample_filter,
            ZeroBasedHalfOpen { start: 999, end: 3001 }, // Include position 3000
            ZeroBasedHalfOpen { start: 999, end: 3001 }, // Include position 3000
            adjusted_sequence_length,
            Arc::clone(&position_allele_map),
            chromosome.clone(),
            true, // is_filtered_set
            &reference_sequence,
            &vec![], // Empty TranscriptAnnotationCDS for test
        ).unwrap();
    
        // Correctly unwrap the Option to access the inner tuple
        let (segsites, _w_theta, _pi, _n_hap, _site_diversity) = match _result_group1 {
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
        let position_allele_map = Arc::new(Mutex::new(HashMap::new()));
        let chromosome = "1".to_string();
        let (fasta_file, _cds_regions) = setup_test_data();
        let region = ZeroBasedHalfOpen { start: 999, end: 3001 };
        let reference_sequence = read_reference_sequence(fasta_file.path(), "1", region)
            .expect("Failed to read reference sequence");

        // Manually populate the position_allele_map
        {
            let mut pam = position_allele_map.lock();
            pam.insert(1000, ('A', 'T'));
            pam.insert(2000, ('C', 'G'));
            pam.insert(3000, ('G', 'A'));
        }
    
        let _result_group1 = process_variants(
            &variants,
            &sample_names,
            1, // haplotype_group=1
            &sample_filter,
            ZeroBasedHalfOpen { start: 999, end: 3001 }, // Include position 3000
            ZeroBasedHalfOpen { start: 999, end: 3001 }, // Include position 3000
            adjusted_sequence_length,
            Arc::clone(&position_allele_map),
            chromosome.clone(),
            true, // is_filtered_set
            &reference_sequence,
            &vec![], // Empty TranscriptAnnotationCDS for test
        ).unwrap();
    
        // Correctly unwrap the Option to access the inner tuple
        let (_segsites, w_theta, _pi, _n_hap, _site_diversity) = match _result_group1 {
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
        let position_allele_map = Arc::new(Mutex::new(HashMap::new()));
        let chromosome = "1".to_string();
        let (fasta_file, _cds_regions) = setup_test_data();
        let region = ZeroBasedHalfOpen { start: 999, end: 3001 };
        let reference_sequence = read_reference_sequence(fasta_file.path(), "1", region)
            .expect("Failed to read reference sequence");
        
        // Manually populate the position_allele_map
        {
            let mut pam = position_allele_map.lock();
            pam.insert(1000, ('A', 'T'));
            pam.insert(2000, ('C', 'G'));
            pam.insert(3000, ('G', 'A'));
        }
    
        // Process variants for haplotype_group=1 (Group 1)
        let _result_group1 = process_variants(
            &variants,
            &sample_names,
            1, // haplotype_group=1
            &sample_filter_unfiltered,
            ZeroBasedHalfOpen { start: 999, end: 3001 },
            ZeroBasedHalfOpen { start: 999, end: 3001 },
            None,
            Arc::clone(&position_allele_map),
            chromosome.clone(),
            true, // is_filtered_set
            &reference_sequence,
            &vec![], // Empty TranscriptAnnotationCDS for test
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

    #[test]
    fn test_per_site_falsta_includes_hudson_components() {
        struct DirGuard {
            original: PathBuf,
        }

        impl Drop for DirGuard {
            fn drop(&mut self) {
                std::env::set_current_dir(&self.original)
                    .expect("failed to restore working directory");
            }
        }

        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let original_dir = std::env::current_dir().expect("failed to get current dir");
        std::env::set_current_dir(temp_dir.path()).expect("failed to change working directory");
        let _dir_guard = DirGuard {
            original: original_dir,
        };

        let vcf_dir = temp_dir.path().join("vcf");
        std::fs::create_dir_all(&vcf_dir).expect("failed to create vcf dir");
        let vcf_path = vcf_dir.join("chr1.vcf");
        let mut vcf_file = File::create(&vcf_path).expect("failed to create vcf");
        writeln!(vcf_file, "##fileformat=VCFv4.2").unwrap();
        writeln!(
            vcf_file,
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSampleA\tSampleB"
        )
        .unwrap();
        writeln!(
            vcf_file,
            "chr1\t1\t.\tA\tG\t.\tPASS\t.\tGT:GQ\t0|0:99\t1|1:99"
        )
        .unwrap();
        writeln!(
            vcf_file,
            "chr1\t2\t.\tC\tT\t.\tPASS\t.\tGT:GQ\t0|1:99\t1|0:99"
        )
        .unwrap();
        writeln!(
            vcf_file,
            "chr1\t3\t.\tG\tA\t.\tPASS\t.\tGT:GQ\t1|1:99\t0|0:99"
        )
        .unwrap();

        let fasta_path = temp_dir.path().join("reference.fa");
        let mut fasta_file = File::create(&fasta_path).expect("failed to create fasta");
        writeln!(fasta_file, ">chr1").unwrap();
        writeln!(fasta_file, "ACGTACGTACGT").unwrap();
        drop(fasta_file);

        let fai_path = temp_dir.path().join("reference.fa.fai");
        let mut fai_file = File::create(&fai_path).expect("failed to create fai");
        writeln!(fai_file, "chr1\t12\t6\t12\t13").unwrap();

        let gtf_path = temp_dir.path().join("annotations.gtf");
        let mut gtf_file = File::create(&gtf_path).expect("failed to create gtf");
        writeln!(
            gtf_file,
            "chr1\tsource\tCDS\t1\t3\t.\t+\t0\tgene_id \"GENE1\"; transcript_id \"TRANS1\"; gene_name \"GENE1\";"
        )
        .unwrap();

        let config_path = temp_dir.path().join("config.tsv");
        let mut config_file = File::create(&config_path).expect("failed to create config");
        writeln!(
            config_file,
            "seqnames\tstart\tend\tPOS\torig_ID\tverdict\tcateg\tSampleA\tSampleB"
        )
        .unwrap();
        writeln!(config_file, "chr1\t1\t3\t1\tid1\tpass\tinv\t0|0\t1|1").unwrap();

        let config_entries = parse_config_file(&config_path).expect("failed to parse config");

        struct ProgressGuard;

        impl Drop for ProgressGuard {
            fn drop(&mut self) {
                crate::progress::finish_all();
            }
        }

        crate::progress::init_global_progress(config_entries.len());
        let _progress_guard = ProgressGuard;

        let args = Args {
            vcf_folder: vcf_dir.to_string_lossy().into_owned(),
            chr: None,
            region: None,
            config_file: None,
            output_file: None,
            min_gq: 30,
            mask_file: None,
            allow_file: None,
            reference_path: fasta_path.to_string_lossy().into_owned(),
            gtf_path: gtf_path.to_string_lossy().into_owned(),
            enable_pca: false,
            pca_components: 10,
            pca_output: "pca_results.tsv".to_string(),
            enable_fst: true,
            fst_populations: None,
        };

        let output_csv = temp_dir.path().join("results.csv");
        process_config_entries(
            &config_entries,
            &args.vcf_folder,
            &output_csv,
            args.min_gq,
            None,
            None,
            &args,
        )
        .expect("process_config_entries failed");

        let falsta_path = temp_dir
            .path()
            .join("per_site_fst_output.falsta");
        assert!(falsta_path.exists(), "per-site falsta was not created");
        let contents = std::fs::read_to_string(&falsta_path)
            .expect("failed to read per-site falsta");
        let lines: Vec<&str> = contents.lines().collect();

        let fst_header = ">hudson_pairwise_fst_hap_0v1_chr_1_start_1_end_3";
        let fst_index = lines
            .iter()
            .position(|line| *line == fst_header)
            .expect("missing hudson per-site fst header");
        let fst_values: Vec<f64> = lines[fst_index + 1]
            .split(',')
            .map(|v| v.parse::<f64>().expect("invalid FST value"))
            .collect();
        assert_eq!(fst_values.len(), 3);
        assert!((fst_values[0] - 1.0).abs() < 1e-6);
        assert!((fst_values[1] + 1.0).abs() < 1e-6, "FST should retain negative values");
        assert!((fst_values[2] - 1.0).abs() < 1e-6);

        let numerator_header = ">hudson_pairwise_fst_hap_0v1_numerator_chr_1_start_1_end_3";
        let numerator_index = lines
            .iter()
            .position(|line| *line == numerator_header)
            .expect("missing hudson numerator header");
        let numerator_values: Vec<f64> = lines[numerator_index + 1]
            .split(',')
            .map(|v| v.parse::<f64>().expect("invalid numerator value"))
            .collect();
        assert_eq!(numerator_values.len(), 3);
        assert!((numerator_values[0] - 1.0).abs() < 1e-6);
        assert!((numerator_values[1] + 0.5).abs() < 1e-6);
        assert!((numerator_values[2] - 1.0).abs() < 1e-6);

        let denominator_header = ">hudson_pairwise_fst_hap_0v1_denominator_chr_1_start_1_end_3";
        let denominator_index = lines
            .iter()
            .position(|line| *line == denominator_header)
            .expect("missing hudson denominator header");
        let denominator_values: Vec<f64> = lines[denominator_index + 1]
            .split(',')
            .map(|v| v.parse::<f64>().expect("invalid denominator value"))
            .collect();
        assert_eq!(denominator_values.len(), 3);
        assert!((denominator_values[0] - 1.0).abs() < 1e-6);
        assert!((denominator_values[1] - 0.5).abs() < 1e-6);
        assert!((denominator_values[2] - 1.0).abs() < 1e-6);
    }
}
