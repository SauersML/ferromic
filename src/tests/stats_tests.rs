use std::collections::HashMap;
use std::path::PathBuf;

// Helper function to create a Variant for testing
fn create_variant(position: i64, genotypes: Vec<Option<Vec<u8>>>) -> Variant {
    Variant { position, genotypes }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use tempfile::{NamedTempFile, tempdir};
    use super::*;

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
    fn test_calculate_pairwise_differences() {
        let variants = vec![
            create_variant(1, vec![Some(vec![0, 0]), Some(vec![0, 1]), Some(vec![1, 1])]),
            create_variant(2, vec![Some(vec![0, 0]), Some(vec![0, 0]), Some(vec![0, 1])]),
            create_variant(3, vec![Some(vec![0, 1]), Some(vec![1, 1]), Some(vec![0, 0])]),
        ];

        let result = calculate_pairwise_differences(&variants, 3);

        // Check the number of pairwise comparisons
        assert_eq!(result.len(), 3);

        // Check specific pairwise differences
        assert!(result.contains(&((0, 1), 2, vec![1, 3])));
        assert!(result.contains(&((0, 2), 3, vec![1, 2, 3])));
        assert!(result.contains(&((1, 2), 3, vec![1, 2, 3])));

        // Test with no variants
        let empty_result = calculate_pairwise_differences(&[], 3);
        assert_eq!(empty_result.len(), 3);
        assert!(empty_result.iter().all(|(_, count, positions)| *count == 0 && positions.is_empty()));

        // Test with missing data
        let missing_data_variants = vec![
            create_variant(1, vec![Some(vec![0, 0]), None, Some(vec![1, 1])]),
            create_variant(2, vec![Some(vec![0, 1]), Some(vec![0, 1]), None]),
        ];
        let missing_data_result = calculate_pairwise_differences(&missing_data_variants, 3);
        assert_eq!(missing_data_result.len(), 3);
        // Check that missing data is handled correctly
        assert!(missing_data_result.iter().any(|((i, j), count, _)| (*i == 0 && *j == 1 && *count == 0)));
    }

    #[test]
    fn test_extract_sample_id() {
        // Standard cases where sample IDs are extracted correctly
        assert_eq!(extract_sample_id("sample_123"), "123");
        assert_eq!(extract_sample_id("sample_with_multiple_underscores_456"), "456");
        
        assert_eq!(extract_sample_id("nomore_underscore"), "underscore");
        
        // Edge cases
        assert_eq!(extract_sample_id(""), "");
        assert_eq!(extract_sample_id("_"), "");
        assert_eq!(extract_sample_id("sample_"), "");
        
        // Complex sample names
        assert_eq!(extract_sample_id("EAS_JPT_NA18939"), "NA18939");
        assert_eq!(extract_sample_id("AMR_PEL_HG02059"), "HG02059");
        
        // Extra cases
        assert_eq!(extract_sample_id("singlepart"), "singlepart");
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
    fn test_calculate_watterson_theta() {
        // Test with typical values
        // Number of segregating sites = 10, number of haplotypes = 5, sequence length = 1000
        // Harmonic number a1 = 1 + 1/2 + 1/3 + 1/4 = 2.083333...
        // Expected theta = 10 / (2.083333 * 1000) ≈ 0.0048
        assert!((calculate_watterson_theta(10, 5, 1000) - 0.0048).abs() < 1e-6);
    
        // Test with no segregating sites
        assert_eq!(calculate_watterson_theta(0, 5, 1000), 0.0);
    
        // Test with large number of segregating sites
        // Number of segregating sites = 1000, number of haplotypes = 100, sequence length = 10000
        // Harmonic number a1 ≈ 5.1773775
        // Expected theta = 1000 / (5.1773775 * 10000) ≈ 0.0193
        assert!((calculate_watterson_theta(1000, 100, 10000) - 0.0193).abs() < 1e-4);
    
        // Test with minimum possible sample size (2)
        // Number of segregating sites = 5, number of haplotypes = 2, sequence length = 1000
        // Harmonic number a1 = 1
        // Expected theta = 5 / (1 * 1000) = 0.005
        assert!((calculate_watterson_theta(5, 2, 1000) - 0.005).abs() < 1e-6);
    
        // Test with very large sequence length
        // Number of segregating sites = 100, number of haplotypes = 10, sequence length = 1,000,000
        // Harmonic number a1 ≈ 2.8289682539682537
        // Expected theta = 100 / (2.8289682539682537 * 1,000,000) ≈ 0.00003534
        assert!((calculate_watterson_theta(100, 10, 1_000_000) - 0.00003534).abs() < 1e-8);
    
        // Test with n = 1, expecting infinity
        let theta_n1 = calculate_watterson_theta(100, 1, 1000);
        assert!(theta_n1.is_infinite(), "Expected Watterson's theta to be infinity when n = 1, got {}", theta_n1);
        
        // Test with n = 0, expecting infinity
        let theta_n0 = calculate_watterson_theta(0, 0, 1000);
        assert!(theta_n0.is_infinite(), "Expected Watterson's theta to be infinity when n = 0, got {}", theta_n0);
    }

    #[test]
    fn test_calculate_pi() {
        // Test with typical values
        // Total pairwise differences = 15, number of haplotypes = 5, sequence length = 1000
        // Expected pi = 15 / (10 * 1000) = 0.0015
        assert!((calculate_pi(15, 5, 1000) - 0.0015).abs() < 1e-6);
    
        // Test with no pairwise differences
        assert_eq!(calculate_pi(0, 5, 1000), 0.0);
    
        // Test with large number of pairwise differences
        // Total pairwise differences = 10000, number of haplotypes = 100, sequence length = 10000
        // Expected pi = 10000 / (4950 * 10000) ≈ 0.00020202
        assert!((calculate_pi(10000, 100, 10000) - 0.00020202).abs() < 1e-6);
    
        // Test with minimum possible sample size (2)
        // Total pairwise differences = 5, number of haplotypes = 2, sequence length = 1000
        // Expected pi = 5 / (1 * 1000) = 0.005
        assert!((calculate_pi(5, 2, 1000) - 0.005).abs() < 1e-6);
    
        // Test with very large sequence length
        // Total pairwise differences = 1000, number of haplotypes = 10, sequence length = 1,000,000
        // Expected pi = 1000 / (45 * 1,000,000) ≈ 0.0000222222
        assert!((calculate_pi(1000, 10, 1_000_000) - 0.0000222222).abs() < 1e-9);

        // Test with n = 1, expecting infinity
        let pi_n1 = calculate_pi(100, 1, 1000);
        assert!(pi_n1.is_infinite(), "Expected pi to be infinity when n = 1, got {}", pi_n1);
        
        // Test with n = 0, expecting infinity
        let pi_n0 = calculate_pi(0, 0, 1000);
        assert!(pi_n0.is_infinite(), "Expected pi to be infinity when n = 0, got {}", pi_n0);
    }

    #[test]
    fn test_parse_region() {
        assert!(matches!(parse_region("1-1000"), Ok((1, 1000))));
        assert!(matches!(parse_region("1000000-2000000"), Ok((1000000, 2000000))));

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
    
        // Test valid variant
        let valid_line = "chr1\t1000\t.\tA\tT\t.\tPASS\t.\tGT\t0|0\t0|1\t1|1";
        let result = parse_variant(valid_line, "1", 1, 2000, &mut missing_data_info, &sample_names);
        assert!(result.is_ok());
        if let Ok(Some(variant)) = result {
            assert_eq!(variant.position, 1000);
            assert_eq!(variant.genotypes, vec![Some(vec![0, 0]), Some(vec![0, 1]), Some(vec![1, 1])]);
        }
    
        // Test variant outside region
        let out_of_range = "chr1\t3000\t.\tA\tT\t.\tPASS\t.\tGT\t0|0\t0|1\t1|1";
        assert!(parse_variant(out_of_range, "1", 1, 2000, &mut missing_data_info, &sample_names).unwrap().is_none());
    
        // Test different chromosome
        let diff_chr = "chr2\t1000\t.\tA\tT\t.\tPASS\t.\tGT\t0|0\t0|1\t1|1";
        assert!(parse_variant(diff_chr, "1", 1, 2000, &mut missing_data_info, &sample_names).unwrap().is_none());
    
        // Test missing data
        let missing_data = "chr1\t1000\t.\tA\tT\t.\tPASS\t.\tGT\t0|0\t.|.\t1|1";
        let result = parse_variant(missing_data, "1", 1, 2000, &mut missing_data_info, &sample_names);
        assert!(result.is_ok());
        if let Ok(Some(variant)) = result {
            assert_eq!(variant.genotypes, vec![Some(vec![0, 0]), None, Some(vec![1, 1])]);
        }
    
        // Test invalid format (fewer fields than required)
        let invalid_format = "chr1\t1000\t.\tA\tT\t.\tPASS\t.\tGT\t0|0"; // Only 10 fields, expecting 12 for 3 samples
        assert!(parse_variant(invalid_format, "1", 1, 2000, &mut missing_data_info, &sample_names).is_err());
    
        // Test multi-allelic site
        let multi_allelic = "chr1\t1000\t.\tA\tT,G\t.\tPASS\t.\tGT\t0|1\t1|2\t0|2";
        let result = parse_variant(multi_allelic, "1", 1, 2000, &mut missing_data_info, &sample_names);
        assert!(result.is_ok());
        if let Ok(Some(variant)) = result {
            assert_eq!(variant.position, 1000);
            assert_eq!(variant.genotypes, vec![Some(vec![0, 1]), Some(vec![1, 2]), Some(vec![0, 2])]);
        }
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
    
        // Test haplotype group 0
        let result_0 = process_variants(&variants, &sample_names, 0, &sample_filter, 1000, 3000);
        assert!(result_0.is_ok());
        if let Ok((num_segsites, w_theta, pi)) = result_0 {
            assert_eq!(num_segsites, 2);
            // Expected theta = 2 / (1.5 * 2001) ≈ 0.000666
            assert!((w_theta - 0.000666).abs() < 1e-6);
            // Expected pi = 4 / (3 * 2001) ≈ 0.000666
            assert!((pi - 0.000666).abs() < 1e-6);
        }
        
        // Test haplotype group 1
        let result_1 = process_variants(&variants, &sample_names, 1, &sample_filter, 1000, 3000);
        assert!(result_1.is_ok());
        if let Ok((num_segsites, w_theta, pi)) = result_1 {
            assert_eq!(num_segsites, 2);
            // Expected theta = 2 / (1.5 * 2001) ≈ 0.000666
            assert!((w_theta - 0.000666).abs() < 1e-6);
            // Expected pi = 4 / (3 * 2001) ≈ 0.000666
            assert!((pi - 0.000666).abs() < 1e-6);
        }

    
        // Test with empty variants
        let empty_result = process_variants(&[], &sample_names, 0, &sample_filter, 1000, 3000);
        assert!(empty_result.is_ok());
        if let Ok((num_segsites, w_theta, pi)) = empty_result {
            assert_eq!(num_segsites, 0);
            assert_eq!(w_theta, 0.0);
            assert_eq!(pi, 0.0);
        }
    
        // Test with invalid haplotype group
        let invalid_group = process_variants(&variants, &sample_names, 2, &sample_filter, 1000, 3000);
        assert!(invalid_group.is_err());
    
        // Test with missing samples
        let mut missing_sample_filter = HashMap::new();
        missing_sample_filter.insert("SAMPLE4".to_string(), (0, 1));
        let missing_result = process_variants(&variants, &sample_names, 0, &missing_sample_filter, 1000, 3000);
        assert!(missing_result.is_err());
    }

    #[test]
    fn test_parse_config_file_with_noreads() {
        use std::io::Write;
    
        let config_content = "seqnames\tstart\tend\tPOS\torig_ID\tverdict\tcateg\tSAMPLE1\tSAMPLE2\tSAMPLE3\n\
                              chr1\t1000\t2000\t1500\ttest_id\tpass\tinv\t0|1\tnoreads\t1|0";
        let path = tempfile::NamedTempFile::new().unwrap();
        write!(path.as_file(), "{}", config_content).unwrap();
        
        let result = parse_config_file(path.path());
        assert!(result.is_ok());
        let entries = result.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].samples.len(), 2);  // SAMPLE2 should be skipped
        assert!(entries[0].samples.contains_key("SAMPLE1"));
        assert!(entries[0].samples.contains_key("SAMPLE3"));
        assert!(!entries[0].samples.contains_key("SAMPLE2"));
        
        assert_eq!(entries[0].seqname, "chr1");
        assert_eq!(entries[0].start, 1000);
        assert_eq!(entries[0].end, 2000);
        assert_eq!(entries[0].samples["SAMPLE1"], (0, 1));
        assert_eq!(entries[0].samples["SAMPLE3"], (1, 0));
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
        if let Ok((num_segsites, w_theta, pi)) = result {
            assert_eq!(num_segsites, 0);
            assert_eq!(w_theta, 0.0);
            assert_eq!(pi, 0.0);
        }
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
        assert!(find_vcf_file(temp_path.to_str().unwrap(), "1").is_ok());
        assert!(find_vcf_file(temp_path.to_str().unwrap(), "2").is_ok());
        assert!(find_vcf_file(temp_path.to_str().unwrap(), "10").is_ok());
    
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
    chr1\t1000\t.\tA\tT\t.\tPASS\t.\tGT\t0|0\t0|1\t0|1\t1|0
    chr1\t1500\t.\tG\tC\t.\tPASS\t.\tGT\t0|1\t1|1\t1|0\t0|1
    chr1\t2000\t.\tT\tA\t.\tPASS\t.\tGT\t1|1\t0|1\t1|1\t0|0";
        let vcf_file = vcf_folder.join("chr1.vcf");
        let mut file = File::create(&vcf_file).unwrap();
        writeln!(file, "{}", vcf_content).unwrap();
        
        // Define config entries with at least two haplotypes per group
        let config_entries = vec![
            ConfigEntry {
                seqname: "chr1".to_string(),
                start: 1000,
                end: 2000,
                samples: {
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
        let result = process_config_entries(&config_entries, vcf_folder.to_str().unwrap(), &output_file);
        
        assert!(result.is_ok());
        assert!(output_file.exists());
        
        // Read and check the output file content
        let output_content = fs::read_to_string(&output_file).unwrap();
        let lines: Vec<&str> = output_content.lines().collect();
        assert_eq!(lines.len(), 2); // Header + 1 data line
        assert!(lines[0].starts_with("chr,region_start,region_end"));
        assert!(lines[1].starts_with("chr1,1000,2000"));
        
        
        // Clean up
        temp_dir.close().unwrap();
    }
}
