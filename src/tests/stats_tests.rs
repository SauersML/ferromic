// Helper function to create a Variant for testing
fn create_variant(position: i64, genotypes: Vec<Option<Vec<u8>>>) -> Variant {
    Variant { position, genotypes }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::{NamedTempFile, tempdir};
    use std::fs::{self, File};
    use std::io::{self, Write};
    use std::path::PathBuf;

    #[test]
    fn test_count_segregating_sites() {
        let variants = vec![
            create_variant(1, vec![Some(vec![0]), Some(vec![0]), Some(vec![1])]),
            create_variant(2, vec![Some(vec![0]), Some(vec![0]), Some(vec![0])]),
            create_variant(3, vec![Some(vec![0]), Some(vec![1]), Some(vec![0])]),
            create_variant(4, vec![Some(vec![1]), Some(vec![1]), Some(vec![1])]),
        ];

        assert_eq!(count_segregating_sites(&variants), 2);

        // Test with no variants
        assert_eq!(count_segregating_sites(&[]), 0);

        // Test with all homozygous sites
        let homozygous_variants = vec![
            create_variant(1, vec![Some(vec![0]), Some(vec![0]), Some(vec![0])]),
            create_variant(2, vec![Some(vec![1]), Some(vec![1]), Some(vec![1])]),
        ];
        assert_eq!(count_segregating_sites(&homozygous_variants), 0);

        // Test with missing data
        let missing_data_variants = vec![
            create_variant(1, vec![Some(vec![0]), None, Some(vec![1])]),
            create_variant(2, vec![Some(vec![0]), Some(vec![1]), None]),
        ];
        assert_eq!(count_segregating_sites(&missing_data_variants), 2);
    }

    #[test]
    fn test_calculate_pairwise_differences() {
        let variants = vec![
            create_variant(1, vec![Some(vec![0]), Some(vec![1]), Some(vec![1])]),
            create_variant(2, vec![Some(vec![0]), Some(vec![0]), Some(vec![1])]),
            create_variant(3, vec![Some(vec![1]), Some(vec![1]), Some(vec![0])]),
        ];

        let result = calculate_pairwise_differences(&variants, 3);

        // Check the number of pairwise comparisons
        assert_eq!(result.len(), 3);

        // Check specific pairwise differences
        assert!(result.contains(&((0, 1), 2, vec![1, 3])));
        assert!(result.contains(&((0, 2), 2, vec![2, 3])));
        assert!(result.contains(&((1, 2), 1, vec![2])));

        // Test with no variants
        let empty_result = calculate_pairwise_differences(&[], 3);
        assert_eq!(empty_result.len(), 3);
        assert!(empty_result.iter().all(|(_, count, positions)| *count == 0 && positions.is_empty()));

        // Test with missing data
        let missing_data_variants = vec![
            create_variant(1, vec![Some(vec![0]), None, Some(vec![1])]),
            create_variant(2, vec![Some(vec![1]), Some(vec![1]), None]),
        ];
        let missing_data_result = calculate_pairwise_differences(&missing_data_variants, 3);
        assert_eq!(missing_data_result.len(), 3);
        // Missing data pairs should be skipped
    }

    #[test]
    fn test_extract_sample_id() {
        // Standard cases
        assert_eq!(extract_sample_id("sample_123"), "123");
        assert_eq!(extract_sample_id("sample_with_multiple_underscores_456"), "456");

        // Edge cases
        assert_eq!(extract_sample_id("singlepart"), "singlepart");
        assert_eq!(extract_sample_id(""), "");

        // Complex sample names
        assert_eq!(extract_sample_id("EAS_JPT_NA18939"), "NA18939");
        assert_eq!(extract_sample_id("AMR_PEL_HG02059"), "HG02059");
    }

    #[test]
    fn test_harmonic() {
        assert_eq!(harmonic(1), 1.0);
        assert!((harmonic(2) - 1.5).abs() < 1e-10);
        assert!((harmonic(3) - (1.0 + 0.5 + 1.0 / 3.0)).abs() < 1e-10);
        assert!((harmonic(10) - 2.928968254).abs() < 1e-6);
    }

    #[test]
    fn test_calculate_watterson_theta() {
        // Typical values
        assert!((calculate_watterson_theta(10, 5, 1000) - 0.0048).abs() < 1e-4);

        // No segregating sites
        assert_eq!(calculate_watterson_theta(0, 5, 1000), 0.0);

        // Large sample size
        assert!((calculate_watterson_theta(1000, 100, 10000) - 0.01927).abs() < 1e-5);

        // Minimum sample size
        assert!((calculate_watterson_theta(5, 2, 1000) - 0.005).abs() < 1e-6);

        // Sequence length zero
        let theta_seq_len_zero = calculate_watterson_theta(100, 10, 0);
        assert!(theta_seq_len_zero.is_infinite());
    }

    #[test]
    fn test_calculate_pi() {
        // Typical values
        assert!((calculate_pi(15, 5, 1000) - 0.0015).abs() < 1e-6);

        // No pairwise differences
        assert_eq!(calculate_pi(0, 5, 1000), 0.0);

        // Large sample size
        assert!((calculate_pi(10000, 100, 10000) - 0.00020202).abs() < 1e-6);

        // Minimum sample size
        assert!((calculate_pi(5, 2, 1000) - 0.005).abs() < 1e-6);

        // Sequence length zero
        let pi_seq_len_zero = calculate_pi(100, 10, 0);
        assert!(pi_seq_len_zero.is_infinite());
    }

    #[test]
    fn test_parse_region() {
        assert_eq!(parse_region("1000-2000").unwrap(), (1000, 2000));

        // Invalid formats
        assert!(parse_region("1000").is_err());
        assert!(parse_region("a-b").is_err());

        // Invalid range
        assert!(parse_region("2000-1000").is_err());
    }

    #[test]
    fn test_validate_vcf_header() {
        let valid_header = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1";
        assert!(validate_vcf_header(valid_header).is_ok());

        let invalid_header = "CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT";
        assert!(validate_vcf_header(invalid_header).is_err());
    }

    #[test]
    fn test_parse_variant() {
        let sample_names = vec!["SAMPLE1".to_string(), "SAMPLE2".to_string()];
        let mut missing_data_info = MissingDataInfo::default();
        let min_gq = 30;

        let valid_line = "chr1\t1000\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:35\t0|1:40";
        let result = parse_variant(valid_line, "1", 1, 2000, &mut missing_data_info, &sample_names, min_gq);
        assert!(result.is_ok());
        if let Ok(Some(variant)) = result {
            assert_eq!(variant.position, 1000);
            assert_eq!(variant.genotypes, vec![Some(vec![0, 0]), Some(vec![0, 1])]);
        }

        // Variant with low GQ
        let low_gq_line = "chr1\t1000\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:25\t0|1:40";
        let result = parse_variant(low_gq_line, "1", 1, 2000, &mut missing_data_info, &sample_names, min_gq);
        assert!(result.unwrap().is_none());

        // Invalid format
        let invalid_line = "chr1\t1000\t.\tA\tT\t.\tPASS\t.\tGT\t0|0\t0|1";
        let result = parse_variant(invalid_line, "1", 1, 2000, &mut missing_data_info, &sample_names, min_gq);
        assert!(result.is_err());
    }

    #[test]
    fn test_process_variants() {
        let variants = vec![
            create_variant(1000, vec![Some(vec![0]), Some(vec![1])]),
            create_variant(2000, vec![Some(vec![0]), Some(vec![0])]),
            create_variant(3000, vec![Some(vec![1]), Some(vec![1])]),
        ];
        let sample_names = vec!["SAMPLE1".to_string(), "SAMPLE2".to_string()];
        let mut sample_filter = HashMap::new();
        sample_filter.insert("SAMPLE1".to_string(), (0, 1));
        sample_filter.insert("SAMPLE2".to_string(), (1, 0));

        let result = process_variants(&variants, &sample_names, 0, &sample_filter, 1000, 3000);
        assert!(result.is_ok());
        if let Ok((num_segsites, w_theta, pi, num_haplotypes, allele_frequency)) = result {
            assert_eq!(num_segsites, 2);
            assert!(w_theta > 0.0);
            assert!(pi > 0.0);
            assert_eq!(num_haplotypes, 2);
            assert_eq!(allele_frequency, 0.5);
        }
    }

    #[test]
    fn test_parse_config_file() {
        let config_content = "\
seqnames\tstart\tend\tPOS\torig_ID\tverdict\tcateg\tSample1\tSample2\n\
chr1\t1000\t2000\t.\t.\t.\t.\t0|1_lowconf\t1|1\n\
chr1\t3000\t4000\t.\t.\t.\t.\t0|0\t0|1\n";

        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, "{}", config_content).unwrap();

        let config_entries = parse_config_file(temp_file.path()).unwrap();

        assert_eq!(config_entries.len(), 2);

        // First entry
        let entry1 = &config_entries[0];
        assert_eq!(entry1.samples_unfiltered.len(), 2);
        assert_eq!(entry1.samples_filtered.len(), 1); // Only Sample2 matches exact genotypes

        // Second entry
        let entry2 = &config_entries[1];
        assert_eq!(entry2.samples_unfiltered.len(), 2);
        assert_eq!(entry2.samples_filtered.len(), 2);
    }

    #[test]
    fn test_find_vcf_file() {
        use std::fs::File;

        let temp_dir = tempdir().unwrap();
        let temp_path = temp_dir.path();

        File::create(temp_path.join("chr1.vcf")).unwrap();
        File::create(temp_path.join("chr2.vcf.gz")).unwrap();

        let vcf_path = find_vcf_file(temp_path.to_str().unwrap(), "1").unwrap();
        assert!(vcf_path.ends_with("chr1.vcf"));

        let vcf_path = find_vcf_file(temp_path.to_str().unwrap(), "2").unwrap();
        assert!(vcf_path.ends_with("chr2.vcf.gz"));
    }

    #[test]
    fn test_open_vcf_reader() {
        let temp_dir = tempdir().unwrap();
        let vcf_path = temp_dir.path().join("test.vcf");
        File::create(&vcf_path).unwrap();

        let reader = open_vcf_reader(&vcf_path);
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
    }

    #[test]
    fn test_haplotype_processing() {
        let variants = vec![
            create_variant(1, vec![Some(vec![0, 1]), Some(vec![1, 0])]),
            create_variant(2, vec![Some(vec![1, 1]), Some(vec![0, 0])]),
        ];
        let sample_names = vec!["Sample1".to_string(), "Sample2".to_string()];
        let mut sample_filter = HashMap::new();
        sample_filter.insert("Sample1".to_string(), (0, 1)); // Left haplotype in group 0, right in group 1
        sample_filter.insert("Sample2".to_string(), (1, 0)); // Left haplotype in group 1, right in group 0

        // Process haplotype group 0
        let result_group0 = process_variants(
            &variants,
            &sample_names,
            0,
            &sample_filter,
            1,
            2,
        ).unwrap();

        // Check that haplotypes are correctly assigned
        // In group 0, we have Sample1 left haplotype and Sample2 right haplotype
        // So we should have two haplotypes in total
        assert_eq!(result_group0.3, 2); // num_haplotypes
    }

}
