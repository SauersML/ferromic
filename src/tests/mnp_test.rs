#[cfg(test)]
mod tests {
    use crate::process::{
        FilteringStats, MissingDataInfo, process_variant, ZeroBasedHalfOpen,
    };

    #[test]
    fn test_mnp_filtering_mixed_snp_mnp() {
        let sample_names = vec!["Sample1".to_string()];
        let mut missing_data_info = MissingDataInfo::default();
        let mut filtering_stats = FilteringStats::default();
        let min_gq = 30;
        let indices = vec![9];

        // A mixed variant: A -> G (SNP), TT (MNP)
        // This should be rejected because of the 'TT'.
        let variant_line = "chr1\t1000\t.\tA\tG,TT\t.\tPASS\t.\tGT:GQ\t1|2:40";
        let region = ZeroBasedHalfOpen { start: 999, end: 2000 };
        let regions = [region];

        let result = process_variant(
            variant_line,
            "1",
            &regions,
            &mut missing_data_info,
            &sample_names,
            &indices,
            min_gq,
            &mut filtering_stats,
            None,
            None,
        );

        assert!(result.is_ok());
        let variant_opt = result.unwrap();

        // Expect None because it contains an MNP
        assert!(variant_opt.is_none(), "Expected mixed SNP/MNP variant to be filtered out");
        assert_eq!(filtering_stats.mnp_variants, 1);
    }
}
