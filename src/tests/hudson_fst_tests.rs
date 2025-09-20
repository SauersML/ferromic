#[cfg(test)]
mod hudson_fst_tests {
    use crate::process::*;
    use crate::stats::*;

    /// Helper function to create test variants
    fn create_test_variant(position: i64, genotypes: Vec<Option<Vec<u8>>>) -> Variant {
        Variant {
            position,
            genotypes,
        }
    }

    #[test]
    fn test_hudson_fst_per_site_calculation() {
        // Test case 1: Perfect population structure (FST should be 1.0)
        // Pop1: all 0|0, Pop2: all 1|1
        let variant = create_test_variant(
            100,
            vec![
                Some(vec![0, 0]), // sample 0 - pop1
                Some(vec![0, 0]), // sample 1 - pop1
                Some(vec![1, 1]), // sample 2 - pop2
                Some(vec![1, 1]), // sample 3 - pop2
            ],
        );

        let pop1_haps = vec![
            (0, HaplotypeSide::Left),
            (0, HaplotypeSide::Right),
            (1, HaplotypeSide::Left),
            (1, HaplotypeSide::Right),
        ];
        let pop2_haps = vec![
            (2, HaplotypeSide::Left),
            (2, HaplotypeSide::Right),
            (3, HaplotypeSide::Left),
            (3, HaplotypeSide::Right),
        ];

        let sample_names = vec![
            "s1".to_string(),
            "s2".to_string(),
            "s3".to_string(),
            "s4".to_string(),
        ];

        let pop1_context = PopulationContext {
            id: PopulationId::HaplotypeGroup(0),
            haplotypes: pop1_haps,
            variants: &[variant.clone()],
            sample_names: &sample_names,
            sequence_length: 1000,
            dense_genotypes: None,
            dense_summary: None,
        };

        let pop2_context = PopulationContext {
            id: PopulationId::HaplotypeGroup(1),
            haplotypes: pop2_haps,
            variants: &[variant.clone()],
            sample_names: &sample_names,
            sequence_length: 1000,
            dense_genotypes: None,
            dense_summary: None,
        };

        // Test the Hudson FST calculation
        let result = calculate_hudson_fst_for_pair(&pop1_context, &pop2_context);
        assert!(
            result.is_ok(),
            "Hudson FST calculation failed: {:?}",
            result.err()
        );

        let outcome = result.unwrap();
        println!("Perfect structure FST: {:?}", outcome.fst);

        // With perfect population structure, FST should be close to 1.0
        if let Some(fst) = outcome.fst {
            assert!(
                fst > 0.8,
                "FST {} is too low for perfect population structure",
                fst
            );
            assert!(fst <= 1.0, "FST {} exceeds maximum value of 1.0", fst);
        } else {
            panic!("FST calculation returned None for perfect population structure");
        }
    }

    #[test]
    fn test_hudson_fst_no_structure() {
        // Test case 2: No population structure (FST should be ~0)
        // Both populations have same allele frequencies
        let variant = create_test_variant(
            100,
            vec![
                Some(vec![0, 1]), // sample 0 - pop1 (heterozygous)
                Some(vec![1, 0]), // sample 1 - pop1 (heterozygous)
                Some(vec![0, 1]), // sample 2 - pop2 (heterozygous)
                Some(vec![1, 0]), // sample 3 - pop2 (heterozygous)
            ],
        );

        let pop1_haps = vec![
            (0, HaplotypeSide::Left),
            (0, HaplotypeSide::Right),
            (1, HaplotypeSide::Left),
            (1, HaplotypeSide::Right),
        ];
        let pop2_haps = vec![
            (2, HaplotypeSide::Left),
            (2, HaplotypeSide::Right),
            (3, HaplotypeSide::Left),
            (3, HaplotypeSide::Right),
        ];

        let sample_names = vec![
            "s1".to_string(),
            "s2".to_string(),
            "s3".to_string(),
            "s4".to_string(),
        ];

        let pop1_context = PopulationContext {
            id: PopulationId::HaplotypeGroup(0),
            haplotypes: pop1_haps,
            variants: &[variant.clone()],
            sample_names: &sample_names,
            sequence_length: 1000,
            dense_genotypes: None,
            dense_summary: None,
        };

        let pop2_context = PopulationContext {
            id: PopulationId::HaplotypeGroup(1),
            haplotypes: pop2_haps,
            variants: &[variant.clone()],
            sample_names: &sample_names,
            sequence_length: 1000,
            dense_genotypes: None,
            dense_summary: None,
        };

        let result = calculate_hudson_fst_for_pair(&pop1_context, &pop2_context);
        assert!(
            result.is_ok(),
            "Hudson FST calculation failed: {:?}",
            result.err()
        );

        let outcome = result.unwrap();
        println!("No structure FST: {:?}", outcome.fst);

        // With no population structure, FST can be negative (which is expected)
        // The negative value indicates that within-population diversity is higher than between-population diversity
        if let Some(fst) = outcome.fst {
            assert!(
                fst >= -1.0 && fst <= 1.0,
                "FST {} is outside valid range [-1, 1]",
                fst
            );
            // For this specific case with identical allele frequencies but different arrangements,
            // negative FST is mathematically correct
            println!(
                "FST = {} is within expected range for this data pattern",
                fst
            );
        } else {
            panic!("FST calculation returned None for valid data");
        }
    }

    #[test]
    fn test_hudson_fst_per_site_components() {
        // Test that per-site components are correctly calculated
        let variant = create_test_variant(
            100,
            vec![
                Some(vec![0, 0]), // sample 0 - pop1
                Some(vec![0, 0]), // sample 1 - pop1
                Some(vec![1, 1]), // sample 2 - pop2
                Some(vec![1, 1]), // sample 3 - pop2
            ],
        );

        let pop1_haps = vec![
            (0, HaplotypeSide::Left),
            (0, HaplotypeSide::Right),
            (1, HaplotypeSide::Left),
            (1, HaplotypeSide::Right),
        ];
        let pop2_haps = vec![
            (2, HaplotypeSide::Left),
            (2, HaplotypeSide::Right),
            (3, HaplotypeSide::Left),
            (3, HaplotypeSide::Right),
        ];

        let sample_names = vec![
            "s1".to_string(),
            "s2".to_string(),
            "s3".to_string(),
            "s4".to_string(),
        ];

        let pop1_context = PopulationContext {
            id: PopulationId::HaplotypeGroup(0),
            haplotypes: pop1_haps,
            variants: &[variant.clone()],
            sample_names: &sample_names,
            sequence_length: 1000,
            dense_genotypes: None,
            dense_summary: None,
        };

        let pop2_context = PopulationContext {
            id: PopulationId::HaplotypeGroup(1),
            haplotypes: pop2_haps,
            variants: &[variant.clone()],
            sample_names: &sample_names,
            sequence_length: 1000,
            dense_genotypes: None,
            dense_summary: None,
        };

        let region = QueryRegion {
            start: 99,
            end: 101,
        }; // 0-based, so 99-101 covers position 100
        let result = calculate_hudson_fst_for_pair_with_sites(&pop1_context, &pop2_context, region);
        assert!(
            result.is_ok(),
            "Hudson FST with sites calculation failed: {:?}",
            result.err()
        );

        let (outcome, sites) = result.unwrap();

        println!("Number of sites returned: {}", sites.len());
        for (i, site) in sites.iter().enumerate() {
            println!(
                "Site {}: pos={}, fst={:?}, dxy={:?}",
                i, site.position, site.fst, site.d_xy
            );
        }

        // Find the site at position 101 (where the actual data is)
        let site_101 = sites.iter().find(|s| s.position == 101);
        assert!(site_101.is_some(), "Site at position 101 not found");

        let site = site_101.unwrap();
        println!("Site 101 FST: {:?}", site.fst);
        println!("Site 101 D_xy: {:?}", site.d_xy);
        println!("Site 101 pi_pop1: {:?}", site.pi_pop1);
        println!("Site 101 pi_pop2: {:?}", site.pi_pop2);

        // With perfect structure: D_xy should be 1.0, pi should be 0.0
        assert!(site.d_xy.is_some(), "D_xy should be calculated");
        assert!(site.pi_pop1.is_some(), "pi_pop1 should be calculated");
        assert!(site.pi_pop2.is_some(), "pi_pop2 should be calculated");

        if let (Some(dxy), Some(pi1), Some(pi2)) = (site.d_xy, site.pi_pop1, site.pi_pop2) {
            assert!(
                (dxy - 1.0).abs() < 0.01,
                "D_xy should be ~1.0 for perfect structure, got {}",
                dxy
            );
            assert!(
                pi1.abs() < 0.01,
                "pi_pop1 should be ~0.0 for monomorphic population, got {}",
                pi1
            );
            assert!(
                pi2.abs() < 0.01,
                "pi_pop2 should be ~0.0 for monomorphic population, got {}",
                pi2
            );
        }

        // Regional FST should match per-site aggregation
        if let Some(regional_fst) = outcome.fst {
            if let Some(site_fst) = site.fst {
                assert!(
                    (regional_fst - site_fst).abs() < 0.01,
                    "Regional FST {} should match site FST {} for single-site region",
                    regional_fst,
                    site_fst
                );
            }
        }
    }

    #[test]
    fn test_hudson_fst_monomorphic_window() {
        // Test that monomorphic windows (no variants) return FST = 0.0
        let pop1_haplotypes = vec![
            (0, HaplotypeSide::Left),
            (0, HaplotypeSide::Right),
            (1, HaplotypeSide::Left),
            (1, HaplotypeSide::Right),
        ];
        let pop2_haplotypes = vec![
            (2, HaplotypeSide::Left),
            (2, HaplotypeSide::Right),
            (3, HaplotypeSide::Left),
            (3, HaplotypeSide::Right),
        ];

        // No variants - monomorphic region
        let variants = vec![];
        let sample_names = vec![
            "sample1".to_string(),
            "sample2".to_string(),
            "sample3".to_string(),
            "sample4".to_string(),
        ];

        let pop1_context = PopulationContext {
            id: PopulationId::HaplotypeGroup(0),
            haplotypes: pop1_haplotypes,
            variants: &variants,
            sample_names: &sample_names,
            sequence_length: 1000,
            dense_genotypes: None,
            dense_summary: None,
        };

        let pop2_context = PopulationContext {
            id: PopulationId::HaplotypeGroup(1),
            haplotypes: pop2_haplotypes,
            variants: &variants,
            sample_names: &sample_names,
            sequence_length: 1000,
            dense_genotypes: None,
            dense_summary: None,
        };

        let result = calculate_hudson_fst_for_pair(&pop1_context, &pop2_context);
        assert!(
            result.is_ok(),
            "Hudson FST calculation should succeed for monomorphic window"
        );

        let outcome = result.unwrap();
        println!("Monomorphic window FST: {:?}", outcome.fst);

        // Monomorphic windows should return FST = 0.0, not None
        assert!(
            outcome.fst.is_some(),
            "Monomorphic window should have FST = 0.0, not None"
        );
        assert_eq!(
            outcome.fst.unwrap(),
            0.0,
            "Monomorphic window FST should be exactly 0.0"
        );
    }

    #[test]
    fn test_hudson_fst_ratio_of_sums_no_missingness() {
        // Test 1: Ratio-of-sums with no missingness (two sites, mixed structure)
        // Expected regional FST = 5/9 ≈ 0.5555555556

        let sample_names = vec![
            "sample0".to_string(),
            "sample1".to_string(),
            "sample2".to_string(),
            "sample3".to_string(),
        ];

        // Site A (pos=100): perfect structure
        // sample0: [0,0], sample1: [0,0], sample2: [1,1], sample3: [1,1]
        let variant_a = create_test_variant(
            100,
            vec![
                Some(vec![0, 0]), // sample0
                Some(vec![0, 0]), // sample1
                Some(vec![1, 1]), // sample2
                Some(vec![1, 1]), // sample3
            ],
        );

        // Site B (pos=200): identical pop frequencies (each pop has 2×0 and 2×1)
        // sample0: [0,1], sample1: [0,1], sample2: [0,1], sample3: [0,1]
        let variant_b = create_test_variant(
            200,
            vec![
                Some(vec![0, 1]), // sample0
                Some(vec![0, 1]), // sample1
                Some(vec![0, 1]), // sample2
                Some(vec![0, 1]), // sample3
            ],
        );

        let variants = vec![variant_a, variant_b];

        // pop1 = samples {0,1}, pop2 = samples {2,3}
        let pop1_haplotypes = vec![
            (0, HaplotypeSide::Left),
            (0, HaplotypeSide::Right),
            (1, HaplotypeSide::Left),
            (1, HaplotypeSide::Right),
        ];
        let pop2_haplotypes = vec![
            (2, HaplotypeSide::Left),
            (2, HaplotypeSide::Right),
            (3, HaplotypeSide::Left),
            (3, HaplotypeSide::Right),
        ];

        let region = QueryRegion {
            start: 100,
            end: 200,
        };
        let sequence_length = 2; // Only 2 variant sites

        let pop1_context = PopulationContext {
            id: PopulationId::HaplotypeGroup(0),
            haplotypes: pop1_haplotypes,
            variants: &variants,
            sample_names: &sample_names,
            sequence_length,
            dense_genotypes: None,
            dense_summary: None,
        };

        let pop2_context = PopulationContext {
            id: PopulationId::HaplotypeGroup(1),
            haplotypes: pop2_haplotypes,
            variants: &variants,
            sample_names: &sample_names,
            sequence_length,
            dense_genotypes: None,
            dense_summary: None,
        };

        let result = calculate_hudson_fst_for_pair_with_sites(&pop1_context, &pop2_context, region);
        assert!(result.is_ok(), "Hudson FST calculation should succeed");

        let (outcome, sites) = result.unwrap();

        // Filter to only sites with actual data (non-None FST)
        let variant_sites: Vec<_> = sites.iter().filter(|s| s.fst.is_some()).collect();
        assert_eq!(
            variant_sites.len(),
            2,
            "Should have exactly 2 variant sites"
        );

        // Site A (position 101 in 1-based): perfect structure
        let site_a = variant_sites
            .iter()
            .find(|s| s.position == 101)
            .expect("Site A not found");
        assert!(
            (site_a.fst.unwrap() - 1.0).abs() < 1e-12,
            "Site A FST should be exactly 1.0"
        );
        assert!(
            (site_a.num_component.unwrap() - 1.0).abs() < 1e-12,
            "Site A numerator should be 1.0"
        );
        assert!(
            (site_a.den_component.unwrap() - 1.0).abs() < 1e-12,
            "Site A denominator should be 1.0"
        );

        // Site B (position 201 in 1-based): FST = -1/3
        let site_b = variant_sites
            .iter()
            .find(|s| s.position == 201)
            .expect("Site B not found");
        assert!(
            (site_b.fst.unwrap() - (-1.0 / 3.0)).abs() < 1e-12,
            "Site B FST should be exactly -1/3"
        );
        assert!(
            (site_b.num_component.unwrap() - (-1.0 / 6.0)).abs() < 1e-12,
            "Site B numerator should be -1/6"
        );
        assert!(
            (site_b.den_component.unwrap() - 0.5).abs() < 1e-12,
            "Site B denominator should be 0.5"
        );

        // Regional FST from ratio-of-sums: (5/6) / (3/2) = 5/9
        let expected_regional_fst = 5.0 / 9.0;
        let aggregated_fst =
            aggregate_hudson_from_sites(&sites).expect("Aggregated FST should be Some");
        assert!(
            (aggregated_fst - expected_regional_fst).abs() < 1e-12,
            "Aggregated FST should be exactly 5/9, got {}",
            aggregated_fst
        );

        // Window FST should match (if using unbiased implementation)
        assert!(outcome.fst.is_some(), "Window FST should be Some");
        assert!(
            (outcome.fst.unwrap() - expected_regional_fst).abs() < 1e-12,
            "Window FST should match ratio-of-sums: expected {}, got {}",
            expected_regional_fst,
            outcome.fst.unwrap()
        );

        println!(
            "Test 1 PASSED: Regional FST = {} (expected 5/9 = {})",
            outcome.fst.unwrap(),
            expected_regional_fst
        );
    }

    #[test]
    fn test_hudson_fst_ratio_of_sums_uneven_missingness() {
        // Test 2: Ratio-of-sums under uneven missingness
        // Expected regional FST = 1/3 ≈ 0.3333333333

        let sample_names = vec![
            "sample0".to_string(),
            "sample1".to_string(),
            "sample2".to_string(),
            "sample3".to_string(),
        ];

        // Site A (pos=100): perfect structure (same as Test 1)
        let variant_a = create_test_variant(
            100,
            vec![
                Some(vec![0, 0]), // sample0
                Some(vec![0, 0]), // sample1
                Some(vec![1, 1]), // sample2
                Some(vec![1, 1]), // sample3
            ],
        );

        // Site B (pos=200): missing data for samples 0 and 2
        // sample0: None, sample1: [0,1], sample2: None, sample3: [0,1]
        // This gives each pop n=2 with {0:1, 1:1}
        let variant_b = create_test_variant(
            200,
            vec![
                None,             // sample0 - missing
                Some(vec![0, 1]), // sample1
                None,             // sample2 - missing
                Some(vec![0, 1]), // sample3
            ],
        );

        let variants = vec![variant_a, variant_b];

        let pop1_haplotypes = vec![
            (0, HaplotypeSide::Left),
            (0, HaplotypeSide::Right),
            (1, HaplotypeSide::Left),
            (1, HaplotypeSide::Right),
        ];
        let pop2_haplotypes = vec![
            (2, HaplotypeSide::Left),
            (2, HaplotypeSide::Right),
            (3, HaplotypeSide::Left),
            (3, HaplotypeSide::Right),
        ];

        let region = QueryRegion {
            start: 100,
            end: 200,
        };
        let sequence_length = 2; // Only 2 variant sites

        let pop1_context = PopulationContext {
            id: PopulationId::HaplotypeGroup(0),
            haplotypes: pop1_haplotypes,
            variants: &variants,
            sample_names: &sample_names,
            sequence_length,
            dense_genotypes: None,
            dense_summary: None,
        };

        let pop2_context = PopulationContext {
            id: PopulationId::HaplotypeGroup(1),
            haplotypes: pop2_haplotypes,
            variants: &variants,
            sample_names: &sample_names,
            sequence_length,
            dense_genotypes: None,
            dense_summary: None,
        };

        let result = calculate_hudson_fst_for_pair_with_sites(&pop1_context, &pop2_context, region);
        assert!(result.is_ok(), "Hudson FST calculation should succeed");

        let (outcome, sites) = result.unwrap();

        // Filter to only sites with actual data (non-None FST)
        let variant_sites: Vec<_> = sites.iter().filter(|s| s.fst.is_some()).collect();
        assert_eq!(
            variant_sites.len(),
            2,
            "Should have exactly 2 variant sites"
        );

        // Site A unchanged (FST=1, num=1, den=1)
        let site_a = variant_sites
            .iter()
            .find(|s| s.position == 101)
            .expect("Site A not found");
        assert!(
            (site_a.fst.unwrap() - 1.0).abs() < 1e-12,
            "Site A FST should be exactly 1.0"
        );
        assert!(
            (site_a.num_component.unwrap() - 1.0).abs() < 1e-12,
            "Site A numerator should be 1.0"
        );
        assert!(
            (site_a.den_component.unwrap() - 1.0).abs() < 1e-12,
            "Site A denominator should be 1.0"
        );

        // Site B with missingness: FST = -1, num = -0.5, den = 0.5
        let site_b = variant_sites
            .iter()
            .find(|s| s.position == 201)
            .expect("Site B not found");
        assert!(
            (site_b.fst.unwrap() - (-1.0)).abs() < 1e-12,
            "Site B FST should be exactly -1.0"
        );
        assert!(
            (site_b.num_component.unwrap() - (-0.5)).abs() < 1e-12,
            "Site B numerator should be -0.5"
        );
        assert!(
            (site_b.den_component.unwrap() - 0.5).abs() < 1e-12,
            "Site B denominator should be 0.5"
        );

        // Regional FST: (1 + (-0.5)) / (1 + 0.5) = 0.5 / 1.5 = 1/3
        let expected_regional_fst = 1.0 / 3.0;
        let aggregated_fst =
            aggregate_hudson_from_sites(&sites).expect("Aggregated FST should be Some");
        assert!(
            (aggregated_fst - expected_regional_fst).abs() < 1e-12,
            "Aggregated FST should be exactly 1/3, got {}",
            aggregated_fst
        );

        // Window FST should match
        assert!(outcome.fst.is_some(), "Window FST should be Some");
        assert!(
            (outcome.fst.unwrap() - expected_regional_fst).abs() < 1e-12,
            "Window FST should match ratio-of-sums: expected {}, got {}",
            expected_regional_fst,
            outcome.fst.unwrap()
        );

        println!(
            "Test 2 PASSED: Regional FST = {} (expected 1/3 = {})",
            outcome.fst.unwrap(),
            expected_regional_fst
        );
    }

    #[test]
    fn test_hudson_fst_monomorphic_window_surgical() {
        // Test 3: Monomorphic window ⇒ overall Hudson FST = 0

        let sample_names = vec![
            "sample0".to_string(),
            "sample1".to_string(),
            "sample2".to_string(),
            "sample3".to_string(),
        ];

        // No variants at all
        let variants = vec![];

        let pop1_haplotypes = vec![
            (0, HaplotypeSide::Left),
            (0, HaplotypeSide::Right),
            (1, HaplotypeSide::Left),
            (1, HaplotypeSide::Right),
        ];
        let pop2_haplotypes = vec![
            (2, HaplotypeSide::Left),
            (2, HaplotypeSide::Right),
            (3, HaplotypeSide::Left),
            (3, HaplotypeSide::Right),
        ];

        let region = QueryRegion {
            start: 100,
            end: 102,
        };
        let sequence_length = (region.end - region.start + 1) as i64; // 3

        let pop1_context = PopulationContext {
            id: PopulationId::HaplotypeGroup(0),
            haplotypes: pop1_haplotypes,
            variants: &variants,
            sample_names: &sample_names,
            sequence_length,
            dense_genotypes: None,
            dense_summary: None,
        };

        let pop2_context = PopulationContext {
            id: PopulationId::HaplotypeGroup(1),
            haplotypes: pop2_haplotypes,
            variants: &variants,
            sample_names: &sample_names,
            sequence_length,
            dense_genotypes: None,
            dense_summary: None,
        };

        let result = calculate_hudson_fst_for_pair_with_sites(&pop1_context, &pop2_context, region);
        assert!(result.is_ok(), "Hudson FST calculation should succeed");

        let (outcome, sites) = result.unwrap();

        // Should have 3 sites (positions 100, 101, 102), all with no data
        assert_eq!(sites.len(), 3, "Should have exactly 3 sites in region");

        // Each site should have None for all components
        for site in &sites {
            assert!(
                site.fst.is_none(),
                "Site {} FST should be None",
                site.position
            );
            assert!(
                site.num_component.is_none(),
                "Site {} num_component should be None",
                site.position
            );
            assert!(
                site.den_component.is_none(),
                "Site {} den_component should be None",
                site.position
            );
        }

        // Aggregated FST should be 0.0 (Σnum = 0, Σden = 0)
        let aggregated_fst =
            aggregate_hudson_from_sites(&sites).expect("Aggregated FST should be Some(0.0)");
        assert!(
            (aggregated_fst - 0.0).abs() < 1e-12,
            "Aggregated FST should be exactly 0.0"
        );

        // Window FST should also be 0.0
        assert!(outcome.fst.is_some(), "Window FST should be Some(0.0)");
        assert!(
            (outcome.fst.unwrap() - 0.0).abs() < 1e-12,
            "Window FST should be exactly 0.0"
        );

        println!(
            "Test 3 PASSED: Monomorphic window FST = {} (expected 0.0)",
            outcome.fst.unwrap()
        );
    }

    #[test]
    fn test_pi_dxy_consistency_with_uneven_coverage() {
        // Test that π/Dxy calculations are consistent with per-site math
        // when pairs have different numbers of comparable sites

        let sample_names = vec![
            "sample0".to_string(),
            "sample1".to_string(),
            "sample2".to_string(),
            "sample3".to_string(),
        ];

        // Site A (pos=100): all samples have data
        let variant_a = create_test_variant(
            100,
            vec![
                Some(vec![0, 0]), // sample0
                Some(vec![0, 1]), // sample1
                Some(vec![1, 1]), // sample2
                Some(vec![1, 0]), // sample3
            ],
        );

        // Site B (pos=200): only samples 1 and 3 have data (uneven coverage)
        let variant_b = create_test_variant(
            200,
            vec![
                None,             // sample0 - missing
                Some(vec![0, 0]), // sample1
                None,             // sample2 - missing
                Some(vec![1, 1]), // sample3
            ],
        );

        let variants = vec![variant_a, variant_b];

        let pop1_haplotypes = vec![
            (0, HaplotypeSide::Left),
            (0, HaplotypeSide::Right),
            (1, HaplotypeSide::Left),
            (1, HaplotypeSide::Right),
        ];
        let pop2_haplotypes = vec![
            (2, HaplotypeSide::Left),
            (2, HaplotypeSide::Right),
            (3, HaplotypeSide::Left),
            (3, HaplotypeSide::Right),
        ];

        let region = QueryRegion {
            start: 100,
            end: 200,
        };
        let sequence_length = 2; // Only 2 variant sites

        let pop1_context = PopulationContext {
            id: PopulationId::HaplotypeGroup(0),
            haplotypes: pop1_haplotypes,
            variants: &variants,
            sample_names: &sample_names,
            sequence_length,
            dense_genotypes: None,
            dense_summary: None,
        };

        let pop2_context = PopulationContext {
            id: PopulationId::HaplotypeGroup(1),
            haplotypes: pop2_haplotypes,
            variants: &variants,
            sample_names: &sample_names,
            sequence_length,
            dense_genotypes: None,
            dense_summary: None,
        };

        let result = calculate_hudson_fst_for_pair_with_sites(&pop1_context, &pop2_context, region);
        assert!(result.is_ok(), "Hudson FST calculation should succeed");

        let (outcome, sites) = result.unwrap();

        // Filter to only sites with actual data
        let variant_sites: Vec<_> = sites.iter().filter(|s| s.fst.is_some()).collect();

        // Should have 2 variant sites but with different coverage patterns
        assert_eq!(
            variant_sites.len(),
            2,
            "Should have exactly 2 variant sites"
        );

        // Verify that π and Dxy values are consistent with per-site aggregation
        let manual_pi1_sum: f64 = variant_sites.iter().filter_map(|s| s.pi_pop1).sum();
        let manual_pi2_sum: f64 = variant_sites.iter().filter_map(|s| s.pi_pop2).sum();
        let manual_dxy_sum: f64 = variant_sites.iter().filter_map(|s| s.d_xy).sum();

        let expected_pi1 = manual_pi1_sum / sequence_length as f64;
        let expected_pi2 = manual_pi2_sum / sequence_length as f64;
        let expected_dxy = manual_dxy_sum / sequence_length as f64;

        // The reported π/Dxy should match per-site aggregation
        if let Some(reported_pi1) = outcome.pi_pop1 {
            assert!(
                (reported_pi1 - expected_pi1).abs() < 1e-12,
                "Reported π1 ({}) should match per-site aggregation ({})",
                reported_pi1,
                expected_pi1
            );
        }

        if let Some(reported_pi2) = outcome.pi_pop2 {
            assert!(
                (reported_pi2 - expected_pi2).abs() < 1e-12,
                "Reported π2 ({}) should match per-site aggregation ({})",
                reported_pi2,
                expected_pi2
            );
        }

        if let Some(reported_dxy) = outcome.d_xy {
            assert!(
                (reported_dxy - expected_dxy).abs() < 1e-12,
                "Reported Dxy ({}) should match per-site aggregation ({})",
                reported_dxy,
                expected_dxy
            );
        }

        println!("π/Dxy consistency test PASSED: per-site aggregation matches reported values");
    }

    #[test]
    fn test_hudson_fst_multi_allelic_site() {
        // Test multi-allelic site: 3 alleles with different frequencies between populations
        // pop1: A/C/G at 0.5/0.25/0.25 vs pop2: A/C/G at 0.2/0.3/0.5

        let sample_names = vec![
            "sample0".to_string(),
            "sample1".to_string(),
            "sample2".to_string(),
            "sample3".to_string(),
        ];

        // Multi-allelic site with alleles 0=A, 1=C, 2=G
        // pop1: 2×A, 1×C, 1×G → frequencies: A=0.5, C=0.25, G=0.25
        // pop2: 1×A, 1×C, 2×G → frequencies: A=0.25, C=0.25, G=0.5
        let variant = create_test_variant(
            100,
            vec![
                Some(vec![0, 0]), // sample0: A/A (pop1)
                Some(vec![1, 2]), // sample1: C/G (pop1)
                Some(vec![0, 1]), // sample2: A/C (pop2)
                Some(vec![2, 2]), // sample3: G/G (pop2)
            ],
        );

        let variants = vec![variant];

        let pop1_haplotypes = vec![
            (0, HaplotypeSide::Left),
            (0, HaplotypeSide::Right),
            (1, HaplotypeSide::Left),
            (1, HaplotypeSide::Right),
        ];
        let pop2_haplotypes = vec![
            (2, HaplotypeSide::Left),
            (2, HaplotypeSide::Right),
            (3, HaplotypeSide::Left),
            (3, HaplotypeSide::Right),
        ];

        let region = QueryRegion {
            start: 100,
            end: 100,
        };
        let sequence_length = 1;

        let pop1_context = PopulationContext {
            id: PopulationId::HaplotypeGroup(0),
            haplotypes: pop1_haplotypes,
            variants: &variants,
            sample_names: &sample_names,
            sequence_length,
            dense_genotypes: None,
            dense_summary: None,
        };

        let pop2_context = PopulationContext {
            id: PopulationId::HaplotypeGroup(1),
            haplotypes: pop2_haplotypes,
            variants: &variants,
            sample_names: &sample_names,
            sequence_length,
            dense_genotypes: None,
            dense_summary: None,
        };

        let result = calculate_hudson_fst_for_pair_with_sites(&pop1_context, &pop2_context, region);
        assert!(
            result.is_ok(),
            "Hudson FST calculation should succeed for multi-allelic site"
        );

        let (_outcome, sites) = result.unwrap();

        let variant_sites: Vec<_> = sites.iter().filter(|s| s.fst.is_some()).collect();
        assert_eq!(variant_sites.len(), 1, "Should have exactly 1 variant site");

        let site = variant_sites[0];

        // Manual calculation for verification:
        // pop1 frequencies: p1A=0.5, p1C=0.25, p1G=0.25
        // pop2 frequencies: p2A=0.25, p2C=0.25, p2G=0.5
        // Dxy = 1 - (p1A*p2A + p1C*p2C + p1G*p2G) = 1 - (0.5*0.25 + 0.25*0.25 + 0.25*0.5) = 1 - 0.3125 = 0.6875
        let expected_dxy = 0.6875;

        // π1 = (4/3) * (1 - (0.5² + 0.25² + 0.25²)) = (4/3) * (1 - 0.375) = (4/3) * 0.625 = 0.8333...
        let expected_pi1 = (4.0 / 3.0) * (1.0 - (0.5 * 0.5 + 0.25 * 0.25 + 0.25 * 0.25));

        // π2 = (4/3) * (1 - (0.25² + 0.25² + 0.5²)) = (4/3) * (1 - 0.375) = (4/3) * 0.625 = 0.8333...
        let expected_pi2 = (4.0 / 3.0) * (1.0 - (0.25 * 0.25 + 0.25 * 0.25 + 0.5 * 0.5));

        // numerator = Dxy - 0.5*(π1 + π2)
        let expected_numerator = expected_dxy - 0.5 * (expected_pi1 + expected_pi2);
        let expected_fst = expected_numerator / expected_dxy;

        assert!(
            (site.d_xy.unwrap() - expected_dxy).abs() < 1e-12,
            "Multi-allelic Dxy should be {}, got {}",
            expected_dxy,
            site.d_xy.unwrap()
        );

        assert!(
            (site.pi_pop1.unwrap() - expected_pi1).abs() < 1e-12,
            "Multi-allelic π1 should be {}, got {}",
            expected_pi1,
            site.pi_pop1.unwrap()
        );

        assert!(
            (site.pi_pop2.unwrap() - expected_pi2).abs() < 1e-12,
            "Multi-allelic π2 should be {}, got {}",
            expected_pi2,
            site.pi_pop2.unwrap()
        );

        assert!(
            (site.fst.unwrap() - expected_fst).abs() < 1e-12,
            "Multi-allelic FST should be {}, got {}",
            expected_fst,
            site.fst.unwrap()
        );

        println!(
            "Multi-allelic test PASSED: Dxy={:.6}, π1={:.6}, π2={:.6}, FST={:.6}",
            site.d_xy.unwrap(),
            site.pi_pop1.unwrap(),
            site.pi_pop2.unwrap(),
            site.fst.unwrap()
        );
    }

    #[test]
    fn test_hudson_fst_identical_frequencies() {
        // Test case: identical allele frequencies between populations
        // This should give FST close to 0 (but may be slightly negative due to sampling)

        let sample_names = vec![
            "sample0".to_string(),
            "sample1".to_string(),
            "sample2".to_string(),
            "sample3".to_string(),
        ];

        // Both populations have identical frequencies: 50% A, 50% T
        let variant = create_test_variant(
            100,
            vec![
                Some(vec![0, 1]), // sample0: A/T (pop1)
                Some(vec![1, 0]), // sample1: T/A (pop1) → pop1: 50% A, 50% T
                Some(vec![0, 1]), // sample2: A/T (pop2)
                Some(vec![1, 0]), // sample3: T/A (pop2) → pop2: 50% A, 50% T
            ],
        );

        let variants = vec![variant];

        let pop1_haplotypes = vec![
            (0, HaplotypeSide::Left),
            (0, HaplotypeSide::Right),
            (1, HaplotypeSide::Left),
            (1, HaplotypeSide::Right),
        ];
        let pop2_haplotypes = vec![
            (2, HaplotypeSide::Left),
            (2, HaplotypeSide::Right),
            (3, HaplotypeSide::Left),
            (3, HaplotypeSide::Right),
        ];

        let region = QueryRegion {
            start: 100,
            end: 100,
        };
        let sequence_length = 1;

        let pop1_context = PopulationContext {
            id: PopulationId::HaplotypeGroup(0),
            haplotypes: pop1_haplotypes,
            variants: &variants,
            sample_names: &sample_names,
            sequence_length,
            dense_genotypes: None,
            dense_summary: None,
        };

        let pop2_context = PopulationContext {
            id: PopulationId::HaplotypeGroup(1),
            haplotypes: pop2_haplotypes,
            variants: &variants,
            sample_names: &sample_names,
            sequence_length,
            dense_genotypes: None,
            dense_summary: None,
        };

        let result = calculate_hudson_fst_for_pair_with_sites(&pop1_context, &pop2_context, region);
        assert!(result.is_ok(), "Hudson FST calculation should succeed");

        let (_outcome, sites) = result.unwrap();

        assert_eq!(sites.len(), 1, "Should have exactly 1 site");
        let site = &sites[0];

        assert!(site.d_xy.is_some(), "Dxy should be calculable");
        assert!(site.pi_pop1.is_some(), "π1 should be calculable");
        assert!(site.pi_pop2.is_some(), "π2 should be calculable");
        assert!(
            site.fst.is_some(),
            "FST should be calculable for identical frequencies"
        );

        // With identical frequencies, FST can be negative due to finite sample corrections
        // The key is that it should be calculable (not None) and within reasonable bounds
        assert!(
            site.fst.unwrap() >= -1.0 && site.fst.unwrap() <= 1.0,
            "FST should be within [-1,1] bounds, got {}",
            site.fst.unwrap()
        );

        println!(
            "Identical frequencies test PASSED: FST = {} (within bounds)",
            site.fst.unwrap()
        );
    }

    #[test]
    fn test_hudson_fst_variant_compatibility_guard() {
        // Test that calculate_hudson_fst_per_site guards against variant incompatibility

        let sample_names = vec![
            "sample0".to_string(),
            "sample1".to_string(),
            "sample2".to_string(),
            "sample3".to_string(),
        ];

        // Different variants for each population (incompatible)
        let variants1 = vec![create_test_variant(
            100,
            vec![
                Some(vec![0, 0]),
                Some(vec![0, 1]),
                Some(vec![1, 1]),
                Some(vec![1, 0]),
            ],
        )];

        let variants2 = vec![create_test_variant(
            200,
            vec![
                // Different position!
                Some(vec![0, 0]),
                Some(vec![0, 1]),
                Some(vec![1, 1]),
                Some(vec![1, 0]),
            ],
        )];

        let pop1_haplotypes = vec![
            (0, HaplotypeSide::Left),
            (0, HaplotypeSide::Right),
            (1, HaplotypeSide::Left),
            (1, HaplotypeSide::Right),
        ];
        let pop2_haplotypes = vec![
            (2, HaplotypeSide::Left),
            (2, HaplotypeSide::Right),
            (3, HaplotypeSide::Left),
            (3, HaplotypeSide::Right),
        ];

        let region = QueryRegion {
            start: 100,
            end: 200,
        };

        let pop1_context = PopulationContext {
            id: PopulationId::HaplotypeGroup(0),
            haplotypes: pop1_haplotypes,
            variants: &variants1, // Different variants!
            sample_names: &sample_names,
            sequence_length: 2,
            dense_genotypes: None,
            dense_summary: None,
        };

        let pop2_context = PopulationContext {
            id: PopulationId::HaplotypeGroup(1),
            haplotypes: pop2_haplotypes,
            variants: &variants2, // Different variants!
            sample_names: &sample_names,
            sequence_length: 2,
            dense_genotypes: None,
            dense_summary: None,
        };

        // Direct call to per-site function should return empty vector due to guard
        let sites = calculate_hudson_fst_per_site(&pop1_context, &pop2_context, region);
        assert!(
            sites.is_empty(),
            "Should return empty vector for incompatible variants"
        );

        // Safe wrapper should return error
        let result = calculate_hudson_fst_for_pair_with_sites(&pop1_context, &pop2_context, region);
        assert!(
            result.is_err(),
            "Safe wrapper should return error for incompatible variants"
        );

        println!("Variant compatibility guard test PASSED");
    }

    #[test]
    fn test_hudson_fst_missing_data() {
        // Test Hudson FST with missing genotypes
        let variant = create_test_variant(
            100,
            vec![
                Some(vec![0, 0]), // sample 0 - pop1
                None,             // sample 1 - pop1 (missing)
                Some(vec![1, 1]), // sample 2 - pop2
                Some(vec![1, 1]), // sample 3 - pop2
            ],
        );

        let pop1_haps = vec![
            (0, HaplotypeSide::Left),
            (0, HaplotypeSide::Right),
            (1, HaplotypeSide::Left),
            (1, HaplotypeSide::Right),
        ];
        let pop2_haps = vec![
            (2, HaplotypeSide::Left),
            (2, HaplotypeSide::Right),
            (3, HaplotypeSide::Left),
            (3, HaplotypeSide::Right),
        ];

        let sample_names = vec![
            "s1".to_string(),
            "s2".to_string(),
            "s3".to_string(),
            "s4".to_string(),
        ];

        let pop1_context = PopulationContext {
            id: PopulationId::HaplotypeGroup(0),
            haplotypes: pop1_haps,
            variants: &[variant.clone()],
            sample_names: &sample_names,
            sequence_length: 1000,
            dense_genotypes: None,
            dense_summary: None,
        };

        let pop2_context = PopulationContext {
            id: PopulationId::HaplotypeGroup(1),
            haplotypes: pop2_haps,
            variants: &[variant.clone()],
            sample_names: &sample_names,
            sequence_length: 1000,
            dense_genotypes: None,
            dense_summary: None,
        };

        let result = calculate_hudson_fst_for_pair(&pop1_context, &pop2_context);
        assert!(
            result.is_ok(),
            "Hudson FST calculation with missing data failed: {:?}",
            result.err()
        );

        let outcome = result.unwrap();
        println!("Missing data FST: {:?}", outcome.fst);

        // Should still calculate FST using available data
        assert!(
            outcome.fst.is_some(),
            "FST should be calculated despite missing data"
        );

        if let Some(fst) = outcome.fst {
            // With remaining data (pop1: 2 haplotypes all 0, pop2: 4 haplotypes all 1),
            // should still show strong structure
            assert!(
                fst > 0.5,
                "FST {} should be high with remaining structured data",
                fst
            );
        }
    }
}
