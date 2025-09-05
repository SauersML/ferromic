#[cfg(test)]
mod hudson_fst_tests {
    use crate::process::*;
    use crate::stats::*;
    use std::collections::HashMap;

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
        let variant = create_test_variant(100, vec![
            Some(vec![0, 0]), // sample 0 - pop1
            Some(vec![0, 0]), // sample 1 - pop1  
            Some(vec![1, 1]), // sample 2 - pop2
            Some(vec![1, 1]), // sample 3 - pop2
        ]);

        let pop1_haps = vec![(0, HaplotypeSide::Left), (0, HaplotypeSide::Right), 
                            (1, HaplotypeSide::Left), (1, HaplotypeSide::Right)];
        let pop2_haps = vec![(2, HaplotypeSide::Left), (2, HaplotypeSide::Right),
                            (3, HaplotypeSide::Left), (3, HaplotypeSide::Right)];

        let sample_names = vec!["s1".to_string(), "s2".to_string(), "s3".to_string(), "s4".to_string()];
        
        let pop1_context = PopulationContext {
            id: PopulationId::HaplotypeGroup(0),
            haplotypes: pop1_haps,
            variants: &[variant.clone()],
            sample_names: &sample_names,
            sequence_length: 1000,
        };

        let pop2_context = PopulationContext {
            id: PopulationId::HaplotypeGroup(1), 
            haplotypes: pop2_haps,
            variants: &[variant.clone()],
            sample_names: &sample_names,
            sequence_length: 1000,
        };

        // Test the Hudson FST calculation
        let result = calculate_hudson_fst_for_pair(&pop1_context, &pop2_context);
        assert!(result.is_ok(), "Hudson FST calculation failed: {:?}", result.err());

        let outcome = result.unwrap();
        println!("Perfect structure FST: {:?}", outcome.fst);
        
        // With perfect population structure, FST should be close to 1.0
        if let Some(fst) = outcome.fst {
            assert!(fst > 0.8, "FST {} is too low for perfect population structure", fst);
            assert!(fst <= 1.0, "FST {} exceeds maximum value of 1.0", fst);
        } else {
            panic!("FST calculation returned None for perfect population structure");
        }
    }

    #[test]
    fn test_hudson_fst_no_structure() {
        // Test case 2: No population structure (FST should be ~0)
        // Both populations have same allele frequencies
        let variant = create_test_variant(100, vec![
            Some(vec![0, 1]), // sample 0 - pop1 (heterozygous)
            Some(vec![1, 0]), // sample 1 - pop1 (heterozygous)
            Some(vec![0, 1]), // sample 2 - pop2 (heterozygous) 
            Some(vec![1, 0]), // sample 3 - pop2 (heterozygous)
        ]);

        let pop1_haps = vec![(0, HaplotypeSide::Left), (0, HaplotypeSide::Right),
                            (1, HaplotypeSide::Left), (1, HaplotypeSide::Right)];
        let pop2_haps = vec![(2, HaplotypeSide::Left), (2, HaplotypeSide::Right),
                            (3, HaplotypeSide::Left), (3, HaplotypeSide::Right)];

        let sample_names = vec!["s1".to_string(), "s2".to_string(), "s3".to_string(), "s4".to_string()];
        
        let pop1_context = PopulationContext {
            id: PopulationId::HaplotypeGroup(0),
            haplotypes: pop1_haps,
            variants: &[variant.clone()],
            sample_names: &sample_names,
            sequence_length: 1000,
        };

        let pop2_context = PopulationContext {
            id: PopulationId::HaplotypeGroup(1),
            haplotypes: pop2_haps,
            variants: &[variant.clone()],
            sample_names: &sample_names,
            sequence_length: 1000,
        };

        let result = calculate_hudson_fst_for_pair(&pop1_context, &pop2_context);
        assert!(result.is_ok(), "Hudson FST calculation failed: {:?}", result.err());

        let outcome = result.unwrap();
        println!("No structure FST: {:?}", outcome.fst);
        
        // With no population structure, FST can be negative (which is expected)
        // The negative value indicates that within-population diversity is higher than between-population diversity
        if let Some(fst) = outcome.fst {
            assert!(fst >= -1.0 && fst <= 1.0, "FST {} is outside valid range [-1, 1]", fst);
            // For this specific case with identical allele frequencies but different arrangements,
            // negative FST is mathematically correct
            println!("FST = {} is within expected range for this data pattern", fst);
        } else {
            panic!("FST calculation returned None for valid data");
        }
    }

    #[test] 
    fn test_hudson_fst_per_site_components() {
        // Test that per-site components are correctly calculated
        let variant = create_test_variant(100, vec![
            Some(vec![0, 0]), // sample 0 - pop1
            Some(vec![0, 0]), // sample 1 - pop1
            Some(vec![1, 1]), // sample 2 - pop2
            Some(vec![1, 1]), // sample 3 - pop2
        ]);

        let pop1_haps = vec![(0, HaplotypeSide::Left), (0, HaplotypeSide::Right),
                            (1, HaplotypeSide::Left), (1, HaplotypeSide::Right)];
        let pop2_haps = vec![(2, HaplotypeSide::Left), (2, HaplotypeSide::Right),
                            (3, HaplotypeSide::Left), (3, HaplotypeSide::Right)];

        let sample_names = vec!["s1".to_string(), "s2".to_string(), "s3".to_string(), "s4".to_string()];
        
        let pop1_context = PopulationContext {
            id: PopulationId::HaplotypeGroup(0),
            haplotypes: pop1_haps,
            variants: &[variant.clone()],
            sample_names: &sample_names,
            sequence_length: 1000,
        };

        let pop2_context = PopulationContext {
            id: PopulationId::HaplotypeGroup(1),
            haplotypes: pop2_haps,
            variants: &[variant.clone()],
            sample_names: &sample_names,
            sequence_length: 1000,
        };

        let region = QueryRegion { start: 99, end: 101 }; // 0-based, so 99-101 covers position 100
        let result = calculate_hudson_fst_for_pair_with_sites(&pop1_context, &pop2_context, region);
        assert!(result.is_ok(), "Hudson FST with sites calculation failed: {:?}", result.err());

        let (outcome, sites) = result.unwrap();
        
        println!("Number of sites returned: {}", sites.len());
        for (i, site) in sites.iter().enumerate() {
            println!("Site {}: pos={}, fst={:?}, dxy={:?}", i, site.position, site.fst, site.d_xy);
        }
        
        // Find the site at position 100
        let site_100 = sites.iter().find(|s| s.position == 100);
        assert!(site_100.is_some(), "Site at position 100 not found");
        
        let site = site_100.unwrap();
        println!("Site 100 FST: {:?}", site.fst);
        println!("Site 100 D_xy: {:?}", site.d_xy);
        println!("Site 100 pi_pop1: {:?}", site.pi_pop1);
        println!("Site 100 pi_pop2: {:?}", site.pi_pop2);
        
        // With perfect structure: D_xy should be 1.0, pi should be 0.0
        assert!(site.d_xy.is_some(), "D_xy should be calculated");
        assert!(site.pi_pop1.is_some(), "pi_pop1 should be calculated");
        assert!(site.pi_pop2.is_some(), "pi_pop2 should be calculated");
        
        if let (Some(dxy), Some(pi1), Some(pi2)) = (site.d_xy, site.pi_pop1, site.pi_pop2) {
            assert!((dxy - 1.0).abs() < 0.01, "D_xy should be ~1.0 for perfect structure, got {}", dxy);
            assert!(pi1.abs() < 0.01, "pi_pop1 should be ~0.0 for monomorphic population, got {}", pi1);
            assert!(pi2.abs() < 0.01, "pi_pop2 should be ~0.0 for monomorphic population, got {}", pi2);
        }
        
        // Regional FST should match per-site aggregation
        if let Some(regional_fst) = outcome.fst {
            if let Some(site_fst) = site.fst {
                assert!((regional_fst - site_fst).abs() < 0.01, 
                    "Regional FST {} should match site FST {} for single-site region", 
                    regional_fst, site_fst);
            }
        }
    }

    fn validate_falsta_content(content: &str) {
        let lines: Vec<&str> = content.lines().collect();
        
        // Should have headers and data lines
        assert!(!lines.is_empty(), "FALSTA file is empty");

        let mut found_hudson_header = false;
        let mut found_hudson_data = false;

        for (i, line) in lines.iter().enumerate() {
            if line.starts_with(">hudson_pairwise_fst_hap_0v1_chr_") {
                found_hudson_header = true;
                
                // Validate header format
                assert!(line.contains("chr_chr1_start_50_end_550"), 
                    "Hudson header format incorrect: {}", line);

                // Next line should be data
                if i + 1 < lines.len() {
                    let data_line = lines[i + 1];
                    found_hudson_data = true;
                    
                    // Validate data format
                    let values: Vec<&str> = data_line.split(',').collect();
                    assert!(!values.is_empty(), "Hudson data line is empty");
                    
                    // Should have 501 values (positions 50-550 inclusive)
                    assert_eq!(values.len(), 501, 
                        "Expected 501 values for region 50-550, got {}", values.len());

                    // Check that values are either "NA" or valid floats
                    let mut valid_fst_count = 0;
                    let mut na_count = 0;
                    
                    for (pos, value) in values.iter().enumerate() {
                        if *value == "NA" {
                            na_count += 1;
                        } else {
                            match value.parse::<f64>() {
                                Ok(fst_val) => {
                                    valid_fst_count += 1;
                                    // FST should be between -1 and 1 (though negative values are rare)
                                    assert!(fst_val >= -1.0 && fst_val <= 1.0, 
                                        "FST value {} at position {} is outside valid range [-1, 1]", 
                                        fst_val, pos + 50);
                                }
                                Err(_) => panic!("Invalid FST value '{}' at position {}", value, pos + 50),
                            }
                        }
                    }

                    println!("Hudson FST validation: {} valid FST values, {} NA values", 
                        valid_fst_count, na_count);
                    
                    // Should have some valid FST values (not all NA)
                    assert!(valid_fst_count > 0, "All Hudson FST values are NA");
                    
                    // Should have some variant sites with FST calculations
                    // Based on our test VCF, we have variants at positions 100, 200, 300, 400, 500
                    // These should have FST values, not NA
                    let variant_positions = [100, 200, 300, 400, 500];
                    for &var_pos in &variant_positions {
                        let idx = (var_pos - 50) as usize; // Convert to 0-based index in the array
                        if idx < values.len() {
                            let value = values[idx];
                            if value != "NA" {
                                let fst_val: f64 = value.parse().expect("Failed to parse FST value");
                                println!("Position {}: Hudson FST = {}", var_pos, fst_val);
                            }
                        }
                    }
                }
                break;
            }
        }

        assert!(found_hudson_header, "Hudson FST header not found in FALSTA file");
        assert!(found_hudson_data, "Hudson FST data not found in FALSTA file");
    }

    fn validate_csv_hudson_columns(content: &str) {
        let lines: Vec<&str> = content.lines().collect();
        assert!(!lines.is_empty(), "CSV file is empty");

        // Check header line
        let header = lines[0];
        println!("CSV header: {}", header);

        // Should contain Hudson FST columns
        assert!(header.contains("hudson_fst_hap_group_0v1"), 
            "CSV missing hudson_fst_hap_group_0v1 column");
        assert!(header.contains("hudson_dxy_hap_group_0v1"), 
            "CSV missing hudson_dxy_hap_group_0v1 column");
        assert!(header.contains("hudson_pi_hap_group_0"), 
            "CSV missing hudson_pi_hap_group_0 column");
        assert!(header.contains("hudson_pi_hap_group_1"), 
            "CSV missing hudson_pi_hap_group_1 column");

        // Check data line
        if lines.len() > 1 {
            let data_line = lines[1];
            println!("CSV data: {}", data_line);
            
            let values: Vec<&str> = data_line.split(',').collect();
            assert!(!values.is_empty(), "CSV data line is empty");

            // Find Hudson FST column and validate its value
            let header_cols: Vec<&str> = header.split(',').collect();
            for (i, col_name) in header_cols.iter().enumerate() {
                if col_name.contains("hudson_fst_hap_group_0v1") && i < values.len() {
                    let fst_value = values[i];
                    if fst_value != "NA" && !fst_value.is_empty() {
                        match fst_value.parse::<f64>() {
                            Ok(fst) => {
                                println!("Regional Hudson FST: {}", fst);
                                assert!(fst >= -1.0 && fst <= 1.0, 
                                    "Regional Hudson FST {} is outside valid range", fst);
                            }
                            Err(_) => panic!("Invalid Hudson FST value in CSV: '{}'", fst_value),
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_hudson_fst_missing_data() {
        // Test Hudson FST with missing genotypes
        let variant = create_test_variant(100, vec![
            Some(vec![0, 0]), // sample 0 - pop1
            None,             // sample 1 - pop1 (missing)
            Some(vec![1, 1]), // sample 2 - pop2
            Some(vec![1, 1]), // sample 3 - pop2
        ]);

        let pop1_haps = vec![(0, HaplotypeSide::Left), (0, HaplotypeSide::Right),
                            (1, HaplotypeSide::Left), (1, HaplotypeSide::Right)];
        let pop2_haps = vec![(2, HaplotypeSide::Left), (2, HaplotypeSide::Right),
                            (3, HaplotypeSide::Left), (3, HaplotypeSide::Right)];

        let sample_names = vec!["s1".to_string(), "s2".to_string(), "s3".to_string(), "s4".to_string()];
        
        let pop1_context = PopulationContext {
            id: PopulationId::HaplotypeGroup(0),
            haplotypes: pop1_haps,
            variants: &[variant.clone()],
            sample_names: &sample_names,
            sequence_length: 1000,
        };

        let pop2_context = PopulationContext {
            id: PopulationId::HaplotypeGroup(1),
            haplotypes: pop2_haps,
            variants: &[variant.clone()],
            sample_names: &sample_names,
            sequence_length: 1000,
        };

        let result = calculate_hudson_fst_for_pair(&pop1_context, &pop2_context);
        assert!(result.is_ok(), "Hudson FST calculation with missing data failed: {:?}", result.err());

        let outcome = result.unwrap();
        println!("Missing data FST: {:?}", outcome.fst);
        
        // Should still calculate FST using available data
        assert!(outcome.fst.is_some(), "FST should be calculated despite missing data");
        
        if let Some(fst) = outcome.fst {
            // With remaining data (pop1: 2 haplotypes all 0, pop2: 4 haplotypes all 1), 
            // should still show strong structure
            assert!(fst > 0.5, "FST {} should be high with remaining structured data", fst);
        }
    }
}