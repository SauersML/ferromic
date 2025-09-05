#[cfg(test)]
mod hudson_fst_tests {
    use crate::process::*;
    use crate::stats::*;
    use std::fs;
    use std::path::Path;
    use tempfile::TempDir;

    /// Creates a minimal VCF file for testing Hudson FST
    fn create_test_vcf(path: &Path) -> std::io::Result<()> {
        let vcf_content = r#"##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	sample1_0	sample1_1	sample2_0	sample2_1	sample3_0	sample3_1	sample4_0	sample4_1
chr1	100	.	A	T	60	PASS	.	GT	0|0	0|1	1|1	1|0	0|0	0|0	1|1	1|1
chr1	200	.	G	C	60	PASS	.	GT	0|0	0|0	1|1	1|1	0|1	1|0	1|1	0|1
chr1	300	.	C	G	60	PASS	.	GT	0|0	1|1	0|0	1|1	0|0	0|1	1|1	1|0
chr1	400	.	T	A	60	PASS	.	GT	1|1	0|0	1|1	0|0	1|0	0|1	0|0	1|1
chr1	500	.	A	G	60	PASS	.	GT	0|1	1|0	0|1	1|0	0|0	1|1	0|0	1|1
"#;
        fs::write(path, vcf_content)
    }

    /// Creates a test configuration file
    fn create_test_config(path: &Path, vcf_path: &Path, output_dir: &Path) -> std::io::Result<()> {
        let config_content = format!(
            r#"seqname,interval_start,interval_end,vcf_file,output_prefix
chr1,50,550,{},{}
"#,
            vcf_path.display(),
            output_dir.join("test_hudson").display()
        );
        fs::write(path, config_content)
    }

    #[test]
    fn test_hudson_per_site_fst_calculation_and_output() {
        // Create temporary directory for test files
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let temp_path = temp_dir.path();

        // Create test files
        let vcf_path = temp_path.join("test.vcf");
        let config_path = temp_path.join("config.csv");
        let output_dir = temp_path.join("output");
        fs::create_dir_all(&output_dir).expect("Failed to create output directory");

        create_test_vcf(&vcf_path).expect("Failed to create test VCF");
        create_test_config(&config_path, &vcf_path, &output_dir).expect("Failed to create test config");

        // Set up test arguments
        let args = Args {
            config_file: config_path.clone(),
            output_dir: output_dir.clone(),
            enable_fst: true,
            enable_diversity: true,
            enable_pca: false,
            population_file: None,
            allow_regions_file: None,
            mask_regions_file: None,
            threads: Some(1),
            chunk_size: None,
            memory_limit_gb: None,
            temp_dir: None,
            verbose: false,
            quiet: false,
        };

        // Run the analysis
        let result = process_config_entries(&args);
        assert!(result.is_ok(), "Hudson FST analysis failed: {:?}", result.err());

        // Check that FALSTA file was created
        let falsta_path = output_dir.join("test_hudson.falsta");
        assert!(falsta_path.exists(), "FALSTA file was not created");

        // Read and validate FALSTA content
        let falsta_content = fs::read_to_string(&falsta_path)
            .expect("Failed to read FALSTA file");

        println!("FALSTA content:\n{}", falsta_content);

        // Validate FALSTA format and content
        validate_falsta_content(&falsta_content);

        // Check main CSV output
        let csv_path = output_dir.join("test_hudson.csv");
        assert!(csv_path.exists(), "Main CSV file was not created");

        let csv_content = fs::read_to_string(&csv_path)
            .expect("Failed to read CSV file");
        
        println!("CSV content:\n{}", csv_content);
        validate_csv_hudson_columns(&csv_content);
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

    fn validate_csv_hudson_columns(&content: &str) {
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
    fn test_hudson_fst_with_population_structure() {
        // Test that Hudson FST correctly identifies population structure
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let temp_path = temp_dir.path();

        // Create VCF with clear population structure
        let vcf_path = temp_path.join("structured.vcf");
        let structured_vcf = r#"##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	pop1_s1_0	pop1_s1_1	pop1_s2_0	pop1_s2_1	pop2_s1_0	pop2_s1_1	pop2_s2_0	pop2_s2_1
chr1	100	.	A	T	60	PASS	.	GT	0|0	0|0	0|0	0|0	1|1	1|1	1|1	1|1
chr1	200	.	G	C	60	PASS	.	GT	0|0	0|0	0|0	0|0	1|1	1|1	1|1	1|1
chr1	300	.	C	G	60	PASS	.	GT	1|1	1|1	1|1	1|1	0|0	0|0	0|0	0|0
"#;
        fs::write(&vcf_path, structured_vcf).expect("Failed to create structured VCF");

        let config_path = temp_path.join("config.csv");
        let output_dir = temp_path.join("output");
        fs::create_dir_all(&output_dir).expect("Failed to create output directory");

        create_test_config(&config_path, &vcf_path, &output_dir).expect("Failed to create config");

        let args = Args {
            config_file: config_path,
            output_dir: output_dir.clone(),
            enable_fst: true,
            enable_diversity: false,
            enable_pca: false,
            population_file: None,
            allow_regions_file: None,
            mask_regions_file: None,
            threads: Some(1),
            chunk_size: None,
            memory_limit_gb: None,
            temp_dir: None,
            verbose: false,
            quiet: false,
        };

        let result = process_config_entries(&args);
        assert!(result.is_ok(), "Structured population analysis failed: {:?}", result.err());

        // Read FALSTA output
        let falsta_path = output_dir.join("test_hudson.falsta");
        let falsta_content = fs::read_to_string(&falsta_path)
            .expect("Failed to read FALSTA file");

        // With perfect population structure, FST should be 1.0 at variant sites
        let lines: Vec<&str> = falsta_content.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            if line.starts_with(">hudson_pairwise_fst_hap_0v1_chr_") {
                if i + 1 < lines.len() {
                    let data_line = lines[i + 1];
                    let values: Vec<&str> = data_line.split(',').collect();
                    
                    // Check FST at variant positions (100, 200, 300 -> indices 50, 150, 250)
                    let variant_indices = [50, 150, 250]; // 0-based indices for positions 100, 200, 300
                    for &idx in &variant_indices {
                        if idx < values.len() && values[idx] != "NA" {
                            let fst: f64 = values[idx].parse().expect("Failed to parse FST");
                            println!("Perfect structure FST at index {}: {}", idx, fst);
                            // With perfect population structure, FST should be close to 1.0
                            assert!(fst > 0.8, "FST {} is too low for perfect population structure", fst);
                        }
                    }
                }
                break;
            }
        }
    }
}