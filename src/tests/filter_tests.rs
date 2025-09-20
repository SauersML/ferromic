// tests/filter_tests.rs

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tempfile::tempdir;

#[test]
fn test_variant_filtering_unit() -> Result<(), Box<dyn std::error::Error>> {
    use crate::process::{process_variant, FilteringStats, MissingDataInfo, ZeroBasedHalfOpen};
    use parking_lot::Mutex;
    use std::collections::HashMap;

    // Test the core filtering functionality directly (unit test)
    let sample_names = vec!["SAMPLE1".to_string(), "SAMPLE2".to_string()];
    let mut missing_data_info = MissingDataInfo::default();
    let mut filtering_stats = FilteringStats::default();
    let position_allele_map = Mutex::new(HashMap::new());
    let region = ZeroBasedHalfOpen {
        start: 999,
        end: 2000,
    };

    // Test variant with high GQ values (should pass)
    let high_gq_variant = "chr1\t1000\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:50\t0|1:60";
    let result_high = process_variant(
        high_gq_variant,
        "1",
        region,
        &mut missing_data_info,
        &sample_names,
        30, // min_gq threshold
        &mut filtering_stats,
        None,
        None,
        &position_allele_map,
    );

    assert!(
        result_high.is_ok(),
        "High GQ variant should be processed successfully"
    );
    let (variant_high, is_valid_high) = result_high.unwrap().unwrap();
    assert!(is_valid_high, "High GQ variant should be marked as valid");
    assert_eq!(variant_high.position, 999); // 1-based to 0-based conversion

    // Test variant with low GQ values (should be filtered)
    let low_gq_variant = "chr1\t1001\t.\tA\tT\t.\tPASS\t.\tGT:GQ\t0|0:20\t0|1:25";
    let result_low = process_variant(
        low_gq_variant,
        "1",
        region,
        &mut missing_data_info,
        &sample_names,
        30, // min_gq threshold
        &mut filtering_stats,
        None,
        None,
        &position_allele_map,
    );

    assert!(
        result_low.is_ok(),
        "Low GQ variant should be processed successfully"
    );
    let (variant_low, is_valid_low) = result_low.unwrap().unwrap();
    assert!(
        !is_valid_low,
        "Low GQ variant should be marked as invalid (filtered)"
    );
    assert_eq!(variant_low.position, 1000); // 1-based to 0-based conversion

    // Verify filtering stats were updated
    assert!(
        filtering_stats.low_gq_variants > 0,
        "Filtering stats should record low GQ variants"
    );

    Ok(())
}

#[test]
fn test_variant_filtering_cli_integration() -> Result<(), Box<dyn std::error::Error>> {
    // Comprehensive CLI integration test - restored from original
    let dir = tempdir()?;
    let temp_path = dir.path();

    let allow_file_path = temp_path.join("test_allow.tsv");
    let config_file_path = temp_path.join("test_config.tsv");
    let vcf_folder_path = temp_path.join("vcfs_test");
    fs::create_dir(&vcf_folder_path)?;

    let output_file_path = temp_path.join("output_stats.csv");

    // Create test_allow.tsv
    let allow_content = "\
chr1\t100\t200
chr22\t900\t950
chr22\t1000\t2000
chr22\t2000\t3000
chr22\t10731880\t11731885
chr3\t1000\t20000
chr3\t200000\t200600
";
    fs::write(&allow_file_path, allow_content)?;

    // Create test_config.tsv
    let config_content = "\
seqnames\tstart\tend\tPOS\torig_ID\tverdict\tcateg\tHG00096\tHG00171\tHG00268
chr1\t1\t1000\t13113386\tchr1-13084312-INV-62181\tpass\tinv\t1|1\t1|0\t1|1
chr22\t100\t2000\t13257750\tchr22-13190915-INV-133672\tpass\tinv\t0|0\t1|0\t0|0
chr22\t2500\t5000\t13257755\tchr22-13190955-INV-133622\tpass\tinv\t0|0\t1|0\t0|1
chr22\t10711885\t10832100\t13257775\tchr22-13190975-INV-133672\tpass\tinv\t1|0\t1|0\t0|1
chr3\t500\t10010\t13257750\tchr3-13190915-INV-133672\tpass\tinv\t0|1\t0|0\t0|0
chr3\t5000\t6000\t13254750\tchr3-13180915-INV-133672\tpass\tinv\t0|1\t0|0\t0|0
chr3\t200100\t200900\t21204260\tchr3-21203898-INV-862\tpass\tinv\t0|0_lowconf\t0|0_lowconf\t0|1
chr17\t2000\t4000\t25346670\tchr17-25338356-INV-24067\tpass\tinv\t0|0\t0|0\t0|0
chr1\t26641622\t26646431\t26644026\tchr1-26639853-INV-8324\tMISO\tinv\t1|1\t1|1\t1|1
chr1\t43593641\t43594291\t43593966\tchr1-43593626-INV-710\tlowconf-Mendelfail\tinv\t0|1_lowconf\t1|1\t1|1
chr1\t60776841\t60778677\t60777759\tchr1-60775308-INV-5023\tpass\tinv\t0|1_lowconf\t0|0\t0|0
chr1\t81650508\t81707447\t81678978\tchr1-81642914-INV-66617\tpass\tinv\t0|0\t0|0\t0|0
";
    fs::write(&config_file_path, config_content)?;

    // Create simple VCF files for testing
    let vcf_header = "##fileformat=VCFv4.2\n##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"Genotype Quality\">\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tHG00096\tHG00171\tHG00268\n";

    // chr22.test.vcf
    let chr22_vcf_content = format!("{}chr22\t1234\t.\tG\tA\t.\tPASS\t.\tGT:GQ\t0|1:50\t1|0:60\t0|0:40\nchr22\t10731885\t.\tC\tT\t.\tPASS\t.\tGT:GQ\t0|0:50\t0|0:60\t0|0:40\n", vcf_header);
    fs::write(vcf_folder_path.join("chr22.test.vcf"), chr22_vcf_content)?;

    // chr3.test.vcf
    let chr3_vcf_content = format!("{}chr3\t10000\t.\tA\tG\t.\tPASS\t.\tGT:GQ\t0|0:50\t1|0:60\t0|0:40\nchr3\t200500\t.\tG\tA\t.\tPASS\t.\tGT:GQ\t0|0:20\t0|0:25\t0|0:30\n", vcf_header);
    fs::write(vcf_folder_path.join("chr3.test.vcf"), chr3_vcf_content)?;

    // chr17.test.vcf
    let chr17_vcf_content = format!(
        "{}chr17\t2400\t.\tG\tT\t.\tPASS\t.\tGT:GQ\t0|0:50\t1|0:60\t0|0:40\n",
        vcf_header
    );
    fs::write(vcf_folder_path.join("chr17.test.vcf"), chr17_vcf_content)?;

    // Create reference and GTF files
    let reference_file_path = temp_path.join("reference.fasta");
    let gtf_file_path = temp_path.join("annotations.gtf");

    // Generate reference file content for all chromosomes
    let sequence = "ACTACGTACGGATCG"; // Repeatable sequence pattern
    let mut reference_content = String::new();
    let mut gtf_content = String::new();

    for chr_num in 1..=22 {
        let full_sequence = sequence.repeat(1000); // Shorter for testing
        reference_content.push_str(&format!(">chr{}\n{}\n", chr_num, full_sequence));
        gtf_content.push_str(&format!(
            "chr{}\t.\tgene\t1\t1000\t.\t+\t.\tgene_id \"gene_chr{}\"; gene_name \"gene_chr{}\";\n",
            chr_num, chr_num, chr_num
        ));
    }

    // Chromosomes X and Y
    let full_sequence = sequence.repeat(1000); // Shorter for testing
    reference_content.push_str(&format!(
        ">chrX\n{}\n>chrY\n{}\n",
        full_sequence, full_sequence
    ));
    gtf_content.push_str(
        "chrX\t.\tgene\t1\t1000\t.\t+\t.\tgene_id \"gene_chrX\"; gene_name \"gene_chrX\";\n",
    );
    gtf_content.push_str(
        "chrY\t.\tgene\t1\t1000\t.\t+\t.\tgene_id \"gene_chrY\"; gene_name \"gene_chrY\";\n",
    );

    // Write the reference and GTF content to files
    fs::write(&reference_file_path, reference_content)?;
    fs::write(&gtf_file_path, gtf_content)?;

    // Determine the path to the run_vcf binary
    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let run_vcf_binary = if cfg!(windows) {
        project_root
            .join("target")
            .join("release")
            .join("run_vcf.exe")
    } else {
        project_root.join("target").join("release").join("run_vcf")
    };

    // Skip test if binary doesn't exist (not built in release mode)
    if !run_vcf_binary.exists() {
        println!(
            "Skipping CLI integration test - run_vcf binary not found at {:?}",
            run_vcf_binary
        );
        println!("Build with 'cargo build --release' to enable this test");
        return Ok(());
    }

    // Execute the run_vcf binary with updated arguments
    let mut cmd = Command::new(&run_vcf_binary);
    cmd.arg("--vcf_folder")
        .arg(&vcf_folder_path)
        .arg("--reference")
        .arg(&reference_file_path)
        .arg("--gtf")
        .arg(&gtf_file_path)
        .arg("--config_file")
        .arg(&config_file_path)
        .arg("--output_file")
        .arg(&output_file_path)
        .arg("--min_gq")
        .arg("30")
        .arg("--allow_file")
        .arg(&allow_file_path);

    // Execute and capture output
    let output = cmd.output()?;

    // Check that the command executed successfully
    if !output.status.success() {
        println!("Command failed with status: {}", output.status);
        println!("Stdout: {}", String::from_utf8_lossy(&output.stdout));
        println!("Stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!("CLI command failed");
    }

    // Verify that output file was created
    assert!(
        output_file_path.exists(),
        "Output CSV file should be created"
    );

    // Read and validate output file exists and has content
    let output_csv = fs::read_to_string(&output_file_path)?;
    assert!(!output_csv.is_empty(), "Output CSV should not be empty");

    // Basic validation - should have header line
    assert!(
        output_csv.contains("chromosome") || output_csv.contains("chr"),
        "Output CSV should contain chromosome information"
    );

    // Clean up
    dir.close()?;

    Ok(())
}
