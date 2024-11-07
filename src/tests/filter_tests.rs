// tests/filter_tests.rs

use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tempfile::tempdir;

#[test]
fn test_variant_filtering_output() -> Result<(), Box<dyn std::error::Error>> {
    // Create a temporary directory for the test environment
    let dir = tempdir()?;
    let temp_path = dir.path();

    let allow_file_path = temp_path.join("test_allow.tsv");
    let config_file_path = temp_path.join("test_config.tsv");
    let vcf_folder_path = temp_path.join("vcfs_test");
    fs::create_dir(&vcf_folder_path)?;

    let output_file_path = temp_path.join("output_stats.csv");

    // ../test_allow.tsv
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

    // ../test_config.tsv
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

    // ../vcfs_test/chr22.test.vcf
    let chr22_vcf_content = "\
##fileformat=VCFv4.2
##INFO=<ID=AC,Number=A,Type=Integer,Description=\"Allele count in genotypes\">
##INFO=<ID=AN,Number=1,Type=Integer,Description=\"Total number of alleles in called genotypes\">
##FORMAT=<ID=AD,Number=2,Type=Integer,Description=\"Allelic depths (number of reads in each observed allele)\">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Total read depth\">
##FORMAT=<ID=FT,Number=1,Type=String,Description=\"Variant filters\">
##FORMAT=<ID=QUAL,Number=1,Type=Float,Description=\".\">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"Phred scaled genotype quality computed by whatshap genotyping algorithm.\">
##FORMAT=<ID=GL,Number=G,Type=Float,Description=\"log10-scaled likelihoods for genotypes: 0/0,0/1,1/1, computed by whatshap genotyping algorithm.\">
##FORMAT=<ID=PS,Number=1,Type=Integer,Description=\"Phase set identifier\">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	EUR_GBR_HG00096	EUR_FIN_HG00171	EUR_FIN_HG00268
chr22	1234	.	G	A	.	.	AA=C;VT=SNP;AN=6;AC=0	GT:GQ:GL:PS	0|1:469:0,-42.9443,-213.646:.	1|0:498:0,-49.7769,-315.657:.	0|0:276:0,-27.5614,-170.553:.
chr22	1253	.	G	T	.	.	AA=C;VT=SNP;AN=6;AC=0	GT:GQ:GL:PS	0|1:469:0,-46.9343,-213.646:.	1|0:498:0,-49.7769,-315.657:.	1|1:276:0,-27.5614,-151.073:.
chr22	10731885	.	C	T	.	.	AA=C;VT=SNP;AN=6;AC=0	GT:GQ:GL:PS	0|0:469:0,-46.9443,-213.646:.	0|0:498:0,-49.7769,-315.657:.	0|0:276:0,-27.5614,-141.056:.
chr22	10732039	.	C	T	.	.	AA=C;VT=SNP;AN=6;AC=0	GT:GQ:GL:PS	0|0:438:0,-40.7527,-202.874:.	1|0:480:0,-48.0367,-309.11:.	0|0:323:0,-32.3138,-164.506:.
chr22	10832039	.	A	G	.	.	AA=C;VT=SNP;AN=6;AC=0	GT:GQ:GL:PS	0|0:408:0,-41.7547,-202.874:.	0|1:480:0,-48.0367,-309.11:.	0|0:323:0,-32.3138,-162.569:.
";
    fs::write(vcf_folder_path.join("chr22.test.vcf"), chr22_vcf_content)?;

    // ../vcfs_test/chr3.test.vcf
    let chr3_vcf_content = "\
##fileformat=VCFv4.2
##INFO=<ID=AC,Number=A,Type=Integer,Description=\"Allele count in genotypes\">
##INFO=<ID=AN,Number=1,Type=Integer,Description=\"Total number of alleles in called genotypes\">
##FORMAT=<ID=AD,Number=2,Type=Integer,Description=\"Allelic depths (number of reads in each observed allele)\">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Total read depth\">
##FORMAT=<ID=FT,Number=1,Type=String,Description=\"Variant filters\">
##FORMAT=<ID=QUAL,Number=1,Type=Float,Description=\".\">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"Phred scaled genotype quality computed by whatshap genotyping algorithm.\">
##FORMAT=<ID=GL,Number=G,Type=Float,Description=\"log10-scaled likelihoods for genotypes: 0/0,0/1,1/1, computed by whatshap genotyping algorithm.\">
##FORMAT=<ID=PS,Number=1,Type=Integer,Description=\"Phase set identifier\">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	EUR_GBR_HG00096	EUR_FIN_HG00171	EUR_FIN_HG00268
chr3	10000	.	A	G	.	.	AA=C;VT=SNP;AN=6;AC=0	GT:GQ:GL:PS	0|0:469:0,-46.9443,-213.646:.	1|0:498:0,-49.7769,-315.657:.	0|0:276:0,-27.5614,-170.553:.
chr3	10100	.	G	A	.	.	AA=C;VT=SNP;AN=6;AC=0	GT:GQ:GL:PS	0|0:469:0,-46.9443,-213.646:.	1|0:498:0,-49.7769,-315.657:.	0|0:276:0,-27.5614,-151.073:.
chr3	200400	.	C	T	.	.	AA=C;VT=SNP;AN=6;AC=0	GT:GQ:GL:PS	0|0:469:0,-46.9443,-213.646:.	0|0:498:0,-49.7769,-315.657:.	0|1:276:0,-27.5614,-141.056:.
chr3	200500	.	G	A	.	.	AA=C;VT=SNP;AN=6;AC=0	GT:GQ:GL:PS	0|0:20:0,-40.7527,-150.874:.	0|0:480:0,-48.0367,-309.11:.	0|0:323:0,-32.3138,-164.506:.
chr3	200700	.	A	T	.	.	AA=C;VT=SNP;AN=6;AC=0	GT:GQ:GL:PS	0|1:408:0,-40.7527,-202.874:.	0|0:480:0,-48.0367,-309.11:.	0|1:323:0,-32.3138,-162.569:.
";
    fs::write(vcf_folder_path.join("chr3.test.vcf"), chr3_vcf_content)?;

    // ../vcfs_test/chr17.test.vcf
    let chr17_vcf_content = "\
##fileformat=VCFv4.2
##INFO=<ID=AC,Number=A,Type=Integer,Description=\"Allele count in genotypes\">
##INFO=<ID=AN,Number=1,Type=Integer,Description=\"Total number of alleles in called genotypes\">
##FORMAT=<ID=AD,Number=2,Type=Integer,Description=\"Allelic depths (number of reads in each observed allele)\">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Total read depth\">
##FORMAT=<ID=FT,Number=1,Type=String,Description=\"Variant filters\">
##FORMAT=<ID=QUAL,Number=1,Type=Float,Description=\".\">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"Phred scaled genotype quality computed by whatshap genotyping algorithm.\">
##FORMAT=<ID=GL,Number=G,Type=Float,Description=\"log10-scaled likelihoods for genotypes: 0/0,0/1,1/1, computed by whatshap genotyping algorithm.\">
##FORMAT=<ID=PS,Number=1,Type=Integer,Description=\"Phase set identifier\">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	EUR_GBR_HG00096	EUR_FIN_HG00171	EUR_FIN_HG00268
chr17	150	    .	A	G	.	.	AA=C;VT=SNP;AN=6;AC=0	GT:GQ:GL:PS	0|0:469:0,-46.9443,-213.646:.	0|0:498:0,-49.7769,-315.657:.	0|0:276:0,-27.5614,-141.056:.
chr17	2400	.	G	T	.	.	AA=C;VT=SNP;AN=6;AC=0	GT:GQ:GL:PS	0|0:452:0,-46.9443,-213.646:.	1|0:498:0,-49.7769,-315.657:.	0|0:276:0,-27.5614,-170.553:.
chr17	2800	.	G	T	.	.	AA=C;VT=SNP;AN=6;AC=0	GT:GQ:GL:PS	0|0:479:0,-46.9443,-213.646:.	0|0:498:0,-49.7769,-315.657:.	0|0:276:0,-27.5614,-151.073:.
chr17	3100	.	A	G	.	.	AA=C;VT=SNP;AN=6;AC=0	GT:GQ:GL:PS	0|0:469:0,-46.9443,-213.646:.	0|0:498:0,-49.7769,-315.657:.	0|0:276:0,-27.5614,-141.056:.
chr17	3600	.	T	A	.	.	AA=C;VT=SNP;AN=6;AC=0	GT:GQ:GL:PS	0|0:408:0,-40.7527,-202.874:.	0|0:480:0,-48.0367,-309.11:.	0|0:323:0,-32.3138,-164.506:.
chr17	3900	.	G	C	.	.	AA=C;VT=SNP;AN=6;AC=0	GT:GQ:GL:PS	0|0:408:0,-40.7527,-202.874:.	0|0:480:0,-48.0367,-309.11:.	0|1:323:0,-32.3138,-162.569:.
chr17	4400	.	G	T	.	.	AA=C;VT=SNP;AN=6;AC=0	GT:GQ:GL:PS	0|0:452:0,-46.9443,-213.646:.	1|0:498:0,-49.7769,-315.657:.	0|0:276:0,-27.5614,-170.553:.
chr17	5800	.	G	T	.	.	AA=C;VT=SNP;AN=6;AC=0	GT:GQ:GL:PS	0|0:479:0,-46.9443,-213.646:.	0|0:498:0,-49.7769,-315.657:.	0|0:276:0,-27.5614,-151.073:.
chr17	6100	.	A	G	.	.	AA=C;VT=SNP;AN=6;AC=0	GT:GQ:GL:PS	0|0:469:0,-46.9443,-213.646:.	0|0:498:0,-49.7769,-315.657:.	0|0:276:0,-27.5614,-141.056:.
chr17	7600	.	T	A	.	.	AA=C;VT=SNP;AN=6;AC=0	GT:GQ:GL:PS	0|0:408:0,-40.7527,-202.874:.	0|0:480:0,-48.0367,-309.11:.	0|0:323:0,-32.3138,-164.506:.
chr17	10910	.	G	C	.	.	AA=C;VT=SNP;AN=6;AC=0	GT:GQ:GL:PS	0|0:408:0,-40.7527,-202.874:.	0|0:480:0,-48.0367,-309.11:.	0|1:323:0,-32.3138,-162.569:.
chr17	36004	.	T	A	.	.	AA=C;VT=SNP;AN=6;AC=0	GT:GQ:GL:PS	0|0:408:0,-40.7527,-202.874:.	0|0:480:0,-48.0367,-309.11:.	0|0:323:0,-32.3138,-164.506:.
chr17	39003	.	G	C	.	.	AA=C;VT=SNP;AN=6;AC=0	GT:GQ:GL:PS	0|0:408:0,-40.7527,-202.874:.	0|0:480:0,-48.0367,-309.11:.	0|1:323:0,-32.3138,-162.569:.
chr17	44002	.	G	T	.	.	AA=C;VT=SNP;AN=6;AC=0	GT:GQ:GL:PS	0|0:452:0,-46.9443,-213.646:.	1|0:498:0,-49.7769,-315.657:.	0|0:276:0,-27.5614,-170.553:.
chr17	58001	.	G	T	.	.	AA=C;VT=SNP;AN=6;AC=0	GT:GQ:GL:PS	0|0:479:0,-46.9443,-213.646:.	0|0:498:0,-49.7769,-315.657:.	0|0:276:0,-27.5614,-151.073:.
";
    fs::write(vcf_folder_path.join("chr17.test.vcf"), chr17_vcf_content)?;

    // Determine the path to the `vcf_stats` binary
    // This assumes that the test is being run from the project root and the binary is built in release mode
    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let vcf_stats_binary = if cfg!(windows) {
        project_root.join("target").join("release").join("vcf_stats.exe")
    } else {
        project_root.join("target").join("release").join("vcf_stats")
    };

    assert!(
        vcf_stats_binary.exists(),
        "vcf_stats binary not found at {:?}. Please build the project before running tests using `cargo build --release`.",
        vcf_stats_binary
    );

    // Create temporary reference and GFF files
    let reference_file_path = temp_path.join("reference.fasta");
    let gff_file_path = temp_path.join("annotations.gff");
    
    // Write content to the reference and GFF files
    let long_sequence = "ACTACGTACGGATCG".repeat(81708457);
    fs::write(&reference_file_path, format!(">chr1\n{}", long_sequence))?;
    fs::write(&gff_file_path, "chr1\t.\tgene\t1\t1000\t.\t+\t.\tID=gene0;Name=gene0")?;

    // Execute the `vcf_stats` binary with the test files as arguments
    let mut cmd = Command::new(&vcf_stats_binary);
    cmd.arg("--vcf_folder")
        .arg(&vcf_folder_path)
        .arg("--reference")
        .arg(&reference_file_path)
        .arg("--gff")
        .arg(&gff_file_path)
        .arg("--config_file")
        .arg(&config_file_path)
        .arg("--output_file")
        .arg(&output_file_path)
        .arg("--min_gq")
        .arg("30")
        .arg("--allow_file")
        .arg(&allow_file_path);


    // Capture and assert the `stdout` contains the expected output statements
    cmd.assert()
        .success()
        .stderr(predicate::str::contains("Error finding VCF file for 1: NoVcfFiles"))
        .stdout(predicate::str::contains("Filtering Statistics:"))
        .stdout(predicate::str::contains("Total variants processed: 5"))
        .stdout(predicate::str::contains("Filtered due to allow: 0"))
        .stdout(predicate::str::contains("Filtered due to mask: 0"))
        .stdout(predicate::str::contains("Low GQ variants: 0"))
        .stdout(predicate::str::contains("Filtered variants: 0 (0.00%)"))
        .stdout(predicate::str::contains("Filtered due to allow: 1"))
        .stdout(predicate::str::contains("Low GQ variants: 1"))
        .stdout(predicate::str::contains("Filtered variants: 2 (40.00%)"))
        .stdout(predicate::str::contains("Filtered due to allow: 5"))
        .stdout(predicate::str::contains("Filtered variants: 5 (100.00%)"));

    let output_csv = fs::read_to_string(&output_file_path)?;

    // Clean up the temporary directory
    dir.close()?;

    Ok(())
}
