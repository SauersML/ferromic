
use crate::transcripts::{prepare_to_write_cds, TranscriptAnnotationCDS};
use crate::process::ZeroBasedHalfOpen;
use std::collections::HashMap;
use tempfile::TempDir;

#[test]
fn test_partial_overlap_writes_to_special_dir() {
    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path();

    let hap_group = 0;

    // Create a dummy CDS region
    let cds = TranscriptAnnotationCDS {
        transcript_id: "ENST_PARTIAL".to_string(),
        gene_id: "GENE_PARTIAL".to_string(),
        gene_name: "GENE_PARTIAL".to_string(),
        strand: '+',
        frames: vec![0],
        segments: vec![ZeroBasedHalfOpen { start: 100, end: 200 }],
    };

    // Case 1: Partial Overlap
    // Inversion is 150-250. CDS is 100-202 (length 102, divisible by 3).
    // Overlap is 150-202.
    // CDS is NOT fully inside inversion (100 < 150).
    // Update CDS to have valid length
    let mut cds_valid = cds.clone();
    cds_valid.segments = vec![ZeroBasedHalfOpen { start: 100, end: 202 }];

    let inversion_interval = ZeroBasedHalfOpen { start: 150, end: 250 };
    // hap_region must include everything we care about for the function to find segments.
    // Let's say hap_region is 0-300.
    let hap_region = ZeroBasedHalfOpen { start: 0, end: 300 };

    let mut hap_sequences = HashMap::new();
    // 300 bytes of 'A' (AAA is valid codon)
    let mut seq_data = vec![b'A'; 300];
    // Must start with ATG at CDS start (100)
    seq_data[100] = b'A';
    seq_data[101] = b'T';
    seq_data[102] = b'G';

    hap_sequences.insert("Sample1_L".to_string(), seq_data);

    prepare_to_write_cds(
        hap_group,
        &[cds_valid.clone()],
        &hap_sequences,
        "1",
        hap_region,
        inversion_interval,
        temp_path,
    ).unwrap();

    // Check if "partial_overlap" directory exists
    let partial_dir = temp_path.join("partial_overlap");
    assert!(partial_dir.exists(), "partial_overlap directory should be created");

    // The file should be in partial_overlap, not in temp_path root
    // Filename format: group0_GENEPARTIAL_GENEPARTIAL_ENSTPARTIAL_chr1_cds_start101_cds_end202_inv_start151_inv_end250.phy.gz
    // Note: sanitize_id removes underscores from IDs.
    let filename = "group0_GENEPARTIAL_GENEPARTIAL_ENSTPARTIAL_chr1_cds_start101_cds_end202_inv_start151_inv_end250.phy.gz";

    let file_in_partial = partial_dir.join(filename);
    assert!(file_in_partial.exists(), "File should exist in partial_overlap directory");

    let file_in_root = temp_path.join(filename);
    assert!(!file_in_root.exists(), "File should NOT exist in root directory");

    // Case 2: Full Overlap (Inside)
    // Inversion 50-250. CDS 100-202. Fully inside.
    let inversion_interval_full = ZeroBasedHalfOpen { start: 50, end: 250 };

    let cds_full = TranscriptAnnotationCDS {
        transcript_id: "ENST_FULL".to_string(),
        gene_id: "GENE_FULL".to_string(),
        gene_name: "GENE_FULL".to_string(),
        strand: '+',
        frames: vec![0],
        segments: vec![ZeroBasedHalfOpen { start: 100, end: 202 }],
    };

    prepare_to_write_cds(
        hap_group,
        &[cds_full.clone()],
        &hap_sequences,
        "1",
        hap_region,
        inversion_interval_full,
        temp_path,
    ).unwrap();

    let filename_full = "group0_GENEFULL_GENEFULL_ENSTFULL_chr1_cds_start101_cds_end202_inv_start51_inv_end250.phy.gz";
    let file_full_in_root = temp_path.join(filename_full);
    assert!(file_full_in_root.exists(), "Full overlap file should exist in root directory");

    let file_full_in_partial = partial_dir.join(filename_full);
    assert!(!file_full_in_partial.exists(), "Full overlap file should NOT exist in partial directory");

}
