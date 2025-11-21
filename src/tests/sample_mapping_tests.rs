use crate::process::map_sample_names_to_indices;

#[test]
fn maps_core_and_full_sample_ids() {
    let samples = vec![
        "AFR_ACB_HG12345".to_string(),
        "HG54321".to_string(),
    ];

    let mapping = map_sample_names_to_indices(&samples).expect("mapping should succeed");

    // Core ID should resolve to the same index as the full VCF name
    assert_eq!(mapping.get("HG12345"), Some(&0));
    assert_eq!(mapping.get("AFR_ACB_HG12345"), Some(&0));

    // Entries without prefixes should still work
    assert_eq!(mapping.get("HG54321"), Some(&1));

}
