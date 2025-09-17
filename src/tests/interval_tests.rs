use crate::process::{ZeroBasedHalfOpen, ZeroBasedPosition};

#[test]
fn from_0based_inclusive_handles_end_before_start() {
    let interval = ZeroBasedHalfOpen::from_0based_inclusive(10, 5);
    assert_eq!(interval.start, 10);
    assert_eq!(interval.end, 10);
    assert_eq!(interval.len(), 0);
}

#[test]
fn from_0based_inclusive_clamps_negative_coordinates() {
    let interval = ZeroBasedHalfOpen::from_0based_inclusive(-5, -3);
    assert_eq!(interval.start, 0);
    assert_eq!(interval.end, 0);
}

#[test]
fn from_0based_inclusive_keeps_valid_range() {
    let interval = ZeroBasedHalfOpen::from_0based_inclusive(-5, 2);
    assert_eq!(interval.start, 0);
    assert_eq!(interval.end, 3);
}

#[test]
fn from_0based_inclusive_single_point_has_length_one() {
    let interval = ZeroBasedHalfOpen::from_0based_inclusive(7, 7);
    assert_eq!(interval.start, 7);
    assert_eq!(interval.end, 8);
    assert_eq!(interval.len(), 1);
}

#[test]
fn from_0based_inclusive_allows_large_positive_inputs_without_overflow() {
    let interval = ZeroBasedHalfOpen::from_0based_inclusive(i64::MAX, i64::MAX);
    assert!(interval.end >= interval.start);
    assert_eq!(interval.len(), 0);
}

#[test]
fn from_0based_inclusive_produces_expected_half_open_length() {
    let interval = ZeroBasedHalfOpen::from_0based_inclusive(2, 6);
    assert_eq!(interval.start, 2);
    assert_eq!(interval.end, 7);
    assert_eq!(interval.len(), 5);
}

#[test]
fn from_0based_point_matches_inclusive_conversion() {
    let inclusive = ZeroBasedHalfOpen::from_0based_inclusive(12, 12);
    let point = ZeroBasedHalfOpen::from_0based_point(12);
    assert_eq!(inclusive.start, point.start);
    assert_eq!(inclusive.end, point.end);
}

#[test]
fn len_returns_zero_for_reversed_intervals() {
    let interval = ZeroBasedHalfOpen { start: 10, end: 5 };
    assert_eq!(interval.len(), 0);
}

#[test]
fn from_1based_inclusive_clamps_start_below_one() {
    let interval = ZeroBasedHalfOpen::from_1based_inclusive(-3, 5);
    assert_eq!(interval.start, 0);
    assert_eq!(interval.end, 5);
}

#[test]
fn from_1based_inclusive_clamps_end_before_start() {
    let interval = ZeroBasedHalfOpen::from_1based_inclusive(5, 2);
    assert_eq!(interval.start, 4);
    assert_eq!(interval.end, 5);
    assert_eq!(interval.len(), 1);
}

#[test]
fn zero_based_position_to_one_based_matches_expectations() {
    let position = ZeroBasedPosition(0);
    assert_eq!(position.to_one_based(), 1);
    let position = ZeroBasedPosition(41);
    assert_eq!(position.to_one_based(), 42);
}

#[test]
fn inclusive_and_half_open_conversions_align() {
    let inclusive = ZeroBasedHalfOpen::from_0based_inclusive(4, 9);
    let one_based = ZeroBasedHalfOpen::from_1based_inclusive(5, 10);
    assert_eq!(inclusive.start, one_based.start);
    assert_eq!(inclusive.end, one_based.end);
}

#[test]
fn half_open_intervals_slice_ascii_sequences() {
    let dna = b"ACGTACGT";
    let interval = ZeroBasedHalfOpen::from_0based_inclusive(2, 5);
    assert_eq!(&dna[interval.start..interval.end], b"GTAC");
}
