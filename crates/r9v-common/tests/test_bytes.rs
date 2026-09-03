// SPDX-License-Identifier: Apache-2.0
//! Tests for strict byte-size parsing and formatting (Spec 12 §3, Spec 14 §2, CONVENTIONS.md §1.3).

use std::str::FromStr;

use r9v_common::{
    format_byte_size, parse_byte_size, ByteSize, ByteSizeError, GIB, KIB, MIB, PIB, TIB,
};

#[test]
fn parse_valid_plain_numbers() {
    assert_eq!(parse_byte_size("0").unwrap(), 0);
    assert_eq!(parse_byte_size("1024").unwrap(), 1024);
    assert_eq!(parse_byte_size("1024B").unwrap(), 1024);
    assert_eq!(parse_byte_size("1024 B").unwrap(), 1024);
    assert_eq!(parse_byte_size("  500 bytes ").unwrap(), 500);
}

#[test]
fn parse_valid_binary_units() {
    assert_eq!(parse_byte_size("4KiB").unwrap(), 4 * KIB);
    assert_eq!(parse_byte_size("4 KiB").unwrap(), 4 * KIB);
    assert_eq!(parse_byte_size("512MiB").unwrap(), 512 * MIB);
    assert_eq!(parse_byte_size("512 MiB").unwrap(), 512 * MIB);
    assert_eq!(parse_byte_size("32GiB").unwrap(), 32 * GIB);
    assert_eq!(parse_byte_size("32 GiB").unwrap(), 32 * GIB);
    assert_eq!(parse_byte_size("2TiB").unwrap(), 2 * TIB);
    assert_eq!(parse_byte_size("1PiB").unwrap(), PIB);
}

#[test]
fn parse_valid_shorthand_units_and_case_insensitivity() {
    // Unqualified memory-unit aliases follow the binary-unit decision documented by the parser.
    assert_eq!(parse_byte_size("4K").unwrap(), 4 * KIB);
    assert_eq!(parse_byte_size("4KB").unwrap(), 4 * KIB);
    assert_eq!(parse_byte_size("4kb").unwrap(), 4 * KIB);
    assert_eq!(parse_byte_size("512M").unwrap(), 512 * MIB);
    assert_eq!(parse_byte_size("512MB").unwrap(), 512 * MIB);
    assert_eq!(parse_byte_size("512mb").unwrap(), 512 * MIB);
    assert_eq!(parse_byte_size("4G").unwrap(), 4 * GIB);
    assert_eq!(parse_byte_size("4GB").unwrap(), 4 * GIB);
    assert_eq!(parse_byte_size("4gb").unwrap(), 4 * GIB);
    assert_eq!(parse_byte_size("2T").unwrap(), 2 * TIB);
    assert_eq!(parse_byte_size("2TB").unwrap(), 2 * TIB);
}

#[test]
fn parse_valid_fractional_sizes_with_exact_integral_bytes() {
    // Exact decimal calculations
    assert_eq!(parse_byte_size("1.5 GiB").unwrap(), 1_610_612_736);
    assert_eq!(parse_byte_size("0.5 MiB").unwrap(), 524_288);
    assert_eq!(parse_byte_size("2.25 KiB").unwrap(), 2_304);
    assert_eq!(parse_byte_size("1.000 GiB").unwrap(), 1_073_741_824);
}

#[test]
fn parse_failure_fractional_byte() {
    // Non-integral byte amounts must be rejected with FractionalByte
    assert!(matches!(
        parse_byte_size("0.5 B"),
        Err(ByteSizeError::FractionalByte { .. })
    ));
    assert!(matches!(
        parse_byte_size("1.1 B"),
        Err(ByteSizeError::FractionalByte { .. })
    ));
    assert!(matches!(
        parse_byte_size("1.3 KiB"),
        Err(ByteSizeError::FractionalByte { .. })
    ));
    assert!(matches!(
        parse_byte_size("1.1 GiB"),
        Err(ByteSizeError::FractionalByte { .. })
    ));
}

#[test]
fn parse_failure_empty_or_whitespace() {
    assert_eq!(parse_byte_size(""), Err(ByteSizeError::Empty));
    assert_eq!(parse_byte_size("   \t\n "), Err(ByteSizeError::Empty));
}

#[test]
fn parse_failure_missing_number() {
    assert!(matches!(
        parse_byte_size("GB"),
        Err(ByteSizeError::MissingNumber { .. })
    ));
    assert!(matches!(
        parse_byte_size("bytes"),
        Err(ByteSizeError::MissingNumber { .. })
    ));
}

#[test]
fn parse_failure_negative() {
    assert!(matches!(
        parse_byte_size("-1024"),
        Err(ByteSizeError::Negative { .. })
    ));
    assert!(matches!(
        parse_byte_size("-4GB"),
        Err(ByteSizeError::Negative { .. })
    ));
}

#[test]
fn parse_failure_invalid_number() {
    assert!(matches!(
        parse_byte_size("1.2.3 GB"),
        Err(ByteSizeError::InvalidNumber { .. })
    ));
}

#[test]
fn parse_failure_unknown_unit() {
    assert!(matches!(
        parse_byte_size("4 foo"),
        Err(ByteSizeError::UnknownUnit { unit, .. }) if unit == "foo"
    ));
    assert!(matches!(
        parse_byte_size("16 Q"),
        Err(ByteSizeError::UnknownUnit { unit, .. }) if unit == "Q"
    ));
}

#[test]
fn parse_failure_trailing_characters() {
    assert!(matches!(
        parse_byte_size("4 GB extra"),
        Err(ByteSizeError::TrailingCharacters { trailing, .. }) if trailing == "extra"
    ));
}

#[test]
fn parse_failure_overflow() {
    let huge = format!("{} GiB", u64::MAX);
    assert!(matches!(
        parse_byte_size(&huge),
        Err(ByteSizeError::Overflow { .. })
    ));

    // u64::MAX parses successfully
    assert_eq!(parse_byte_size("18446744073709551615 B").unwrap(), u64::MAX);

    // u64::MAX + 1 classified as Overflow
    assert!(matches!(
        parse_byte_size("18446744073709551616 B"),
        Err(ByteSizeError::Overflow { .. })
    ));

    // u64::MAX + 1 without unit classified as Overflow
    assert!(matches!(
        parse_byte_size("18446744073709551616"),
        Err(ByteSizeError::Overflow { .. })
    ));

    // Exceeding u128 magnitude classified as Overflow
    assert!(matches!(
        parse_byte_size("340282366920938463463374607431768211456 B"),
        Err(ByteSizeError::Overflow { .. })
    ));

    // Decimal with integer part exceeding u64 classified as Overflow
    assert!(matches!(
        parse_byte_size("18446744073709551616.0 B"),
        Err(ByteSizeError::Overflow { .. })
    ));
}

#[test]
fn format_byte_size_units() {
    assert_eq!(format_byte_size(0), "0 B");
    assert_eq!(format_byte_size(512), "512 B");
    assert_eq!(format_byte_size(KIB), "1 KiB");
    assert_eq!(format_byte_size(4 * KIB), "4 KiB");
    assert_eq!(format_byte_size(512 * MIB), "512 MiB");
    assert_eq!(format_byte_size(32 * GIB), "32 GiB");
    assert_eq!(format_byte_size(2 * TIB), "2 TiB");
    assert_eq!(format_byte_size(PIB), "1 PiB");

    // Fractional format
    assert_eq!(format_byte_size(1_610_612_736), "1.50 GiB");
}

#[test]
fn byte_size_wrapper() {
    let bs = ByteSize::from_str("32 GiB").unwrap();
    assert_eq!(bs.as_u64(), 32 * GIB);
    assert_eq!(format!("{bs}"), "32 GiB");

    let bs2 = ByteSize::new(1024);
    assert_eq!(bs2.as_u64(), 1024);
}
