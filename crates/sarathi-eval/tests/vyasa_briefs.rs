// SPDX-License-Identifier: Apache-2.0
//! The committed Theme Studio brief set parses and has the composition the measurement protocol fixes.

use sarathi_eval::brief::{Split, parse_brief_set};

const SET: &str = include_str!("../../../evals/vyasa/briefs.jsonl");

#[test]
fn the_vyasa_set_has_thirty_briefs_split_twenty_ten() {
    let briefs = parse_brief_set(SET).unwrap();
    assert_eq!(briefs.len(), 30);
    let tuning = briefs.iter().filter(|b| b.split == Split::Tuning).count();
    assert_eq!(tuning, 10);
    assert_eq!(briefs.len() - tuning, 20);
}

#[test]
fn the_set_has_the_planned_kinds() {
    let briefs = parse_brief_set(SET).unwrap();
    for (prefix, n) in [("plain-", 10), ("vague-", 10), ("tight-", 5), ("adv-", 5)] {
        let got = briefs.iter().filter(|b| b.id.starts_with(prefix)).count();
        assert_eq!(got, n, "{prefix}");
    }
}

#[test]
fn every_kind_has_briefs_in_both_splits() {
    let briefs = parse_brief_set(SET).unwrap();
    for prefix in ["plain-", "vague-", "tight-", "adv-"] {
        for split in [Split::Tuning, Split::Measurement] {
            assert!(
                briefs
                    .iter()
                    .any(|b| b.id.starts_with(prefix) && b.split == split),
                "{prefix} {split:?}"
            );
        }
    }
}

#[test]
fn adv_04_names_its_injected_fault_and_no_other_brief_does() {
    let briefs = parse_brief_set(SET).unwrap();
    let faulty: Vec<_> = briefs
        .iter()
        .filter(|b| b.fault.is_some())
        .map(|b| (b.id.as_str(), b.fault.as_deref()))
        .collect();
    assert_eq!(faulty, vec![("adv-04", Some("malformed_steps"))]);
}
