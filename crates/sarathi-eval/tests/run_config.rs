// SPDX-License-Identifier: Apache-2.0
//! The committed Theme Studio run configuration parses with its three slots and caps, broken configurations are
//! refused, and a budgeted run stops a slot at its cap while the other slots run on.

use sarathi_eval::brief::{Brief, Split, parse_brief_set};
use sarathi_eval::config::{
    CappedRunError, ConfigError, RunConfig, run_within_caps, spent_by_slot,
};
use sarathi_eval::metrics::{Purpose, TokenUsage};
use sarathi_eval::runner::{Acceptance, Arm, Harness, HarnessError, HarnessOutput};

const A9: &str = include_str!("../../../evals/vyasa/a9-run.json");
const SET: &str = include_str!("../../../evals/vyasa/briefs.jsonl");

#[test]
fn the_committed_configuration_has_three_slots_within_the_total_cap() {
    let c = RunConfig::from_json(A9).unwrap();
    assert_eq!(c.runs, 3);
    assert_eq!(c.harnesses, ["old", "new"]);
    let slots: Vec<(&str, &str, u64, bool)> = c
        .slots
        .iter()
        .map(|s| (s.slot.as_str(), s.model.as_str(), s.cap_usd, s.native_tools))
        .collect();
    assert_eq!(
        slots,
        [
            ("strong", "claude-sonnet-5-5", 170, true),
            ("cheap-tools", "gpt-5-mini", 20, true),
            ("fallback-no-tools", "gemini-2.5-flash", 20, false),
        ]
    );
    assert_eq!(c.total_cap_usd, 210);
    // Every brief of the set allows every configured slot, so the whole set can run on every arm.
    for b in parse_brief_set(SET).unwrap() {
        for s in &c.slots {
            assert!(
                b.allowed_models.contains(&s.slot),
                "{} does not allow {}",
                b.id,
                s.slot
            );
        }
    }
    // Every model is priced, and the strong model's cache writes are priced above its input.
    let prices = c.price_table();
    assert_eq!(prices.models.len(), 3);
    let strong = prices.models["claude-sonnet-5-5"];
    assert!(strong.cache_write.is_some_and(|w| w > strong.input));
}

fn with(edit: impl FnOnce(&mut serde_json::Value)) -> Result<RunConfig, ConfigError> {
    let mut v: serde_json::Value =
        serde_json::from_str(A9).map_err(|e| ConfigError::Parse(e.to_string()))?;
    edit(&mut v);
    RunConfig::from_json(&v.to_string())
}

#[test]
fn broken_configurations_are_refused() {
    use serde_json::json;
    assert!(matches!(
        with(|v| v["version"] = json!(2)),
        Err(ConfigError::Version(2))
    ));
    assert!(matches!(
        with(|v| v["version"] = json!(0)),
        Err(ConfigError::Version(0))
    ));
    for (why, edit) in [
        (
            "no runs",
            Box::new(|v: &mut serde_json::Value| v["runs"] = json!(0)) as Box<dyn Fn(&mut _)>,
        ),
        (
            "no harness",
            Box::new(|v: &mut serde_json::Value| v["harnesses"] = json!([])),
        ),
        (
            "harness with a dot",
            Box::new(|v: &mut serde_json::Value| v["harnesses"] = json!(["old.1", "new"])),
        ),
        (
            "harness twice",
            Box::new(|v: &mut serde_json::Value| v["harnesses"] = json!(["old", "old"])),
        ),
        (
            "slot twice",
            Box::new(|v: &mut serde_json::Value| v["slots"][1]["slot"] = json!("strong")),
        ),
        (
            "empty model",
            Box::new(|v: &mut serde_json::Value| v["slots"][0]["model"] = json!(" ")),
        ),
        (
            "zero cap",
            Box::new(|v: &mut serde_json::Value| v["slots"][2]["cap_usd"] = json!(0)),
        ),
        (
            "caps over total",
            Box::new(|v: &mut serde_json::Value| v["total_cap_usd"] = json!(209)),
        ),
    ] {
        assert!(
            matches!(with(|v| edit(v)), Err(ConfigError::Invalid(_))),
            "{why}"
        );
    }
    assert!(
        matches!(with(|v| v["extra"] = json!(1)), Err(ConfigError::Parse(_))),
        "unknown field"
    );
}

/// Every run spends one million uncached input tokens.
struct Spender;
impl Harness for Spender {
    fn run(&self, _b: &Brief, _m: &str, _run: u32) -> Result<HarnessOutput, HarnessError> {
        Ok(HarnessOutput {
            usage: TokenUsage {
                input_tokens: 1_000_000,
                ..TokenUsage::default()
            },
            ..HarnessOutput::default()
        })
    }
}
struct Always;
impl Acceptance for Always {
    fn handles(&self, _: &str) -> bool {
        true
    }

    fn accepts(&self, _b: &Brief, _o: &HarnessOutput) -> bool {
        true
    }
}

fn briefs(n: usize) -> Vec<Brief> {
    (0..n)
        .map(|i| Brief {
            id: format!("b{i:02}"),
            split: Split::Measurement,
            site_context: String::new(),
            text: "t".into(),
            allowed_models: vec![
                "strong".into(),
                "cheap-tools".into(),
                "fallback-no-tools".into(),
            ],
            acceptance: "structural".into(),
            fault: None,
        })
        .collect()
}

fn arm<'a>(c: &RunConfig, harness: &str, slot: &str, h: &'a dyn Harness) -> Arm<'a> {
    Arm {
        name: RunConfig::arm_name(harness, slot),
        slot: slot.into(),
        // An unknown slot gets no model, which `run_within_caps` refuses.
        model: c.slot(slot).map_or_else(String::new, |s| s.model.clone()),
        harness: h,
    }
}

#[test]
fn a_slot_stops_at_its_cap_and_the_others_run_on() {
    let mut c = RunConfig::from_json(A9).unwrap();
    // $2 per run on `strong`, two harnesses: a group (one brief, one repetition) costs $4 against a $5 cap. The first
    // group leaves $4 spent, below the cap, so the second runs whole and crosses it; then the slot stops.
    c.slots[0].cap_usd = 5;
    let h = Spender;
    let arms = [
        arm(&c, "old", "strong", &h),
        arm(&c, "new", "strong", &h),
        arm(&c, "new", "cheap-tools", &h),
    ];
    let bs = briefs(4);
    let out = run_within_caps(&c, &bs, &arms, &Always, Purpose::Measure, &[]).unwrap();
    let strong = out.records.iter().filter(|r| r.slot == "strong").count();
    assert_eq!(
        strong, 4,
        "groups run while spend is below the cap, so one group may cross it"
    );
    let cheap = out
        .records
        .iter()
        .filter(|r| r.slot == "cheap-tools")
        .count();
    assert_eq!(cheap, 4 * 3, "the cheap slot ran every run");
    assert_eq!(
        out.stopped.iter().map(String::as_str).collect::<Vec<_>>(),
        ["strong"]
    );
    assert!(out.slot_stopped("strong"));
    assert!(!out.slot_stopped("cheap-tools"));
    assert_eq!(out.stopped_slots(), ["strong"]);
    let spent = spent_by_slot(&out.records, &c.price_table()).unwrap();
    assert_eq!(spent["strong"], 8_000_000);
    assert_paired(&out.records, "strong", &["old", "new"]);
}

/// The harnesses of `slot` ran the same (brief, repetition) set.
fn assert_paired(records: &[sarathi_eval::metrics::RunRecord], slot: &str, harnesses: &[&str]) {
    use std::collections::BTreeSet;
    let set = |h: &str| -> BTreeSet<(&str, u32)> {
        records
            .iter()
            .filter(|r| r.slot == slot && r.arm == RunConfig::arm_name(h, slot))
            .map(|r| (r.brief.as_str(), r.run))
            .collect()
    };
    let first = set(harnesses[0]);
    assert!(!first.is_empty(), "slot {slot}: no runs");
    for h in &harnesses[1..] {
        assert_eq!(
            set(h),
            first,
            "slot {slot}: {h} and {} ran different (brief, run) sets",
            harnesses[0]
        );
    }
}

#[test]
fn a_stop_never_leaves_a_slot_unpaired() {
    // Every cap from one to nine $2 runs, with odd briefs reversing the arm order, so the crossing run falls on either
    // harness: the stopped slot's two harnesses still ran the same (brief, repetition) set.
    let h = Spender;
    let bs = briefs(5);
    for cap in 1..=9 {
        let mut c = RunConfig::from_json(A9).unwrap();
        c.slots[0].cap_usd = cap;
        let arms = [
            arm(&c, "old", "strong", &h),
            arm(&c, "new", "cheap-tools", &h),
            arm(&c, "new", "strong", &h),
            arm(&c, "old", "cheap-tools", &h),
        ];
        let out = run_within_caps(&c, &bs, &arms, &Always, Purpose::Measure, &[]).unwrap();
        assert!(out.stopped.contains("strong"), "cap ${cap}");
        let strong = out.records.iter().filter(|r| r.slot == "strong").count() as u64;
        // Whole groups of two runs, and at most one group past the cap.
        assert_eq!(strong % 2, 0, "cap ${cap}");
        // $2 a run: the slot reached its cap, and it was below the cap before its last group ($4).
        assert!(
            strong * 2 >= cap && strong * 2 < cap + 4,
            "cap ${cap}: {strong} runs"
        );
        assert_paired(&out.records, "strong", &["old", "new"]);
        assert_paired(&out.records, "cheap-tools", &["old", "new"]);
    }
}

#[test]
fn earlier_spend_counts_against_the_cap() {
    let mut c = RunConfig::from_json(A9).unwrap();
    c.slots[0].cap_usd = 5;
    let h = Spender;
    let arms = [arm(&c, "new", "strong", &h)];
    let bs = briefs(2);
    let smoke = run_within_caps(&c, &bs, &arms, &Always, Purpose::Measure, &[]).unwrap();
    assert_eq!(smoke.records.len(), 3);
    let again = run_within_caps(&c, &bs, &arms, &Always, Purpose::Measure, &smoke.records).unwrap();
    assert!(again.records.is_empty(), "the cap is already spent");
    assert!(again.stopped.contains("strong"));
}

#[test]
fn an_arm_must_match_its_configured_slot() {
    let c = RunConfig::from_json(A9).unwrap();
    let h = Spender;
    let mut wrong = arm(&c, "new", "strong", &h);
    wrong.model = "another-model".into();
    let bs = briefs(1);
    let e = run_within_caps(&c, &bs, &[wrong], &Always, Purpose::Measure, &[]).unwrap_err();
    assert!(matches!(e, CappedRunError::Arm { .. }), "{e}");
    let mut renamed = arm(&c, "new", "strong", &h);
    renamed.name = "newer.strong".into();
    assert!(
        matches!(
            run_within_caps(&c, &bs, &[renamed], &Always, Purpose::Measure, &[]),
            Err(CappedRunError::Arm { .. })
        ),
        "an arm of an unconfigured harness"
    );
    let mut unknown = arm(&c, "new", "huge", &h);
    unknown.model = "claude-sonnet-5-5".into();
    assert!(matches!(
        run_within_caps(&c, &bs, &[unknown], &Always, Purpose::Measure, &[]),
        Err(CappedRunError::Arm { .. })
    ));
}
