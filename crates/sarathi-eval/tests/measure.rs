// SPDX-License-Identifier: Apache-2.0
//! A budgeted run becomes its report: stats for every arm, the kill line over the old and new harness of each slot,
//! and the slots stopped at their cap reaching both the report and the kill line.

use sarathi_eval::brief::{Brief, Split};
use sarathi_eval::config::{RunConfig, run_within_caps};
use sarathi_eval::killline::{CostSource, Outcome};
use sarathi_eval::measure::{CostTolerance, Judging, MeasureError, report};
use sarathi_eval::metrics::{Purpose, TokenUsage};
use sarathi_eval::report::{to_json, to_markdown};
use sarathi_eval::runner::{Acceptance, Arm, Harness, HarnessError, HarnessOutput};

const A9: &str = include_str!("../../../evals/vyasa/a9-run.json");

const JUDGING: Judging<'static> = Judging {
    old: "old",
    new: "new",
    strong_slot: "strong",
    cost: CostTolerance::Fraction(0.05),
};

/// Every run spends one million uncached input tokens ($2 on `strong`). It produces `no` for the brief numbers in
/// `misses` on repetition 0 and `ok` otherwise.
struct Spender {
    misses: &'static [usize],
}
impl Harness for Spender {
    fn run(&self, b: &Brief, _m: &str, run: u32) -> Result<HarnessOutput, HarnessError> {
        let n: usize = b.id[1..].parse().unwrap_or(usize::MAX);
        let miss = run == 0 && self.misses.contains(&n);
        Ok(HarnessOutput {
            artifact: if miss { "no" } else { "ok" }.into(),
            usage: TokenUsage {
                input_tokens: 1_000_000,
                ..TokenUsage::default()
            },
            ..HarnessOutput::default()
        })
    }
}
struct Ok_;
impl Acceptance for Ok_ {
    fn handles(&self, _: &str) -> bool {
        true
    }

    fn accepts(&self, _b: &Brief, o: &HarnessOutput) -> bool {
        o.artifact == "ok"
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

/// Every configured (harness, slot) arm, `old` on `old_h` and `new` on `new_h`.
fn arms<'a>(c: &RunConfig, old_h: &'a dyn Harness, new_h: &'a dyn Harness) -> Vec<Arm<'a>> {
    let mut v = Vec::new();
    for s in &c.slots {
        for (h, harness) in [("old", old_h), ("new", new_h)] {
            v.push(Arm {
                name: RunConfig::arm_name(h, &s.slot),
                slot: s.slot.clone(),
                model: s.model.clone(),
                harness,
            });
        }
    }
    v
}

#[test]
fn a_full_run_reports_every_arm_and_judges_every_slot() {
    let c = RunConfig::from_json(A9).unwrap();
    let (old, new) = (Spender { misses: &[0, 1] }, Spender { misses: &[0] });
    let bs = briefs(4);
    let run = run_within_caps(&c, &bs, &arms(&c, &old, &new), &Ok_, Purpose::Measure, &[]).unwrap();
    let r = report(&c, &run, Some(&JUDGING), "2026-11-28", 0).unwrap();
    let names: Vec<&str> = r.arms.iter().map(|a| a.arm.as_str()).collect();
    assert_eq!(
        names,
        [
            "old.strong",
            "new.strong",
            "old.cheap-tools",
            "new.cheap-tools",
            "old.fallback-no-tools",
            "new.fallback-no-tools",
        ]
    );
    assert!(r.arms.iter().all(|a| a.runs == 12));
    assert!(r.stopped.is_empty());
    let k = r.kill_line.as_ref().unwrap();
    assert_eq!(k.outcome, Outcome::Pass, "{:?}", k.notes);
    let judged: Vec<(&str, bool)> = k.arms.iter().map(|a| (a.name.as_str(), a.strong)).collect();
    assert_eq!(
        judged,
        [
            ("strong", true),
            ("cheap-tools", false),
            ("fallback-no-tools", false)
        ]
    );
    // The accept tolerance is the old arm's spread; the cost tolerance is the judging rule's.
    let old_strong = &r.arms[0];
    assert!(old_strong.spread > 0.0);
    assert!((k.arms[0].tolerance.accept - old_strong.spread).abs() < 1e-12);
    assert!((k.arms[0].tolerance.cost_fraction - 0.05).abs() < 1e-12);
    // The same run gives the same report.
    let again = report(&c, &run, Some(&JUDGING), "2026-11-28", 0).unwrap();
    assert_eq!(to_json(&r).unwrap(), to_json(&again).unwrap());
}

/// Every run is accepted; repetition `r` spends `costs[r]` million uncached input tokens.
struct CostBy {
    costs: [u64; 3],
}
impl Harness for CostBy {
    fn run(&self, _b: &Brief, _m: &str, run: u32) -> Result<HarnessOutput, HarnessError> {
        Ok(HarnessOutput {
            artifact: "ok".into(),
            usage: TokenUsage {
                input_tokens: self.costs[run as usize % 3] * 1_000_000,
                ..TokenUsage::default()
            },
            ..HarnessOutput::default()
        })
    }
}

#[test]
fn a_strong_slot_stopped_before_its_first_run_fails_the_kill_line() {
    let mut c = RunConfig::from_json(A9).unwrap();
    c.slots[0].cap_usd = 1;
    let h = Spender { misses: &[] };
    let bs = briefs(2);
    let all = arms(&c, &h, &h);
    // An earlier pass already spent the strong slot's cap.
    let earlier = run_within_caps(&c, &bs, &all[..2], &Ok_, Purpose::Measure, &[]).unwrap();
    let run = run_within_caps(&c, &bs, &all, &Ok_, Purpose::Measure, &earlier.records).unwrap();
    let r = report(&c, &run, Some(&JUDGING), "2026-11-28", 0).unwrap();
    assert_eq!(r.stopped, ["strong"]);
    assert_eq!((r.arms[0].runs, r.arms[1].runs), (0, 0));
    let k = r.kill_line.unwrap();
    assert_eq!(k.outcome, Outcome::Fail, "{:?}", k.notes);
    assert!(k.arms.iter().all(|a| a.name != "strong"));
}

#[test]
fn the_old_cost_spread_is_the_cost_tolerance_when_chosen() {
    let c = RunConfig::from_json(A9).unwrap();
    let judging = Judging {
        cost: CostTolerance::OldSpread,
        ..JUDGING
    };
    // Old: 1, 2, 3 million a run (mean 2, std sqrt(2/3)): a spread of about 40.8%.
    let old = CostBy { costs: [1, 2, 3] };
    let spread = (2.0f64 / 3.0).sqrt() / 2.0;
    let bs = briefs(2);
    // New: a third dearer than the old mean, within the spread: passes; a fixed 5% fails it.
    let within = CostBy { costs: [3, 2, 3] };
    let run = run_within_caps(
        &c,
        &bs,
        &arms(&c, &old, &within),
        &Ok_,
        Purpose::Measure,
        &[],
    )
    .unwrap();
    let r = report(&c, &run, Some(&judging), "2026-11-28", 0).unwrap();
    let k = r.kill_line.as_ref().unwrap();
    assert_eq!(k.outcome, Outcome::Pass, "{:?}", k.notes);
    assert!((r.arms[0].cost_spread - spread).abs() < 1e-9);
    assert!((k.arms[0].tolerance.cost_fraction - spread).abs() < 1e-9);
    assert_eq!(k.arms[0].tolerance.cost_source, CostSource::OldSpread);
    assert!(
        to_markdown(&r).contains("% (old spread)): accept ok, cost ok"),
        "{}",
        to_markdown(&r)
    );
    let fixed = report(&c, &run, Some(&JUDGING), "2026-11-28", 0).unwrap();
    assert_eq!(fixed.kill_line.unwrap().outcome, Outcome::Fail);
    // New: 50% dearer, beyond the spread: fails on cost.
    let beyond = CostBy { costs: [3, 3, 3] };
    let run = run_within_caps(
        &c,
        &bs,
        &arms(&c, &old, &beyond),
        &Ok_,
        Purpose::Measure,
        &[],
    )
    .unwrap();
    let k = report(&c, &run, Some(&judging), "2026-11-28", 0)
        .unwrap()
        .kill_line
        .unwrap();
    assert_eq!(k.outcome, Outcome::Fail);
    assert!(
        k.notes
            .iter()
            .any(|n| n.contains("strong arm `strong`: cost per accepted run")),
        "{:?}",
        k.notes
    );
}

/// Every run spends one million uncached input tokens; only repetition 0 is accepted.
struct FirstOnly;
impl Harness for FirstOnly {
    fn run(&self, _b: &Brief, _m: &str, run: u32) -> Result<HarnessOutput, HarnessError> {
        Ok(HarnessOutput {
            artifact: if run == 0 { "ok" } else { "no" }.into(),
            usage: TokenUsage {
                input_tokens: 1_000_000,
                ..TokenUsage::default()
            },
            ..HarnessOutput::default()
        })
    }
}

#[test]
fn the_old_cost_spread_without_two_accepted_runs_is_strict_and_noted() {
    let c = RunConfig::from_json(A9).unwrap();
    let judging = Judging {
        cost: CostTolerance::OldSpread,
        ..JUDGING
    };
    let bs = briefs(2);
    let run = run_within_caps(
        &c,
        &bs,
        &arms(&c, &FirstOnly, &FirstOnly),
        &Ok_,
        Purpose::Measure,
        &[],
    )
    .unwrap();
    let r = report(&c, &run, Some(&judging), "2026-11-28", 0).unwrap();
    let k = r.kill_line.as_ref().unwrap();
    // Same cost on both sides: within a strict tolerance, so the outcome does not change; the note says why it is 0.
    assert_eq!(k.outcome, Outcome::Pass, "{:?}", k.notes);
    let strong = k.arms.iter().find(|a| a.name == "strong").unwrap();
    assert_eq!(strong.tolerance.cost_source, CostSource::OldSpreadMissing);
    assert!(strong.tolerance.cost_fraction.abs() < f64::EPSILON);
    assert!(
        k.notes
            .iter()
            .any(|n| n.starts_with("arm `strong`: no old brief has two accepted runs")),
        "{:?}",
        k.notes
    );
    let md = to_markdown(&r);
    assert!(
        md.contains("cost 0.0% (old spread missing, strict)"),
        "{md}"
    );
    // A fixed fraction is labelled as such.
    let fixed = report(&c, &run, Some(&JUDGING), "2026-11-28", 0).unwrap();
    assert!(to_markdown(&fixed).contains("cost 5.0% (fixed)"));
}

#[test]
fn the_new_core_losing_on_the_strong_slot_fails() {
    let c = RunConfig::from_json(A9).unwrap();
    let (old, new) = (Spender { misses: &[] }, Spender { misses: &[0, 1, 2] });
    let bs = briefs(4);
    let run = run_within_caps(&c, &bs, &arms(&c, &old, &new), &Ok_, Purpose::Measure, &[]).unwrap();
    let r = report(&c, &run, Some(&JUDGING), "2026-11-28", 0).unwrap();
    let k = r.kill_line.unwrap();
    assert_eq!(k.outcome, Outcome::Fail);
    assert!(
        k.notes
            .iter()
            .any(|n| n.contains("strong arm `strong`: accept rate")),
        "{:?}",
        k.notes
    );
}

#[test]
fn a_stopped_strong_slot_reaches_the_report_and_fails_the_kill_line() {
    let mut c = RunConfig::from_json(A9).unwrap();
    // $2 a run, two harnesses: the strong slot stops after its second group.
    c.slots[0].cap_usd = 5;
    let h = Spender { misses: &[] };
    let bs = briefs(4);
    let run = run_within_caps(&c, &bs, &arms(&c, &h, &h), &Ok_, Purpose::Measure, &[]).unwrap();
    let r = report(&c, &run, Some(&JUDGING), "2026-11-28", 0).unwrap();
    assert_eq!(r.stopped, ["strong"]);
    assert_eq!((r.arms[0].runs, r.arms[1].runs), (2, 2));
    let k = r.kill_line.as_ref().unwrap();
    assert_eq!(k.outcome, Outcome::Fail);
    assert!(
        k.notes
            .iter()
            .any(|n| n.contains("strong arm `strong`: stopped at its spend cap")),
        "{:?}",
        k.notes
    );
    let md = to_markdown(&r);
    assert!(
        md.contains("- slot strong: stopped at its spend cap"),
        "{md}"
    );
    assert!(md.contains("Kill line: **FAIL**"), "{md}");
}

#[test]
fn a_stopped_other_slot_is_a_finding() {
    let mut c = RunConfig::from_json(A9).unwrap();
    // gpt-5-mini input is $0.25 per million: a group of two runs is $0.50 against a $1 cap.
    c.slots[1].cap_usd = 1;
    let h = Spender { misses: &[] };
    let bs = briefs(4);
    let run = run_within_caps(&c, &bs, &arms(&c, &h, &h), &Ok_, Purpose::Measure, &[]).unwrap();
    let r = report(&c, &run, Some(&JUDGING), "2026-11-28", 0).unwrap();
    assert_eq!(r.stopped, ["cheap-tools"]);
    let k = r.kill_line.unwrap();
    assert_eq!(k.outcome, Outcome::Pass, "{:?}", k.notes);
    assert!(
        k.notes
            .iter()
            .any(|n| n.contains("finding, arm `cheap-tools`: stopped at its spend cap")),
        "{:?}",
        k.notes
    );
}

#[test]
fn a_slot_stopped_before_its_first_run_is_left_out_of_the_kill_line() {
    let mut c = RunConfig::from_json(A9).unwrap();
    c.slots[2].cap_usd = 1;
    let h = Spender { misses: &[] };
    let bs = briefs(2);
    let all = arms(&c, &h, &h);
    // An earlier pass already spent the fallback slot's cap.
    let earlier = run_within_caps(&c, &bs, &all[4..], &Ok_, Purpose::Measure, &[]).unwrap();
    let run = run_within_caps(&c, &bs, &all, &Ok_, Purpose::Measure, &earlier.records).unwrap();
    let r = report(&c, &run, Some(&JUDGING), "2026-11-28", 0).unwrap();
    assert_eq!(r.stopped, ["fallback-no-tools"]);
    assert_eq!((r.arms[4].runs, r.arms[5].runs), (0, 0));
    let k = r.kill_line.unwrap();
    assert_eq!(k.outcome, Outcome::Pass, "{:?}", k.notes);
    assert!(k.arms.iter().all(|a| a.name != "fallback-no-tools"));
    assert!(
        k.notes.iter().any(|n| n
            == "arm `fallback-no-tools`: stopped at its spend cap before its first run; left out of the kill line"),
        "{:?}",
        k.notes
    );
}

#[test]
fn without_judging_there_is_no_kill_line() {
    let c = RunConfig::from_json(A9).unwrap();
    let h = Spender { misses: &[] };
    let bs = briefs(1);
    let run = run_within_caps(&c, &bs, &arms(&c, &h, &h), &Ok_, Purpose::Measure, &[]).unwrap();
    let r = report(&c, &run, None, "2026-11-28", 2).unwrap();
    assert!(r.kill_line.is_none());
    assert_eq!(r.deviations, 2);
    assert!(to_markdown(&r).contains("Kill line: not evaluated"));
}

#[test]
fn a_judging_rule_that_does_not_fit_is_refused() {
    let c = RunConfig::from_json(A9).unwrap();
    let h = Spender { misses: &[] };
    let bs = briefs(1);
    let run = run_within_caps(&c, &bs, &arms(&c, &h, &h), &Ok_, Purpose::Measure, &[]).unwrap();
    for (why, j) in [
        (
            "unknown old",
            Judging {
                old: "older",
                ..JUDGING
            },
        ),
        (
            "unknown new",
            Judging {
                new: "x",
                ..JUDGING
            },
        ),
        (
            "same harness",
            Judging {
                new: "old",
                ..JUDGING
            },
        ),
        (
            "unknown strong slot",
            Judging {
                strong_slot: "huge",
                ..JUDGING
            },
        ),
        (
            "negative cost",
            Judging {
                cost: CostTolerance::Fraction(-0.1),
                ..JUDGING
            },
        ),
        (
            "NaN cost",
            Judging {
                cost: CostTolerance::Fraction(f64::NAN),
                ..JUDGING
            },
        ),
        (
            "infinite cost",
            Judging {
                cost: CostTolerance::Fraction(f64::INFINITY),
                ..JUDGING
            },
        ),
    ] {
        assert!(
            matches!(
                report(&c, &run, Some(&j), "2026-11-28", 0),
                Err(MeasureError::Judging(_))
            ),
            "{why}"
        );
    }
}
