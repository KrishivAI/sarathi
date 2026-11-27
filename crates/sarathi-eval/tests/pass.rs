// SPDX-License-Identifier: Apache-2.0
//! Passes over one root with the committed configuration and brief set: a dry run calls and writes nothing, the smoke
//! run, tuning and the measurement each keep their own records, and every record counts towards the caps.

use std::cell::Cell;
use std::fs::File;
use std::path::{Path, PathBuf};

use sarathi_eval::brief::{Brief, parse_brief_set};
use sarathi_eval::config::{CappedRun, RunConfig, SlotConfig};
use sarathi_eval::measure::{CostTolerance, Judging};
use sarathi_eval::metrics::{Purpose, TokenUsage};
use sarathi_eval::pass::{Pass, PassError, PassKind, Summary};
use sarathi_eval::report::Report;
use sarathi_eval::runner::{
    Acceptance, Harness, HarnessError, HarnessOutput, read_records, record_path,
};
use sarathi_eval::structural::Structural;

const A9: &str = include_str!("../../../evals/vyasa/a9-run.json");
const SET: &str = include_str!("../../../evals/vyasa/briefs.jsonl");
const FRI: &str = "2026-11-27";
const SAT: &str = "2026-11-28";
const SMOKE: PassKind<'static> = PassKind::Smoke {
    briefs: &["plain-02", "adv-03"],
};
const TUNE: PassKind<'static> = PassKind::Tune {
    round: 1,
    slot: "strong",
};
const MEASURE: PassKind<'static> = PassKind::Measure {
    judging: Judging {
        old: "old",
        new: "new",
        strong_slot: "strong",
        cost: CostTolerance::OldSpread,
    },
};

/// Every run spends `tokens` uncached input tokens and is counted.
struct Spender {
    tokens: u64,
    calls: Cell<usize>,
}
impl Spender {
    fn new(tokens: u64) -> Self {
        Self {
            tokens,
            calls: Cell::new(0),
        }
    }
}
impl Harness for Spender {
    fn run(&self, _: &Brief, _: &str, _: u32) -> Result<HarnessOutput, HarnessError> {
        self.calls.set(self.calls.get() + 1);
        Ok(HarnessOutput {
            usage: TokenUsage {
                input_tokens: self.tokens,
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

    fn accepts(&self, _: &Brief, _: &HarnessOutput) -> bool {
        true
    }
}

struct Fixture {
    config: RunConfig,
    set: Vec<Brief>,
    root: PathBuf,
}
impl Fixture {
    fn new(name: &str) -> Option<Self> {
        let root = std::env::temp_dir().join(format!("sarathi-pass-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        Some(Self {
            config: RunConfig::from_json(A9).ok()?,
            set: parse_brief_set(SET).ok()?,
            root,
        })
    }

    fn dry<'a>(&'a self, kind: PassKind<'a>, date: &'a str) -> Result<Summary, PassError> {
        Pass::prepare(&self.config, &self.set, kind, &Always, &self.root, date)
            .map(|p| p.summary().clone())
    }

    fn run<'a>(
        &'a self,
        kind: PassKind<'a>,
        date: &'a str,
        h: &Spender,
    ) -> Result<(CappedRun, Report), PassError> {
        Pass::prepare(&self.config, &self.set, kind, &Always, &self.root, date)?
            .run(|_: &str, _: &SlotConfig| h as &dyn Harness, 0)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn files(dir: &Path) -> usize {
    std::fs::read_dir(dir).map_or(0, |d| {
        d.flatten()
            .map(|e| {
                let p = e.path();
                if p.is_dir() { files(&p) } else { 1 }
            })
            .sum()
    })
}

#[test]
fn a_dry_run_says_what_would_run_and_touches_nothing() {
    let f = Fixture::new("dry").unwrap();
    let s = f.dry(SMOKE, FRI).unwrap();
    assert_eq!(s.dir, f.root.join("smoke"));
    let slots: Vec<_> = s
        .slots
        .iter()
        .map(|s| (s.slot.as_str(), s.runs_planned, s.runs_done, s.spent_micro))
        .collect();
    assert_eq!(
        slots,
        [
            ("strong", 4, 0, 0),
            ("cheap-tools", 4, 0, 0),
            ("fallback-no-tools", 4, 0, 0),
        ]
    );
    assert_eq!(s.slots[0].cap_micro, 170_000_000);
    let t = f.dry(TUNE, FRI).unwrap();
    assert_eq!(t.slots.len(), 1);
    assert_eq!(t.slots[0].runs_planned, 20);
    let m = f.dry(MEASURE, SAT).unwrap();
    assert!(
        m.slots
            .iter()
            .all(|s| s.runs_planned == 120 && s.projected_micro.is_none())
    );
    assert!(!f.root.exists());
}

#[test]
fn the_passes_share_one_root_and_every_record_counts_towards_the_caps() {
    let f = Fixture::new("flow").unwrap();
    let h = Spender::new(100_000);
    let (smoke, report) = f.run(SMOKE, FRI, &h).unwrap();
    assert_eq!((smoke.records.len(), h.calls.get()), (12, 12));
    assert!(report.kill_line.is_none());
    for file in ["report.md", "report.json"] {
        assert!(
            f.root.join("smoke").join(FRI).join(file).is_file(),
            "{file}"
        );
    }
    assert_eq!(
        read_records(&f.root.join("smoke"), FRI, Purpose::Measure)
            .unwrap()
            .len(),
        12
    );
    // $0.20 a strong run: four smoke runs.
    assert_eq!(f.dry(SMOKE, FRI).unwrap().slots[0].spent_micro, 800_000);

    // Tuning runs on one slot; the smoke run's records on other slots do not disturb it.
    let (tune, _) = f.run(TUNE, FRI, &h).unwrap();
    assert_eq!(tune.records.len(), 20);
    assert_eq!(f.dry(TUNE, FRI).unwrap().slots[0].spent_micro, 4_800_000);

    let m = f.dry(MEASURE, SAT).unwrap();
    assert_eq!(m.slots[0].spent_micro, 4_800_000);
    assert_eq!(m.slots[0].projected_micro, Some(4_800_000 + 120 * 200_000));
    // cheap-tools: $0.025 a run.
    assert_eq!(m.slots[1].projected_micro, Some(100_000 + 120 * 25_000));

    let before = h.calls.get();
    let (measure, report) = f.run(MEASURE, SAT, &h).unwrap();
    assert_eq!(measure.records.len(), 360);
    assert_eq!(h.calls.get() - before, 360);
    assert!(report.kill_line.is_some());
    let md = std::fs::read_to_string(f.root.join("measure").join(SAT).join("report.md")).unwrap();
    assert!(md.contains("Kill line: **"), "{md}");
    let done = f.dry(MEASURE, SAT).unwrap();
    assert!(done.slots.iter().all(|s| s.runs_done == 120));
}

#[test]
fn a_stopped_pass_resumes_under_its_date_without_paying_twice() {
    let f = Fixture::new("resume").unwrap();
    let h = Spender::new(1_000);
    let (whole, _) = f.run(SMOKE, FRI, &h).unwrap();
    // One record lost before its save: only that run is made again.
    std::fs::remove_file(record_path(&f.root.join("smoke"), FRI, &whole.records[5])).unwrap();
    assert_eq!(
        f.dry(SMOKE, FRI)
            .unwrap()
            .slots
            .iter()
            .map(|s| s.runs_done)
            .sum::<usize>(),
        11
    );
    let e = f.run(SMOKE, SAT, &h).unwrap_err();
    assert!(
        matches!(&e, PassError::Dir(m) if m.contains(FRI) && m.contains(SAT)),
        "{e}"
    );
    assert_eq!(h.calls.get(), 12);
    let (again, _) = f.run(SMOKE, FRI, &h).unwrap();
    assert_eq!(h.calls.get(), 13);
    assert_eq!(again, whole);
}

#[test]
fn the_measurement_waits_for_a_smoke_run_that_fits_the_caps() {
    let f = Fixture::new("ready").unwrap();
    let h = Spender::new(1_000);
    let e = f.run(MEASURE, SAT, &h).unwrap_err();
    assert!(
        matches!(&e, PassError::NotReady(m) if m.contains("smoke")),
        "{e}"
    );
    assert_eq!(h.calls.get(), 0);

    // $2 a strong run, 120 to go: $248 is over the $170 cap.
    let dear = Spender::new(1_000_000);
    f.run(SMOKE, FRI, &dear).unwrap();
    let e = f.run(MEASURE, SAT, &h).unwrap_err();
    assert!(
        matches!(&e, PassError::NotReady(m) if m.contains("`strong`") && m.contains("248000000")),
        "{e}"
    );
    assert_eq!(h.calls.get(), 0);
}

#[test]
fn failed_smoke_runs_do_not_price_the_measurement() {
    struct Down;
    impl Harness for Down {
        fn run(&self, _: &Brief, _: &str, _: u32) -> Result<HarnessOutput, HarnessError> {
            Err(HarnessError::new("no key"))
        }
    }
    let f = Fixture::new("down").unwrap();
    Pass::prepare(&f.config, &f.set, SMOKE, &Always, &f.root, FRI)
        .unwrap()
        .run(|_: &str, _: &SlotConfig| &Down as &dyn Harness, 0)
        .unwrap();
    assert_eq!(f.dry(MEASURE, SAT).unwrap().slots[0].projected_micro, None);
    let h = Spender::new(1_000);
    let e = f.run(MEASURE, SAT, &h).unwrap_err();
    assert!(matches!(e, PassError::NotReady(_)), "{e}");
    assert_eq!(h.calls.get(), 0);
}

#[test]
fn a_record_on_a_slot_the_configuration_lacks_stops_the_pass() {
    let mut f = Fixture::new("lost").unwrap();
    let h = Spender::new(1_000);
    f.run(SMOKE, FRI, &h).unwrap();
    f.config.slots.retain(|s| s.slot != "cheap-tools");
    let e = f.dry(TUNE, FRI).unwrap_err();
    assert!(
        matches!(&e, PassError::Dir(m) if m.contains("`cheap-tools`")),
        "{e}"
    );
}

#[test]
fn a_smoke_run_on_another_model_does_not_price_the_measurement() {
    let mut f = Fixture::new("model").unwrap();
    let h = Spender::new(1_000);
    f.run(SMOKE, FRI, &h).unwrap();
    // The slot's model changes: its old records can no longer be priced, so nothing runs.
    f.config.slots[0].model = "another-model".into();
    let e = f.run(MEASURE, SAT, &h).unwrap_err();
    assert!(matches!(e, PassError::Metrics(_)), "{e}");
    assert_eq!(h.calls.get(), 12);
}

#[test]
fn a_started_measurement_is_resumed_whatever_it_is_projected_to_cost() {
    let f = Fixture::new("started").unwrap();
    let cheap = Spender::new(100_000);
    f.run(SMOKE, FRI, &cheap).unwrap();
    // The measurement costs ten times the smoke run, so every slot stops at its cap.
    let dear = Spender::new(1_000_000);
    let (first, report) = f.run(MEASURE, SAT, &dear).unwrap();
    assert!(first.slot_stopped("strong"));
    assert_eq!(
        report.stopped,
        ["cheap-tools", "fallback-no-tools", "strong"]
    );
    std::fs::remove_file(record_path(&f.root.join("measure"), SAT, &first.records[0])).unwrap();
    f.run(MEASURE, SAT, &dear).unwrap();
}

#[test]
fn anything_unknown_under_the_root_stops_the_pass() {
    let f = Fixture::new("stray").unwrap();
    let h = Spender::new(1_000);
    f.run(SMOKE, FRI, &h).unwrap();
    for stray in [
        "notes.txt".to_owned(),
        "tune-0/x".to_owned(),
        "tune-01/x".to_owned(),
        "smoke-2/x".to_owned(),
        format!("smoke/{SAT}/x"),
        format!("smoke/{FRI}/other/x"),
        format!("smoke/{FRI}/measure/old.strong/adv-03/1.json.tmp"),
    ] {
        let path = f.root.join(&stray);
        // The first directory this stray adds, to take it away again.
        let added = path
            .ancestors()
            .take_while(|p| !p.exists())
            .last()
            .unwrap()
            .to_owned();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "x").unwrap();
        let before = files(&f.root);
        let e = f.run(TUNE, FRI, &h).unwrap_err();
        assert!(
            matches!(e, PassError::Dir(_) | PassError::Runner(_)),
            "{stray}: {e}"
        );
        assert_eq!((h.calls.get(), files(&f.root)), (12, before), "{stray}");
        if added.is_dir() {
            std::fs::remove_dir_all(added).unwrap();
        } else {
            std::fs::remove_file(added).unwrap();
        }
    }
    assert!(f.dry(TUNE, FRI).is_ok());
}

#[test]
fn one_pass_at_a_time_holds_the_root() {
    let f = Fixture::new("lock").unwrap();
    std::fs::create_dir_all(&f.root).unwrap();
    let other = File::create(f.root.join("lock")).unwrap();
    other.lock().unwrap();
    let h = Spender::new(1_000);
    let e = f.run(SMOKE, FRI, &h).unwrap_err();
    assert!(matches!(e, PassError::Locked(_)), "{e}");
    assert_eq!(h.calls.get(), 0);
    drop(other);
    f.run(SMOKE, FRI, &h).unwrap();
}

#[test]
fn records_added_after_the_dry_run_are_seen_under_the_lock() {
    let f = Fixture::new("stale").unwrap();
    let h = Spender::new(1_000);
    let stale = Pass::prepare(&f.config, &f.set, SMOKE, &Always, &f.root, FRI).unwrap();
    f.run(SMOKE, FRI, &h).unwrap();
    stale
        .run(|_: &str, _: &SlotConfig| &h as &dyn Harness, 0)
        .unwrap();
    assert_eq!(h.calls.get(), 12);
}

#[test]
fn a_request_that_does_not_fit_is_refused_before_any_run() {
    let f = Fixture::new("invalid").unwrap();
    for kind in [
        PassKind::Smoke {
            briefs: &["no-such"],
        },
        PassKind::Smoke {
            briefs: &["adv-03", "adv-03"],
        },
        PassKind::Tune {
            round: 0,
            slot: "strong",
        },
        PassKind::Tune {
            round: 1,
            slot: "huge",
        },
    ] {
        let e = f.dry(kind, FRI).unwrap_err();
        assert!(matches!(e, PassError::Invalid(_)), "{e}");
    }
    // The split guard: a tuning brief is no part of the smoke run.
    let e = f
        .dry(
            PassKind::Smoke {
                briefs: &["adv-01"],
            },
            FRI,
        )
        .unwrap_err();
    assert!(matches!(e, PassError::Runner(_)), "{e}");
    assert!(f.dry(PassKind::Smoke { briefs: &[] }, FRI).is_err());
    // A judging rule the report would refuse is refused before the money is spent.
    let bad = PassKind::Measure {
        judging: Judging {
            old: "old",
            new: "old",
            strong_slot: "strong",
            cost: CostTolerance::OldSpread,
        },
    };
    assert!(matches!(f.dry(bad, SAT), Err(PassError::Measure(_))));
    // A date that would leave the directory.
    let h = Spender::new(1);
    assert!(matches!(f.dry(SMOKE, "../x"), Err(PassError::Runner(_))));
    assert!(f.run(SMOKE, "../x", &h).is_err());
    assert_eq!(h.calls.get(), 0);
}

#[test]
fn a_brief_whose_acceptance_check_the_pass_does_not_make_is_refused_before_any_run() {
    let mut f = Fixture::new("check").unwrap();
    let check = Structural::default();
    let prepare = |f: &Fixture| {
        Pass::prepare(&f.config, &f.set, SMOKE, &check, &f.root, FRI).map(|p| p.summary().clone())
    };
    // The committed set names only the structural check.
    prepare(&f).unwrap();
    // A brief outside this pass counts too: a later pass must not be the first to find it.
    let b = f.set.iter_mut().find(|b| b.id == "plain-05").unwrap();
    b.acceptance = "structual".into();
    let e = prepare(&f).unwrap_err();
    assert!(
        matches!(&e, PassError::Invalid(m) if m.contains("plain-05")),
        "{e}"
    );
    assert!(!f.root.exists());
}
