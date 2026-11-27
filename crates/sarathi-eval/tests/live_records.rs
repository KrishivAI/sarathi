// SPDX-License-Identifier: Apache-2.0
//! A budgeted pass that saves each record as its run ends can be stopped anywhere and picked up again: no spend record
//! is lost, no run is made twice, and the result is that of a pass that never stopped.

use std::cell::RefCell;
use std::path::PathBuf;

use sarathi_eval::brief::{Brief, Split};
use sarathi_eval::config::{
    CappedRun, CappedRunError, ConfigError, Journal, RunConfig, run_within_caps,
    run_within_caps_journaled,
};
use sarathi_eval::metrics::{Purpose, RunRecord, TokenUsage};
use sarathi_eval::runner::{
    Acceptance, Arm, Harness, HarnessError, HarnessOutput, RunnerError, read_records, record_path,
    save_record,
};

const A9: &str = include_str!("../../../evals/vyasa/a9-run.json");
const DATE: &str = "2026-11-28";

/// One million uncached input tokens a run ($2 on `strong`), and a log of the runs made.
struct Spender<'a> {
    name: &'static str,
    log: &'a RefCell<Vec<String>>,
}
impl Harness for Spender<'_> {
    fn run(&self, b: &Brief, _m: &str, run: u32) -> Result<HarnessOutput, HarnessError> {
        self.log
            .borrow_mut()
            .push(format!("run {} {} {run}", self.name, b.id));
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
            allowed_models: vec!["strong".into()],
            acceptance: "structural".into(),
            fault: None,
        })
        .collect()
}

/// The A9 configuration with a `strong` cap of `cap` dollars.
fn config(cap: u64) -> Result<RunConfig, ConfigError> {
    let mut c = RunConfig::from_json(A9)?;
    c.slots[0].cap_usd = cap;
    Ok(c)
}

/// One pass of `old` and `new` on `strong` over `bs`.
fn pass(
    c: &RunConfig,
    bs: &[Brief],
    log: &RefCell<Vec<String>>,
    earlier: &[RunRecord],
    done: &[RunRecord],
    save: &mut dyn FnMut(&RunRecord) -> Result<(), RunnerError>,
) -> Result<CappedRun, CappedRunError> {
    let old = Spender { name: "old", log };
    let new = Spender { name: "new", log };
    let model = c.slots[0].model.clone();
    let arms = [("old", &old), ("new", &new)].map(|(h, harness)| Arm {
        name: RunConfig::arm_name(h, "strong"),
        slot: "strong".into(),
        model: model.clone(),
        harness,
    });
    run_within_caps_journaled(
        c,
        bs,
        &arms,
        &Always,
        Purpose::Measure,
        earlier,
        Journal { done, save },
    )
}

fn temp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sarathi-eval-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

#[test]
fn each_record_is_saved_before_the_next_run_starts() {
    let c = config(170).unwrap();
    let log = RefCell::new(Vec::new());
    let out = pass(&c, &briefs(2), &log, &[], &[], &mut |r| {
        log.borrow_mut()
            .push(format!("save {} {} {}", r.arm, r.brief, r.run));
        Ok(())
    })
    .unwrap();
    let log = log.into_inner();
    assert_eq!(out.records.len(), 12);
    assert_eq!(log.len(), 24);
    for pair in log.chunks(2) {
        let run: Vec<&str> = pair[0].split(' ').collect();
        assert_eq!(run[0], "run", "{pair:?}");
        assert_eq!(
            pair[1],
            format!("save {}.strong {} {}", run[1], run[2], run[3])
        );
    }
}

#[test]
fn a_record_that_cannot_be_saved_stops_the_pass() {
    let c = config(170).unwrap();
    let log = RefCell::new(Vec::new());
    let mut saves = 0;
    let e = pass(&c, &briefs(3), &log, &[], &[], &mut |_| {
        saves += 1;
        if saves < 2 {
            Ok(())
        } else {
            Err(RunnerError::Io("disk full".into()))
        }
    })
    .unwrap_err();
    assert_eq!(e, CappedRunError::Save(RunnerError::Io("disk full".into())));
    assert_eq!(log.borrow().len(), 2, "nothing runs after the failed save");
}

#[test]
fn a_pass_stopped_after_any_run_resumes_to_the_same_result() {
    // A $9 cap stops the pass part way (three groups of two $2 runs), so the resumed pass has to reach the same stop.
    let c = config(9).unwrap();
    let bs = briefs(3);
    let whole_log = RefCell::new(Vec::new());
    let whole = pass(&c, &bs, &whole_log, &[], &[], &mut |_| Ok(())).unwrap();
    assert_eq!(whole.records.len(), 6);
    assert!(whole.slot_stopped("strong"));
    for after in 0..=whole.records.len() {
        let dir = temp(&format!("resume-{after}"));
        let log = RefCell::new(Vec::new());
        // The first process: its save works `after` times, then it dies.
        let mut saves = 0;
        let first = pass(&c, &bs, &log, &[], &[], &mut |r| {
            if saves == after {
                return Err(RunnerError::Io("killed".into()));
            }
            saves += 1;
            save_record(&dir, DATE, r)
        });
        assert_eq!(first.is_err(), after < whole.records.len(), "{after}");
        let done = read_records(&dir, DATE, Purpose::Measure).unwrap();
        assert_eq!(done.len(), after);
        log.borrow_mut().clear();
        let out = pass(&c, &bs, &log, &[], &done, &mut |r| {
            save_record(&dir, DATE, r)
        })
        .unwrap();
        assert_eq!(out, whole, "stopped after {after}");
        assert_eq!(
            *log.borrow(),
            whole_log.borrow()[after..],
            "only the missing runs are made, stopped after {after}"
        );
        assert_eq!(
            read_records(&dir, DATE, Purpose::Measure).unwrap().len(),
            whole.records.len()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[test]
fn a_started_group_is_finished_even_past_the_cap() {
    // The first run of the group spent past the $1 cap before the stop: its pair still runs, so the arms stay paired.
    let c = config(1).unwrap();
    let bs = briefs(2);
    let log = RefCell::new(Vec::new());
    let whole = pass(&c, &bs, &log, &[], &[], &mut |_| Ok(())).unwrap();
    assert_eq!(whole.records.len(), 2);
    // Either member of the group may be the one on disk.
    for kept in 0..2 {
        log.borrow_mut().clear();
        let done = [whole.records[kept].clone()];
        let out = pass(&c, &bs, &log, &[], &done, &mut |_| Ok(())).unwrap();
        assert_eq!(out, whole, "kept {kept}");
        assert_eq!(log.borrow().len(), 1);
    }
}

#[test]
fn earlier_passes_and_done_runs_both_count_towards_the_cap() {
    let c = config(9).unwrap();
    let bs = briefs(3);
    let log = RefCell::new(Vec::new());
    let whole = pass(&c, &bs, &log, &[], &[], &mut |_| Ok(())).unwrap();
    // $6 spent by an earlier pass, $4 by the first group of this one: $10 is past the $9 cap, so no group starts.
    let earlier = whole.records[..3].to_vec();
    let done = whole.records[..2].to_vec();
    log.borrow_mut().clear();
    let out = pass(&c, &bs, &log, &earlier, &done, &mut |_| Ok(())).unwrap();
    assert_eq!(out.records, done);
    assert!(out.slot_stopped("strong"));
    assert!(log.borrow().is_empty());
}

/// One change that makes a record foreign to the pass.
type Edit = fn(&mut RunRecord);

#[test]
fn a_done_record_that_is_not_part_of_the_pass_is_refused_before_any_run() {
    let c = config(170).unwrap();
    let bs = briefs(1);
    let log = RefCell::new(Vec::new());
    let whole = pass(&c, &bs, &log, &[], &[], &mut |_| Ok(())).unwrap();
    let good = whole.records[0].clone();
    let edits: [(&str, Edit); 6] = [
        ("brief", |r| r.brief = "b99".into()),
        ("arm", |r| r.arm = "other.strong".into()),
        ("run", |r| r.run = 3),
        ("model", |r| r.model = "another-model".into()),
        ("slot", |r| r.slot = "cheap-tools".into()),
        ("purpose", |r| r.purpose = Purpose::Tune),
    ];
    for (why, edit) in edits {
        let mut bad = good.clone();
        edit(&mut bad);
        log.borrow_mut().clear();
        let e = pass(&c, &bs, &log, &[], &[bad], &mut |_| Ok(())).unwrap_err();
        assert!(matches!(e, CappedRunError::Done { .. }), "{why}: {e}");
        assert!(log.borrow().is_empty(), "{why}");
    }
    log.borrow_mut().clear();
    let twice = [good.clone(), good];
    let e = pass(&c, &bs, &log, &[], &twice, &mut |_| Ok(())).unwrap_err();
    assert!(matches!(e, CappedRunError::Done { .. }), "twice: {e}");
    assert!(log.borrow().is_empty());
}

#[test]
fn a_pass_without_a_journal_is_unchanged() {
    let c = config(9).unwrap();
    let bs = briefs(3);
    let log = RefCell::new(Vec::new());
    let journaled = pass(&c, &bs, &log, &[], &[], &mut |_| Ok(())).unwrap();
    let h = Spender {
        name: "old",
        log: &log,
    };
    let n = Spender {
        name: "new",
        log: &log,
    };
    let arms = [("old", &h), ("new", &n)].map(|(name, harness)| Arm {
        name: RunConfig::arm_name(name, "strong"),
        slot: "strong".into(),
        model: c.slots[0].model.clone(),
        harness,
    });
    let plain = run_within_caps(&c, &bs, &arms, &Always, Purpose::Measure, &[]).unwrap();
    assert_eq!(plain, journaled);
}

/// The first record of a pass.
fn record() -> Option<RunRecord> {
    let c = config(170).ok()?;
    let log = RefCell::new(Vec::new());
    let out = pass(&c, &briefs(1), &log, &[], &[], &mut |_| Ok(())).ok()?;
    out.records.into_iter().next()
}

#[test]
fn a_saved_record_reads_back_and_is_never_overwritten() {
    let dir = temp("save");
    assert_eq!(
        read_records(&dir, DATE, Purpose::Measure).unwrap(),
        [],
        "nothing saved yet"
    );
    let r = record().unwrap();
    save_record(&dir, DATE, &r).unwrap();
    assert_eq!(
        read_records(&dir, DATE, Purpose::Measure).unwrap(),
        std::slice::from_ref(&r)
    );
    assert_eq!(read_records(&dir, DATE, Purpose::Tune).unwrap(), []);
    let mut other = r.clone();
    other.usage.input_tokens = 7;
    assert!(matches!(
        save_record(&dir, DATE, &other),
        Err(RunnerError::Io(_))
    ));
    assert_eq!(
        read_records(&dir, DATE, Purpose::Measure).unwrap(),
        [r],
        "the first record stands"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn ids_that_leave_the_directory_are_refused() {
    let dir = temp("leave");
    assert!(matches!(
        read_records(&dir, "../x", Purpose::Measure),
        Err(RunnerError::Invalid(_))
    ));
    let mut r = record().unwrap();
    r.brief = "../b".into();
    assert!(matches!(
        save_record(&dir, DATE, &r),
        Err(RunnerError::Invalid(_))
    ));
    assert!(!dir.exists());
}

#[test]
fn a_file_that_is_not_a_whole_record_in_its_place_is_an_error_not_skipped() {
    let r = record().unwrap();
    let good = serde_json::to_string(&r).unwrap();
    let mut newer = r.clone();
    newer.version = 2;
    let mut zero = r.clone();
    zero.version = 0;
    let mut moved = r.clone();
    moved.run = 1;
    for (why, file, text) in [
        ("cut short", "0.json", good[..good.len() / 2].to_string()),
        ("left over from a save", "1.json.tmp", good.clone()),
        (
            "a newer version",
            "0.json",
            serde_json::to_string(&newer).unwrap(),
        ),
        (
            "a version that never was",
            "0.json",
            serde_json::to_string(&zero).unwrap(),
        ),
        (
            "in another run's place",
            "0.json",
            serde_json::to_string(&moved).unwrap(),
        ),
    ] {
        let dir = temp("broken");
        let path = record_path(&dir, DATE, &r).with_file_name(file);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, text).unwrap();
        let e = read_records(&dir, DATE, Purpose::Measure).unwrap_err();
        assert!(e.to_string().contains(file), "{why}: {e}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
