// SPDX-License-Identifier: Apache-2.0
//! The brief runner: the `Harness` and `Acceptance` traits, the run plan (interleaved across arms), the split guard and
//! the run directory. A run that fails is recorded as a failed run, never dropped. Nothing here knows which provider a
//! harness uses; a harness decides that, so the same runner measures a recorded run and a live one.

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::brief::{Brief, Split};
use crate::metrics::{RunRecord, TokenUsage};

/// What a harness reports for one run of one brief.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct HarnessOutput {
    /// What the run produced, handed to the acceptance check.
    pub artifact: String,
    /// Operations the run issued that were valid.
    pub valid_ops: u32,
    /// Operations the run issued.
    pub total_ops: u32,
    /// Tokens used.
    pub usage: TokenUsage,
    /// Model calls made.
    pub steps: u32,
    /// Wall time in milliseconds, measured by the harness (so tests inject it).
    pub wall_ms: u64,
}

/// A harness could not finish a run.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct HarnessError(pub String);

/// An agent harness under measurement.
pub trait Harness {
    /// Run `brief` on `model`; `run` is the repetition number from 0.
    ///
    /// # Errors
    /// [`HarnessError`] when the run failed; the runner records it as a failed run.
    fn run(&self, brief: &Brief, model: &str, run: u32) -> Result<HarnessOutput, HarnessError>;
}

/// The check that decides whether a run's artifact is accepted. Named by `Brief::acceptance`.
pub trait Acceptance {
    /// Whether `out` satisfies `brief`.
    fn accepts(&self, brief: &Brief, out: &HarnessOutput) -> bool;
}

/// One arm: a name, a model, and the harness that runs it.
pub struct Arm<'a> {
    /// Arm name in the report; every brief run on this arm must list it in `allowed_models` (briefs name arms, not
    /// model ids, so the same set runs against whichever model a run configures for the arm).
    pub name: String,
    /// Model id, recorded with every run and used to price it.
    pub model: String,
    /// The harness.
    pub harness: &'a dyn Harness,
}

/// Why briefs are being run. A tuning pass never touches the measurement split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Purpose {
    /// Developing against the tuning split.
    Tune,
    /// The measurement that decides the kill line.
    Measure,
}

/// The runner refused a plan.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RunnerError {
    /// A tuning pass was given a measurement brief.
    #[error("brief {0} is in the measurement split and may not be run to tune")]
    MeasurementInTuning(String),
    /// A measurement pass was given a tuning brief.
    #[error("brief {0} is in the tuning split and is not part of the measurement")]
    TuningInMeasurement(String),
    /// A brief does not allow an arm.
    #[error("brief {brief} does not allow arm {arm}")]
    ArmNotAllowed {
        /// The brief id.
        brief: String,
        /// The arm name.
        arm: String,
    },
    /// Two arms share a name, or an id would leave the run directory.
    #[error("{0}")]
    Invalid(String),
    /// No arms or no repetitions.
    #[error("a run needs at least one arm and one repetition")]
    Empty,
    /// Writing the run directory failed.
    #[error("run directory: {0}")]
    Io(String),
}

/// The split guard: the briefs must all belong to the split `purpose` allows.
///
/// # Errors
/// [`RunnerError::MeasurementInTuning`] or [`RunnerError::TuningInMeasurement`] naming the first offender.
pub fn guard_split(briefs: &[Brief], purpose: Purpose) -> Result<(), RunnerError> {
    for b in briefs {
        match (purpose, b.split) {
            (Purpose::Tune, Split::Measurement) => {
                return Err(RunnerError::MeasurementInTuning(b.id.clone()));
            }
            (Purpose::Measure, Split::Tuning) => {
                return Err(RunnerError::TuningInMeasurement(b.id.clone()));
            }
            _ => {}
        }
    }
    Ok(())
}

/// One planned run: indices into the briefs and arms, and the repetition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Planned {
    /// Index into the brief list.
    pub brief: usize,
    /// Index into the arm list.
    pub arm: usize,
    /// Repetition, from 0.
    pub run: u32,
}

/// The order runs execute in: repetition by repetition, brief by brief (sorted by id), the arms alternating their order
/// on odd briefs so that drift in the provider (load, price, a model update) falls on every arm alike.
#[must_use]
pub fn plan(briefs: &[Brief], arms: usize, k: u32) -> Vec<Planned> {
    let mut order: Vec<usize> = (0..briefs.len()).collect();
    order.sort_by(|a, b| briefs[*a].id.cmp(&briefs[*b].id));
    let mut out = Vec::with_capacity(briefs.len() * arms * k as usize);
    for run in 0..k {
        for (pos, &brief) in order.iter().enumerate() {
            let arm_order: Vec<usize> = if pos % 2 == 0 {
                (0..arms).collect()
            } else {
                (0..arms).rev().collect()
            };
            for arm in arm_order {
                out.push(Planned { brief, arm, run });
            }
        }
    }
    out
}

/// Run every planned run and return one record per run, in plan order.
///
/// # Errors
/// A refused plan ([`RunnerError`]); a failing run is not an error, it is a record with `error` set.
pub fn run_briefs(
    briefs: &[Brief],
    arms: &[Arm<'_>],
    acceptance: &dyn Acceptance,
    k: u32,
    purpose: Purpose,
) -> Result<Vec<RunRecord>, RunnerError> {
    if arms.is_empty() || k == 0 || briefs.is_empty() {
        return Err(RunnerError::Empty);
    }
    guard_split(briefs, purpose)?;
    for (i, a) in arms.iter().enumerate() {
        check_component("arm name", &a.name)?;
        if arms[..i].iter().any(|p| p.name == a.name) {
            return Err(RunnerError::Invalid(format!(
                "arm name `{}` is used twice",
                a.name
            )));
        }
    }
    for b in briefs {
        check_component("brief id", &b.id)?;
        for a in arms {
            if !b.allowed_models.iter().any(|m| m == &a.name) {
                return Err(RunnerError::ArmNotAllowed {
                    brief: b.id.clone(),
                    arm: a.name.clone(),
                });
            }
        }
    }
    let mut records = Vec::new();
    for p in plan(briefs, arms.len(), k) {
        let (brief, arm) = (&briefs[p.brief], &arms[p.arm]);
        let mut rec = RunRecord {
            brief: brief.id.clone(),
            arm: arm.name.clone(),
            model: arm.model.clone(),
            run: p.run,
            accepted: false,
            valid_ops: 0,
            total_ops: 0,
            usage: TokenUsage::default(),
            steps: 0,
            wall_ms: 0,
            error: None,
        };
        match arm.harness.run(brief, &arm.model, p.run) {
            Ok(out) => {
                rec.accepted = acceptance.accepts(brief, &out);
                rec.valid_ops = out.valid_ops;
                rec.total_ops = out.total_ops;
                rec.usage = out.usage;
                rec.steps = out.steps;
                rec.wall_ms = out.wall_ms;
            }
            Err(e) => rec.error = Some(e.0),
        }
        records.push(rec);
    }
    Ok(records)
}

/// An id becomes a directory name, so it must be a single plain path component.
fn check_component(what: &str, id: &str) -> Result<(), RunnerError> {
    let bad = id.is_empty()
        || id == "."
        || id == ".."
        || id.contains(['/', '\\', '\0'])
        || id.contains(':');
    if bad {
        return Err(RunnerError::Invalid(format!(
            "{what} `{id}` is not a plain name (no path separators, `:` or `..`)"
        )));
    }
    Ok(())
}

/// Where a record is written: `<root>/<date>/<arm>/<brief>/<run>.json`.
#[must_use]
pub fn record_path(root: &Path, date: &str, r: &RunRecord) -> PathBuf {
    root.join(date)
        .join(&r.arm)
        .join(&r.brief)
        .join(format!("{}.json", r.run))
}

/// Write each record under the run directory (pretty JSON, trailing newline).
///
/// # Errors
/// [`RunnerError::Io`] on a failed create or write.
pub fn write_records(root: &Path, date: &str, records: &[RunRecord]) -> Result<(), RunnerError> {
    for r in records {
        let path = record_path(root, date, r);
        let io = |e: std::io::Error| RunnerError::Io(format!("{}: {e}", path.display()));
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(io)?;
        }
        let mut s = serde_json::to_string_pretty(&Wrapper(r))
            .map_err(|e| RunnerError::Io(e.to_string()))?;
        s.push('\n');
        std::fs::write(&path, s).map_err(io)?;
    }
    Ok(())
}

#[derive(Serialize)]
#[serde(transparent)]
struct Wrapper<'a>(&'a RunRecord);

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;

    fn brief(id: &str, split: Split) -> Brief {
        Brief {
            id: id.into(),
            split,
            site_context: "ctx".into(),
            text: "t".into(),
            allowed_models: vec!["new".into()],
            acceptance: "a".into(),
            fault: None,
        }
    }

    struct Fake {
        log: RefCell<Vec<(String, u32)>>,
        fail_on: Option<&'static str>,
    }
    impl Harness for Fake {
        fn run(&self, b: &Brief, _m: &str, run: u32) -> Result<HarnessOutput, HarnessError> {
            self.log.borrow_mut().push((b.id.clone(), run));
            if self.fail_on == Some(b.id.as_str()) {
                return Err(HarnessError("provider down".into()));
            }
            Ok(HarnessOutput {
                artifact: "ok".into(),
                valid_ops: 2,
                total_ops: 2,
                steps: 3,
                wall_ms: 10,
                ..Default::default()
            })
        }
    }
    struct AcceptOk;
    impl Acceptance for AcceptOk {
        fn accepts(&self, _b: &Brief, o: &HarnessOutput) -> bool {
            o.artifact == "ok"
        }
    }

    #[test]
    fn plan_alternates_arm_order_by_brief() {
        let bs = [
            brief("b", Split::Measurement),
            brief("a", Split::Measurement),
        ];
        let p = plan(&bs, 2, 1);
        let seq: Vec<(usize, usize)> = p.iter().map(|x| (x.brief, x.arm)).collect();
        // sorted: a (index 1) first with arms 0,1; then b (index 0) with arms 1,0
        assert_eq!(seq, vec![(1, 0), (1, 1), (0, 1), (0, 0)]);
    }

    #[test]
    fn plan_repeats_whole_set_per_run() {
        let bs = [brief("a", Split::Tuning)];
        let p = plan(&bs, 1, 3);
        assert_eq!(p.iter().map(|x| x.run).collect::<Vec<_>>(), vec![0, 1, 2]);
    }

    #[test]
    fn split_guard_names_the_offender() {
        let bs = [brief("a", Split::Tuning), brief("m1", Split::Measurement)];
        assert_eq!(
            guard_split(&bs, Purpose::Tune),
            Err(RunnerError::MeasurementInTuning("m1".into()))
        );
        assert_eq!(
            guard_split(&bs, Purpose::Measure),
            Err(RunnerError::TuningInMeasurement("a".into()))
        );
        assert!(guard_split(&bs[..1], Purpose::Tune).is_ok());
    }

    #[test]
    fn failed_run_is_recorded_not_dropped() {
        let h = Fake {
            log: RefCell::new(vec![]),
            fail_on: Some("a"),
        };
        let arms = [Arm {
            name: "new".into(),
            model: "m".into(),
            harness: &h,
        }];
        let bs = [
            brief("a", Split::Measurement),
            brief("b", Split::Measurement),
        ];
        let recs = run_briefs(&bs, &arms, &AcceptOk, 2, Purpose::Measure).unwrap_or_default();
        assert_eq!(recs.len(), 4);
        let a0 = &recs[0];
        assert_eq!((a0.brief.as_str(), a0.accepted), ("a", false));
        assert_eq!(a0.error.as_deref(), Some("provider down"));
        assert!(recs[1].accepted && recs[1].error.is_none());
    }

    #[test]
    fn an_arm_a_brief_does_not_allow_is_refused_before_any_run() {
        let h = Fake {
            log: RefCell::new(vec![]),
            fail_on: None,
        };
        let arms = [Arm {
            name: "x".into(),
            model: "m".into(),
            harness: &h,
        }];
        let bs = [brief("a", Split::Tuning)];
        let e = run_briefs(&bs, &arms, &AcceptOk, 1, Purpose::Tune).err();
        assert_eq!(
            e,
            Some(RunnerError::ArmNotAllowed {
                brief: "a".into(),
                arm: "x".into()
            })
        );
        assert!(h.log.borrow().is_empty());
    }

    #[test]
    fn ids_that_would_leave_the_run_directory_and_duplicate_arms_are_refused() {
        let h = Fake {
            log: RefCell::new(vec![]),
            fail_on: None,
        };
        let arm = |name: &str| Arm {
            name: name.into(),
            model: "m".into(),
            harness: &h,
        };
        for bad in ["/etc/evil", "..", "a/b", "a\\b", ""] {
            let mut b = brief(bad, Split::Tuning);
            b.allowed_models = vec!["new".into()];
            let e = run_briefs(&[b], &[arm("new")], &AcceptOk, 1, Purpose::Tune).err();
            assert!(matches!(e, Some(RunnerError::Invalid(_))), "{bad}: {e:?}");
        }
        let bs = [brief("a", Split::Tuning)];
        let e = run_briefs(&bs, &[arm("new"), arm("new")], &AcceptOk, 1, Purpose::Tune).err();
        assert!(matches!(e, Some(RunnerError::Invalid(_))), "{e:?}");
        assert!(h.log.borrow().is_empty());
    }

    #[test]
    fn empty_plan_is_refused() {
        let bs = [brief("a", Split::Tuning)];
        assert_eq!(
            run_briefs(&bs, &[], &AcceptOk, 1, Purpose::Tune).err(),
            Some(RunnerError::Empty)
        );
    }

    #[test]
    fn record_path_layout() {
        let r = RunRecord {
            brief: "b1".into(),
            arm: "old".into(),
            model: "m".into(),
            run: 2,
            accepted: true,
            valid_ops: 0,
            total_ops: 0,
            usage: TokenUsage::default(),
            steps: 0,
            wall_ms: 0,
            error: None,
        };
        assert_eq!(
            record_path(Path::new("runs"), "2026-11-28", &r),
            Path::new("runs/2026-11-28/old/b1/2.json")
        );
    }
}
