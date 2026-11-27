// SPDX-License-Identifier: Apache-2.0
//! One pass of a measurement that spends money: the smoke run, a tuning round or the measurement, over one directory
//! that holds every pass's records.
//!
//! The directory is `<root>/<pass>/<date>/`, with `<pass>` one of `smoke`, `tune-<round>` and `measure`; below the date
//! are the records ([`crate::runner::save_record`]) and the pass's `report.md` and `report.json`. Every record under
//! the root counts towards the spend caps, so the earlier passes are read from disk and never listed by hand. A pass
//! has one date: a stopped pass is resumed under the date it started with.
//!
//! [`Pass::prepare`] only reads: its [`Summary`] is the dry run. [`Pass::run`] is the only function here that calls a
//! harness. It takes the root's lock, reads the directory again, and saves each record as its run ends.

use std::collections::BTreeSet;
use std::fs::File;
use std::path::{Path, PathBuf};

use crate::brief::Brief;
use crate::config::{
    CappedRun, CappedRunError, Journal, RunConfig, SlotConfig, done_runs,
    run_within_caps_journaled, spent_by_slot,
};
use crate::measure::{self, Judging, MeasureError};
use crate::metrics::{MetricsError, Purpose, RunRecord};
use crate::report::{Report, to_json, to_markdown};
use crate::runner::{
    Acceptance, Arm, Harness, HarnessError, HarnessOutput, RunnerError, check_component,
    check_plan, read_records, save_record,
};

/// The name of the smoke pass's directory.
const SMOKE: &str = "smoke";
/// The name of the measurement's directory.
const MEASURE: &str = "measure";
/// The prefix of a tuning round's directory.
const TUNE: &str = "tune-";
/// The lock file in the root.
const LOCK: &str = "lock";
/// The files a pass writes beside its records.
const REPORTS: [&str; 2] = ["report.md", "report.json"];

/// Which pass to make.
#[derive(Debug, Clone, Copy)]
pub enum PassKind<'a> {
    /// The small run first: the named measurement briefs, once on every slot and harness. Its records price the
    /// measurement.
    Smoke {
        /// Brief ids, all in the measurement split.
        briefs: &'a [&'a str],
    },
    /// A tuning round: every tuning brief, once on one slot, every harness.
    Tune {
        /// The round, from 1; each round has its own directory.
        round: u32,
        /// The slot it runs on.
        slot: &'a str,
    },
    /// The measurement: every measurement brief on every slot and harness, as often as the configuration says, judged
    /// by a rule fixed before the first run.
    Measure {
        /// The kill-line rule.
        judging: Judging<'a>,
    },
}

impl PassKind<'_> {
    fn dir_name(&self) -> String {
        match self {
            Self::Smoke { .. } => SMOKE.into(),
            Self::Tune { round, .. } => format!("{TUNE}{round}"),
            Self::Measure { .. } => MEASURE.into(),
        }
    }
}

/// A pass that cannot be made, or stopped.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum PassError {
    /// The request does not fit the brief set or the configuration.
    #[error("pass: {0}")]
    Invalid(String),
    /// The root holds something that cannot be counted, or cannot be read or written.
    #[error("pass directory: {0}")]
    Dir(String),
    /// Another process is running a pass on this root.
    #[error("another pass holds the lock of {0}")]
    Locked(String),
    /// The measurement may not start (yet).
    #[error("the measurement may not start: {0}")]
    NotReady(String),
    /// A record is not this pass's, a record could not be saved, or the budgeted run refused to start.
    #[error(transparent)]
    Run(#[from] CappedRunError),
    /// The runner refused the plan, or a record could not be read.
    #[error(transparent)]
    Runner(#[from] RunnerError),
    /// A record could not be priced.
    #[error(transparent)]
    Metrics(#[from] MetricsError),
    /// The report could not be made.
    #[error(transparent)]
    Measure(#[from] MeasureError),
}

/// One slot of a pass before it runs. Money is in micro-dollars.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotSummary {
    /// The slot.
    pub slot: String,
    /// The model it runs.
    pub model: String,
    /// Runs the pass plans on this slot.
    pub runs_planned: usize,
    /// How many of them are on disk.
    pub runs_done: usize,
    /// Spend of every record under the root on this slot.
    pub spent_micro: u64,
    /// The slot's cap.
    pub cap_micro: u64,
    /// For the measurement: the spend so far plus the missing runs at the smoke run's mean cost per arm. `None` for
    /// other passes, and when an arm of the slot has no finished smoke run.
    pub projected_micro: Option<u64>,
}

/// What a pass would do: the dry run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Summary {
    /// The pass's directory, `<root>/<pass>`.
    pub dir: PathBuf,
    /// One entry per slot of the pass, in configuration order.
    pub slots: Vec<SlotSummary>,
}

/// A checked pass and what the root already holds.
pub struct Pass<'a> {
    /// The check that judges the runs, and that handles every brief's `acceptance` name.
    acceptance: &'a dyn Acceptance,
    /// The configuration and brief set as given, to check again under the lock.
    given: (&'a RunConfig, &'a [Brief]),
    /// The configuration narrowed to this pass (its slots and repetitions).
    config: RunConfig,
    briefs: Vec<Brief>,
    kind: PassKind<'a>,
    purpose: Purpose,
    root: &'a Path,
    date: &'a str,
    /// This pass's records.
    done: Vec<RunRecord>,
    /// Other passes' records on this pass's slots.
    earlier: Vec<RunRecord>,
    summary: Summary,
}

impl std::fmt::Debug for Pass<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pass")
            .field("kind", &self.kind)
            .field("date", &self.date)
            .field("summary", &self.summary)
            .finish_non_exhaustive()
    }
}

/// Stands in for a harness where only the plan is checked.
struct Unused;
impl Harness for Unused {
    fn run(&self, _: &Brief, _: &str, _: u32) -> Result<HarnessOutput, HarnessError> {
        Err(HarnessError::new("a dry run makes no run"))
    }
}

impl<'a> Pass<'a> {
    /// Check the pass against the brief set `set`, the configuration, the check that will judge it and the records
    /// under `root`. Reads only.
    ///
    /// # Errors
    /// [`PassError::Invalid`] for a brief anywhere in `set` whose `acceptance` name `acceptance` does not handle, an
    /// unknown or repeated smoke brief, tuning round 0 or an unknown slot; [`PassError::Runner`] for a tuning brief in
    /// the smoke run;
    /// [`PassError::Dir`] for anything under the root that is not a pass's records or reports, or a second date for
    /// this pass; the runner's, the records' and the judging rule's errors as their variants.
    pub fn prepare(
        config: &'a RunConfig,
        set: &'a [Brief],
        kind: PassKind<'a>,
        acceptance: &'a dyn Acceptance,
        root: &'a Path,
        date: &'a str,
    ) -> Result<Self, PassError> {
        // Before any spend: a date that no record could be saved under.
        check_component("date", date)?;
        // Also before any spend, and over the whole set, so a later pass cannot be the first to find it.
        if let Some(b) = set.iter().find(|b| !acceptance.handles(&b.acceptance)) {
            return Err(PassError::Invalid(format!(
                "brief `{}` names the acceptance check `{}`, which this pass's check does not handle",
                b.id, b.acceptance
            )));
        }
        let given = (config, set);
        let (config, briefs, purpose) = narrow(config, set, &kind)?;
        let name = kind.dir_name();
        let mut done = Vec::new();
        let mut earlier = Vec::new();
        let mut smoke = Vec::new();
        for (pass, started, records) in read_root(root)? {
            if pass == name {
                if started != date {
                    return Err(PassError::Dir(format!(
                        "pass `{name}` started on {started}: resume it with that date, not {date}"
                    )));
                }
                done = records;
            } else {
                if pass == SMOKE {
                    smoke.clone_from(&records);
                }
                earlier.extend(records);
            }
        }
        let all = given.0;
        if let Some(r) = earlier.iter().find(|r| all.slot(&r.slot).is_none()) {
            return Err(PassError::Dir(format!(
                "the record of brief `{}`, arm `{}`, is on slot `{}`, which is not configured: its spend cannot be \
                 counted",
                r.brief, r.arm, r.slot
            )));
        }
        // Spend on a slot this pass does not run cannot move its caps (and its model may have no price here).
        earlier.retain(|r| config.slot(&r.slot).is_some());

        let unused = Unused;
        let arms = arms(&config, |_, _| &unused);
        check_plan(&briefs, &arms, config.runs, purpose)?;
        done_runs(&done, &briefs, &arms, config.runs, purpose)?;

        let prices = config.price_table();
        let mut spent = spent_by_slot(&earlier, &prices)?;
        for (slot, cost) in spent_by_slot(&done, &prices)? {
            let s = spent.entry(slot).or_default();
            *s = s.saturating_add(cost);
        }
        let per_arm = briefs.len() * config.runs as usize;
        let mut slots = Vec::new();
        for s in &config.slots {
            let spent_micro = spent.get(&s.slot).copied().unwrap_or(0);
            let projected_micro = match kind {
                PassKind::Measure { .. } => {
                    project(&config, s, &smoke, &done, per_arm, spent_micro)?
                }
                _ => None,
            };
            slots.push(SlotSummary {
                slot: s.slot.clone(),
                model: s.model.clone(),
                runs_planned: per_arm * config.harnesses.len(),
                runs_done: done.iter().filter(|r| r.slot == s.slot).count(),
                spent_micro,
                cap_micro: s.cap_usd.saturating_mul(1_000_000),
                projected_micro,
            });
        }
        let summary = Summary {
            dir: root.join(&name),
            slots,
        };
        Ok(Self {
            acceptance,
            given,
            config,
            briefs,
            kind,
            purpose,
            root,
            date,
            done,
            earlier,
            summary,
        })
    }

    /// What [`Pass::run`] would do.
    #[must_use]
    pub fn summary(&self) -> &Summary {
        &self.summary
    }

    /// Make the missing runs, saving each record as it ends, then write the pass's reports. `harness` gives the harness
    /// of a (harness name, slot); the arm's name and model come from the configuration.
    ///
    /// A measurement with no record yet starts only if every arm has a finished smoke run and every slot's projected spend
    /// is within its cap. A started one is resumed whatever the projection: the caps still stop it.
    ///
    /// # Errors
    /// [`PassError::Locked`] while another pass runs on the root; [`PassError::NotReady`] as above; otherwise as
    /// [`Pass::prepare`], plus a record or report that cannot be written.
    pub fn run<'h>(
        self,
        harness: impl Fn(&str, &SlotConfig) -> &'h dyn Harness,
        deviations: u32,
    ) -> Result<(CappedRun, Report), PassError> {
        let dir_err =
            |path: &Path, e: std::io::Error| PassError::Dir(format!("{}: {e}", path.display()));
        std::fs::create_dir_all(self.root).map_err(|e| dir_err(self.root, e))?;
        let lock_path = self.root.join(LOCK);
        // Held until this function returns.
        let lock = File::options()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&lock_path)
            .map_err(|e| dir_err(&lock_path, e))?;
        lock.try_lock().map_err(|e| match e {
            std::fs::TryLockError::WouldBlock => PassError::Locked(self.root.display().to_string()),
            std::fs::TryLockError::Error(e) => dir_err(&lock_path, e),
        })?;
        // Another pass may have added records between `prepare` and the lock.
        let (config, set) = self.given;
        let this = Pass::prepare(
            config,
            set,
            self.kind,
            self.acceptance,
            self.root,
            self.date,
        )?;
        if matches!(this.kind, PassKind::Measure { .. }) && this.done.is_empty() {
            ready(&this.summary)?;
        }
        let arms = arms(&this.config, harness);
        let pass_dir = this.summary.dir.clone();
        let date = this.date;
        let mut save = |r: &RunRecord| save_record(&pass_dir, date, r);
        let run = run_within_caps_journaled(
            &this.config,
            &this.briefs,
            &arms,
            this.acceptance,
            this.purpose,
            &this.earlier,
            Journal {
                done: &this.done,
                save: &mut save,
            },
        )?;
        let judging = match &this.kind {
            PassKind::Measure { judging } => Some(judging),
            _ => None,
        };
        let report = measure::report(&this.config, &run, judging, date, deviations)?;
        let out = pass_dir.join(date);
        std::fs::create_dir_all(&out).map_err(|e| dir_err(&out, e))?;
        let json = to_json(&report).map_err(|e| PassError::Dir(e.to_string()))?;
        for (file, text) in REPORTS.iter().zip([to_markdown(&report), json]) {
            let path = out.join(file);
            std::fs::write(&path, text).map_err(|e| dir_err(&path, e))?;
        }
        Ok((run, report))
    }
}

/// The configuration, briefs and purpose of one pass.
fn narrow(
    config: &RunConfig,
    set: &[Brief],
    kind: &PassKind<'_>,
) -> Result<(RunConfig, Vec<Brief>, Purpose), PassError> {
    let mut c = config.clone();
    let split = |s| set.iter().filter(move |b| b.split == s).cloned().collect();
    let (briefs, purpose): (Vec<Brief>, _) = match kind {
        PassKind::Smoke { briefs } => {
            c.runs = 1;
            let mut out = Vec::new();
            let mut seen = BTreeSet::new();
            for id in *briefs {
                let Some(b) = set.iter().find(|b| b.id == *id) else {
                    return Err(PassError::Invalid(format!("no brief `{id}` in the set")));
                };
                if !seen.insert(id) {
                    return Err(PassError::Invalid(format!("brief `{id}` is named twice")));
                }
                out.push(b.clone());
            }
            (out, Purpose::Measure)
        }
        PassKind::Tune { round, slot } => {
            if *round == 0 {
                return Err(PassError::Invalid("tuning rounds count from 1".into()));
            }
            c.runs = 1;
            c.slots.retain(|s| s.slot == *slot);
            if c.slots.is_empty() {
                return Err(PassError::Invalid(format!(
                    "slot `{slot}` is not configured"
                )));
            }
            (split(crate::brief::Split::Tuning), Purpose::Tune)
        }
        PassKind::Measure { judging } => {
            // Before any spend: a rule the report would refuse afterwards.
            measure::check(&c, judging)?;
            (split(crate::brief::Split::Measurement), Purpose::Measure)
        }
    };
    Ok((c, briefs, purpose))
}

/// Every arm of `config`: each harness on each slot.
fn arms<'h>(
    config: &RunConfig,
    harness: impl Fn(&str, &SlotConfig) -> &'h dyn Harness,
) -> Vec<Arm<'h>> {
    config
        .slots
        .iter()
        .flat_map(|s| config.harnesses.iter().map(move |h| (h, s)))
        .map(|(h, s)| Arm {
            name: RunConfig::arm_name(h, &s.slot),
            slot: s.slot.clone(),
            model: s.model.clone(),
            harness: harness(h, s),
        })
        .collect()
}

/// The spend of `slot` once its missing runs are made, each at the mean cost of its arm's smoke runs on the same model
/// that finished; `None` if an arm has none. A failed run is left out: one that never reached the model costs nothing
/// and would price the measurement at nothing.
fn project(
    config: &RunConfig,
    slot: &SlotConfig,
    smoke: &[RunRecord],
    done: &[RunRecord],
    per_arm: usize,
    spent: u64,
) -> Result<Option<u64>, PassError> {
    let prices = config.price_table();
    let mut total = spent;
    for h in &config.harnesses {
        let arm = RunConfig::arm_name(h, &slot.slot);
        let mut sum = 0u64;
        let mut n = 0u64;
        for r in smoke
            .iter()
            .filter(|r| r.arm == arm && r.model == slot.model && r.error.is_none())
        {
            sum = sum.saturating_add(prices.cost_micro(r)?);
            n += 1;
        }
        if n == 0 {
            return Ok(None);
        }
        let made = done.iter().filter(|r| r.arm == arm).count();
        let missing = per_arm.saturating_sub(made) as u64;
        total = total.saturating_add(sum.div_ceil(n).saturating_mul(missing));
    }
    Ok(Some(total))
}

/// The start condition of the measurement.
fn ready(summary: &Summary) -> Result<(), PassError> {
    for s in &summary.slots {
        let Some(p) = s.projected_micro else {
            return Err(PassError::NotReady(format!(
                "slot `{}` has an arm with no finished smoke run on model `{}`: make the smoke run first",
                s.slot, s.model
            )));
        };
        if p > s.cap_micro {
            return Err(PassError::NotReady(format!(
                "slot `{}` is projected to spend {p} micro-dollars, over its cap of {}",
                s.slot, s.cap_micro
            )));
        }
    }
    Ok(())
}

/// Every pass under `root` as (pass, date, records). A pass with no date yet is left out.
fn read_root(root: &Path) -> Result<Vec<(String, String, Vec<RunRecord>)>, PassError> {
    let mut out = Vec::new();
    match root.try_exists() {
        Ok(false) => return Ok(out),
        Ok(true) => {}
        Err(e) => return Err(PassError::Dir(format!("{}: {e}", root.display()))),
    }
    for (pass, path) in named_entries(root)? {
        if pass == LOCK {
            continue;
        }
        let round = pass.strip_prefix(TUNE).and_then(|n| n.parse::<u32>().ok());
        let known = pass == SMOKE
            || pass == MEASURE
            || round.is_some_and(|n| n > 0 && pass == format!("{TUNE}{n}"));
        if !known {
            return Err(unknown(&path));
        }
        let mut dates = named_entries(&path)?;
        if dates.len() > 1 {
            return Err(PassError::Dir(format!(
                "{}: a pass has one date, found {}",
                path.display(),
                dates.len()
            )));
        }
        let Some((date, dated)) = dates.pop() else {
            continue;
        };
        let mut records = Vec::new();
        for (name, entry) in named_entries(&dated)? {
            let purpose = [Purpose::Tune, Purpose::Measure]
                .into_iter()
                .find(|p| p.dir() == name);
            match purpose {
                Some(p) => records.extend(read_records(&path, &date, p)?),
                None if REPORTS.contains(&name.as_str()) => {}
                None => return Err(unknown(&entry)),
            }
        }
        out.push((pass, date, records));
    }
    Ok(out)
}

fn unknown(path: &Path) -> PassError {
    PassError::Dir(format!(
        "{}: not part of a pass; a record that is skipped is spend that goes uncounted",
        path.display()
    ))
}

/// The entries of `dir` with their names, sorted.
fn named_entries(dir: &Path) -> Result<Vec<(String, PathBuf)>, PassError> {
    let io = |e: std::io::Error| PassError::Dir(format!("{}: {e}", dir.display()));
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).map_err(io)? {
        let e = e.map_err(io)?;
        let name = e
            .file_name()
            .into_string()
            .map_err(|_| unknown(&e.path()))?;
        out.push((name, e.path()));
    }
    out.sort();
    Ok(out)
}
