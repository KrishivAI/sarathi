// SPDX-License-Identifier: Apache-2.0
//! A run configuration file ([`RunConfig`]): which model fills each slot, its price, and the most a slot may spend; and
//! a budgeted run ([`run_within_caps`]) that stops a slot once its spend reaches its cap.
//!
//! The file is JSON with a `version` (this crate writes [`RUN_CONFIG_VERSION`] and reads it and the one before). Money
//! is whole dollars for caps and micro-dollars per million tokens for prices ([`Price`]), so every sum is exact.
//!
//! A cap is checked once per group: the runs of one brief and repetition on one slot, one per harness. A group whose
//! slot is still below its cap (from the cost of the runs already recorded) runs whole; once the slot reaches its cap,
//! every later group of the slot is skipped. So the harnesses of a stopped slot always ran the same (brief, repetition)
//! set and their stats stay comparable, and a slot can overshoot its cap by at most one group (one run per harness).
//! That bound is in runs, not dollars: it bounds money only while one run's cost is bounded, which the provider-side
//! spending limits on the keys ensure. Those limits are the hard stop; the cap is the measurement's own.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::brief::Brief;
use crate::metrics::{MetricsError, Price, PriceTable, Purpose, RunRecord};
use crate::runner::{Acceptance, Arm, Planned, RunnerError, check_plan, plan, run_one};

/// The run-configuration format version this crate writes.
pub const RUN_CONFIG_VERSION: u32 = 1;

/// One slot of a run: the model that fills it and what it may spend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SlotConfig {
    /// The slot name briefs list in `allowed_models` (`strong`, `cheap-tools`, `fallback-no-tools`).
    pub slot: String,
    /// The provider's model id, recorded with every run.
    pub model: String,
    /// Whether the harness offers the model native tool calls; `false` runs the JSON step protocol.
    pub native_tools: bool,
    /// The most this slot may spend over every harness and pass, in whole dollars.
    pub cap_usd: u64,
    /// The model's price.
    pub price: Price,
}

/// A run configuration: the slots, the harnesses that each run every slot, and the caps.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    /// Format version.
    pub version: u32,
    /// Runs per brief, per harness, per slot (`k`).
    pub runs: u32,
    /// The harness names; each runs every slot, as arm `<harness>.<slot>`.
    pub harnesses: Vec<String>,
    /// The slots.
    pub slots: Vec<SlotConfig>,
    /// The most the whole run may spend, in whole dollars; at least the sum of the slot caps. Only that sum is
    /// checked: real spend can pass it by each slot's overshoot (one group per slot, see the module doc).
    pub total_cap_usd: u64,
    /// Free text: where the prices were read and when (recorded, not interpreted).
    #[serde(default)]
    pub prices_checked: String,
}

/// A run configuration that cannot be used.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ConfigError {
    /// Not valid JSON for this format.
    #[error("run configuration: {0}")]
    Parse(String),
    /// A version this crate does not read.
    #[error(
        "run configuration version {0} (this crate reads {RUN_CONFIG_VERSION} and the version before)"
    )]
    Version(u32),
    /// A field that breaks a rule of the format.
    #[error("run configuration: {0}")]
    Invalid(String),
}

impl RunConfig {
    /// Read and check a configuration.
    ///
    /// # Errors
    /// [`ConfigError`] for malformed JSON, an unknown version, or a broken rule: no runs, no harness or slot, a
    /// name used twice, an empty model id, a zero cap, or slot caps adding up to more than the total cap.
    pub fn from_json(text: &str) -> Result<Self, ConfigError> {
        let c: RunConfig =
            serde_json::from_str(text).map_err(|e| ConfigError::Parse(e.to_string()))?;
        c.check()?;
        Ok(c)
    }

    fn check(&self) -> Result<(), ConfigError> {
        let bad = |m: String| Err(ConfigError::Invalid(m));
        if self.version == 0 || self.version > RUN_CONFIG_VERSION {
            return Err(ConfigError::Version(self.version));
        }
        if self.runs == 0 {
            return bad("`runs` must be at least 1".into());
        }
        if self.harnesses.is_empty() || self.slots.is_empty() {
            return bad("at least one harness and one slot are needed".into());
        }
        unique("harness", self.harnesses.iter().map(String::as_str))?;
        unique("slot", self.slots.iter().map(|s| s.slot.as_str()))?;
        let mut sum = 0u64;
        for s in &self.slots {
            if s.model.trim().is_empty() {
                return bad(format!("slot `{}` has no model", s.slot));
            }
            if s.cap_usd == 0 {
                return bad(format!("slot `{}` has a zero cap", s.slot));
            }
            sum = sum.saturating_add(s.cap_usd);
        }
        if sum > self.total_cap_usd {
            return bad(format!(
                "slot caps add up to ${sum}, more than the total cap ${}",
                self.total_cap_usd
            ));
        }
        Ok(())
    }

    /// The price table of every slot's model (one table for every harness, so both are priced alike).
    #[must_use]
    pub fn price_table(&self) -> PriceTable {
        PriceTable {
            models: self
                .slots
                .iter()
                .map(|s| (s.model.clone(), s.price))
                .collect(),
        }
    }

    /// The slot named `slot`.
    #[must_use]
    pub fn slot(&self, slot: &str) -> Option<&SlotConfig> {
        self.slots.iter().find(|s| s.slot == slot)
    }

    /// The arm name of a harness in a slot.
    #[must_use]
    pub fn arm_name(harness: &str, slot: &str) -> String {
        format!("{harness}.{slot}")
    }
}

fn unique<'a>(what: &str, names: impl Iterator<Item = &'a str>) -> Result<(), ConfigError> {
    let mut seen = BTreeSet::new();
    for n in names {
        // Arm names (`<harness>.<slot>`) become directory names: plain names only, and no `.` inside a harness name
        // or a slot name, so an arm name splits back into the two.
        if n.trim().is_empty() || n.contains(['/', '\\', '\0', ':', '.']) {
            return Err(ConfigError::Invalid(format!(
                "{what} name `{n}` is not a plain name (no path separators, `:` or `.`)"
            )));
        }
        if !seen.insert(n) {
            return Err(ConfigError::Invalid(format!("{what} `{n}` is named twice")));
        }
    }
    Ok(())
}

/// Spend per slot, in micro-dollars, of `records` priced by `prices`.
///
/// # Errors
/// [`MetricsError::NoPrice`] for a record whose model has no price.
pub fn spent_by_slot(
    records: &[RunRecord],
    prices: &PriceTable,
) -> Result<BTreeMap<String, u64>, MetricsError> {
    let mut spent = BTreeMap::new();
    for r in records {
        let c = prices.cost_micro(r)?;
        let s: &mut u64 = spent.entry(r.slot.clone()).or_default();
        *s = s.saturating_add(c);
    }
    Ok(spent)
}

/// A budgeted run's records and the slots it stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CappedRun {
    /// Every run made, in plan order.
    pub records: Vec<RunRecord>,
    /// Slots whose cap was reached: their remaining groups were not run, so their arms are incomplete. Pass them on to
    /// the report ([`CappedRun::stopped_slots`]) and the kill line ([`CappedRun::slot_stopped`]).
    pub stopped: BTreeSet<String>,
}

impl CappedRun {
    /// Whether `slot` stopped at its cap: the value for [`crate::killline::ArmPair::stopped`] of the arm pair on
    /// that slot.
    #[must_use]
    pub fn slot_stopped(&self, slot: &str) -> bool {
        self.stopped.contains(slot)
    }

    /// The stopped slots, sorted: the value for [`crate::report::Report::stopped`].
    #[must_use]
    pub fn stopped_slots(&self) -> Vec<String> {
        self.stopped.iter().cloned().collect()
    }
}

/// What a budgeted pass has on disk and where its new records go ([`run_within_caps_journaled`]).
pub struct Journal<'a> {
    /// The records of this pass already made, by a process that stopped before the end
    /// ([`crate::runner::read_records`]); empty for a new pass. Their runs are not made again.
    pub done: &'a [RunRecord],
    /// Called with each new record before the next run starts ([`crate::runner::save_record`]).
    pub save: &'a mut dyn FnMut(&RunRecord) -> Result<(), RunnerError>,
}

/// A budgeted run refused before it started, or stopped because a record could not be saved.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CappedRunError {
    /// The runner refused the plan.
    #[error(transparent)]
    Runner(#[from] RunnerError),
    /// An earlier record could not be priced.
    #[error(transparent)]
    Metrics(#[from] MetricsError),
    /// An arm does not match the configuration.
    #[error("arm `{arm}`: {why}")]
    Arm {
        /// The arm's name.
        arm: String,
        /// What does not match.
        why: String,
    },
    /// A record handed in as done is not a run of this pass.
    #[error("done record (arm `{arm}`, brief `{brief}`, run {run}): {why}")]
    Done {
        /// The record's arm.
        arm: String,
        /// The record's brief.
        brief: String,
        /// The record's repetition.
        run: u32,
        /// What does not match.
        why: &'static str,
    },
    /// A new record could not be saved, so no further run was made. The run it was for was made (and billed) and is
    /// not on disk.
    #[error("a record could not be saved, so no further run was made: {0}")]
    Save(#[source] RunnerError),
}

/// Run `briefs` on `arms` as [`crate::runner::run_briefs`] plans them, one run at a time, skipping every later group
/// of a slot (its runs of one brief and repetition) once that slot's spend (`earlier` records included: an earlier
/// pass of the same measurement, the smoke run) reaches its cap. The cap is checked before a group's first run, and a
/// group that starts runs whole.
///
/// # Errors
/// [`CappedRunError`] if an arm's name is not `<harness>.<slot>` for a configured harness, its slot is not in
/// `config`, its model is not the slot's model, an earlier record cannot be priced, or the runner refuses the plan
/// ([`check_plan`]).
pub fn run_within_caps(
    config: &RunConfig,
    briefs: &[Brief],
    arms: &[Arm<'_>],
    acceptance: &dyn Acceptance,
    purpose: Purpose,
    earlier: &[RunRecord],
) -> Result<CappedRun, CappedRunError> {
    let journal = Journal {
        done: &[],
        save: &mut |_| Ok(()),
    };
    run_within_caps_journaled(config, briefs, arms, acceptance, purpose, earlier, journal)
}

/// [`run_within_caps`] for a pass that must survive a stop, such as one that spends money: each record is handed to
/// [`Journal::save`] as its run ends, and a pass started again with the saved records as [`Journal::done`] makes only
/// the missing runs. A done run counts towards its slot's cap, a group with a done run is finished whatever the spend,
/// and the result is that of a pass that never stopped. `earlier` holds other passes only: a record in both would be
/// counted twice. A run cut off before its record was saved is made again and its first spend is in no record, which
/// the provider-side limits on the keys bound.
///
/// # Errors
/// As [`run_within_caps`]; [`CappedRunError::Done`] (before any run) if a done record is not a run of this pass (its
/// purpose, arm, slot, model, brief or repetition) or is there twice; [`CappedRunError::Save`] at the first record
/// that cannot be saved.
pub fn run_within_caps_journaled(
    config: &RunConfig,
    briefs: &[Brief],
    arms: &[Arm<'_>],
    acceptance: &dyn Acceptance,
    purpose: Purpose,
    earlier: &[RunRecord],
    journal: Journal<'_>,
) -> Result<CappedRun, CappedRunError> {
    for a in arms {
        if !config
            .harnesses
            .iter()
            .any(|h| a.name == RunConfig::arm_name(h, &a.slot))
        {
            return Err(CappedRunError::Arm {
                arm: a.name.clone(),
                why: format!(
                    "not `<harness>.{}` for a configured harness ({})",
                    a.slot,
                    config.harnesses.join(", ")
                ),
            });
        }
        let Some(s) = config.slot(&a.slot) else {
            return Err(CappedRunError::Arm {
                arm: a.name.clone(),
                why: format!("slot `{}` is not configured", a.slot),
            });
        };
        if s.model != a.model {
            return Err(CappedRunError::Arm {
                arm: a.name.clone(),
                why: format!(
                    "model `{}`, but slot `{}` is configured for `{}`",
                    a.model, a.slot, s.model
                ),
            });
        }
    }
    let prices = config.price_table();
    let mut spent = spent_by_slot(earlier, &prices)?;
    // The runner's own checks and plan order, then the runs one by one.
    check_plan(briefs, arms, config.runs, purpose)?;
    let done = done_runs(journal.done, briefs, arms, config.runs, purpose)?;
    for (slot, cost) in spent_by_slot(journal.done, &prices)? {
        let s = spent.entry(slot).or_default();
        *s = s.saturating_add(cost);
    }
    let mut out = CappedRun {
        records: Vec::new(),
        stopped: BTreeSet::new(),
    };
    let order = plan(briefs, arms.len(), config.runs);
    let key = |p: &Planned| {
        (
            arms[p.arm].name.as_str(),
            briefs[p.brief].id.as_str(),
            p.run,
        )
    };
    // Whether each (repetition, brief, slot) group runs, decided at its first new run in plan order; a group with a
    // done run has started, so it runs.
    let mut groups: BTreeMap<(u32, usize, &str), bool> = order
        .iter()
        .filter(|p| done.contains_key(&key(p)))
        .map(|p| ((p.run, p.brief, arms[p.arm].slot.as_str()), true))
        .collect();
    for p in &order {
        if let Some(&r) = done.get(&key(p)) {
            out.records.push(r.clone());
            continue;
        }
        let arm = &arms[p.arm];
        let so_far = spent.get(&arm.slot).copied().unwrap_or(0);
        let runs = *groups
            .entry((p.run, p.brief, &arm.slot))
            .or_insert_with(|| {
                // Every arm's slot was checked to be configured above, so the `0` (stop at once) is never used.
                let cap = config
                    .slot(&arm.slot)
                    .map_or(0, |s| s.cap_usd.saturating_mul(1_000_000));
                so_far < cap
            });
        if !runs {
            out.stopped.insert(arm.slot.clone());
            continue;
        }
        let rec = run_one(&briefs[p.brief], arm, acceptance, p.run, purpose);
        (journal.save)(&rec).map_err(CappedRunError::Save)?;
        let c = prices.cost_micro(&rec)?;
        spent.insert(arm.slot.clone(), so_far.saturating_add(c));
        out.records.push(rec);
    }
    Ok(out)
}

/// The `done` records by (arm, brief, repetition), each checked to be a run of this pass and there once.
pub(crate) fn done_runs<'r>(
    done: &'r [RunRecord],
    briefs: &[Brief],
    arms: &[Arm<'_>],
    runs: u32,
    purpose: Purpose,
) -> Result<BTreeMap<(&'r str, &'r str, u32), &'r RunRecord>, CappedRunError> {
    let mut out = BTreeMap::new();
    for r in done {
        let refuse = |why| CappedRunError::Done {
            arm: r.arm.clone(),
            brief: r.brief.clone(),
            run: r.run,
            why,
        };
        if r.purpose != purpose {
            return Err(refuse("made for another purpose"));
        }
        let Some(arm) = arms.iter().find(|a| a.name == r.arm) else {
            return Err(refuse("no such arm in this pass"));
        };
        if r.slot != arm.slot || r.model != arm.model {
            return Err(refuse("made on another slot or model than the arm's"));
        }
        if !briefs.iter().any(|b| b.id == r.brief) {
            return Err(refuse("no such brief in this pass"));
        }
        if r.run >= runs {
            return Err(refuse("a repetition this pass does not make"));
        }
        if out
            .insert((r.arm.as_str(), r.brief.as_str(), r.run), r)
            .is_some()
        {
            return Err(refuse("there twice"));
        }
    }
    Ok(out)
}
