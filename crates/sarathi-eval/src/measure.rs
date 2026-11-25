// SPDX-License-Identifier: Apache-2.0
//! From a budgeted run ([`CappedRun`]) to its [`Report`]: the stats of every arm, the kill line over the arm pairs
//! (one per slot, the old harness against the new one), and the slots stopped at their cap.
//!
//! The judging rule ([`Judging`]: which harness is old and which new, the strong slot, the cost tolerance) is an input
//! stated before the run; the accept tolerance of each pair is the old arm's own spread
//! ([`Tolerance::from_old_spread`]), the cost tolerance either the old arm's own cost spread or a fixed fraction
//! ([`CostTolerance`]). A slot that stopped at its cap reaches the kill line as
//! [`ArmPair::stopped`] and the report as [`Report::stopped`]; a slot with no run on either harness (stopped before
//! its first group) is left out of the kill line with a note, since there is nothing to judge.

use crate::config::{CappedRun, RunConfig};
use crate::killline::{ArmPair, Tolerance, evaluate};
use crate::metrics::{ArmStats, MetricsError, arm_stats};
use crate::report::Report;

/// How the kill line judges a run: chosen before the run, recorded in the report through each arm's tolerance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Judging<'a> {
    /// The existing harness (a name in [`RunConfig::harnesses`]).
    pub old: &'a str,
    /// The new core (a name in [`RunConfig::harnesses`]).
    pub new: &'a str,
    /// The slot the kill line is decided on (a name in [`RunConfig::slots`]).
    pub strong_slot: &'a str,
    /// How far above the old cost per accepted run the new one may rise.
    pub cost: CostTolerance,
}

/// The cost side of the kill line's tolerance, chosen before the run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CostTolerance {
    /// The old arm's own run-to-run cost spread ([`ArmStats::cost_spread`]), per arm: the same rule as the accept
    /// tolerance ([`Tolerance::from_old_spreads`]).
    OldSpread,
    /// A fixed fraction for every arm (0.05 is 5%; 0 is strict: not higher at all).
    Fraction(f64),
}

/// A report that cannot be built.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum MeasureError {
    /// A record could not be priced, or an arm mixes purposes.
    #[error(transparent)]
    Metrics(#[from] MetricsError),
    /// The judging rule does not fit the configuration.
    #[error("judging: {0}")]
    Judging(String),
}

/// The report of `run`, made under `config`: one stats entry per arm (slots in configuration order, harnesses in
/// configuration order within a slot), the stopped slots, and, with `judging`, the kill line.
///
/// Without `judging` the kill line is left out (one harness measured, or a run that is not judged, such as the smoke
/// run).
///
/// # Errors
/// [`MeasureError::Metrics`] if a record cannot be priced or an arm mixes purposes; [`MeasureError::Judging`] if the
/// old or new harness or the strong slot is not configured, old and new are the same, or the cost tolerance is not a
/// finite fraction of at least 0.
pub fn report(
    config: &RunConfig,
    run: &CappedRun,
    judging: Option<&Judging<'_>>,
    date: &str,
    deviations: u32,
) -> Result<Report, MeasureError> {
    if let Some(j) = judging {
        check(config, j)?;
    }
    let prices = config.price_table();
    let mut arms = Vec::with_capacity(config.slots.len() * config.harnesses.len());
    for s in &config.slots {
        for h in &config.harnesses {
            arms.push(arm_stats(
                &RunConfig::arm_name(h, &s.slot),
                &run.records,
                &prices,
            )?);
        }
    }
    let kill_line = judging.map(|j| {
        let find = |harness: &str, slot: &str| -> Option<&ArmStats> {
            let name = RunConfig::arm_name(harness, slot);
            arms.iter().find(|a| a.arm == name)
        };
        let mut pairs = Vec::new();
        let mut empty = Vec::new();
        for s in &config.slots {
            // `check` made sure both harnesses are configured, so both arms are in `arms`.
            let (Some(old), Some(new)) = (find(j.old, &s.slot), find(j.new, &s.slot)) else {
                continue;
            };
            if old.runs == 0 && new.runs == 0 {
                empty.push(s.slot.as_str());
                continue;
            }
            pairs.push(ArmPair {
                name: &s.slot,
                strong: s.slot == j.strong_slot,
                old,
                new,
                tolerance: match j.cost {
                    CostTolerance::OldSpread => Tolerance::from_old_spreads(old),
                    CostTolerance::Fraction(f) => Tolerance::from_old_spread(old, f),
                },
                stopped: run.slot_stopped(&s.slot),
            });
        }
        let mut k = evaluate(&pairs);
        for slot in empty {
            let why = if run.slot_stopped(slot) {
                "stopped at its spend cap before its first run"
            } else {
                "no runs"
            };
            k.notes
                .push(format!("arm `{slot}`: {why}; left out of the kill line"));
        }
        k
    });
    Ok(Report {
        date: date.to_owned(),
        arms,
        prices,
        kill_line,
        deviations,
        stopped: run.stopped_slots(),
    })
}

fn check(config: &RunConfig, j: &Judging<'_>) -> Result<(), MeasureError> {
    let bad = |m: String| Err(MeasureError::Judging(m));
    for (what, h) in [("old", j.old), ("new", j.new)] {
        if !config.harnesses.iter().any(|c| c == h) {
            return bad(format!(
                "{what} harness `{h}` is not configured ({})",
                config.harnesses.join(", ")
            ));
        }
    }
    if j.old == j.new {
        return bad(format!("old and new are the same harness `{}`", j.old));
    }
    if config.slot(j.strong_slot).is_none() {
        return bad(format!("strong slot `{}` is not configured", j.strong_slot));
    }
    if let CostTolerance::Fraction(f) = j.cost
        && (!f.is_finite() || f < 0.0)
    {
        return bad(format!(
            "cost tolerance {f} is not a fraction of at least 0"
        ));
    }
    Ok(())
}
