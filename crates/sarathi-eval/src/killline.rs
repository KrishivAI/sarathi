// SPDX-License-Identifier: Apache-2.0
//! The kill line (the kill-line rule) as one pure function. The tolerance is an explicit input, so a decision is made on a rule
//! stated before the run and is never argued after it. The tolerance is chosen before the first run and recorded with the report.

use serde::{Deserialize, Serialize};

use crate::metrics::{ArmStats, Purpose};

/// The tolerances the rule uses.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Tolerance {
    /// How far below the old accept rate the new one may fall (absolute, 0..1). The default is the old
    /// harness's own spread: [`Tolerance::from_old_spread`].
    pub accept: f64,
    /// How far above the old cost per accepted run the new one may rise, as a fraction (0.05 is 5%).
    pub cost_fraction: f64,
}

impl Tolerance {
    /// The default accept tolerance (the old arm's spread) with the given cost fraction.
    #[must_use]
    pub fn from_old_spread(old: &ArmStats, cost_fraction: f64) -> Self {
        Self {
            accept: old.spread,
            cost_fraction,
        }
    }

    /// Both tolerances from the old arm's own run-to-run spread: accept rate ([`ArmStats::spread`]) and cost per
    /// accepted run ([`ArmStats::cost_spread`]), so neither number is picked by hand.
    #[must_use]
    pub fn from_old_spreads(old: &ArmStats) -> Self {
        Self {
            accept: old.spread,
            cost_fraction: old.cost_spread,
        }
    }
}

/// One model arm measured under both harnesses.
#[derive(Debug, Clone, Copy)]
pub struct ArmPair<'a> {
    /// Model arm name (e.g. `strong`).
    pub name: &'a str,
    /// The strong-model arm: both conditions gate here (the kill-line rule).
    pub strong: bool,
    /// The existing harness.
    pub old: &'a ArmStats,
    /// The new core.
    pub new: &'a ArmStats,
    /// This arm's tolerance: the old harness's own spread differs per arm, so one tolerance for all arms would
    /// misjudge the arms whose spread is larger or smaller.
    pub tolerance: Tolerance,
    /// The slot stopped at its spend cap ([`crate::config::CappedRun::stopped`]), so both arms are incomplete. A
    /// stopped strong arm fails the rule (it cannot decide on part of the measurement); on another arm it is a note.
    pub stopped: bool,
}

/// How one arm fared.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArmVerdict {
    /// Model arm name.
    pub name: String,
    /// Strong-model arm.
    pub strong: bool,
    /// The tolerance this arm was judged by.
    pub tolerance: Tolerance,
    /// Accept rate within tolerance of the old one (or above it).
    pub accept_ok: bool,
    /// Cost per accepted run within tolerance of the old one (or below it).
    pub cost_ok: bool,
}

/// The decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Outcome {
    /// Both conditions hold on the strong arm and the other arms do not both miss on accept rate.
    Pass,
    /// Stop and keep the old harness.
    Fail,
}

/// The decision and the reasons for it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KillLine {
    /// Pass or fail.
    pub outcome: Outcome,
    /// Per arm.
    pub arms: Vec<ArmVerdict>,
    /// Why it failed, or findings on a pass (a miss on a non-strong arm).
    pub notes: Vec<String>,
}

fn accept_ok(p: &ArmPair<'_>) -> bool {
    let t = &p.tolerance;
    p.new.accept_rate + t.accept >= p.old.accept_rate - 1e-12
}

fn cost_ok(p: &ArmPair<'_>) -> bool {
    let t = &p.tolerance;
    match (p.old.cost_per_accepted_micro, p.new.cost_per_accepted_micro) {
        // Nothing accepted by the new arm while the old one accepted: not "equal on cost".
        (Some(_), None) => false,
        (None, _) => true,
        (Some(o), Some(n)) => n as f64 <= o as f64 * (1.0 + t.cost_fraction) * (1.0 + 1e-9),
    }
}

/// Evaluate the rule. Strong arm: accept rate and cost must both hold. Other arms: a miss is a note, unless the accept
/// rate misses on every non-strong arm (and there are at least two), which fails. No strong arm is a failure: the rule
/// cannot be applied.
#[must_use]
pub fn evaluate(pairs: &[ArmPair<'_>]) -> KillLine {
    let arms: Vec<ArmVerdict> = pairs
        .iter()
        .map(|p| ArmVerdict {
            name: p.name.to_owned(),
            strong: p.strong,
            tolerance: p.tolerance,
            accept_ok: accept_ok(p),
            cost_ok: cost_ok(p),
        })
        .collect();
    let mut notes = Vec::new();
    let mut fail = false;
    for p in pairs {
        for (which, s) in [("old", p.old), ("new", p.new)] {
            if s.purpose != Some(Purpose::Measure) {
                fail = true;
                notes.push(format!(
                    "arm `{}` ({which}): the stats are not from a measurement pass; the kill line decides on the measurement only",
                    p.name
                ));
            }
        }
    }
    for p in pairs.iter().filter(|p| p.stopped) {
        if p.strong {
            fail = true;
            notes.push(format!(
                "strong arm `{}`: stopped at its spend cap, so the measurement is incomplete and cannot decide",
                p.name
            ));
        } else {
            notes.push(format!(
                "finding, arm `{}`: stopped at its spend cap; judged on the runs made before the stop",
                p.name
            ));
        }
    }
    if !arms.iter().any(|a| a.strong) {
        fail = true;
        notes.push("no strong-model arm: the rule cannot be applied".to_owned());
    }
    for a in &arms {
        if a.strong {
            if !a.accept_ok {
                fail = true;
                notes.push(format!(
                    "strong arm `{}`: accept rate is lower than the old harness's beyond tolerance",
                    a.name
                ));
            }
            if !a.cost_ok {
                fail = true;
                notes.push(format!(
                    "strong arm `{}`: cost per accepted run is higher beyond tolerance",
                    a.name
                ));
            }
        } else {
            if !a.accept_ok {
                notes.push(format!(
                    "finding, arm `{}`: accept rate is lower beyond tolerance",
                    a.name
                ));
            }
            if !a.cost_ok {
                notes.push(format!(
                    "finding, arm `{}`: cost per accepted run is higher beyond tolerance",
                    a.name
                ));
            }
        }
    }
    let others: Vec<&ArmVerdict> = arms.iter().filter(|a| !a.strong).collect();
    if others.len() >= 2 && others.iter().all(|a| !a.accept_ok) {
        fail = true;
        notes.push("accept rate is lower beyond tolerance on every non-strong arm".to_owned());
    }
    KillLine {
        outcome: if fail { Outcome::Fail } else { Outcome::Pass },
        arms,
        notes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arm(rate: f64, cost: Option<u64>) -> ArmStats {
        ArmStats {
            arm: "x".into(),
            purpose: Some(Purpose::Measure),
            models: vec![],
            runs: 60,
            accepted: 0,
            accept_rate: rate,
            spread: 0.1,
            pass_all_rate: 0.0,
            valid_ops_rate: 1.0,
            cost_micro: 0,
            cost_per_accepted_micro: cost,
            cost_spread: 0.0,
            steps_mean: 0.0,
            wall_p50_ms: 0,
            wall_p95_ms: 0,
            cache_read_share: 0.0,
            briefs: vec![],
        }
    }

    const T: Tolerance = Tolerance {
        accept: 0.1,
        cost_fraction: 0.05,
    };

    fn pair<'a>(name: &'a str, strong: bool, old: &'a ArmStats, new: &'a ArmStats) -> ArmPair<'a> {
        pair_t(name, strong, old, new, T)
    }

    fn pair_t<'a>(
        name: &'a str,
        strong: bool,
        old: &'a ArmStats,
        new: &'a ArmStats,
        tolerance: Tolerance,
    ) -> ArmPair<'a> {
        ArmPair {
            name,
            strong,
            old,
            new,
            tolerance,
            stopped: false,
        }
    }

    #[test]
    fn higher_accept_and_lower_cost_passes() {
        let (o, n) = (arm(0.7, Some(100)), arm(0.9, Some(80)));
        assert_eq!(
            evaluate(&[pair("strong", true, &o, &n)]).outcome,
            Outcome::Pass
        );
    }

    #[test]
    fn accept_drop_within_spread_passes_and_beyond_fails() {
        let o = arm(0.8, Some(100));
        let (within, beyond) = (arm(0.71, Some(100)), arm(0.69, Some(100)));
        assert_eq!(
            evaluate(&[pair("s", true, &o, &within)]).outcome,
            Outcome::Pass
        );
        assert_eq!(
            evaluate(&[pair("s", true, &o, &beyond)]).outcome,
            Outcome::Fail
        );
    }

    #[test]
    fn cost_over_tolerance_fails_on_strong_only() {
        let o = arm(0.8, Some(100));
        let n = arm(0.8, Some(106));
        assert_eq!(evaluate(&[pair("s", true, &o, &n)]).outcome, Outcome::Fail);
        let kl = evaluate(&[pair("s", true, &o, &o), pair("cheap", false, &o, &n)]);
        assert_eq!(kl.outcome, Outcome::Pass);
        assert!(
            kl.notes
                .iter()
                .any(|m| m.starts_with("finding, arm `cheap`"))
        );
    }

    #[test]
    fn accept_miss_on_both_other_arms_fails() {
        let o = arm(0.8, Some(100));
        let bad = arm(0.5, Some(100));
        let kl = evaluate(&[
            pair("s", true, &o, &o),
            pair("a", false, &o, &bad),
            pair("b", false, &o, &bad),
        ]);
        assert_eq!(kl.outcome, Outcome::Fail);
        let one = evaluate(&[
            pair("s", true, &o, &o),
            pair("a", false, &o, &bad),
            pair("b", false, &o, &o),
        ]);
        assert_eq!(one.outcome, Outcome::Pass);
    }

    #[test]
    fn zero_accepted_cases() {
        let none = arm(0.0, None);
        let some = arm(0.5, Some(100));
        let t = Tolerance {
            accept: 1.0,
            cost_fraction: 0.0,
        };
        assert_eq!(
            evaluate(&[pair_t("s", true, &some, &none, t)]).outcome,
            Outcome::Fail
        );
        assert_eq!(
            evaluate(&[pair_t("s", true, &none, &some, t)]).outcome,
            Outcome::Pass
        );
        assert_eq!(
            evaluate(&[pair_t("s", true, &none, &none, t)]).outcome,
            Outcome::Pass
        );
    }

    #[test]
    fn no_strong_arm_fails() {
        let o = arm(0.8, Some(100));
        assert_eq!(evaluate(&[pair("a", false, &o, &o)]).outcome, Outcome::Fail);
    }

    #[test]
    fn tolerance_defaults_to_the_old_spread() {
        let o = arm(0.8, Some(100));
        assert_eq!(
            Tolerance::from_old_spread(&o, 0.05),
            Tolerance {
                accept: 0.1,
                cost_fraction: 0.05
            }
        );
    }

    #[test]
    fn each_arm_is_judged_by_its_own_spread() {
        // The strong arm's old spread is small; the other arms' are large. A 0.10 drop on both other arms is within
        // their own spread, so it is not a miss and the outcome stays Pass.
        let o = arm(0.8, Some(100));
        let drop = arm(0.7, Some(100));
        let tight = Tolerance {
            accept: 0.05,
            cost_fraction: 0.05,
        };
        let wide = Tolerance {
            accept: 0.30,
            cost_fraction: 0.05,
        };
        let kl = evaluate(&[
            pair_t("s", true, &o, &o, tight),
            pair_t("a", false, &o, &drop, wide),
            pair_t("b", false, &o, &drop, wide),
        ]);
        assert_eq!(kl.outcome, Outcome::Pass, "{:?}", kl.notes);
        let strict = evaluate(&[
            pair_t("s", true, &o, &o, tight),
            pair_t("a", false, &o, &drop, tight),
            pair_t("b", false, &o, &drop, tight),
        ]);
        assert_eq!(strict.outcome, Outcome::Fail);
    }

    #[test]
    fn cost_exactly_at_the_tolerance_passes() {
        let o = arm(0.8, Some(100));
        let n = arm(0.8, Some(105));
        assert_eq!(evaluate(&[pair("s", true, &o, &n)]).outcome, Outcome::Pass);
    }

    #[test]
    fn a_stopped_strong_arm_fails_and_a_stopped_other_arm_is_a_finding() {
        let o = arm(0.8, Some(100));
        let mut strong = pair("s", true, &o, &o);
        strong.stopped = true;
        let kl = evaluate(&[strong]);
        assert_eq!(kl.outcome, Outcome::Fail);
        assert!(
            kl.notes
                .iter()
                .any(|n| n.contains("stopped at its spend cap"))
        );
        let mut other = pair("cheap", false, &o, &o);
        other.stopped = true;
        let kl = evaluate(&[pair("s", true, &o, &o), other]);
        assert_eq!(kl.outcome, Outcome::Pass);
        assert!(
            kl.notes
                .iter()
                .any(|n| n.starts_with("finding, arm `cheap`: stopped"))
        );
    }

    #[test]
    fn stats_from_a_tuning_pass_cannot_decide_the_kill_line() {
        let o = arm(0.9, Some(100));
        let mut n = arm(0.9, Some(100));
        n.purpose = Some(Purpose::Tune);
        let kl = evaluate(&[pair("s", true, &o, &n)]);
        assert_eq!(kl.outcome, Outcome::Fail);
        assert!(
            kl.notes
                .iter()
                .any(|x| x.contains("not from a measurement pass"))
        );
    }
}
