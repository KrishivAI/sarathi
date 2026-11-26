// SPDX-License-Identifier: Apache-2.0
//! The run report `report.md` for people and `report.json` for diffing. Both are pure functions of the
//! stats, so the same runs give byte-identical output.

use serde::Serialize;

use crate::killline::{KillLine, Outcome};
use crate::metrics::{ArmStats, PriceTable};

/// Everything a report shows.
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    /// Run date, `YYYY-MM-DD` (an input, so the output is reproducible).
    pub date: String,
    /// Stats per arm, in the order given.
    pub arms: Vec<ArmStats>,
    /// The price table the costs were computed with (micro-dollars per million tokens, by model id).
    pub prices: PriceTable,
    /// The kill-line decision, when both harnesses were measured.
    pub kill_line: Option<KillLine>,
    /// Entries in `deviations.md`: changes made after the first run (the evaluation protocol).
    pub deviations: u32,
    /// Slots stopped at their spend cap ([`crate::config::CappedRun::stopped`]): their arms are incomplete. Left out
    /// of `report.json` when empty, so a run that never hit a cap reports as before.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stopped: Vec<String>,
}

/// `report.json`: pretty, stable key order, trailing newline.
///
/// # Errors
/// Serialisation failure (not expected for these types).
pub fn to_json(r: &Report) -> Result<String, serde_json::Error> {
    let mut s = serde_json::to_string_pretty(r)?;
    s.push('\n');
    Ok(s)
}

fn micro_to_dollars(m: u64) -> String {
    format!("${}.{:06}", m / 1_000_000, m % 1_000_000)
}

fn pct(x: f64) -> String {
    format!("{:.1}%", x * 100.0)
}

/// `report.md`: the table per arm, then the verdict and the deviation count.
#[must_use]
pub fn to_markdown(r: &Report) -> String {
    let mut o = format!("# Eval report {}\n\n", r.date);
    o.push_str("| arm | runs | accept | spread | pass^k | valid ops | cost / accepted | steps | wall p50 | wall p95 | cache read |\n");
    o.push_str("|---|---|---|---|---|---|---|---|---|---|---|\n");
    for a in &r.arms {
        let cpa = a
            .cost_per_accepted_micro
            .map_or_else(|| "n/a".to_owned(), micro_to_dollars);
        o.push_str(&format!(
            "| {} | {} | {} | {:.3} | {} | {} | {} | {:.1} | {} ms | {} ms | {} |\n",
            a.arm,
            a.runs,
            pct(a.accept_rate),
            a.spread,
            pct(a.pass_all_rate),
            pct(a.valid_ops_rate),
            cpa,
            a.steps_mean,
            a.wall_p50_ms,
            a.wall_p95_ms,
            pct(a.cache_read_share),
        ));
    }
    o.push('\n');
    for a in &r.arms {
        o.push_str(&format!("- {} models: {}\n", a.arm, a.models.join(", ")));
    }
    for (m, p) in &r.prices.models {
        o.push_str(&format!(
            "- price {m}: input {}, output {}, cached input {} (micro-dollars per million tokens)\n",
            p.input, p.output, p.cached_input
        ));
    }
    for s in &r.stopped {
        o.push_str(&format!(
            "- slot {s}: stopped at its spend cap; its arms are incomplete\n"
        ));
    }
    o.push('\n');
    match &r.kill_line {
        None => o.push_str("Kill line: not evaluated (one harness measured).\n"),
        Some(k) => {
            let word = match k.outcome {
                Outcome::Pass => "PASS",
                Outcome::Fail => "FAIL",
            };
            o.push_str(&format!("Kill line: **{word}**\n\n"));
            for a in &k.arms {
                let kind = if a.strong { "strong" } else { "other" };
                o.push_str(&format!(
                    "- {} ({kind}, tolerance: accept {:.3}, cost {:.1}% ({})): accept {}, cost {}\n",
                    a.name,
                    a.tolerance.accept,
                    a.tolerance.cost_fraction * 100.0,
                    a.tolerance.cost_source.label(),
                    if a.accept_ok { "ok" } else { "MISS" },
                    if a.cost_ok { "ok" } else { "MISS" }
                ));
            }
            for n in &k.notes {
                o.push_str(&format!("- note: {n}\n"));
            }
        }
    }
    o.push_str(&format!(
        "\nDeviations from the committed protocol: {}\n",
        r.deviations
    ));
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::killline::{ArmPair, Tolerance, evaluate};
    use crate::metrics::{PriceTable, RunRecord, TokenUsage, arm_stats};

    fn runs(arm: &str, accepted: &[bool]) -> Vec<RunRecord> {
        accepted
            .iter()
            .enumerate()
            .map(|(i, a)| RunRecord {
                version: crate::metrics::RUN_RECORD_VERSION,
                purpose: crate::metrics::Purpose::Measure,
                brief: "b".into(),
                arm: arm.into(),
                slot: String::new(),
                model: "m".into(),
                run: i as u32,
                accepted: *a,
                valid_ops: 1,
                total_ops: 1,
                usage: TokenUsage::default(),
                steps: 2,
                wall_ms: 10,
                error: None,
            })
            .collect()
    }

    fn report() -> Report {
        let mut prices = PriceTable::default();
        prices.models.insert(
            "m".into(),
            crate::metrics::Price {
                input: 1,
                output: 1,
                cached_input: 1,
                cache_write: None,
            },
        );
        let old = arm_stats(
            "old/strong",
            &runs("old/strong", &[true, true, false]),
            &prices,
        )
        .unwrap();
        let new = arm_stats(
            "new/strong",
            &runs("new/strong", &[true, true, true]),
            &prices,
        )
        .unwrap();
        let kl = evaluate(&[ArmPair {
            name: "strong",
            strong: true,
            old: &old,
            new: &new,
            tolerance: Tolerance::from_old_spread(&old, 0.05),
            stopped: false,
        }]);
        Report {
            date: "2026-11-24".into(),
            arms: vec![old, new],
            prices,
            kill_line: Some(kl),
            deviations: 0,
            stopped: vec![],
        }
    }

    #[test]
    fn same_input_gives_byte_identical_output() {
        let (a, b) = (report(), report());
        assert_eq!(to_json(&a).unwrap(), to_json(&b).unwrap());
        assert_eq!(to_markdown(&a), to_markdown(&b));
    }

    #[test]
    fn markdown_shows_the_table_verdict_and_deviations() {
        let m = to_markdown(&report());
        assert!(m.contains("| old/strong | 3 | 66.7% |"), "{m}");
        assert!(m.contains("| new/strong | 3 | 100.0% |"), "{m}");
        assert!(m.contains("Kill line: **PASS**"));
        assert!(m.contains("Deviations from the committed protocol: 0"));
        assert!(m.contains("$0.000000"), "{m}");
    }

    #[test]
    fn an_arm_with_nothing_accepted_prints_na() {
        let mut r = report();
        r.arms[0].cost_per_accepted_micro = None;
        assert!(to_markdown(&r).contains("n/a"));
    }

    #[test]
    fn a_stopped_slot_is_reported_and_kept_out_of_json_when_none() {
        let mut r = report();
        assert!(!to_json(&r).unwrap().contains("stopped"));
        r.stopped = vec!["strong".into()];
        assert!(
            to_markdown(&r)
                .contains("- slot strong: stopped at its spend cap; its arms are incomplete")
        );
        assert!(
            to_json(&r)
                .unwrap()
                .contains("\"stopped\": [\n    \"strong\"\n  ]")
        );
    }

    #[test]
    fn no_kill_line_says_so() {
        let mut r = report();
        r.kill_line = None;
        assert!(to_markdown(&r).contains("not evaluated"));
    }

    #[test]
    fn dollars_format() {
        assert_eq!(micro_to_dollars(10_800_000), "$10.800000");
        assert_eq!(micro_to_dollars(5), "$0.000005");
    }
}
