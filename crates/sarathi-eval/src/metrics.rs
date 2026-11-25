// SPDX-License-Identifier: Apache-2.0
//! Per-run records and the standard run metrics: per brief, then per arm. Pure functions over records; time and prices
//! arrive as data, so the results are deterministic.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Token counts of one run, as the provider reported them. `input_tokens` excludes the cached part.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Uncached input tokens.
    pub input_tokens: u64,
    /// Output tokens.
    pub output_tokens: u64,
    /// Input tokens read from the prompt cache.
    pub cached_input_tokens: u64,
    /// Input tokens written to the prompt cache (billed above the uncached rate by providers that charge for writes).
    /// Absent in records written before it was counted: read as 0.
    #[serde(default)]
    pub cache_write_input_tokens: u64,
}

/// The run-record format version this crate writes. It reads this version and the one before.
pub const RUN_RECORD_VERSION: u32 = 1;

/// Why briefs are being run. A tuning pass never touches the measurement split.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Purpose {
    /// Developing against the tuning split.
    Tune,
    /// The measurement that decides the kill line.
    Measure,
}

impl Purpose {
    /// The directory name under the run date.
    #[must_use]
    pub fn dir(self) -> &'static str {
        match self {
            Self::Tune => "tune",
            Self::Measure => "measure",
        }
    }
}

/// What one run of one brief by one arm produced.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunRecord {
    /// Record format version, [`RUN_RECORD_VERSION`].
    pub version: u32,
    /// Whether this run was a tuning pass or the measurement.
    pub purpose: Purpose,
    /// Brief id.
    pub brief: String,
    /// Arm name (harness and model, e.g. `new/strong`).
    pub arm: String,
    /// The slot the arm filled (the name briefs list in `allowed_models`); empty in records written before slots.
    #[serde(default)]
    pub slot: String,
    /// Model id, the key into the [`PriceTable`].
    pub model: String,
    /// Run number within the brief and arm, from 0.
    pub run: u32,
    /// The acceptance check passed. A run that errored is `false`, never dropped.
    pub accepted: bool,
    /// Proposed operations that validated.
    pub valid_ops: u32,
    /// Operations proposed.
    pub total_ops: u32,
    /// Tokens used.
    pub usage: TokenUsage,
    /// Model calls made.
    pub steps: u32,
    /// Wall time in milliseconds.
    pub wall_ms: u64,
    /// Why the run failed before the acceptance check, if it did.
    pub error: Option<String>,
}

/// Price of one model, in micro-dollars per million tokens (integers keep the sums exact).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Price {
    /// Uncached input.
    pub input: u64,
    /// Output.
    pub output: u64,
    /// Cache read.
    pub cached_input: u64,
    /// Cache write. `None` (a table written before it was priced, or a provider that does not charge for writes)
    /// prices a written token as uncached input, so a write is never cheaper than sending the token uncached.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write: Option<u64>,
}

/// One price table for both arms (the metrics table). A model missing from it is an error, never a free run.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PriceTable {
    /// Price by model id.
    pub models: BTreeMap<String, Price>,
}

/// A run could not be priced.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MetricsError {
    /// The price table has no entry for the run's model.
    #[error("no price for model `{0}`")]
    NoPrice(String),
    /// The arm's runs mix tuning and measurement passes, or are not all one purpose.
    #[error(
        "arm `{0}` has runs from more than one purpose (tuning and measurement must not be mixed)"
    )]
    MixedPurposes(String),
}

impl PriceTable {
    /// Cost of a run in micro-dollars, rounded down.
    ///
    /// # Errors
    /// [`MetricsError::NoPrice`] when the model is not in the table.
    pub fn cost_micro(&self, run: &RunRecord) -> Result<u64, MetricsError> {
        let p = self
            .models
            .get(&run.model)
            .ok_or_else(|| MetricsError::NoPrice(run.model.clone()))?;
        let u = run.usage;
        let total = u128::from(u.input_tokens) * u128::from(p.input)
            + u128::from(u.output_tokens) * u128::from(p.output)
            + u128::from(u.cached_input_tokens) * u128::from(p.cached_input)
            + u128::from(u.cache_write_input_tokens) * u128::from(p.cache_write.unwrap_or(p.input));
        Ok(u64::try_from(total / 1_000_000).unwrap_or(u64::MAX))
    }
}

/// The metrics of one brief over its `k` runs in one arm.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BriefStats {
    /// Brief id.
    pub brief: String,
    /// Runs counted (`k`).
    pub runs: u32,
    /// Runs accepted.
    pub accepted: u32,
    /// Mean accept indicator over the runs.
    pub accept_rate: f64,
    /// Standard deviation of the accept indicator over the runs (population form; 0 for `k = 1`).
    pub accept_std: f64,
    /// Every run accepted (`pass^k`).
    pub pass_all: bool,
    /// Total cost of the runs, micro-dollars.
    pub cost_micro: u64,
}

/// The metrics of one arm over all its briefs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArmStats {
    /// Arm name.
    pub arm: String,
    /// The purpose of every run counted (`None` when the arm has no runs). The kill line refuses anything but
    /// the measurement.
    pub purpose: Option<Purpose>,
    /// Concrete model ids the runs used, sorted: the model per run is part of the result.
    pub models: Vec<String>,
    /// Runs counted.
    pub runs: u32,
    /// Runs accepted.
    pub accepted: u32,
    /// Mean over briefs of the per-brief accept rate (the kill line's primary number).
    pub accept_rate: f64,
    /// Mean over briefs of the per-brief accept standard deviation: the arm's run-to-run spread (the kill-line rule).
    pub spread: f64,
    /// Briefs with `pass^k`, over briefs.
    pub pass_all_rate: f64,
    /// Valid operations over proposed operations (1.0 when none were proposed).
    pub valid_ops_rate: f64,
    /// Total cost, micro-dollars.
    pub cost_micro: u64,
    /// Cost per accepted run, micro-dollars; `None` when nothing was accepted.
    pub cost_per_accepted_micro: Option<u64>,
    /// Mean model calls per run.
    pub steps_mean: f64,
    /// Median wall time, milliseconds.
    pub wall_p50_ms: u64,
    /// 95th percentile wall time, milliseconds.
    pub wall_p95_ms: u64,
    /// Cached input over all input tokens (0.0 when there were none). Reported, not gating.
    pub cache_read_share: f64,
    /// Per-brief numbers, ordered by brief id.
    pub briefs: Vec<BriefStats>,
}

/// Nearest-rank percentile of a sorted slice; `p` in `1..=100`. Empty input gives 0.
#[must_use]
pub fn percentile(sorted: &[u64], p: u32) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let n = sorted.len();
    let rank = (n * p as usize).div_ceil(100).clamp(1, n);
    sorted[rank - 1]
}

fn ratio(num: u64, den: u64, empty: f64) -> f64 {
    if den == 0 {
        empty
    } else {
        num as f64 / den as f64
    }
}

/// Aggregate the runs of one arm. `runs` may be in any order; briefs come out sorted by id.
///
/// # Errors
/// [`MetricsError::NoPrice`] when a run's model has no price.
pub fn arm_stats(
    arm: &str,
    runs: &[RunRecord],
    prices: &PriceTable,
) -> Result<ArmStats, MetricsError> {
    let purpose = runs.iter().find(|r| r.arm == arm).map(|r| r.purpose);
    if runs
        .iter()
        .any(|r| r.arm == arm && Some(r.purpose) != purpose)
    {
        return Err(MetricsError::MixedPurposes(arm.to_owned()));
    }
    let mut models: Vec<String> = runs
        .iter()
        .filter(|r| r.arm == arm)
        .map(|r| r.model.clone())
        .collect();
    models.sort_unstable();
    models.dedup();
    let mut by_brief: BTreeMap<&str, Vec<&RunRecord>> = BTreeMap::new();
    for r in runs.iter().filter(|r| r.arm == arm) {
        by_brief.entry(r.brief.as_str()).or_default().push(r);
    }
    let mut briefs = Vec::with_capacity(by_brief.len());
    let mut total_cost = 0u64;
    for (id, rs) in &by_brief {
        let mut cost = 0u64;
        for r in rs {
            cost = cost.saturating_add(prices.cost_micro(r)?);
        }
        total_cost = total_cost.saturating_add(cost);
        let k = u32::try_from(rs.len()).unwrap_or(u32::MAX);
        let accepted = u32::try_from(rs.iter().filter(|r| r.accepted).count()).unwrap_or(u32::MAX);
        let p = f64::from(accepted) / f64::from(k);
        briefs.push(BriefStats {
            brief: (*id).to_owned(),
            runs: k,
            accepted,
            accept_rate: p,
            accept_std: (p * (1.0 - p)).sqrt(),
            pass_all: accepted == k,
            cost_micro: cost,
        });
    }
    let own: Vec<&RunRecord> = runs.iter().filter(|r| r.arm == arm).collect();
    let n_runs = u32::try_from(own.len()).unwrap_or(u32::MAX);
    let accepted = u32::try_from(own.iter().filter(|r| r.accepted).count()).unwrap_or(u32::MAX);
    let nb = briefs.len().max(1) as f64;
    let valid: u64 = own.iter().map(|r| u64::from(r.valid_ops)).sum();
    let total_ops: u64 = own.iter().map(|r| u64::from(r.total_ops)).sum();
    let input: u64 = own.iter().map(|r| r.usage.input_tokens).sum();
    let cached: u64 = own.iter().map(|r| r.usage.cached_input_tokens).sum();
    let steps: u64 = own.iter().map(|r| u64::from(r.steps)).sum();
    let mut walls: Vec<u64> = own.iter().map(|r| r.wall_ms).collect();
    walls.sort_unstable();
    Ok(ArmStats {
        arm: arm.to_owned(),
        purpose,
        models,
        runs: n_runs,
        accepted,
        accept_rate: briefs.iter().map(|b| b.accept_rate).sum::<f64>() / nb,
        spread: briefs.iter().map(|b| b.accept_std).sum::<f64>() / nb,
        pass_all_rate: briefs.iter().filter(|b| b.pass_all).count() as f64 / nb,
        valid_ops_rate: ratio(valid, total_ops, 1.0),
        cost_micro: total_cost,
        cost_per_accepted_micro: (accepted > 0).then(|| total_cost / u64::from(accepted)),
        steps_mean: ratio(steps, u64::from(n_runs), 0.0),
        wall_p50_ms: percentile(&walls, 50),
        wall_p95_ms: percentile(&walls, 95),
        cache_read_share: ratio(cached, input + cached, 0.0),
        briefs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(brief: &str, arm: &str, n: u32, accepted: bool) -> RunRecord {
        RunRecord {
            version: RUN_RECORD_VERSION,
            purpose: Purpose::Measure,
            brief: brief.into(),
            arm: arm.into(),
            slot: String::new(),
            model: "m".into(),
            run: n,
            accepted,
            valid_ops: 3,
            total_ops: 4,
            usage: TokenUsage {
                input_tokens: 1_000_000,
                output_tokens: 500_000,
                cached_input_tokens: 1_000_000,
                cache_write_input_tokens: 0,
            },
            steps: 4,
            wall_ms: 1000 * u64::from(n + 1),
            error: None,
        }
    }

    fn prices() -> PriceTable {
        let mut models = BTreeMap::new();
        models.insert(
            "m".to_owned(),
            Price {
                input: 3_000_000,
                output: 15_000_000,
                cached_input: 300_000,
                cache_write: None,
            },
        );
        PriceTable { models }
    }

    #[test]
    fn cost_is_exact_integer_micro_dollars() {
        // 1M in at $3 + 0.5M out at $15 + 1M cached at $0.30 = $10.80.
        assert_eq!(prices().cost_micro(&run("b", "a", 0, true)), Ok(10_800_000));
    }

    #[test]
    fn cache_writes_cost_their_price_or_the_input_price() {
        let mut r = run("b", "a", 0, true);
        r.usage = TokenUsage {
            cache_write_input_tokens: 1_000_000,
            ..TokenUsage::default()
        };
        // No write price: a written token costs what an uncached one does ($3).
        assert_eq!(prices().cost_micro(&r), Ok(3_000_000));
        let mut t = prices();
        if let Some(p) = t.models.get_mut("m") {
            p.cache_write = Some(3_750_000);
        }
        assert_eq!(t.cost_micro(&r), Ok(3_750_000));
    }

    #[test]
    fn records_and_prices_without_cache_writes_still_read() {
        let u: TokenUsage =
            serde_json::from_str(r#"{"input_tokens":1,"output_tokens":2,"cached_input_tokens":3}"#)
                .unwrap();
        assert_eq!(u.cache_write_input_tokens, 0);
        let p: Price = serde_json::from_str(r#"{"input":1,"output":2,"cached_input":3}"#).unwrap();
        assert_eq!(p.cache_write, None);
    }

    #[test]
    fn unknown_model_is_an_error_not_free() {
        let mut r = run("b", "a", 0, true);
        r.model = "x".into();
        assert_eq!(
            prices().cost_micro(&r),
            Err(MetricsError::NoPrice("x".into()))
        );
    }

    #[test]
    fn mixed_runs_give_rate_spread_and_pass_all() {
        let runs = [
            run("b1", "a", 0, true),
            run("b1", "a", 1, false),
            run("b1", "a", 2, true),
            run("b2", "a", 0, true),
            run("b2", "a", 1, true),
            run("b2", "a", 2, true),
            run("b1", "other", 0, true),
        ];
        let s = arm_stats("a", &runs, &prices()).unwrap();
        assert_eq!((s.runs, s.accepted), (6, 5));
        assert!((s.accept_rate - (2.0 / 3.0 + 1.0) / 2.0).abs() < 1e-12);
        let std_b1 = (2.0f64 / 3.0 * (1.0 / 3.0)).sqrt();
        assert!((s.spread - std_b1 / 2.0).abs() < 1e-12);
        assert!((s.pass_all_rate - 0.5).abs() < 1e-12);
        assert_eq!(
            s.briefs
                .iter()
                .map(|b| b.brief.as_str())
                .collect::<Vec<_>>(),
            ["b1", "b2"]
        );
        assert_eq!(s.cost_per_accepted_micro, Some(6 * 10_800_000 / 5));
        assert!((s.valid_ops_rate - 0.75).abs() < 1e-12);
        assert!((s.cache_read_share - 0.5).abs() < 1e-12);
        assert!((s.steps_mean - 4.0).abs() < 1e-12);
    }

    #[test]
    fn zero_accepted_has_no_cost_per_accepted_and_zero_spread() {
        let runs = [run("b", "a", 0, false), run("b", "a", 1, false)];
        let s = arm_stats("a", &runs, &prices()).unwrap();
        assert_eq!(s.cost_per_accepted_micro, None);
        assert_eq!(s.spread, 0.0);
        assert!(!s.briefs[0].pass_all);
    }

    #[test]
    fn single_run_has_zero_spread() {
        let s = arm_stats("a", &[run("b", "a", 0, true)], &prices()).unwrap();
        assert_eq!(s.spread, 0.0);
        assert_eq!(s.accept_rate, 1.0);
    }

    #[test]
    fn empty_arm_is_all_zero_not_a_division_error() {
        let s = arm_stats("a", &[], &prices()).unwrap();
        assert_eq!((s.runs, s.accepted, s.wall_p95_ms), (0, 0, 0));
        assert_eq!(s.accept_rate, 0.0);
        assert_eq!(s.valid_ops_rate, 1.0);
    }

    #[test]
    fn percentile_is_nearest_rank() {
        let v = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(percentile(&v, 50), 5);
        assert_eq!(percentile(&v, 95), 10);
        assert_eq!(percentile(&[7], 95), 7);
        assert_eq!(percentile(&[], 50), 0);
    }

    #[test]
    fn stats_refuse_a_mix_of_tuning_and_measurement_runs() {
        let mut a = run("b", "x", 0, true);
        let mut b = run("b", "x", 1, true);
        a.purpose = Purpose::Tune;
        b.purpose = Purpose::Measure;
        let mut t = PriceTable::default();
        t.models.insert(
            "m".into(),
            Price {
                input: 1,
                output: 1,
                cached_input: 1,
                cache_write: None,
            },
        );
        assert_eq!(
            arm_stats("x", &[a, b], &t).err(),
            Some(MetricsError::MixedPurposes("x".into()))
        );
    }
}
