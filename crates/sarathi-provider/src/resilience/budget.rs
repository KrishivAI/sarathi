// Derived from Vyasa `crates/ai/src/budget.rs` @ 2403be1 (MIT OR Apache-2.0); see PROVENANCE.md
// SPDX-License-Identifier: Apache-2.0
//! Spend caps, checked before a call.
//!
//! Amounts are whole micro-dollars (`u64`), so sums and comparisons are exact (the original used `f64` dollars).
//! The caller keeps the spend and chooses the window (a run, a day, a calendar month): this module only compares.
//! Concurrent calls may each pass the check and together overrun a cap by at most one call each, as in the original.

use std::collections::BTreeMap;

use crate::http::{Error, ErrorKind};

/// One US dollar in micro-dollars.
pub const MICROS_PER_USD: u64 = 1_000_000;

/// The caps; `None` and a missing purpose are uncapped.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BudgetCaps {
    /// The cap on all spend, in micro-dollars.
    pub total_micros: Option<u64>,
    /// A cap per purpose (`"agent"`, `"compaction"`, `"judge"`, ...), in micro-dollars.
    pub per_purpose_micros: BTreeMap<String, u64>,
}

/// The verdict of [`evaluate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetDecision {
    /// Under every cap: go ahead.
    Allowed,
    /// The total cap is reached.
    TotalExceeded,
    /// The purpose's cap is reached.
    PurposeExceeded,
}

/// Compares the spend so far with the caps. A cap is reached when the spend equals it.
pub fn evaluate(
    caps: &BudgetCaps,
    total_spent_micros: u64,
    purpose: &str,
    purpose_spent_micros: u64,
) -> BudgetDecision {
    if caps
        .total_micros
        .is_some_and(|cap| total_spent_micros >= cap)
    {
        return BudgetDecision::TotalExceeded;
    }
    if caps
        .per_purpose_micros
        .get(purpose)
        .is_some_and(|cap| purpose_spent_micros >= *cap)
    {
        return BudgetDecision::PurposeExceeded;
    }
    BudgetDecision::Allowed
}

impl BudgetDecision {
    /// `Ok` when allowed, else an [`ErrorKind::BudgetExceeded`] error naming the cap and the purpose (never retried,
    /// never failed over).
    ///
    /// # Errors
    /// A cap is reached.
    pub fn check(self, purpose: &str) -> Result<(), Error> {
        let which = match self {
            BudgetDecision::Allowed => return Ok(()),
            BudgetDecision::TotalExceeded => "the total",
            BudgetDecision::PurposeExceeded => "the purpose's",
        };
        Err(Error::new(
            ErrorKind::BudgetExceeded,
            format!("{which} spend cap is reached (purpose {purpose})"),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps() -> BudgetCaps {
        BudgetCaps {
            total_micros: Some(5 * MICROS_PER_USD),
            per_purpose_micros: BTreeMap::from([("judge".to_string(), MICROS_PER_USD)]),
        }
    }

    #[test]
    fn under_every_cap_is_allowed() {
        assert_eq!(
            evaluate(&caps(), 4_999_999, "judge", 999_999),
            BudgetDecision::Allowed
        );
        assert_eq!(
            evaluate(&caps(), 0, "agent", u64::MAX),
            BudgetDecision::Allowed
        );
        assert_eq!(
            evaluate(&BudgetCaps::default(), u64::MAX, "x", u64::MAX),
            BudgetDecision::Allowed
        );
    }

    #[test]
    fn reaching_a_cap_stops_the_call_total_first() {
        assert_eq!(
            evaluate(&caps(), 5 * MICROS_PER_USD, "judge", MICROS_PER_USD),
            BudgetDecision::TotalExceeded
        );
        assert_eq!(
            evaluate(&caps(), 0, "judge", MICROS_PER_USD),
            BudgetDecision::PurposeExceeded
        );
    }

    #[test]
    fn check_gives_a_budget_error() {
        assert!(BudgetDecision::Allowed.check("agent").is_ok());
        let e = BudgetDecision::PurposeExceeded.check("judge").unwrap_err();
        assert_eq!(e.kind, ErrorKind::BudgetExceeded);
        assert!(!e.retryable);
        assert!(e.message.contains("judge"), "{e}");
        let e = BudgetDecision::TotalExceeded.check("agent").unwrap_err();
        assert!(e.message.contains("total"), "{e}");
    }
}
