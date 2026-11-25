// Derived from Vyasa `crates/ai/src/circuit.rs` @ 2403be1 (MIT OR Apache-2.0); see PROVENANCE.md
// SPDX-License-Identifier: Apache-2.0
//! A consecutive-failure circuit breaker, one per slot of a failover chain.

use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use tokio::time::Instant;

use crate::http::{Error, ErrorKind};

#[derive(Debug)]
struct State {
    failures: u32,
    opened_at: Option<Instant>,
}

/// Opens after `threshold` consecutive outage failures ([`CircuitBreaker::trips`]); after `cooldown` it lets calls
/// through again (half-open). A success closes it; a failure while half-open opens it for another `cooldown`.
///
/// Cloning shares the state: keep one long-lived breaker per slot and hand clones to each call. A breaker built
/// fresh per call never opens.
///
/// Time is [`tokio::time::Instant`], so a test with a paused clock controls the cooldown.
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    threshold: u32,
    cooldown: Duration,
    state: Arc<Mutex<State>>,
}

impl CircuitBreaker {
    /// A closed breaker. A `threshold` of 0 is read as 1.
    pub fn new(threshold: u32, cooldown: Duration) -> Self {
        CircuitBreaker {
            threshold: threshold.max(1),
            cooldown,
            state: Arc::new(Mutex::new(State {
                failures: 0,
                opened_at: None,
            })),
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        // The state is two plain numbers: a panic elsewhere cannot leave it half-written.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Whether a call may go ahead now: closed, or open for at least `cooldown` (a probe).
    pub fn allows(&self) -> bool {
        match self.state().opened_at {
            None => true,
            Some(at) => at.elapsed() >= self.cooldown,
        }
    }

    /// Whether the breaker is open (a probe may still be allowed once the cooldown has passed).
    pub fn is_open(&self) -> bool {
        self.state().opened_at.is_some()
    }

    /// A success: the count goes back to zero and the breaker closes.
    pub fn record_success(&self) {
        let mut st = self.state();
        st.failures = 0;
        st.opened_at = None;
    }

    /// A failure: once `threshold` failures come in a row the breaker opens. A failed probe opens it again for a
    /// full `cooldown` (the original kept the first opening time, so after one cooldown it never closed again).
    pub fn record_failure(&self) {
        let mut st = self.state();
        st.failures = st.failures.saturating_add(1);
        if st.failures >= self.threshold {
            st.opened_at = Some(Instant::now());
        }
    }

    /// Records `result` if it says something about the slot's health: a success, or an error that [`trips`]. Other
    /// errors (a bad request, a refusal of the key, a history mismatch) leave the count alone.
    ///
    /// [`trips`]: CircuitBreaker::trips
    pub fn record<T>(&self, result: &Result<T, Error>) {
        match result {
            Ok(_) => self.record_success(),
            Err(e) if Self::trips(e) => self.record_failure(),
            Err(_) => {}
        }
    }

    /// Whether `e` counts as an outage of the slot: the provider failed (5xx, an overloaded event), the connection
    /// failed or was cut, or no answer came in time. A rate limit does not: the slot is up, and failover handles it.
    pub fn trips(e: &Error) -> bool {
        matches!(
            e.kind,
            ErrorKind::Upstream | ErrorKind::Network | ErrorKind::Timeout
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(start_paused = true)]
    async fn opens_after_the_threshold_and_lets_a_probe_through_after_the_cooldown() {
        let b = CircuitBreaker::new(3, Duration::from_secs(20));
        b.record_failure();
        b.record_failure();
        assert!(b.allows(), "below the threshold it stays closed");
        b.record_failure();
        assert!(!b.allows(), "open after 3 failures in a row");
        tokio::time::advance(Duration::from_secs(20)).await;
        assert!(b.allows(), "a probe after the cooldown");
        assert!(b.is_open());
    }

    #[tokio::test(start_paused = true)]
    async fn a_failed_probe_opens_it_for_another_cooldown() {
        let b = CircuitBreaker::new(1, Duration::from_secs(10));
        b.record_failure();
        tokio::time::advance(Duration::from_secs(10)).await;
        assert!(b.allows());
        b.record_failure();
        assert!(!b.allows(), "the failed probe reopened it");
        tokio::time::advance(Duration::from_secs(10)).await;
        assert!(b.allows());
        b.record_success();
        assert!(!b.is_open());
    }

    #[test]
    fn a_success_resets_the_count() {
        let b = CircuitBreaker::new(2, Duration::from_secs(60));
        b.record_failure();
        b.record_success();
        b.record_failure();
        assert!(b.allows());
    }

    #[test]
    fn a_clone_is_the_same_breaker() {
        let a = CircuitBreaker::new(2, Duration::from_secs(60));
        let b = a.clone();
        a.record_failure();
        b.record_failure();
        assert!(!a.allows() && !b.allows());
        a.record_success();
        assert!(b.allows());
    }

    #[test]
    fn only_outages_trip_it() {
        let b = CircuitBreaker::new(1, Duration::from_secs(60));
        for kind in [
            ErrorKind::RateLimited,
            ErrorKind::InvalidRequest,
            ErrorKind::Auth,
            ErrorKind::HistoryMismatch,
            ErrorKind::Malformed,
        ] {
            b.record::<()>(&Err(Error::new(kind, "x")));
            assert!(b.allows(), "{kind} must not trip it");
        }
        b.record::<()>(&Err(Error::new(ErrorKind::Upstream, "x")));
        assert!(!b.allows());
        b.record(&Ok(()));
        assert!(b.allows());
    }

    #[test]
    fn a_zero_threshold_is_one() {
        let b = CircuitBreaker::new(0, Duration::from_secs(60));
        assert!(b.allows());
        b.record_failure();
        assert!(!b.allows());
    }
}
