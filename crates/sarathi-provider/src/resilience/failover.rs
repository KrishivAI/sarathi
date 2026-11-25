// Derived from Vyasa `crates/ai/src/failover.rs` @ 2403be1 (MIT OR Apache-2.0); see PROVENANCE.md
// SPDX-License-Identifier: Apache-2.0
//! An ordered chain of slots (a client and a model, each with its breaker): retried per slot, then failed over.

use std::fmt;
use std::future::Future;
use std::time::Duration;

use crate::http::{Client, Error, ErrorKind};
use crate::resilience::circuit::CircuitBreaker;
use crate::resilience::retry::RetryPolicy;

/// The breaker a [`Slot::new`] gets: open after 5 outage failures in a row, probe again after 60 s (Vyasa's values).
pub const DEFAULT_BREAKER: (u32, Duration) = (5, Duration::from_secs(60));

/// One entry of a [`Chain`]: where to call, which model to ask for, and the slot's breaker.
#[derive(Debug, Clone)]
pub struct Slot {
    /// A name for logs and [`Attempt`]s (`"anthropic/claude-opus-5-5"`, `"gateway"`).
    pub name: String,
    /// The client for the slot's target.
    pub client: Client,
    /// The model id to send to this slot.
    pub model: String,
    /// The slot's breaker; clones share it, so keep the chain (or its slots) for the life of the host.
    pub breaker: CircuitBreaker,
}

impl Slot {
    /// A slot with a new [`DEFAULT_BREAKER`].
    pub fn new(name: impl Into<String>, client: Client, model: impl Into<String>) -> Slot {
        Slot {
            name: name.into(),
            client,
            model: model.into(),
            breaker: CircuitBreaker::new(DEFAULT_BREAKER.0, DEFAULT_BREAKER.1),
        }
    }

    /// The same slot with `breaker` (shared with whoever else holds a clone of it).
    pub fn with_breaker(mut self, breaker: CircuitBreaker) -> Slot {
        self.breaker = breaker;
        self
    }
}

/// A slot's last error, as the chain left it.
#[derive(Debug, Clone, PartialEq)]
pub struct Attempt {
    /// The slot's index in the chain.
    pub slot: usize,
    /// The slot's name.
    pub name: String,
    /// The error after the slot's retries, or [`ErrorKind::CircuitOpen`] when it was skipped.
    pub error: Error,
}

/// A successful call, with what failed before it.
#[derive(Debug, Clone, PartialEq)]
pub struct Answered<T> {
    /// The answer.
    pub value: T,
    /// The index of the slot that answered: which provider and model the answer (and its cost) belongs to.
    pub slot: usize,
    /// The slots that failed or were skipped first, in order.
    pub failed: Vec<Attempt>,
}

/// Every slot failed or was skipped, or one failed in a way no other slot can mend.
#[derive(Debug, Clone, PartialEq)]
pub struct Failed {
    /// One entry per slot tried or skipped, in order; never empty.
    pub attempts: Vec<Attempt>,
}

impl Failed {
    /// The error worth showing: the most actionable of the attempts (a refused key before an outage, an outage
    /// before an open circuit). Its message is the slot's own; the others are in [`Failed::attempts`].
    pub fn error(&self) -> &Error {
        // `attempts` is never empty; `Chain::new` refuses an empty chain.
        self.attempts
            .iter()
            .min_by_key(|a| rank(a.error.kind))
            .map_or(&FALLBACK, |a| &a.error)
    }

    /// [`Failed::error`] with the number of slots tried in its message.
    pub fn into_error(self) -> Error {
        let tried = self.attempts.len();
        let mut e = self.error().clone();
        e.message = format!("{} ({tried} tried)", e.message);
        e
    }
}

static FALLBACK: Error = Error {
    kind: ErrorKind::Network,
    message: String::new(),
    status: None,
    retryable: true,
    retry_after: None,
};

impl fmt::Display for Failed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({} tried)", self.error(), self.attempts.len())
    }
}

impl std::error::Error for Failed {}

impl From<Failed> for Error {
    fn from(f: Failed) -> Error {
        f.into_error()
    }
}

/// Lower is more actionable: what a person can fix first, then what the provider did, then what the chain did.
fn rank(kind: ErrorKind) -> u8 {
    match kind {
        ErrorKind::Auth => 0,
        ErrorKind::Permission => 1,
        ErrorKind::BudgetExceeded => 2,
        ErrorKind::NotFound => 3,
        ErrorKind::InvalidRequest | ErrorKind::ContextOverflow => 4,
        ErrorKind::HistoryMismatch => 5,
        ErrorKind::Unassemblable => 6,
        ErrorKind::Refusal => 7,
        ErrorKind::RateLimited => 8,
        ErrorKind::Upstream => 9,
        ErrorKind::Timeout => 10,
        ErrorKind::Network => 11,
        ErrorKind::Malformed => 12,
        ErrorKind::CircuitOpen => 13,
        // Kinds a provider call does not produce (tool, log and run-loop failures, and kinds a newer version adds).
        _ => 14,
    }
}

/// Whether the next slot may succeed where this one failed. Not when the request itself is at fault (it is the same
/// for every slot), when the history check failed (a Sarathi bug, never hidden by trying elsewhere), or when the
/// budget is spent.
fn fails_over(kind: ErrorKind) -> bool {
    !matches!(
        kind,
        ErrorKind::InvalidRequest | ErrorKind::HistoryMismatch | ErrorKind::BudgetExceeded
    )
}

/// Slots in order of preference, and the retry policy each one gets.
///
/// For each slot whose breaker allows it, the call is retried by the policy, every attempt is recorded on the
/// breaker ([`CircuitBreaker::record`]), and a failure moves on to the next slot unless no slot can mend it (a bad
/// request, a history mismatch, a spent budget: [`Chain::run`] returns at once).
///
/// The chain does not decide whether a different model may continue a conversation: the call closure builds the
/// request for the slot it is given (its `model`), so a caller whose history holds another model's thinking blocks
/// keeps such slots out of the chain. A retry repeats the whole call: nothing of a failed turn is stored.
#[derive(Debug, Clone)]
pub struct Chain {
    slots: Vec<Slot>,
    retry: RetryPolicy,
}

impl Chain {
    /// A chain over `slots`, with [`RetryPolicy::default`].
    ///
    /// # Errors
    /// `slots` is empty ([`ErrorKind::InvalidRequest`]).
    pub fn new(slots: Vec<Slot>) -> Result<Chain, Error> {
        if slots.is_empty() {
            return Err(Error::new(
                ErrorKind::InvalidRequest,
                "a failover chain needs at least one slot",
            ));
        }
        Ok(Chain {
            slots,
            retry: RetryPolicy::default(),
        })
    }

    /// The same chain with `retry` for each slot.
    pub fn with_retry(mut self, retry: RetryPolicy) -> Chain {
        self.retry = retry;
        self
    }

    /// The slots, in order.
    pub fn slots(&self) -> &[Slot] {
        &self.slots
    }

    /// The first slot's model: what a caller puts in its request before a call is made.
    pub fn primary_model(&self) -> &str {
        self.slots.first().map_or("", |s| s.model.as_str())
    }

    /// Runs `call(slot, attempt)` through the chain (see [`Chain`]). `attempt` counts from 1 within each slot.
    ///
    /// # Errors
    /// [`Failed`] with one [`Attempt`] per slot tried or skipped.
    pub async fn run<'a, T, F, Fut>(&'a self, mut call: F) -> Result<Answered<T>, Failed>
    where
        F: FnMut(&'a Slot, u32) -> Fut,
        Fut: Future<Output = Result<T, Error>>,
    {
        let mut failed = Vec::new();
        for (i, slot) in self.slots.iter().enumerate() {
            if !slot.breaker.allows() {
                failed.push(Attempt {
                    slot: i,
                    name: slot.name.clone(),
                    error: Error::new(
                        ErrorKind::CircuitOpen,
                        format!("{}: the circuit is open", slot.name),
                    ),
                });
                continue;
            }
            let result = self
                .retry
                .run(|attempt| {
                    let fut = call(slot, attempt);
                    async move {
                        let r = fut.await;
                        slot.breaker.record(&r);
                        r
                    }
                })
                .await;
            match result {
                Ok(value) => {
                    return Ok(Answered {
                        value,
                        slot: i,
                        failed,
                    });
                }
                Err(error) => {
                    let stop = !fails_over(error.kind);
                    failed.push(Attempt {
                        slot: i,
                        name: slot.name.clone(),
                        error,
                    });
                    if stop {
                        break;
                    }
                }
            }
        }
        Err(Failed { attempts: failed })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::Target;
    use std::sync::Mutex;

    fn slot(name: &str) -> Slot {
        Slot::new(
            name,
            Client::new(Target::anthropic("k")).unwrap(),
            format!("{name}-model"),
        )
    }

    fn open(slot: &Slot) {
        for _ in 0..DEFAULT_BREAKER.0 {
            slot.breaker.record_failure();
        }
    }

    fn chain(names: &[&str]) -> Chain {
        Chain::new(names.iter().map(|n| slot(n)).collect())
            .unwrap()
            .with_retry(RetryPolicy::default().without_jitter())
    }

    /// Each slot fails with its scripted kinds in turn, then answers with its model.
    fn scripted(
        script: &[(&str, &[ErrorKind])],
    ) -> impl FnMut(&Slot, u32) -> std::future::Ready<Result<String, Error>> {
        let script: Vec<(String, Vec<ErrorKind>)> = script
            .iter()
            .map(|(n, k)| (n.to_string(), k.to_vec()))
            .collect();
        move |slot, attempt| {
            let kinds = &script.iter().find(|(n, _)| *n == slot.name).unwrap().1;
            let i = usize::try_from(attempt - 1).unwrap();
            std::future::ready(match kinds.get(i) {
                Some(k) => Err(Error::new(*k, format!("{} failed", slot.name))),
                None => Ok(slot.model.clone()),
            })
        }
    }

    #[tokio::test(start_paused = true)]
    async fn a_slot_that_keeps_failing_falls_over_to_the_next() {
        let c = chain(&["a", "b"]);
        let up = ErrorKind::Upstream;
        let out = c
            .run(scripted(&[("a", &[up, up, up]), ("b", &[])]))
            .await
            .unwrap();
        assert_eq!(out.value, "b-model");
        assert_eq!(out.slot, 1);
        assert_eq!(out.failed.len(), 1);
        assert_eq!(out.failed[0].error.kind, up);
        assert!(
            c.slots()[0].breaker.allows(),
            "3 failures do not open a threshold-5 breaker"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_retry_within_the_slot_comes_first() {
        let c = chain(&["a", "b"]);
        let out = c
            .run(scripted(&[("a", &[ErrorKind::RateLimited]), ("b", &[])]))
            .await
            .unwrap();
        assert_eq!((out.slot, out.failed.len()), (0, 0));
    }

    #[tokio::test(start_paused = true)]
    async fn an_open_circuit_is_skipped_and_said() {
        let c = chain(&["a", "b"]);
        open(&c.slots()[0]);
        let out = c.run(scripted(&[("a", &[]), ("b", &[])])).await.unwrap();
        assert_eq!(out.slot, 1);
        assert_eq!(out.failed[0].error.kind, ErrorKind::CircuitOpen);
    }

    #[tokio::test(start_paused = true)]
    async fn outages_open_the_breaker_across_runs() {
        let c = chain(&["a", "b"]).with_retry(RetryPolicy::none());
        let up: &[ErrorKind] = &[ErrorKind::Upstream; 10];
        for _ in 0..5 {
            let out = c.run(scripted(&[("a", up), ("b", &[])])).await.unwrap();
            assert_eq!(out.slot, 1);
        }
        assert!(
            !c.slots()[0].breaker.allows(),
            "5 outages in a row opened it"
        );
        let calls = Mutex::new(Vec::new());
        let out = c
            .run(|slot, _| {
                calls.lock().unwrap().push(slot.name.clone());
                std::future::ready(Ok::<_, Error>(()))
            })
            .await
            .unwrap();
        assert_eq!(out.slot, 1);
        assert_eq!(
            *calls.lock().unwrap(),
            ["b"],
            "the open slot was not called"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn no_failover_when_no_other_slot_can_mend_it() {
        for kind in [
            ErrorKind::InvalidRequest,
            ErrorKind::HistoryMismatch,
            ErrorKind::BudgetExceeded,
        ] {
            let c = chain(&["a", "b"]);
            let e = c
                .run(scripted(&[("a", &[kind]), ("b", &[])]))
                .await
                .unwrap_err();
            assert_eq!(e.attempts.len(), 1, "{kind}");
            assert_eq!(e.error().kind, kind);
        }
    }

    #[tokio::test(start_paused = true)]
    async fn everything_failed_reports_the_most_actionable_error() {
        let c = chain(&["a", "b", "c"]);
        open(&c.slots()[2]);
        let up = ErrorKind::Upstream;
        let e = c
            .run(scripted(&[
                ("a", &[up, up, up]),
                ("b", &[ErrorKind::Auth]),
                ("c", &[]),
            ]))
            .await
            .unwrap_err();
        let kinds: Vec<_> = e.attempts.iter().map(|a| a.error.kind).collect();
        assert_eq!(kinds, [up, ErrorKind::Auth, ErrorKind::CircuitOpen]);
        assert_eq!(e.error().kind, ErrorKind::Auth);
        assert_eq!(e.error().message, "b failed");
        let err: Error = e.into();
        assert_eq!(err.message, "b failed (3 tried)");
    }

    #[test]
    fn an_empty_chain_is_refused() {
        assert_eq!(
            Chain::new(vec![]).unwrap_err().kind,
            ErrorKind::InvalidRequest
        );
        assert_eq!(chain(&["a"]).primary_model(), "a-model");
    }
}
