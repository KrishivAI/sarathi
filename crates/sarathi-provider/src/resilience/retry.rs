// SPDX-License-Identifier: Apache-2.0
//! Retries with exponential backoff, honouring `Retry-After`.
//!
//! What may be retried is the error's own word ([`Error::retryable`]): a rate limit, a provider failure, a failed or
//! cut connection, a timeout. A cut stream is [`ErrorKind::Network`] on both paths: nothing of a
//! failed turn is stored, so the whole call is simply made again. Never retried: a bad request, a refused key, a
//! history mismatch (a Sarathi bug), an unknown stream delta, an open circuit, a spent budget.

use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::http::Error;
#[cfg(doc)]
use crate::http::ErrorKind;

/// How often and how long to wait. [`RetryPolicy::default`]: 3 attempts, 0.5 s doubling up to 8 s, with jitter; a
/// `Retry-After` of up to 60 s is waited for, a longer one ends the retries (the failover chain moves on).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetryPolicy {
    /// Attempts in all, the first included; 0 is read as 1.
    pub max_attempts: u32,
    /// The wait after the first failure; it doubles after each further one.
    pub base_delay: Duration,
    /// The longest backoff wait.
    pub max_delay: Duration,
    /// The longest `Retry-After` the policy waits for; past it the error is returned at once.
    pub max_retry_after: Duration,
    /// Spread each backoff wait over its upper half, so clients that failed together do not retry together.
    pub jitter: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        RetryPolicy {
            max_attempts: 3,
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(8),
            max_retry_after: Duration::from_secs(60),
            jitter: true,
        }
    }
}

impl RetryPolicy {
    /// One attempt, no retry.
    pub fn none() -> Self {
        RetryPolicy {
            max_attempts: 1,
            ..RetryPolicy::default()
        }
    }

    /// The same policy without jitter: waits are exact (tests, replays).
    pub fn without_jitter(mut self) -> Self {
        self.jitter = false;
        self
    }

    /// The wait before trying again after attempt `attempt` (counted from 1) failed with `e`, or `None` when the
    /// policy gives up: `e` is not retryable, the attempts are used up, or `Retry-After` asks for more than
    /// [`RetryPolicy::max_retry_after`]. The wait is the backoff or `Retry-After`, whichever is longer.
    pub fn delay(&self, attempt: u32, e: &Error) -> Option<Duration> {
        if !e.retryable || attempt >= self.max_attempts.max(1) {
            return None;
        }
        let doublings = attempt.saturating_sub(1).min(31);
        let backoff = self
            .base_delay
            .saturating_mul(1u32 << doublings)
            .min(self.max_delay);
        let backoff = if self.jitter {
            spread(backoff)
        } else {
            backoff
        };
        match e.retry_after {
            Some(asked) if asked > self.max_retry_after => None,
            Some(asked) => Some(asked.max(backoff)),
            None => Some(backoff),
        }
    }

    /// Calls `call(attempt)` until it succeeds or [`RetryPolicy::delay`] gives up, sleeping in between; returns the
    /// last error.
    ///
    /// # Errors
    /// The last attempt's error.
    pub async fn run<T, F, Fut>(&self, call: F) -> Result<T, Error>
    where
        F: FnMut(u32) -> Fut,
        Fut: Future<Output = Result<T, Error>>,
    {
        self.run_with(call, |_, _, _| {}).await
    }

    /// As [`RetryPolicy::run`], telling `on_retry(attempt, error, wait)` before each wait (for the host's log).
    ///
    /// # Errors
    /// The last attempt's error.
    pub async fn run_with<T, F, Fut, R>(&self, mut call: F, mut on_retry: R) -> Result<T, Error>
    where
        F: FnMut(u32) -> Fut,
        Fut: Future<Output = Result<T, Error>>,
        R: FnMut(u32, &Error, Duration),
    {
        let mut attempt = 1;
        loop {
            match call(attempt).await {
                Ok(v) => return Ok(v),
                Err(e) => match self.delay(attempt, &e) {
                    Some(wait) => {
                        on_retry(attempt, &e, wait);
                        tokio::time::sleep(wait).await;
                        attempt += 1;
                    }
                    None => return Err(e),
                },
            }
        }
    }
}

/// `d` scaled by a pseudo-random factor in [0.5, 1.0]. Not for anything but spreading retries.
fn spread(d: Duration) -> Duration {
    static STATE: AtomicU64 = AtomicU64::new(0);
    let mut seed = STATE.load(Ordering::Relaxed);
    if seed == 0 {
        seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0x9e37_79b9_7f4a_7c15, |t| {
                u64::try_from(t.as_nanos() & u128::from(u64::MAX)).unwrap_or(1)
            })
            | 1;
    }
    // splitmix64
    let next = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    STATE.store(next, Ordering::Relaxed);
    let mut z = next;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    // 1024 steps between one half and the whole.
    let step = u32::try_from(z % 1025).unwrap_or(1024);
    // `d / 2 / 1024 * step` is at most `d / 2`, so nothing overflows, even for `Duration::MAX`.
    (d / 2).saturating_add((d / 2 / 1024).saturating_mul(step))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::ErrorKind;
    use std::sync::atomic::AtomicU32;

    fn err(kind: ErrorKind) -> Error {
        Error::new(kind, "x")
    }

    fn after(kind: ErrorKind, secs: u64) -> Error {
        Error {
            retry_after: Some(Duration::from_secs(secs)),
            ..err(kind)
        }
    }

    #[test]
    fn backoff_doubles_up_to_the_cap() {
        let p = RetryPolicy {
            max_attempts: 10,
            ..RetryPolicy::default().without_jitter()
        };
        let waits: Vec<_> = (1..7)
            .map(|a| p.delay(a, &err(ErrorKind::Upstream)).unwrap())
            .collect();
        let ms = |n| Duration::from_millis(n);
        assert_eq!(
            waits,
            [ms(500), ms(1000), ms(2000), ms(4000), ms(8000), ms(8000)]
        );
    }

    #[test]
    fn gives_up_when_not_retryable_or_out_of_attempts() {
        let p = RetryPolicy::default().without_jitter();
        for kind in [
            ErrorKind::InvalidRequest,
            ErrorKind::Auth,
            ErrorKind::HistoryMismatch,
            ErrorKind::Unassemblable,
            ErrorKind::CircuitOpen,
            ErrorKind::BudgetExceeded,
            ErrorKind::Malformed,
        ] {
            assert_eq!(p.delay(1, &err(kind)), None, "{kind}");
        }
        assert!(p.delay(2, &err(ErrorKind::Network)).is_some());
        assert_eq!(p.delay(3, &err(ErrorKind::Network)), None);
        assert_eq!(RetryPolicy::none().delay(1, &err(ErrorKind::Network)), None);
        let zero = RetryPolicy {
            max_attempts: 0,
            ..RetryPolicy::default()
        };
        assert_eq!(zero.delay(1, &err(ErrorKind::Network)), None);
    }

    #[test]
    fn retry_after_is_honoured_up_to_its_cap() {
        let p = RetryPolicy::default().without_jitter();
        assert_eq!(
            p.delay(1, &after(ErrorKind::RateLimited, 7)),
            Some(Duration::from_secs(7))
        );
        // A short Retry-After never shortens the backoff.
        let short = Error {
            retry_after: Some(Duration::from_millis(1)),
            ..err(ErrorKind::RateLimited)
        };
        assert_eq!(p.delay(2, &short), Some(Duration::from_secs(1)));
        assert_eq!(p.delay(1, &after(ErrorKind::RateLimited, 61)), None);
    }

    #[test]
    fn jitter_stays_in_the_upper_half() {
        let p = RetryPolicy::default();
        for _ in 0..200 {
            let d = p.delay(2, &err(ErrorKind::Upstream)).unwrap();
            assert!(
                d >= Duration::from_millis(500) && d <= Duration::from_millis(1000),
                "{d:?}"
            );
        }
    }

    #[test]
    fn jitter_does_not_overflow_on_the_longest_delays() {
        let p = RetryPolicy {
            max_attempts: 40,
            base_delay: Duration::MAX,
            max_delay: Duration::MAX,
            ..RetryPolicy::default()
        };
        for attempt in [1, 2, 39] {
            let d = p.delay(attempt, &err(ErrorKind::Upstream)).unwrap();
            assert!(d >= Duration::MAX / 2, "{d:?}");
        }
    }

    #[tokio::test(start_paused = true)]
    async fn run_retries_then_succeeds_and_reports_each_wait() {
        let calls = AtomicU32::new(0);
        let mut seen = Vec::new();
        let start = tokio::time::Instant::now();
        let out = RetryPolicy::default()
            .without_jitter()
            .run_with(
                |attempt| {
                    calls.fetch_add(1, Ordering::Relaxed);
                    async move {
                        if attempt < 3 {
                            Err(err(ErrorKind::Upstream))
                        } else {
                            Ok(attempt)
                        }
                    }
                },
                |a, e, w| seen.push((a, e.kind, w)),
            )
            .await;
        assert_eq!(out.unwrap(), 3);
        assert_eq!(calls.load(Ordering::Relaxed), 3);
        assert_eq!(
            seen,
            [
                (1, ErrorKind::Upstream, Duration::from_millis(500)),
                (2, ErrorKind::Upstream, Duration::from_millis(1000)),
            ]
        );
        assert_eq!(start.elapsed(), Duration::from_millis(1500));
    }

    #[tokio::test(start_paused = true)]
    async fn run_returns_the_last_error() {
        let out: Result<(), _> = RetryPolicy::default()
            .without_jitter()
            .run(|attempt| async move {
                Err(if attempt == 1 {
                    err(ErrorKind::Network)
                } else {
                    err(ErrorKind::InvalidRequest)
                })
            })
            .await;
        assert_eq!(out.unwrap_err().kind, ErrorKind::InvalidRequest);
    }
}
