// SPDX-License-Identifier: Apache-2.0
//! The kinds of failure every layer reports: a provider call, a tool, the log, the run loop.

use std::fmt;

use serde::{Deserialize, Serialize};

/// What kind of failure an error is; decides retry, fallback and what a host shows. The names ([`ErrorKind::as_str`])
/// are stable: they appear in logs and events. A name this version does not know reads as [`ErrorKind::Unknown`], so
/// an older reader keeps reading a log a newer writer added kinds to.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    /// The key was refused (401).
    Auth,
    /// The key may not do this (403).
    Permission,
    /// The model or route does not exist (404).
    NotFound,
    /// The request is wrong and will fail again as is (400, 422, translation errors).
    InvalidRequest,
    /// The request is longer than the model's context window; the run loop compacts and tries again.
    ContextOverflow,
    /// Too many requests (429); see `Retry-After`.
    RateLimited,
    /// The provider failed (5xx, an overloaded event).
    Upstream,
    /// The connection failed or was cut before the answer was complete.
    Network,
    /// No answer, or no next chunk, within the timeout.
    Timeout,
    /// The answer could not be read, or exceeded a size cap.
    Malformed,
    /// The API refused a request because an earlier assistant turn's bytes differ from what it sent. A bug in the
    /// caller: never retried.
    HistoryMismatch,
    /// The model refused (`stop_reason: "refusal"`).
    Refusal,
    /// A stream sent a delta type the assembler does not know; the turn cannot be stored byte-stable.
    Unassemblable,
    /// A circuit breaker is open: the call was not made.
    CircuitOpen,
    /// A spend cap is reached: the call was not made.
    BudgetExceeded,
    /// A tool failed to run (not a tool result that reports an error: that is an answer).
    Tool,
    /// The permission rules or the host denied a tool call.
    PermissionDenied,
    /// A guard stopped the run (steps, time, spend, a repeated call).
    Guard,
    /// The log could not be read or written.
    Log,
    /// The run was interrupted.
    Cancelled,
    /// A broken invariant inside Sarathi: a bug, never retried.
    Internal,
    /// A kind this version does not know (read only; written as `unknown`).
    #[serde(other)]
    Unknown,
}

impl ErrorKind {
    /// Every kind this version writes, in declaration order ([`ErrorKind::Unknown`] is not one).
    pub const ALL: &'static [ErrorKind] = &[
        ErrorKind::Auth,
        ErrorKind::Permission,
        ErrorKind::NotFound,
        ErrorKind::InvalidRequest,
        ErrorKind::ContextOverflow,
        ErrorKind::RateLimited,
        ErrorKind::Upstream,
        ErrorKind::Network,
        ErrorKind::Timeout,
        ErrorKind::Malformed,
        ErrorKind::HistoryMismatch,
        ErrorKind::Refusal,
        ErrorKind::Unassemblable,
        ErrorKind::CircuitOpen,
        ErrorKind::BudgetExceeded,
        ErrorKind::Tool,
        ErrorKind::PermissionDenied,
        ErrorKind::Guard,
        ErrorKind::Log,
        ErrorKind::Cancelled,
        ErrorKind::Internal,
    ];

    /// The kind's stable name (`auth`, `rate_limited`, ...), the same as its serde name.
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorKind::Auth => "auth",
            ErrorKind::Permission => "permission",
            ErrorKind::NotFound => "not_found",
            ErrorKind::InvalidRequest => "invalid_request",
            ErrorKind::ContextOverflow => "context_overflow",
            ErrorKind::RateLimited => "rate_limited",
            ErrorKind::Upstream => "upstream",
            ErrorKind::Network => "network",
            ErrorKind::Timeout => "timeout",
            ErrorKind::Malformed => "malformed",
            ErrorKind::HistoryMismatch => "history_mismatch",
            ErrorKind::Refusal => "refusal",
            ErrorKind::Unassemblable => "unassemblable",
            ErrorKind::CircuitOpen => "circuit_open",
            ErrorKind::BudgetExceeded => "budget_exceeded",
            ErrorKind::Tool => "tool",
            ErrorKind::PermissionDenied => "permission_denied",
            ErrorKind::Guard => "guard",
            ErrorKind::Log => "log",
            ErrorKind::Cancelled => "cancelled",
            ErrorKind::Internal => "internal",
            ErrorKind::Unknown => "unknown",
        }
    }

    /// Whether the same call may succeed when made again.
    pub fn retryable(self) -> bool {
        matches!(
            self,
            ErrorKind::RateLimited | ErrorKind::Upstream | ErrorKind::Network | ErrorKind::Timeout
        )
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_stable_and_match_serde() {
        let names: Vec<_> = ErrorKind::ALL.iter().map(|k| k.as_str()).collect();
        assert_eq!(
            names,
            [
                "auth",
                "permission",
                "not_found",
                "invalid_request",
                "context_overflow",
                "rate_limited",
                "upstream",
                "network",
                "timeout",
                "malformed",
                "history_mismatch",
                "refusal",
                "unassemblable",
                "circuit_open",
                "budget_exceeded",
                "tool",
                "permission_denied",
                "guard",
                "log",
                "cancelled",
                "internal",
            ]
        );
        for &k in ErrorKind::ALL {
            let json = serde_json::to_string(&k).unwrap();
            assert_eq!(json, format!("\"{k}\""));
            assert_eq!(serde_json::from_str::<ErrorKind>(&json).unwrap(), k);
        }
    }

    #[test]
    fn an_unknown_name_reads_as_unknown() {
        assert_eq!(
            serde_json::from_str::<ErrorKind>(r#""overloaded""#).unwrap(),
            ErrorKind::Unknown
        );
        assert!(!ErrorKind::Unknown.retryable());
        assert!(!ErrorKind::ALL.contains(&ErrorKind::Unknown));
    }

    #[test]
    fn only_transient_kinds_are_retryable() {
        let retryable: Vec<_> = ErrorKind::ALL
            .iter()
            .copied()
            .filter(|k| k.retryable())
            .collect();
        assert_eq!(
            retryable,
            [
                ErrorKind::RateLimited,
                ErrorKind::Upstream,
                ErrorKind::Network,
                ErrorKind::Timeout
            ]
        );
    }
}
