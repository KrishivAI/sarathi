// SPDX-License-Identifier: Apache-2.0
//! [`ContextError`]: every way a context operation can fail.

use sarathi_log::LogError;
use sarathi_protocol::RawJsonError;

/// A context operation that failed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ContextError {
    /// The log refused or could not answer.
    #[error(transparent)]
    Log(#[from] LogError),
    /// A tool call of the last assistant message has no result, or a model call is in flight: compaction happens only
    /// between requests, never inside a tool round.
    #[error("not at a request boundary: a tool call has no result")]
    NotAtBoundary,
    /// The current segment holds no user message to cut at.
    #[error("the current segment has no user message")]
    NoUserMessage,
    /// Keeping the requested turns would drop nothing.
    #[error("nothing to drop: the segment already starts at the kept turns")]
    NothingToDrop,
    /// Even keeping only the current turn does not fit. The turn fails; nothing is retried.
    #[error("the request does not fit even with only the current turn kept")]
    DoesNotFit,
    /// A call id that cannot name a spill file.
    #[error("call id `{0}` cannot name a spill file")]
    BadCallId(String),
    /// No spilled output for the handle.
    #[error("no spilled output `{0}`")]
    SpillNotFound(String),
    /// A ranged read asked for more than the per-call cap.
    #[error("read of {asked} bytes is over the cap of {cap}")]
    ReadTooLarge {
        /// The length asked for.
        asked: usize,
        /// The cap.
        cap: usize,
    },
    /// Spilled bytes whose hash is not the one the log recorded.
    #[error("spilled output `{0}` does not match its recorded hash")]
    SpillMismatch(String),
    /// A value that is not the JSON it should be.
    #[error("{0}")]
    Json(#[from] RawJsonError),
    /// Turn parameters that cannot take a `compaction` member.
    #[error("compaction params: {0}")]
    BadParams(String),
    /// Server compaction instructions past the provider's limit.
    #[error("compaction instructions of {0} characters are over the limit")]
    InstructionsTooLong(usize),
    /// No compaction call is pending, or it has no response yet.
    #[error("no compaction call with a response is pending")]
    NoCompaction,
    /// The compaction call answered without a usable summary.
    #[error("the compaction call returned no summary (stop reason {stop_reason:?})")]
    CompactionFailed {
        /// The response's stop reason, if it had one.
        stop_reason: Option<String>,
    },
    /// File IO failed.
    #[error("spill IO: {0}")]
    Io(#[from] std::io::Error),
}
