// SPDX-License-Identifier: Apache-2.0
//! [`LogError`]: every way reading or writing a session log can fail.

use sarathi_protocol::Seq;

/// A log operation that failed. No variant is retried by the store itself.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LogError {
    /// The session's next seq is not the one the writer expected: another writer appended first.
    #[error("append conflict: expected next seq {expected}, the log is at {actual}")]
    Conflict {
        /// The seq the writer passed.
        expected: u64,
        /// The session's actual next seq.
        actual: u64,
    },
    /// No session with this id.
    #[error("no session {0}")]
    NotFound(String),
    /// The session is held by another writer (another process, or another store over the same directory).
    #[error("session {0} is locked by another writer")]
    Locked(String),
    /// The log was written by a newer format than this reader supports. It is never guessed at.
    #[error("log format {found} is newer than this reader (reads {oldest} to {supported})")]
    NewerFormat {
        /// The format in the header.
        found: u32,
        /// The oldest format this reader reads.
        oldest: u32,
        /// The newest format this reader reads.
        supported: u32,
    },
    /// A line that is not the last one does not decode. Never repaired; the host decides.
    #[error("corrupt log line {line}: {detail}")]
    Corrupt {
        /// The 0-based line number (0 is the header).
        line: u64,
        /// What is wrong with it.
        detail: String,
    },
    /// An entry of a kind this reader does not know is marked `must_understand`.
    #[error("entry {} has kind `{kind}`, which this reader does not know and must understand", .seq.0)]
    MustUnderstand {
        /// The entry's seq.
        seq: Seq,
        /// The unknown kind.
        kind: String,
    },
    /// An entry that breaks a log invariant (seqs not dense, a segment that does not start with `segment.start`, a
    /// reference to a seq that is not a slice, …). The append is refused whole.
    #[error("invalid entry {}: {detail}", .seq.0)]
    Invalid {
        /// The entry's seq.
        seq: Seq,
        /// The broken invariant.
        detail: String,
    },
    /// A fork point inside a tool round or a pending model call.
    #[error("seq {} is not a turn boundary", .0.0)]
    NotABoundary(Seq),
    /// A model request's recipe does not rebuild to the recorded hash.
    #[error("model request {}: recipe rebuilds to {actual}, the entry records {recorded}", .seq.0)]
    HashMismatch {
        /// The `model.request` entry.
        seq: Seq,
        /// The hash in the entry.
        recorded: String,
        /// The hash of the rebuilt body.
        actual: String,
    },
    /// File IO failed.
    #[error("log IO: {0}")]
    Io(#[from] std::io::Error),
}
