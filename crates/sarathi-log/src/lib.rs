// SPDX-License-Identifier: Apache-2.0
//! The append-only session log (layer S1): the source of truth a run is resumed, forked, audited and replayed from.
//!
//! A session is a header and a dense sequence of [`Entry`]s. Nothing is ever edited: compaction, a provider protocol
//! switch and a fork each start a new segment ([`Body::SegmentStart`]) whose seed names the slices it begins with.
//! The next model request is a projection of the log ([`Log::projection`]), and every request the log records can
//! be rebuilt and checked against the hash of the body that was sent ([`Log::verify_requests`]).
//!
//! Stores: [`MemoryStore`], and [`JsonlStore`] (feature `jsonl`, on by default), one JSONL file per session. Both
//! append optimistically ([`LogStore::append`] names the seq it expects next) and keep the same invariants.
//!
//! The on-disk format carries a version ([`FORMAT_VERSION`]); a reader of format N reads N and N−1 and refuses newer
//! formats. Within a version, an unknown entry kind is kept and skipped, unless it is marked `must_understand`.

mod entry;
mod error;
#[cfg(feature = "jsonl")]
mod jsonl;
mod line;
mod log;
mod store;

pub use entry::{
    Body, CallRef, Effect, Entry, EntryKind, ErrorKind, FORMAT_VERSION, ForkPoint, LogHeader,
    OLDEST_READABLE, Outcome, Recipe, SEGMENT_SEED, SegmentId, SegmentReason, SessionId, Strategy,
    new_header, new_session_id, rfc3339_utc,
};
pub use error::LogError;
#[cfg(feature = "jsonl")]
pub use jsonl::JsonlStore;
pub use line::{ReadLog, decode_header, encode_entry, encode_header, read_log, render_log};
pub use log::{
    CUT_SHORT, CallStatus, Log, OpenCall, Projection, RequestParts, ResumeState, Slice, sha256_hex,
};
pub use store::{Durability, Loaded, LogStore, MemoryStore, Recovered, fork};
