// SPDX-License-Identifier: Apache-2.0
//! Context management over a session log (layer S2): what to do when the next request will not fit.
//!
//! - [`deterministic_seed`] and [`shrink_to_fit`]: overflow recovery that needs no model call. A new segment keeps
//!   the session's first user message, a fixed note, and the last turns; kept assistant turns are re-sent without
//!   their thinking blocks, because a kept turn that still carries its thinking fails the provider's check that
//!   thinking is only ever replayed in its original place.
//! - [`SpillStore`] with [`spill_output`] and [`read_output`]: tool output above a size limit is stored outside the
//!   log, and the `tool.result` slice holds a bounded preview and a handle the model can read ranges through.
//!
//! - [`before_request`] with [`estimate`]: whether the next request fits, needs an exact count, or is due a
//!   compaction; [`compaction_request`] and [`compaction_seed`] turn a compaction call (server block or client
//!   summary) into a new segment, and [`pending_compaction`] finds one a restart left open.
//!
//! The crate makes no model calls: the run loop makes the calls it asks for and appends the entries it returns.

mod compact;
mod error;
mod fallback;
mod plan;
mod spill;

pub use compact::{
    COMPACTION_BETA, CompactionRequest, Compactor, MAX_INSTRUCTIONS, PendingCompaction,
    SUMMARY_PREFIX, SUMMARY_PROMPT, carries_compaction_block, compaction_request, compaction_seed,
    pending_compaction,
};
pub use error::ContextError;
pub use fallback::{DROPPED_NOTE, Keep, Seed, deterministic_seed, shrink_to_fit, strip_thinking};
pub use plan::{ContextAction, ContextProfile, before_request, estimate, overflow_seed};
#[cfg(feature = "fs")]
pub use spill::FsSpill;
pub use spill::{
    MemorySpill, SpillLimits, SpillRecord, SpillStore, Spilled, read_output, spill_output,
    verify_spill,
};
