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
//! The crate makes no model calls and decides nothing about when to compact: the run loop does, and appends the
//! entries this crate returns.

mod error;
mod fallback;
mod spill;

pub use error::ContextError;
pub use fallback::{DROPPED_NOTE, Keep, Seed, deterministic_seed, shrink_to_fit, strip_thinking};
#[cfg(feature = "fs")]
pub use spill::FsSpill;
pub use spill::{
    MemorySpill, SpillLimits, SpillRecord, SpillStore, Spilled, read_output, spill_output,
    verify_spill,
};
