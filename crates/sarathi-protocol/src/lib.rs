// SPDX-License-Identifier: Apache-2.0
//! The types every Sarathi crate shares (layer S0): raw JSON kept as received, message slices, the view of a model
//! answer and its content blocks (thinking and unknown block types included), ids, the commands a host sends
//! ([`Op`]), the reports it gets ([`Event`]), the error kinds, and the session log's header and entry kinds. No IO.
//!
//! The rule the types serve: provider bytes survive a round trip untouched. A content block is never parsed into a
//! tree and written again; it is kept as [`RawJson`] and described by a [`BlockView`] of ranges into it, so a block
//! type this crate does not know ([`BlockKind::Opaque`]) goes back to the provider exactly as it came.

mod error;
mod event;
mod ids;
mod log;
mod message;
mod op;
mod raw;
mod view;

pub use error::ErrorKind;
pub use event::{DeltaKind, Event, Outcome, SegmentReason};
pub use ids::{BadSessionId, CallId, SegmentId, SessionId, TurnId};
pub use log::{
    BadSha256, EntryHead, EntryKind, FORMAT_VERSION, ForkPoint, HeaderError, LogHeader,
    MUST_UNDERSTAND, OLDEST_READABLE, Sha256, UnknownRequired, VersionError, check_version,
    read_header,
};
pub use message::{MessageSlice, Role, Seq};
pub use op::{Decision, Op, ToolOutput};
pub use raw::{RawJson, RawJsonError};
pub use view::{BlockKind, BlockView, ResponseView, Usage};
