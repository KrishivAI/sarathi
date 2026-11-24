// SPDX-License-Identifier: Apache-2.0
//! The log's entries: the header line and one [`Entry`] per later line.

use std::ops::Range;

use sarathi_protocol::{RawJson, Seq, Usage};
use serde::{Deserialize, Serialize};

pub use sarathi_protocol::{
    EntryKind, ErrorKind, FORMAT_VERSION, ForkPoint, LogHeader, OLDEST_READABLE, Outcome,
    SegmentId, SegmentReason, SessionId,
};

/// A new session id: a UUID v7, so ids sort by creation time and make sortable directory names.
pub fn new_session_id() -> SessionId {
    SessionId(uuid::Uuid::now_v7().as_u128())
}

/// A header for a new session in the current format, created `created_ms` milliseconds after the Unix epoch.
pub fn new_header(
    session: SessionId,
    created_ms: u64,
    forked_from: Option<ForkPoint>,
) -> LogHeader {
    let header = LogHeader::new(
        session,
        rfc3339_utc(created_ms),
        concat!("sarathi-log ", env!("CARGO_PKG_VERSION")),
    );
    match forked_from {
        Some(at) => header.forked(at),
        None => header,
    }
}

/// `ms` milliseconds after the Unix epoch as RFC 3339 in UTC, to the second (`2026-11-25T08:00:00Z`).
pub fn rfc3339_utc(ms: u64) -> String {
    let secs = ms / 1000;
    let (days, rem) = (secs / 86_400, secs % 86_400);
    // Civil date from days since 1970-01-01 (H. Hinnant's algorithm), in unsigned arithmetic: days are never negative.
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z % 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + u64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

/// One line of the log after the header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// Position in the session: dense, from 1 (0 is the header).
    pub seq: Seq,
    /// The segment the entry belongs to: 0 at session start, +1 at each [`Body::SegmentStart`].
    pub seg: SegmentId,
    /// When it was written, milliseconds since the Unix epoch. Informational; nothing orders by it.
    pub ts: u64,
    /// What it records.
    pub body: Body,
}

/// Whether a tool changes anything outside the conversation. Decides what resume does with a call that started but
/// has no result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Effect {
    /// Read-only: safe to run again.
    Read,
    /// Side-effecting: never run again once started.
    Write,
}

/// How a compaction made its seed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Strategy {
    /// The provider compacted (a compaction block in an assistant turn).
    Server,
    /// A model wrote a summary on the client side.
    ClientSummary,
    /// No model call: existing slices kept verbatim with a fixed note. Overflow recovery.
    Deterministic,
}

/// A tool call an assistant message asked for.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CallRef {
    /// The provider's call id (`toolu_…`), or one the core made for a protocol without ids.
    pub id: Box<str>,
    /// The tool's name.
    pub name: Box<str>,
}

/// How a request body is rebuilt from the log: which prefix, which slices in order, and the per-call parameters.
///
/// The body is a function of the recipe and the log, so the log stores the recipe and the SHA-256 of the body that
/// went out instead of the body: storing every body would grow the log with the square of its turns.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Recipe {
    /// The `session.prefix` entry.
    pub prefix: Seq,
    /// The slice entries, in the order sent (`message.user`, `message.assistant`, `tool.result`).
    pub messages: Vec<Seq>,
    /// The per-call parameters (model, `max_tokens`, …) as rendered once.
    pub params: RawJson,
}

/// What an entry records. The `kind` on the line is in each variant's doc.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Body {
    /// `session.prefix`: the parts of every request that never change within a segment.
    SessionPrefix {
        /// The provider protocol the slices are in (`anthropic-messages`, `openai-chat`).
        protocol: Box<str>,
        /// The `system` value as sent.
        system: RawJson,
        /// The `tools` value as sent.
        tools: RawJson,
    },
    /// `message.user`: a user message or a steer, rendered once.
    UserMessage {
        /// The message's `content` value.
        content: RawJson,
        /// It arrived as a steer during a turn.
        steer: bool,
    },
    /// `model.request`: a call about to be made.
    ModelRequest {
        /// How to rebuild the body.
        recipe: Recipe,
        /// Lower-case hex SHA-256 of the body as sent.
        sha256: String,
    },
    /// `model.response`: a call's answer as received.
    ModelResponse {
        /// The body as received (streaming: the assembled message).
        body: RawJson,
        /// The byte range of the `content` array in `body`.
        content: Range<u32>,
        /// The stream text as received (streaming only).
        sse: Option<String>,
        /// The provider's request id.
        request_id: Option<Box<str>>,
        /// Token counts.
        usage: Usage,
    },
    /// `model.failure`: a call that ended without a usable answer.
    ///
    /// The error kind and a `turn.end`'s outcome are kept as the names read, not as enums, so a line from a newer
    /// writer re-encodes to the same bytes (fork hashes are over encoded lines); [`Body::error_kind`] and
    /// [`Body::outcome`] read them.
    ModelFailure {
        /// The error kind's name ([`ErrorKind`]).
        error: Box<str>,
        /// The HTTP status, if there was one.
        status: Option<u16>,
        /// What went wrong, with secrets removed.
        detail: String,
        /// The provider's request id.
        request_id: Option<Box<str>>,
        /// Retries made before giving up.
        retries: u32,
    },
    /// `message.assistant`: a response accepted into the conversation.
    AssistantMessage {
        /// The `model.response` entry whose `content` is the assistant message.
        from: Seq,
        /// The client tool calls it asks for, in order.
        calls: Vec<CallRef>,
    },
    /// `tool.ask`: a permission question put to the host.
    ToolAsk {
        /// The call.
        call: Box<str>,
        /// The tool.
        tool: Box<str>,
        /// The call's input as the model sent it.
        input: RawJson,
    },
    /// `tool.decision`: the host's answer to a `tool.ask`. The log does not interpret it.
    ToolDecision {
        /// The call.
        call: Box<str>,
        /// The decision as the tools layer renders it.
        decision: RawJson,
    },
    /// `tool.start`: a tool call is about to run. Durable before a [`Effect::Write`] tool runs.
    ToolStart {
        /// The call.
        call: Box<str>,
        /// The tool.
        tool: Box<str>,
        /// Whether running it again is safe.
        effect: Effect,
    },
    /// `tool.result`: a call's result block, rendered once.
    ToolResult {
        /// The call.
        call: Box<str>,
        /// The result block as it goes into the next request.
        block: RawJson,
        /// The result is an error.
        is_error: bool,
        /// Where the full output was spilled, if it was.
        spill: Option<RawJson>,
    },
    /// `segment.start`: a new request projection starts here.
    SegmentStart {
        /// Why.
        reason: SegmentReason,
        /// How the seed was made (compaction only).
        strategy: Option<Strategy>,
        /// The slices the segment starts with, by seq: new entries or existing ones referenced as they are.
        seed: Vec<Seq>,
        /// The seqs whose slices the projection no longer sends. Nothing in it is changed.
        replaces: Option<Range<Seq>>,
    },
    /// `turn.end`: a turn finished.
    TurnEnd {
        /// The outcome's name ([`Outcome`]: `finished`, `ran_out`, `interrupted`, `refused`, `failed`).
        outcome: Box<str>,
    },
    /// `note`: host or store metadata the core never reads.
    Note {
        /// Anything.
        data: RawJson,
    },
    /// A kind this reader does not know. Kept as its whole line and skipped by the projection.
    Unknown {
        /// The kind.
        kind: Box<str>,
        /// The whole line as read.
        line: RawJson,
    },
}

impl Body {
    /// The `kind` written on the line.
    pub fn kind(&self) -> EntryKind {
        match self {
            Body::SessionPrefix { .. } => EntryKind::SessionPrefix,
            Body::UserMessage { .. } => EntryKind::MessageUser,
            Body::ModelRequest { .. } => EntryKind::ModelRequest,
            Body::ModelResponse { .. } => EntryKind::ModelResponse,
            Body::ModelFailure { .. } => EntryKind::ModelFailure,
            Body::AssistantMessage { .. } => EntryKind::MessageAssistant,
            Body::ToolAsk { .. } => EntryKind::ToolAsk,
            Body::ToolDecision { .. } => EntryKind::ToolDecision,
            Body::ToolStart { .. } => EntryKind::ToolStart,
            Body::ToolResult { .. } => EntryKind::ToolResult,
            Body::SegmentStart { .. } => EntryKind::SegmentStart,
            Body::TurnEnd { .. } => EntryKind::TurnEnd,
            Body::Note { .. } => EntryKind::Note,
            Body::Unknown { kind, .. } => EntryKind::from_name(kind),
        }
    }

    /// A `turn.end`'s outcome; [`Outcome::Unknown`] for a name this version does not know. `None` for other kinds.
    pub fn outcome(&self) -> Option<Outcome> {
        match self {
            Body::TurnEnd { outcome } => Some(by_name(outcome).unwrap_or(Outcome::Unknown)),
            _ => None,
        }
    }

    /// A `model.failure`'s error kind; [`ErrorKind::Unknown`] for a name this version does not know. `None` for
    /// other kinds.
    pub fn error_kind(&self) -> Option<ErrorKind> {
        match self {
            Body::ModelFailure { error, .. } => Some(by_name(error).unwrap_or(ErrorKind::Unknown)),
            _ => None,
        }
    }

    /// The entry is a message slice: something a request projection sends.
    pub fn is_slice(&self) -> bool {
        matches!(
            self,
            Body::UserMessage { .. } | Body::AssistantMessage { .. } | Body::ToolResult { .. }
        )
    }
}

/// A unit enum of the protocol read from its name.
fn by_name<T: serde::de::DeserializeOwned>(name: &str) -> Option<T> {
    serde_json::from_value(serde_json::Value::String(name.to_owned())).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc3339_dates() {
        assert_eq!(rfc3339_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339_utc(951_782_400_000), "2000-02-29T00:00:00Z");
        assert_eq!(rfc3339_utc(1_795_593_600_999), "2026-11-25T08:00:00Z");
        assert_eq!(rfc3339_utc(4_107_542_399_000), "2100-02-28T23:59:59Z");
    }

    #[test]
    fn names_read_as_kinds() {
        let end = |o: &str| Body::TurnEnd { outcome: o.into() }.outcome();
        assert_eq!(end("ran_out"), Some(Outcome::RanOut));
        assert_eq!(end("snoozed"), Some(Outcome::Unknown));
        let fail = Body::ModelFailure {
            error: "context_overflow".into(),
            status: None,
            detail: String::new(),
            request_id: None,
            retries: 0,
        };
        assert_eq!(fail.error_kind(), Some(ErrorKind::ContextOverflow));
        assert_eq!(fail.outcome(), None);
        assert_eq!(
            Body::Note {
                data: RawJson::string("x")
            }
            .kind(),
            EntryKind::Note
        );
    }
}
