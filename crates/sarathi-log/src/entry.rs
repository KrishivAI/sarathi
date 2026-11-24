// SPDX-License-Identifier: Apache-2.0
//! The log's entries: the header line and one [`Entry`] per later line.

use std::fmt;
use std::ops::Range;
use std::str::FromStr;

use sarathi_protocol::{RawJson, Seq, Usage};
use serde::{Deserialize, Serialize};

/// The format this crate writes.
pub const FORMAT_VERSION: u32 = 1;

/// The oldest format this crate reads. A reader of format N reads N and N−1; format 1 is the first, so there is no
/// N−1 yet.
pub const OLDEST_READABLE: u32 = 1;

/// A session's id: a UUID v7, so ids sort by creation time and make sortable directory names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionId(uuid::Uuid);

impl SessionId {
    /// A new time-ordered id.
    pub fn new() -> Self {
        SessionId(uuid::Uuid::now_v7())
    }
}

impl Default for SessionId {
    fn default() -> Self {
        SessionId::new()
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.hyphenated().fmt(f)
    }
}

impl FromStr for SessionId {
    type Err = uuid::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        uuid::Uuid::parse_str(s).map(SessionId)
    }
}

/// Where a forked session came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForkedFrom {
    /// The parent session.
    pub session: SessionId,
    /// The last parent entry copied into the child.
    pub seq: Seq,
    /// Lower-case hex SHA-256 of the parent's entry lines `1..=seq`, each with its `\n`.
    pub sha256: String,
}

/// Line 0 of every log.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Header {
    /// The format version ([`FORMAT_VERSION`] when written by this crate).
    pub sarathi_log: u32,
    /// The session.
    pub session: SessionId,
    /// Creation time, milliseconds since the Unix epoch.
    pub created: u64,
    /// The crate and version that created the log (`sarathi-log 0.1.0`). Informational.
    pub writer: String,
    /// Set when the session is a fork.
    pub forked_from: Option<ForkedFrom>,
}

impl Header {
    /// A header for a new session, in the current format.
    pub fn new(session: SessionId, created: u64, forked_from: Option<ForkedFrom>) -> Self {
        Header {
            sarathi_log: FORMAT_VERSION,
            session,
            created,
            writer: concat!("sarathi-log ", env!("CARGO_PKG_VERSION")).to_owned(),
            forked_from,
        }
    }
}

/// One line of the log after the header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// Position in the session: dense, from 1 (0 is the header).
    pub seq: Seq,
    /// The segment the entry belongs to: 0 at session start, +1 at each [`Body::SegmentStart`].
    pub seg: u32,
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

/// Why a segment started.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SegmentReason {
    /// The context was compacted.
    Compaction,
    /// The session moved to another provider protocol; the seed is the history translated once.
    ProtocolSwitch,
    /// The session is a fork; the seed is the parent's projection at the fork point.
    Fork,
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
    ModelFailure {
        /// The error kind's name.
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
        /// The outcome's name (`finished`, `ran_out`, `interrupted`, `refused`, `failed`).
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
    pub fn kind(&self) -> &str {
        match self {
            Body::SessionPrefix { .. } => "session.prefix",
            Body::UserMessage { .. } => "message.user",
            Body::ModelRequest { .. } => "model.request",
            Body::ModelResponse { .. } => "model.response",
            Body::ModelFailure { .. } => "model.failure",
            Body::AssistantMessage { .. } => "message.assistant",
            Body::ToolAsk { .. } => "tool.ask",
            Body::ToolDecision { .. } => "tool.decision",
            Body::ToolStart { .. } => "tool.start",
            Body::ToolResult { .. } => "tool.result",
            Body::SegmentStart { .. } => "segment.start",
            Body::TurnEnd { .. } => "turn.end",
            Body::Note { .. } => "note",
            Body::Unknown { kind, .. } => kind,
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
