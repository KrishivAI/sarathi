// SPDX-License-Identifier: Apache-2.0
//! What the run loop reports to a host.
//!
//! Events are not the log: an event is what a host shows or forwards; the log is what resume needs. Most events have
//! a log entry behind them; streaming deltas are events only. Serialised externally tagged, as [`crate::Op`]. The
//! small enums written into log entries ([`DeltaKind`], [`SegmentReason`], [`Outcome`]) read a name they do not know
//! as `Unknown`, so an older reader keeps reading a newer log.

use serde::{Deserialize, Serialize};

use crate::error::ErrorKind;
use crate::ids::{CallId, SegmentId, TurnId};
use crate::message::Seq;
use crate::raw::RawJson;
use crate::view::{ResponseView, Usage};

/// A report from the run loop.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Event {
    /// A turn began.
    TurnStarted {
        /// The turn.
        turn: TurnId,
    },
    /// A model call is about to be sent.
    ModelCall {
        /// The log entry of the request.
        seq: Seq,
        /// The provider (`anthropic`, `openai`, a gateway's name).
        provider: Box<str>,
        /// The model id sent.
        model: Box<str>,
    },
    /// A piece of a streamed answer; never logged.
    Delta {
        /// The content block's index in the answer.
        block: u32,
        /// What the piece is.
        kind: DeltaKind,
        /// The piece.
        text: Box<str>,
    },
    /// The model answered and the answer was stored.
    AssistantMessage {
        /// The log entry of the answer.
        seq: Seq,
        /// What the answer holds.
        view: ResponseView,
    },
    /// A tool call needs the host's permission; answer with [`crate::Op::ToolDecision`].
    PermissionAsked {
        /// The call.
        call: CallId,
        /// The tool's name.
        tool: Box<str>,
        /// The call's input, as the model wrote it.
        input: RawJson,
    },
    /// A tool call began.
    ToolStarted {
        /// The call.
        call: CallId,
        /// The tool's name.
        tool: Box<str>,
    },
    /// A tool call ended.
    ToolFinished {
        /// The call.
        call: CallId,
        /// The result reports a failure.
        is_error: bool,
        /// The start of the result, for display.
        preview: Box<str>,
    },
    /// A new segment began.
    SegmentStarted {
        /// The segment.
        segment: SegmentId,
        /// Why.
        reason: SegmentReason,
    },
    /// What a model call used.
    Usage {
        /// The log entry of the answer.
        seq: Seq,
        /// The token counts.
        usage: Usage,
        /// The cost in micro-dollars, when the model's prices are known.
        cost_micros: Option<u64>,
    },
    /// A failure the run loop reports (the turn may go on, see [`Event::Done`]).
    Error {
        /// What kind of failure.
        kind: ErrorKind,
        /// A readable detail, holding no credential.
        detail: Box<str>,
    },
    /// A turn ended.
    Done {
        /// The turn.
        turn: TurnId,
        /// How it ended.
        outcome: Outcome,
    },
}

/// What a streamed piece is.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeltaKind {
    /// Answer text.
    Text,
    /// Thinking text.
    Thinking,
    /// A piece of a tool call's JSON input.
    InputJson,
    /// A name this version does not know (read only; written as `unknown`).
    #[serde(other)]
    Unknown,
}

/// Why a segment began.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SegmentReason {
    /// The context was compacted.
    Compaction,
    /// The session moved to another provider protocol; the history was translated once.
    ProtocolSwitch,
    /// The session was forked from another.
    Fork,
    /// A name this version does not know (read only; written as `unknown`).
    #[serde(other)]
    Unknown,
}

/// How a turn ended.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    /// The model finished, and the completion gate (if any) passed.
    Finished,
    /// A guard stopped it (steps, time, spend).
    RanOut,
    /// The host interrupted it.
    Interrupted,
    /// The model refused.
    Refused,
    /// An error ended it.
    Failed,
    /// A name this version does not know (read only; written as `unknown`).
    #[serde(other)]
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view::{BlockKind, BlockView};

    #[test]
    fn every_event_round_trips() {
        // Escapes, a number spelling and inner whitespace: a re-render would change them.
        let input = RawJson::from_text(HAZARD).unwrap();
        let events = [
            Event::TurnStarted {
                turn: TurnId(Seq(2)),
            },
            Event::ModelCall {
                seq: Seq(3),
                provider: "anthropic".into(),
                model: "claude-opus-5-5".into(),
            },
            Event::Delta {
                block: 0,
                kind: DeltaKind::Thinking,
                text: "Let me".into(),
            },
            Event::AssistantMessage {
                seq: Seq(4),
                view: ResponseView {
                    stop_reason: Some("tool_use".into()),
                    stop_details: Some(RawJson::from_text(HAZARD).unwrap()),
                    blocks: vec![BlockView {
                        kind: BlockKind::Opaque {
                            block_type: "future_block".into(),
                        },
                        range: 1..20,
                    }],
                    ..ResponseView::default()
                },
            },
            Event::PermissionAsked {
                call: "toolu_01".into(),
                tool: "edit".into(),
                input: input.clone(),
            },
            Event::ToolStarted {
                call: "toolu_01".into(),
                tool: "edit".into(),
            },
            Event::ToolFinished {
                call: "toolu_01".into(),
                is_error: false,
                preview: "1 line changed".into(),
            },
            Event::SegmentStarted {
                segment: SegmentId(1),
                reason: SegmentReason::Compaction,
            },
            Event::Usage {
                seq: Seq(4),
                usage: Usage::default(),
                cost_micros: Some(1_200),
            },
            Event::Error {
                kind: ErrorKind::RateLimited,
                detail: "slow down".into(),
            },
            Event::Done {
                turn: TurnId(Seq(2)),
                outcome: Outcome::Finished,
            },
        ];
        for e in events {
            let json = serde_json::to_string(&e).unwrap();
            if matches!(
                e,
                Event::PermissionAsked { .. } | Event::AssistantMessage { .. }
            ) {
                assert!(json.contains(HAZARD), "{json}");
            }
            assert_eq!(serde_json::from_str::<Event>(&json).unwrap(), e, "{json}");
        }
    }

    const HAZARD: &str = r#"{"z" : 1,"a":"\u00e9\/","n":1e5}"#;

    #[test]
    fn unknown_names_of_logged_enums_read_as_unknown() {
        let e: Event = serde_json::from_str(r#"{"done":{"turn":1,"outcome":"paused"}}"#).unwrap();
        assert_eq!(
            e,
            Event::Done {
                turn: TurnId(Seq(1)),
                outcome: Outcome::Unknown
            }
        );
        let s: SegmentReason = serde_json::from_str(r#""rebase""#).unwrap();
        assert_eq!(s, SegmentReason::Unknown);
        let d: DeltaKind = serde_json::from_str(r#""citation""#).unwrap();
        assert_eq!(d, DeltaKind::Unknown);
    }

    #[test]
    fn a_block_view_has_a_fixed_wire_shape() {
        let v = BlockView {
            kind: BlockKind::Text,
            range: 3..9,
        };
        let json = serde_json::to_string(&v).unwrap();
        assert_eq!(
            json,
            r#"{"kind":{"type":"text"},"range":{"start":3,"end":9}}"#
        );
        assert_eq!(serde_json::from_str::<BlockView>(&json).unwrap(), v);
    }

    #[test]
    fn names_on_the_wire() {
        let e = Event::Done {
            turn: TurnId(Seq(9)),
            outcome: Outcome::RanOut,
        };
        assert_eq!(
            serde_json::to_string(&e).unwrap(),
            r#"{"done":{"turn":9,"outcome":"ran_out"}}"#
        );
        let s = Event::SegmentStarted {
            segment: SegmentId(2),
            reason: SegmentReason::ProtocolSwitch,
        };
        assert_eq!(
            serde_json::to_string(&s).unwrap(),
            r#"{"segment_started":{"segment":2,"reason":"protocol_switch"}}"#
        );
    }
}
