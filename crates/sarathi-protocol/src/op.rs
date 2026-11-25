// SPDX-License-Identifier: Apache-2.0
//! What a host sends the run loop.
//!
//! Serialised externally tagged (`{"user_message":{"content":...}}`): [`RawJson`] keeps its bytes only when serde
//! does not buffer the enum's content, which internally tagged, adjacently tagged and untagged enums do.

use serde::{Deserialize, Serialize};

use crate::ids::CallId;
use crate::raw::RawJson;

/// A command to the run loop.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Op {
    /// A user message, rendered once by the host for the session's protocol and stored as is.
    UserMessage {
        /// The message content, as the provider protocol wants it.
        content: RawJson,
    },
    /// A message for the running turn, applied at the next tool boundary (queued until then).
    Steer {
        /// The message content, as for [`Op::UserMessage`].
        content: RawJson,
    },
    /// Stop the running model or tool call.
    Interrupt,
    /// The answer to a pending permission ask ([`crate::Event::PermissionAsked`]).
    ToolDecision {
        /// The call the answer is for.
        call: CallId,
        /// The answer.
        decision: Decision,
    },
    /// The result of a tool the host ran itself.
    ToolOutput {
        /// The call the result is for.
        call: CallId,
        /// The result.
        output: ToolOutput,
    },
    /// Finish the running step, then stop.
    Shutdown,
}

/// A host's answer to a permission ask.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    /// Run this call.
    AllowOnce,
    /// Run this call and later calls of the same tool, for the rest of the session.
    AllowAlways,
    /// Do not run it; the model is told why.
    Deny {
        /// What the model is told.
        reason: Box<str>,
    },
}

/// A tool's result, rendered once for the session's protocol.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolOutput {
    /// The result content, as the provider protocol wants it in a tool result.
    pub content: RawJson,
    /// The tool reports a failure (the model sees it as an error result).
    pub is_error: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    const HAZARD: &str = r#"{"z":1,"a":"é\/","n":1e5}"#;

    fn raw(s: &str) -> RawJson {
        RawJson::from_text(s).unwrap()
    }

    #[test]
    fn every_op_round_trips_with_its_raw_bytes_kept() {
        let ops = [
            Op::UserMessage {
                content: raw(HAZARD),
            },
            Op::Steer {
                content: raw(HAZARD),
            },
            Op::Interrupt,
            Op::ToolDecision {
                call: "toolu_01".into(),
                decision: Decision::AllowOnce,
            },
            Op::ToolDecision {
                call: "toolu_01".into(),
                decision: Decision::AllowAlways,
            },
            Op::ToolDecision {
                call: "s4-0".into(),
                decision: Decision::Deny {
                    reason: "outside the project".into(),
                },
            },
            Op::ToolOutput {
                call: "toolu_02".into(),
                output: ToolOutput {
                    content: raw(HAZARD),
                    is_error: true,
                },
            },
            Op::Shutdown,
        ];
        for op in ops {
            let json = serde_json::to_string(&op).unwrap();
            if matches!(
                op,
                Op::UserMessage { .. } | Op::Steer { .. } | Op::ToolOutput { .. }
            ) {
                assert!(json.contains(HAZARD), "{json}");
            }
            assert_eq!(serde_json::from_str::<Op>(&json).unwrap(), op, "{json}");
        }
    }

    #[test]
    fn the_wire_shape_is_externally_tagged() {
        assert_eq!(
            serde_json::to_string(&Op::Interrupt).unwrap(),
            r#""interrupt""#
        );
        let op = Op::UserMessage {
            content: raw(r#"[{"type":"text","text":"hi"}]"#),
        };
        assert_eq!(
            serde_json::to_string(&op).unwrap(),
            r#"{"user_message":{"content":[{"type":"text","text":"hi"}]}}"#
        );
        let back: Op = serde_json::from_str(r#"{"steer":{"content":{"k":"v" }}}"#).unwrap();
        let Op::Steer { content } = back else {
            panic!("{back:?}")
        };
        assert_eq!(
            content.as_str(),
            r#"{"k":"v" }"#,
            "whitespace inside the raw part is kept"
        );
    }
}
