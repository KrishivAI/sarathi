// SPDX-License-Identifier: Apache-2.0
//! The shallow view of a model response: what the core reads, never what is sent back.

use std::ops::Range;

use serde::{Deserialize, Serialize};

use crate::raw::RawJson;

/// Token counts of one call, as the provider reported them. Missing and `null` counts are 0.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(default)]
pub struct Usage {
    /// Input tokens that were not read from or written to the cache.
    #[serde(deserialize_with = "null_as_zero")]
    pub input_tokens: u64,
    /// Output tokens, thinking included.
    #[serde(deserialize_with = "null_as_zero")]
    pub output_tokens: u64,
    /// Input tokens written to the prompt cache.
    #[serde(deserialize_with = "null_as_zero")]
    pub cache_creation_input_tokens: u64,
    /// Input tokens read from the prompt cache.
    #[serde(deserialize_with = "null_as_zero")]
    pub cache_read_input_tokens: u64,
}

/// A count the provider may send as `null` (the SDKs type them as nullable).
fn null_as_zero<'de, D: serde::Deserializer<'de>>(d: D) -> Result<u64, D::Error> {
    Option::<u64>::deserialize(d).map(Option::unwrap_or_default)
}

/// What one content block is, as far as the core needs to know.
///
/// The core never needs to understand a block to send it back, because it sends the stored slice, not this view:
/// a block type Sarathi does not know is [`BlockKind::Opaque`] and goes back byte for byte.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BlockKind {
    /// `text`.
    Text,
    /// `thinking` (its text may be empty: the signature alone is kept).
    Thinking,
    /// `redacted_thinking`.
    RedactedThinking,
    /// `tool_use`: a client tool the host runs.
    ToolUse {
        /// The call id (`toolu_…`).
        id: Box<str>,
        /// The tool name.
        name: Box<str>,
        /// The byte range of `input` in the response body.
        input: Range<u32>,
    },
    /// `server_tool_use` or `mcp_tool_use`: a tool the provider runs.
    ServerToolUse {
        /// The call id.
        id: Box<str>,
        /// The tool name.
        name: Box<str>,
    },
    /// A provider-run tool's result (`web_search_tool_result`, `mcp_tool_result`, …).
    ServerToolResult,
    /// `compaction`.
    Compaction,
    /// Any other block type, kept whole.
    Opaque {
        /// The block's `type`.
        block_type: Box<str>,
    },
}

/// One content block: what it is and where its bytes are.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockView {
    /// What the block is.
    pub kind: BlockKind,
    /// The byte range of the whole block in the response body.
    pub range: Range<u32>,
}

/// The parts of a response the core reads: the stop reason, usage and the blocks. Ranges point into the response
/// body; a body over 4 GiB is refused, so they fit `u32`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseView {
    /// The message id (`msg_…`).
    pub id: Option<Box<str>>,
    /// The model that answered.
    pub model: Option<Box<str>>,
    /// Why the model stopped (`end_turn`, `tool_use`, `max_tokens`, `refusal`, `pause_turn`, …).
    pub stop_reason: Option<Box<str>>,
    /// `stop_details` as sent (the refusal category, for one).
    pub stop_details: Option<RawJson>,
    /// Token counts.
    pub usage: Usage,
    /// The content blocks, in order.
    pub blocks: Vec<BlockView>,
}

impl ResponseView {
    /// The client tool calls, in order: `(id, name, input range)`.
    pub fn tool_uses(&self) -> impl Iterator<Item = (&str, &str, Range<u32>)> {
        self.blocks.iter().filter_map(|b| match &b.kind {
            BlockKind::ToolUse { id, name, input } => Some((&**id, &**name, input.clone())),
            _ => None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_uses_lists_client_calls_only() {
        let v = ResponseView {
            blocks: vec![
                BlockView {
                    kind: BlockKind::Text,
                    range: 0..5,
                },
                BlockView {
                    kind: BlockKind::ToolUse {
                        id: "toolu_1".into(),
                        name: "edit".into(),
                        input: 7..9,
                    },
                    range: 6..10,
                },
                BlockView {
                    kind: BlockKind::ServerToolUse {
                        id: "srvtoolu_1".into(),
                        name: "web_search".into(),
                    },
                    range: 11..20,
                },
            ],
            ..ResponseView::default()
        };
        let calls: Vec<_> = v.tool_uses().collect();
        assert_eq!(calls, vec![("toolu_1", "edit", 7..9)]);
    }

    #[test]
    fn usage_defaults_missing_counts() {
        let u: Usage = serde_json::from_str(
            r#"{"input_tokens":3,"cache_read_input_tokens":null,"server_tool_use":{"x":1}}"#,
        )
        .unwrap();
        assert_eq!(
            u,
            Usage {
                input_tokens: 3,
                ..Usage::default()
            }
        );
        let k = serde_json::to_string(&BlockKind::Opaque {
            block_type: "x".into(),
        })
        .unwrap();
        assert_eq!(k, r#"{"type":"opaque","block_type":"x"}"#);
    }
}
