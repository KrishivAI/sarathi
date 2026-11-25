// Derived from Ultrafast Gateway @ c9c3eaf (MIT); see PROVENANCE.md
// SPDX-License-Identifier: Apache-2.0
use serde::{Deserialize, Serialize};

use crate::wire::error::TranslateError;

/// Who a message is from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Instructions to the model.
    System,
    /// The person or host.
    User,
    /// The model.
    Assistant,
    /// A tool's result.
    Tool,
}

/// Where an image comes from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ImageSource {
    /// An http(s) URL, passed on as is.
    Url(String),
    /// `data:<media_type>;base64,<data>`, split.
    Base64 {
        /// The image media type.
        media_type: String,
        /// The base64 data.
        data: String,
    },
}

/// One part of a message's content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Part {
    /// Text.
    Text(String),
    /// An image.
    Image(ImageSource),
}

/// A tool call the model made.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// The provider's call id.
    pub id: String,
    /// The tool's name.
    pub name: String,
    /// The arguments as the JSON text the model produced.
    pub arguments: String,
}

/// A tool the model may call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tool {
    /// The name the model calls it by.
    pub name: String,
    /// What the tool does, for the model.
    pub description: Option<String>,
    /// JSON Schema of the arguments; `{"type":"object"}` when not given.
    pub parameters: serde_json::Value,
    /// OpenAI's `function.strict` (schema-enforced arguments). Only the
    /// OpenAI format sends it; Anthropic has no such setting and ignores it.
    #[serde(default)]
    pub strict: Option<bool>,
    /// A freeform (Codex `custom`) tool: its input is one text, not a JSON
    /// object. Every provider is given it as a function with one string
    /// parameter, `input`.
    #[serde(default)]
    pub freeform: bool,
}

impl Tool {
    /// The parameter schema of a freeform tool.
    pub fn freeform_parameters() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
        })
    }

    /// The JSON Schema the provider is given: the tool's own, or the one
    /// string parameter of a freeform tool.
    pub fn wire_parameters(&self) -> std::borrow::Cow<'_, serde_json::Value> {
        if self.freeform {
            std::borrow::Cow::Owned(Self::freeform_parameters())
        } else {
            std::borrow::Cow::Borrowed(&self.parameters)
        }
    }
}

/// How the model must choose among the tools.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ToolChoice {
    /// The model decides.
    Auto,
    /// No tool may be called.
    None,
    /// Some tool must be called.
    Required,
    /// This tool must be called.
    Tool(String),
}

/// One message of a conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// Who it is from.
    pub role: Role,
    /// May be empty for an assistant message with tool calls.
    pub content: Vec<Part>,
    /// An optional participant name.
    pub name: Option<String>,
    /// Assistant only.
    pub tool_calls: Vec<ToolCall>,
    /// `Role::Tool` only, required there.
    pub tool_call_id: Option<String>,
}

impl Message {
    /// One text part, no tools.
    pub fn text(role: Role, text: impl Into<String>) -> Self {
        Message {
            role,
            content: vec![Part::Text(text.into())],
            name: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    /// Concatenation of the text parts.
    pub fn joined_text(&self) -> String {
        let mut out = String::new();
        for p in &self.content {
            if let Part::Text(t) = p {
                out.push_str(t);
            }
        }
        out
    }

    /// Whether any part is an image.
    pub fn has_images(&self) -> bool {
        self.content.iter().any(|p| matches!(p, Part::Image(_)))
    }
}

/// The image media types every provider accepts.
pub const IMAGE_TYPES: &[&str] = &["image/png", "image/jpeg", "image/gif", "image/webp"];

/// The alphabet and the length: standard or URL-safe (not both), with `=`
/// padding (at most two) at the very end; a length of 1 mod 4 is impossible
/// and padded data is a multiple of 4.
fn valid_base64(data: &str) -> bool {
    let body = data.trim_end_matches('=');
    let padding = data.len() - body.len();
    if body.is_empty() || padding > 2 {
        return false;
    }
    // One character is never a whole byte; padded data fills its last group.
    if data.len() % 4 == 1 || (padding > 0 && !data.len().is_multiple_of(4)) {
        return false;
    }
    let std = body.bytes().any(|b| matches!(b, b'+' | b'/'));
    let url = body.bytes().any(|b| matches!(b, b'-' | b'_'));
    !(std && url)
        && body
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'/' | b'-' | b'_'))
}

/// Checks the media type and the alphabet of base64 image data.
pub(crate) fn base64_source(media_type: &str, data: &str) -> Result<ImageSource, TranslateError> {
    if !IMAGE_TYPES.contains(&media_type) {
        return Err(TranslateError::InvalidRequest(format!(
            "image type '{media_type}' is not supported; use one of {}",
            IMAGE_TYPES.join(", ")
        )));
    }
    let valid = valid_base64(data);
    if !valid {
        return Err(TranslateError::InvalidRequest(
            "image data is not valid base64".to_string(),
        ));
    }
    Ok(ImageSource::Base64 {
        media_type: media_type.to_string(),
        data: data.to_string(),
    })
}

/// Parses `data:<type>;base64,<data>` or an http(s) URL.
pub fn image_source(url: &str) -> Result<ImageSource, TranslateError> {
    let bad = |m: &str| TranslateError::InvalidRequest(m.to_string());
    if let Some(rest) = url.strip_prefix("data:") {
        let (meta, data) = rest
            .split_once(',')
            .ok_or_else(|| bad("image data URL must be base64 encoded"))?;
        let media_type = meta
            .strip_suffix(";base64")
            .ok_or_else(|| bad("image data URL must be base64 encoded"))?;
        return base64_source(media_type, data);
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        return Ok(ImageSource::Url(url.to_string()));
    }
    Err(bad("image url must be an http(s) URL or a base64 data URL"))
}

/// A provider-neutral chat request.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    /// The model to call.
    pub model: String,
    /// The conversation, in order.
    pub messages: Vec<Message>,
    /// The most tokens the answer may hold.
    pub max_tokens: Option<u32>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus sampling probability.
    pub top_p: Option<f32>,
    /// Sequences that end the answer.
    pub stop: Option<Vec<String>>,
    /// Whether the answer is streamed.
    pub stream: bool,
    /// Tools the model may call.
    pub tools: Vec<Tool>,
    /// How the model must choose among the tools.
    pub tool_choice: Option<ToolChoice>,
    /// Whether several tool calls may come in one answer.
    pub parallel_tool_calls: Option<bool>,
    /// How the answer text must be shaped (structured outputs).
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    /// OpenAI's `reasoning_effort` (`none` .. `max`). Only the OpenAI format
    /// can express it; the Anthropic path refuses a call that sets it.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// Names of what translation left out of the caller's request (fields,
    /// tools, content blocks), for the call's log row. Empty for a request
    /// that was taken whole. Forms: `context_management`, `cache_control`,
    /// `tools[2]:web_search_20250305`, `messages[1].content[0]:thinking`.
    #[serde(default)]
    pub dropped: Vec<String>,
}

/// OpenAI's `response_format`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResponseFormat {
    /// Plain text.
    Text,
    /// Any JSON object.
    JsonObject,
    /// JSON that matches `schema` (a full JSON Schema).
    JsonSchema {
        /// The schema's name.
        name: String,
        /// The JSON Schema.
        schema: serde_json::Value,
        /// OpenAI's `strict`; other providers cannot express it and ignore it.
        strict: Option<bool>,
        /// What the schema is for.
        description: Option<String>,
    },
}

/// `input_tokens` is the uncached input (Anthropic's meaning); tokens read
/// from or written to a prompt cache are counted apart.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Input tokens not read from or written to a cache.
    pub input_tokens: u32,
    /// Output tokens.
    pub output_tokens: u32,
    /// Input tokens read from a prompt cache.
    #[serde(default)]
    pub cache_read_input_tokens: u32,
    /// Input tokens written to a prompt cache.
    #[serde(default)]
    pub cache_write_input_tokens: u32,
}

impl Usage {
    /// Every token the call sent in, cached or not: what limits, budgets
    /// and counters see as input volume.
    pub fn input_volume(&self) -> u64 {
        u64::from(self.input_tokens)
            + u64::from(self.cache_read_input_tokens)
            + u64::from(self.cache_write_input_tokens)
    }

    /// Input volume plus output.
    pub fn total_volume(&self) -> u64 {
        self.input_volume() + u64::from(self.output_tokens)
    }
}

/// Why the answer ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    /// The model finished, or hit a stop sequence.
    Stop,
    /// The token limit was reached.
    Length,
    /// The model called tools.
    ToolCalls,
    /// A provider filter cut the answer.
    ContentFilter,
}

impl FinishReason {
    /// The reason's OpenAI name (`stop`, `length`, `tool_calls`, `content_filter`).
    pub fn as_openai(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::ToolCalls => "tool_calls",
            FinishReason::ContentFilter => "content_filter",
        }
    }
}

/// A provider-neutral chat answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatResponse {
    /// The provider's answer id.
    pub id: String,
    /// The model that answered.
    pub model: String,
    /// The answer text.
    pub content: String,
    /// Tool calls, in order.
    pub tool_calls: Vec<ToolCall>,
    /// Why the answer ended, if the provider said.
    pub finish_reason: Option<FinishReason>,
    /// Token counts, if the provider gave them.
    pub usage: Option<Usage>,
}

/// One decoded event of a streamed answer.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    /// More answer text.
    Delta {
        /// The text.
        text: String,
    },
    /// A tool call begins. `index` counts tool calls of this answer from 0.
    ToolCallStart {
        /// The call's index.
        index: u32,
        /// The provider's call id.
        id: String,
        /// The tool's name.
        name: String,
    },
    /// More argument text for the call at `index`.
    ToolCallDelta {
        /// The call's index.
        index: u32,
        /// The argument text.
        arguments: String,
    },
    /// The answer ended.
    Done {
        /// Why, if the provider said.
        finish_reason: Option<FinishReason>,
        /// Token counts, if the provider gave them.
        usage: Option<Usage>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_source_parses_both_forms() {
        assert_eq!(
            image_source("https://x.test/a.png").unwrap(),
            ImageSource::Url("https://x.test/a.png".into())
        );
        assert_eq!(
            image_source("http://x.test/a.png").unwrap(),
            ImageSource::Url("http://x.test/a.png".into())
        );
        assert_eq!(
            image_source("data:image/png;base64,QUJD").unwrap(),
            ImageSource::Base64 {
                media_type: "image/png".into(),
                data: "QUJD".into()
            }
        );
        assert!(image_source("ftp://x/a.png").is_err());
        assert!(image_source("data:image/svg+xml;base64,QQ==").is_err());
    }

    #[test]
    fn base64_padding_and_alphabet_are_checked() {
        for ok in ["QUJD", "QQ==", "QUI=", "QQ", "QUI", "a-b_", "a+b/"] {
            assert!(
                image_source(&format!("data:image/png;base64,{ok}")).is_ok(),
                "{ok}"
            );
        }
        for bad in [
            "=QUJD", "QU=JD", "QQ===", "=", "a+b_", "a-b/", "QUJDx", "Q", "QUJDx=", "QQ=", "QUJ=Q",
        ] {
            assert!(
                image_source(&format!("data:image/png;base64,{bad}")).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn joined_text_skips_images() {
        let m = Message {
            role: Role::User,
            content: vec![
                Part::Text("a".into()),
                Part::Image(ImageSource::Url("https://x/y.png".into())),
                Part::Text("b".into()),
            ],
            name: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        };
        assert_eq!(m.joined_text(), "ab");
        assert!(m.has_images());
        assert!(!Message::text(Role::User, "x").has_images());
    }
}
