// SPDX-License-Identifier: Apache-2.0
//! The OpenAI chat-completions call without IO, rendered once and spliced, like [`super::native`].
//!
//! The translated path ([`super::provider`]) goes through `serde_json::Value`, which re-orders keys and re-spells
//! escapes, so a conversation re-rendered from it changes bytes from one call to the next and an OpenAI-compatible
//! prefix cache misses. Here a body is written from byte slices made once: the per-call parameters ([`ChatParams`]),
//! `system` and `tools` as given, and each message as it was first rendered (an assistant message from the answer it
//! came in). [`parse`] reads an answer into the shallow [`ResponseView`] without changing it.

use std::fmt;
use std::ops::Range;

use bytes::Bytes;
use sarathi_protocol::{BlockKind, BlockView, RawJson, ResponseView, Usage};
use serde::Deserialize;
use serde_json::value::RawValue;

use super::native::NativeError;
use super::native::response::range;

/// Request keys Sarathi writes itself, or that would change what an answer holds (`n`: only the first choice is
/// read; the legacy `functions` call style; `stream_options` without a stream); [`ChatParams::with_extra`] refuses
/// them.
pub const RESERVED_KEYS: [&str; 10] = [
    "model",
    "max_tokens",
    "max_completion_tokens",
    "messages",
    "tools",
    "stream",
    "stream_options",
    "n",
    "functions",
    "function_call",
];

/// The name the body gives the output limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenField {
    /// `max_tokens`: every OpenAI-compatible server knows it.
    MaxTokens,
    /// `max_completion_tokens`: OpenAI itself, whose reasoning models refuse `max_tokens`.
    MaxCompletionTokens,
}

impl TokenField {
    /// The member's name.
    pub fn as_str(self) -> &'static str {
        match self {
            TokenField::MaxTokens => "max_tokens",
            TokenField::MaxCompletionTokens => "max_completion_tokens",
        }
    }

    /// The field named `name`.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "max_tokens" => Some(TokenField::MaxTokens),
            "max_completion_tokens" => Some(TokenField::MaxCompletionTokens),
            _ => None,
        }
    }
}

/// The per-call parameters: model, output limit and the caller's extra members, in their order and bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatParams {
    model: String,
    max_tokens: u32,
    field: TokenField,
    extra: Vec<(String, RawJson)>,
}

impl ChatParams {
    /// `model` and an output limit of `max_tokens` under the name `field`, nothing else.
    pub fn new(model: impl Into<String>, max_tokens: u32, field: TokenField) -> Self {
        ChatParams {
            model: model.into(),
            max_tokens,
            field,
            extra: Vec::new(),
        }
    }

    /// Adds the members of `extra`, a JSON object (`response_format`, `temperature`, `tool_choice`,
    /// `parallel_tool_calls`, …), written into the body in their order with their values' bytes unchanged.
    ///
    /// # Errors
    /// `extra` is not an object, repeats a key, or names a key in [`RESERVED_KEYS`].
    pub fn with_extra(mut self, extra: &RawJson) -> Result<Self, NativeError> {
        let members = super::native::members(extra)?;
        if let Some((k, _)) = members
            .iter()
            .find(|(k, _)| RESERVED_KEYS.contains(&k.as_str()))
        {
            return Err(NativeError::Invalid(format!(
                "`{k}` is written by Sarathi and cannot be an extra parameter"
            )));
        }
        self.extra = members;
        Ok(self)
    }

    /// The model.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The most output tokens.
    pub fn max_tokens(&self) -> u32 {
        self.max_tokens
    }

    /// The output limit's name.
    pub fn field(&self) -> TokenField {
        self.field
    }
}

/// A chat-completions request: the parameters, `system` and `tools` once, then whole messages as spliced bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatBody {
    params: ChatParams,
    system: Option<RawJson>,
    tools: Option<RawJson>,
    messages: Vec<RawJson>,
}

impl ChatBody {
    /// A request with no messages yet. `system` is the system message's `content` value (a string or an array of
    /// text parts); `tools` the `tools` array.
    pub fn new(params: ChatParams, system: Option<RawJson>, tools: Option<RawJson>) -> Self {
        ChatBody {
            params,
            system,
            tools,
            messages: Vec::new(),
        }
    }

    /// Appends a whole message (`{"role":…,…}`), its bytes unchanged.
    pub fn push(&mut self, message: RawJson) {
        self.messages.push(message);
    }

    /// The messages after the system message.
    pub fn messages(&self) -> &[RawJson] {
        &self.messages
    }

    /// The body: `model`, the output limit, the extra members, `messages` (the system message first), `tools`.
    /// Never `stream`: the answer comes whole.
    pub fn body(&self) -> Vec<u8> {
        let len = self.messages.iter().map(|m| m.len() + 1).sum::<usize>()
            + self.system.as_ref().map_or(0, RawJson::len)
            + self.tools.as_ref().map_or(0, RawJson::len)
            + self
                .params
                .extra
                .iter()
                .map(|(k, v)| k.len() + v.len() + 4)
                .sum::<usize>()
            + 128;
        let mut out = String::with_capacity(len);
        out.push_str(r#"{"model":"#);
        out.push_str(RawJson::string(&self.params.model).as_str());
        out.push_str(",\"");
        out.push_str(self.params.field.as_str());
        out.push_str("\":");
        out.push_str(&self.params.max_tokens.to_string());
        for (k, v) in &self.params.extra {
            out.push(',');
            out.push_str(RawJson::string(k).as_str());
            out.push(':');
            out.push_str(v.as_str());
        }
        out.push_str(r#","messages":["#);
        let mut first = true;
        if let Some(system) = &self.system {
            out.push_str(r#"{"role":"system","content":"#);
            out.push_str(system.as_str());
            out.push('}');
            first = false;
        }
        for m in &self.messages {
            if !first {
                out.push(',');
            }
            out.push_str(m.as_str());
            first = false;
        }
        out.push(']');
        if let Some(tools) = &self.tools {
            out.push_str(r#","tools":"#);
            out.push_str(tools.as_str());
        }
        out.push('}');
        out.into_bytes()
    }
}

/// A chat-completions answer: the body as received, where its first choice's `message` is, and the view.
#[derive(Clone, PartialEq, Eq)]
pub struct ChatAnswer {
    /// The body exactly as received.
    pub raw: Bytes,
    /// The byte range of `choices[0].message` in `raw`: the assistant message is re-sent from it.
    pub message: Range<u32>,
    /// The shallow view; its ranges point into `raw`. A text block's range is the `content` string (or a text
    /// part), a tool call's input range is its `arguments` value: a string whose text is the JSON input, or (some compatible
    /// servers) the input object itself.
    pub view: ResponseView,
    /// The `x-request-id` header.
    pub request_id: Option<String>,
}

/// Never prints the body, which holds user content.
impl fmt::Debug for ChatAnswer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChatAnswer")
            .field("raw", &format_args!("{} bytes", self.raw.len()))
            .field("message", &self.message)
            .field("view", &self.view)
            .field("request_id", &self.request_id)
            .finish()
    }
}

#[derive(Deserialize)]
struct Top<'a> {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(borrow)]
    choices: Vec<Choice<'a>>,
    #[serde(default)]
    usage: Option<WireUsage>,
}

#[derive(Deserialize)]
struct Choice<'a> {
    #[serde(borrow)]
    message: &'a RawValue,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct Message<'a> {
    #[serde(default, borrow)]
    content: Option<&'a RawValue>,
    #[serde(default, borrow)]
    tool_calls: Option<Vec<WireCall<'a>>>,
    #[serde(default)]
    refusal: Option<String>,
}

#[derive(Deserialize)]
struct WireCall<'a> {
    id: String,
    #[serde(borrow)]
    function: Function<'a>,
}

#[derive(Deserialize)]
struct Function<'a> {
    name: String,
    #[serde(borrow)]
    arguments: &'a RawValue,
}

#[derive(Deserialize)]
struct Part<'a> {
    #[serde(rename = "type")]
    kind: std::borrow::Cow<'a, str>,
}

#[derive(Deserialize, Default)]
struct WireUsage {
    #[serde(default)]
    prompt_tokens: Option<u64>,
    #[serde(default)]
    completion_tokens: Option<u64>,
    #[serde(default)]
    prompt_tokens_details: Option<Details>,
}

#[derive(Deserialize, Default)]
struct Details {
    #[serde(default)]
    cached_tokens: Option<u64>,
}

/// Reads `raw`, a chat-completions answer body, into a [`ChatAnswer`] with no request id. The bytes are kept as
/// given; parsing only locates ranges and reads scalars.
///
/// The view speaks the same words as a Messages answer where the core reads them: a non-empty `refusal`, or a
/// `content_filter` finish, is `stop_reason: "refusal"`; any other finish reason is kept as sent (`stop`,
/// `tool_calls`, `length`). Usage: `input_tokens` is the prompt tokens not read from the cache, `cache_read_input_tokens`
/// the cached ones.
///
/// # Errors
/// The body is over 4 GiB, not UTF-8, not an object with a non-empty `choices` array whose first `message` is an
/// object, or a tool call has no `id`, `function.name` or `function.arguments`, or its `arguments` is neither a string
/// nor an object.
pub fn parse(raw: Bytes) -> Result<ChatAnswer, NativeError> {
    let malformed = |m: String| NativeError::Malformed(m);
    if u32::try_from(raw.len()).is_err() {
        return Err(malformed("the body is larger than 4 GiB".into()));
    }
    let text = std::str::from_utf8(&raw).map_err(|e| malformed(format!("not UTF-8: {e}")))?;
    let top: Top<'_> =
        serde_json::from_str(text).map_err(|e| malformed(format!("not a chat completion: {e}")))?;
    let Some(choice) = top.choices.first() else {
        return Err(malformed("the answer has no choices".into()));
    };
    let message: Message<'_> = serde_json::from_str(choice.message.get())
        .map_err(|e| malformed(format!("the message cannot be read: {e}")))?;
    let mut blocks = Vec::new();
    match message.content {
        Some(c) if c.get().starts_with('"') => blocks.push(BlockView {
            kind: BlockKind::Text,
            range: range(text, c.get())?,
        }),
        Some(c) if c.get().starts_with('[') => {
            let parts: Vec<&RawValue> = serde_json::from_str(c.get())
                .map_err(|e| malformed(format!("`content` cannot be read: {e}")))?;
            for p in parts {
                let head: Part<'_> = serde_json::from_str(p.get())
                    .map_err(|e| malformed(format!("a content part without a type: {e}")))?;
                let kind = if head.kind == "text" {
                    BlockKind::Text
                } else {
                    BlockKind::Opaque {
                        block_type: head.kind.as_ref().into(),
                    }
                };
                blocks.push(BlockView {
                    kind,
                    range: range(text, p.get())?,
                });
            }
        }
        Some(c) if c.get() != "null" => {
            return Err(malformed(
                "`content` is neither a string, an array nor null".into(),
            ));
        }
        _ => {}
    }
    for call in message.tool_calls.unwrap_or_default() {
        let args = call.function.arguments.get();
        if !args.starts_with('"') && !args.starts_with('{') {
            return Err(malformed(format!(
                "the arguments of tool call {} are neither a string nor an object",
                call.id
            )));
        }
        let input = range(text, call.function.arguments.get())?;
        blocks.push(BlockView {
            kind: BlockKind::ToolUse {
                id: call.id.into(),
                name: call.function.name.into(),
                input: input.clone(),
            },
            range: input,
        });
    }
    let refused = message.refusal.as_deref().is_some_and(|r| !r.is_empty())
        || choice.finish_reason.as_deref() == Some("content_filter");
    let stop_reason = if refused {
        Some("refusal".into())
    } else {
        choice.finish_reason.clone().map(Into::into)
    };
    let usage = top.usage.unwrap_or_default();
    let cached = usage
        .prompt_tokens_details
        .and_then(|d| d.cached_tokens)
        .unwrap_or(0);
    let view = ResponseView {
        id: top.id.map(Into::into),
        model: top.model.map(Into::into),
        stop_reason,
        stop_details: None,
        usage: Usage {
            input_tokens: usage.prompt_tokens.unwrap_or(0).saturating_sub(cached),
            output_tokens: usage.completion_tokens.unwrap_or(0),
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: cached,
        },
        blocks,
    };
    Ok(ChatAnswer {
        message: range(text, choice.message.get())?,
        view,
        request_id: None,
        raw,
    })
}

/// The assistant message to re-send, from the stored `message` object of an answer: `{"role":"assistant",
/// "content":…}` plus `"tool_calls":…` when `with_calls` and it has calls, each value's bytes as received. Every
/// other member (`annotations`, a provider's reasoning text) stays in the log and is not sent back.
///
/// Any answer [`parse`] accepted gives a message that can be sent: with no calls and no content, the content is the
/// `refusal` string when there is one (a refused step stays in the conversation as what the model said), else `""`.
/// `with_calls` is false when the step's calls were not read as calls (the JSON-object protocol): calls without
/// their `tool` messages would be refused by the server on every later request.
///
/// # Errors
/// `stored` is not a JSON object.
pub fn assistant_message(stored: &RawJson, with_calls: bool) -> Result<RawJson, NativeError> {
    #[derive(Deserialize)]
    struct Stored<'a> {
        #[serde(default, borrow)]
        content: Option<&'a RawValue>,
        #[serde(default, borrow)]
        tool_calls: Option<&'a RawValue>,
        #[serde(default, borrow)]
        refusal: Option<&'a RawValue>,
    }
    if !stored.as_str().starts_with('{') {
        return Err(NativeError::Invalid(
            "a stored assistant message is not a JSON object".into(),
        ));
    }
    let m: Stored<'_> = serde_json::from_str(stored.as_str())
        .map_err(|e| NativeError::Invalid(format!("a stored assistant message: {e}")))?;
    let calls = m
        .tool_calls
        .map(RawValue::get)
        .filter(|c| with_calls && *c != "null" && *c != "[]");
    let content = m.content.map(RawValue::get).filter(|c| *c != "null");
    let refusal = m
        .refusal
        .map(RawValue::get)
        .filter(|r| r.starts_with('"') && *r != r#""""#);
    let content = match (content, &calls) {
        (Some(c), _) => c,
        (None, Some(_)) => "null",
        (None, None) => refusal.unwrap_or(r#""""#),
    };
    let mut out = String::from(r#"{"role":"assistant","content":"#);
    out.push_str(content);
    if let Some(calls) = calls {
        out.push_str(r#","tool_calls":"#);
        out.push_str(calls);
    }
    out.push('}');
    RawJson::from_string(out).map_err(|e| NativeError::Invalid(e.to_string()))
}

/// The provider error a 2xx body carries instead of an answer (`{"error":{"message":…,"code":…}}`, sent by routers
/// whose upstream failed after the status was written), with `error.code` as its status when that is an HTTP error
/// number, else 502; `None` for a body with `choices` or without `error`.
pub(crate) fn error_envelope(body: &[u8]) -> Option<super::error::TranslateError> {
    #[derive(Deserialize)]
    struct Envelope {
        #[serde(default)]
        choices: Option<serde::de::IgnoredAny>,
        error: Option<Inner>,
    }
    #[derive(Deserialize)]
    struct Inner {
        #[serde(default)]
        code: Option<serde_json::Value>,
    }
    let e: Envelope = serde_json::from_slice(body).ok()?;
    if e.choices.is_some() {
        return None;
    }
    let code = e.error?.code;
    let named = |name: &str| {
        let n = name.to_ascii_lowercase();
        if n.contains("rate") {
            429
        } else if ["server", "overload", "unavailable", "timeout", "upstream"]
            .iter()
            .any(|w| n.contains(w))
        {
            502
        } else {
            400
        }
    };
    let status = match &code {
        Some(serde_json::Value::Number(n)) => n
            .as_u64()
            .and_then(|c| u16::try_from(c).ok())
            .filter(|c| (400..600).contains(c))
            .unwrap_or(502),
        Some(serde_json::Value::String(c)) => c
            .parse::<u16>()
            .ok()
            .filter(|c| (400..600).contains(c))
            .unwrap_or_else(|| named(c)),
        _ => 502,
    };
    Some(super::provider::provider_error(status, body))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ANSWER: &str = concat!(
        r#"{"id":"chatcmpl-1","object":"chat.completion","model":"m-1","choices":[{"index":0,"#,
        r#""message":{"role":"assistant","content":"Hi \u00e9","refusal":null,"tool_calls":["#,
        r#"{"id":"call_1","type":"function","function":{"name":"look","arguments":"{\"z\": 1,\"a\":2}"}}]},"#,
        r#""finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":4,"#,
        r#""prompt_tokens_details":{"cached_tokens":6}}}"#
    );

    fn cut(a: &ChatAnswer, r: &Range<u32>) -> String {
        String::from_utf8(a.raw[r.start as usize..r.end as usize].to_vec()).unwrap()
    }

    #[test]
    fn an_answer_is_located_not_changed() {
        let a = parse(Bytes::from_static(ANSWER.as_bytes())).unwrap();
        assert_eq!(&a.raw[..], ANSWER.as_bytes());
        assert!(cut(&a, &a.message).starts_with(r#"{"role":"assistant","content":"Hi \u00e9""#));
        assert_eq!(a.view.id.as_deref(), Some("chatcmpl-1"));
        assert_eq!(a.view.model.as_deref(), Some("m-1"));
        assert_eq!(a.view.stop_reason.as_deref(), Some("tool_calls"));
        assert_eq!(
            a.view.usage,
            Usage {
                input_tokens: 4,
                output_tokens: 4,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 6,
            }
        );
        assert_eq!(a.view.blocks.len(), 2);
        assert_eq!(a.view.blocks[0].kind, BlockKind::Text);
        assert_eq!(cut(&a, &a.view.blocks[0].range), r#""Hi \u00e9""#);
        let (id, name, input) = a.view.tool_uses().next().unwrap();
        assert_eq!((id, name), ("call_1", "look"));
        assert_eq!(cut(&a, &input), r#""{\"z\": 1,\"a\":2}""#);
    }

    #[test]
    fn refusals_and_filters_read_as_refusal() {
        let refused = r#"{"choices":[{"message":{"role":"assistant","content":null,"refusal":"no"},"finish_reason":"stop"}]}"#;
        let filtered = r#"{"choices":[{"message":{"role":"assistant","content":""},"finish_reason":"content_filter"}]}"#;
        for body in [refused, filtered] {
            let a = parse(Bytes::copy_from_slice(body.as_bytes())).unwrap();
            assert_eq!(a.view.stop_reason.as_deref(), Some("refusal"), "{body}");
        }
    }

    #[test]
    fn content_parts_are_blocks() {
        let body = r#"{"choices":[{"message":{"role":"assistant","content":[{"type":"text","text":"a"},{"type":"audio","x":1}]},"finish_reason":"stop"}]}"#;
        let a = parse(Bytes::copy_from_slice(body.as_bytes())).unwrap();
        assert_eq!(a.view.blocks[0].kind, BlockKind::Text);
        assert_eq!(
            cut(&a, &a.view.blocks[0].range),
            r#"{"type":"text","text":"a"}"#
        );
        assert_eq!(
            a.view.blocks[1].kind,
            BlockKind::Opaque {
                block_type: "audio".into()
            }
        );
    }

    #[test]
    fn unreadable_answers_say_why() {
        for (body, why) in [
            (r#"{"choices":[]}"#, "no choices"),
            (r#"{"error":{"message":"x"}}"#, "not a chat completion"),
            (
                r#"{"choices":[{"message":{"content":7}}]}"#,
                "neither a string",
            ),
            (
                r#"{"choices":[{"message":{"content":null,"tool_calls":[{"id":"c","function":{"name":"f","arguments":[1]}}]}}]}"#,
                "neither a string nor an object",
            ),
            (
                r#"{"choices":[{"message":{"content":null,"tool_calls":[{"id":"c","function":{"name":"f"}}]}}]}"#,
                "cannot be read",
            ),
        ] {
            let e = parse(Bytes::copy_from_slice(body.as_bytes())).unwrap_err();
            assert!(e.to_string().contains(why), "{body}: {e}");
        }
    }

    #[test]
    fn the_body_splices_its_parts_in_order() {
        let params = ChatParams::new("m-1", 64, TokenField::MaxCompletionTokens)
            .with_extra(
                &RawJson::from_text(
                    r#"{"temperature":0.0,"response_format":{"type":"json_object"}}"#,
                )
                .unwrap(),
            )
            .unwrap();
        let mut b = ChatBody::new(
            params,
            Some(RawJson::string("be brief")),
            Some(RawJson::from_text(r#"[{"type":"function"}]"#).unwrap()),
        );
        b.push(RawJson::from_text(r#"{"role":"user","content":"hi \u00e9"}"#).unwrap());
        b.push(RawJson::from_text(r#"{"role":"assistant","content":"yo"}"#).unwrap());
        assert_eq!(
            String::from_utf8(b.body()).unwrap(),
            concat!(
                r#"{"model":"m-1","max_completion_tokens":64,"temperature":0.0,"response_format":{"type":"json_object"},"#,
                r#""messages":[{"role":"system","content":"be brief"},{"role":"user","content":"hi \u00e9"},"#,
                r#"{"role":"assistant","content":"yo"}],"tools":[{"type":"function"}]}"#
            )
        );
        let bare = ChatBody::new(ChatParams::new("m", 1, TokenField::MaxTokens), None, None);
        assert_eq!(
            String::from_utf8(bare.body()).unwrap(),
            r#"{"model":"m","max_tokens":1,"messages":[]}"#
        );
    }

    #[test]
    fn sarathi_keys_cannot_be_extra() {
        for key in RESERVED_KEYS {
            let extra = RawJson::from_string(format!(r#"{{"{key}":1}}"#)).unwrap();
            let e = ChatParams::new("m", 1, TokenField::MaxTokens)
                .with_extra(&extra)
                .unwrap_err();
            assert!(e.to_string().contains(key), "{e}");
        }
        let twice = RawJson::from_text(r#"{"a":1,"a":2}"#).unwrap();
        assert!(
            ChatParams::new("m", 1, TokenField::MaxTokens)
                .with_extra(&twice)
                .is_err()
        );
    }

    #[test]
    fn the_assistant_message_keeps_content_and_calls_only() {
        let a = parse(Bytes::from_static(ANSWER.as_bytes())).unwrap();
        let stored = RawJson::from_text(&cut(&a, &a.message)).unwrap();
        assert_eq!(
            assistant_message(&stored, true).unwrap().as_str(),
            concat!(
                r#"{"role":"assistant","content":"Hi \u00e9","tool_calls":[{"id":"call_1","type":"function","#,
                r#""function":{"name":"look","arguments":"{\"z\": 1,\"a\":2}"}}]}"#
            )
        );
        let text =
            RawJson::from_text(r#"{"role":"assistant","content":"x","reasoning_content":"r"}"#)
                .unwrap();
        assert_eq!(
            assistant_message(&text, true).unwrap().as_str(),
            r#"{"role":"assistant","content":"x"}"#
        );
        let empty = RawJson::from_text(r#"{"role":"assistant","content":null}"#).unwrap();
        assert_eq!(
            assistant_message(&empty, true).unwrap().as_str(),
            r#"{"role":"assistant","content":""}"#
        );
        let refused =
            RawJson::from_text(r#"{"role":"assistant","content":null,"refusal":"No \u00e9."}"#)
                .unwrap();
        assert_eq!(
            assistant_message(&refused, true).unwrap().as_str(),
            r#"{"role":"assistant","content":"No \u00e9."}"#
        );
        // Calls the step did not read are not sent back.
        assert_eq!(
            assistant_message(&stored, false).unwrap().as_str(),
            r#"{"role":"assistant","content":"Hi \u00e9"}"#
        );
        assert!(assistant_message(&RawJson::from_text("[1]").unwrap(), true).is_err());
    }

    #[test]
    fn object_arguments_are_the_input() {
        let body = r#"{"choices":[{"message":{"content":null,"tool_calls":[{"id":"c","function":{"name":"f","arguments":{"z" :1}}}]},"finish_reason":"tool_calls"}]}"#;
        let a = parse(Bytes::copy_from_slice(body.as_bytes())).unwrap();
        let (_, _, input) = a.view.tool_uses().next().unwrap();
        assert_eq!(cut(&a, &input), r#"{"z" :1}"#);
    }

    #[test]
    fn an_error_envelope_in_a_success_is_a_provider_error() {
        let e = error_envelope(br#"{"error":{"message":"slow","code":429}}"#).unwrap();
        assert!(e.to_string().contains("slow"), "{e}");
        assert!(matches!(
            e,
            crate::wire::error::TranslateError::Provider {
                status: 429,
                retryable: true,
                ..
            }
        ));
        let provider = |body: &[u8]| match error_envelope(body) {
            Some(crate::wire::error::TranslateError::Provider {
                status, retryable, ..
            }) => (status, retryable),
            other => panic!("{other:?}"),
        };
        assert_eq!(
            provider(br#"{"error":{"message":"x","code":"context_length_exceeded"}}"#),
            (400, false)
        );
        assert_eq!(
            provider(br#"{"error":{"message":"x","code":"server_error"}}"#),
            (502, true)
        );
        assert_eq!(provider(br#"{"error":{"message":"x"}}"#), (502, true));
        assert!(error_envelope(br#"{"choices":[],"error":null}"#).is_none());
        assert!(error_envelope(b"not json").is_none());
    }

    #[test]
    fn token_field_names_round_trip() {
        for f in [TokenField::MaxTokens, TokenField::MaxCompletionTokens] {
            assert_eq!(TokenField::parse(f.as_str()), Some(f));
        }
        assert_eq!(TokenField::parse("x"), None);
    }
}
