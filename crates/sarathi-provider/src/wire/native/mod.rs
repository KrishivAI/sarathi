// SPDX-License-Identifier: Apache-2.0
//! The Anthropic-native Messages call, without IO: render once, splice forever.
//!
//! A request body is written from byte slices that were each made exactly once: the session's `system` and `tools`
//! ([`SessionPrefix`]), one [`MessageSlice`] per message, then the per-call parameters ([`CallParams`]). Nothing here
//! holds the conversation as a value it re-renders, so every earlier message goes out with the bytes it had the
//! first time, which is what the API's history check and the prompt cache compare. [`response::parse`] reads a body
//! into the shallow [`ResponseView`](sarathi_protocol::ResponseView) without changing it; [`stream::Assembler`] builds the message from SSE events.

pub mod response;
pub mod stream;

use std::collections::BTreeMap;

use sarathi_protocol::{MessageSlice, RawJson};

pub use response::{MessagesResponse, RateLimitHeaders};
pub use stream::Assembler;

/// The `anthropic-version` sent unless the caller names another.
pub const API_VERSION: &str = "2023-06-01";

/// Request keys Sarathi writes itself; [`CallParams::with_extra`] refuses them.
pub const RESERVED_KEYS: [&str; 6] = [
    "model",
    "max_tokens",
    "system",
    "tools",
    "messages",
    "stream",
];

/// The extra members a `count_tokens` body keeps; the endpoint refuses generation-only ones (`metadata`,
/// `temperature`, …). Not yet confirmed against the live API.
pub const COUNT_TOKENS_KEYS: [&str; 5] = [
    "tool_choice",
    "thinking",
    "mcp_servers",
    "context_management",
    "output_config",
];

/// A failure in the native path: a request that cannot be built, or an answer that cannot be read.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum NativeError {
    /// The request cannot be built as given.
    #[error("invalid request: {0}")]
    Invalid(String),
    /// The answer is not a Messages API answer Sarathi can read.
    #[error("malformed response: {0}")]
    Malformed(String),
    /// The stream ended before `message_stop`: the turn was cut, and making the call again may succeed.
    #[error("cut stream: {0}")]
    Cut(String),
    /// A stream sent a delta type the assembler does not know. Guessing a merge would write bytes the API never
    /// sent into an immutable prefix, so the turn fails.
    #[error("cannot assemble a `{delta_type}` delta into a `{block_type}` block")]
    UnassemblableDelta {
        /// The block's `type`.
        block_type: String,
        /// The delta's `type`.
        delta_type: String,
    },
    /// The provider sent an `error` event in the stream.
    #[error("provider error {status} ({error_type}): {message}")]
    Provider {
        /// The HTTP status the error type stands for (`overloaded_error` is 529).
        status: u16,
        /// The error's `type`.
        error_type: String,
        /// The provider's message.
        message: String,
    },
}

/// `system` and `tools` as rendered at session start: the same bytes on every call of the session (design rule 1).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionPrefix {
    system: Option<RawJson>,
    tools: Option<RawJson>,
}

impl SessionPrefix {
    /// A prefix of `system` (a string or an array of text blocks) and `tools` (an array), either optional.
    pub fn new(system: Option<RawJson>, tools: Option<RawJson>) -> Self {
        SessionPrefix { system, tools }
    }

    /// The `system` bytes.
    pub fn system(&self) -> Option<&RawJson> {
        self.system.as_ref()
    }

    /// The `tools` bytes.
    pub fn tools(&self) -> Option<&RawJson> {
        self.tools.as_ref()
    }
}

/// The parameters of one call: everything outside `system`, `tools` and `messages`. These may change between calls
/// without breaking the history check. Sarathi owns `stream`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallParams {
    model: String,
    max_tokens: u32,
    /// The members of the caller's object, in order, values exactly as given.
    extra: Vec<(String, RawJson)>,
}

impl CallParams {
    /// `model` and `max_tokens`, nothing else.
    pub fn new(model: impl Into<String>, max_tokens: u32) -> Self {
        CallParams {
            model: model.into(),
            max_tokens,
            extra: Vec::new(),
        }
    }

    /// Adds the members of `extra`, a JSON object (`thinking`, `output_config`, `tool_choice`, `metadata`,
    /// `context_management`, …), written into the body in their order with their values' bytes unchanged. A
    /// `count_tokens` body takes only the members named in [`COUNT_TOKENS_KEYS`].
    ///
    /// # Errors
    /// `extra` is not an object, repeats a key, or names a key in [`RESERVED_KEYS`].
    pub fn with_extra(mut self, extra: &RawJson) -> Result<Self, NativeError> {
        let members = members(extra)?;
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
}

/// The members of a JSON object, refusing duplicates.
pub(crate) fn members(value: &RawJson) -> Result<Vec<(String, RawJson)>, NativeError> {
    let entries = response::entries(value.as_str())
        .map_err(|_| NativeError::Invalid("the extra parameters must be a JSON object".into()))?;
    let mut seen = BTreeMap::new();
    for (k, _) in &entries {
        if seen.insert(k.clone(), ()).is_some() {
            return Err(NativeError::Invalid(format!("`{k}` is given twice")));
        }
    }
    Ok(entries
        .into_iter()
        .map(|(k, v)| (k, RawJson::from(v)))
        .collect())
}

/// Which endpoint a body is for. `count_tokens` takes neither `max_tokens` nor `stream`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endpoint {
    /// `POST /v1/messages`, answered whole.
    Messages,
    /// `POST /v1/messages`, answered as SSE.
    MessagesStream,
    /// `POST /v1/messages/count_tokens`.
    CountTokens,
}

impl Endpoint {
    /// The path under the API's base URL.
    pub fn path(self) -> &'static str {
        match self {
            Endpoint::Messages | Endpoint::MessagesStream => "/v1/messages",
            Endpoint::CountTokens => "/v1/messages/count_tokens",
        }
    }
}

/// One Messages API call, as slices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessagesRequest {
    /// `system` and `tools`, fixed for the session.
    pub prefix: SessionPrefix,
    /// One slice per message, in order. Appended to, never edited.
    pub messages: Vec<MessageSlice>,
    /// The per-call parameters.
    pub params: CallParams,
    /// `anthropic-beta` values, sent comma-joined in one header.
    pub betas: Vec<String>,
    /// `anthropic-version`; [`API_VERSION`] when `None`.
    pub version: Option<String>,
}

impl MessagesRequest {
    /// A request with no messages, betas or version override.
    pub fn new(prefix: SessionPrefix, params: CallParams) -> Self {
        MessagesRequest {
            prefix,
            messages: Vec::new(),
            params,
            betas: Vec::new(),
            version: None,
        }
    }

    /// Appends a message slice.
    pub fn push(&mut self, slice: MessageSlice) {
        self.messages.push(slice);
    }

    /// The body for `endpoint`, written from the slices: no `Value`, no `Serialize` of history.
    pub fn body(&self, endpoint: Endpoint) -> Vec<u8> {
        let len = self
            .messages
            .iter()
            .map(|m| m.bytes.len() + 1)
            .sum::<usize>()
            + self.prefix.system.as_ref().map_or(0, RawJson::len)
            + self.prefix.tools.as_ref().map_or(0, RawJson::len)
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
        if endpoint != Endpoint::CountTokens {
            out.push_str(r#","max_tokens":"#);
            out.push_str(&self.params.max_tokens.to_string());
        }
        if let Some(system) = &self.prefix.system {
            out.push_str(r#","system":"#);
            out.push_str(system.as_str());
        }
        if let Some(tools) = &self.prefix.tools {
            out.push_str(r#","tools":"#);
            out.push_str(tools.as_str());
        }
        out.push_str(r#","messages":["#);
        for (i, m) in self.messages.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(m.bytes.as_str());
        }
        out.push(']');
        for (k, v) in &self.params.extra {
            if endpoint == Endpoint::CountTokens && !COUNT_TOKENS_KEYS.contains(&k.as_str()) {
                continue;
            }
            out.push(',');
            out.push_str(RawJson::string(k).as_str());
            out.push(':');
            out.push_str(v.as_str());
        }
        if endpoint == Endpoint::MessagesStream {
            out.push_str(r#","stream":true"#);
        }
        out.push('}');
        out.into_bytes()
    }

    /// The headers this request sets besides the credential: `anthropic-version`, `anthropic-beta` (when any) and
    /// `content-type`.
    pub fn headers(&self) -> Vec<(String, String)> {
        let mut h = vec![
            ("content-type".to_string(), "application/json".to_string()),
            (
                "anthropic-version".to_string(),
                self.version
                    .clone()
                    .unwrap_or_else(|| API_VERSION.to_string()),
            ),
        ];
        if !self.betas.is_empty() {
            h.push(("anthropic-beta".to_string(), self.betas.join(",")));
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sarathi_protocol::{Role, Seq};

    fn raw(s: &str) -> RawJson {
        RawJson::from_text(s).unwrap()
    }

    fn request() -> MessagesRequest {
        let prefix = SessionPrefix::new(
            Some(raw(r#"[{"type":"text","text":"You are terse.\/"}]"#)),
            Some(raw(
                r#"[{"name":"edit","input_schema":{"type":"object","z":1,"a":2}}]"#,
            )),
        );
        let params = CallParams::new("claude-opus-5-5", 1024)
            .with_extra(&raw(
                r#"{ "thinking":{"type":"adaptive"}, "metadata":{"user_id":"u\u00e9"} }"#,
            ))
            .unwrap();
        let mut r = MessagesRequest::new(prefix, params);
        r.push(MessageSlice::user_text(Seq(2), "hi").unwrap());
        r.push(
            MessageSlice::new(
                Seq(5),
                Role::Assistant,
                &raw(r#"[{"type":"text","text":"yo"}]"#),
            )
            .unwrap(),
        );
        r
    }

    #[test]
    fn the_body_is_the_slices_spliced() {
        let body = String::from_utf8(request().body(Endpoint::Messages)).unwrap();
        assert_eq!(
            body,
            concat!(
                r#"{"model":"claude-opus-5-5","max_tokens":1024,"#,
                r#""system":[{"type":"text","text":"You are terse.\/"}],"#,
                r#""tools":[{"name":"edit","input_schema":{"type":"object","z":1,"a":2}}],"#,
                r#""messages":[{"role":"user","content":"hi"},{"role":"assistant","content":[{"type":"text","text":"yo"}]}],"#,
                r#""thinking":{"type":"adaptive"},"metadata":{"user_id":"u\u00e9"}}"#
            )
        );
        let _: serde_json::Value = serde_json::from_str(&body).unwrap();
    }

    #[test]
    fn stream_and_count_tokens_bodies() {
        let r = request();
        let s = String::from_utf8(r.body(Endpoint::MessagesStream)).unwrap();
        assert!(s.ends_with(r#","stream":true}"#), "{s}");
        let c = String::from_utf8(r.body(Endpoint::CountTokens)).unwrap();
        assert!(!c.contains("max_tokens") && !c.contains("stream"), "{c}");
        assert!(
            c.contains(r#""thinking":"#) && !c.contains("metadata"),
            "{c}"
        );
        assert_eq!(Endpoint::CountTokens.path(), "/v1/messages/count_tokens");
        assert_eq!(Endpoint::MessagesStream.path(), "/v1/messages");
    }

    #[test]
    fn an_empty_request_is_valid_json() {
        let r = MessagesRequest::new(SessionPrefix::default(), CallParams::new("m", 1));
        assert_eq!(
            String::from_utf8(r.body(Endpoint::Messages)).unwrap(),
            r#"{"model":"m","max_tokens":1,"messages":[]}"#
        );
        let r = MessagesRequest::new(
            SessionPrefix::default(),
            CallParams::new("m", 1).with_extra(&raw("{ }")).unwrap(),
        );
        assert_eq!(
            String::from_utf8(r.body(Endpoint::Messages)).unwrap(),
            r#"{"model":"m","max_tokens":1,"messages":[]}"#
        );
    }

    #[test]
    fn extra_refuses_reserved_repeated_and_non_objects() {
        let p = || CallParams::new("m", 1);
        assert!(matches!(
            p().with_extra(&raw(r#"{"stream":false}"#)),
            Err(NativeError::Invalid(_))
        ));
        assert!(matches!(
            p().with_extra(&raw(r#"{"system":"x"}"#)),
            Err(NativeError::Invalid(_))
        ));
        assert!(matches!(
            p().with_extra(&raw(r#"{"a":1,"a":2}"#)),
            Err(NativeError::Invalid(_))
        ));
        assert!(matches!(
            p().with_extra(&raw("[1]")),
            Err(NativeError::Invalid(_))
        ));
        assert_eq!(p().model(), "m");
        assert_eq!(p().max_tokens(), 1);
    }

    #[test]
    fn headers_default_the_version_and_join_betas() {
        let mut r = request();
        assert_eq!(
            r.headers(),
            vec![
                ("content-type".into(), "application/json".into()),
                ("anthropic-version".into(), API_VERSION.into()),
            ]
        );
        r.betas = vec!["a-1".into(), "b-2".into()];
        r.version = Some("2099-01-01".into());
        let h = r.headers();
        assert!(h.contains(&("anthropic-beta".into(), "a-1,b-2".into())));
        assert!(h.contains(&("anthropic-version".into(), "2099-01-01".into())));
        assert!(r.prefix.system().unwrap().as_str().ends_with(r#"\/"}]"#));
        assert!(r.prefix.tools().is_some());
    }
}
