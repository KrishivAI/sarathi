// SPDX-License-Identifier: Apache-2.0
//! OpenAI-style chat completions as the driver's model ([`OpenAiCaller`]), and the request assembler it uses.
//!
//! For every model that is not Claude: OpenAI, and the OpenAI-compatible APIs (Groq, Mistral, OpenRouter, Ollama, a
//! gateway). Both step protocols run on it: native tool calls, and one JSON object per step for models without
//! reliable tool calls (with [`step_format`] when the server takes a JSON schema).
//!
//! A request is spliced from the log's stored slices, as for the Messages API ([`crate::anthropic`]), so the same
//! log gives the same bytes and an earlier message goes out each time with the bytes it first had (render once):
//! `system` and `tools` from the session prefix; each assistant message from the `message` object it was received in
//! (its `content` and `tool_calls` values, bytes unchanged; reasoning text and other members are not sent back); each
//! native tool result as a `tool` message; a JSON-object observation, and any message added at that tool boundary
//! (a steer, a correction), as one user message after the results. A user message of one text has a string
//! `content`; a block this model cannot take (an image) is said in words, so the session goes on.
//!
//! Under this protocol a `model.response` entry's `content` range is the whole `message` object, not a `content`
//! array: an OpenAI assistant turn keeps its calls beside its content.

use std::ops::Range;

use bytes::Bytes;
use sarathi_log::{RequestParts, Slice};
use sarathi_protocol::{CallId, ErrorKind, RawJson};
use sarathi_provider::wire::openai_chat::{assistant_message, parse};
use sarathi_provider::{Chain, ChatAnswer, ChatBody, ChatParams, TokenField};
use serde::Deserialize;
use serde_json::value::RawValue;

use crate::driver::{BoxFuture, CallFailure, ModelCaller, Received};
use crate::profile::{ToolDef, step_schema};

/// The protocol's name in the session prefix.
pub const PROTOCOL: &str = "openai-chat";

/// OpenAI-style chat completions through a provider chain (retries, circuit breaker, failover between slots).
///
/// Every slot of the chain serves the same model and names the output limit the same way: the request is hashed
/// before it is sent, so the body must not change between slots.
#[derive(Debug)]
pub struct OpenAiCaller {
    chain: Chain,
    params: RawJson,
}

impl OpenAiCaller {
    /// A caller of the chain's model, answering with at most `max_tokens`.
    ///
    /// # Errors
    /// A slot is not an OpenAI-compatible target, or the chain's slots serve different models or name the output
    /// limit differently (OpenAI itself and another server).
    pub fn new(chain: Chain, max_tokens: u32) -> Result<Self, CallFailure> {
        let model = chain.primary_model().to_owned();
        let Some(first) = chain.slots().first() else {
            return Err(invalid("the chain has no slots"));
        };
        let field = first.client.chat_token_field();
        for slot in chain.slots() {
            if slot.model != model {
                return Err(invalid(format!(
                    "slot {} serves {}, not {model}: one chain serves one model",
                    slot.name, slot.model
                )));
            }
            if !slot.client.speaks_openai_chat() {
                return Err(invalid(format!(
                    "slot {} is not an OpenAI-compatible target: Claude turns go through the Messages API",
                    slot.name
                )));
            }
            if slot.client.chat_token_field() != field {
                return Err(invalid(format!(
                    "slot {} names the output limit `{}`, the first slot `{}`: their bodies would differ",
                    slot.name,
                    slot.client.chat_token_field().as_str(),
                    field.as_str()
                )));
            }
        }
        Ok(OpenAiCaller {
            chain,
            params: params_value(&ChatParams::new(model, max_tokens, field), None),
        })
    }

    /// The same caller sending `extra` request members on every call (`response_format`, `temperature`,
    /// `tool_choice`, `parallel_tool_calls`, …), in their order, bytes unchanged.
    ///
    /// # Errors
    /// `extra` is not an object, repeats a key or names one Sarathi writes.
    pub fn with_extra(mut self, extra: &RawJson) -> Result<Self, CallFailure> {
        let p = chat_params(&self.params).map_err(invalid)?;
        p.clone()
            .with_extra(extra)
            .map_err(|e| invalid(e.to_string()))?;
        self.params = params_value(&p, Some(extra));
        Ok(self)
    }
}

/// The extra member that asks the server for one step object per answer (the JSON-object protocol's
/// [`step_schema`] as `response_format`), for a model whose profile has `structured_output`. Pass it, merged with
/// any other members, to [`OpenAiCaller::with_extra`].
pub fn step_format() -> RawJson {
    let text = format!(
        r#"{{"response_format":{{"type":"json_schema","json_schema":{{"name":"step","schema":{}}}}}}}"#,
        step_schema().as_str()
    );
    // The schema is a JSON object, so the whole is one.
    crate::built_json(text)
}

impl ModelCaller for OpenAiCaller {
    type Prepared = ChatBody;

    fn protocol(&self) -> &str {
        PROTOCOL
    }

    fn params(&self) -> RawJson {
        self.params.clone()
    }

    fn render_tools(&self, tools: &[ToolDef]) -> Result<Option<RawJson>, CallFailure> {
        render_tools(tools)
    }

    fn result_block(&self, call: &CallId, content: &RawJson, is_error: bool) -> RawJson {
        result_block(call, content, is_error)
    }

    fn prepare(&self, parts: &RequestParts<'_>) -> Result<(Vec<u8>, ChatBody), CallFailure> {
        let req = request(parts).map_err(invalid)?;
        Ok((req.body(), req))
    }

    fn send(&self, request: ChatBody) -> BoxFuture<'_, Result<Received, CallFailure>> {
        Box::pin(async move {
            let request = &request;
            match self
                .chain
                .run(|slot, _| async move { slot.client.chat_spliced(request).await })
                .await
            {
                Ok(answered) => received(answered.value),
                Err(failed) => {
                    let retries =
                        u32::try_from(failed.attempts.len().saturating_sub(1)).unwrap_or(u32::MAX);
                    let e = failed.into_error();
                    Err(CallFailure {
                        kind: kind_of(e.kind),
                        detail: e.message.into(),
                        status: e.status,
                        request_id: None,
                        retries,
                    })
                }
            }
        })
    }

    fn read(&self, body: &RawJson, content: Range<u32>) -> Result<Received, CallFailure> {
        let mut r = answer(body)?;
        r.content = content;
        Ok(r)
    }
}

/// A chat-completions answer body (as received, or as stored in the log) read as an answer.
///
/// # Errors
/// The body is not a chat completion.
pub fn answer(body: &RawJson) -> Result<Received, CallFailure> {
    let a = parse(Bytes::copy_from_slice(body.as_str().as_bytes()))
        .map_err(|e| CallFailure::new(ErrorKind::Malformed, e.to_string()))?;
    Ok(Received::new(body.clone(), a.message, a.view))
}

/// The request `parts` describe, for chat completions.
///
/// # Errors
/// The parts are of another protocol, or a stored value is not what its kind holds.
pub fn request(parts: &RequestParts<'_>) -> Result<ChatBody, String> {
    if parts.protocol != PROTOCOL {
        return Err(format!(
            "the session's protocol is {}, not {PROTOCOL}",
            parts.protocol
        ));
    }
    let present = |v: &RawJson| (v.as_str() != "null").then(|| v.clone());
    let mut req = ChatBody::new(
        chat_params(parts.params)?,
        present(parts.system),
        present(parts.tools),
    );
    // What follows an assistant message's calls: its `tool` messages, then one user message of the texts that
    // joined at the tool boundary (JSON-object observations, steers, corrections).
    let mut boundary: Option<(Vec<RawJson>, Vec<String>)> = None;
    let flush = |req: &mut ChatBody, boundary: &mut Option<(Vec<RawJson>, Vec<String>)>| {
        if let Some((tools, texts)) = boundary.take() {
            for t in tools {
                req.push(t);
            }
            if !texts.is_empty() {
                req.push(user_of(&texts)?);
            }
        }
        Ok::<(), String>(())
    };
    for (_, slice) in &parts.messages {
        match slice {
            Slice::ToolResult { block, .. } => {
                let (tools, texts) = boundary.get_or_insert_with(Default::default);
                if is_tool_message(block) {
                    tools.push((*block).clone());
                } else {
                    texts.push(block_text(block)?);
                }
            }
            Slice::User { content } => match &mut boundary {
                Some((_, texts)) => texts.extend(user_texts(content)?),
                None => req.push(user_of(&user_texts(content)?)?),
            },
            Slice::Assistant { content, calls } => {
                flush(&mut req, &mut boundary)?;
                // Calls go back only when the request offers tools (the native protocol); under the JSON-object
                // protocol a step's call is read from its text, and stray `tool_calls` would have no `tool` message.
                let with_calls = !calls.is_empty() && parts.tools.as_str() != "null";
                req.push(assistant_message(content, with_calls).map_err(|e| e.to_string())?);
            }
        }
    }
    flush(&mut req, &mut boundary)?;
    Ok(req)
}

/// Whether a stored tool-result block is a whole `tool` message (native calls) rather than a text block.
fn is_tool_message(block: &RawJson) -> bool {
    #[derive(Deserialize)]
    struct Head<'a> {
        #[serde(default, borrow)]
        role: Option<std::borrow::Cow<'a, str>>,
    }
    serde_json::from_str::<Head<'_>>(block.as_str())
        .ok()
        .and_then(|h| h.role)
        .is_some_and(|r| r == "tool")
}

/// A user message of `texts` (each a JSON string's bytes): one text as a string `content`, the form every
/// OpenAI-compatible server takes; several as an array of text parts.
fn user_of(texts: &[String]) -> Result<RawJson, String> {
    match texts {
        [one] => json(format!(r#"{{"role":"user","content":{one}}}"#)),
        _ => {
            let parts: Vec<String> = texts
                .iter()
                .map(|t| format!(r#"{{"type":"text","text":{t}}}"#))
                .collect();
            json(format!(
                r#"{{"role":"user","content":[{}]}}"#,
                parts.join(",")
            ))
        }
    }
}

/// A user message's content as texts (JSON strings' bytes, unchanged): a string is one, an array gives one per
/// block.
fn user_texts(content: &RawJson) -> Result<Vec<String>, String> {
    let text = content.as_str();
    if text.starts_with('"') {
        return Ok(vec![text.to_owned()]);
    }
    let blocks: Vec<&RawValue> = serde_json::from_str(text).map_err(|e| e.to_string())?;
    blocks
        .into_iter()
        .map(|b| block_text(&RawJson::from_text(b.get()).map_err(|e| e.to_string())?))
        .collect()
}

/// A content block's text for this model: a text block's `text` bytes unchanged (any other member, such as
/// `cache_control` or `citations`, left out); any other block said in words, as a tool result's is. The text is
/// a pure function of the stored block, so a logged request still rebuilds, and the session goes on.
///
/// # Errors
/// The block is not an object with a `type`.
fn block_text(block: &RawJson) -> Result<String, String> {
    #[derive(Deserialize)]
    struct Block<'a> {
        #[serde(rename = "type")]
        kind: std::borrow::Cow<'a, str>,
        #[serde(default, borrow)]
        text: Option<&'a RawValue>,
    }
    let b: Block<'_> = serde_json::from_str(block.as_str())
        .map_err(|e| format!("a content block cannot be read: {e}"))?;
    Ok(match (b.kind.as_ref(), b.text) {
        ("text", Some(t)) if t.get().starts_with('"') => t.get().to_owned(),
        (kind, _) => RawJson::string(&not_shown(kind)).as_str().to_owned(),
    })
}

/// What this model reads in place of a block it cannot take.
fn not_shown(kind: &str) -> String {
    format!("[a {kind} block, not shown to this model]")
}

/// The `tools` value for `tools` (`function` tools, each schema's bytes as given), or `None` for no tools.
///
/// # Errors
/// The rendered list is not JSON (a tool's schema is not valid JSON).
pub fn render_tools(tools: &[ToolDef]) -> Result<Option<RawJson>, CallFailure> {
    if tools.is_empty() {
        return Ok(None);
    }
    let items: Vec<String> = tools
        .iter()
        .map(|t| {
            format!(
                r#"{{"type":"function","function":{{"name":{},"description":{},"parameters":{}}}}}"#,
                RawJson::string(&t.name).as_str(),
                RawJson::string(&t.description).as_str(),
                t.input_schema.as_str()
            )
        })
        .collect();
    RawJson::from_string(format!("[{}]", items.join(",")))
        .map(Some)
        .map_err(|e| invalid(format!("the tools cannot be rendered: {e}")))
}

/// A `tool` message answering `call`. Its content is text: a string result as it is; the text blocks of an array
/// result joined by newlines, any other block said in words; any other value as its JSON. A failed call's text
/// starts `error: `, since the format has no error flag.
pub fn result_block(call: &CallId, content: &RawJson, is_error: bool) -> RawJson {
    let mut text = result_text(content);
    if is_error {
        text.insert_str(0, "error: ");
    }
    let msg = format!(
        r#"{{"role":"tool","tool_call_id":{},"content":{}}}"#,
        RawJson::string(call.as_str()).as_str(),
        RawJson::string(&text).as_str()
    );
    // Both parts are JSON strings, so the message is one.
    crate::built_json(msg)
}

fn result_text(content: &RawJson) -> String {
    #[derive(Deserialize)]
    struct Block {
        #[serde(rename = "type", default)]
        kind: Option<String>,
        #[serde(default)]
        text: Option<String>,
    }
    let raw = content.as_str();
    if raw.starts_with('"') {
        return serde_json::from_str(raw).unwrap_or_else(|_| raw.to_owned());
    }
    if raw.starts_with('[')
        && let Ok(blocks) = serde_json::from_str::<Vec<Block>>(raw)
    {
        return blocks
            .into_iter()
            .map(|b| match (b.kind.as_deref(), b.text) {
                (Some("text"), Some(t)) => t,
                (kind, _) => format!(
                    "[a {} block, not shown to this model]",
                    kind.unwrap_or("typeless")
                ),
            })
            .collect::<Vec<_>>()
            .join("\n");
    }
    raw.to_owned()
}

/// The recipe's parameters: `{"model":…,"max_tokens":…,"token_field":…,"extra":{…}}`, `extra` as given.
fn params_value(p: &ChatParams, extra: Option<&RawJson>) -> RawJson {
    let extra = extra.map_or(String::new(), |e| format!(r#","extra":{}"#, e.as_str()));
    let text = format!(
        r#"{{"model":{},"max_tokens":{},"token_field":"{}"{extra}}}"#,
        RawJson::string(p.model()).as_str(),
        p.max_tokens(),
        p.field().as_str()
    );
    crate::built_json(text)
}

fn chat_params(params: &RawJson) -> Result<ChatParams, String> {
    #[derive(Deserialize)]
    struct Params<'a> {
        model: String,
        max_tokens: u32,
        token_field: String,
        #[serde(borrow)]
        extra: Option<&'a RawValue>,
    }
    let p: Params<'_> = serde_json::from_str(params.as_str())
        .map_err(|e| format!("the call parameters cannot be read: {e}"))?;
    let field = TokenField::parse(&p.token_field)
        .ok_or_else(|| format!("`{}` is not an output limit's name", p.token_field))?;
    let base = ChatParams::new(p.model, p.max_tokens, field);
    match p.extra {
        None => Ok(base),
        Some(extra) => {
            let extra = RawJson::from_text(extra.get()).map_err(|e| e.to_string())?;
            base.with_extra(&extra).map_err(|e| e.to_string())
        }
    }
}

/// The answer as the log stores it. Its ranges are measured again on the stored text, as [`answer`] does on a
/// resume: the stored body has no outer whitespace, which a server may have sent.
fn received(a: ChatAnswer) -> Result<Received, CallFailure> {
    let body = RawJson::from_slice(&a.raw)
        .map_err(|e| CallFailure::new(ErrorKind::Malformed, e.to_string()))?;
    let mut out = answer(&body)?;
    out.request_id = a.request_id.map(Into::into);
    Ok(out)
}

fn json(text: String) -> Result<RawJson, String> {
    RawJson::from_string(text).map_err(|e| e.to_string())
}

fn kind_of(kind: sarathi_provider::ErrorKind) -> ErrorKind {
    serde_json::from_value(serde_json::Value::String(kind.as_str().to_owned()))
        .unwrap_or(ErrorKind::Unknown)
}

fn invalid(detail: impl Into<Box<str>>) -> CallFailure {
    CallFailure::new(ErrorKind::InvalidRequest, detail)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sarathi_protocol::Seq;

    fn raw(s: &str) -> RawJson {
        RawJson::from_text(s).unwrap()
    }

    const PARAMS: &str =
        r#"{"model":"m-1","max_tokens":64,"token_field":"max_tokens","extra":{"temperature":0}}"#;

    fn parts<'a>(
        system: &'a RawJson,
        tools: &'a RawJson,
        params: &'a RawJson,
        messages: Vec<Slice<'a>>,
    ) -> RequestParts<'a> {
        RequestParts {
            protocol: PROTOCOL,
            system,
            tools,
            messages: messages
                .into_iter()
                .enumerate()
                .map(|(i, s)| (Seq(u64::try_from(i).unwrap() + 1), s))
                .collect(),
            params,
        }
    }

    #[test]
    fn a_native_conversation_is_spliced_with_tool_messages_then_the_boundary() {
        let system = RawJson::string("sys");
        let tools = render_tools(&[ToolDef::new(
            "look",
            "reads",
            raw(r#"{"type":"object","properties":{"z":{},"a":{}}}"#),
            crate::Effect::Read,
        )])
        .unwrap()
        .unwrap();
        let params = raw(PARAMS);
        let user = raw(r#""hi \u00e9""#);
        let stored = raw(
            r#"{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"look","arguments":"{}"}}],"refusal":null}"#,
        );
        let result = result_block(&CallId::from("call_1"), &raw(r#""seen""#), false);
        let calls = [sarathi_log::CallRef {
            id: "call_1".into(),
            name: "look".into(),
        }];
        let steer =
            raw(r#"[{"type":"text","text":"also b","cache_control":{"type":"ephemeral"}}]"#);
        let p = parts(
            &system,
            &tools,
            &params,
            vec![
                Slice::User { content: &user },
                Slice::Assistant {
                    content: stored,
                    calls: &calls,
                },
                Slice::ToolResult {
                    call: "call_1",
                    block: &result,
                },
                Slice::User { content: &steer },
            ],
        );
        let body = String::from_utf8(request(&p).unwrap().body()).unwrap();
        assert_eq!(
            body,
            concat!(
                r#"{"model":"m-1","max_tokens":64,"temperature":0,"messages":[{"role":"system","content":"sys"},"#,
                r#"{"role":"user","content":"hi \u00e9"},"#,
                r#"{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"look","arguments":"{}"}}]},"#,
                r#"{"role":"tool","tool_call_id":"call_1","content":"seen"},"#,
                r#"{"role":"user","content":"also b"}],"#,
                r#""tools":[{"type":"function","function":{"name":"look","description":"reads","parameters":{"type":"object","properties":{"z":{},"a":{}}}}}]}"#
            )
        );
        // The same parts give the same bytes.
        assert_eq!(request(&p).unwrap().body(), body.into_bytes());
    }

    #[test]
    fn a_json_object_observation_is_a_user_message() {
        let none = RawJson::null();
        let system = RawJson::string("sys");
        let params = raw(PARAMS);
        let user = raw(r#""go""#);
        let stored = raw(r#"{"role":"assistant","content":"{\"tool\":\"look\"}"}"#);
        let obs = raw(r#"{"type":"text","text":"observation:\nok"}"#);
        let p = parts(
            &system,
            &none,
            &params,
            vec![
                Slice::User { content: &user },
                Slice::Assistant {
                    content: stored,
                    calls: &[],
                },
                Slice::ToolResult {
                    call: "s2-0",
                    block: &obs,
                },
            ],
        );
        let body = String::from_utf8(request(&p).unwrap().body()).unwrap();
        assert!(
            body.ends_with(concat!(
                r#"{"role":"assistant","content":"{\"tool\":\"look\"}"},"#,
                r#"{"role":"user","content":"observation:\nok"}]}"#
            )),
            "{body}"
        );
        assert!(!body.contains(r#""tools""#), "{body}");
    }

    #[test]
    fn what_cannot_be_sent_is_said_or_refused() {
        let none = RawJson::null();
        let system = RawJson::string("sys");
        let params = raw(PARAMS);
        let image = raw(r#"[{"type":"image","source":{}}]"#);
        let p = parts(
            &system,
            &none,
            &params,
            vec![Slice::User { content: &image }],
        );
        let body = String::from_utf8(request(&p).unwrap().body()).unwrap();
        assert!(
            body.contains(
                r#"{"role":"user","content":"[a image block, not shown to this model]"}"#
            ),
            "{body}"
        );
        let mut p = parts(&system, &none, &params, vec![]);
        p.protocol = "anthropic-messages";
        assert!(request(&p).unwrap_err().contains("not openai-chat"));
        let bad = raw(r#"{"model":"m","max_tokens":1,"token_field":"tokens"}"#);
        let p = parts(&system, &none, &bad, vec![]);
        assert!(request(&p).unwrap_err().contains("output limit"));
    }

    #[test]
    fn tool_results_are_text() {
        let id = CallId::from("call_9");
        assert_eq!(
            result_block(&id, &raw(r#""no such file""#), true).as_str(),
            r#"{"role":"tool","tool_call_id":"call_9","content":"error: no such file"}"#
        );
        assert_eq!(
            result_block(
                &id,
                &raw(r#"[{"type":"text","text":"a"},{"type":"image"},{"type":"text","text":"b"}]"#),
                false
            )
            .as_str(),
            r#"{"role":"tool","tool_call_id":"call_9","content":"a\n[a image block, not shown to this model]\nb"}"#
        );
        assert_eq!(
            result_block(&id, &raw(r#"{"n":1}"#), false).as_str(),
            r#"{"role":"tool","tool_call_id":"call_9","content":"{\"n\":1}"}"#
        );
    }

    #[test]
    fn the_step_format_carries_the_step_schema() {
        let f = step_format();
        assert!(
            f.as_str()
                .starts_with(r#"{"response_format":{"type":"json_schema""#)
        );
        assert!(f.as_str().contains(step_schema().as_str()));
    }

    #[test]
    fn a_stored_answer_reads_back() {
        let body = raw(
            r#"{"choices":[{"message":{"role":"assistant","content":"x"},"finish_reason":"stop"}]}"#,
        );
        let r = answer(&body).unwrap();
        assert_eq!(r.body, body);
        let m = &body.as_str()[r.content.start as usize..r.content.end as usize];
        assert_eq!(m, r#"{"role":"assistant","content":"x"}"#);
        assert_eq!(
            crate::step::native_text(&r.body, &r.view).as_deref(),
            Some("x")
        );
        assert!(answer(&raw(r#"{"x":1}"#)).is_err());
    }
}
