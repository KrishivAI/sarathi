// SPDX-License-Identifier: Apache-2.0
//! The Anthropic Messages API as the driver's model ([`AnthropicCaller`]), and the request assembler it uses.
//!
//! A request is spliced from the log's stored slices ([`request`]): `system` and `tools` from the session prefix,
//! each assistant message as the `content` bytes it was received with (thinking blocks, signatures and unknown
//! blocks included), and each run of tool results as one user message, with any message added at that tool
//! boundary (a steer, a correction) after the results in the same message. The same log gives the same bytes, so a
//! logged request is checked by building it again.

use std::ops::Range;

use bytes::Bytes;
use sarathi_context::COMPACTION_BETA;
use sarathi_log::{RequestParts, Slice};
use sarathi_protocol::{CallId, ErrorKind, MessageSlice, RawJson, Role, Seq};
use sarathi_provider::wire::native::response::parse;
use sarathi_provider::{
    CallParams, Chain, Endpoint, MessagesRequest, MessagesResponse, SessionPrefix,
};
use serde_json::value::RawValue;

use crate::driver::{BoxFuture, CallFailure, ModelCaller, Received};
use crate::profile::ToolDef;

/// The protocol's name in the session prefix.
pub const PROTOCOL: &str = "anthropic-messages";

/// The Anthropic Messages API through a provider chain (retries, circuit breaker, failover between slots).
///
/// Every slot of the chain serves the same model: the request is hashed before it is sent, so the body must not
/// change between slots. Failover to another model is a protocol switch, which starts a new segment.
#[derive(Debug)]
pub struct AnthropicCaller {
    chain: Chain,
    params: RawJson,
    betas: Vec<String>,
}

impl AnthropicCaller {
    /// A caller of the chain's model, answering with at most `max_tokens`.
    ///
    /// # Errors
    /// The chain's slots serve different models.
    pub fn new(chain: Chain, max_tokens: u32) -> Result<Self, CallFailure> {
        let model = chain.primary_model().to_owned();
        if let Some(other) = chain.slots().iter().find(|s| s.model != model) {
            return Err(CallFailure::new(
                ErrorKind::InvalidRequest,
                format!(
                    "slot {} serves {}, not {model}: one chain serves one model",
                    other.name, other.model
                ),
            ));
        }
        Ok(AnthropicCaller {
            chain,
            params: params_value(&model, max_tokens, None),
            betas: Vec::new(),
        })
    }

    /// The same caller sending `extra` request members on every call (`thinking`, `output_config`, …), in their
    /// order, bytes unchanged.
    ///
    /// # Errors
    /// `extra` is not an object, repeats a key or names one Sarathi writes.
    pub fn with_extra(mut self, extra: &RawJson) -> Result<Self, CallFailure> {
        let p = call_params(&self.params).map_err(invalid)?;
        p.clone()
            .with_extra(extra)
            .map_err(|e| invalid(e.to_string()))?;
        self.params = params_value(p.model(), p.max_tokens(), Some(extra));
        Ok(self)
    }

    /// The same caller sending these `anthropic-beta` values.
    #[must_use]
    pub fn with_betas(mut self, betas: Vec<String>) -> Self {
        self.betas = betas;
        self
    }
}

impl ModelCaller for AnthropicCaller {
    type Prepared = MessagesRequest;

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

    fn prepare(&self, parts: &RequestParts<'_>) -> Result<(Vec<u8>, MessagesRequest), CallFailure> {
        let mut req = request(parts).map_err(invalid)?;
        req.betas.clone_from(&self.betas);
        if needs_compaction_beta(parts) && !req.betas.iter().any(|b| b == COMPACTION_BETA) {
            req.betas.push(COMPACTION_BETA.to_owned());
        }
        Ok((req.body(Endpoint::Messages), req))
    }

    fn send(&self, request: MessagesRequest) -> BoxFuture<'_, Result<Received, CallFailure>> {
        Box::pin(async move {
            let request = &request;
            match self
                .chain
                .run(|slot, _| async move { slot.client.messages(request).await })
                .await
            {
                Ok(answered) => received(answered.value),
                Err(failed) => Err(chain_failure(failed)),
            }
        })
    }

    /// `POST /v1/messages/count_tokens` through the chain, with the request [`ModelCaller::prepare`] builds (the same
    /// betas, so the same history check). A failure goes back to the driver, which uses its estimate instead.
    fn count_tokens(
        &self,
        parts: &RequestParts<'_>,
    ) -> Option<BoxFuture<'_, Result<u64, CallFailure>>> {
        let prepared = self.prepare(parts);
        Some(Box::pin(async move {
            let (_, request) = prepared?;
            let request = &request;
            match self
                .chain
                .run(|slot, _| async move { slot.client.count_tokens(request).await })
                .await
            {
                Ok(answered) => Ok(answered.value.input_tokens),
                Err(failed) => Err(chain_failure(failed)),
            }
        }))
    }

    fn read(&self, body: &RawJson, content: Range<u32>) -> Result<Received, CallFailure> {
        let mut r = answer(body)?;
        r.content = content;
        Ok(r)
    }
}

/// A call the chain gave up on, as the driver's failure: the last attempt's error, and the retries before it.
fn chain_failure(failed: sarathi_provider::resilience::Failed) -> CallFailure {
    let retries = u32::try_from(failed.attempts.len().saturating_sub(1)).unwrap_or(u32::MAX);
    let e = failed.into_error();
    CallFailure {
        kind: kind_of(e.kind),
        detail: e.message.into(),
        status: e.status,
        request_id: None,
        retries,
    }
}

/// A Messages API answer body (as received, or as stored in the log) read as an answer.
///
/// # Errors
/// The body is not a Messages answer.
pub fn answer(body: &RawJson) -> Result<Received, CallFailure> {
    let r = parse(Bytes::copy_from_slice(body.as_str().as_bytes()))
        .map_err(|e| CallFailure::new(ErrorKind::Malformed, e.to_string()))?;
    Ok(Received::new(body.clone(), r.content, r.view))
}

/// The request `parts` describe, for the Messages API.
///
/// # Errors
/// The parts are of another protocol, or a stored value is not what its kind holds.
pub fn request(parts: &RequestParts<'_>) -> Result<MessagesRequest, String> {
    if parts.protocol != PROTOCOL {
        return Err(format!(
            "the session's protocol is {}, not {PROTOCOL}",
            parts.protocol
        ));
    }
    let present = |v: &RawJson| (v.as_str() != "null").then(|| v.clone());
    let prefix = SessionPrefix::new(present(parts.system), present(parts.tools));
    let mut req = MessagesRequest::new(prefix, call_params(parts.params)?);
    // A run of tool results, and what joins it at the tool boundary: one user message.
    let mut group: Option<(Seq, Vec<String>)> = None;
    let flush = |req: &mut MessagesRequest, group: &mut Option<(Seq, Vec<String>)>| {
        if let Some((seq, blocks)) = group.take() {
            let content = RawJson::from_string(format!("[{}]", blocks.join(",")))
                .map_err(|e| e.to_string())?;
            req.push(MessageSlice::new(seq, Role::User, &content).map_err(|e| e.to_string())?);
        }
        Ok::<(), String>(())
    };
    for (seq, slice) in &parts.messages {
        match slice {
            Slice::ToolResult { block, .. } => {
                group
                    .get_or_insert_with(|| (*seq, Vec::new()))
                    .1
                    .push(block.as_str().to_owned());
            }
            Slice::User { content } => match &mut group {
                Some((_, blocks)) => blocks.extend(user_blocks(content)?),
                None => req
                    .push(MessageSlice::new(*seq, Role::User, content).map_err(|e| e.to_string())?),
            },
            Slice::Assistant { content, .. } => {
                flush(&mut req, &mut group)?;
                req.push(
                    MessageSlice::new(*seq, Role::Assistant, content).map_err(|e| e.to_string())?,
                );
            }
        }
    }
    flush(&mut req, &mut group)?;
    Ok(req)
}

/// A user message's content as blocks: a string becomes one text block, an array gives its blocks as they are.
fn user_blocks(content: &RawJson) -> Result<Vec<String>, String> {
    let text = content.as_str();
    if text.starts_with('"') {
        return Ok(vec![format!(r#"{{"type":"text","text":{text}}}"#)]);
    }
    let blocks: Vec<&RawValue> = serde_json::from_str(text).map_err(|e| e.to_string())?;
    Ok(blocks.into_iter().map(|b| b.get().to_owned()).collect())
}

/// The `tools` value for `tools`, or `None` for no tools.
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
                r#"{{"name":{},"description":{},"input_schema":{}}}"#,
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

/// A `tool_result` block.
pub fn result_block(call: &CallId, content: &RawJson, is_error: bool) -> RawJson {
    let error = if is_error { r#","is_error":true"# } else { "" };
    let text = format!(
        r#"{{"type":"tool_result","tool_use_id":{},"content":{}{error}}}"#,
        RawJson::string(call.as_str()).as_str(),
        content.as_str()
    );
    // Both parts are JSON values, so the block is one.
    crate::built_json(text)
}

/// The recipe's parameters: `{"model":…,"max_tokens":…,"extra":{…}}`, `extra` as given.
fn params_value(model: &str, max_tokens: u32, extra: Option<&RawJson>) -> RawJson {
    let extra = extra.map_or(String::new(), |e| format!(r#","extra":{}"#, e.as_str()));
    let text = format!(
        r#"{{"model":{},"max_tokens":{max_tokens}{extra}}}"#,
        RawJson::string(model).as_str()
    );
    crate::built_json(text)
}

/// The recipe's parameters as stored: `extra` holds the request members besides `model` and `max_tokens`; a
/// compaction call's `compaction` member ([`sarathi_context::compaction_request`] adds it at the top) is sent after
/// them. Any other member is refused, so nothing a recipe holds is left out of the request without a word.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Params<'a> {
    model: String,
    max_tokens: u32,
    #[serde(borrow)]
    extra: Option<&'a RawValue>,
    #[serde(borrow)]
    compaction: Option<&'a RawValue>,
}

fn read_params(params: &RawJson) -> Result<Params<'_>, String> {
    serde_json::from_str(params.as_str())
        .map_err(|e| format!("the call parameters cannot be read: {e}"))
}

fn call_params(params: &RawJson) -> Result<CallParams, String> {
    let p = read_params(params)?;
    let base = CallParams::new(p.model, p.max_tokens);
    let extra = match (p.extra, p.compaction) {
        (None, None) => return Ok(base),
        (Some(e), None) => e.get().to_owned(),
        (e, Some(c)) => {
            let member = format!(r#""compaction":{}"#, c.get());
            let inner = match e {
                None => "",
                Some(e) => e
                    .get()
                    .trim()
                    .strip_prefix('{')
                    .and_then(|t| t.strip_suffix('}'))
                    .ok_or("`extra` is not an object")?
                    .trim(),
            };
            if inner.is_empty() {
                format!("{{{member}}}")
            } else {
                format!("{{{inner},{member}}}")
            }
        }
    };
    let extra = RawJson::from_text(&extra).map_err(|e| e.to_string())?;
    base.with_extra(&extra).map_err(|e| e.to_string())
}

/// The request asks for a server compaction or sends a compaction block back: it needs [`COMPACTION_BETA`].
fn needs_compaction_beta(parts: &RequestParts<'_>) -> bool {
    #[derive(serde::Deserialize)]
    struct Extra<'a> {
        #[serde(borrow)]
        compaction: Option<&'a RawValue>,
    }
    #[derive(serde::Deserialize)]
    struct Block<'a> {
        #[serde(rename = "type", borrow)]
        kind: Option<std::borrow::Cow<'a, str>>,
    }
    let asks = read_params(parts.params).is_ok_and(|p| {
        p.compaction.is_some()
            || p.extra.is_some_and(|e| {
                serde_json::from_str::<Extra<'_>>(e.get()).is_ok_and(|x| x.compaction.is_some())
            })
    });
    asks || parts.messages.iter().any(|(_, slice)| match slice {
        Slice::Assistant { content, .. } => {
            serde_json::from_str::<Vec<Block<'_>>>(content.as_str())
                .is_ok_and(|bs| bs.iter().any(|b| b.kind.as_deref() == Some("compaction")))
        }
        _ => false,
    })
}

fn received(r: MessagesResponse) -> Result<Received, CallFailure> {
    let body = RawJson::from_slice(&r.raw)
        .map_err(|e| CallFailure::new(ErrorKind::Malformed, e.to_string()))?;
    let mut out = Received::new(body, r.content, r.view);
    out.sse = r.sse.map(|b| String::from_utf8_lossy(&b).into_owned());
    out.request_id = r.request_id.map(Into::into);
    Ok(out)
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

    fn raw(s: &str) -> RawJson {
        RawJson::from_text(s).unwrap()
    }

    fn parts<'a>(
        params: &'a RawJson,
        system: &'a RawJson,
        messages: Vec<(Seq, Slice<'a>)>,
    ) -> RequestParts<'a> {
        RequestParts {
            protocol: PROTOCOL,
            system,
            tools: system,
            messages,
            params,
        }
    }

    fn body(params: &RawJson) -> String {
        let user = raw(r#""hi""#);
        let null = RawJson::null();
        let p = parts(
            params,
            &null,
            vec![(Seq(2), Slice::User { content: &user })],
        );
        String::from_utf8(request(&p).unwrap().body(Endpoint::Messages)).unwrap()
    }

    #[test]
    fn a_compaction_member_is_sent_after_the_extras() {
        let b = body(&raw(
            r#"{"model":"m","max_tokens":9,"extra":{"thinking":{"type":"enabled"}},"compaction":{"type":"summarize"}}"#,
        ));
        assert!(
            b.contains(r#""thinking":{"type":"enabled"},"compaction":{"type":"summarize"}"#),
            "{b}"
        );
        let b = body(&raw(
            r#"{"model":"m","max_tokens":9,"compaction":{"type":"summarize"}}"#,
        ));
        assert!(b.contains(r#""compaction":{"type":"summarize"}"#), "{b}");
        let b = body(&raw(
            r#"{"model":"m","max_tokens":9,"extra":{ },"compaction":{"type":"summarize"}}"#,
        ));
        assert!(b.contains(r#""compaction":{"type":"summarize"}"#), "{b}");
    }

    #[test]
    fn a_member_the_recipe_cannot_send_is_refused() {
        let p = raw(r#"{"model":"m","max_tokens":9,"temperature":0.5}"#);
        assert!(call_params(&p).is_err());
        let p = raw(
            r#"{"model":"m","max_tokens":9,"extra":{"compaction":{}},"compaction":{"type":"summarize"}}"#,
        );
        assert!(call_params(&p).is_err(), "two compaction members");
    }

    #[test]
    fn the_beta_goes_with_a_compaction_call_or_a_block_sent_back() {
        let null = RawJson::null();
        let plain = raw(r#"{"model":"m","max_tokens":9}"#);
        let asks = raw(r#"{"model":"m","max_tokens":9,"compaction":{"type":"summarize"}}"#);
        let user = raw(r#""hi""#);
        let text = raw(r#"[{"type":"text","text":"ok"}]"#);
        let block = raw(r#"[{"type":"compaction","content":"c","signature":"s"}]"#);
        let assistant = |content: &RawJson| Slice::Assistant {
            content: content.clone(),
            calls: &[],
        };
        assert!(!needs_compaction_beta(&parts(
            &plain,
            &null,
            vec![
                (Seq(2), Slice::User { content: &user }),
                (Seq(3), assistant(&text))
            ]
        )));
        assert!(needs_compaction_beta(&parts(&asks, &null, vec![])));
        assert!(needs_compaction_beta(&parts(
            &plain,
            &null,
            vec![
                (Seq(2), assistant(&block)),
                (Seq(3), Slice::User { content: &user })
            ]
        )));
    }
}
