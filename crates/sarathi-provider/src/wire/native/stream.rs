// SPDX-License-Identifier: Apache-2.0
//! The assembler from Messages SSE events to the message.
//!
//! Each block starts from the raw `content_block` of its `content_block_start`, keys in their order. Known deltas
//! append into the field they name; a block that gets no delta is kept byte for byte as started (an empty `thinking`
//! with only a signature, a whole server-tool result). An unknown delta type fails the turn; unknown event types and
//! `ping` are skipped. The assembled message is written once, then read like a non-streamed body, so the result is
//! byte-stable from then on.

use bytes::Bytes;
use serde::Deserialize;
use serde_json::value::RawValue;

use super::NativeError;
use super::response::{MessagesResponse, entries, parse};
use crate::wire::sse::{SseEvent, SseParser};

/// The type of an event, read first so an unknown event is skipped whatever its other fields hold.
#[derive(Deserialize)]
struct Kind {
    #[serde(rename = "type")]
    kind: String,
}

/// The event types the assembler reads; every other type is skipped.
const KNOWN_EVENTS: [&str; 8] = [
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
    "error",
    "ping",
];

type Entries = Vec<(String, Box<RawValue>)>;

/// What a known delta appends to.
#[derive(Debug)]
enum Acc {
    /// A string field (`text`, `thinking`, `signature`): decoded text, re-rendered once at the block's end.
    Text(String),
    /// `input`: the `partial_json` pieces, joined as text.
    Json(String),
    /// `citations`: whole citation objects, appended to the array.
    Items(Vec<Box<RawValue>>),
}

#[derive(Debug)]
struct Block {
    block_type: String,
    start: Box<RawValue>,
    entries: Entries,
    /// Fields that got deltas, in the order of their first delta.
    accs: Vec<(String, Acc)>,
    /// The block as written at `content_block_stop`.
    done: Option<String>,
}

#[derive(Deserialize)]
struct Event {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    index: Option<usize>,
    #[serde(default)]
    message: Option<Box<RawValue>>,
    #[serde(default)]
    content_block: Option<Box<RawValue>>,
    #[serde(default)]
    delta: Option<Box<RawValue>>,
    #[serde(default)]
    usage: Option<Box<RawValue>>,
    #[serde(default)]
    error: Option<ErrorBody>,
}

#[derive(Deserialize)]
struct ErrorBody {
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    message: String,
}

#[derive(Deserialize)]
struct Delta {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    thinking: Option<String>,
    #[serde(default)]
    signature: Option<String>,
    #[serde(default)]
    partial_json: Option<String>,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    citation: Option<Box<RawValue>>,
}

/// Builds one message from a Messages SSE stream, keeping the stream's bytes as received.
#[derive(Debug)]
pub struct Assembler {
    parser: SseParser,
    sse: Vec<u8>,
    max_sse: usize,
    message: Option<Entries>,
    blocks: Vec<Block>,
    /// `message_delta.delta` members, replaced or appended over the start's.
    top: Entries,
    /// `message_delta.usage` members, over the start's usage.
    usage: Entries,
    stopped: bool,
    skipped: Vec<String>,
}

impl Assembler {
    /// An assembler that keeps at most `max_sse` bytes of stream.
    pub fn new(max_sse: usize) -> Self {
        Assembler {
            // One event may be as large as the whole stream (a start-only document or compaction block).
            parser: SseParser::with_max_event_bytes(max_sse),
            sse: Vec::new(),
            max_sse,
            message: None,
            blocks: Vec::new(),
            top: Vec::new(),
            usage: Vec::new(),
            stopped: false,
            skipped: Vec::new(),
        }
    }

    /// Feeds a chunk of the stream, cut anywhere; returns the complete events it finished, for a host to show.
    ///
    /// # Errors
    /// The stream is over its cap, an event is over the SSE parser's cap or not a Messages event, a delta is of an
    /// unknown type ([`NativeError::UnassemblableDelta`]), or the provider sent an `error` event.
    pub fn feed(&mut self, chunk: &[u8]) -> Result<Vec<SseEvent>, NativeError> {
        if self.sse.len() + chunk.len() > self.max_sse {
            return Err(NativeError::Malformed(format!(
                "the stream is larger than the limit of {} bytes",
                self.max_sse
            )));
        }
        self.sse.extend_from_slice(chunk);
        let events = self.parser.feed(chunk);
        if self.parser.overflowed() {
            return Err(NativeError::Malformed(
                "an event is larger than the SSE limit".into(),
            ));
        }
        for e in &events {
            self.event(e)?;
        }
        Ok(events)
    }

    /// Whether `message_stop` has arrived.
    pub fn is_complete(&self) -> bool {
        self.stopped
    }

    /// Event types that were skipped as unknown, in order (for the host to log).
    pub fn skipped(&self) -> &[String] {
        &self.skipped
    }

    /// The assembled message, with the stream's bytes in [`MessagesResponse::sse`].
    ///
    /// # Errors
    /// The stream ended before `message_stop` ([`NativeError::Cut`]), or a block was never stopped.
    pub fn finish(self) -> Result<MessagesResponse, NativeError> {
        if !self.stopped {
            return Err(NativeError::Cut(
                "the stream ended before `message_stop`".into(),
            ));
        }
        let Some(message) = self.message else {
            return Err(NativeError::Malformed("no `message_start`".into()));
        };
        let mut content = String::from("[");
        for (i, b) in self.blocks.iter().enumerate() {
            let Some(done) = &b.done else {
                return Err(NativeError::Malformed(format!(
                    "block {i} was never stopped"
                )));
            };
            if i > 0 {
                content.push(',');
            }
            content.push_str(done);
        }
        content.push(']');
        let usage = message
            .iter()
            .find(|(k, _)| k == "usage")
            .map(|(_, v)| merge(v.get(), &self.usage))
            .transpose()?
            .unwrap_or_else(|| object(self.usage.iter().map(|(k, v)| (k.as_str(), v.get()))));
        let mut out: Vec<(&str, &str)> = Vec::new();
        for (k, v) in &message {
            let v = match k.as_str() {
                "content" => content.as_str(),
                "usage" => usage.as_str(),
                _ => self
                    .top
                    .iter()
                    .find(|(t, _)| t == k)
                    .map_or(v.get(), |(_, d)| d.get()),
            };
            out.push((k, v));
        }
        for (k, v) in &self.top {
            if !message.iter().any(|(m, _)| m == k) {
                out.push((k, v.get()));
            }
        }
        if !message.iter().any(|(m, _)| m == "content") {
            out.push(("content", &content));
        }
        if !message.iter().any(|(m, _)| m == "usage") && !self.usage.is_empty() {
            out.push(("usage", &usage));
        }
        let body = object(out);
        let mut r = parse(Bytes::from(body.into_bytes()))?;
        r.sse = Some(Bytes::from(self.sse));
        Ok(r)
    }

    fn event(&mut self, e: &SseEvent) -> Result<(), NativeError> {
        let not_an_event = |err: serde_json::Error| {
            NativeError::Malformed(format!("an event that is not a Messages event: {err}"))
        };
        let kind: Kind = serde_json::from_str(&e.data).map_err(not_an_event)?;
        if self.stopped {
            return Ok(());
        }
        if !KNOWN_EVENTS.contains(&kind.kind.as_str()) {
            self.skipped.push(kind.kind);
            return Ok(());
        }
        let ev: Event = serde_json::from_str(&e.data).map_err(not_an_event)?;
        match ev.kind.as_str() {
            "message_start" => {
                if self.message.is_some() {
                    return Err(NativeError::Malformed("a second `message_start`".into()));
                }
                let m = ev.message.ok_or_else(|| {
                    NativeError::Malformed("`message_start` without `message`".into())
                })?;
                self.message = Some(object_entries(m.get())?);
            }
            "content_block_start" => {
                self.started()?;
                let i = index(&ev)?;
                if i != self.blocks.len() {
                    return Err(NativeError::Malformed(format!(
                        "block {i} started, block {} expected",
                        self.blocks.len()
                    )));
                }
                let start = ev.content_block.ok_or_else(|| {
                    NativeError::Malformed("`content_block_start` without a block".into())
                })?;
                let entries = object_entries(start.get())?;
                let block_type = entries
                    .iter()
                    .find(|(k, _)| k == "type")
                    .and_then(|(_, v)| serde_json::from_str::<String>(v.get()).ok())
                    .ok_or_else(|| NativeError::Malformed("a block without a type".into()))?;
                self.blocks.push(Block {
                    block_type,
                    start,
                    entries,
                    accs: Vec::new(),
                    done: None,
                });
            }
            "content_block_delta" => {
                let i = index(&ev)?;
                let delta = ev.delta.ok_or_else(|| {
                    NativeError::Malformed("`content_block_delta` without a delta".into())
                })?;
                self.open(i)?.delta(delta.get())?;
            }
            "content_block_stop" => {
                let i = index(&ev)?;
                let b = self.open(i)?;
                b.done = Some(b.render()?);
            }
            "message_delta" => {
                self.started()?;
                if let Some(d) = ev.delta {
                    over(&mut self.top, object_entries(d.get())?);
                }
                if let Some(u) = ev.usage {
                    over(&mut self.usage, object_entries(u.get())?);
                }
            }
            "message_stop" => {
                self.started()?;
                self.stopped = true;
            }
            "error" => {
                let err = ev.error.unwrap_or(ErrorBody {
                    kind: String::new(),
                    message: String::new(),
                });
                return Err(NativeError::Provider {
                    status: status_of(&err.kind),
                    error_type: err.kind,
                    message: err.message,
                });
            }
            _ => {}
        }
        Ok(())
    }

    fn started(&self) -> Result<(), NativeError> {
        if self.message.is_none() {
            return Err(NativeError::Malformed(
                "an event before `message_start`".into(),
            ));
        }
        Ok(())
    }

    fn open(&mut self, i: usize) -> Result<&mut Block, NativeError> {
        match self.blocks.get_mut(i) {
            Some(b) if b.done.is_none() => Ok(b),
            Some(_) => Err(NativeError::Malformed(format!(
                "block {i} is already stopped"
            ))),
            None => Err(NativeError::Malformed(format!(
                "block {i} was never started"
            ))),
        }
    }
}

impl Block {
    fn delta(&mut self, raw: &str) -> Result<(), NativeError> {
        let d: Delta = serde_json::from_str(raw)
            .map_err(|e| NativeError::Malformed(format!("a delta without a type: {e}")))?;
        let missing = |f: &str| NativeError::Malformed(format!("a `{}` without `{f}`", d.kind));
        let (field, piece) = match d.kind.as_str() {
            "text_delta" => ("text", Piece::Text(d.text.ok_or_else(|| missing("text"))?)),
            "thinking_delta" => (
                "thinking",
                Piece::Text(d.thinking.ok_or_else(|| missing("thinking"))?),
            ),
            "signature_delta" => (
                "signature",
                Piece::Text(d.signature.ok_or_else(|| missing("signature"))?),
            ),
            "input_json_delta" => (
                "input",
                Piece::Json(d.partial_json.ok_or_else(|| missing("partial_json"))?),
            ),
            // Server-side compaction streams its summary as one or more `content` pieces.
            "compaction_delta" => (
                "content",
                Piece::Text(d.content.ok_or_else(|| missing("content"))?),
            ),
            "citations_delta" => (
                "citations",
                Piece::Item(d.citation.ok_or_else(|| missing("citation"))?),
            ),
            _ => {
                return Err(NativeError::UnassemblableDelta {
                    block_type: self.block_type.clone(),
                    delta_type: d.kind,
                });
            }
        };
        let pos = match self.accs.iter().position(|(k, _)| k == field) {
            Some(p) => p,
            None => {
                let acc = match &piece {
                    Piece::Text(_) => Acc::Text(String::new()),
                    Piece::Json(_) => Acc::Json(String::new()),
                    Piece::Item(_) => Acc::Items(Vec::new()),
                };
                self.accs.push((field.to_string(), acc));
                self.accs.len() - 1
            }
        };
        let Some((_, acc)) = self.accs.get_mut(pos) else {
            return Err(NativeError::Malformed(format!(
                "no accumulator for `{field}`"
            )));
        };
        match (acc, piece) {
            (Acc::Text(s), Piece::Text(p)) | (Acc::Json(s), Piece::Json(p)) => s.push_str(&p),
            (Acc::Items(v), Piece::Item(p)) => v.push(p),
            _ => {
                return Err(NativeError::Malformed(format!(
                    "mixed delta types for `{field}`"
                )));
            }
        }
        Ok(())
    }

    /// The block as it goes into the message: the start's bytes when nothing was appended, else the start's members
    /// in order with the appended fields written once.
    fn render(&self) -> Result<String, NativeError> {
        if self.accs.is_empty() {
            return Ok(self.start.get().to_string());
        }
        let mut written: Vec<(String, String)> = Vec::new();
        for (k, v) in &self.entries {
            let value = match self.accs.iter().find(|(a, _)| a == k) {
                Some((_, acc)) => finish(acc, Some(v.get()))?,
                None => v.get().to_string(),
            };
            written.push((k.clone(), value));
        }
        for (k, acc) in &self.accs {
            if !self.entries.iter().any(|(e, _)| e == k) {
                written.push((k.clone(), finish(acc, None)?));
            }
        }
        let text = object(written.iter().map(|(k, v)| (k.as_str(), v.as_str())));
        // Validate once: a bad `partial_json` must not reach the log.
        RawValue::from_string(text)
            .map(|v| v.get().to_string())
            .map_err(|e| {
                NativeError::Malformed(format!(
                    "block `{}` is not valid JSON: {e}",
                    self.block_type
                ))
            })
    }
}

enum Piece {
    Text(String),
    Json(String),
    Item(Box<RawValue>),
}

/// The value of an appended field, given the start's value of it (if the start had the field).
fn finish(acc: &Acc, initial: Option<&str>) -> Result<String, NativeError> {
    match acc {
        Acc::Text(s) => {
            let mut full = match initial {
                Some(v) => serde_json::from_str::<Option<String>>(v)
                    .map_err(|_| {
                        NativeError::Malformed("a delta into a field that is not a string".into())
                    })?
                    .unwrap_or_default(),
                None => String::new(),
            };
            full.push_str(s);
            Ok(serde_json::to_string(&full).unwrap_or_default())
        }
        Acc::Json(s) if s.trim().is_empty() => Ok(initial.unwrap_or("{}").to_string()),
        Acc::Json(s) => {
            // One JSON object on its own, so pieces cannot close `input` early and add members to the block.
            let v = RawValue::from_string(s.trim().to_string()).map_err(|e| {
                NativeError::Malformed(format!("`input` is not one JSON value: {e}"))
            })?;
            if !v.get().starts_with('{') {
                return Err(NativeError::Malformed(
                    "`input` is not a JSON object".into(),
                ));
            }
            Ok(v.get().to_string())
        }
        Acc::Items(items) => {
            let new = items.iter().map(|i| i.get()).collect::<Vec<_>>().join(",");
            match initial.map(str::trim) {
                None | Some("null") => Ok(format!("[{new}]")),
                Some(arr) if arr.starts_with('[') && arr.ends_with(']') => {
                    let inner = arr
                        .strip_prefix('[')
                        .and_then(|a| a.strip_suffix(']'))
                        .unwrap_or_default()
                        .trim();
                    Ok(if inner.is_empty() {
                        format!("[{new}]")
                    } else {
                        format!("[{inner},{new}]")
                    })
                }
                Some(_) => Err(NativeError::Malformed(
                    "a citation into a field that is not an array".into(),
                )),
            }
        }
    }
}

/// `{"k":v,…}` from members whose values are JSON text; keys are written as JSON strings.
fn object<'a>(members: impl IntoIterator<Item = (&'a str, &'a str)>) -> String {
    let mut out = String::from("{");
    for (i, (k, v)) in members.into_iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&serde_json::to_string(k).unwrap_or_default());
        out.push(':');
        out.push_str(v);
    }
    out.push('}');
    out
}

/// `base`'s members with `over`'s replacing same-named ones and the rest appended.
fn merge(base: &str, over: &Entries) -> Result<String, NativeError> {
    let mut members = object_entries(base)?;
    self::over(
        &mut members,
        over.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
    );
    Ok(object(members.iter().map(|(k, v)| (k.as_str(), v.get()))))
}

fn over(base: &mut Entries, new: Entries) {
    for (k, v) in new {
        match base.iter_mut().find(|(b, _)| *b == k) {
            Some(slot) => slot.1 = v,
            None => base.push((k, v)),
        }
    }
}

fn object_entries(text: &str) -> Result<Entries, NativeError> {
    entries(text).map_err(|e| NativeError::Malformed(format!("not a JSON object: {e}")))
}

fn index(ev: &Event) -> Result<usize, NativeError> {
    ev.index
        .ok_or_else(|| NativeError::Malformed(format!("`{}` without `index`", ev.kind)))
}

/// The HTTP status an in-stream error type stands for.
fn status_of(error_type: &str) -> u16 {
    match error_type {
        "invalid_request_error" => 400,
        "authentication_error" => 401,
        "billing_error" => 402,
        "permission_error" => 403,
        "not_found_error" => 404,
        "request_too_large" => 413,
        "rate_limit_error" => 429,
        "timeout_error" => 504,
        "overloaded_error" => 529,
        _ => 500,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sse(events: &[&str]) -> String {
        events
            .iter()
            .map(|d| {
                let kind = serde_json::from_str::<serde_json::Value>(d).unwrap()["type"]
                    .as_str()
                    .unwrap()
                    .to_string();
                format!("event: {kind}\ndata: {d}\n\n")
            })
            .collect()
    }

    const START: &str = r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":9,"output_tokens":1}}}"#;
    const STOP: &str = r#"{"type":"message_stop"}"#;

    fn run(events: &[&str]) -> Result<MessagesResponse, NativeError> {
        let mut a = Assembler::new(1 << 20);
        a.feed(sse(events).as_bytes())?;
        a.finish()
    }

    #[test]
    fn assembles_thinking_text_and_tool_use() {
        let r = run(&[
            START,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me \u00e9"}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"think."}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"EqQB"}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"ping"}"#,
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"edit","input":{}}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":""}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"z\": 1e5,"}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":" \"a\":\"\\u00e9\"}"}}"#,
            r#"{"type":"content_block_stop","index":1}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":42}}"#,
            STOP,
        ])
        .unwrap();
        assert_eq!(
            std::str::from_utf8(r.content_bytes()).unwrap(),
            concat!(
                r#"[{"type":"thinking","thinking":"Let me éthink.","signature":"EqQB"},"#,
                r#"{"type":"tool_use","id":"toolu_1","name":"edit","input":{"z": 1e5, "a":"\u00e9"}}]"#
            )
        );
        assert_eq!(r.view.stop_reason.as_deref(), Some("tool_use"));
        assert_eq!(r.view.usage.input_tokens, 9);
        assert_eq!(r.view.usage.output_tokens, 42);
        let body = std::str::from_utf8(&r.raw).unwrap();
        assert!(
            body.starts_with(
                r#"{"id":"msg_1","type":"message","role":"assistant","model":"m","content":["#
            ),
            "{body}"
        );
        assert!(body.ends_with(r#""stop_reason":"tool_use","stop_sequence":null,"usage":{"input_tokens":9,"output_tokens":42}}"#), "{body}");
        assert!(r.sse.is_some());
    }

    #[test]
    fn a_block_without_deltas_is_kept_as_started() {
        let block = r#"{"type":"thinking", "thinking":"","signature":"sig\/1"}"#;
        let r = run(&[
            START,
            &format!(r#"{{"type":"content_block_start","index":0,"content_block":{block}}}"#),
            r#"{"type":"content_block_stop","index":0}"#,
            STOP,
        ])
        .unwrap();
        assert_eq!(
            std::str::from_utf8(r.content_bytes()).unwrap(),
            format!("[{block}]")
        );
    }

    #[test]
    fn citations_append_and_unknown_deltas_fail() {
        let r = run(&[
            START,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"","citations":[]}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"char_location","cited_text":"x"}}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"a"}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            STOP,
        ])
        .unwrap();
        assert_eq!(
            std::str::from_utf8(r.content_bytes()).unwrap(),
            r#"[{"type":"text","text":"a","citations":[{"type":"char_location","cited_text":"x"}]}]"#
        );
        let e = run(&[
            START,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"future_delta","x":"y"}}"#,
        ])
        .unwrap_err();
        assert_eq!(
            e,
            NativeError::UnassemblableDelta {
                block_type: "text".into(),
                delta_type: "future_delta".into()
            }
        );
    }

    #[test]
    fn compaction_deltas_fill_the_content() {
        let r = run(&[
            START,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"compaction","content":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"compaction_delta","content":"Sum"}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"compaction_delta","content":"mary."}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            STOP,
        ])
        .unwrap();
        assert_eq!(
            std::str::from_utf8(r.content_bytes()).unwrap(),
            r#"[{"type":"compaction","content":"Summary."}]"#
        );
    }

    #[test]
    fn input_pieces_cannot_add_members_or_change_the_type() {
        let tool = |pieces: &[&str]| {
            let mut evs = vec![
                START.to_string(),
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t","name":"n","input":{}}}"#.to_string(),
            ];
            for p in pieces {
                evs.push(format!(
                    r#"{{"type":"content_block_delta","index":0,"delta":{{"type":"input_json_delta","partial_json":{}}}}}"#,
                    serde_json::to_string(p).unwrap()
                ));
            }
            evs.push(r#"{"type":"content_block_stop","index":0}"#.into());
            evs.push(STOP.into());
            let refs: Vec<&str> = evs.iter().map(String::as_str).collect();
            run(&refs)
        };
        assert!(matches!(
            tool(&[r#"{"a":1},"evil":2"#]),
            Err(NativeError::Malformed(_))
        ));
        assert!(matches!(tool(&["[1]"]), Err(NativeError::Malformed(_))));
        assert!(tool(&[r#"{"a":"#, "1}"]).is_ok());
    }

    #[test]
    fn an_event_may_be_as_large_as_the_stream_cap() {
        let big = "x".repeat(2 << 20);
        let block = format!(r#"{{"type":"document_result","text":"{big}"}}"#);
        let mut a = Assembler::new(8 << 20);
        a.feed(
            sse(&[
                START,
                &format!(r#"{{"type":"content_block_start","index":0,"content_block":{block}}}"#),
                r#"{"type":"content_block_stop","index":0}"#,
                STOP,
            ])
            .as_bytes(),
        )
        .unwrap();
        assert_eq!(
            a.finish().unwrap().content_bytes(),
            format!("[{block}]").as_bytes()
        );
    }

    #[test]
    fn an_unknown_event_with_odd_fields_is_skipped() {
        let mut a = Assembler::new(1 << 20);
        a.feed(sse(&[START, r#"{"type":"future_event","index":"x","delta":7}"#]).as_bytes())
            .unwrap();
        assert_eq!(a.skipped(), ["future_event"]);
    }

    #[test]
    fn unknown_events_are_skipped_and_errors_surface() {
        let mut a = Assembler::new(1 << 20);
        a.feed(sse(&[START, r#"{"type":"future_event","x":1}"#]).as_bytes())
            .unwrap();
        assert_eq!(a.skipped(), ["future_event"]);
        assert!(!a.is_complete());
        let e = a
            .feed(sse(&[r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#]).as_bytes())
            .unwrap_err();
        assert!(
            matches!(e, NativeError::Provider { status: 529, .. }),
            "{e:?}"
        );
        assert!(matches!(a.finish(), Err(NativeError::Cut(_))));
    }

    #[test]
    fn order_and_shape_errors() {
        let bad = |evs: &[&str]| run(evs).unwrap_err();
        assert!(matches!(
            bad(&[r#"{"type":"message_stop"}"#]),
            NativeError::Malformed(_)
        ));
        assert!(matches!(bad(&[START, START]), NativeError::Malformed(_)));
        assert!(matches!(
            bad(&[
                START,
                r#"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#
            ]),
            NativeError::Malformed(_)
        ));
        assert!(matches!(
            bad(&[START, r#"{"type":"content_block_stop","index":0}"#]),
            NativeError::Malformed(_)
        ));
        assert!(matches!(
            bad(&[
                START,
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
                STOP
            ]),
            NativeError::Malformed(_)
        ));
        assert!(matches!(
            bad(&[
                START,
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t","name":"n","input":{}}}"#,
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"a\":"}}"#,
                r#"{"type":"content_block_stop","index":0}"#,
            ]),
            NativeError::Malformed(_)
        ));
        let mut a = Assembler::new(1 << 20);
        assert!(matches!(
            a.feed(b"data: not json\n\n"),
            Err(NativeError::Malformed(_))
        ));
        let mut a = Assembler::new(10);
        assert!(matches!(
            a.feed(START.as_bytes()),
            Err(NativeError::Malformed(_))
        ));
    }

    #[test]
    fn error_type_statuses() {
        assert_eq!(status_of("rate_limit_error"), 429);
        assert_eq!(status_of("invalid_request_error"), 400);
        assert_eq!(status_of("whatever"), 500);
    }
}
