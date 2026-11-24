// SPDX-License-Identifier: Apache-2.0
//! The line codec: one [`Entry`] to one JSON line and back, with every stored slice byte for byte.
//!
//! A line is fixed keys followed by the entry's fields, written by concatenation:
//!
//! ```text
//! {"seq":7,"seg":0,"ts":1764057600000,"kind":"tool.result","call":"toolu_1","block":<slice>,"is_error":false}
//! ```
//!
//! A slice is written as its exact bytes, since a JSON value is valid where a JSON value is expected. JSONL forbids a
//! raw newline, and a provider may put `\n` or `\r` in the whitespace of a body (never inside a string: a JSON string
//! cannot hold a raw control character). Such a slice is written as a JSON string holding its text instead, and the
//! line lists the field in `"esc":[…]`, so the reader knows to unwrap it. Both forms read back as identical bytes.
//! The list is outside the slice, so no slice value can be mistaken for an escaped one.

use std::collections::BTreeMap;
use std::ops::Range;

use sarathi_protocol::{RawJson, Seq, Usage};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::value::RawValue;

use crate::entry::{Body, CallRef, Effect, Entry, LogHeader, Recipe, SegmentReason, Strategy};
use crate::error::LogError;
use crate::log::Log;
use sarathi_protocol::{HeaderError, SegmentId};

/// Why a line did not decode.
#[derive(Debug)]
enum DecodeError {
    /// Not a well-formed entry.
    Bad(String),
    /// A well-formed entry of an unknown kind that must be understood.
    MustUnderstand { seq: Seq, kind: String },
}

impl DecodeError {
    fn into_log_error(self, line: u64) -> LogError {
        match self {
            DecodeError::Bad(detail) => LogError::Corrupt { line, detail },
            DecodeError::MustUnderstand { seq, kind } => LogError::MustUnderstand { seq, kind },
        }
    }
}

fn bad(detail: impl Into<String>) -> DecodeError {
    DecodeError::Bad(detail.into())
}

/// Renders the header line, without its `\n`.
///
/// # Errors
/// Never in practice: the header has only strings and integers.
pub fn encode_header(header: &LogHeader) -> Result<String, LogError> {
    serde_json::to_string(header).map_err(|e| LogError::Invalid {
        seq: Seq(0),
        detail: e.to_string(),
    })
}

/// Reads the header line and checks its format version.
///
/// # Errors
/// [`LogError::Format`] for a format this reader does not read; [`LogError::Corrupt`] for a line that is not a
/// header.
pub fn decode_header(line: &str) -> Result<LogHeader, LogError> {
    sarathi_protocol::read_header(line).map_err(|e| match e {
        HeaderError::Version(v) => LogError::Format(v),
        e => LogError::Corrupt {
            line: 0,
            detail: format!("header: {e}"),
        },
    })
}

struct LineWriter {
    buf: String,
    esc: Vec<&'static str>,
}

impl LineWriter {
    fn new(e: &Entry) -> Self {
        let mut buf = String::with_capacity(128);
        buf.push_str("{\"seq\":");
        buf.push_str(&e.seq.0.to_string());
        buf.push_str(",\"seg\":");
        buf.push_str(&e.seg.to_string());
        buf.push_str(",\"ts\":");
        buf.push_str(&e.ts.to_string());
        buf.push_str(",\"kind\":");
        push_json_string(&mut buf, e.body.kind().as_str());
        LineWriter {
            buf,
            esc: Vec::new(),
        }
    }

    fn key(&mut self, name: &str) {
        self.buf.push(',');
        push_json_string(&mut self.buf, name);
        self.buf.push(':');
    }

    /// A stored slice: its bytes, or its text as a string when it holds a raw line break.
    fn slice(&mut self, name: &'static str, value: &RawJson) {
        self.key(name);
        let text = value.as_str();
        if text.contains(['\n', '\r']) {
            push_json_string(&mut self.buf, text);
            self.esc.push(name);
        } else {
            self.buf.push_str(text);
        }
    }

    fn opt_slice(&mut self, name: &'static str, value: Option<&RawJson>) {
        if let Some(v) = value {
            self.slice(name, v);
        }
    }

    /// A value Sarathi renders itself (ids, numbers, enums): compact, never a line break.
    fn value<T: Serialize + ?Sized>(&mut self, name: &str, value: &T) -> Result<(), LogError> {
        self.key(name);
        let text = serde_json::to_string(value).map_err(|e| LogError::Invalid {
            seq: Seq(0),
            detail: format!("field {name}: {e}"),
        })?;
        self.buf.push_str(&text);
        Ok(())
    }

    fn opt_value<T: Serialize + ?Sized>(
        &mut self,
        name: &str,
        value: Option<&T>,
    ) -> Result<(), LogError> {
        match value {
            Some(v) => self.value(name, v),
            None => Ok(()),
        }
    }

    fn finish(mut self) -> String {
        if !self.esc.is_empty() {
            self.buf.push_str(",\"esc\":[");
            for (i, name) in self.esc.iter().enumerate() {
                if i > 0 {
                    self.buf.push(',');
                }
                push_json_string(&mut self.buf, name);
            }
            self.buf.push(']');
        }
        self.buf.push('}');
        self.buf
    }
}

fn push_json_string(buf: &mut String, s: &str) {
    // Rendering a `&str` cannot fail.
    buf.push_str(RawJson::string(s).as_str());
}

/// Renders one entry as a line, without its `\n`.
///
/// # Errors
/// [`LogError::Invalid`] if a typed field does not serialise (never for values this crate builds).
pub fn encode_entry(e: &Entry) -> Result<String, LogError> {
    let invalid = |detail: String| LogError::Invalid { seq: e.seq, detail };
    if let Body::Unknown { line, .. } = &e.body {
        // Kept as read: the line already holds its seq and kind.
        if line.as_str().contains(['\n', '\r']) {
            return Err(invalid("an unknown entry's line holds a line break".into()));
        }
        return Ok(line.as_str().to_owned());
    }
    let mut w = LineWriter::new(e);
    let r = (|| {
        match &e.body {
            Body::SessionPrefix {
                protocol,
                system,
                tools,
            } => {
                w.value("protocol", protocol)?;
                w.slice("system", system);
                w.slice("tools", tools);
            }
            Body::UserMessage { content, steer } => {
                w.slice("content", content);
                if *steer {
                    w.value("steer", &true)?;
                }
            }
            Body::ModelRequest { recipe, sha256 } => {
                w.value("prefix", &recipe.prefix)?;
                w.value("messages", &recipe.messages)?;
                w.slice("params", &recipe.params);
                w.value("sha256", sha256)?;
            }
            Body::ModelResponse {
                body,
                content,
                sse,
                request_id,
                usage,
            } => {
                w.slice("body", body);
                w.value("content", &[content.start, content.end])?;
                w.opt_value("sse", sse.as_ref())?;
                w.opt_value("request_id", request_id.as_ref())?;
                w.value("usage", usage)?;
            }
            Body::ModelFailure {
                error,
                status,
                detail,
                request_id,
                retries,
            } => {
                w.value("error", error)?;
                w.opt_value("status", status.as_ref())?;
                w.value("detail", detail)?;
                w.opt_value("request_id", request_id.as_ref())?;
                w.value("retries", retries)?;
            }
            Body::AssistantMessage { from, calls } => {
                w.value("from", from)?;
                w.value("calls", calls)?;
            }
            Body::ToolAsk { call, tool, input } => {
                w.value("call", call)?;
                w.value("tool", tool)?;
                w.slice("input", input);
            }
            Body::ToolDecision { call, decision } => {
                w.value("call", call)?;
                w.slice("decision", decision);
            }
            Body::ToolStart { call, tool, effect } => {
                w.value("call", call)?;
                w.value("tool", tool)?;
                w.value("effect", effect)?;
            }
            Body::ToolResult {
                call,
                block,
                is_error,
                spill,
            } => {
                w.value("call", call)?;
                w.slice("block", block);
                w.value("is_error", is_error)?;
                w.opt_slice("spill", spill.as_ref());
            }
            Body::SegmentSeed {
                role,
                content,
                from,
                calls,
            } => {
                w.value("must_understand", &true)?;
                w.value("role", role)?;
                w.slice("content", content);
                w.opt_value("from", from.as_ref())?;
                if !calls.is_empty() {
                    w.value("calls", calls)?;
                }
            }
            Body::SegmentStart {
                reason,
                strategy,
                seed,
                replaces,
            } => {
                w.value("reason", reason)?;
                w.opt_value("strategy", strategy.as_ref())?;
                w.value("seed", seed)?;
                if let Some(r) = replaces {
                    w.value("replaces", &[r.start, r.end])?;
                }
            }
            Body::TurnEnd { outcome } => w.value("outcome", outcome)?,
            Body::Note { data } => w.slice("data", data),
            Body::Unknown { .. } => {}
        }
        Ok::<(), LogError>(())
    })();
    r.map_err(|err| match err {
        LogError::Invalid { detail, .. } => invalid(detail),
        other => other,
    })?;
    Ok(w.finish())
}

struct Fields<'a> {
    map: BTreeMap<String, &'a RawValue>,
    esc: Vec<String>,
}

impl<'a> Fields<'a> {
    fn raw(&self, name: &str) -> Result<&'a RawValue, DecodeError> {
        self.map
            .get(name)
            .copied()
            .ok_or_else(|| bad(format!("missing field `{name}`")))
    }

    fn typed<T: DeserializeOwned>(&self, name: &str) -> Result<T, DecodeError> {
        serde_json::from_str(self.raw(name)?.get()).map_err(|e| bad(format!("field `{name}`: {e}")))
    }

    fn opt_typed<T: DeserializeOwned>(&self, name: &str) -> Result<Option<T>, DecodeError> {
        match self.map.get(name) {
            None => Ok(None),
            Some(r) => serde_json::from_str(r.get())
                .map(Some)
                .map_err(|e| bad(format!("field `{name}`: {e}"))),
        }
    }

    fn slice(&self, name: &str) -> Result<RawJson, DecodeError> {
        let raw = self.raw(name)?;
        if self.esc.iter().any(|n| n == name) {
            let text: String = serde_json::from_str(raw.get())
                .map_err(|e| bad(format!("escaped field `{name}`: {e}")))?;
            RawJson::from_string(text).map_err(|e| bad(format!("escaped field `{name}`: {e}")))
        } else {
            Ok(RawJson::from(raw.to_owned()))
        }
    }

    fn opt_slice(&self, name: &str) -> Result<Option<RawJson>, DecodeError> {
        if self.map.contains_key(name) {
            self.slice(name).map(Some)
        } else {
            Ok(None)
        }
    }

    fn range_u32(&self, name: &str) -> Result<Range<u32>, DecodeError> {
        let [a, b]: [u32; 2] = self.typed(name)?;
        if a > b {
            return Err(bad(format!("field `{name}`: start after end")));
        }
        Ok(a..b)
    }
}

/// Reads one entry line (without its `\n`).
fn decode_entry(line: &str) -> Result<Entry, DecodeError> {
    let map: BTreeMap<String, &RawValue> =
        serde_json::from_str(line).map_err(|e| bad(e.to_string()))?;
    let mut f = Fields {
        map,
        esc: Vec::new(),
    };
    f.esc = f.opt_typed("esc")?.unwrap_or_default();
    let seq: Seq = f.typed("seq")?;
    let seg: u32 = f.typed("seg")?;
    let ts: u64 = f.typed("ts")?;
    let kind: String = f.typed("kind")?;
    let body = match kind.as_str() {
        "session.prefix" => Body::SessionPrefix {
            protocol: f.typed("protocol")?,
            system: f.slice("system")?,
            tools: f.slice("tools")?,
        },
        "message.user" => Body::UserMessage {
            content: f.slice("content")?,
            steer: f.opt_typed("steer")?.unwrap_or(false),
        },
        "model.request" => Body::ModelRequest {
            recipe: Recipe {
                prefix: f.typed("prefix")?,
                messages: f.typed("messages")?,
                params: f.slice("params")?,
            },
            sha256: f.typed("sha256")?,
        },
        "model.response" => {
            let body = f.slice("body")?;
            let content = f.range_u32("content")?;
            if content_slice(&body, &content).is_none() {
                return Err(bad(
                    "model.response: `content` is not a JSON value in `body`",
                ));
            }
            Body::ModelResponse {
                body,
                content,
                sse: f.opt_typed("sse")?,
                request_id: f.opt_typed("request_id")?,
                usage: f.typed::<Usage>("usage")?,
            }
        }
        "model.failure" => Body::ModelFailure {
            error: f.typed("error")?,
            status: f.opt_typed("status")?,
            detail: f.typed("detail")?,
            request_id: f.opt_typed("request_id")?,
            retries: f.typed("retries")?,
        },
        "message.assistant" => Body::AssistantMessage {
            from: f.typed("from")?,
            calls: f.typed::<Vec<CallRef>>("calls")?,
        },
        "tool.ask" => Body::ToolAsk {
            call: f.typed("call")?,
            tool: f.typed("tool")?,
            input: f.slice("input")?,
        },
        "tool.decision" => Body::ToolDecision {
            call: f.typed("call")?,
            decision: f.slice("decision")?,
        },
        "tool.start" => Body::ToolStart {
            call: f.typed("call")?,
            tool: f.typed("tool")?,
            effect: f.typed::<Effect>("effect")?,
        },
        "tool.result" => Body::ToolResult {
            call: f.typed("call")?,
            block: f.slice("block")?,
            is_error: f.typed("is_error")?,
            spill: f.opt_slice("spill")?,
        },
        "segment.start" => Body::SegmentStart {
            reason: f.typed::<SegmentReason>("reason")?,
            strategy: f.opt_typed::<Strategy>("strategy")?,
            seed: f.typed("seed")?,
            replaces: f.opt_typed::<[Seq; 2]>("replaces")?.map(|[a, b]| a..b),
        },
        "turn.end" => Body::TurnEnd {
            outcome: f.typed("outcome")?,
        },
        "segment.seed" => Body::SegmentSeed {
            role: f.typed("role")?,
            content: f.slice("content")?,
            from: f.opt_typed("from")?,
            calls: f.opt_typed::<Vec<CallRef>>("calls")?.unwrap_or_default(),
        },
        "note" => Body::Note {
            data: f.slice("data")?,
        },
        _ => {
            if f.opt_typed::<bool>("must_understand")?.unwrap_or(false) {
                return Err(DecodeError::MustUnderstand { seq, kind });
            }
            Body::Unknown {
                kind: kind.into(),
                line: RawJson::from_text(line).map_err(|e| bad(e.to_string()))?,
            }
        }
    };
    Ok(Entry {
        seq,
        seg: SegmentId(seg),
        ts,
        body,
    })
}

/// The assistant `content` value inside a response body, if the range holds one JSON value.
pub(crate) fn content_slice(body: &RawJson, range: &Range<u32>) -> Option<RawJson> {
    let start = usize::try_from(range.start).ok()?;
    let end = usize::try_from(range.end).ok()?;
    let text = body.as_str().get(start..end)?;
    RawJson::from_text(text).ok()
}

/// A log read from bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadLog {
    /// Every complete entry.
    pub log: Log,
    /// Length of the bytes up to and including the last complete line.
    pub complete_len: u64,
    /// Bytes after it: a torn last line, not read.
    pub torn_bytes: u64,
}

/// Renders a whole log as the JSONL store writes it: the header line, then one line per entry, each with its `\n`.
///
/// # Errors
/// As [`encode_entry`].
pub fn render_log(log: &Log) -> Result<String, LogError> {
    let mut s = encode_header(log.header())?;
    s.push('\n');
    for e in log.entries() {
        s.push_str(&encode_entry(e)?);
        s.push('\n');
    }
    Ok(s)
}

/// Reads a whole log as the JSONL store writes it (header line, then one entry per line). A last line without its `\n` is a torn tail (a crash mid-write) and is not read; any complete
/// line that does not decode, or breaks a log invariant, is corruption.
///
/// # Errors
/// [`LogError::Corrupt`], [`LogError::Format`] or [`LogError::MustUnderstand`].
pub fn read_log(bytes: &[u8]) -> Result<ReadLog, LogError> {
    let good_len = bytes.iter().rposition(|b| *b == b'\n').map_or(0, |i| i + 1);
    let torn_bytes = (bytes.len() - good_len) as u64;
    if good_len == 0 {
        return Err(LogError::Corrupt {
            line: 0,
            detail: "no complete header line".into(),
        });
    }
    // Every complete line, without the final `\n`.
    let mut lines = bytes[..good_len - 1].split(|b| *b == b'\n');
    let text = |line: u64, b: &[u8]| -> Result<String, LogError> {
        String::from_utf8(b.to_vec()).map_err(|e| LogError::Corrupt {
            line,
            detail: e.to_string(),
        })
    };
    let header = decode_header(&text(0, lines.next().unwrap_or_default())?)?;
    let mut log = Log::new(header);
    for (i, raw) in lines.enumerate() {
        let n = i as u64 + 1;
        let e = decode_entry(&text(n, raw)?).map_err(|e| e.into_log_error(n))?;
        let seq = e.seq;
        log.append(seq, vec![e]).map_err(|err| LogError::Corrupt {
            line: n,
            detail: err.to_string(),
        })?;
    }
    Ok(ReadLog {
        log,
        complete_len: good_len as u64,
        torn_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(body: Body) -> Entry {
        Entry {
            seq: Seq(3),
            seg: SegmentId(1),
            ts: 42,
            body,
        }
    }

    fn round_trip(e: &Entry) -> String {
        let line = encode_entry(e).unwrap();
        assert!(!line.contains('\n'), "{line}");
        let back = decode_entry(&line).unwrap();
        assert_eq!(&back, e);
        assert_eq!(encode_entry(&back).unwrap(), line);
        line
    }

    #[test]
    fn a_slice_with_a_line_break_is_escaped_and_comes_back_whole() {
        let pretty = RawJson::from_text("{\n  \"a\": [1e5,\r\n \"\\u00e9\"]\n}").unwrap();
        let line = round_trip(&entry(Body::Note { data: pretty }));
        assert!(line.ends_with(r#","esc":["data"]}"#), "{line}");
    }

    #[test]
    fn an_object_shaped_like_an_escape_is_not_unwrapped() {
        let data = RawJson::from_text(r#"{"esc":"x"}"#).unwrap();
        let line = round_trip(&entry(Body::Note { data }));
        assert_eq!(
            line,
            r#"{"seq":3,"seg":1,"ts":42,"kind":"note","data":{"esc":"x"}}"#
        );
    }

    #[test]
    fn every_kind_round_trips() {
        let raw = |s: &str| RawJson::from_text(s).unwrap();
        let body = raw(r#"{"id":"m","content":[{"type":"text","text":"hi"}],"z":1.0}"#);
        let start = u32::try_from(body.as_str().find('[').unwrap()).unwrap();
        let end = u32::try_from(body.as_str().rfind(']').unwrap() + 1).unwrap();
        let content = start..end;
        assert_eq!(
            content_slice(&body, &content).unwrap().as_str(),
            r#"[{"type":"text","text":"hi"}]"#
        );
        for b in [
            Body::SessionPrefix {
                protocol: "anthropic-messages".into(),
                system: raw(r#""be brief""#),
                tools: raw("[]"),
            },
            Body::UserMessage {
                content: raw(r#""hi""#),
                steer: true,
            },
            Body::ModelRequest {
                recipe: Recipe {
                    prefix: Seq(1),
                    messages: vec![Seq(2)],
                    params: raw(r#"{"max_tokens":10}"#),
                },
                sha256: "ab".into(),
            },
            Body::ModelResponse {
                body,
                content,
                sse: Some("event: x\ndata: {}\n\n".into()),
                request_id: Some("req_1".into()),
                usage: Usage {
                    input_tokens: 4,
                    ..Usage::default()
                },
            },
            Body::ModelFailure {
                error: "overloaded".into(),
                status: Some(529),
                detail: "busy".into(),
                request_id: None,
                retries: 2,
            },
            Body::AssistantMessage {
                from: Seq(2),
                calls: vec![CallRef {
                    id: "toolu_1".into(),
                    name: "edit".into(),
                }],
            },
            Body::SegmentSeed {
                role: sarathi_protocol::Role::Assistant,
                content: raw("[{\"type\":\"text\",\"text\":\"a\\nb\"},\n {\"z\":1e5}]"),
                from: Some(Seq(2)),
                calls: vec![CallRef {
                    id: "toolu_1".into(),
                    name: "edit".into(),
                }],
            },
            Body::SegmentSeed {
                role: sarathi_protocol::Role::User,
                content: raw(r#""note""#),
                from: None,
                calls: vec![],
            },
            Body::ToolAsk {
                call: "toolu_1".into(),
                tool: "edit".into(),
                input: raw(r#"{"path":"a"}"#),
            },
            Body::ToolDecision {
                call: "toolu_1".into(),
                decision: raw(r#"{"allow":true}"#),
            },
            Body::ToolStart {
                call: "toolu_1".into(),
                tool: "edit".into(),
                effect: Effect::Write,
            },
            Body::ToolResult {
                call: "toolu_1".into(),
                block: raw(r#"{"type":"tool_result","tool_use_id":"toolu_1","content":"ok"}"#),
                is_error: false,
                spill: Some(raw(r#"{"path":"spill/1"}"#)),
            },
            Body::SegmentStart {
                reason: SegmentReason::Compaction,
                strategy: Some(Strategy::Deterministic),
                seed: vec![Seq(2)],
                replaces: Some(Seq(1)..Seq(3)),
            },
            Body::TurnEnd {
                outcome: "finished".into(),
            },
        ] {
            round_trip(&entry(b));
        }
    }

    #[test]
    fn unknown_kinds_are_kept_unless_they_must_be_understood() {
        let line = r#"{"seq":3,"seg":1,"ts":42,"kind":"later.thing","x":[1e5]}"#;
        let e = decode_entry(line).unwrap();
        assert_eq!(e.body.kind().as_str(), "later.thing");
        assert_eq!(encode_entry(&e).unwrap(), line);
        let must = r#"{"seq":3,"seg":1,"ts":42,"kind":"later.thing","must_understand":true}"#;
        assert!(matches!(
            decode_entry(must),
            Err(DecodeError::MustUnderstand { seq: Seq(3), .. })
        ));
    }

    #[test]
    fn a_content_range_outside_the_body_is_refused() {
        let line = r#"{"seq":3,"seg":1,"ts":42,"kind":"model.response","body":{"content":[]},"content":[0,99],"usage":{}}"#;
        assert!(matches!(decode_entry(line), Err(DecodeError::Bad(_))));
    }

    #[test]
    fn headers_check_their_version() {
        let h = crate::new_header(crate::new_session_id(), 1, None);
        let line = encode_header(&h).unwrap();
        assert!(
            line.starts_with(r#"{"sarathi_log":1,"session":""#),
            "{line}"
        );
        assert_eq!(decode_header(&line).unwrap(), h);
        let newer = line.replacen("\"sarathi_log\":1", "\"sarathi_log\":2", 1);
        assert!(matches!(
            decode_header(&newer),
            Err(LogError::Format(
                sarathi_protocol::VersionError::NewerFormat { found: 2, .. }
            ))
        ));
        assert!(matches!(
            decode_header("{}"),
            Err(LogError::Corrupt { line: 0, .. })
        ));
    }
}
