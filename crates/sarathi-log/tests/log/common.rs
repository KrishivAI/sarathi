// SPDX-License-Identifier: Apache-2.0
//! Shared test helpers: an entry builder that tracks seqs and segments, and a toy request assembler.

use sarathi_log::{
    Body, CallRef, Effect, Entry, Log, RequestParts, SegmentReason, Slice, Strategy, sha256_hex,
};
use sarathi_protocol::{RawJson, Seq, Usage};

pub(crate) fn raw(s: &str) -> RawJson {
    RawJson::from_text(s).unwrap()
}

/// The toy assembler: `{"system":…,"tools":…,"messages":[…],"params":…}`, consecutive tool results grouped into one
/// user message, as the Messages API wants. Byte-exact concatenation, like the real one.
pub(crate) fn assemble(p: &RequestParts<'_>) -> Vec<u8> {
    let mut out = String::new();
    out.push_str(r#"{"system":"#);
    out.push_str(p.system.as_str());
    out.push_str(r#","tools":"#);
    out.push_str(p.tools.as_str());
    out.push_str(r#","messages":["#);
    let mut first = true;
    let mut open_results = false;
    for (seq, s) in &p.messages {
        match s {
            Slice::ToolResult { block, .. } => {
                if open_results {
                    out.push(',');
                } else {
                    if !first {
                        out.push(',');
                    }
                    out.push_str(r#"{"role":"user","content":["#);
                    open_results = true;
                }
                out.push_str(block.as_str());
            }
            other => {
                if open_results {
                    out.push_str("]}");
                    open_results = false;
                }
                if !first {
                    out.push(',');
                }
                out.push_str(other.to_message(*seq).unwrap().bytes.as_str());
            }
        }
        first = false;
    }
    if open_results {
        out.push_str("]}");
    }
    out.push_str(r#"],"params":"#);
    out.push_str(p.params.as_str());
    out.push('}');
    out.into_bytes()
}

/// Builds entries in order against a [`Log`] mirror, so seqs, segments and recipes are always right.
pub(crate) struct Script {
    pub(crate) log: Log,
    pub(crate) ts: u64,
}

impl Script {
    pub(crate) fn new(log: Log) -> Self {
        Script {
            log,
            ts: 1_764_057_600_000,
        }
    }

    /// Appends one body to the mirror and returns the entry for the store.
    pub(crate) fn push(&mut self, body: Body) -> Entry {
        let seg = match body {
            Body::SegmentStart { .. } => self.log.segment() + 1,
            _ => self.log.segment(),
        };
        self.ts += 1;
        let e = Entry {
            seq: self.log.next_seq(),
            seg,
            ts: self.ts,
            body,
        };
        self.log.append(e.seq, vec![e.clone()]).unwrap();
        e
    }

    pub(crate) fn prefix(&mut self) -> Entry {
        self.push(Body::SessionPrefix {
            protocol: "anthropic-messages".into(),
            system: raw(r#"[{"type":"text","text":"You edit themes.","cache_control":{"type":"ephemeral"}}]"#),
            tools: raw(r#"[{"name":"edit","input_schema":{"type":"object"}},{"name":"look","input_schema":{"type":"object"}}]"#),
        })
    }

    pub(crate) fn user(&mut self, text: &str) -> Entry {
        self.push(Body::UserMessage {
            content: RawJson::string(text),
            steer: false,
        })
    }

    pub(crate) fn request(&mut self) -> Entry {
        let recipe = self
            .log
            .next_recipe(raw(r#"{"model":"m","max_tokens":1024}"#))
            .unwrap();
        let sha256 = sha256_hex(&assemble(&self.log.request_parts(&recipe).unwrap()));
        self.push(Body::ModelRequest { recipe, sha256 })
    }

    /// A response whose content is `content` (a JSON array), and the assistant message accepting it.
    pub(crate) fn respond(&mut self, content: &str, calls: &[(&str, &str)]) -> Vec<Entry> {
        let body = format!(
            r#"{{"id":"msg_1","type":"message","role":"assistant","content":{content},"stop_reason":"{}","usage":{{"input_tokens":10,"output_tokens":5}}}}"#,
            if calls.is_empty() {
                "end_turn"
            } else {
                "tool_use"
            }
        );
        let start = u32::try_from(body.find(r#""content":"#).unwrap() + 10).unwrap();
        let end = start + u32::try_from(content.len()).unwrap();
        let r = self.push(Body::ModelResponse {
            body: RawJson::from_string(body).unwrap(),
            content: start..end,
            sse: None,
            request_id: Some("req_1".into()),
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
                ..Usage::default()
            },
        });
        let a = self.push(Body::AssistantMessage {
            from: r.seq,
            calls: calls
                .iter()
                .map(|(id, name)| CallRef {
                    id: (*id).into(),
                    name: (*name).into(),
                })
                .collect(),
        });
        vec![r, a]
    }

    pub(crate) fn start(&mut self, call: &str, tool: &str, effect: Effect) -> Entry {
        self.push(Body::ToolStart {
            call: call.into(),
            tool: tool.into(),
            effect,
        })
    }

    pub(crate) fn result(&mut self, call: &str, text: &str) -> Entry {
        let block = format!(
            r#"{{"type":"tool_result","tool_use_id":"{call}","content":{}}}"#,
            RawJson::string(text)
        );
        self.push(Body::ToolResult {
            call: call.into(),
            block: RawJson::from_string(block).unwrap(),
            is_error: false,
            spill: None,
        })
    }

    pub(crate) fn end(&mut self) -> Entry {
        self.push(Body::TurnEnd {
            outcome: "finished".into(),
        })
    }

    pub(crate) fn compact(&mut self, seed: Vec<Seq>, replaces: std::ops::Range<Seq>) -> Entry {
        self.push(Body::SegmentStart {
            reason: SegmentReason::Compaction,
            strategy: Some(Strategy::Deterministic),
            seed,
            replaces: Some(replaces),
        })
    }

    /// A whole turn: user text, a request, a response with one write call, its start and result, a request, a final
    /// response, the end. Returns the entries in order.
    pub(crate) fn tool_turn(&mut self, text: &str, call: &str) -> Vec<Entry> {
        let mut v = vec![self.user(text), self.request()];
        v.extend(self.respond(
            &format!(
                r#"[{{"type":"thinking","thinking":"","signature":"sig"}},{{"type":"tool_use","id":"{call}","name":"edit","input":{{"path":"a.css","z":1e5}}}}]"#
            ),
            &[(call, "edit")],
        ));
        v.push(self.start(call, "edit", Effect::Write));
        v.push(self.result(call, "ok \u{e9}"));
        v.push(self.request());
        v.extend(self.respond(
            r#"[{"type":"text","text":"Done."},{"type":"future_block","x":[1.0]}]"#,
            &[],
        ));
        v.push(self.end());
        v
    }
}
