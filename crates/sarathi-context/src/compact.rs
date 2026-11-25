// SPDX-License-Identifier: Apache-2.0
//! Compaction by a model call: the request ([`compaction_request`]), the new segment made from its answer
//! ([`compaction_seed`]) and how a restarted run finds a compaction call left open ([`pending_compaction`]).
//!
//! Two compactors:
//! - [`Compactor::Server`]: the provider summarizes (Anthropic on-demand compaction). The request is the
//!   conversation as it stands plus a `compaction` parameter; the answer is one `compaction` block, which becomes the
//!   whole seed of the next segment, byte for byte (the provider rejects a block whose bytes changed).
//! - [`Compactor::ClientSummary`]: any protocol. The request is the conversation plus a fixed summary prompt; the
//!   answer's text becomes one user message, the whole seed of the next segment. No thinking is carried over.
//!
//! Both are logged like any call (`model.request`, `model.response`); the answer goes into the seed instead of a
//! `message.assistant`, and the seed's `segment.start` answers the request for resume.

use std::collections::BTreeMap;
use std::ops::Range;

use sarathi_log::{Body, Entry, Log, Recipe, SegmentId, SegmentReason, Strategy};
use sarathi_protocol::{RawJson, Role, Seq};
use serde::Deserialize;
use serde_json::value::RawValue;

use crate::error::ContextError;
use crate::fallback::{Seed, next_seq, out_of_seqs};

/// The beta flag a request needs when it asks for server compaction or carries a compaction block
/// ([`carries_compaction_block`]).
pub const COMPACTION_BETA: &str = "compact-2026-09-04";

/// The longest `instructions` a server compaction takes, in characters.
pub const MAX_INSTRUCTIONS: usize = 16_384;

/// The user message a client summary request ends with.
pub const SUMMARY_PROMPT: &str = "Stop the task for a moment and write a handoff summary of this conversation for \
whoever continues it, who will see only your summary. Do not call any tools. Include: the user's requests and \
constraints, in their words where they matter; what has been done, with the files, commands and results that \
matter; decisions made and why; what is still open or failing; and the exact next step. Leave out greetings and \
anything already superseded. Answer with the summary only.";

/// What a client summary seed starts with, before the summary's text.
pub const SUMMARY_PREFIX: &str = "[The earlier conversation was replaced by this summary to fit the context \
window.]\n\n";

/// Who writes the summary a compaction seeds the next segment with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Compactor {
    /// The provider, with its own compaction (a `compaction` block).
    Server,
    /// The model, asked with [`SUMMARY_PROMPT`]; the answer becomes a user message.
    ClientSummary,
}

impl Compactor {
    /// The strategy the seed's `segment.start` records.
    pub fn strategy(self) -> Strategy {
        match self {
            Compactor::Server => Strategy::Server,
            Compactor::ClientSummary => Strategy::ClientSummary,
        }
    }

    /// The beta flag its compaction call needs, if any.
    pub fn beta(self) -> Option<&'static str> {
        match self {
            Compactor::Server => Some(COMPACTION_BETA),
            Compactor::ClientSummary => None,
        }
    }

    fn of(strategy: Strategy) -> Option<Compactor> {
        match strategy {
            Strategy::Server => Some(Compactor::Server),
            Strategy::ClientSummary => Some(Compactor::ClientSummary),
            _ => None,
        }
    }
}

/// A compaction call to make: render [`recipe`](Self::recipe), append [`entries`](Self::entries) and
/// [`model_request`](Self::model_request) in one batch at [`Log::next_seq`], then make the call. The request is marked
/// as a compaction call, so its answer can only become a seed ([`compaction_seed`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionRequest {
    /// Who summarizes.
    pub compactor: Compactor,
    /// Entries to append before the request (a client summary's prompt), possibly none.
    pub entries: Vec<Entry>,
    /// The request.
    pub recipe: Recipe,
    /// A beta flag the request needs, if any.
    pub beta: Option<&'static str>,
}

impl CompactionRequest {
    /// The `model.request` body to log for this call, with the SHA-256 of the body as sent.
    pub fn model_request(&self, sha256: String) -> Body {
        Body::ModelRequest {
            recipe: self.recipe.clone(),
            sha256,
            compaction: Some(self.compactor.strategy()),
        }
    }
}

/// The compaction call for the log as it stands, with the turn's `params` (same model and settings as a turn).
///
/// For [`Compactor::Server`], `params` gets a `compaction` member (`{"type":"summarize"}`, with `instructions` if
/// given); for [`Compactor::ClientSummary`], a `segment.seed` user message with [`SUMMARY_PROMPT`] is appended and
/// sent last. `instructions` are for the server only and ignored by the client summary; an empty one is none.
///
/// A server compaction runs only between turns: its block must be followed by the next user message. A client
/// summary may also run inside a turn (after a tool round): the model continues from the summary.
///
/// # Errors
/// [`ContextError::NotAtBoundary`] if a tool call has no result or a model call is in flight, or, for the server, a
/// turn is open;
/// [`ContextError::BadParams`] if `params` is not a JSON object or already has a `compaction` member;
/// [`ContextError::InstructionsTooLong`] past [`MAX_INSTRUCTIONS`]; [`ContextError::Log`] if no `session.prefix` was
/// written.
pub fn compaction_request(
    log: &Log,
    compactor: Compactor,
    params: &RawJson,
    instructions: Option<&str>,
    ts: u64,
) -> Result<CompactionRequest, ContextError> {
    let resume = log.resume();
    if !resume.at_boundary() || (compactor == Compactor::Server && resume.in_turn) {
        return Err(ContextError::NotAtBoundary);
    }
    match compactor {
        Compactor::Server => {
            let mut member = String::from(r#"{"type":"summarize""#);
            if let Some(i) = instructions.filter(|i| !i.is_empty()) {
                let chars = i.chars().count();
                if chars > MAX_INSTRUCTIONS {
                    return Err(ContextError::InstructionsTooLong(chars));
                }
                member.push_str(r#","instructions":"#);
                member.push_str(RawJson::string(i).as_str());
            }
            member.push('}');
            let params = with_member(params, "compaction", &member)?;
            Ok(CompactionRequest {
                compactor,
                entries: Vec::new(),
                recipe: log.next_recipe(params)?,
                beta: compactor.beta(),
            })
        }
        Compactor::ClientSummary => {
            let seq = log.next_seq();
            let prompt = Entry {
                seq,
                seg: log.segment(),
                ts,
                body: Body::SegmentSeed {
                    role: Role::User,
                    content: RawJson::string(SUMMARY_PROMPT),
                    from: None,
                    calls: Vec::new(),
                },
            };
            let mut recipe = log.next_recipe(params.clone())?;
            recipe.messages.push(seq);
            Ok(CompactionRequest {
                compactor,
                entries: vec![prompt],
                recipe,
                beta: compactor.beta(),
            })
        }
    }
}

/// A compaction call the log shows as made and not yet turned into a segment: a restarted run continues from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PendingCompaction {
    /// The `model.request`.
    pub request: Seq,
    /// Who was asked.
    pub compactor: Compactor,
    /// The `model.response`, if one was logged: call [`compaction_seed`] instead of calling again.
    pub response: Option<Seq>,
}

/// The compaction call in flight, if the log's unanswered `model.request` is one.
pub fn pending_compaction(log: &Log) -> Option<PendingCompaction> {
    let request = log.resume().pending_model?;
    let compactor = compactor_of(log, request)?;
    let response = log
        .entries()
        .iter()
        .rev()
        .take_while(|e| e.seq > request)
        .find(|e| matches!(e.body, Body::ModelResponse { .. }))
        .map(|e| e.seq);
    Some(PendingCompaction {
        request,
        compactor,
        response,
    })
}

/// Which compactor a `model.request` asked, if it is a compaction call (its `compaction` field).
pub(crate) fn compactor_of(log: &Log, request: Seq) -> Option<Compactor> {
    match log.get(request).map(|e| &e.body) {
        Some(Body::ModelRequest {
            compaction: Some(c),
            ..
        }) => Compactor::of(*c),
        _ => None,
    }
}

/// The new segment made from the pending compaction call's answer ([`pending_compaction`]).
///
/// Server: the seed is a `segment.seed` assistant slice holding the response's `content` byte for byte, which must be
/// exactly one `compaction` block with non-empty `content` and `signature`. Client summary: the seed is a
/// `segment.seed` user message, [`SUMMARY_PREFIX`] and the answer's text; an answer that was cut short, refused or
/// asks for a tool is no summary. Either way `replaces` covers the old segment, and the `segment.start`
/// answers the request. Append the entries in one batch at [`Log::next_seq`].
///
/// # Errors
/// [`ContextError::NoCompaction`] if no compaction call is pending or it has no response yet;
/// [`ContextError::CompactionFailed`] if the answer is not a usable summary (an empty block, a reply, a refusal):
/// log a `model.failure` for the request, keep the segment, and try again later or fall back to
/// [`crate::deterministic_seed`].
pub fn compaction_seed(log: &Log, ts: u64) -> Result<Seed, ContextError> {
    let pending = pending_compaction(log).ok_or(ContextError::NoCompaction)?;
    let response = pending.response.ok_or(ContextError::NoCompaction)?;
    let Some(Body::ModelResponse { body, content, .. }) = log.get(response).map(|e| &e.body) else {
        return Err(ContextError::NoCompaction);
    };
    let blocks = content_of(body, content).ok_or_else(|| failed(body))?;
    let (role, content) = match pending.compactor {
        Compactor::Server => {
            if !is_compaction_block_list(&blocks) {
                return Err(failed(body));
            }
            (Role::Assistant, blocks)
        }
        Compactor::ClientSummary => {
            let text = text_of(&blocks);
            let cut = stop_of(body).is_some_and(|r| NOT_A_SUMMARY.contains(&r.as_str()));
            if cut || text.trim().is_empty() || block_types(&blocks).iter().any(|t| t == "tool_use")
            {
                return Err(failed(body));
            }
            (
                Role::User,
                RawJson::string(&format!("{SUMMARY_PREFIX}{text}")),
            )
        }
    };
    let seg = log.segment();
    let mut next = log.next_seq();
    let s = next_seq(&mut next)?;
    let first_of_segment = log
        .entries()
        .iter()
        .find(|e| e.seg == seg)
        .map_or(Seq(1), |e| e.seq);
    let new_seg = seg.0.checked_add(1).ok_or_else(|| out_of_seqs(next))?;
    let entries = vec![
        Entry {
            seq: s,
            seg,
            ts,
            body: Body::SegmentSeed {
                role,
                content,
                from: Some(response),
                calls: Vec::new(),
            },
        },
        Entry {
            seq: next,
            seg: SegmentId(new_seg),
            ts,
            body: Body::SegmentStart {
                reason: SegmentReason::Compaction,
                strategy: Some(pending.compactor.strategy()),
                seed: vec![s],
                replaces: Some(first_of_segment..s),
            },
        },
    ];
    Ok(Seed {
        entries,
        kept_turns: 0,
        kept_last: false,
        kept_head: false,
    })
}

/// The projection sends a server compaction block, so every request needs [`COMPACTION_BETA`].
pub fn carries_compaction_block(log: &Log) -> bool {
    log.projection().messages.iter().any(|s| {
        matches!(log.slice(*s), Ok(sarathi_log::Slice::Assistant { content, .. })
            if block_types(&content).iter().any(|t| t == "compaction"))
    })
}

/// Stop reasons of an answer that is no complete summary: cut short, refused or asking for a tool (Anthropic
/// `stop_reason`, OpenAI `finish_reason`).
const NOT_A_SUMMARY: &[&str] = &[
    "max_tokens",
    "model_context_window_exceeded",
    "refusal",
    "tool_use",
    "pause_turn",
    "length",
    "content_filter",
    "tool_calls",
    "function_call",
];

/// The answer's stop reason: Anthropic `stop_reason`, or the first OpenAI choice's `finish_reason`.
fn stop_of(body: &RawJson) -> Option<String> {
    #[derive(Deserialize)]
    struct Choice {
        #[serde(default)]
        finish_reason: Option<String>,
    }
    #[derive(Deserialize)]
    struct Stop {
        #[serde(default)]
        stop_reason: Option<String>,
        #[serde(default)]
        choices: Vec<Choice>,
    }
    let s = serde_json::from_str::<Stop>(body.as_str()).ok()?;
    s.stop_reason
        .or_else(|| s.choices.into_iter().next().and_then(|c| c.finish_reason))
}

fn failed(body: &RawJson) -> ContextError {
    ContextError::CompactionFailed {
        stop_reason: stop_of(body),
    }
}

fn content_of(body: &RawJson, range: &Range<u32>) -> Option<RawJson> {
    let start = usize::try_from(range.start).ok()?;
    let end = usize::try_from(range.end).ok()?;
    RawJson::from_text(body.as_str().get(start..end)?).ok()
}

#[derive(Deserialize)]
struct Kind<'a> {
    #[serde(rename = "type", borrow, default)]
    kind: Option<std::borrow::Cow<'a, str>>,
}

#[derive(Deserialize)]
struct TextBlock<'a> {
    #[serde(rename = "type", borrow, default)]
    kind: Option<std::borrow::Cow<'a, str>>,
    #[serde(borrow, default)]
    text: Option<std::borrow::Cow<'a, str>>,
}

#[derive(Deserialize)]
struct CompactionBlock<'a> {
    #[serde(rename = "type", borrow, default)]
    kind: Option<std::borrow::Cow<'a, str>>,
    #[serde(borrow, default)]
    content: Option<std::borrow::Cow<'a, str>>,
    #[serde(borrow, default)]
    signature: Option<std::borrow::Cow<'a, str>>,
}

fn block_types(content: &RawJson) -> Vec<String> {
    serde_json::from_str::<Vec<Kind<'_>>>(content.as_str())
        .map(|bs| {
            bs.into_iter()
                .filter_map(|b| b.kind.map(std::borrow::Cow::into_owned))
                .collect()
        })
        .unwrap_or_default()
}

/// Exactly one block, a `compaction` block with non-empty `content` text and `signature`.
fn is_compaction_block_list(content: &RawJson) -> bool {
    matches!(
        serde_json::from_str::<Vec<CompactionBlock<'_>>>(content.as_str()).as_deref(),
        Ok([b]) if b.kind.as_deref() == Some("compaction")
            && b.content.as_deref().is_some_and(|c| !c.trim().is_empty())
            && b.signature.as_deref().is_some_and(|g| !g.is_empty())
    )
}

/// A content value's text: the string itself, or its `text` blocks joined by blank lines.
fn text_of(content: &RawJson) -> String {
    if let Ok(s) = serde_json::from_str::<String>(content.as_str()) {
        return s;
    }
    serde_json::from_str::<Vec<TextBlock<'_>>>(content.as_str())
        .map(|bs| {
            bs.iter()
                .filter(|b| b.kind.as_deref() == Some("text"))
                .filter_map(|b| b.text.as_deref())
                .collect::<Vec<_>>()
                .join("\n\n")
        })
        .unwrap_or_default()
}

/// `params` (a JSON object) with one more member, the other bytes unchanged.
fn with_member(params: &RawJson, name: &str, value: &str) -> Result<RawJson, ContextError> {
    let bad = |why: &str| ContextError::BadParams(why.to_owned());
    let members = serde_json::from_str::<BTreeMap<String, &RawValue>>(params.as_str())
        .map_err(|_| bad("not a JSON object"))?;
    if members.contains_key(name) {
        return Err(ContextError::BadParams(format!(
            "already has a `{name}` member"
        )));
    }
    let text = params.as_str().trim_end();
    let body = text
        .strip_suffix('}')
        .ok_or_else(|| bad("not a JSON object"))?;
    let sep = if members.is_empty() { "" } else { "," };
    let key = RawJson::string(name);
    Ok(RawJson::from_string(format!(
        "{body}{sep}{}:{value}}}",
        key.as_str()
    ))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(s: &str) -> RawJson {
        RawJson::from_text(s).unwrap()
    }

    #[test]
    fn a_member_is_added_and_the_other_bytes_kept() {
        let p = with_member(&raw(r#"{"model":"m", "z":1e5 }"#), "compaction", "{}").unwrap();
        assert_eq!(p.as_str(), r#"{"model":"m", "z":1e5 ,"compaction":{}}"#);
        let p = with_member(&raw("{ } "), "compaction", "{}").unwrap();
        assert_eq!(p.as_str(), r#"{ "compaction":{}}"#);
    }

    #[test]
    fn params_that_are_not_an_object_or_already_compact_are_refused() {
        for bad in ["[]", r#""x""#, r#"{"compaction":{}}"#] {
            assert!(matches!(
                with_member(&raw(bad), "compaction", "{}"),
                Err(ContextError::BadParams(_))
            ));
        }
    }

    #[test]
    fn only_one_non_empty_compaction_block_is_a_summary() {
        assert!(is_compaction_block_list(&raw(
            r#"[{"type":"compaction","content":"sum","signature":"s"}]"#
        )));
        for no in [
            r#"[{"type":"compaction","content":"","signature":"s"}]"#,
            r#"[{"type":"compaction","content":null,"signature":"s"}]"#,
            r#"[{"type":"compaction","content":"a"}]"#,
            r#"[{"type":"compaction","content":"a","signature":""}]"#,
            r#"[{"type":"text","text":"hi"}]"#,
            r#"[{"type":"compaction","content":"a","signature":"s"},{"type":"text","text":"b"}]"#,
            "[]",
            r#""text""#,
        ] {
            assert!(!is_compaction_block_list(&raw(no)), "{no}");
        }
    }

    #[test]
    fn text_is_read_from_a_string_or_text_blocks() {
        assert_eq!(text_of(&raw(r#""plain""#)), "plain");
        assert_eq!(
            text_of(&raw(
                r#"[{"type":"thinking","thinking":"x"},{"type":"text","text":"a"},{"type":"text","text":"b"}]"#
            )),
            "a\n\nb"
        );
        assert_eq!(text_of(&raw("{}")), "");
    }
}
