// SPDX-License-Identifier: Apache-2.0
//! The core's records as log entries, and a core rebuilt from a log.

use sarathi_log::{Body, CallRef, CallStatus, Log};
use sarathi_protocol::{CallId, Decision, ErrorKind, Outcome, RawJson, Seq, TurnId};
use serde::Serialize;

use super::DriverConfig;
use super::seams::{CallFailure, ModelCaller, Received};
use crate::core::{Action, CUT_SHORT, Core, CoreConfig, OpenTurn, Record, ResumeAt};
use crate::profile::{Effect, StepProtocol, protocol_text};
use crate::step;

/// What the driver holds for the records that need more than the core knows.
#[derive(Debug, Default)]
pub(super) struct Held {
    /// The answer of the call just made, until its `model.response` is written.
    pub(super) answer: Option<Received>,
    /// The failure of the call just made, until its `model.failure` is written.
    pub(super) failure: Option<CallFailure>,
}

/// A record as the log's entry body.
pub(super) fn body_of<C: ModelCaller>(
    record: Record,
    caller: &C,
    protocol: StepProtocol,
    held: &mut Held,
) -> Result<Body, String> {
    Ok(match record {
        Record::UserMessage { content, steer } => Body::UserMessage { content, steer },
        Record::ModelResponse => {
            let r = held
                .answer
                .take()
                .ok_or("a model.response was asked for with no answer held")?;
            Body::ModelResponse {
                body: r.body,
                content: r.content,
                sse: r.sse,
                request_id: r.request_id,
                usage: r.view.usage,
            }
        }
        Record::ModelFailure { kind, detail } => {
            let f = held.failure.take().filter(|f| f.kind == kind);
            Body::ModelFailure {
                error: kind.as_str().into(),
                status: f.as_ref().and_then(|f| f.status),
                detail: detail.into(),
                request_id: f.as_ref().and_then(|f| f.request_id.clone()),
                retries: f.map_or(0, |f| f.retries),
            }
        }
        Record::AssistantMessage { from, calls } => Body::AssistantMessage {
            from,
            calls: calls
                .into_iter()
                .map(|c| CallRef {
                    id: c.call.as_str().into(),
                    name: c.tool,
                })
                .collect(),
        },
        Record::ToolAsk { call, tool, input } => Body::ToolAsk {
            call: call.as_str().into(),
            tool,
            input,
        },
        Record::ToolDecision { call, decision } => Body::ToolDecision {
            call: call.as_str().into(),
            decision: RawJson::render(&decision).map_err(|e| e.to_string())?,
        },
        Record::ToolStart { call, tool, effect } => Body::ToolStart {
            call: call.as_str().into(),
            tool,
            effect: match effect {
                Effect::Read => sarathi_log::Effect::Read,
                Effect::Write => sarathi_log::Effect::Write,
            },
        },
        Record::ToolResult {
            call,
            content,
            is_error,
        } => Body::ToolResult {
            block: match protocol {
                StepProtocol::Native => caller.result_block(&call, &content, is_error),
                StepProtocol::JsonObject => text_block(&content)?,
            },
            call: call.as_str().into(),
            is_error,
            spill: None,
        },
        Record::TurnEnd { outcome } => Body::TurnEnd {
            outcome: name_of(&outcome)?,
        },
    })
}

/// A text block of `content`, a JSON string (an observation under the JSON-object protocol).
fn text_block(content: &RawJson) -> Result<RawJson, String> {
    let text = if content.as_str().starts_with('"') {
        content.clone()
    } else {
        RawJson::string(content.as_str())
    };
    RawJson::from_string(format!(r#"{{"type":"text","text":{}}}"#, text.as_str()))
        .map_err(|e| e.to_string())
}

/// The serde name of a unit variant.
fn name_of<T: Serialize>(value: &T) -> Result<Box<str>, String> {
    match serde_json::to_value(value) {
        Ok(serde_json::Value::String(s)) => Ok(s.into()),
        Ok(other) => Err(format!("{other} is not a name")),
        Err(e) => Err(e.to_string()),
    }
}

/// The session prefix for `config`: the system prompt (with the step protocol's text under the JSON-object
/// protocol) and the tools, rendered once by the caller.
///
/// # Errors
/// The caller cannot render the tools.
pub(super) fn prefix_body<C: ModelCaller>(
    config: &DriverConfig,
    caller: &C,
) -> Result<Body, String> {
    let mut system = config.system.to_string();
    let tools = match config.profile.protocol {
        StepProtocol::Native => caller
            .render_tools(&config.tools)
            .map_err(|f| f.detail.to_string())?,
        StepProtocol::JsonObject => {
            system.push_str(&protocol_text(&config.tools));
            None
        }
    };
    Ok(Body::SessionPrefix {
        protocol: caller.protocol().into(),
        system: RawJson::string(&system),
        tools: tools.unwrap_or_else(RawJson::null),
    })
}

/// A `turn.end` with `outcome`.
pub(super) fn turn_end(outcome: Outcome) -> Body {
    Body::TurnEnd {
        outcome: name_of(&outcome).unwrap_or_else(|_| "failed".into()),
    }
}

/// What closes an open turn that failed: a result for every call without one (a started side-effecting call is
/// answered as cut short, the rest as skipped), then `turn.end` `failed`. Empty if no turn is open.
pub(super) fn closing<C: ModelCaller>(
    log: &Log,
    caller: &C,
    protocol: StepProtocol,
) -> Result<Vec<Body>, String> {
    let resume = log.resume();
    if !resume.in_turn {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for c in resume.calls.iter().filter(|c| c.status != CallStatus::Done) {
        let text = if c.status == CallStatus::StartedWrite {
            CUT_SHORT.to_owned()
        } else {
            step::skipped("the turn failed")
        };
        let call = CallId::from(&*c.call.id);
        let block = match protocol {
            StepProtocol::Native => caller.result_block(&call, &RawJson::string(&text), true),
            StepProtocol::JsonObject => {
                text_block(&RawJson::string(&format!("observation:\n{text}")))?
            }
        };
        out.push(Body::ToolResult {
            call: c.call.id.clone(),
            block,
            is_error: true,
            spill: None,
        });
    }
    if resume.pending_model.is_some() {
        // The request that was written has no answer: say so, so the log is at a boundary again.
        out.push(Body::ModelFailure {
            error: ErrorKind::Log.as_str().into(),
            status: None,
            detail: "the turn failed before the call was answered".into(),
            request_id: None,
            retries: 0,
        });
    }
    out.push(turn_end(Outcome::Failed));
    Ok(out)
}

/// How a core starts from a log.
#[derive(Debug)]
pub(super) enum Start {
    /// At rest, between turns.
    Rest(Core),
    /// In a turn a restart cut short, with what to do first.
    Open(Core, Vec<Action>),
    /// The turn's last model call failed and the turn has no end: write this end first, then start again.
    Close(Outcome),
}

/// Rebuilds the core from `log`. A turn left open is resumed: calls answered stay answered, a started side-effecting
/// call is answered as cut short and never run again, everything else runs again.
pub(super) fn start<C: ModelCaller>(
    config: &DriverConfig,
    log: &Log,
    caller: &C,
) -> Result<Start, String> {
    let projection = log.projection();
    let prefix = projection.prefix.ok_or("the log has no session.prefix")?;
    if log.get(prefix).map(|e| &e.body) != Some(&prefix_body(config, caller)?) {
        return Err(
            "the session was started with another protocol, system prompt or tools than this config gives".into(),
        );
    }
    let mut core_config = CoreConfig::new(config.profile.clone(), config.tools.clone(), prefix);
    core_config.limits = config.limits.clone();
    core_config.history = projection.messages;
    core_config.next_seq = log.next_seq();
    core_config.allowed_always = allowed_always(log);

    let resume = log.resume();
    if !resume.in_turn {
        return Ok(Start::Rest(Core::new(core_config)));
    }
    let entries = log.entries();
    let turn = entries
        .iter()
        .rev()
        .find(|e| matches!(e.body, Body::UserMessage { steer: false, .. }))
        .map(|e| e.seq)
        .ok_or("an open turn without its user message")?;
    let in_turn = || entries.iter().filter(move |e| e.seq > turn);
    let steps = u32::try_from(
        in_turn()
            .filter(|e| matches!(e.body, Body::AssistantMessage { .. }))
            .count(),
    )
    .unwrap_or(u32::MAX);
    let open = |at| OpenTurn::new(TurnId(turn), steps, at);

    if resume.pending_model.is_some() {
        let (core, actions) = Core::resume(core_config, open(ResumeAt::Call));
        return Ok(Start::Open(core, actions));
    }
    // The last thing that happened to a model call of the turn's own: compaction calls and their failures are
    // passed over (a compaction call that failed leaves the turn going).
    let last = in_turn().rev().find(|e| match &e.body {
        Body::ModelRequest { compaction, .. } => compaction.is_none(),
        Body::AssistantMessage { .. } => true,
        Body::ModelFailure { .. } => !answers_compaction(log, e.seq),
        _ => false,
    });
    let assistant = match last.map(|e| (e.seq, &e.body)) {
        Some((_, b @ Body::ModelFailure { .. })) => {
            return Ok(Start::Close(match b.error_kind() {
                Some(ErrorKind::Cancelled) => Outcome::Interrupted,
                Some(ErrorKind::BudgetExceeded) => Outcome::RanOut,
                _ => Outcome::Failed,
            }));
        }
        Some((seq, Body::AssistantMessage { from, .. })) => Some((seq, *from)),
        _ => None,
    };
    let all_done = resume.calls.iter().all(|c| c.status == CallStatus::Done);
    let added_since = |since: Seq| {
        entries
            .iter()
            .any(|e| e.seq > since && matches!(e.body, Body::UserMessage { .. }))
    };
    // A segment that began after the answer (a compaction or a fallback seed inside the turn) holds the turn's
    // results already: the next thing is the call.
    let reseated = |since: Seq| {
        entries
            .iter()
            .any(|e| e.seq > since && matches!(e.body, Body::SegmentStart { .. }))
    };
    let at = match assistant {
        Some((seq, _)) if all_done && (added_since(seq) || reseated(seq)) => ResumeAt::Call,
        None => ResumeAt::Call,
        Some((seq, from)) => {
            let Some(Body::ModelResponse { body, content, .. }) = log.get(from).map(|e| &e.body)
            else {
                return Err(format!(
                    "message.assistant {seq} does not point at a model.response"
                ));
            };
            let answer = caller
                .read(body, content.clone())
                .map_err(|f| format!("the answer at {from} cannot be read: {}", f.detail))?
                .answer();
            let answer = Box::new(answer);
            let ids = |want: CallStatus| {
                resume
                    .calls
                    .iter()
                    .filter(|c| c.status == want)
                    .map(|c| CallId::from(&*c.call.id))
                    .collect()
            };
            ResumeAt::Answer {
                assistant: seq,
                answer,
                answered: ids(CallStatus::Done),
                cut_short: ids(CallStatus::StartedWrite),
            }
        }
    };
    let (core, actions) = Core::resume(core_config, open(at));
    Ok(Start::Open(core, actions))
}

/// The `model.failure` at `failure` answers a compaction call: the nearest `model.request` before it is one.
fn answers_compaction(log: &Log, failure: Seq) -> bool {
    log.entries()
        .iter()
        .rev()
        .skip_while(|e| e.seq >= failure)
        .find_map(|e| match &e.body {
            Body::ModelRequest { compaction, .. } => Some(compaction.is_some()),
            _ => None,
        })
        .unwrap_or(false)
}

/// Tools the host allowed for the rest of the session.
fn allowed_always(log: &Log) -> Vec<Box<str>> {
    let mut asked: Vec<(&str, &str)> = Vec::new();
    let mut out: Vec<Box<str>> = Vec::new();
    for e in log.entries() {
        match &e.body {
            Body::ToolAsk { call, tool, .. } => asked.push((call, tool)),
            Body::ToolDecision { call, decision } => {
                let always = matches!(
                    serde_json::from_str::<Decision>(decision.as_str()),
                    Ok(Decision::AllowAlways)
                );
                if let Some((_, tool)) = asked.iter().rev().find(|(c, _)| c == &&**call)
                    && always
                    && !out.iter().any(|t| &**t == *tool)
                {
                    out.push((*tool).into());
                }
            }
            _ => {}
        }
    }
    out
}
