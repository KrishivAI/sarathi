// SPDX-License-Identifier: Apache-2.0
//! Context management in the driver: the next request is checked against the model's window
//! ([`sarathi_context::before_request`]) and, when it is due, compacted by a model call or cut down by the
//! deterministic fallback, each making a new segment; tool output over the spill limit is stored outside the log.
//!
//! The check runs at two points: between turns, before the next user message is written (so a server compaction
//! block is followed by it), and inside a turn before each model call (a client summary or the overflow fallback
//! only). A compaction call a restart left open is finished first ([`sarathi_context::pending_compaction`]): an answer
//! already logged becomes the seed, otherwise the logged request is sent again.
//!
//! The compaction call is awaited in place: ops sent meanwhile (an interrupt) are read once it returns. In a turn
//! the next step reads them; between turns, an interrupt sent during the check ends the turn before it starts
//! ([`super::DriverError::Interrupted`]).
//!
//! A request that cannot fit even after the fallback is not sent: between turns the op is refused
//! ([`super::DriverError::DoesNotFit`]); in a turn the call is aborted, so the turn ends failed with kind `guard`.

use sarathi_context::{
    Compactor, ContextAction, ContextError, ContextProfile, before_request, compaction_request,
    compaction_seed, estimate, overflow_seed, pending_compaction, spill_output,
};
use sarathi_log::{Body, Entry, LogStore, SessionId, sha256_hex};
use sarathi_protocol::{ErrorKind, Event, RawJson, Seq};

use super::seams::{CallFailure, Erased, ModelCaller, Spill};
use super::{Writer, now_ms};
use crate::core::{Durability, Record};

/// Passes of the check at most: each pass ends it or makes a segment, and a segment made by a compaction cannot
/// compact again before its first answer, so a few always suffice.
const MAX_PASSES: usize = 6;

/// What the check left.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Checked {
    /// Something was appended: the core must take the new segment.
    pub(super) changed: bool,
    /// The next request cannot fit the window even after the fallback dropped all it may: it must not be sent.
    pub(super) does_not_fit: bool,
    /// What the compaction calls the check made cost, in micro-dollars (a call without a known cost counts as zero):
    /// the driver adds it to the turn's spend.
    pub(super) spent_micros: u64,
}

/// Brings the log's context within the profile before the next request: finishes a compaction call left open, then
/// counts, compacts or falls back as [`before_request`] says, until the request may go as it is.
///
/// With no profile, only a compaction call left open is finished (a session resumed without the profile it ran with
/// must not send a turn after an unanswered compaction request).
///
/// A log not at a boundary (a model call without an answer that is no compaction call) is left as it is.
///
/// # Errors
/// A write failed: the caller treats it like any failed write.
pub(super) async fn maintain<S: LogStore, C: ModelCaller>(
    w: &mut Writer<'_, S>,
    caller: &C,
    profile: Option<&ContextProfile>,
    on_event: &mut (dyn FnMut(Event) + Send),
) -> Result<Checked, String> {
    let start = w.log.next_seq();
    let mut counted = None;
    let mut does_not_fit = false;
    let mut spent_micros = 0u64;
    let mut decided = false;
    for _ in 0..MAX_PASSES {
        if let Some(p) = pending_compaction(w.log) {
            if p.response.is_none() {
                let cost = resend(w, caller, p.request, on_event).await?;
                spent_micros = spent_micros.saturating_add(cost);
            }
            if pending_compaction(w.log).is_some_and(|p| p.response.is_some()) {
                seed(w, on_event).await?;
            }
            counted = None;
            continue;
        }
        decided = true;
        let Some(profile) = profile else {
            break;
        };
        let action = match before_request(w.log, profile, counted) {
            Ok(a) => a,
            Err(ContextError::NotAtBoundary) => break,
            Err(e) => {
                does_not_fit = matches!(e, ContextError::DoesNotFit);
                report(on_event, &e);
                break;
            }
        };
        match action {
            ContextAction::CountTokens => counted = Some(count(w, caller).await),
            ContextAction::Compact(c) => {
                let Some(cost) = compact(w, caller, c, on_event).await? else {
                    break;
                };
                spent_micros = spent_micros.saturating_add(cost);
                counted = None;
            }
            ContextAction::Overflow => match overflow_seed(w.log, profile, now_ms()) {
                Ok(seed) => {
                    append_seed(w, seed.entries, on_event).await?;
                    counted = None;
                }
                Err(e) => {
                    // The request is over the limit and nothing more can go: sending it would only fail.
                    does_not_fit =
                        matches!(e, ContextError::DoesNotFit | ContextError::NothingToDrop);
                    report(on_event, &e);
                    break;
                }
            },
            _ => break,
        }
        decided = false;
    }
    if !decided {
        on_event(Event::Error {
            kind: ErrorKind::Internal,
            detail: format!("context: the check stopped undecided after {MAX_PASSES} passes")
                .into(),
        });
    }
    Ok(Checked {
        changed: w.log.next_seq() != start,
        does_not_fit,
        spent_micros,
    })
}

/// The next request's size: the caller's count, or the estimate when it cannot count.
async fn count<S: LogStore, C: ModelCaller>(w: &Writer<'_, S>, caller: &C) -> u64 {
    let guess = estimate(w.log);
    let Ok(recipe) = w.log.next_recipe(caller.params()) else {
        return guess;
    };
    let fut = match w.log.request_parts(&recipe) {
        Ok(parts) => caller.count_tokens(&parts),
        Err(_) => None,
    };
    match fut {
        Some(f) => f.await.unwrap_or(guess),
        None => guess,
    }
}

/// Makes the compaction call: appends its entries and `model.request`, sends it, appends the answer or the failure.
/// `Some` with the call's cost (zero when unknown, or when the call failed); `None`: the call could not be built
/// (reported), so the check stops.
async fn compact<S: LogStore, C: ModelCaller>(
    w: &mut Writer<'_, S>,
    caller: &C,
    compactor: Compactor,
    on_event: &mut (dyn FnMut(Event) + Send),
) -> Result<Option<u64>, String> {
    let ts = now_ms();
    let req = match compaction_request(w.log, compactor, &caller.params(), None, ts) {
        Ok(r) => r,
        Err(e) => {
            report(on_event, &e);
            return Ok(None);
        }
    };
    let at = w.log.next_seq();
    // The request names the client summary's prompt, which is not in the log yet: render against a copy holding it.
    let ahead;
    let log = if req.entries.is_empty() {
        &*w.log
    } else {
        let mut copy = w.log.clone();
        if let Err(e) = copy.append(at, req.entries.clone()) {
            report(on_event, &ContextError::Log(e));
            return Ok(None);
        }
        ahead = copy;
        &ahead
    };
    let prepared = match log.request_parts(&req.recipe) {
        Ok(parts) => caller.prepare(&parts),
        Err(e) => {
            report(on_event, &ContextError::Log(e));
            return Ok(None);
        }
    };
    let (body, request) = match prepared {
        Ok(p) => p,
        Err(f) => {
            on_event(Event::Error {
                kind: f.kind,
                detail: format!("the compaction call cannot be built: {}", f.detail).into(),
            });
            return Ok(None);
        }
    };
    let mut entries = req.entries.clone();
    let seq = Seq(at.0 + entries.len() as u64);
    entries.push(Entry {
        seq,
        seg: w.log.segment(),
        ts,
        body: req.model_request(sha256_hex(&body)),
    });
    w.write_entries(at, entries, Durability::Buffered).await?;
    call(w, caller, request, on_event).await.map(Some)
}

/// Sends the compaction request at `request` again, as logged, and gives its cost as [`call`] does. A request that
/// no longer builds to the logged bytes is answered with a failure instead.
async fn resend<S: LogStore, C: ModelCaller>(
    w: &mut Writer<'_, S>,
    caller: &C,
    request: Seq,
    on_event: &mut (dyn FnMut(Event) + Send),
) -> Result<u64, String> {
    let Some(Body::ModelRequest { recipe, sha256, .. }) = w.log.get(request).map(|e| &e.body)
    else {
        return Ok(0);
    };
    let prepared = w
        .log
        .request_parts(recipe)
        .map_err(|e| CallFailure::new(ErrorKind::Unassemblable, e.to_string()))
        .and_then(|parts| caller.prepare(&parts));
    match prepared {
        Ok((body, req)) if sha256_hex(&body) == *sha256 => call(w, caller, req, on_event).await,
        Ok(_) => {
            let f = CallFailure::new(
                ErrorKind::HistoryMismatch,
                "the compaction request no longer builds to the bytes logged",
            );
            failure(w, &f).await.map(|()| 0)
        }
        Err(f) => failure(w, &f).await.map(|()| 0),
    }
}

/// Sends a logged compaction request and appends its `model.response` or `model.failure`. An answer is reported as
/// [`Event::Usage`], like the turn's own calls, and its cost returned (zero when unknown or when the call failed).
async fn call<S: LogStore, C: ModelCaller>(
    w: &mut Writer<'_, S>,
    caller: &C,
    request: C::Prepared,
    on_event: &mut (dyn FnMut(Event) + Send),
) -> Result<u64, String> {
    match caller.send(request).await {
        Ok(r) => {
            let at = w.log.next_seq();
            let usage = r.view.usage;
            let cost_micros = r.cost_micros;
            let body = Body::ModelResponse {
                body: r.body,
                content: r.content,
                sse: r.sse,
                request_id: r.request_id,
                usage,
            };
            w.write(at, vec![body], Durability::Durable).await?;
            on_event(Event::Usage {
                seq: at,
                usage,
                cost_micros,
            });
            Ok(cost_micros.unwrap_or(0))
        }
        Err(f) => failure(w, &f).await.map(|()| 0),
    }
}

async fn failure<S: LogStore>(w: &mut Writer<'_, S>, f: &CallFailure) -> Result<(), String> {
    let at = w.log.next_seq();
    let body = Body::ModelFailure {
        error: f.kind.as_str().into(),
        status: f.status,
        detail: f.detail.to_string(),
        request_id: f.request_id.clone(),
        retries: f.retries,
    };
    w.write(at, vec![body], Durability::Durable).await
}

/// Turns the answered compaction call into a new segment; an answer that is no summary is logged as the call's
/// failure, so the segment goes on and the next check falls back if it must.
async fn seed<S: LogStore>(
    w: &mut Writer<'_, S>,
    on_event: &mut (dyn FnMut(Event) + Send),
) -> Result<(), String> {
    match compaction_seed(w.log, now_ms()) {
        Ok(seed) => append_seed(w, seed.entries, on_event).await,
        Err(e) => {
            report(on_event, &e);
            let kind = match &e {
                ContextError::CompactionFailed {
                    stop_reason: Some(r),
                } if r == "refusal" => ErrorKind::Refusal,
                _ => ErrorKind::Malformed,
            };
            failure(w, &CallFailure::new(kind, e.to_string())).await
        }
    }
}

/// Appends a seed (its entries carry their own seqs and segments) and reports the segment it starts.
async fn append_seed<S: LogStore>(
    w: &mut Writer<'_, S>,
    entries: Vec<Entry>,
    on_event: &mut (dyn FnMut(Event) + Send),
) -> Result<(), String> {
    let Some(at) = entries.first().map(|e| e.seq) else {
        return Ok(());
    };
    let started: Vec<Event> = entries
        .iter()
        .filter_map(|e| match &e.body {
            Body::SegmentStart { reason, .. } => Some(Event::SegmentStarted {
                segment: e.seg,
                reason: *reason,
            }),
            _ => None,
        })
        .collect();
    w.write_entries(at, entries, Durability::Durable).await?;
    for e in started {
        on_event(e);
    }
    Ok(())
}

fn report(on_event: &mut (dyn FnMut(Event) + Send), e: &ContextError) {
    let kind = match e {
        ContextError::DoesNotFit | ContextError::NothingToDrop | ContextError::NoUserMessage => {
            ErrorKind::ContextOverflow
        }
        ContextError::Log(_) => ErrorKind::Log,
        _ => ErrorKind::Internal,
    };
    on_event(Event::Error {
        kind,
        detail: format!("context: {e}").into(),
    });
}

/// Spills the tool results among `records` that are over the limit: each one's content becomes the preview, and
/// the returned list holds, per record, the `spill` field its entry gets. The spill is durable before the entry that
/// names it is written. A spill that fails is reported and the output stays in the log.
///
/// Only string content spills: a result given as an array of blocks stays in the log as it is, since a text preview
/// would turn its non-text blocks (images, documents) into their JSON source.
pub(super) async fn spill_results(
    records: &mut [Record],
    spill: &Spill,
    session: SessionId,
    on_event: &mut (dyn FnMut(Event) + Send),
) -> Vec<Option<RawJson>> {
    let mut out = Vec::with_capacity(records.len());
    for r in records.iter_mut() {
        let mut field = None;
        if let Record::ToolResult { call, content, .. } = r {
            let Ok(text) = serde_json::from_str::<String>(content.as_str()) else {
                out.push(None);
                continue;
            };
            let store = Erased(spill.store.clone());
            match spill_output(&store, session, call.as_str(), &text, &spill.limits).await {
                Ok(Some(s)) => match s.record.to_raw() {
                    Ok(raw) => {
                        *content = RawJson::string(&s.preview);
                        field = Some(raw);
                    }
                    Err(e) => report(on_event, &e),
                },
                Ok(None) => {}
                Err(e) => report(on_event, &e),
            }
        }
        out.push(field);
    }
    out
}
