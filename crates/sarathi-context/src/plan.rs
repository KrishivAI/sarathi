// SPDX-License-Identifier: Apache-2.0
//! When to compact: the size estimate ([`estimate`]) and the decision before a request ([`before_request`]).

use sarathi_log::{Body, Log, Slice};
use sarathi_protocol::{ErrorKind, Seq};

use crate::compact::{Compactor, block_types, compactor_of};
use crate::error::ContextError;
use crate::fallback::{Seed, shrink_to_fit};

/// How one model's context is managed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextProfile {
    /// The context window, in tokens.
    pub window: u64,
    /// Tokens kept free for the answer: a request over `window - max_output` does not fit.
    pub max_output: u64,
    /// Compact once the estimate reaches this percentage of [`limit`](Self::limit).
    pub compact_at_percent: u8,
    /// An estimate within this percentage of a limit is checked with an exact count first.
    pub count_margin_percent: u8,
    /// Turns before the last one the overflow fallback keeps, at most.
    pub max_kept_turns: usize,
    /// Who summarizes.
    pub compactor: Compactor,
}

impl ContextProfile {
    /// A profile for a window, with the defaults: compact at 75%, exact count within 10%, keep up to 4 turns on
    /// overflow, server compaction.
    pub fn new(window: u64, max_output: u64) -> Self {
        ContextProfile {
            window,
            max_output,
            compact_at_percent: 75,
            count_margin_percent: 10,
            max_kept_turns: 4,
            compactor: Compactor::Server,
        }
    }

    /// The most tokens a request may hold.
    pub fn limit(&self) -> u64 {
        self.window.saturating_sub(self.max_output)
    }

    /// The size at which a compaction is due: [`compact_at_percent`](Self::compact_at_percent) of the limit, so it
    /// always comes before the limit.
    pub fn compact_at(&self) -> u64 {
        percent(self.limit(), self.compact_at_percent)
    }
}

/// What to do before the next request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ContextAction {
    /// Send it as it is.
    None,
    /// The estimate is close to a limit: count the request's tokens and ask again with the count.
    CountTokens,
    /// Compact with this compactor first ([`crate::compaction_request`]).
    Compact(Compactor),
    /// The request does not fit: make a deterministic seed first ([`overflow_seed`]), no model call.
    Overflow,
}

/// The request's size in tokens, estimated without a tokenizer.
///
/// With an accepted answer in the current segment that reports usage: that call's input (uncached, cache reads and
/// writes) and output tokens, plus 2 tokens per 7 bytes of every slice it did not hold. Otherwise (no answer yet, or
/// a provider that reports no usage): 2 per 7 bytes of the prefix and every slice. An answer holding a `compaction`
/// block is read as one without usage: its usage is billed usage and includes the input that was compacted away.
pub fn estimate(log: &Log) -> u64 {
    let seg = log.segment();
    let last = log
        .entries()
        .iter()
        .rev()
        .take_while(|e| e.seg == seg)
        .find_map(|e| match &e.body {
            Body::AssistantMessage { from, .. } => match log.get(*from).map(|r| &r.body) {
                Some(Body::ModelResponse { usage, .. }) => Some((e.seq, *from, usage)),
                _ => None,
            },
            _ => None,
        });
    let projection = log.projection();
    let mut bytes = 0usize;
    let used = last
        .filter(|(answer, _, _)| !holds_compaction(log, *answer))
        .map_or(0, |(_, _, u)| {
            u.input_tokens
                .saturating_add(u.cache_read_input_tokens)
                .saturating_add(u.cache_creation_input_tokens)
                .saturating_add(u.output_tokens)
        });
    let base = match last {
        Some((answer, from, _)) if used > 0 => {
            let sent = request_before(log, from)
                .map(|r| r.messages.as_slice())
                .unwrap_or(&[]);
            for s in &projection.messages {
                if *s != answer && !sent.contains(s) {
                    bytes = bytes.saturating_add(slice_len(log, *s));
                }
            }
            used
        }
        _ => {
            if let Some(Body::SessionPrefix { system, tools, .. }) =
                projection.prefix.and_then(|p| log.get(p)).map(|e| &e.body)
            {
                bytes = system.len().saturating_add(tools.len());
            }
            for s in &projection.messages {
                bytes = bytes.saturating_add(slice_len(log, *s));
            }
            0
        }
    };
    let bytes = u64::try_from(bytes).unwrap_or(u64::MAX);
    base.saturating_add(bytes.saturating_mul(2).div_ceil(7))
}

/// Whether the assistant slice at `seq` holds a `compaction` block.
fn holds_compaction(log: &Log, seq: Seq) -> bool {
    matches!(log.slice(seq), Ok(Slice::Assistant { content, .. })
        if block_types(&content).iter().any(|t| t == "compaction"))
}

/// What to do before the next request, from the [`estimate`] or, after [`ContextAction::CountTokens`], the exact
/// `counted` size. Ask after each `turn.end` and before each model call.
///
/// In order:
/// - The last request of the segment was refused as too large: [`ContextAction::Overflow`] (the provider knows best).
/// - Close to the limit or to the compaction point, with no count: [`ContextAction::CountTokens`].
/// - Past the window: overflow (not even a compaction call fits).
/// - Compaction allowed and at or past [`ContextProfile::compact_at`]: [`ContextAction::Compact`], also past the
///   limit, because a summary keeps more than the fallback does.
/// - Past [`ContextProfile::limit`]: overflow.
///
/// Compaction is allowed when the segment has an answer of its own after its last compaction call (a segment just
/// made, or one whose compaction failed, waits for the next answer instead of asking again), and either no turn is
/// open or the compactor is [`Compactor::ClientSummary`]: a server block must be followed by the next user message,
/// so a long tool loop with server compaction can only overflow (in-turn server compaction is not supported).
///
/// # Errors
/// [`ContextError::NotAtBoundary`] if a tool call has no result or a model call is in flight.
pub fn before_request(
    log: &Log,
    profile: &ContextProfile,
    counted: Option<u64>,
) -> Result<ContextAction, ContextError> {
    let resume = log.resume();
    if !resume.at_boundary() {
        return Err(ContextError::NotAtBoundary);
    }
    if last_request_overflowed(log) {
        return Ok(ContextAction::Overflow);
    }
    let tokens = counted.unwrap_or_else(|| estimate(log));
    let may_compact = answered_since_compaction(log)
        && (!resume.in_turn || profile.compactor == Compactor::ClientSummary);
    let near = |limit: u64| {
        let margin = percent(limit, profile.count_margin_percent);
        tokens.abs_diff(limit) <= margin
    };
    if counted.is_none() && (near(profile.limit()) || (may_compact && near(profile.compact_at()))) {
        return Ok(ContextAction::CountTokens);
    }
    Ok(if tokens > profile.window {
        ContextAction::Overflow
    } else if may_compact && tokens >= profile.compact_at() {
        ContextAction::Compact(profile.compactor)
    } else if tokens > profile.limit() {
        ContextAction::Overflow
    } else {
        ContextAction::None
    })
}

/// The deterministic seed that keeps the most turns (up to [`ContextProfile::max_kept_turns`]) whose new segment's
/// [`estimate`] fits the target: [`ContextProfile::limit`], or, right after the provider refused the segment's last
/// request as too large, at most half the current estimate (the estimate was too low, so each refusal must at least
/// halve the request; a session that cannot shrink ends in [`ContextError::DoesNotFit`] instead of looping).
///
/// # Errors
/// As [`shrink_to_fit`].
pub fn overflow_seed(log: &Log, profile: &ContextProfile, ts: u64) -> Result<Seed, ContextError> {
    let target = if last_request_overflowed(log) {
        profile.limit().min(estimate(log) / 2)
    } else {
        profile.limit()
    };
    shrink_to_fit(log, profile.max_kept_turns, ts, |candidate| {
        estimate(candidate) <= target
    })
}

/// The current segment's last model call failed as too large for the context window.
fn last_request_overflowed(log: &Log) -> bool {
    let seg = log.segment();
    log.entries()
        .iter()
        .rev()
        .take_while(|e| e.seg == seg)
        .find(|e| {
            matches!(
                e.body,
                Body::ModelFailure { .. } | Body::ModelResponse { .. } | Body::ModelRequest { .. }
            )
        })
        .is_some_and(|e| e.body.error_kind() == Some(ErrorKind::ContextOverflow))
}

/// The recipe of the `model.request` nearest before `response`.
fn request_before(log: &Log, response: Seq) -> Option<&sarathi_log::Recipe> {
    log.entries()
        .iter()
        .rev()
        .skip_while(|e| e.seq >= response)
        .find_map(|e| match &e.body {
            Body::ModelRequest { recipe, .. } => Some(recipe),
            _ => None,
        })
}

/// The current segment has an accepted answer after its last compaction call (or has no compaction call).
fn answered_since_compaction(log: &Log) -> bool {
    let seg = log.segment();
    for e in log.entries().iter().rev().take_while(|e| e.seg == seg) {
        match e.body {
            Body::AssistantMessage { .. } => return true,
            Body::ModelRequest { .. } if compactor_of(log, e.seq).is_some() => return false,
            _ => {}
        }
    }
    false
}

fn slice_len(log: &Log, seq: Seq) -> usize {
    match log.slice(seq) {
        Ok(Slice::User { content }) => content.len(),
        Ok(Slice::Assistant { content, .. }) => content.len(),
        Ok(Slice::ToolResult { block, .. }) => block.len(),
        Err(_) => 0,
    }
}

fn percent(of: u64, pct: u8) -> u64 {
    u64::try_from(u128::from(of) * u128::from(pct) / 100).unwrap_or(u64::MAX)
}
