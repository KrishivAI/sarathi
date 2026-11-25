// SPDX-License-Identifier: Apache-2.0
//! When to compact: the size estimate ([`estimate`]) and the decision before a request ([`before_request`]).

use sarathi_log::{Body, Log, Slice};

use crate::compact::{Compactor, compactor_of};
use crate::error::ContextError;
use crate::fallback::{Seed, shrink_to_fit};

/// How one model's context is managed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextProfile {
    /// The context window, in tokens.
    pub window: u64,
    /// Tokens kept free for the answer: a request over `window - max_output` does not fit.
    pub max_output: u64,
    /// Compact between turns once the estimate reaches this percentage of `window`.
    pub compact_at_percent: u8,
    /// An estimate within this percentage of a limit is checked with an exact count first.
    pub count_margin_percent: u8,
    /// Turns before the current one the overflow fallback keeps, at most.
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

    /// The size at which a compaction is due.
    pub fn compact_at(&self) -> u64 {
        percent(self.window, self.compact_at_percent)
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

/// The request's size in tokens, estimated without a tokenizer: the last accepted answer of the current segment's
/// reported usage (input, cache reads and writes, output), plus 2 tokens per 7 bytes of every slice after it; with no
/// answer in the segment yet, 2 per 7 bytes of the prefix and every slice.
pub fn estimate(log: &Log) -> u64 {
    let seg = log.segment();
    let last = log
        .entries()
        .iter()
        .rev()
        .take_while(|e| e.seg == seg)
        .find_map(|e| match &e.body {
            Body::AssistantMessage { from, .. } => match log.get(*from).map(|r| &r.body) {
                Some(Body::ModelResponse { usage, .. }) => Some((e.seq, usage)),
                _ => None,
            },
            _ => None,
        });
    let projection = log.projection();
    let mut bytes = 0usize;
    let base = if let Some((after, usage)) = last {
        for s in projection.messages.iter().filter(|s| **s > after) {
            bytes = bytes.saturating_add(slice_len(log, *s));
        }
        usage
            .input_tokens
            .saturating_add(usage.cache_read_input_tokens)
            .saturating_add(usage.cache_creation_input_tokens)
            .saturating_add(usage.output_tokens)
    } else {
        if let Some(Body::SessionPrefix { system, tools, .. }) =
            projection.prefix.and_then(|p| log.get(p)).map(|e| &e.body)
        {
            bytes = system.len().saturating_add(tools.len());
        }
        for s in &projection.messages {
            bytes = bytes.saturating_add(slice_len(log, *s));
        }
        0
    };
    let bytes = u64::try_from(bytes).unwrap_or(u64::MAX);
    base.saturating_add(bytes.saturating_mul(2).div_ceil(7))
}

/// What to do before the next request, from the [`estimate`] or, after [`ContextAction::CountTokens`], the exact
/// `counted` size. Ask after each `turn.end` and before each model call.
///
/// A request over [`ContextProfile::limit`] overflows. Compaction is due at [`ContextProfile::compact_at`], but only
/// between turns (no turn open, so a server block is followed by the next user message) and only if the segment has
/// an answer of its own after its last compaction call: a segment just made, or one whose compaction failed, waits for
/// the next turn instead of asking again.
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
    let tokens = counted.unwrap_or_else(|| estimate(log));
    let may_compact = !resume.in_turn && answered_since_compaction(log);
    let near = |limit: u64| {
        let margin = percent(limit, profile.count_margin_percent);
        tokens.abs_diff(limit) <= margin
    };
    if counted.is_none() && (near(profile.limit()) || (may_compact && near(profile.compact_at()))) {
        return Ok(ContextAction::CountTokens);
    }
    Ok(if tokens > profile.limit() {
        ContextAction::Overflow
    } else if may_compact && tokens >= profile.compact_at() {
        ContextAction::Compact(profile.compactor)
    } else {
        ContextAction::None
    })
}

/// The deterministic seed that keeps the most turns (up to [`ContextProfile::max_kept_turns`]) for which the
/// [`estimate`] of the new segment fits [`ContextProfile::limit`].
///
/// # Errors
/// As [`shrink_to_fit`].
pub fn overflow_seed(log: &Log, profile: &ContextProfile, ts: u64) -> Result<Seed, ContextError> {
    shrink_to_fit(log, profile.max_kept_turns, ts, |candidate| {
        estimate(candidate) <= profile.limit()
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

fn slice_len(log: &Log, seq: sarathi_protocol::Seq) -> usize {
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
