// SPDX-License-Identifier: Apache-2.0
//! The deterministic overflow fallback: a new segment that keeps the first user message (or the segment's head), a
//! note and the last turns, made without a model call.

use sarathi_log::{Body, Entry, Log, LogError, SegmentId, SegmentReason, Slice, Strategy};
use sarathi_protocol::{RawJson, Role, Seq};
use serde::Deserialize;
use serde_json::value::RawValue;

use crate::error::ContextError;

/// The user message a deterministic seed puts before the kept turns. It says only what holds for every seed: the
/// message before it may be the session's first user message, the old segment's head (an earlier summary or
/// compaction block), or nothing, when the first user message is among the kept turns.
pub const DROPPED_NOTE: &str = "[Earlier turns of this conversation were removed to fit the context window. The \
turns after this note are the most recent ones.]";

/// The entries a compaction appends, in order: new `segment.seed` slices, then the `segment.start` that seeds the
/// new segment with them. Append them in one batch at [`Log::next_seq`] of the log they were made from. A batch cut
/// short before its `segment.start` changes nothing: seed slices are sent only through a seed that names them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Seed {
    /// The entries.
    pub entries: Vec<Entry>,
    /// Earlier turns of the old segment kept besides the last one.
    pub kept_turns: usize,
    /// The old segment's last turn was kept (always, while a turn is open).
    pub kept_last: bool,
    /// The old segment's head (the slices before its first turn, such as a server compaction block) was kept.
    pub kept_head: bool,
}

impl Seed {
    /// The seqs the new segment starts with.
    pub fn seed(&self) -> &[Seq] {
        match self.entries.last().map(|e| &e.body) {
            Some(Body::SegmentStart { seed, .. }) => seed,
            _ => &[],
        }
    }
}

/// What to keep of the current segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Keep {
    /// Turns before the last one to keep, at most (fewer when the segment has fewer).
    pub turns: usize,
    /// Keep the last turn. While a turn is open it is the current one and always kept, whatever this says; between
    /// turns it is complete and may be dropped like the others (the next user message starts a new turn), which
    /// is the only way on when the last turn alone does not fit.
    pub last: bool,
    /// Keep the segment's head: the slices before its first turn, such as a server compaction block, which stays
    /// first. Without it, the session's first user message is put first instead.
    pub head: bool,
}

/// The deterministic seed that keeps the last turn (see [`Keep::last`]) and up to `keep.turns` turns before it.
///
/// The new segment's seed is: the old segment's head if it has one and `keep.head` is set, otherwise the session's
/// first user message (by seq, unless it is in the kept range); a new [`DROPPED_NOTE`] user message; then the kept
/// turns' slices. User and tool-result slices are referenced by seq. The current turn is kept as it is, thinking
/// included: within a tool-use turn the provider needs its thinking blocks back unmodified. An earlier kept turn's
/// assistant slice with thinking blocks gets a `segment.seed` copy without them ([`strip_thinking`]); one with
/// nothing but thinking is left out. Turns start at a user message that is not a steer, so the cut never splits a
/// tool round.
///
/// # Errors
/// [`ContextError::NotAtBoundary`] if a tool call has no result or a model call is in flight;
/// [`ContextError::NoUserMessage`] if the segment has no turn start; [`ContextError::NothingToDrop`] if the last
/// turn is kept and nothing before it can be dropped under `keep`; [`ContextError::Log`] for a log that does not
/// resolve its own slices or is out of seqs.
pub fn deterministic_seed(log: &Log, keep: Keep, ts: u64) -> Result<Seed, ContextError> {
    if !log.resume().at_boundary() {
        return Err(ContextError::NotAtBoundary);
    }
    let shape = Shape::of(log)?;
    let keep_head = keep.head && shape.head > 0;
    let keep_last = keep.last || log.resume().in_turn;
    let last = *shape.starts.last().ok_or(ContextError::NoUserMessage)?;
    let before_last = shape.starts.len() - 1;
    // The kept range must leave something out: the last turn, or else turn 0 (and the head, if it is kept).
    let most = if !keep_last {
        before_last
    } else if keep_head || shape.head == 0 {
        before_last
            .checked_sub(1)
            .ok_or(ContextError::NothingToDrop)?
    } else {
        before_last
    };
    let kept_turns = keep.turns.min(most);
    let cut = shape.starts[before_last - kept_turns];
    let end = if keep_last {
        shape.messages.len()
    } else {
        last
    };
    let messages = &shape.messages;
    let kept = &messages[cut..end];

    let seg = log.segment();
    let mut next = log.next_seq();
    let mut entries = Vec::new();
    let mut seed = Vec::new();
    let mut push = |entries: &mut Vec<Entry>, body: Body| -> Result<Seq, ContextError> {
        let seq = next_seq(&mut next)?;
        entries.push(Entry { seq, seg, ts, body });
        Ok(seq)
    };
    if keep_head {
        seed.extend_from_slice(&messages[..shape.head]);
    } else if let Some(f) = log
        .entries()
        .iter()
        .find(|e| is_turn_start(log, e.seq))
        .map(|e| e.seq)
        .filter(|f| !kept.contains(f))
    {
        seed.push(f);
    }
    seed.push(push(
        &mut entries,
        Body::SegmentSeed {
            role: Role::User,
            content: RawJson::string(DROPPED_NOTE),
            from: None,
            calls: Vec::new(),
        },
    )?);
    for (i, &s) in kept.iter().enumerate() {
        let earlier_turn = cut + i < last;
        match log.get(s).map(|e| &e.body) {
            Some(Body::AssistantMessage { calls, .. }) if earlier_turn => {
                let Slice::Assistant { content, .. } = log.slice(s)? else {
                    return Err(not_a_slice(s));
                };
                match strip_thinking(&content)? {
                    None => seed.push(s),
                    Some(stripped) if stripped.as_str() == "[]" => {}
                    Some(stripped) => seed.push(push(
                        &mut entries,
                        Body::SegmentSeed {
                            role: Role::Assistant,
                            content: stripped,
                            from: Some(s),
                            calls: calls.clone(),
                        },
                    )?),
                }
            }
            _ => seed.push(s),
        }
    }
    let first_of_segment = log
        .entries()
        .iter()
        .find(|e| e.seg == seg)
        .map_or(Seq(1), |e| e.seq);
    let replaces = first_of_segment..log.next_seq();
    let new_seg = seg.0.checked_add(1).ok_or_else(|| out_of_seqs(next))?;
    entries.push(Entry {
        seq: next,
        seg: SegmentId(new_seg),
        ts,
        body: Body::SegmentStart {
            reason: SegmentReason::Compaction,
            strategy: Some(Strategy::Deterministic),
            seed,
            replaces: Some(replaces),
        },
    });
    Ok(Seed {
        entries,
        kept_turns,
        kept_last: keep_last,
        kept_head: keep_head,
    })
}

/// The deterministic seed that keeps the most for which `fits` says the log with the seed appended fits. With the
/// segment's head kept: the last turn and from `max_turns` earlier turns down to none, then (only between turns) the
/// same with the last turn dropped too, so a complete turn too large for the window does not block every later one;
/// then the same two passes without the head. The head is dropped only when nothing with it fits.
///
/// # Errors
/// [`ContextError::DoesNotFit`] if no seed fits (while a turn is open: even the current turn alone does not fit);
/// any error of [`deterministic_seed`] other than [`ContextError::NothingToDrop`].
pub fn shrink_to_fit(
    log: &Log,
    max_turns: usize,
    ts: u64,
    mut fits: impl FnMut(&Log) -> bool,
) -> Result<Seed, ContextError> {
    // Without a head, keeping it or not is the same seed: one pass.
    let passes: &[bool] = if Shape::of(log)?.head > 0 {
        &[true, false]
    } else {
        &[false]
    };
    let lasts: &[bool] = if log.resume().in_turn {
        &[true]
    } else {
        &[true, false]
    };
    let mut tried = false;
    for (&head, &last) in passes
        .iter()
        .flat_map(|h| lasts.iter().map(move |l| (h, l)))
    {
        let mut turns = max_turns;
        loop {
            let seed = match deterministic_seed(log, Keep { turns, head, last }, ts) {
                Ok(seed) => seed,
                Err(ContextError::NothingToDrop) => break,
                Err(e) => return Err(e),
            };
            tried = true;
            let mut candidate = log.clone();
            candidate.append(log.next_seq(), seed.entries.clone())?;
            if fits(&candidate) {
                return Ok(seed);
            }
            if seed.kept_turns == 0 {
                break;
            }
            // A seed keeps at most the turns the segment has, so the passes are bounded by the turn count.
            turns = seed.kept_turns - 1;
        }
    }
    Err(if tried {
        ContextError::DoesNotFit
    } else {
        ContextError::NothingToDrop
    })
}

/// The current segment's projection split into its head and its turns.
struct Shape {
    messages: Vec<Seq>,
    /// Slices before the first turn start.
    head: usize,
    /// Indexes in `messages` of the turn starts.
    starts: Vec<usize>,
}

impl Shape {
    fn of(log: &Log) -> Result<Shape, ContextError> {
        let messages = log.projection().messages;
        let starts: Vec<usize> = messages
            .iter()
            .enumerate()
            .filter(|(_, s)| is_turn_start(log, **s))
            .map(|(i, _)| i)
            .collect();
        let Some(&head) = starts.first() else {
            return Err(ContextError::NoUserMessage);
        };
        Ok(Shape {
            messages,
            head,
            starts,
        })
    }
}

/// `*next`, and moves `next` on by one.
pub(crate) fn next_seq(next: &mut Seq) -> Result<Seq, ContextError> {
    let seq = *next;
    *next = Seq(next.0.checked_add(1).ok_or_else(|| out_of_seqs(seq))?);
    Ok(seq)
}

pub(crate) fn out_of_seqs(seq: Seq) -> ContextError {
    ContextError::Log(LogError::Invalid {
        seq,
        detail: "out of seqs or segments".into(),
    })
}

fn not_a_slice(seq: Seq) -> ContextError {
    ContextError::Log(LogError::Invalid {
        seq,
        detail: "an assistant message that is not an assistant slice".into(),
    })
}

/// `content` (a JSON array of content blocks) without its `thinking` and `redacted_thinking` blocks, every other
/// block's bytes unchanged. `None` if there is nothing to remove or `content` is not an array.
///
/// # Errors
/// Never for content read from a valid log.
pub fn strip_thinking(content: &RawJson) -> Result<Option<RawJson>, ContextError> {
    #[derive(Deserialize)]
    struct Kind<'a> {
        #[serde(rename = "type", borrow, default)]
        kind: Option<std::borrow::Cow<'a, str>>,
    }
    let Ok(blocks) = serde_json::from_str::<Vec<&RawValue>>(content.as_str()) else {
        return Ok(None);
    };
    let is_thinking = |b: &RawValue| {
        serde_json::from_str::<Kind<'_>>(b.get())
            .ok()
            .and_then(|k| k.kind)
            .is_some_and(|k| k == "thinking" || k == "redacted_thinking")
    };
    if !blocks.iter().any(|b| is_thinking(b)) {
        return Ok(None);
    }
    let kept: Vec<&str> = blocks
        .iter()
        .filter(|b| !is_thinking(b))
        .map(|b| b.get())
        .collect();
    Ok(Some(RawJson::from_string(format!("[{}]", kept.join(",")))?))
}

fn is_turn_start(log: &Log, seq: Seq) -> bool {
    matches!(
        log.get(seq).map(|e| &e.body),
        Some(Body::UserMessage { steer: false, .. })
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(s: &str) -> RawJson {
        RawJson::from_text(s).unwrap()
    }

    #[test]
    fn thinking_is_removed_and_other_bytes_are_kept() {
        let c = raw(
            r#"[{"type":"thinking","thinking":"x","signature":"s"}, {"type":"text","text":"é","z":1e5},{"type":"redacted_thinking","data":"d"},{"type":"tool_use","id":"t","name":"n","input":{}}]"#,
        );
        assert_eq!(
            strip_thinking(&c).unwrap().unwrap().as_str(),
            r#"[{"type":"text","text":"é","z":1e5},{"type":"tool_use","id":"t","name":"n","input":{}}]"#
        );
    }

    #[test]
    fn content_without_thinking_or_not_an_array_is_left_alone() {
        assert_eq!(
            strip_thinking(&raw(r#"[{"type":"text","text":"a"}]"#)).unwrap(),
            None
        );
        assert_eq!(strip_thinking(&raw(r#""plain text""#)).unwrap(), None);
        assert_eq!(strip_thinking(&raw(r#"[{"no_type":1}]"#)).unwrap(), None);
    }

    #[test]
    fn only_thinking_leaves_an_empty_array() {
        let s = strip_thinking(&raw(
            r#"[{"type":"thinking","thinking":"","signature":"s"}]"#,
        ))
        .unwrap()
        .unwrap();
        assert_eq!(s.as_str(), "[]");
    }
}
