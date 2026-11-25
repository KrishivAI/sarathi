// SPDX-License-Identifier: Apache-2.0
//! [`Log`]: one session's entries in memory, the invariants every append keeps, the request projection and resume.

use sarathi_protocol::{MessageSlice, RawJson, Role, Seq};
use sha2::{Digest, Sha256};

use crate::entry::{
    Body, CallRef, Effect, Entry, LogHeader, Recipe, SegmentId, SegmentReason, Strategy,
};
use crate::error::LogError;
use crate::line::{content_slice, encode_entry};

/// The text resume puts in the result of a side-effecting call that started before a crash and has no result.
pub const CUT_SHORT: &str =
    "The call was cut short by a restart. Its effect is unknown; check before running it again.";

/// One session: its header and entries, in seq order. Every way of adding entries keeps the invariants checked by
/// [`Log::append`], so a `Log` read from a store is one that store would have accepted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Log {
    header: LogHeader,
    entries: Vec<Entry>,
}

/// One message slice as a request carries it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Slice<'a> {
    /// A user message (or steer): its `content`.
    User {
        /// The `content` value.
        content: &'a RawJson,
    },
    /// An assistant message: the `content` of the response it came from, byte for byte.
    Assistant {
        /// The `content` value.
        content: RawJson,
        /// The client tool calls it asks for.
        calls: &'a [CallRef],
    },
    /// One tool result block. The assembler groups consecutive results into one user message.
    ToolResult {
        /// The call it answers.
        call: &'a str,
        /// The block.
        block: &'a RawJson,
    },
}

impl Slice<'_> {
    /// The slice as a whole message (`{"role":…,"content":…}`). A tool result is wrapped alone; an assembler that
    /// groups results builds its own message instead.
    ///
    /// # Errors
    /// Never for slices read from a valid log.
    pub fn to_message(&self, seq: Seq) -> Result<MessageSlice, sarathi_protocol::RawJsonError> {
        match self {
            Slice::User { content } => MessageSlice::new(seq, Role::User, content),
            Slice::Assistant { content, .. } => MessageSlice::new(seq, Role::Assistant, content),
            Slice::ToolResult { block, .. } => {
                let content = RawJson::from_string(format!("[{}]", block.as_str()))?;
                MessageSlice::new(seq, Role::User, &content)
            }
        }
    }
}

/// Everything a request is built from, resolved from a [`Recipe`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestParts<'a> {
    /// The provider protocol of the prefix.
    pub protocol: &'a str,
    /// `system` as sent.
    pub system: &'a RawJson,
    /// `tools` as sent.
    pub tools: &'a RawJson,
    /// The slices, in order, with their seqs.
    pub messages: Vec<(Seq, Slice<'a>)>,
    /// The per-call parameters.
    pub params: &'a RawJson,
}

/// What the next request sends: the current segment's prefix and slices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Projection {
    /// The current segment.
    pub segment: SegmentId,
    /// The `session.prefix` entry in force, if one was written.
    pub prefix: Option<Seq>,
    /// The slices: the segment's seed, then the segment's own slices, in seq order.
    pub messages: Vec<Seq>,
}

/// Where a tool call of the last assistant message stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CallStatus {
    /// Asked for, never started: may run.
    NotStarted,
    /// A read-only call started and has no result: run it again.
    StartedRead,
    /// A side-effecting call started and has no result: never run again; answer it with [`CUT_SHORT`].
    StartedWrite,
    /// Answered.
    Done,
}

/// One call of the last assistant message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenCall {
    /// The call.
    pub call: CallRef,
    /// Where it stands.
    pub status: CallStatus,
}

/// The state a run continues from after a restart.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResumeState {
    /// The next seq to append.
    pub next: Seq,
    /// The current segment.
    pub segment: SegmentId,
    /// A user message started a turn that has no `turn.end` yet.
    pub in_turn: bool,
    /// The calls of the last assistant message, if the turn is still open.
    pub calls: Vec<OpenCall>,
    /// A `model.request` with no accepted answer and no failure: make the call again.
    pub pending_model: Option<Seq>,
}

impl ResumeState {
    /// No call is waiting for a result and no model call is in flight: a fork or a new user message may follow.
    pub fn at_boundary(&self) -> bool {
        self.pending_model.is_none() && self.calls.iter().all(|c| c.status == CallStatus::Done)
    }
}

impl Log {
    /// An empty session.
    pub fn new(header: LogHeader) -> Self {
        Log {
            header,
            entries: Vec::new(),
        }
    }

    /// The header.
    pub fn header(&self) -> &LogHeader {
        &self.header
    }

    /// The entries, from seq 1.
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// The seq the next entry gets.
    pub fn next_seq(&self) -> Seq {
        Seq(self.entries.len() as u64 + 1)
    }

    /// The current segment.
    pub fn segment(&self) -> SegmentId {
        self.entries.last().map_or(SegmentId(0), |e| e.seg)
    }

    /// The entry at `seq`.
    pub fn get(&self, seq: Seq) -> Option<&Entry> {
        let i = usize::try_from(seq.0).ok()?.checked_sub(1)?;
        self.entries.get(i)
    }

    /// Appends `entries` if the next seq is `expected` and every entry keeps the log's invariants; all or none.
    ///
    /// Invariants: seqs are dense; an entry stays in the current segment, except a `segment.start`, which opens the
    /// next one; references (`from`, `seed`, a recipe's `prefix` and `messages`) point back to entries of the right
    /// kind; a `segment.start`'s `replaces` lies before it and carries a `strategy` exactly when it is a compaction;
    /// a `tool.start` or `tool.result` names a call the last assistant message asked for; a `turn.end` comes only
    /// once every such call has its `tool.result`; a `model.request`'s `sha256` is 64 lower-case hex digits.
    ///
    /// A request's recipe is not rebuilt on append (the assembler lives with the provider): a log can hold a request
    /// whose recipe does not rebuild to its hash, and [`Log::verify_requests`] finds it.
    ///
    /// # Errors
    /// [`LogError::Conflict`] if `expected` is not the next seq; [`LogError::Invalid`] for a broken invariant.
    pub fn append(&mut self, expected: Seq, entries: Vec<Entry>) -> Result<Seq, LogError> {
        let next = self.next_seq();
        if expected != next {
            return Err(LogError::Conflict {
                expected: expected.0,
                actual: next.0,
            });
        }
        let len = self.entries.len();
        for e in entries {
            if let Err(err) = self.check(&e) {
                self.entries.truncate(len);
                return Err(err);
            }
            self.entries.push(e);
        }
        Ok(self.next_seq())
    }

    #[cfg(feature = "jsonl")]
    pub(crate) fn truncate_to(&mut self, len: usize) {
        self.entries.truncate(len);
    }

    fn check(&self, e: &Entry) -> Result<(), LogError> {
        let invalid = |detail: String| LogError::Invalid { seq: e.seq, detail };
        if e.seq != self.next_seq() {
            return Err(invalid(format!("expected seq {}", self.next_seq().0)));
        }
        let seg = self.segment();
        match &e.body {
            Body::SegmentStart {
                reason,
                strategy,
                seed,
                replaces,
            } => {
                match (reason, strategy) {
                    (SegmentReason::Compaction, None) => {
                        return Err(invalid(
                            "a compaction segment.start names its `strategy`".into(),
                        ));
                    }
                    (SegmentReason::ProtocolSwitch | SegmentReason::Fork, Some(_)) => {
                        return Err(invalid(
                            "only a compaction segment.start has a `strategy`".into(),
                        ));
                    }
                    _ => {}
                }
                if e.seg.0 != seg.0 + 1 {
                    return Err(invalid(format!(
                        "segment.start must open segment {}",
                        seg.0 + 1
                    )));
                }
                for s in seed {
                    self.expect_slice(*s).map_err(invalid)?;
                }
                if let Some(r) = replaces
                    && (r.start > r.end || r.end > e.seq)
                {
                    return Err(invalid("`replaces` must lie before the entry".into()));
                }
            }
            _ if e.seg != seg => {
                return Err(invalid(format!("entry must stay in segment {seg}")));
            }
            Body::AssistantMessage { from, .. } => {
                if !matches!(
                    self.get(*from).map(|f| &f.body),
                    Some(Body::ModelResponse { .. })
                ) {
                    return Err(invalid(format!(
                        "`from` {} is not a model.response",
                        from.0
                    )));
                }
                if self.compaction_of(*from).is_some() {
                    return Err(invalid(format!(
                        "`from` {} answers a compaction call: it goes into a seed",
                        from.0
                    )));
                }
            }
            Body::SegmentSeed { from: Some(f), .. } => {
                if !matches!(
                    self.get(*f).map(|x| &x.body),
                    Some(Body::ModelResponse { .. })
                ) {
                    self.expect_slice(*f).map_err(|_| {
                        invalid(format!(
                            "`from` {} is not a message slice or a model.response",
                            f.0
                        ))
                    })?;
                }
            }
            Body::ModelRequest {
                recipe,
                sha256,
                compaction,
            } => {
                if *compaction == Some(Strategy::Deterministic) {
                    return Err(invalid("a deterministic seed makes no model call".into()));
                }
                match self.get(recipe.prefix).map(|p| &p.body) {
                    Some(Body::SessionPrefix { .. }) => {}
                    _ => {
                        return Err(invalid(format!(
                            "`prefix` {} is not a session.prefix",
                            recipe.prefix.0
                        )));
                    }
                }
                for s in &recipe.messages {
                    self.expect_slice(*s).map_err(invalid)?;
                }
                if sha256.len() != 64
                    || !sha256
                        .bytes()
                        .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
                {
                    return Err(invalid("`sha256` is not 64 lower-case hex digits".into()));
                }
            }
            Body::ToolStart { call, .. } | Body::ToolResult { call, .. } => {
                if !self
                    .asked()
                    .is_some_and(|(_, asked)| asked.iter().any(|c| *c.id == **call))
                {
                    return Err(invalid(format!(
                        "no open assistant message asked for call `{call}`"
                    )));
                }
            }
            Body::TurnEnd { .. } => {
                if let Some((at, asked)) = self.asked()
                    && let Some(open) = asked.iter().find(|c| !self.answered_after(at, &c.id))
                {
                    return Err(invalid(format!(
                        "call `{}` has no tool.result: answer it before the turn ends",
                        open.id
                    )));
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// The last assistant message of the open turn and the calls it asked for: found walking back to the nearest
    /// `message.assistant`, stopping at a `turn.end` or a user message that starts a turn.
    fn asked(&self) -> Option<(usize, &[CallRef])> {
        for (i, e) in self.entries.iter().enumerate().rev() {
            match &e.body {
                Body::AssistantMessage { calls, .. } => return Some((i, calls)),
                Body::TurnEnd { .. } | Body::UserMessage { steer: false, .. } => return None,
                _ => {}
            }
        }
        None
    }

    /// A `tool.result` for `call` follows the entry at index `at`.
    fn answered_after(&self, at: usize, call: &str) -> bool {
        self.entries.get(at + 1..).is_some_and(|rest| {
            rest.iter()
                .any(|e| matches!(&e.body, Body::ToolResult { call: c, .. } if **c == *call))
        })
    }

    /// The compactor of the call `response` answers, if that call is a compaction call: the `model.request` nearest
    /// before it.
    pub(crate) fn compaction_of(&self, response: Seq) -> Option<Strategy> {
        let n = usize::try_from(response.0).ok()?;
        self.entries
            .get(..n.saturating_sub(1))?
            .iter()
            .rev()
            .find_map(|e| match &e.body {
                Body::ModelRequest { compaction, .. } => Some(*compaction),
                _ => None,
            })?
    }

    fn expect_slice(&self, seq: Seq) -> Result<(), String> {
        match self.get(seq) {
            Some(e) if e.body.is_slice() => Ok(()),
            _ => Err(format!("seq {} is not a message slice", seq.0)),
        }
    }

    /// The slice an entry holds.
    ///
    /// # Errors
    /// [`LogError::Invalid`] if `seq` is not a slice entry.
    pub fn slice(&self, seq: Seq) -> Result<Slice<'_>, LogError> {
        let not = |what: &str| LogError::Invalid {
            seq,
            detail: what.to_owned(),
        };
        match self.get(seq).map(|e| &e.body) {
            Some(Body::UserMessage { content, .. }) => Ok(Slice::User { content }),
            Some(Body::ToolResult { call, block, .. }) => Ok(Slice::ToolResult { call, block }),
            Some(Body::SegmentSeed {
                role: Role::User,
                content,
                ..
            }) => Ok(Slice::User { content }),
            Some(Body::SegmentSeed { content, calls, .. }) => Ok(Slice::Assistant {
                content: content.clone(),
                calls,
            }),
            Some(Body::AssistantMessage { from, calls }) => {
                match self.get(*from).map(|e| &e.body) {
                    Some(Body::ModelResponse { body, content, .. }) => Ok(Slice::Assistant {
                        content: content_slice(body, content)
                            .ok_or_else(|| not("the response's content range is not a value"))?,
                        calls,
                    }),
                    _ => Err(not("`from` is not a model.response")),
                }
            }
            _ => Err(not("not a message slice")),
        }
    }

    /// The current segment's prefix and slices: what the next request sends.
    pub fn projection(&self) -> Projection {
        let segment = self.segment();
        let mut messages = Vec::new();
        let mut prefix = None;
        for e in &self.entries {
            if let Body::SessionPrefix { .. } = e.body {
                prefix = Some(e.seq);
            }
            if e.seg != segment {
                continue;
            }
            match &e.body {
                Body::SegmentStart { seed, .. } => messages.extend_from_slice(seed),
                // Sent only through a seed that names it.
                Body::SegmentSeed { .. } => {}
                b if b.is_slice() => messages.push(e.seq),
                _ => {}
            }
        }
        Projection {
            segment,
            prefix,
            messages,
        }
    }

    /// The recipe of the next request: the projection with `params`.
    ///
    /// # Errors
    /// [`LogError::Invalid`] if no `session.prefix` was written.
    pub fn next_recipe(&self, params: RawJson) -> Result<Recipe, LogError> {
        let p = self.projection();
        let prefix = p.prefix.ok_or_else(|| LogError::Invalid {
            seq: self.next_seq(),
            detail: "no session.prefix before the first request".into(),
        })?;
        Ok(Recipe {
            prefix,
            messages: p.messages,
            params,
        })
    }

    /// Resolves a recipe to the parts an assembler writes a body from.
    ///
    /// # Errors
    /// [`LogError::Invalid`] if the recipe points at entries of the wrong kind.
    pub fn request_parts<'a>(&'a self, recipe: &'a Recipe) -> Result<RequestParts<'a>, LogError> {
        let Some(Body::SessionPrefix {
            protocol,
            system,
            tools,
        }) = self.get(recipe.prefix).map(|e| &e.body)
        else {
            return Err(LogError::Invalid {
                seq: recipe.prefix,
                detail: "not a session.prefix".into(),
            });
        };
        let messages = recipe
            .messages
            .iter()
            .map(|s| self.slice(*s).map(|sl| (*s, sl)))
            .collect::<Result<_, _>>()?;
        Ok(RequestParts {
            protocol,
            system,
            tools,
            messages,
            params: &recipe.params,
        })
    }

    /// Checks that every `model.request` rebuilds, through `assemble`, to the body hash it records.
    ///
    /// # Errors
    /// [`LogError::HashMismatch`] for the first request that does not.
    pub fn verify_requests(
        &self,
        assemble: &dyn Fn(&RequestParts<'_>) -> Vec<u8>,
    ) -> Result<(), LogError> {
        for e in &self.entries {
            if let Body::ModelRequest { recipe, sha256, .. } = &e.body {
                let actual = sha256_hex(&assemble(&self.request_parts(recipe)?));
                if &actual != sha256 {
                    return Err(LogError::HashMismatch {
                        seq: e.seq,
                        recorded: sha256.clone(),
                        actual,
                    });
                }
            }
        }
        Ok(())
    }

    /// Where a run stands: what a restarted core must do before its next model call.
    pub fn resume(&self) -> ResumeState {
        let mut in_turn = false;
        let mut calls: Vec<OpenCall> = Vec::new();
        let mut last_request: Option<Seq> = None;
        let mut answered = false;
        for e in &self.entries {
            match &e.body {
                // A new turn's request must carry its message: an older unanswered request is never made again.
                Body::UserMessage { steer: false, .. } => {
                    in_turn = true;
                    last_request = None;
                }
                // The end of a turn closes it, model call included (an interrupt or a budget stop during a call writes
                // no answer). Its tool calls are all answered: `turn.end` is refused otherwise.
                Body::TurnEnd { .. } => {
                    in_turn = false;
                    calls.clear();
                    last_request = None;
                }
                Body::ModelRequest { .. } => {
                    last_request = Some(e.seq);
                    answered = false;
                }
                Body::ModelFailure { .. } => answered = true,
                // A compaction call's answer goes into the seed instead of a `message.assistant`: a `segment.start`
                // whose seed copies a response after the request answers it. Any other start answers nothing.
                Body::SegmentStart { seed, .. } => {
                    if let Some(r) = last_request
                        && seed.iter().any(|s| {
                            matches!(self.get(*s).map(|x| &x.body),
                                Some(Body::SegmentSeed { from: Some(f), .. })
                                    if *f > r && matches!(self.get(*f).map(|x| &x.body), Some(Body::ModelResponse { .. })))
                        })
                    {
                        answered = true;
                    }
                }
                Body::AssistantMessage { from, calls: asked } => {
                    if last_request.is_some_and(|r| *from > r) {
                        answered = true;
                    }
                    calls = asked
                        .iter()
                        .map(|c| OpenCall {
                            call: c.clone(),
                            status: CallStatus::NotStarted,
                        })
                        .collect();
                }
                Body::ToolStart { call, effect, .. } => {
                    if let Some(c) = calls.iter_mut().find(|c| *c.call.id == **call)
                        && c.status == CallStatus::NotStarted
                    {
                        c.status = match effect {
                            Effect::Read => CallStatus::StartedRead,
                            Effect::Write => CallStatus::StartedWrite,
                        };
                    }
                }
                Body::ToolResult { call, .. } => {
                    if let Some(c) = calls.iter_mut().find(|c| *c.call.id == **call) {
                        c.status = CallStatus::Done;
                    }
                }
                _ => {}
            }
        }
        ResumeState {
            next: self.next_seq(),
            segment: self.segment(),
            in_turn,
            calls,
            pending_model: last_request.filter(|_| !answered),
        }
    }

    /// The log as it was after `at`: the header and entries `1..=at`.
    ///
    /// # Errors
    /// [`LogError::Invalid`] if `at` is past the end.
    pub fn up_to(&self, at: Seq) -> Result<Log, LogError> {
        let n = usize::try_from(at.0).unwrap_or(usize::MAX);
        if n > self.entries.len() {
            return Err(LogError::Invalid {
                seq: at,
                detail: "past the end of the log".into(),
            });
        }
        Ok(Log {
            header: self.header.clone(),
            entries: self.entries[..n].to_vec(),
        })
    }

    /// Checks a forked session's copy of its parent: its lines `1..=at` hash to the digest its header records. A log
    /// that is not a fork, or holds fewer than `at` entries (a fork cut short before its copy was written), passes.
    ///
    /// # Errors
    /// [`LogError::ForkMismatch`] if the copy differs.
    pub fn verify_fork(&self) -> Result<(), LogError> {
        let Some(from) = &self.header.forked_from else {
            return Ok(());
        };
        if usize::try_from(from.seq.0).map_or(true, |n| n > self.entries.len()) {
            return Ok(());
        }
        let actual = self.lines_sha256(from.seq)?;
        if actual == from.sha256 {
            Ok(())
        } else {
            Err(LogError::ForkMismatch {
                at: from.seq,
                recorded: from.sha256,
                actual,
            })
        }
    }

    /// Lower-case hex SHA-256 of the entry lines `1..=at`, each with its `\n`, as the JSONL store writes them.
    ///
    /// # Errors
    /// [`LogError::Invalid`] if `at` is past the end.
    pub fn lines_sha256(&self, at: Seq) -> Result<sarathi_protocol::Sha256, LogError> {
        let part = self.up_to(at)?;
        let mut h = Sha256::new();
        for e in &part.entries {
            h.update(encode_entry(e)?.as_bytes());
            h.update(b"\n");
        }
        Ok(sarathi_protocol::Sha256(h.finalize().into()))
    }
}

/// Lower-case hex SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(char::from(DIGITS[usize::from(b >> 4)]));
        s.push(char::from(DIGITS[usize::from(b & 0xf)]));
    }
    s
}
