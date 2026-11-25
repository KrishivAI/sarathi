// SPDX-License-Identifier: Apache-2.0
//! A long scripted run: 60 turns with tool rounds through a small window, the way a run loop drives this crate.
//! A test-only loop stands in for the core and a scripted provider for the model. It checks that the run compacts
//! more than once with each compactor, that every request after a compaction starts with its one seed slice and
//! holds nothing it replaced, that requests inside a segment only ever append (the prefix check), that a restart at
//! any entry of a compaction ends in the same projection, and that an overflow is recovered without a model call.

#![expect(clippy::unwrap_used, clippy::panic, reason = "test code")]

use sarathi_context::{
    Compactor, ContextAction, ContextProfile, before_request, carries_compaction_block,
    compaction_request, compaction_seed, estimate, overflow_seed, pending_compaction,
};
use sarathi_log::{
    Body, CallRef, Effect, Entry, Log, Recipe, SegmentId, Slice, Strategy, new_header,
    new_session_id,
};
use sarathi_protocol::{RawJson, Seq, Usage};

const TURNS: u32 = 60;
/// Bytes of each tool result: about 430 tokens by the 2-per-7 estimate, so a compaction every few turns.
const RESULT_BYTES: usize = 1_500;

fn raw(s: &str) -> RawJson {
    RawJson::from_text(s).unwrap()
}

/// The text a projected slice sends.
fn text(log: &Log, seq: Seq) -> String {
    match log.slice(seq).unwrap() {
        Slice::User { content } => content.as_str().to_owned(),
        Slice::Assistant { content, .. } => content.as_str().to_owned(),
        Slice::ToolResult { block, .. } => block.as_str().to_owned(),
    }
}

fn projected(log: &Log) -> Vec<String> {
    log.projection()
        .messages
        .iter()
        .map(|s| text(log, *s))
        .collect()
}

/// Tokens a scripted provider reports for a request: 2 per 7 bytes, the same rule the estimate falls back to.
fn tokens(bytes: usize) -> u64 {
    u64::try_from(bytes * 2).unwrap().div_ceil(7)
}

/// One compaction batch: the first entry it wrote, its `segment.start`, and the projection on either side.
struct Batch {
    index: u32,
    compactor: Compactor,
    first: Seq,
    last: Seq,
    before: Vec<String>,
    after: Vec<String>,
}

/// The stand-in core: decides before every model call and between turns, over an in-memory log.
#[derive(Clone)]
struct Run {
    log: Log,
    profile: ContextProfile,
    /// Model calls made (turn requests and compaction calls).
    calls: u32,
    compactions: u32,
    overflows: u32,
    /// The turn request whose first call the provider refuses as too large.
    refuse_at: Option<u32>,
    /// The last turn request in the current segment, for the prefix check.
    last_request: Option<(SegmentId, Vec<Seq>)>,
    /// The first seq of each compaction batch.
    batch_starts: Vec<Seq>,
}

impl Run {
    fn new(profile: ContextProfile) -> Self {
        let mut run = Run {
            log: Log::new(new_header(new_session_id(), 1_764_057_600_000, None)),
            profile,
            calls: 0,
            compactions: 0,
            overflows: 0,
            refuse_at: None,
            last_request: None,
            batch_starts: Vec::new(),
        };
        run.push(Body::SessionPrefix {
            protocol: "anthropic-messages".into(),
            system: raw(r#""be brief""#),
            tools: raw("[]"),
        });
        run
    }

    fn ts(&self) -> u64 {
        1_000 + self.log.next_seq().0
    }

    fn push(&mut self, body: Body) -> Seq {
        let seq = self.log.next_seq();
        let seg = match body {
            Body::SegmentStart { .. } => SegmentId(self.log.segment().0 + 1),
            _ => self.log.segment(),
        };
        let ts = self.ts();
        self.log
            .append(seq, vec![Entry { seq, seg, ts, body }])
            .unwrap();
        seq
    }

    fn append(&mut self, entries: Vec<Entry>) {
        self.log.append(self.log.next_seq(), entries).unwrap();
    }

    /// A `model.response` for an Anthropic message body; `content` is the JSON array of its blocks.
    fn response(&mut self, content: &str, stop: &str, usage: Usage) -> Seq {
        let head = r#"{"id":"msg","content":"#;
        let body = format!(r#"{head}{content},"stop_reason":"{stop}"}}"#);
        let start = u32::try_from(head.len()).unwrap();
        let end = start + u32::try_from(content.len()).unwrap();
        self.push(Body::ModelResponse {
            body: RawJson::from_string(body).unwrap(),
            content: start..end,
            sse: None,
            request_id: None,
            usage,
        })
    }

    /// The scripted count of a request: what the provider would report as its input tokens.
    fn count(&self, recipe: &Recipe) -> u64 {
        let prefix = r#""be brief""#.len() + "[]".len();
        tokens(
            prefix
                + recipe
                    .messages
                    .iter()
                    .map(|s| text(&self.log, *s).len())
                    .sum::<usize>(),
        )
    }

    /// Asks the context crate what to do before a model call and does it, the way the core does.
    fn prepare(&mut self) {
        let mut counted = None;
        for _ in 0..4 {
            match before_request(&self.log, &self.profile, counted).unwrap() {
                ContextAction::None => return,
                ContextAction::CountTokens => {
                    let recipe = self.log.next_recipe(raw("{}")).unwrap();
                    counted = Some(self.count(&recipe));
                }
                ContextAction::Compact(c) => {
                    self.compact(c);
                    counted = None;
                }
                ContextAction::Overflow => {
                    let calls = self.calls;
                    let seed = overflow_seed(&self.log, &self.profile, self.ts()).unwrap();
                    self.append(seed.entries);
                    self.overflows += 1;
                    assert_eq!(self.calls, calls, "overflow recovery makes no model call");
                    self.check_overflow_segment();
                    return;
                }
                other => panic!("unexpected {other:?}"),
            }
        }
        panic!("no decision settles after 4 rounds");
    }

    /// Renders and logs a compaction call in one batch (the request entries with their `model.request`).
    fn compaction_call(&mut self, c: Compactor) {
        let req = compaction_request(
            &self.log,
            c,
            &raw(r#"{"model":"m","max_tokens":1024}"#),
            None,
            self.ts(),
        )
        .unwrap();
        self.batch_starts.push(self.log.next_seq());
        let mut batch = req.entries.clone();
        let seq = Seq(self.log.next_seq().0 + u64::try_from(batch.len()).unwrap());
        batch.push(Entry {
            seq,
            seg: self.log.segment(),
            ts: self.ts(),
            body: req.model_request("0".repeat(64)),
        });
        self.append(batch);
    }

    /// The scripted provider's answer to compaction call `index`: the same bytes whenever it is asked again.
    fn compaction_answer(&mut self, c: Compactor, index: u32) {
        self.calls += 1;
        let usage = Usage {
            output_tokens: 20,
            ..Usage::default()
        };
        match c {
            Compactor::Server => self.response(
                &format!(
                    r#"[{{"type":"compaction","content":"Summary {index}: earlier tasks done.","signature":"c2ln{index}"}}]"#
                ),
                "compaction",
                usage,
            ),
            _ => self.response(
                &format!(r#"[{{"type":"text","text":"Summary {index}: earlier tasks done."}}]"#),
                "end_turn",
                usage,
            ),
        };
    }

    fn compact(&mut self, c: Compactor) {
        let index = self.compactions;
        self.compaction_call(c);
        self.compaction_answer(c, index);
        let seed = compaction_seed(&self.log, self.ts()).unwrap();
        self.append(seed.entries);
        self.compactions += 1;
    }

    /// One turn request: the recipe is checked, logged, and answered (or refused as too large).
    fn call(&mut self, turn: u32, content: &str, calls: &[&str]) -> bool {
        self.prepare();
        let recipe = self.log.next_recipe(raw(r#"{"model":"m"}"#)).unwrap();
        self.check_request(&recipe);
        let input = self.count(&recipe);
        self.push(Body::ModelRequest {
            recipe,
            sha256: "0".repeat(64),
            compaction: None,
        });
        self.calls += 1;
        if self.refuse_at == Some(turn) {
            self.refuse_at = None;
            self.push(Body::ModelFailure {
                error: "context_overflow".into(),
                status: Some(400),
                detail: "prompt is too long".into(),
                request_id: None,
                retries: 0,
            });
            return false;
        }
        let from = self.response(
            content,
            if calls.is_empty() {
                "end_turn"
            } else {
                "tool_use"
            },
            Usage {
                input_tokens: input,
                output_tokens: tokens(content.len()),
                ..Usage::default()
            },
        );
        self.push(Body::AssistantMessage {
            from,
            calls: calls
                .iter()
                .map(|c| CallRef {
                    id: (*c).into(),
                    name: "edit".into(),
                })
                .collect(),
        });
        true
    }

    /// A turn: a task, an answer that thinks and calls a tool, the tool's large result, a closing answer.
    fn turn(&mut self, n: u32) {
        // Between turns first: a server block can only be asked for here, before the next user message.
        if self
            .log
            .entries()
            .iter()
            .any(|e| matches!(e.body, Body::TurnEnd { .. }))
        {
            self.prepare();
        }
        self.push(Body::UserMessage {
            content: RawJson::string(&format!("task {n}")),
            steer: false,
        });
        let call = format!("toolu_{n}");
        let first = format!(
            r#"[{{"type":"thinking","thinking":"plan {n}","signature":"sig{n}"}},{{"type":"tool_use","id":"{call}","name":"edit","input":{{"n":{n}}}}}]"#
        );
        while !self.call(n, &first, &[&call]) {}
        self.push(Body::ToolStart {
            call: call.as_str().into(),
            tool: "edit".into(),
            effect: Effect::Write,
        });
        let output = "x".repeat(RESULT_BYTES);
        self.push(Body::ToolResult {
            call: call.as_str().into(),
            block: raw(&format!(
                r#"{{"type":"tool_result","tool_use_id":"{call}","content":"{output}"}}"#
            )),
            is_error: false,
            spill: None,
        });
        while !self.call(n, &format!(r#"[{{"type":"text","text":"done {n}"}}]"#), &[]) {}
        self.push(Body::TurnEnd {
            outcome: "finished".into(),
        });
    }

    /// The current segment's `segment.start`, if the session has one.
    fn segment_start(&self) -> Option<(Seq, Option<Strategy>, Vec<Seq>)> {
        self.log.entries().iter().rev().find_map(|e| match &e.body {
            Body::SegmentStart { strategy, seed, .. } => Some((e.seq, *strategy, seed.clone())),
            _ => None,
        })
    }

    fn check_request(&mut self, recipe: &Recipe) {
        let seg = self.log.segment();
        // The prefix check: within a segment, every request repeats the one before it and only appends.
        if let Some((s, before)) = &self.last_request
            && *s == seg
        {
            assert!(
                recipe.messages.starts_with(before),
                "a request in segment {seg:?} changed what an earlier one sent"
            );
        }
        self.last_request = Some((seg, recipe.messages.clone()));
        let blocks = recipe
            .messages
            .iter()
            .filter(|s| text(&self.log, **s).contains(r#""type":"compaction""#))
            .count();
        let Some((start, strategy, seed)) = self.segment_start() else {
            assert_eq!(blocks, 0);
            return;
        };
        if matches!(strategy, Some(Strategy::Server | Strategy::ClientSummary)) {
            // After a compaction: the one seed slice first, then only what came after the segment began.
            assert_eq!(seed.len(), 1);
            assert_eq!(recipe.messages[0], seed[0]);
            assert!(
                recipe.messages[1..].iter().all(|s| *s > start),
                "a replaced slice is sent again"
            );
        }
        match strategy {
            Some(Strategy::Server) => {
                assert_eq!(blocks, 1, "exactly one block after a server compaction")
            }
            Some(Strategy::ClientSummary) => assert_eq!(blocks, 0),
            // An overflow seed keeps an earlier server block as its head, first.
            _ => assert!(
                blocks == 0
                    || (blocks == 1
                        && text(&self.log, recipe.messages[0]).contains(r#""type":"compaction""#))
            ),
        }
        assert_eq!(carries_compaction_block(&self.log), blocks == 1);
    }

    /// After an overflow seed: kept earlier turns carry no thinking, and no tool result is sent without its call.
    fn check_overflow_segment(&self) {
        let (_, strategy, _) = self.segment_start().unwrap();
        assert_eq!(strategy, Some(Strategy::Deterministic));
        let texts = projected(&self.log);
        let current = texts
            .iter()
            .rposition(|t| t.starts_with(r#""task "#))
            .unwrap();
        for (i, t) in texts.iter().enumerate() {
            if i < current {
                assert!(
                    !t.contains(r#""type":"thinking""#),
                    "kept turn {i} still thinks"
                );
            }
            if let Some(rest) = t.strip_prefix(r#"{"type":"tool_result","tool_use_id":""#) {
                let id = &rest[..rest.find('"').unwrap()];
                let asked = format!(r#""id":"{id}""#);
                assert!(
                    texts[..i].iter().any(|a| a.contains(&asked)),
                    "tool result {id} kept without its call"
                );
            }
        }
    }
}

/// Runs 60 turns and returns the run with every compaction batch it made.
fn long_run(profile: ContextProfile, refuse_at: Option<u32>) -> (Run, Vec<Batch>) {
    let mut run = Run::new(profile);
    run.refuse_at = refuse_at;
    for n in 1..=TURNS {
        run.turn(n);
    }
    let batches = run
        .batch_starts
        .iter()
        .enumerate()
        .map(|(index, first)| {
            // The batch ends at the first `segment.start` after its request.
            let last = run
                .log
                .entries()
                .iter()
                .find(|e| e.seq > *first && matches!(e.body, Body::SegmentStart { .. }))
                .unwrap()
                .seq;
            Batch {
                index: u32::try_from(index).unwrap(),
                compactor: run.profile.compactor,
                first: *first,
                last,
                before: projected(&run.log.up_to(Seq(first.0 - 1)).unwrap()),
                after: projected(&run.log.up_to(last).unwrap()),
            }
        })
        .collect();
    (run, batches)
}

/// A restarted core: finishes an open compaction the way the core's resume does, else decides again.
fn resume(log: Log, profile: &ContextProfile, batch: &Batch) -> Log {
    let mut run = Run::new(profile.clone());
    run.log = log;
    run.compactions = batch.index;
    match pending_compaction(&run.log) {
        Some(p) => {
            if p.response.is_none() {
                // The call is open: model calls have no side effects, so it is made again.
                run.compaction_answer(p.compactor, batch.index);
            }
            let seed = compaction_seed(&run.log, run.ts()).unwrap();
            run.append(seed.entries);
        }
        None => {
            if !matches!(
                run.log.entries().last().unwrap().body,
                Body::SegmentStart { .. }
            ) {
                run.prepare();
            }
        }
    }
    run.log
}

fn check_long_run(compactor: Compactor) {
    let mut profile = ContextProfile::new(6_000, 1_000);
    profile.compactor = compactor;
    let (run, batches) = long_run(profile.clone(), None);
    assert!(
        run.compactions >= 2,
        "{compactor:?}: {} compactions",
        run.compactions
    );
    assert_eq!(usize::try_from(run.compactions).unwrap(), batches.len());
    assert_eq!(run.overflows, 0, "{compactor:?}: compaction comes first");
    assert!(estimate(&run.log) <= profile.limit());
    for batch in &batches {
        assert_eq!(batch.compactor, compactor);
        assert_ne!(batch.before, batch.after);
        // A restart after any entry of the batch, or just before it, ends in the same projection.
        for cut in batch.first.0 - 1..=batch.last.0 {
            let log = run.log.up_to(Seq(cut)).unwrap();
            let resumed = resume(log, &profile, batch);
            assert_eq!(
                projected(&resumed),
                batch.after,
                "{compactor:?}: compaction {} cut at {cut}",
                batch.index
            );
        }
    }
}

#[test]
fn sixty_turns_with_server_compaction() {
    check_long_run(Compactor::Server);
}

#[test]
fn sixty_turns_with_client_summaries() {
    check_long_run(Compactor::ClientSummary);
}

#[test]
fn a_refused_request_mid_run_is_recovered_without_a_model_call() {
    let profile = ContextProfile::new(6_000, 1_000);
    let (run, _) = long_run(profile, Some(30));
    assert_eq!(run.overflows, 1);
    assert!(run.compactions >= 2);
    // Every request was answered once, except the refused one, which was made again after the seed.
    let requests = run
        .log
        .entries()
        .iter()
        .filter(|e| matches!(e.body, Body::ModelRequest { .. }))
        .count();
    assert_eq!(u32::try_from(requests).unwrap(), run.calls);
    assert_eq!(run.calls, 2 * TURNS + 1 + run.compactions);
}
