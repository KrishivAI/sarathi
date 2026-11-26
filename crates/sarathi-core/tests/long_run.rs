// SPDX-License-Identifier: Apache-2.0
//! A long run through the driver: 60 turns, each a task, a thinking answer that calls a tool, a large tool result
//! and a closing answer, through a small window, with each compactor. The model answers from what it is sent, so a
//! restarted run asks and gets the same as an unbroken one.
//!
//! It checks that the run compacts more than once, that every request after a compaction starts with its one seed
//! slice and holds nothing it replaced, that requests inside a segment only ever append (the prefix check), that a
//! restart after any append of a compaction, or inside one (a prefix of its entries written), ends in the same
//! conversation as the unbroken run, and that a request the
//! provider refused as too large is recovered without a model call.

#![expect(clippy::unwrap_used, clippy::panic, reason = "test code")]

pub mod common;

use std::future::Future;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use common::*;
use sarathi_context::{SUMMARY_PROMPT, carries_compaction_block, estimate};
use sarathi_core::anthropic::{self, PROTOCOL};
use sarathi_core::driver::{
    BoxFuture, CallFailure, Compactor, ContextProfile, Driver, DriverConfig, ModelCaller, Received,
    StopSignal, Tools,
};
use sarathi_core::{ModelProfile, ToolDef, Verdict};
use sarathi_log::{
    Body, Durability, Entry, ForkPoint, Loaded, Log, LogError, LogHeader, LogStore, MemoryStore,
    RequestParts, SessionId, Slice, Strategy,
};
use sarathi_protocol::{CallId, ErrorKind, Outcome, RawJson, Seq, ToolOutput};

const TURNS: u32 = 60;
/// Bytes of each tool result: about 430 tokens by the 2-per-7 estimate, so a compaction every few turns.
const RESULT_BYTES: usize = 1_500;

/// Tokens the scripted provider reports for `bytes` sent: 2 per 7, the rule the estimate falls back to.
fn tokens(bytes: usize) -> u64 {
    u64::try_from(bytes * 2).unwrap().div_ceil(7)
}

fn text_of(body: &[u8]) -> String {
    String::from_utf8(body.to_vec()).unwrap()
}

/// The highest `task <n>` a request mentions: the task the run is on.
fn task_of(text: &str) -> u32 {
    text.match_indices("task ")
        .filter_map(|(i, m)| {
            let digits: String = text[i + m.len()..]
                .chars()
                .take_while(char::is_ascii_digit)
                .collect();
            digits.parse().ok()
        })
        .max()
        .unwrap_or(0)
}

/// Whether a request already holds the answer to task `n`'s tool call (its result, or a summary that says so).
fn answered(text: &str, n: u32) -> bool {
    text.contains(&format!(r#""tool_use_id":"toolu_{n}""#))
        || text.contains(&format!("toolu_{n} answered"))
}

fn message(content: &str, stop: &str, input: u64, output: u64) -> String {
    format!(
        r#"{{"id":"msg_01","type":"message","role":"assistant","model":"scripted","content":[{content}],"stop_reason":"{stop}","usage":{{"input_tokens":{input},"output_tokens":{output}}}}}"#
    )
}

/// The model: answers a request from its bytes alone, and keeps every body it was sent.
///
/// A compaction call gets a block (server) or a summary (client) naming the task the run is on; a turn request whose
/// task's tool call has no answer yet gets a thinking block and an `edit` call (a write); one that has it gets `done <n>`.
/// With `refuse`, the first request of that task is refused once as too large.
#[derive(Debug, Clone, Default)]
struct Long {
    bodies: Arc<Mutex<Vec<Vec<u8>>>>,
    refuse: Option<u32>,
    refused: Arc<AtomicBool>,
}

impl Long {
    fn bodies(&self) -> Vec<String> {
        self.bodies
            .lock()
            .unwrap()
            .iter()
            .map(|b| text_of(b))
            .collect()
    }

    fn compaction_calls(&self) -> usize {
        self.bodies()
            .iter()
            .filter(|t| is_server_call(t) || is_summary_call(t))
            .count()
    }

    fn reply(&self, text: &str) -> Result<String, CallFailure> {
        let n = task_of(text);
        let input = tokens(text.len());
        if is_server_call(text) {
            let block = format!(
                r#"{{"type":"compaction","content":"Summary up to task {n}: earlier tasks done.","signature":"c2ln{n}"}}"#
            );
            return Ok(message(&block, "compaction", input, 20));
        }
        if is_summary_call(text) {
            let open = if answered(text, n) {
                format!(" toolu_{n} answered.")
            } else {
                String::new()
            };
            let said = format!("Summary up to task {n}: earlier tasks done.{open}");
            let block = format!(
                r#"{{"type":"text","text":{}}}"#,
                RawJson::string(&said).as_str()
            );
            return Ok(message(&block, "end_turn", input, 20));
        }
        if answered(text, n) {
            let block = format!(r#"{{"type":"text","text":"done {n}"}}"#);
            return Ok(message(&block, "end_turn", input, 5));
        }
        if self.refuse == Some(n) && !self.refused.swap(true, Ordering::SeqCst) {
            return Err(CallFailure::new(
                ErrorKind::ContextOverflow,
                "prompt is too long",
            ));
        }
        let blocks = format!(
            r#"{{"type":"thinking","thinking":"plan {n}","signature":"sig{n}"}},{{"type":"tool_use","id":"toolu_{n}","name":"edit","input":{{"n":{n}}}}}"#
        );
        Ok(message(&blocks, "tool_use", input, 30))
    }
}

fn is_server_call(text: &str) -> bool {
    text.contains(r#""compaction":{"type":"summarize"}"#)
}

fn is_summary_call(text: &str) -> bool {
    text.contains(RawJson::string(SUMMARY_PROMPT).as_str())
}

impl ModelCaller for Long {
    type Prepared = Vec<u8>;

    fn protocol(&self) -> &str {
        PROTOCOL
    }

    fn params(&self) -> RawJson {
        raw(r#"{"model":"scripted","max_tokens":1024}"#)
    }

    fn render_tools(&self, tools: &[ToolDef]) -> Result<Option<RawJson>, CallFailure> {
        anthropic::render_tools(tools)
    }

    fn result_block(&self, call: &CallId, content: &RawJson, is_error: bool) -> RawJson {
        anthropic::result_block(call, content, is_error)
    }

    fn prepare(&self, parts: &RequestParts<'_>) -> Result<(Vec<u8>, Vec<u8>), CallFailure> {
        let body = assemble(parts);
        Ok((body.clone(), body))
    }

    fn send(&self, body: Vec<u8>) -> BoxFuture<'_, Result<Received, CallFailure>> {
        let text = text_of(&body);
        self.bodies.lock().unwrap().push(body);
        let reply = self.reply(&text);
        Box::pin(async move { anthropic::answer(&raw(&reply?)) })
    }

    fn read(&self, body: &RawJson, content: Range<u32>) -> Result<Received, CallFailure> {
        let mut r = anthropic::answer(body)?;
        r.content = content;
        Ok(r)
    }

    /// The provider's count is the same rule as the reported usage.
    fn count_tokens(
        &self,
        parts: &RequestParts<'_>,
    ) -> Option<BoxFuture<'_, Result<u64, CallFailure>>> {
        let n = tokens(assemble(parts).len());
        Some(Box::pin(async move { Ok(n) }))
    }
}

/// Every tool returns [`RESULT_BYTES`] of output (the run calls `edit`, a write, so each call is started durably).
#[derive(Debug, Clone, Default)]
struct Big;

impl Tools for Big {
    fn evaluate(&self, _tool: &str, _input: &RawJson) -> Verdict {
        Verdict::Allow
    }

    fn run(
        &self,
        _call: CallId,
        _tool: Box<str>,
        _input: RawJson,
        _stop: StopSignal,
    ) -> BoxFuture<'_, ToolOutput> {
        Box::pin(async {
            ToolOutput {
                content: RawJson::string(&"x".repeat(RESULT_BYTES)),
                is_error: false,
            }
        })
    }
}

/// A store that records the first seq of every append it is asked for, so a test can find the append that wrote
/// an entry.
#[derive(Debug, Clone, Default)]
struct Recording {
    inner: Shared,
    starts: Arc<Mutex<Vec<Seq>>>,
}

impl LogStore for Recording {
    fn create(
        &self,
        created: u64,
        forked_from: Option<ForkPoint>,
    ) -> impl Future<Output = Result<LogHeader, LogError>> + Send {
        self.inner.create(created, forked_from)
    }

    async fn append(
        &self,
        session: SessionId,
        expected: Seq,
        entries: Vec<Entry>,
        durability: Durability,
    ) -> Result<Seq, LogError> {
        self.starts.lock().unwrap().push(expected);
        self.inner
            .append(session, expected, entries, durability)
            .await
    }

    fn load(&self, session: SessionId) -> impl Future<Output = Result<Loaded, LogError>> + Send {
        self.inner.load(session)
    }
}

fn profile(compactor: Compactor) -> ContextProfile {
    let mut p = ContextProfile::new(6_000, 1_000);
    p.compactor = compactor;
    p
}

fn cfg(context: &ContextProfile) -> DriverConfig {
    config(ModelProfile::native())
        .with_limits(limits(10))
        .with_context(context.clone())
}

/// The text a projected slice sends.
fn slice_text(log: &Log, seq: Seq) -> String {
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
        .map(|s| slice_text(log, *s))
        .collect()
}

/// Whether the log holds the user message of task `n`.
fn has_task(log: &Log, n: u32) -> bool {
    let task = RawJson::string(&format!("task {n}"));
    log.entries().iter().any(|e| {
        matches!(&e.body, Body::UserMessage { content, .. } if content.as_str() == task.as_str())
    })
}

/// Runs tasks `from..=TURNS` (finishing an open turn first); a task the log does not yet hold is sent again.
/// A turn of task `refused` may end unfinished (its request was refused); any other that does not finish stops the
/// run (the store stopped): `false`.
async fn run_from(
    d: &mut Driver<impl LogStore, Long, Big, Host>,
    from: u32,
    refused: Option<u32>,
) -> bool {
    if d.in_turn() {
        match d.continue_turn(&mut |_| {}).await {
            Ok(Some(end)) if end.outcome == Outcome::Finished => {}
            _ => return false,
        }
    }
    for n in from..=TURNS {
        if has_task(d.log(), n) {
            continue;
        }
        match d.run_turn(user(&format!("task {n}")), &mut |_| {}).await {
            Ok(end) if end.outcome == Outcome::Finished => {}
            // The refused request ends its turn; the run goes on with the next task.
            Ok(_) if refused == Some(n) => {}
            _ => return false,
        }
    }
    true
}

/// The segment-start entries of the log: `(seq, strategy, seed)`.
fn segments(log: &Log) -> Vec<(Seq, Option<Strategy>, Vec<Seq>)> {
    log.entries()
        .iter()
        .filter_map(|e| match &e.body {
            Body::SegmentStart { strategy, seed, .. } => Some((e.seq, *strategy, seed.clone())),
            _ => None,
        })
        .collect()
}

/// Checks every turn request of the log as it was sent: the prefix check within a segment, the seed first after a
/// compaction with nothing it replaced, one block after a server compaction, and after an overflow seed no tool
/// result without its call and, in the first request, no thinking in the turns it kept.
fn check_requests(log: &Log) {
    let mut last: Option<(Seq, Vec<Seq>)> = None;
    let starts = segments(log);
    for e in log.entries() {
        let Body::ModelRequest {
            recipe, compaction, ..
        } = &e.body
        else {
            continue;
        };
        if compaction.is_some() {
            continue;
        }
        let segment = starts.iter().rev().find(|s| s.0 < e.seq);
        let seg = segment.map_or(Seq(0), |s| s.0);
        let first_in_segment = last.as_ref().is_none_or(|(s, _)| *s != seg);
        if let Some((s, before)) = &last
            && *s == seg
        {
            assert!(
                recipe.messages.starts_with(before),
                "request {:?} changed what an earlier one in its segment sent",
                e.seq
            );
        }
        last = Some((seg, recipe.messages.clone()));
        let texts: Vec<String> = recipe
            .messages
            .iter()
            .map(|s| slice_text(log, *s))
            .collect();
        let blocks = texts
            .iter()
            .filter(|t| t.contains(r#""type":"compaction""#))
            .count();
        let before = log.up_to(Seq(e.seq.0 - 1)).unwrap();
        assert_eq!(
            carries_compaction_block(&before),
            blocks == 1,
            "{:?}",
            e.seq
        );
        let Some((start, strategy, seed)) = segment else {
            assert_eq!(blocks, 0);
            continue;
        };
        match strategy {
            Some(Strategy::Server | Strategy::ClientSummary) => {
                assert_eq!(seed.len(), 1);
                assert_eq!(recipe.messages[0], seed[0]);
                assert!(
                    recipe.messages[1..].iter().all(|s| s > start),
                    "request {:?} sends a replaced slice again",
                    e.seq
                );
                let want = usize::from(*strategy == Some(Strategy::Server));
                assert_eq!(blocks, want, "request {:?}", e.seq);
            }
            _ => {
                // An overflow seed keeps an earlier server block as its head, first.
                assert!(
                    blocks == 0 || (blocks == 1 && texts[0].contains(r#""type":"compaction""#))
                );
                let current = texts
                    .iter()
                    .rposition(|t| t.starts_with(r#""task "#))
                    .unwrap();
                for (i, t) in texts.iter().enumerate() {
                    // The seed's own request: the turns it kept (all before the current one) were stripped.
                    if first_in_segment && i < current {
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
    }
}

/// The projection right after each segment start: what every compaction left.
fn projections_at_segments(log: &Log) -> Vec<Vec<String>> {
    segments(log)
        .iter()
        .map(|s| projected(&log.up_to(s.0).unwrap()))
        .collect()
}

/// The compactions of the log: the seqs from each compaction request to the segment start it ends in.
fn compactions(log: &Log) -> Vec<(Seq, Seq)> {
    let starts = segments(log);
    log.entries()
        .iter()
        .filter(|e| {
            matches!(
                &e.body,
                Body::ModelRequest {
                    compaction: Some(_),
                    ..
                }
            )
        })
        .map(|e| {
            let end = starts.iter().find(|s| s.0 > e.seq).unwrap().0;
            (e.seq, end)
        })
        .collect()
}

/// An unbroken run: the store, the model and the log.
async fn unbroken(context: &ContextProfile, refuse: Option<u32>) -> (Recording, Long, Log) {
    let store = Recording::default();
    let model = Long {
        refuse,
        ..Long::default()
    };
    let (mut d, _) = Driver::new(cfg(context), store.clone(), model.clone(), Big, Host(None))
        .await
        .unwrap();
    assert!(run_from(&mut d, 1, refuse).await);
    let log = d.log().clone();
    check_log(&log);
    let stored = store.inner.inner.load(d.session()).await.unwrap().log;
    assert_eq!(stored, log);
    (store, model, log)
}

async fn check_long_run(compactor: Compactor) {
    let context = profile(compactor);
    let (store, model, log) = unbroken(&context, None).await;
    let batches = compactions(&log);
    assert!(
        batches.len() >= 2,
        "{compactor:?}: {} compactions",
        batches.len()
    );
    assert_eq!(model.compaction_calls(), batches.len());
    let starts = segments(&log);
    assert_eq!(
        starts.len(),
        batches.len(),
        "{compactor:?}: compaction comes first"
    );
    let want = match compactor {
        Compactor::Server => Strategy::Server,
        _ => Strategy::ClientSummary,
    };
    assert!(starts.iter().all(|s| s.1 == Some(want)), "{starts:?}");
    assert!(estimate(&log) <= context.limit());
    check_requests(&log);
    let full = projected(&log);
    let at_starts = projections_at_segments(&log);

    // A restart after any append of a compaction, or just before it, ends in the same conversation.
    let appends = store.starts.lock().unwrap().clone();
    let mut sent_again = 0;
    for (first, last) in &batches {
        // From the append that wrote the batch's first entry (a client summary writes its prompt with the request),
        // through one past the append of its segment start.
        let lo = appends.iter().rposition(|s| s <= first).unwrap();
        let hi = appends.iter().rposition(|s| s <= last).unwrap() + 1;
        let (lo, hi) = (u32::try_from(lo).unwrap(), u32::try_from(hi).unwrap());
        for n in lo..=hi {
            let model = Long::default();
            let store = Recording {
                inner: Shared::default().stopping_after(n),
                ..Recording::default()
            };
            let (mut d, _) =
                Driver::new(cfg(&context), store.clone(), model.clone(), Big, Host(None))
                    .await
                    .unwrap();
            assert!(
                !run_from(&mut d, 1, None).await,
                "{compactor:?} n={n}: the store stopped"
            );
            let session = d.session();
            drop(d);
            let (mut again, _) = Driver::resume(
                cfg(&context),
                store.inner.reopened(),
                session,
                model.clone(),
                Big,
                Host(None),
            )
            .await
            .unwrap();
            assert!(run_from(&mut again, 1, None).await, "{compactor:?} n={n}");
            let resumed = again.log();
            check_log(resumed);
            check_requests(resumed);
            assert_eq!(projected(resumed), full, "{compactor:?} n={n}");
            assert_eq!(
                projections_at_segments(resumed),
                at_starts,
                "{compactor:?} n={n}"
            );
            assert_eq!(
                compactions(resumed).len(),
                batches.len(),
                "{compactor:?} n={n}"
            );
            let calls = model.compaction_calls();
            assert!(calls <= batches.len() + 1, "{compactor:?} n={n}: {calls}");
            sent_again += usize::from(calls == batches.len() + 1);
        }
    }
    assert!(
        sent_again > 0,
        "{compactor:?}: some stop fell between a compaction request and its answer"
    );
}

/// A store that writes only the first `keep` entries of append `at` and then dies, as a file store can leave a
/// prefix of a batch after a crash.
#[derive(Debug, Clone)]
struct Torn {
    inner: Arc<MemoryStore>,
    count: Arc<AtomicU32>,
    at: u32,
    keep: usize,
}

impl LogStore for Torn {
    fn create(
        &self,
        created: u64,
        forked_from: Option<ForkPoint>,
    ) -> impl Future<Output = Result<LogHeader, LogError>> + Send {
        self.inner.create(created, forked_from)
    }

    async fn append(
        &self,
        session: SessionId,
        expected: Seq,
        mut entries: Vec<Entry>,
        durability: Durability,
    ) -> Result<Seq, LogError> {
        let n = self.count.fetch_add(1, Ordering::SeqCst);
        if n < self.at {
            return self
                .inner
                .append(session, expected, entries, durability)
                .await;
        }
        if n == self.at {
            entries.truncate(self.keep);
            if !entries.is_empty() {
                self.inner
                    .append(session, expected, entries, durability)
                    .await?;
            }
        }
        Err(LogError::Io(std::io::Error::other("the process is gone")))
    }

    fn load(&self, session: SessionId) -> impl Future<Output = Result<Loaded, LogError>> + Send {
        self.inner.load(session)
    }
}

/// A crash inside any multi-entry append of a compaction (a prefix of its entries on disk) ends, after a resume, in
/// the same conversation as the unbroken run.
async fn check_torn_appends(compactor: Compactor) {
    let context = profile(compactor);
    let (store, _, log) = unbroken(&context, None).await;
    let full = projected(&log);
    let at_starts = projections_at_segments(&log);
    let batches = compactions(&log);
    let appends = store.starts.lock().unwrap().clone();
    let mut torn = 0;
    for (first, last) in &batches {
        let lo = appends.iter().rposition(|s| s <= first).unwrap();
        let hi = appends.iter().rposition(|s| s <= last).unwrap();
        for i in lo..=hi {
            let end = appends.get(i + 1).map_or(log.next_seq().0, |s| s.0);
            let size = usize::try_from(end - appends[i].0).unwrap();
            for keep in 1..size {
                torn += 1;
                let memory = Arc::new(MemoryStore::default());
                let store = Torn {
                    inner: memory.clone(),
                    count: Arc::default(),
                    at: u32::try_from(i).unwrap(),
                    keep,
                };
                let model = Long::default();
                let (mut d, _) = Driver::new(cfg(&context), store, model.clone(), Big, Host(None))
                    .await
                    .unwrap();
                let at = format!("{compactor:?} append {i} keep {keep}/{size}");
                assert!(!run_from(&mut d, 1, None).await, "{at}: the store stopped");
                let session = d.session();
                drop(d);
                let reopened = Shared {
                    inner: memory,
                    ..Shared::default()
                };
                let (mut again, _) =
                    Driver::resume(cfg(&context), reopened, session, model, Big, Host(None))
                        .await
                        .unwrap_or_else(|e| panic!("{at}: {e}"));
                assert!(run_from(&mut again, 1, None).await, "{at}");
                let resumed = again.log();
                check_log(resumed);
                check_requests(resumed);
                assert_eq!(projected(resumed), full, "{at}");
                assert_eq!(projections_at_segments(resumed), at_starts, "{at}");
            }
        }
    }
    assert!(
        torn > 0,
        "{compactor:?}: no compaction append holds more than one entry"
    );
}

#[tokio::test]
async fn a_crash_inside_a_compaction_append_with_server_compaction() {
    check_torn_appends(Compactor::Server).await;
}

#[tokio::test]
async fn a_crash_inside_a_compaction_append_with_client_summaries() {
    check_torn_appends(Compactor::ClientSummary).await;
}

#[tokio::test]
async fn sixty_turns_with_server_compaction() {
    check_long_run(Compactor::Server).await;
}

#[tokio::test]
async fn sixty_turns_with_client_summaries() {
    check_long_run(Compactor::ClientSummary).await;
}

#[tokio::test]
async fn a_refused_request_mid_run_is_recovered_without_a_model_call() {
    let context = profile(Compactor::Server);
    // Refuse the task just before the unbroken run's second compaction: its segment is then at its largest, so the
    // seed (at most half of it) still keeps earlier turns.
    let (_, _, plain) = unbroken(&context, None).await;
    let second = compactions(&plain)[1].0;
    let refuse = plain
        .entries()
        .iter()
        .find_map(|e| match &e.body {
            Body::UserMessage { content, .. } if e.seq > second => Some(task_of(content.as_str())),
            _ => None,
        })
        .unwrap()
        - 1;
    let (_, model, log) = unbroken(&context, Some(refuse)).await;
    let starts = segments(&log);
    let overflows = starts
        .iter()
        .filter(|s| s.1 == Some(Strategy::Deterministic))
        .count();
    assert_eq!(overflows, 1, "{starts:?}");
    // The seed kept earlier turns, so its request's checks (no thinking, no lone tool result) had something to check.
    let seed = starts
        .iter()
        .find(|s| s.1 == Some(Strategy::Deterministic))
        .unwrap();
    let kept = log
        .entries()
        .iter()
        .find_map(|e| match &e.body {
            Body::ModelRequest { recipe, .. } if e.seq > seed.0 => Some(recipe.messages.clone()),
            _ => None,
        })
        .unwrap();
    assert!(
        kept.iter()
            .any(|s| slice_text(&log, *s).contains(r#""type":"tool_use""#)),
        "the overflow seed kept no earlier turn (refused task {refuse})"
    );
    // The seed's request starts with the server block of the compaction before it.
    assert!(slice_text(&log, kept[0]).contains(r#""type":"compaction""#));
    // The turn ended on the refusal: one model failure, a context overflow. The task is not sent again.
    // The log records the kind by its name.
    let failures: Vec<&str> = log
        .entries()
        .iter()
        .filter_map(|e| match &e.body {
            Body::ModelFailure { error, .. } => Some(&**error),
            _ => None,
        })
        .collect();
    assert_eq!(failures, [ErrorKind::ContextOverflow.as_str()]);
    let batches = compactions(&log);
    assert!(batches.len() >= 2);
    check_requests(&log);
    // Every request was sent once: two per turn, but one for the refused turn, and one per compaction. The overflow
    // seed made no call.
    let requests = log
        .entries()
        .iter()
        .filter(|e| matches!(e.body, Body::ModelRequest { .. }))
        .count();
    assert_eq!(model.bodies().len(), requests);
    assert_eq!(
        requests,
        2 * usize::try_from(TURNS).unwrap() - 1 + batches.len()
    );
}
