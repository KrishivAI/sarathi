// SPDX-License-Identifier: Apache-2.0
//! Context management through the driver: compaction between turns with each compactor, a client summary inside
//! a long tool loop, a compaction answer that is no summary, the overflow fallback after the provider refused a
//! request, a restart after every append of a compaction, and tool output spilled outside the log.

#![expect(clippy::unwrap_used, clippy::panic, reason = "test code")]

pub mod common;

use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::*;
use sarathi_context::{
    ContextError, MemorySpill, SUMMARY_PREFIX, SUMMARY_PROMPT, SpillRecord, SpillStore, read_output,
};
use sarathi_core::anthropic::{self, PROTOCOL};
use sarathi_core::driver::{
    BoxFuture, CallFailure, Compactor, ContextProfile, Driver, DriverConfig, DriverError,
    ModelCaller, Received, Spill, SpillLimits,
};
use sarathi_core::{ModelProfile, ToolDef};
use sarathi_log::{Body, Log, LogStore, RequestParts, SegmentReason, SessionId, Strategy};
use sarathi_protocol::{CallId, ErrorKind, Event, Op, Outcome, RawJson};

/// A window of 1000 tokens with 200 kept for the answer: the limit is 800, compaction is due at 600.
fn profile(compactor: Compactor) -> ContextProfile {
    let mut p = ContextProfile::new(1000, 200);
    p.compactor = compactor;
    p
}

/// A Messages answer of `blocks` that reports `input` input tokens.
fn using(blocks: &str, stop: &str, input: u64) -> Reply {
    Reply::Body(format!(
        r#"{{"id":"msg_01","type":"message","role":"assistant","model":"scripted","content":[{blocks}],"stop_reason":"{stop}","usage":{{"input_tokens":{input},"output_tokens":5}}}}"#
    ))
}

fn said(text: &str, input: u64) -> Reply {
    using(
        &format!(
            r#"{{"type":"text","text":{}}}"#,
            RawJson::string(text).as_str()
        ),
        "end_turn",
        input,
    )
}

fn echo_call(n: u32, input: u64) -> Reply {
    using(
        &format!(r#"{{"type":"tool_use","id":"toolu_{n}","name":"echo","input":{{"n":{n}}}}}"#),
        "tool_use",
        input,
    )
}

const BLOCK: &str = r#"{"type":"compaction","content":"The user said hello; the model answered.","signature":"sig-1"}"#;

fn compacted() -> Reply {
    using(BLOCK, "compaction", 700)
}

fn text_of(body: &[u8]) -> String {
    String::from_utf8(body.to_vec()).unwrap()
}

async fn driver_with(
    context: ContextProfile,
    replies: Vec<Reply>,
    tools: TestTools,
) -> (TestDriver, Scripted, Shared) {
    let model = Scripted::new(replies);
    let store = Shared::default();
    let config = config(ModelProfile::native())
        .with_limits(limits(10))
        .with_context(context);
    let (d, _) = Driver::new(config, store.clone(), model.clone(), tools, Host(None))
        .await
        .unwrap();
    (d, model, store)
}

fn segment_starts(log: &Log) -> Vec<(Option<Strategy>, SegmentReason)> {
    log.entries()
        .iter()
        .filter_map(|e| match &e.body {
            Body::SegmentStart {
                strategy, reason, ..
            } => Some((*strategy, *reason)),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn a_server_compaction_between_turns_seeds_the_next_turn_with_its_block() {
    let replies = vec![said("Hello.", 700), compacted(), said("Second.", 50)];
    let (mut d, model, store) =
        driver_with(profile(Compactor::Server), replies, TestTools::default()).await;
    let first = d
        .run_turn(user("Say hello first."), &mut |_| {})
        .await
        .unwrap();
    assert_eq!(first.outcome, Outcome::Finished);
    let mut events = Vec::new();
    let second = d
        .run_turn(user("Now the next thing."), &mut sink(&mut events))
        .await
        .unwrap();
    assert_eq!(second.reply.as_deref(), Some("Second."));
    let bodies = model.bodies();
    assert_eq!(bodies.len(), 3);
    let call = text_of(&bodies[1]);
    assert!(
        call.contains(r#""compaction":{"type":"summarize"}"#),
        "{call}"
    );
    assert!(call.contains("Say hello first."), "{call}");
    let next = text_of(&bodies[2]);
    assert!(
        next.contains(BLOCK),
        "the block goes back byte for byte: {next}"
    );
    assert!(next.contains("Now the next thing."), "{next}");
    assert!(!next.contains("Say hello first."), "{next}");
    assert_eq!(
        segment_starts(d.log()),
        vec![(Some(Strategy::Server), SegmentReason::Compaction)]
    );
    assert!(events.iter().any(|e| matches!(
        e,
        Event::SegmentStarted {
            reason: SegmentReason::Compaction,
            ..
        }
    )));
    check_log(d.log());
    let stored = store.inner.load(d.session()).await.unwrap().log;
    assert_eq!(&stored, d.log());
}

#[tokio::test]
async fn a_client_summary_between_turns_becomes_the_next_segments_first_message() {
    let replies = vec![
        said("Hello.", 700),
        said("They greeted; nothing is open.", 700),
        said("Second.", 50),
    ];
    let (mut d, model, _) = driver_with(
        profile(Compactor::ClientSummary),
        replies,
        TestTools::default(),
    )
    .await;
    d.run_turn(user("Say hello first."), &mut |_| {})
        .await
        .unwrap();
    let second = d
        .run_turn(user("Now the next thing."), &mut |_| {})
        .await
        .unwrap();
    assert_eq!(second.reply.as_deref(), Some("Second."));
    let bodies = model.bodies();
    assert_eq!(bodies.len(), 3);
    let call = text_of(&bodies[1]);
    assert!(
        call.contains(RawJson::string(SUMMARY_PROMPT).as_str()),
        "{call}"
    );
    assert!(!call.contains(r#""compaction""#), "{call}");
    let next = text_of(&bodies[2]);
    let prefix = RawJson::string(SUMMARY_PREFIX);
    let prefix = prefix.as_str().trim_end_matches('"');
    assert!(next.contains(prefix), "{next}");
    assert!(next.contains("They greeted; nothing is open."), "{next}");
    assert!(!next.contains("Say hello first."), "{next}");
    assert_eq!(
        segment_starts(d.log()),
        vec![(Some(Strategy::ClientSummary), SegmentReason::Compaction)]
    );
    check_log(d.log());
}

#[tokio::test]
async fn a_long_tool_loop_is_summarized_in_the_turn_and_goes_on() {
    let replies = vec![
        echo_call(1, 50),
        echo_call(2, 700),
        said("Summary: two echoes ran; one more is due.", 700),
        echo_call(3, 60),
        said("Done.", 70),
    ];
    let tools = TestTools::default();
    let (mut d, model, _) =
        driver_with(profile(Compactor::ClientSummary), replies, tools.clone()).await;
    let mut events = Vec::new();
    let end = d
        .run_turn(user("Echo three times."), &mut sink(&mut events))
        .await
        .unwrap();
    assert_eq!(end.outcome, Outcome::Finished, "{events:#?}");
    assert_eq!(end.reply.as_deref(), Some("Done."));
    assert_eq!(tools.tally.ran.lock().unwrap().len(), 3);
    let bodies = model.bodies();
    assert_eq!(bodies.len(), 5);
    let after = text_of(&bodies[3]);
    assert!(after.contains("two echoes ran"), "{after}");
    assert!(!after.contains("toolu_1"), "{after}");
    assert_eq!(
        segment_starts(d.log()),
        vec![(Some(Strategy::ClientSummary), SegmentReason::Compaction)]
    );
    let turns = events
        .iter()
        .filter(|e| matches!(e, Event::TurnStarted { .. }))
        .count();
    assert_eq!(turns, 1, "the turn is not started again");
    check_log(d.log());
}

/// Usage events: the seq of each answer the host is told about, and its cost.
fn usage_of(events: &[Event]) -> Vec<(u64, Option<u64>)> {
    events
        .iter()
        .filter_map(|e| match e {
            Event::Usage {
                seq, cost_micros, ..
            } => Some((seq.0, *cost_micros)),
            _ => None,
        })
        .collect()
}

/// A driver whose turns may spend at most `cap` micro-dollars.
async fn capped(context: ContextProfile, cap: u64, replies: Vec<Reply>) -> (TestDriver, Scripted) {
    let model = Scripted::new(replies);
    let mut limits = limits(10);
    limits.max_cost_micros = Some(cap);
    let config = config(ModelProfile::native())
        .with_limits(limits)
        .with_context(context);
    let (d, _) = Driver::new(
        config,
        Shared::default(),
        model.clone(),
        TestTools::default(),
        Host(None),
    )
    .await
    .unwrap();
    (d, model)
}

#[tokio::test]
async fn a_summary_in_the_turn_is_reported_as_usage_and_counts_against_the_spend_cap() {
    let replies = vec![
        echo_call(1, 50).priced(100),
        echo_call(2, 700).priced(100),
        said("Summary: two echoes ran; one more is due.", 700).priced(100),
        echo_call(3, 60).priced(100),
        said("Done.", 70).priced(100),
    ];
    let (mut d, model) = capped(profile(Compactor::ClientSummary), 300, replies).await;
    let mut events = Vec::new();
    let end = d
        .run_turn(user("Echo three times."), &mut sink(&mut events))
        .await
        .unwrap();
    assert_eq!(end.outcome, Outcome::RanOut, "{events:#?}");
    assert_eq!(model.left(), 2, "no call after the summary reached the cap");
    let usage = usage_of(&events);
    assert_eq!(usage.len(), 3, "{usage:?}");
    assert!(usage.iter().all(|(_, c)| *c == Some(100)), "{usage:?}");
    let summary = usage[2].0;
    assert!(
        matches!(
            d.log().get(sarathi_protocol::Seq(summary)).map(|e| &e.body),
            Some(Body::ModelResponse { .. })
        ),
        "the third usage event names the summary's answer"
    );
    assert!(events.iter().any(|e| matches!(
        e,
        Event::Error {
            kind: ErrorKind::BudgetExceeded,
            ..
        }
    )));
    check_log(d.log());
}

#[tokio::test]
async fn a_compaction_before_a_turn_is_reported_as_usage_and_charged_to_that_turn() {
    let replies = vec![
        said("Hello.", 700).priced(100),
        compacted().priced(100),
        echo_call(1, 50).priced(100),
        said("never", 60).priced(100),
    ];
    let (mut d, model) = capped(profile(Compactor::Server), 150, replies).await;
    let first = d
        .run_turn(user("Say hello first."), &mut |_| {})
        .await
        .unwrap();
    assert_eq!(first.outcome, Outcome::Finished);
    let mut events = Vec::new();
    let second = d
        .run_turn(user("Now echo."), &mut sink(&mut events))
        .await
        .unwrap();
    assert_eq!(second.outcome, Outcome::RanOut, "{events:#?}");
    assert_eq!(
        model.left(),
        1,
        "the compaction and the first call reach the cap"
    );
    assert_eq!(
        usage_of(&events)
            .iter()
            .map(|(_, c)| *c)
            .collect::<Vec<_>>(),
        vec![Some(100), Some(100)],
        "the compaction call is reported, then the turn's call"
    );
    check_log(d.log());
}

#[tokio::test]
async fn a_compaction_answer_that_is_no_summary_fails_alone_and_the_turn_goes_on() {
    let replies = vec![
        echo_call(1, 700),
        using(r#"{"type":"text","text":"I cannot."}"#, "refusal", 700),
        said("Done.", 710),
    ];
    let (mut d, model, _) = driver_with(
        profile(Compactor::ClientSummary),
        replies,
        TestTools::default(),
    )
    .await;
    let mut events = Vec::new();
    let end = d
        .run_turn(user("Echo once."), &mut sink(&mut events))
        .await
        .unwrap();
    assert_eq!(end.outcome, Outcome::Finished, "{events:#?}");
    assert_eq!(model.bodies().len(), 3, "the segment waits for an answer");
    assert!(segment_starts(d.log()).is_empty());
    let failure = d
        .log()
        .entries()
        .iter()
        .find_map(|e| match &e.body {
            Body::ModelFailure { error, .. } => Some(error.to_string()),
            _ => None,
        })
        .unwrap();
    assert_eq!(failure, "refusal");
    assert!(events.iter().any(|e| matches!(e, Event::Error { .. })));
    check_log(d.log());
}

#[tokio::test]
async fn a_request_the_provider_refused_as_too_large_is_cut_down_without_a_model_call() {
    // Compaction never due before the window: only the fallback acts.
    let mut p = profile(Compactor::Server);
    p.compact_at_percent = 255;
    let replies = vec![
        said("First.", 300),
        Reply::Fail(ErrorKind::ContextOverflow),
        said("Third.", 50),
    ];
    let (mut d, model, _) = driver_with(p, replies, TestTools::default()).await;
    d.run_turn(user("The first request."), &mut |_| {})
        .await
        .unwrap();
    let refused = d
        .run_turn(user("The second request."), &mut |_| {})
        .await
        .unwrap();
    assert_ne!(refused.outcome, Outcome::Finished);
    let mut events = Vec::new();
    let third = d
        .run_turn(user("The third request."), &mut sink(&mut events))
        .await
        .unwrap();
    assert_eq!(third.reply.as_deref(), Some("Third."), "{events:#?}");
    let bodies = model.bodies();
    assert_eq!(bodies.len(), 3, "no compaction call");
    let last = text_of(&bodies[2]);
    assert!(last.contains("The third request."), "{last}");
    assert!(
        last.contains("The first request."),
        "the head is kept: {last}"
    );
    assert!(
        last.contains("Earlier turns of this conversation were removed"),
        "{last}"
    );
    assert!(
        !last.contains(r#""text":"First.""#),
        "the answered turn is dropped: {last}"
    );
    assert_eq!(segment_starts(d.log()).len(), 1);
    check_log(d.log());
}

/// Answers a compaction call with a block and anything else with a short text, whatever the order: a restarted run
/// may send a compaction call again.
///
/// With `tool_loop`, the session is a client-summary one: two echo calls, the second reporting a large input, then
/// done; the summary call is answered with a summary, and the turn finishes after it.
#[derive(Debug, Clone, Default)]
struct Auto {
    bodies: Arc<Mutex<Vec<Vec<u8>>>>,
    tool_loop: bool,
}

impl Auto {
    fn compaction_calls(&self) -> usize {
        self.bodies
            .lock()
            .unwrap()
            .iter()
            .filter(|b| {
                let t = text_of(b);
                t.contains(r#""compaction":{"type":"summarize"}"#)
                    || t.contains(RawJson::string(SUMMARY_PROMPT).as_str())
            })
            .count()
    }

    fn tool_loop_reply(text: &str) -> Reply {
        if text.contains(RawJson::string(SUMMARY_PROMPT).as_str()) {
            said("SUMMARY-MARK: two echoes ran.", 700)
        } else if text.contains("SUMMARY-MARK") || text.contains("toolu_2") {
            said("Done.", 50)
        } else if text.contains("toolu_1") {
            echo_call(2, 700)
        } else {
            echo_call(1, 50)
        }
    }
}

impl ModelCaller for Auto {
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
        let reply = if self.tool_loop {
            Self::tool_loop_reply(&text)
        } else if text.contains(r#""compaction":{"type":"summarize"}"#) {
            compacted()
        } else if text.contains("Say hello first.") && !text.contains("Hello.") {
            said("Hello.", 700)
        } else {
            said("Fine.", 50)
        };
        let Reply::Body(b) = reply else {
            unreachable!()
        };
        Box::pin(async move { anthropic::answer(&raw(&b)) })
    }

    fn read(&self, body: &RawJson, content: Range<u32>) -> Result<Received, CallFailure> {
        let mut r = anthropic::answer(body)?;
        r.content = content;
        Ok(r)
    }
}

#[tokio::test]
async fn a_restart_after_any_append_of_a_compaction_finishes_it_once() {
    let messages = ["Say hello first.", "Now the next thing.", "And the last."];
    let cfg = || {
        config(ModelProfile::native())
            .with_limits(limits(10))
            .with_context(profile(Compactor::Server))
    };
    // A run without a stop counts the appends.
    let full = Shared::default();
    let (mut d, _) = Driver::new(
        cfg(),
        full.clone(),
        Auto::default(),
        TestTools::default(),
        Host(None),
    )
    .await
    .unwrap();
    for m in messages {
        d.run_turn(user(m), &mut |_| {}).await.unwrap();
    }
    let total = full.appends.load(std::sync::atomic::Ordering::SeqCst);
    assert!(total > 8, "{total}");

    let mut sent_again = 0;
    for n in 1..total {
        let model = Auto::default();
        let store = Shared::default().stopping_after(n);
        let (mut d, _) = Driver::new(
            cfg(),
            store.clone(),
            model.clone(),
            TestTools::default(),
            Host(None),
        )
        .await
        .unwrap();
        for m in messages {
            match d.run_turn(user(m), &mut |_| {}).await {
                Ok(end) if end.outcome == Outcome::Finished => {}
                _ => break,
            }
        }
        let session = d.session();
        drop(d);
        let (mut again, _) = Driver::resume(
            cfg(),
            store.reopened(),
            session,
            model.clone(),
            TestTools::default(),
            Host(None),
        )
        .await
        .unwrap();
        if again.in_turn() {
            again.continue_turn(&mut |_| {}).await.unwrap();
        }
        let end = again
            .run_turn(user("One more."), &mut |_| {})
            .await
            .unwrap_or_else(|e| panic!("n={n}: {e}"));
        assert_eq!(end.outcome, Outcome::Finished, "n={n}");
        let log = store.inner.load(session).await.unwrap().log;
        assert!(log.resume().at_boundary() && !log.resume().in_turn, "n={n}");
        assert!(segment_starts(&log).len() <= 1, "n={n}: {:#?}", kinds(&log));
        assert!(model.compaction_calls() <= 2, "n={n}");
        sent_again += usize::from(model.compaction_calls() == 2);
        check_log(&log);
    }
    assert!(
        sent_again > 0,
        "some stop fell between a compaction request and its answer"
    );
}

#[tokio::test]
async fn a_restart_after_any_append_of_a_summary_inside_a_turn_finishes_the_turn() {
    let cfg = || {
        config(ModelProfile::native())
            .with_limits(limits(10))
            .with_context(profile(Compactor::ClientSummary))
    };
    let auto = || Auto {
        tool_loop: true,
        ..Auto::default()
    };
    let full = Shared::default();
    let model = auto();
    let (mut d, _) = Driver::new(
        cfg(),
        full.clone(),
        model.clone(),
        TestTools::default(),
        Host(None),
    )
    .await
    .unwrap();
    let end = d.run_turn(user("Echo twice."), &mut |_| {}).await.unwrap();
    assert_eq!(end.reply.as_deref(), Some("Done."));
    assert_eq!(model.compaction_calls(), 1);
    assert_eq!(segment_starts(d.log()).len(), 1);
    let total = full.appends.load(std::sync::atomic::Ordering::SeqCst);

    let mut sent_again = 0;
    for n in 1..total {
        let model = auto();
        let store = Shared::default().stopping_after(n);
        let (mut d, _) = Driver::new(
            cfg(),
            store.clone(),
            model.clone(),
            TestTools::default(),
            Host(None),
        )
        .await
        .unwrap();
        let _ = d.run_turn(user("Echo twice."), &mut |_| {}).await;
        let session = d.session();
        drop(d);
        let (mut again, _) = Driver::resume(
            cfg(),
            store.reopened(),
            session,
            model.clone(),
            TestTools::default(),
            Host(None),
        )
        .await
        .unwrap();
        if again.in_turn() {
            let end = again.continue_turn(&mut |_| {}).await.unwrap().unwrap();
            assert_eq!(end.outcome, Outcome::Finished, "n={n}: {end:?}");
        }
        let end = again
            .run_turn(user("One more."), &mut |_| {})
            .await
            .unwrap_or_else(|e| panic!("n={n}: {e}"));
        assert_eq!(end.outcome, Outcome::Finished, "n={n}");
        let log = store.inner.load(session).await.unwrap().log;
        assert!(log.resume().at_boundary() && !log.resume().in_turn, "n={n}");
        assert!(segment_starts(&log).len() <= 1, "n={n}: {:#?}", kinds(&log));
        assert!(model.compaction_calls() <= 2, "n={n}");
        sent_again += usize::from(model.compaction_calls() == 2);
        check_log(&log);
    }
    assert!(
        sent_again > 0,
        "some stop fell between the summary request and its answer"
    );
}

/// A spill store the test can read after the driver wrote to it.
#[derive(Debug, Clone, Default)]
struct SharedSpill(Arc<MemorySpill>);

impl SpillStore for SharedSpill {
    async fn put(
        &self,
        session: SessionId,
        call: &str,
        bytes: Vec<u8>,
    ) -> Result<(), ContextError> {
        self.0.put(session, call, bytes).await
    }

    async fn read(
        &self,
        session: SessionId,
        call: &str,
        offset: u64,
        len: usize,
    ) -> Result<Vec<u8>, ContextError> {
        self.0.read(session, call, offset, len).await
    }

    async fn copy_session(&self, from: SessionId, to: SessionId) -> Result<(), ContextError> {
        self.0.copy_session(from, to).await
    }
}

#[tokio::test]
async fn a_large_tool_output_spills_and_the_model_sees_a_preview() {
    let big = "x".repeat(5000);
    let call = using(
        &format!(r#"{{"type":"tool_use","id":"toolu_1","name":"echo","input":{{"msg":"{big}"}}}}"#),
        "tool_use",
        50,
    );
    let caps = SpillLimits {
        spill_bytes: 1000,
        preview_bytes: 200,
        read_cap: 4096,
    };
    let spill = SharedSpill::default();
    let model = Scripted::new(vec![call, said("Seen.", 60)]);
    let store = Shared::default();
    let config = config(ModelProfile::native())
        .with_limits(limits(10))
        .with_spill(Spill::new(spill.clone(), caps));
    let (mut d, _) = Driver::new(
        config,
        store.clone(),
        model.clone(),
        TestTools::default(),
        Host(None),
    )
    .await
    .unwrap();
    let end = d.run_turn(user("Echo it."), &mut |_| {}).await.unwrap();
    assert_eq!(end.reply.as_deref(), Some("Seen."));
    let (block, record) = d
        .log()
        .entries()
        .iter()
        .find_map(|e| match &e.body {
            Body::ToolResult { block, spill, .. } => Some((block.clone(), spill.clone())),
            _ => None,
        })
        .unwrap();
    let record = SpillRecord::from_raw(&record.unwrap()).unwrap();
    assert_eq!(record.handle, "spill:toolu_1");
    assert!(record.bytes > 5000);
    assert!(
        block.as_str().contains("spill:toolu_1"),
        "{}",
        block.as_str()
    );
    assert!(block.as_str().len() < 1000, "{}", block.as_str());
    let second = text_of(&model.bodies()[1]);
    assert_eq!(
        second.matches(&big).count(),
        1,
        "the call's input only; the result is the preview"
    );
    let head = read_output(&spill, d.session(), &record.handle, 0, 100, &caps)
        .await
        .unwrap();
    assert!(head.starts_with("echoed: "), "{head}");
    check_log(d.log());
}

#[tokio::test]
async fn a_first_request_over_the_window_with_nothing_to_drop_fails_its_turn_unsent() {
    // The system prompt alone is past the window: no fallback can make it fit, so nothing is sent.
    let model = Scripted::new(vec![said("Never.", 10)]);
    let config = DriverConfig::new(ModelProfile::native(), defs(), "x".repeat(8000))
        .with_limits(limits(10))
        .with_context(profile(Compactor::Server));
    let (mut d, _) = Driver::new(
        config,
        Shared::default(),
        model.clone(),
        TestTools::default(),
        Host(None),
    )
    .await
    .unwrap();
    let mut events = Vec::new();
    let end = d
        .run_turn(user("Hello."), &mut sink(&mut events))
        .await
        .unwrap();
    assert_eq!(end.outcome, Outcome::Failed, "{events:#?}");
    assert!(model.bodies().is_empty(), "no request is sent");
    assert!(events.iter().any(|e| matches!(
        e,
        Event::Error {
            kind: ErrorKind::Guard,
            ..
        }
    )));
    assert!(!d.in_turn());
    // The next turn is refused before it starts, without a call either.
    let refused = d.run_turn(user("Hello again."), &mut |_| {}).await;
    assert!(
        matches!(refused, Err(DriverError::DoesNotFit)),
        "{refused:?}"
    );
    assert!(model.bodies().is_empty());
    check_log(d.log());
}

#[tokio::test]
async fn a_turn_too_large_for_the_window_fails_as_a_guard_without_a_model_call() {
    let mut p = profile(Compactor::Server);
    p.compact_at_percent = 255;
    let replies = vec![said("First.", 50), said("Third.", 50)];
    let (mut d, model, _) = driver_with(p, replies, TestTools::default()).await;
    d.run_turn(user("The first request."), &mut |_| {})
        .await
        .unwrap();
    let mut events = Vec::new();
    let huge = d
        .run_turn(user(&"y".repeat(8000)), &mut sink(&mut events))
        .await
        .unwrap();
    assert_eq!(huge.outcome, Outcome::Failed, "{events:#?}");
    assert_eq!(model.bodies().len(), 1, "the doomed request is not sent");
    assert!(
        events.iter().any(|e| matches!(
            e,
            Event::Error {
                kind: ErrorKind::Guard,
                ..
            }
        )),
        "{events:#?}"
    );
    // Between turns the fallback may drop that turn: the session goes on.
    let third = d
        .run_turn(user("The third request."), &mut |_| {})
        .await
        .unwrap();
    assert_eq!(third.reply.as_deref(), Some("Third."));
    assert_eq!(model.bodies().len(), 2);
    check_log(d.log());
}

#[tokio::test(start_paused = true)]
async fn an_interrupt_during_the_compaction_before_a_turn_ends_it_before_it_starts() {
    let replies = vec![
        said("Hello.", 700),
        Reply::Slow(Duration::from_secs(10), body_of(compacted())),
        said("Later.", 50),
    ];
    let (mut d, model, _) =
        driver_with(profile(Compactor::Server), replies, TestTools::default()).await;
    d.run_turn(user("Say hello first."), &mut |_| {})
        .await
        .unwrap();
    // An interrupt sent before the call is aimed at the last turn: dropped.
    let handle = d.handle();
    assert!(handle.send(Op::Interrupt));
    let interrupt = async {
        tokio::time::sleep(Duration::from_secs(1)).await;
        assert!(handle.send(Op::Interrupt));
    };
    let mut quiet = |_| {};
    let (refused, ()) = tokio::join!(d.run_turn(user("Now this."), &mut quiet), interrupt);
    assert!(
        matches!(refused, Err(DriverError::Interrupted)),
        "{refused:?}"
    );
    assert_eq!(model.bodies().len(), 2, "the compaction call finished");
    assert_eq!(segment_starts(d.log()).len(), 1, "its segment is kept");
    assert!(!d.in_turn());
    let later = d
        .run_turn(user("Now this, again."), &mut |_| {})
        .await
        .unwrap();
    assert_eq!(later.reply.as_deref(), Some("Later."));
    check_log(d.log());
}

#[tokio::test(start_paused = true)]
async fn a_turn_dropped_during_the_compaction_before_it_counts_as_open_and_goes_on() {
    let replies = vec![
        said("Hello.", 700),
        Reply::Slow(Duration::from_secs(10), body_of(compacted())),
        compacted(),
        said("Second.", 50),
    ];
    let (mut d, model, store) =
        driver_with(profile(Compactor::Server), replies, TestTools::default()).await;
    d.run_turn(user("Say hello first."), &mut |_| {})
        .await
        .unwrap();
    let mut quiet = |_| {};
    let dropped = tokio::time::timeout(
        Duration::from_secs(1),
        d.run_turn(user("Now the next thing."), &mut quiet),
    )
    .await;
    assert!(dropped.is_err(), "the compaction call was still out");
    assert!(d.in_turn(), "the store is ahead of the core: read it again");
    let end = d
        .run_turn(user("Now the next thing."), &mut |_| {})
        .await
        .unwrap();
    assert_eq!(end.reply.as_deref(), Some("Second."));
    assert_eq!(
        model.bodies().len(),
        4,
        "the open compaction call was sent again"
    );
    check_log(d.log());
    let stored = store.inner.load(d.session()).await.unwrap().log;
    assert_eq!(&stored, d.log());
}

#[tokio::test(start_paused = true)]
async fn a_compaction_left_open_is_finished_by_a_resume_without_the_profile() {
    let replies = vec![
        said("Hello.", 700),
        Reply::Slow(Duration::from_secs(10), body_of(compacted())),
        compacted(),
        said("Second.", 50),
    ];
    let (mut d, model, store) =
        driver_with(profile(Compactor::Server), replies, TestTools::default()).await;
    d.run_turn(user("Say hello first."), &mut |_| {})
        .await
        .unwrap();
    let mut quiet = |_| {};
    let dropped = tokio::time::timeout(
        Duration::from_secs(1),
        d.run_turn(user("Now the next thing."), &mut quiet),
    )
    .await;
    assert!(dropped.is_err());
    let session = d.session();
    drop(d);
    let plain = config(ModelProfile::native()).with_limits(limits(10));
    let (mut again, _) = Driver::resume(
        plain,
        store.reopened(),
        session,
        model.clone(),
        TestTools::default(),
        Host(None),
    )
    .await
    .unwrap();
    let end = again
        .run_turn(user("Now the next thing."), &mut |_| {})
        .await
        .unwrap();
    assert_eq!(end.reply.as_deref(), Some("Second."));
    let bodies = model.bodies();
    assert_eq!(bodies.len(), 4);
    let next = text_of(&bodies[3]);
    assert!(
        next.contains(BLOCK),
        "the turn follows the finished compaction: {next}"
    );
    assert_eq!(segment_starts(again.log()).len(), 1);
    check_log(again.log());
}

fn body_of(reply: Reply) -> String {
    match reply {
        Reply::Body(b) => b,
        other => panic!("not a body: {other:?}"),
    }
}

#[tokio::test(start_paused = true)]
async fn a_steer_sent_before_a_turn_dropped_during_its_compaction_reaches_the_next_turn() {
    let replies = vec![
        said("Hello.", 700),
        Reply::Slow(Duration::from_secs(10), body_of(compacted())),
        compacted(),
        said("Second.", 50),
        said("Red it is.", 50),
    ];
    let (mut d, model, _) =
        driver_with(profile(Compactor::Server), replies, TestTools::default()).await;
    d.run_turn(user("Say hello first."), &mut |_| {})
        .await
        .unwrap();
    let handle = d.handle();
    assert!(handle.send(Op::Steer {
        content: RawJson::string("use red"),
    }));
    let mut quiet = |_| {};
    let dropped = tokio::time::timeout(
        Duration::from_secs(1),
        d.run_turn(user("Now the next thing."), &mut quiet),
    )
    .await;
    assert!(dropped.is_err());
    let end = d
        .run_turn(user("Now the next thing."), &mut |_| {})
        .await
        .unwrap();
    // The steer was held through the dropped turn; the model's answer is not the turn's end while a held message
    // may still go out, so the steer is sent and the turn goes on to the next answer.
    let sent = model
        .bodies()
        .iter()
        .any(|b| text_of(b).contains("use red"));
    assert!(sent, "the steer reaches the next turn: {end:?}");
    assert!(end.unsent.is_empty(), "nothing comes back unsent: {end:?}");
    assert_eq!(end.outcome, Outcome::Finished);
    assert_eq!(end.reply.as_deref(), Some("Red it is."));
    check_log(d.log());
}
