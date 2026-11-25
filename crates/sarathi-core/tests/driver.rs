// SPDX-License-Identifier: Apache-2.0
//! The driver end to end on a scripted model: the loop tests of the agent it replaces, on both step protocols; the
//! native round trip (the answer re-sent byte for byte); permission asks, steers and interrupts through ops; and
//! resume after the store stopped at every append of a turn.

#![expect(clippy::unwrap_used, clippy::panic, reason = "test code")]

pub mod common;

use std::sync::atomic::Ordering;
use std::time::Duration;

use common::*;
use sarathi_core::driver::Driver;
use sarathi_core::{CUT_SHORT, ModelProfile, Verdict};
use sarathi_log::{Body, LogStore};
use sarathi_protocol::{Decision, ErrorKind, Event, Op, Outcome, RawJson};

/// One step of a scripted model under `profile`: call `tool` with `input`.
fn call(profile: &ModelProfile, n: u32, tool: &str, input: &str) -> Reply {
    if profile.protocol == sarathi_core::StepProtocol::Native {
        native(None, &[(&format!("toolu_{n}"), tool, input)], "tool_use")
    } else {
        text(&format!(
            r#"{{"thought":"step {n}","tool":"{tool}","input":{input},"reply":""}}"#
        ))
    }
}

/// The closing step under `profile`.
fn done(profile: &ModelProfile, reply: &str) -> Reply {
    if profile.protocol == sarathi_core::StepProtocol::Native {
        native(Some(reply), &[], "end_turn")
    } else {
        text(&format!(
            r#"{{"thought":"finished","tool":"done","input":{{}},"reply":{}}}"#,
            RawJson::string(reply).as_str()
        ))
    }
}

fn profiles() -> [ModelProfile; 2] {
    [ModelProfile::json_object(true), ModelProfile::native()]
}

async fn driver(
    profile: ModelProfile,
    max_steps: u32,
    replies: Vec<Reply>,
    tools: TestTools,
) -> (TestDriver, Scripted, Shared) {
    let model = Scripted::new(replies);
    let store = Shared::default();
    let config = config(profile).with_limits(limits(max_steps));
    let (d, _) = Driver::new(config, store.clone(), model.clone(), tools, Host(None))
        .await
        .unwrap();
    (d, model, store)
}

#[tokio::test]
async fn acts_then_finishes_and_records_the_steps() {
    for p in profiles() {
        let tools = TestTools::default();
        let replies = vec![call(&p, 1, "echo", r#"{"msg":"hi"}"#), done(&p, "All set.")];
        let (mut d, model, store) = driver(p.clone(), 5, replies, tools.clone()).await;
        let mut events = Vec::new();
        let end = d
            .run_turn(user("Say hello."), &mut sink(&mut events))
            .await
            .unwrap();
        assert_eq!(end.outcome, Outcome::Finished, "{p:?}");
        assert_eq!(end.reply.as_deref(), Some("All set."));
        let r = results(d.log());
        assert_eq!(r.len(), 1);
        assert!(r[0].1.contains("echoed"), "{r:?}");
        assert_eq!(
            tools.tally.ran.lock().unwrap().len(),
            1,
            "the tool really ran"
        );
        assert_eq!(model.left(), 0);
        check_log(d.log());
        let stored = store.inner.load(d.session()).await.unwrap().log;
        assert_eq!(&stored, d.log(), "the driver's copy is the store's log");
        assert!(matches!(
            events.last(),
            Some(Event::Done {
                outcome: Outcome::Finished,
                ..
            })
        ));
    }
}

#[tokio::test]
async fn an_unknown_tool_becomes_a_correction_the_model_reads() {
    for p in profiles() {
        let tools = TestTools::default();
        let replies = vec![call(&p, 1, "paint", "{}"), done(&p, "Fine.")];
        let (mut d, model, _) = driver(p.clone(), 5, replies, tools.clone()).await;
        let end = d.run_turn(user("Paint it."), &mut |_| {}).await.unwrap();
        let r = results(d.log());
        assert_eq!(r.len(), 1);
        assert!(r[0].1.contains("not a tool"), "{r:?}");
        assert!(r[0].2);
        assert!(tools.tally.ran.lock().unwrap().is_empty(), "nothing ran");
        assert_eq!(end.reply.as_deref(), Some("Fine."));
        let second = String::from_utf8(model.bodies()[1].clone()).unwrap();
        assert!(
            second.contains("not a tool"),
            "the model reads the correction: {p:?}"
        );
        check_log(d.log());
    }
}

#[tokio::test]
async fn the_step_cap_ends_the_run_honestly() {
    for p in profiles() {
        let replies = (1..=5).map(|n| call(&p, n, "echo", "{}")).collect();
        let (mut d, _, _) = driver(p.clone(), 3, replies, TestTools::default()).await;
        let end = d.run_turn(user("Loop."), &mut |_| {}).await.unwrap();
        assert_eq!(end.outcome, Outcome::RanOut, "{p:?}");
        assert!(end.reply.as_deref().unwrap().contains("ran out of steps"));
        let answers = d
            .log()
            .entries()
            .iter()
            .filter(|e| matches!(e.body, Body::AssistantMessage { .. }))
            .count();
        assert_eq!(answers, 3);
        check_log(d.log());
    }
}

#[tokio::test]
async fn a_readable_tool_refusal_is_an_observation_not_a_crash() {
    for p in profiles() {
        let tools = TestTools {
            edit_refuses: true,
            ..TestTools::default()
        };
        let replies = vec![call(&p, 1, "edit", "{}"), done(&p, "Adjusted.")];
        let (mut d, _, _) = driver(p.clone(), 5, replies, tools).await;
        let end = d.run_turn(user("Edit."), &mut |_| {}).await.unwrap();
        assert_eq!(end.reply.as_deref(), Some("Adjusted."), "{p:?}");
        let r = results(d.log());
        assert!(r[0].1.contains("already exists") && r[0].2, "{r:?}");
        check_log(d.log());
    }
}

#[tokio::test]
async fn invalid_json_retries_then_succeeds() {
    let p = ModelProfile::json_object(true);
    let replies = vec![text("not json at all"), done(&p, "Recovered.")];
    let (mut d, model, _) = driver(p, 5, replies, TestTools::default()).await;
    let end = d.run_turn(user("Go."), &mut |_| {}).await.unwrap();
    assert_eq!(end.outcome, Outcome::Finished);
    assert_eq!(end.reply.as_deref(), Some("Recovered."));
    let second = String::from_utf8(model.bodies()[1].clone()).unwrap();
    assert!(second.contains("unusable"), "{second}");
    check_log(d.log());
}

#[tokio::test]
async fn schema_mismatch_fails_after_three_in_a_row() {
    let p = ModelProfile::json_object(true);
    let replies = vec![text("{}"), text("[]"), text("nope"), done(&p, "never")];
    let (mut d, model, _) = driver(p, 10, replies, TestTools::default()).await;
    let mut events = Vec::new();
    let end = d
        .run_turn(user("Go."), &mut sink(&mut events))
        .await
        .unwrap();
    assert_eq!(end.outcome, Outcome::Failed);
    assert_eq!(model.left(), 1, "the fourth answer is never asked for");
    assert!(events.iter().any(|e| matches!(
        e,
        Event::Error {
            kind: ErrorKind::Guard,
            ..
        }
    )));
    check_log(d.log());
}

#[tokio::test]
async fn transport_error_bubbles_immediately() {
    let p = ModelProfile::native();
    let replies = vec![Reply::Fail(ErrorKind::Upstream), done(&p, "never")];
    let (mut d, model, _) = driver(p, 5, replies, TestTools::default()).await;
    let end = d.run_turn(user("Go."), &mut |_| {}).await.unwrap();
    assert_eq!(end.outcome, Outcome::Failed);
    assert_eq!(model.bodies().len(), 1);
    let failure = d
        .log()
        .entries()
        .iter()
        .find_map(|e| e.body.error_kind())
        .unwrap();
    assert_eq!(failure, ErrorKind::Upstream);
    check_log(d.log());
}

/// The answer goes back exactly as received (thinking and its signature, escapes, block order), the session prefix
/// and the first message are the same bytes on turn 2, and the step's tool results are one user message.
#[tokio::test]
async fn the_native_round_trip_resends_the_answer_byte_for_byte() {
    let p = ModelProfile::native();
    let content = concat!(
        r#"{"type":"thinking","thinking":"Look first, then édit.\n","signature":"c2lnbmVk/+=="},"#,
        r#"{"type":"text","text":"Checking \"both\"  now."},"#,
        r#"{"type":"tool_use","id":"toolu_a","name":"look","input":{"z":1,"a":"\/"}},"#,
        r#"{"type":"tool_use","id":"toolu_b","name":"edit","input":{"title":"Hi 😀"}}"#
    );
    let replies = vec![Reply::Body(message(content, "tool_use")), done(&p, "Done.")];
    let (mut d, model, _) = driver(p, 5, replies, TestTools::default()).await;
    let end = d.run_turn(user("Go."), &mut |_| {}).await.unwrap();
    assert_eq!(end.outcome, Outcome::Finished);
    let bodies = model.bodies();
    let (b1, b2) = (&bodies[0], &bodies[1]);
    let start = find(b1, br#""messages":["#).unwrap() + br#""messages":["#.len();
    let first = br#"{"role":"user","content":"Go."}"#;
    assert_eq!(&b1[start..start + first.len()], first);
    let end1 = start + first.len();
    assert_eq!(
        &b2[..end1],
        &b1[..end1],
        "the prefix and turn 1 are the same bytes"
    );
    let rest = String::from_utf8(b2[end1..].to_vec()).unwrap();
    let assistant = format!(r#",{{"role":"assistant","content":[{content}]}}"#);
    assert!(rest.starts_with(&assistant), "{rest}");
    let results = concat!(
        r#",{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_a","content":"seen"},"#,
        r#"{"type":"tool_result","tool_use_id":"toolu_b","content":"edited"}]}]"#
    );
    assert!(rest[assistant.len()..].starts_with(results), "{rest}");
    check_log(d.log());
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

#[tokio::test]
async fn an_ask_is_answered_by_an_op_from_the_host() {
    let p = ModelProfile::native();
    let mut tools = TestTools::default();
    tools.verdicts.insert("edit".into(), Verdict::Ask);
    let replies = vec![call(&p, 1, "edit", "{}"), done(&p, "Edited.")];
    let (mut d, _, _) = driver(p, 5, replies, tools.clone()).await;
    let handle = d.handle();
    let mut asked = 0;
    let end = d
        .run_turn(user("Edit."), &mut |e| {
            if let Event::PermissionAsked { call, .. } = e {
                asked += 1;
                handle.send(Op::ToolDecision {
                    call,
                    decision: Decision::AllowOnce,
                });
            }
        })
        .await
        .unwrap();
    assert_eq!(asked, 1);
    assert_eq!(end.outcome, Outcome::Finished);
    assert_eq!(tools.tally.writes.load(Ordering::SeqCst), 1);
    let k = kinds(d.log());
    let ask = k.iter().position(|k| k == "tool.ask").unwrap();
    assert_eq!(k[ask + 1], "tool.decision");
    check_log(d.log());
}

#[tokio::test(start_paused = true)]
async fn a_steer_during_a_model_call_joins_the_next_request_after_the_results() {
    let p = ModelProfile::native();
    let Reply::Body(first) = call(&p, 1, "look", "{}") else {
        unreachable!()
    };
    let replies = vec![Reply::Slow(Duration::from_secs(5), first), done(&p, "Red.")];
    let (mut d, model, _) = driver(p, 5, replies, TestTools::default()).await;
    let handle = d.handle();
    let end = d
        .run_turn(user("Make it nice."), &mut |e| {
            if let Event::TurnStarted { .. } = e {
                handle.send(Op::Steer {
                    content: RawJson::string("use red"),
                });
            }
        })
        .await
        .unwrap();
    assert_eq!(end.outcome, Outcome::Finished);
    let second = String::from_utf8(model.bodies()[1].clone()).unwrap();
    assert!(
        second.contains(
            r#"{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"seen"},{"type":"text","text":"use red"}]}"#
        ),
        "the steer is in the results' user message: {second}"
    );
    check_log(d.log());
}

#[tokio::test(start_paused = true)]
async fn an_interrupt_lets_a_write_stop_within_its_grace() {
    let p = ModelProfile::native();
    let tools = TestTools {
        edit_takes: Some(Duration::from_secs(300)),
        edit_stops: true,
        ..TestTools::default()
    };
    let replies = vec![call(&p, 1, "edit", "{}"), done(&p, "never")];
    let (mut d, model, _) = driver(p, 5, replies, tools.clone()).await;
    let handle = d.handle();
    let end = d
        .run_turn(user("Edit."), &mut |e| {
            if let Event::ToolStarted { .. } = e {
                handle.send(Op::Interrupt);
            }
        })
        .await
        .unwrap();
    assert_eq!(end.outcome, Outcome::Interrupted);
    assert_eq!(model.left(), 1, "no request after the interrupt");
    assert_eq!(tools.tally.writes.load(Ordering::SeqCst), 1);
    assert_eq!(results(d.log()).len(), 1);
    check_log(d.log());
}

#[tokio::test(start_paused = true)]
async fn a_write_that_ignores_the_stop_is_abandoned_after_its_grace() {
    let p = ModelProfile::native();
    let tools = TestTools {
        edit_takes: Some(Duration::from_secs(300)),
        edit_stops: false,
        ..TestTools::default()
    };
    let replies = vec![call(&p, 1, "edit", "{}")];
    let (mut d, _, _) = driver(p, 5, replies, tools).await;
    let handle = d.handle();
    let started = tokio::time::Instant::now();
    let end = d
        .run_turn(user("Edit."), &mut |e| {
            if let Event::ToolStarted { .. } = e {
                handle.send(Op::Interrupt);
            }
        })
        .await
        .unwrap();
    assert_eq!(end.outcome, Outcome::Interrupted);
    let waited = started.elapsed();
    assert!(
        waited >= Duration::from_secs(30) && waited < Duration::from_secs(300),
        "{waited:?}"
    );
    let r = results(d.log());
    assert!(r[0].1.contains("effect is unknown") && r[0].2, "{r:?}");
    check_log(d.log());
}

/// The scenario every resume test runs: a step that reads and writes, then a final answer.
fn scenario(p: &ModelProfile) -> Vec<Reply> {
    if p.protocol == sarathi_core::StepProtocol::Native {
        vec![
            native(
                Some("Both."),
                &[("toolu_1", "look", "{}"), ("toolu_2", "edit", r#"{"t":1}"#)],
                "tool_use",
            ),
            done(p, "Done."),
            done(p, "Done again."),
        ]
    } else {
        vec![
            call(p, 1, "look", "{}"),
            call(p, 2, "edit", r#"{"t":1}"#),
            done(p, "Done."),
            done(p, "Done again."),
        ]
    }
}

/// For every append of the scenario's turn: the store stops there, a new driver resumes the session from what was
/// written and finishes the turn. The write ran at most once, every tool use has one result, every request rebuilds
/// to its hash, and the turn ends.
#[tokio::test]
async fn a_turn_stopped_at_any_append_resumes_and_finishes() {
    for p in profiles() {
        // The appends a whole run makes, counting the prefix.
        let base = Shared::default();
        let (mut whole, _) = Driver::new(
            config(p.clone()),
            base.clone(),
            Scripted::new(scenario(&p)),
            TestTools::default(),
            Host(None),
        )
        .await
        .unwrap();
        whole.run_turn(user("Go."), &mut |_| {}).await.unwrap();
        let total = base.appends.load(Ordering::SeqCst);
        assert!(total > 5, "{total}");

        for n in 2..total {
            let tools = TestTools::default();
            let model = Scripted::new(scenario(&p));
            let store = Shared::default().stopping_after(n);
            let (mut d, _) = Driver::new(
                config(p.clone()),
                store.clone(),
                model.clone(),
                tools.clone(),
                Host(None),
            )
            .await
            .unwrap();
            let mut events = Vec::new();
            let first = d.run_turn(user("Go."), &mut sink(&mut events)).await;
            let first = first.unwrap_or_else(|e| panic!("{p:?} n={n}: {e}: {events:#?}"));
            assert_eq!(first.outcome, Outcome::Failed, "{p:?} n={n}");
            let session = d.session();
            drop(d);

            let (mut again, _) = Driver::resume(
                config(p.clone()),
                store.reopened(),
                session,
                model.clone(),
                tools.clone(),
                Host(None),
            )
            .await
            .unwrap();
            if again.in_turn() {
                let end = again.continue_turn(&mut |_| {}).await.unwrap().unwrap();
                assert!(
                    matches!(end.outcome, Outcome::Finished),
                    "{p:?} n={n}: {end:?}"
                );
            }
            let log = store.inner.load(session).await.unwrap().log;
            assert!(
                log.resume().at_boundary() && !log.resume().in_turn,
                "{p:?} n={n}"
            );
            check_log(&log);
            assert!(
                tools.tally.writes.load(Ordering::SeqCst) <= 1,
                "{p:?} n={n}"
            );
            // A write whose start reached the log and whose result did not is answered as cut short.
            let cut = results(&log).iter().any(|r| r.1.contains(CUT_SHORT));
            if cut {
                assert_eq!(tools.tally.writes.load(Ordering::SeqCst), 1, "{p:?} n={n}");
            }
            // The resumed driver goes on with the next turn.
            let next = again.run_turn(user("Again."), &mut |_| {}).await;
            assert!(next.is_ok(), "{p:?} n={n}: {next:?}");
        }
    }
}

#[tokio::test(start_paused = true)]
async fn a_dropped_turn_during_a_model_call_is_a_restart() {
    let p = ModelProfile::native();
    let Reply::Body(first) = call(&p, 1, "look", "{}") else {
        unreachable!()
    };
    let replies = vec![
        Reply::Slow(Duration::from_secs(50), first),
        done(&p, "Done."),
        done(&p, "Two."),
    ];
    let (mut d, _, store) = driver(p, 5, replies, TestTools::default()).await;
    let dropped =
        tokio::time::timeout(Duration::from_secs(1), d.run_turn(user("Go."), &mut |_| {})).await;
    assert!(dropped.is_err());
    assert!(d.in_turn(), "a dropped turn is open");
    let refused = d.run_turn(user("Two?"), &mut |_| {}).await;
    assert!(matches!(
        refused,
        Err(sarathi_core::driver::DriverError::TurnOpen)
    ));
    let end = d.continue_turn(&mut |_| {}).await.unwrap().unwrap();
    assert_eq!(end.outcome, Outcome::Finished);
    assert_eq!(end.reply.as_deref(), Some("Done."));
    let two = d.run_turn(user("Two?"), &mut |_| {}).await.unwrap();
    assert_eq!(two.reply.as_deref(), Some("Two."));
    check_log(&store.inner.load(d.session()).await.unwrap().log);
}

#[tokio::test(start_paused = true)]
async fn a_dropped_turn_during_a_write_answers_it_as_cut_short() {
    let p = ModelProfile::native();
    let tools = TestTools {
        edit_takes: Some(Duration::from_secs(50)),
        ..TestTools::default()
    };
    let replies = vec![call(&p, 1, "edit", "{}"), done(&p, "Checked.")];
    let (mut d, _, store) = driver(p, 5, replies, tools.clone()).await;
    let dropped = tokio::time::timeout(
        Duration::from_secs(1),
        d.run_turn(user("Edit."), &mut |_| {}),
    )
    .await;
    assert!(dropped.is_err());
    let end = d.continue_turn(&mut |_| {}).await.unwrap().unwrap();
    assert_eq!(end.outcome, Outcome::Finished);
    assert_eq!(tools.tally.writes.load(Ordering::SeqCst), 1);
    let log = store.inner.load(d.session()).await.unwrap().log;
    let r = results(&log);
    assert_eq!(r.len(), 1);
    assert!(r[0].1.contains(CUT_SHORT) && r[0].2, "{r:?}");
    check_log(&log);
}

#[tokio::test(start_paused = true)]
async fn an_interrupt_sent_after_a_turn_ended_does_not_cancel_the_next() {
    let p = ModelProfile::native();
    let Reply::Body(two) = done(&p, "Two.") else {
        unreachable!()
    };
    let replies = vec![done(&p, "One."), Reply::Slow(Duration::from_secs(5), two)];
    let (mut d, _, _) = driver(p, 5, replies, TestTools::default()).await;
    let handle = d.handle();
    d.run_turn(user("One?"), &mut |_| {}).await.unwrap();
    handle.send(Op::Interrupt);
    let end = d.run_turn(user("Two?"), &mut |_| {}).await.unwrap();
    assert_eq!(end.outcome, Outcome::Finished);
    assert_eq!(end.reply.as_deref(), Some("Two."));
}

/// One append fails and the store then works again, on the same driver: the turn the host was told failed is
/// closed as failed in the log, and the next turn starts clean.
#[tokio::test]
async fn a_failed_write_closes_the_turn_and_the_next_one_starts_clean() {
    for p in profiles() {
        let base = Shared::default();
        let (mut whole, _) = Driver::new(
            config(p.clone()),
            base.clone(),
            Scripted::new(scenario(&p)),
            TestTools::default(),
            Host(None),
        )
        .await
        .unwrap();
        whole.run_turn(user("Go."), &mut |_| {}).await.unwrap();
        let total = base.appends.load(Ordering::SeqCst);
        for n in 1..total {
            let tools = TestTools::default();
            let mut replies = scenario(&p);
            replies.push(done(&p, "Next."));
            let store = Shared {
                fail_once: Some(n),
                ..Shared::default()
            };
            let (mut d, _) = Driver::new(
                config(p.clone()),
                store.clone(),
                Scripted::new(replies),
                tools.clone(),
                Host(None),
            )
            .await
            .unwrap();
            let first = d.run_turn(user("Go."), &mut |_| {}).await.unwrap();
            assert_eq!(first.outcome, Outcome::Failed, "{p:?} n={n}");
            assert!(!d.in_turn(), "{p:?} n={n}");
            let log = store.inner.load(d.session()).await.unwrap().log;
            assert!(!log.resume().in_turn, "{p:?} n={n}: closed in the log");
            if log.entries().len() > 1 {
                assert_eq!(
                    kinds(&log).last().map(String::as_str),
                    Some("turn.end"),
                    "{p:?} n={n}"
                );
            }
            check_log(&log);
            let next = d.run_turn(user("Again."), &mut |_| {}).await.unwrap();
            assert!(
                matches!(next.outcome, Outcome::Finished),
                "{p:?} n={n}: {next:?}"
            );
            let log = store.inner.load(d.session()).await.unwrap().log;
            check_log(&log);
            assert!(
                tools.tally.writes.load(Ordering::SeqCst) <= 2,
                "{p:?} n={n}"
            );
        }
    }
}

#[tokio::test]
async fn an_ask_no_one_can_answer_interrupts_the_turn() {
    let p = ModelProfile::native();
    let mut tools = TestTools::default();
    tools.verdicts.insert("edit".into(), Verdict::Ask);
    let replies = vec![call(&p, 1, "edit", "{}"), done(&p, "never")];
    // The handle `Driver::new` returns is dropped: no decision can come.
    let (mut d, _, _) = driver(p, 5, replies, tools.clone()).await;
    let end = tokio::time::timeout(
        Duration::from_secs(10),
        d.run_turn(user("Edit."), &mut |_| {}),
    )
    .await
    .expect("the turn does not hang")
    .unwrap();
    assert_eq!(end.outcome, Outcome::Interrupted);
    assert_eq!(tools.tally.writes.load(Ordering::SeqCst), 0);
    check_log(d.log());
}

#[tokio::test(start_paused = true)]
async fn an_interrupt_during_a_model_call_ends_the_turn_and_the_next_one_runs() {
    let p = ModelProfile::native();
    let Reply::Body(slow) = done(&p, "never") else {
        unreachable!()
    };
    let replies = vec![Reply::Slow(Duration::from_secs(60), slow), done(&p, "Two.")];
    let (mut d, _, _) = driver(p, 5, replies, TestTools::default()).await;
    let handle = d.handle();
    let end = d
        .run_turn(user("Go."), &mut |e| {
            if let Event::TurnStarted { .. } = e {
                handle.send(Op::Interrupt);
            }
        })
        .await
        .unwrap();
    assert_eq!(end.outcome, Outcome::Interrupted);
    let failure = d.log().entries().iter().find_map(|e| e.body.error_kind());
    assert_eq!(failure, Some(ErrorKind::Cancelled));
    let two = d.run_turn(user("Two?"), &mut |_| {}).await.unwrap();
    assert_eq!(two.reply.as_deref(), Some("Two."));
    check_log(d.log());
}

#[tokio::test(start_paused = true)]
async fn a_second_interrupt_during_the_grace_drops_the_write_at_once() {
    let p = ModelProfile::native();
    let tools = TestTools {
        edit_takes: Some(Duration::from_secs(300)),
        ..TestTools::default()
    };
    let replies = vec![native(
        None,
        &[("toolu_1", "edit", "{}"), ("toolu_2", "look", "{}")],
        "tool_use",
    )];
    let (mut d, _, _) = driver(p, 5, replies, tools).await;
    let handle = d.handle();
    let started = tokio::time::Instant::now();
    let end = d
        .run_turn(user("Edit."), &mut |e| {
            if let Event::ToolStarted { .. } = e {
                handle.send(Op::Interrupt);
                handle.send(Op::Interrupt);
            }
        })
        .await
        .unwrap();
    assert_eq!(end.outcome, Outcome::Interrupted);
    assert!(
        started.elapsed() < Duration::from_secs(30),
        "{:?}",
        started.elapsed()
    );
    let r = results(d.log());
    assert!(r[0].1.contains("effect is unknown"), "{r:?}");
    assert!(r[1].1.contains("skipped: interrupted"), "{r:?}");
    check_log(d.log());
}

#[tokio::test]
async fn allow_always_holds_after_a_resume() {
    let p = ModelProfile::native();
    let mut tools = TestTools::default();
    tools.verdicts.insert("edit".into(), Verdict::Ask);
    let model = Scripted::new(vec![
        call(&p, 1, "edit", "{}"),
        done(&p, "One."),
        call(&p, 2, "edit", r#"{"again":true}"#),
        done(&p, "Two."),
    ]);
    let store = Shared::default();
    let (mut d, _) = Driver::new(
        config(p.clone()),
        store.clone(),
        model.clone(),
        tools.clone(),
        Host(Some(Decision::AllowAlways)),
    )
    .await
    .unwrap();
    d.run_turn(user("One."), &mut |_| {}).await.unwrap();
    let session = d.session();
    drop(d);
    // No one answers now: a second ask would interrupt the turn.
    let (mut again, _) =
        Driver::resume(config(p), store, session, model, tools.clone(), Host(None))
            .await
            .unwrap();
    let mut events = Vec::new();
    let end = again
        .run_turn(user("Two."), &mut sink(&mut events))
        .await
        .unwrap();
    assert_eq!(end.reply.as_deref(), Some("Two."));
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, Event::PermissionAsked { .. }))
    );
    assert_eq!(tools.tally.writes.load(Ordering::SeqCst), 2);
}

fn slow_look(p: &ModelProfile) -> Scripted {
    let Reply::Body(first) = call(p, 1, "look", "{}") else {
        unreachable!()
    };
    Scripted::new(vec![
        Reply::Slow(Duration::from_secs(5), first),
        done(p, "Red."),
        done(p, "Red."),
    ])
}

#[tokio::test(start_paused = true)]
async fn a_resume_with_a_steer_after_the_last_answer_makes_the_next_call() {
    let p = ModelProfile::native();
    let model = slow_look(&p);
    let base = Shared::default();
    // The steer arrives during the first (slow) call, so it is written after the look's result.
    let (mut whole, _) = Driver::new(
        config(p.clone()),
        base.clone(),
        model.clone(),
        TestTools::default(),
        Host(None),
    )
    .await
    .unwrap();
    let handle = whole.handle();
    whole
        .run_turn(user("Go."), &mut |e| {
            if let Event::TurnStarted { .. } = e {
                handle.send(Op::Steer {
                    content: RawJson::string("use red"),
                });
            }
        })
        .await
        .unwrap();
    // The steer is written right after the result; stop the store at the request that follows it.
    let log = base.inner.load(whole.session()).await.unwrap().log;
    let k = kinds(&log);
    let steer = log
        .entries()
        .iter()
        .position(|e| matches!(e.body, Body::UserMessage { steer: true, .. }))
        .unwrap();
    assert_eq!(k[steer + 1], "model.request");
    let mut stop_at = 0;
    for n in 2..20 {
        let store = Shared::default().stopping_after(n);
        let (mut d, _) = Driver::new(
            config(p.clone()),
            store.clone(),
            slow_look(&p),
            TestTools::default(),
            Host(None),
        )
        .await
        .unwrap();
        let h = d.handle();
        let _ = d
            .run_turn(user("Go."), &mut |e| {
                if let Event::TurnStarted { .. } = e {
                    h.send(Op::Steer {
                        content: RawJson::string("use red"),
                    });
                }
            })
            .await;
        let partial = store.inner.load(d.session()).await.unwrap().log;
        if kinds(&partial).last().map(String::as_str) == Some("message.user")
            && partial.entries().len() > 2
        {
            stop_at = n;
            let (mut again, _) = Driver::resume(
                config(p.clone()),
                store.reopened(),
                d.session(),
                Scripted::new(vec![done(&p, "Red.")]),
                TestTools::default(),
                Host(None),
            )
            .await
            .unwrap();
            assert!(again.in_turn());
            let end = again.continue_turn(&mut |_| {}).await.unwrap().unwrap();
            assert_eq!(end.reply.as_deref(), Some("Red."));
            let log = store.inner.load(d.session()).await.unwrap().log;
            check_log(&log);
            break;
        }
    }
    assert!(
        stop_at > 0,
        "no stop point left the steer as the last entry"
    );
}

#[tokio::test]
async fn a_resume_with_another_config_is_refused() {
    let p = ModelProfile::native();
    let (d, model, store) = driver(p.clone(), 5, vec![], TestTools::default()).await;
    let session = d.session();
    drop(d);
    let other = sarathi_core::driver::DriverConfig::new(p, defs(), "You are someone else.");
    let r = Driver::resume(
        other,
        store,
        session,
        model,
        TestTools::default(),
        Host(None),
    )
    .await;
    assert!(matches!(
        r,
        Err(sarathi_core::driver::DriverError::Session(_))
    ));
}

/// A resumed turn that ends in its first actions (the final answer is on disk, its end is not) while a steer waits:
/// the steer joins the next turn, and the next turn runs.
#[tokio::test(start_paused = true)]
async fn a_late_steer_on_a_resumed_turn_that_ends_at_once_waits_for_the_next_turn() {
    let p = ModelProfile::native();
    let model = Scripted::new(vec![done(&p, "One."), done(&p, "Two."), done(&p, "Three.")]);
    let store = Shared::default().stopping_after(4);
    let (mut d, _) = Driver::new(
        config(p.clone()),
        store.clone(),
        model.clone(),
        TestTools::default(),
        Host(None),
    )
    .await
    .unwrap();
    d.run_turn(user("One?"), &mut |_| {}).await.unwrap();
    let session = d.session();
    drop(d);
    let (mut again, h) = Driver::resume(
        config(p),
        store.reopened(),
        session,
        model.clone(),
        TestTools::default(),
        Host(None),
    )
    .await
    .unwrap();
    h.send(Op::Steer {
        content: RawJson::string("and also this"),
    });
    let end = again.continue_turn(&mut |_| {}).await.unwrap().unwrap();
    assert_eq!(end.reply.as_deref(), Some("One."));
    assert!(!again.in_turn());
    assert_eq!(
        kinds(again.log()).last().map(String::as_str),
        Some("turn.end")
    );
    let next = tokio::time::timeout(
        Duration::from_secs(60),
        again.run_turn(user("Next?"), &mut |_| {}),
    )
    .await
    .expect("the next turn does not hang")
    .unwrap();
    // The held steer keeps the turn going after its first answer.
    assert_eq!(next.reply.as_deref(), Some("Three."));
    let last = String::from_utf8(model.bodies().last().unwrap().clone()).unwrap();
    assert!(
        last.contains("and also this") && last.contains("Next?"),
        "{last}"
    );
    check_log(again.log());
}

#[tokio::test]
async fn a_turn_whose_first_write_fails_was_reported_started_before_it_ends() {
    let model = Scripted::new(vec![done(&ModelProfile::native(), "never")]);
    let store = Shared {
        fail_once: Some(1),
        ..Shared::default()
    };
    let (mut d, _) = Driver::new(
        config(ModelProfile::native()),
        store,
        model,
        TestTools::default(),
        Host(None),
    )
    .await
    .unwrap();
    let mut events = Vec::new();
    let end = d
        .run_turn(user("Go."), &mut sink(&mut events))
        .await
        .unwrap();
    assert_eq!(end.outcome, Outcome::Failed);
    let started = events
        .iter()
        .position(|e| matches!(e, Event::TurnStarted { .. }))
        .unwrap_or_else(|| panic!("{events:#?}"));
    let done = events
        .iter()
        .position(|e| matches!(e, Event::Done { .. }))
        .unwrap();
    assert!(started < done, "{events:#?}");
}
