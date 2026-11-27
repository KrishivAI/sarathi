// Derived from Vyasa `crates/ai/src/agent.rs` @ 2403be1 (MIT OR Apache-2.0); see PROVENANCE.md
// SPDX-License-Identifier: Apache-2.0
//! The core's turns as input and action sequences.

use std::time::Duration;

use sarathi_protocol::{BlockKind, BlockView, Decision, ErrorKind, Event, Op, Outcome, RawJson};

use super::*;
use crate::profile::{Effect, ModelProfile, ToolDef};

fn raw(s: &str) -> RawJson {
    RawJson::from_text(s).unwrap()
}

fn tools() -> Vec<ToolDef> {
    vec![
        ToolDef::new(
            "look",
            "Shows the page.",
            raw(r#"{"type":"object"}"#),
            Effect::Read,
        ),
        ToolDef::new(
            "edit",
            "Changes the page.",
            raw(r#"{"type":"object"}"#),
            Effect::Write,
        ),
    ]
}

/// A native answer: optional text, then `tool_use` blocks `(id, name, input)`, with exact ranges.
pub(super) fn native(text: Option<&str>, calls: &[(&str, &str, &str)], stop: &str) -> Answer {
    let mut body = String::from(r#"{"content":["#);
    let mut blocks = Vec::new();
    let pos = |s: &String| u32::try_from(s.len()).unwrap();
    if let Some(t) = text {
        let start = pos(&body);
        body.push_str(&format!(
            r#"{{"type":"text","text":{}}}"#,
            serde_json::to_string(t).unwrap()
        ));
        blocks.push(BlockView {
            kind: BlockKind::Text,
            range: start..pos(&body),
        });
    }
    for (id, name, input) in calls {
        if body.ends_with('}') {
            body.push(',');
        }
        let start = pos(&body);
        body.push_str(&format!(
            r#"{{"type":"tool_use","id":"{id}","name":"{name}","input":"#
        ));
        let i0 = pos(&body);
        body.push_str(input);
        let i1 = pos(&body);
        body.push('}');
        blocks.push(BlockView {
            kind: BlockKind::ToolUse {
                id: (*id).into(),
                name: (*name).into(),
                input: i0..i1,
            },
            range: start..pos(&body),
        });
    }
    body.push_str(&format!(r#"],"stop_reason":"{stop}"}}"#));
    let view = ResponseView {
        stop_reason: Some(stop.into()),
        blocks,
        ..ResponseView::default()
    };
    Answer::new(raw(&body), view)
}

fn json(text: &str) -> Answer {
    Answer::new(RawJson::string(text), ResponseView::default()).with_text(text)
}

fn user(text: &str) -> Input {
    Input::Op(Op::UserMessage {
        content: RawJson::string(text),
    })
}

/// A core with a check that every seq it hands out is the next one.
pub(super) struct Harness {
    pub(super) core: Core,
    expected: u64,
    calling: Seq,
    pub(super) log: Vec<(Seq, Record, Durability)>,
}

impl Harness {
    pub(super) fn new(profile: ModelProfile, limits: Limits) -> Self {
        let mut config = CoreConfig::new(profile, tools(), Seq(1));
        config.limits = limits;
        Harness {
            core: Core::new(config),
            expected: 2,
            calling: Seq(0),
            log: Vec::new(),
        }
    }

    fn native() -> Self {
        Harness::new(ModelProfile::native(), Limits::default())
    }

    fn json() -> Self {
        Harness::new(ModelProfile::json_object(true), Limits::default())
    }

    /// Answers the model call in flight.
    pub(super) fn answer(&mut self, answer: Answer) -> Vec<Action> {
        let at = self.calling;
        self.send(Input::ModelAnswered { at, answer })
    }

    /// Fails the model call in flight.
    pub(super) fn fail(&mut self, kind: ErrorKind, detail: &str) -> Vec<Action> {
        let at = self.calling;
        self.send(Input::ModelFailed {
            at,
            kind,
            detail: detail.into(),
        })
    }

    pub(super) fn send(&mut self, input: Input) -> Vec<Action> {
        let actions = self.core.step(input);
        self.track(actions)
    }

    /// A core resumed mid-turn with `history` and the log's next seq `next`.
    fn resumed(
        profile: ModelProfile,
        history: Vec<Seq>,
        next: u64,
        open: OpenTurn,
    ) -> (Self, Vec<Action>) {
        let mut config = CoreConfig::new(profile, tools(), Seq(1));
        config.history = history;
        config.next_seq = Seq(next);
        let (core, actions) = Core::resume(config, open);
        let mut h = Harness {
            core,
            expected: next,
            calling: Seq(0),
            log: Vec::new(),
        };
        let actions = h.track(actions);
        (h, actions)
    }

    fn track(&mut self, actions: Vec<Action>) -> Vec<Action> {
        for a in &actions {
            match a {
                Action::Append {
                    at,
                    records,
                    durability,
                } => {
                    assert_eq!(at.0, self.expected, "appends are dense: {actions:#?}");
                    for (i, r) in records.iter().enumerate() {
                        self.log
                            .push((Seq(at.0 + i as u64), r.clone(), *durability));
                    }
                    self.expected += records.len() as u64;
                }
                Action::CallModel { at, .. } => {
                    assert_eq!(at.0, self.expected, "the request takes the next seq");
                    self.calling = *at;
                    self.expected += 1;
                }
                _ => {}
            }
        }
        assert_eq!(self.core.next_seq().0, self.expected);
        actions
    }
}

fn appended(actions: &[Action]) -> Vec<Record> {
    actions
        .iter()
        .filter_map(|a| match a {
            Action::Append { records, .. } => Some(records.clone()),
            _ => None,
        })
        .flatten()
        .collect()
}

fn results(actions: &[Action]) -> Vec<(String, String, bool)> {
    appended(actions)
        .into_iter()
        .filter_map(|r| match r {
            Record::ToolResult {
                call,
                content,
                is_error,
            } => Some((
                call.to_string(),
                serde_json::from_str::<String>(content.as_str())
                    .unwrap_or_else(|_| content.as_str().to_owned()),
                is_error,
            )),
            _ => None,
        })
        .collect()
}

fn call_model(actions: &[Action]) -> Option<&Recipe> {
    actions.iter().find_map(|a| match a {
        Action::CallModel { recipe, .. } => Some(recipe),
        _ => None,
    })
}

fn evaluate(actions: &[Action]) -> Option<CallId> {
    actions.iter().find_map(|a| match a {
        Action::Evaluate { call, .. } => Some(call.clone()),
        _ => None,
    })
}

fn run_tool(actions: &[Action]) -> Option<(CallId, Effect)> {
    actions.iter().find_map(|a| match a {
        Action::RunTool { call, effect, .. } => Some((call.clone(), *effect)),
        _ => None,
    })
}

fn turn_end(actions: &[Action]) -> Option<TurnEnd> {
    actions.iter().find_map(|a| match a {
        Action::EndTurn(end) => Some(end.clone()),
        _ => None,
    })
}

fn errors(actions: &[Action]) -> Vec<ErrorKind> {
    actions
        .iter()
        .filter_map(|a| match a {
            Action::Emit(Event::Error { kind, .. }) => Some(*kind),
            _ => None,
        })
        .collect()
}

fn allow(h: &mut Harness, call: &CallId) -> Vec<Action> {
    h.send(Input::Verdict {
        call: call.clone(),
        verdict: Verdict::Allow,
    })
}

fn done(h: &mut Harness, call: &CallId, text: &str, is_error: bool) -> Vec<Action> {
    h.send(Input::ToolDone {
        call: call.clone(),
        output: ToolOutput {
            content: RawJson::string(text),
            is_error,
        },
    })
}

/// Allows and runs the evaluated call, returning what its result led to.
fn allow_and_run(h: &mut Harness, actions: &[Action], text: &str) -> Vec<Action> {
    let call = evaluate(actions).expect("a call to evaluate");
    let started = allow(h, &call);
    let (ran, _) = run_tool(&started).expect("the call runs");
    assert_eq!(ran, call);
    done(h, &call, text, false)
}

#[test]
fn a_native_turn_runs_each_call_in_order_then_finishes() {
    let mut h = Harness::native();
    let a = h.send(user("fix the title"));
    assert_eq!(
        a[0],
        Action::Emit(Event::TurnStarted {
            turn: TurnId(Seq(2))
        }),
        "the turn is reported before its first write"
    );
    assert!(matches!(a[1], Action::Append { at: Seq(2), .. }));
    assert_eq!(
        call_model(&a).unwrap(),
        &Recipe {
            prefix: Seq(1),
            messages: vec![Seq(2)]
        }
    );
    assert_eq!(h.core.phase(), Phase::Calling);

    let answer = native(
        Some("Looking first."),
        &[
            ("toolu_1", "look", "{}"),
            ("toolu_2", "edit", r#"{"title" : "Hi"}"#),
        ],
        "tool_use",
    );
    let a = h.answer(answer);
    assert_eq!(
        a[0],
        Action::Append {
            at: Seq(4),
            records: vec![
                Record::ModelResponse,
                Record::AssistantMessage {
                    from: Seq(4),
                    calls: vec![
                        AskedCall {
                            call: "toolu_1".into(),
                            tool: "look".into()
                        },
                        AskedCall {
                            call: "toolu_2".into(),
                            tool: "edit".into()
                        },
                    ],
                }
            ],
            durability: Durability::Durable,
        },
        "the answer and its acceptance are one durable batch"
    );
    assert!(matches!(
        a[1],
        Action::Emit(Event::AssistantMessage { seq: Seq(5), .. })
    ));
    assert_eq!(evaluate(&a).unwrap().as_str(), "toolu_1");
    assert_eq!(h.core.phase(), Phase::Deciding);

    let a = allow(&mut h, &"toolu_1".into());
    assert_eq!(
        a[0],
        Action::Append {
            at: Seq(6),
            records: vec![Record::ToolStart {
                call: "toolu_1".into(),
                tool: "look".into(),
                effect: Effect::Read
            }],
            durability: Durability::Buffered,
        }
    );
    assert_eq!(h.core.phase(), Phase::Running);
    let a = done(&mut h, &"toolu_1".into(), "a page", false);
    assert_eq!(results(&a), [("toolu_1".into(), "a page".into(), false)]);
    assert_eq!(evaluate(&a).unwrap().as_str(), "toolu_2");

    let a = allow(&mut h, &"toolu_2".into());
    let Action::Append { durability, .. } = &a[0] else {
        panic!("{a:?}")
    };
    assert_eq!(
        *durability,
        Durability::Durable,
        "a write tool's start is durable"
    );
    let Some(Action::RunTool { input, .. }) =
        a.iter().find(|a| matches!(a, Action::RunTool { .. }))
    else {
        panic!()
    };
    assert_eq!(
        input.as_str(),
        r#"{"title" : "Hi"}"#,
        "the input bytes as the model wrote them"
    );
    let a = done(&mut h, &"toolu_2".into(), "1 line changed", false);
    let Action::Append { durability, .. } = &a[0] else {
        panic!()
    };
    assert_eq!(
        *durability,
        Durability::Durable,
        "a write tool's result is durable"
    );
    assert_eq!(
        call_model(&a).unwrap().messages,
        [Seq(2), Seq(5), Seq(7), Seq(9)],
        "user message, assistant message, both results"
    );

    let a = h.answer(native(Some("Done: title fixed."), &[], "end_turn"));
    let end = turn_end(&a).unwrap();
    assert_eq!(end.outcome, Outcome::Finished);
    assert_eq!(end.reply.as_deref(), Some("Done: title fixed."));
    assert!(h.log.iter().any(|(_, r, d)| {
        *r == Record::TurnEnd {
            outcome: Outcome::Finished,
        } && *d == Durability::Durable
    }));
    assert_eq!(h.core.phase(), Phase::Idle);
}

#[test]
fn acts_then_finishes_and_records_the_steps() {
    let mut h = Harness::json();
    h.send(user("make it blue"));
    let a = h.answer(json(r#"{"thought":"check","tool":"look","input":{}}"#));
    let call = evaluate(&a).unwrap();
    assert_eq!(
        call.as_str(),
        "s5-0",
        "the call id is made from the assistant message's seq"
    );
    let a = allow_and_run(&mut h, &a, "a white page");
    assert_eq!(
        results(&a),
        [("s5-0".into(), "observation:\na white page".into(), false)]
    );
    let a = h.answer(json(
        r#"{"thought":"ok","tool":"done","reply":"It is blue now."}"#,
    ));
    let end = turn_end(&a).unwrap();
    assert_eq!(end.outcome, Outcome::Finished);
    assert_eq!(end.reply.as_deref(), Some("It is blue now."));
}

#[test]
fn an_unknown_tool_becomes_a_correction_the_model_reads() {
    let mut h = Harness::json();
    h.send(user("go"));
    let a = h.answer(json(r#"{"thought":"t","tool":"paint","input":{}}"#));
    assert_eq!(
        results(&a),
        [(
            "s5-0".into(),
            "observation:\n\"paint\" is not a tool. The tools are: look, edit — or \"done\"."
                .into(),
            true
        )]
    );
    assert!(call_model(&a).is_some(), "the model is asked again");
    let a = h.answer(json(r#"{"thought":"t","tool":"done","reply":"ok"}"#));
    assert_eq!(turn_end(&a).unwrap().outcome, Outcome::Finished);
}

#[test]
fn the_step_cap_ends_the_run_honestly() {
    let mut h = Harness::new(ModelProfile::json_object(true), Limits::with_max_steps(2));
    h.send(user("go"));
    for n in 0..2 {
        let a = h.answer(json(&format!(
            r#"{{"thought":"t","tool":"look","input":{{"n":{n}}}}}"#
        )));
        let a = allow_and_run(&mut h, &a, "seen");
        if n == 1 {
            assert!(call_model(&a).is_none());
            let end = turn_end(&a).unwrap();
            assert_eq!(end.outcome, Outcome::RanOut);
            assert!(end.reply.unwrap().contains("ran out of steps"));
        }
    }
}

#[test]
fn a_readable_tool_refusal_is_an_observation_not_a_crash() {
    let mut h = Harness::json();
    h.send(user("go"));
    // Two failures, then a refused tool: the refusal resets the count, so two more failures do not end the turn.
    for _ in 0..2 {
        h.answer(json("not json"));
    }
    let a = h.answer(json(r#"{"thought":"t","tool":"edit","input":{}}"#));
    let call = evaluate(&a).unwrap();
    allow(&mut h, &call);
    let a = done(&mut h, &call, "no such section", true);
    assert_eq!(
        results(&a)[0].1,
        "observation:\nThe tool refused: no such section"
    );
    for _ in 0..2 {
        let a = h.answer(json("still not json"));
        assert!(turn_end(&a).is_none());
    }
}

#[test]
fn invalid_json_retries_then_succeeds() {
    let mut h = Harness::json();
    h.send(user("go"));
    let a = h.answer(json("sure!"));
    let correction = appended(&a)
        .into_iter()
        .find_map(|r| match r {
            Record::UserMessage { content, .. } => Some(content),
            _ => None,
        })
        .unwrap();
    assert!(
        correction
            .as_str()
            .starts_with(r#""That reply was unusable ("#)
    );
    let recipe = call_model(&a).unwrap();
    assert_eq!(
        recipe.messages,
        [Seq(2), Seq(5), Seq(6)],
        "user, the bad answer, the correction"
    );
    let a = h.answer(json(r#"{"thought":"t","tool":"done"}"#));
    let end = turn_end(&a).unwrap();
    assert_eq!(end.outcome, Outcome::Finished);
    assert_eq!(end.reply.as_deref(), Some("Done."));
}

#[test]
fn schema_mismatch_fails_after_three_in_a_row() {
    let mut h = Harness::json();
    h.send(user("go"));
    h.answer(json("{}"));
    h.answer(json(r#"{"tool":"look"}"#));
    let a = h.answer(json("[]"));
    assert_eq!(errors(&a), [ErrorKind::Guard]);
    assert_eq!(turn_end(&a).unwrap().outcome, Outcome::Failed);
    assert!(call_model(&a).is_none());
}

#[test]
fn transport_error_bubbles_immediately() {
    let mut h = Harness::json();
    h.send(user("go"));
    let a = h.fail(ErrorKind::Upstream, "overloaded");
    assert_eq!(
        appended(&a)[0],
        Record::ModelFailure {
            kind: ErrorKind::Upstream,
            detail: "overloaded".into()
        }
    );
    assert_eq!(errors(&a), [ErrorKind::Upstream]);
    assert_eq!(turn_end(&a).unwrap().outcome, Outcome::Failed);
}

#[test]
fn a_third_identical_call_is_not_run() {
    let mut h = Harness::native();
    h.send(user("go"));
    for n in 1..=2 {
        let id = format!("toolu_{n}");
        let a = h.answer(native(None, &[(&id, "look", r#"{"p":1}"#)], "tool_use"));
        allow_and_run(&mut h, &a, "same page");
    }
    let a = h.answer(native(
        None,
        &[("toolu_3", "look", r#"{"p":1}"#)],
        "tool_use",
    ));
    assert!(evaluate(&a).is_none());
    let r = results(&a);
    assert_eq!(r[0].0, "toolu_3");
    assert!(r[0].1.contains("identical to the previous two calls") && r[0].2);
    assert!(call_model(&a).is_some());
    // A different input runs again.
    let a = h.answer(native(
        None,
        &[("toolu_4", "look", r#"{"p":2}"#)],
        "tool_use",
    ));
    assert!(evaluate(&a).is_some());
}

#[test]
fn unknown_tools_three_times_fail_and_answer_every_call() {
    let mut h = Harness::native();
    h.send(user("go"));
    let calls = [
        ("toolu_1", "paint", "{}"),
        ("toolu_2", "draw", "{}"),
        ("toolu_3", "sing", "{}"),
        ("toolu_4", "look", "{}"),
    ];
    let a = h.answer(native(None, &calls, "tool_use"));
    let r = results(&a);
    assert_eq!(r.len(), 4, "every tool use is answered: {r:?}");
    assert!(r[0].1.ends_with("The tools are: look, edit."));
    assert_eq!(r[3].1, "skipped: the turn ended after repeated failures");
    assert_eq!(errors(&a), [ErrorKind::Guard]);
    assert_eq!(turn_end(&a).unwrap().outcome, Outcome::Failed);
}

#[test]
fn an_unreadable_input_is_answered_and_counted() {
    let mut h = Harness::native();
    h.send(user("go"));
    let mut answer = native(None, &[("toolu_1", "look", "{}")], "tool_use");
    if let BlockKind::ToolUse { input, .. } = &mut answer.view.blocks[0].kind {
        input.end -= 1;
    }
    let a = h.answer(answer);
    let r = results(&a);
    assert_eq!(r[0].1, "the call's input could not be read as JSON");
    assert!(call_model(&a).is_some());
}

#[test]
fn permissions_deny_ask_and_allow_always() {
    let mut h = Harness::native();
    h.send(user("go"));
    let a = h.answer(native(None, &[("toolu_1", "edit", "{}")], "tool_use"));
    let call = evaluate(&a).unwrap();
    let a = h.send(Input::Verdict {
        call: call.clone(),
        verdict: Verdict::Deny("outside the project".into()),
    });
    assert_eq!(
        results(&a),
        [("toolu_1".into(), "outside the project".into(), true)]
    );
    assert!(run_tool(&a).is_none());

    let a = h.answer(native(None, &[("toolu_2", "edit", "{}")], "tool_use"));
    let a = h.send(Input::Verdict {
        call: evaluate(&a).unwrap(),
        verdict: Verdict::Ask,
    });
    assert_eq!(
        appended(&a),
        [Record::ToolAsk {
            call: "toolu_2".into(),
            tool: "edit".into(),
            input: raw("{}")
        }]
    );
    assert!(matches!(a[1], Action::Emit(Event::PermissionAsked { .. })));
    assert_eq!(h.core.phase(), Phase::Asking);
    // A decision for another call is refused and changes nothing.
    let a = h.send(Input::Op(Op::ToolDecision {
        call: "toolu_9".into(),
        decision: Decision::AllowOnce,
    }));
    assert_eq!(errors(&a), [ErrorKind::InvalidRequest]);
    assert_eq!(h.core.phase(), Phase::Asking);
    let a = h.send(Input::Op(Op::ToolDecision {
        call: "toolu_2".into(),
        decision: Decision::AllowAlways,
    }));
    assert_eq!(
        appended(&a)[0],
        Record::ToolDecision {
            call: "toolu_2".into(),
            decision: Decision::AllowAlways
        }
    );
    assert!(run_tool(&a).is_some());
    let a = done(&mut h, &"toolu_2".into(), "changed", false);

    // Later asks for the same tool run without asking.
    let _ = a;
    let a = h.answer(native(
        None,
        &[("toolu_3", "edit", r#"{"x":1}"#)],
        "tool_use",
    ));
    let a = h.send(Input::Verdict {
        call: evaluate(&a).unwrap(),
        verdict: Verdict::Ask,
    });
    assert!(run_tool(&a).is_some());
    done(&mut h, &"toolu_3".into(), "changed", false);

    let a = h.answer(native(None, &[("toolu_4", "look", "{}")], "tool_use"));
    h.send(Input::Verdict {
        call: evaluate(&a).unwrap(),
        verdict: Verdict::Ask,
    });
    let a = h.send(Input::Op(Op::ToolDecision {
        call: "toolu_4".into(),
        decision: Decision::Deny {
            reason: "not now".into(),
        },
    }));
    assert_eq!(results(&a), [("toolu_4".into(), "not now".into(), true)]);
}

#[test]
fn a_steer_waits_for_the_tool_boundary() {
    let mut h = Harness::native();
    h.send(user("go"));
    let a = h.send(Input::Op(Op::Steer {
        content: RawJson::string("use red"),
    }));
    assert!(a.is_empty(), "held while the model answers: {a:?}");
    let a = h.answer(native(None, &[("toolu_1", "look", "{}")], "tool_use"));
    assert!(
        appended(&a)
            .iter()
            .all(|r| !matches!(r, Record::UserMessage { .. }))
    );
    let a = allow_and_run(&mut h, &a, "seen");
    let recs = appended(&a);
    assert!(matches!(recs[0], Record::ToolResult { .. }));
    assert_eq!(
        recs[1],
        Record::UserMessage {
            content: RawJson::string("use red"),
            steer: true,
        },
        "after the step's last result"
    );
    assert_eq!(
        call_model(&a).unwrap().messages,
        [Seq(2), Seq(5), Seq(7), Seq(8)]
    );
}

#[test]
fn a_steer_keeps_a_finishing_turn_going() {
    let mut h = Harness::native();
    h.send(user("go"));
    h.send(Input::Op(Op::UserMessage {
        content: RawJson::string("and the footer"),
    }));
    let a = h.answer(native(Some("Done."), &[], "end_turn"));
    assert!(turn_end(&a).is_none());
    assert!(call_model(&a).is_some());
    let a = h.answer(native(Some("Footer too."), &[], "end_turn"));
    assert_eq!(turn_end(&a).unwrap().outcome, Outcome::Finished);
}

#[test]
fn a_steer_at_rest_starts_a_turn() {
    let mut h = Harness::native();
    let a = h.send(Input::Op(Op::Steer {
        content: RawJson::string("hi"),
    }));
    assert!(call_model(&a).is_some());
}

#[test]
fn interrupting_a_model_call_drops_it() {
    let mut h = Harness::native();
    h.send(user("go"));
    h.send(Input::Op(Op::Steer {
        content: RawJson::string("later"),
    }));
    let a = h.send(Input::Op(Op::Interrupt));
    assert_eq!(a[0], Action::Cancel(Running::Model { at: Seq(3) }));
    assert_eq!(
        appended(&a)[0],
        Record::ModelFailure {
            kind: ErrorKind::Cancelled,
            detail: "interrupted".into()
        }
    );
    let end = turn_end(&a).unwrap();
    assert_eq!(end.outcome, Outcome::Interrupted);
    assert_eq!(
        end.unsent,
        [RawJson::string("later")],
        "a held steer goes back to the host"
    );
    // A late answer is refused.
    let a = h.answer(native(None, &[], "end_turn"));
    assert_eq!(errors(&a), [ErrorKind::InvalidRequest]);
}

#[test]
fn interrupting_a_read_tool_drops_it_and_skips_the_rest() {
    let mut h = Harness::native();
    h.send(user("go"));
    let a = h.answer(native(
        None,
        &[("toolu_1", "look", "{}"), ("toolu_2", "edit", "{}")],
        "tool_use",
    ));
    allow(&mut h, &evaluate(&a).unwrap());
    let a = h.send(Input::Op(Op::Interrupt));
    assert_eq!(
        a,
        [Action::Cancel(Running::Tool {
            call: "toolu_1".into(),
            grace: None
        })]
    );
    assert_eq!(h.core.phase(), Phase::Stopping);
    let a = h.send(Input::ToolAbandoned {
        call: "toolu_1".into(),
    });
    assert_eq!(
        results(&a),
        [
            ("toolu_1".into(), "interrupted".into(), true),
            ("toolu_2".into(), "skipped: interrupted".into(), true)
        ]
    );
    assert_eq!(turn_end(&a).unwrap().outcome, Outcome::Interrupted);
}

#[test]
fn an_interrupted_write_tool_gets_its_grace() {
    let mut h = Harness::native();
    h.send(user("go"));
    let a = h.answer(native(None, &[("toolu_1", "edit", "{}")], "tool_use"));
    allow(&mut h, &evaluate(&a).unwrap());
    let a = h.send(Input::Op(Op::Interrupt));
    assert_eq!(
        a,
        [Action::Cancel(Running::Tool {
            call: "toolu_1".into(),
            grace: Some(Duration::from_secs(30))
        })]
    );
    // It stops in time: its own result is kept.
    let a = done(&mut h, &"toolu_1".into(), "saved", false);
    assert_eq!(results(&a), [("toolu_1".into(), "saved".into(), false)]);
    assert_eq!(turn_end(&a).unwrap().outcome, Outcome::Interrupted);

    // Another turn: a second interrupt drops it, and its effect is unknown.
    h.send(user("again"));
    let a = h.answer(native(None, &[("toolu_2", "edit", "{}")], "tool_use"));
    allow(&mut h, &evaluate(&a).unwrap());
    h.send(Input::Op(Op::Interrupt));
    let a = h.send(Input::Op(Op::Interrupt));
    assert_eq!(
        a,
        [Action::Cancel(Running::Tool {
            call: "toolu_2".into(),
            grace: None
        })]
    );
    assert!(h.send(Input::Op(Op::Interrupt)).is_empty());
    let a = h.send(Input::ToolAbandoned {
        call: "toolu_2".into(),
    });
    assert_eq!(
        results(&a),
        [(
            "toolu_2".into(),
            "interrupted; its effect is unknown".into(),
            true
        )]
    );
    let Action::Append { durability, .. } = &a[0] else {
        panic!()
    };
    assert_eq!(*durability, Durability::Durable);
}

#[test]
fn interrupting_an_ask_answers_the_call() {
    let mut h = Harness::native();
    h.send(user("go"));
    let a = h.answer(native(None, &[("toolu_1", "edit", "{}")], "tool_use"));
    h.send(Input::Verdict {
        call: evaluate(&a).unwrap(),
        verdict: Verdict::Ask,
    });
    let a = h.send(Input::Op(Op::Interrupt));
    assert_eq!(
        results(&a),
        [("toolu_1".into(), "skipped: interrupted".into(), true)]
    );
    assert_eq!(turn_end(&a).unwrap().outcome, Outcome::Interrupted);
}

#[test]
fn the_spend_cap_stops_the_next_call() {
    let limits = Limits {
        max_cost_micros: Some(100),
        ..Limits::default()
    };
    let mut h = Harness::new(ModelProfile::native(), limits);
    h.send(user("go"));
    let answer = native(None, &[("toolu_1", "look", "{}")], "tool_use").with_cost(150);
    let a = h.answer(answer);
    let a = allow_and_run(&mut h, &a, "seen");
    assert!(call_model(&a).is_none());
    assert_eq!(errors(&a), [ErrorKind::BudgetExceeded]);
    assert_eq!(turn_end(&a).unwrap().outcome, Outcome::RanOut);
}

#[test]
fn a_refusal_ends_the_turn() {
    let mut h = Harness::native();
    h.send(user("go"));
    let a = h.answer(native(Some("I can't."), &[], "refusal"));
    assert_eq!(turn_end(&a).unwrap().outcome, Outcome::Refused);
}

#[test]
fn a_paused_turn_is_continued() {
    let mut h = Harness::native();
    h.send(user("go"));
    let a = h.answer(native(Some("searching"), &[], "pause_turn"));
    assert_eq!(call_model(&a).unwrap().messages, [Seq(2), Seq(5)]);
}

#[test]
fn shutdown_finishes_the_step_then_stops() {
    let mut h = Harness::native();
    h.send(user("go"));
    let a = h.answer(native(None, &[("toolu_1", "look", "{}")], "tool_use"));
    assert!(h.send(Input::Op(Op::Shutdown)).is_empty());
    let a = allow_and_run(&mut h, &a, "seen");
    assert!(call_model(&a).is_none());
    assert_eq!(turn_end(&a).unwrap().outcome, Outcome::Interrupted);
    assert!(
        h.send(Input::Op(Op::Shutdown)).is_empty(),
        "nothing to stop at rest"
    );
    assert!(h.send(Input::Op(Op::Interrupt)).is_empty());
}

#[test]
fn inputs_out_of_place_are_refused() {
    let mut h = Harness::native();
    for input in [
        Input::ModelAnswered {
            at: Seq(3),
            answer: native(None, &[], "end_turn"),
        },
        Input::ModelFailed {
            at: Seq(3),
            kind: ErrorKind::Network,
            detail: "x".into(),
        },
        Input::Verdict {
            call: "a".into(),
            verdict: Verdict::Allow,
        },
        Input::ToolAbandoned { call: "a".into() },
        Input::Op(Op::ToolOutput {
            call: "a".into(),
            output: ToolOutput {
                content: RawJson::null(),
                is_error: false,
            },
        }),
    ] {
        let a = h.send(input);
        assert_eq!(errors(&a), [ErrorKind::InvalidRequest]);
    }
    assert_eq!(h.core.phase(), Phase::Idle);
    assert!(h.log.is_empty());
}

#[test]
fn limits_are_clamped_and_a_history_carries_on() {
    let mut config = CoreConfig::new(ModelProfile::native(), tools(), Seq(1));
    config.limits = Limits::with_max_steps(99);
    config.history = vec![Seq(2), Seq(5)];
    config.next_seq = Seq(7);
    let mut core = Core::new(config);
    assert_eq!(core.limits().max_steps, MAX_STEPS_CEILING);
    let a = core.step(user("next"));
    assert_eq!(call_model(&a).unwrap().messages, [Seq(2), Seq(5), Seq(7)]);
    assert_eq!(core.history(), [Seq(2), Seq(5), Seq(7)]);
    let zero = Core::new(CoreConfig {
        limits: Limits::with_max_steps(0),
        ..CoreConfig::new(ModelProfile::native(), tools(), Seq(1))
    });
    assert_eq!(zero.limits().max_steps, 1);
}

#[test]
fn an_unknown_tool_on_three_steps_in_a_row_fails_the_turn() {
    for json_profile in [false, true] {
        let mut h = if json_profile {
            Harness::json()
        } else {
            Harness::native()
        };
        h.send(user("go"));
        for n in 0..3 {
            let a = if json_profile {
                h.answer(json(r#"{"thought":"t","tool":"paint","input":{}}"#))
            } else {
                let id = format!("toolu_{n}");
                h.answer(native(None, &[(&id, "paint", "{}")], "tool_use"))
            };
            let end = turn_end(&a);
            if n < 2 {
                assert!(end.is_none(), "step {n}");
            } else {
                assert_eq!(end.unwrap().outcome, Outcome::Failed);
                assert_eq!(errors(&a), [ErrorKind::Guard]);
            }
        }
    }
}

#[test]
fn identical_calls_past_the_limit_count_until_the_turn_fails() {
    let mut h = Harness::native();
    h.send(user("go"));
    for n in 1..=5 {
        let id = format!("toolu_{n}");
        let a = h.answer(native(None, &[(&id, "look", "{}")], "tool_use"));
        match n {
            1 | 2 => {
                allow_and_run(&mut h, &a, "same");
            }
            3 | 4 => assert!(evaluate(&a).is_none() && turn_end(&a).is_none(), "call {n}"),
            _ => assert_eq!(turn_end(&a).unwrap().outcome, Outcome::Failed),
        }
    }
}

#[test]
fn progress_between_failures_resets_the_count() {
    let mut h = Harness::native();
    h.send(user("go"));
    h.answer(native(None, &[("toolu_1", "paint", "{}")], "tool_use"));
    let a = h.answer(native(None, &[("toolu_2", "look", "{}")], "tool_use"));
    allow_and_run(&mut h, &a, "seen");
    h.answer(native(None, &[("toolu_3", "paint", "{}")], "tool_use"));
    let a = h.answer(native(None, &[("toolu_4", "paint", "{}")], "tool_use"));
    assert!(turn_end(&a).is_none());
}

#[test]
fn an_answer_for_a_cancelled_call_is_refused() {
    let mut h = Harness::native();
    h.send(user("go"));
    let old = h.calling;
    h.send(Input::Op(Op::Interrupt));
    h.send(user("again"));
    assert_ne!(h.calling, old);
    let a = h.send(Input::ModelAnswered {
        at: old,
        answer: native(None, &[("toolu_1", "edit", "{}")], "tool_use"),
    });
    assert_eq!(errors(&a), [ErrorKind::InvalidRequest]);
    assert!(appended(&a).is_empty());
    assert_eq!(h.core.phase(), Phase::Calling);
    let a = h.send(Input::ModelFailed {
        at: old,
        kind: ErrorKind::Network,
        detail: "late".into(),
    });
    assert_eq!(errors(&a), [ErrorKind::InvalidRequest]);
    assert_eq!(h.core.phase(), Phase::Calling);
}

#[test]
fn an_aborted_turn_appends_nothing_and_rebases() {
    let mut h = Harness::native();
    h.send(user("go"));
    h.send(Input::Op(Op::Steer {
        content: RawJson::string("later"),
    }));
    // The request at 3 was not written: the store's next seq is still 3.
    let a = h.core.step(Input::Aborted {
        kind: ErrorKind::Log,
        detail: "disk full".into(),
        next_seq: Seq(3),
    });
    assert!(appended(&a).is_empty());
    assert_eq!(errors(&a), [ErrorKind::Log]);
    let end = turn_end(&a).unwrap();
    assert_eq!(end.outcome, Outcome::Failed);
    assert_eq!(end.unsent, [RawJson::string("later")]);
    assert_eq!(h.core.next_seq(), Seq(3));
    assert_eq!(h.core.history(), [Seq(2)]);
    // A rebase below a slice forgets it.
    assert!(!h.core.needs_rebuild());
    let a = h.core.step(Input::Aborted {
        kind: ErrorKind::Log,
        detail: "lost".into(),
        next_seq: Seq(2),
    });
    assert!(turn_end(&a).is_none(), "no turn to end");
    assert!(h.core.history().is_empty());
}

#[test]
fn an_abort_with_open_calls_needs_a_rebuild() {
    let mut h = Harness::native();
    h.send(user("go"));
    let a = h.answer(native(
        None,
        &[("toolu_1", "edit", "{}"), ("toolu_2", "look", "{}")],
        "tool_use",
    ));
    allow(&mut h, &evaluate(&a).unwrap());
    // The write tool's result could not be appended.
    let a = h.core.step(Input::Aborted {
        kind: ErrorKind::Log,
        detail: "disk full".into(),
        next_seq: Seq(7),
    });
    assert_eq!(turn_end(&a).unwrap().outcome, Outcome::Failed);
    assert!(h.core.needs_rebuild());
    let a = h.core.step(user("again"));
    assert_eq!(errors(&a), [ErrorKind::Internal]);
    assert!(call_model(&a).is_none(), "no request with unanswered calls");
}

#[test]
fn a_shutdown_before_the_verdict_never_asks() {
    let mut h = Harness::native();
    h.send(user("go"));
    let a = h.answer(native(None, &[("toolu_1", "edit", "{}")], "tool_use"));
    h.send(Input::Op(Op::Shutdown));
    let a = h.send(Input::Verdict {
        call: evaluate(&a).unwrap(),
        verdict: Verdict::Ask,
    });
    assert!(
        a.iter()
            .all(|a| !matches!(a, Action::Emit(Event::PermissionAsked { .. })))
    );
    assert_eq!(
        results(&a),
        [("toolu_1".into(), "skipped: interrupted".into(), true)]
    );
    assert_eq!(turn_end(&a).unwrap().outcome, Outcome::Interrupted);
}

#[test]
fn a_refusal_with_one_id_twice_answers_it_once() {
    let mut h = Harness::native();
    h.send(user("go"));
    let a = h.answer(native(
        None,
        &[("toolu_1", "look", "{}"), ("toolu_1", "edit", "{}")],
        "refusal",
    ));
    assert_eq!(results(&a).len(), 1);
    assert_eq!(turn_end(&a).unwrap().outcome, Outcome::Refused);
}

#[test]
fn tools_allowed_for_the_session_carry_over() {
    let mut config = CoreConfig::new(ModelProfile::native(), tools(), Seq(1));
    config.allowed_always = vec!["edit".into()];
    let mut h = Harness::native();
    h.core = Core::new(config);
    h.send(user("go"));
    let a = h.answer(native(None, &[("toolu_1", "edit", "{}")], "tool_use"));
    let a = h.send(Input::Verdict {
        call: evaluate(&a).unwrap(),
        verdict: Verdict::Ask,
    });
    assert!(
        run_tool(&a).is_some(),
        "no ask for a tool allowed for the session"
    );
    done(&mut h, &"toolu_1".into(), "ok", false);
    // The rules still deny.
    let a = h.answer(native(
        None,
        &[("toolu_2", "edit", r#"{"x":1}"#)],
        "tool_use",
    ));
    let a = h.send(Input::Verdict {
        call: evaluate(&a).unwrap(),
        verdict: Verdict::Deny("read-only mode".into()),
    });
    assert!(run_tool(&a).is_none());
    assert_eq!(results(&a)[0].1, "read-only mode");
}

#[test]
fn a_held_steer_at_the_step_cap_keeps_the_final_reply() {
    let mut h = Harness::new(ModelProfile::native(), Limits::with_max_steps(1));
    h.send(user("go"));
    h.send(Input::Op(Op::Steer {
        content: RawJson::string("and blue"),
    }));
    let a = h.answer(native(Some("All done."), &[], "end_turn"));
    let end = turn_end(&a).unwrap();
    assert_eq!(end.outcome, Outcome::Finished);
    assert_eq!(end.reply.as_deref(), Some("All done."));
    assert_eq!(end.unsent, [RawJson::string("and blue")]);
    assert!(
        appended(&a)
            .iter()
            .all(|r| !matches!(r, Record::UserMessage { .. }))
    );
}

#[test]
fn a_provider_spend_cap_ends_the_turn_ran_out() {
    let mut h = Harness::native();
    h.send(user("go"));
    h.send(Input::Op(Op::Steer {
        content: RawJson::string("later"),
    }));
    let a = h.fail(ErrorKind::BudgetExceeded, "monthly cap");
    let end = turn_end(&a).unwrap();
    assert_eq!(end.outcome, Outcome::RanOut);
    assert_eq!(
        end.unsent,
        [RawJson::string("later")],
        "held messages go back on a failure too"
    );
}

#[test]
fn a_read_tool_that_returns_after_its_cancel_keeps_its_result() {
    let mut h = Harness::native();
    h.send(user("go"));
    let a = h.answer(native(
        None,
        &[("toolu_1", "look", "{}"), ("toolu_2", "look", r#"{"b":1}"#)],
        "tool_use",
    ));
    allow(&mut h, &evaluate(&a).unwrap());
    h.send(Input::Op(Op::Interrupt));
    let a = done(&mut h, &"toolu_1".into(), "seen", false);
    assert_eq!(
        results(&a),
        [
            ("toolu_1".into(), "seen".into(), false),
            ("toolu_2".into(), "skipped: interrupted".into(), true)
        ]
    );
    assert_eq!(turn_end(&a).unwrap().outcome, Outcome::Interrupted);
    let a = h.send(Input::ToolAbandoned {
        call: "toolu_1".into(),
    });
    assert_eq!(errors(&a), [ErrorKind::InvalidRequest]);
}

#[test]
fn two_calls_with_one_id_end_the_turn() {
    let mut h = Harness::native();
    h.send(user("go"));
    let a = h.answer(native(
        None,
        &[("toolu_1", "look", "{}"), ("toolu_1", "edit", "{}")],
        "tool_use",
    ));
    let r = results(&a);
    assert_eq!(r.len(), 1, "each id answered once: {r:?}");
    assert_eq!(errors(&a), [ErrorKind::Malformed]);
    assert_eq!(turn_end(&a).unwrap().outcome, Outcome::Failed);
}

#[test]
fn a_shutdown_during_an_ask_ends_the_turn() {
    let mut h = Harness::native();
    h.send(user("go"));
    let a = h.answer(native(None, &[("toolu_1", "edit", "{}")], "tool_use"));
    h.send(Input::Verdict {
        call: evaluate(&a).unwrap(),
        verdict: Verdict::Ask,
    });
    let a = h.send(Input::Op(Op::Shutdown));
    assert_eq!(
        results(&a),
        [("toolu_1".into(), "skipped: interrupted".into(), true)]
    );
    assert_eq!(turn_end(&a).unwrap().outcome, Outcome::Interrupted);
}

#[test]
fn a_refusal_under_the_json_protocol_is_not_a_correction() {
    let mut h = Harness::json();
    h.send(user("go"));
    let mut answer = json("I won't do that.");
    answer.view.stop_reason = Some("refusal".into());
    let a = h.answer(answer);
    assert_eq!(turn_end(&a).unwrap().outcome, Outcome::Refused);
    assert!(
        appended(&a)
            .iter()
            .all(|r| !matches!(r, Record::UserMessage { .. }))
    );
}

#[test]
fn a_native_answer_without_text_has_no_reply() {
    let mut h = Harness::native();
    h.send(user("go"));
    let a = h.answer(native(None, &[], "end_turn"));
    assert_eq!(turn_end(&a).unwrap().reply, None);
}

mod resume {
    use super::*;

    fn open(steps: u32, at: ResumeAt) -> OpenTurn {
        OpenTurn {
            turn: TurnId(Seq(2)),
            steps,
            at,
        }
    }

    fn at_answer(answer: Answer, answered: &[&str], cut_short: &[&str]) -> ResumeAt {
        ResumeAt::Answer {
            assistant: Seq(5),
            answer: Box::new(answer),
            answered: answered.iter().map(|c| CallId::from(*c)).collect(),
            cut_short: cut_short.iter().map(|c| CallId::from(*c)).collect(),
        }
    }

    #[derive(Debug)]
    struct Never;
    impl Completion for Never {
        fn check(&self) -> Option<String> {
            Some("not yet".into())
        }
    }

    /// The refusal count is memory only: a turn resumed after a crash gets the full budget again, then ends `RanOut`.
    #[test]
    fn a_resumed_turn_has_its_full_completion_refusals_again() {
        let mut config = CoreConfig::new(ModelProfile::native(), tools(), Seq(1));
        config.history = vec![Seq(2)];
        config.next_seq = Seq(5);
        config.completion = Some(Arc::new(Never));
        let (core, actions) = Core::resume(config, open(1, ResumeAt::Call));
        let mut h = Harness {
            core,
            expected: 5,
            calling: Seq(0),
            log: Vec::new(),
        };
        let mut a = h.track(actions);
        let mut sent_back = 0;
        // Bounded: if the cap regresses the test fails here instead of looping forever.
        for _ in 0..=MAX_COMPLETION_REFUSALS {
            a = h.answer(native(Some("Done."), &[], "end_turn"));
            if turn_end(&a).is_some() {
                break;
            }
            sent_back += 1;
        }
        assert_eq!(sent_back, MAX_COMPLETION_REFUSALS);
        assert_eq!(turn_end(&a).unwrap().outcome, Outcome::RanOut);
    }

    #[test]
    fn a_started_write_is_answered_cut_short_and_never_run() {
        let answer = native(
            None,
            &[
                ("toolu_1", "look", "{}"),
                ("toolu_2", "edit", r#"{"title":"Hi"}"#),
                ("toolu_3", "look", r#"{"again":true}"#),
            ],
            "tool_use",
        );
        let (mut h, a) = Harness::resumed(
            ModelProfile::native(),
            vec![Seq(2), Seq(5), Seq(7)],
            9,
            open(1, at_answer(answer, &["toolu_1"], &["toolu_2"])),
        );
        assert!(matches!(a[0], Action::Emit(Event::TurnStarted { .. })));
        assert_eq!(
            results(&a),
            vec![("toolu_2".into(), CUT_SHORT.into(), true)],
            "only the cut-short write is answered, and never run"
        );
        assert!(a.iter().any(|x| matches!(
            x,
            Action::Append { durability: Durability::Durable, records, .. }
                if matches!(records[..], [Record::ToolResult { .. }])
        )));
        assert!(run_tool(&a).is_none());
        let next = evaluate(&a).unwrap();
        assert_eq!(
            next.as_str(),
            "toolu_3",
            "the call not started goes through the rules"
        );
        let a = allow(&mut h, &next);
        assert_eq!(run_tool(&a), Some((next.clone(), Effect::Read)));
        let a = done(&mut h, &next, "seen", false);
        let recipe = call_model(&a).unwrap();
        assert_eq!(
            recipe.messages,
            vec![Seq(2), Seq(5), Seq(7), Seq(9), Seq(11)]
        );
    }

    #[test]
    fn a_started_read_runs_again() {
        let answer = native(None, &[("toolu_1", "look", "{}")], "tool_use");
        let (_, a) = Harness::resumed(
            ModelProfile::native(),
            vec![Seq(2), Seq(5)],
            7,
            open(1, at_answer(answer, &[], &[])),
        );
        assert_eq!(evaluate(&a).unwrap().as_str(), "toolu_1");
        assert!(results(&a).is_empty());
    }

    #[test]
    fn every_call_answered_goes_on_to_the_next_request() {
        let answer = native(None, &[("toolu_1", "look", "{}")], "tool_use");
        let (_, a) = Harness::resumed(
            ModelProfile::native(),
            vec![Seq(2), Seq(5), Seq(7)],
            8,
            open(1, at_answer(answer, &["toolu_1"], &[])),
        );
        let recipe = call_model(&a).unwrap();
        assert_eq!(recipe.messages, vec![Seq(2), Seq(5), Seq(7)]);
        assert!(
            turn_end(&a).is_none(),
            "a step whose calls are all answered does not end the turn"
        );
    }

    #[test]
    fn a_request_without_an_answer_is_made_again() {
        let (h, a) = Harness::resumed(
            ModelProfile::native(),
            vec![Seq(2)],
            4,
            open(0, ResumeAt::Call),
        );
        let Some(Action::CallModel { at, recipe }) =
            a.iter().find(|x| matches!(x, Action::CallModel { .. }))
        else {
            panic!("no call: {a:#?}");
        };
        assert_eq!(
            *at,
            Seq(4),
            "a new request entry; the unanswered one stays as it is"
        );
        assert_eq!(recipe.messages, vec![Seq(2)]);
        assert_eq!(h.core.phase(), Phase::Calling);
    }

    #[test]
    fn an_accepted_final_answer_ends_the_turn() {
        let answer = native(Some("All done."), &[], "end_turn");
        let (h, a) = Harness::resumed(
            ModelProfile::native(),
            vec![Seq(2), Seq(5)],
            6,
            open(1, at_answer(answer, &[], &[])),
        );
        let end = turn_end(&a).unwrap();
        assert_eq!(end.outcome, Outcome::Finished);
        assert_eq!(end.reply.as_deref(), Some("All done."));
        assert_eq!(h.core.phase(), Phase::Idle);
    }

    #[test]
    fn the_step_cap_counts_the_steps_before_the_restart() {
        let (_, a) = Harness::resumed(
            ModelProfile::native(),
            vec![Seq(2)],
            4,
            open(MAX_STEPS_CEILING, ResumeAt::Call),
        );
        assert!(call_model(&a).is_none());
        assert_eq!(turn_end(&a).unwrap().outcome, Outcome::RanOut);
    }

    #[test]
    fn a_json_step_is_read_again_with_its_made_id() {
        let step = r#"{"thought":"change it","tool":"edit","input":{"title":"Hi"}}"#;
        let (_, a) = Harness::resumed(
            ModelProfile::json_object(true),
            vec![Seq(2), Seq(5)],
            7,
            open(1, at_answer(json(step), &[], &["s5-0"])),
        );
        let r = results(&a);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, "s5-0");
        assert!(r[0].1.contains(CUT_SHORT), "{r:?}");
        assert!(r[0].2);
    }

    #[test]
    fn the_log_lists_a_json_steps_call() {
        let mut h = Harness::json();
        h.send(user("Change the title."));
        let step = r#"{"thought":"change it","tool":"edit","input":{"title":"Hi"}}"#;
        let a = h.answer(json(step));
        let calls = appended(&a).into_iter().find_map(|r| match r {
            Record::AssistantMessage { calls, .. } => Some(calls),
            _ => None,
        });
        assert_eq!(
            calls,
            Some(vec![AskedCall {
                call: CallId::made(Seq(5), 0),
                tool: "edit".into(),
            }])
        );
    }

    #[test]
    fn a_refusal_whose_results_are_written_still_ends_refused() {
        let answer = native(None, &[("toolu_1", "look", "{}")], "refusal");
        let (_, a) = Harness::resumed(
            ModelProfile::native(),
            vec![Seq(2), Seq(5), Seq(6)],
            7,
            open(1, at_answer(answer, &["toolu_1"], &[])),
        );
        assert!(call_model(&a).is_none(), "{a:#?}");
        assert!(
            results(&a).is_empty(),
            "the written result is not written again"
        );
        assert_eq!(turn_end(&a).unwrap().outcome, Outcome::Refused);
    }

    #[test]
    fn a_repeated_call_id_still_fails_the_turn_after_a_restart() {
        let answer = native(
            None,
            &[("toolu_1", "look", "{}"), ("toolu_1", "edit", "{}")],
            "tool_use",
        );
        let (_, a) = Harness::resumed(
            ModelProfile::native(),
            vec![Seq(2), Seq(5), Seq(6)],
            7,
            open(1, at_answer(answer, &["toolu_1"], &[])),
        );
        assert!(call_model(&a).is_none() && run_tool(&a).is_none(), "{a:#?}");
        assert!(results(&a).is_empty());
        assert_eq!(turn_end(&a).unwrap().outcome, Outcome::Failed);
    }
}

#[test]
fn a_call_reseated_on_a_new_segment_is_made_again_after_it() {
    let mut h = Harness::native();
    let first = h.send(user("Go."));
    let at = h.calling;
    assert!(call_model(&first).is_some());
    // Refused for a call that is not the one waiting, or a seq not past it.
    assert_eq!(
        errors(&h.core.reseat(Seq(at.0 + 1), vec![], Seq(at.0 + 5))),
        vec![ErrorKind::InvalidRequest]
    );
    assert_eq!(
        errors(&h.core.reseat(at, vec![], at)),
        vec![ErrorKind::InvalidRequest]
    );
    // The driver appended a seed at `at`, `at + 1`: the call comes after it, from the new history.
    let again = h.core.reseat(at, vec![Seq(at.0)], Seq(at.0 + 2));
    let Some(Action::CallModel {
        at: again_at,
        recipe,
    }) = again.first()
    else {
        panic!("{again:#?}");
    };
    assert_eq!(*again_at, Seq(at.0 + 2));
    assert_eq!(recipe.messages, vec![Seq(at.0)]);
    assert_eq!(h.core.next_seq(), Seq(at.0 + 3));
    assert_eq!(h.core.phase(), Phase::Calling);
    // The answer to the call made again is taken.
    let answered = h.core.step(Input::ModelAnswered {
        at: *again_at,
        answer: native(Some("Done."), &[], "end_turn"),
    });
    assert!(turn_end(&answered).is_some(), "{answered:#?}");
}

#[test]
fn a_user_message_that_no_provider_takes_is_refused_when_sent() {
    for bad in [r#"{"text":"hi"}"#, "[]", r#"["hi"]"#, "7", "null"] {
        let mut h = Harness::native();
        let a = h.send(Input::Op(Op::UserMessage { content: raw(bad) }));
        assert_eq!(errors(&a), vec![ErrorKind::InvalidRequest], "{bad}");
        assert!(appended(&a).is_empty(), "{bad}");
        assert_eq!(h.core.phase(), Phase::Idle);
    }
    let mut h = Harness::native();
    let a = h.send(Input::Op(Op::UserMessage {
        content: raw(r#"[{"type":"text","text":"hi"}]"#),
    }));
    assert!(call_model(&a).is_some());
    // A steer is checked the same way, and the turn goes on without it.
    let a = h.send(Input::Op(Op::Steer {
        content: raw(r#"{"text":"red"}"#),
    }));
    assert_eq!(errors(&a), vec![ErrorKind::InvalidRequest]);
    assert_eq!(h.core.phase(), Phase::Calling);
}

#[test]
fn a_refused_json_step_that_names_a_call_still_answers_it() {
    let mut h = Harness::json();
    h.send(user("look"));
    let step = r#"{"thought":"t","tool":"look","input":{},"reply":""}"#;
    let view = ResponseView {
        stop_reason: Some("refusal".into()),
        ..ResponseView::default()
    };
    let a = h.answer(Answer::new(RawJson::string(step), view).with_text(step));
    let calls = appended(&a)
        .into_iter()
        .find_map(|r| match r {
            Record::AssistantMessage { calls, .. } => Some(calls),
            _ => None,
        })
        .unwrap();
    assert_eq!(calls.len(), 1, "the log lists the call");
    let r = results(&a);
    assert_eq!(r.len(), 1, "and answers it: {a:#?}");
    assert!(r[0].2, "as an error");
    assert_eq!(turn_end(&a).unwrap().outcome, Outcome::Refused);
}
