// SPDX-License-Identifier: Apache-2.0
//! A property over random runs: whatever the model answers and whatever the host does in between (verdicts,
//! decisions, steers, interrupts, failures), no model request is made, and no turn ends, while a tool call of an
//! accepted answer has no result; every call gets exactly one result; a write tool's start is durable before it runs;
//! and seqs are dense.

#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helpers outside #[test] functions"
)]

use std::collections::{BTreeMap, HashSet};

use proptest::prelude::*;
use sarathi_core::{
    Action, Answer, Core, CoreConfig, Durability, Effect, Input, Limits, ModelProfile, Record,
    Running, ToolDef, Verdict,
};
use sarathi_protocol::{
    BlockKind, BlockView, CallId, Decision, ErrorKind, Event, Op, RawJson, ResponseView, Seq,
    ToolOutput,
};

const NAMES: [&str; 4] = ["look", "edit", "paint", "draw"];
const STEER: &str = r#""steer""#;
const INPUTS: [&str; 3] = ["{}", r#"{"a":1}"#, "{}"];

fn tools() -> Vec<ToolDef> {
    let schema = RawJson::from_text(r#"{"type":"object"}"#).unwrap();
    vec![
        ToolDef::new("look", "Reads.", schema.clone(), Effect::Read),
        ToolDef::new("edit", "Writes.", schema, Effect::Write),
    ]
}

/// A choice stream: each draw takes the next byte (0 once it runs out).
struct Dice(Vec<u8>, usize);

impl Dice {
    fn roll(&mut self, n: u8) -> u8 {
        let b = self.0.get(self.1).copied().unwrap_or(0);
        self.1 += 1;
        b % n
    }
}

/// What the core waits for, read from its last actions.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Waiting {
    Idle,
    Model(Seq),
    Verdict(CallId),
    Decision(CallId),
    Tool(CallId),
    Stopping(CallId),
}

struct World {
    core: Core,
    json: bool,
    next: u64,
    waiting: Waiting,
    /// Calls of accepted answers, by id, and the results seen for each.
    calls: BTreeMap<CallId, u32>,
    started_durable: HashSet<CallId>,
    answers: u32,
    max_steps: u32,
    turn_calls: u32,
    steers_sent: usize,
    steers_added: usize,
    steers_back: usize,
    rebuilt: bool,
}

impl World {
    fn new(json: bool, max_steps: u32, cap: Option<u64>) -> Self {
        let profile = if json {
            ModelProfile::json_object(false)
        } else {
            ModelProfile::native()
        };
        let mut config = CoreConfig::new(profile, tools(), Seq(1));
        config.limits = Limits::with_max_steps(max_steps);
        config.limits.max_cost_micros = cap;
        World {
            core: Core::new(config),
            json,
            next: 2,
            waiting: Waiting::Idle,
            calls: BTreeMap::new(),
            started_durable: HashSet::new(),
            answers: 0,
            max_steps,
            turn_calls: 0,
            steers_sent: 0,
            steers_added: 0,
            steers_back: 0,
            rebuilt: false,
        }
    }

    fn unanswered(&self) -> Vec<&CallId> {
        self.calls
            .iter()
            .filter(|(_, n)| **n == 0)
            .map(|(c, _)| c)
            .collect()
    }

    fn send(&mut self, input: Input) {
        let actions = self.core.step(input);
        for a in actions {
            match a {
                Action::Append {
                    at,
                    records,
                    durability,
                } => {
                    assert_eq!(at.0, self.next, "dense seqs");
                    for r in &records {
                        match r {
                            Record::AssistantMessage { .. } => self.answers += 1,
                            Record::UserMessage { content, .. } if content.as_str() == STEER => {
                                self.steers_added += 1;
                            }
                            Record::ToolResult { call, .. } => {
                                let n =
                                    self.calls.get_mut(call).expect("a result for a known call");
                                *n += 1;
                                assert_eq!(*n, 1, "exactly one result for {call}");
                            }
                            Record::ToolStart {
                                call,
                                effect: Effect::Write,
                                ..
                            } if durability == Durability::Durable => {
                                self.started_durable.insert(call.clone());
                            }
                            _ => {}
                        }
                    }
                    self.next += records.len() as u64;
                }
                Action::CallModel { at, recipe } => {
                    assert_eq!(at.0, self.next);
                    self.next += 1;
                    assert!(
                        self.unanswered().is_empty(),
                        "a request with unanswered calls {:?}",
                        self.unanswered()
                    );
                    assert!(recipe.messages.windows(2).all(|w| w[0] < w[1]));
                    self.waiting = Waiting::Model(at);
                    self.turn_calls += 1;
                    assert!(
                        self.turn_calls <= self.max_steps,
                        "more calls than the step cap"
                    );
                }
                Action::Evaluate { call, .. } => self.waiting = Waiting::Verdict(call),
                Action::Emit(Event::PermissionAsked { call, .. }) => {
                    self.waiting = Waiting::Decision(call);
                }
                Action::RunTool { call, effect, .. } => {
                    if effect == Effect::Write {
                        assert!(
                            self.started_durable.contains(&call),
                            "durable start before {call}"
                        );
                    }
                    self.waiting = Waiting::Tool(call);
                }
                Action::Cancel(Running::Tool { call, .. }) => {
                    self.waiting = Waiting::Stopping(call)
                }
                Action::EndTurn(end) => {
                    assert!(
                        self.unanswered().is_empty(),
                        "a turn ended with unanswered calls"
                    );
                    self.steers_back += end.unsent.len();
                    self.turn_calls = 0;
                    self.waiting = Waiting::Idle;
                }
                _ => {}
            }
        }
        assert_eq!(self.core.next_seq(), Seq(self.next));
    }

    fn answer(&mut self, dice: &mut Dice, at: Seq) -> Input {
        // The seq the assistant message will get: after the response entry.
        let assistant = Seq(self.next + 1);
        if self.json {
            let text = match dice.roll(4) {
                0 => r#"{"thought":"t","tool":"done","reply":"ok"}"#.to_owned(),
                1 => "not json".to_owned(),
                _ => format!(
                    r#"{{"thought":"t","tool":"{}","input":{}}}"#,
                    NAMES[usize::from(dice.roll(4))],
                    INPUTS[usize::from(dice.roll(3))]
                ),
            };
            if let Ok(step) = serde_json::from_str::<serde_json::Value>(&text)
                && step["tool"] != "done"
            {
                self.calls.insert(CallId::made(assistant, 0), 0);
            }
            return Input::ModelAnswered {
                at,
                answer: Answer::new(RawJson::string(&text), ResponseView::default())
                    .with_text(text),
            };
        }
        let n = dice.roll(7);
        let stop = match dice.roll(8) {
            0 => "refusal",
            1 => "pause_turn",
            _ if n == 0 => "end_turn",
            _ => "tool_use",
        };
        let mut body = String::from(r#"{"content":["#);
        let mut blocks = Vec::new();
        for i in 0..n {
            if i > 0 {
                body.push(',');
            }
            let id = format!("toolu_{}_{i}", self.answers);
            let name = NAMES[usize::from(dice.roll(4))];
            let input = INPUTS[usize::from(dice.roll(3))];
            let start = u32::try_from(body.len()).unwrap();
            body.push_str(&format!(
                r#"{{"type":"tool_use","id":"{id}","name":"{name}","input":"#
            ));
            let i0 = u32::try_from(body.len()).unwrap();
            body.push_str(input);
            let i1 = u32::try_from(body.len()).unwrap() - u32::from(dice.roll(10) == 0);
            body.push('}');
            blocks.push(BlockView {
                kind: BlockKind::ToolUse {
                    id: id.as_str().into(),
                    name: name.into(),
                    input: i0..i1,
                },
                range: start..u32::try_from(body.len()).unwrap(),
            });
            self.calls.insert(id.into(), 0);
        }
        body.push_str("]}");
        let view = ResponseView {
            stop_reason: Some(stop.into()),
            blocks,
            ..ResponseView::default()
        };
        Input::ModelAnswered {
            at,
            answer: Answer::new(RawJson::from_text(&body).unwrap(), view).with_cost(10),
        }
    }

    /// The driver could not write something. Before a request is written, or at rest, the core goes on; with
    /// calls open it must be rebuilt, which ends this run. Returns whether the run goes on.
    fn abort(&mut self) -> bool {
        let next = match self.waiting {
            Waiting::Model(at) => at,
            _ => Seq(self.next),
        };
        let actions = self.core.step(Input::Aborted {
            kind: ErrorKind::Log,
            detail: "disk full".into(),
            next_seq: next,
        });
        assert!(
            actions
                .iter()
                .all(|a| matches!(a, Action::Emit(_) | Action::EndTurn(_))),
            "an abort appends nothing: {actions:?}"
        );
        for a in &actions {
            if let Action::EndTurn(end) = a {
                self.steers_back += end.unsent.len();
                self.turn_calls = 0;
            }
        }
        if self.core.needs_rebuild() {
            assert!(
                !matches!(self.waiting, Waiting::Idle | Waiting::Model(_)),
                "only open calls need a rebuild"
            );
            self.rebuilt = true;
            return false;
        }
        assert!(self.unanswered().is_empty());
        self.next = next.0;
        self.waiting = Waiting::Idle;
        true
    }

    /// An input for the wrong call or the wrong stage: refused, nothing changes.
    fn stray(&mut self, dice: &mut Dice) {
        let wrong: CallId = "stray".into();
        let wrong_seq = Seq(self.next + 100);
        let input = match dice.roll(6) {
            0 => Input::Verdict {
                call: wrong,
                verdict: Verdict::Allow,
            },
            1 => Input::ToolAbandoned { call: wrong },
            2 => Input::ToolDone {
                call: wrong,
                output: ToolOutput {
                    content: RawJson::null(),
                    is_error: false,
                },
            },
            3 => Input::Op(Op::ToolDecision {
                call: wrong,
                decision: Decision::AllowOnce,
            }),
            4 => Input::ModelFailed {
                at: wrong_seq,
                kind: ErrorKind::Network,
                detail: "late".into(),
            },
            _ => Input::ModelAnswered {
                at: wrong_seq,
                answer: Answer::new(RawJson::from_text("{}").unwrap(), ResponseView::default()),
            },
        };
        let (phase, next) = (self.core.phase(), self.core.next_seq());
        let actions = self.core.step(input);
        assert!(
            actions
                .iter()
                .all(|a| matches!(a, Action::Emit(Event::Error { .. }))),
            "a stray input only reports: {actions:?}"
        );
        assert_eq!((self.core.phase(), self.core.next_seq()), (phase, next));
    }

    fn act(&mut self, dice: &mut Dice) -> bool {
        // Now and then the host does something in between.
        match dice.roll(12) {
            0 => {
                self.send(Input::Op(Op::Interrupt));
                return true;
            }
            1 => {
                self.steers_sent += 1;
                self.send(Input::Op(Op::Steer {
                    content: RawJson::from_text(STEER).unwrap(),
                }));
                return true;
            }
            3 => {
                self.stray(dice);
                return true;
            }
            4 if dice.roll(4) == 0 => return self.abort(),
            2 => {
                self.send(Input::Op(Op::Shutdown));
                return true;
            }
            _ => {}
        }
        match self.waiting.clone() {
            Waiting::Idle => {
                if dice.roll(6) == 0 {
                    return false;
                }
                self.send(Input::Op(Op::UserMessage {
                    content: RawJson::string("go"),
                }));
            }
            Waiting::Model(at) => {
                let input = if dice.roll(10) == 0 {
                    Input::ModelFailed {
                        at,
                        kind: ErrorKind::Network,
                        detail: "cut".into(),
                    }
                } else {
                    self.answer(dice, at)
                };
                self.send(input);
            }
            Waiting::Verdict(call) => {
                let verdict = match dice.roll(3) {
                    0 => Verdict::Allow,
                    1 => Verdict::Ask,
                    _ => Verdict::Deny("no".into()),
                };
                self.send(Input::Verdict { call, verdict });
            }
            Waiting::Decision(call) => {
                let decision = match dice.roll(3) {
                    0 => Decision::AllowOnce,
                    1 => Decision::AllowAlways,
                    _ => Decision::Deny {
                        reason: "no".into(),
                    },
                };
                self.send(Input::Op(Op::ToolDecision { call, decision }));
            }
            Waiting::Tool(call) | Waiting::Stopping(call) => {
                let input = if matches!(self.waiting, Waiting::Stopping(_)) && dice.roll(2) == 0 {
                    Input::ToolAbandoned { call }
                } else {
                    let output = ToolOutput {
                        content: RawJson::string("out"),
                        is_error: dice.roll(4) == 0,
                    };
                    if dice.roll(2) == 0 {
                        Input::ToolDone { call, output }
                    } else {
                        Input::Op(Op::ToolOutput { call, output })
                    }
                };
                self.send(input);
            }
        }
        true
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn no_request_leaves_a_tool_call_unanswered(
        bytes in proptest::collection::vec(any::<u8>(), 0..400),
        json in any::<bool>(),
        max_steps in 1u32..6,
        cap in proptest::option::of(5u64..40),
    ) {
        let mut world = World::new(json, max_steps, cap);
        let mut dice = Dice(bytes, 0);
        for _ in 0..300 {
            if !world.act(&mut dice) {
                break;
            }
        }
        if world.rebuilt {
            // The log is closed by the rebuild, which is the driver's.
            return Ok(());
        }
        // Wherever the run stopped, an interrupt leaves no call open once the work in flight reports back.
        world.send(Input::Op(Op::Interrupt));
        if let Waiting::Stopping(call) = world.waiting.clone() {
            world.send(Input::ToolAbandoned { call });
        }
        prop_assert_eq!(world.waiting.clone(), Waiting::Idle);
        prop_assert!(world.unanswered().is_empty());
        prop_assert_eq!(
            world.steers_sent,
            world.steers_added + world.steers_back,
            "every steer is added or returned"
        );
    }
}
