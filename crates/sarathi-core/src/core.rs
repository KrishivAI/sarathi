// Derived from Vyasa `crates/ai/src/agent.rs` @ 2403be1 (MIT OR Apache-2.0); see PROVENANCE.md
// SPDX-License-Identifier: Apache-2.0
//! The run loop as a state machine that does no IO.
//!
//! [`Core::step`] takes one [`Input`] (a host command, a model answer, a permission verdict, a tool result) and
//! returns the [`Action`]s a driver carries out, in order: entries to append to the session log, the model call to
//! make, the tool to evaluate or run, a call to cancel, events for the host, the end of a turn. Every decision is made
//! here, so the loop is tested with plain inputs and outputs.
//!
//! The core numbers log entries itself: each [`Action::Append`] names the seq of its first record, and
//! [`Action::CallModel`] names the seq the driver writes the request at. The driver appends exactly what it is told,
//! at the seqs it is told, and fails the turn if the store disagrees.
//!
//! Rules the core keeps:
//! - every tool call of an answer gets a result before the next model request, including denied, skipped and
//!   interrupted calls (the Messages API refuses a request with an unanswered tool use);
//! - a model answer and the assistant message that accepts it are appended in one durable batch, and a write tool's
//!   start mark is durable before the tool runs, so after a crash a started write is never run again;
//! - a steer or a user message sent while a turn runs is held and added at the next tool boundary, after the step's
//!   last tool result and before the next request.

use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use sarathi_protocol::{
    CallId, Decision, ErrorKind, Event, Op, Outcome, RawJson, ResponseView, Seq, ToolOutput, TurnId,
};

use crate::profile::{Effect, ModelProfile, StepProtocol, ToolDef};
use crate::step::{self, JsonStep};

/// The most steps a turn may take, whatever the host asks for.
pub const MAX_STEPS_CEILING: u32 = 24;

/// The limits a turn runs under.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Limits {
    /// Model calls per turn; clamped to `1..=`[`MAX_STEPS_CEILING`].
    pub max_steps: u32,
    /// Failures in a row (an unusable answer, an unknown tool, a repeated call, an unreadable input) that end the
    /// turn, counted across steps. Progress resets the count: a call handed to the permission rules, a tool result
    /// of any kind, a paused answer that continues.
    pub max_failures: u32,
    /// Identical calls in a row (same tool, byte-equal input) at which the last is not run, whether or not the
    /// earlier ones ran (a denied call counts). Unknown and unreadable calls do not break a run of identical calls.
    pub repeat_limit: u32,
    /// The spend cap of a turn in micro-dollars, checked before every model call. An answer whose cost is not
    /// known counts as zero, so the cap does not hold for a model without prices. The driver's compaction calls
    /// count too ([`Core::charge`]): one inside a turn before the call it preceded, one before a turn from that
    /// turn's second call on.
    pub max_cost_micros: Option<u64>,
    /// How long an interrupted write tool is given to stop before it is abandoned.
    pub write_grace: Duration,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_steps: MAX_STEPS_CEILING,
            max_failures: 3,
            repeat_limit: 3,
            max_cost_micros: None,
            write_grace: Duration::from_secs(30),
        }
    }
}

impl Limits {
    /// The defaults with `max_steps` steps (clamped).
    pub fn with_max_steps(max_steps: u32) -> Self {
        Limits {
            max_steps,
            ..Limits::default()
        }
    }
}

/// What the core starts from: the profile, the tools, the limits, and where the session's log stands. The core
/// starts at rest, between turns; a turn cut short by a crash is closed by the driver before the core is made.
///
/// Tool names are matched exactly and the first definition of a name wins; under [`StepProtocol::JsonObject`] a
/// tool named [`crate::DONE_TOOL`] can never be called.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct CoreConfig {
    /// The model's profile.
    pub profile: ModelProfile,
    /// The tools the model may call.
    pub tools: Vec<ToolDef>,
    /// The limits of each turn.
    pub limits: Limits,
    /// The seq of the segment's `session.prefix` entry.
    pub prefix: Seq,
    /// The seqs of the segment's message slices so far (user messages, assistant messages, tool results), in order.
    pub history: Vec<Seq>,
    /// The seq the next appended entry gets.
    pub next_seq: Seq,
    /// Tools the host allowed for the rest of the session (`allow_always` decisions read back from the log).
    pub allowed_always: Vec<Box<str>>,
    /// The host's check on a finish, if it has one.
    pub completion: Option<Arc<dyn Completion>>,
}

impl CoreConfig {
    /// A fresh session whose prefix entry is at `prefix`, with default limits and an empty history.
    pub fn new(profile: ModelProfile, tools: Vec<ToolDef>, prefix: Seq) -> Self {
        CoreConfig {
            profile,
            tools,
            limits: Limits::default(),
            prefix,
            history: Vec::new(),
            next_seq: Seq(prefix.0 + 1),
            allowed_always: Vec::new(),
            completion: None,
        }
    }
}

/// A permission verdict on one call, as the rules give it: the tools crate's ([`sarathi_tools::Verdict`]); an ask is
/// sent to the host as [`Event::PermissionAsked`] and answered by [`Op::ToolDecision`].
pub use sarathi_tools::Verdict;

/// A model answer as the core reads it.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Answer {
    /// The answer body as stored; the view's ranges point into it.
    pub body: RawJson,
    /// What the answer holds.
    pub view: ResponseView,
    /// The answer's text, when the driver read it from a body the view does not describe (a translated
    /// protocol). `None`: the core joins the view's `text` blocks.
    pub text: Option<Box<str>>,
    /// The call's cost in micro-dollars, when known.
    pub cost_micros: Option<u64>,
}

impl Answer {
    /// An answer with a view of `body`.
    pub fn new(body: RawJson, view: ResponseView) -> Self {
        Answer {
            body,
            view,
            text: None,
            cost_micros: None,
        }
    }

    /// The same answer with its text given.
    #[must_use]
    pub fn with_text(mut self, text: impl Into<Box<str>>) -> Self {
        self.text = Some(text.into());
        self
    }

    /// The same answer with its cost.
    #[must_use]
    pub fn with_cost(mut self, cost_micros: u64) -> Self {
        self.cost_micros = Some(cost_micros);
        self
    }
}

/// A host's check on a finish: the model stopped calling tools and asks to end the turn.
///
/// Called by the core (no IO, no waiting) when a turn would end [`Outcome::Finished`]. `Some(text)` refuses: `text`
/// goes to the model as a message and the turn goes on; at most [`MAX_COMPLETION_REFUSALS`] in a turn, after which
/// the turn ends [`Outcome::RanOut`]. The check reads state the host's tools keep (see
/// [`crate::toolbox::ToolboxTools::with_gate`]); it is not in the log, so a resumed session starts it afresh.
pub trait Completion: std::fmt::Debug + Send + Sync {
    /// `None` lets the turn finish.
    fn check(&self) -> Option<String>;
}

/// How many times in one turn a [`Completion`] may send the model back before the turn ends [`Outcome::RanOut`].
pub const MAX_COMPLETION_REFUSALS: u32 = 3;

/// What the core is told.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Input {
    /// A host command.
    Op(Op),
    /// The model call answered.
    ModelAnswered {
        /// The seq of the call's request ([`Action::CallModel`]'s `at`).
        at: Seq,
        /// The answer.
        answer: Answer,
    },
    /// The model call failed (after the provider's retries and failover). A spend cap reached
    /// ([`ErrorKind::BudgetExceeded`]) ends the turn [`Outcome::RanOut`]; any other failure, [`Outcome::Failed`].
    ModelFailed {
        /// The seq of the call's request.
        at: Seq,
        /// The kind of failure.
        kind: ErrorKind,
        /// A readable detail, holding no credential.
        detail: Box<str>,
    },
    /// The permission verdict on the call being evaluated.
    Verdict {
        /// The call.
        call: CallId,
        /// The verdict.
        verdict: Verdict,
    },
    /// A tool returned.
    ToolDone {
        /// The call.
        call: CallId,
        /// Its result, rendered for the session's protocol.
        output: ToolOutput,
    },
    /// An action could not be carried out (an append or the request could not be written, the request could not
    /// be built). The turn is dropped without appending anything more; the core goes on from `next_seq`, the seq
    /// the store will give its next entry, and forgets history slices at or after it. Only an abort at rest at the
    /// core's own next seq, or before the request was written (`next_seq` is the request's seq), is recoverable;
    /// any other leaves the core needing a rebuild from the log ([`Core::needs_rebuild`]).
    Aborted {
        /// The kind of failure.
        kind: ErrorKind,
        /// A readable detail.
        detail: Box<str>,
        /// The store's next seq.
        next_seq: Seq,
    },
    /// A cancelled tool was dropped without a result (at once for a read tool, after the grace for a write tool).
    ToolAbandoned {
        /// The call.
        call: CallId,
    },
}

/// How soon an append must be on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Durability {
    /// Written; made durable by the next durable append.
    Buffered,
    /// On disk before the driver goes on.
    Durable,
}

/// An entry the core asks the driver to append, by its meaning; the driver writes it in the log's form.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Record {
    /// `message.user`: a user message, a steer, or a correction the core wrote for the model.
    UserMessage {
        /// The message content, rendered once.
        content: RawJson,
        /// Added during a turn (a steer, a message held until the tool boundary, a correction), not the message
        /// that starts one.
        steer: bool,
    },
    /// `model.response`: the answer to the call just made, as the driver received it (the driver holds it).
    ModelResponse,
    /// `model.failure`: the call just made failed or was cancelled.
    ModelFailure {
        /// The kind of failure.
        kind: ErrorKind,
        /// A readable detail.
        detail: Box<str>,
    },
    /// `message.assistant`: the response at `from` becomes an assistant slice.
    AssistantMessage {
        /// The `model.response` entry.
        from: Seq,
        /// The tool calls the step asks for, in order: each gets a `tool.result` before the next request.
        calls: Vec<AskedCall>,
    },
    /// `tool.ask`: the host is asked whether a call may run.
    ToolAsk {
        /// The call.
        call: CallId,
        /// The tool.
        tool: Box<str>,
        /// Its input.
        input: RawJson,
    },
    /// `tool.decision`: the host's answer.
    ToolDecision {
        /// The call.
        call: CallId,
        /// The answer.
        decision: Decision,
    },
    /// `tool.start`: a call is about to run.
    ToolStart {
        /// The call.
        call: CallId,
        /// The tool.
        tool: Box<str>,
        /// Its effect.
        effect: Effect,
    },
    /// `tool.result`: a call's result, rendered once. Under [`StepProtocol::JsonObject`] the content is the text of
    /// a user message (`observation:` ...), which the request assembler sends as one.
    ToolResult {
        /// The call.
        call: CallId,
        /// The result content.
        content: RawJson,
        /// The result reports a failure.
        is_error: bool,
    },
    /// `turn.end`.
    TurnEnd {
        /// How the turn ended.
        outcome: Outcome,
    },
}

/// A tool call an accepted answer asks for, as the log lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AskedCall {
    /// The call id: the provider's, or one the core made ([`CallId::made`]).
    pub call: CallId,
    /// The tool's name as the model wrote it.
    pub tool: Box<str>,
}

/// A turn a restart cut short, as the driver read it from the log ([`Core::resume`]).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenTurn {
    /// The turn: the seq of the user message that started it.
    pub turn: TurnId,
    /// The answers the turn accepted so far.
    pub steps: u32,
    /// Where it stopped.
    pub at: ResumeAt,
}

impl OpenTurn {
    /// The turn started at `turn`, with `steps` answers accepted, going on from `at`.
    pub fn new(turn: TurnId, steps: u32, at: ResumeAt) -> Self {
        OpenTurn { turn, steps, at }
    }
}

/// Where a cut-short turn goes on from.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResumeAt {
    /// Make the next model call: a request was written with no answer, or messages were added after the last one.
    Call,
    /// The answer accepted at `assistant` is worked again: calls in `answered` already have a result and are
    /// skipped, calls in `cut_short` started a side-effecting tool and get the cut-short result without running,
    /// and the rest go through the permission rules and run (a read that started runs again).
    Answer {
        /// The `message.assistant` entry.
        assistant: Seq,
        /// The answer, read back from its `model.response`.
        answer: Box<Answer>,
        /// Calls with a result.
        answered: Vec<CallId>,
        /// Side-effecting calls that started and have no result.
        cut_short: Vec<CallId>,
    },
}

/// The result text of a side-effecting call that started before a restart and has no result (the log's).
pub use sarathi_log::CUT_SHORT;

/// How a request is built from the log: the prefix entry and the message slices, in order.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recipe {
    /// The `session.prefix` entry.
    pub prefix: Seq,
    /// The message slices.
    pub messages: Vec<Seq>,
}

/// Work in flight that an interrupt cancels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Running {
    /// The model call whose request is at `at`: drop it; nothing of it is stored.
    Model {
        /// The request entry.
        at: Seq,
    },
    /// A tool call. `grace: None`: drop it now. `Some(d)`: signal it to stop and give it `d` to return. Either way,
    /// report [`Input::ToolDone`] if it returned and [`Input::ToolAbandoned`] if it was dropped.
    Tool {
        /// The call.
        call: CallId,
        /// How long it may take to stop.
        grace: Option<Duration>,
    },
}

/// How a turn ended, for the host.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnEnd {
    /// The turn.
    pub turn: TurnId,
    /// How it ended.
    pub outcome: Outcome,
    /// The closing text: the model's final reply, or the core's when a guard stopped the turn.
    pub reply: Option<Box<str>>,
    /// Steers and user messages that arrived during the turn and were not added to it; the host may send them again.
    pub unsent: Vec<RawJson>,
}

impl TurnEnd {
    /// A turn that failed with no reply and nothing held.
    pub fn failed(turn: TurnId) -> Self {
        TurnEnd {
            turn,
            outcome: Outcome::Failed,
            reply: None,
            unsent: Vec::new(),
        }
    }
}

/// What the driver does next.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Append `records` at seqs `at`, `at + 1`, ...
    Append {
        /// The seq of the first record.
        at: Seq,
        /// The records, in order.
        records: Vec<Record>,
        /// How soon they must be on disk.
        durability: Durability,
    },
    /// Build the request from `recipe`, append it as `model.request` at `at` (buffered), and make the call; answer
    /// with [`Input::ModelAnswered`] or [`Input::ModelFailed`] naming `at`, or [`Input::Aborted`] if the request
    /// was not written. A call cancelled by [`Action::Cancel`] is answered with nothing.
    CallModel {
        /// The seq of the request entry.
        at: Seq,
        /// How to build it.
        recipe: Recipe,
    },
    /// Ask the permission rules about a call; answer with [`Input::Verdict`]. An interrupt while the verdict or the
    /// host's decision is pending answers the call itself and emits no [`Action::Cancel`]: a verdict or decision
    /// that arrives later is refused with an [`ErrorKind::InvalidRequest`] event, which a host may ignore.
    Evaluate {
        /// The call.
        call: CallId,
        /// The tool.
        tool: Box<str>,
        /// Its input.
        input: RawJson,
    },
    /// Run a call; answer with [`Input::ToolDone`].
    RunTool {
        /// The call.
        call: CallId,
        /// The tool.
        tool: Box<str>,
        /// Its input.
        input: RawJson,
        /// Its effect.
        effect: Effect,
    },
    /// Cancel work in flight.
    Cancel(Running),
    /// Report to the host.
    Emit(Event),
    /// The turn ended.
    EndTurn(TurnEnd),
}

/// Where the core is, for a host or a test.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase {
    /// No turn is running.
    Idle,
    /// A model call is in flight.
    Calling,
    /// A call's permission is being evaluated.
    Deciding,
    /// The host was asked whether a call may run.
    Asking,
    /// A tool is running.
    Running,
    /// An interrupted tool is being stopped.
    Stopping,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CallKind {
    Known(Effect),
    Unknown,
    Unreadable,
    /// Started before a restart, side-effecting, no result: answered without running.
    CutShort,
}

#[derive(Debug, Clone)]
struct Call {
    id: CallId,
    tool: Box<str>,
    input: RawJson,
    kind: CallKind,
}

/// An answer's step, read once.
#[derive(Debug)]
enum ReadStep {
    Native(VecDeque<Call>),
    Json(Result<JsonRead, String>),
    /// Every call of the step was answered before a restart: go on to the next request.
    Answered,
}

#[derive(Debug)]
enum JsonRead {
    Done(Box<str>),
    Call(Call),
}

/// Calls a restart found answered, or started as side-effecting and not answered.
#[derive(Debug, Default)]
struct Done {
    answered: HashSet<CallId>,
    cut_short: HashSet<CallId>,
}

impl ReadStep {
    /// The calls the log lists for the step.
    fn asked(&self) -> Vec<AskedCall> {
        let asked = |c: &Call| AskedCall {
            call: c.id.clone(),
            tool: c.tool.clone(),
        };
        match self {
            ReadStep::Native(calls) => calls.iter().map(asked).collect(),
            ReadStep::Json(Ok(JsonRead::Call(c))) => vec![asked(c)],
            ReadStep::Json(_) | ReadStep::Answered => Vec::new(),
        }
    }

    /// The step with the calls in `done` taken out or marked cut short.
    fn marked(self, done: &Done) -> ReadStep {
        if done.answered.is_empty() && done.cut_short.is_empty() {
            return self;
        }
        let mark = |mut c: Call| {
            if done.answered.contains(&c.id) {
                return None;
            }
            if done.cut_short.contains(&c.id) {
                c.kind = CallKind::CutShort;
            }
            Some(c)
        };
        match self {
            ReadStep::Native(calls) => {
                let left: VecDeque<Call> = calls.into_iter().filter_map(mark).collect();
                if left.is_empty() {
                    ReadStep::Answered
                } else {
                    ReadStep::Native(left)
                }
            }
            ReadStep::Json(Ok(JsonRead::Call(c))) => match mark(c) {
                Some(c) => ReadStep::Json(Ok(JsonRead::Call(c))),
                None => ReadStep::Answered,
            },
            other => other,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CallState {
    Evaluating,
    Asking,
    Running(Effect),
    Stopping { effect: Effect, hard: bool },
}

#[derive(Debug)]
enum Stage {
    Calling {
        at: Seq,
    },
    Tools {
        calls: VecDeque<Call>,
        current: Option<(Call, CallState)>,
    },
}

#[derive(Debug)]
struct Turn {
    id: TurnId,
    steps: u32,
    spent_micros: u64,
    queued: Vec<RawJson>,
    shutdown: bool,
    /// Finishes the [`Completion`] sent back this turn.
    refused: u32,
    stage: Stage,
}

/// The run loop's state machine. See the module documentation.
#[derive(Debug)]
pub struct Core {
    profile: ModelProfile,
    tools: Vec<ToolDef>,
    limits: Limits,
    prefix: Seq,
    history: Vec<Seq>,
    next_seq: Seq,
    failures: u32,
    recent: VecDeque<(Box<str>, RawJson)>,
    allowed_always: HashSet<Box<str>>,
    turn: Option<Turn>,
    broken: bool,
    completion: Option<Arc<dyn Completion>>,
}

impl Core {
    /// A core at rest.
    pub fn new(config: CoreConfig) -> Self {
        let mut limits = config.limits;
        limits.max_steps = limits.max_steps.clamp(1, MAX_STEPS_CEILING);
        limits.max_failures = limits.max_failures.max(1);
        limits.repeat_limit = limits.repeat_limit.max(2);
        Core {
            profile: config.profile,
            tools: config.tools,
            limits,
            prefix: config.prefix,
            history: config.history,
            next_seq: config.next_seq,
            failures: 0,
            recent: VecDeque::new(),
            allowed_always: config.allowed_always.into_iter().collect(),
            turn: None,
            broken: false,
            completion: config.completion,
        }
    }

    /// A core in the middle of a turn a restart cut short, and what to do first. The config is as for
    /// [`Core::new`], with `next_seq` the log's next seq and `history` the segment's slices so far. The turn's spend
    /// and its count of failures in a row start again from zero.
    pub fn resume(config: CoreConfig, open: OpenTurn) -> (Core, Vec<Action>) {
        let mut core = Core::new(config);
        let mut out = Vec::new();
        core.turn = Some(Turn {
            id: open.turn,
            steps: open.steps,
            spent_micros: 0,
            queued: Vec::new(),
            shutdown: false,
            refused: 0,
            stage: Stage::Calling { at: core.next_seq },
        });
        out.push(Action::Emit(Event::TurnStarted { turn: open.turn }));
        match open.at {
            ResumeAt::Call => core.next_call(&mut out),
            ResumeAt::Answer {
                assistant,
                answer,
                answered,
                cut_short,
            } => {
                let read = core.read_step(&answer, assistant);
                let done = Done {
                    answered: answered.into_iter().collect(),
                    cut_short: cut_short.into_iter().collect(),
                };
                core.work(&mut out, &answer, read, &done);
            }
        }
        (core, out)
    }

    /// Where the core is.
    pub fn phase(&self) -> Phase {
        match &self.turn {
            None => Phase::Idle,
            Some(t) => match &t.stage {
                Stage::Calling { .. } => Phase::Calling,
                Stage::Tools { current, .. } => match current {
                    None | Some((_, CallState::Evaluating)) => Phase::Deciding,
                    Some((_, CallState::Asking)) => Phase::Asking,
                    Some((_, CallState::Running(_))) => Phase::Running,
                    Some((_, CallState::Stopping { .. })) => Phase::Stopping,
                },
            },
        }
    }

    /// The seq the next appended entry gets.
    pub fn next_seq(&self) -> Seq {
        self.next_seq
    }

    /// The segment's message slices so far.
    pub fn history(&self) -> &[Seq] {
        &self.history
    }

    /// Whether an aborted turn left the log with tool calls the core cannot answer: every input is then refused,
    /// and the driver makes a new core from the log (which closes the open calls first).
    pub fn needs_rebuild(&self) -> bool {
        self.broken
    }

    /// The driver began a new segment before making the call the core asked for at `at` (a compaction or the
    /// overflow fallback, whose entries the driver appended itself at `at` and after): the core takes the new
    /// segment's slices as its `history` and asks for the call again at `next_seq`, the log's next seq. The turn's
    /// steps, spend and held messages carry on. Refused unless the core waits for the call at `at` and `next_seq`
    /// is past it.
    ///
    /// The spend cap is checked again first, so a compaction call [`Core::charge`]d to the turn counts: at the cap,
    /// the turn ends [`Outcome::RanOut`] instead of making the call.
    pub fn reseat(&mut self, at: Seq, history: Vec<Seq>, next_seq: Seq) -> Vec<Action> {
        let mut out = Vec::new();
        let calling = matches!(
            &self.turn,
            Some(Turn { stage: Stage::Calling { at: a }, .. }) if *a == at
        );
        if self.broken || !calling || next_seq <= at {
            refuse(
                &mut out,
                &format!("no model call at {at} waits to be made again at {next_seq}"),
            );
            return out;
        }
        self.history = history;
        self.next_seq = next_seq;
        if self.cap_reached(&mut out) {
            return out;
        }
        let Some(t) = &mut self.turn else {
            return out;
        };
        let at = next_seq;
        self.next_seq = Seq(at.0 + 1);
        t.stage = Stage::Calling { at };
        out.push(Action::CallModel {
            at,
            recipe: Recipe {
                prefix: self.prefix,
                messages: self.history.clone(),
            },
        });
        out
    }

    /// Adds a model call the driver made for the running turn outside the core (a compaction call) to the turn's
    /// spend; it counts against [`Limits::max_cost_micros`] from the next model call on. Nothing happens with no turn
    /// running.
    pub fn charge(&mut self, cost_micros: u64) {
        if let Some(t) = &mut self.turn {
            t.spent_micros = t.spent_micros.saturating_add(cost_micros);
        }
    }

    /// At the spend cap, ends the turn [`Outcome::RanOut`] and says so.
    fn cap_reached(&mut self, out: &mut Vec<Action>) -> bool {
        let Some(t) = &self.turn else { return false };
        match self.limits.max_cost_micros {
            Some(cap) if t.spent_micros >= cap => {
                out.push(Action::Emit(Event::Error {
                    kind: ErrorKind::BudgetExceeded,
                    detail: format!("the turn's spend cap of {cap} micro-dollars is reached")
                        .into(),
                }));
                self.end(out, Outcome::RanOut, None);
                true
            }
            _ => false,
        }
    }

    /// The limits in force (clamped).
    pub fn limits(&self) -> &Limits {
        &self.limits
    }

    /// Takes one input and returns what to do, in order.
    pub fn step(&mut self, input: Input) -> Vec<Action> {
        let mut out = Vec::new();
        if self.broken {
            out.push(Action::Emit(Event::Error {
                kind: ErrorKind::Internal,
                detail: "the core must be rebuilt from the log after an aborted turn".into(),
            }));
            return out;
        }
        match input {
            Input::Op(op) => self.on_op(&mut out, op),
            Input::ModelAnswered { at, answer } => self.on_answer(&mut out, at, answer),
            Input::ModelFailed { at, kind, detail } => {
                self.on_model_failed(&mut out, at, kind, detail);
            }
            Input::Aborted {
                kind,
                detail,
                next_seq,
            } => self.on_aborted(&mut out, kind, detail, next_seq),
            Input::Verdict { call, verdict } => self.on_verdict(&mut out, &call, verdict),
            Input::ToolDone { call, output } => self.on_tool_done(&mut out, &call, &output),
            Input::ToolAbandoned { call } => self.on_tool_abandoned(&mut out, &call),
        }
        out
    }

    fn on_op(&mut self, out: &mut Vec<Action>, op: Op) {
        match op {
            Op::UserMessage { content } | Op::Steer { content } if !user_content(&content) => {
                // Taken, it would be in the log for good and fail every later request that groups it.
                refuse(
                    out,
                    "a user message's content is a string or a non-empty array of content blocks",
                );
            }
            Op::UserMessage { content } | Op::Steer { content } => match &mut self.turn {
                None => self.start_turn(out, content),
                Some(t) => t.queued.push(content),
            },
            Op::Interrupt => self.on_interrupt(out),
            Op::ToolDecision { call, decision } => self.on_decision(out, &call, decision),
            Op::ToolOutput { call, output } => self.on_tool_done(out, &call, &output),
            Op::Shutdown => {
                if let Some(t) = &mut self.turn {
                    t.shutdown = true;
                }
                // A host that is shutting down may never answer an ask.
                if self.phase() == Phase::Asking {
                    self.on_interrupt(out);
                }
            }
            other => refuse(out, &format!("the run loop does not take {other:?}")),
        }
    }

    fn start_turn(&mut self, out: &mut Vec<Action>, content: RawJson) {
        self.failures = 0;
        self.recent.clear();
        // The turn is the user message's seq. It is reported before the message is written, so a host that hears
        // the turn end (a failed write) has heard it start.
        let id = TurnId(self.next_seq);
        out.push(Action::Emit(Event::TurnStarted { turn: id }));
        let at = self.append(
            out,
            vec![Record::UserMessage {
                content,
                steer: false,
            }],
            Durability::Buffered,
        );
        self.turn = Some(Turn {
            id,
            steps: 0,
            spent_micros: 0,
            queued: Vec::new(),
            shutdown: false,
            refused: 0,
            // Replaced by `next_call` at once.
            stage: Stage::Calling { at },
        });
        self.next_call(out);
    }

    /// The tool boundary: add held messages, then make the next call, unless a guard or a shutdown ends the turn.
    fn next_call(&mut self, out: &mut Vec<Action>) {
        let Some(t) = &mut self.turn else { return };
        if t.shutdown {
            return self.end(out, Outcome::Interrupted, None);
        }
        if !t.queued.is_empty() {
            let records = t
                .queued
                .drain(..)
                .map(|content| Record::UserMessage {
                    content,
                    steer: true,
                })
                .collect();
            self.append(out, records, Durability::Buffered);
        }
        let Some(t) = &mut self.turn else { return };
        if t.steps >= self.limits.max_steps {
            let reply = step::ran_out(self.limits.max_steps);
            return self.end(out, Outcome::RanOut, Some(reply.into()));
        }
        if self.cap_reached(out) {
            return;
        }
        let Some(t) = &mut self.turn else { return };
        let at = self.next_seq;
        self.next_seq = Seq(at.0 + 1);
        t.stage = Stage::Calling { at };
        out.push(Action::CallModel {
            at,
            recipe: Recipe {
                prefix: self.prefix,
                messages: self.history.clone(),
            },
        });
    }

    fn on_answer(&mut self, out: &mut Vec<Action>, at: Seq, answer: Answer) {
        let Some(t) = &mut self.turn else {
            return refuse(out, "a model answer arrived with no turn running");
        };
        if !matches!(t.stage, Stage::Calling { at: a } if a == at) {
            return refuse(
                out,
                &format!("no model call with request {at} is in flight"),
            );
        }
        t.steps += 1;
        t.spent_micros = t
            .spent_micros
            .saturating_add(answer.cost_micros.unwrap_or(0));
        let response = self.next_seq;
        let assistant = Seq(response.0 + 1);
        let read = self.read_step(&answer, assistant);
        self.append(
            out,
            vec![
                Record::ModelResponse,
                Record::AssistantMessage {
                    from: response,
                    calls: read.asked(),
                },
            ],
            Durability::Durable,
        );
        out.push(Action::Emit(Event::AssistantMessage {
            seq: assistant,
            view: answer.view.clone(),
        }));
        out.push(Action::Emit(Event::Usage {
            seq: response,
            usage: answer.view.usage,
            cost_micros: answer.cost_micros,
        }));
        self.work(out, &answer, read, &Done::default());
    }

    /// Reads the step an answer holds, once: its calls are listed in the log before any of them is worked.
    fn read_step(&self, answer: &Answer, assistant: Seq) -> ReadStep {
        match self.profile.protocol {
            StepProtocol::Native => ReadStep::Native(self.native_calls(answer)),
            StepProtocol::JsonObject => {
                let text = answer
                    .text
                    .as_deref()
                    .map(str::to_owned)
                    .or_else(|| step::native_text(&answer.body, &answer.view))
                    .unwrap_or_default();
                ReadStep::Json(step::read_json_step(&text).map(|s| match s {
                    JsonStep::Done { reply } => JsonRead::Done(reply),
                    JsonStep::Call { tool, input } => {
                        let kind = match self.tool(&tool) {
                            Some(def) => CallKind::Known(def.effect),
                            None => CallKind::Unknown,
                        };
                        JsonRead::Call(Call {
                            id: CallId::made(assistant, 0),
                            tool,
                            input,
                            kind,
                        })
                    }
                }))
            }
        }
    }

    /// Works an accepted answer: ends the turn, or starts on its calls. `done` names calls a restart found
    /// answered or cut short.
    ///
    /// A refusal and an answer that repeats a call id end the turn whatever a restart found answered: they are
    /// decided on the whole step, and only the calls still without a result are answered.
    fn work(&mut self, out: &mut Vec<Action>, answer: &Answer, read: ReadStep, done: &Done) {
        let unanswered = |calls: VecDeque<Call>| -> VecDeque<Call> {
            let mut seen = HashSet::new();
            calls
                .into_iter()
                .filter(|c| seen.insert(c.id.clone()) && !done.answered.contains(&c.id))
                .collect()
        };
        if answer.view.stop_reason.as_deref() == Some("refusal") {
            // The log lists the step's calls with the answer: each gets a result, so none is left open.
            let calls = match read {
                ReadStep::Native(calls) => calls,
                ReadStep::Json(Ok(JsonRead::Call(call))) => VecDeque::from([call]),
                ReadStep::Json(_) | ReadStep::Answered => VecDeque::new(),
            };
            self.set_tools(unanswered(calls));
            self.skip_rest(out, "the model refused");
            return self.end(out, Outcome::Refused, None);
        }
        if let ReadStep::Native(calls) = &read {
            let mut seen = HashSet::new();
            if !calls.iter().all(|c| seen.insert(c.id.clone())) {
                // Results could not be told apart: answer each id once and stop.
                out.push(Action::Emit(Event::Error {
                    kind: ErrorKind::Malformed,
                    detail: "the answer has two tool calls with one id".into(),
                }));
                let ReadStep::Native(calls) = read else {
                    return;
                };
                self.set_tools(unanswered(calls));
                self.skip_rest(out, "the answer repeated a call id");
                return self.end(out, Outcome::Failed, None);
            }
        }
        match read.marked(done) {
            ReadStep::Json(r) => self.json_step(out, r),
            ReadStep::Native(calls) => self.native_step(out, answer, calls),
            ReadStep::Answered => {
                self.set_tools(VecDeque::new());
                self.advance(out);
            }
        }
    }

    fn native_calls(&self, answer: &Answer) -> VecDeque<Call> {
        answer
            .view
            .tool_uses()
            .map(|(id, name, range)| {
                let input = step::native_input(&answer.body, range);
                let kind = match (&input, self.tool(name)) {
                    (None, _) => CallKind::Unreadable,
                    (Some(_), None) => CallKind::Unknown,
                    (Some(_), Some(def)) => CallKind::Known(def.effect),
                };
                Call {
                    id: id.into(),
                    tool: name.into(),
                    input: input.unwrap_or_else(RawJson::null),
                    kind,
                }
            })
            .collect()
    }

    fn native_step(&mut self, out: &mut Vec<Action>, answer: &Answer, calls: VecDeque<Call>) {
        let stop = answer.view.stop_reason.as_deref();
        if calls.is_empty() {
            if stop == Some("pause_turn") {
                self.failures = 0;
                return self.next_call(out);
            }
            let reply = answer
                .text
                .clone()
                .or_else(|| step::native_text(&answer.body, &answer.view).map(Into::into))
                .filter(|t| !t.is_empty());
            return self.finish(out, reply);
        }
        self.set_tools(calls);
        self.advance(out);
    }

    fn json_step(&mut self, out: &mut Vec<Action>, read: Result<JsonRead, String>) {
        match read {
            Err(reason) => {
                if self.fail(out, &format!("the model's reply was unusable: {reason}")) {
                    return;
                }
                let content = RawJson::string(&step::unusable_reply(&reason));
                self.append(
                    out,
                    vec![Record::UserMessage {
                        content,
                        steer: true,
                    }],
                    Durability::Buffered,
                );
                self.next_call(out);
            }
            Ok(JsonRead::Done(reply)) => self.finish(out, Some(reply)),
            Ok(JsonRead::Call(call)) => {
                self.set_tools(VecDeque::from([call]));
                self.advance(out);
            }
        }
    }

    /// The model finished; a message held during the turn keeps it going if another call is allowed, else it goes
    /// back to the host in [`TurnEnd::unsent`].
    fn finish(&mut self, out: &mut Vec<Action>, reply: Option<Box<str>>) {
        let held = self.turn.as_ref().is_some_and(|t| {
            !t.queued.is_empty()
                && !t.shutdown
                && t.steps < self.limits.max_steps
                && self
                    .limits
                    .max_cost_micros
                    .is_none_or(|cap| t.spent_micros < cap)
        });
        if held {
            self.failures = 0;
            self.next_call(out);
            return;
        }
        if let Some(text) = self.completion_refusal() {
            let Some(t) = self.turn.as_mut() else { return };
            t.refused += 1;
            if t.refused > MAX_COMPLETION_REFUSALS {
                self.end(out, Outcome::RanOut, reply);
                return;
            }
            self.failures = 0;
            self.append(
                out,
                vec![Record::UserMessage {
                    content: RawJson::string(&text),
                    steer: true,
                }],
                Durability::Buffered,
            );
            self.next_call(out);
        } else {
            self.end(out, Outcome::Finished, reply);
        }
    }

    /// The host's refusal of this finish, unless the turn is stopping anyway (shutdown, step or spend cap).
    fn completion_refusal(&self) -> Option<String> {
        let t = self.turn.as_ref()?;
        let stopping = t.shutdown
            || t.steps >= self.limits.max_steps
            || self
                .limits
                .max_cost_micros
                .is_some_and(|cap| t.spent_micros >= cap);
        if stopping {
            return None;
        }
        self.completion.as_ref()?.check()
    }

    fn set_tools(&mut self, calls: VecDeque<Call>) {
        if let Some(t) = &mut self.turn {
            t.stage = Stage::Tools {
                calls,
                current: None,
            };
        }
    }

    /// Takes the next call of the step; when none is left, goes to the next model call.
    fn advance(&mut self, out: &mut Vec<Action>) {
        loop {
            let Some(Turn {
                stage: Stage::Tools { calls, current },
                ..
            }) = &mut self.turn
            else {
                return;
            };
            if current.is_some() {
                return;
            }
            let Some(call) = calls.pop_front() else {
                return self.next_call(out);
            };
            match call.kind {
                CallKind::CutShort => {
                    self.core_result(out, &call, CUT_SHORT, Durability::Durable);
                }
                CallKind::Unreadable => {
                    let text = "the call's input could not be read as JSON";
                    self.core_result(out, &call, text, Durability::Buffered);
                    if self.fail(out, text) {
                        return;
                    }
                }
                CallKind::Unknown => {
                    let json = self.profile.protocol == StepProtocol::JsonObject;
                    let names = self.tools.iter().map(|t| &*t.name);
                    let text = step::unknown_tool(&call.tool, names, json);
                    self.core_result(out, &call, &text, Durability::Buffered);
                    if self.fail(
                        out,
                        &format!("the model called an unknown tool, {:?}", call.tool),
                    ) {
                        return;
                    }
                }
                CallKind::Known(_) => {
                    let repeated = self.recent.len() + 1 >= self.limits.repeat_limit as usize
                        && self
                            .recent
                            .iter()
                            .rev()
                            .take(self.limits.repeat_limit as usize - 1)
                            .all(|(tool, input)| *tool == call.tool && *input == call.input);
                    self.recent
                        .push_back((call.tool.clone(), call.input.clone()));
                    while self.recent.len() > self.limits.repeat_limit as usize {
                        self.recent.pop_front();
                    }
                    if repeated {
                        self.core_result(out, &call, step::REPEATED_CALL, Durability::Buffered);
                        if self.fail(
                            out,
                            &format!("the model repeated a call to {:?}", call.tool),
                        ) {
                            return;
                        }
                        continue;
                    }
                    self.failures = 0;
                    out.push(Action::Evaluate {
                        call: call.id.clone(),
                        tool: call.tool.clone(),
                        input: call.input.clone(),
                    });
                    *current = Some((call, CallState::Evaluating));
                    return;
                }
            }
        }
    }

    /// Counts a failure; at the limit, skips the step's other calls and ends the turn. Returns whether it ended.
    fn fail(&mut self, out: &mut Vec<Action>, detail: &str) -> bool {
        self.failures += 1;
        if self.failures < self.limits.max_failures {
            return false;
        }
        out.push(Action::Emit(Event::Error {
            kind: ErrorKind::Guard,
            detail: format!("{} failures in a row; the last: {detail}", self.failures).into(),
        }));
        self.skip_rest(out, "the turn ended after repeated failures");
        self.end(out, Outcome::Failed, None);
        true
    }

    fn on_verdict(&mut self, out: &mut Vec<Action>, id: &CallId, verdict: Verdict) {
        let Some(call) = self.current(id, |s| s == CallState::Evaluating) else {
            return refuse(out, &format!("no call {id} is being evaluated"));
        };
        match verdict {
            Verdict::Allow => self.start_tool(out, call),
            Verdict::Ask if self.allowed_always.contains(&call.tool) => self.start_tool(out, call),
            // A host that is shutting down may never answer an ask.
            Verdict::Ask if self.turn.as_ref().is_some_and(|t| t.shutdown) => {
                self.set_state(CallState::Asking);
                self.on_interrupt(out);
            }
            Verdict::Ask => {
                self.append(
                    out,
                    vec![Record::ToolAsk {
                        call: call.id.clone(),
                        tool: call.tool.clone(),
                        input: call.input.clone(),
                    }],
                    Durability::Buffered,
                );
                out.push(Action::Emit(Event::PermissionAsked {
                    call: call.id,
                    tool: call.tool,
                    input: call.input,
                }));
                self.set_state(CallState::Asking);
            }
            Verdict::Deny(reason) => self.deny(out, &call, &reason),
        }
    }

    fn on_decision(&mut self, out: &mut Vec<Action>, id: &CallId, decision: Decision) {
        let Some(call) = self.current(id, |s| s == CallState::Asking) else {
            return refuse(out, &format!("no permission ask is pending for call {id}"));
        };
        self.append(
            out,
            vec![Record::ToolDecision {
                call: call.id.clone(),
                decision: decision.clone(),
            }],
            Durability::Buffered,
        );
        match decision {
            Decision::AllowOnce => self.start_tool(out, call),
            Decision::AllowAlways => {
                self.allowed_always.insert(call.tool.clone());
                self.start_tool(out, call);
            }
            Decision::Deny { reason } => self.deny(out, &call, &reason),
            _ => self.deny(out, &call, "the host's answer was not understood"),
        }
    }

    fn deny(&mut self, out: &mut Vec<Action>, call: &Call, reason: &str) {
        self.core_result(out, call, reason, Durability::Buffered);
        self.clear_current();
        self.advance(out);
    }

    fn start_tool(&mut self, out: &mut Vec<Action>, call: Call) {
        let CallKind::Known(effect) = call.kind else {
            return refuse(out, "only a known tool is run");
        };
        self.append(
            out,
            vec![Record::ToolStart {
                call: call.id.clone(),
                tool: call.tool.clone(),
                effect,
            }],
            durability(effect),
        );
        out.push(Action::Emit(Event::ToolStarted {
            call: call.id.clone(),
            tool: call.tool.clone(),
        }));
        out.push(Action::RunTool {
            call: call.id,
            tool: call.tool,
            input: call.input,
            effect,
        });
        self.set_state(CallState::Running(effect));
    }

    fn on_tool_done(&mut self, out: &mut Vec<Action>, id: &CallId, output: &ToolOutput) {
        let running = |s| matches!(s, CallState::Running(_) | CallState::Stopping { .. });
        let Some(call) = self.current(id, running) else {
            return refuse(out, &format!("no call {id} is running"));
        };
        let state = self.state();
        let CallKind::Known(effect) = call.kind else {
            return refuse(out, "only a known tool runs");
        };
        self.failures = 0;
        let content = match self.profile.protocol {
            StepProtocol::JsonObject => {
                RawJson::string(&step::observation(&output.content, output.is_error))
            }
            StepProtocol::Native => output.content.clone(),
        };
        let preview = preview_of(&output.content);
        self.result(
            out,
            &call,
            content,
            output.is_error,
            preview,
            durability(effect),
        );
        self.clear_current();
        if matches!(state, Some(CallState::Stopping { .. })) {
            self.skip_rest(out, step::INTERRUPTED);
            return self.end(out, Outcome::Interrupted, None);
        }
        self.advance(out);
    }

    fn on_tool_abandoned(&mut self, out: &mut Vec<Action>, id: &CallId) {
        let Some(call) = self.current(id, |s| matches!(s, CallState::Stopping { .. })) else {
            return refuse(out, &format!("no call {id} is being stopped"));
        };
        let Some(CallState::Stopping { effect, .. }) = self.state() else {
            return;
        };
        let text = match effect {
            Effect::Write => step::INTERRUPTED_EFFECT_UNKNOWN,
            Effect::Read => step::INTERRUPTED,
        };
        self.core_result(out, &call, text, durability(effect));
        self.clear_current();
        self.skip_rest(out, step::INTERRUPTED);
        self.end(out, Outcome::Interrupted, None);
    }

    fn on_interrupt(&mut self, out: &mut Vec<Action>) {
        let Some(t) = &mut self.turn else { return };
        match &mut t.stage {
            Stage::Calling { at } => {
                out.push(Action::Cancel(Running::Model { at: *at }));
                self.append(
                    out,
                    vec![Record::ModelFailure {
                        kind: ErrorKind::Cancelled,
                        detail: "interrupted".into(),
                    }],
                    Durability::Buffered,
                );
                self.end(out, Outcome::Interrupted, None);
            }
            Stage::Tools { current, .. } => match current {
                None => {
                    self.skip_rest(out, step::INTERRUPTED);
                    self.end(out, Outcome::Interrupted, None);
                }
                Some((call, state)) => match *state {
                    CallState::Evaluating | CallState::Asking => {
                        let call = call.clone();
                        self.core_result(
                            out,
                            &call,
                            &step::skipped(step::INTERRUPTED),
                            Durability::Buffered,
                        );
                        self.clear_current();
                        self.skip_rest(out, step::INTERRUPTED);
                        self.end(out, Outcome::Interrupted, None);
                    }
                    CallState::Running(effect) => {
                        let grace = (effect == Effect::Write).then_some(self.limits.write_grace);
                        out.push(Action::Cancel(Running::Tool {
                            call: call.id.clone(),
                            grace,
                        }));
                        *state = CallState::Stopping {
                            effect,
                            hard: grace.is_none(),
                        };
                    }
                    CallState::Stopping {
                        effect,
                        hard: false,
                    } => {
                        out.push(Action::Cancel(Running::Tool {
                            call: call.id.clone(),
                            grace: None,
                        }));
                        *state = CallState::Stopping { effect, hard: true };
                    }
                    CallState::Stopping { hard: true, .. } => {}
                },
            },
        }
    }

    fn on_model_failed(
        &mut self,
        out: &mut Vec<Action>,
        at: Seq,
        kind: ErrorKind,
        detail: Box<str>,
    ) {
        let calling = matches!(
            &self.turn,
            Some(Turn { stage: Stage::Calling { at: a }, .. }) if *a == at
        );
        if !calling {
            return refuse(
                out,
                &format!("no model call with request {at} is in flight"),
            );
        }
        self.append(
            out,
            vec![Record::ModelFailure {
                kind,
                detail: detail.clone(),
            }],
            Durability::Buffered,
        );
        out.push(Action::Emit(Event::Error { kind, detail }));
        let outcome = match kind {
            ErrorKind::BudgetExceeded => Outcome::RanOut,
            _ => Outcome::Failed,
        };
        self.end(out, outcome, None);
    }

    fn on_aborted(&mut self, out: &mut Vec<Action>, kind: ErrorKind, detail: Box<str>, next: Seq) {
        // Recoverable only if no call of an accepted answer is open: at rest at the core's own next seq, or a
        // request that was not written. Anything else leaves the log in a state only a rebuild can close.
        let recoverable = match &self.turn {
            None => next == self.next_seq,
            Some(Turn {
                stage: Stage::Calling { at },
                ..
            }) => next == *at,
            Some(_) => false,
        };
        self.broken = !recoverable;
        self.next_seq = next;
        self.history.retain(|s| *s < next);
        out.push(Action::Emit(Event::Error { kind, detail }));
        let Some(t) = self.turn.take() else { return };
        out.push(Action::Emit(Event::Done {
            turn: t.id,
            outcome: Outcome::Failed,
        }));
        out.push(Action::EndTurn(TurnEnd {
            turn: t.id,
            outcome: Outcome::Failed,
            reply: None,
            unsent: t.queued,
        }));
    }

    /// Answers every call of the step not yet taken.
    fn skip_rest(&mut self, out: &mut Vec<Action>, why: &str) {
        let Some(Turn {
            stage: Stage::Tools { calls, .. },
            ..
        }) = &mut self.turn
        else {
            return;
        };
        let rest: Vec<Call> = calls.drain(..).collect();
        let text = step::skipped(why);
        for call in &rest {
            self.core_result(out, call, &text, Durability::Buffered);
        }
    }

    fn end(&mut self, out: &mut Vec<Action>, outcome: Outcome, reply: Option<Box<str>>) {
        let Some(t) = self.turn.take() else { return };
        self.append(out, vec![Record::TurnEnd { outcome }], Durability::Durable);
        out.push(Action::Emit(Event::Done {
            turn: t.id,
            outcome,
        }));
        out.push(Action::EndTurn(TurnEnd {
            turn: t.id,
            outcome,
            reply,
            unsent: t.queued,
        }));
    }

    /// A result the core writes itself (a refusal, a skip, an interrupt), always an error result.
    fn core_result(&mut self, out: &mut Vec<Action>, call: &Call, text: &str, d: Durability) {
        let content = match self.profile.protocol {
            StepProtocol::JsonObject => RawJson::string(&format!("observation:\n{text}")),
            StepProtocol::Native => RawJson::string(text),
        };
        self.result(out, call, content, true, preview(text), d);
    }

    fn result(
        &mut self,
        out: &mut Vec<Action>,
        call: &Call,
        content: RawJson,
        is_error: bool,
        preview: Box<str>,
        d: Durability,
    ) {
        self.append(
            out,
            vec![Record::ToolResult {
                call: call.id.clone(),
                content,
                is_error,
            }],
            d,
        );
        out.push(Action::Emit(Event::ToolFinished {
            call: call.id.clone(),
            is_error,
            preview,
        }));
    }

    /// Appends records at the next seqs; message slices join the history. Returns the first seq.
    fn append(&mut self, out: &mut Vec<Action>, records: Vec<Record>, d: Durability) -> Seq {
        let at = self.next_seq;
        for (i, r) in records.iter().enumerate() {
            if matches!(
                r,
                Record::UserMessage { .. }
                    | Record::AssistantMessage { .. }
                    | Record::ToolResult { .. }
            ) {
                self.history.push(Seq(at.0 + i as u64));
            }
        }
        self.next_seq = Seq(at.0 + records.len() as u64);
        out.push(Action::Append {
            at,
            records,
            durability: d,
        });
        at
    }

    fn tool(&self, name: &str) -> Option<&ToolDef> {
        self.tools.iter().find(|t| &*t.name == name)
    }

    fn current(&self, id: &CallId, want: impl Fn(CallState) -> bool) -> Option<Call> {
        match &self.turn {
            Some(Turn {
                stage:
                    Stage::Tools {
                        current: Some((call, state)),
                        ..
                    },
                ..
            }) if call.id == *id && want(*state) => Some(call.clone()),
            _ => None,
        }
    }

    fn state(&self) -> Option<CallState> {
        match &self.turn {
            Some(Turn {
                stage:
                    Stage::Tools {
                        current: Some((_, state)),
                        ..
                    },
                ..
            }) => Some(*state),
            _ => None,
        }
    }

    fn set_state(&mut self, new: CallState) {
        if let Some(Turn {
            stage:
                Stage::Tools {
                    current: Some((_, state)),
                    ..
                },
            ..
        }) = &mut self.turn
        {
            *state = new;
        }
    }

    fn clear_current(&mut self) {
        if let Some(Turn {
            stage: Stage::Tools { current, .. },
            ..
        }) = &mut self.turn
        {
            *current = None;
        }
    }
}

fn durability(effect: Effect) -> Durability {
    match effect {
        Effect::Write => Durability::Durable,
        Effect::Read => Durability::Buffered,
    }
}

/// A user message's content as every provider takes it: a string, or a non-empty array of blocks (objects).
fn user_content(content: &RawJson) -> bool {
    let text = content.as_str();
    if text.starts_with('"') {
        return true;
    }
    serde_json::from_str::<Vec<&serde_json::value::RawValue>>(text)
        .is_ok_and(|blocks| !blocks.is_empty() && blocks.iter().all(|b| b.get().starts_with('{')))
}

/// An input that does not fit the core's state: reported, otherwise ignored.
fn refuse(out: &mut Vec<Action>, detail: &str) {
    out.push(Action::Emit(Event::Error {
        kind: ErrorKind::InvalidRequest,
        detail: detail.into(),
    }));
}

const PREVIEW_CHARS: usize = 200;

fn preview(text: &str) -> Box<str> {
    text.chars().take(PREVIEW_CHARS).collect::<String>().into()
}

fn preview_of(content: &RawJson) -> Box<str> {
    match serde_json::from_str::<String>(content.as_str()) {
        Ok(text) => preview(&text),
        Err(_) => preview(content.as_str()),
    }
}

#[cfg(test)]
mod tests;
