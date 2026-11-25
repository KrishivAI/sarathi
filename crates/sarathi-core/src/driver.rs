// SPDX-License-Identifier: Apache-2.0
//! The driver: carries out the core's actions against the session log, a model and the tools.
//!
//! [`Driver`] owns the IO the core leaves out. It appends exactly the records the core asks for, at the seqs the core
//! names, builds each request from the log (so a logged request can be rebuilt and checked against its hash), makes
//! the model call, evaluates and runs tool calls one at a time, and reports events to the host. Ops sent through a
//! [`Handle`] (a steer, an interrupt, a permission decision) reach the core while a call or a tool is in flight.
//!
//! A session that stopped in the middle of a turn (a crash, a killed process) is continued with [`Driver::resume`]:
//! a call that has a result keeps it, a side-effecting call that started and has none is answered as cut short and
//! never run again, and everything else runs again, so no tool use is left without a result.

mod context;
mod entries;
mod seams;

use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

use sarathi_log::{Body, Entry, Log, LogError, LogStore, SessionId, sha256_hex};
use sarathi_protocol::{CallId, ErrorKind, Event, Op, Outcome, Seq, ToolOutput, TurnId};
use tokio::sync::mpsc;
use tokio::time::Instant;

use self::entries::{Held, Start};
pub use self::seams::{
    AskByOp, AskReply, BoxFuture, CallFailure, DenyAsks, ModelCaller, PermissionHost, Received,
    Spill, StopSignal, Tools,
};
use crate::core::{Action, Core, CoreConfig, Durability, Input, Limits, Running, TurnEnd};
use crate::profile::{ModelProfile, ToolDef};
pub use sarathi_context::{Compactor, ContextProfile, SpillLimits};

/// What a session runs with: the model profile, the tools, the limits and the system prompt.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct DriverConfig {
    /// How steps are read and tool results rendered.
    pub profile: ModelProfile,
    /// The tools the model may call. Under [`crate::StepProtocol::JsonObject`] they are described in the system prompt.
    pub tools: Vec<ToolDef>,
    /// The limits every turn runs under.
    pub limits: Limits,
    /// The host's system prompt.
    pub system: Box<str>,
    /// How the session's context is kept within the model's window: compaction, the overflow fallback. `None`: the
    /// driver sends every request as it is, and a request too large fails as the provider answers.
    pub context: Option<ContextProfile>,
    /// Where tool output over a size limit goes; `None`: every output stays in the log.
    pub spill: Option<Spill>,
}

impl DriverConfig {
    /// A config with the default limits.
    pub fn new(profile: ModelProfile, tools: Vec<ToolDef>, system: impl Into<Box<str>>) -> Self {
        DriverConfig {
            profile,
            tools,
            limits: Limits::default(),
            system: system.into(),
            context: None,
            spill: None,
        }
    }

    /// The same config managing the context under `profile`.
    #[must_use]
    pub fn with_context(mut self, profile: ContextProfile) -> Self {
        self.context = Some(profile);
        self
    }

    /// The same config spilling large tool output to `spill`.
    #[must_use]
    pub fn with_spill(mut self, spill: Spill) -> Self {
        self.spill = Some(spill);
        self
    }

    /// The same config with `limits`.
    #[must_use]
    pub fn with_limits(mut self, limits: Limits) -> Self {
        self.limits = limits;
        self
    }
}

/// Why the driver could not go on.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum DriverError {
    /// The session log refused a read or a write outside a turn.
    #[error("session log: {0}")]
    Log(#[from] LogError),
    /// The log does not hold a session the driver can continue.
    #[error("the session cannot be continued: {0}")]
    Session(String),
    /// A turn cut short by a restart is still open: finish it with [`Driver::continue_turn`] first.
    #[error("a turn cut short by a restart is still open")]
    TurnOpen,
    /// The context check before the turn could not write to the log (the op is not run: send it again), or the
    /// context profile does not suit the caller.
    #[error("context management: {0}")]
    Context(String),
    /// The next request cannot fit the model's window even after the context fallback dropped all it may (the op is
    /// not run; the session needs a larger window or a smaller turn).
    #[error("the next request does not fit the model's context window")]
    DoesNotFit,
    /// An interrupt came while the context check before the turn ran (a compaction call): the op is not run.
    #[error("interrupted before the turn started")]
    Interrupted,
    /// The op did not start a turn (the core refused it; see the events).
    #[error("the op did not start a turn")]
    NotStarted,
}

/// Sends ops to a running driver from any task.
///
/// The driver runs every tool call through its toolbox, so an [`Op::ToolOutput`] for the call it is running is
/// refused with an [`Event::Error`] and the call keeps running.
#[derive(Debug, Clone)]
pub struct Handle {
    tx: mpsc::UnboundedSender<Op>,
}

impl Handle {
    /// Sends `op`; it reaches the core at the driver's next wait. Returns `false` if the driver is gone.
    pub fn send(&self, op: Op) -> bool {
        self.tx.send(op).is_ok()
    }
}

#[cfg(feature = "fault-injection")]
type FaultHook = Box<dyn FnMut(Seq) + Send>;

/// Runs a session's turns. See the module documentation.
pub struct Driver<S, C, T, P> {
    store: S,
    caller: C,
    tools: T,
    permissions: P,
    config: DriverConfig,
    session: SessionId,
    log: Log,
    core: Core,
    /// The first actions of a resumed turn, not yet carried out.
    pending: Vec<Action>,
    /// A turn was being driven and its future was dropped: the core and the log copy may be behind the store, so
    /// the next call rebuilds from the store, as after a restart.
    dirty: bool,
    /// A write failed during a turn the host was told failed: the next call closes that turn in the log first.
    failed: bool,
    ops: mpsc::UnboundedReceiver<Op>,
    /// Ops read between turns that wait for a turn to join.
    late: Vec<Op>,
    /// Only weak: when every host handle is gone, the channel closes, so a turn waiting on an op can tell.
    weak: mpsc::WeakUnboundedSender<Op>,
    #[cfg(feature = "fault-injection")]
    fault: Option<FaultHook>,
}

impl<S, C, T, P> std::fmt::Debug for Driver<S, C, T, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Driver")
            .field("session", &self.session)
            .field("next_seq", &self.log.next_seq())
            .field("phase", &self.core.phase())
            .finish_non_exhaustive()
    }
}

impl<S, C, T, P> Driver<S, C, T, P>
where
    S: LogStore,
    C: ModelCaller,
    T: Tools,
    P: PermissionHost,
{
    /// Starts a new session: creates its log and writes the session prefix (the system prompt and the tools, rendered
    /// once for every request of the session).
    ///
    /// # Errors
    /// The store refused the session.
    pub async fn new(
        config: DriverConfig,
        store: S,
        caller: C,
        tools: T,
        permissions: P,
    ) -> Result<(Self, Handle), DriverError> {
        check_context(&config, &caller)?;
        // Rendered before the session exists, so a config that cannot start one leaves no empty session behind.
        let body = entries::prefix_body(&config, &caller).map_err(DriverError::Session)?;
        let header = store.create(now_ms(), None).await?;
        let prefix = Entry {
            seq: Seq(1),
            seg: sarathi_log::SegmentId(0),
            ts: now_ms(),
            body,
        };
        let session = header.session;
        store
            .append(
                session,
                Seq(1),
                vec![prefix.clone()],
                sarathi_log::Durability::Durable,
            )
            .await?;
        let mut log = Log::new(header);
        log.append(Seq(1), vec![prefix])?;
        let mut core_config = CoreConfig::new(config.profile.clone(), config.tools.clone(), Seq(1));
        core_config.limits = config.limits.clone();
        core_config.next_seq = Seq(2);
        let core = Core::new(core_config);
        Ok(Self::make(
            config,
            store,
            caller,
            tools,
            permissions,
            log,
            core,
            Vec::new(),
        ))
    }

    /// Continues `session` from its log. A turn the restart cut short stays open ([`Driver::in_turn`]); finish it
    /// with [`Driver::continue_turn`] before the next [`Driver::run_turn`].
    ///
    /// # Errors
    /// The store cannot read the session, or the log does not hold one the driver can continue.
    pub async fn resume(
        config: DriverConfig,
        store: S,
        session: SessionId,
        caller: C,
        tools: T,
        permissions: P,
    ) -> Result<(Self, Handle), DriverError> {
        check_context(&config, &caller)?;
        let log = store.load(session).await?.log;
        let (log, core, pending) = Self::start(&config, &store, &caller, log).await?;
        Ok(Self::make(
            config,
            store,
            caller,
            tools,
            permissions,
            log,
            core,
            pending,
        ))
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the parts both constructors share"
    )]
    fn make(
        config: DriverConfig,
        store: S,
        caller: C,
        tools: T,
        permissions: P,
        log: Log,
        core: Core,
        pending: Vec<Action>,
    ) -> (Self, Handle) {
        let (tx, ops) = mpsc::unbounded_channel();
        let weak = tx.downgrade();
        let handle = Handle { tx };
        let session = log.header().session;
        let driver = Driver {
            store,
            caller,
            tools,
            permissions,
            config,
            session,
            log,
            core,
            pending,
            dirty: false,
            failed: false,
            ops,
            late: Vec::new(),
            weak,
            #[cfg(feature = "fault-injection")]
            fault: None,
        };
        (driver, handle)
    }

    /// A core from `log`, closing first a turn whose last model call failed before its end was written.
    async fn start(
        config: &DriverConfig,
        store: &S,
        caller: &C,
        mut log: Log,
    ) -> Result<(Log, Core, Vec<Action>), DriverError> {
        loop {
            match entries::start(config, &log, caller).map_err(DriverError::Session)? {
                Start::Rest(core) => return Ok((log, core, Vec::new())),
                Start::Open(core, actions) => return Ok((log, core, actions)),
                Start::Close(outcome) => {
                    let at = log.next_seq();
                    let end = Entry {
                        seq: at,
                        seg: log.segment(),
                        ts: now_ms(),
                        body: entries::turn_end(outcome),
                    };
                    let session = log.header().session;
                    store
                        .append(
                            session,
                            at,
                            vec![end.clone()],
                            sarathi_log::Durability::Durable,
                        )
                        .await?;
                    log.append(at, vec![end])?;
                }
            }
        }
    }

    /// The session.
    pub fn session(&self) -> SessionId {
        self.session
    }

    /// The session's log as the driver wrote it.
    pub fn log(&self) -> &Log {
        &self.log
    }

    /// A handle for sending ops. Once every handle is dropped, a turn waiting for an op (an ask left pending) is
    /// interrupted, since no answer can come; a later call here opens a new channel.
    pub fn handle(&mut self) -> Handle {
        if let Some(tx) = self.weak.upgrade() {
            return Handle { tx };
        }
        while let Ok(op) = self.ops.try_recv() {
            if matches!(op, Op::Steer { .. } | Op::UserMessage { .. } | Op::Shutdown) {
                self.late.push(op);
            }
        }
        let (tx, ops) = mpsc::unbounded_channel();
        self.ops = ops;
        self.weak = tx.downgrade();
        Handle { tx }
    }

    /// Whether a turn is open and waits for [`Driver::continue_turn`]: one a restart cut short, or one whose future
    /// was dropped (until the next call reads the log, a dropped turn counts as open).
    pub fn in_turn(&self) -> bool {
        !self.pending.is_empty() || self.dirty
    }

    /// Calls `hook` with the next seq after every append that reached the store. For crash tests only.
    #[cfg(feature = "fault-injection")]
    pub fn on_append(&mut self, hook: impl FnMut(Seq) + Send + 'static) {
        self.fault = Some(Box::new(hook));
    }

    /// Runs one turn: `op` is the user message (or steer) that starts it. Events go to `on_event` as they happen.
    ///
    /// Ops sent between turns are read first: an interrupt or a decision aimed at the turn that ended is dropped;
    /// a steer, a user message or a shutdown joins this turn (a steer or message at its first tool boundary).
    ///
    /// Dropping the returned future is a restart: the turn stays open ([`Driver::in_turn`]), and the next call reads
    /// the log to see what it holds. A tool in flight is dropped with it, so a side-effecting call is answered as
    /// cut short. A failed write ends the turn [`Outcome::Failed`], and the log records it as failed before the
    /// next turn.
    ///
    /// With a context profile, the context check runs first ([`DriverConfig::with_context`]): it may make a compaction
    /// call before the turn starts. A compaction call a restart left open is finished first even without a profile.
    ///
    /// # Errors
    /// [`DriverError::TurnOpen`] if a turn is still open (the op is not run: send it again after
    /// [`Driver::continue_turn`]), [`DriverError::NotStarted`] if the core refused the op,
    /// [`DriverError::Context`] if the context check could not write to the log (the op is not run: send it again),
    /// [`DriverError::DoesNotFit`] if the next request cannot fit the window, [`DriverError::Interrupted`] if an
    /// interrupt came during the context check (the op is not run), and the store's errors when the session has to
    /// be read or closed first.
    pub async fn run_turn(
        &mut self,
        op: Op,
        on_event: &mut (dyn FnMut(Event) + Send),
    ) -> Result<TurnEnd, DriverError> {
        self.settle().await?;
        if !self.pending.is_empty() {
            return Err(DriverError::TurnOpen);
        }
        // Ops sent before this call wait in `self.late` (an interrupt among them, aimed at the last turn, is dropped),
        // so only an interrupt sent during the check below counts, and an error or a dropped future loses none.
        self.late = self.late_ops();
        self.between_turns(on_event).await?;
        if self.interrupted_during_check() {
            return Err(DriverError::Interrupted);
        }
        let late = std::mem::take(&mut self.late);
        let mut first = self.core.step(Input::Op(op));
        self.join_turn(&mut first, late);
        self.drive_turn(first, on_event)
            .await
            .ok_or(DriverError::NotStarted)
    }

    /// Finishes an open turn: one a restart cut short, or one whose future was dropped. `Ok(None)` if no turn was
    /// open.
    ///
    /// The context check runs before each model call of the turn, as in [`Driver::run_turn`]. After a restart in the
    /// middle of the turn's own model call, that call is made again without a check (the log is not at a boundary,
    /// so nothing may be cut), as it was checked before it was first made.
    ///
    /// # Errors
    /// The store's errors when the session has to be read or closed first.
    pub async fn continue_turn(
        &mut self,
        on_event: &mut (dyn FnMut(Event) + Send),
    ) -> Result<Option<TurnEnd>, DriverError> {
        self.settle().await?;
        let mut first = std::mem::take(&mut self.pending);
        if first.is_empty() {
            return Ok(None);
        }
        let late = self.late_ops();
        self.join_turn(&mut first, late);
        Ok(self.drive_turn(first, on_event).await)
    }

    /// Before a turn: closes a failed turn in the log, or rebuilds after a dropped one.
    async fn settle(&mut self) -> Result<(), DriverError> {
        if self.failed {
            let log = self.store.load(self.session).await?.log;
            let closing = entries::closing(&log, &self.caller, self.config.profile.protocol)
                .map_err(DriverError::Session)?;
            let mut log = log;
            if !closing.is_empty() {
                let at = log.next_seq();
                let seg = log.segment();
                let ts = now_ms();
                let entries: Vec<Entry> = closing
                    .into_iter()
                    .zip(at.0..)
                    .map(|(body, seq)| Entry {
                        seq: Seq(seq),
                        seg,
                        ts,
                        body,
                    })
                    .collect();
                self.store
                    .append(
                        self.session,
                        at,
                        entries.clone(),
                        sarathi_log::Durability::Durable,
                    )
                    .await?;
                log.append(at, entries)?;
            }
            self.failed = false;
            self.rebuild(log).await?;
        } else if self.dirty || self.core.needs_rebuild() {
            let log = self.store.load(self.session).await?.log;
            self.rebuild(log).await?;
        }
        Ok(())
    }

    /// The context check between turns: a compaction or a fallback seed starts a new segment the core is rebuilt on.
    /// The driver counts as dirty while it runs, so a dropped future leaves the next call to read the log again.
    async fn between_turns(
        &mut self,
        on_event: &mut (dyn FnMut(Event) + Send),
    ) -> Result<(), DriverError> {
        let profile = self.config.context.clone();
        self.dirty = true;
        let mut w = Writer {
            store: &self.store,
            session: self.session,
            log: &mut self.log,
            #[cfg(feature = "fault-injection")]
            fault: &mut self.fault,
        };
        let checked = context::maintain(&mut w, &self.caller, profile.as_ref(), on_event).await;
        match checked {
            Ok(c) => {
                if c.changed {
                    self.rebuild(self.log.clone()).await?;
                } else {
                    self.dirty = false;
                }
                if c.does_not_fit {
                    return Err(DriverError::DoesNotFit);
                }
                Ok(())
            }
            // The store's copy may be ahead of ours: `dirty` stays set, so it is read again before the next turn.
            Err(detail) => Err(DriverError::Context(detail)),
        }
    }

    /// Reads the ops that came during the check before a turn: true if an interrupt was among them. The rest join
    /// the late ops.
    fn interrupted_during_check(&mut self) -> bool {
        let mut interrupted = false;
        while let Ok(op) = self.ops.try_recv() {
            match op {
                Op::Interrupt => interrupted = true,
                Op::Steer { .. } | Op::UserMessage { .. } | Op::Shutdown => self.late.push(op),
                _ => {}
            }
        }
        interrupted
    }

    async fn rebuild(&mut self, log: Log) -> Result<(), DriverError> {
        let (log, core, pending) =
            Self::start(&self.config, &self.store, &self.caller, log).await?;
        self.log = log;
        self.core = core;
        self.pending = pending;
        self.dirty = false;
        Ok(())
    }

    /// The ops sent since the last turn that still mean something for the next one.
    fn late_ops(&mut self) -> Vec<Op> {
        let mut late = std::mem::take(&mut self.late);
        while let Ok(op) = self.ops.try_recv() {
            if matches!(op, Op::Steer { .. } | Op::UserMessage { .. } | Op::Shutdown) {
                late.push(op);
            }
        }
        late
    }

    /// Feeds late ops to the turn the core is running; with no turn running (a resumed turn that ends in its first
    /// actions), they wait for the next turn instead of starting one here.
    fn join_turn(&mut self, first: &mut Vec<Action>, late: Vec<Op>) {
        if self.core.phase() == crate::core::Phase::Idle {
            self.late = late;
            return;
        }
        for op in late {
            first.extend(self.core.step(Input::Op(op)));
        }
    }

    /// Drives a turn, marking the driver dirty until it returns normally.
    async fn drive_turn(
        &mut self,
        first: Vec<Action>,
        on_event: &mut (dyn FnMut(Event) + Send),
    ) -> Option<TurnEnd> {
        self.dirty = true;
        let (end, failed) = self.drive(first, on_event).await;
        // A drive that returns with the core still in a turn left work it did not carry out: rebuild from the log.
        self.dirty = self.core.phase() != crate::core::Phase::Idle;
        if failed {
            self.failed = true;
            // Close it now if the store answers again, so the log agrees with what the host was told.
            let _ = self.settle().await;
        }
        end
    }

    /// Carries out `first` and every action that follows, until the turn ends or the core waits with nothing in
    /// flight and no turn running.
    async fn drive(
        &mut self,
        first: Vec<Action>,
        on_event: &mut (dyn FnMut(Event) + Send),
    ) -> (Option<TurnEnd>, bool) {
        let Driver {
            store,
            caller,
            tools,
            permissions,
            config,
            session,
            log,
            core,
            ops,
            #[cfg(feature = "fault-injection")]
            fault,
            ..
        } = self;
        let mut w = Writer {
            store,
            session: *session,
            log,
            #[cfg(feature = "fault-injection")]
            fault,
        };
        let protocol = config.profile.protocol;
        let mut held = Held::default();
        let mut queue: VecDeque<Action> = first.into();
        let mut flight: Option<Flight<'_>> = None;
        let mut ended = None;
        let mut turn = None;
        let mut failed = false;
        let mut closed = false;
        let mut nudged = false;
        loop {
            while let Some(action) = queue.pop_front() {
                match action {
                    Action::Append {
                        at,
                        mut records,
                        durability,
                    } => {
                        let spilled = match &config.spill {
                            Some(spill) => {
                                context::spill_results(&mut records, spill, *session, on_event)
                                    .await
                            }
                            None => Vec::new(),
                        };
                        let bodies: Result<Vec<Body>, String> = records
                            .into_iter()
                            .zip(spilled.into_iter().chain(std::iter::repeat(None)))
                            .map(|(r, field)| {
                                let mut body = entries::body_of(r, caller, protocol, &mut held)?;
                                if let Body::ToolResult { spill, .. } = &mut body {
                                    *spill = field;
                                }
                                Ok(body)
                            })
                            .collect();
                        let written = match bodies {
                            Ok(b) => w.write(at, b, durability).await,
                            Err(e) => Err(e),
                        };
                        if let Err(detail) = written {
                            flight = None;
                            failed = true;
                            abort(core, &mut queue, w.aborted(detail), turn);
                        }
                    }
                    Action::CallModel { at, recipe } => {
                        match context::maintain(&mut w, caller, config.context.as_ref(), on_event)
                            .await
                        {
                            Ok(c) if c.does_not_fit => {
                                // Sending it would only fail at the provider: the turn ends here, as a guard.
                                failed = true;
                                let aborted = Input::Aborted {
                                    kind: ErrorKind::Guard,
                                    detail:
                                        "the next request does not fit the model's context window"
                                            .into(),
                                    next_seq: w.log.next_seq(),
                                };
                                abort(core, &mut queue, aborted, turn);
                                continue;
                            }
                            Ok(c) if c.changed => {
                                // A new segment (or a failed compaction call) sits at `at`: make the call again
                                // after it, from the new segment.
                                let history = w.log.projection().messages;
                                let again = core.reseat(at, history, w.log.next_seq());
                                if !again.iter().any(|a| matches!(a, Action::CallModel { .. })) {
                                    // The core refused: nothing would be in flight and the turn would hang.
                                    for a in again {
                                        if let Action::Emit(e) = a {
                                            on_event(e);
                                        }
                                    }
                                    failed = true;
                                    let aborted = Input::Aborted {
                                        kind: ErrorKind::Internal,
                                        detail: "the core did not take the new segment".into(),
                                        next_seq: w.log.next_seq(),
                                    };
                                    abort(core, &mut queue, aborted, turn);
                                    continue;
                                }
                                for a in again.into_iter().rev() {
                                    queue.push_front(a);
                                }
                                continue;
                            }
                            Ok(_) => {}
                            Err(detail) => {
                                failed = true;
                                abort(core, &mut queue, w.aborted(detail), turn);
                                continue;
                            }
                        }
                        let recipe = sarathi_log::Recipe {
                            prefix: recipe.prefix,
                            messages: recipe.messages,
                            params: caller.params(),
                        };
                        let prepared = match w.log.request_parts(&recipe) {
                            Ok(parts) => caller.prepare(&parts).map_err(|f| f.detail.to_string()),
                            Err(e) => Err(e.to_string()),
                        };
                        let sent = match prepared {
                            Ok((body, request)) => {
                                let sha256 = sha256_hex(&body);
                                let entry = Body::ModelRequest {
                                    recipe,
                                    sha256,
                                    compaction: None,
                                };
                                w.write(at, vec![entry], Durability::Buffered)
                                    .await
                                    .map(|()| request)
                            }
                            Err(e) => Err(e),
                        };
                        match sent {
                            Ok(request) => {
                                flight = Some(Flight::Model {
                                    at,
                                    fut: caller.send(request),
                                });
                            }
                            Err(detail) => {
                                failed = true;
                                abort(core, &mut queue, w.aborted(detail), turn);
                            }
                        }
                    }
                    Action::Evaluate { call, tool, input } => {
                        let verdict = tools.evaluate(&tool, &input);
                        queue.extend(core.step(Input::Verdict { call, verdict }));
                    }
                    Action::RunTool {
                        call, tool, input, ..
                    } => {
                        let stop = StopSignal::new();
                        let fut = tools.run(call.clone(), tool, input, stop.clone());
                        flight = Some(Flight::Tool {
                            call,
                            fut,
                            stop,
                            deadline: None,
                        });
                    }
                    Action::Cancel(Running::Model { at }) => {
                        if matches!(&flight, Some(Flight::Model { at: a, .. }) if *a == at) {
                            flight = None;
                        }
                    }
                    Action::Cancel(Running::Tool { call, grace }) => {
                        let running =
                            matches!(&flight, Some(Flight::Tool { call: c, .. }) if *c == call);
                        match (running, grace) {
                            (true, Some(grace)) => {
                                if let Some(Flight::Tool { stop, deadline, .. }) = &mut flight {
                                    stop.fire();
                                    *deadline = Some(Instant::now() + grace);
                                }
                            }
                            (true, None) => {
                                if let Some(Flight::Tool { stop, .. }) = &flight {
                                    stop.fire();
                                }
                                flight = None;
                                queue.extend(core.step(Input::ToolAbandoned { call }));
                            }
                            (false, _) => {
                                queue.extend(core.step(Input::ToolAbandoned { call }));
                            }
                        }
                    }
                    Action::Emit(event) => {
                        if let Event::TurnStarted { turn: t } = &event {
                            turn = Some(*t);
                        }
                        let ask = match &event {
                            Event::PermissionAsked { call, tool, input } => {
                                Some((call.clone(), permissions.ask(call, tool, input)))
                            }
                            _ => None,
                        };
                        on_event(event);
                        if let Some((call, AskReply::Decided(decision))) = ask {
                            queue.extend(core.step(Input::Op(Op::ToolDecision { call, decision })));
                        }
                    }
                    Action::EndTurn(end) => ended = Some(end),
                }
            }
            if ended.is_some() {
                return (ended, failed);
            }
            if flight.is_none() && core.phase() == crate::core::Phase::Idle {
                return (None, failed);
            }
            if closed && flight.is_none() {
                // No op can come and nothing is in flight: the core waits for an answer no one can send.
                if nudged {
                    return (turn.map(TurnEnd::failed), failed);
                }
                nudged = true;
                queue.extend(core.step(Input::Op(Op::Interrupt)));
                continue;
            }
            let input = tokio::select! {
                landed = land(&mut flight) => {
                    flight = None;
                    match landed {
                        Landed::Model(at, Ok(received)) => {
                            let answer = received.answer();
                            held.answer = Some(received);
                            Input::ModelAnswered { at, answer }
                        }
                        Landed::Model(at, Err(failure)) => {
                            let input = Input::ModelFailed {
                                at,
                                kind: failure.kind,
                                detail: failure.detail.clone(),
                            };
                            held.failure = Some(failure);
                            input
                        }
                        Landed::Tool(call, output) => Input::ToolDone { call, output },
                        Landed::Abandoned(call) => Input::ToolAbandoned { call },
                    }
                }
                op = ops.recv(), if !closed => match op {
                    // The driver's toolbox runs this call: an answer from the host would end the call while its
                    // future is still running, and the next call would drop that future mid-run.
                    Some(Op::ToolOutput { call, .. })
                        if matches!(&flight, Some(Flight::Tool { call: c, .. }) if *c == call) =>
                    {
                        on_event(Event::Error {
                            kind: ErrorKind::InvalidRequest,
                            detail: format!("call {call} is run by the driver's toolbox, not the host").into(),
                        });
                        continue;
                    }
                    Some(op) => Input::Op(op),
                    None => {
                        closed = true;
                        continue;
                    }
                },
            };
            queue.extend(core.step(input));
        }
    }
}

/// Server compaction is the Anthropic Messages API's: with another protocol, the profile must summarize itself.
fn check_context<C: ModelCaller>(config: &DriverConfig, caller: &C) -> Result<(), DriverError> {
    match &config.context {
        Some(p)
            if p.compactor == Compactor::Server && caller.protocol() != "anthropic-messages" =>
        {
            Err(DriverError::Context(format!(
                "server compaction needs the anthropic-messages protocol, not {}: use Compactor::ClientSummary",
                caller.protocol()
            )))
        }
        _ => Ok(()),
    }
}

/// A write failed: nothing after it in the batch is carried out, and the core drops the turn. If the core had already
/// ended the turn in the same step (its end was among the dropped actions), the turn ends as failed here.
fn abort(core: &mut Core, queue: &mut VecDeque<Action>, aborted: Input, turn: Option<TurnId>) {
    queue.clear();
    let actions = core.step(aborted);
    let ends = actions.iter().any(|a| matches!(a, Action::EndTurn(_)));
    queue.extend(actions);
    if let (false, Some(turn)) = (ends, turn) {
        queue.push_back(Action::Emit(Event::Done {
            turn,
            outcome: Outcome::Failed,
        }));
        queue.push_back(Action::EndTurn(TurnEnd::failed(turn)));
    }
}

/// Appends to the store and keeps the driver's copy of the log in step.
struct Writer<'a, S> {
    store: &'a S,
    session: SessionId,
    log: &'a mut Log,
    #[cfg(feature = "fault-injection")]
    fault: &'a mut Option<FaultHook>,
}

impl<S: LogStore> Writer<'_, S> {
    async fn write(&mut self, at: Seq, bodies: Vec<Body>, d: Durability) -> Result<(), String> {
        let seg = self.log.segment();
        let ts = now_ms();
        let entries: Vec<Entry> = bodies
            .into_iter()
            .zip(at.0..)
            .map(|(body, seq)| Entry {
                seq: Seq(seq),
                seg,
                ts,
                body,
            })
            .collect();
        self.write_entries(at, entries, d).await
    }

    /// Appends whole entries (with their own segments: a seed ends one segment and starts the next).
    async fn write_entries(
        &mut self,
        at: Seq,
        entries: Vec<Entry>,
        d: Durability,
    ) -> Result<(), String> {
        let durability = match d {
            Durability::Buffered => sarathi_log::Durability::Buffered,
            Durability::Durable => sarathi_log::Durability::Durable,
        };
        let next = self
            .store
            .append(self.session, at, entries.clone(), durability)
            .await
            .map_err(|e| e.to_string())?;
        self.log.append(at, entries).map_err(|e| e.to_string())?;
        if next != self.log.next_seq() {
            return Err(format!(
                "the store's next seq is {next}, the driver's {}",
                self.log.next_seq()
            ));
        }
        #[cfg(feature = "fault-injection")]
        if let Some(hook) = self.fault.as_mut() {
            hook(next);
        }
        Ok(())
    }

    fn aborted(&self, detail: String) -> Input {
        Input::Aborted {
            kind: ErrorKind::Log,
            detail: detail.into(),
            next_seq: self.log.next_seq(),
        }
    }
}

/// Work in flight: at most one model call or one tool call.
enum Flight<'a> {
    Model {
        at: Seq,
        fut: BoxFuture<'a, Result<Received, CallFailure>>,
    },
    Tool {
        call: CallId,
        fut: BoxFuture<'a, ToolOutput>,
        stop: StopSignal,
        /// Set when an interrupt gave a side-effecting tool a grace period to stop.
        deadline: Option<Instant>,
    },
}

enum Landed {
    Model(Seq, Result<Received, CallFailure>),
    Tool(CallId, ToolOutput),
    Abandoned(CallId),
}

/// Waits for the work in flight; never returns if there is none.
async fn land(flight: &mut Option<Flight<'_>>) -> Landed {
    match flight {
        None => std::future::pending().await,
        Some(Flight::Model { at, fut }) => Landed::Model(*at, fut.await),
        Some(Flight::Tool {
            call,
            fut,
            deadline,
            ..
        }) => match deadline {
            None => Landed::Tool(call.clone(), fut.await),
            Some(d) => tokio::select! {
                output = fut => Landed::Tool(call.clone(), output),
                () = tokio::time::sleep_until(*d) => Landed::Abandoned(call.clone()),
            },
        },
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}
