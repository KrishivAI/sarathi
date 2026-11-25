// SPDX-License-Identifier: Apache-2.0
//! The seams a host fills besides the session log: the model, the tools and the permission host.

use std::future::Future;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use sarathi_context::{ContextError, SpillLimits, SpillStore};
use sarathi_log::{RequestParts, SessionId};
use sarathi_protocol::{CallId, Decision, ErrorKind, RawJson, ResponseView, ToolOutput};
use tokio::sync::Notify;

use crate::core::{Answer, Verdict};
use crate::profile::ToolDef;

/// A boxed future that can move between threads.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// A model answer as received, with what the log stores of it.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Received {
    /// The body as received (streaming: the assembled message).
    pub body: RawJson,
    /// The byte range in `body` of the value that becomes the assistant message's `content`.
    pub content: Range<u32>,
    /// What the answer holds; its ranges point into `body`.
    pub view: ResponseView,
    /// The answer's text, when the view does not describe the body.
    pub text: Option<Box<str>>,
    /// The stream as received, for a streamed answer.
    pub sse: Option<String>,
    /// The provider's request id.
    pub request_id: Option<Box<str>>,
    /// The call's cost in micro-dollars, when known.
    pub cost_micros: Option<u64>,
}

impl Received {
    /// An answer of `body` whose `content` is at `content`, described by `view`.
    pub fn new(body: RawJson, content: Range<u32>, view: ResponseView) -> Self {
        Received {
            body,
            content,
            view,
            text: None,
            sse: None,
            request_id: None,
            cost_micros: None,
        }
    }

    /// The same answer with its provider request id.
    #[must_use]
    pub fn with_request_id(mut self, id: impl Into<Box<str>>) -> Self {
        self.request_id = Some(id.into());
        self
    }

    /// The same answer with its cost.
    #[must_use]
    pub fn with_cost(mut self, cost_micros: u64) -> Self {
        self.cost_micros = Some(cost_micros);
        self
    }

    /// The answer as the core reads it.
    pub(crate) fn answer(&self) -> Answer {
        let mut a = Answer::new(self.body.clone(), self.view.clone());
        a.text.clone_from(&self.text);
        a.cost_micros = self.cost_micros;
        a
    }
}

/// A model call that ended without an answer, after the caller's retries and failover.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallFailure {
    /// The kind of failure.
    pub kind: ErrorKind,
    /// What went wrong, holding no credential.
    pub detail: Box<str>,
    /// The HTTP status, if the server answered.
    pub status: Option<u16>,
    /// The provider's request id.
    pub request_id: Option<Box<str>>,
    /// Attempts made before giving up, after the first.
    pub retries: u32,
}

impl CallFailure {
    /// A failure of `kind`.
    pub fn new(kind: ErrorKind, detail: impl Into<Box<str>>) -> Self {
        CallFailure {
            kind,
            detail: detail.into(),
            status: None,
            request_id: None,
            retries: 0,
        }
    }
}

/// How the driver reaches a model: one provider protocol for the whole session.
///
/// The caller renders what goes into the session prefix and the tool results, builds a request body from the log's
/// [`RequestParts`] (the same parts give the same bytes, so a logged request can be checked against its hash), and
/// makes the call. It never stores anything; the driver logs what it returns.
pub trait ModelCaller: Send + Sync {
    /// A request ready to send.
    type Prepared: Send;

    /// The provider protocol's name in the log (`anthropic-messages`).
    fn protocol(&self) -> &str;

    /// The per-call parameters as the log's recipe stores them.
    fn params(&self) -> RawJson;

    /// The request's `tools` value for these tools, or `None` when the request carries none.
    ///
    /// # Errors
    /// The tools cannot be rendered: the session is not started, rather than started without them.
    fn render_tools(&self, tools: &[ToolDef]) -> Result<Option<RawJson>, CallFailure>;

    /// The block a tool result goes back as.
    fn result_block(&self, call: &CallId, content: &RawJson, is_error: bool) -> RawJson;

    /// Builds the request from `parts`: the body exactly as it will be sent, and the request.
    ///
    /// # Errors
    /// The parts cannot make a request of this protocol.
    fn prepare(&self, parts: &RequestParts<'_>) -> Result<(Vec<u8>, Self::Prepared), CallFailure>;

    /// Makes the call.
    fn send(&self, request: Self::Prepared) -> BoxFuture<'_, Result<Received, CallFailure>>;

    /// Reads an answer stored in the log back (`body` and its `content` range), for a resumed turn.
    ///
    /// # Errors
    /// The body is not an answer of this protocol.
    fn read(&self, body: &RawJson, content: Range<u32>) -> Result<Received, CallFailure>;

    /// Counts the input tokens of the request `parts` make, when the provider can (Anthropic `count_tokens`), for a
    /// context check close to a limit. The future must not borrow `parts`. `None` (the default): the driver goes on
    /// with its estimate, as if it were the count.
    fn count_tokens(
        &self,
        parts: &RequestParts<'_>,
    ) -> Option<BoxFuture<'_, Result<u64, CallFailure>>> {
        let _ = parts;
        None
    }
}

/// Where the driver spills tool output over a size limit ([`sarathi_context::SpillStore`]), with the limits.
#[derive(Clone)]
pub struct Spill {
    pub(super) store: Arc<dyn ErasedSpill>,
    pub(super) limits: SpillLimits,
}

impl Spill {
    /// Spills to `store` under `limits`.
    pub fn new<S: SpillStore + 'static>(store: S, limits: SpillLimits) -> Self {
        Spill {
            store: Arc::new(store),
            limits,
        }
    }

    /// The limits.
    pub fn limits(&self) -> &SpillLimits {
        &self.limits
    }
}

impl std::fmt::Debug for Spill {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Spill")
            .field("limits", &self.limits)
            .finish_non_exhaustive()
    }
}

/// [`SpillStore`] with boxed futures, so a config can hold any store.
pub(super) trait ErasedSpill: Send + Sync {
    fn put<'a>(
        &'a self,
        session: SessionId,
        call: &'a str,
        bytes: Vec<u8>,
    ) -> BoxFuture<'a, Result<(), ContextError>>;
    fn read<'a>(
        &'a self,
        session: SessionId,
        call: &'a str,
        offset: u64,
        len: usize,
    ) -> BoxFuture<'a, Result<Vec<u8>, ContextError>>;
    fn copy_session(
        &self,
        from: SessionId,
        to: SessionId,
    ) -> BoxFuture<'_, Result<(), ContextError>>;
}

impl<S: SpillStore> ErasedSpill for S {
    fn put<'a>(
        &'a self,
        session: SessionId,
        call: &'a str,
        bytes: Vec<u8>,
    ) -> BoxFuture<'a, Result<(), ContextError>> {
        Box::pin(SpillStore::put(self, session, call, bytes))
    }
    fn read<'a>(
        &'a self,
        session: SessionId,
        call: &'a str,
        offset: u64,
        len: usize,
    ) -> BoxFuture<'a, Result<Vec<u8>, ContextError>> {
        Box::pin(SpillStore::read(self, session, call, offset, len))
    }
    fn copy_session(
        &self,
        from: SessionId,
        to: SessionId,
    ) -> BoxFuture<'_, Result<(), ContextError>> {
        Box::pin(SpillStore::copy_session(self, from, to))
    }
}

/// An erased store seen as a [`SpillStore`] again, for the context crate's functions.
pub(super) struct Erased(pub(super) Arc<dyn ErasedSpill>);

impl SpillStore for Erased {
    fn put(
        &self,
        session: SessionId,
        call: &str,
        bytes: Vec<u8>,
    ) -> impl Future<Output = Result<(), ContextError>> + Send {
        // The context crate's futures do not borrow the call id: own it.
        let (inner, call) = (self.0.clone(), call.to_owned());
        async move { inner.put(session, &call, bytes).await }
    }
    fn read(
        &self,
        session: SessionId,
        call: &str,
        offset: u64,
        len: usize,
    ) -> impl Future<Output = Result<Vec<u8>, ContextError>> + Send {
        let (inner, call) = (self.0.clone(), call.to_owned());
        async move { inner.read(session, &call, offset, len).await }
    }
    fn copy_session(
        &self,
        from: SessionId,
        to: SessionId,
    ) -> impl Future<Output = Result<(), ContextError>> + Send {
        self.0.copy_session(from, to)
    }
}

/// The tools of a session and the permission rules over them. Implementations may borrow (a host's toolbox that
/// lives for one request); calls are made one at a time.
pub trait Tools: Send + Sync {
    /// The permission rules' verdict on a call. IO-free.
    fn evaluate(&self, tool: &str, input: &RawJson) -> Verdict;

    /// Runs a call. `stop` fires when the host interrupts the turn; a side-effecting tool should return soon after,
    /// a read-only one is simply dropped.
    fn run(
        &self,
        call: CallId,
        tool: Box<str>,
        input: RawJson,
        stop: StopSignal,
    ) -> BoxFuture<'_, ToolOutput>;
}

/// The host's answer to a permission question.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AskReply {
    /// Decided now.
    Decided(Decision),
    /// The host asks a person: the answer comes later as [`sarathi_protocol::Op::ToolDecision`].
    Pending,
}

/// Who answers when a permission rule says to ask.
pub trait PermissionHost: Send + Sync {
    /// Asked once per call that needs a decision.
    fn ask(&self, call: &CallId, tool: &str, input: &RawJson) -> AskReply;
}

/// A host with no one to ask: every question is answered with a denial.
#[derive(Debug, Clone, Copy, Default)]
pub struct DenyAsks;

impl PermissionHost for DenyAsks {
    fn ask(&self, _call: &CallId, _tool: &str, _input: &RawJson) -> AskReply {
        AskReply::Decided(Decision::Deny {
            reason: "no one is available to allow this call".into(),
        })
    }
}

/// A host whose questions are answered later, through ops.
#[derive(Debug, Clone, Copy, Default)]
pub struct AskByOp;

impl PermissionHost for AskByOp {
    fn ask(&self, _call: &CallId, _tool: &str, _input: &RawJson) -> AskReply {
        AskReply::Pending
    }
}

/// Fires once, when an interrupt asks a running tool to stop.
#[derive(Debug, Clone, Default)]
pub struct StopSignal(Arc<StopInner>);

#[derive(Debug, Default)]
struct StopInner {
    fired: AtomicBool,
    notify: Notify,
}

impl StopSignal {
    /// A signal that has not fired.
    pub fn new() -> Self {
        StopSignal::default()
    }

    /// A signal that has already fired: for a host testing how its tool stops.
    pub fn fired() -> Self {
        let s = StopSignal::default();
        s.fire();
        s
    }

    /// Whether it fired.
    pub fn is_stopped(&self) -> bool {
        self.0.fired.load(Ordering::Acquire)
    }

    /// Waits until it fires.
    pub async fn stopped(&self) {
        loop {
            // Registered before the check, so a fire between the check and the wait is not missed.
            let mut notified = std::pin::pin!(self.0.notify.notified());
            notified.as_mut().enable();
            if self.is_stopped() {
                return;
            }
            notified.await;
        }
    }

    pub(crate) fn fire(&self) {
        self.0.fired.store(true, Ordering::Release);
        self.0.notify.notify_waiters();
    }
}
