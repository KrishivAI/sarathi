// SPDX-License-Identifier: Apache-2.0
//! Test seams for the driver: a scripted model on the Anthropic wire format, tools that count their calls, and a
//! store that stops accepting appends after a given number.
#![expect(
    missing_docs,
    reason = "test support, documented where it is not obvious"
)]

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use sarathi_core::anthropic::{self, PROTOCOL};
use sarathi_core::driver::{
    BoxFuture, CallFailure, Driver, DriverConfig, ModelCaller, PermissionHost, Received,
    StopSignal, Tools,
};
use sarathi_core::{Effect, Limits, ModelProfile, ToolDef, Verdict};
use sarathi_log::{
    Body, Entry, ForkPoint, LogError, LogHeader, LogStore, MemoryStore, RequestParts,
};
use sarathi_log::{Durability, Loaded, Log, SessionId};
use sarathi_protocol::{CallId, ErrorKind, Event, Op, RawJson, Seq, ToolOutput};
use sarathi_provider::{Endpoint, MessagesRequest};

pub fn raw(s: &str) -> RawJson {
    RawJson::from_text(s).unwrap()
}

/// One scripted answer.
#[derive(Debug, Clone)]
pub enum Reply {
    /// A Messages answer body.
    Body(String),
    /// The call fails with this kind.
    Fail(ErrorKind),
    /// The call answers after this long (on the test clock).
    Slow(Duration, String),
}

/// A Messages answer: optional text, then `tool_use` blocks `(id, name, input)`.
pub fn native(text: Option<&str>, calls: &[(&str, &str, &str)], stop: &str) -> Reply {
    let mut blocks: Vec<String> = Vec::new();
    if let Some(t) = text {
        blocks.push(format!(
            r#"{{"type":"text","text":{}}}"#,
            RawJson::string(t).as_str()
        ));
    }
    for (id, name, input) in calls {
        blocks.push(format!(
            r#"{{"type":"tool_use","id":"{id}","name":"{name}","input":{input}}}"#
        ));
    }
    Reply::Body(message(&blocks.join(","), stop))
}

/// A Messages answer whose text is `text` (a JSON step under the fallback protocol).
pub fn text(text: &str) -> Reply {
    native(Some(text), &[], "end_turn")
}

pub fn message(content: &str, stop: &str) -> String {
    format!(
        r#"{{"id":"msg_01","type":"message","role":"assistant","model":"scripted","content":[{content}],"stop_reason":"{stop}","usage":{{"input_tokens":10,"output_tokens":5}}}}"#
    )
}

/// The model: answers in order, and keeps every body it was sent.
#[derive(Debug, Default)]
pub struct Script {
    replies: Mutex<VecDeque<Reply>>,
    pub bodies: Mutex<Vec<Vec<u8>>>,
}

#[derive(Debug, Clone, Default)]
pub struct Scripted(pub Arc<Script>);

impl Scripted {
    pub fn new(replies: Vec<Reply>) -> Self {
        Scripted(Arc::new(Script {
            replies: Mutex::new(replies.into()),
            bodies: Mutex::new(Vec::new()),
        }))
    }

    pub fn bodies(&self) -> Vec<Vec<u8>> {
        self.0.bodies.lock().unwrap().clone()
    }

    pub fn left(&self) -> usize {
        self.0.replies.lock().unwrap().len()
    }
}

impl ModelCaller for Scripted {
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
        self.0.bodies.lock().unwrap().push(body);
        let reply = self.0.replies.lock().unwrap().pop_front();
        Box::pin(async move {
            match reply {
                None => Err(CallFailure::new(ErrorKind::Upstream, "script exhausted")),
                Some(Reply::Fail(kind)) => Err(CallFailure::new(kind, "scripted failure")),
                Some(Reply::Body(b)) => anthropic::answer(&raw(&b)),
                Some(Reply::Slow(d, b)) => {
                    tokio::time::sleep(d).await;
                    anthropic::answer(&raw(&b))
                }
            }
        })
    }

    fn read(&self, body: &RawJson, content: Range<u32>) -> Result<Received, CallFailure> {
        let mut r = anthropic::answer(body)?;
        r.content = content;
        Ok(r)
    }
}

/// The body the Anthropic assembler writes for `parts`.
pub fn assemble(parts: &RequestParts<'_>) -> Vec<u8> {
    let req: MessagesRequest = anthropic::request(parts).unwrap();
    req.body(Endpoint::Messages)
}

/// What the test tools do.
#[derive(Debug, Default)]
pub struct Tally {
    /// Every call that ran: `(tool, input)`.
    pub ran: Mutex<Vec<(String, String)>>,
    /// Runs of the side-effecting tool.
    pub writes: AtomicU32,
}

/// `echo` and `look` read, `edit` writes. Verdicts default to allow.
#[derive(Debug, Clone, Default)]
pub struct TestTools {
    pub tally: Arc<Tally>,
    pub verdicts: HashMap<String, Verdict>,
    /// `edit` refuses with a readable error.
    pub edit_refuses: bool,
    /// `edit` takes this long; with `edit_stops`, it returns early when asked to stop.
    pub edit_takes: Option<Duration>,
    pub edit_stops: bool,
    /// A file whose number `edit` increments: the side effect a crash test counts across processes.
    pub counter: Option<PathBuf>,
    /// `edit` aborts the process right after its effect, before it returns.
    pub abort_after_effect: bool,
}

pub fn defs() -> Vec<ToolDef> {
    let obj = || raw(r#"{"type":"object"}"#);
    vec![
        ToolDef::new("echo", "repeats the input back", obj(), Effect::Read),
        ToolDef::new("look", "shows the page", obj(), Effect::Read),
        ToolDef::new("edit", "changes the page", obj(), Effect::Write),
    ]
}

impl Tools for TestTools {
    fn evaluate(&self, tool: &str, _input: &RawJson) -> Verdict {
        self.verdicts.get(tool).cloned().unwrap_or(Verdict::Allow)
    }

    fn run(
        &self,
        _call: CallId,
        tool: Box<str>,
        input: RawJson,
        stop: StopSignal,
    ) -> BoxFuture<'_, ToolOutput> {
        self.tally
            .ran
            .lock()
            .unwrap()
            .push((tool.to_string(), input.as_str().to_owned()));
        Box::pin(async move {
            let ok = |s: &str| ToolOutput {
                content: RawJson::string(s),
                is_error: false,
            };
            match &*tool {
                "echo" => ok(&format!("echoed: {}", input.as_str())),
                "look" => ok("seen"),
                "edit" if self.edit_refuses => ToolOutput {
                    content: RawJson::string("that id already exists"),
                    is_error: true,
                },
                "edit" => {
                    self.tally.writes.fetch_add(1, Ordering::SeqCst);
                    if let Some(path) = &self.counter {
                        let n: u32 = std::fs::read_to_string(path)
                            .ok()
                            .and_then(|s| s.trim().parse().ok())
                            .unwrap_or(0);
                        std::fs::write(path, (n + 1).to_string()).unwrap();
                    }
                    if self.abort_after_effect {
                        std::process::abort();
                    }
                    if let Some(d) = self.edit_takes {
                        if self.edit_stops {
                            tokio::select! {
                                () = tokio::time::sleep(d) => {}
                                () = stop.stopped() => return ok("stopped early"),
                            }
                        } else {
                            tokio::time::sleep(d).await;
                        }
                    }
                    ok("edited")
                }
                other => panic!("no tool {other}"),
            }
        })
    }
}

/// Answers every question with this decision, or leaves it pending.
#[derive(Debug, Clone)]
pub struct Host(pub Option<sarathi_protocol::Decision>);

impl PermissionHost for Host {
    fn ask(&self, _: &CallId, _: &str, _: &RawJson) -> sarathi_core::driver::AskReply {
        match &self.0 {
            Some(d) => sarathi_core::driver::AskReply::Decided(d.clone()),
            None => sarathi_core::driver::AskReply::Pending,
        }
    }
}

/// A memory store shared between drivers; after `limit` appends it refuses every write, as a process that died
/// at that point would have stopped writing.
#[derive(Debug, Clone, Default)]
pub struct Shared {
    pub inner: Arc<MemoryStore>,
    pub appends: Arc<AtomicU32>,
    pub limit: Option<u32>,
    /// Refuses only this append, then works again.
    pub fail_once: Option<u32>,
}

impl Shared {
    pub fn stopping_after(&self, limit: u32) -> Shared {
        Shared {
            inner: self.inner.clone(),
            appends: Arc::new(AtomicU32::new(0)),
            limit: Some(limit),
            fail_once: None,
        }
    }

    pub fn reopened(&self) -> Shared {
        Shared {
            inner: self.inner.clone(),
            appends: Arc::new(AtomicU32::new(0)),
            limit: None,
            fail_once: None,
        }
    }
}

impl LogStore for Shared {
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
        let n = self.appends.fetch_add(1, Ordering::SeqCst);
        if self.limit.is_some_and(|l| n >= l) || self.fail_once == Some(n) {
            return Err(LogError::Io(std::io::Error::other("the process is gone")));
        }
        self.inner
            .append(session, expected, entries, durability)
            .await
    }

    fn load(&self, session: SessionId) -> impl Future<Output = Result<Loaded, LogError>> + Send {
        self.inner.load(session)
    }
}

pub fn config(profile: ModelProfile) -> DriverConfig {
    DriverConfig::new(profile, defs(), "You are a test.")
}

pub fn limits(max_steps: u32) -> Limits {
    Limits::with_max_steps(max_steps)
}

pub fn user(text: &str) -> Op {
    Op::UserMessage {
        content: RawJson::string(text),
    }
}

/// Collects events.
pub fn sink(events: &mut Vec<Event>) -> impl FnMut(Event) + Send + '_ {
    move |e| events.push(e)
}

/// The log's tool results as `(call, block, is_error)`.
pub fn results(log: &Log) -> Vec<(String, String, bool)> {
    log.entries()
        .iter()
        .filter_map(|e| match &e.body {
            Body::ToolResult {
                call,
                block,
                is_error,
                ..
            } => Some((call.to_string(), block.as_str().to_owned(), *is_error)),
            _ => None,
        })
        .collect()
}

/// The kinds of the log's entries, in order.
pub fn kinds(log: &Log) -> Vec<String> {
    log.entries()
        .iter()
        .map(|e| e.body.kind().as_str().to_owned())
        .collect()
}

/// Every tool use the log's assistant messages asked for has exactly one result, and every request rebuilds to
/// the hash it records.
pub fn check_log(log: &Log) {
    let mut asked: Vec<String> = Vec::new();
    for e in log.entries() {
        if let Body::AssistantMessage { calls, .. } = &e.body {
            asked.extend(calls.iter().map(|c| c.id.to_string()));
        }
    }
    let answered: Vec<String> = results(log).into_iter().map(|r| r.0).collect();
    for id in &asked {
        assert_eq!(
            answered.iter().filter(|a| *a == id).count(),
            1,
            "call {id} needs exactly one result: {:#?}",
            kinds(log)
        );
    }
    log.verify_requests(&|parts| assemble(parts)).unwrap();
}

pub type TestDriver = Driver<Shared, Scripted, TestTools, Host>;
