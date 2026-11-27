// SPDX-License-Identifier: Apache-2.0
//! Scripted, record and replay decorators over a [`ModelCaller`].
//!
//! [`Recording`] wraps a live caller and keeps every call as a cassette [`Entry`] (the request body as sent, the answer
//! as received). [`Replaying`] answers the same calls from a [`Cassette`] without touching a network: each call must
//! carry the next recorded request byte for byte, or it fails naming the first differing field.
//!
//! What a cassette keeps of an answer: the assembled message body (a streamed answer is stored as the message the
//! stream assembled to, not as its frames) and a failure's kind class (by HTTP status when the server answered, a cut
//! stream otherwise). The cost of a call is not stored: the runner prices a run from its usage.
//!
//! A replay is not a byte-for-byte copy of the recorded log: a failure other than an HTTP status comes back as a cut
//! stream (`Network`), an answer carries no `request_id`, `sse` or `retries`, and calls are matched in the order
//! `send` is reached (the driver's calls are sequential, so that is the recorded order). The requests themselves match.
//! Native token counts are not recorded; [`Recording`] reports none, so recording and replay both use the estimate.

use std::collections::VecDeque;
use std::ops::Range;
use std::sync::{Arc, Mutex, MutexGuard};

use sarathi_core::ToolDef;
use sarathi_core::driver::{BoxFuture, CallFailure, ModelCaller, Received};
use sarathi_log::RequestParts;
use sarathi_protocol::{CallId, ErrorKind, RawJson};
use sarathi_provider::wire::classify::kind_of_status;

use crate::cassette::{Cassette, CassetteHeader, Entry, Response, match_request};

/// A request ready to send, with its body kept for the cassette.
#[derive(Debug)]
pub struct Sent<P> {
    request: String,
    inner: P,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    // A panic while holding the lock leaves plain data behind; carry on with it.
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn request_text(body: &[u8]) -> Result<String, CallFailure> {
    String::from_utf8(body.to_vec())
        .map_err(|_| CallFailure::new(ErrorKind::InvalidRequest, "a request body is not UTF-8"))
}

/// Records every call of the wrapped caller.
#[derive(Debug)]
pub struct Recording<C> {
    inner: C,
    entries: Arc<Mutex<Vec<Entry>>>,
}

impl<C> Recording<C> {
    /// Records the calls of `inner`.
    pub fn new(inner: C) -> Self {
        Recording {
            inner,
            entries: Arc::default(),
        }
    }

    /// The calls so far as a cassette for `model`, recorded on `recorded_on` (a date).
    #[must_use]
    pub fn cassette(&self, model: &str, recorded_on: &str) -> Cassette {
        Cassette {
            header: CassetteHeader::new(model, recorded_on),
            entries: lock(&self.entries).clone(),
        }
    }
}

impl<C: ModelCaller> ModelCaller for Recording<C> {
    type Prepared = Sent<C::Prepared>;

    fn protocol(&self) -> &str {
        self.inner.protocol()
    }

    fn params(&self) -> RawJson {
        self.inner.params()
    }

    fn render_tools(&self, tools: &[ToolDef]) -> Result<Option<RawJson>, CallFailure> {
        self.inner.render_tools(tools)
    }

    fn result_block(&self, call: &CallId, content: &RawJson, is_error: bool) -> RawJson {
        self.inner.result_block(call, content, is_error)
    }

    fn prepare(&self, parts: &RequestParts<'_>) -> Result<(Vec<u8>, Self::Prepared), CallFailure> {
        let (body, inner) = self.inner.prepare(parts)?;
        let request = request_text(&body)?;
        Ok((body, Sent { request, inner }))
    }

    fn send(&self, sent: Self::Prepared) -> BoxFuture<'_, Result<Received, CallFailure>> {
        let Sent { request, inner } = sent;
        let call = self.inner.send(inner);
        Box::pin(async move {
            let result = call.await;
            let response = match &result {
                Ok(r) => Response::Body {
                    status: 200,
                    body: r.body.as_str().to_owned(),
                },
                Err(f) => match f.status {
                    Some(status) => Response::Body {
                        status,
                        body: f.detail.to_string(),
                    },
                    None => Response::Stream {
                        status: 200,
                        frames: Vec::new(),
                        complete: false,
                    },
                },
            };
            let mut entries = lock(&self.entries);
            let seq = u32::try_from(entries.len()).unwrap_or(u32::MAX);
            entries.push(Entry::new(seq, &request, response));
            drop(entries);
            result
        })
    }

    fn read(&self, body: &RawJson, content: Range<u32>) -> Result<Received, CallFailure> {
        self.inner.read(body, content)
    }

    /// Always `None`: a native token count is not kept in the cassette, so a recorded run decides near the context
    /// limit from the estimate, exactly as its replay will. Forwarding it would let the recording compact where the
    /// replay does not, and the replay would then fail with a request mismatch.
    fn count_tokens(
        &self,
        _parts: &RequestParts<'_>,
    ) -> Option<BoxFuture<'_, Result<u64, CallFailure>>> {
        None
    }
}

/// Reads a recorded answer body as an answer (for the Messages API, `sarathi_core::anthropic::answer`).
pub type ParseAnswer = fn(&RawJson) -> Result<Received, CallFailure>;

/// Answers the wrapped caller's calls from a cassette, in order, without a network call.
///
/// The wrapped caller still builds the requests (the same log gives the same bytes), so a changed prompt, tool list or
/// parameter shows as a mismatch on the call where it first matters.
#[derive(Debug)]
pub struct Replaying<C> {
    inner: C,
    parse: ParseAnswer,
    queue: Mutex<VecDeque<Entry>>,
}

impl<C> Replaying<C> {
    /// Replays `cassette` through `inner`'s request building; `parse` reads a recorded body as an answer.
    pub fn new(inner: C, cassette: Cassette, parse: ParseAnswer) -> Self {
        Replaying {
            inner,
            parse,
            queue: Mutex::new(cassette.entries.into()),
        }
    }

    /// Recorded calls not yet replayed. A finished run that leaves some behind took a different path.
    #[must_use]
    pub fn remaining(&self) -> usize {
        lock(&self.queue).len()
    }
}

impl<C: ModelCaller> ModelCaller for Replaying<C> {
    type Prepared = Sent<C::Prepared>;

    fn protocol(&self) -> &str {
        self.inner.protocol()
    }

    fn params(&self) -> RawJson {
        self.inner.params()
    }

    fn render_tools(&self, tools: &[ToolDef]) -> Result<Option<RawJson>, CallFailure> {
        self.inner.render_tools(tools)
    }

    fn result_block(&self, call: &CallId, content: &RawJson, is_error: bool) -> RawJson {
        self.inner.result_block(call, content, is_error)
    }

    fn prepare(&self, parts: &RequestParts<'_>) -> Result<(Vec<u8>, Self::Prepared), CallFailure> {
        let (body, inner) = self.inner.prepare(parts)?;
        let request = request_text(&body)?;
        Ok((body, Sent { request, inner }))
    }

    fn send(&self, sent: Self::Prepared) -> BoxFuture<'_, Result<Received, CallFailure>> {
        let next = lock(&self.queue).pop_front();
        let parse = self.parse;
        Box::pin(async move {
            let Some(entry) = next else {
                return Err(CallFailure::new(
                    ErrorKind::InvalidRequest,
                    "the cassette has no more calls: the run made one the recording did not",
                ));
            };
            match_request(&entry, &sent.request).map_err(|e| {
                CallFailure::new(
                    ErrorKind::InvalidRequest,
                    format!("call {} does not match the cassette: {e}", entry.seq),
                )
            })?;
            match entry.response {
                Response::Body { status, body } if status < 300 => {
                    let raw = RawJson::from_string(body).map_err(|e| {
                        CallFailure::new(ErrorKind::Malformed, format!("recorded body: {e}"))
                    })?;
                    parse(&raw)
                }
                Response::Body { status, body } => {
                    let mut f = CallFailure::new(kind_of_status(status), body);
                    f.status = Some(status);
                    Err(f)
                }
                Response::Stream { .. } => Err(CallFailure::new(
                    ErrorKind::Network,
                    "the recorded stream was cut before the answer was complete",
                )),
            }
        })
    }

    fn read(&self, body: &RawJson, content: Range<u32>) -> Result<Received, CallFailure> {
        self.inner.read(body, content)
    }
}

/// Answers the wrapped caller's calls from a list, in order, and keeps every request body it was sent.
///
/// The wrapped caller still builds the requests, so a test sees the bytes of the real protocol; only the call itself is
/// scripted. A run that makes more calls than the script has fails rather than reaching a network.
///
/// It offers no native token count (the default `count_tokens`), so a scripted run decides near the context limit from
/// the estimate, as a recording and its replay do.
#[derive(Debug)]
pub struct Scripted<C> {
    inner: C,
    replies: Mutex<VecDeque<Result<Received, CallFailure>>>,
    sent: Mutex<Vec<String>>,
}

impl<C> Scripted<C> {
    /// Scripts `replies` (an answer or a failure per call) through `inner`'s request building.
    pub fn new(inner: C, replies: impl IntoIterator<Item = Result<Received, CallFailure>>) -> Self {
        Scripted {
            inner,
            replies: Mutex::new(replies.into_iter().collect()),
            sent: Mutex::default(),
        }
    }

    /// The request bodies sent so far, in call order.
    #[must_use]
    pub fn requests(&self) -> Vec<String> {
        lock(&self.sent).clone()
    }

    /// Scripted replies not yet used.
    #[must_use]
    pub fn remaining(&self) -> usize {
        lock(&self.replies).len()
    }
}

impl<C: ModelCaller> ModelCaller for Scripted<C> {
    type Prepared = Sent<C::Prepared>;

    fn protocol(&self) -> &str {
        self.inner.protocol()
    }

    fn params(&self) -> RawJson {
        self.inner.params()
    }

    fn render_tools(&self, tools: &[ToolDef]) -> Result<Option<RawJson>, CallFailure> {
        self.inner.render_tools(tools)
    }

    fn result_block(&self, call: &CallId, content: &RawJson, is_error: bool) -> RawJson {
        self.inner.result_block(call, content, is_error)
    }

    fn prepare(&self, parts: &RequestParts<'_>) -> Result<(Vec<u8>, Self::Prepared), CallFailure> {
        let (body, inner) = self.inner.prepare(parts)?;
        let request = request_text(&body)?;
        Ok((body, Sent { request, inner }))
    }

    fn send(&self, sent: Self::Prepared) -> BoxFuture<'_, Result<Received, CallFailure>> {
        lock(&self.sent).push(sent.request);
        let next = lock(&self.replies).pop_front();
        Box::pin(async move {
            next.unwrap_or_else(|| {
                Err(CallFailure::new(
                    ErrorKind::InvalidRequest,
                    "the script has no more replies: the run made a call the test did not expect",
                ))
            })
        })
    }

    fn read(&self, body: &RawJson, content: Range<u32>) -> Result<Received, CallFailure> {
        self.inner.read(body, content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sarathi_core::anthropic::{AnthropicCaller, answer};
    use sarathi_log::Slice;
    use sarathi_protocol::Seq;
    use sarathi_provider::{Chain, Client, Slot, Target};

    const ANSWER: &str = r#"{"id":"m1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":1}}"#;

    /// A caller that answers from a list and builds no request itself: the tests hand `send` a [`Sent`] directly. It
    /// offers a native token count, as a live provider caller does.
    struct Live(Mutex<VecDeque<Result<Received, CallFailure>>>);

    impl ModelCaller for Live {
        type Prepared = ();
        fn protocol(&self) -> &str {
            "anthropic-messages"
        }
        fn params(&self) -> RawJson {
            RawJson::null()
        }
        fn render_tools(&self, _: &[ToolDef]) -> Result<Option<RawJson>, CallFailure> {
            Ok(None)
        }
        fn result_block(&self, _: &CallId, content: &RawJson, _: bool) -> RawJson {
            content.clone()
        }
        fn prepare(&self, _: &RequestParts<'_>) -> Result<(Vec<u8>, ()), CallFailure> {
            Err(CallFailure::new(ErrorKind::InvalidRequest, "unused"))
        }
        fn send(&self, (): ()) -> BoxFuture<'_, Result<Received, CallFailure>> {
            let r = lock(&self.0).pop_front();
            Box::pin(async move {
                r.unwrap_or_else(|| Err(CallFailure::new(ErrorKind::Upstream, "empty")))
            })
        }
        fn read(&self, body: &RawJson, _: Range<u32>) -> Result<Received, CallFailure> {
            answer(body)
        }
        fn count_tokens(
            &self,
            _: &RequestParts<'_>,
        ) -> Option<BoxFuture<'_, Result<u64, CallFailure>>> {
            Some(Box::pin(async { Ok(1) }))
        }
    }

    fn sent(request: &str) -> Sent<()> {
        Sent {
            request: request.into(),
            inner: (),
        }
    }

    fn ok() -> Result<Received, CallFailure> {
        answer(&RawJson::from_text(ANSWER).unwrap())
    }

    fn failure(status: Option<u16>) -> Result<Received, CallFailure> {
        let mut f = CallFailure::new(ErrorKind::Upstream, "overloaded");
        f.status = status;
        Err(f)
    }

    #[tokio::test]
    async fn every_decorator_sends_the_request_a_real_caller_builds() {
        let real = || {
            let client = Client::new(Target::anthropic("sk-test")).unwrap();
            let chain = Chain::new(vec![Slot::new("main", client, "m")]).unwrap();
            AnthropicCaller::new(chain, 256).unwrap()
        };
        let (null, params) = (RawJson::null(), real().params());
        let content = RawJson::from_text(r#"[{"type":"text","text":"hi"}]"#).unwrap();
        let parts = RequestParts {
            protocol: "anthropic-messages",
            system: &null,
            tools: &null,
            messages: vec![(Seq(1), Slice::User { content: &content })],
            params: &params,
        };
        let (body, _) = real().prepare(&parts).unwrap();
        let text = String::from_utf8(body.clone()).unwrap();
        assert!(text.contains(content.as_str()), "{text}");

        let rec = Recording::new(real());
        let (recorded, held) = rec.prepare(&parts).unwrap();
        assert_eq!((&recorded, &held.request), (&body, &text));

        let rep = Replaying::new(real(), rec.cassette("m", "2026-11-27"), answer);
        let (replayed, held) = rep.prepare(&parts).unwrap();
        assert_eq!((&replayed, &held.request), (&body, &text));

        // Only the script is sent: a real caller's own `send` would reach the network.
        let script = Scripted::new(real(), [ok()]);
        let (scripted, held) = script.prepare(&parts).unwrap();
        assert_eq!(scripted, body);
        script.send(held).await.unwrap();
        assert_eq!(script.requests(), [text]);
    }

    #[tokio::test]
    async fn a_recorded_run_replays_to_the_same_answers_and_failures() {
        let rec = Recording::new(Live(Mutex::new(
            vec![ok(), failure(Some(503)), failure(None)].into(),
        )));
        let first = rec.send(sent(r#"{"a":1}"#)).await.unwrap();
        assert!(rec.send(sent(r#"{"a":2}"#)).await.is_err());
        assert!(rec.send(sent(r#"{"a":3}"#)).await.is_err());
        let cassette = rec.cassette("m", "2026-11-26");
        assert_eq!(cassette.entries.len(), 3);

        let rep = Replaying::new(Live(Mutex::new(VecDeque::new())), cassette, answer);
        let again = rep.send(sent(r#"{"a":1}"#)).await.unwrap();
        assert_eq!(again.body, first.body);
        assert_eq!(again.content, first.content);
        let f = rep.send(sent(r#"{"a":2}"#)).await.unwrap_err();
        assert_eq!((f.kind, f.status), (ErrorKind::Upstream, Some(503)));
        let f = rep.send(sent(r#"{"a":3}"#)).await.unwrap_err();
        assert_eq!(f.kind, ErrorKind::Network);
        assert_eq!(rep.remaining(), 0);
    }

    #[tokio::test]
    async fn a_changed_request_names_its_field_and_a_longer_run_is_refused() {
        let rec = Recording::new(Live(Mutex::new(vec![ok()].into())));
        rec.send(sent(r#"{"system":"a","n":1}"#)).await.unwrap();
        let rep = Replaying::new(
            Live(Mutex::new(VecDeque::new())),
            rec.cassette("m", "2026-11-26"),
            answer,
        );
        let f = rep.send(sent(r#"{"system":"b","n":1}"#)).await.unwrap_err();
        assert_eq!(f.kind, ErrorKind::InvalidRequest);
        assert!(f.detail.contains("system"), "{}", f.detail);
        let f = rep.send(sent("{}")).await.unwrap_err();
        assert!(f.detail.contains("no more calls"), "{}", f.detail);
    }

    #[tokio::test]
    async fn a_recording_round_trips_through_the_cassette_file_format() {
        let rec = Recording::new(Live(Mutex::new(vec![ok()].into())));
        rec.send(sent(r#"{"n":1}"#)).await.unwrap();
        let text = crate::cassette::write(&rec.cassette("m", "2026-11-26")).unwrap();
        let parsed = crate::cassette::parse(&text).unwrap();
        let rep = Replaying::new(Live(Mutex::new(VecDeque::new())), parsed, answer);
        assert!(rep.send(sent(r#"{"n":1}"#)).await.is_ok());
    }

    #[tokio::test]
    async fn a_script_answers_in_order_keeps_requests_and_refuses_a_longer_run() {
        let s = Scripted::new(
            Live(Mutex::new(VecDeque::new())),
            [ok(), failure(Some(529))],
        );
        assert!(s.send(sent(r#"{"n":1}"#)).await.is_ok());
        let f = s.send(sent(r#"{"n":2}"#)).await.unwrap_err();
        assert_eq!((f.kind, f.status), (ErrorKind::Upstream, Some(529)));
        assert_eq!(s.remaining(), 0);
        let f = s.send(sent(r#"{"n":3}"#)).await.unwrap_err();
        assert!(f.detail.contains("no more replies"), "{}", f.detail);
        assert_eq!(s.requests(), [r#"{"n":1}"#, r#"{"n":2}"#, r#"{"n":3}"#]);
    }

    #[test]
    fn no_decorator_forwards_the_native_count_so_a_recording_decides_like_its_replay() {
        let live = || Live(Mutex::new(VecDeque::new()));
        let rec = Recording::new(live());
        let rep = Replaying::new(live(), rec.cassette("m", "2026-11-26"), answer);
        let script = Scripted::new(live(), []);
        let (null, tools) = (RawJson::null(), RawJson::null());
        let parts = RequestParts {
            protocol: "anthropic-messages",
            system: &null,
            tools: &tools,
            messages: Vec::new(),
            params: &null,
        };
        assert!(live().count_tokens(&parts).is_some());
        assert!(rec.count_tokens(&parts).is_none());
        assert!(rep.count_tokens(&parts).is_none());
        assert!(script.count_tokens(&parts).is_none());
    }
}
