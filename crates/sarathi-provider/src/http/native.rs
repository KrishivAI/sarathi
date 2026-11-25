// SPDX-License-Identifier: Apache-2.0
//! The Anthropic-native calls: `messages`, `messages_stream` and `count_tokens`.
//!
//! The body is [`MessagesRequest::body`], sent as is; the answer is kept as received ([`MessagesResponse::raw`]) or,
//! for a stream, assembled once from the SSE events ([`Assembler`]) with the stream's bytes kept alongside.

use std::time::Duration;

use bytes::Bytes;
use futures::{Stream, StreamExt};

use crate::http::client::{Client, check_redirect, read};
use crate::http::error::{Error, ErrorKind};
use crate::http::target::Target;
use crate::wire::native::response::parse;
use crate::wire::native::{
    Assembler, Endpoint, MessagesRequest, MessagesResponse, NativeError, RateLimitHeaders,
};
use crate::wire::provider::{ProviderKind, provider_error};
use crate::wire::sse::SseEvent;

/// One item of a native stream: each event as it completes (for a host to show), then the assembled message.
#[derive(Debug, Clone, PartialEq)]
pub enum NativeEvent {
    /// A complete SSE event, as parsed.
    Event(SseEvent),
    /// The assembled message; the last item of a stream that did not fail.
    Done(Box<MessagesResponse>),
}

/// A `count_tokens` answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenCount {
    /// The input tokens the request would use.
    pub input_tokens: u64,
    /// The answer body as received.
    pub raw: Bytes,
}

/// Lower-cased fragments of the API's 400 message when an earlier turn's thinking blocks do not match what it sent.
/// The last is the signature check (an edited thinking text no longer matches its signature).
const HISTORY_MISMATCH: [&[&str]; 3] = [
    &["thinking", "cannot be modified"],
    &["prefix_mismatch"],
    &["invalid", "signature", "thinking"],
];

impl Client {
    /// One non-streaming Messages call. A refusal is an answer, not an error: see [`MessagesResponse::refusal`].
    ///
    /// # Errors
    /// The target is not Anthropic or a gateway; any transport or provider failure; a 400 from the history check is
    /// [`ErrorKind::HistoryMismatch`].
    pub async fn messages(&self, request: &MessagesRequest) -> Result<MessagesResponse, Error> {
        self.messages_inner(request)
            .await
            .map_err(|e| self.scrub(e))
    }

    async fn messages_inner(&self, request: &MessagesRequest) -> Result<MessagesResponse, Error> {
        let resp = self.native_send(request, Endpoint::Messages).await?;
        let (request_id, rate_limits) = meta(resp.headers());
        let (status, retry_after, body) = read(resp, self.max_response_bytes).await?;
        check_status(status, &body, retry_after)?;
        let mut r = parse(body).map_err(from_native)?;
        r.request_id = request_id;
        r.rate_limits = rate_limits;
        Ok(r)
    }

    /// A streamed Messages call: every event in order, then [`NativeEvent::Done`] with the assembled message, or at
    /// most one error. A stream that ends before `message_stop` ends with an error, never silently.
    ///
    /// # Errors
    /// As [`Client::messages`], before the first event; afterwards each failure is the stream's last item.
    pub async fn messages_stream(
        &self,
        request: &MessagesRequest,
    ) -> Result<impl Stream<Item = Result<NativeEvent, Error>> + use<>, Error> {
        self.stream_native(request).await.map_err(|e| self.scrub(e))
    }

    async fn stream_native(
        &self,
        request: &MessagesRequest,
    ) -> Result<impl Stream<Item = Result<NativeEvent, Error>> + use<>, Error> {
        let send = self.native_send(request, Endpoint::MessagesStream);
        let resp = match tokio::time::timeout(self.timeout, send).await {
            Ok(r) => r?,
            Err(_) => return Err(Error::new(ErrorKind::Timeout, "the request timed out")),
        };
        let (request_id, rate_limits) = meta(resp.headers());
        if resp.status().as_u16() >= 300 {
            let read = read(resp, self.max_response_bytes);
            let (status, retry_after, body) = match tokio::time::timeout(self.timeout, read).await {
                Ok(r) => r?,
                Err(_) => return Err(Error::new(ErrorKind::Timeout, "the request timed out")),
            };
            check_status(status, &body, retry_after)?;
            return Err(Error::new(ErrorKind::Malformed, "unexpected answer"));
        }
        let key = self.target.key().to_string();
        let scrub = move |e: Error| e.scrubbed(Some(&key));
        let idle = self.timeout;
        let mut asm = Assembler::new(self.max_response_bytes);
        Ok(async_stream::stream! {
            let mut body = Box::pin(resp.bytes_stream());
            while !asm.is_complete() {
                let next = match tokio::time::timeout(idle, body.next()).await {
                    Ok(next) => next,
                    Err(_) => {
                        yield Err(scrub(Error::new(ErrorKind::Timeout, "the stream went quiet")));
                        return;
                    }
                };
                let chunk = match next {
                    Some(Ok(chunk)) => chunk,
                    Some(Err(e)) => {
                        yield Err(scrub(Error::from_reqwest(e)));
                        return;
                    }
                    None => break,
                };
                match asm.feed(&chunk) {
                    Ok(events) => {
                        for e in events {
                            yield Ok(NativeEvent::Event(e));
                        }
                    }
                    Err(e) => {
                        yield Err(scrub(from_native(e)));
                        return;
                    }
                }
            }
            match asm.finish() {
                Ok(mut r) => {
                    r.request_id = request_id;
                    r.rate_limits = rate_limits;
                    yield Ok(NativeEvent::Done(Box::new(r)));
                }
                Err(e) => yield Err(scrub(from_native(e))),
            }
        })
    }

    /// `POST /v1/messages/count_tokens` with the same slices (it runs the same history check).
    ///
    /// # Errors
    /// As [`Client::messages`]; an answer without `input_tokens` is [`ErrorKind::Malformed`].
    pub async fn count_tokens(&self, request: &MessagesRequest) -> Result<TokenCount, Error> {
        self.count_inner(request).await.map_err(|e| self.scrub(e))
    }

    async fn count_inner(&self, request: &MessagesRequest) -> Result<TokenCount, Error> {
        #[derive(serde::Deserialize)]
        struct Count {
            input_tokens: u64,
        }
        let resp = self.native_send(request, Endpoint::CountTokens).await?;
        let (status, retry_after, body) = read(resp, self.max_response_bytes).await?;
        check_status(status, &body, retry_after)?;
        let c: Count = serde_json::from_slice(&body)
            .map_err(|e| Error::new(ErrorKind::Malformed, format!("not a token count: {e}")))?;
        Ok(TokenCount {
            input_tokens: c.input_tokens,
            raw: body,
        })
    }

    async fn native_send(
        &self,
        request: &MessagesRequest,
        endpoint: Endpoint,
    ) -> Result<reqwest::Response, Error> {
        let url = native_url(&self.target, endpoint)?;
        let mut b = self.http.post(url).body(request.body(endpoint));
        for (k, v) in request.headers() {
            b = b.header(k, v);
        }
        let key = self.target.key();
        if !key.is_empty() {
            b = b.header("x-api-key", key);
        }
        if endpoint == Endpoint::MessagesStream {
            b = b.header("accept", "text/event-stream");
        } else {
            b = b.timeout(self.timeout);
        }
        b.send().await.map_err(Error::from_reqwest)
    }
}

impl MessagesResponse {
    /// The refusal as an [`ErrorKind::Refusal`] error, when the model refused; the message holds `stop_details`.
    pub fn refusal(&self) -> Option<Error> {
        self.is_refusal().then(|| {
            let details = self
                .view
                .stop_details
                .as_ref()
                .map_or_else(String::new, |d| format!(": {d}"));
            Error::new(ErrorKind::Refusal, format!("the model refused{details}"))
        })
    }
}

/// Where a native call goes: `{base}/v1/messages…` for Anthropic, the gateway's root plus the same path for a
/// gateway (it forwards natively).
fn native_url(target: &Target, endpoint: Endpoint) -> Result<String, Error> {
    match target {
        Target::Gateway { base_url, .. } => {
            let base = base_url.trim_end_matches('/');
            let root = base
                .strip_suffix("/v1")
                .unwrap_or(base)
                .trim_end_matches('/');
            Ok(format!("{root}{}", endpoint.path()))
        }
        Target::Provider {
            kind: ProviderKind::Anthropic,
            base_url,
            ..
        } => Ok(format!(
            "{}{}",
            base_url.trim_end_matches('/'),
            endpoint.path()
        )),
        Target::Provider { .. } => Err(Error::new(
            ErrorKind::InvalidRequest,
            "the native Messages call needs an Anthropic or gateway target",
        )),
    }
}

fn meta(headers: &reqwest::header::HeaderMap) -> (Option<String>, RateLimitHeaders) {
    let id = headers
        .get("request-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let limits = headers
        .iter()
        .filter(|(k, _)| RateLimitHeaders::keeps(k.as_str()))
        .filter_map(|(k, v)| Some((k.as_str().to_string(), v.to_str().ok()?.to_string())))
        .collect();
    (id, RateLimitHeaders(limits))
}

/// An answer with status 300 or more as its error; a 400 from the history check as [`ErrorKind::HistoryMismatch`].
fn check_status(status: u16, body: &[u8], retry_after: Option<Duration>) -> Result<(), Error> {
    check_redirect(status)?;
    if status < 300 {
        return Ok(());
    }
    let mut e = Error::from_translate(provider_error(status, body), retry_after);
    let lower = e.message.to_ascii_lowercase();
    if status == 400
        && HISTORY_MISMATCH
            .iter()
            .any(|all| all.iter().all(|f| lower.contains(f)))
    {
        e.kind = ErrorKind::HistoryMismatch;
        e.retryable = false;
    }
    Err(e)
}

pub(super) fn from_native(e: NativeError) -> Error {
    match e {
        NativeError::Invalid(m) => Error::new(ErrorKind::InvalidRequest, m),
        NativeError::Malformed(m) => Error::new(ErrorKind::Malformed, m),
        NativeError::Cut(m) => Error::new(ErrorKind::Network, m),
        e @ NativeError::UnassemblableDelta { .. } => {
            Error::new(ErrorKind::Unassemblable, e.to_string())
        }
        NativeError::Provider {
            status, message, ..
        } => Error::from_status(status, message, None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urls_per_target() {
        let a = Target::anthropic("k").with_base_url("https://x.test/");
        assert_eq!(
            native_url(&a, Endpoint::Messages).unwrap(),
            "https://x.test/v1/messages"
        );
        let g = Target::gateway("https://gw.test/v1/", "k");
        assert_eq!(
            native_url(&g, Endpoint::CountTokens).unwrap(),
            "https://gw.test/v1/messages/count_tokens"
        );
        let e = native_url(&Target::openai("k"), Endpoint::Messages).unwrap_err();
        assert_eq!(e.kind, ErrorKind::InvalidRequest);
    }

    #[test]
    fn history_mismatch_is_its_own_kind() {
        let body = br#"{"type":"error","error":{"type":"invalid_request_error","message":"messages.1.content.0: `thinking` or `redacted_thinking` blocks in the latest assistant message cannot be modified."}}"#;
        let e = check_status(400, body, None).unwrap_err();
        assert_eq!(e.kind, ErrorKind::HistoryMismatch);
        assert!(!e.retryable);
        let sig = br#"{"type":"error","error":{"type":"invalid_request_error","message":"messages.1.content.0: Invalid `signature` in `thinking` block"}}"#;
        assert_eq!(
            check_status(400, sig, None).unwrap_err().kind,
            ErrorKind::HistoryMismatch
        );
        let e = check_status(400, br#"{"error":{"message":"bad"}}"#, None).unwrap_err();
        assert_eq!(e.kind, ErrorKind::InvalidRequest);
        assert!(check_status(200, b"", None).is_ok());
    }

    #[test]
    fn native_errors_map_to_kinds() {
        let u = from_native(NativeError::UnassemblableDelta {
            block_type: "text".into(),
            delta_type: "x".into(),
        });
        assert_eq!(u.kind, ErrorKind::Unassemblable);
        assert!(!u.retryable);
        let p = from_native(NativeError::Provider {
            status: 529,
            error_type: "overloaded_error".into(),
            message: "Overloaded".into(),
        });
        assert_eq!(
            (p.kind, p.retryable, p.status),
            (ErrorKind::Upstream, true, Some(529))
        );
        assert_eq!(
            from_native(NativeError::Invalid("x".into())).kind,
            ErrorKind::InvalidRequest
        );
        assert_eq!(
            from_native(NativeError::Malformed("x".into())).kind,
            ErrorKind::Malformed
        );
    }

    #[test]
    fn refusal_is_an_error_on_request() {
        let r = parse(Bytes::from_static(
            br#"{"content":[],"stop_reason":"refusal","stop_details":{"category":"cyber"}}"#,
        ))
        .unwrap();
        let e = r.refusal().unwrap();
        assert_eq!(e.kind, ErrorKind::Refusal);
        assert!(e.message.contains("cyber"), "{e}");
        let ok = parse(Bytes::from_static(
            br#"{"content":[],"stop_reason":"end_turn"}"#,
        ))
        .unwrap();
        assert!(ok.refusal().is_none());
    }
}
