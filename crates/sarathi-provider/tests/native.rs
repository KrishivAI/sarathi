// SPDX-License-Identifier: Apache-2.0
//! The round-trip test of the Anthropic-native path, on the scripted server, offline.
//!
//! For every fixture in `tests/fixtures/anthropic-native/`: the stored answer equals the bytes the server sent
//! (capture), and turn 2 is built from the stored slices, so its `system`, `tools` and earlier messages equal turn
//! 1's byte for byte and the assistant message is the content exactly as received (re-send). Checked as bytes,
//! never as parsed JSON. The fixtures are synthetic until recorded answers replace them.
#![expect(clippy::unwrap_used, reason = "test code")]

mod common;

use std::path::PathBuf;
use std::time::Duration;

use common::*;
use futures::StreamExt;
use sarathi_protocol::{BlockKind, MessageSlice, RawJson, Role, Seq};
use sarathi_provider::{
    Assembler, CallParams, Client, Endpoint, Error, ErrorKind, MessagesRequest, MessagesResponse,
    NativeEvent, SessionPrefix, Target,
};

const KEY: &str = "sk-ant-test-key";
const FINAL: &str = r#"{"id":"msg_09","type":"message","role":"assistant","model":"claude-opus-5-5","content":[{"type":"text","text":"Bye."}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}"#;
const NON_STREAMED: [&str; 10] = [
    "thinking-signed",
    "thinking-empty",
    "redacted-thinking",
    "interleaved",
    "unknown-block",
    "escape-hazards",
    "server-tools",
    "citations",
    "compaction",
    "mid-system",
];
const STREAMED: [&str; 4] = [
    "stream-thinking-tool",
    "stream-empty-thinking",
    "stream-compaction",
    "stream-server-tools",
];

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/anthropic-native")
}

fn fixture(case: &str, file: &str) -> Vec<u8> {
    std::fs::read(dir().join(case).join(file)).unwrap()
}

fn case_toml(case: &str) -> String {
    String::from_utf8(fixture(case, "case.toml")).unwrap()
}

fn raw(bytes: &[u8]) -> RawJson {
    RawJson::from_slice(bytes).unwrap()
}

/// A small seam, so the same assertions can later drive another client (for example a comparison with another library).
trait RoundTrip {
    async fn call(&self, request: &MessagesRequest) -> Result<MessagesResponse, Error>;
}

struct Whole(Client);
struct Streamed(Client);

impl RoundTrip for Whole {
    async fn call(&self, request: &MessagesRequest) -> Result<MessagesResponse, Error> {
        self.0.messages(request).await
    }
}

impl RoundTrip for Streamed {
    async fn call(&self, request: &MessagesRequest) -> Result<MessagesResponse, Error> {
        let mut s = Box::pin(self.0.messages_stream(request).await?);
        let mut done = None;
        while let Some(item) = s.next().await {
            if let NativeEvent::Done(r) = item? {
                done = Some(*r);
            }
        }
        Ok(done.unwrap())
    }
}

fn client(server: &Server) -> Client {
    Client::new(Target::anthropic(KEY).with_base_url(&server.url))
        .unwrap()
        .with_timeout(Duration::from_secs(5))
}

fn first_request() -> MessagesRequest {
    let prefix = SessionPrefix::new(
        Some(raw(&std::fs::read(dir().join("system.json")).unwrap())),
        Some(raw(&std::fs::read(dir().join("tools.json")).unwrap())),
    );
    let params = CallParams::new("claude-opus-5-5", 2048)
        .with_extra(&raw(
            br#"{"thinking":{"type":"adaptive"},"output_config":{"effort":"low"}}"#,
        ))
        .unwrap();
    let mut r = MessagesRequest::new(prefix, params);
    r.betas = vec!["interleaved-thinking-2025-05-14".into()];
    r.push(MessageSlice::user_text(Seq(2), "Go.").unwrap());
    r
}

/// Appends what turn 2 needs after `r1`: the assistant slice, a system message if the case asks for one, then the
/// tool results (one per `tool_use`) or a user message.
fn second_request(
    mut req: MessagesRequest,
    r1: &MessagesResponse,
    mid_system: bool,
) -> MessagesRequest {
    req.push(r1.assistant_slice(Seq(5)).unwrap());
    if mid_system {
        req.push(MessageSlice::new(Seq(6), Role::System, &RawJson::string("Be brief.")).unwrap());
    }
    let results: Vec<_> = r1
        .view
        .tool_uses()
        .map(|(id, _, _)| serde_json::json!({"type": "tool_result", "tool_use_id": id, "content": "ok"}))
        .collect();
    if results.is_empty() {
        req.push(MessageSlice::user_text(Seq(7), "Next.").unwrap());
    } else {
        req.push(
            MessageSlice::new(Seq(7), Role::User, &RawJson::render(&results).unwrap()).unwrap(),
        );
    }
    req
}

/// `system` and `tools` go out as the fixture files' bytes, on every turn.
fn assert_prefix_on_wire(case: &str, body: &[u8]) {
    for (key, file) in [("system", "system.json"), ("tools", "tools.json")] {
        let mut want = format!(r#""{key}":"#).into_bytes();
        want.extend_from_slice(&std::fs::read(dir().join(file)).unwrap());
        assert!(
            find(body, &want).is_some(),
            "{case}: `{key}` is not the fixture's bytes"
        );
    }
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Assertion 2 (re-send): everything up to the end of turn 1's first message is the same bytes, and the next bytes
/// are the assistant slice made of the received content.
fn assert_resent(case: &str, body1: &[u8], body2: &[u8], content: &[u8]) {
    let m0 = MessageSlice::user_text(Seq(2), "Go.").unwrap();
    let start = find(body1, br#""messages":["#).unwrap() + br#""messages":["#.len();
    let end = start + m0.bytes.len();
    assert_eq!(&body1[start..end], m0.bytes.as_bytes(), "{case}");
    assert_eq!(
        &body2[..end],
        &body1[..end],
        "{case}: system, tools or turn 1 changed"
    );
    let mut assistant = br#",{"role":"assistant","content":"#.to_vec();
    assistant.extend_from_slice(content);
    assistant.push(b'}');
    assert!(
        body2[end..].starts_with(&assistant),
        "{case}: the assistant message is not the received content\n{}",
        String::from_utf8_lossy(&body2[end..])
    );
}

#[tokio::test]
async fn non_streamed_turns_are_stored_and_resent_byte_for_byte() {
    for case in NON_STREAMED {
        let toml = case_toml(case);
        assert!(toml.contains(r#"source = "synthetic""#), "{case}: {toml}");
        let response = fixture(case, "response.json");
        let server = serve_seq(vec![
            Script::json(200, std::str::from_utf8(&response).unwrap())
                .header("request-id", "req_01"),
            Script::json(200, FINAL),
        ])
        .await;
        let rt = Whole(client(&server));
        let req = first_request();
        let r1 = rt.call(&req).await.unwrap();
        // 1. Capture.
        assert_eq!(r1.raw.as_ref(), response.as_slice(), "{case}");
        assert_eq!(r1.request_id.as_deref(), Some("req_01"));
        let body1 = server.bodies()[0].clone();
        assert_eq!(
            body1,
            req.body(Endpoint::Messages),
            "{case}: turn 1 is not the assembled body"
        );
        assert_prefix_on_wire(case, &body1);
        // 2. Re-send.
        let req2 = second_request(req, &r1, toml.contains("mid_system = true"));
        let r2 = rt.call(&req2).await.unwrap();
        assert_eq!(r2.view.stop_reason.as_deref(), Some("end_turn"));
        let body2 = server.bodies()[1].clone();
        assert_eq!(body2, req2.body(Endpoint::Messages));
        assert_resent(case, &body1, &body2, r1.content_bytes());
        assert_prefix_on_wire(case, &body2);
        if case == "mid-system" {
            assert!(find(&body2, br#"{"role":"system","content":"Be brief."}"#).is_some());
        }
    }
}

#[tokio::test]
async fn the_views_name_what_each_case_holds() {
    let kinds = |case: &str| {
        let r =
            sarathi_provider::wire::native::response::parse(fixture(case, "response.json").into())
                .unwrap();
        r.view
            .blocks
            .into_iter()
            .map(|b| b.kind)
            .collect::<Vec<_>>()
    };
    assert_eq!(kinds("thinking-empty")[0], BlockKind::Thinking);
    assert_eq!(kinds("redacted-thinking")[0], BlockKind::RedactedThinking);
    assert_eq!(kinds("compaction")[0], BlockKind::Compaction);
    assert_eq!(
        kinds("unknown-block")[0],
        BlockKind::Opaque {
            block_type: "future_widget".into()
        }
    );
    assert_eq!(kinds("server-tools")[1], BlockKind::ServerToolResult);
    let order: Vec<_> = kinds("interleaved")
        .iter()
        .map(|k| match k {
            BlockKind::Thinking => "thinking".to_string(),
            BlockKind::Text => "text".to_string(),
            BlockKind::ToolUse { id, .. } => id.to_string(),
            other => format!("{other:?}"),
        })
        .collect();
    assert_eq!(
        order,
        ["thinking", "text", "thinking", "toolu_01", "toolu_02"]
    );
}

#[test]
fn a_value_round_trip_would_lose_the_escape_hazards() {
    // 3. Teeth: the same content through `serde_json::Value` and back is different bytes, so "simplifying" the
    // path to a `Value` fails the re-send assertion.
    let r = sarathi_provider::wire::native::response::parse(
        fixture("escape-hazards", "response.json").into(),
    )
    .unwrap();
    let content = r.content_bytes();
    let v: serde_json::Value = serde_json::from_slice(content).unwrap();
    assert_ne!(serde_json::to_vec(&v).unwrap(), content);
}

/// Cut points that are awkward for a parser: inside a UTF-8 character, between `\r` and `\n`, inside `data:`.
fn awkward_parts(sse: &[u8]) -> Vec<Vec<u8>> {
    let mut cuts = vec![];
    if let Some(i) = sse.iter().position(|b| *b >= 0x80) {
        cuts.push(i + 1);
    }
    if let Some(i) = find(sse, b"\r\n") {
        cuts.push(i + 1);
    }
    if let Some(i) = find(sse, b"data:") {
        cuts.push(i + 2);
    }
    cuts.push(sse.len() / 2);
    cuts.sort_unstable();
    cuts.dedup();
    let mut parts = vec![];
    let mut at = 0;
    for c in cuts {
        parts.push(sse[at..c].to_vec());
        at = c;
    }
    parts.push(sse[at..].to_vec());
    parts
}

#[tokio::test]
async fn streamed_turns_assemble_byte_stable_and_resend() {
    for case in STREAMED {
        let sse = fixture(case, "response.sse");
        let expected = fixture(case, "expected-content.json");
        let server = serve_seq(vec![
            Script::sse(awkward_parts(&sse)),
            Script::json(200, FINAL),
        ])
        .await;
        let rt = Streamed(client(&server));
        let req = first_request();
        let r1 = rt.call(&req).await.unwrap();
        // 4. The assembled content is the expected bytes, the stream is kept as received.
        assert_eq!(
            r1.content_bytes(),
            expected.as_slice(),
            "{case}: {}",
            String::from_utf8_lossy(r1.content_bytes())
        );
        assert_eq!(r1.sse.as_deref(), Some(sse.as_slice()), "{case}");
        let body1 = server.bodies()[0].clone();
        assert_eq!(body1, req.body(Endpoint::MessagesStream));
        assert!(server.requests()[0].contains("accept: text/event-stream"));
        let req2 = second_request(req, &r1, false);
        let r2 = Whole(client(&server)).call(&req2).await.unwrap();
        assert_eq!(r2.view.stop_reason.as_deref(), Some("end_turn"));
        let body2 = server.bodies()[1].clone();
        assert_eq!(body2, req2.body(Endpoint::Messages));
        // The stream body ends with `"stream":true`, turn 2's does not: compare up to the end of turn 1's message.
        assert_resent(case, &body1, &body2, &expected);
        assert_prefix_on_wire(case, &body2);
    }
}

#[test]
fn assembly_does_not_depend_on_how_the_stream_is_cut() {
    for case in STREAMED {
        let sse = fixture(case, "response.sse");
        let expected = fixture(case, "expected-content.json");
        let mut a = Assembler::new(1 << 20);
        for b in &sse {
            a.feed(std::slice::from_ref(b)).unwrap();
        }
        let r = a.finish().unwrap();
        assert_eq!(r.content_bytes(), expected.as_slice(), "{case}");
        // Re-reading the assembled body gives the same bytes: it is stored once and stable from then on.
        let again = sarathi_provider::wire::native::response::parse(r.raw.clone()).unwrap();
        assert_eq!(again.raw, r.raw);
    }
}

#[tokio::test]
async fn an_unknown_delta_fails_the_turn() {
    let case = "stream-unknown-delta";
    assert!(case_toml(case).contains(r#"fails = "unassemblable""#));
    let server = serve(Script::sse(vec![fixture(case, "response.sse")])).await;
    let e = Streamed(client(&server))
        .call(&first_request())
        .await
        .unwrap_err();
    assert_eq!(e.kind, ErrorKind::Unassemblable, "{e}");
    assert!(!e.retryable);
    assert!(e.message.contains("future_delta"), "{e}");
}

#[tokio::test]
async fn headers_on_the_wire_and_metadata_back() {
    let server = serve(
        Script::json(200, FINAL)
            .header("request-id", "req_7")
            .header("anthropic-ratelimit-requests-remaining", "99"),
    )
    .await;
    let r = client(&server).messages(&first_request()).await.unwrap();
    let head = server.only();
    assert!(head.starts_with("post /v1/messages "), "{head}");
    assert!(head.contains(&format!("x-api-key: {KEY}")));
    assert!(head.contains("anthropic-version: 2023-06-01"));
    assert!(head.contains("anthropic-beta: interleaved-thinking-2025-05-14"));
    assert!(head.contains("content-type: application/json"));
    assert_eq!(r.request_id.as_deref(), Some("req_7"));
    assert_eq!(
        r.rate_limits.get("anthropic-ratelimit-requests-remaining"),
        Some("99")
    );
}

#[tokio::test]
async fn count_tokens_uses_the_same_slices() {
    let server = serve(Script::json(200, r#"{"input_tokens":321}"#)).await;
    let req = first_request();
    let c = client(&server).count_tokens(&req).await.unwrap();
    assert_eq!(c.input_tokens, 321);
    assert!(server.only().starts_with("post /v1/messages/count_tokens "));
    assert_eq!(server.bodies()[0], req.body(Endpoint::CountTokens));
    let bad = serve(Script::json(200, r#"{"tokens":1}"#)).await;
    assert_eq!(
        client(&bad).count_tokens(&req).await.unwrap_err().kind,
        ErrorKind::Malformed
    );
}

#[tokio::test]
async fn error_answers_are_classified() {
    let mismatch = r#"{"type":"error","error":{"type":"invalid_request_error","message":"messages.1.content.0: `thinking` or `redacted_thinking` blocks in the latest assistant message cannot be modified. These blocks must remain as they were in the original response."}}"#;
    let cases = [
        (
            Script::json(400, mismatch),
            ErrorKind::HistoryMismatch,
            false,
            None,
        ),
        (
            Script::json(
                429,
                r#"{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}"#,
            )
            .header("retry-after", "7"),
            ErrorKind::RateLimited,
            true,
            Some(Duration::from_secs(7)),
        ),
        (
            Script::json(
                529,
                r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
            ),
            ErrorKind::Upstream,
            true,
            None,
        ),
        (
            Script::json(
                401,
                &format!(r#"{{"error":{{"message":"invalid x-api-key {KEY}"}}}}"#),
            ),
            ErrorKind::Auth,
            false,
            None,
        ),
    ];
    for (script, kind, retryable, retry_after) in cases {
        let server = serve(script).await;
        let e = client(&server)
            .messages(&first_request())
            .await
            .unwrap_err();
        assert_eq!(
            (e.kind, e.retryable, e.retry_after),
            (kind, retryable, retry_after),
            "{e}"
        );
        assert!(!e.message.contains(KEY), "the key leaked: {e}");
        // The stream entry point classifies the same answers the same way.
        let e = client(&server)
            .messages_stream(&first_request())
            .await
            .err()
            .unwrap();
        assert_eq!(e.kind, kind, "{e}");
    }
}

#[tokio::test]
async fn stream_failures_end_the_stream_with_one_error() {
    let start = r#"{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","model":"m","content":[],"usage":{"input_tokens":1}}}"#;
    let overloaded = format!(
        "event: message_start\ndata: {start}\n\nevent: error\ndata: {}\n\n",
        r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#
    );
    let incomplete = format!("event: message_start\ndata: {start}\n\n");
    let leaky = format!(
        "event: message_start\ndata: {start}\n\nevent: error\ndata: {{\"type\":\"error\",\"error\":{{\"type\":\"api_error\",\"message\":\"bad key {KEY}\"}}}}\n\n"
    );
    let cases = [
        (Script::sse(vec![leaky.into_bytes()]), ErrorKind::Upstream),
        (
            Script::sse(vec![overloaded.into_bytes()]),
            ErrorKind::Upstream,
        ),
        (
            Script::sse(vec![incomplete.clone().into_bytes()]),
            ErrorKind::Network,
        ),
        (
            Script::sse(vec![incomplete.into_bytes()]).cut(),
            ErrorKind::Network,
        ),
    ];
    for (script, kind) in cases {
        let server = serve(script).await;
        let mut s = Box::pin(
            client(&server)
                .messages_stream(&first_request())
                .await
                .unwrap(),
        );
        let mut errors = vec![];
        while let Some(item) = s.next().await {
            match item {
                Ok(NativeEvent::Event(_)) => {}
                Ok(NativeEvent::Done(_)) => panic!("a failed stream must not finish"),
                Err(e) => errors.push(e),
            }
        }
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert_eq!(errors[0].kind, kind, "{}", errors[0]);
        assert!(
            !errors[0].message.contains(KEY),
            "the key leaked: {}",
            errors[0]
        );
    }
}

#[tokio::test]
async fn a_quiet_stream_times_out() {
    let start = r#"{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","model":"m","content":[]}}"#;
    let server = serve(Script::sse(vec![format!("data: {start}\n\n").into_bytes()]).stall()).await;
    let c = client(&server).with_timeout(Duration::from_millis(200));
    let mut s = Box::pin(c.messages_stream(&first_request()).await.unwrap());
    let mut last = None;
    while let Some(item) = s.next().await {
        last = Some(item);
    }
    assert_eq!(last.unwrap().unwrap_err().kind, ErrorKind::Timeout);
}

#[tokio::test]
async fn a_gateway_target_posts_to_its_native_route() {
    let server = serve(Script::json(200, FINAL)).await;
    let c = Client::new(Target::gateway(format!("{}/v1", server.url), KEY)).unwrap();
    c.messages(&first_request()).await.unwrap();
    let head = server.only();
    assert!(head.starts_with("post /v1/messages "), "{head}");
    assert!(head.contains(&format!("x-api-key: {KEY}")));
}

#[tokio::test]
async fn an_openai_target_has_no_native_call() {
    let c = Client::new(Target::openai(KEY)).unwrap();
    let e = c.messages(&first_request()).await.unwrap_err();
    assert_eq!(e.kind, ErrorKind::InvalidRequest);
}
