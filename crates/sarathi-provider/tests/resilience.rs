// SPDX-License-Identifier: Apache-2.0
//! Retries, failover and the gateway profile against the scripted server, offline.
#![expect(clippy::unwrap_used, reason = "test code")]

mod common;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use common::*;
use futures::StreamExt;
use sarathi_protocol::{MessageSlice, RawJson, Seq};
use sarathi_provider::resilience::Answered;
use sarathi_provider::{
    CallParams, Chain, Client, Endpoint, Error, ErrorKind, MessagesRequest, MessagesResponse,
    NativeEvent, RetryPolicy, SessionPrefix, Slot, Target,
};

const KEY: &str = "sk-ant-test-key";
const FINAL: &str = r#"{"id":"msg_09","type":"message","role":"assistant","model":"claude-opus-5-5","content":[{"type":"text","text":"Bye."}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}"#;
const OVERLOADED: &str =
    r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#;
const MISMATCH: &str = r#"{"type":"error","error":{"type":"invalid_request_error","message":"messages.1.content.0: `thinking` or `redacted_thinking` blocks in the latest assistant message cannot be modified. These blocks must remain as they were in the original response."}}"#;

fn native(case: &str, file: &str) -> Vec<u8> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/anthropic-native");
    std::fs::read(dir.join(case).join(file)).unwrap()
}

fn anthropic(server: &Server) -> Client {
    Client::new(Target::anthropic(KEY).with_base_url(&server.url))
        .unwrap()
        .with_timeout(Duration::from_secs(5))
}

fn request(model: &str) -> MessagesRequest {
    let mut r = MessagesRequest::new(SessionPrefix::new(None, None), CallParams::new(model, 64));
    r.push(MessageSlice::user_text(Seq(1), "Go.").unwrap());
    r
}

/// Short waits, no jitter.
fn fast(attempts: u32) -> RetryPolicy {
    RetryPolicy {
        max_attempts: attempts,
        base_delay: Duration::from_millis(1),
        max_delay: Duration::from_millis(4),
        ..RetryPolicy::default().without_jitter()
    }
}

async fn ask(
    chain: &Chain,
) -> Result<Answered<MessagesResponse>, sarathi_provider::resilience::Failed> {
    chain
        .run(|slot, _| async move { slot.client.messages(&request(&slot.model)).await })
        .await
}

/// A streamed turn read to its end: the assembled message, or the stream's error.
async fn streamed(client: &Client, req: &MessagesRequest) -> Result<MessagesResponse, Error> {
    let mut s = Box::pin(client.messages_stream(req).await?);
    while let Some(item) = s.next().await {
        if let NativeEvent::Done(r) = item? {
            return Ok(*r);
        }
    }
    Err(Error::new(ErrorKind::Malformed, "no Done"))
}

#[tokio::test]
async fn an_overloaded_answer_is_retried_on_the_same_slot_with_the_same_bytes() {
    let server = serve_seq(vec![
        Script::json(529, OVERLOADED),
        Script::json(200, FINAL),
    ])
    .await;
    let chain = Chain::new(vec![Slot::new("a", anthropic(&server), "claude-opus-5-5")])
        .unwrap()
        .with_retry(fast(3));
    let out = ask(&chain).await.unwrap();
    assert_eq!((out.slot, out.failed.len()), (0, 0));
    assert_eq!(out.value.raw.as_ref(), FINAL.as_bytes());
    let bodies = server.bodies();
    assert_eq!(bodies.len(), 2);
    assert_eq!(bodies[0], bodies[1], "a retry sends the same bytes");
    assert_eq!(
        bodies[0],
        request("claude-opus-5-5").body(Endpoint::Messages)
    );
}

#[tokio::test]
async fn retry_after_is_waited_for() {
    let limited = Script::json(
        429,
        r#"{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}"#,
    )
    .header("retry-after", "1");
    let server = serve_seq(vec![limited, Script::json(200, FINAL)]).await;
    let chain = Chain::new(vec![Slot::new("a", anthropic(&server), "m")])
        .unwrap()
        .with_retry(fast(2));
    let start = Instant::now();
    ask(&chain).await.unwrap();
    assert!(
        start.elapsed() >= Duration::from_secs(1),
        "{:?}",
        start.elapsed()
    );
}

#[tokio::test]
async fn a_long_retry_after_moves_on_to_the_next_slot_at_once() {
    let limited = Script::json(
        429,
        r#"{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}"#,
    )
    .header("retry-after", "3600");
    let first = serve(limited).await;
    let second = serve(Script::json(200, FINAL)).await;
    let chain = Chain::new(vec![
        Slot::new("a", anthropic(&first), "claude-opus-5-5"),
        Slot::new("b", anthropic(&second), "claude-sonnet-5-5"),
    ])
    .unwrap()
    .with_retry(fast(3));
    // Honouring the 3600 s Retry-After would sleep an hour; moving on takes no wait at all. A bound far below the hour
    // and far above any scheduling delay checks which one happened without timing the work (a wall-clock bound of a
    // few seconds failed on a loaded machine).
    let out = tokio::time::timeout(Duration::from_secs(120), ask(&chain))
        .await
        .expect("the chain waited for the 3600 s Retry-After instead of moving on")
        .unwrap();
    assert_eq!(out.slot, 1);
    assert_eq!(out.failed[0].error.kind, ErrorKind::RateLimited);
    assert_eq!(first.requests().len(), 1, "not retried");
    assert!(
        String::from_utf8(second.bodies()[0].clone())
            .unwrap()
            .contains(r#""model":"claude-sonnet-5-5""#),
        "the second slot is asked for its own model"
    );
}

#[tokio::test]
async fn a_dead_slot_fails_over_and_trips_its_breaker() {
    let dead = {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        format!("http://{}", l.local_addr().unwrap())
    };
    let live = serve(Script::json(200, FINAL)).await;
    let dead_client = Client::new(Target::anthropic(KEY).with_base_url(&dead)).unwrap();
    let chain = Chain::new(vec![
        Slot::new("dead", dead_client, "m").with_breaker(sarathi_provider::CircuitBreaker::new(
            2,
            Duration::from_secs(60),
        )),
        Slot::new("live", anthropic(&live), "m"),
    ])
    .unwrap()
    .with_retry(fast(2));
    let out = ask(&chain).await.unwrap();
    assert_eq!(out.slot, 1);
    assert_eq!(out.failed[0].error.kind, ErrorKind::Network);
    assert!(
        !chain.slots()[0].breaker.allows(),
        "2 failed connects opened it"
    );
    let out = ask(&chain).await.unwrap();
    assert_eq!(out.failed[0].error.kind, ErrorKind::CircuitOpen);
}

#[tokio::test]
async fn a_history_mismatch_is_neither_retried_nor_failed_over() {
    let first = serve(Script::json(400, MISMATCH)).await;
    let second = serve(Script::json(200, FINAL)).await;
    let chain = Chain::new(vec![
        Slot::new("a", anthropic(&first), "m"),
        Slot::new("b", anthropic(&second), "m"),
    ])
    .unwrap()
    .with_retry(fast(3));
    let e = ask(&chain).await.unwrap_err();
    assert_eq!(e.error().kind, ErrorKind::HistoryMismatch);
    assert_eq!(first.requests().len(), 1);
    assert!(second.requests().is_empty(), "never hidden by another slot");
}

#[tokio::test]
async fn a_cut_stream_is_made_again_and_the_second_answer_is_stored() {
    let sse = native("stream-thinking-tool", "response.sse");
    let half = sse[..sse.len() / 2].to_vec();
    let server = serve_seq(vec![
        Script::sse(vec![half]).cut(),
        Script::sse(vec![sse.clone()]),
    ])
    .await;
    let chain = Chain::new(vec![Slot::new("a", anthropic(&server), "m")])
        .unwrap()
        .with_retry(fast(2));
    let out = chain
        .run(|slot, _| async move { streamed(&slot.client, &request(&slot.model)).await })
        .await
        .unwrap();
    assert_eq!(server.requests().len(), 2);
    assert_eq!(
        out.value.content_bytes(),
        native("stream-thinking-tool", "expected-content.json").as_slice()
    );
}

/// The gateway is not byte-exact on the request, so its profile says so and a caller
/// compares JSON values. Sarathi still sends the same bytes to a gateway as to Anthropic.
#[tokio::test]
async fn the_gateway_profile_compares_json_values_not_bytes() {
    let gateway = Target::gateway("https://gw.test", KEY);
    assert!(gateway.capabilities().native_messages);
    assert!(!gateway.capabilities().byte_exact_request);
    let direct = Target::anthropic(KEY).capabilities();
    assert!(direct.native_messages && direct.byte_exact_request);
    let openai = Target::openai(KEY).capabilities();
    assert!(!openai.native_messages && openai.byte_exact_request);

    // Turn 2 with the escape hazards in its history, sent through a gateway target.
    let r1 = sarathi_provider::wire::native::response::parse(
        native("escape-hazards", "response.json").into(),
    )
    .unwrap();
    let mut req = request("claude-opus-5-5");
    req.push(r1.assistant_slice(Seq(2)).unwrap());
    req.push(MessageSlice::user_text(Seq(3), "Next.").unwrap());
    let server = serve(Script::json(200, FINAL)).await;
    let c = Client::new(Target::gateway(format!("{}/v1", server.url), KEY)).unwrap();
    c.messages(&req).await.unwrap();
    let sent = server.bodies().remove(0);
    assert_eq!(
        sent,
        req.body(Endpoint::Messages),
        "Sarathi's side is byte-exact"
    );

    // What the gateway forwards at c9c3eaf: the body parsed into a map and written again with `model` replaced.
    let mut forwarded: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&sent).unwrap();
    forwarded.insert("model".into(), "claude-opus-5-5".into());
    let forwarded = serde_json::to_vec(&forwarded).unwrap();
    assert_ne!(forwarded, sent, "the gateway changes the request bytes");
    let as_value = |b: &[u8]| serde_json::from_slice::<serde_json::Value>(b).unwrap();
    assert_eq!(as_value(&forwarded), as_value(&sent), "but not the content");
    let stored = RawJson::from_slice(r1.content_bytes()).unwrap();
    let forwarded_assistant = &as_value(&forwarded)["messages"][1]["content"];
    assert_eq!(*forwarded_assistant, as_value(stored.as_bytes()));
}
