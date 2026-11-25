// SPDX-License-Identifier: Apache-2.0
//! The spliced chat-completions call on a scripted server: the body goes out byte for byte and the answer is kept as
//! received.
#![expect(clippy::unwrap_used, reason = "test code")]

mod common;

use common::*;
use sarathi_protocol::RawJson;
use sarathi_provider::{ChatBody, ChatParams, Client, ErrorKind, Target, TokenField};

const ANSWER: &str = r#"{"id":"chatcmpl-9","model":"m-1","choices":[{"index":0,"message":{"role":"assistant","content":"{\"thought\":\"t\",\"tool\":\"done\",\"reply\":\"ok \u00e9\"}"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3}}"#;

fn request() -> ChatBody {
    let params = ChatParams::new("m-1", 32, TokenField::MaxTokens)
        .with_extra(&RawJson::from_text(r#"{"temperature":0}"#).unwrap())
        .unwrap();
    let mut b = ChatBody::new(params, Some(RawJson::string("sys")), None);
    b.push(RawJson::from_text(r#"{"role":"user","content":"a \u00e9 \/ b"}"#).unwrap());
    b
}

#[tokio::test]
async fn the_body_goes_out_as_spliced_and_the_answer_comes_back_as_received() {
    let s = serve(Script::json(200, ANSWER).header("x-request-id", "req_7")).await;
    let c = Client::new(Target::openai_compatible(format!("{}/v1", s.url), "sk-x")).unwrap();
    let req = request();
    let a = c.chat_spliced(&req).await.unwrap();
    assert_eq!(&a.raw[..], ANSWER.as_bytes());
    assert_eq!(a.request_id.as_deref(), Some("req_7"));
    assert_eq!(a.view.stop_reason.as_deref(), Some("stop"));
    assert_eq!(s.bodies(), vec![req.body()]);
    let head = s.only();
    assert!(head.starts_with("post /v1/chat/completions "), "{head}");
    assert!(head.contains("authorization: bearer sk-x"), "{head}");
    assert!(!head.contains("accept: text/event-stream"), "{head}");
}

#[tokio::test]
async fn a_gateway_is_called_under_v1() {
    let s = serve(Script::json(200, ANSWER)).await;
    let c = Client::new(Target::gateway(&s.url, "uf-key")).unwrap();
    assert_eq!(c.chat_token_field(), TokenField::MaxTokens);
    c.chat_spliced(&request()).await.unwrap();
    assert!(s.only().starts_with("post /v1/chat/completions "));
}

#[tokio::test]
async fn provider_errors_are_classified() {
    let s =
        serve(Script::json(429, r#"{"error":{"message":"slow down"}}"#).header("retry-after", "3"))
            .await;
    let c = Client::new(Target::openai_compatible(&s.url, "sk-x")).unwrap();
    let e = c.chat_spliced(&request()).await.unwrap_err();
    assert_eq!(e.kind, ErrorKind::RateLimited);
    assert!(e.retryable);
    assert_eq!(e.retry_after, Some(std::time::Duration::from_secs(3)));
    assert!(e.message.contains("slow down"), "{e}");

    let s = serve(Script::json(
        200,
        r#"{"error":{"message":"upstream busy","code":429}}"#,
    ))
    .await;
    let c = Client::new(Target::openai_compatible(&s.url, "sk-x")).unwrap();
    let e = c.chat_spliced(&request()).await.unwrap_err();
    assert_eq!(e.kind, ErrorKind::RateLimited);
    assert!(e.retryable);
    assert!(e.message.contains("upstream busy"), "{e}");

    let s = serve(Script::json(200, r#"{"choices":[]}"#)).await;
    let c = Client::new(Target::openai_compatible(&s.url, "sk-x")).unwrap();
    let e = c.chat_spliced(&request()).await.unwrap_err();
    assert_eq!(e.kind, ErrorKind::Malformed);
}

#[tokio::test]
async fn an_anthropic_target_is_refused_before_any_call() {
    let c = Client::new(Target::anthropic("k").with_base_url("http://127.0.0.1:9")).unwrap();
    let e = c.chat_spliced(&request()).await.unwrap_err();
    assert_eq!(e.kind, ErrorKind::InvalidRequest);
}

#[test]
fn only_openai_kind_targets_speak_openai_chat() {
    assert!(
        Client::new(Target::openai("k"))
            .unwrap()
            .speaks_openai_chat()
    );
    assert!(
        Client::new(Target::gateway("http://g", "k"))
            .unwrap()
            .speaks_openai_chat()
    );
    assert!(
        !Client::new(Target::anthropic("k"))
            .unwrap()
            .speaks_openai_chat()
    );
}

#[test]
fn openai_itself_names_the_limit_max_completion_tokens() {
    let c = Client::new(Target::openai("k")).unwrap();
    assert_eq!(c.chat_token_field(), TokenField::MaxCompletionTokens);
    let c = Client::new(Target::openai_compatible(
        "https://api.groq.com/openai/v1",
        "k",
    ))
    .unwrap();
    assert_eq!(c.chat_token_field(), TokenField::MaxTokens);
}
