// SPDX-License-Identifier: Apache-2.0
//! The driver on the Anthropic Messages API over HTTP: a context check close to the compaction point counts the
//! request's tokens with `count_tokens` and goes on with the count, not the estimate.

#![expect(clippy::unwrap_used, clippy::panic, reason = "test code")]

pub mod common;

use std::sync::{Arc, Mutex};

use common::*;
use sarathi_core::ModelProfile;
use sarathi_core::anthropic::AnthropicCaller;
use sarathi_core::driver::{Compactor, ContextProfile, Driver};
use sarathi_protocol::Outcome;
use sarathi_provider::{Chain, Client, Slot, Target};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// A server answering `count_tokens` with `count` and each other request with the next of `answers`; it keeps the
/// path of every request, in order.
async fn serve(count: u64, answers: Vec<String>) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let paths = Arc::new(Mutex::new(Vec::new()));
    let seen = paths.clone();
    let answers = Arc::new(Mutex::new(answers.into_iter()));
    tokio::spawn(async move {
        while let Ok((mut sock, _)) = listener.accept().await {
            let seen = seen.clone();
            let answers = answers.clone();
            tokio::spawn(async move {
                let mut buf = Vec::new();
                let mut tmp = [0u8; 4096];
                let path = loop {
                    let k = sock.read(&mut tmp).await.unwrap();
                    buf.extend_from_slice(&tmp[..k]);
                    let Some(end) = buf.windows(4).position(|w| w == b"\r\n\r\n") else {
                        continue;
                    };
                    let head = String::from_utf8_lossy(&buf[..end]).to_ascii_lowercase();
                    let len: usize = head
                        .lines()
                        .find_map(|l| l.strip_prefix("content-length: "))
                        .unwrap()
                        .trim()
                        .parse()
                        .unwrap();
                    if buf.len() >= end + 4 + len {
                        break head.split(' ').nth(1).unwrap().to_owned();
                    }
                };
                let answer = if path.ends_with("/count_tokens") {
                    format!(r#"{{"input_tokens":{count}}}"#)
                } else {
                    answers.lock().unwrap().next().unwrap()
                };
                seen.lock().unwrap().push(path);
                let resp = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{answer}",
                    answer.len()
                );
                sock.write_all(resp.as_bytes()).await.unwrap();
            });
        }
    });
    (url, paths)
}

/// A Messages answer saying `text` that reports `input` input tokens.
fn said(text: &str, input: u64) -> String {
    format!(
        r#"{{"id":"msg_01","type":"message","role":"assistant","model":"m-1","content":[{{"type":"text","text":"{text}"}}],"stop_reason":"end_turn","usage":{{"input_tokens":{input},"output_tokens":5}}}}"#
    )
}

#[tokio::test]
async fn a_check_near_the_compaction_point_counts_with_the_endpoint_and_goes_on_with_the_count() {
    // The limit is 800 and compaction is due at 600. The first answer reports 600 input tokens, so the estimate
    // for the next request is just past the compaction point; the endpoint counts 100, well under it.
    let (url, paths) = serve(100, vec![said("Hello.", 600), said("Second.", 120)]).await;
    let client = Client::new(Target::anthropic("k").with_base_url(url)).unwrap();
    let chain = Chain::new(vec![Slot::new("main", client, "m-1")]).unwrap();
    let mut context = ContextProfile::new(1000, 200);
    context.compactor = Compactor::Server;
    let config = config(ModelProfile::native())
        .with_limits(limits(10))
        .with_context(context);
    let caller = AnthropicCaller::new(chain, 256).unwrap();
    let (mut d, _) = Driver::new(
        config,
        Shared::default(),
        caller,
        TestTools::default(),
        Host(None),
    )
    .await
    .unwrap();
    let first = d.run_turn(user("Hi."), &mut |_| {}).await.unwrap();
    assert_eq!(first.outcome, Outcome::Finished);
    let second = d.run_turn(user("Next."), &mut |_| {}).await.unwrap();
    assert_eq!(second.outcome, Outcome::Finished);
    assert_eq!(second.reply.as_deref(), Some("Second."));
    assert_eq!(
        *paths.lock().unwrap(),
        vec!["/v1/messages", "/v1/messages/count_tokens", "/v1/messages"],
        "counted, then sent without a compaction call"
    );
    check_log(d.log());
}
