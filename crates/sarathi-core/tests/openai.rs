// SPDX-License-Identifier: Apache-2.0
//! The driver on OpenAI-style chat completions over HTTP, on both step protocols: every request is spliced from the
//! log (the earlier messages go out with the bytes they first had), the answer is re-sent from the message it came
//! in, and every logged request rebuilds to its hash.

#![expect(clippy::unwrap_used, clippy::panic, reason = "test code")]

pub mod common;

use std::sync::{Arc, Mutex};

use common::*;
use sarathi_core::ModelProfile;
use sarathi_core::driver::Driver;
use sarathi_core::openai::{self, OpenAiCaller};
use sarathi_log::RequestParts;
use sarathi_protocol::{Outcome, RawJson};
use sarathi_provider::{Chain, Client, Slot, Target};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// A server answering the n-th request with the n-th body; it keeps every request body as received.
async fn serve(answers: Vec<String>) -> (String, Arc<Mutex<Vec<Vec<u8>>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let seen = bodies.clone();
    tokio::spawn(async move {
        let mut n = 0;
        while let Ok((mut sock, _)) = listener.accept().await {
            let answer = answers[n.min(answers.len() - 1)].clone();
            n += 1;
            let seen = seen.clone();
            tokio::spawn(async move {
                let mut buf = Vec::new();
                let mut tmp = [0u8; 4096];
                let body = loop {
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
                        break buf[end + 4..end + 4 + len].to_vec();
                    }
                };
                seen.lock().unwrap().push(body);
                let resp = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{answer}",
                    answer.len()
                );
                sock.write_all(resp.as_bytes()).await.unwrap();
            });
        }
    });
    (url, bodies)
}

/// A chat completion whose message has `content` (a JSON value) and `calls` (`(id, name, arguments text)`).
fn completion(content: &str, calls: &[(&str, &str, &str)], finish: &str) -> String {
    let calls = if calls.is_empty() {
        String::new()
    } else {
        let items: Vec<String> = calls
            .iter()
            .map(|(id, name, args)| {
                format!(
                    r#"{{"id":"{id}","type":"function","function":{{"name":"{name}","arguments":{}}}}}"#,
                    RawJson::string(args).as_str()
                )
            })
            .collect();
        format!(r#","tool_calls":[{}]"#, items.join(","))
    };
    format!(
        r#"{{"id":"chatcmpl-1","object":"chat.completion","model":"m-1","choices":[{{"index":0,"message":{{"role":"assistant","content":{content}{calls},"refusal":null}},"finish_reason":"{finish}"}}],"usage":{{"prompt_tokens":9,"completion_tokens":2}}}}"#
    )
}

fn caller(url: &str) -> OpenAiCaller {
    let client = Client::new(Target::openai_compatible(url, "sk-test")).unwrap();
    let chain = Chain::new(vec![Slot::new("main", client, "m-1")]).unwrap();
    OpenAiCaller::new(chain, 256)
        .unwrap()
        .with_extra(&openai::step_format())
        .unwrap()
}

fn rebuild(parts: &RequestParts<'_>) -> Vec<u8> {
    openai::request(parts).unwrap().body()
}

fn find(hay: &[u8], needle: &[u8]) -> usize {
    hay.windows(needle.len())
        .position(|w| w == needle)
        .unwrap_or_else(|| {
            panic!(
                "{} not in {}",
                String::from_utf8_lossy(needle),
                String::from_utf8_lossy(hay)
            )
        })
}

async fn run(
    profile: ModelProfile,
    answers: Vec<String>,
) -> (
    Vec<Vec<u8>>,
    TestTools,
    Outcome,
    Option<Box<str>>,
    sarathi_log::Log,
) {
    let (url, bodies) = serve(answers).await;
    let tools = TestTools::default();
    let (mut d, _) = Driver::new(
        config(profile),
        Shared::default(),
        caller(&url),
        tools.clone(),
        Host(None),
    )
    .await
    .unwrap();
    let end = d.run_turn(user("Go é."), &mut |_| {}).await.unwrap();
    let log = d.log().clone();
    let got = bodies.lock().unwrap().clone();
    (got, tools, end.outcome, end.reply, log)
}

#[tokio::test]
async fn a_json_object_turn_runs_on_chat_completions() {
    let step = r#"{"thought":"look first","tool":"look","input":{"z": 1,"a":"\/"}}"#;
    let finish = r#"{"thought":"ok","tool":"done","reply":"All done."}"#;
    let answers = vec![
        completion(RawJson::string(step).as_str(), &[], "stop"),
        completion(RawJson::string(finish).as_str(), &[], "stop"),
    ];
    let (bodies, tools, outcome, reply, log) = run(ModelProfile::json_object(true), answers).await;
    assert_eq!(outcome, Outcome::Finished);
    assert_eq!(reply.as_deref(), Some("All done."));
    let ran = tools.tally.ran.lock().unwrap().clone();
    assert_eq!(
        ran,
        vec![("look".to_owned(), r#"{"z": 1,"a":"\/"}"#.to_owned())]
    );

    let (b1, b2) = (&bodies[0], &bodies[1]);
    assert!(b1.starts_with(
        br#"{"model":"m-1","max_tokens":256,"response_format":{"type":"json_schema""#
    ));
    let user = r#"{"role":"user","content":"Go é."}"#.as_bytes();
    let end1 = find(b1, user) + user.len();
    assert_eq!(
        &b2[..end1],
        &b1[..end1],
        "the prefix and the first message are the same bytes"
    );
    let rest = String::from_utf8(b2[end1..].to_vec()).unwrap();
    let assistant = format!(
        r#",{{"role":"assistant","content":{}}},{{"role":"user","content":"observation:\nseen"}}]}}"#,
        RawJson::string(step).as_str()
    );
    assert_eq!(rest, assistant);
    log.verify_requests(&rebuild).unwrap();
    assert!(b1.windows(7).all(|w| w != br#""tools""#));
}

#[tokio::test]
async fn a_native_turn_sends_tool_messages_and_resends_the_calls() {
    let answers = vec![
        completion(
            r#""Checking.""#,
            &[("call_a", "look", r#"{"z": 1}"#), ("call_b", "echo", "")],
            "tool_calls",
        ),
        completion(r#""Done.""#, &[], "stop"),
    ];
    let (bodies, tools, outcome, reply, log) = run(ModelProfile::native(), answers).await;
    assert_eq!(outcome, Outcome::Finished);
    assert_eq!(reply.as_deref(), Some("Done."));
    let ran = tools.tally.ran.lock().unwrap().clone();
    assert_eq!(
        ran,
        vec![
            ("look".to_owned(), r#"{"z": 1}"#.to_owned()),
            ("echo".to_owned(), "{}".to_owned())
        ]
    );
    let b2 = String::from_utf8(bodies[1].clone()).unwrap();
    let tail = concat!(
        r#"{"role":"assistant","content":"Checking.","tool_calls":[{"id":"call_a","type":"function","function":{"name":"look","arguments":"{\"z\": 1}"}},"#,
        r#"{"id":"call_b","type":"function","function":{"name":"echo","arguments":""}}]},"#,
        r#"{"role":"tool","tool_call_id":"call_a","content":"seen"},"#,
        r#"{"role":"tool","tool_call_id":"call_b","content":"echoed: {}"}],"#,
        r#""tools":[{"type":"function","function":{"name":"#
    );
    assert!(b2.contains(tail), "{b2}");
    assert!(b2.ends_with("}]}"), "{b2}");
    log.verify_requests(&rebuild).unwrap();
    let k = kinds(&log);
    assert_eq!(k.last().map(String::as_str), Some("turn.end"), "{k:?}");
}

/// Two turns on one driver: what the second request holds after `first` answered the first.
async fn two_turns(
    profile: ModelProfile,
    first: String,
    first_user: sarathi_protocol::Op,
) -> (Outcome, Outcome, Vec<Vec<u8>>, sarathi_log::Log) {
    let (url, bodies) = serve(vec![first, completion(r#""Fine.""#, &[], "stop")]).await;
    let (mut d, _) = Driver::new(
        config(profile),
        Shared::default(),
        caller(&url),
        TestTools::default(),
        Host(None),
    )
    .await
    .unwrap();
    let one = d.run_turn(first_user, &mut |_| {}).await.unwrap().outcome;
    let two = d
        .run_turn(user("Again."), &mut |_| {})
        .await
        .unwrap()
        .outcome;
    let log = d.log().clone();
    let got = bodies.lock().unwrap().clone();
    (one, two, got, log)
}

#[tokio::test]
async fn a_refusal_stays_in_the_conversation_and_the_next_turn_runs() {
    let refused = r#"{"choices":[{"message":{"role":"assistant","content":null,"refusal":"I can't."},"finish_reason":"stop"}]}"#;
    for (body, sent) in [
        (refused, r#"{"role":"assistant","content":"I can't."}"#),
        (
            r#"{"choices":[{"message":{"role":"assistant","content":null},"finish_reason":"length"}]}"#,
            r#"{"role":"assistant","content":""}"#,
        ),
    ] {
        let (one, two, bodies, log) =
            two_turns(ModelProfile::native(), body.to_owned(), user("Go.")).await;
        assert!(
            matches!(one, Outcome::Refused | Outcome::Finished),
            "{one:?}"
        );
        assert_eq!(two, Outcome::Finished);
        let b2 = String::from_utf8(bodies[1].clone()).unwrap();
        assert!(b2.contains(sent), "{b2}");
        log.verify_requests(&rebuild).unwrap();
    }
}

#[tokio::test]
async fn an_answer_with_outer_whitespace_is_stored_and_read() {
    let padded = format!("\n\n   {}", completion(r#""Hi.""#, &[], "stop"));
    let (one, two, _, log) = two_turns(ModelProfile::native(), padded, user("Go.")).await;
    assert_eq!((one, two), (Outcome::Finished, Outcome::Finished));
    log.verify_requests(&rebuild).unwrap();
}

#[tokio::test]
async fn an_image_is_said_in_words_and_the_session_goes_on() {
    let op = sarathi_protocol::Op::UserMessage {
        content: RawJson::from_text(
            r#"[{"type":"text","text":"See:"},{"type":"image","source":{"type":"base64","data":"QUJD"}}]"#,
        )
        .unwrap(),
    };
    let (one, two, bodies, log) = two_turns(
        ModelProfile::native(),
        completion(r#""Ok.""#, &[], "stop"),
        op,
    )
    .await;
    assert_eq!((one, two), (Outcome::Finished, Outcome::Finished));
    let b1 = String::from_utf8(bodies[0].clone()).unwrap();
    assert!(
        b1.contains(r#"{"role":"user","content":[{"type":"text","text":"See:"},{"type":"text","text":"[a image block, not shown to this model]"}]}"#),
        "{b1}"
    );
    log.verify_requests(&rebuild).unwrap();
}
