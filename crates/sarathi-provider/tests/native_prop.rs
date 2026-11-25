// SPDX-License-Identifier: Apache-2.0
//! The round-trip property: random block sequences (known and opaque types, strings with
//! escapes and non-BMP characters) as a non-streamed body and as an SSE stream cut anywhere; capture and re-send hold
//! for both. Wire level, no IO.
#![expect(clippy::unwrap_used, reason = "test code")]

use proptest::prelude::*;
use sarathi_protocol::Seq;
use sarathi_provider::wire::native::response::parse;
use sarathi_provider::{Assembler, CallParams, Endpoint, MessagesRequest, SessionPrefix};

#[derive(Debug, Clone)]
enum Block {
    Text(String),
    Thinking(String, String),
    ToolUse(String),
    Opaque(String),
}

/// JSON literals whose spelling a `Value` round trip changes. Built at run time so the source holds no escapes.
fn hazards() -> Vec<String> {
    let b = '\\';
    vec![
        "1e5".into(),
        "-0".into(),
        "1.0".into(),
        "12345678901234567890".into(),
        format!("\"{b}u00e9\""),
        format!("\"{b}/\""),
        format!("\"{b}ud83d{b}ude00\""),
        "null".into(),
        r#"[true,{"b":1,"a":2}]"#.into(),
    ]
}

fn text() -> impl Strategy<Value = String> {
    let chars = prop_oneof![
        Just('a'),
        Just(' '),
        Just('"'),
        Just('\\'),
        Just('/'),
        Just('\n'),
        Just('\u{e9}'),
        Just('\u{1f600}'),
        Just('\u{7}'),
    ];
    prop::collection::vec(chars, 0..12).prop_map(|v| v.into_iter().collect())
}

/// An object with keys in a random (not sorted) order and hazard values.
fn object() -> impl Strategy<Value = String> {
    let n = hazards().len();
    prop::collection::vec((0..5usize, 0..n), 0..5).prop_map(move |members| {
        let h = hazards();
        let mut seen = vec![];
        let mut parts = vec![];
        for (k, v) in members {
            let key = ["zeta", "alpha", "m", "b", "a"][k];
            if !seen.contains(&key) {
                seen.push(key);
                parts.push(format!("\"{key}\":{}", h[v]));
            }
        }
        format!("{{{}}}", parts.join(","))
    })
}

fn block() -> impl Strategy<Value = Block> {
    prop_oneof![
        text().prop_map(Block::Text),
        (text(), "[A-Za-z0-9+/]{1,12}").prop_map(|(t, s)| Block::Thinking(t, s)),
        object().prop_map(Block::ToolUse),
        object().prop_map(|o| {
            let inner = o.trim_start_matches('{');
            let sep = if inner == "}" { "" } else { "," };
            Block::Opaque(format!(r#"{{"type":"future_block"{sep}{inner}"#))
        }),
    ]
}

fn js(s: &str) -> String {
    serde_json::to_string(s).unwrap()
}

/// The block as a non-streamed body holds it (with `\/` spellings, which a `Value` would drop).
fn whole(i: usize, b: &Block) -> String {
    match b {
        Block::Text(t) => format!(r#"{{"type":"text","text":{}}}"#, js(t).replace('/', "\\/")),
        Block::Thinking(t, s) => format!(
            r#"{{"type":"thinking","thinking":{},"signature":{}}}"#,
            js(t),
            js(s)
        ),
        Block::ToolUse(input) => {
            format!(r#"{{"type":"tool_use","id":"toolu_{i}","name":"t","input":{input}}}"#)
        }
        Block::Opaque(o) => o.clone(),
    }
}

/// Splits `s` at char boundaries chosen by `cuts`.
fn pieces(s: &str, cuts: &[prop::sample::Index]) -> Vec<String> {
    let bounds: Vec<usize> = s.char_indices().map(|(i, _)| i).skip(1).collect();
    let mut at: Vec<usize> = cuts
        .iter()
        .filter(|_| !bounds.is_empty())
        .map(|c| bounds[c.index(bounds.len())])
        .collect();
    at.sort_unstable();
    at.dedup();
    let mut out = vec![];
    let mut from = 0;
    for a in at {
        out.push(s[from..a].to_string());
        from = a;
    }
    out.push(s[from..].to_string());
    out
}

/// The SSE events for `b`, and the block the assembler must produce.
fn streamed(i: usize, b: &Block, cuts: &[prop::sample::Index]) -> (Vec<String>, String) {
    let start =
        |cb: &str| format!(r#"{{"type":"content_block_start","index":{i},"content_block":{cb}}}"#);
    let delta = |d: String| format!(r#"{{"type":"content_block_delta","index":{i},"delta":{d}}}"#);
    let stop = format!(r#"{{"type":"content_block_stop","index":{i}}}"#);
    let mut ev = vec![];
    let expected = match b {
        Block::Text(t) => {
            ev.push(start(r#"{"type":"text","text":""}"#));
            if !t.is_empty() {
                for p in pieces(t, cuts) {
                    ev.push(delta(format!(
                        r#"{{"type":"text_delta","text":{}}}"#,
                        js(&p)
                    )));
                }
            }
            format!(r#"{{"type":"text","text":{}}}"#, js(t))
        }
        Block::Thinking(t, s) => {
            ev.push(start(r#"{"type":"thinking","thinking":"","signature":""}"#));
            if !t.is_empty() {
                for p in pieces(t, cuts) {
                    ev.push(delta(format!(
                        r#"{{"type":"thinking_delta","thinking":{}}}"#,
                        js(&p)
                    )));
                }
            }
            ev.push(delta(format!(
                r#"{{"type":"signature_delta","signature":{}}}"#,
                js(s)
            )));
            format!(
                r#"{{"type":"thinking","thinking":{},"signature":{}}}"#,
                js(t),
                js(s)
            )
        }
        Block::ToolUse(input) => {
            ev.push(start(&format!(
                r#"{{"type":"tool_use","id":"toolu_{i}","name":"t","input":{{}}}}"#
            )));
            for p in pieces(input, cuts) {
                ev.push(delta(format!(
                    r#"{{"type":"input_json_delta","partial_json":{}}}"#,
                    js(&p)
                )));
            }
            format!(r#"{{"type":"tool_use","id":"toolu_{i}","name":"t","input":{input}}}"#)
        }
        Block::Opaque(o) => {
            ev.push(start(o));
            o.clone()
        }
    };
    ev.push(stop);
    (ev, expected)
}

fn cut_points() -> impl Strategy<Value = Vec<prop::sample::Index>> {
    prop::collection::vec(any::<prop::sample::Index>(), 0..6)
}

proptest! {
    #[test]
    fn a_non_streamed_body_is_captured_and_resent(blocks in prop::collection::vec(block(), 0..6)) {
        let content = format!(
            "[{}]",
            blocks.iter().enumerate().map(|(i, b)| whole(i, b)).collect::<Vec<_>>().join(",")
        );
        let body = format!(r#"{{"type":"message","content":{content},"stop_reason":"end_turn"}}"#);
        let r = parse(body.clone().into_bytes().into()).unwrap();
        prop_assert_eq!(r.raw.as_ref(), body.as_bytes());
        prop_assert_eq!(r.content_bytes(), content.as_bytes());
        prop_assert_eq!(r.view.blocks.len(), blocks.len());
        let mut req = MessagesRequest::new(SessionPrefix::default(), CallParams::new("m", 8));
        req.push(r.assistant_slice(Seq(1)).unwrap());
        let sent = String::from_utf8(req.body(Endpoint::Messages)).unwrap();
        let expected = format!(r#""messages":[{{"role":"assistant","content":{content}}}]"#);
        prop_assert!(sent.contains(&expected));
    }

    #[test]
    fn a_stream_cut_anywhere_assembles_the_same_bytes(
        blocks in prop::collection::vec(block(), 0..6),
        text_cuts in cut_points(),
        chunk_cuts in cut_points(),
        crlf in any::<bool>(),
    ) {
        let eol = if crlf { "\r\n" } else { "\n" };
        let mut events = vec![
            r#"{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","model":"m","content":[],"stop_reason":null,"usage":{"input_tokens":1}}}"#.to_string(),
        ];
        let mut expected = vec![];
        for (i, b) in blocks.iter().enumerate() {
            let (ev, ex) = streamed(i, b, &text_cuts);
            events.extend(ev);
            expected.push(ex);
        }
        events.push(r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}"#.into());
        events.push(r#"{"type":"message_stop"}"#.into());
        let sse: String = events.iter().map(|e| format!("data: {e}{eol}{eol}")).collect();
        let bytes = sse.as_bytes();
        let mut at: Vec<usize> = chunk_cuts.iter().map(|c| c.index(bytes.len())).collect();
        at.sort_unstable();
        at.dedup();
        let mut a = Assembler::new(1 << 24);
        let mut from = 0;
        for c in at.into_iter().chain([bytes.len()]) {
            a.feed(&bytes[from..c]).unwrap();
            from = c;
        }
        let r = a.finish().unwrap();
        let want = format!("[{}]", expected.join(","));
        prop_assert_eq!(std::str::from_utf8(r.content_bytes()).unwrap(), want.as_str());
        prop_assert_eq!(r.sse.as_deref(), Some(bytes));
        let again = parse(r.raw.clone()).unwrap();
        prop_assert_eq!(again.content_bytes(), r.content_bytes());
        let mut req = MessagesRequest::new(SessionPrefix::default(), CallParams::new("m", 8));
        req.push(r.assistant_slice(Seq(1)).unwrap());
        let sent = String::from_utf8(req.body(Endpoint::Messages)).unwrap();
        let resent = format!(r#""messages":[{{"role":"assistant","content":{want}}}]"#);
        prop_assert!(sent.contains(&resent));
    }
}
