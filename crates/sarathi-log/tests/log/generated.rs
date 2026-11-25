// SPDX-License-Identifier: Apache-2.0
//! Byte identity over generated JSON: documents with arbitrary whitespace (`\r\n` included), escapes, surrogate
//! pairs, raw unicode and number spellings (`1e5`, `-0`, `0.10`) go into a session as user messages, response
//! contents and tool result blocks, through the file, and come back byte for byte.

use sarathi_log::{Body, Durability, JsonlStore, Log, LogStore, Slice};
use sarathi_protocol::RawJson;

use crate::common::Script;

/// splitmix64: a fixed seed, so a failure names the document that broke.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn pick<'a>(&mut self, from: &[&'a str]) -> &'a str {
        from[usize::try_from(self.next() % from.len() as u64).unwrap()]
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

const WS: &[&str] = &["", "", " ", "\n", "\r\n", "\t", "  \r\n  "];
const NUMBERS: &[&str] = &[
    "0",
    "-0",
    "1e5",
    "1E+5",
    "-1.5e-3",
    "0.10",
    "12345678901234567890",
    "-7",
];
const PIECES: &[&str] = &[
    "a",
    "theme",
    " ",
    "\\\"",
    "\\\\",
    "\\/",
    "\\b",
    "\\f",
    "\\n",
    "\\r",
    "\\t",
    "\\u00e9",
    "\\ud83d\\ude00",
    "é",
    "😀",
    "\\u0000",
    "{\\\"x\\\":1}",
];

fn string(r: &mut Rng, out: &mut String) {
    out.push('"');
    for _ in 0..r.below(6) {
        out.push_str(r.pick(PIECES));
    }
    out.push('"');
}

fn value(r: &mut Rng, depth: u32, out: &mut String) {
    let kind = if depth == 0 { r.below(4) } else { r.below(6) };
    match kind {
        0 => string(r, out),
        1 => out.push_str(r.pick(NUMBERS)),
        2 => out.push_str(r.pick(&["true", "false", "null"])),
        3 => out.push_str(r.pick(&["[]", "{}", "[ ]", "{\r\n}"])),
        4 => {
            out.push('[');
            for i in 0..=r.below(3) {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(r.pick(WS));
                value(r, depth - 1, out);
                out.push_str(r.pick(WS));
            }
            out.push(']');
        }
        _ => {
            out.push('{');
            for i in 0..=r.below(3) {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(r.pick(WS));
                string(r, out);
                out.push_str(r.pick(WS));
                out.push(':');
                out.push_str(r.pick(WS));
                value(r, depth - 1, out);
                out.push_str(r.pick(WS));
            }
            out.push('}');
        }
    }
}

fn doc(r: &mut Rng) -> String {
    let mut s = String::new();
    value(r, 3, &mut s);
    s
}

#[tokio::test(flavor = "multi_thread")]
async fn generated_documents_come_back_byte_for_byte_from_the_file() {
    let root = tempfile::tempdir().unwrap();
    let store = JsonlStore::new(root.path());
    let h = store.create(1, None).await.unwrap();
    let id = h.session;
    let mut s = Script::new(Log::new(h));
    let mut r = Rng(0x5a7a_7e1f);
    let mut v = vec![s.prefix()];
    let mut sent = Vec::new();
    for turn in 0..200 {
        let (user, content, result) = (doc(&mut r), doc(&mut r), doc(&mut r));
        let call = format!("toolu_{turn}");
        let u = s.push(Body::UserMessage {
            content: RawJson::from_text(&user).unwrap(),
            steer: false,
        });
        v.push(u.clone());
        v.push(s.request());
        let array = format!("[{}{content}{}]", r.pick(WS), r.pick(WS));
        let ra = s.respond(&array, &[(&call, "edit")]);
        let assistant = ra[1].seq;
        v.extend(ra);
        let block =
            format!(r#"{{"type":"tool_result","tool_use_id":"{call}","content":{result}}}"#);
        let t = s.push(Body::ToolResult {
            call: call.as_str().into(),
            block: RawJson::from_text(&block).unwrap(),
            is_error: false,
            spill: None,
        });
        v.push(t.clone());
        v.push(s.end());
        sent.push((u.seq, user, assistant, array, t.seq, block));
    }
    store
        .append(id, sarathi_protocol::Seq(1), v, Durability::Durable)
        .await
        .unwrap();
    drop(store);
    let loaded = JsonlStore::new(root.path()).read(id).await.unwrap();
    assert_eq!(loaded.recovered, None);
    assert_eq!(loaded.log, s.log);
    for (u, user, a, array, t, block) in sent {
        let Slice::User { content } = loaded.log.slice(u).unwrap() else {
            panic!("{u:?} is a user message");
        };
        assert_eq!(content.as_str(), user);
        let Slice::Assistant { content, .. } = loaded.log.slice(a).unwrap() else {
            panic!("{a:?} is an assistant message");
        };
        assert_eq!(content.as_str(), array);
        let Slice::ToolResult { block: b, .. } = loaded.log.slice(t).unwrap() else {
            panic!("{t:?} is a tool result");
        };
        assert_eq!(b.as_str(), block);
    }
}
