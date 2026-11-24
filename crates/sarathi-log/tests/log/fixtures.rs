// SPDX-License-Identifier: Apache-2.0
//! The committed format fixtures (`fixtures/log/v<N>/`). Every fixture of every format this reader supports must
//! load, rebuild every model request to its recorded hash, and resume. A format change adds a new directory and keeps
//! the old ones green: that is the "a log written by this commit is read by the next" test.
//!
//! Regenerate the current format's fixtures with `SARATHI_LOG_WRITE_FIXTURES=1 cargo test -p sarathi-log fixtures`;
//! never regenerate an older format's.

use std::path::{Path, PathBuf};

use sarathi_log::{
    Body, Durability, JsonlStore, Log, LogStore, MemoryStore, decode_header, fork, render_log,
};
use sarathi_protocol::{RawJson, Seq, Usage};

use crate::common::{Script, assemble};

fn dir(version: u32) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/log/v{version}"))
}

fn render(log: &Log) -> String {
    render_log(log).unwrap()
}

async fn session(store: &MemoryStore) -> Script {
    let h = store.create(1_764_057_600_000, None).await.unwrap();
    let mut s = Script::new(Log::new(h));
    s.prefix();
    s
}

async fn save(store: &MemoryStore, s: &Script) {
    let entries = s.log.entries().to_vec();
    store
        .append(s.log.header().session, Seq(1), entries, Durability::Durable)
        .await
        .unwrap();
}

/// Writes the current format's fixtures.
async fn write_fixtures(out: &Path) {
    std::fs::create_dir_all(out).unwrap();
    let store = MemoryStore::new();
    let write = |name: &str, text: &str| std::fs::write(out.join(name), text).unwrap();

    let mut plain = session(&store).await;
    plain.user("Make the header blue.");
    plain.request();
    plain.respond(r#"[{"type":"text","text":"Which blue?"}]"#, &[]);
    plain.end();
    write("plain.jsonl", &render(&plain.log));

    let mut tool = session(&store).await;
    tool.tool_turn("Make the header blue.", "toolu_01");
    let tool_text = render(&tool.log);
    write("tool.jsonl", &tool_text);

    // A streamed turn keeps the stream text; a pretty-printed body is stored escaped.
    let mut streamed = session(&store).await;
    streamed.user("hi");
    streamed.request();
    let body = "{\n  \"id\": \"msg_2\",\n  \"content\": [{\"type\": \"text\", \"text\": \"h\\u00e9\"}],\n  \"stop_reason\": \"end_turn\"\n}";
    let start = u32::try_from(body.find('[').unwrap()).unwrap();
    let end = u32::try_from(body.rfind(']').unwrap() + 1).unwrap();
    let r = streamed.push(Body::ModelResponse {
        body: RawJson::from_text(body).unwrap(),
        content: start..end,
        sse: Some("event: message_start\ndata: {\"type\":\"message_start\"}\n\n".into()),
        request_id: Some("req_2".into()),
        usage: Usage {
            input_tokens: 3,
            output_tokens: 2,
            ..Usage::default()
        },
    });
    streamed.push(Body::AssistantMessage {
        from: r.seq,
        calls: vec![],
    });
    streamed.end();
    write("streamed.jsonl", &render(&streamed.log));

    let mut compacted = session(&store).await;
    compacted.tool_turn("one", "toolu_01");
    compacted.tool_turn("two", "toolu_02");
    compacted.compact(vec![Seq(2), Seq(20)], Seq(1)..Seq(22));
    compacted.user("three");
    compacted.request();
    write("compaction.jsonl", &render(&compacted.log));

    let mut parent = session(&store).await;
    parent.tool_turn("one", "toolu_01");
    parent.tool_turn("two", "toolu_02");
    save(&store, &parent).await;
    let child = fork(
        &store,
        parent.log.header().session,
        Seq(11),
        1_764_057_700_000,
    )
    .await
    .unwrap();
    write("fork-parent.jsonl", &render(&parent.log));
    write(
        "fork-child.jsonl",
        &render(&store.load(child).await.unwrap().log),
    );

    // A later kind this reader does not know, not marked must_understand: kept and skipped.
    let mut unknown = render(&plain.log);
    let next = plain.log.next_seq().0;
    unknown.push_str(&format!(
        "{{\"seq\":{next},\"seg\":0,\"ts\":1764057600100,\"kind\":\"later.kind\",\"x\":[1e5]}}\n"
    ));
    write("unknown-kind.jsonl", &unknown);

    // A crash in the middle of the last line.
    let cut = tool_text.len() - 40;
    write("torn-tail.jsonl", &tool_text[..cut]);
}

#[tokio::test(flavor = "multi_thread")]
async fn fixtures_of_every_supported_format_load_rebuild_and_resume() {
    if std::env::var_os("SARATHI_LOG_WRITE_FIXTURES").is_some() {
        write_fixtures(&dir(sarathi_log::FORMAT_VERSION)).await;
    }
    let mut seen = 0;
    for version in sarathi_log::OLDEST_READABLE..=sarathi_log::FORMAT_VERSION {
        let mut files: Vec<_> = std::fs::read_dir(dir(version))
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        files.sort();
        for path in files {
            check(&path).await;
            seen += 1;
        }
    }
    assert!(seen >= 8, "fixtures missing: {seen}");
}

async fn check(path: &Path) {
    let name = path.file_name().unwrap().to_string_lossy().into_owned();
    let text = std::fs::read_to_string(path).unwrap();
    let header = decode_header(text.lines().next().unwrap()).unwrap();
    let root = tempfile::tempdir().unwrap();
    let store = JsonlStore::new(root.path());
    let copy = store.log_path(header.session);
    std::fs::create_dir_all(copy.parent().unwrap()).unwrap();
    std::fs::write(&copy, &text).unwrap();
    let loaded = store.load(header.session).await.unwrap();
    let log = &loaded.log;
    log.verify_requests(&assemble)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    if name == "torn-tail.jsonl" {
        let r = loaded.recovered.unwrap();
        assert!(r.repaired && r.truncated_bytes > 0, "{name}");
        return;
    }
    assert_eq!(loaded.recovered, None, "{name}");
    // This writer renders the same bytes: the format did not drift within its version.
    assert_eq!(render(log), text, "{name}");
    let resume = log.resume();
    assert!(
        resume
            .calls
            .iter()
            .all(|c| c.status != sarathi_log::CallStatus::StartedWrite),
        "{name}"
    );
    match name.as_str() {
        "compaction.jsonl" => {
            assert_eq!(log.projection().messages, [Seq(2), Seq(20), Seq(23)]);
            assert_eq!(resume.pending_model, Some(Seq(24)));
        }
        "fork-child.jsonl" => {
            let from = log.header().forked_from.clone().unwrap();
            assert_eq!(from.seq, Seq(11));
            assert!(resume.at_boundary());
        }
        "unknown-kind.jsonl" => {
            assert_eq!(log.entries().last().unwrap().body.kind(), "later.kind");
            assert!(resume.at_boundary());
        }
        _ => assert!(resume.at_boundary(), "{name}"),
    }
}
