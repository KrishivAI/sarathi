// SPDX-License-Identifier: Apache-2.0
//! The JSONL store: torn tails, corruption, the session lock, reopening, format versions, escaped slices.

use std::path::Path;

use sarathi_log::{
    Body, Durability, JsonlStore, Log, LogError, LogStore, Recovered, SessionId, encode_entry,
};
use sarathi_protocol::{RawJson, Seq};

use crate::common::{Script, raw};

/// Writes a session with a prefix and one tool turn; returns its id and the store's root.
async fn written(root: &Path) -> (SessionId, Script) {
    let store = JsonlStore::new(root);
    let h = store.create(1, None).await.unwrap();
    let mut s = Script::new(Log::new(h));
    let mut v = vec![s.prefix()];
    v.extend(s.tool_turn("x", "toolu_01"));
    store
        .append(s.log.header().session, Seq(1), v, Durability::Durable)
        .await
        .unwrap();
    (s.log.header().session, s)
}

#[tokio::test(flavor = "multi_thread")]
async fn a_torn_last_line_is_dropped_at_every_cut_and_the_loss_is_recorded() {
    let src = tempfile::tempdir().unwrap();
    let (id, s) = written(src.path()).await;
    let bytes = std::fs::read(JsonlStore::new(src.path()).log_path(id)).unwrap();
    let last_start = bytes[..bytes.len() - 1]
        .iter()
        .rposition(|b| *b == b'\n')
        .unwrap()
        + 1;
    let before = s.log.up_to(Seq(10)).unwrap();
    for cut in last_start..bytes.len() {
        let root = tempfile::tempdir().unwrap();
        let store = JsonlStore::new(root.path());
        let path = store.log_path(id);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &bytes[..cut]).unwrap();
        let loaded = store.load(id).await.unwrap();
        let torn = (cut - last_start) as u64;
        if torn == 0 {
            assert_eq!(loaded.recovered, None, "cut {cut}");
            assert_eq!(loaded.log, before);
            continue;
        }
        assert_eq!(
            loaded.recovered,
            Some(Recovered {
                truncated_bytes: torn,
                repaired: true
            }),
            "cut {cut}"
        );
        assert_eq!(&loaded.log.entries()[..10], before.entries());
        let Body::Note { data } = &loaded.log.entries()[10].body else {
            panic!("cut {cut}: the loss is recorded as a note");
        };
        assert!(
            data.as_str()
                .contains(&format!("\"truncated_bytes\":{torn}"))
        );
        // The writer continues after the note; a fresh store reads it all with nothing to recover.
        store
            .append(id, Seq(12), vec![], Durability::Durable)
            .await
            .unwrap();
        drop(store);
        let again = JsonlStore::new(root.path()).load(id).await.unwrap();
        assert_eq!(again.recovered, None);
        assert_eq!(again.log, loaded.log);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_bad_line_before_the_last_is_corruption() {
    let root = tempfile::tempdir().unwrap();
    let (id, _) = written(root.path()).await;
    let path = JsonlStore::new(root.path()).log_path(id);
    let text = std::fs::read_to_string(&path).unwrap();
    let mut lines: Vec<&str> = text.lines().collect();
    lines[3] = "{\"seq\":3,";
    std::fs::write(&path, lines.join("\n") + "\n").unwrap();
    let r = JsonlStore::new(root.path()).load(id).await;
    assert!(matches!(r, Err(LogError::Corrupt { line: 3, .. })), "{r:?}");
    // A complete last line that does not parse is corruption too, not a torn tail.
    lines[3] = text.lines().nth(3).unwrap();
    let n = lines.len();
    lines[n - 1] = "{not json";
    std::fs::write(&path, lines.join("\n") + "\n").unwrap();
    let r = JsonlStore::new(root.path()).load(id).await;
    assert!(matches!(r, Err(LogError::Corrupt { .. })), "{r:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_second_store_reads_but_cannot_write_a_held_session() {
    let root = tempfile::tempdir().unwrap();
    let first = JsonlStore::new(root.path());
    let h = first.create(1, None).await.unwrap();
    let mut s = Script::new(Log::new(h));
    let p = s.prefix();
    first
        .append(
            s.log.header().session,
            Seq(1),
            vec![p],
            Durability::Buffered,
        )
        .await
        .unwrap();
    let second = JsonlStore::new(root.path());
    let id = s.log.header().session;
    assert_eq!(second.load(id).await.unwrap().log, s.log);
    let u = s.user("hi");
    assert!(matches!(
        second
            .append(id, Seq(2), vec![u.clone()], Durability::Buffered)
            .await,
        Err(LogError::Locked(_))
    ));
    drop(first);
    // The lock goes with the store that held it.
    second
        .append(id, Seq(2), vec![u], Durability::Durable)
        .await
        .unwrap();
    assert_eq!(second.load(id).await.unwrap().log, s.log);
}

#[tokio::test(flavor = "multi_thread")]
async fn unknown_sessions_and_newer_formats_are_refused() {
    let root = tempfile::tempdir().unwrap();
    let store = JsonlStore::new(root.path());
    assert!(matches!(
        store.load(sarathi_log::new_session_id()).await,
        Err(LogError::NotFound(_))
    ));
    let (id, _) = written(root.path()).await;
    let path = store.log_path(id);
    let text = std::fs::read_to_string(&path).unwrap();
    std::fs::write(
        &path,
        text.replacen("\"sarathi_log\":1", "\"sarathi_log\":2", 1),
    )
    .unwrap();
    assert!(matches!(
        JsonlStore::new(root.path()).load(id).await,
        Err(LogError::Format(
            sarathi_protocol::VersionError::NewerFormat {
                found: 2,
                supported: 1,
            }
        ))
    ));
}

#[tokio::test(flavor = "multi_thread")]
async fn pretty_printed_slices_come_back_byte_for_byte_from_the_file() {
    let root = tempfile::tempdir().unwrap();
    let store = JsonlStore::new(root.path());
    let h = store.create(1, None).await.unwrap();
    let id = h.session;
    let mut s = Script::new(Log::new(h));
    let pretty =
        "{\n  \"type\": \"text\",\r\n  \"text\": \"\\u00e9\\/ \\ud83d\\ude00\",\n  \"n\": -0\n}";
    let e = s.push(Body::Note { data: raw(pretty) });
    let line = encode_entry(&e).unwrap();
    assert!(line.ends_with(r#","esc":["data"]}"#));
    store
        .append(id, Seq(1), vec![e], Durability::Durable)
        .await
        .unwrap();
    drop(store);
    let log = JsonlStore::new(root.path()).load(id).await.unwrap().log;
    let Body::Note { data } = &log.entries()[0].body else {
        panic!("a note");
    };
    assert_eq!(data.as_str(), pretty);
    assert_eq!(data, &RawJson::from_text(pretty).unwrap());
}
