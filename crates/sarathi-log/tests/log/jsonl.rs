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

#[tokio::test(flavor = "multi_thread")]
async fn a_read_only_read_takes_no_lock_and_repairs_nothing() {
    let root = tempfile::tempdir().unwrap();
    let (id, s) = written(root.path()).await;
    let path = JsonlStore::new(root.path()).log_path(id);
    let mut torn = std::fs::read(&path).unwrap();
    torn.extend_from_slice(b"{\"seq\":12,");
    std::fs::write(&path, &torn).unwrap();
    let reader = JsonlStore::new(root.path());
    let loaded = reader.read(id).await.unwrap();
    assert_eq!(loaded.log, s.log);
    assert_eq!(
        loaded.recovered,
        Some(Recovered {
            truncated_bytes: 10,
            repaired: false,
        })
    );
    assert_eq!(std::fs::read(&path).unwrap(), torn, "the file is unchanged");
    // The reader holds nothing: a writer opens the session while it is alive, and repairs the tail.
    let writer = JsonlStore::new(root.path());
    let w = writer.load(id).await.unwrap();
    assert!(w.recovered.unwrap().repaired);
    assert!(matches!(
        reader.read(sarathi_log::new_session_id()).await,
        Err(LogError::NotFound(_))
    ));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_released_session_can_be_written_by_another_store() {
    let root = tempfile::tempdir().unwrap();
    let (id, mut s) = written(root.path()).await;
    let first = JsonlStore::new(root.path());
    first.load(id).await.unwrap();
    let second = JsonlStore::new(root.path());
    let u = s.user("again");
    assert!(matches!(
        second
            .append(id, u.seq, vec![u.clone()], Durability::Durable)
            .await,
        Err(LogError::Locked(_))
    ));
    assert!(first.release(id));
    assert!(!first.release(id));
    second
        .append(id, u.seq, vec![u], Durability::Durable)
        .await
        .unwrap();
    // The first store reads it again from the file (and cannot write it while the second holds it).
    assert_eq!(first.load(id).await.unwrap().log, s.log);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_fork_whose_copy_was_changed_is_refused() {
    let root = tempfile::tempdir().unwrap();
    let (id, _) = written(root.path()).await;
    let store = JsonlStore::new(root.path());
    let child = sarathi_log::fork(&store, id, Seq(11), 7).await.unwrap();
    assert!(store.load(child).await.unwrap().log.verify_fork().is_ok());
    drop(store);
    let path = JsonlStore::new(root.path()).log_path(child);
    let text = std::fs::read_to_string(&path).unwrap();
    // The user message of the copied turn, edited in place: every line still decodes and keeps the invariants.
    std::fs::write(&path, text.replacen("\"x\"", "\"y\"", 1)).unwrap();
    let r = JsonlStore::new(root.path()).read(child).await;
    assert!(
        matches!(r, Err(LogError::ForkMismatch { at: Seq(11), .. })),
        "{r:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn an_unknown_must_understand_kind_and_an_older_format_are_refused_on_load() {
    let root = tempfile::tempdir().unwrap();
    let (id, _) = written(root.path()).await;
    let path = JsonlStore::new(root.path()).log_path(id);
    let text = std::fs::read_to_string(&path).unwrap();
    let line = r#"{"seq":12,"seg":0,"ts":1,"kind":"later.kind","must_understand":true,"x":1}"#;
    std::fs::write(&path, format!("{text}{line}\n")).unwrap();
    let r = JsonlStore::new(root.path()).load(id).await;
    assert!(
        matches!(&r, Err(LogError::MustUnderstand { seq: Seq(12), kind }) if kind == "later.kind"),
        "{r:?}"
    );
    std::fs::write(
        &path,
        text.replacen("\"sarathi_log\":1", "\"sarathi_log\":0", 1),
    )
    .unwrap();
    let r = JsonlStore::new(root.path()).load(id).await;
    assert!(
        matches!(
            r,
            Err(LogError::Format(
                sarathi_protocol::VersionError::OlderFormat { .. }
            ))
        ),
        "{r:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_crash_inside_a_batch_leaves_whole_entries_that_read_as_a_valid_log() {
    let root = tempfile::tempdir().unwrap();
    // `written` appends the prefix and a whole tool turn as one batch; a fork appends its copy and its start as one.
    let (id, s) = written(root.path()).await;
    let store = JsonlStore::new(root.path());
    let child = sarathi_log::fork(&store, id, Seq(11), 7).await.unwrap();
    drop(store);
    for (session, total) in [(id, 11), (child, 12)] {
        let path = JsonlStore::new(root.path()).log_path(session);
        let full = std::fs::read_to_string(&path).unwrap();
        for k in 0..=total {
            let cut: String = full.lines().take(k + 1).map(|l| format!("{l}\n")).collect();
            std::fs::write(&path, &cut).unwrap();
            let loaded = JsonlStore::new(root.path()).read(session).await.unwrap();
            assert_eq!(loaded.recovered, None, "{session} cut after entry {k}");
            assert_eq!(loaded.log.entries().len(), k);
            if session == id {
                assert_eq!(loaded.log.entries(), &s.log.entries()[..k]);
            }
        }
        std::fs::write(&path, &full).unwrap();
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_fork_cut_short_stays_readable_and_cannot_be_finished_over_a_changed_copy() {
    let root = tempfile::tempdir().unwrap();
    let (id, _) = written(root.path()).await;
    let store = JsonlStore::new(root.path());
    let child = sarathi_log::fork(&store, id, Seq(11), 7).await.unwrap();
    let start = store.load(child).await.unwrap().log.entries()[11].clone();
    drop(store);
    let path = JsonlStore::new(root.path()).log_path(child);
    let full = std::fs::read_to_string(&path).unwrap();
    // Cut inside the copy's last line (seq 11): the writer cuts the tear and records it as seq 11.
    let keep: String = full.lines().take(11).map(|l| format!("{l}\n")).collect();
    let torn = format!("{keep}{}", &full.lines().nth(11).unwrap()[..20]);
    std::fs::write(&path, &torn).unwrap();
    let writer = JsonlStore::new(root.path());
    let repaired = writer.load(child).await.unwrap();
    assert!(repaired.recovered.unwrap().repaired);
    assert!(matches!(repaired.log.entries()[10].body, Body::Note { .. }));
    // It reads again, and again.
    assert_eq!(writer.load(child).await.unwrap().log, repaired.log);
    drop(writer);
    assert_eq!(
        JsonlStore::new(root.path()).read(child).await.unwrap().log,
        repaired.log
    );
    // Finishing the fork over the changed copy is refused; so is anything else after the copy.
    let writer = JsonlStore::new(root.path());
    let r = writer
        .append(child, Seq(12), vec![start], Durability::Durable)
        .await;
    assert!(
        matches!(r, Err(LogError::ForkMismatch { at: Seq(11), .. })),
        "{r:?}"
    );
    let note = sarathi_log::Entry {
        seq: Seq(12),
        seg: sarathi_log::SegmentId(0),
        ts: 1,
        body: Body::Note { data: raw("{}") },
    };
    let r = writer
        .append(child, Seq(12), vec![note], Durability::Durable)
        .await;
    assert!(
        matches!(r, Err(LogError::Invalid { seq: Seq(12), .. })),
        "{r:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_fork_torn_at_every_byte_of_its_last_lines_loads_and_reloads() {
    let root = tempfile::tempdir().unwrap();
    let (id, _) = written(root.path()).await;
    let store = JsonlStore::new(root.path());
    let child = sarathi_log::fork(&store, id, Seq(11), 7).await.unwrap();
    drop(store);
    let path = JsonlStore::new(root.path()).log_path(child);
    let full = std::fs::read(&path).unwrap();
    // Line starts: the last copied entry (seq 11) and the fork's segment.start (seq 12).
    let starts: Vec<usize> = std::iter::once(0)
        .chain(
            full.iter()
                .enumerate()
                .filter(|(_, b)| **b == b'\n')
                .map(|(i, _)| i + 1),
        )
        .collect();
    for cut in starts[11]..full.len() - 1 {
        if starts.contains(&cut) {
            continue;
        }
        std::fs::write(&path, &full[..cut]).unwrap();
        let first = JsonlStore::new(root.path()).load(child).await;
        let first = first.unwrap_or_else(|e| panic!("cut at {cut}: {e}"));
        assert!(first.recovered.unwrap().repaired, "cut at {cut}");
        let again = JsonlStore::new(root.path()).load(child).await;
        let again = again.unwrap_or_else(|e| panic!("cut at {cut}, reload: {e}"));
        assert_eq!(again.log, first.log, "cut at {cut}");
        assert_eq!(again.recovered, None, "cut at {cut}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_fork_point_at_the_end_of_the_seq_range_does_not_overflow() {
    let root = tempfile::tempdir().unwrap();
    let (id, _) = written(root.path()).await;
    let store = JsonlStore::new(root.path());
    let child = sarathi_log::fork(&store, id, Seq(11), 7).await.unwrap();
    drop(store);
    let path = JsonlStore::new(root.path()).log_path(child);
    let text = std::fs::read_to_string(&path).unwrap();
    std::fs::write(
        &path,
        text.replacen("\"seq\":11,", &format!("\"seq\":{},", u64::MAX), 1),
    )
    .unwrap();
    // Whatever it reads as, reading it does not overflow.
    let _ = JsonlStore::new(root.path()).read(child).await;
}
