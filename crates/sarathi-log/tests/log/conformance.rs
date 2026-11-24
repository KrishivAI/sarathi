// SPDX-License-Identifier: Apache-2.0
//! Tests every store must pass: run against [`MemoryStore`] and [`JsonlStore`].

use std::sync::Arc;

use sarathi_log::{
    Body, CallStatus, Durability, Effect, JsonlStore, Log, LogError, LogStore, MemoryStore,
    SegmentReason, Slice, encode_entry, fork, sha256_hex,
};
use sarathi_protocol::Seq;

use crate::common::{Script, assemble, raw};

async fn new_session<S: LogStore>(store: &S) -> Script {
    let h = store.create(1_764_057_600_000, None).await.unwrap();
    Script::new(Log::new(h))
}

async fn append<S: LogStore>(store: &S, s: &Script, entries: Vec<sarathi_log::Entry>) {
    let expected = entries[0].seq;
    store
        .append(
            s.log.header().session,
            expected,
            entries,
            Durability::Durable,
        )
        .await
        .unwrap();
}

async fn byte_identity<S: LogStore>(store: &S) {
    let mut s = new_session(store).await;
    let mut v = vec![s.prefix()];
    v.extend(s.tool_turn("make the header blue", "toolu_01"));
    append(store, &s, v).await;
    let loaded = store.load(s.log.header().session).await.unwrap();
    assert_eq!(loaded.recovered, None);
    assert_eq!(loaded.log, s.log);
    for (a, b) in loaded.log.entries().iter().zip(s.log.entries()) {
        assert_eq!(encode_entry(a).unwrap(), encode_entry(b).unwrap());
    }
    // The assistant slice is the response's content, byte for byte (empty thinking, 1e5, unknown block kept).
    let Slice::Assistant { content, .. } = loaded.log.slice(Seq(5)).unwrap() else {
        panic!("seq 5 is the first assistant message");
    };
    assert!(content.as_str().contains(r#""thinking":"""#));
    assert!(content.as_str().contains("1e5"));
    loaded.log.verify_requests(&assemble).unwrap();
}

async fn recipe_hash_detects_tampering<S: LogStore>(store: &S) {
    let mut s = new_session(store).await;
    let mut v = vec![s.prefix()];
    v.extend(s.tool_turn("x", "toolu_01"));
    append(store, &s, v).await;
    let log = store.load(s.log.header().session).await.unwrap().log;
    let mut entries = log.entries().to_vec();
    for e in &mut entries {
        if let Body::ToolResult { block, .. } = &mut e.body {
            *block = raw(r#"{"type":"tool_result","tool_use_id":"toolu_01","content":"ok"}"#);
        }
    }
    let mut tampered = Log::new(log.header().clone());
    tampered.append(Seq(1), entries).unwrap();
    assert!(matches!(
        tampered.verify_requests(&assemble),
        Err(LogError::HashMismatch { seq: Seq(8), .. })
    ));
}

async fn optimistic_append<S: LogStore + 'static>(store: Arc<S>) {
    let mut s = new_session(&*store).await;
    let e = s.prefix();
    let id = s.log.header().session;
    let tasks: Vec<_> = (0..8)
        .map(|_| {
            let (store, e) = (Arc::clone(&store), e.clone());
            tokio::spawn(async move {
                store
                    .append(id, Seq(1), vec![e], Durability::Buffered)
                    .await
            })
        })
        .collect();
    let mut ok = 0;
    for t in tasks {
        match t.await.unwrap() {
            Ok(next) => {
                assert_eq!(next, Seq(2));
                ok += 1;
            }
            Err(LogError::Conflict {
                expected: 1,
                actual: 2,
            }) => {}
            Err(other) => panic!("{other}"),
        }
    }
    assert_eq!(ok, 1);
    assert_eq!(store.load(id).await.unwrap().log.entries().len(), 1);
}

async fn invariants_refuse_the_whole_batch<S: LogStore>(store: &S) {
    let mut s = new_session(store).await;
    let p = s.prefix();
    let id = s.log.header().session;
    let mut wrong_seg = s.user("hi");
    wrong_seg.seg = 1;
    let r = store
        .append(id, Seq(1), vec![p.clone(), wrong_seg], Durability::Durable)
        .await;
    assert!(
        matches!(r, Err(LogError::Invalid { seq: Seq(2), .. })),
        "{r:?}"
    );
    let dangling = sarathi_log::Entry {
        seq: Seq(2),
        seg: 0,
        ts: 0,
        body: Body::AssistantMessage {
            from: Seq(1),
            calls: vec![],
        },
    };
    let r = store
        .append(id, Seq(1), vec![p, dangling], Durability::Durable)
        .await;
    assert!(
        matches!(r, Err(LogError::Invalid { seq: Seq(2), .. })),
        "{r:?}"
    );
    assert!(store.load(id).await.unwrap().log.entries().is_empty());
}

async fn resume_at_every_point<S: LogStore>(store: &S) {
    let mut s = new_session(store).await;
    let mut v = vec![s.prefix()];
    v.extend(s.tool_turn("x", "toolu_01"));
    append(store, &s, v).await;
    let log = store.load(s.log.header().session).await.unwrap().log;
    // 1 prefix, 2 user, 3 request, 4 response, 5 assistant, 6 start, 7 result, 8 request, 9 response,
    // 10 assistant, 11 end.
    let expect: [(u64, Option<CallStatus>, Option<u64>, bool); 11] = [
        (1, None, None, false),
        (2, None, None, true),
        (3, None, Some(3), true),
        (4, None, Some(3), true),
        (5, Some(CallStatus::NotStarted), None, true),
        (6, Some(CallStatus::StartedWrite), None, true),
        (7, Some(CallStatus::Done), None, true),
        (8, Some(CallStatus::Done), Some(8), true),
        (9, Some(CallStatus::Done), Some(8), true),
        (10, None, None, true),
        (11, None, None, false),
    ];
    for (at, call, pending, in_turn) in expect {
        let r = log.up_to(Seq(at)).unwrap().resume();
        assert_eq!(r.calls.first().map(|c| c.status), call, "at {at}");
        assert_eq!(r.pending_model, pending.map(Seq), "at {at}");
        assert_eq!(r.in_turn, in_turn, "at {at}");
        assert_eq!(r.next, Seq(at + 1));
    }
}

async fn a_started_read_is_run_again<S: LogStore>(store: &S) {
    let mut s = new_session(store).await;
    let mut v = vec![s.prefix(), s.user("look"), s.request()];
    v.extend(s.respond(
        r#"[{"type":"tool_use","id":"toolu_r","name":"look","input":{}},{"type":"tool_use","id":"toolu_w","name":"edit","input":{}}]"#,
        &[("toolu_r", "look"), ("toolu_w", "edit")],
    ));
    v.push(s.start("toolu_r", "look", Effect::Read));
    append(store, &s, v).await;
    let r = store
        .load(s.log.header().session)
        .await
        .unwrap()
        .log
        .resume();
    let st: Vec<_> = r.calls.iter().map(|c| c.status).collect();
    assert_eq!(st, [CallStatus::StartedRead, CallStatus::NotStarted]);
    assert!(!r.at_boundary());
}

async fn fork_projects_the_parents_next_request<S: LogStore>(store: &S) {
    let mut s = new_session(store).await;
    let mut v = vec![s.prefix()];
    v.extend(s.tool_turn("one", "toolu_01"));
    v.extend(s.tool_turn("two", "toolu_02"));
    append(store, &s, v).await;
    let parent = s.log.header().session;
    let at = Seq(11); // the first turn's end
    let child = fork(store, parent, at, 7).await.unwrap();
    let child_log = store.load(child).await.unwrap().log;
    let from = child_log.header().forked_from.clone().unwrap();
    assert_eq!((from.session, from.seq), (parent, at));
    assert_eq!(from.sha256, s.log.lines_sha256(at).unwrap());
    let params = raw(r#"{"model":"m"}"#);
    let at_parent = s.log.up_to(at).unwrap();
    let body = |log: &Log| {
        let r = log.next_recipe(params.clone()).unwrap();
        sha256_hex(&assemble(&log.request_parts(&r).unwrap()))
    };
    assert_eq!(body(&child_log), body(&at_parent));
    assert!(matches!(
        child_log.entries().last().unwrap().body,
        Body::SegmentStart {
            reason: SegmentReason::Fork,
            ..
        }
    ));
    // Inside the second turn's tool round: refused.
    for mid in [13, 14, 16, 18] {
        assert!(matches!(
            fork(store, parent, Seq(mid), 7).await,
            Err(LogError::NotABoundary(_))
        ));
    }
}

async fn a_compaction_segment_projects_its_seed_and_its_own_slices<S: LogStore>(store: &S) {
    let mut s = new_session(store).await;
    let mut v = vec![s.prefix()];
    v.extend(s.tool_turn("one", "toolu_01"));
    v.extend(s.tool_turn("two", "toolu_02"));
    // Keep the first user message and the second turn's final answer verbatim.
    v.push(s.compact(vec![Seq(2), Seq(20)], Seq(1)..Seq(22)));
    v.push(s.user("three"));
    append(store, &s, v).await;
    let log = store.load(s.log.header().session).await.unwrap().log;
    let p = log.projection();
    assert_eq!(p.segment, 1);
    assert_eq!(p.prefix, Some(Seq(1)));
    assert_eq!(p.messages, [Seq(2), Seq(20), Seq(23)]);
    // Nothing replaced is gone.
    assert!(matches!(log.slice(Seq(7)), Ok(Slice::ToolResult { .. })));
    let r = log.next_recipe(raw("{}")).unwrap();
    assert_eq!(r.messages, p.messages);
}

macro_rules! conformance {
    ($($name:ident),* $(,)?) => {
        mod memory {
            $(#[tokio::test(flavor = "multi_thread")]
            async fn $name() {
                super::$name(&sarathi_log::MemoryStore::new()).await;
            })*
        }
        mod jsonl {
            $(#[tokio::test(flavor = "multi_thread")]
            async fn $name() {
                let dir = tempfile::tempdir().unwrap();
                super::$name(&sarathi_log::JsonlStore::new(dir.path())).await;
            })*
        }
    };
}

conformance!(
    byte_identity,
    recipe_hash_detects_tampering,
    invariants_refuse_the_whole_batch,
    resume_at_every_point,
    a_started_read_is_run_again,
    fork_projects_the_parents_next_request,
    a_compaction_segment_projects_its_seed_and_its_own_slices,
);

#[tokio::test(flavor = "multi_thread")]
async fn optimistic_append_memory() {
    optimistic_append(Arc::new(MemoryStore::new())).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn optimistic_append_jsonl() {
    let dir = tempfile::tempdir().unwrap();
    optimistic_append(Arc::new(JsonlStore::new(dir.path()))).await;
}
