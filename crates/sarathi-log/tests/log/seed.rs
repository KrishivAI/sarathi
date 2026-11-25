// SPDX-License-Identifier: Apache-2.0
//! A compaction call's answer as a seed: the marked `model.request`, the `segment.seed` naming its `model.response`,
//! and the `segment.start` that answers the call for resume.

use sarathi_log::{
    Body, Entry, Log, LogError, SegmentId, SegmentReason, Strategy, new_header, new_session_id,
    sha256_hex,
};
use sarathi_protocol::{Role, Seq, Usage};

use crate::common::{Script, assemble, raw};

fn session() -> Script {
    let mut s = Script::new(Log::new(new_header(
        new_session_id(),
        1_764_057_600_000,
        None,
    )));
    s.prefix();
    s.user("task");
    s.request();
    s.respond(r#"[{"type":"text","text":"done"}]"#, &[]);
    s.end();
    s
}

const BLOCK: &str = r#"[{"type":"compaction","content":"sum","signature":"c2ln"}]"#;

/// A compaction request and its response; returns their seqs.
fn compaction_call(s: &mut Script) -> (Seq, Seq) {
    let recipe = s
        .log
        .next_recipe(raw(r#"{"compaction":{"type":"summarize"}}"#))
        .unwrap();
    let sha256 = sha256_hex(&assemble(&s.log.request_parts(&recipe).unwrap()));
    let req = s
        .push(Body::ModelRequest {
            recipe,
            sha256,
            compaction: Some(Strategy::Server),
        })
        .seq;
    let body = format!(r#"{{"content":{BLOCK},"stop_reason":"compaction"}}"#);
    let resp = s
        .push(Body::ModelResponse {
            content: 11..u32::try_from(11 + BLOCK.len()).unwrap(),
            body: raw(&body),
            sse: None,
            request_id: None,
            usage: Usage::default(),
        })
        .seq;
    (req, resp)
}

fn seed_at(log: &Log, from: Seq) -> Entry {
    Entry {
        seq: log.next_seq(),
        seg: log.segment(),
        ts: 1,
        body: Body::SegmentSeed {
            role: Role::Assistant,
            content: raw(BLOCK),
            from: Some(from),
            calls: Vec::new(),
        },
    }
}

fn start_at(log: &Log, seed: Vec<Seq>) -> Entry {
    Entry {
        seq: log.next_seq(),
        seg: SegmentId(log.segment().0 + 1),
        ts: 1,
        body: Body::SegmentStart {
            reason: SegmentReason::Compaction,
            strategy: Some(Strategy::Server),
            seed,
            replaces: Some(Seq(1)..log.next_seq()),
        },
    }
}

#[test]
fn the_segment_start_that_seeds_the_answer_answers_the_compaction_call() {
    let mut s = session();
    let (req, resp) = compaction_call(&mut s);
    let mut log = s.log.clone();
    assert_eq!(log.resume().pending_model, Some(req));
    let seed = seed_at(&log, resp);
    log.append(log.next_seq(), vec![seed.clone()]).unwrap();
    assert_eq!(
        log.resume().pending_model,
        Some(req),
        "a seed slice alone answers nothing"
    );
    let start = start_at(&log, vec![seed.seq]);
    log.append(log.next_seq(), vec![start]).unwrap();
    assert_eq!(log.resume().pending_model, None);
    assert!(log.resume().at_boundary());
    assert_eq!(log.projection().messages, vec![seed.seq]);
}

#[test]
fn a_segment_start_with_any_other_seed_leaves_a_lost_call_pending() {
    let mut s = session();
    let user = Seq(2);
    let req = s.request().seq;
    let mut log = s.log.clone();
    let start = start_at(&log, vec![user]);
    log.append(log.next_seq(), vec![start]).unwrap();
    assert_eq!(log.resume().pending_model, Some(req));
}

#[test]
fn a_compaction_answer_is_never_an_assistant_message() {
    let mut s = session();
    let (_, resp) = compaction_call(&mut s);
    let mut log = s.log.clone();
    let e = Entry {
        seq: log.next_seq(),
        seg: log.segment(),
        ts: 1,
        body: Body::AssistantMessage {
            from: resp,
            calls: Vec::new(),
        },
    };
    assert!(matches!(
        log.append(log.next_seq(), vec![e]),
        Err(LogError::Invalid { detail, .. }) if detail.contains("compaction call")
    ));
}

#[test]
fn a_seed_slice_is_made_from_a_slice_or_a_model_response_only() {
    let s = session();
    let find = |f: fn(&Body) -> bool| s.log.entries().iter().find(|e| f(&e.body)).unwrap().seq;
    let resp = find(|b| matches!(b, Body::ModelResponse { .. }));
    let request = find(|b| matches!(b, Body::ModelRequest { .. }));
    let mut ok = s.log.clone();
    ok.append(ok.next_seq(), vec![seed_at(&s.log, resp)])
        .unwrap();
    let mut bad = s.log.clone();
    assert!(matches!(
        bad.append(bad.next_seq(), vec![seed_at(&s.log, request)]),
        Err(LogError::Invalid { detail, .. }) if detail.contains("model.response")
    ));
}

#[test]
fn a_deterministic_compaction_call_is_refused() {
    let s = session();
    let mut log = s.log.clone();
    let recipe = log.next_recipe(raw("{}")).unwrap();
    let e = Entry {
        seq: log.next_seq(),
        seg: log.segment(),
        ts: 1,
        body: Body::ModelRequest {
            recipe,
            sha256: "0".repeat(64),
            compaction: Some(Strategy::Deterministic),
        },
    };
    assert!(log.append(log.next_seq(), vec![e]).is_err());
}
