// SPDX-License-Identifier: Apache-2.0
//! A compaction call's answer as a seed: `segment.seed` naming a `model.response`, and the `segment.start` that
//! answers the call for resume.

use sarathi_log::{
    Body, Entry, Log, LogError, SegmentId, SegmentReason, Strategy, new_header, new_session_id,
};
use sarathi_protocol::{Role, Seq};

use crate::common::{Script, raw};

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

fn seed_at(log: &Log, from: Seq) -> Entry {
    Entry {
        seq: log.next_seq(),
        seg: log.segment(),
        ts: 1,
        body: Body::SegmentSeed {
            role: Role::Assistant,
            content: raw(r#"[{"type":"compaction","content":"sum"}]"#),
            from: Some(from),
            calls: Vec::new(),
        },
    }
}

#[test]
fn a_segment_start_answers_the_compaction_call_before_it() {
    let mut s = session();
    let req = s.request();
    let resp = s.respond(r#"[{"type":"compaction","content":"sum"}]"#, &[]);
    // The compaction answer is not accepted as a message: drop the `message.assistant` the helper wrote.
    let mut log = s.log.up_to(resp[0].seq).unwrap();
    assert_eq!(log.resume().pending_model, Some(req.seq));
    let seed = seed_at(&log, resp[0].seq);
    let start = Entry {
        seq: Seq(seed.seq.0 + 1),
        seg: SegmentId(1),
        ts: 1,
        body: Body::SegmentStart {
            reason: SegmentReason::Compaction,
            strategy: Some(Strategy::Server),
            seed: vec![seed.seq],
            replaces: Some(Seq(1)..seed.seq),
        },
    };
    log.append(log.next_seq(), vec![seed.clone()]).unwrap();
    assert_eq!(
        log.resume().pending_model,
        Some(req.seq),
        "a seed slice alone answers nothing"
    );
    log.append(log.next_seq(), vec![start]).unwrap();
    assert_eq!(log.resume().pending_model, None);
    assert!(log.resume().at_boundary());
    assert_eq!(log.projection().messages, vec![seed.seq]);
}

#[test]
fn a_seed_slice_is_made_from_a_slice_or_a_model_response_only() {
    let s = session();
    let resp = s
        .log
        .entries()
        .iter()
        .find(|e| matches!(e.body, Body::ModelResponse { .. }))
        .unwrap()
        .seq;
    let mut ok = s.log.clone();
    ok.append(ok.next_seq(), vec![seed_at(&s.log, resp)])
        .unwrap();
    let request = s
        .log
        .entries()
        .iter()
        .find(|e| matches!(e.body, Body::ModelRequest { .. }))
        .unwrap()
        .seq;
    let mut bad = s.log.clone();
    assert!(matches!(
        bad.append(bad.next_seq(), vec![seed_at(&s.log, request)]),
        Err(LogError::Invalid { detail, .. }) if detail.contains("model.response")
    ));
}
