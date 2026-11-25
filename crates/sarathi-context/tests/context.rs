// SPDX-License-Identifier: Apache-2.0
//! The deterministic fallback, compaction by a model call and the size decision over scripted logs, and both spill
//! stores.

#![expect(clippy::unwrap_used, clippy::panic, reason = "test code")]

use sarathi_context::{
    COMPACTION_BETA, CompactionRequest, Compactor, ContextAction, ContextError, ContextProfile,
    DROPPED_NOTE, FsSpill, Keep, MAX_INSTRUCTIONS, MemorySpill, SUMMARY_PREFIX, SpillLimits,
    SpillRecord, SpillStore, before_request, carries_compaction_block, compaction_request,
    compaction_seed, deterministic_seed, estimate, overflow_seed, pending_compaction, read_output,
    shrink_to_fit, spill_output, verify_spill,
};
use sarathi_log::{
    Body, CallRef, Durability, Effect, Entry, JsonlStore, Log, LogStore, SegmentId, SegmentReason,
    Slice, Strategy, new_header, new_session_id,
};
use sarathi_protocol::{RawJson, Seq, Usage};

fn raw(s: &str) -> RawJson {
    RawJson::from_text(s).unwrap()
}

/// A log built entry by entry.
struct Script {
    log: Log,
}

impl Script {
    fn new() -> Self {
        let mut s = Script {
            log: Log::new(new_header(new_session_id(), 1_764_057_600_000, None)),
        };
        s.push(Body::SessionPrefix {
            protocol: "anthropic-messages".into(),
            system: raw(r#""be brief""#),
            tools: raw("[]"),
        });
        s
    }

    fn push(&mut self, body: Body) -> Seq {
        let seq = self.log.next_seq();
        let seg = match body {
            Body::SegmentStart { .. } => SegmentId(self.log.segment().0 + 1),
            _ => self.log.segment(),
        };
        self.log
            .append(
                seq,
                vec![Entry {
                    seq,
                    seg,
                    ts: 1,
                    body,
                }],
            )
            .unwrap();
        seq
    }

    fn user(&mut self, text: &str) -> Seq {
        self.push(Body::UserMessage {
            content: RawJson::string(text),
            steer: false,
        })
    }

    fn respond(&mut self, content: &str, calls: &[&str]) -> Seq {
        let recipe = self.log.next_recipe(raw("{}")).unwrap();
        self.push(Body::ModelRequest {
            recipe,
            sha256: "0".repeat(64),
            compaction: None,
        });
        let body = format!(r#"{{"id":"m","content":{content}}}"#);
        let start = u32::try_from(body.find('[').unwrap()).unwrap();
        let end = u32::try_from(body.len() - 1).unwrap();
        let from = self.push(Body::ModelResponse {
            body: RawJson::from_string(body).unwrap(),
            content: start..end,
            sse: None,
            request_id: None,
            usage: Usage::default(),
        });
        self.push(Body::AssistantMessage {
            from,
            calls: calls
                .iter()
                .map(|c| CallRef {
                    id: (*c).into(),
                    name: "edit".into(),
                })
                .collect(),
        })
    }

    fn result(&mut self, call: &str) -> Seq {
        self.push(Body::ToolStart {
            call: call.into(),
            tool: "edit".into(),
            effect: Effect::Write,
        });
        self.push(Body::ToolResult {
            call: call.into(),
            block: raw(&format!(
                r#"{{"type":"tool_result","tool_use_id":"{call}","content":"ok"}}"#
            )),
            is_error: false,
            spill: None,
        })
    }

    /// A finished turn with one tool round; the first response thinks.
    fn turn(&mut self, n: u32) -> Vec<Seq> {
        let call = format!("toolu_{n}");
        let u = self.user(&format!("task {n}"));
        let a1 = self.respond(
            &format!(
                r#"[{{"type":"thinking","thinking":"plan {n}","signature":"sig{n}"}},{{"type":"tool_use","id":"{call}","name":"edit","input":{{"z":1e5}}}}]"#
            ),
            &[&call],
        );
        let r = self.result(&call);
        let a2 = self.respond(&format!(r#"[{{"type":"text","text":"done {n}"}}]"#), &[]);
        self.push(Body::TurnEnd {
            outcome: "finished".into(),
        });
        vec![u, a1, r, a2]
    }

    fn apply(&mut self, seed: &sarathi_context::Seed) {
        self.log
            .append(self.log.next_seq(), seed.entries.clone())
            .unwrap();
    }
}

fn text(log: &Log, seq: Seq) -> String {
    match log.slice(seq).unwrap() {
        Slice::User { content } => content.as_str().to_owned(),
        Slice::Assistant { content, .. } => content.as_str().to_owned(),
        Slice::ToolResult { block, .. } => block.as_str().to_owned(),
    }
}

const KEEP_HEAD: bool = true;

fn keep(turns: usize) -> Keep {
    Keep {
        turns,
        head: KEEP_HEAD,
        last: true,
    }
}

#[test]
fn the_seed_keeps_the_first_message_a_note_and_the_last_turns() {
    let mut s = Script::new();
    let t1 = s.turn(1);
    let _t2 = s.turn(2);
    let t3 = s.turn(3);
    // The current turn: a user message and one answered tool round, before the next request.
    let u4 = s.user("task 4");
    let a4 = s.respond(
        r#"[{"type":"redacted_thinking","data":"x"},{"type":"tool_use","id":"toolu_4","name":"edit","input":{}}]"#,
        &["toolu_4"],
    );
    let r4 = s.result("toolu_4");
    let before = s.log.next_seq();

    let seed = deterministic_seed(&s.log, keep(1), 7).unwrap();
    assert_eq!(seed.kept_turns, 1);
    assert!(!seed.kept_head);
    s.apply(&seed);
    let log = &s.log;
    let Some(Entry {
        seg,
        body:
            Body::SegmentStart {
                reason,
                strategy,
                replaces,
                ..
            },
        ..
    }) = log.entries().last()
    else {
        panic!("the seed ends with a segment.start");
    };
    assert_eq!(*seg, SegmentId(1));
    assert_eq!(*reason, SegmentReason::Compaction);
    assert_eq!(*strategy, Some(Strategy::Deterministic));
    assert_eq!(replaces.clone(), Some(Seq(1)..before));

    let p = log.projection().messages;
    // First user message by reference, the note, turn 3 (user, stripped copy, result, answer), turn 4 as it is.
    assert_eq!(p.len(), 2 + 4 + 3);
    assert_eq!(p[0], t1[0]);
    assert!(matches!(
        log.get(p[1]).unwrap().body,
        Body::SegmentSeed { .. }
    ));
    assert_eq!(text(log, p[1]), RawJson::string(DROPPED_NOTE).as_str());
    assert_eq!(p[2], t3[0]);
    assert!(p[3] >= before, "an earlier turn's thinking turn is a copy");
    assert_eq!(
        text(log, p[3]),
        r#"[{"type":"tool_use","id":"toolu_3","name":"edit","input":{"z":1e5}}]"#
    );
    assert_eq!(
        &p[4..6],
        &t3[2..4],
        "results and plain answers by reference"
    );
    // The current tool loop is sent as it was, thinking included: the provider needs it back unmodified.
    assert_eq!(&p[6..], &[u4, a4, r4]);
    assert!(text(log, a4).contains("redacted_thinking"));
    assert!(text(log, t3[1]).contains("plan 3"));
    // The copies open no calls and the note opens no turn: the run continues with its next request.
    let r = log.resume();
    assert!(r.at_boundary());
    assert!(r.in_turn);
}

#[test]
fn keeping_only_the_current_turn_and_compacting_twice() {
    let mut s = Script::new();
    let t1 = s.turn(1);
    s.turn(2);
    let u3 = s.user("task 3");
    let seed = deterministic_seed(&s.log, keep(0), 7).unwrap();
    s.apply(&seed);
    assert_eq!(s.log.projection().messages.len(), 3);
    assert_eq!(s.log.projection().messages[2], u3);
    // The turn goes on in the new segment, then overflows again in the next turn.
    let a = s.respond(
        r#"[{"type":"thinking","thinking":"t","signature":"s"},{"type":"text","text":"a"}]"#,
        &[],
    );
    s.push(Body::TurnEnd {
        outcome: "finished".into(),
    });
    let u4 = s.user("task 4");
    let second = deterministic_seed(&s.log, keep(1), 8).unwrap();
    s.apply(&second);
    let p = s.log.projection().messages;
    assert_eq!(s.log.segment(), SegmentId(2));
    // First message, a new note, turn 3 (its user message and the stripped answer), turn 4.
    assert_eq!(p[0], t1[0]);
    assert_eq!(p[2], u3);
    assert!(p[3] > a);
    assert_eq!(text(&s.log, p[3]), r#"[{"type":"text","text":"a"}]"#);
    assert_eq!(p[4], u4);
    assert_eq!(p.len(), 5);
}

#[test]
fn a_seed_at_a_turn_boundary_opens_no_turn() {
    let mut s = Script::new();
    for n in 1..=3 {
        s.turn(n);
    }
    assert!(!s.log.resume().in_turn);
    let seed = deterministic_seed(&s.log, keep(0), 1).unwrap();
    s.apply(&seed);
    assert!(!s.log.resume().in_turn);
    assert!(s.log.resume().at_boundary());
}

#[test]
fn a_tool_round_or_a_model_call_in_flight_is_never_cut() {
    let mut s = Script::new();
    s.turn(1);
    s.user("task 2");
    s.respond(
        r#"[{"type":"tool_use","id":"toolu_9","name":"edit","input":{}}]"#,
        &["toolu_9"],
    );
    assert!(matches!(
        deterministic_seed(&s.log, keep(0), 1),
        Err(ContextError::NotAtBoundary)
    ));
    let mut s = Script::new();
    s.turn(1);
    s.user("task 2");
    let recipe = s.log.next_recipe(raw("{}")).unwrap();
    s.push(Body::ModelRequest {
        recipe,
        sha256: "0".repeat(64),
        compaction: None,
    });
    assert!(matches!(
        deterministic_seed(&s.log, keep(0), 1),
        Err(ContextError::NotAtBoundary)
    ));
}

#[test]
fn a_segment_that_is_only_the_current_turn_has_nothing_to_drop() {
    let mut s = Script::new();
    s.user("task 1");
    assert!(matches!(
        deterministic_seed(&s.log, keep(3), 1),
        Err(ContextError::NothingToDrop)
    ));
    assert!(matches!(
        shrink_to_fit(&s.log, 3, 1, |_| true),
        Err(ContextError::NothingToDrop)
    ));
    let empty = Script::new();
    assert!(matches!(
        deterministic_seed(&empty.log, keep(0), 1),
        Err(ContextError::NoUserMessage)
    ));
}

#[test]
fn shrink_to_fit_keeps_the_most_turns_that_fit() {
    let mut s = Script::new();
    for n in 1..=6 {
        s.turn(n);
    }
    s.user("task 7");
    let len = |log: &Log| log.projection().messages.len();
    // Each kept turn is four slices; first message + note + current turn is three.
    let seed = shrink_to_fit(&s.log, 10, 1, |l| len(l) <= 3 + 4 * 2).unwrap();
    assert_eq!(seed.kept_turns, 2);
    let seed = shrink_to_fit(&s.log, 10, 1, |l| len(l) <= 3).unwrap();
    assert_eq!(seed.kept_turns, 0);
    assert!(matches!(
        shrink_to_fit(&s.log, 10, 1, |_| false),
        Err(ContextError::DoesNotFit)
    ));
    // "As many as fit" is bounded by the segment's turns, not by the number asked for.
    let mut tries = 0;
    let t = std::time::Instant::now();
    assert!(matches!(
        shrink_to_fit(&s.log, usize::MAX, 1, |_| {
            tries += 1;
            false
        }),
        Err(ContextError::DoesNotFit)
    ));
    assert_eq!(tries, 6);
    assert!(t.elapsed() < std::time::Duration::from_secs(5));
}

/// Between turns the last turn is complete: when it alone does not fit, it is dropped too, so the session goes on
/// with the first message, the note and the next user message. While a turn is open it is never dropped.
#[test]
fn between_turns_a_last_turn_too_large_is_dropped_but_never_an_open_one() {
    let mut s = Script::new();
    let t1 = s.turn(1);
    let t2 = s.turn(2);
    let big = s.user(&"y".repeat(8000));
    s.push(Body::TurnEnd {
        outcome: "failed".into(),
    });
    assert!(!s.log.resume().in_turn);
    let small = |l: &Log| !l.projection().messages.contains(&big);

    // Keeping the last turn is tried first: it wins whenever it fits.
    let seed = shrink_to_fit(&s.log, 4, 1, |_| true).unwrap();
    assert!(seed.kept_last);
    assert_eq!(
        seed.kept_turns, 1,
        "turn 1 is dropped to leave something out"
    );

    // It does not fit: the last turn goes, the earlier ones stay as far as they fit.
    let seed = shrink_to_fit(&s.log, 4, 1, small).unwrap();
    assert!(!seed.kept_last);
    assert_eq!(seed.kept_turns, 2);
    let mut l = s.log.clone();
    l.append(l.next_seq(), seed.entries.clone()).unwrap();
    let p = l.projection().messages;
    assert!(!p.contains(&big));
    assert_eq!(
        p[1], t1[0],
        "after the note, turn 1 is kept whole, first message included"
    );
    assert!(p.contains(&t2[3]));
    assert!(l.resume().at_boundary());
    assert!(!l.resume().in_turn);

    // Down to no turn at all: the first message and the note.
    let seed = shrink_to_fit(&s.log, 4, 1, |l| {
        small(l) && l.projection().messages.len() <= 2
    })
    .unwrap();
    assert!(!seed.kept_last);
    assert_eq!(seed.kept_turns, 0);
    let mut l = s.log.clone();
    l.append(l.next_seq(), seed.entries.clone()).unwrap();
    let p = l.projection().messages;
    assert_eq!(p.len(), 2);
    assert_eq!(p[0], t1[0]);

    // Asked directly, the same seed comes back; an explicit `last: false` is honoured between turns only.
    let direct = deterministic_seed(
        &s.log,
        Keep {
            turns: 0,
            head: KEEP_HEAD,
            last: false,
        },
        1,
    )
    .unwrap();
    assert_eq!(direct, seed);

    // While a turn is open, the current turn stays: dropping it is never tried.
    let open = &mut s;
    let u = open.user("task 4");
    let mut tries = 0;
    let seed = shrink_to_fit(&open.log, 4, 1, |l| {
        tries += 1;
        l.projection().messages.contains(&u)
    })
    .unwrap();
    assert!(seed.kept_last);
    let in_turn = deterministic_seed(
        &open.log,
        Keep {
            turns: 0,
            head: KEEP_HEAD,
            last: false,
        },
        1,
    )
    .unwrap();
    assert!(
        in_turn.kept_last,
        "an open turn is kept whatever `last` says"
    );
    assert!(matches!(
        shrink_to_fit(&open.log, 4, 1, |l| !l.projection().messages.contains(&u)),
        Err(ContextError::DoesNotFit)
    ));
    assert!(tries > 0);
}

/// A one-turn session whose first message alone does not fit cannot go on: the first message is always kept.
#[test]
fn a_first_message_too_large_still_does_not_fit_between_turns() {
    let mut s = Script::new();
    let big = s.user(&"y".repeat(8000));
    s.push(Body::TurnEnd {
        outcome: "failed".into(),
    });
    assert!(matches!(
        shrink_to_fit(&s.log, 4, 1, |l| !l.projection().messages.contains(&big)),
        Err(ContextError::DoesNotFit)
    ));
    // Something fits once the last turn is dropped: the first message and the note.
    let seed = shrink_to_fit(&s.log, 4, 1, |_| true).unwrap();
    assert!(!seed.kept_last, "keeping the only turn leaves nothing out");
    assert_eq!(seed.kept_turns, 0);
}

/// A segment that starts with a server compaction block: the block is the head, kept first.
#[test]
fn a_server_compaction_head_is_kept_first_and_dropped_last() {
    let mut s = Script::new();
    s.turn(1);
    s.turn(2);
    s.user("task 3");
    let block = s.respond(
        r#"[{"type":"compaction","content":"summary","signature":"sig"}]"#,
        &[],
    );
    let end = s.log.next_seq();
    s.push(Body::SegmentStart {
        reason: SegmentReason::Compaction,
        strategy: Some(Strategy::Server),
        seed: vec![block],
        replaces: Some(Seq(1)..end),
    });
    let t3 = s.turn(3);
    let t4 = s.turn(4);
    let u5 = s.user("task 5");
    // Asking for more turns than there are still drops the oldest one; the head stays.
    let seed = deterministic_seed(&s.log, keep(5), 1).unwrap();
    assert_eq!(seed.kept_turns, 1);
    assert_eq!(seed, deterministic_seed(&s.log, keep(1), 1).unwrap());
    assert!(seed.kept_head);
    let mut l = s.log.clone();
    l.append(l.next_seq(), seed.entries.clone()).unwrap();
    let p = l.projection().messages;
    assert_eq!(p[0], block, "the block stays first");
    assert!(matches!(
        l.get(p[1]).unwrap().body,
        Body::SegmentSeed { .. }
    ));
    assert_eq!(p[2], t4[0]);
    assert_eq!(*p.last().unwrap(), u5);
    assert!(!p.contains(&t3[0]));
    // When nothing with the head fits, the head goes first (before any turn) and the first user message comes back.
    let seed = shrink_to_fit(&s.log, 9, 1, |l| !l.projection().messages.contains(&block)).unwrap();
    assert!(!seed.kept_head);
    assert_eq!(seed.kept_turns, 2);
    let mut l = s.log.clone();
    l.append(l.next_seq(), seed.entries).unwrap();
    let p = l.projection().messages;
    assert_eq!(p.len(), 2 + 4 + 4 + 1);
    assert_eq!(p[0], Seq(2), "the session's first user message");
    assert_eq!(p[2], t3[0]);
    assert_eq!(*p.last().unwrap(), u5);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_seed_appends_in_one_batch_and_reads_back_from_a_file() {
    let dir = tempfile::tempdir().unwrap();
    let store = JsonlStore::new(dir.path());
    let header = store.create(1_764_057_600_000, None).await.unwrap();
    let mut s = Script {
        log: Log::new(header.clone()),
    };
    s.push(Body::SessionPrefix {
        protocol: "anthropic-messages".into(),
        system: raw(r#""be brief""#),
        tools: raw("[]"),
    });
    s.turn(1);
    s.turn(2);
    s.user("task 3");
    store
        .append(
            header.session,
            Seq(1),
            s.log.entries().to_vec(),
            Durability::Durable,
        )
        .await
        .unwrap();
    let seed = deterministic_seed(&s.log, keep(1), 1).unwrap();
    let at = s.log.next_seq();
    store
        .append(
            header.session,
            at,
            seed.entries.clone(),
            Durability::Durable,
        )
        .await
        .unwrap();
    s.apply(&seed);
    let loaded = store.load(header.session).await.unwrap().log;
    assert_eq!(loaded, s.log);
    assert_eq!(loaded.projection(), s.log.projection());
}

async fn spill_round_trip<S: SpillStore>(store: &S) {
    let session = new_session_id();
    let limits = SpillLimits {
        spill_bytes: 1000,
        preview_bytes: 200,
        read_cap: 64,
    };
    assert_eq!(
        spill_output(store, session, "toolu_1", "short", &limits)
            .await
            .unwrap(),
        None
    );
    let out: String = (0..500).map(|i| format!("{i:04} é\n")).collect();
    let spilled = spill_output(store, session, "toolu_1", &out, &limits)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(spilled.record.handle, "spill:toolu_1");
    assert_eq!(spilled.record.bytes, out.len() as u64);
    assert!(spilled.preview.starts_with("0000 é\n"));
    assert!(spilled.preview.ends_with("0499 é\n"));
    assert!(spilled.preview.contains("spill:toolu_1"));
    assert!(spilled.preview.len() < 400);
    let back = SpillRecord::from_raw(&spilled.record.to_raw().unwrap()).unwrap();
    assert_eq!(back, spilled.record);
    verify_spill(store, session, &spilled.record).await.unwrap();

    // Ranged reads of whole characters, with a cap.
    let mut all = String::new();
    let mut offset = 0u64;
    loop {
        let part = read_output(store, session, "spill:toolu_1", offset, 64, &limits)
            .await
            .unwrap();
        if part.is_empty() {
            break;
        }
        offset += part.len() as u64;
        all.push_str(&part);
    }
    assert_eq!(all, out);
    // A start inside a character moves to the next one.
    let mid = read_output(store, session, "spill:toolu_1", 6, 10, &limits)
        .await
        .unwrap();
    assert!(mid.starts_with('\n'), "{mid:?}");
    assert!(matches!(
        read_output(store, session, "spill:toolu_1", 0, 65, &limits).await,
        Err(ContextError::ReadTooLarge { asked: 65, cap: 64 })
    ));
    assert!(matches!(
        read_output(store, session, "spill:nope", 0, 8, &limits).await,
        Err(ContextError::SpillNotFound(_))
    ));
    // Any call id works, including ones that are not file names.
    for call in ["../x", "functions.edit:0", ".a"] {
        let sp = spill_output(store, session, call, &out, &limits)
            .await
            .unwrap()
            .unwrap();
        verify_spill(store, session, &sp.record).await.unwrap();
    }
    assert!(matches!(
        spill_output(store, session, "", &out, &limits).await,
        Err(ContextError::BadCallId(_))
    ));
    // A read shorter than the next character still moves on by one character.
    let pos = u64::try_from(out.find('é').unwrap()).unwrap();
    assert_eq!(
        read_output(store, session, "spill:toolu_1", pos, 1, &limits)
            .await
            .unwrap(),
        "é"
    );

    // A fork's copy reads and checks the same.
    let child = new_session_id();
    store.copy_session(session, child).await.unwrap();
    verify_spill(store, child, &spilled.record).await.unwrap();
    for call in ["../x", "functions.edit:0", ".a"] {
        let r = read_output(store, child, &format!("spill:{call}"), 0, 8, &limits)
            .await
            .unwrap();
        assert_eq!(r, out[..8]);
    }
    // Different bytes fail the check.
    store
        .put(child, "toolu_1", out.replace("0007", "7000").into_bytes())
        .await
        .unwrap();
    assert!(matches!(
        verify_spill(store, child, &spilled.record).await,
        Err(ContextError::SpillMismatch(_))
    ));
}

#[tokio::test(flavor = "multi_thread")]
async fn spill_round_trip_memory() {
    spill_round_trip(&MemorySpill::new()).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn spill_round_trip_fs() {
    let dir = tempfile::tempdir().unwrap();
    let store = FsSpill::new(dir.path());
    spill_round_trip(&store).await;
    let any = std::fs::read_dir(dir.path())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    assert!(
        store
            .path(sarathi_protocol::SessionId(0), "toolu_1")
            .ends_with("746f6f6c755f31")
    );
    assert!(any.join("spill").join("746f6f6c755f31").is_file());
    assert!(
        std::fs::read_dir(any.join("spill")).unwrap().all(|e| !e
            .unwrap()
            .file_name()
            .to_string_lossy()
            .ends_with(".tmp")),
        "no temporary file is left"
    );
}

// Compaction by a model call.

impl Script {
    /// Logs `req` as a `model.request` and `body` (an Anthropic message) as its `model.response`; returns the
    /// response's seq.
    fn call(&mut self, req: &CompactionRequest, body: &str) -> Seq {
        self.log
            .append(self.log.next_seq(), req.entries.clone())
            .unwrap();
        self.push(req.model_request("0".repeat(64)));
        self.response(body)
    }

    fn response(&mut self, body: &str) -> Seq {
        // Every test body has `stop_reason` right after `content`.
        let start = body.find(r#""content":"#).unwrap() + r#""content":"#.len();
        let end = body.find(r#","stop_reason""#).unwrap();
        self.push(Body::ModelResponse {
            body: raw(body),
            content: u32::try_from(start).unwrap()..u32::try_from(end).unwrap(),
            sse: None,
            request_id: None,
            usage: Usage::default(),
        })
    }

    /// A finished turn whose single answer reports `input` input tokens.
    fn sized_turn(&mut self, n: u32, input: u64) {
        self.user(&format!("task {n}"));
        let recipe = self.log.next_recipe(raw("{}")).unwrap();
        self.push(Body::ModelRequest {
            recipe,
            sha256: "0".repeat(64),
            compaction: None,
        });
        let body = format!(r#"{{"content":[{{"type":"text","text":"done {n}"}}]}}"#);
        let from = self.push(Body::ModelResponse {
            content: 11..u32::try_from(body.len() - 1).unwrap(),
            body: RawJson::from_string(body).unwrap(),
            sse: None,
            request_id: None,
            usage: Usage {
                input_tokens: input,
                ..Usage::default()
            },
        });
        self.push(Body::AssistantMessage {
            from,
            calls: Vec::new(),
        });
        self.push(Body::TurnEnd {
            outcome: "finished".into(),
        });
    }
}

const BLOCK: &str =
    r#"[{"type":"compaction","content":"Summary: tasks 1-3 done.","signature":"c2ln"}]"#;

fn server_answer(content: &str, stop: &str) -> String {
    format!(
        r#"{{"id":"msg_c","content":{content},"stop_reason":"{stop}","usage":{{"input_tokens":0}}}}"#
    )
}

fn last_start(log: &Log) -> (Strategy, Vec<Seq>, std::ops::Range<Seq>) {
    match &log.entries().last().unwrap().body {
        Body::SegmentStart {
            strategy: Some(st),
            seed,
            replaces: Some(r),
            ..
        } => (*st, seed.clone(), r.clone()),
        other => panic!("expected a segment.start, got {other:?}"),
    }
}

#[test]
fn a_server_compaction_seeds_the_next_segment_with_the_block_byte_for_byte() {
    let mut s = Script::new();
    for n in 1..=3 {
        s.turn(n);
    }
    let before = s.log.projection().messages;
    let req = compaction_request(
        &s.log,
        Compactor::Server,
        &raw(r#"{"model":"m","max_tokens":10}"#),
        Some("Keep file paths."),
        5,
    )
    .unwrap();
    assert!(req.entries.is_empty());
    assert_eq!(req.beta, Some(COMPACTION_BETA));
    assert_eq!(req.recipe.messages, before);
    assert_eq!(
        req.recipe.params.as_str(),
        r#"{"model":"m","max_tokens":10,"compaction":{"type":"summarize","instructions":"Keep file paths."}}"#
    );
    assert!(!carries_compaction_block(&s.log));

    let resp = s.call(&req, &server_answer(BLOCK, "compaction"));
    let pending = pending_compaction(&s.log).unwrap();
    assert_eq!(pending.compactor, Compactor::Server);
    assert_eq!(pending.response, Some(resp));
    assert!(
        !s.log.resume().at_boundary(),
        "the call is open until its seed"
    );

    let seed = compaction_seed(&s.log, 6).unwrap();
    let first_new = s.log.next_seq();
    s.apply(&seed);
    let (strategy, seed_seqs, replaces) = last_start(&s.log);
    assert_eq!(strategy, Strategy::Server);
    assert_eq!(seed_seqs, vec![first_new]);
    assert_eq!(replaces, Seq(1)..first_new);
    assert_eq!(s.log.projection().messages, vec![first_new]);
    assert_eq!(text(&s.log, first_new), BLOCK, "the block's bytes are kept");
    assert!(matches!(
        s.log.get(first_new).unwrap().body,
        Body::SegmentSeed { from: Some(f), .. } if f == resp
    ));
    assert!(s.log.resume().at_boundary());
    assert_eq!(pending_compaction(&s.log), None);
    assert!(carries_compaction_block(&s.log));

    // The next turn sends the block first, then its own slices; a second compaction replaces the block too.
    s.turn(4);
    let req = compaction_request(&s.log, Compactor::Server, &raw("{}"), None, 7).unwrap();
    assert_eq!(req.recipe.messages[0], first_new);
    assert_eq!(
        req.recipe.params.as_str(),
        r#"{"compaction":{"type":"summarize"}}"#
    );
    s.call(&req, &server_answer(BLOCK, "compaction"));
    let start_of_segment_1 = Seq(first_new.0 + 1);
    let seed = compaction_seed(&s.log, 8).unwrap();
    let newest = s.log.next_seq();
    s.apply(&seed);
    let (_, seed_seqs, replaces) = last_start(&s.log);
    assert_eq!(seed_seqs, vec![newest], "only the newest block");
    assert_eq!(replaces, start_of_segment_1..newest);
}

#[test]
fn a_server_answer_without_a_block_fails_and_is_not_asked_again_until_the_next_turn() {
    let mut s = Script::new();
    for n in 1..=3 {
        s.turn(n);
    }
    let req = compaction_request(&s.log, Compactor::Server, &raw("{}"), None, 5).unwrap();
    for (content, stop) in [
        ("[]", "max_tokens"),
        (r#"[{"type":"text","text":"Sure!"}]"#, "end_turn"),
        (r#"[{"type":"compaction","content":""}]"#, "compaction"),
    ] {
        let mut t = Script { log: s.log.clone() };
        t.call(&req, &server_answer(content, stop));
        match compaction_seed(&t.log, 6) {
            Err(ContextError::CompactionFailed { stop_reason }) => {
                assert_eq!(stop_reason.as_deref(), Some(stop));
            }
            other => panic!("{content}: {other:?}"),
        }
    }
    s.call(&req, &server_answer("[]", "max_tokens"));
    s.push(Body::ModelFailure {
        error: "invalid_response".into(),
        status: None,
        detail: "no compaction block".into(),
        request_id: None,
        retries: 0,
    });
    assert_eq!(pending_compaction(&s.log), None);
    let mut tiny = ContextProfile::new(10_000, 0);
    tiny.compact_at_percent = 1;
    assert_eq!(
        before_request(&s.log, &tiny, Some(5_000)).unwrap(),
        ContextAction::None,
        "no second try in the same gap between turns"
    );
    s.turn(4);
    assert_eq!(
        before_request(&s.log, &tiny, Some(5_000)).unwrap(),
        ContextAction::Compact(Compactor::Server)
    );
}

#[test]
fn a_client_summary_seeds_the_next_segment_with_one_user_message() {
    let mut s = Script::new();
    for n in 1..=2 {
        s.turn(n);
    }
    let before = s.log.projection().messages;
    let req =
        compaction_request(&s.log, Compactor::ClientSummary, &raw("{}"), Some("x"), 5).unwrap();
    assert_eq!(req.beta, None);
    assert_eq!(req.entries.len(), 1);
    let prompt = req.entries[0].seq;
    assert_eq!(req.recipe.messages[..before.len()], before[..]);
    assert_eq!(req.recipe.messages.last(), Some(&prompt));
    assert_eq!(req.recipe.params.as_str(), "{}");

    s.call(
        &req,
        r#"{"id":"m","content":[{"type":"text","text":"The user wants X."},{"type":"text","text":"Next: Y."}],"stop_reason":"end_turn"}"#,
    );
    assert_eq!(
        s.log.projection().messages,
        before,
        "the prompt is not part of the conversation"
    );
    assert_eq!(
        pending_compaction(&s.log).unwrap().compactor,
        Compactor::ClientSummary
    );
    let seed = compaction_seed(&s.log, 6).unwrap();
    let summary = s.log.next_seq();
    s.apply(&seed);
    let (strategy, seed_seqs, _) = last_start(&s.log);
    assert_eq!(strategy, Strategy::ClientSummary);
    assert_eq!(seed_seqs, vec![summary]);
    assert_eq!(
        text(&s.log, summary),
        RawJson::string(&format!("{SUMMARY_PREFIX}The user wants X.\n\nNext: Y.")).as_str()
    );
    assert!(s.log.resume().at_boundary());
    assert!(!carries_compaction_block(&s.log));

    // An empty answer is no summary.
    let mut t = Script::new();
    t.turn(1);
    let req = compaction_request(&t.log, Compactor::ClientSummary, &raw("{}"), None, 5).unwrap();
    for body in [
        r#"{"content":[{"type":"text","text":"  "}],"stop_reason":"end_turn"}"#,
        r#"{"content":[{"type":"text","text":"The user wants"}],"stop_reason":"max_tokens"}"#,
        r#"{"content":[{"type":"text","text":"Let me check."},{"type":"tool_use","id":"t","name":"look","input":{}}],"stop_reason":"end_turn"}"#,
        r#"{"content":"The user wants","stop_reason":null,"choices":[{"finish_reason":"length"}]}"#,
    ] {
        let mut u = Script { log: t.log.clone() };
        u.call(&req, body);
        assert!(
            matches!(
                compaction_seed(&u.log, 6),
                Err(ContextError::CompactionFailed { .. })
            ),
            "{body}"
        );
    }
}

#[test]
fn a_compaction_cut_short_at_any_entry_resumes_from_where_it_stopped() {
    for compactor in [Compactor::Server, Compactor::ClientSummary] {
        let mut s = Script::new();
        for n in 1..=2 {
            s.turn(n);
        }
        let old = s.log.projection().messages;
        let at_start = s.log.next_seq();
        let req = compaction_request(&s.log, compactor, &raw("{}"), None, 5).unwrap();
        let body = if compactor == Compactor::Server {
            server_answer(BLOCK, "compaction")
        } else {
            r#"{"content":[{"type":"text","text":"sum"}],"stop_reason":"end_turn"}"#.to_owned()
        };
        s.call(&req, &body);
        let seed = compaction_seed(&s.log, 6).unwrap();
        s.apply(&seed);
        let done = s.log.projection().messages;
        let end = s.log.next_seq().0 - 1;
        for cut in at_start.0 - 1..=end {
            let log = s.log.up_to(Seq(cut)).unwrap();
            let has_start = cut == end;
            assert_eq!(
                log.projection().messages,
                if has_start { done.clone() } else { old.clone() },
                "{compactor:?} cut at {cut}"
            );
            let pending = pending_compaction(&log);
            let has_request = log
                .entries()
                .iter()
                .skip(usize::try_from(at_start.0).unwrap() - 1)
                .any(|e| matches!(e.body, Body::ModelRequest { .. }));
            assert_eq!(
                pending.is_some(),
                has_request && !has_start,
                "{compactor:?} cut at {cut}"
            );
            if let Some(p) = pending
                && p.response.is_some()
            {
                // The answer is in the log: the seed is made again without a new call, the same as before.
                let again = compaction_seed(&log, 6).unwrap();
                let mut resumed = log.clone();
                resumed.append(resumed.next_seq(), again.entries).unwrap();
                assert_eq!(
                    text(&resumed, resumed.projection().messages[0]),
                    text(&s.log, done[0])
                );
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_compacted_log_reads_back_from_a_file() {
    let dir = tempfile::tempdir().unwrap();
    let store = JsonlStore::new(dir.path());
    let header = store.create(1_764_057_600_000, None).await.unwrap();
    let mut s = Script {
        log: Log::new(header.clone()),
    };
    s.push(Body::SessionPrefix {
        protocol: "anthropic-messages".into(),
        system: raw(r#""be brief""#),
        tools: raw("[]"),
    });
    s.turn(1);
    let req = compaction_request(&s.log, Compactor::Server, &raw("{}"), None, 5).unwrap();
    s.call(&req, &server_answer(BLOCK, "compaction"));
    let seed = compaction_seed(&s.log, 6).unwrap();
    s.apply(&seed);
    store
        .append(
            header.session,
            Seq(1),
            s.log.entries().to_vec(),
            Durability::Durable,
        )
        .await
        .unwrap();
    let back = store.load(header.session).await.unwrap().log;
    assert_eq!(back, s.log);
    assert!(carries_compaction_block(&back));
}

#[test]
fn the_estimate_is_the_last_usage_plus_the_bytes_after_it() {
    let mut s = Script::new();
    // No answer yet: 2 tokens per 7 bytes of the prefix and slices.
    let u = s.user("abcdefg");
    let bytes = r#""be brief""#.len() + "[]".len() + RawJson::string("abcdefg").len();
    assert_eq!(
        estimate(&s.log),
        u64::try_from(bytes * 2).unwrap().div_ceil(7)
    );
    let _ = u;
    s.sized_turn(1, 1_000);
    assert_eq!(estimate(&s.log), 1_000);
    s.user("abcdefg");
    assert_eq!(estimate(&s.log), 1_000 + 3);
}

#[test]
fn before_request_counts_near_a_point_compacts_before_the_limit_and_overflows_past_it() {
    // A large answer budget (32% of the window): compaction still comes first.
    let mut p = ContextProfile::new(200_000, 64_000);
    assert_eq!(p.limit(), 136_000);
    assert_eq!(p.compact_at(), 102_000);
    let mut s = Script::new();
    s.sized_turn(1, 50_000);
    assert_eq!(
        before_request(&s.log, &p, None).unwrap(),
        ContextAction::None
    );
    s.sized_turn(2, 100_000);
    assert_eq!(
        before_request(&s.log, &p, None).unwrap(),
        ContextAction::CountTokens,
        "within 10% of 102,000"
    );
    let at = |n| before_request(&s.log, &p, Some(n)).unwrap();
    assert_eq!(at(101_000), ContextAction::None);
    assert_eq!(at(103_000), ContextAction::Compact(Compactor::Server));
    assert_eq!(
        at(150_000),
        ContextAction::Compact(Compactor::Server),
        "past the limit, a summary still beats the fallback"
    );
    assert_eq!(at(200_001), ContextAction::Overflow, "past the window");

    // Inside a turn, server compaction is not possible: past the limit it overflows.
    s.user("next");
    let at = |p: &ContextProfile, n| before_request(&s.log, p, Some(n)).unwrap();
    assert_eq!(at(&p, 120_000), ContextAction::None);
    assert_eq!(at(&p, 140_000), ContextAction::Overflow);
    // A client summary may run inside a turn.
    p.compactor = Compactor::ClientSummary;
    assert_eq!(
        at(&p, 120_000),
        ContextAction::Compact(Compactor::ClientSummary)
    );

    // An open tool call: no decision at all.
    s.respond(
        r#"[{"type":"tool_use","id":"toolu_x","name":"edit","input":{}}]"#,
        &["toolu_x"],
    );
    assert!(matches!(
        before_request(&s.log, &p, None),
        Err(ContextError::NotAtBoundary)
    ));
}

#[test]
fn a_provider_overflow_forces_the_fallback_and_each_one_at_least_halves_the_request() {
    let mut s = Script::new();
    for n in 1..=6 {
        s.turn(n);
    }
    let p = ContextProfile::new(1_000_000, 0);
    let mut kept = usize::MAX;
    for round in 0..20 {
        s.user(&format!("round {round}"));
        let recipe = s.log.next_recipe(raw("{}")).unwrap();
        s.push(Body::ModelRequest {
            recipe,
            sha256: "0".repeat(64),
            compaction: None,
        });
        s.push(Body::ModelFailure {
            error: "context_overflow".into(),
            status: Some(400),
            detail: "prompt is too long".into(),
            request_id: None,
            retries: 0,
        });
        assert_eq!(
            before_request(&s.log, &p, Some(10)).unwrap(),
            ContextAction::Overflow,
            "the provider's refusal outranks the count"
        );
        match overflow_seed(&s.log, &p, 9) {
            Ok(seed) => {
                let before = estimate(&s.log);
                s.apply(&seed);
                assert!(estimate(&s.log) <= before / 2, "round {round}");
                assert!(seed.kept_turns <= kept);
                kept = seed.kept_turns;
            }
            Err(ContextError::DoesNotFit) => return,
            Err(e) => panic!("round {round}: {e}"),
        }
    }
    panic!("no end after 20 refusals");
}

#[test]
fn the_estimate_counts_bytes_when_usage_is_missing_and_after_a_compaction() {
    let mut s = Script::new();
    // `turn` reports no usage: the whole projection is counted by bytes.
    s.turn(1);
    let one = estimate(&s.log);
    assert!(one > 0);
    s.turn(2);
    assert!(estimate(&s.log) > one, "grows with the bytes");

    let req = compaction_request(&s.log, Compactor::Server, &raw("{}"), None, 5).unwrap();
    s.call(&req, &server_answer(BLOCK, "compaction"));
    let seed = compaction_seed(&s.log, 6).unwrap();
    s.apply(&seed);
    let bytes = r#""be brief""#.len() + "[]".len() + BLOCK.len();
    assert_eq!(
        estimate(&s.log),
        u64::try_from(bytes * 2).unwrap().div_ceil(7)
    );
}

#[test]
fn a_steer_sent_after_the_last_request_is_counted() {
    let mut s = Script::new();
    s.user("task");
    let recipe = s.log.next_recipe(raw("{}")).unwrap();
    s.push(Body::ModelRequest {
        recipe,
        sha256: "0".repeat(64),
        compaction: None,
    });
    s.push(Body::UserMessage {
        content: RawJson::string("abcdefg"),
        steer: true,
    });
    let body = r#"{"content":[{"type":"text","text":"ok"}]}"#;
    let from = s.push(Body::ModelResponse {
        content: 11..u32::try_from(body.len() - 1).unwrap(),
        body: raw(body),
        sse: None,
        request_id: None,
        usage: Usage {
            input_tokens: 100,
            ..Usage::default()
        },
    });
    s.push(Body::AssistantMessage {
        from,
        calls: Vec::new(),
    });
    assert_eq!(estimate(&s.log), 100 + 3);
}

#[test]
fn a_turn_request_in_flight_is_no_compaction() {
    let mut s = Script::new();
    s.turn(1);
    s.user("task 2");
    let recipe = s.log.next_recipe(raw(r#"{"model":"m"}"#)).unwrap();
    s.push(Body::ModelRequest {
        recipe,
        sha256: "0".repeat(64),
        compaction: None,
    });
    assert!(s.log.resume().pending_model.is_some());
    assert_eq!(pending_compaction(&s.log), None);
    // The first request after an overflow seed sends an earlier segment's seed slice last: still no compaction.
    let mut t = Script::new();
    for n in 1..=3 {
        t.turn(n);
    }
    let seed = deterministic_seed(&t.log, keep(0), 1).unwrap();
    t.apply(&seed);
    let recipe = t.log.next_recipe(raw("{}")).unwrap();
    t.push(Body::ModelRequest {
        recipe,
        sha256: "0".repeat(64),
        compaction: None,
    });
    assert_eq!(pending_compaction(&t.log), None);
}

#[test]
fn the_overflow_seed_fits_the_limit_by_estimate() {
    let mut s = Script::new();
    for n in 1..=6 {
        s.turn(n);
    }
    let mut p = ContextProfile::new(400, 0);
    p.max_kept_turns = 5;
    let seed = overflow_seed(&s.log, &p, 9).unwrap();
    s.apply(&seed);
    assert!(estimate(&s.log) <= 400);
    assert!(seed.kept_turns < 5);
    let tiny = ContextProfile::new(1, 0);
    let mut t = Script::new();
    for n in 1..=2 {
        t.turn(n);
    }
    assert!(matches!(
        overflow_seed(&t.log, &tiny, 9),
        Err(ContextError::DoesNotFit)
    ));
}

#[test]
fn compaction_needs_a_boundary_and_object_params() {
    let mut s = Script::new();
    s.user("task");
    s.respond(
        r#"[{"type":"tool_use","id":"toolu_1","name":"edit","input":{}}]"#,
        &["toolu_1"],
    );
    assert!(matches!(
        compaction_request(&s.log, Compactor::Server, &raw("{}"), None, 1),
        Err(ContextError::NotAtBoundary)
    ));
    s.result("toolu_1");
    assert!(
        matches!(
            compaction_request(&s.log, Compactor::Server, &raw("{}"), None, 1),
            Err(ContextError::NotAtBoundary)
        ),
        "a server block needs the next user message after it"
    );
    let client = compaction_request(&s.log, Compactor::ClientSummary, &raw("{}"), None, 1).unwrap();
    assert_eq!(client.compactor, Compactor::ClientSummary);
    s.respond(r#"[{"type":"text","text":"done"}]"#, &[]);
    s.push(Body::TurnEnd {
        outcome: "finished".into(),
    });
    assert!(matches!(
        compaction_request(&s.log, Compactor::Server, &raw("[]"), None, 1),
        Err(ContextError::BadParams(_))
    ));
    assert_eq!(
        compaction_request(&s.log, Compactor::Server, &raw("{}"), Some(""), 1)
            .unwrap()
            .recipe
            .params
            .as_str(),
        r#"{"compaction":{"type":"summarize"}}"#,
        "empty instructions are none"
    );
    let long = "x".repeat(MAX_INSTRUCTIONS + 1);
    assert!(matches!(
        compaction_request(&s.log, Compactor::Server, &raw("{}"), Some(&long), 1),
        Err(ContextError::InstructionsTooLong(n)) if n == MAX_INSTRUCTIONS + 1
    ));
    assert!(matches!(
        compaction_seed(&s.log, 1),
        Err(ContextError::NoCompaction)
    ));
}
