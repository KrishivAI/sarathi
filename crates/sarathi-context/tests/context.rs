// SPDX-License-Identifier: Apache-2.0
//! The deterministic fallback over scripted logs, and both spill stores.

#![expect(clippy::unwrap_used, reason = "test code")]

use sarathi_context::{
    ContextError, DROPPED_NOTE, FsSpill, Keep, MemorySpill, SpillLimits, SpillRecord, SpillStore,
    deterministic_seed, read_output, shrink_to_fit, spill_output, verify_spill,
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
