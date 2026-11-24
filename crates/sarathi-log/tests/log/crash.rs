// SPDX-License-Identifier: Apache-2.0
//! The crash rule: a child process writes a scripted write-tool turn and aborts after each entry in turn; the parent
//! resumes every cut and checks that a started side-effecting call is never run again and that every call is
//! answered before the next model request.

use std::path::Path;
use std::process::Command;

use sarathi_log::{Body, CUT_SHORT, CallStatus, Durability, JsonlStore, Log, LogStore, SessionId};
use sarathi_protocol::{RawJson, Seq};

use crate::common::Script;

const DIR: &str = "SARATHI_LOG_CRASH_DIR";
const AT: &str = "SARATHI_LOG_CRASH_AT";

/// The entries a scripted turn writes, with the durability the crash rule gives each.
fn durability(body: &Body) -> Durability {
    match body {
        Body::ModelResponse { .. }
        | Body::AssistantMessage { .. }
        | Body::ToolStart { .. }
        | Body::TurnEnd { .. } => Durability::Durable,
        _ => Durability::Buffered,
    }
}

/// The child: runs only when the parent sets the variables.
#[test]
fn crash_child() {
    let (Ok(dir), Ok(at)) = (std::env::var(DIR), std::env::var(AT)) else {
        return;
    };
    let at: usize = at.parse().unwrap();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    rt.block_on(async {
        let store = JsonlStore::new(&dir);
        let h = store.create(1, None).await.unwrap();
        std::fs::write(Path::new(&dir).join("session"), h.session.to_string()).unwrap();
        let mut s = Script::new(Log::new(h));
        let mut entries = vec![s.prefix()];
        entries.extend(s.tool_turn("make it blue", "toolu_01"));
        for (i, e) in entries.into_iter().enumerate() {
            let d = durability(&e.body);
            store
                .append(s.log.header().session, e.seq, vec![e], d)
                .await
                .unwrap();
            if i + 1 == at {
                std::process::abort();
            }
        }
    });
}

#[tokio::test(flavor = "multi_thread")]
async fn a_started_write_is_never_run_again_after_a_crash() {
    let exe = std::env::current_exe().unwrap();
    for at in 1..=11 {
        let dir = tempfile::tempdir().unwrap();
        let status = Command::new(&exe)
            .args([
                "--exact",
                "crash::crash_child",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(DIR, dir.path())
            .env(AT, at.to_string())
            .status()
            .unwrap();
        assert!(!status.success(), "the child aborts after entry {at}");
        let id: SessionId = std::fs::read_to_string(dir.path().join("session"))
            .unwrap()
            .parse()
            .unwrap();
        let store = JsonlStore::new(dir.path());
        let loaded = store.load(id).await.unwrap();
        assert_eq!(loaded.recovered, None, "whole lines only: at {at}");
        let log = loaded.log;
        assert_eq!(log.entries().len(), at);
        let started = log
            .entries()
            .iter()
            .any(|e| matches!(e.body, Body::ToolStart { .. }));

        // What a restarted core does: answer a started write with CUT_SHORT, run the others.
        let r = log.resume();
        let mut s = Script::new(log);
        let mut ran = Vec::new();
        let mut answers = Vec::new();
        for c in &r.calls {
            match c.status {
                CallStatus::StartedWrite => {
                    answers.push(s.push(Body::ToolResult {
                        call: c.call.id.clone(),
                        block: RawJson::string(CUT_SHORT),
                        is_error: true,
                        spill: None,
                    }));
                }
                CallStatus::NotStarted | CallStatus::StartedRead => {
                    ran.push(c.call.id.clone());
                    answers.push(s.result(&c.call.id, "ran"));
                }
                CallStatus::Done => {}
            }
        }
        if started {
            assert!(ran.is_empty(), "at {at}: a started write call ran again");
        }
        if !answers.is_empty() {
            store
                .append(id, answers[0].seq, answers, Durability::Durable)
                .await
                .unwrap();
        }
        let after = store.load(id).await.unwrap().log.resume();
        assert!(
            after.calls.iter().all(|c| c.status == CallStatus::Done),
            "at {at}: every call is answered before the next request"
        );
        assert_eq!(after.pending_model, r.pending_model);
        assert_eq!(r.next, Seq(at as u64 + 1));
    }
}
