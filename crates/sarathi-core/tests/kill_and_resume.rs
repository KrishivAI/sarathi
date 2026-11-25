// SPDX-License-Identifier: Apache-2.0
//! The crash rule with a real crash: the test binary runs itself as a child that drives a turn on the JSONL store
//! and aborts the process right after the n-th append, for every n of the turn. The parent then resumes the session
//! in-process and finishes the turn. The side-effecting tool (which counts its runs in a file) ran at most once,
//! every tool use has exactly one result, and every logged request rebuilds to its hash.

#![expect(clippy::unwrap_used, clippy::panic, reason = "test code")]

pub mod common;

use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use common::*;
use sarathi_core::ModelProfile;
use sarathi_core::driver::Driver;
use sarathi_log::{JsonlStore, LogStore, SessionId};
use sarathi_protocol::Outcome;

const AT: &str = "SARATHI_TEST_ABORT_AT";
const DIR: &str = "SARATHI_TEST_DIR";

fn scenario() -> Vec<Reply> {
    vec![
        native(
            Some("Both."),
            &[("toolu_1", "look", "{}"), ("toolu_2", "edit", r#"{"t":1}"#)],
            "tool_use",
        ),
        native(Some("Done."), &[], "end_turn"),
        native(Some("Done again."), &[], "end_turn"),
    ]
}

const IN_EDIT: &str = "SARATHI_TEST_ABORT_IN_EDIT";

fn tools(dir: &Path) -> TestTools {
    TestTools {
        counter: Some(dir.join("writes")),
        abort_after_effect: std::env::var(IN_EDIT).is_ok(),
        ..TestTools::default()
    }
}

fn writes(dir: &Path) -> u32 {
    std::fs::read_to_string(dir.join("writes")).map_or(0, |s| s.trim().parse().unwrap())
}

/// The child: runs the turn and aborts after append `AT` (0: never). Does nothing outside a parent's run.
#[tokio::test]
async fn child() {
    let (Ok(at), Ok(dir)) = (std::env::var(AT), std::env::var(DIR)) else {
        return;
    };
    let at: u32 = at.parse().unwrap();
    let dir = Path::new(&dir).to_owned();
    let (mut d, _) = Driver::new(
        config(ModelProfile::native()),
        JsonlStore::new(dir.join("logs")),
        Scripted::new(scenario()),
        tools(&dir),
        Host(None),
    )
    .await
    .unwrap();
    std::fs::write(dir.join("session"), d.session().to_string()).unwrap();
    let count = Arc::new(AtomicU32::new(0));
    let n = count.clone();
    d.on_append(move |_| {
        if n.fetch_add(1, Ordering::SeqCst) + 1 == at {
            std::process::abort();
        }
    });
    d.run_turn(user("Go."), &mut |_| {}).await.unwrap();
    let appends = count.load(Ordering::SeqCst);
    std::fs::write(dir.join("appends"), appends.to_string()).unwrap();
}

fn run_child(dir: &Path, at: u32) -> std::process::ExitStatus {
    child_command(dir, at).status().unwrap()
}

fn child_command(dir: &Path, at: u32) -> Command {
    let mut c = Command::new(std::env::current_exe().unwrap());
    c.args(["--exact", "child", "--nocapture"])
        .env(AT, at.to_string())
        .env(DIR, dir)
        .env_remove(IN_EDIT);
    c
}

/// The log's results, after the resume of `dir`.
async fn results_of(dir: &Path) -> Vec<(String, String, bool)> {
    let session: SessionId = std::fs::read_to_string(dir.join("session"))
        .unwrap()
        .parse()
        .unwrap();
    let log = JsonlStore::new(dir.join("logs"))
        .load(session)
        .await
        .unwrap()
        .log;
    results(&log)
}

async fn resume_and_check(dir: &Path, at: u32) {
    let session: SessionId = std::fs::read_to_string(dir.join("session"))
        .unwrap()
        .parse()
        .unwrap();
    let store = JsonlStore::new(dir.join("logs"));
    let (mut d, _) = Driver::resume(
        config(ModelProfile::native()),
        store,
        session,
        Scripted::new(scenario()[1..].to_vec()),
        tools(dir),
        Host(None),
    )
    .await
    .unwrap();
    if d.in_turn() {
        let end = d.continue_turn(&mut |_| {}).await.unwrap().unwrap();
        assert_eq!(end.outcome, Outcome::Finished, "abort at {at}: {end:?}");
    }
    let log = JsonlStore::new(dir.join("logs"))
        .load(session)
        .await
        .unwrap()
        .log;
    assert!(!log.resume().in_turn, "abort at {at}: the turn ended");
    check_log(&log);
    assert!(
        writes(dir) <= 1,
        "abort at {at}: the write ran {} times",
        writes(dir)
    );
}

#[tokio::test]
async fn a_turn_killed_after_any_append_resumes_without_running_a_write_twice() {
    // How many appends the whole turn makes: a child that never aborts.
    let whole = tempfile::tempdir().unwrap();
    assert!(run_child(whole.path(), 0).success());
    let total: u32 = std::fs::read_to_string(whole.path().join("appends"))
        .unwrap()
        .parse()
        .unwrap();
    assert!(total >= 6, "{total}");
    assert_eq!(writes(whole.path()), 1);

    let mut cut_short = 0;
    for at in 1..=total {
        let dir = tempfile::tempdir().unwrap();
        let status = run_child(dir.path(), at);
        assert!(!status.success(), "the child aborts at append {at}");
        resume_and_check(dir.path(), at).await;
        if results_of(dir.path())
            .await
            .iter()
            .any(|r| r.1.contains(sarathi_core::CUT_SHORT))
        {
            cut_short += 1;
        }
    }
    assert!(
        cut_short > 0,
        "some abort point leaves a started write without its result"
    );
}

/// The window the crash rule exists for: the write's effect happened, its result was never written.
#[tokio::test]
async fn a_turn_killed_between_a_writes_effect_and_its_result_never_runs_it_again() {
    let dir = tempfile::tempdir().unwrap();
    let status = child_command(dir.path(), 0)
        .env(IN_EDIT, "1")
        .status()
        .unwrap();
    assert!(!status.success(), "the child aborts inside the write");
    assert_eq!(writes(dir.path()), 1);
    resume_and_check(dir.path(), 0).await;
    assert_eq!(writes(dir.path()), 1, "the write is not run again");
    let r = results_of(dir.path()).await;
    let edit: Vec<_> = r.iter().filter(|r| r.0 == "toolu_2").collect();
    assert_eq!(edit.len(), 1, "{r:?}");
    assert!(
        edit[0].1.contains(sarathi_core::CUT_SHORT) && edit[0].2,
        "{r:?}"
    );
}
