// SPDX-License-Identifier: Apache-2.0
//! The completion gate through the driver: a finish with no plan, or with evidence older than the last edit, is sent
//! back to the model with what is missing; `look` evidence recorded after the last edit lets it finish; a failed
//! `look` is no evidence; the refusals of one turn are capped.

#![expect(clippy::unwrap_used, clippy::panic, reason = "test code")]

pub mod common;

use std::sync::{Arc, Mutex};

use common::*;
use sarathi_core::driver::{Driver, DriverConfig};
use sarathi_core::gate::EvidenceKinds;
use sarathi_core::toolbox::{SharedGate, ToolboxTools};
use sarathi_core::{MAX_COMPLETION_REFUSALS, ModelProfile};
use sarathi_protocol::{Decision, Outcome};
use sarathi_tools::{BoxFuture, Hooks, Mode, Policy, ToolSpec, Toolbox};
use serde_json::{Value, json};

/// A page: `look` shows it (evidence of kind `look`; it fails while `broken`), `edit` sets it.
#[derive(Debug, Clone, Default)]
struct Page {
    text: Arc<Mutex<String>>,
    broken: Arc<Mutex<bool>>,
}

impl Toolbox for Page {
    fn tools(&self) -> Vec<ToolSpec> {
        vec![
            ToolSpec::new("look", "shows the page", json!({"type":"object"}))
                .read_only()
                .evidence("look"),
            ToolSpec::new("edit", "changes the page", json!({"type":"object"})).edits(),
        ]
    }

    fn call<'a>(&'a mut self, name: &str, input: &Value) -> BoxFuture<'a, Result<String, String>> {
        let out = match name {
            "look" if *self.broken.lock().unwrap() => Err("the render failed".to_owned()),
            "look" => Ok(format!("page: {}", self.text.lock().unwrap())),
            "edit" => {
                *self.text.lock().unwrap() = input["text"].as_str().unwrap_or("").to_owned();
                Ok("edited".to_owned())
            }
            other => panic!("no tool {other}"),
        };
        Box::pin(std::future::ready(out))
    }
}

fn call(n: u32, tool: &str, input: &str) -> Reply {
    native(None, &[(&format!("toolu_{n}"), tool, input)], "tool_use")
}

fn plan(n: u32) -> Reply {
    call(
        n,
        "plan",
        r#"{"criteria":[{"id":"hero","text":"the hero shows","kind":"look"}]}"#,
    )
}

fn finish() -> Reply {
    native(Some("Done."), &[], "end_turn")
}

async fn run(
    page: &Page,
    replies: Vec<Reply>,
) -> (sarathi_log::Log, Scripted, Outcome, SharedGate) {
    run_with_steps(page, replies, 20).await
}

async fn run_with_steps(
    page: &Page,
    replies: Vec<Reply>,
    max_steps: u32,
) -> (sarathi_log::Log, Scripted, Outcome, SharedGate) {
    let (tools, gate) =
        ToolboxTools::new(page.clone(), Policy::new(Mode::AcceptEdits), Hooks::new())
            .with_gate(EvidenceKinds::new(["look"]));
    let defs = tools.defs().unwrap();
    let model = Scripted::new(replies);
    let config = DriverConfig::new(ModelProfile::native(), defs, "You are a test.")
        .with_limits(limits(max_steps))
        .with_completion(Arc::new(gate.clone()));
    let (mut d, _) = Driver::new(
        config,
        Shared::default(),
        model.clone(),
        tools,
        Host(Some(Decision::AllowOnce)),
    )
    .await
    .unwrap();
    let end = d
        .run_turn(user("Work on the page."), &mut |_| {})
        .await
        .unwrap();
    check_log(d.log());
    (d.log().clone(), model, end.outcome, gate)
}

/// How many requests the model got, and the text of the user messages the core wrote (steers) after the first.
fn steers(log: &sarathi_log::Log) -> Vec<String> {
    log.entries()
        .iter()
        .filter_map(|e| match &e.body {
            sarathi_log::Body::UserMessage {
                steer: true,
                content,
                ..
            } => Some(content.as_str().to_owned()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn a_finish_without_a_plan_is_sent_back_and_the_plan_then_look_lets_it_finish() {
    let page = Page::default();
    let replies = vec![
        finish(),
        plan(1),
        call(2, "edit", r#"{"text":"hero"}"#),
        call(3, "look", "{}"),
        finish(),
    ];
    let (log, model, outcome, gate) = run(&page, replies).await;
    assert_eq!(outcome, Outcome::Finished);
    assert_eq!(model.left(), 0);
    let s = steers(&log);
    assert_eq!(s.len(), 1, "{s:?}");
    assert!(s[0].contains("state a plan first"), "{s:?}");
    assert!(gate.verdict().is_accepted());
}

#[tokio::test]
async fn an_edit_after_the_look_makes_the_evidence_stale() {
    let page = Page::default();
    let replies = vec![
        plan(1),
        call(2, "look", "{}"),
        call(3, "edit", r#"{"text":"again"}"#),
        finish(),
        call(4, "look", "{}"),
        finish(),
    ];
    let (log, _, outcome, _) = run(&page, replies).await;
    assert_eq!(outcome, Outcome::Finished);
    let s = steers(&log);
    assert_eq!(s.len(), 1, "{s:?}");
    assert!(
        s[0].contains("hero") && s[0].contains("older than your last edit"),
        "{s:?}"
    );
}

#[tokio::test]
async fn a_look_before_the_plan_and_a_failed_look_are_no_evidence() {
    let page = Page::default();
    let replies = vec![
        call(1, "look", "{}"),
        plan(2),
        finish(),
        call(3, "look", "{}"),
        finish(),
    ];
    let (log, _, outcome, _) = run(&page, replies).await;
    assert_eq!(outcome, Outcome::Finished);
    let s = steers(&log);
    assert_eq!(s.len(), 1, "{s:?}");
    assert!(s[0].contains("no evidence yet"), "{s:?}");

    *page.broken.lock().unwrap() = true;
    let replies = vec![plan(1), call(2, "look", "{}"), finish(), finish()];
    let (log, _, _, gate) = run(&page, replies).await;
    assert!(!gate.verdict().is_accepted());
    assert!(steers(&log)[0].contains("no evidence yet"));
}

#[tokio::test]
async fn a_model_that_keeps_finishing_is_stopped_after_the_cap() {
    let page = Page::default();
    let replies = (0..=MAX_COMPLETION_REFUSALS).map(|_| finish()).collect();
    let (log, model, outcome, _) = run(&page, replies).await;
    assert_eq!(outcome, Outcome::RanOut);
    assert_eq!(model.left(), 0);
    assert_eq!(steers(&log).len(), MAX_COMPLETION_REFUSALS as usize);
}

#[tokio::test]
async fn a_refused_finish_at_the_step_cap_ends_ran_out_not_finished() {
    let page = Page::default();
    let (log, model, outcome, gate) = run_with_steps(&page, vec![finish()], 1).await;
    assert_eq!(outcome, Outcome::RanOut);
    assert_eq!(model.left(), 0);
    assert!(steers(&log).is_empty(), "no call is left to answer a steer");
    assert!(!gate.verdict().is_accepted());
}

#[tokio::test]
async fn a_bad_plan_is_an_error_result_the_model_reads() {
    let page = Page::default();
    let bad = call(
        1,
        "plan",
        r#"{"criteria":[{"id":"hero","text":"x","kind":"smell"}]}"#,
    );
    let (log, _, _, _) = run(&page, vec![bad, plan(2), call(3, "look", "{}"), finish()]).await;
    let first = results(&log).remove(0);
    assert!(first.2, "an error result: {first:?}");
    assert!(first.1.contains("smell"), "{first:?}");
}
