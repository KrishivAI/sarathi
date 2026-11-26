// SPDX-License-Identifier: Apache-2.0
//! The completion gate with the rest of the stack: a `look` the policy denies is no evidence; the plan and its
//! evidence carry across turns of one driver, and an edit in a later turn makes the old evidence stale.

#![expect(clippy::unwrap_used, clippy::panic, reason = "test code")]

pub mod common;

use std::sync::{Arc, Mutex};

use common::*;
use sarathi_core::ModelProfile;
use sarathi_core::driver::{Driver, DriverConfig};
use sarathi_core::gate::EvidenceKinds;
use sarathi_core::toolbox::{SharedGate, ToolboxTools};
use sarathi_protocol::{Decision, Outcome};
use sarathi_tools::{BoxFuture, Effect, Hooks, Mode, Policy, Rule, Scope, ToolSpec, Toolbox};
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

async fn driver(
    page: &Page,
    policy: Policy,
    replies: Vec<Reply>,
) -> (
    Driver<Shared, Scripted, ToolboxTools<Page>, Host>,
    Scripted,
    SharedGate,
) {
    let (tools, gate) = ToolboxTools::new(page.clone(), policy, Hooks::new())
        .with_gate(EvidenceKinds::new(["look"]))
        .unwrap();
    let defs = tools.defs().unwrap();
    let model = Scripted::new(replies);
    let config = DriverConfig::new(ModelProfile::native(), defs, "You are a test.")
        .with_limits(limits(20))
        .with_completion(Arc::new(gate.clone()));
    let (d, _) = Driver::new(
        config,
        Shared::default(),
        model.clone(),
        tools,
        Host(Some(Decision::AllowOnce)),
    )
    .await
    .unwrap();
    (d, model, gate)
}

/// The text of the user messages the core wrote after the first (the steers: a finish sent back).
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
async fn a_look_the_policy_denies_is_no_evidence() {
    let page = Page::default();
    let mut policy = Policy::new(Mode::AcceptEdits);
    policy.add(Scope::Project, Rule::parse(Effect::Deny, "look").unwrap());
    let replies = vec![
        plan(1),
        call(2, "look", "{}"),
        finish(),
        finish(),
        finish(),
        finish(),
    ];
    let (mut d, _, gate) = driver(&page, policy, replies).await;
    let end = d.run_turn(user("Work."), &mut |_| {}).await.unwrap();
    check_log(d.log());
    assert_eq!(end.outcome, Outcome::RanOut, "no evidence is ever recorded");
    assert!(!gate.verdict().is_accepted());
    assert_eq!(
        steers(d.log()).len(),
        sarathi_core::MAX_COMPLETION_REFUSALS as usize,
        "the run ended by the gate's refusal limit, not the step limit"
    );
    let first = results(d.log())
        .into_iter()
        .find(|r| r.0 == "toolu_2")
        .unwrap();
    assert!(first.2, "the denied look is an error result: {first:?}");
}

#[tokio::test]
async fn the_plan_carries_over_and_an_edit_in_a_later_turn_makes_the_evidence_stale() {
    let page = Page::default();
    let replies = vec![
        plan(1),
        call(2, "look", "{}"),
        finish(),
        // turn 2
        call(3, "edit", r#"{"text":"new"}"#),
        finish(),
        call(4, "look", "{}"),
        finish(),
    ];
    let (mut d, model, gate) = driver(&page, Policy::new(Mode::AcceptEdits), replies).await;
    let one = d.run_turn(user("Work."), &mut |_| {}).await.unwrap();
    assert_eq!(one.outcome, Outcome::Finished);
    assert!(gate.verdict().is_accepted());
    let two = d.run_turn(user("Change it."), &mut |_| {}).await.unwrap();
    check_log(d.log());
    assert_eq!(two.outcome, Outcome::Finished);
    assert_eq!(model.left(), 0);
    let steers = steers(d.log());
    assert_eq!(
        steers.len(),
        1,
        "only the finish after the edit is sent back: {steers:?}"
    );
    assert!(gate.verdict().is_accepted());
}

#[tokio::test]
async fn a_look_that_fails_records_no_evidence() {
    let page = Page::default();
    *page.broken.lock().unwrap() = true;
    let replies = vec![
        plan(1),
        call(2, "look", "{}"),
        finish(),
        finish(),
        finish(),
        finish(),
    ];
    let (mut d, _, gate) = driver(&page, Policy::new(Mode::AcceptEdits), replies).await;
    let end = d.run_turn(user("Work."), &mut |_| {}).await.unwrap();
    check_log(d.log());
    assert_eq!(end.outcome, Outcome::RanOut);
    assert!(
        !gate.verdict().is_accepted(),
        "a failed look is no evidence"
    );
    let first = results(d.log())
        .into_iter()
        .find(|r| r.0 == "toolu_2")
        .unwrap();
    assert!(first.2, "the failed look is an error result: {first:?}");
}

#[tokio::test]
async fn a_host_tool_named_plan_is_refused_by_the_gate() {
    #[derive(Debug)]
    struct Planner;
    impl Toolbox for Planner {
        fn tools(&self) -> Vec<ToolSpec> {
            vec![ToolSpec::new("plan", "the host's own", json!({"type":"object"})).read_only()]
        }
        fn call<'a>(&'a mut self, _: &str, _: &Value) -> BoxFuture<'a, Result<String, String>> {
            Box::pin(std::future::ready(Ok(String::new())))
        }
    }
    let tools = ToolboxTools::new(Planner, Policy::new(Mode::AcceptEdits), Hooks::new());
    assert!(tools.with_gate(EvidenceKinds::new(["look"])).is_err());
}
