// SPDX-License-Identifier: Apache-2.0
//! The tools crate's toolbox, policy and hooks through the driver: a read runs without an ask, a write is asked, a
//! deny rule on the call's subject and plan mode refuse without running, hooks block, rewrite (checked again) and
//! replace output, and a toolbox's `Err` is an observation.

#![expect(clippy::unwrap_used, clippy::panic, reason = "test code")]

pub mod common;

use std::sync::{Arc, Mutex};

use common::*;
use sarathi_core::driver::{Driver, DriverConfig};
use sarathi_core::toolbox::ToolboxTools;
use sarathi_core::{Effect, ModelProfile};
use sarathi_protocol::{Decision, Outcome};
use sarathi_tools::{
    BoxFuture, Hook, Hooks, Mode, Policy, PostAction, PostTool, PreAction, PreTool, Rule, Scope,
    ToolSpec, Toolbox,
};
use serde_json::{Value, json};

/// A page the tools share: `look` reads it, `edit` sets it (the subject is the input's `path`).
#[derive(Debug, Clone, Default)]
struct Page {
    text: Arc<Mutex<String>>,
    edits: Arc<Mutex<u32>>,
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
            "look" => Ok(format!("page: {}", self.text.lock().unwrap())),
            "edit" => match input.get("text").and_then(Value::as_str) {
                Some(t) => {
                    *self.text.lock().unwrap() = t.to_owned();
                    *self.edits.lock().unwrap() += 1;
                    Ok("edited".to_owned())
                }
                None => Err("edit needs a `text`".to_owned()),
            },
            other => panic!("no tool {other}"),
        };
        Box::pin(std::future::ready(out))
    }

    fn subject(&self, name: &str, input: &Value) -> Option<String> {
        (name == "edit")
            .then(|| input.get("path").and_then(Value::as_str).map(str::to_owned))
            .flatten()
    }
}

fn edit_call(n: u32, path: &str, text: &str) -> Reply {
    native(
        None,
        &[(
            &format!("toolu_{n}"),
            "edit",
            &format!(r#"{{"path":"{path}","text":"{text}"}}"#),
        )],
        "tool_use",
    )
}

fn look_call(n: u32) -> Reply {
    native(None, &[(&format!("toolu_{n}"), "look", "{}")], "tool_use")
}

fn finished(text: &str) -> Reply {
    native(Some(text), &[], "end_turn")
}

async fn run(
    tools: ToolboxTools<Page>,
    host: Option<Decision>,
    replies: Vec<Reply>,
) -> (sarathi_log::Log, Scripted, Outcome) {
    let defs = tools.defs().unwrap();
    let model = Scripted::new(replies);
    let config =
        DriverConfig::new(ModelProfile::native(), defs, "You are a test.").with_limits(limits(10));
    let (mut d, _) = Driver::new(config, Shared::default(), model.clone(), tools, Host(host))
        .await
        .unwrap();
    let end = d
        .run_turn(user("Work on the page."), &mut |_| {})
        .await
        .unwrap();
    check_log(d.log());
    (d.log().clone(), model, end.outcome)
}

fn result_texts(log: &sarathi_log::Log) -> Vec<(String, bool)> {
    results(log).into_iter().map(|(_, b, e)| (b, e)).collect()
}

#[test]
fn a_read_only_tool_reads_and_every_other_one_writes() {
    let tools = ToolboxTools::new(Page::default(), Policy::new(Mode::Default), Hooks::new());
    let defs = tools.defs().unwrap();
    assert_eq!(defs.len(), 2);
    assert_eq!(defs[0].effect, Effect::Read);
    assert_eq!(defs[1].effect, Effect::Write);
    assert_eq!(tools.specs()[0].evidence.as_deref(), Some("look"));
}

#[tokio::test]
async fn a_read_runs_unasked_and_a_write_runs_once_the_host_allows_it() {
    let page = Page::default();
    let tools = ToolboxTools::new(page.clone(), Policy::new(Mode::Default), Hooks::new());
    let replies = vec![
        look_call(1),
        edit_call(2, "index.md", "hello"),
        finished("Done."),
    ];
    let (log, _, outcome) = run(tools, Some(Decision::AllowOnce), replies).await;
    assert_eq!(outcome, Outcome::Finished);
    assert_eq!(*page.text.lock().unwrap(), "hello");
    let asks = log
        .entries()
        .iter()
        .filter(|e| matches!(e.body, sarathi_log::Body::ToolAsk { .. }))
        .count();
    assert_eq!(asks, 1, "only the write was asked");
}

#[tokio::test]
async fn a_deny_rule_on_the_subject_and_plan_mode_refuse_without_running() {
    let mut policy = Policy::new(Mode::Bypass);
    policy.add(
        Scope::Project,
        Rule::parse(sarathi_tools::Effect::Deny, "edit(secret/*)").unwrap(),
    );
    let page = Page::default();
    let tools = ToolboxTools::new(page.clone(), policy, Hooks::new());
    let replies = vec![
        edit_call(1, "secret/keys.md", "x"),
        edit_call(2, "index.md", "fine"),
        finished("Done."),
    ];
    let (log, _, _) = run(tools, None, replies).await;
    let r = result_texts(&log);
    assert!(r[0].1 && r[0].0.contains("refused"), "{r:?}");
    assert!(!r[1].1, "{r:?}");
    assert_eq!(*page.edits.lock().unwrap(), 1, "the denied edit never ran");

    let page = Page::default();
    let tools = ToolboxTools::new(page.clone(), Policy::new(Mode::Plan), Hooks::new());
    let replies = vec![edit_call(1, "index.md", "x"), finished("Planned.")];
    let (log, _, _) = run(tools, None, replies).await;
    assert!(result_texts(&log)[0].0.contains("plan mode"));
    assert_eq!(*page.edits.lock().unwrap(), 0);
}

#[tokio::test]
async fn a_failing_tool_is_an_observation() {
    let tools = ToolboxTools::new(Page::default(), Policy::new(Mode::Bypass), Hooks::new());
    let replies = vec![
        native(None, &[("toolu_1", "edit", "{}")], "tool_use"),
        finished("Recovered."),
    ];
    let (log, model, outcome) = run(tools, None, replies).await;
    assert_eq!(outcome, Outcome::Finished);
    let r = result_texts(&log);
    assert!(r[0].1 && r[0].0.contains("needs a `text`"), "{r:?}");
    let second = String::from_utf8(model.bodies()[1].clone()).unwrap();
    assert!(second.contains("needs a `text`"));
}

/// Blocks edits of `blocked.md`, sends edits of `draft.md` to `secret/draft.md`, and tags every look's output.
struct Guard;

impl Hook for Guard {
    fn pre_tool<'a>(&'a self, e: &'a PreTool<'a>) -> BoxFuture<'a, PreAction> {
        let path = e.input.get("path").and_then(Value::as_str);
        let action = match path {
            Some("blocked.md") => PreAction::Block("blocked.md is read-only".into()),
            Some("draft.md") => {
                let mut input = e.input.clone();
                input["path"] = json!("secret/draft.md");
                PreAction::Rewrite(input)
            }
            _ => PreAction::Continue,
        };
        Box::pin(std::future::ready(action))
    }

    fn post_tool<'a>(&'a self, e: &'a PostTool<'a>) -> BoxFuture<'a, PostAction> {
        let action = if e.tool == "look" {
            PostAction::Replace(format!("[checked] {}", e.output))
        } else {
            PostAction::Keep
        };
        Box::pin(std::future::ready(action))
    }
}

#[tokio::test]
async fn hooks_block_rewrite_through_the_policy_again_and_replace_output() {
    let mut policy = Policy::new(Mode::Bypass);
    policy.add(
        Scope::Project,
        Rule::parse(sarathi_tools::Effect::Deny, "edit(secret/*)").unwrap(),
    );
    let mut hooks = Hooks::new();
    hooks.add(Arc::new(Guard));
    let page = Page::default();
    let tools = ToolboxTools::new(page.clone(), policy, hooks);
    let replies = vec![
        edit_call(1, "blocked.md", "x"),
        edit_call(2, "draft.md", "y"),
        look_call(3),
        finished("Done."),
    ];
    let (log, _, outcome) = run(tools, None, replies).await;
    assert_eq!(outcome, Outcome::Finished);
    let r = result_texts(&log);
    assert!(r[0].1 && r[0].0.contains("read-only"), "{r:?}");
    assert!(r[1].1 && r[1].0.contains("rewrote"), "{r:?}");
    assert!(r[2].0.contains("[checked] page: "), "{r:?}");
    assert_eq!(*page.edits.lock().unwrap(), 0);
}

#[tokio::test]
async fn a_rewrite_the_policy_allows_runs_with_the_new_input() {
    let mut hooks = Hooks::new();
    hooks.add(Arc::new(Guard));
    let page = Page::default();
    let tools = ToolboxTools::new(page.clone(), Policy::new(Mode::Bypass), hooks);
    let replies = vec![edit_call(1, "draft.md", "y"), finished("Done.")];
    let (log, _, outcome) = run(tools, None, replies).await;
    assert_eq!(outcome, Outcome::Finished);
    let r = result_texts(&log);
    assert!(!r[0].1 && r[0].0.contains("edited"), "{r:?}");
    assert_eq!(*page.text.lock().unwrap(), "y");
    assert_eq!(*page.edits.lock().unwrap(), 1);
}

#[tokio::test]
async fn a_rewrite_the_policy_would_ask_about_is_refused_though_the_host_allowed_the_call() {
    let mut hooks = Hooks::new();
    hooks.add(Arc::new(Guard));
    let page = Page::default();
    let tools = ToolboxTools::new(page.clone(), Policy::new(Mode::Default), hooks);
    let replies = vec![edit_call(1, "draft.md", "y"), finished("Done.")];
    let (log, _, outcome) = run(tools, Some(Decision::AllowOnce), replies).await;
    assert_eq!(outcome, Outcome::Finished);
    let r = result_texts(&log);
    assert!(r[0].1 && r[0].0.contains("needs permission"), "{r:?}");
    assert_eq!(*page.edits.lock().unwrap(), 0);
}
