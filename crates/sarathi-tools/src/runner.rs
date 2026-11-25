// SPDX-License-Identifier: Apache-2.0
//! The runner: one tool call through hooks, the permission gate and the tool.

use sarathi_protocol::{CallId, RawJson, ToolOutput};
use serde_json::Value;

use crate::gate::{Access, Gate};
use crate::hook::Hooks;
use crate::tool::{ToolSpec, Toolbox};

/// What became of a call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Disposition {
    /// The tool ran (its own failure, if any, is in [`Executed::is_error`]).
    Ran,
    /// The model named no tool of this session. Nothing ran.
    Unknown,
    /// A pre-tool hook blocked the call. Nothing ran.
    Blocked,
    /// The permission gate refused the call. Nothing ran.
    Denied,
}

/// The result of one call, in the form the run loop needs.
#[derive(Debug, Clone, PartialEq)]
pub struct Executed {
    /// What happened.
    pub disposition: Disposition,
    /// What the model reads (after post-tool hooks for a call that ran).
    pub text: String,
    /// The model sees an error result: a tool failure, a refusal or an unknown tool.
    pub is_error: bool,
    /// The input the tool ran with (after pre-tool hooks), for the log.
    pub input: Value,
    /// The tool's evidence kind for a completion check, when it ran.
    pub evidence: Option<String>,
}

impl Executed {
    /// The result as the protocol's tool output, one JSON string.
    pub fn to_output(&self) -> ToolOutput {
        ToolOutput {
            content: RawJson::string(&self.text),
            is_error: self.is_error,
        }
    }

    fn refused(disposition: Disposition, text: String, input: Value) -> Self {
        Executed {
            disposition,
            text,
            is_error: true,
            input,
            evidence: None,
        }
    }
}

/// A toolbox behind hooks and a permission gate.
pub struct ToolRunner<T: Toolbox> {
    toolbox: T,
    specs: Vec<ToolSpec>,
    gate: Gate,
    hooks: Hooks,
}

impl<T: Toolbox> ToolRunner<T> {
    /// Wraps `toolbox`; its tool list is read once, here, and stays fixed for the session.
    pub fn new(toolbox: T, gate: Gate, hooks: Hooks) -> Self {
        let specs = toolbox.tools();
        ToolRunner {
            toolbox,
            specs,
            gate,
            hooks,
        }
    }

    /// The tools the session offers.
    pub fn specs(&self) -> &[ToolSpec] {
        &self.specs
    }

    /// The gate, to change the mode or add rules mid-session.
    pub fn gate_mut(&mut self) -> &mut Gate {
        &mut self.gate
    }

    /// The hooks, for the stop check.
    pub fn hooks(&self) -> &Hooks {
        &self.hooks
    }

    /// Runs one call: pre-tool hooks, the permission check, the tool, post-tool hooks. Never an error: every outcome
    /// is a result the model can read. The caller writes the start mark before this and the result after it.
    pub async fn run(&mut self, call: &CallId, name: &str, input: Value) -> Executed {
        let Some(spec) = self.specs.iter().find(|s| s.name == name).cloned() else {
            let names: Vec<&str> = self.specs.iter().map(|s| s.name.as_str()).collect();
            return Executed::refused(
                Disposition::Unknown,
                format!("unknown tool `{name}`; the tools are: {}", names.join(", ")),
                input,
            );
        };
        let input = match self.hooks.run_pre(call, name, &input).await {
            Ok(v) => v,
            Err(reason) => return Executed::refused(Disposition::Blocked, reason, input),
        };
        let subject = self.toolbox.subject(name, &input);
        if let Access::Denied(reason) = self.gate.check(call, &spec, &input, subject).await {
            return Executed::refused(Disposition::Denied, reason, input);
        }
        let (text, is_error) = match self.toolbox.call(name, &input).await {
            Ok(t) => (t, false),
            Err(t) => (t, true),
        };
        let text = self
            .hooks
            .run_post(call, name, &input, text, is_error)
            .await;
        Executed {
            disposition: Disposition::Ran,
            text,
            is_error,
            input,
            evidence: spec.evidence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::{AskRequest, PermissionHost};
    use crate::hook::{Hook, PostAction, PostTool, PreAction, PreTool};
    use crate::policy::{Mode, Policy, Scope};
    use crate::rule::{Effect, Rule};
    use crate::testutil::block_on;
    use crate::tool::{BoxFuture, Tool, ToolSet};
    use sarathi_protocol::Decision;
    use serde_json::json;
    use std::sync::{Arc, Mutex};

    /// A tool that counts how often it really ran.
    struct Counter(Arc<Mutex<u32>>, bool);
    impl Tool for Counter {
        fn spec(&self) -> ToolSpec {
            let s = ToolSpec::new("look", "d", json!({})).evidence("look");
            if self.1 { s.read_only() } else { s }
        }
        fn subject(&self, input: &Value) -> Option<String> {
            input.get("path")?.as_str().map(str::to_owned)
        }
        fn call<'a>(&'a self, input: &'a Value) -> BoxFuture<'a, Result<String, String>> {
            *self.0.lock().unwrap() += 1;
            let fail = input.get("fail").is_some();
            Box::pin(async move {
                if fail {
                    Err("boom".into())
                } else {
                    Ok("seen".into())
                }
            })
        }
    }

    struct Yes;
    impl PermissionHost for Yes {
        fn ask<'a>(&'a self, _: &'a AskRequest) -> BoxFuture<'a, Decision> {
            Box::pin(std::future::ready(Decision::AllowOnce))
        }
    }

    fn runner(read_only: bool, gate: Gate, hooks: Hooks) -> (ToolRunner<ToolSet>, Arc<Mutex<u32>>) {
        let n = Arc::new(Mutex::new(0));
        let mut set = ToolSet::new();
        set.add(Counter(n.clone(), read_only)).unwrap();
        (ToolRunner::new(set, gate, hooks), n)
    }
    fn id() -> CallId {
        "c".into()
    }

    #[test]
    fn a_permitted_call_runs_and_carries_its_evidence_kind() {
        let (mut r, n) = runner(true, Gate::new(Policy::default(), None), Hooks::new());
        let out = block_on(r.run(&id(), "look", json!({"path":"a"})));
        assert_eq!(out.disposition, Disposition::Ran);
        assert_eq!((out.text.as_str(), out.is_error), ("seen", false));
        assert_eq!(out.evidence.as_deref(), Some("look"));
        assert_eq!(*n.lock().unwrap(), 1);
        assert_eq!(out.to_output().content.as_str(), r#""seen""#);
    }

    #[test]
    fn a_tool_failure_is_an_observation() {
        let (mut r, _) = runner(true, Gate::new(Policy::default(), None), Hooks::new());
        let out = block_on(r.run(&id(), "look", json!({"fail":1})));
        assert_eq!(out.disposition, Disposition::Ran);
        assert!(out.is_error && out.text == "boom");
    }

    #[test]
    fn a_refused_call_never_reaches_the_tool() {
        // Not read-only, default mode, no host: the ask fails closed.
        let (mut r, n) = runner(false, Gate::new(Policy::default(), None), Hooks::new());
        let out = block_on(r.run(&id(), "look", json!({})));
        assert_eq!(out.disposition, Disposition::Denied);
        assert!(out.is_error && out.evidence.is_none());
        assert_eq!(*n.lock().unwrap(), 0);

        // A pattern rule on the subject the tool reports.
        let mut p = Policy::new(Mode::Bypass);
        p.add(
            Scope::Project,
            Rule::parse(Effect::Deny, "look(secret/*)").unwrap(),
        );
        let (mut r, n) = runner(true, Gate::new(p, None), Hooks::new());
        assert_eq!(
            block_on(r.run(&id(), "look", json!({"path":"secret/key"}))).disposition,
            Disposition::Denied
        );
        assert_eq!(
            block_on(r.run(&id(), "look", json!({"path":"public/a"}))).disposition,
            Disposition::Ran
        );
        assert_eq!(*n.lock().unwrap(), 1);
    }

    #[test]
    fn an_ask_goes_to_the_host_and_an_unknown_tool_lists_the_real_ones() {
        let gate = Gate::new(Policy::default(), Some(Arc::new(Yes)));
        let (mut r, n) = runner(false, gate, Hooks::new());
        assert_eq!(
            block_on(r.run(&id(), "look", json!({}))).disposition,
            Disposition::Ran
        );
        assert_eq!(*n.lock().unwrap(), 1);
        let out = block_on(r.run(&id(), "lok", json!({})));
        assert_eq!(out.disposition, Disposition::Unknown);
        assert!(out.text.contains("the tools are: look"), "{}", out.text);
    }

    struct Guard;
    impl Hook for Guard {
        fn pre_tool<'a>(&'a self, e: &'a PreTool<'a>) -> BoxFuture<'a, PreAction> {
            let a = match e.input.get("path").and_then(Value::as_str) {
                Some("..") => PreAction::Block("path escapes the project".into()),
                Some("old") => PreAction::Rewrite(json!({"path":"new"})),
                _ => PreAction::Continue,
            };
            Box::pin(std::future::ready(a))
        }
        fn post_tool<'a>(&'a self, e: &'a PostTool<'a>) -> BoxFuture<'a, PostAction> {
            Box::pin(std::future::ready(PostAction::Replace(format!(
                "[{}]",
                e.output
            ))))
        }
    }

    #[test]
    fn hooks_block_rewrite_and_post_process() {
        let mut hooks = Hooks::new();
        hooks.add(Arc::new(Guard));
        let (mut r, n) = runner(true, Gate::new(Policy::default(), None), hooks);
        let blocked = block_on(r.run(&id(), "look", json!({"path":".."})));
        assert_eq!(blocked.disposition, Disposition::Blocked);
        assert_eq!(*n.lock().unwrap(), 0);
        let out = block_on(r.run(&id(), "look", json!({"path":"old"})));
        assert_eq!(out.input, json!({"path":"new"}));
        assert_eq!(out.text, "[seen]");
    }
}

#[cfg(test)]
mod toolbox_tests {
    use super::*;
    use crate::policy::Policy;
    use crate::testutil::block_on;
    use crate::tool::BoxFuture;
    use serde_json::json;

    /// The shape of a host's own toolbox: one object, mutable state shared by its tools.
    struct Draft(Vec<String>);
    impl Toolbox for Draft {
        fn tools(&self) -> Vec<ToolSpec> {
            vec![
                ToolSpec::new("edit", "adds a line", json!({})).edits(),
                ToolSpec::new("look", "reads the draft", json!({})).read_only(),
            ]
        }
        fn call<'a>(
            &'a mut self,
            name: &str,
            input: &Value,
        ) -> BoxFuture<'a, Result<String, String>> {
            let r = match name {
                "edit" => {
                    self.0
                        .push(input["line"].as_str().unwrap_or_default().to_owned());
                    Ok("ok".to_owned())
                }
                "look" => Ok(self.0.join("\n")),
                _ => Err(format!("unknown tool `{name}`")),
            };
            Box::pin(std::future::ready(r))
        }
    }

    #[test]
    fn a_stateful_toolbox_plugs_in_unchanged() {
        let policy = Policy::new(crate::Mode::AcceptEdits);
        let mut r = ToolRunner::new(Draft(Vec::new()), Gate::new(policy, None), Hooks::new());
        let id: CallId = "c".into();
        block_on(r.run(&id, "edit", json!({"line":"a"})));
        block_on(r.run(&id, "edit", json!({"line":"b"})));
        assert_eq!(block_on(r.run(&id, "look", json!({}))).text, "a\nb");
    }
}
