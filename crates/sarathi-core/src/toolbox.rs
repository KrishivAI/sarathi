// SPDX-License-Identifier: Apache-2.0
//! A host's tools from the tools crate behind the driver's [`Tools`] seam: a [`Toolbox`], the [`Policy`] over it and
//! the [`Hooks`] around each call ([`ToolboxTools`]).
//!
//! The tools crate's [`sarathi_tools::ToolRunner`] does a whole call in one future (hooks, permission, tool). The core
//! splits a call instead, so an ask can be logged and answered through ops and a restart can resume it: the verdict
//! comes first and without IO ([`Tools::evaluate`]: the policy on the call's subject), an ask goes to the host as an
//! event, and [`Tools::run`] runs the pre-tool hooks, the tool and the post-tool hooks. The host's "allow always" is
//! kept by the core (it survives a restart in the log), not by a session rule.
//!
//! Two consequences of the split: a pre-tool hook runs after the permission check, and a hook that rewrites the input
//! sends the rewritten call through the policy again, where only an allow lets it run (the host agreed to the call
//! the model made, not to the rewrite). The host's "allow always" is the core's, not the policy's, so it does not
//! cover a rewrite either: a hook that rewrites a tool the policy would ask about refuses it on every call. The log
//! keeps the model's input; the rewritten input that ran is not recorded. A toolbox has no stop signal: an
//! interrupted side-effecting call runs until the driver's grace period ends and is then dropped.

use std::sync::{Arc, Mutex as StdMutex, MutexGuard, PoisonError};

use sarathi_protocol::{CallId, RawJson, ToolOutput};
use sarathi_tools::{Hooks, Policy, ToolSpec, Toolbox, Verdict};
use serde_json::Value;
use tokio::sync::Mutex;

use crate::core::Completion;
use crate::driver::{BoxFuture, StopSignal, Tools};
use crate::gate::{Criterion, EvidenceKinds, Gate};
use crate::profile::{Effect, ToolDef};

/// The name of the tool [`ToolboxTools::with_gate`] adds: the model states the criteria a finish is checked against.
pub const PLAN_TOOL: &str = "plan";

/// The completion gate shared by the tools that feed it and the core that asks it ([`Completion`]).
#[derive(Debug, Clone)]
pub struct SharedGate(Arc<StdMutex<Gate>>);

impl SharedGate {
    fn lock(&self) -> MutexGuard<'_, Gate> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The gate's verdict now.
    #[must_use]
    pub fn verdict(&self) -> crate::gate::Verdict {
        self.lock().check()
    }
}

impl Completion for SharedGate {
    fn check(&self) -> Option<String> {
        let v = self.lock().check();
        (!v.is_accepted()).then(|| v.to_string())
    }
}

/// What the `plan` tool takes: criteria, each with an id, the words, and the evidence kind that shows it.
#[derive(serde::Deserialize)]
struct PlanInput {
    criteria: Vec<PlanCriterion>,
}

#[derive(serde::Deserialize)]
struct PlanCriterion {
    id: String,
    text: String,
    kind: String,
}

/// A [`Toolbox`] with its policy and hooks, as the driver's [`Tools`]. Calls are made one at a time.
pub struct ToolboxTools<T: Toolbox> {
    toolbox: Mutex<T>,
    specs: Vec<ToolSpec>,
    policy: Policy,
    hooks: Hooks,
    gate: Option<SharedGate>,
}

impl<T: Toolbox> std::fmt::Debug for ToolboxTools<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolboxTools")
            .field(
                "tools",
                &self.specs.iter().map(|s| &s.name).collect::<Vec<_>>(),
            )
            .field("mode", &self.policy.mode())
            .finish_non_exhaustive()
    }
}

/// The host already has a tool named [`PLAN_TOOL`], which the gate's own tool would shadow.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("the host has a tool named `plan`; the completion gate needs that name for its own tool")]
pub struct PlanNameTaken;

/// A tool's schema is not one a request can carry.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("the schema of tool `{tool}` cannot be rendered: {detail}")]
pub struct BadSchema {
    /// The tool.
    pub tool: Box<str>,
    /// What went wrong.
    pub detail: Box<str>,
}

impl<T: Toolbox> ToolboxTools<T> {
    /// The toolbox's tools (read once: they are the session's fixed prefix) under `policy`, with `hooks`.
    pub fn new(toolbox: T, policy: Policy, hooks: Hooks) -> Self {
        let specs = toolbox.tools();
        ToolboxTools {
            toolbox: Mutex::new(toolbox),
            specs,
            policy,
            hooks,
            gate: None,
        }
    }

    /// The same tools feeding a completion gate for the evidence `kinds` the host offers (the kinds its tools name
    /// with [`ToolSpec::evidence`]), and a `plan` tool ([`PLAN_TOOL`], read-only) through which the model states the
    /// criteria a finish is checked against. Call before [`ToolboxTools::defs`]: the plan tool is part of the
    /// session's fixed tool list.
    ///
    /// The gate sees every call that ran: a tool that is not read-only counts as an edit (failed or not, since a
    /// failed write may have changed something), and a successful result of a tool with an evidence kind is evidence
    /// of that kind for every criterion asking for it, fresh when no edit follows. Pass the returned [`SharedGate`]
    /// to the driver ([`crate::driver::DriverConfig::with_completion`]): a model that stops without a plan or
    /// without fresh evidence is told what is missing and goes on, up to [`crate::MAX_COMPLETION_REFUSALS`] times a
    /// turn. The gate is memory only: a resumed session starts with no plan and the model is asked for one.
    ///
    /// # Errors
    /// [`PlanNameTaken`] when the host's toolbox already has a tool named [`PLAN_TOOL`].
    pub fn with_gate(mut self, kinds: EvidenceKinds) -> Result<(Self, SharedGate), PlanNameTaken> {
        if self.specs.iter().any(|s| s.name == PLAN_TOOL) {
            return Err(PlanNameTaken);
        }
        let gate = SharedGate(Arc::new(StdMutex::new(Gate::new(kinds.clone()))));
        let kinds_text = kinds.iter().collect::<Vec<_>>().join(", ");
        self.specs.push(
            ToolSpec::new(
                PLAN_TOOL,
                format!(
                    "State what must hold before you finish, as criteria with an id, the words, and the kind of \
                     evidence that shows it (kinds: {kinds_text}). Evidence counts only when gathered after your \
                     last edit. Stating a plan again replaces the old one."
                ),
                serde_json::json!({
                    "type": "object",
                    "properties": {"criteria": {"type": "array", "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "text": {"type": "string"},
                            "kind": {"type": "string"}
                        },
                        "required": ["id", "text", "kind"]
                    }}},
                    "required": ["criteria"]
                }),
            )
            .read_only(),
        );
        self.gate = Some(gate.clone());
        Ok((self, gate))
    }

    /// Runs the `plan` tool: `Ok` is the note the model reads, `Err` why the plan was not taken.
    fn plan(gate: &SharedGate, input: &Value) -> Result<String, String> {
        let plan: PlanInput = serde_json::from_value(input.clone())
            .map_err(|e| format!("the plan needs `criteria` (id, text, kind): {e}"))?;
        let criteria: Vec<Criterion> = plan
            .criteria
            .into_iter()
            .map(|c| Criterion {
                id: c.id.into(),
                text: c.text.into(),
                kind: c.kind.into(),
            })
            .collect();
        let n = criteria.len();
        gate.lock().set_plan(criteria).map_err(|e| e.to_string())?;
        Ok(format!(
            "Plan taken: {n} criteria. Gather the evidence after your last edit, then finish."
        ))
    }

    /// The tools as the driver's config lists them ([`crate::driver::DriverConfig::new`]): a read-only tool reads,
    /// every other one writes (its start is durable and a restart never runs it twice).
    ///
    /// # Errors
    /// A schema that cannot be rendered as JSON.
    pub fn defs(&self) -> Result<Vec<ToolDef>, BadSchema> {
        self.specs
            .iter()
            .map(|s| {
                let schema = RawJson::render(&s.input_schema).map_err(|e| BadSchema {
                    tool: s.name.as_str().into(),
                    detail: e.to_string().into(),
                })?;
                let effect = if s.read_only {
                    Effect::Read
                } else {
                    Effect::Write
                };
                Ok(ToolDef::new(
                    s.name.as_str(),
                    s.description.as_str(),
                    schema,
                    effect,
                ))
            })
            .collect()
    }

    /// The tools' specs, with their evidence kinds.
    pub fn specs(&self) -> &[ToolSpec] {
        &self.specs
    }

    /// The policy.
    pub fn policy(&self) -> &Policy {
        &self.policy
    }

    fn spec(&self, tool: &str) -> Option<&ToolSpec> {
        self.specs.iter().find(|s| s.name == tool)
    }
}

impl<T: Toolbox> Tools for ToolboxTools<T> {
    fn evaluate(&self, tool: &str, input: &RawJson) -> Verdict {
        let Some(spec) = self.spec(tool) else {
            return Verdict::Deny(format!("`{tool}` is not a tool of this session"));
        };
        let Ok(input) = serde_json::from_str::<Value>(input.as_str()) else {
            return Verdict::Deny(format!("the input of `{tool}` is not JSON"));
        };
        // Calls are one at a time, so the toolbox is free here; if it is not, refuse rather than check a rule
        // pattern against no subject (which could let a call through that a pattern denies).
        let Ok(toolbox) = self.toolbox.try_lock() else {
            return Verdict::Deny(format!(
                "`{tool}` cannot be checked while another call runs"
            ));
        };
        let subject = toolbox.subject(tool, &input);
        self.policy.evaluate(spec, subject.as_deref())
    }

    fn run(
        &self,
        call: CallId,
        tool: Box<str>,
        input: RawJson,
        stop: StopSignal,
    ) -> BoxFuture<'_, ToolOutput> {
        let _ = stop;
        Box::pin(async move {
            let refused = |text: String| ToolOutput {
                content: RawJson::string(&text),
                is_error: true,
            };
            let Some(spec) = self.spec(&tool) else {
                return refused(format!("`{tool}` is not a tool of this session"));
            };
            let planning = self.gate.is_some() && tool.as_ref() == PLAN_TOOL;
            let Ok(asked) = serde_json::from_str::<Value>(input.as_str()) else {
                return refused(format!("the input of `{tool}` is not JSON"));
            };
            let input = match self.hooks.run_pre(&call, &tool, &asked).await {
                Ok(v) => v,
                Err(reason) => return refused(reason),
            };
            let mut toolbox = self.toolbox.lock().await;
            if input != asked {
                let subject = toolbox.subject(&tool, &input);
                match self.policy.evaluate(spec, subject.as_deref()) {
                    Verdict::Allow => {}
                    Verdict::Deny(reason) => {
                        return refused(format!("the call as a hook rewrote it: {reason}"));
                    }
                    Verdict::Ask => {
                        return refused(format!(
                            "the call as a hook rewrote it needs permission the host was not asked for: `{tool}`"
                        ));
                    }
                }
            }
            let result = match &self.gate {
                Some(gate) if planning => Self::plan(gate, &input),
                _ => toolbox.call(&tool, &input).await,
            };
            let (text, is_error) = match result {
                Ok(t) => (t, false),
                Err(t) => (t, true),
            };
            drop(toolbox);
            if let (Some(gate), false) = (&self.gate, planning) {
                let mut g = gate.lock();
                if !spec.read_only {
                    g.record_edit();
                }
                if let (Some(kind), false) = (&spec.evidence, is_error) {
                    g.record_evidence_of_kind(kind, &text);
                }
            }
            let text = self
                .hooks
                .run_post(&call, &tool, &input, text, is_error)
                .await;
            ToolOutput {
                content: RawJson::string(&text),
                is_error,
            }
        })
    }
}
