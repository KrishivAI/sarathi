// SPDX-License-Identifier: Apache-2.0
//! The tool seam: a [`Toolbox`] offers tools and runs them; a [`Tool`] is one tool, and a [`ToolSet`] makes a toolbox
//! of tools.

use std::future::Future;
use std::pin::Pin;

use serde_json::Value;

/// A boxed, sendable future, the shape every async method here returns (no macro, no extra dependency).
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// One tool as the model sees it, plus what the permission layer needs to know about it.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolSpec {
    /// The name the model calls.
    pub name: String,
    /// What it does and when to reach for it.
    pub description: String,
    /// JSON Schema of the input.
    pub input_schema: Value,
    /// The tool changes nothing. Read-only calls need no ask in the default mode. (Calls run one at a time; the flag is for the mode.)
    pub read_only: bool,
    /// The tool edits the workspace (allowed without an ask in [`Mode::AcceptEdits`](crate::Mode::AcceptEdits)).
    pub edits: bool,
    /// What the output is evidence of for a completion check (for example `look`), if anything.
    pub evidence: Option<String>,
}

impl ToolSpec {
    /// A tool that is neither read-only nor an edit, with no evidence kind.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
    ) -> Self {
        ToolSpec {
            name: name.into(),
            description: description.into(),
            input_schema,
            read_only: false,
            edits: false,
            evidence: None,
        }
    }

    /// Marks the tool read-only.
    #[must_use]
    pub fn read_only(mut self) -> Self {
        self.read_only = true;
        self
    }

    /// Marks the tool as editing the workspace.
    #[must_use]
    pub fn edits(mut self) -> Self {
        self.edits = true;
        self
    }

    /// Tags the tool's output as evidence of `kind` for a completion check.
    #[must_use]
    pub fn evidence(mut self, kind: impl Into<String>) -> Self {
        self.evidence = Some(kind.into());
        self
    }
}

/// The result of a tool call: text for the model, and whether it is a failure the model should see as one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolResult {
    /// What the model reads.
    pub content: String,
    /// The call failed (the model recovers; the run goes on).
    pub is_error: bool,
}

impl From<Result<String, String>> for ToolResult {
    fn from(r: Result<String, String>) -> Self {
        match r {
            Ok(content) => ToolResult {
                content,
                is_error: false,
            },
            Err(content) => ToolResult {
                content,
                is_error: true,
            },
        }
    }
}

/// A surface's tools plus the state they act on.
///
/// One object rather than one per tool, because the tools of a surface share mutable state (a draft, accumulated
/// warnings) and one owner needs no locks. The shape is the usual host-side toolbox on purpose: `Err` from [`call`](Self::call)
/// is an *observation* the model reads and recovers from, not a failure of the run.
pub trait Toolbox: Send {
    /// The tools on offer. Called once per session: the list is part of the session's fixed prefix.
    fn tools(&self) -> Vec<ToolSpec>;

    /// Runs one tool. `Err` is an observation, not a run failure.
    fn call<'a>(&'a mut self, name: &str, input: &Value) -> BoxFuture<'a, Result<String, String>>;

    /// What a rule pattern is matched against for this call: the command, path or URL. `None` when the call has no
    /// such thing (then only a pattern-less rule matches it).
    fn subject(&self, name: &str, input: &Value) -> Option<String> {
        let _ = (name, input);
        None
    }
}

/// A tool that needs no state beyond itself.
pub trait Tool: Send + Sync {
    /// The tool as the model sees it.
    fn spec(&self) -> ToolSpec;

    /// What a rule pattern is matched against (see [`Toolbox::subject`]).
    fn subject(&self, input: &Value) -> Option<String> {
        let _ = input;
        None
    }

    /// Runs the tool. `Err` is an observation.
    fn call<'a>(&'a self, input: &'a Value) -> BoxFuture<'a, Result<String, String>>;
}

/// Two tools share a name.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ToolError {
    /// A name is used twice.
    #[error("two tools are named `{0}`")]
    Duplicate(Box<str>),
}

/// A [`Toolbox`] made of independent [`Tool`]s.
#[derive(Default)]
pub struct ToolSet {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolSet {
    /// An empty set.
    pub fn new() -> Self {
        ToolSet::default()
    }

    /// Adds a tool.
    ///
    /// # Errors
    /// [`ToolError::Duplicate`] when a tool of that name is already in the set.
    pub fn add(&mut self, tool: impl Tool + 'static) -> Result<&mut Self, ToolError> {
        let name = tool.spec().name;
        if self.find(&name).is_some() {
            return Err(ToolError::Duplicate(name.into()));
        }
        self.tools.push(Box::new(tool));
        Ok(self)
    }

    fn find(&self, name: &str) -> Option<&dyn Tool> {
        self.tools
            .iter()
            .find(|t| t.spec().name == name)
            .map(AsRef::as_ref)
    }
}

impl Toolbox for ToolSet {
    fn tools(&self) -> Vec<ToolSpec> {
        self.tools.iter().map(|t| t.spec()).collect()
    }

    fn call<'a>(&'a mut self, name: &str, input: &Value) -> BoxFuture<'a, Result<String, String>> {
        let Some(tool) = self.find(name) else {
            let msg = format!("unknown tool `{name}`");
            return Box::pin(std::future::ready(Err(msg)));
        };
        let input = input.clone();
        Box::pin(async move { tool.call(&input).await })
    }

    fn subject(&self, name: &str, input: &Value) -> Option<String> {
        self.find(name)?.subject(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::block_on;
    use serde_json::json;

    struct Echo;
    impl Tool for Echo {
        fn spec(&self) -> ToolSpec {
            ToolSpec::new("echo", "repeats", json!({"type":"object"})).read_only()
        }
        fn subject(&self, input: &Value) -> Option<String> {
            input.get("text")?.as_str().map(str::to_owned)
        }
        fn call<'a>(&'a self, input: &'a Value) -> BoxFuture<'a, Result<String, String>> {
            Box::pin(async move {
                input
                    .get("text")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .ok_or_else(|| "give `text`".to_owned())
            })
        }
    }

    #[test]
    fn a_set_dispatches_and_reports_unknown_tools_as_observations() {
        let mut set = ToolSet::new();
        set.add(Echo).unwrap();
        assert!(matches!(set.add(Echo), Err(ToolError::Duplicate(n)) if &*n == "echo"));
        assert_eq!(set.tools().len(), 1);
        assert_eq!(
            block_on(set.call("echo", &json!({"text":"hi"}))),
            Ok("hi".into())
        );
        assert_eq!(
            block_on(set.call("echo", &json!({}))),
            Err("give `text`".into())
        );
        assert_eq!(
            block_on(set.call("nope", &json!({}))),
            Err("unknown tool `nope`".into())
        );
        assert_eq!(
            set.subject("echo", &json!({"text":"x"})).as_deref(),
            Some("x")
        );
        assert_eq!(set.subject("nope", &json!({})), None);
    }

    #[test]
    fn spec_builders_and_result_conversion() {
        let s = ToolSpec::new("look", "d", json!({}))
            .read_only()
            .evidence("look");
        assert!(s.read_only && !s.edits);
        assert_eq!(s.evidence.as_deref(), Some("look"));
        assert!(ToolSpec::new("e", "d", json!({})).edits().edits);
        assert!(!ToolResult::from(Ok("x".to_owned())).is_error);
        assert!(ToolResult::from(Err("x".to_owned())).is_error);
    }
}
