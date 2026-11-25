// SPDX-License-Identifier: Apache-2.0
//! Hooks: code the host runs before a tool call, after it, and when the model tries to stop.

use std::sync::Arc;

use sarathi_protocol::CallId;
use serde_json::Value;

use crate::tool::BoxFuture;

/// What a pre-tool hook sees.
#[derive(Debug)]
pub struct PreTool<'a> {
    /// The call.
    pub call: &'a CallId,
    /// The tool.
    pub tool: &'a str,
    /// The input as the previous hook left it.
    pub input: &'a Value,
}

/// What a pre-tool hook decides.
#[derive(Debug, Clone, PartialEq)]
pub enum PreAction {
    /// Go on.
    Continue,
    /// Do not run the call; the model is told `reason`.
    Block(String),
    /// Run the call with this input instead.
    Rewrite(Value),
}

/// What a post-tool hook sees.
#[derive(Debug)]
pub struct PostTool<'a> {
    /// The call.
    pub call: &'a CallId,
    /// The tool.
    pub tool: &'a str,
    /// The input the tool ran with.
    pub input: &'a Value,
    /// What the tool (or the previous hook) returned.
    pub output: &'a str,
    /// The call failed.
    pub is_error: bool,
}

/// What a post-tool hook decides.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PostAction {
    /// Keep the output.
    Keep,
    /// Show the model this instead (redaction, a note appended).
    Replace(String),
}

/// What a stop hook sees: the model wants to end the run.
#[derive(Debug)]
pub struct StopEvent<'a> {
    /// The model's closing reply.
    pub reply: &'a str,
}

/// What a stop hook decides.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopAction {
    /// Let the run end.
    Allow,
    /// Send the model back to work with this message (a completion check that failed).
    Continue(String),
}

/// A hook. Every method has a do-nothing default, so a hook implements only the events it cares about.
pub trait Hook: Send + Sync {
    /// Before a call, ahead of the permission check.
    fn pre_tool<'a>(&'a self, event: &'a PreTool<'a>) -> BoxFuture<'a, PreAction> {
        let _ = event;
        Box::pin(std::future::ready(PreAction::Continue))
    }

    /// After a call ran (or failed).
    fn post_tool<'a>(&'a self, event: &'a PostTool<'a>) -> BoxFuture<'a, PostAction> {
        let _ = event;
        Box::pin(std::future::ready(PostAction::Keep))
    }

    /// When the model tries to end the run.
    fn stop<'a>(&'a self, event: &'a StopEvent<'a>) -> BoxFuture<'a, StopAction> {
        let _ = event;
        Box::pin(std::future::ready(StopAction::Allow))
    }
}

/// The hooks of a session, run in the order added.
#[derive(Default, Clone)]
pub struct Hooks {
    hooks: Vec<Arc<dyn Hook>>,
}

impl Hooks {
    /// No hooks.
    pub fn new() -> Self {
        Hooks::default()
    }

    /// Adds a hook after the existing ones.
    pub fn add(&mut self, hook: Arc<dyn Hook>) {
        self.hooks.push(hook);
    }

    /// Runs the pre-tool hooks. The first block ends the chain; a rewrite changes the input the next hook sees.
    /// Returns the (possibly rewritten) input, or the block's reason.
    pub async fn run_pre(&self, call: &CallId, tool: &str, input: &Value) -> Result<Value, String> {
        let mut input = input.clone();
        for h in &self.hooks {
            let event = PreTool {
                call,
                tool,
                input: &input,
            };
            match h.pre_tool(&event).await {
                PreAction::Continue => {}
                PreAction::Block(reason) => return Err(reason),
                PreAction::Rewrite(v) => input = v,
            }
        }
        Ok(input)
    }

    /// Runs the post-tool hooks in order; each sees the previous one's output.
    pub async fn run_post(
        &self,
        call: &CallId,
        tool: &str,
        input: &Value,
        output: String,
        is_error: bool,
    ) -> String {
        let mut output = output;
        for h in &self.hooks {
            let event = PostTool {
                call,
                tool,
                input,
                output: &output,
                is_error,
            };
            if let PostAction::Replace(text) = h.post_tool(&event).await {
                output = text;
            }
        }
        output
    }

    /// Runs the stop hooks; the first that sends the model back to work wins.
    pub async fn run_stop(&self, reply: &str) -> StopAction {
        let event = StopEvent { reply };
        for h in &self.hooks {
            if let StopAction::Continue(msg) = h.stop(&event).await {
                return StopAction::Continue(msg);
            }
        }
        StopAction::Allow
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::block_on;
    use serde_json::json;

    struct Rewrite;
    impl Hook for Rewrite {
        fn pre_tool<'a>(&'a self, e: &'a PreTool<'a>) -> BoxFuture<'a, PreAction> {
            let mut v = e.input.clone();
            v["seen"] = json!(true);
            Box::pin(std::future::ready(PreAction::Rewrite(v)))
        }
        fn post_tool<'a>(&'a self, e: &'a PostTool<'a>) -> BoxFuture<'a, PostAction> {
            let t = format!("{}!", e.output);
            Box::pin(std::future::ready(PostAction::Replace(t)))
        }
    }
    struct Blocker;
    impl Hook for Blocker {
        fn pre_tool<'a>(&'a self, e: &'a PreTool<'a>) -> BoxFuture<'a, PreAction> {
            let act = if e.input.get("seen").is_some() {
                PreAction::Block("rewritten input reached the next hook".into())
            } else {
                PreAction::Continue
            };
            Box::pin(std::future::ready(act))
        }
        fn stop<'a>(&'a self, _: &'a StopEvent<'a>) -> BoxFuture<'a, StopAction> {
            Box::pin(std::future::ready(StopAction::Continue("not done".into())))
        }
    }

    #[test]
    fn hooks_chain_in_order() {
        let id: CallId = "c".into();
        let mut hs = Hooks::new();
        assert_eq!(
            block_on(hs.run_pre(&id, "t", &json!({"a":1}))),
            Ok(json!({"a":1}))
        );
        hs.add(Arc::new(Rewrite));
        assert_eq!(
            block_on(hs.run_pre(&id, "t", &json!({}))),
            Ok(json!({"seen":true}))
        );
        assert_eq!(
            block_on(hs.run_post(&id, "t", &json!({}), "x".into(), false)),
            "x!"
        );
        assert_eq!(block_on(hs.run_stop("done")), StopAction::Allow);
        hs.add(Arc::new(Blocker));
        assert_eq!(
            block_on(hs.run_pre(&id, "t", &json!({}))),
            Err("rewritten input reached the next hook".into())
        );
        assert_eq!(
            block_on(hs.run_stop("done")),
            StopAction::Continue("not done".into())
        );
    }
}
