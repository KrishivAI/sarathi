// Derived from Vyasa `crates/ai/src/agent.rs` @ 2403be1 (MIT OR Apache-2.0); see PROVENANCE.md
// SPDX-License-Identifier: Apache-2.0
//! Model profiles, step protocols and the tools the core knows.
//!
//! A host picks a [`ModelProfile`] per model; the core does not detect one. The profile decides how a step is read
//! from an answer ([`StepProtocol`]) and how a tool result is rendered for the model.

use std::fmt::Write as _;

use sarathi_protocol::RawJson;

/// How the model says what it wants to do next.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StepProtocol {
    /// The provider's own tool calls: a step is the assistant turn, text and zero or more `tool_use` blocks, with
    /// the provider's call ids. Tools go in the request's `tools` field.
    Native,
    /// For models without reliable tool calls: a step is one JSON object `{thought, tool, input, reply}` in the
    /// answer text, and `tool: "done"` ends the turn with `reply`. Tools are described in the system prompt
    /// ([`protocol_text`]); a tool result goes back as a user message.
    JsonObject,
}

/// What the core and its driver need to know about a model.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelProfile {
    /// How a step is read.
    pub protocol: StepProtocol,
    /// The provider takes a JSON schema for the answer ([`step_schema`]); read by the driver for
    /// [`StepProtocol::JsonObject`].
    pub structured_output: bool,
    /// The model may ask for several tool calls in one step. The core runs them in order, one at a time, either way;
    /// the driver reads this to set the request's parallel-tool parameter.
    pub parallel_tools: bool,
}

impl ModelProfile {
    /// A profile for a model with native tool calls.
    pub fn native() -> Self {
        ModelProfile {
            protocol: StepProtocol::Native,
            structured_output: false,
            parallel_tools: true,
        }
    }

    /// A profile for the JSON-object fallback; `structured_output` when the provider enforces a schema.
    pub fn json_object(structured_output: bool) -> Self {
        ModelProfile {
            protocol: StepProtocol::JsonObject,
            structured_output,
            parallel_tools: false,
        }
    }
}

/// Whether a tool changes anything outside the session.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Effect {
    /// Reads only: running it again after a crash is safe, and an interrupt drops it at once.
    Read,
    /// Has side effects: its start is made durable before it runs, a crash never re-runs it, and an interrupt
    /// waits a grace period for it to stop.
    Write,
}

impl Effect {
    /// The name in the log (`read`, `write`).
    pub fn as_str(self) -> &'static str {
        match self {
            Effect::Read => "read",
            Effect::Write => "write",
        }
    }
}

/// One tool as the model sees it, and its effect.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolDef {
    /// The name the model calls (`edit`, `look`).
    pub name: Box<str>,
    /// What it does and when to use it.
    pub description: Box<str>,
    /// JSON Schema of its input.
    pub input_schema: RawJson,
    /// Whether it has side effects.
    pub effect: Effect,
}

impl ToolDef {
    /// A tool definition.
    pub fn new(
        name: impl Into<Box<str>>,
        description: impl Into<Box<str>>,
        input_schema: RawJson,
        effect: Effect,
    ) -> Self {
        ToolDef {
            name: name.into(),
            description: description.into(),
            input_schema,
            effect,
        }
    }
}

/// The tool name that ends a turn under [`StepProtocol::JsonObject`].
pub const DONE_TOOL: &str = "done";

/// The schema of one step under [`StepProtocol::JsonObject`], for providers with structured output.
pub fn step_schema() -> RawJson {
    // A constant a test parses: the fallback is never taken.
    RawJson::from_text(STEP_SCHEMA).unwrap_or_else(|_| RawJson::null())
}

const STEP_SCHEMA: &str = r#"{"type":"object","properties":{"thought":{"type":"string"},"tool":{"type":"string"},"input":{"type":"object"},"reply":{"type":"string"}},"required":["thought","tool"],"additionalProperties":false}"#;

/// The protocol text a host appends to its system prompt under [`StepProtocol::JsonObject`], naming every tool.
/// Rendered once per session: the prefix never changes.
pub fn protocol_text(tools: &[ToolDef]) -> String {
    let mut out = String::from(
        "\n\nYou work in steps. Each reply is ONE action as JSON: \
         {\"thought\", \"tool\", \"input\", \"reply\"}.\n\
         - thought: why this action, briefly.\n\
         - tool: one tool name from the list below, or \"done\".\n\
         - input: that tool's input object (omit or {} for \"done\").\n\
         - reply: only with \"done\" — your closing message to the person, \
         plain prose, naming what you changed and anything you chose not \
         to do.\n\
         Each tool's result comes back as a message starting \"observation:\". \
         Finish as soon as the task is honestly done — steps cost real money.\n\nTOOLS\n",
    );
    for t in tools {
        let _ = write!(
            out,
            "- {} — {}\n  input schema: {}\n",
            t.name,
            t.description,
            t.input_schema.as_str()
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_step_schema_is_valid_json_and_requires_thought_and_tool() {
        let schema = step_schema();
        assert_eq!(schema.as_str(), STEP_SCHEMA);
        let v: serde_json::Value = serde_json::from_str(schema.as_str()).unwrap();
        assert_eq!(v["required"], serde_json::json!(["thought", "tool"]));
    }

    #[test]
    fn the_protocol_text_names_every_tool_with_its_schema() {
        let tools = [
            ToolDef::new(
                "look",
                "Shows the page.",
                RawJson::from_text(r#"{"type":"object"}"#).unwrap(),
                Effect::Read,
            ),
            ToolDef::new(
                "edit",
                "Changes a section.",
                RawJson::from_text(r#"{"type":"object","properties":{"id":{"type":"string"}}}"#)
                    .unwrap(),
                Effect::Write,
            ),
        ];
        let text = protocol_text(&tools);
        assert!(text.contains("- look — Shows the page.\n  input schema: {\"type\":\"object\"}\n"));
        assert!(
            text.contains(
                r#"input schema: {"type":"object","properties":{"id":{"type":"string"}}}"#
            )
        );
        assert!(text.contains("or \"done\""));
        assert_eq!(protocol_text(&tools), text, "rendered the same every time");
    }

    #[test]
    fn profiles_and_effects() {
        assert_eq!(ModelProfile::native().protocol, StepProtocol::Native);
        let j = ModelProfile::json_object(true);
        assert_eq!(j.protocol, StepProtocol::JsonObject);
        assert!(j.structured_output && !j.parallel_tools);
        assert_eq!(Effect::Read.as_str(), "read");
        assert_eq!(Effect::Write.as_str(), "write");
    }
}
