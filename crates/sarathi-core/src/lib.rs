// SPDX-License-Identifier: Apache-2.0
//! The run loop as an IO-free state machine (layer S3).
//!
//! [`Core`] decides everything a turn does: what to append to the session log, which model call to make, which tool
//! call to evaluate and run, when a guard or an interrupt ends the turn. A driver carries out its [`Action`]s and feeds
//! back each result as an [`Input`]. Two step protocols are supported ([`StepProtocol`]): the provider's native tool
//! calls, and one JSON object per step for models without them.
//!
//! [`driver::Driver`] carries the actions out against the session log, a model ([`driver::ModelCaller`]; the
//! Anthropic Messages API is [`anthropic::AnthropicCaller`], feature `anthropic`; OpenAI-style chat completions
//! [`openai::OpenAiCaller`], feature `openai`) and the tools ([`toolbox::ToolboxTools`] serves the tools crate's
//! toolbox, policy and hooks).

#[cfg(feature = "anthropic")]
pub mod anthropic;
mod core;
pub mod driver;
pub mod gate;
#[cfg(feature = "openai")]
pub mod openai;
mod profile;
mod step;
pub mod toolbox;

pub use crate::core::{
    Action, Answer, AskedCall, CUT_SHORT, Completion, Core, CoreConfig, Durability, Input, Limits,
    MAX_COMPLETION_REFUSALS, MAX_STEPS_CEILING, OpenTurn, Phase, Recipe, Record, ResumeAt, Running,
    TurnEnd, Verdict,
};
pub use crate::profile::{
    DONE_TOOL, Effect, ModelProfile, StepProtocol, ToolDef, protocol_text, step_schema,
};

/// JSON the crate put together from parts that are JSON themselves, so it parses; `null` if it ever did not, which a
/// debug build reports.
#[cfg(any(feature = "anthropic", feature = "openai"))]
pub(crate) fn built_json(text: String) -> sarathi_protocol::RawJson {
    let parsed = sarathi_protocol::RawJson::from_string(text);
    debug_assert!(parsed.is_ok(), "built JSON that does not parse: {parsed:?}");
    parsed.unwrap_or_else(|_| sarathi_protocol::RawJson::null())
}
