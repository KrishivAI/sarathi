// SPDX-License-Identifier: Apache-2.0
//! Tools, permissions and hooks (layer S2).
//!
//! Three of the four host seams live here: the tools a host offers ([`Toolbox`]), the permission callback
//! ([`PermissionHost`]) and the hooks that run around a call ([`Hook`]). A call goes through [`ToolRunner`]:
//! pre-tool hooks, then the permission check (rules, mode, host), then the tool, then post-tool hooks. A tool's own
//! failure is an observation the model reads, never an error of the run.
//!
//! Rules are `Tool(pattern)` with an effect (allow, ask, deny); deny always wins, then ask, then allow; a call no rule
//! matches falls to the mode ([`Mode`]).

mod gate;
mod hook;
mod policy;
mod rule;
mod runner;
mod tool;

pub use gate::{Access, AskRequest, Gate, PermissionHost};
pub use hook::{Hook, Hooks, PostAction, PostTool, PreAction, PreTool, StopAction, StopEvent};
pub use policy::{Mode, Policy, Scope, Verdict};
pub use rule::{Effect, Rule, RuleError};
pub use runner::{Disposition, Executed, Prepared, Ready, ToolRunner};
pub use tool::{BoxFuture, Tool, ToolError, ToolResult, ToolSet, ToolSpec, Toolbox};

#[cfg(test)]
mod testutil;
