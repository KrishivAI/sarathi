# sarathi-core (S3)
The IO-free run loop (`Core`) and its driver (`Driver`): guards, steering, interrupts, resume after a crash, context
management (compaction, the overflow fallback, tool-output spill, through `sarathi-context`); later the completion
gate.

## Layout
- `src/core.rs`: `Core::step(Input) -> Vec<Action>`; the turn's stages (model call, then each tool call in order:
  evaluate, ask, run, stop); guards (step cap, failures in a row, repeated call, spend cap); steering at the tool
  boundary; interrupts (a write tool gets `write_grace`). `src/core/tests.rs`: turns as input/action sequences.
- `src/profile.rs`: `ModelProfile`, `StepProtocol` (`Native`, `JsonObject`), `ToolDef`, `Effect`, the JSON-step
  schema and protocol text.
- `src/step.rs`: reading a JSON step, cutting native text and inputs from the answer body, the wordings the model reads.
- `src/driver.rs`: `Driver` (`new`, `resume`, `run_turn`, `continue_turn`, `handle`), the action loop over one flight
  (a model call or a tool call) and the op channel, the failed-write and dropped-future handling.
  `src/driver/seams.rs`: `ModelCaller` (with `count_tokens`, default none), `Tools`, `PermissionHost`, `StopSignal`,
  `Received`, `CallFailure`, `Spill` (any `SpillStore`, type-erased).
  `src/driver/context.rs`: the context check before a model call (between turns and inside a turn), the compaction
  call, a compaction left open by a restart, the overflow seed, the tool-output spill.
  `src/driver/entries.rs`: records as log entries, the session prefix, a core rebuilt from a log (resume), the
  entries that close a failed turn.
- `src/toolbox.rs`: `ToolboxTools`, the tools crate's `Toolbox` + `Policy` + `Hooks` as the driver's `Tools`, and
  their `ToolDef`s (read-only reads, everything else writes).
- `src/anthropic.rs` (feature `anthropic`, default): `AnthropicCaller` over a provider chain of one model, and the
  request assembler from the log's parts.
- `src/openai.rs` (feature `openai`, default): the OpenAI-style chat caller, with the request spliced from slices
  rendered once.
- `tests/driver.rs` (+ `tests/common/`): the loop tests on both protocols, the native round trip, asks, steers and
  interrupts, failed writes and dropped futures, resume at every append. `tests/kill_and_resume.rs` (feature
  `fault-injection`): a child process aborted after every append, and inside a write, on the JSONL store.
- `tests/context.rs`: compaction between turns (both compactors), a summary in a tool loop, an answer that is no
  summary, the fallback after a refusal, restarts at every append of a compaction (between turns and in a turn),
  spill.
- `tests/long_run.rs`: 60 turns through a small window with each compactor and a model that answers from what it is
  sent: the prefix check per segment, the seed first after a compaction, a restart after every append of every
  compaction, and after a crash inside one (a prefix of its entries written), ending in the same conversation, and a
  refused request recovered without a model call.
- `tests/openai.rs`: the OpenAI-style caller through the driver.
- `tests/toolbox.rs`: a toolbox through the driver: asks, a deny rule on the subject, plan mode, hooks, `Err` as an
  observation.
- `tests/tool_uses_answered.rs`: property over random runs (every tool call answered before a request or a turn end,
  one result per call, durable start before a write tool, dense seqs).

## Rules
- The core numbers entries: each `Append` names its first seq and `CallModel` the request's seq; the driver appends
  exactly that; when an append or the request cannot be written it sends `Input::Aborted` with the store's next
  seq (recoverable only at rest or before the request is written; otherwise `Core::needs_rebuild()` and the
  driver makes a new core from the log). Model answers and failures name the request's seq, so a cancelled call's
  late answer is refused.
- Every tool call of an accepted answer gets one `ToolResult` before the next `CallModel` and before `EndTurn`.
- Answer + `AssistantMessage` are one durable batch; a write tool's `ToolStart` and `ToolResult` are durable.
- `Verdict` is the tools crate's. A host's toolbox, policy and hooks reach the driver through `toolbox::ToolboxTools`
  (the `Tools` seam): the verdict is the policy on the call's subject, IO-free; an ask goes out as an event and is
  answered by an op; `run` does pre-tool hooks, the tool, post-tool hooks. A hook's rewrite goes through the policy
  again and runs only on an allow. "Allow always" is the core's (from the log), not a session rule.
- The driver writes nothing the core did not ask for, except: the session prefix (`new`), a `turn.end` for a turn
  whose last model call failed before its end was written (resume), the closing results and `turn.end` of a
  turn that failed on a write, and context management's entries (a compaction call, its answer or failure, a seed).
- Context entries go in only at a boundary: between turns (before the user message, so a server compaction block is
  followed by it) or at a `CallModel` before its request is written; then `Core::reseat` makes the call again after
  them (between turns the core is rebuilt from the log). A compaction call and its failure never end or count in a
  turn (`entries::start` passes over them). Server compaction needs `anthropic-messages` (checked at `new`/`resume`);
  the Anthropic caller adds `COMPACTION_BETA` itself when a request asks for compaction or sends a block back.
- A spilled output is in the spill store (durable) before the `tool.result` that names it is written.
- `fault-injection` is for tests only; never on in a host's build.
- A request that cannot fit even after the fallback is never sent: between turns `DriverError::DoesNotFit` (op not
  run), in a turn an abort with kind `guard`. An interrupt during the check before a turn ends it before it starts
  (`DriverError::Interrupted`); in a turn, one during a compaction call is read when the call returns.
- Only string tool output spills; a result given as blocks stays inline.

- A compaction call is reported as `Event::Usage` and its cost is charged to the turn (`Core::charge`); the driver
  refuses an `Op::ToolOutput` for the call its toolbox runs.

Status: core, driver, both callers, the toolbox adapter (`ToolboxTools`) and the context hooks written. Not yet: the
stop hook is not yet called by the completion gate.
