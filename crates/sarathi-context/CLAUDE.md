# sarathi-context (S2)
Context management over a session log: the deterministic overflow fallback (`fallback.rs`), and the tool-output
spill store with its bounded preview and ranged reads (`spill.rs`; `FsSpill` behind feature `fs`, default on;
`MemorySpill`). No provider and no model calls: the run loop asks this crate what to append and makes any model call
itself.

Rules:
- Nothing in the log is changed. A compaction is a new segment whose seed is appended in one batch with its
  `segment.start`, so a crash leaves either the old segment or the new one.
- The deterministic fallback never calls a model and runs only between requests (no open call, no model call in
  flight). It keeps user and tool-result slices by seq reference and the current turn as it is, thinking included
  (a tool loop needs its thinking back unmodified). Earlier kept turns' assistant slices are re-rendered once
  without thinking, as `segment.seed` entries (each other block's bytes unchanged). The cut never splits a tool
  round: kept turns start at a user message. A segment's head (e.g. a server compaction block) is kept first, and
  dropped only when nothing with it fits.
- Spilled output is durable before the `tool.result` that names it; the result carries the output's SHA-256.
- Spill files are named by the hex of the call id (any non-empty id up to 120 bytes), never by the id itself.
- A fork copies spill files before the child log is written.

Tests: unit tests per module; `tests/context.rs` runs the fallback over scripted logs and the spill stores.
