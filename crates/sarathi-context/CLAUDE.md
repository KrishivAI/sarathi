# sarathi-context (S2)
Context management over a session log: the size estimate and the decision before each request (`plan.rs`:
`estimate`, `ContextProfile`, `before_request`, `overflow_seed`), compaction by a model call (`compact.rs`: server
block or client summary; `compaction_request`, `compaction_seed`, `pending_compaction`), the deterministic overflow
fallback (`fallback.rs`), and the tool-output spill store with its bounded preview and ranged reads (`spill.rs`;
`FsSpill` behind feature `fs`, default on; `MemorySpill`). No provider and no model calls: the run loop asks this
crate what to append and makes any model call itself.

Rules:
- Nothing in the log is changed. A compaction is a new segment whose seed is appended in one batch with its
  `segment.start`, so a crash leaves either the old segment or the new one.
- The deterministic fallback never calls a model and runs only between requests (no open call, no model call in
  flight). It keeps user and tool-result slices by seq reference and the current turn as it is, thinking included
  (a tool loop needs its thinking back unmodified). Earlier kept turns' assistant slices are re-rendered once
  without thinking, as `segment.seed` entries (each other block's bytes unchanged). The cut never splits a tool
  round: kept turns start at a user message. A segment's head (e.g. a server compaction block) is kept first, and
  dropped only when nothing with it fits.
- A compaction call is a `model.request` marked `compaction`; its answer only ever becomes a seed (the log refuses
  a `message.assistant` from it). On resume the run loop checks `pending_compaction` before anything else: with a
  logged response it calls `compaction_seed` instead of calling again.
- A server block is kept byte for byte and is the segment's head; while the projection carries one
  (`carries_compaction_block`) every request sends the beta flag. Later turn params never carry `compaction`.
- Server compaction runs only between turns (its block needs the next user message after it); a client summary may
  also run inside a turn. A long single-turn tool loop with server compaction can only overflow.
- Compaction is asked for at most once per gap: a segment needs an answer of its own after its last compaction call.
  After a provider overflow, the fallback must at least halve the estimate, so repeated refusals end in
  `DoesNotFit` instead of a loop.
- Spilled output is durable before the `tool.result` that names it; the result carries the output's SHA-256.
- Spill files are named by the hex of the call id (any non-empty id up to 120 bytes), never by the id itself.
- A fork copies spill files before the child log is written.

Tests: unit tests per module; `tests/context.rs` runs the fallback, both compactors (crash at every entry), the
decision and the estimate over scripted logs, and the spill stores.
