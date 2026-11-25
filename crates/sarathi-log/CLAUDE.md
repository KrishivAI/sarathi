# sarathi-log (S1)
The append-only session log: entries and the line codec (`entry.rs`, `line.rs`), the in-memory `Log` with its
invariants, request projection, recipe check and resume (`log.rs`), the `LogStore` seam, `MemoryStore` and `fork`
(`store.rs`), and the JSONL file store (`jsonl.rs`, feature `jsonl`, default on).

Rules:
- Nothing is edited or deleted. Compaction, a protocol switch and a fork start a new segment (`segment.start`).
- A stored slice keeps its bytes: written raw, or as a string listed in the line's `esc` when it holds a line break.
- `segment.seed` (written with `must_understand`) holds a slice made for a compaction seed: a note, or an earlier
  turn re-rendered without thinking. It is sent only through a `segment.start` that names it, and resume ignores it,
  so a seed batch cut short leaves the old segment as it was.
- Format changes are recorded in `docs/design/log-format.md`.
- `model.request` stores a recipe and the SHA-256 of the body sent. The store does not rebuild it on append (the
  assembler lives in the provider crate): hosts run `Log::verify_requests`.
- `turn.end` needs every call of the turn's last assistant message answered, and closes a pending model call.
- `JsonlStore::load`/`append` lock and repair the session until `release`; `JsonlStore::read` is read-only.
- A crash inside an append can keep a whole-line prefix of the batch: order batches so every prefix is resumable.
- Crash rule support: `resume` reports each open call as not started, started read (run again) or started write
  (never run again; answer with `CUT_SHORT`).
- Format version in the header; a reader of N reads N and N−1. Changing the meaning of a kind or the line framing
  bumps the version and adds `fixtures/log/v<N>/`; old fixture directories stay and stay green. Regenerate only the
  current version's fixtures (`SARATHI_LOG_WRITE_FIXTURES=1`).

Tests: `tests/log/` (one binary, needs `jsonl`): store conformance on both stores, torn tail at every byte, lock,
crash matrix (the test binary re-runs itself as an aborting child), committed fixtures.
