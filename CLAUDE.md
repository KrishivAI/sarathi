# CLAUDE.md — sarathi

Sarathi: an embeddable agent harness in Rust (run loop, context and compaction, tools and permissions, append-only
session log, completion checks, evals). A harness, not a workflow framework, not a model gateway. Apache-2.0, DCO.
Design notes per crate go in `docs/design/`.

## Read first
1. This file, then `crates/<crate>/CLAUDE.md` for each crate you touch.
2. `README.md` (scope and layers) and the design rules below.
3. `git status`, `git log --oneline -10`, `gh pr list` before changing anything.

## Commands
```text
cargo build --workspace                                               # toolchain pinned: rust-toolchain.toml (1.99.1); MSRV 1.94
cargo nextest run --workspace --all-features --locked --no-tests=pass # CI form
cargo test --workspace --doc --locked
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo deny check                       # licences, OpenSSL/anyhow/rig-core bans, advisories
cargo machete                          # unused deps
cargo hack check --each-feature --workspace --locked   # features stay additive
python3 scripts/layering.py            # S0..S5 layer rule (scripts/layers.toml)
scripts/spdx.sh; scripts/file-size.sh  # SPDX headers; no .rs file over 2,000 lines
cargo +1.94 check --workspace --locked # MSRV (CI job `msrv`)
```
**Every commit needs a DCO sign-off: `git commit -s`.** Source files start with `SPDX-License-Identifier: Apache-2.0`.
Lint exceptions are `#[expect(lint, reason = "...")]`, never a bare `#[allow]`. Never push to `main`; a PR per task.
`git config core.hooksPath .githooks` once per clone enables the commit-msg hook (AI-attribution check, DCO sign-off).

## Layering (enforced by `scripts/layering.py`)
A crate depends only on crates in the same or a lower layer.
- S0 protocol · S1 provider, log · S2 tools, context · S3 core, mcp, skills · S4 eval · S5 `sarathi` facade
A new crate: add it to `scripts/layers.toml` and the workspace, and write its `CLAUDE.md`.

## Design rules (each is a review line)
1. **Immutable prefix**: the log stores bytes sent and received per provider; `system` and `tools` fixed per session;
   compaction is a new segment with a recorded summary, never an edit.
2. **IO-free core**: a state machine returns `CallModel`, `CallTools` or `Done`; a driver does the IO.
3. **Crash rule**: a side-effecting tool starts only after the model reply that asked for it and a start mark are on
   disk; after a crash a started call is answered "cut short by the restart" and is not re-run.
4. **Versioned on-disk log**, N reads N−1.
5. **Capability flags per provider profile**, used where present and never required.
Four seams carry every host difference: tools, permission callback, store, event sink.

## Hard rules
- `unsafe` is forbidden in every crate except a future Python-bindings crate, each block with `// SAFETY:`.
- No `unwrap`/`expect`/`panic` in library code; `thiserror` errors; no `anyhow`; no OpenSSL (`rustls` only); no `rig-core`.
- Cargo features are additive only. All dependencies in `[workspace.dependencies]`, `default-features = false`;
  `rmcp` and every HTTP crate pinned exactly; a bump is its own PR.
- Public surface and on-disk formats change only deliberately (a design note in `docs/design/`).
- **No live model spend** in tests or CI: tests use the scripted provider or recorded cassettes. A live call is a
  named, explicitly approved step, never a default.
- Copied code: `PROVENANCE.md` + `NOTICE`.
- No bot or assistant trailers in commits or PR text (`.github/scripts/check-ai-attribution.sh` fails the PR).
  Commit identity: a human author.
- Never commit secrets. Never push to `main`; no force pushes; no crates.io publish.

## Process
One task = one branch = one small PR (≤ ~2,000 counted lines). Keep changes to one crate where you can. A reviewer
checks each PR against the design rules above.
