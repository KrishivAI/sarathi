# Sarathi

An embeddable agent harness in Rust: the run loop, context and compaction, tools and permissions, an append-only
session log, completion checks and evals. Sarathi runs agents safely with **your** tools inside **your** application.
It is not a workflow framework and not a model gateway.

> **Status: early and pre-release.** The workspace builds and the crate boundaries exist, but most crates are stubs and
> the API is unstable. Nothing is published to crates.io yet.

## Design principles

- **Small core.** An IO-free state machine returns `CallModel`, `CallTools` or `Done`; a driver does the IO.
- **Four host seams.** Everything that differs between hosts goes through tools, a permission callback, a store or an
  event sink.
- **Append-only sessions.** The log keeps the exact bytes sent and received per provider. The system prompt and tool
  list are fixed for a session; compaction adds a new segment with a recorded summary and never edits history.
- **Crash-safe tools.** A side-effecting tool starts only after the model reply that asked for it is on disk; after a
  crash a started call is reported as cut short and is not re-run.
- **Completion checks.** A run may finish only when its stated criteria have evidence gathered after the last edit.
- **Evals built in.** A scripted provider and recorded cassettes make runs reproducible offline. Tests never spend
  money on a live model.
- **Versioned on-disk formats.** A release reads the format of the one before it.

## Crates

| Layer | Crate | Purpose |
|---|---|---|
| S0 | `sarathi-protocol` | ops, events, log entry schema, content blocks |
| S1 | `sarathi-provider`, `sarathi-log` | model wire clients; the append-only log and stores |
| S2 | `sarathi-tools`, `sarathi-context` | tool trait, permission rules, hooks; overflow fallback, compaction seeds, output spill |
| S3 | `sarathi-core`, `sarathi-mcp`, `sarathi-skills` | the run loop; MCP client; skill and agent definitions |
| S4 | `sarathi-eval` | scripted provider, cassettes, brief runner, metrics |
| S5 | `sarathi` | the facade crate |

A crate depends only on crates in its own or a lower layer (`scripts/layering.py` checks it).

## Quick start

```text
git clone https://github.com/KrishivAI/sarathi
cd sarathi
cargo build --workspace
cargo nextest run --workspace --all-features --locked --no-tests=pass
```

The toolchain is pinned in `rust-toolchain.toml`; the minimum supported Rust version is 1.94.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Commits are signed off under the Developer Certificate of Origin.

## Licence

Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). Parts of the provider layer are derived from MIT-licensed
code; see [PROVENANCE.md](PROVENANCE.md).
