# Provenance

Where copied code came from. One entry per copy; a copied file carries a header line naming this file.

## Ultrafast Gateway

The provider layer (`sarathi-provider`) is derived from the MIT-licensed Ultrafast Gateway
(<https://github.com/techgopal/ultrafast-ai-gateway>, crates `ufaigw-translate` and `ufaigw-client`).

**Status: copied** into `crates/sarathi-provider`.

| Field | Value |
|---|---|
| Source | `github.com/techgopal/ultrafast-ai-gateway` |
| Source licence | MIT |
| Source commit | `c9c3eafa5c03ae4ac503b4b544a48774bbffe500` (tag `v3.1.0`) |
| Taken | provider clients, request and response translation, SSE streaming, tool-call mapping, timeouts and error classification (the copied client does not retry; retries, failover and the circuit breaker are `src/resilience/`, below) |
| Left | console, storage, admin API, routing tables, gateway-side budgets |
| Renames | crate `ufaigw-translate` → module `sarathi_provider::wire`, `ufaigw-client` → module `sarathi_provider::http` (one crate); no type carried a gateway prefix, so no type was renamed; the `x-uf-tags` header keeps its name (it is the gateway's wire protocol) |
| Licence here | Apache-2.0; the MIT notice is kept in `NOTICE` for any file not rewritten |

### Files

| Copied to (`crates/sarathi-provider/`) | From (`crates/` at c9c3eaf) | Changes |
|---|---|---|
| `src/wire/{classify,error,sse,tags,types}.rs` | `translate/src/` | module paths; docs added; doc text naming removed providers or the gateway's other clients rewritten; `ErrorKind` is `sarathi_protocol::ErrorKind` (re-exported), and `ErrorKind::of_status` became `classify::kind_of_status` |
| `src/wire/provider/mod.rs` | `translate/src/provider/mod.rs` | Gemini, Azure and Responses arms removed (`api_version`, `path_segment`, the Gemini stream state and `StreamDecoder::finish`, which only Gemini used); docs |
| `src/wire/provider/{anthropic,openai}.rs` | `translate/src/provider/` | the two ingress round-trip tests removed (ingress is the gateway's server side; the behaviour is covered by direct tests); a key-order assertion compares values (no `preserve_order`) and two tests pin the sorted-key effect; `openai::body` takes `model: &str` (the `None` case was Azure's); one index replaced by `get` |
| `src/http/{client,error,request,stream,target}.rs` | `client/src/` | `embed` removed; Gemini and Azure targets removed; `Client::new` returns `Result` (a non-retryable `Network` error) instead of panicking; the stream no longer calls the removed `finish`; capability flags per target |
| `tests/{chat,errors,stream,fixtures}.rs`, `tests/common/mod.rs` | `client/tests/` | Gemini, Azure and embeddings cases removed; the redirect test counts 2 requests (no embed call); the embed size-cap half of a test removed (chat has its own); the key-scrub test's Gemini and embed cases become Anthropic `chat` and `chat_stream`; `Client::new(..).unwrap()`; lint attributes; helpers `pub(crate)` |
| `tests/fixtures/{anthropic,openai}/*` | `translate/tests/fixtures/` | none |
| `tests/fixtures/parity/*.json` | `clients/fixtures/` | Gemini, Azure and embeddings cases removed (re-serialised); a stream that ends before its end event is `network`, retryable (was `malformed`), in `streams.json` |

Not copied: `translate/src/{audio,images,embeddings}.rs`, `translate/src/ingress/`, `translate/src/provider/{gemini,azure,responses}.rs`,
`translate/tests/fixtures/{gemini,azure}/`, `client-py`, `client-wasm`, `gateway`, `client/tests/{embed,parity}.rs`.
The two crates' `lib.rs` files were replaced by `src/lib.rs`, `src/wire/mod.rs` and `src/http/mod.rs` (the client
crate's doc moved into `lib.rs`). Written here, not copied: the Anthropic-native call (`src/wire/native/`,
`src/http/native.rs`), the spliced OpenAI chat call (`src/{wire,http}/openai_chat.rs`), `src/resilience/{mod,retry}.rs`
and `tests/{native,native_prop,openai_chat,resilience}.rs` with `tests/fixtures/anthropic-native/`.

Whole-copy changes: edition 2021 → 2024, so rustfmt and clippy rewrote some code mechanically (nested `if let`
collapsed into let-chains in `provider/{mod,anthropic,openai}.rs`, import order); behaviour is the same. Workspace
change: `serde_json` without `preserve_order` (feature unification would turn it on for every embedding host); the
translated path's keys therefore come back sorted, which two tests pin.

After the copy the two code bases diverge on purpose: no shared crate, no obligation to port fixes either way. Each
copied file keeps a header line `// Derived from Ultrafast Gateway @ <short hash> (MIT); see PROVENANCE.md` above its
`SPDX-License-Identifier: Apache-2.0` line.

## Vyasa

Three files of `sarathi-provider::resilience` and three of `sarathi-core` (the run loop) are rewritten from Vyasa
(<https://github.com/vyasa-cms/vyasa>).

| Field | Value |
|---|---|
| Source | `github.com/vyasa-cms/vyasa`, `crates/ai/src/{circuit,failover,budget}.rs` |
| Source licence | MIT OR Apache-2.0 (used under Apache-2.0) |
| Source commit | `2403be1a61f4e3147019671c49b95c02afe91a99` |

| Written to (`crates/sarathi-provider/src/resilience/`) | From | Changes |
|---|---|---|
| `circuit.rs` | `circuit.rs` | `tokio::time::Instant` (tests use a paused clock); a failed probe opens the breaker for a new cooldown (the original kept the first opening time, so once cooled it never refused again); `record` and `trips` decide which errors count (outages: `upstream`, `network`, `timeout`); a threshold of 0 is 1 |
| `failover.rs` | `failover.rs` | against Sarathi's `Error` and `Client`, not `LlmProvider`; a slot is a client and a model; retries per slot (`retry.rs`, new); no failover on `invalid_request`, `history_mismatch`, `budget_exceeded`; the result names the slot that answered and every failed attempt (the original logged them with `tracing`); the most actionable error is ranked by `ErrorKind` |
| `budget.rs` | `budget.rs` | whole micro-dollars (`u64`) instead of `f64` dollars; the window is the caller's (the original fixed the UTC month); the error is `ErrorKind::BudgetExceeded` |

Each file keeps a header line naming its source above its SPDX line.

### The run loop (`sarathi-core`), from `crates/ai/src/agent.rs` @ `2403be1`

Rewritten as a state machine with no IO (Vyasa's loop is one async function).

| Written to (`crates/sarathi-core/src/`) | From | Changes |
|---|---|---|
| `step.rs` | `agent.rs`: `StepReply`, the correction, unknown-tool, refusal and ran-out wordings | the step is read with its `input` bytes kept as written; a fenced step is read; the unknown-tool text has a form without `"done"` for native tool calls |
| `profile.rs` | `agent.rs`: `step_schema`, `protocol` | the schema is a constant; the protocol text says results come back as `observation:` messages and drops the page-specific sentence (the host's system prompt carries it) |
| `core.rs` | `agent.rs`: `run_agent`, `MAX_STEPS_CEILING`, `MAX_CONSECUTIVE_FAILURES` | a multi-turn conversation instead of a transcript rebuilt every step (the prefix stays byte-stable); no observation truncation or collapsing (context size is handled elsewhere); native tool calls, permissions, steering, interrupts, a spend cap and a repeated-call guard are new |

## Rules

- Code copied from any other project adds a row here and an entry in `NOTICE`.
- Design ideas borrowed from other projects are credited in documentation, not copied as code.
