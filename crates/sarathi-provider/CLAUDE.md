# sarathi-provider (S1)
Wire clients and translation (Anthropic, OpenAI-compatible), SSE, retries, failover, circuit breaker, budget, the
Anthropic-native `messages()` call, provider profiles. Derived in part from the Ultrafast Gateway and Vyasa
(`PROVENANCE.md`). Test that must always pass: the round-trip fixture test (`tests/native.rs`, `tests/native_prop.rs`).

Layout: `src/wire/` is the no-IO half (build request bytes, parse answers and SSE; `wire::provider` holds the
translated per-format code, `wire::native` the Anthropic-native call, `wire::openai_chat` the spliced OpenAI chat
call), `src/http/` the reqwest client, `src/resilience/` retries, the circuit breaker, failover `Chain` and the budget
check. Keep IO out of `wire`. Copied files keep their `Derived from …` header line (`PROVENANCE.md`).

Rules: `ErrorKind` is `sarathi_protocol::ErrorKind` (non-exhaustive: matches need a wildcard arm). A stream that ends
before its end event is `ErrorKind::Network` (retryable) on both paths. Native path: never parse history into a
`Value` and write it again; the turn is kept as the bytes received. Fixtures under `tests/fixtures/anthropic-native/`
are synthetic (`case.toml` says so) until recorded answers replace them; tests never call a live model.
