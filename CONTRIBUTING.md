# Contributing to Sarathi

Thank you for your interest. Sarathi is an embeddable agent harness in Rust; see [README.md](README.md) for scope.

- **DCO:** sign off every commit (`git commit -s`). CI fails a commit without a `Signed-off-by` line.
- **Attribution:** commits and PR text must not carry bot or assistant trailers
  (`.github/scripts/check-ai-attribution.sh` fails the PR).
- **SPDX:** every `.rs`, `.sh` and `.py` file starts with `SPDX-License-Identifier: Apache-2.0` (within the first 5 lines).
- **One change, one branch, one PR**, kept small (at most about 2,000 counted lines). Never push to `main`; open a PR.
- **No live model calls** in tests or CI: use the scripted provider or recorded cassettes.
- **Commands** (all must pass):

```text
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo nextest run --workspace --all-features --locked --no-tests=pass
cargo test --workspace --doc --locked
cargo deny check && cargo machete && cargo hack check --each-feature --workspace --locked
python3 scripts/layering.py && scripts/spdx.sh && scripts/file-size.sh
```

Enable the hooks once per clone: `git config core.hooksPath .githooks`.
