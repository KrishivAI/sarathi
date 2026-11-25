## Summary

<!-- What and why. Link the issue if there is one. -->

## Tests run

```text

```

## Checklist

- [ ] Commits are signed off (`git commit -s`).
- [ ] No AI attribution in commits or this text.
- [ ] Source files carry `SPDX-License-Identifier: Apache-2.0`.
- [ ] Lint exceptions use `#[expect(lint, reason = "...")]`, never `#[allow]`; no `unsafe`.
- [ ] No `unwrap`/`expect`/`panic` in library code; errors use `thiserror`.
- [ ] On-disk formats carry a version; N reads N-1.
- [ ] New dependency: licence on the `deny.toml` list, maintained, no OpenSSL, exact pin for HTTP and MCP crates.
- [ ] Copied code: `PROVENANCE.md` and `NOTICE` updated.
