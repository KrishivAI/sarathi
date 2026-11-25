# Provenance

Where copied code came from. One entry per copy; a copied file carries a header line naming this file.

## Ultrafast Gateway

The provider layer (`sarathi-provider`) is derived from the MIT-licensed Ultrafast Gateway
(<https://github.com/techgopal/ultrafast-ai-gateway>, crates `ufaigw-translate` and `ufaigw-client`).

**Status: nothing has been copied into this repository yet.**

| Field | Value |
|---|---|
| Source | `github.com/techgopal/ultrafast-ai-gateway` |
| Source licence | MIT |
| Source commit | `c9c3eafa5c03ae4ac503b4b544a48774bbffe500` (the commit the copy is taken from) |
| Taken | provider clients, request and response translation, SSE streaming, tool-call mapping, retries and timeouts |
| Left | console, storage, admin API, routing tables, gateway-side budgets |
| Renames | crates and types are renamed to `sarathi-*` / `Sarathi*`; listed here when the copy lands |
| Licence here | Apache-2.0; the MIT notice is kept in `NOTICE` for any file not rewritten |

After the copy the two code bases diverge on purpose: no shared crate, no obligation to port fixes either way. Each
copied file keeps a header line `// Derived from Ultrafast Gateway @ <short hash> (MIT); see PROVENANCE.md` above its
`SPDX-License-Identifier: Apache-2.0` line.

## Rules

- Code copied from any other project adds a row here and an entry in `NOTICE`.
- Design ideas borrowed from other projects are credited in documentation, not copied as code.
