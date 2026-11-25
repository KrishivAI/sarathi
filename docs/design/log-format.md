# Session log format

The on-disk format of `sarathi-log`: one JSONL file per session, a header line, then one entry per line. This note
records the format's rules and every change to it. The code is the reference for field details
(`crates/sarathi-log/src/line.rs`); committed fixtures for each version are in `crates/sarathi-log/fixtures/log/`.

## Versioning
- The header's `sarathi_log` is the format version. A reader of version N reads N and N-1 and refuses anything else
  (newer formats are never guessed at).
- Within a version, an entry of an unknown kind is kept as its whole line and skipped, unless it carries
  `"must_understand": true`, in which case the reader refuses the log.
- Changing what an existing kind or field means, or how lines are framed, needs a new version and a new fixture
  directory; old fixture directories stay and stay green.
- A new kind that an older reader could misread if it skipped it is written with `must_understand`.
- Readers ignore unknown fields of a known kind and do not keep them, so a new field on a known kind is only safe
  when ignoring it gives the same meaning. Otherwise it is a new kind or a new version.
- Fork hashes (`forked_from.sha256`) are over the parent's encoded entry lines, so a reader must re-encode a line
  it read to the same bytes. Names of open sets (error kinds, outcomes) are therefore stored as the names read.

## Version 1
**Not frozen until the first release.** Until then version 1 may still change in place; logs written by earlier
development builds may not load.

Header: `sarathi_log`, `session` (UUID v7), `created` (RFC 3339 UTC, to the second), `writer`, `forked_from`
(`session`, `seq`, `sha256` of the parent's lines 1..=seq) or null.

Entry fields: `seq` (dense from 1), `seg` (segment, +1 at each `segment.start`), `ts` (ms since the epoch,
informational), `kind`, then the kind's fields; `esc` lists stored slices written as strings because they hold a
raw line break.

Kinds: `session.prefix`, `message.user`, `model.request`, `model.response`, `model.failure`, `message.assistant`,
`tool.ask`, `tool.decision`, `tool.start`, `tool.result`, `segment.seed` (`must_understand`), `segment.start`,
`turn.end`, `note`.

Invariants checked on every append and on every read (a log read is one a store would accept): dense seqs; an entry
stays in the current segment unless it is a `segment.start`, which opens the next; references (`from`, `seed`, a
recipe's `prefix` and `messages`) point back to entries of the right kind; a `segment.start` has a `strategy`
exactly when its reason is `compaction`; `tool.start` and `tool.result` name a call the open turn's last
`message.assistant` asked for; `turn.end` comes only after every such call has its `tool.result` (an interrupt
answers open calls first); `model.request.sha256` is 64 lower-case hex digits. A fork's copy of its parent's lines
must hash to `forked_from.sha256` (a fork cut short before its copy is complete is read as it is).

Not checked on append: that a request's recipe rebuilds to its `sha256`. The body is assembled by the provider
layer, so a log can hold a request whose recipe does not rebuild; hosts that need the guarantee run
`Log::verify_requests` with their assembler (the eval runner and the fixture tests do).

Resume: `turn.end` closes the turn, including a model call that has no answer (an interrupt or a budget stop
during the call); a user message that starts a turn also leaves any older unanswered request behind.

Crash atomicity: a refused batch writes nothing, but a crash during an append can leave a prefix of the batch in
whole lines, and the reader accepts it. Writers order every batch so each prefix is a state they can resume from.

Store surface: `LogStore` has `create(created, forked_from)`, `append(session, expected, entries, durability)` and
`load(session)` (whole session); `fork` is a function over any store. The JSONL store's `load` and `append` open the
session for writing (lock, torn-tail repair, kept until `release` or drop); `JsonlStore::read` takes no lock and
changes nothing.

### Changes before the first release
- `created` changed from integer milliseconds to RFC 3339 text; `forked_from.sha256` is a typed digest (same text).
- `segment.seed` added: a slice made for a compaction seed (`role`, `content`, optional `from`, optional `calls`).
  It is projected only through a `segment.start` that names it and ignored by resume, so a seed batch cut short
  before its `segment.start` leaves the old segment's projection and resume unchanged.
- `model.request` has an optional `compaction` (`server` or `client_summary`) on a compaction call. Its
  `model.response` is never accepted as a `message.assistant`: a `segment.seed` copies it (`from` names the
  response: a server compaction block byte for byte, or a client summary), and the `segment.start` whose seed holds
  that copy answers the request for resume. A `segment.start` with any other seed answers nothing.
