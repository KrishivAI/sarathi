# sarathi-protocol (S0)
The types every crate shares: `RawJson`, message slices (`MessageSlice`, `Seq`, `Role`), the response view
(`ResponseView`, `BlockView`, `BlockKind` with the opaque variant for unknown block types, `Usage`), ids
(`SessionId`, `SegmentId`, `CallId`, `TurnId`), `Op` (host to run loop), `Event` (run loop to host), `ErrorKind`, and
the log's header, format version, entry head and kinds. Depends on nothing in the workspace. No IO.

Rules: raw provider bytes survive a round trip untouched, so `RawJson` is never parsed into a tree and written again.
Enums that hold `RawJson` are externally tagged (serde buffers internally tagged, adjacently tagged and untagged
enums, which loses the raw bytes); every such variant has a round-trip test. Never `#[serde(flatten)]` on a type that
holds `RawJson`: it re-renders the bytes without an error. Public enums and error types are `#[non_exhaustive]`.
Structs other crates build with literals (`Usage`, `ResponseView`, `BlockView`, `ToolOutput`) are fixed shape: a new
field is a deliberate breaking change; the stored ones read a missing field as its default. Error kind names and
entry kind names are stable: they are written to logs; enums written into log entries read an unknown name as
`Unknown` (or `Other`). An unknown `Role` is an error (a message is never re-sent under a guessed role).
