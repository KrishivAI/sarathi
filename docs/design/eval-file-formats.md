# Eval crate: on-disk formats

`sarathi-eval` reads and writes three file formats. Each carries (or should carry) a version, and a reader of version N
reads N and N-1. Version 1 is the first of each, so no N-1 exists yet; the rules below say what a change must do.

## Brief set (`*.jsonl`)

- Line 1 is the header `{"sarathi_brief_set": 2}`; every further line is one brief. Written version: `BRIEF_SET_VERSION`.
- Brief fields: `id` (unique, a plain path component), `split` (`tuning` | `measurement`), `site_context`, `text`,
  `allowed_models` (the *slots* a brief may run on, such as `strong`; not model ids), `acceptance` (name of the check), and from version 2 an optional `fault` (a fault the harness must inject, e.g.
  `malformed_steps`). Version 1 sets, which have no `fault`, still read.
- Unknown fields are refused (`deny_unknown_fields`) in the header and in every brief, so a newer file is never read
  silently as an older one. A newer version number is refused with `BriefError::Version`.
- N-1 rule: a change that adds a field bumps the version; the new reader accepts the old file by defaulting the
  field, and the old reader refuses the new file by version.

## Cassette (`*.jsonl`)

- Line 1 is the header `{"format":"sarathi.cassette","version":1,"model":...,"recorded_on":"YYYY-MM-DD"}`; further lines
  are entries with a `seq` that must count up from 0, a request (or its hash) and the recorded response (a complete
  body or frames with optional `offset_ms`). Written version: `CASSETTE_VERSION`.
- Unknown fields are refused everywhere; a wrong `format` or a `version` of 0 or above the reader's is refused.
- N-1 rule as above. Replay compares the request's JSON and its bytes; equal JSON with different bytes is a mismatch.
- Recording refuses to write text that looks like a secret.

## Run record (`<root>/<date>/<purpose>/<arm>/<brief>/<run>.json`)

- One pretty-printed JSON object per run: `brief`, `arm` (unique name, e.g. `new.strong`), `slot` (the slot the arm
  filled), `model` (the concrete id; the price key), `run`, `accepted`, `valid_ops`, `total_ops`, `usage`, `steps`,
  `wall_ms`, `error`.
- **Versioned from the start of this change:** `version` (1, `RUN_RECORD_VERSION`) and `purpose` (`tune` | `measure`) are required fields; the path is `<root>/<date>/<purpose>/<arm>/<brief>/<run>.json`. Unknown fields are still ignored, so the next field change bumps `version`. `slot` is `#[serde(default)]` (empty when absent). A tuning pass and the measurement never share a directory, `arm_stats` refuses a mix of purposes, and the kill line fails on stats that are not from a measurement pass.
- Components of the path (`date`, `arm`, `brief`) are checked to be plain names before anything is written.
- A pass that spends money saves each record as its run ends (`save_record`): written and synced to
  `<run>.json.tmp`, then renamed, and never over an existing record. One writer per run directory.
- `read_records` reads a whole pass back (version N and N-1) and skips nothing, because a skipped record is spend that
  goes uncounted: a file that is not a whole record (a left-over `.tmp` included), an unknown version or a record that
  is not at its own path is an error naming the file.
