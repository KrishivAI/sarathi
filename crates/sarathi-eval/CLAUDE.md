# sarathi-eval (S4)
Offline evaluation of a harness. Never makes a live model call
unless a caller passes a live provider explicitly.

## Layout
- `metrics.rs`: `RunRecord`, `PriceTable` (micro-dollars, exact), per-brief and per-arm stats (the metrics table).
- `killline.rs`: the kill-line rule as one pure function; the tolerance is an input.
- `brief.rs`: brief-set JSONL format (header line, then briefs; unknown fields refused; split tuning/measurement).
- `cassette.rs`: cassette JSONL format, reader and writer, request match naming the first differing field, secret scan.
- `report.rs`: `report.md` and `report.json`, byte-stable for the same input.
- `runner.rs`: `Harness` and `Acceptance` traits, the run plan (arm order alternates by brief), the split guard (tuning vs measurement), `run_briefs` (a failed run is a record), the run directory.

Not yet built: the scripted provider and the record and replay decorators (they need the provider trait).
