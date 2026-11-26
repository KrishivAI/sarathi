# sarathi-eval (S4)
Offline evaluation of a harness. Never makes a live model call
unless a caller passes a live provider explicitly.

## Layout
- `metrics.rs`: `RunRecord`, `PriceTable` (micro-dollars, exact; cache writes have their own price, or the input price when
  none is given), per-brief and per-arm stats (the metrics table).
- `config.rs`: the run-configuration file (`RunConfig`: slot -> model, native tools or not, price, cap in whole dollars, total cap;
  versioned, unknown fields refused) and `run_within_caps` (the runner's plan, run by run; a slot stops once its spend,
  earlier records included, reaches its cap, so it can overshoot by one run; arms are `<harness>.<slot>`).
- `killline.rs`: the kill-line rule as one pure function; the tolerance is an input.
- `brief.rs`: brief-set JSONL format (header line, then briefs; unknown fields refused; split tuning/measurement).
- `cassette.rs`: cassette JSONL format, reader and writer, request match naming the first differing field, secret scan.
- `report.rs`: `report.md` and `report.json`, byte-stable for the same input.
- `runner.rs`: `Harness` and `Acceptance` traits, the run plan (arm order alternates by brief), the split guard (tuning vs measurement), `run_briefs` (a failed run is a record; `check_plan` and `run_one` are its two halves), the run directory.

- `caller.rs`: `Recording<C>` and `Replaying<C>`, decorators over a core `ModelCaller`: record every call as a cassette entry; replay in order with a byte-for-byte request match (a mismatch names the field). Streams are kept as the assembled message; a failure keeps its status class.
