# sarathi-eval (S4)
Offline evaluation of a harness. Never makes a live model call
unless a caller passes a live provider explicitly.

## Layout
- `metrics.rs`: `RunRecord`, `PriceTable` (micro-dollars, exact; cache writes have their own price, or the input price when
  none is given), per-brief and per-arm stats (the metrics table), including the cost spread of accepted runs (a brief
  needs two accepted runs to count).
- `config.rs`: the run-configuration file (`RunConfig`: slot -> model, native tools or not, price, cap in whole dollars, total cap;
  versioned, unknown fields refused) and `run_within_caps` (the runner's plan, run by run; the cap is checked per group,
  one brief and repetition on one slot over every harness, so a stopped slot stays paired and overshoots by at most one
  group; earlier records count; arms are `<harness>.<slot>`; `CappedRun::stopped` goes to `Report::stopped` and
  `ArmPair::stopped`). A pass that spends money uses `run_within_caps_journaled` with a `Journal`: every record is
  saved as its run ends and a restarted pass takes the saved ones as `done`, so only the missing runs are made.
- `pass.rs`: `Pass`, one pass that spends money (`PassKind` smoke, tuning round, measurement) over one root that holds
  every pass's records: earlier spend is read from disk, `prepare` is the dry run (`Summary`), `run` locks the root,
  saves each record and writes the reports; the measurement starts only after a smoke run whose finished runs, projected,
  are within every cap.
- `structural.rs`: the `Structural` acceptance check over the host-written `Proposal` artifact (valid operation, no
  validator error, required sections kept, contrast).
- `killline.rs`: the kill-line rule as one pure function; the tolerance is an input.
- `measure.rs`: a `CappedRun` to its `Report`: stats per arm (slots, then harnesses, in configuration order), the kill
  line over the old/new pair of each slot under a `Judging` rule stated before the run (old, new, strong slot, cost
  tolerance `CostTolerance::OldSpread` or a fixed `Fraction`; accept tolerance = the old arm's spread), `stopped` passed to both; a slot with no runs is left out with a note. Each `Tolerance` carries its `CostSource` (fixed, old spread, or old spread missing: no old brief has two accepted runs, so 0 and a kill-line note); report.md prints it.
- `brief.rs`: brief-set JSONL format (header line, then briefs; unknown fields refused; split tuning/measurement).
- `cassette.rs`: cassette JSONL format, reader and writer, request match naming the first differing field, secret scan.
- `report.rs`: `report.md` and `report.json`, byte-stable for the same input.
- `runner.rs`: `Harness` and `Acceptance` traits, the run plan (arm order alternates by brief), the split guard (tuning vs measurement), `run_briefs` (a failed run is a record; `check_plan` and `run_one` are its two halves), the run directory (`save_record` for one record, whole and never over another; `read_records` reads a pass back and refuses anything it cannot count).

- `caller.rs`: `Recording<C>` and `Replaying<C>`, decorators over a core `ModelCaller`: record every call as a cassette entry; replay in order with a byte-for-byte request match (a mismatch names the field). Streams are kept as the assembled message; a failure keeps its status class.
