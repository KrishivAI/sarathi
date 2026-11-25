# Theme Studio brief set

`briefs.jsonl` is the fixed brief set for comparing two harnesses on the same theme-design task: 30 briefs, 20
`measurement` and 10 `tuning`. Only `tuning` briefs may drive a prompt or profile change; `measurement` briefs decide
the kill line and are never tuned against. Composition: 10 plain, 10 vague, 5 tightly constrained, 5 adversarial
(injection text in site content, an oversized page, a model returning malformed steps).

- `allowed_models` names three slots (`strong`, `cheap-tools`, `fallback-no-tools`); an arm fills a slot, and the run
  configuration gives the arm a concrete model id, which is recorded per run.
- `acceptance` is `structural` (valid operations, contrast, required sections); a person-labelled sample calibrates it.
- The file is committed before any run. A change after the first run is logged as a deviation, not edited in.
- The adversarial briefs carry their hazard in `site_context` (injection text inside site content; `adv-03` is an
  oversized page, about 43 KB of API reference entries, which fits a documentation site). `adv-04` has no hazard in the brief itself: its `fault` field
  (`malformed_steps`, brief-set version 2) tells the harness or the scripted provider to inject the malformed steps,
  so the model is never told it is being tested. A harness that ignores `fault` runs adv-04 as a plain brief.

## Run configuration

`a9-run.json` is the run configuration for the measurement (`sarathi_eval::config::RunConfig`): three runs per brief per
harness per slot, harnesses `old` and `new`, and one model per slot with its price and spending cap:

| Slot | Model | Native tools | Cap |
|---|---|---|---|
| `strong` | `claude-sonnet-5-5` | yes | $170 |
| `cheap-tools` | `gpt-5-mini` | yes | $20 |
| `fallback-no-tools` | `gemini-2.5-flash` | no (JSON step protocol) | $20 |

The total cap is $210. A cap covers every pass on its slot (the smoke run, tuning rounds and the measurement), so a
later pass is run with the earlier records counted. The cap is checked once per group (one brief and repetition on
one slot, both harnesses), so a slot overshoots by at most one group and its two harnesses always ran the same briefs.
A slot that reaches its cap stops: its remaining groups are not run, the report lists it as incomplete, and the kill
line fails on a stopped strong slot. Keys get provider-side limits at or below their slot's cap as the hard stop.
The prices are checked against the providers' pages on the run day (`prices_checked` records where they came from).
