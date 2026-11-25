# Theme Studio brief set

`briefs.jsonl` is the fixed brief set for comparing two harnesses on the same theme-design task: 30 briefs, 20
`measurement` and 10 `tuning`. Only `tuning` briefs may drive a prompt or profile change; `measurement` briefs decide
the kill line and are never tuned against. Composition: 10 plain, 10 vague, 5 tightly constrained, 5 adversarial
(injection text in site content, an oversized page, a model returning malformed steps).

- `allowed_models` names three arms (`strong`, `cheap-tools`, `fallback-no-tools`); the run configuration maps each arm
  to a concrete model id, which is recorded per run.
- `acceptance` is `structural` (valid operations, contrast, required sections); a person-labelled sample calibrates it.
- The file is committed before any run. A change after the first run is logged as a deviation, not edited in.
