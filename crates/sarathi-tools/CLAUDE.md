# sarathi-tools (S2)
Tools, permissions and hooks. Depends on `sarathi-protocol` only.
- `tool.rs`: `Toolbox` (a surface's tools plus shared state; `Err` from `call` is an observation), `Tool` and
  `ToolSet` (independent tools), `ToolSpec` (schema, `read_only`, `edits`, evidence kind).
- `rule.rs`, `policy.rs`: `Tool(pattern)` rules with allow/ask/deny; deny beats ask beats allow across scopes; a call
  no rule matches falls to the `Mode` (default, accept-edits, plan, bypass).
- `gate.rs`: the policy plus a host callback (`PermissionHost`, may stay pending); no host means an ask is a refusal.
  "Allow always" adds a session rule.
- `hook.rs`: pre-tool (block or rewrite), post-tool (replace output), stop (send the model back to work).
- `runner.rs`: `ToolRunner::prepare` (hooks + gate, may wait for a person) then `execute` (tool + post hooks) so a host logs the decision and the start mark between; `run` does both. Never an error, always a result.
Rules: the tool list is read once per session (fixed prefix); a refused call never reaches the tool.
