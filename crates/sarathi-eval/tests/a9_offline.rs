// SPDX-License-Identifier: Apache-2.0
//! The A9 offline run: all 30 Theme Studio briefs (`evals/vyasa/briefs.jsonl`) under the A9 run configuration
//! (`evals/vyasa/a9-run.json`): the 10 tuning briefs as the first tuning round (one run per harness on `strong`), then
//! the 20 measurement briefs, driven through the real core `Driver`, recorded to cassettes by a scripted model and
//! then replayed from them. It proves the measurement pipeline (arms, caps, pricing, cassettes, report, kill line)
//! end to end before any live spend.
//!
//! The scripted model and the theme tools are stand-ins: the numbers say nothing about a real model or about Vyasa's
//! agent. The `old` arm is a scripted stand-in for the existing harness (it re-reads the page, and uses no prompt
//! cache); the live run replaces it with the real adapter.

#![expect(clippy::unwrap_used, reason = "test code")]

use std::collections::BTreeMap;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use sarathi_core::anthropic::{self, PROTOCOL};
use sarathi_core::driver::{
    AskReply, BoxFuture, CallFailure, Driver, DriverConfig, ModelCaller, PermissionHost, Received,
    StopSignal, Tools,
};
use sarathi_core::{Effect, Limits, ModelProfile, ToolDef, Verdict};
use sarathi_eval::brief::{Brief, Split};
use sarathi_eval::caller::{Recording, Replaying, Scripted};
use sarathi_eval::cassette::{self, Cassette, Response};
use sarathi_eval::config::{RunConfig, run_within_caps};
use sarathi_eval::killline::Outcome;
use sarathi_eval::measure::{CostTolerance, Judging, report};
use sarathi_eval::metrics::{Purpose, RunRecord, TokenUsage};
use sarathi_eval::report::{to_json, to_markdown};
use sarathi_eval::runner::{Acceptance, Arm, Harness, HarnessError, HarnessOutput};
use sarathi_log::{MemoryStore, RequestParts};
use sarathi_protocol::{CallId, ErrorKind, Event, Op, RawJson, ToolOutput};
use sarathi_provider::{Endpoint, MessagesRequest};

const BRIEFS: &str = include_str!("../../../evals/vyasa/briefs.jsonl");
const A9: &str = include_str!("../../../evals/vyasa/a9-run.json");
const DATE: &str = "2026-11-27";

const JUDGING: Judging<'static> = Judging {
    old: "old",
    new: "new",
    strong_slot: "strong",
    // A stand-in until Q-S6 (the cost tolerance) is answered; it is not the rule.
    cost: CostTolerance::Fraction(0.05),
};

fn raw(s: &str) -> RawJson {
    RawJson::from_text(s).unwrap()
}

// ---------------------------------------------------------------------------------------------------------------
// The model seam: the real Anthropic request building, no network.

/// Builds the real request bytes and never sends: `Scripted` or `Replaying` answers in its place.
struct Builder;

impl ModelCaller for Builder {
    type Prepared = Vec<u8>;

    fn protocol(&self) -> &str {
        PROTOCOL
    }

    fn params(&self) -> RawJson {
        raw(r#"{"model":"offline","max_tokens":4096}"#)
    }

    fn render_tools(&self, tools: &[ToolDef]) -> Result<Option<RawJson>, CallFailure> {
        anthropic::render_tools(tools)
    }

    fn result_block(&self, call: &CallId, content: &RawJson, is_error: bool) -> RawJson {
        anthropic::result_block(call, content, is_error)
    }

    fn prepare(&self, parts: &RequestParts<'_>) -> Result<(Vec<u8>, Vec<u8>), CallFailure> {
        let req: MessagesRequest = anthropic::request(parts)
            .map_err(|e| CallFailure::new(ErrorKind::InvalidRequest, e))?;
        let body = req.body(Endpoint::Messages);
        Ok((body.clone(), body))
    }

    fn send(&self, _: Vec<u8>) -> BoxFuture<'_, Result<Received, CallFailure>> {
        Box::pin(async {
            Err(CallFailure::new(
                ErrorKind::Upstream,
                "offline: nothing is sent",
            ))
        })
    }

    fn read(&self, body: &RawJson, content: Range<u32>) -> Result<Received, CallFailure> {
        let mut r = anthropic::answer(body)?;
        r.content = content;
        Ok(r)
    }
}

/// A caller the driver owns and the harness can still read (the recording) after the turn.
struct Shared<C>(Arc<C>);

impl<C: ModelCaller> ModelCaller for Shared<C> {
    type Prepared = C::Prepared;

    fn protocol(&self) -> &str {
        self.0.protocol()
    }

    fn params(&self) -> RawJson {
        self.0.params()
    }

    fn render_tools(&self, tools: &[ToolDef]) -> Result<Option<RawJson>, CallFailure> {
        self.0.render_tools(tools)
    }

    fn result_block(&self, call: &CallId, content: &RawJson, is_error: bool) -> RawJson {
        self.0.result_block(call, content, is_error)
    }

    fn prepare(&self, parts: &RequestParts<'_>) -> Result<(Vec<u8>, Self::Prepared), CallFailure> {
        self.0.prepare(parts)
    }

    fn send(&self, p: Self::Prepared) -> BoxFuture<'_, Result<Received, CallFailure>> {
        self.0.send(p)
    }

    fn read(&self, body: &RawJson, content: Range<u32>) -> Result<Received, CallFailure> {
        self.0.read(body, content)
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The theme tools: `look` reads, `set_token` writes and counts valid and invalid operations.

const TOKENS: [&str; 5] = [
    "color.background",
    "color.text",
    "font.body",
    "color.accent",
    "radius.card",
];
const REQUIRED: [&str; 3] = ["color.background", "color.text", "font.body"];

#[derive(Debug, Default)]
struct Page {
    set: BTreeMap<String, String>,
    valid: u32,
    total: u32,
}

#[derive(Debug, Clone, Default)]
struct ThemeTools(Arc<Mutex<Page>>);

fn valid_value(token: &str, value: &str) -> bool {
    match token {
        "color.background" | "color.text" | "color.accent" => hex(value).is_some(),
        "font.body" => !value.is_empty() && value.len() < 60,
        "radius.card" => value.ends_with("px") && value[..value.len() - 2].parse::<u32>().is_ok(),
        _ => false,
    }
}

fn hex(v: &str) -> Option<[u8; 3]> {
    let h = v.strip_prefix('#')?;
    if h.len() != 6 {
        return None;
    }
    let n = u32::from_str_radix(h, 16).ok()?;
    Some([(n >> 16) as u8, (n >> 8) as u8, n as u8])
}

/// WCAG relative luminance contrast of two `#rrggbb` colours.
fn contrast(a: &str, b: &str) -> f64 {
    let lum = |c: [u8; 3]| {
        let f = |v: u8| {
            let s = f64::from(v) / 255.0;
            if s <= 0.039_28 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            }
        };
        0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2])
    };
    let (la, lb) = (lum(hex(a).unwrap_or([0; 3])), lum(hex(b).unwrap_or([0; 3])));
    (la.max(lb) + 0.05) / (la.min(lb) + 0.05)
}

impl Tools for ThemeTools {
    fn evaluate(&self, _tool: &str, _input: &RawJson) -> Verdict {
        Verdict::Allow
    }

    fn run(
        &self,
        _call: CallId,
        tool: Box<str>,
        input: RawJson,
        _stop: StopSignal,
    ) -> BoxFuture<'_, ToolOutput> {
        Box::pin(async move {
            let text = |s: &str, is_error: bool| ToolOutput {
                content: RawJson::string(s),
                is_error,
            };
            match &*tool {
                "look" => text("the page: header, article column, sidebar, footer", false),
                "set_token" => {
                    let mut page = self.0.lock().unwrap();
                    page.total += 1;
                    let v: serde_json::Value =
                        serde_json::from_str(input.as_str()).unwrap_or_default();
                    let (t, val) = (
                        v["token"].as_str().unwrap_or_default(),
                        v["value"].as_str().unwrap_or_default(),
                    );
                    if valid_value(t, val) {
                        page.valid += 1;
                        page.set.insert(t.to_owned(), val.to_owned());
                        text("set", false)
                    } else {
                        text("that token or value is not valid", true)
                    }
                }
                other => text(&format!("no tool {other}"), true),
            }
        })
    }
}

struct Allow;
impl PermissionHost for Allow {
    fn ask(&self, _: &CallId, _: &str, _: &RawJson) -> AskReply {
        AskReply::Decided(sarathi_protocol::Decision::AllowOnce)
    }
}

fn defs() -> Vec<ToolDef> {
    let obj = || raw(r#"{"type":"object"}"#);
    vec![
        ToolDef::new("look", "shows the page", obj(), Effect::Read),
        ToolDef::new(
            "set_token",
            "sets a design token: {\"token\", \"value\"}",
            obj(),
            Effect::Write,
        ),
    ]
}

// ---------------------------------------------------------------------------------------------------------------
// The script: what the stand-in model answers for one run.

/// One scripted model step: its calls and its token usage.
struct Step {
    calls: Vec<(String, String)>,
    reply: String,
    usage: [u64; 4],
    /// The answer is malformed under the slot's step protocol (the `malformed_steps` fault), so the core recovers.
    garbled: bool,
}

fn seed(brief: &str) -> u64 {
    brief.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The steps of one run. `new` reads the page once and hits the prompt cache; `old` reads it twice and never does.
/// A fraction of runs pick a low-contrast pair (rejected by the acceptance check) and a fraction make one invalid
/// operation (a refused tool call, counted in the validity metric).
fn steps(brief: &Brief, harness: &str, run: u32) -> Vec<Step> {
    let h = seed(&brief.id);
    let new = harness == "new";
    let low = if new {
        (h + u64::from(run)).is_multiple_of(13)
    } else {
        (h + u64::from(run)).is_multiple_of(9)
    };
    let (bg, fg) = if low {
        ("#777777", "#808080")
    } else {
        ("#fdfaf5", "#1c1b19")
    };
    let sets = 3 + usize::try_from(h % 3).unwrap_or(0);
    let mut calls: Vec<(String, String)> = vec![("look".into(), "{}".into())];
    if !new {
        calls.push(("look".into(), "{}".into()));
    }
    let bad = (h / 7 + u64::from(run)).is_multiple_of(5) || brief.fault.is_some();
    if bad {
        calls.push((
            "set_token".into(),
            r##"{"token":"colour.bg","value":"#fff"}"##.into(),
        ));
    }
    for t in &TOKENS[..sets] {
        let value = match *t {
            "color.background" => bg,
            "color.text" => fg,
            "font.body" => "Georgia, serif",
            "color.accent" => "#2f6f8f",
            _ => "6px",
        };
        calls.push((
            "set_token".into(),
            format!(r#"{{"token":"{t}","value":"{value}"}}"#),
        ));
    }
    let n = calls.len() as u64;
    let mut out: Vec<Step> = calls
        .into_iter()
        .enumerate()
        .map(|(i, c)| {
            let i = i as u64;
            let usage = if new {
                // The prefix is written to the cache on the first call and read after; only the new turn is uncached.
                if i == 0 {
                    [1_000, 600, 0, 8_000]
                } else {
                    [1_000 + 400 * i, 600, 8_000, 0]
                }
            } else {
                [9_000 + 2_000 * i, 600, 0, 0]
            };
            Step {
                calls: vec![c],
                reply: String::new(),
                usage,
                garbled: false,
            }
        })
        .collect();
    out.push(Step {
        calls: Vec::new(),
        reply: "The theme is updated.".into(),
        usage: if new {
            [1_000 + 400 * n, 300, 8_000, 0]
        } else {
            [9_000 + 2_000 * n, 300, 0, 0]
        },
        garbled: false,
    });
    if brief.fault.as_deref() == Some("malformed_steps") {
        // The first answer is malformed; the core asks again and the run still finishes.
        out.insert(
            0,
            Step {
                calls: Vec::new(),
                reply: String::new(),
                usage: [1_000, 50, 0, 0],
                garbled: true,
            },
        );
    }
    out
}

/// A step as the model's answer under the slot's step protocol.
fn answer(step: &Step, n: usize, native: bool) -> Result<Received, CallFailure> {
    let usage = format!(
        r#""usage":{{"input_tokens":{},"output_tokens":{},"cache_read_input_tokens":{},"cache_creation_input_tokens":{}}}"#,
        step.usage[0], step.usage[1], step.usage[2], step.usage[3]
    );
    let (content, stop) = if step.garbled && native {
        // A `tool_use` whose input is not an object.
        (
            r#"{"type":"tool_use","id":"toolu_bad","name":"set_token","input":"not an object"}"#
                .to_owned(),
            "tool_use",
        )
    } else if step.garbled {
        (
            format!(
                r#"{{"type":"text","text":{}}}"#,
                RawJson::string("this is not json {").as_str()
            ),
            "end_turn",
        )
    } else if native {
        let mut blocks: Vec<String> = Vec::new();
        if !step.reply.is_empty() {
            blocks.push(format!(
                r#"{{"type":"text","text":{}}}"#,
                RawJson::string(&step.reply).as_str()
            ));
        }
        for (i, (tool, input)) in step.calls.iter().enumerate() {
            blocks.push(format!(
                r#"{{"type":"tool_use","id":"toolu_{n}_{i}","name":"{tool}","input":{input}}}"#
            ));
        }
        (
            blocks.join(","),
            if step.calls.is_empty() {
                "end_turn"
            } else {
                "tool_use"
            },
        )
    } else {
        let (tool, input) = step
            .calls
            .first()
            .map_or(("done", "{}"), |(t, i)| (t.as_str(), i.as_str()));
        let text = format!(
            r#"{{"thought":"step {n}","tool":"{tool}","input":{input},"reply":{}}}"#,
            RawJson::string(&step.reply).as_str()
        );
        (
            format!(
                r#"{{"type":"text","text":{}}}"#,
                RawJson::string(&text).as_str()
            ),
            "end_turn",
        )
    };
    let body = format!(
        r#"{{"id":"msg_{n}","type":"message","role":"assistant","model":"offline","content":[{content}],"stop_reason":"{stop}",{usage}}}"#
    );
    anthropic::answer(&raw(&body))
}

/// The token usage a cassette's answers report.
fn usage_of(c: &Cassette) -> TokenUsage {
    let mut u = TokenUsage::default();
    for e in &c.entries {
        let Response::Body { body, .. } = &e.response else {
            continue;
        };
        let v: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
        let n = |k: &str| v["usage"][k].as_u64().unwrap_or(0);
        u.input_tokens += n("input_tokens");
        u.output_tokens += n("output_tokens");
        u.cached_input_tokens += n("cache_read_input_tokens");
        u.cache_write_input_tokens += n("cache_creation_input_tokens");
    }
    u
}

// ---------------------------------------------------------------------------------------------------------------
// The harness: one driver turn per run, recorded or replayed.

#[derive(Clone)]
enum Mode {
    /// Answer from the script and write a cassette per run under the directory.
    Record(PathBuf),
    /// Answer from the cassettes under the directory only.
    Replay(PathBuf),
}

struct Theme {
    name: &'static str,
    slot: &'static str,
    mode: Mode,
}

fn cassette_path(dir: &Path, arm: &str, brief: &str, run: u32) -> PathBuf {
    dir.join(arm).join(brief).join(format!("{run}.jsonl"))
}

impl Theme {
    async fn drive<C: ModelCaller>(
        &self,
        brief: &Brief,
        caller: C,
        native: bool,
    ) -> Result<ThemeTools, HarnessError> {
        let tools = ThemeTools::default();
        let profile = if native {
            ModelProfile::native()
        } else {
            ModelProfile::json_object(true)
        };
        let config = DriverConfig::new(profile, defs(), "You design themes for a website.")
            .with_limits(Limits::with_max_steps(12));
        let store = MemoryStore::default();
        let (mut driver, _handle) = Driver::new(config, store, caller, tools.clone(), Allow)
            .await
            .map_err(|e| HarnessError::new(e.to_string()))?;
        let op = Op::UserMessage {
            content: RawJson::string(&format!("{}\n\nSite: {}", brief.text, brief.site_context)),
        };
        let end = driver
            .run_turn(op, &mut |_: Event| {})
            .await
            .map_err(|e| HarnessError::new(e.to_string()))?;
        if end.outcome != sarathi_protocol::Outcome::Finished {
            return Err(HarnessError::new(format!(
                "the turn ended {:?}",
                end.outcome
            )));
        }
        Ok(tools)
    }
}

impl Harness for Theme {
    fn run(&self, brief: &Brief, _model: &str, run: u32) -> Result<HarnessOutput, HarnessError> {
        let native = self.slot != "fallback-no-tools";
        let arm = RunConfig::arm_name(self.name, self.slot);
        let path = cassette_path(
            match &self.mode {
                Mode::Record(d) | Mode::Replay(d) => d,
            },
            &arm,
            &brief.id,
            run,
        );
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .map_err(|e| HarnessError::new(e.to_string()))?;
        let (tools, cassette) = match &self.mode {
            Mode::Record(_) => {
                let replies: Vec<_> = steps(brief, self.name, run)
                    .iter()
                    .enumerate()
                    .map(|(n, s)| answer(s, n, native))
                    .collect();
                let rec = Arc::new(Recording::new(Scripted::new(Builder, replies)));
                let tools = rt.block_on(self.drive(brief, Shared(rec.clone()), native))?;
                let c = rec.cassette(&format!("offline-{}", self.slot), DATE);
                let text = cassette::write(&c).map_err(|e| HarnessError::new(e.to_string()))?;
                if let Some(secret) = cassette::find_secret(&text) {
                    return Err(HarnessError::new(format!(
                        "a cassette holds a secret: {secret}"
                    )));
                }
                std::fs::create_dir_all(path.parent().unwrap())
                    .map_err(|e| HarnessError::new(e.to_string()))?;
                std::fs::write(&path, text).map_err(|e| HarnessError::new(e.to_string()))?;
                (tools, c)
            }
            Mode::Replay(_) => {
                let text = std::fs::read_to_string(&path)
                    .map_err(|e| HarnessError::new(format!("{}: {e}", path.display())))?;
                let c = cassette::parse(&text).map_err(|e| HarnessError::new(e.to_string()))?;
                let rep = Arc::new(Replaying::new(Builder, c.clone(), anthropic::answer));
                let tools = rt.block_on(self.drive(brief, Shared(rep.clone()), native))?;
                if rep.remaining() != 0 {
                    return Err(HarnessError::new(
                        "the run made fewer calls than the cassette holds",
                    ));
                }
                (tools, c)
            }
        };
        let page = tools.0.lock().unwrap();
        let artifact = page
            .set
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(";");
        Ok(HarnessOutput {
            artifact,
            valid_ops: page.valid,
            total_ops: page.total,
            usage: usage_of(&cassette),
            steps: u32::try_from(cassette.entries.len()).unwrap_or(u32::MAX),
            // Wall time is not part of a replay's identity: a fixed number keeps the two reports comparable.
            wall_ms: 0,
        })
    }
}

/// The structural acceptance check: the required tokens are set and text on background has contrast of at least 4.5.
struct Structural;
impl Acceptance for Structural {
    fn accepts(&self, _brief: &Brief, out: &HarnessOutput) -> bool {
        let set: BTreeMap<&str, &str> = out
            .artifact
            .split(';')
            .filter_map(|kv| kv.split_once('='))
            .collect();
        REQUIRED.iter().all(|t| set.contains_key(t))
            && contrast(set["color.background"], set["color.text"]) >= 4.5
    }
}

fn briefs_of(split: Split) -> Vec<Brief> {
    let mut lines = BRIEFS.lines();
    lines.next().unwrap();
    lines
        .map(|l| serde_json::from_str::<Brief>(l).unwrap())
        .filter(|b| b.split == split)
        .collect()
}

fn arms<'a>(c: &RunConfig, harnesses: &'a [Theme]) -> Vec<Arm<'a>> {
    harnesses
        .iter()
        .map(|h| {
            let s = c.slot(h.slot).unwrap();
            Arm {
                name: RunConfig::arm_name(h.name, h.slot),
                slot: h.slot.into(),
                model: s.model.clone(),
                harness: h,
            }
        })
        .collect()
}

fn themes(slots: &[&'static str], mode: &Mode) -> Vec<Theme> {
    let mut hs = Vec::new();
    for &slot in slots {
        for name in ["old", "new"] {
            hs.push(Theme {
                name,
                slot,
                mode: mode.clone(),
            });
        }
    }
    hs
}

fn slot_names(c: &RunConfig) -> Vec<&'static str> {
    c.slots
        .iter()
        .map(|s| &*Box::leak(s.slot.clone().into_boxed_str()))
        .collect()
}

/// The first tuning round as the labelling sheet plans it: the tuning briefs, both harnesses, one run, on `strong`
/// only. Its records are not part of the measurement; they count only against the `strong` cap.
fn tune(c: &RunConfig, briefs: &[Brief], mode: &Mode) -> (Vec<RunRecord>, String) {
    let mut t = c.clone();
    t.runs = 1;
    t.slots.retain(|s| s.slot == JUDGING.strong_slot);
    let hs = themes(&[JUDGING.strong_slot], mode);
    let run = run_within_caps(&t, briefs, &arms(&t, &hs), &Structural, Purpose::Tune, &[]).unwrap();
    assert!(
        run.stopped.is_empty(),
        "no cap is near offline: {:?}",
        run.stopped
    );
    let r = report(&t, &run, None, DATE, 0).unwrap();
    assert!(r.kill_line.is_none(), "a tuning round has no kill line");
    (run.records, to_markdown(&r))
}

/// The measurement: every slot, both harnesses, `runs` runs, after the tuning round (`earlier`, for the caps).
fn pass(c: &RunConfig, briefs: &[Brief], mode: &Mode, earlier: &[RunRecord]) -> (String, String) {
    let hs = themes(&slot_names(c), mode);
    let run = run_within_caps(
        c,
        briefs,
        &arms(c, &hs),
        &Structural,
        Purpose::Measure,
        earlier,
    )
    .unwrap();
    let r = report(c, &run, Some(&JUDGING), DATE, 0).unwrap();
    assert!(
        r.stopped.is_empty(),
        "no cap is near offline: {:?}",
        r.stopped
    );
    let k = r.kill_line.as_ref().unwrap();
    eprintln!("kill line: {:?}\n{:#?}", k.outcome, k.notes);
    assert!(matches!(k.outcome, Outcome::Pass | Outcome::Fail));
    (to_json(&r).unwrap(), to_markdown(&r))
}

/// The whole offline run of one mode: the tuning round, then the measurement.
fn all(c: &RunConfig, mode: &Mode) -> (Vec<RunRecord>, String, String, String) {
    let tuning = briefs_of(Split::Tuning);
    assert_eq!(tuning.len(), 10);
    let measurement = briefs_of(Split::Measurement);
    assert_eq!(measurement.len(), 20);
    let (tuned, tune_md) = tune(c, &tuning, mode);
    let (json, md) = pass(c, &measurement, mode, &tuned);
    (tuned, tune_md, json, md)
}

#[test]
fn all_30_briefs_run_offline_on_cassettes_and_replay_to_the_same_reports() {
    let c = RunConfig::from_json(A9).unwrap();
    let dir = std::env::temp_dir().join(format!("sarathi-a9-offline-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let (recorded_tuned, recorded_tune_md, recorded_json, recorded_md) =
        all(&c, &Mode::Record(dir.clone()));
    let (replayed_tuned, replayed_tune_md, replayed_json, replayed_md) =
        all(&c, &Mode::Replay(dir.clone()));
    assert_eq!(
        recorded_tuned, replayed_tuned,
        "a replay reproduces the tuning round"
    );
    assert_eq!(recorded_tune_md, replayed_tune_md);
    assert_eq!(
        recorded_json, replayed_json,
        "a replay reproduces the recorded run's report"
    );
    assert_eq!(recorded_md, replayed_md);
    // 10 tuning briefs x 1 run x 2 harnesses on `strong`, all tuning records (the guard refuses a measurement brief).
    assert_eq!(recorded_tuned.len(), 10 * 2);
    assert!(recorded_tuned.iter().all(|r| r.purpose == Purpose::Tune));
    // Plus 20 briefs x 3 runs x 3 slots x 2 harnesses, one `<run>.jsonl` cassette each (arm / brief / run).
    let files = std::fs::read_dir(&dir)
        .unwrap()
        .flat_map(|arm| std::fs::read_dir(arm.unwrap().path()).unwrap())
        .flat_map(|brief| std::fs::read_dir(brief.unwrap().path()).unwrap())
        .filter(|f| {
            f.as_ref()
                .unwrap()
                .path()
                .extension()
                .is_some_and(|e| e == "jsonl")
        })
        .count();
    assert_eq!(files, 10 * 2 + 6 * 20 * 3);
    if let Ok(out) = std::env::var("SARATHI_A9_OFFLINE_REPORT") {
        std::fs::write(out, &recorded_md).unwrap();
    }
    if let Ok(out) = std::env::var("SARATHI_A9_OFFLINE_TUNING_REPORT") {
        std::fs::write(out, &recorded_tune_md).unwrap();
    }
    let _ = std::fs::remove_dir_all(&dir);
}
