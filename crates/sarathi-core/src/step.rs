// Derived from Vyasa `crates/ai/src/agent.rs` @ 2403be1 (MIT OR Apache-2.0); see PROVENANCE.md
// SPDX-License-Identifier: Apache-2.0
//! Reading a step from an answer, and the texts the model reads when a step goes wrong.

use sarathi_protocol::{RawJson, ResponseView};
use serde::Deserialize;
use serde_json::value::RawValue;

use crate::profile::DONE_TOOL;

/// A step read under [`crate::StepProtocol::JsonObject`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum JsonStep {
    /// Call `tool` with `input` (its bytes as the model wrote them; `{}` when missing).
    Call { tool: Box<str>, input: RawJson },
    /// `tool: "done"`: the turn ends with this reply.
    Done { reply: Box<str> },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire<'a> {
    #[serde(rename = "thought")]
    _thought: String,
    tool: String,
    #[serde(default, borrow)]
    input: Option<&'a RawValue>,
    #[serde(default)]
    reply: String,
}

/// Reads one JSON step from the answer text. A step wrapped in a Markdown code fence (a bare fence or one tagged
/// `json`, nothing before it) is read too; models without structured output often add one.
///
/// # Errors
/// A readable reason the reply is unusable, for the correction the model reads.
pub(crate) fn read_json_step(text: &str) -> Result<JsonStep, String> {
    let body = strip_fence(text.trim());
    let wire: Wire<'_> = serde_json::from_str(body).map_err(|e| e.to_string())?;
    if wire.tool == DONE_TOOL {
        let reply = wire.reply.trim();
        return Ok(JsonStep::Done {
            reply: if reply.is_empty() { "Done." } else { reply }.into(),
        });
    }
    let input = match wire.input {
        Some(raw) if raw.get() != "null" => {
            if !raw.get().starts_with('{') {
                return Err("`input` must be a JSON object".to_owned());
            }
            RawJson::from_text(raw.get()).map_err(|e| e.to_string())?
        }
        _ => RawJson::from_text("{}").map_err(|e| e.to_string())?,
    };
    Ok(JsonStep::Call {
        tool: wire.tool.into(),
        input,
    })
}

fn strip_fence(text: &str) -> &str {
    let Some(rest) = text.strip_prefix("```") else {
        return text;
    };
    let rest = rest.strip_prefix("json").unwrap_or(rest);
    rest.strip_suffix("```").unwrap_or(rest).trim()
}

/// The text of an answer's `text` blocks under the native protocol, joined; `None` when a block cannot be read.
pub(crate) fn native_text(body: &RawJson, view: &ResponseView) -> Option<String> {
    #[derive(Deserialize)]
    struct TextBlock<'a> {
        #[serde(borrow)]
        text: std::borrow::Cow<'a, str>,
    }
    let mut out = String::new();
    for b in &view.blocks {
        if b.kind != sarathi_protocol::BlockKind::Text {
            continue;
        }
        let slice = body
            .as_str()
            .get(b.range.start as usize..b.range.end as usize)?;
        // A Messages text block, or (an `openai-chat` answer) the `content` string itself.
        if slice.starts_with('"') {
            let text: std::borrow::Cow<'_, str> = serde_json::from_str(slice).ok()?;
            out.push_str(&text);
        } else {
            let block: TextBlock<'_> = serde_json::from_str(slice).ok()?;
            out.push_str(&block.text);
        }
    }
    Some(out)
}

/// The input bytes of a native tool call, cut from the answer body. A string there (an `openai-chat` call's
/// `arguments`) holds the input as text: its text is the input, as the model wrote it, when that is a JSON object.
/// The rule goes by the value, not the protocol: a Messages `input` is always an object, so a string there (never sent
/// by the API) is read the same way rather than passed on as a string.
pub(crate) fn native_input(body: &RawJson, range: std::ops::Range<u32>) -> Option<RawJson> {
    let slice = body
        .as_str()
        .get(range.start as usize..range.end as usize)?;
    if slice.starts_with('"') {
        let text: String = serde_json::from_str(slice).ok()?;
        let text = if text.trim().is_empty() {
            "{}"
        } else {
            text.trim()
        };
        return RawJson::from_text(text)
            .ok()
            .filter(|v| v.as_str().starts_with('{'));
    }
    RawJson::from_text(slice).ok()
}

/// What the model reads after an answer it cannot use as a step.
pub(crate) fn unusable_reply(reason: &str) -> String {
    format!(
        "That reply was unusable ({reason}). Answer with exactly one JSON object matching the schema."
    )
}

/// What the model reads after calling a tool that does not exist.
pub(crate) fn unknown_tool<'a>(
    tool: &str,
    tools: impl Iterator<Item = &'a str>,
    json_object: bool,
) -> String {
    let names = tools.collect::<Vec<_>>().join(", ");
    if json_object {
        format!("\"{tool}\" is not a tool. The tools are: {names} — or \"done\".")
    } else {
        format!("\"{tool}\" is not a tool. The tools are: {names}.")
    }
}

/// What the model reads instead of a third identical call in a row.
pub(crate) const REPEATED_CALL: &str =
    "This call is identical to the previous two calls and was not run; change approach or finish.";

/// The result of a read tool the host interrupted.
pub(crate) const INTERRUPTED: &str = "interrupted";

/// The result of a write tool that did not stop within its grace period after an interrupt.
pub(crate) const INTERRUPTED_EFFECT_UNKNOWN: &str = "interrupted; its effect is unknown";

/// The result of a call the turn ended before reaching.
pub(crate) fn skipped(why: &str) -> String {
    format!("skipped: {why}")
}

/// The closing text of a turn stopped by the step cap.
pub(crate) fn ran_out(max_steps: u32) -> String {
    format!(
        "I ran out of steps before finishing cleanly ({max_steps} used). \
         What is applied so far is what the steps show; ask again to continue."
    )
}

/// How a tool result is shown to the model under [`crate::StepProtocol::JsonObject`]: a user message starting
/// `observation:`, the text of a string result or the JSON of any other, a failure said as a refusal.
pub(crate) fn observation(content: &RawJson, is_error: bool) -> String {
    let text = serde_json::from_str::<String>(content.as_str())
        .unwrap_or_else(|_| content.as_str().to_owned());
    if is_error {
        format!("observation:\nThe tool refused: {text}")
    } else {
        format!("observation:\n{text}")
    }
}

#[cfg(test)]
mod tests {
    use sarathi_protocol::{BlockKind, BlockView};

    use super::*;

    #[test]
    fn a_call_keeps_the_input_bytes_as_written() {
        let step =
            read_json_step(r#"{"thought":"t","tool":"edit","input":{"z" : 1,"a":"é"}}"#).unwrap();
        let JsonStep::Call { tool, input } = step else {
            panic!("{step:?}")
        };
        assert_eq!(&*tool, "edit");
        assert_eq!(input.as_str(), r#"{"z" : 1,"a":"é"}"#);
    }

    #[test]
    fn missing_or_null_input_is_an_empty_object() {
        for text in [
            r#"{"thought":"t","tool":"look"}"#,
            r#"{"thought":"t","tool":"look","input":null}"#,
        ] {
            let JsonStep::Call { input, .. } = read_json_step(text).unwrap() else {
                panic!()
            };
            assert_eq!(input.as_str(), "{}");
        }
    }

    #[test]
    fn done_carries_the_reply_or_a_default() {
        assert_eq!(
            read_json_step(r#"{"thought":"t","tool":"done","reply":"  All set. "}"#).unwrap(),
            JsonStep::Done {
                reply: "All set.".into()
            }
        );
        assert_eq!(
            read_json_step(r#"{"thought":"t","tool":"done"}"#).unwrap(),
            JsonStep::Done {
                reply: "Done.".into()
            }
        );
    }

    #[test]
    fn a_fenced_step_is_read() {
        let text = "```json\n{\"thought\":\"t\",\"tool\":\"done\",\"reply\":\"ok\"}\n```";
        assert!(matches!(read_json_step(text), Ok(JsonStep::Done { .. })));
        let bare = "```\n{\"thought\":\"t\",\"tool\":\"done\"}\n```";
        assert!(matches!(read_json_step(bare), Ok(JsonStep::Done { .. })));
    }

    #[test]
    fn unusable_replies_say_why() {
        for text in [
            "sure, I will edit it",
            r#"{"tool":"edit"}"#,
            r#"{"thought":"t","tool":"edit","input":[1]}"#,
            r#"{"thought":"t","tool":"edit","extra":1}"#,
            r#"{"thought":"t","tool":"edit"} {"thought":"u","tool":"look"}"#,
        ] {
            let e = read_json_step(text).unwrap_err();
            assert!(!e.is_empty(), "{text}");
        }
        assert_eq!(
            unusable_reply("expected value"),
            "That reply was unusable (expected value). Answer with exactly one JSON object matching the schema."
        );
    }

    #[test]
    fn native_text_and_input_are_cut_from_the_body() {
        let body = RawJson::from_text(
            r#"{"content":[{"type":"text","text":"Hi é"},{"type":"tool_use","id":"toolu_1","name":"look","input":{"a" :1}}]}"#,
        )
        .unwrap();
        let s = body.as_str();
        let t0 = u32::try_from(s.find(r#"{"type":"text""#).unwrap()).unwrap();
        let t1 = u32::try_from(s.find(r#"},{"type":"tool_use""#).unwrap() + 1).unwrap();
        let i0 = u32::try_from(s.find(r#"{"a" :1}"#).unwrap()).unwrap();
        let view = ResponseView {
            blocks: vec![
                BlockView {
                    kind: BlockKind::Text,
                    range: t0..t1,
                },
                BlockView {
                    kind: BlockKind::ToolUse {
                        id: "toolu_1".into(),
                        name: "look".into(),
                        input: i0..i0 + 8,
                    },
                    range: t1 + 1..u32::try_from(s.len()).unwrap() - 2,
                },
            ],
            ..ResponseView::default()
        };
        assert_eq!(native_text(&body, &view).unwrap(), "Hi é");
        assert_eq!(
            native_input(&body, i0..i0 + 8).unwrap().as_str(),
            r#"{"a" :1}"#
        );
        assert!(native_input(&body, i0..i0 + 3).is_none());
    }

    #[test]
    fn string_text_and_string_arguments_are_read_as_their_text() {
        let text = r#"{"m":{"content":"Hi \u00e9","args":"{\"z\" : 1}","bad":"[1]","empty":""}}"#;
        let body = RawJson::from_text(text).unwrap();
        let at = |needle: &str| {
            let s = u32::try_from(text.find(needle).unwrap()).unwrap();
            s..s + u32::try_from(needle.len()).unwrap()
        };
        let view = ResponseView {
            blocks: vec![sarathi_protocol::BlockView {
                kind: sarathi_protocol::BlockKind::Text,
                range: at(r#""Hi \u00e9""#),
            }],
            ..ResponseView::default()
        };
        assert_eq!(native_text(&body, &view).as_deref(), Some("Hi é"));
        assert_eq!(
            native_input(&body, at(r#""{\"z\" : 1}""#))
                .unwrap()
                .as_str(),
            r#"{"z" : 1}"#
        );
        assert!(native_input(&body, at(r#""[1]""#)).is_none());
        assert_eq!(native_input(&body, at(r#""""#)).unwrap().as_str(), "{}");
        assert!(native_input(&body, 0..10_000).is_none());
    }

    #[test]
    fn wordings() {
        let tools = ["look", "edit"];
        assert_eq!(
            unknown_tool("paint", tools.into_iter(), true),
            "\"paint\" is not a tool. The tools are: look, edit — or \"done\"."
        );
        assert_eq!(
            unknown_tool("paint", tools.into_iter(), false),
            "\"paint\" is not a tool. The tools are: look, edit."
        );
        assert_eq!(skipped("interrupted"), "skipped: interrupted");
        assert!(ran_out(4).contains("(4 used)"));
        assert_eq!(
            observation(&RawJson::string("2 rows"), false),
            "observation:\n2 rows"
        );
        assert_eq!(
            observation(&RawJson::from_text(r#"{"n":2}"#).unwrap(), true),
            "observation:\nThe tool refused: {\"n\":2}"
        );
    }
}
