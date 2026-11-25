// SPDX-License-Identifier: Apache-2.0
//! Reading a Messages answer: locate the content and its blocks by byte range, read the few scalars the core needs,
//! never produce bytes that are sent.

use std::fmt;
use std::ops::Range;

use bytes::Bytes;
use sarathi_protocol::{
    BlockKind, BlockView, MessageSlice, RawJson, ResponseView, Role, Seq, Usage,
};
use serde::Deserialize;
use serde::de::{Deserializer, MapAccess, Visitor};
use serde_json::value::RawValue;

use super::NativeError;

/// The `anthropic-ratelimit-*` and `retry-after` headers of an answer, as sent.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RateLimitHeaders(pub Vec<(String, String)>);

impl RateLimitHeaders {
    /// Whether a header name is one this keeps.
    pub fn keeps(name: &str) -> bool {
        let n = name.to_ascii_lowercase();
        n.starts_with("anthropic-ratelimit-") || n == "retry-after"
    }

    /// The value of `name`, if sent (case ignored).
    pub fn get(&self, name: &str) -> Option<&str> {
        self.0
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

/// A Messages answer: the body as received (or as assembled from a stream), where its content is, and the view.
#[derive(Clone, PartialEq, Eq)]
pub struct MessagesResponse {
    /// The body exactly as received (non-streaming), or the assembled message (streaming).
    pub raw: Bytes,
    /// The byte range of the `content` array in `raw`: the assistant slice is cut from here.
    pub content: Range<u32>,
    /// The shallow view; its ranges point into `raw`.
    pub view: ResponseView,
    /// The SSE bytes as received, for a streamed answer (a session log stores them so a turn can be re-assembled).
    pub sse: Option<Bytes>,
    /// The `request-id` header.
    pub request_id: Option<String>,
    /// The rate-limit headers.
    pub rate_limits: RateLimitHeaders,
}

/// Never prints the body, which holds user content.
impl fmt::Debug for MessagesResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MessagesResponse")
            .field("raw", &format_args!("{} bytes", self.raw.len()))
            .field("content", &self.content)
            .field("view", &self.view)
            .field("sse", &self.sse.as_ref().map(Bytes::len))
            .field("request_id", &self.request_id)
            .finish_non_exhaustive()
    }
}

impl MessagesResponse {
    /// The `content` array's bytes, exactly as received.
    pub fn content_bytes(&self) -> &[u8] {
        self.slice(&self.content)
    }

    /// The bytes of `range` (a block or a tool input of the view); empty if the range is not inside the body.
    pub fn slice(&self, range: &Range<u32>) -> &[u8] {
        let (s, e) = (range.start as usize, range.end as usize);
        self.raw.get(s..e).unwrap_or_default()
    }

    /// The assistant message to store and re-send: `{"role":"assistant","content":` + the content bytes + `}`.
    ///
    /// # Errors
    /// Never for a body that [`parse`] accepted; the slice is validated once like every `RawJson`.
    pub fn assistant_slice(&self, seq: Seq) -> Result<MessageSlice, NativeError> {
        let content = RawJson::from_slice(self.content_bytes())
            .map_err(|e| NativeError::Malformed(format!("content: {e}")))?;
        MessageSlice::new(seq, Role::Assistant, &content)
            .map_err(|e| NativeError::Malformed(e.to_string()))
    }

    /// Whether the model refused (`stop_reason: "refusal"`).
    pub fn is_refusal(&self) -> bool {
        self.view.stop_reason.as_deref() == Some("refusal")
    }
}

/// The members of a JSON object, in their order, each value as its text. Keys are decoded.
pub(crate) fn entries(object: &str) -> Result<Vec<(String, Box<RawValue>)>, serde_json::Error> {
    struct Entries(Vec<(String, Box<RawValue>)>);
    impl<'de> Deserialize<'de> for Entries {
        fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            struct V;
            impl<'de> Visitor<'de> for V {
                type Value = Entries;
                fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.write_str("a JSON object")
                }
                fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Entries, A::Error> {
                    let mut out = Vec::new();
                    while let Some((k, v)) = map.next_entry::<String, Box<RawValue>>()? {
                        out.push((k, v));
                    }
                    Ok(Entries(out))
                }
            }
            d.deserialize_map(V)
        }
    }
    serde_json::from_str::<Entries>(object).map(|e| e.0)
}

#[derive(Deserialize)]
struct Top<'a> {
    #[serde(rename = "type", default)]
    kind: Option<String>,
    #[serde(borrow)]
    content: &'a RawValue,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default, borrow)]
    stop_details: Option<&'a RawValue>,
    #[serde(default)]
    usage: Option<WireUsage>,
}

/// `usage` as sent: the counts, and with server compaction the `iterations` they leave out.
#[derive(Deserialize)]
struct WireUsage {
    #[serde(flatten)]
    counts: Usage,
    #[serde(default, deserialize_with = "lenient_iterations")]
    iterations: Option<Vec<Iteration>>,
}

/// An iteration in a shape this does not read is left out (the others still count), never a reason to refuse the
/// answer (a logged answer is read again on resume).
fn lenient_iterations<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Vec<Iteration>>, D::Error> {
    Ok(match serde_json::Value::deserialize(d)? {
        serde_json::Value::Array(items) => Some(
            items
                .into_iter()
                .filter_map(|i| serde_json::from_value(i).ok())
                .collect(),
        ),
        _ => None,
    })
}

/// One sampling iteration of `usage.iterations`. Missing and `null` counts are 0.
#[derive(Deserialize)]
struct Iteration {
    #[serde(rename = "type", default)]
    kind: Option<String>,
    #[serde(flatten)]
    counts: Usage,
}

impl WireUsage {
    /// The billed counts. The top-level counts leave out compaction iterations (an answer that is only a compaction
    /// reports 0), and the billed total is the sum over all iterations, so each compaction iteration is added to
    /// them. Other iterations are already in the top-level counts.
    fn billed(self) -> Usage {
        let mut u = self.counts;
        for it in self.iterations.unwrap_or_default() {
            if it.kind.as_deref() == Some("compaction") {
                u.input_tokens = u.input_tokens.saturating_add(it.counts.input_tokens);
                u.output_tokens = u.output_tokens.saturating_add(it.counts.output_tokens);
                u.cache_creation_input_tokens = u
                    .cache_creation_input_tokens
                    .saturating_add(it.counts.cache_creation_input_tokens);
                u.cache_read_input_tokens = u
                    .cache_read_input_tokens
                    .saturating_add(it.counts.cache_read_input_tokens);
            }
        }
        u
    }
}

#[derive(Deserialize)]
struct Head<'a> {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default, borrow)]
    input: Option<&'a RawValue>,
}

/// Reads `raw`, a Messages answer body, into a [`MessagesResponse`] with no request id or headers. The bytes are
/// kept as given; parsing only locates ranges and reads scalars.
///
/// # Errors
/// The body is over 4 GiB, not UTF-8, not a JSON object with a `content` array, of a `type` other than `message`, or
/// holds a `tool_use` block without `id`, `name` or `input`.
pub fn parse(raw: Bytes) -> Result<MessagesResponse, NativeError> {
    if u32::try_from(raw.len()).is_err() {
        return Err(NativeError::Malformed(
            "the body is larger than 4 GiB".into(),
        ));
    }
    let text =
        std::str::from_utf8(&raw).map_err(|e| NativeError::Malformed(format!("not UTF-8: {e}")))?;
    let top: Top<'_> = serde_json::from_str(text)
        .map_err(|e| NativeError::Malformed(format!("not a message: {e}")))?;
    if let Some(k) = top.kind.as_deref().filter(|k| *k != "message") {
        return Err(NativeError::Malformed(format!(
            "an answer of type `{k}`, not `message`"
        )));
    }
    let content = range(text, top.content.get())?;
    let items: Vec<&RawValue> = serde_json::from_str(top.content.get())
        .map_err(|e| NativeError::Malformed(format!("`content` is not an array: {e}")))?;
    let mut blocks = Vec::with_capacity(items.len());
    for item in items {
        let head: Head<'_> = serde_json::from_str(item.get())
            .map_err(|e| NativeError::Malformed(format!("a content block without a type: {e}")))?;
        blocks.push(BlockView {
            kind: kind(text, head)?,
            range: range(text, item.get())?,
        });
    }
    let view = ResponseView {
        id: top.id.map(Into::into),
        model: top.model.map(Into::into),
        stop_reason: top.stop_reason.map(Into::into),
        stop_details: top
            .stop_details
            .map(|d| RawJson::from_text(d.get()))
            .transpose()
            .map_err(|e| NativeError::Malformed(e.to_string()))?,
        usage: top.usage.map(WireUsage::billed).unwrap_or_default(),
        blocks,
    };
    Ok(MessagesResponse {
        content,
        view,
        sse: None,
        request_id: None,
        rate_limits: RateLimitHeaders::default(),
        raw,
    })
}

fn kind(base: &str, head: Head<'_>) -> Result<BlockKind, NativeError> {
    Ok(match head.kind.as_str() {
        "text" => BlockKind::Text,
        "thinking" => BlockKind::Thinking,
        "redacted_thinking" => BlockKind::RedactedThinking,
        "compaction" => BlockKind::Compaction,
        "tool_use" => {
            let (Some(id), Some(name), Some(input)) = (head.id, head.name, head.input) else {
                return Err(NativeError::Malformed(
                    "a `tool_use` block needs `id`, `name` and `input`".into(),
                ));
            };
            BlockKind::ToolUse {
                id: id.into(),
                name: name.into(),
                input: range(base, input.get())?,
            }
        }
        "server_tool_use" | "mcp_tool_use" => match (head.id, head.name) {
            (Some(id), Some(name)) => BlockKind::ServerToolUse {
                id: id.into(),
                name: name.into(),
            },
            _ => BlockKind::Opaque {
                block_type: head.kind.into(),
            },
        },
        t if t.ends_with("_tool_result") && t != "tool_result" => BlockKind::ServerToolResult,
        _ => BlockKind::Opaque {
            block_type: head.kind.into(),
        },
    })
}

/// Where `part`, a sub-slice of `base`, lies in `base`.
pub(crate) fn range(base: &str, part: &str) -> Result<Range<u32>, NativeError> {
    let start = part
        .as_ptr()
        .addr()
        .checked_sub(base.as_ptr().addr())
        .filter(|s| s + part.len() <= base.len())
        .ok_or_else(|| NativeError::Malformed("a value outside the body".into()))?;
    let to_u32 = |n: usize| {
        u32::try_from(n).map_err(|_| NativeError::Malformed("the body is larger than 4 GiB".into()))
    };
    Ok(to_u32(start)?..to_u32(start + part.len())?)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BODY: &str = concat!(
        r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5-5","content":"#,
        r#"[{"type":"thinking","thinking":"","signature":"sig"},"#,
        r#"{"type":"tool_use","id":"toolu_1","name":"edit","input":{"z":1e5,"a":"\u00e9"}},"#,
        r#"{"type":"server_tool_use","id":"srv_1","name":"web_search","input":{}},"#,
        r#"{"type":"web_search_tool_result","tool_use_id":"srv_1","content":[]},"#,
        r#"{"type":"future_block","x":[{"y":null}]}],"#,
        r#""stop_reason":"tool_use","stop_details":null,"usage":{"input_tokens":5,"output_tokens":7,"cache_read_input_tokens":2}}"#
    );

    #[test]
    fn locates_content_and_blocks_without_changing_bytes() {
        let r = parse(Bytes::from_static(BODY.as_bytes())).unwrap();
        assert_eq!(r.raw.as_ref(), BODY.as_bytes());
        let content = std::str::from_utf8(r.content_bytes()).unwrap();
        assert!(
            content.starts_with(r#"[{"type":"thinking""#) && content.ends_with(r#"[{"y":null}]}]"#)
        );
        let kinds: Vec<_> = r.view.blocks.iter().map(|b| b.kind.clone()).collect();
        assert_eq!(kinds[0], BlockKind::Thinking);
        assert!(
            matches!(&kinds[1], BlockKind::ToolUse { id, name, .. } if &**id == "toolu_1" && &**name == "edit")
        );
        assert!(
            matches!(&kinds[2], BlockKind::ServerToolUse { name, .. } if &**name == "web_search")
        );
        assert_eq!(kinds[3], BlockKind::ServerToolResult);
        assert_eq!(
            kinds[4],
            BlockKind::Opaque {
                block_type: "future_block".into()
            }
        );
        let (_, _, input) = r.view.tool_uses().next().unwrap();
        assert_eq!(r.slice(&input), br#"{"z":1e5,"a":"\u00e9"}"#);
        assert_eq!(
            r.slice(&r.view.blocks[0].range),
            br#"{"type":"thinking","thinking":"","signature":"sig"}"#
        );
        assert_eq!(r.view.stop_reason.as_deref(), Some("tool_use"));
        assert_eq!(r.view.stop_details, None);
        assert_eq!(r.view.usage.cache_read_input_tokens, 2);
        assert_eq!(r.view.id.as_deref(), Some("msg_1"));
        assert!(!r.is_refusal());
        assert!(format!("{r:?}").contains("bytes"));
    }

    #[test]
    fn the_assistant_slice_is_the_content_bytes() {
        let r = parse(Bytes::from_static(BODY.as_bytes())).unwrap();
        let s = r.assistant_slice(Seq(3)).unwrap();
        let content = std::str::from_utf8(r.content_bytes()).unwrap();
        assert_eq!(
            s.bytes.as_str(),
            format!(r#"{{"role":"assistant","content":{content}}}"#)
        );
    }

    #[test]
    fn refuses_what_is_not_a_message() {
        let bad = |s: &'static str| parse(Bytes::from_static(s.as_bytes())).unwrap_err();
        assert!(matches!(
            bad(r#"{"type":"error","content":[]}"#),
            NativeError::Malformed(_)
        ));
        assert!(matches!(
            bad(r#"{"content":{}}"#),
            NativeError::Malformed(_)
        ));
        assert!(matches!(
            bad(r#"{"stop_reason":"x"}"#),
            NativeError::Malformed(_)
        ));
        assert!(matches!(
            bad(r#"{"content":[{"text":"no type"}]}"#),
            NativeError::Malformed(_)
        ));
        assert!(matches!(
            bad(r#"{"content":[{"type":"tool_use","id":"t"}]}"#),
            NativeError::Malformed(_)
        ));
        assert!(matches!(
            parse(Bytes::from_static(b"\xff")),
            Err(NativeError::Malformed(_))
        ));
    }

    #[test]
    fn refusal_and_stop_details() {
        let r = parse(Bytes::from_static(
            br#"{"content":[],"stop_reason":"refusal","stop_details":{"category":"cyber"}}"#,
        ))
        .unwrap();
        assert!(r.is_refusal());
        assert_eq!(
            r.view.stop_details.unwrap().as_str(),
            r#"{"category":"cyber"}"#
        );
        assert_eq!(r.view.usage, Usage::default());
    }

    #[test]
    fn a_compaction_only_answer_is_billed_from_its_iteration() {
        let r = parse(Bytes::from_static(
            br#"{"content":[{"type":"compaction","content":"s"}],"stop_reason":"compaction","usage":{"input_tokens":0,"output_tokens":0,"iterations":[{"type":"compaction","input_tokens":144,"output_tokens":276}]}}"#,
        ))
        .unwrap();
        assert_eq!(
            r.view.usage,
            Usage {
                input_tokens: 144,
                output_tokens: 276,
                ..Usage::default()
            }
        );
    }

    #[test]
    fn a_compaction_iteration_is_added_to_the_top_level_counts_once() {
        let r = parse(Bytes::from_static(
            br#"{"content":[],"stop_reason":"end_turn","usage":{"input_tokens":23000,"output_tokens":1000,"cache_read_input_tokens":7,"iterations":[{"type":"compaction","input_tokens":180000,"output_tokens":3500},{"type":"message","input_tokens":23000,"output_tokens":1000},{"type":"future","input_tokens":null}]}}"#,
        ))
        .unwrap();
        assert_eq!(
            r.view.usage,
            Usage {
                input_tokens: 203_000,
                output_tokens: 4_500,
                cache_read_input_tokens: 7,
                ..Usage::default()
            }
        );
    }

    #[test]
    fn iterations_in_an_unknown_shape_are_left_out() {
        for usage in [
            r#"{"input_tokens":5,"iterations":{"type":"compaction"}}"#,
            r#"{"input_tokens":5,"iterations":[{"type":"compaction","input_tokens":"many"}]}"#,
            r#"{"input_tokens":5,"iterations":[7]}"#,
            r#"{"input_tokens":5,"iterations":null}"#,
        ] {
            let body = format!(r#"{{"content":[],"usage":{usage}}}"#);
            let r = parse(Bytes::from(body)).unwrap();
            assert_eq!(r.view.usage.input_tokens, 5, "{usage}");
        }
    }

    #[test]
    fn a_bad_iteration_does_not_drop_the_good_ones() {
        let r = parse(Bytes::from_static(
            br#"{"content":[],"usage":{"input_tokens":1,"iterations":[{"type":"compaction","input_tokens":"x"},{"type":"compaction","input_tokens":10}]}}"#,
        ))
        .unwrap();
        assert_eq!(r.view.usage.input_tokens, 11);
    }

    #[test]
    fn a_compaction_iteration_with_null_or_missing_counts_adds_what_it_has() {
        let r = parse(Bytes::from_static(
            br#"{"content":[],"usage":{"input_tokens":1,"iterations":[{"type":"compaction","input_tokens":null,"output_tokens":3,"cache_read_input_tokens":4}]}}"#,
        ))
        .unwrap();
        assert_eq!(
            r.view.usage,
            Usage {
                input_tokens: 1,
                output_tokens: 3,
                cache_read_input_tokens: 4,
                ..Usage::default()
            }
        );
    }

    #[test]
    fn a_server_tool_use_without_id_is_opaque() {
        let r = parse(Bytes::from_static(
            br#"{"content":[{"type":"mcp_tool_use"}]}"#,
        ))
        .unwrap();
        assert_eq!(
            r.view.blocks[0].kind,
            BlockKind::Opaque {
                block_type: "mcp_tool_use".into()
            }
        );
    }

    #[test]
    fn entries_keep_order_and_text() {
        let e = entries(r#"{"z":1e5,"a":"\/"}"#).unwrap();
        let got: Vec<_> = e.iter().map(|(k, v)| (k.as_str(), v.get())).collect();
        assert_eq!(got, vec![("z", "1e5"), ("a", r#""\/""#)]);
        assert!(entries("[1]").is_err());
    }

    #[test]
    fn rate_limit_headers() {
        assert!(RateLimitHeaders::keeps(
            "Anthropic-RateLimit-Requests-Remaining"
        ));
        assert!(RateLimitHeaders::keeps("retry-after"));
        assert!(!RateLimitHeaders::keeps("x-api-key"));
        let h = RateLimitHeaders(vec![("Retry-After".into(), "3".into())]);
        assert_eq!(h.get("retry-after"), Some("3"));
        assert_eq!(h.get("x"), None);
    }
}
