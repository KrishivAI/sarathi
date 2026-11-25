// SPDX-License-Identifier: Apache-2.0
//! The cassette file format: one JSONL file per run, a header line first, then one entry per
//! provider call, in call order. This module is the format only: the reader, the writer, the request match and the
//! secret scan. The record and replay decorators over a `Provider` come with the provider trait.
//!
//! Bodies are stored as text (the wire is JSON). Replay matches by position, then by fingerprint, and a mismatch names
//! the first differing top-level field of the request: it never falls back to a live call.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

/// The cassette version this crate writes. It reads this version and the one before (none exists yet).
pub const CASSETTE_VERSION: u32 = 1;

const MAGIC: &str = "sarathi.cassette";

/// First line of a cassette.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CassetteHeader {
    /// Always `sarathi.cassette`.
    pub format: String,
    /// Format version.
    pub version: u32,
    /// Model id the traffic was recorded against.
    pub model: String,
    /// Recording date, `YYYY-MM-DD`.
    pub recorded_on: String,
}

impl CassetteHeader {
    /// A header for the current version.
    #[must_use]
    pub fn new(model: &str, recorded_on: &str) -> Self {
        Self {
            format: MAGIC.into(),
            version: CASSETTE_VERSION,
            model: model.into(),
            recorded_on: recorded_on.into(),
        }
    }
}

/// One server-sent event with its arrival offset (frames, timing optional).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Frame {
    /// Milliseconds since the response began, when timing was recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset_ms: Option<u64>,
    /// The frame's bytes as text.
    pub data: String,
}

/// What the provider answered.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Response {
    /// A complete body.
    Body {
        /// HTTP status.
        status: u16,
        /// Body text.
        body: String,
    },
    /// A stream, possibly cut short (`complete: false`).
    Stream {
        /// HTTP status.
        status: u16,
        /// Frames in arrival order.
        frames: Vec<Frame>,
        /// The stream ended normally.
        complete: bool,
    },
}

/// One recorded provider call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    /// Position in the cassette, from 0.
    pub seq: u32,
    /// [`fingerprint`] of `request`.
    pub fingerprint: String,
    /// Request body as sent. Auth headers and keys are never part of it.
    pub request: String,
    /// The answer.
    pub response: Response,
}

impl Entry {
    /// An entry for `request` with its fingerprint filled in.
    #[must_use]
    pub fn new(seq: u32, request: &str, response: Response) -> Self {
        Self {
            seq,
            fingerprint: fingerprint(request.as_bytes()),
            request: request.into(),
            response,
        }
    }
}

/// A whole cassette.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cassette {
    /// Header.
    pub header: CassetteHeader,
    /// Entries in call order.
    pub entries: Vec<Entry>,
}

/// A cassette could not be read, written or matched.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CassetteError {
    /// No lines at all.
    #[error("cassette is empty: the first line must be the header")]
    Empty,
    /// The header is not valid or not a cassette.
    #[error("bad cassette header: {0}")]
    BadHeader(String),
    /// A newer version than this reader handles.
    #[error("cassette version {0} is not supported (this reader handles up to {CASSETTE_VERSION})")]
    Version(u32),
    /// An entry line does not parse; usually a write cut off in the middle.
    #[error("entry on line {line} is unreadable (truncated?): {message}")]
    BadEntry {
        /// 1-based line number.
        line: usize,
        /// Parser message.
        message: String,
    },
    /// Entries are not numbered 0, 1, 2, ... in order.
    #[error("entry on line {line} has seq {found}, expected {expected}")]
    OutOfOrder {
        /// 1-based line number.
        line: usize,
        /// Expected seq.
        expected: u32,
        /// Found seq.
        found: u32,
    },
    /// An entry's fingerprint is not that of its request.
    #[error("entry {seq}: fingerprint does not match its request")]
    BadFingerprint {
        /// The entry.
        seq: u32,
    },
    /// A secret-shaped string; the cassette is not written.
    #[error("entry {seq}: {what} looks like a secret")]
    Secret {
        /// The entry.
        seq: u32,
        /// What matched.
        what: String,
    },
    /// The request of a replayed call differs from the recorded one.
    #[error("request differs from the recorded one at `{field}`")]
    Mismatch {
        /// First differing top-level field, or `(body)` when either side is not a JSON object.
        field: String,
    },
    /// Serialising failed.
    #[error("cannot serialise: {0}")]
    Serialise(String),
}

/// A 64-bit FNV-1a hash of the bytes, as 16 hex digits. Not a security hash: it only detects a changed request, and
/// [`match_request`] does the exact comparison.
#[must_use]
pub fn fingerprint(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{h:016x}")
}

/// Parse a cassette. Reads the current version and the one before it.
///
/// # Errors
/// [`CassetteError`] naming the line or entry.
pub fn parse(text: &str) -> Result<Cassette, CassetteError> {
    let mut lines = text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty());
    let (_, first) = lines.next().ok_or(CassetteError::Empty)?;
    let header: CassetteHeader =
        serde_json::from_str(first).map_err(|e| CassetteError::BadHeader(e.to_string()))?;
    if header.format != MAGIC {
        return Err(CassetteError::BadHeader(format!(
            "format `{}`",
            header.format
        )));
    }
    if header.version == 0 || header.version > CASSETTE_VERSION {
        return Err(CassetteError::Version(header.version));
    }
    let mut entries = Vec::new();
    for (i, l) in lines {
        let line = i + 1;
        let e: Entry = serde_json::from_str(l).map_err(|e| CassetteError::BadEntry {
            line,
            message: e.to_string(),
        })?;
        let expected = u32::try_from(entries.len()).unwrap_or(u32::MAX);
        if e.seq != expected {
            return Err(CassetteError::OutOfOrder {
                line,
                expected,
                found: e.seq,
            });
        }
        if e.fingerprint != fingerprint(e.request.as_bytes()) {
            return Err(CassetteError::BadFingerprint { seq: e.seq });
        }
        entries.push(e);
    }
    Ok(Cassette { header, entries })
}

/// Write a cassette. Refuses an entry with a secret-shaped string, one out of order, or a wrong fingerprint.
///
/// # Errors
/// [`CassetteError`].
pub fn write(c: &Cassette) -> Result<String, CassetteError> {
    let ser = |e: serde_json::Error| CassetteError::Serialise(e.to_string());
    let mut out = serde_json::to_string(&c.header).map_err(ser)?;
    out.push('\n');
    for (i, e) in c.entries.iter().enumerate() {
        let expected = u32::try_from(i).unwrap_or(u32::MAX);
        if e.seq != expected {
            return Err(CassetteError::OutOfOrder {
                line: i + 2,
                expected,
                found: e.seq,
            });
        }
        if e.fingerprint != fingerprint(e.request.as_bytes()) {
            return Err(CassetteError::BadFingerprint { seq: e.seq });
        }
        let line = serde_json::to_string(e).map_err(ser)?;
        // The serialised line escapes `\n` and `\t`, which can hide a key behind a backslash letter, so the raw
        // texts are scanned too.
        let raw_hit = find_secret(&e.request).or_else(|| match &e.response {
            Response::Body { body, .. } => find_secret(body),
            Response::Stream { frames, .. } => frames.iter().find_map(|f| find_secret(&f.data)),
        });
        if let Some(what) = find_secret(&line).or(raw_hit) {
            return Err(CassetteError::Secret { seq: e.seq, what });
        }
        out.push_str(&line);
        out.push('\n');
    }
    Ok(out)
}

/// Look for a string shaped like a credential: a provider key prefix, a bearer token, or an auth header name. Returns
/// what matched. A heuristic guard, not a proof: recorded fixtures are also reviewed.
#[must_use]
pub fn find_secret(text: &str) -> Option<String> {
    let lower = text.to_ascii_lowercase();
    for name in ["authorization", "x-api-key", "api-key", "x-goog-api-key"] {
        if lower.contains(name) {
            return Some(format!("header name `{name}`"));
        }
    }
    if lower.contains("bearer ") {
        return Some("a bearer token".into());
    }
    for prefix in ["sk-", "ghp_", "gho_", "xoxb-", "akia", "aiza"] {
        let mut from = 0;
        while let Some(at) = lower[from..].find(prefix) {
            let start = from + at;
            let before_ok = start == 0 || !lower.as_bytes()[start - 1].is_ascii_alphanumeric();
            let run = lower[start + prefix.len()..]
                .bytes()
                .take_while(|b| b.is_ascii_alphanumeric() || *b == b'-' || *b == b'_')
                .count();
            if before_ok && run >= 16 {
                return Some(format!("a `{prefix}` key"));
            }
            from = start + prefix.len();
        }
    }
    None
}

/// Check a request against a recorded entry. Only equal bytes match. Otherwise the error names the first top-level field
/// (in sorted order) whose value differs, which is what a changed system prompt or tool list looks like.
///
/// # Errors
/// [`CassetteError::Mismatch`].
pub fn match_request(entry: &Entry, actual: &str) -> Result<(), CassetteError> {
    if entry.request == actual {
        return Ok(());
    }
    let parse = |s: &str| serde_json::from_str::<serde_json::Value>(s).ok();
    let (Some(serde_json::Value::Object(a)), Some(serde_json::Value::Object(b))) =
        (parse(&entry.request), parse(actual))
    else {
        return Err(CassetteError::Mismatch {
            field: "(body)".into(),
        });
    };
    let keys: BTreeSet<&String> = a.keys().chain(b.keys()).collect();
    for k in keys {
        if a.get(k) != b.get(k) {
            return Err(CassetteError::Mismatch { field: k.clone() });
        }
    }
    // Equal JSON with different bytes (key order, spacing, escapes): byte drift breaks provider-side prompt caching, so
    // it is a mismatch, not a match.
    Err(CassetteError::Mismatch {
        field: "(bytes)".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cassette() -> Cassette {
        let stream = Response::Stream {
            status: 200,
            frames: vec![
                Frame {
                    offset_ms: Some(0),
                    data: "event: message_start".into(),
                },
                Frame {
                    offset_ms: None,
                    data: "event: message_stop".into(),
                },
            ],
            complete: true,
        };
        Cassette {
            header: CassetteHeader::new("claude-x", "2026-11-24"),
            entries: vec![
                Entry::new(
                    0,
                    r#"{"system":"s","tools":[],"messages":[]}"#,
                    Response::Body {
                        status: 200,
                        body: "{}".into(),
                    },
                ),
                Entry::new(1, r#"{"system":"s","tools":[],"messages":[1]}"#, stream),
            ],
        }
    }

    #[test]
    fn write_then_parse_is_identical() {
        let c = cassette();
        let text = write(&c).unwrap();
        assert_eq!(parse(&text).unwrap(), c);
        assert_eq!(
            write(&parse(&text).unwrap()).unwrap(),
            text,
            "byte-identical re-write"
        );
    }

    #[test]
    fn a_truncated_last_entry_names_its_line() {
        let text = write(&cassette()).unwrap();
        let cut = &text[..text.len() - 20];
        assert!(matches!(
            parse(cut),
            Err(CassetteError::BadEntry { line: 3, .. })
        ));
    }

    #[test]
    fn header_rules() {
        assert_eq!(parse(""), Err(CassetteError::Empty));
        let newer =
            r#"{"format":"sarathi.cassette","version":2,"model":"m","recorded_on":"2026-11-24"}"#;
        assert_eq!(parse(newer), Err(CassetteError::Version(2)));
        let other = r#"{"format":"vcr","version":1,"model":"m","recorded_on":"d"}"#;
        assert!(matches!(parse(other), Err(CassetteError::BadHeader(_))));
        let v1 =
            r#"{"format":"sarathi.cassette","version":1,"model":"m","recorded_on":"2026-11-24"}"#;
        assert_eq!(parse(v1).unwrap().entries.len(), 0);
    }

    #[test]
    fn order_and_fingerprint_are_checked() {
        let mut c = cassette();
        c.entries[1].seq = 5;
        assert!(matches!(
            write(&c),
            Err(CassetteError::OutOfOrder {
                found: 5,
                expected: 1,
                ..
            })
        ));
        let mut c = cassette();
        c.entries[0].request.push(' ');
        assert_eq!(write(&c), Err(CassetteError::BadFingerprint { seq: 0 }));
    }

    #[test]
    fn mismatch_names_the_first_differing_field() {
        let e = &cassette().entries[0];
        assert!(match_request(e, e.request.as_str()).is_ok());
        let changed = r#"{"system":"changed","tools":[],"messages":[]}"#;
        assert_eq!(
            match_request(e, changed),
            Err(CassetteError::Mismatch {
                field: "system".into()
            })
        );
        let extra = r#"{"system":"s","tools":[],"messages":[],"temperature":0}"#;
        assert_eq!(
            match_request(e, extra),
            Err(CassetteError::Mismatch {
                field: "temperature".into()
            })
        );
        assert_eq!(
            match_request(e, "not json"),
            Err(CassetteError::Mismatch {
                field: "(body)".into()
            })
        );
        let reordered = r#"{"messages":[],"tools":[],"system":"s"}"#;
        assert_eq!(
            match_request(e, reordered),
            Err(CassetteError::Mismatch {
                field: "(bytes)".into()
            }),
            "equal JSON with different bytes is byte drift"
        );
    }

    #[test]
    fn a_key_right_after_a_newline_in_a_body_is_found() {
        let mut c = cassette();
        c.entries[0].response = Response::Body {
            status: 200,
            body: "line\nsk-abcdefghijklmnopqrstuvwx".into(),
        };
        assert!(matches!(write(&c), Err(CassetteError::Secret { .. })));
    }

    #[test]
    fn secrets_are_refused_at_write() {
        for bad in [
            r#"{"Authorization":"x"}"#,
            "x-api-key",
            "Bearer abc",
            "key sk-ant-api03-abcdefghijklmnop1234",
            "ghp_abcdefghijklmnopqrstuv",
        ] {
            let mut c = cassette();
            c.entries[0] = Entry::new(
                0,
                bad,
                Response::Body {
                    status: 200,
                    body: String::new(),
                },
            );
            assert!(
                matches!(write(&c), Err(CassetteError::Secret { seq: 0, .. })),
                "{bad}"
            );
        }
    }

    #[test]
    fn ordinary_text_is_not_a_secret() {
        assert_eq!(
            find_secret("a task: ask-the-user, then disk-usage, sk-short"),
            None
        );
        assert_eq!(find_secret(&write(&cassette()).unwrap()), None);
    }

    #[test]
    fn fingerprint_is_stable() {
        assert_eq!(fingerprint(b""), "cbf29ce484222325");
        assert_eq!(fingerprint(b"a"), "af63dc4c8601ec8c");
    }
}
