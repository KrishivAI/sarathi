// SPDX-License-Identifier: Apache-2.0
//! The session log's header, its format version and the names of its entry kinds. The store, the line format and the
//! entry bodies belong to the log crate; what is here is what every crate that reads a log agrees on.
//!
//! Versioning: a reader of format N reads N and N-1 and refuses anything newer ([`read_header`]). Within a version an
//! unknown entry kind is kept as it is and skipped, unless the entry says `"must_understand": true`
//! ([`EntryHead::readable`]), in which case the reader refuses the log. New optional kinds need no new version;
//! changing what an existing kind means, or how lines are framed, does.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::ids::{SegmentId, SessionId};
use crate::message::Seq;

/// The log format this crate writes.
pub const FORMAT_VERSION: u32 = 1;

/// The oldest format this crate reads: one before [`FORMAT_VERSION`] (or the first, 1).
pub const OLDEST_READABLE: u32 = if FORMAT_VERSION > 1 {
    FORMAT_VERSION - 1
} else {
    1
};

/// The field of an entry that, when `true`, makes a reader that does not know the entry's kind refuse the log.
pub const MUST_UNDERSTAND: &str = "must_understand";

/// The first line of a log. Read it with [`read_header`], which checks the version before the rest of the shape.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogHeader {
    /// The format version.
    pub sarathi_log: u32,
    /// The session.
    pub session: SessionId,
    /// When the session was created, RFC 3339 in UTC (`2026-11-25T08:00:00Z`).
    pub created: Box<str>,
    /// What wrote the log (`sarathi-log 0.1.0`).
    pub writer: Box<str>,
    /// The session and entry this one was forked from, if it was.
    pub forked_from: Option<ForkPoint>,
}

/// Where a fork started: the parent, its last entry the fork keeps, and the hash of the parent's entries up to it.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForkPoint {
    /// The parent session.
    pub session: SessionId,
    /// The parent's last entry the fork keeps.
    pub seq: Seq,
    /// SHA-256 of the parent's entry lines 1 to `seq` as stored, so a reader can tell the parent was not changed.
    pub sha256: Sha256,
}

impl ForkPoint {
    /// A fork point.
    pub fn new(session: SessionId, seq: Seq, sha256: Sha256) -> Self {
        ForkPoint {
            session,
            seq,
            sha256,
        }
    }
}

/// A SHA-256 digest, written as 64 lower-case hex characters. This crate does not compute it.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Sha256(pub [u8; 32]);

impl fmt::Display for Sha256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for Sha256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sha256({self})")
    }
}

impl std::str::FromStr for Sha256 {
    type Err = BadSha256;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let b = s.as_bytes();
        if b.len() != 64 {
            return Err(BadSha256);
        }
        let nibble = |c: u8| match c {
            b'0'..=b'9' => Ok(c - b'0'),
            b'a'..=b'f' => Ok(c - b'a' + 10),
            _ => Err(BadSha256),
        };
        let mut out = [0u8; 32];
        for (i, pair) in b.as_chunks::<2>().0.iter().enumerate() {
            out[i] = (nibble(pair[0])? << 4) | nibble(pair[1])?;
        }
        Ok(Sha256(out))
    }
}

/// A text that is not 64 lower-case hex characters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("not a SHA-256 digest (64 lower-case hex characters)")]
pub struct BadSha256;

impl Serialize for Sha256 {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for Sha256 {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        String::deserialize(d)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

/// A log this crate will not read.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum VersionError {
    /// Written by a newer version; never guessed at.
    #[error("the log is format {found}, newer than this reader's {supported}")]
    NewerFormat {
        /// The log's version.
        found: u64,
        /// The newest this reader reads.
        supported: u32,
    },
    /// Older than the oldest format this reader keeps.
    #[error("the log is format {found}, older than this reader's oldest, {oldest}")]
    OlderFormat {
        /// The log's version.
        found: u64,
        /// The oldest this reader reads.
        oldest: u32,
    },
}

/// Why a header line was not read.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum HeaderError {
    /// The version is one this crate does not read.
    #[error(transparent)]
    Version(#[from] VersionError),
    /// The line is not a header of a readable version.
    #[error("not a log header: {0}")]
    Malformed(Box<str>),
}

impl LogHeader {
    /// A header of the current format, not forked.
    pub fn new(
        session: SessionId,
        created: impl Into<Box<str>>,
        writer: impl Into<Box<str>>,
    ) -> Self {
        LogHeader {
            sarathi_log: FORMAT_VERSION,
            session,
            created: created.into(),
            writer: writer.into(),
            forked_from: None,
        }
    }

    /// The same header, forked from `at`.
    pub fn forked(mut self, at: ForkPoint) -> Self {
        self.forked_from = Some(at);
        self
    }
}

/// Reads a log's first line: the version first (any other field may have changed shape in a newer format), then the
/// rest of the header.
///
/// # Errors
/// [`HeaderError::Version`] for a version outside [`OLDEST_READABLE`]..=[`FORMAT_VERSION`]; [`HeaderError::Malformed`]
/// for a line that is not a header.
pub fn read_header(line: &str) -> Result<LogHeader, HeaderError> {
    #[derive(Deserialize)]
    struct Version {
        sarathi_log: u64,
    }
    let malformed = |e: serde_json::Error| HeaderError::Malformed(e.to_string().into());
    let v: Version = serde_json::from_str(line).map_err(malformed)?;
    check_version(v.sarathi_log)?;
    serde_json::from_str(line).map_err(malformed)
}

/// Whether this crate reads format `found`.
///
/// # Errors
/// The format is newer than [`FORMAT_VERSION`] or older than [`OLDEST_READABLE`].
pub fn check_version(found: u64) -> Result<(), VersionError> {
    if found > u64::from(FORMAT_VERSION) {
        Err(VersionError::NewerFormat {
            found,
            supported: FORMAT_VERSION,
        })
    } else if found < u64::from(OLDEST_READABLE) {
        Err(VersionError::OlderFormat {
            found,
            oldest: OLDEST_READABLE,
        })
    } else {
        Ok(())
    }
}

/// The fields every entry line starts with; a reader parses this from a line and ignores the body.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntryHead {
    /// The entry's position.
    pub seq: Seq,
    /// Its segment.
    pub seg: SegmentId,
    /// Its kind.
    pub kind: EntryKind,
    /// Refuse the log if the kind is unknown ([`MUST_UNDERSTAND`]).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub must_understand: bool,
}

/// An entry of a kind this crate does not know, marked [`MUST_UNDERSTAND`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("entry {seq} is of kind `{kind}`, which this reader does not know and must understand")]
pub struct UnknownRequired {
    /// The entry.
    pub seq: Seq,
    /// Its kind.
    pub kind: Box<str>,
}

impl EntryHead {
    /// An entry head.
    pub fn new(seq: Seq, seg: SegmentId, kind: EntryKind) -> Self {
        EntryHead {
            seq,
            seg,
            kind,
            must_understand: false,
        }
    }

    /// Whether a reader may go on past this entry: a known kind, or an unknown one it may skip.
    ///
    /// # Errors
    /// The kind is unknown and the entry is marked [`MUST_UNDERSTAND`].
    pub fn readable(&self) -> Result<(), UnknownRequired> {
        if self.must_understand && !self.kind.is_known() {
            Err(UnknownRequired {
                seq: self.seq,
                kind: self.kind.as_str().into(),
            })
        } else {
            Ok(())
        }
    }
}

/// The kind of a log entry (its `kind` field). Build one from a name with [`EntryKind::from_name`]:
/// [`EntryKind::Other`] then only ever holds a name that is not a known kind's.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EntryKind {
    /// The provider protocol with its `system` and `tools` bytes.
    SessionPrefix,
    /// A user (or steer) message, rendered once.
    MessageUser,
    /// How a request was built (the seqs of its slices and its parameters) and the hash of the body sent.
    ModelRequest,
    /// The answer as received, or as assembled from its stream, with the stream's text.
    ModelResponse,
    /// A failed model call.
    ModelFailure,
    /// The assistant message: which response's content it is.
    MessageAssistant,
    /// A permission ask.
    ToolAsk,
    /// The answer to a permission ask.
    ToolDecision,
    /// A tool call is about to run (written before a tool with side effects runs).
    ToolStart,
    /// A tool's result, rendered once.
    ToolResult,
    /// A new segment.
    SegmentStart,
    /// A turn ended.
    TurnEnd,
    /// Host metadata the run loop never reads.
    Note,
    /// A kind this crate does not know.
    Other(Box<str>),
}

impl EntryKind {
    /// Every kind this crate knows.
    pub const ALL: &'static [EntryKind] = &[
        EntryKind::SessionPrefix,
        EntryKind::MessageUser,
        EntryKind::ModelRequest,
        EntryKind::ModelResponse,
        EntryKind::ModelFailure,
        EntryKind::MessageAssistant,
        EntryKind::ToolAsk,
        EntryKind::ToolDecision,
        EntryKind::ToolStart,
        EntryKind::ToolResult,
        EntryKind::SegmentStart,
        EntryKind::TurnEnd,
        EntryKind::Note,
    ];

    /// The name in the log (`message.user`, ...).
    pub fn as_str(&self) -> &str {
        match self {
            EntryKind::SessionPrefix => "session.prefix",
            EntryKind::MessageUser => "message.user",
            EntryKind::ModelRequest => "model.request",
            EntryKind::ModelResponse => "model.response",
            EntryKind::ModelFailure => "model.failure",
            EntryKind::MessageAssistant => "message.assistant",
            EntryKind::ToolAsk => "tool.ask",
            EntryKind::ToolDecision => "tool.decision",
            EntryKind::ToolStart => "tool.start",
            EntryKind::ToolResult => "tool.result",
            EntryKind::SegmentStart => "segment.start",
            EntryKind::TurnEnd => "turn.end",
            EntryKind::Note => "note",
            EntryKind::Other(name) => name,
        }
    }

    /// The kind called `name`.
    pub fn from_name(name: &str) -> EntryKind {
        EntryKind::ALL
            .iter()
            .find(|k| k.as_str() == name)
            .map_or_else(|| EntryKind::Other(name.into()), Clone::clone)
    }

    /// Whether this crate knows the kind.
    pub fn is_known(&self) -> bool {
        !matches!(self, EntryKind::Other(_))
    }
}

impl fmt::Display for EntryKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for EntryKind {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for EntryKind {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Ok(EntryKind::from_name(&String::deserialize(d)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST: &str = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";

    #[test]
    fn the_header_reads_what_it_writes() {
        let h = LogHeader::new(SessionId(1), "2026-11-25T08:00:00Z", "sarathi-log 0.1.0");
        let json = serde_json::to_string(&h).unwrap();
        assert_eq!(
            json,
            r#"{"sarathi_log":1,"session":"00000000-0000-0000-0000-000000000001","created":"2026-11-25T08:00:00Z","writer":"sarathi-log 0.1.0","forked_from":null}"#
        );
        assert_eq!(read_header(&json).unwrap(), h);
        let forked = h.forked(ForkPoint::new(
            SessionId(2),
            Seq(40),
            DIGEST.parse().unwrap(),
        ));
        let json = serde_json::to_string(&forked).unwrap();
        assert!(
            json.ends_with(&format!(
                r#""forked_from":{{"session":"00000000-0000-0000-0000-000000000002","seq":40,"sha256":"{DIGEST}"}}}}"#
            )),
            "{json}"
        );
        assert_eq!(read_header(&json).unwrap(), forked);
    }

    #[test]
    fn a_digest_is_64_lower_case_hex_characters() {
        let d: Sha256 = DIGEST.parse().unwrap();
        assert_eq!(d.to_string(), DIGEST);
        assert_eq!(d.0[1], 0x11);
        for bad in [
            "",
            &DIGEST[..63],
            &DIGEST.to_uppercase(),
            &format!("{DIGEST}0"),
        ] {
            assert!(bad.parse::<Sha256>().is_err(), "{bad}");
        }
    }

    #[test]
    fn the_version_is_checked_before_the_rest_of_the_header() {
        // Tripwire: these literals assume format 1; a new format updates them on purpose.
        assert_eq!(FORMAT_VERSION, 1);
        assert!(check_version(1).is_ok());
        let changed = r#"{"sarathi_log":2,"session":"x","created_at":"later"}"#;
        assert_eq!(
            read_header(changed),
            Err(HeaderError::Version(VersionError::NewerFormat {
                found: 2,
                supported: 1
            }))
        );
        let huge = r#"{"sarathi_log":5000000000}"#;
        assert!(matches!(
            read_header(huge),
            Err(HeaderError::Version(VersionError::NewerFormat {
                found: 5_000_000_000,
                ..
            }))
        ));
        assert_eq!(
            check_version(0),
            Err(VersionError::OlderFormat {
                found: 0,
                oldest: 1
            })
        );
        assert!(matches!(read_header("[]"), Err(HeaderError::Malformed(_))));
        assert!(matches!(
            read_header(r#"{"sarathi_log":1}"#),
            Err(HeaderError::Malformed(_))
        ));
    }

    #[test]
    fn entry_kinds_have_stable_unique_names_and_unknown_ones_are_kept() {
        let mut names = std::collections::BTreeSet::new();
        for kind in EntryKind::ALL {
            let name = kind.as_str();
            assert!(!name.is_empty());
            assert!(names.insert(name), "{name} twice");
            assert_eq!(EntryKind::from_name(name), *kind);
            assert!(kind.is_known());
            let json = serde_json::to_string(kind).unwrap();
            assert_eq!(json, format!("\"{name}\""));
            assert_eq!(serde_json::from_str::<EntryKind>(&json).unwrap(), *kind);
        }
        let other = EntryKind::from_name("tool.progress");
        assert_eq!(other, EntryKind::Other("tool.progress".into()));
        assert!(!other.is_known());
        assert_eq!(serde_json::to_string(&other).unwrap(), r#""tool.progress""#);
    }

    #[test]
    fn must_understand_names_the_field_entry_head_writes() {
        let mut head = EntryHead::new(Seq(1), SegmentId(0), EntryKind::Other("x.y".into()));
        head.must_understand = true;
        let v = serde_json::to_value(&head).unwrap();
        assert_eq!(v.get(MUST_UNDERSTAND), Some(&serde_json::Value::Bool(true)));
    }

    #[test]
    fn an_unknown_entry_is_skipped_unless_it_must_be_understood() {
        let line = r#"{"seq":5,"seg":0,"kind":"tool.progress","body":{"x":1}}"#;
        let head: EntryHead = serde_json::from_str(line).unwrap();
        assert_eq!(head.kind, EntryKind::Other("tool.progress".into()));
        assert!(head.readable().is_ok());
        let line = r#"{"seq":5,"seg":0,"kind":"tool.progress","must_understand":true}"#;
        let head: EntryHead = serde_json::from_str(line).unwrap();
        let e = head.readable().unwrap_err();
        assert_eq!(e.seq, Seq(5));
        assert_eq!(&*e.kind, "tool.progress");
        let known = r#"{"seq":6,"seg":1,"kind":"note","must_understand":true}"#;
        assert!(
            serde_json::from_str::<EntryHead>(known)
                .unwrap()
                .readable()
                .is_ok()
        );
        let plain = EntryHead::new(Seq(2), SegmentId(0), EntryKind::MessageUser);
        assert_eq!(
            serde_json::to_string(&plain).unwrap(),
            r#"{"seq":2,"seg":0,"kind":"message.user"}"#
        );
    }
}
