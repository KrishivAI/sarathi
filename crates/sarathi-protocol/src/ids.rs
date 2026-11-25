// SPDX-License-Identifier: Apache-2.0
//! Identifiers: sessions, segments, tool calls and turns.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::message::Seq;

/// A session: 128 bits, written as a lower-case hyphenated UUID (`0193a1b2-...`). The store makes them (UUID v7, so
/// they sort by creation time); this crate only reads and writes the text form. Upper-case hex is read too; the
/// written form is always lower-case, so it is the one name of the session's directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SessionId(pub u128);

/// A text that is not a hyphenated UUID; holds at most its first 64 characters.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("not a session id: {0:?}")]
pub struct BadSessionId(pub Box<str>);

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let v = self.0;
        write!(
            f,
            "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
            v >> 96,
            (v >> 80) & 0xffff,
            (v >> 64) & 0xffff,
            (v >> 48) & 0xffff,
            v & 0xffff_ffff_ffff
        )
    }
}

impl FromStr for SessionId {
    type Err = BadSessionId;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let bad = || BadSessionId(s.chars().take(64).collect::<String>().into());
        if s.len() != 36 {
            return Err(bad());
        }
        let mut v: u128 = 0;
        for (i, c) in s.bytes().enumerate() {
            if matches!(i, 8 | 13 | 18 | 23) {
                if c != b'-' {
                    return Err(bad());
                }
                continue;
            }
            let nibble = match c {
                b'0'..=b'9' => c - b'0',
                b'a'..=b'f' => c - b'a' + 10,
                b'A'..=b'F' => c - b'A' + 10,
                _ => return Err(bad()),
            };
            v = (v << 4) | u128::from(nibble);
        }
        Ok(SessionId(v))
    }
}

impl Serialize for SessionId {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for SessionId {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        String::deserialize(d)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

/// A segment of a session: 0 at the start, one more for each new segment (compaction, a protocol switch, a fork).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct SegmentId(pub u32);

impl fmt::Display for SegmentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// A tool call: the provider's tool-use id (`toolu_...`, `call_...`), or one the core makes (`s<seq>-<n>`) for a
/// model without native tool calls. Unique within a session.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CallId(pub Box<str>);

impl CallId {
    /// The id the core gives the `n`-th call parsed from the message at `seq`.
    pub fn made(seq: Seq, n: u32) -> CallId {
        CallId(format!("s{}-{n}", seq.0).into())
    }

    /// The id as text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for CallId {
    fn from(s: &str) -> CallId {
        CallId(s.into())
    }
}

impl From<String> for CallId {
    fn from(s: String) -> CallId {
        CallId(s.into())
    }
}

impl From<Box<str>> for CallId {
    fn from(s: Box<str>) -> CallId {
        CallId(s)
    }
}

impl fmt::Display for CallId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A turn: the seq of the entry that started it (the user message or the steer).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TurnId(pub Seq);

impl fmt::Display for TurnId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_session_id_reads_what_it_writes() {
        let id = SessionId(0x0193_a1b2_c3d4_7e5f_8a9b_0c1d_2e3f_4a5b);
        let text = id.to_string();
        assert_eq!(text, "0193a1b2-c3d4-7e5f-8a9b-0c1d2e3f4a5b");
        assert_eq!(text.parse::<SessionId>().unwrap(), id);
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, format!("\"{text}\""));
        assert_eq!(serde_json::from_str::<SessionId>(&json).unwrap(), id);
        assert_eq!(
            SessionId(0).to_string(),
            "00000000-0000-0000-0000-000000000000"
        );
        assert_eq!(
            SessionId(u128::MAX).to_string(),
            "ffffffff-ffff-ffff-ffff-ffffffffffff"
        );
        let upper: SessionId = "0193A1B2-C3D4-7E5F-8A9B-0C1D2E3F4A5B".parse().unwrap();
        assert_eq!(upper, id);
        assert_eq!(upper.to_string(), text, "always written lower-case");
        let escaped: SessionId =
            serde_json::from_str(r#""\u0030193a1b2-c3d4-7e5f-8a9b-0c1d2e3f4a5b""#).unwrap();
        assert_eq!(escaped, id, "an escaped JSON string reads the same");
    }

    #[test]
    fn other_texts_are_not_session_ids() {
        for bad in [
            "",
            "0193a1b2c3d47e5f8a9b0c1d2e3f4a5b",
            "0193a1b2-c3d4-7e5f-8a9b-0c1d2e3f4a5",
            "0193a1b2-c3d4-7e5f-8a9b-0c1d2e3f4a5bc",
            "0193a1b2-c3d4-7e5f-8a9b_0c1d2e3f4a5b",
            "0193a1b2-c3d4-7e5f-8a9b-0c1d2e3f4a5g",
            "+193a1b2-c3d4-7e5f-8a9b-0c1d2e3f4a5b",
            "0193a1b2-c3d4-7e5f-8a9b-0c1d2e3f4a\u{e9}",
        ] {
            assert!(bad.parse::<SessionId>().is_err(), "{bad}");
        }
        assert!(serde_json::from_str::<SessionId>("\"x\"").is_err());
        let long = "x".repeat(1000);
        assert_eq!(long.parse::<SessionId>().unwrap_err().0.len(), 64);
    }

    #[test]
    fn small_ids_are_plain_json_and_print_plainly() {
        assert_eq!(serde_json::to_string(&SegmentId(3)).unwrap(), "3");
        assert_eq!(serde_json::to_string(&TurnId(Seq(7))).unwrap(), "7");
        let made = CallId::made(Seq(12), 0);
        assert_eq!(made.as_str(), "s12-0");
        assert_eq!(serde_json::to_string(&made).unwrap(), "\"s12-0\"");
        assert_eq!(CallId::from("toolu_01"), CallId("toolu_01".into()));
        assert_eq!(
            CallId::from(String::from("a")),
            CallId::from(Box::<str>::from("a"))
        );
        assert_eq!(made.to_string(), "s12-0");
        assert_eq!(SegmentId(3).to_string(), "3");
        assert_eq!(TurnId(Seq(7)).to_string(), "7");
        assert_eq!(Seq(7).to_string(), "7");
    }
}
