// SPDX-License-Identifier: Apache-2.0
//! [`RawJson`]: a JSON value held as the bytes it arrived as.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::value::RawValue;

/// A JSON value held as text, validated once when it is made and never re-rendered.
///
/// Serialising a `RawJson` writes its text as is, so a value that went into a request or a log line comes back with
/// the same bytes: escapes (`\u00e9`, `\/`), number spellings (`1e5`, `-0`, `1.0`) and key order are kept. Whitespace
/// around the value is dropped when it is made; whitespace inside it is kept. Two values are equal when their text
/// is, not when they are equal as JSON.
///
/// Bytes are kept only when the value is read from JSON text (`serde_json::from_str`, `from_slice`, `from_reader`)
/// through types that do not buffer it: not through `serde_json::Value` (`from_value` succeeds but the bytes are
/// already re-rendered), `#[serde(flatten)]`, or internally tagged, adjacently tagged or untagged enums.
#[derive(Clone)]
pub struct RawJson(Box<RawValue>);

/// A value that is not one JSON value.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RawJsonError {
    /// The bytes are not UTF-8.
    #[error("not UTF-8 at byte {0}")]
    NotUtf8(usize),
    /// The text is not exactly one JSON value.
    #[error("not one JSON value: {0}")]
    Invalid(String),
}

impl RawJson {
    /// Validates `text` as one JSON value and keeps it.
    ///
    /// # Errors
    /// `text` is not exactly one JSON value.
    pub fn from_string(text: String) -> Result<Self, RawJsonError> {
        // Reuses the allocation when `text` has no outer whitespace.
        RawValue::from_string(text)
            .map(RawJson)
            .map_err(|e| RawJsonError::Invalid(e.to_string()))
    }

    /// Validates `text` as one JSON value and keeps a copy.
    ///
    /// # Errors
    /// `text` is not exactly one JSON value.
    pub fn from_text(text: &str) -> Result<Self, RawJsonError> {
        serde_json::from_str::<Box<RawValue>>(text)
            .map(RawJson)
            .map_err(|e| RawJsonError::Invalid(e.to_string()))
    }

    /// Validates `bytes` as UTF-8 and as one JSON value, and keeps a copy.
    ///
    /// # Errors
    /// The bytes are not UTF-8, or not exactly one JSON value.
    pub fn from_slice(bytes: &[u8]) -> Result<Self, RawJsonError> {
        let text =
            std::str::from_utf8(bytes).map_err(|e| RawJsonError::NotUtf8(e.valid_up_to()))?;
        Self::from_text(text)
    }

    /// Renders `value` once (compact `serde_json` output). For values Sarathi writes itself, such as a user message;
    /// never for bytes a provider sent.
    ///
    /// # Errors
    /// `value`'s `Serialize` fails (a map with non-string keys, a non-finite float).
    pub fn render<T: Serialize + ?Sized>(value: &T) -> Result<Self, RawJsonError> {
        serde_json::value::to_raw_value(value)
            .map(RawJson)
            .map_err(|e| RawJsonError::Invalid(e.to_string()))
    }

    /// A JSON string holding `text`, rendered once.
    pub fn string(text: &str) -> Self {
        // A `&str` always serialises.
        match serde_json::value::to_raw_value(text) {
            Ok(v) => RawJson(v),
            Err(_) => RawJson::null(),
        }
    }

    /// `null`.
    pub fn null() -> Self {
        RawJson(RawValue::NULL.to_owned())
    }

    /// The text, exactly as held.
    pub fn as_str(&self) -> &str {
        self.0.get()
    }

    /// The bytes, exactly as held.
    pub fn as_bytes(&self) -> &[u8] {
        self.0.get().as_bytes()
    }

    /// The length of the text in bytes.
    pub fn len(&self) -> usize {
        self.0.get().len()
    }

    /// Never true: a JSON value has at least one byte. Present for the `len` convention.
    pub fn is_empty(&self) -> bool {
        self.0.get().is_empty()
    }

    /// The held value as a `serde_json` raw value.
    pub fn as_raw_value(&self) -> &RawValue {
        &self.0
    }

    /// The held value as a boxed `serde_json` raw value.
    pub fn into_raw_value(self) -> Box<RawValue> {
        self.0
    }
}

impl From<Box<RawValue>> for RawJson {
    fn from(v: Box<RawValue>) -> Self {
        RawJson(v)
    }
}

impl PartialEq for RawJson {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

impl Eq for RawJson {}

impl std::hash::Hash for RawJson {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}

impl fmt::Debug for RawJson {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RawJson({})", self.as_str())
    }
}

impl fmt::Display for RawJson {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for RawJson {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for RawJson {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Box::<RawValue>::deserialize(deserializer).map(RawJson)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HAZARDS: &str =
        r#"{"z":1,"a":"\u00e9\/","n":[1e5,-0,1.0,12345678901234567890],"s":"\ud83d\ude00"}"#;

    #[test]
    fn keeps_the_text_exactly() {
        let r = RawJson::from_text(HAZARDS).unwrap();
        assert_eq!(r.as_str(), HAZARDS);
        assert_eq!(serde_json::to_string(&r).unwrap(), HAZARDS);
    }

    #[test]
    fn a_value_round_trip_changes_the_bytes() {
        // The teeth of the whole design: a `Value` re-render is not byte-preserving.
        let v: serde_json::Value = serde_json::from_str(HAZARDS).unwrap();
        assert_ne!(serde_json::to_string(&v).unwrap(), HAZARDS);
    }

    #[test]
    fn survives_inside_a_serialised_struct() {
        #[derive(Serialize, Deserialize)]
        struct Line {
            bytes: RawJson,
        }
        let line = serde_json::to_string(&Line {
            bytes: RawJson::from_text(HAZARDS).unwrap(),
        })
        .unwrap();
        assert_eq!(line, format!(r#"{{"bytes":{HAZARDS}}}"#));
        let back: Line = serde_json::from_str(&line).unwrap();
        assert_eq!(back.bytes.as_str(), HAZARDS);
    }

    #[test]
    fn outer_whitespace_is_dropped_inner_is_kept() {
        let r = RawJson::from_text(" {\"a\" : 1}\n").unwrap();
        assert_eq!(r.as_str(), "{\"a\" : 1}");
    }

    #[test]
    fn refuses_what_is_not_one_value() {
        assert!(matches!(
            RawJson::from_text("{"),
            Err(RawJsonError::Invalid(_))
        ));
        assert!(matches!(
            RawJson::from_text("1 2"),
            Err(RawJsonError::Invalid(_))
        ));
        assert!(matches!(
            RawJson::from_text(""),
            Err(RawJsonError::Invalid(_))
        ));
        assert_eq!(
            RawJson::from_slice(b"\"\xff\""),
            Err(RawJsonError::NotUtf8(1))
        );
    }

    #[test]
    fn string_render_and_null() {
        assert_eq!(RawJson::string("a\"b").as_str(), r#""a\"b""#);
        assert_eq!(RawJson::null().as_str(), "null");
        assert_eq!(RawJson::render(&[1, 2]).unwrap().as_str(), "[1,2]");
        assert_eq!(RawJson::from_string("true".into()).unwrap().len(), 4);
        assert!(!RawJson::null().is_empty());
    }

    #[test]
    fn equality_is_on_text() {
        let a = RawJson::from_text(r#"{"a":1}"#).unwrap();
        let b = RawJson::from_text(r#"{ "a":1}"#).unwrap();
        assert_ne!(a, b);
        assert_eq!(a, a.clone());
        assert_eq!(format!("{a}"), r#"{"a":1}"#);
        assert_eq!(format!("{a:?}"), r#"RawJson({"a":1})"#);
    }
}
