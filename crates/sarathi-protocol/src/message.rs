// SPDX-License-Identifier: Apache-2.0
//! Message slices: one stored, never re-rendered message each.

use serde::{Deserialize, Serialize};

use crate::raw::{RawJson, RawJsonError};

/// The position of an entry in a session log: dense, from 0 (the header). An append names the seq it expects, so two
/// writers cannot both append.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Seq(pub u64);

impl std::fmt::Display for Seq {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// Who a message is from, as the Messages API names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// The user, a tool result or a steer.
    User,
    /// The model.
    Assistant,
    /// A system message between turns (the `mid-system` case).
    System,
}

impl Role {
    /// The wire name (`user`, `assistant`, `system`).
    pub fn as_str(self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => "system",
        }
    }
}

/// One message of a conversation, as the bytes that go into every request that holds it.
///
/// Made once: a user or tool-result message when Sarathi renders it, an assistant message from the `content` bytes
/// the provider sent. A request is assembled from slices; nothing reads a slice back into a typed model and writes it
/// again.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MessageSlice {
    /// The log entry the slice came from.
    pub seq: Seq,
    /// Who the message is from.
    pub role: Role,
    /// The whole message object, `{"role":…,"content":…}`.
    pub bytes: RawJson,
}

impl MessageSlice {
    /// `{"role":"<role>","content":<content>}`, with `content`'s bytes unchanged.
    ///
    /// # Errors
    /// Never for a valid `content`; the joined text is validated once, as every `RawJson` is.
    pub fn new(seq: Seq, role: Role, content: &RawJson) -> Result<Self, RawJsonError> {
        let mut text = String::with_capacity(content.len() + 32);
        text.push_str(r#"{"role":""#);
        text.push_str(role.as_str());
        text.push_str(r#"","content":"#);
        text.push_str(content.as_str());
        text.push('}');
        Ok(MessageSlice {
            seq,
            role,
            bytes: RawJson::from_string(text)?,
        })
    }

    /// A user message whose content is one text string.
    ///
    /// # Errors
    /// Never in practice: a JSON string is valid content (see [`MessageSlice::new`]).
    pub fn user_text(seq: Seq, text: &str) -> Result<Self, RawJsonError> {
        MessageSlice::new(seq, Role::User, &RawJson::string(text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_slice_keeps_content_bytes() {
        let content = RawJson::from_text(r#"[{"type":"text","text":"\u00e9","z":1e5}]"#).unwrap();
        let s = MessageSlice::new(Seq(4), Role::Assistant, &content).unwrap();
        assert_eq!(
            s.bytes.as_str(),
            r#"{"role":"assistant","content":[{"type":"text","text":"\u00e9","z":1e5}]}"#
        );
        assert_eq!(s.seq, Seq(4));
    }

    #[test]
    fn user_text_and_roles() {
        let s = MessageSlice::user_text(Seq(1), "hi \"there\"").unwrap();
        assert_eq!(
            s.bytes.as_str(),
            r#"{"role":"user","content":"hi \"there\""}"#
        );
        assert_eq!(Role::System.as_str(), "system");
        assert_eq!(
            serde_json::to_string(&Role::Assistant).unwrap(),
            r#""assistant""#
        );
        assert_eq!(serde_json::to_string(&Seq(7)).unwrap(), "7");
    }
}
