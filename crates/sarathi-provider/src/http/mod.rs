// SPDX-License-Identifier: Apache-2.0
//! The HTTP client: a gateway or a provider directly, `chat` and `chat_stream`, and the Anthropic-native `messages`, `messages_stream` and `count_tokens`.

mod client;
mod error;
mod native;
mod openai_chat;
mod request;
mod stream;
mod target;

pub use client::{Client, DEFAULT_MAX_RESPONSE_BYTES};
pub use error::{Error, ErrorKind};
pub use native::{NativeEvent, TokenCount};
pub use request::{ChatRequest, MAX_TAGS_BYTES};
pub use target::{Capabilities, Target};
