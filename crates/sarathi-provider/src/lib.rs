// SPDX-License-Identifier: Apache-2.0
//! Provider wire clients and translation (layer S1).
//!
//! Two parts, keeping the IO-free rule inside the crate: [`wire`] builds request bytes and parses response bytes and
//! SSE streams, and does no IO; [`http`] sends them (reqwest, rustls). The client does not retry, route, cache or
//! break circuits; every error says whether trying again could help (`retryable`) and for how long to wait
//! (`retry_after`). Retries, failover, the circuit breaker and the budget check are [`resilience`], built on those
//! two fields.
//!
//! For Anthropic the native call ([`Client::messages`]) is the one that keeps a turn: the request is spliced from
//! stored byte slices and the answer is kept as received, so thinking blocks, unknown block types and every escape
//! go back byte for byte. The translated `chat` path is for other providers.
//!
//! For those providers a harness keeps its turns with [`Client::chat_spliced`], whose body is spliced from slices
//! rendered once ([`wire::openai_chat`]).
//!
//! Derived from Ultrafast Gateway's `ufaigw-translate` and `ufaigw-client`; see `PROVENANCE.md`.

pub mod http;
pub mod resilience;
pub mod wire;

pub use http::{
    Capabilities, Client, DEFAULT_MAX_RESPONSE_BYTES, Error, ErrorKind, NativeEvent, Target,
    TokenCount,
};
pub use http::{ChatRequest, MAX_TAGS_BYTES};
pub use resilience::{Chain, CircuitBreaker, RetryPolicy, Slot};
pub use wire::native::{
    Assembler, CallParams, Endpoint, MessagesRequest, MessagesResponse, NativeError,
    RateLimitHeaders, SessionPrefix,
};
pub use wire::openai_chat::{ChatAnswer, ChatBody, ChatParams, TokenField};
pub use wire::provider::ProviderKind;
/// The common request and response types (re-export of [`wire::types`]).
pub use wire::types;
pub use wire::types::{ImageSource, Part, ResponseFormat, Tool, ToolCall, ToolChoice};
