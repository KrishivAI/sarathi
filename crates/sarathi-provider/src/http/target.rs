// Derived from Ultrafast Gateway @ c9c3eaf (MIT); see PROVENANCE.md
// SPDX-License-Identifier: Apache-2.0
//! Where calls go: a gateway, or a provider directly.

use std::fmt;

use crate::wire::provider::{ProviderKind, gateway_base};

const REDACTED: &str = "[redacted]";

/// What a target can do (capability flags, used where present and never required).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Capabilities {
    /// The Anthropic-native calls ([`Client::messages`](crate::Client::messages) and the rest) are available.
    pub native_messages: bool,
    /// The request bytes Sarathi sends reach the model unchanged. False for the gateway: at c9c3eaf it parses the
    /// native body and writes it again with `model` replaced, so the content is JSON-equal but escapes and number
    /// spellings may change, and an input guardrail may redact earlier turns. A caller comparing what
    /// was sent with what it stored compares JSON values, not bytes, when this is false.
    pub byte_exact_request: bool,
}

impl Target {
    /// The target's [`Capabilities`].
    pub fn capabilities(&self) -> Capabilities {
        match self {
            Target::Gateway { .. } => Capabilities {
                native_messages: true,
                byte_exact_request: false,
            },
            Target::Provider { kind, .. } => Capabilities {
                native_messages: *kind == ProviderKind::Anthropic,
                byte_exact_request: true,
            },
        }
    }
}

/// Where calls go: a gateway, or a provider directly. `Debug` never prints the key.
#[derive(Clone)]
pub enum Target {
    /// An Ultrafast gateway: OpenAI's
    /// wire format under `{base_url}/v1`.
    Gateway {
        /// The gateway's address.
        base_url: String,
        /// The gateway key.
        key: String,
    },
    /// A provider API, called directly.
    Provider {
        /// The provider's wire format.
        kind: ProviderKind,
        /// The API's base URL.
        base_url: String,
        /// The provider API key.
        key: String,
    },
}

impl Target {
    /// `base_url` is the gateway's address; a trailing `/v1` is accepted and ignored.
    pub fn gateway(base_url: impl Into<String>, key: impl Into<String>) -> Self {
        Target::Gateway {
            base_url: base_url.into(),
            key: key.into(),
        }
    }

    /// OpenAI's API at its public address.
    pub fn openai(key: impl Into<String>) -> Self {
        Self::provider(ProviderKind::OpenAi, "https://api.openai.com/v1", key)
    }

    /// Anthropic's Messages API at its public address.
    pub fn anthropic(key: impl Into<String>) -> Self {
        Self::provider(ProviderKind::Anthropic, "https://api.anthropic.com", key)
    }

    /// Any API that speaks OpenAI's format (Groq, Mistral, OpenRouter, Ollama);
    /// `base_url` includes the version segment, e.g. `https://host/v1`.
    pub fn openai_compatible(base_url: impl Into<String>, key: impl Into<String>) -> Self {
        Self::provider(ProviderKind::OpenAi, base_url, key)
    }

    fn provider(kind: ProviderKind, base_url: impl Into<String>, key: impl Into<String>) -> Self {
        Target::Provider {
            kind,
            base_url: base_url.into(),
            key: key.into(),
        }
    }

    /// Replaces the base URL (a proxy, a region, a test server).
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        match &mut self {
            Target::Gateway { base_url, .. } | Target::Provider { base_url, .. } => {
                *base_url = url.into();
            }
        }
        self
    }

    pub(crate) fn key(&self) -> &str {
        match self {
            Target::Gateway { key, .. } | Target::Provider { key, .. } => key,
        }
    }

    pub(crate) fn is_gateway(&self) -> bool {
        matches!(self, Target::Gateway { .. })
    }

    /// The `translate` target for a call to `model`. A gateway is an
    /// OpenAI-kind target under `/v1`.
    pub(crate) fn translate(&self, model: &str) -> crate::wire::provider::Target {
        let key = |k: &str| (!k.is_empty()).then(|| k.to_string());
        match self {
            Target::Gateway { base_url, key: k } => crate::wire::provider::Target {
                kind: ProviderKind::OpenAi,
                base_url: gateway_base(base_url),
                api_key: key(k),
                model: model.to_string(),
            },
            Target::Provider {
                kind,
                base_url,
                key: k,
            } => crate::wire::provider::Target {
                kind: *kind,
                base_url: base_url.clone(),
                api_key: key(k),
                model: model.to_string(),
            },
        }
    }

    pub(crate) fn kind(&self) -> ProviderKind {
        match self {
            Target::Gateway { .. } => ProviderKind::OpenAi,
            Target::Provider { kind, .. } => *kind,
        }
    }
}

/// Never prints the key.
impl fmt::Debug for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Target::Gateway { base_url, .. } => f
                .debug_struct("Gateway")
                .field("base_url", base_url)
                .field("key", &REDACTED)
                .finish(),
            Target::Provider { kind, base_url, .. } => f
                .debug_struct("Provider")
                .field("kind", kind)
                .field("base_url", base_url)
                .field("key", &REDACTED)
                .finish(),
        }
    }
}
