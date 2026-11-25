// Derived from Ultrafast Gateway @ c9c3eaf (MIT); see PROVENANCE.md
// SPDX-License-Identifier: Apache-2.0
use thiserror::Error;

/// A failure in building a request or reading an answer.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum TranslateError {
    /// The request is not valid in the caller's format.
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    /// The request is valid but uses something this target cannot express.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// The provider answered with an error.
    #[error("provider error {status}: {message}")]
    Provider {
        /// The HTTP status.
        status: u16,
        /// Whether the provider's answer says to try again.
        retryable: bool,
        /// The provider's message.
        message: String,
    },
    /// The provider answered with something that could not be read.
    #[error("malformed provider response: {0}")]
    Malformed(String),
}
