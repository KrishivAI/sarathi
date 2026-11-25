// Derived from Ultrafast Gateway @ c9c3eaf (MIT); see PROVENANCE.md
// SPDX-License-Identifier: Apache-2.0
//! How a failed call is classified: the kind, whether to retry, and what a
//! `Retry-After` means. [`classify_answer`] does it in one call for a host
//! that sends its own HTTP.

use crate::wire::error::TranslateError;
use crate::wire::provider::provider_error;

/// What every client says about an answer with a 3xx status.
pub const REDIRECT_MESSAGE: &str = "the server answered with a redirect";

/// A `Retry-After` longer than this is capped (seconds).
pub const RETRY_AFTER_CAP_SECS: u64 = 24 * 60 * 60;

pub use sarathi_protocol::ErrorKind;

/// The kind of an answer with an HTTP status of 300 or more.
pub fn kind_of_status(status: u16) -> ErrorKind {
    match status {
        401 => ErrorKind::Auth,
        403 => ErrorKind::Permission,
        404 => ErrorKind::NotFound,
        408 => ErrorKind::Timeout,
        429 => ErrorKind::RateLimited,
        500.. => ErrorKind::Upstream,
        // Redirects are not followed (they would carry the key onward):
        // the target is wrong, and asking again will not change it.
        300..=399 => ErrorKind::InvalidRequest,
        _ => ErrorKind::InvalidRequest,
    }
}

/// An error, classified. Holds no credential unless the caller's message does:
/// the caller scrubs its own key.
#[derive(Debug, Clone, PartialEq)]
pub struct Classified {
    /// What kind of failure this is.
    pub kind: ErrorKind,
    /// Whether trying again could help.
    pub retryable: bool,
    /// The HTTP status, when there was an answer.
    pub status: Option<u16>,
    /// A readable message.
    pub message: String,
    /// Whole seconds; set for 429 and 503 answers only.
    pub retry_after_secs: Option<u64>,
}

impl Classified {
    /// A classified error of `kind` with no status.
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Classified {
            kind,
            retryable: kind.retryable(),
            status: None,
            message: message.into(),
            retry_after_secs: None,
        }
    }

    /// Classifies an HTTP error status; `retry_after` is kept for 429 and 503 only.
    pub fn from_status(status: u16, message: impl Into<String>, retry_after: Option<u64>) -> Self {
        let kind = kind_of_status(status);
        Classified {
            status: Some(status),
            retry_after_secs: if kind == ErrorKind::RateLimited || status == 503 {
                retry_after
            } else {
                None
            },
            ..Classified::new(kind, message)
        }
    }

    /// Classifies a translation error.
    pub fn from_translate(e: TranslateError, retry_after: Option<u64>) -> Self {
        match e {
            TranslateError::InvalidRequest(m) | TranslateError::Unsupported(m) => {
                Classified::new(ErrorKind::InvalidRequest, m)
            }
            TranslateError::Provider {
                status, message, ..
            } => Classified::from_status(status, message, retry_after),
            TranslateError::Malformed(m) => Classified::new(ErrorKind::Malformed, m),
        }
    }
}

/// `Retry-After` as seconds, capped; an HTTP date is not read.
pub fn parse_retry_after(value: &str) -> Option<u64> {
    let value = value.trim();
    if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    // Digits too many for a number are a very long wait: the cap.
    Some(
        value
            .parse::<u64>()
            .unwrap_or(RETRY_AFTER_CAP_SECS)
            .min(RETRY_AFTER_CAP_SECS),
    )
}

/// `message` with every occurrence of `key` replaced; any non-empty key.
/// Every client runs its error messages through this, so a server that
/// echoes the credential back does not leak it.
pub fn scrub(message: &str, key: &str) -> String {
    if key.is_empty() {
        return message.to_string();
    }
    message.replace(key, "[redacted]")
}

/// An HTTP error answer (any provider's, or a gateway's) from its status,
/// body and `Retry-After` header value.
pub fn classify_answer(status: u16, body: &[u8], retry_after: Option<&str>) -> Classified {
    if (300..400).contains(&status) {
        return Classified::from_status(status, REDIRECT_MESSAGE, None);
    }
    Classified::from_translate(
        provider_error(status, body),
        retry_after.and_then(parse_retry_after),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statuses_map_to_kinds_and_retryability() {
        let cases = [
            (400, ErrorKind::InvalidRequest, false),
            (401, ErrorKind::Auth, false),
            (403, ErrorKind::Permission, false),
            (404, ErrorKind::NotFound, false),
            (408, ErrorKind::Timeout, true),
            (422, ErrorKind::InvalidRequest, false),
            (429, ErrorKind::RateLimited, true),
            (500, ErrorKind::Upstream, true),
            (503, ErrorKind::Upstream, true),
            (302, ErrorKind::InvalidRequest, false),
        ];
        for (status, kind, retry) in cases {
            let c = classify_answer(status, b"{}", None);
            assert_eq!((c.kind, c.retryable, c.status), (kind, retry, Some(status)));
        }
    }

    #[test]
    fn the_message_comes_from_the_body_in_any_shape() {
        let c = classify_answer(401, br#"{"error":{"message":"bad key"}}"#, None);
        assert_eq!(c.message, "bad key");
        let c = classify_answer(502, b"upstream down", None);
        assert_eq!(c.message, "upstream down");
    }

    #[test]
    fn retry_after_is_read_for_rate_limits_only_and_capped() {
        assert_eq!(
            classify_answer(429, b"{}", Some(" 7 ")).retry_after_secs,
            Some(7)
        );
        assert_eq!(
            classify_answer(503, b"{}", Some("5")).retry_after_secs,
            Some(5)
        );
        for status in [400, 401, 500, 502] {
            assert_eq!(
                classify_answer(status, b"{}", Some("7")).retry_after_secs,
                None,
                "{status}"
            );
        }
        assert_eq!(parse_retry_after("999999999"), Some(RETRY_AFTER_CAP_SECS));
        assert_eq!(
            parse_retry_after("99999999999999999999999"),
            Some(RETRY_AFTER_CAP_SECS)
        );
        for bad in ["", "-1", "1.5", "Wed, 21 Oct 2015 07:28:00 GMT", "+3"] {
            assert_eq!(parse_retry_after(bad), None, "{bad}");
        }
    }

    #[test]
    fn a_redirect_has_one_message_for_every_client() {
        for status in [301, 302, 307, 399] {
            let c = classify_answer(status, b"<html>moved</html>", Some("5"));
            assert_eq!(c.message, REDIRECT_MESSAGE);
            assert_eq!(c.retry_after_secs, None);
            assert_eq!((c.kind, c.retryable), (ErrorKind::InvalidRequest, false));
        }
        assert_ne!(classify_answer(400, b"x", None).message, REDIRECT_MESSAGE);
    }

    #[test]
    fn scrub_replaces_every_occurrence_of_any_non_empty_key() {
        assert_eq!(
            scrub("bad k-1 and k-1", "k-1"),
            "bad [redacted] and [redacted]"
        );
        assert_eq!(scrub("a=b", "b"), "a=[redacted]");
        assert_eq!(scrub("nothing here", "zz"), "nothing here");
        assert_eq!(scrub("keep", ""), "keep");
    }

    #[test]
    fn translate_errors_classify() {
        let c = Classified::from_translate(TranslateError::Malformed("x".into()), None);
        assert_eq!(
            (c.kind, c.retryable, c.status),
            (ErrorKind::Malformed, false, None)
        );
        let c = Classified::from_translate(TranslateError::Unsupported("x".into()), None);
        assert_eq!(c.kind, ErrorKind::InvalidRequest);
    }
}
