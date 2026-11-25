// SPDX-License-Identifier: Apache-2.0
//! Integration tests of the session log: the store conformance suite, the JSONL file store, the crash matrix and the
//! committed format fixtures. One binary, so the helpers in `common` are shared without dead-code noise.

#![expect(clippy::unwrap_used, clippy::panic, reason = "test code")]

mod common;
mod conformance;
mod crash;
mod fixtures;
mod generated;
mod jsonl;
mod seed;
