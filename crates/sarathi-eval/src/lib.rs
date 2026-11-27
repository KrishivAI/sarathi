// SPDX-License-Identifier: Apache-2.0
//! Offline evaluation of a harness (layer S4): brief files, cassette files, the record and replay decorators over a
//! model caller, the brief runner and its acceptance check, per-run metrics, spend caps, passes, the kill-line rule and
//! the report. Nothing here makes a model call unless a caller hands it a harness or a caller that does.

pub mod brief;
pub mod caller;
pub mod cassette;
pub mod config;
pub mod killline;
pub mod measure;
pub mod metrics;
pub mod pass;
pub mod report;
pub mod runner;
pub mod structural;
