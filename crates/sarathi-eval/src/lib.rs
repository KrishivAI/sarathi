// SPDX-License-Identifier: Apache-2.0
//! Offline evaluation of a harness (layer S4): brief files, cassette files, per-run metrics, the kill-line rule and the
//! report (the parts that need no provider). The scripted provider, record and replay decorators and the
//! runner follow once the provider trait lands. Nothing here makes a model call.

pub mod brief;
pub mod cassette;
pub mod killline;
pub mod metrics;
pub mod report;
pub mod runner;
