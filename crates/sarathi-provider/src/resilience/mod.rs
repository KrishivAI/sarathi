// SPDX-License-Identifier: Apache-2.0
//! What the client itself does not do: retries ([`RetryPolicy`]), the circuit breaker ([`CircuitBreaker`]),
//! failover across slots ([`Chain`]) and the spend check ([`budget`]). Rewritten from Vyasa's `failover.rs`,
//! `circuit.rs` and `budget.rs` against Sarathi's [`Error`](crate::Error), whose `retryable` and `retry_after` are the
//! seam.
//!
//! Order for a call: the budget check, then [`Chain::run`]; the core adds the call's cost to its spend afterwards.

pub mod budget;
mod circuit;
mod failover;
mod retry;

pub use budget::{BudgetCaps, BudgetDecision, MICROS_PER_USD};
pub use circuit::CircuitBreaker;
pub use failover::{Answered, Attempt, Chain, DEFAULT_BREAKER, Failed, Slot};
pub use retry::RetryPolicy;
