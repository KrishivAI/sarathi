// SPDX-License-Identifier: Apache-2.0
//! A minimal executor for tests: polls until ready, with a no-op waker.

use std::future::Future;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

pub(crate) fn block_on<F: Future>(fut: F) -> F::Output {
    let mut cx = Context::from_waker(Waker::noop());
    let mut fut = pin!(fut);
    loop {
        if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
            return v;
        }
    }
}

/// Pending once, then ready: a host that answers after a while.
pub(crate) struct YieldOnce(pub(crate) bool);
impl Future for YieldOnce {
    type Output = ();
    fn poll(mut self: std::pin::Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.0 {
            Poll::Ready(())
        } else {
            self.0 = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}
