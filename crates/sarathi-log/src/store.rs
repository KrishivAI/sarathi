// SPDX-License-Identifier: Apache-2.0
//! The [`LogStore`] seam, the in-memory store and [`fork`].

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Mutex, PoisonError};

use sarathi_protocol::Seq;

use crate::entry::{Body, Entry, ForkedFrom, Header, SegmentReason, SessionId};
use crate::error::LogError;
use crate::log::Log;

/// When an append returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Durability {
    /// The entries are on disk on return (`fsync` for a file store). Used for the entries the crash rule needs
    /// (`model.response`, `message.assistant` and `tool.start` before a side-effecting tool runs) and for `turn.end`.
    Durable,
    /// Written; made durable by the next durable append. A crash may lose them; resume redoes their work.
    Buffered,
}

/// What a store recovered from when it read a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Recovered {
    /// Bytes of a torn last line that were dropped.
    pub truncated_bytes: u64,
    /// The store cut the file and recorded the loss in a `note` entry. False when another writer holds the session:
    /// the read stops at the last complete line and the file is left as it is.
    pub repaired: bool,
}

/// A session as read from a store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Loaded {
    /// The session.
    pub log: Log,
    /// Set when the read dropped a torn last line.
    pub recovered: Option<Recovered>,
}

/// Where session logs live. One of the four seams a host fills; [`MemoryStore`] and the JSONL file store are the two
/// this crate has.
///
/// **Optimistic append:** an append names the seq it expects to be next. If another writer appended first, the
/// append fails with [`LogError::Conflict`] and writes nothing, so two writers never interleave entries.
pub trait LogStore: Send + Sync {
    /// Creates a session and writes its header.
    fn create(
        &self,
        created: u64,
        forked_from: Option<ForkedFrom>,
    ) -> impl Future<Output = Result<Header, LogError>> + Send;

    /// Appends `entries` if the session's next seq is `expected`; all or none. Returns the next seq after them.
    fn append(
        &self,
        session: SessionId,
        expected: Seq,
        entries: Vec<Entry>,
        durability: Durability,
    ) -> impl Future<Output = Result<Seq, LogError>> + Send;

    /// Reads a whole session.
    fn load(&self, session: SessionId) -> impl Future<Output = Result<Loaded, LogError>> + Send;
}

/// Sessions in memory: for tests, evals and hosts without a disk. Lost when dropped.
#[derive(Debug, Default)]
pub struct MemoryStore {
    sessions: Mutex<HashMap<SessionId, Log>>,
}

impl MemoryStore {
    /// An empty store.
    pub fn new() -> Self {
        MemoryStore::default()
    }
}

impl LogStore for MemoryStore {
    async fn create(
        &self,
        created: u64,
        forked_from: Option<ForkedFrom>,
    ) -> Result<Header, LogError> {
        let header = Header::new(SessionId::new(), created, forked_from);
        let mut map = self.sessions.lock().unwrap_or_else(PoisonError::into_inner);
        map.insert(header.session, Log::new(header.clone()));
        Ok(header)
    }

    async fn append(
        &self,
        session: SessionId,
        expected: Seq,
        entries: Vec<Entry>,
        _durability: Durability,
    ) -> Result<Seq, LogError> {
        let mut map = self.sessions.lock().unwrap_or_else(PoisonError::into_inner);
        let log = map
            .get_mut(&session)
            .ok_or_else(|| LogError::NotFound(session.to_string()))?;
        // The line codec must accept every entry, as the file store's would.
        for e in &entries {
            crate::line::encode_entry(e)?;
        }
        log.append(expected, entries)
    }

    async fn load(&self, session: SessionId) -> Result<Loaded, LogError> {
        let map = self.sessions.lock().unwrap_or_else(PoisonError::into_inner);
        let log = map
            .get(&session)
            .cloned()
            .ok_or_else(|| LogError::NotFound(session.to_string()))?;
        Ok(Loaded {
            log,
            recovered: None,
        })
    }
}

/// Makes a new session that continues `session` from `at`.
///
/// The child's header records the parent, `at` and the hash of the parent's lines `1..=at`. The child holds a copy of
/// those entries (so it stays readable if the parent is archived or deleted), then a `segment.start` with reason
/// `fork` whose seed is the parent's projection at `at`: the child's next request is the one the parent would have
/// sent. `at` must be a turn boundary (no call waiting for a result, no model call in flight).
///
/// # Errors
/// [`LogError::NotABoundary`] for a point inside a tool round; store errors as they come.
pub async fn fork<S: LogStore + ?Sized>(
    store: &S,
    session: SessionId,
    at: Seq,
    ts: u64,
) -> Result<SessionId, LogError> {
    let parent = store.load(session).await?.log.up_to(at)?;
    if at.0 == 0 || !parent.resume().at_boundary() {
        return Err(LogError::NotABoundary(at));
    }
    let forked_from = ForkedFrom {
        session,
        seq: at,
        sha256: parent.lines_sha256(at)?,
    };
    let header = store.create(ts, Some(forked_from)).await?;
    let mut entries = parent.entries().to_vec();
    entries.push(Entry {
        seq: Seq(at.0 + 1),
        seg: parent.segment() + 1,
        ts,
        body: Body::SegmentStart {
            reason: SegmentReason::Fork,
            strategy: None,
            seed: parent.projection().messages,
            replaces: None,
        },
    });
    store
        .append(header.session, Seq(1), entries, Durability::Durable)
        .await?;
    Ok(header.session)
}
