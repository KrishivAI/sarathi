// SPDX-License-Identifier: Apache-2.0
//! [`JsonlStore`]: one directory per session holding `log.jsonl` (line 0 the header, then one entry per line) and
//! `log.lock` (held while this store writes the session).

use std::collections::HashMap;
use std::fs::{File, OpenOptions, TryLockError};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{SystemTime, UNIX_EPOCH};

use sarathi_protocol::{RawJson, Seq};

use crate::entry::{Body, Entry, ForkPoint, LogHeader, SessionId};
use crate::error::LogError;
use crate::line::{ReadLog, encode_entry, encode_header, read_log};
use crate::log::Log;
use crate::store::{Durability, Loaded, LogStore, Recovered};

const LOG_FILE: &str = "log.jsonl";
const LOCK_FILE: &str = "log.lock";

/// Session logs as JSONL files under one root directory.
///
/// [`LogStore::load`] and [`LogStore::append`] open a session for writing: the store takes the session's lock,
/// repairs a torn tail (cuts it and records the loss in a `note`) and keeps the lock, the open file and the session
/// in memory until [`JsonlStore::release`] or until the store is dropped. A second store (in this process or another)
/// over the same session reads it but cannot append: [`LogError::Locked`]. A reader that must never take the lock or
/// change the file (an audit, an eval report) uses [`JsonlStore::read`].
///
/// Opening a session holds the store's session table while the file is read, so the first use of a large session
/// delays other sessions' first use and appends in the same store.
///
/// File IO runs on the blocking pool, so the futures never block an async runtime's worker.
#[derive(Debug, Clone)]
pub struct JsonlStore {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    root: PathBuf,
    writers: Mutex<HashMap<SessionId, Arc<Mutex<Writer>>>>,
}

#[derive(Debug)]
struct Writer {
    file: File,
    /// The file's length after the last complete append.
    len: u64,
    log: Log,
    /// An append failed and its rollback failed too: the file may end in part of a batch. The next use re-opens the
    /// session, which reads that part as a torn tail.
    failed: bool,
    _lock: File,
}

impl JsonlStore {
    /// A store over `root`. The directory is created on the first `create`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        JsonlStore {
            inner: Arc::new(Inner {
                root: root.into(),
                writers: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// Reads a session without taking its lock or changing its file: a torn last line is left as it is and reported
    /// with `repaired: false`.
    ///
    /// # Errors
    /// As [`LogStore::load`].
    pub async fn read(&self, session: SessionId) -> Result<Loaded, LogError> {
        self.blocking(move |inner| inner.read(session)).await
    }

    /// Gives up a session this store opened for writing: its lock, file and in-memory copy. An append in progress
    /// keeps them until it returns. The next `load` or `append` opens the session again. Returns whether the store
    /// held it. Until that append returns, opening the session again (here or in another store) gives
    /// [`LogError::Locked`].
    pub fn release(&self, session: SessionId) -> bool {
        self.inner.writers().remove(&session).is_some()
    }

    /// The path of a session's log file.
    pub fn log_path(&self, session: SessionId) -> PathBuf {
        self.inner.root.join(session.to_string()).join(LOG_FILE)
    }

    async fn blocking<T: Send + 'static>(
        &self,
        f: impl FnOnce(&Inner) -> Result<T, LogError> + Send + 'static,
    ) -> Result<T, LogError> {
        let inner = Arc::clone(&self.inner);
        tokio::task::spawn_blocking(move || f(&inner))
            .await
            .map_err(|e| LogError::Io(io::Error::other(e)))?
    }
}

impl LogStore for JsonlStore {
    async fn create(
        &self,
        created: u64,
        forked_from: Option<ForkPoint>,
    ) -> Result<LogHeader, LogError> {
        self.blocking(move |inner| inner.create(created, forked_from))
            .await
    }

    async fn append(
        &self,
        session: SessionId,
        expected: Seq,
        entries: Vec<Entry>,
        durability: Durability,
    ) -> Result<Seq, LogError> {
        self.blocking(move |inner| inner.append(session, expected, entries, durability))
            .await
    }

    async fn load(&self, session: SessionId) -> Result<Loaded, LogError> {
        self.blocking(move |inner| inner.load(session)).await
    }
}

impl Inner {
    fn dir(&self, session: SessionId) -> PathBuf {
        self.root.join(session.to_string())
    }

    fn writers(&self) -> std::sync::MutexGuard<'_, HashMap<SessionId, Arc<Mutex<Writer>>>> {
        self.writers.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn create(&self, created: u64, forked_from: Option<ForkPoint>) -> Result<LogHeader, LogError> {
        let header = crate::new_header(crate::new_session_id(), created, forked_from);
        let dir = self.dir(header.session);
        std::fs::create_dir_all(&self.root)?;
        std::fs::create_dir(&dir)?;
        let lock = lock_file(&dir, header.session)?;
        let mut file = OpenOptions::new()
            .create_new(true)
            .append(true)
            .open(dir.join(LOG_FILE))?;
        let mut line = encode_header(&header)?;
        line.push('\n');
        file.write_all(line.as_bytes())?;
        file.sync_data()?;
        sync_dir(&dir)?;
        sync_dir(&self.root)?;
        self.writers().insert(
            header.session,
            Arc::new(Mutex::new(Writer {
                file,
                len: line.len() as u64,
                log: Log::new(header.clone()),
                failed: false,
                _lock: lock,
            })),
        );
        Ok(header)
    }

    /// The session's writer, opening (and repairing) the file if this store does not hold it yet.
    fn writer(
        &self,
        session: SessionId,
    ) -> Result<(Arc<Mutex<Writer>>, Option<Recovered>), LogError> {
        // Held while opening, so two first uses of one session in this store do not race for its lock.
        let mut writers = self.writers();
        if let Some(w) = writers.get(&session) {
            return Ok((Arc::clone(w), None));
        }
        let dir = self.dir(session);
        let path = dir.join(LOG_FILE);
        if !path.exists() {
            return Err(LogError::NotFound(session));
        }
        let lock = lock_file(&dir, session)?;
        let read = read_file(&path)?;
        let mut file = OpenOptions::new().append(true).open(&path)?;
        let mut log = read.log;
        let mut len = read.complete_len;
        let mut recovered = None;
        if read.torn_bytes > 0 {
            file.set_len(read.complete_len)?;
            let note = Entry {
                seq: log.next_seq(),
                seg: log.segment(),
                ts: now_ms(),
                body: Body::Note {
                    data: RawJson::render(&serde_json::json!({
                        "store": "torn_tail",
                        "truncated_bytes": read.torn_bytes,
                    }))
                    .map_err(|e| LogError::Io(io::Error::other(e)))?,
                },
            };
            // The note is checked before it is written. Where no note may stand (in a fork, the place of its
            // `segment.start`), the tear is cut and nothing is written: a fork cut short is read as it is.
            let mut line = encode_entry(&note)?;
            line.push('\n');
            if log.append(note.seq, vec![note]).is_ok() {
                file.write_all(line.as_bytes())?;
                len += line.len() as u64;
            }
            file.sync_data()?;
            recovered = Some(Recovered {
                truncated_bytes: read.torn_bytes,
                repaired: true,
            });
        }
        let w = Arc::new(Mutex::new(Writer {
            file,
            len,
            log,
            failed: false,
            _lock: lock,
        }));
        writers.insert(session, Arc::clone(&w));
        Ok((w, recovered))
    }

    fn append(
        &self,
        session: SessionId,
        expected: Seq,
        entries: Vec<Entry>,
        durability: Durability,
    ) -> Result<Seq, LogError> {
        let (w, _) = self.writer(session)?;
        let mut w = w.lock().unwrap_or_else(PoisonError::into_inner);
        if w.failed {
            // Taken from the table by the failed append; appending here would put this batch after its partial
            // bytes, making a bad line in the middle of the file.
            return Err(LogError::Io(io::Error::other(
                "an earlier append to this session failed and could not be undone; retry to re-open it",
            )));
        }
        let mut buf = String::new();
        for e in &entries {
            buf.push_str(&encode_entry(e)?);
            buf.push('\n');
        }
        let len = w.log.entries().len();
        let next = w.log.append(expected, entries)?;
        let written = w
            .file
            .write_all(buf.as_bytes())
            .and_then(|()| match durability {
                Durability::Durable => w.file.sync_data(),
                Durability::Buffered => Ok(()),
            });
        match written {
            Ok(()) => {
                w.len += buf.len() as u64;
                Ok(next)
            }
            Err(e) => {
                // All or none: cut off whatever part of the batch reached the file. If even that fails, drop the
                // writer so the next use re-reads the file (a partial last line is then a torn tail).
                w.log.truncate_to(len);
                let prev = w.len;
                if w.file.set_len(prev).is_err() {
                    w.failed = true;
                    drop(w);
                    self.writers().remove(&session);
                }
                Err(e.into())
            }
        }
    }

    fn load(&self, session: SessionId) -> Result<Loaded, LogError> {
        match self.writer(session) {
            Ok((w, recovered)) => {
                let w = w.lock().unwrap_or_else(PoisonError::into_inner);
                Ok(Loaded {
                    log: w.log.clone(),
                    recovered,
                })
            }
            // Another writer holds it: read up to the last complete line, change nothing.
            Err(LogError::Locked(_)) => self.read(session),
            Err(e) => Err(e),
        }
    }

    fn read(&self, session: SessionId) -> Result<Loaded, LogError> {
        let path = self.dir(session).join(LOG_FILE);
        if !path.exists() {
            return Err(LogError::NotFound(session));
        }
        let read = read_file(&path)?;
        Ok(Loaded {
            log: read.log,
            recovered: (read.torn_bytes > 0).then_some(Recovered {
                truncated_bytes: read.torn_bytes,
                repaired: false,
            }),
        })
    }
}

fn lock_file(dir: &Path, session: SessionId) -> Result<File, LogError> {
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(dir.join(LOCK_FILE))?;
    match lock.try_lock() {
        Ok(()) => Ok(lock),
        Err(TryLockError::WouldBlock) => Err(LogError::Locked(session)),
        Err(TryLockError::Error(e)) => Err(e.into()),
    }
}

fn sync_dir(dir: &Path) -> io::Result<()> {
    // Makes a new entry in the directory durable. Opening a directory to sync it is a Unix thing.
    if cfg!(unix) {
        File::open(dir)?.sync_all()?;
    }
    Ok(())
}

pub(crate) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

/// Reads a log file: the header, every complete line, and the size of a torn tail.
fn read_file(path: &Path) -> Result<ReadLog, LogError> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    read_log(&bytes)
}
