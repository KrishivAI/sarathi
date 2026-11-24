// SPDX-License-Identifier: Apache-2.0
//! Tool-output spill: output over a size limit is stored outside the log, and the `tool.result` slice holds a
//! bounded preview with a handle the model reads ranges through.

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Mutex, PoisonError};

use sarathi_protocol::{RawJson, SessionId, Sha256};
use serde::{Deserialize, Serialize};
use sha2::Digest;

use crate::error::ContextError;

/// Sizes that decide when output spills and how much a preview and a read hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpillLimits {
    /// Output longer than this (bytes) spills. Default 32 KiB.
    pub spill_bytes: usize,
    /// The preview's budget for kept output (bytes), split between the first and the last lines. Default 4 KiB.
    pub preview_bytes: usize,
    /// The most one ranged read returns (bytes). Default 16 KiB.
    pub read_cap: usize,
}

impl Default for SpillLimits {
    fn default() -> Self {
        SpillLimits {
            spill_bytes: 32 * 1024,
            preview_bytes: 4 * 1024,
            read_cap: 16 * 1024,
        }
    }
}

/// What a `tool.result` entry's `spill` field records: where the full output is and how to check it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpillRecord {
    /// The handle the preview names (`spill:<call-id>`).
    pub handle: String,
    /// The full output's length in bytes.
    pub bytes: u64,
    /// SHA-256 of the full output.
    pub sha256: Sha256,
}

impl SpillRecord {
    /// The record as the `tool.result` entry stores it.
    ///
    /// # Errors
    /// Never in practice: the record has only strings and an integer.
    pub fn to_raw(&self) -> Result<RawJson, ContextError> {
        Ok(RawJson::render(self)?)
    }

    /// Reads a stored record.
    ///
    /// # Errors
    /// The value is not a spill record.
    pub fn from_raw(raw: &RawJson) -> Result<Self, ContextError> {
        serde_json::from_str(raw.as_str())
            .map_err(|e| ContextError::Json(sarathi_protocol::RawJsonError::Invalid(e.to_string())))
    }

    /// The call id the handle names.
    ///
    /// # Errors
    /// [`ContextError::SpillNotFound`] for a handle that is not `spill:<call-id>`.
    pub fn call(&self) -> Result<&str, ContextError> {
        call_of(&self.handle)
    }
}

/// Output that spilled: the record for the log and the preview text for the result block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Spilled {
    /// Goes in the `tool.result` entry's `spill` field.
    pub record: SpillRecord,
    /// The text the result block carries instead of the output.
    pub preview: String,
}

/// Where spilled output lives. Writes are durable when `put` returns, so a `tool.result` written after it always
/// names output that can be read.
pub trait SpillStore: Send + Sync {
    /// Stores the output of `call` in `session`. Storing the same call again replaces it.
    fn put(
        &self,
        session: SessionId,
        call: &str,
        bytes: Vec<u8>,
    ) -> impl Future<Output = Result<(), ContextError>> + Send;

    /// Up to `len` bytes of the output of `call` from `offset`; fewer at the end, none past it.
    fn read(
        &self,
        session: SessionId,
        call: &str,
        offset: u64,
        len: usize,
    ) -> impl Future<Output = Result<Vec<u8>, ContextError>> + Send;

    /// Copies every spilled output of `from` to `to`, for a fork.
    fn copy_session(
        &self,
        from: SessionId,
        to: SessionId,
    ) -> impl Future<Output = Result<(), ContextError>> + Send;
}

/// Spills `output` of `call` if it is longer than [`SpillLimits::spill_bytes`]: stores it, then returns the record
/// and the preview. `None`: the output goes into the result block as it is.
///
/// # Errors
/// [`ContextError::BadCallId`] for a call id that cannot name a file; the store's errors.
pub async fn spill_output<S: SpillStore + ?Sized>(
    store: &S,
    session: SessionId,
    call: &str,
    output: &str,
    limits: &SpillLimits,
) -> Result<Option<Spilled>, ContextError> {
    if output.len() <= limits.spill_bytes {
        return Ok(None);
    }
    check_call_id(call)?;
    let record = SpillRecord {
        handle: format!("spill:{call}"),
        bytes: output.len() as u64,
        sha256: Sha256(sha2::Sha256::digest(output.as_bytes()).into()),
    };
    store.put(session, call, output.as_bytes().to_vec()).await?;
    let preview = preview(output, &record, limits);
    Ok(Some(Spilled { record, preview }))
}

/// A ranged read of spilled output by its handle, at most [`SpillLimits::read_cap`] bytes, in whole UTF-8
/// characters: a range that starts inside a character starts at the next one, and one that ends inside a character
/// ends before it, except that a read before the end returns at least one character even when `len` is smaller. A
/// read from a character boundary returns text starting exactly at `offset`, so reading on from `offset` plus the
/// returned text's length walks the whole output; empty text means the end.
///
/// # Errors
/// [`ContextError::ReadTooLarge`] over the cap; [`ContextError::SpillNotFound`] for an unknown handle.
pub async fn read_output<S: SpillStore + ?Sized>(
    store: &S,
    session: SessionId,
    handle: &str,
    offset: u64,
    len: usize,
    limits: &SpillLimits,
) -> Result<String, ContextError> {
    if len > limits.read_cap {
        return Err(ContextError::ReadTooLarge {
            asked: len,
            cap: limits.read_cap,
        });
    }
    let call = call_of(handle)?;
    // Three extra bytes on each side find the character boundaries.
    let back = offset.min(3);
    let mut bytes = store
        .read(session, call, offset - back, len.saturating_add(6))
        .await?;
    let start = usize::try_from(back).unwrap_or(0).min(bytes.len());
    let start = (start..bytes.len())
        .find(|&i| !is_continuation(bytes[i]))
        .unwrap_or(bytes.len());
    let mut end = start.saturating_add(len).min(bytes.len());
    while end > start && end < bytes.len() && is_continuation(bytes[end]) {
        end -= 1;
    }
    if end == start && start < bytes.len() {
        // `len` is smaller than the next character: return that character whole, so a reader always moves on.
        end = (start + 1..bytes.len())
            .find(|&i| !is_continuation(bytes[i]))
            .unwrap_or(bytes.len());
    }
    bytes.truncate(end);
    bytes.drain(..start);
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// Checks the stored output of a record against its hash, for replay and after a fork copy.
///
/// # Errors
/// [`ContextError::SpillMismatch`] if the bytes differ; the store's errors.
pub async fn verify_spill<S: SpillStore + ?Sized>(
    store: &S,
    session: SessionId,
    record: &SpillRecord,
) -> Result<(), ContextError> {
    let call = record.call()?;
    let len = usize::try_from(record.bytes)
        .map_err(|_| ContextError::SpillMismatch(record.handle.clone()))?;
    let bytes = store.read(session, call, 0, len.saturating_add(1)).await?;
    if bytes.len() as u64 != record.bytes || sha2::Sha256::digest(&bytes)[..] != record.sha256.0[..]
    {
        return Err(ContextError::SpillMismatch(record.handle.clone()));
    }
    Ok(())
}

fn is_continuation(b: u8) -> bool {
    b & 0xC0 == 0x80
}

fn call_of(handle: &str) -> Result<&str, ContextError> {
    handle
        .strip_prefix("spill:")
        .filter(|c| check_call_id(c).is_ok())
        .ok_or_else(|| ContextError::SpillNotFound(handle.to_owned()))
}

/// The longest call id a spill takes, in bytes: its file name is twice as long and must stay under 255.
const MAX_CALL_ID: usize = 120;

/// Any non-empty call id up to [`MAX_CALL_ID`] bytes: stores never use it as a file name as it is
/// ([`file_name`]).
fn check_call_id(call: &str) -> Result<(), ContextError> {
    let ok = !call.is_empty() && call.len() <= MAX_CALL_ID;
    if ok {
        Ok(())
    } else {
        Err(ContextError::BadCallId(call.to_owned()))
    }
}

/// The file name of a call's output: the call id's bytes in lower-case hex. Reversible, the same on case-insensitive
/// file systems, and never a temporary file's name (those start with `.`).
#[cfg(feature = "fs")]
fn file_name(call: &str) -> String {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(call.len() * 2);
    for b in call.bytes() {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// The call id a file name encodes, if it is one ([`file_name`]).
#[cfg(feature = "fs")]
fn call_of_file(name: &str) -> Option<String> {
    let b = name.as_bytes();
    if b.is_empty() || !b.len().is_multiple_of(2) || b.len() > MAX_CALL_ID * 2 {
        return None;
    }
    let nibble = |c: u8| match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        _ => None,
    };
    let bytes: Option<Vec<u8>> = b
        .chunks(2)
        .map(|p| Some((nibble(p[0])? << 4) | nibble(p[1])?))
        .collect();
    String::from_utf8(bytes?).ok()
}

/// The first and last whole lines of `output` within the preview budget, and a line naming the handle. The budget
/// is at most half of [`SpillLimits::spill_bytes`], so a preview is always shorter than the output it stands for.
fn preview(output: &str, record: &SpillRecord, limits: &SpillLimits) -> String {
    let half = limits.preview_bytes.min(limits.spill_bytes / 2) / 2;
    let mut head_end = 0;
    for (i, _) in output.match_indices('\n') {
        if i + 1 > half {
            break;
        }
        head_end = i + 1;
    }
    let mut tail_start = output.len();
    for (i, _) in output.rmatch_indices('\n') {
        if i + 1 < head_end || output.len() - (i + 1) > half {
            break;
        }
        tail_start = i + 1;
    }
    // A first or last line longer than the budget: cut it at a character boundary instead.
    if head_end == 0 {
        head_end = floor_char(output, half);
    }
    if tail_start == output.len() {
        tail_start = ceil_char(output, output.len().saturating_sub(half)).max(head_end);
    }
    let omitted = tail_start - head_end;
    format!(
        "{}\n[output truncated: {} bytes in all, {omitted} omitted here; the full output is {}, read it with \
         read_output(handle, offset, len)]\n{}",
        output[..head_end]
            .strip_suffix('\n')
            .unwrap_or(&output[..head_end]),
        record.bytes,
        record.handle,
        &output[tail_start..]
    )
}

fn floor_char(s: &str, mut i: usize) -> usize {
    i = i.min(s.len());
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn ceil_char(s: &str, mut i: usize) -> usize {
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

/// Spilled output in memory: for tests, evals and hosts without a disk.
#[derive(Debug, Default)]
pub struct MemorySpill {
    files: Mutex<HashMap<(SessionId, String), Vec<u8>>>,
}

impl MemorySpill {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }
}

impl SpillStore for MemorySpill {
    async fn put(
        &self,
        session: SessionId,
        call: &str,
        bytes: Vec<u8>,
    ) -> Result<(), ContextError> {
        check_call_id(call)?;
        let mut files = self.files.lock().unwrap_or_else(PoisonError::into_inner);
        files.insert((session, call.to_owned()), bytes);
        Ok(())
    }

    async fn read(
        &self,
        session: SessionId,
        call: &str,
        offset: u64,
        len: usize,
    ) -> Result<Vec<u8>, ContextError> {
        let files = self.files.lock().unwrap_or_else(PoisonError::into_inner);
        let bytes = files
            .get(&(session, call.to_owned()))
            .ok_or_else(|| ContextError::SpillNotFound(format!("spill:{call}")))?;
        let start = usize::try_from(offset)
            .unwrap_or(usize::MAX)
            .min(bytes.len());
        let end = start.saturating_add(len).min(bytes.len());
        Ok(bytes[start..end].to_vec())
    }

    async fn copy_session(&self, from: SessionId, to: SessionId) -> Result<(), ContextError> {
        let mut files = self.files.lock().unwrap_or_else(PoisonError::into_inner);
        let copies: Vec<_> = files
            .iter()
            .filter(|((s, _), _)| *s == from)
            .map(|((_, c), b)| ((to, c.clone()), b.clone()))
            .collect();
        files.extend(copies);
        Ok(())
    }
}

#[cfg(feature = "fs")]
pub use fs::FsSpill;

#[cfg(feature = "fs")]
mod fs {
    use std::fs::{self, File, OpenOptions};
    use std::io::{self, Read, Seek, SeekFrom, Write};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use sarathi_protocol::SessionId;

    use super::{SpillStore, call_of_file, check_call_id, file_name};
    use crate::error::ContextError;

    /// Spilled output on disk, next to the JSONL session logs: `<root>/<session>/spill/<hex of the call id>`. Give it
    /// the root the session logs are under, and a fork's spill files go where its log is. Its methods run file IO on
    /// Tokio's blocking pool, so they need a Tokio runtime.
    ///
    /// For a fork, call [`SpillStore::copy_session`] before the child's log is written, so a child log never names
    /// output its session does not have.
    #[derive(Debug, Clone)]
    pub struct FsSpill {
        root: Arc<PathBuf>,
    }

    impl FsSpill {
        /// A store under `root`.
        pub fn new(root: impl Into<PathBuf>) -> Self {
            FsSpill {
                root: Arc::new(root.into()),
            }
        }

        /// Where the output of `call` in `session` is stored.
        pub fn path(&self, session: SessionId, call: &str) -> PathBuf {
            dir(&self.root, session).join(file_name(call))
        }

        async fn blocking<T: Send + 'static>(
            &self,
            f: impl FnOnce(&Path) -> Result<T, ContextError> + Send + 'static,
        ) -> Result<T, ContextError> {
            let root = Arc::clone(&self.root);
            tokio::task::spawn_blocking(move || f(&root))
                .await
                .map_err(|e| ContextError::Io(io::Error::other(e)))?
        }
    }

    fn dir(root: &Path, session: SessionId) -> PathBuf {
        root.join(session.to_string()).join("spill")
    }

    /// Makes `dir` (`<root>/<session>/spill`) and every directory above it up to `root`, syncing each parent whose
    /// entry is new, so the directory survives a power loss once this returns.
    fn make_dirs(root: &Path, dir: &Path) -> io::Result<()> {
        if dir.is_dir() {
            return Ok(());
        }
        let mut missing = Vec::new();
        let mut d = dir;
        while !d.is_dir() {
            missing.push(d);
            match d.parent() {
                Some(p) if p.starts_with(root) || p == root => d = p,
                _ => break,
            }
        }
        fs::create_dir_all(dir)?;
        for m in missing {
            if let Some(parent) = m.parent() {
                sync_dir(parent)?;
            }
        }
        Ok(())
    }

    fn sync_dir(dir: &Path) -> io::Result<()> {
        // Opening a directory to sync it is a Unix thing.
        if cfg!(unix) {
            File::open(dir)?.sync_all()?;
        }
        Ok(())
    }

    /// Writes `bytes` durably: a temporary file, synced, renamed over the target, the directory synced.
    fn write_durable(root: &Path, dir: &Path, name: &str, bytes: &[u8]) -> io::Result<()> {
        make_dirs(root, dir)?;
        let tmp = dir.join(format!(".{name}.tmp"));
        let mut f = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp)?;
        f.write_all(bytes)?;
        f.sync_data()?;
        fs::rename(&tmp, dir.join(name))?;
        sync_dir(dir)
    }

    impl SpillStore for FsSpill {
        async fn put(
            &self,
            session: SessionId,
            call: &str,
            bytes: Vec<u8>,
        ) -> Result<(), ContextError> {
            check_call_id(call)?;
            let name = file_name(call);
            self.blocking(move |root| Ok(write_durable(root, &dir(root, session), &name, &bytes)?))
                .await
        }

        async fn read(
            &self,
            session: SessionId,
            call: &str,
            offset: u64,
            len: usize,
        ) -> Result<Vec<u8>, ContextError> {
            check_call_id(call)
                .map_err(|_| ContextError::SpillNotFound(format!("spill:{call}")))?;
            let call = call.to_owned();
            self.blocking(move |root| {
                let mut f = match File::open(dir(root, session).join(file_name(&call))) {
                    Ok(f) => f,
                    Err(e) if e.kind() == io::ErrorKind::NotFound => {
                        return Err(ContextError::SpillNotFound(format!("spill:{call}")));
                    }
                    Err(e) => return Err(e.into()),
                };
                f.seek(SeekFrom::Start(offset))?;
                let mut out = Vec::new();
                f.take(len as u64).read_to_end(&mut out)?;
                Ok(out)
            })
            .await
        }

        async fn copy_session(&self, from: SessionId, to: SessionId) -> Result<(), ContextError> {
            self.blocking(move |root| {
                let src = dir(root, from);
                let entries = match fs::read_dir(&src) {
                    Ok(e) => e,
                    Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
                    Err(e) => return Err(e.into()),
                };
                let dst = dir(root, to);
                for e in entries {
                    let e = e?;
                    let name = e.file_name().to_string_lossy().into_owned();
                    // Temporary files and anything else that is not a spill file stay behind.
                    if call_of_file(&name).is_none() {
                        continue;
                    }
                    write_durable(root, &dst, &name, &fs::read(e.path())?)?;
                }
                Ok(())
            })
            .await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(n: u64) -> SpillRecord {
        SpillRecord {
            handle: "spill:t1".into(),
            bytes: n,
            sha256: Sha256([0; 32]),
        }
    }

    #[test]
    fn the_preview_keeps_whole_first_and_last_lines_within_budget() {
        let out: String = (0..1000).map(|i| format!("line {i}\n")).collect();
        let limits = SpillLimits {
            preview_bytes: 100,
            ..SpillLimits::default()
        };
        let p = preview(&out, &record(out.len() as u64), &limits);
        assert!(p.starts_with("line 0\nline 1\n"), "{p}");
        assert!(p.ends_with("line 999\n"), "{p}");
        assert!(p.contains("spill:t1"));
        let kept = p.len() - p.lines().find(|l| l.starts_with("[output")).unwrap().len();
        assert!(kept <= 100 + 2, "{kept}");
    }

    #[test]
    fn a_single_long_line_is_cut_at_character_boundaries() {
        let out = "é".repeat(10_000);
        let limits = SpillLimits {
            preview_bytes: 101,
            ..SpillLimits::default()
        };
        let p = preview(&out, &record(out.len() as u64), &limits);
        assert!(p.starts_with(&"é".repeat(25)));
        assert!(p.ends_with(&"é".repeat(25)));
    }

    #[test]
    fn only_empty_or_overlong_call_ids_are_refused() {
        assert!(check_call_id("").is_err());
        assert!(check_call_id(&"x".repeat(MAX_CALL_ID + 1)).is_err());
        for good in [
            "toolu_01AbC",
            "functions.edit:0",
            "../x",
            ".a",
            "é",
            &"x".repeat(MAX_CALL_ID),
        ] {
            assert!(check_call_id(good).is_ok(), "{good:?}");
        }
    }

    #[cfg(feature = "fs")]
    #[test]
    fn file_names_encode_call_ids_reversibly() {
        for call in [
            "toolu_AbC",
            "toolu_abc",
            "functions.edit:0",
            "../x",
            ".a",
            "é",
        ] {
            let name = file_name(call);
            assert!(
                name.bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
                "{name}"
            );
            assert_eq!(call_of_file(&name).as_deref(), Some(call));
        }
        assert_ne!(file_name("toolu_AbC"), file_name("toolu_abc"));
        for not in ["", "a", ".2e61.tmp", "zz", "6"] {
            assert_eq!(call_of_file(not), None, "{not:?}");
        }
    }
}
