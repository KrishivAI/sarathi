// Derived from Ultrafast Gateway @ c9c3eaf (MIT); see PROVENANCE.md
// SPDX-License-Identifier: Apache-2.0
//! A scripted HTTP server on a free port that records what it was sent.
#![allow(
    dead_code,
    reason = "shared test helpers: each test crate uses a different part"
)]
#![expect(clippy::unwrap_used, reason = "test server")]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

#[derive(Clone)]
pub(crate) struct Script {
    pub(crate) status: u16,
    pub(crate) headers: Vec<(String, String)>,
    /// Body parts, each sent as its own chunk with a pause after it.
    pub(crate) parts: Vec<Vec<u8>>,
    /// False: the connection is cut in the middle of the last chunk.
    pub(crate) clean_end: bool,
    /// Never answer at all.
    pub(crate) hang: bool,
    /// After the parts, keep the connection open and silent.
    pub(crate) stall: bool,
    /// Send `content-length` of this size instead of chunking (the parts may be shorter).
    pub(crate) declared_len: Option<usize>,
}

impl Script {
    pub(crate) fn json(status: u16, body: &str) -> Self {
        Script {
            status,
            headers: vec![("content-type".into(), "application/json".into())],
            parts: vec![body.as_bytes().to_vec()],
            clean_end: true,
            hang: false,
            stall: false,
            declared_len: None,
        }
    }

    pub(crate) fn sse(parts: Vec<Vec<u8>>) -> Self {
        Script {
            status: 200,
            headers: vec![("content-type".into(), "text/event-stream".into())],
            parts,
            clean_end: true,
            hang: false,
            stall: false,
            declared_len: None,
        }
    }

    pub(crate) fn header(mut self, k: &str, v: &str) -> Self {
        self.headers.push((k.into(), v.into()));
        self
    }

    pub(crate) fn stall(mut self) -> Self {
        self.stall = true;
        self
    }

    pub(crate) fn declared_len(mut self, n: usize) -> Self {
        self.declared_len = Some(n);
        self
    }

    pub(crate) fn cut(mut self) -> Self {
        self.clean_end = false;
        self
    }
}

pub(crate) struct Server {
    pub(crate) url: String,
    requests: Arc<Mutex<Vec<String>>>,
    raw: Arc<Mutex<Vec<String>>>,
    bodies: Arc<Mutex<Vec<Vec<u8>>>>,
}

impl Server {
    /// Every request received so far, as head + body text, lower-cased heads.
    pub(crate) fn requests(&self) -> Vec<String> {
        self.requests.lock().unwrap().clone()
    }

    /// The one request received, head + body with the head's case kept.
    pub(crate) fn only_raw(&self) -> String {
        let r = self.raw.lock().unwrap().clone();
        assert_eq!(r.len(), 1, "expected exactly one request: {r:?}");
        r[0].clone()
    }

    /// The bodies of every request received so far, as exact bytes.
    pub(crate) fn bodies(&self) -> Vec<Vec<u8>> {
        self.bodies.lock().unwrap().clone()
    }

    pub(crate) fn only(&self) -> String {
        let r = self.requests();
        assert_eq!(r.len(), 1, "expected exactly one request: {r:?}");
        r[0].clone()
    }
}

pub(crate) async fn serve(script: Script) -> Server {
    serve_seq(vec![script]).await
}

/// Answers the n-th connection with the n-th script (the last one again once they run out).
pub(crate) async fn serve_seq(scripts: Vec<Script>) -> Server {
    assert!(!scripts.is_empty(), "at least one script");
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let requests = Arc::new(Mutex::new(Vec::new()));
    let log = requests.clone();
    let raw = Arc::new(Mutex::new(Vec::new()));
    let raw_log = raw.clone();
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let body_log = bodies.clone();
    tokio::spawn(async move {
        let mut n = 0usize;
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let _ = sock.set_nodelay(true);
            let script = scripts[n.min(scripts.len() - 1)].clone();
            n += 1;
            let log = log.clone();
            let raw_log = raw_log.clone();
            let body_log = body_log.clone();
            tokio::spawn(async move {
                let mut buf = Vec::new();
                let mut tmp = [0u8; 4096];
                loop {
                    let n = sock.read(&mut tmp).await.unwrap_or(0);
                    if n == 0 {
                        return;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                    if let Some(end) = find(&buf, b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&buf[..end]).to_string();
                        let len = head
                            .to_lowercase()
                            .lines()
                            .find_map(|l| l.strip_prefix("content-length:").map(str::to_string))
                            .and_then(|v| v.trim().parse::<usize>().ok())
                            .unwrap_or(0);
                        if buf.len() >= end + 4 + len {
                            body_log
                                .lock()
                                .unwrap()
                                .push(buf[end + 4..end + 4 + len].to_vec());
                            let body = String::from_utf8_lossy(&buf[end + 4..end + 4 + len]);
                            raw_log
                                .lock()
                                .unwrap()
                                .push(format!("{head}\r\n\r\n{body}"));
                            log.lock()
                                .unwrap()
                                .push(format!("{}\r\n\r\n{body}", head.to_lowercase()));
                            break;
                        }
                    }
                }
                if script.hang {
                    tokio::time::sleep(Duration::from_secs(30)).await;
                    return;
                }
                let mut out = format!("HTTP/1.1 {} X\r\n", script.status);
                for (k, v) in &script.headers {
                    out.push_str(&format!("{k}: {v}\r\n"));
                }
                if let Some(n) = script.declared_len {
                    out.push_str(&format!("content-length: {n}\r\nconnection: close\r\n\r\n"));
                    let _ = sock.write_all(out.as_bytes()).await;
                    for part in &script.parts {
                        let _ = sock.write_all(part).await;
                    }
                    let _ = sock.flush().await;
                    tokio::time::sleep(Duration::from_secs(30)).await;
                    return;
                }
                out.push_str("transfer-encoding: chunked\r\nconnection: close\r\n\r\n");
                let _ = sock.write_all(out.as_bytes()).await;
                for (i, part) in script.parts.iter().enumerate() {
                    let cut_here = !script.clean_end && i + 1 == script.parts.len();
                    let declared = part.len() + if cut_here { 50 } else { 0 };
                    let _ = sock.write_all(format!("{declared:x}\r\n").as_bytes()).await;
                    let _ = sock.write_all(part).await;
                    if !cut_here {
                        let _ = sock.write_all(b"\r\n").await;
                    }
                    let _ = sock.flush().await;
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
                if script.stall {
                    tokio::time::sleep(Duration::from_secs(30)).await;
                    return;
                }
                if script.clean_end {
                    let _ = sock.write_all(b"0\r\n\r\n").await;
                }
                let _ = sock.shutdown().await;
            });
        }
    });
    Server {
        url,
        requests,
        raw,
        bodies,
    }
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// A URL where nothing listens.
pub(crate) async fn dead_url() -> String {
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", l.local_addr().unwrap());
    drop(l);
    url
}

pub(crate) const OPENAI_CHAT: &str = r#"{"id":"c1","model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}"#;
pub(crate) const ANTHROPIC_CHAT: &str = r#"{"id":"m1","model":"claude-sonnet-5","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":2}}"#;
