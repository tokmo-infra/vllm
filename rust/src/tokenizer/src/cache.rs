// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Weak};
use std::time::Duration;

use parking_lot::Mutex;
use sha2::{Digest, Sha256};
use vllm_metrics::{METRICS, TokenizerCacheSource};

use crate::{DynTokenizer, IncrementalDecoder, Result, Tokenizer};

struct Entry {
    tokens: Arc<[u32]>,
    special: usize,
    stamp: u64,
    charge: usize,
}

#[derive(Default)]
struct State {
    // Separate namespaces preserve encode_ordinary and added-token policies.
    entries: [HashMap<Arc<str>, Entry>; 3],
    lru: BTreeMap<u64, (usize, Arc<str>)>,
    bytes: usize,
    clock: u64,
}

struct Core {
    state: Mutex<State>,
    max_bytes: usize,
}

impl Core {
    fn clear(&self) {
        let mut state = self.state.lock();
        METRICS.tokenizer_cache.bytes.dec_by(state.bytes as u64);
        *state = State::default();
    }
}

impl Drop for Core {
    fn drop(&mut self) {
        METRICS.tokenizer_cache.bytes.dec_by(self.state.get_mut().bytes as u64);
        METRICS.tokenizer_cache.max_bytes.dec_by(self.max_bytes as u64);
    }
}

/// Model-scoped, byte-bounded LRU that preserves each native encode policy.
///
/// Entries charge UTF-8 bytes, token IDs, and 256 bytes of bookkeeping.
/// This bounds charged cache storage, rather than total process RSS.
pub struct CachedTokenizer {
    inner: DynTokenizer,
    core: Arc<Core>,
}

impl CachedTokenizer {
    pub fn new(inner: DynTokenizer, max_bytes: usize, epoch_file: Option<PathBuf>) -> Self {
        let core = Arc::new(Core {
            state: Mutex::default(),
            max_bytes,
        });
        METRICS.tokenizer_cache.max_bytes.inc_by(max_bytes as u64);
        Self::record("encoded_tokens", 0, 0);
        Self::record("reused_tokens", 0, 0);
        if let Some(path) = epoch_file {
            watch_epoch(Arc::downgrade(&core), path);
        }
        tracing::info!(max_bytes, "Tokenizer segment cache enabled");
        Self { inner, core }
    }

    fn record(source: &'static str, tokens: usize, special: usize) {
        METRICS
            .tokenizer_cache
            .tokens
            .get_or_create(&TokenizerCacheSource { source })
            .inc_by((tokens - special) as u64);
        METRICS
            .tokenizer_cache
            .tokens
            .get_or_create(&TokenizerCacheSource {
                source: "special_tokens",
            })
            .inc_by(special as u64);
    }

    fn encode_cached(&self, text: &str, mode: usize) -> Result<Vec<u32>> {
        {
            let mut state = self.core.state.lock();
            if let Some((key, entry)) = state.entries[mode].get_key_value(text) {
                let (key, tokens, special, stamp) = (
                    key.clone(),
                    entry.tokens.clone(),
                    entry.special,
                    entry.stamp,
                );
                state.lru.remove(&stamp);
                state.clock += 1;
                let stamp = state.clock;
                state.entries[mode].get_mut(text).expect("entry is locked").stamp = stamp;
                state.lru.insert(stamp, (mode, key));
                drop(state);
                Self::record("reused_tokens", tokens.len(), special);
                return Ok(tokens.to_vec());
            }
        }
        // Tokenization is outside the lock; unrelated requests remain parallel.
        let tokens = match mode {
            0 => self.inner.encode_ordinary(text)?,
            1 => self.inner.encode(text, false)?,
            _ => self.inner.encode(text, true)?,
        };
        let special = if mode == 0 {
            0
        } else {
            tokens.iter().filter(|&&id| self.inner.is_special_id(id)).count()
        };
        Self::record("encoded_tokens", tokens.len(), special);
        let charge = text.len().saturating_add(tokens.len().saturating_mul(4)).saturating_add(256);
        if !text.is_empty() && charge <= self.core.max_bytes {
            let mut state = self.core.state.lock();
            // A concurrent miss may already have installed the same entry.
            if !state.entries[mode].contains_key(text) {
                while state.bytes > self.core.max_bytes - charge {
                    let (_, (old_mode, old_key)) =
                        state.lru.pop_first().expect("charged entries have LRU keys");
                    let old = state.entries[old_mode]
                        .remove(old_key.as_ref())
                        .expect("LRU key has an entry");
                    state.bytes -= old.charge;
                    METRICS.tokenizer_cache.bytes.dec_by(old.charge as u64);
                }
                let key: Arc<str> = Arc::from(text);
                state.clock += 1;
                let stamp = state.clock;
                state.lru.insert(stamp, (mode, key.clone()));
                state.entries[mode].insert(
                    key,
                    Entry {
                        tokens: tokens.clone().into(),
                        special,
                        stamp,
                        charge,
                    },
                );
                state.bytes += charge;
                METRICS.tokenizer_cache.bytes.inc_by(charge as u64);
            }
        }
        Ok(tokens)
    }
}

fn watch_epoch(core: Weak<Core>, path: PathBuf) {
    std::thread::spawn(move || {
        let mut previous = None;
        loop {
            let Some(core) = core.upgrade() else { break };
            if let Ok(epoch) = std::fs::read_to_string(&path) {
                let epoch = epoch.trim();
                if previous.as_deref() != Some(epoch) {
                    core.clear();
                    let digest = Sha256::digest(epoch.as_bytes());
                    let mut bytes = [0u8; 8];
                    bytes[2..].copy_from_slice(&digest[..6]);
                    METRICS.tokenizer_cache.epoch.set(u64::from_be_bytes(bytes));
                    previous = Some(epoch.to_owned());
                }
            }
            drop(core);
            std::thread::sleep(Duration::from_secs(1));
        }
    });
}

impl Tokenizer for CachedTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        self.encode_cached(text, if add_special_tokens { 2 } else { 1 })
    }
    fn encode_ordinary(&self, text: &str) -> Result<Vec<u32>> {
        self.encode_cached(text, 0)
    }
    fn decode(&self, ids: &[u32], skip: bool) -> Result<String> {
        self.inner.decode(ids, skip)
    }
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }
    fn added_vocab(&self) -> &[(String, u32)] {
        self.inner.added_vocab()
    }
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn is_special_id(&self, id: u32) -> bool {
        self.inner.is_special_id(id)
    }
    fn create_decode_stream(
        &self,
        ids: &[u32],
        skip: bool,
        min: usize,
    ) -> Box<dyn IncrementalDecoder + '_> {
        self.inner.create_decode_stream(ids, skip, min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::TestTokenizer;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Counted {
        inner: TestTokenizer,
        calls: Arc<AtomicUsize>,
    }

    impl Tokenizer for Counted {
        fn encode(&self, text: &str, special: bool) -> Result<Vec<u32>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.inner.encode(text, special)
        }
        fn encode_ordinary(&self, text: &str) -> Result<Vec<u32>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.inner.encode_ordinary(text)
        }
        fn decode(&self, ids: &[u32], skip: bool) -> Result<String> {
            self.inner.decode(ids, skip)
        }
        fn token_to_id(&self, token: &str) -> Option<u32> {
            self.inner.token_to_id(token)
        }
        fn id_to_token(&self, id: u32) -> Option<String> {
            self.inner.id_to_token(id)
        }
    }

    #[test]
    fn repeat_skips_bpe_and_clear_recomputes_the_same_ids() {
        let calls = Arc::new(AtomicUsize::new(0));
        let cache = CachedTokenizer::new(
            Arc::new(Counted {
                inner: TestTokenizer::new(),
                calls: calls.clone(),
            }),
            8192,
            None,
        );
        let first = cache.encode_ordinary("same prefix 你好🙂").unwrap();
        assert_eq!(cache.encode_ordinary("same prefix 你好🙂").unwrap(), first);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        cache.core.clear();
        assert_eq!(cache.encode_ordinary("same prefix 你好🙂").unwrap(), first);
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn ordinary_and_control_tokens_never_share_an_entry() {
        let inner: DynTokenizer = Arc::new(TestTokenizer::new().with_special_token("<think>", 256));
        let cache = CachedTokenizer::new(inner.clone(), 8192, None);
        for _ in 0..2 {
            for special in [false, true] {
                assert_eq!(
                    cache.encode("<think>你好🙂", special).unwrap(),
                    inner.encode("<think>你好🙂", special).unwrap()
                );
            }
            assert_eq!(
                cache.encode_ordinary("<think>你好🙂").unwrap(),
                inner.encode_ordinary("<think>你好🙂").unwrap()
            );
        }
        assert_eq!(cache.core.state.lock().lru.len(), 3);
    }

    #[test]
    fn hits_return_independent_ids_and_eviction_respects_byte_limit() {
        let cache = CachedTokenizer::new(Arc::new(TestTokenizer::new()), 600, None);
        let mut first = cache.encode_ordinary("one").unwrap();
        first.clear();
        assert_eq!(cache.encode_ordinary("one").unwrap(), vec![111, 110, 101]);
        for text in ["two", "three", "four", "one"] {
            cache.encode_ordinary(text).unwrap();
            assert!(cache.core.state.lock().bytes <= 600);
        }
        cache.encode_ordinary(&"x".repeat(1000)).unwrap();
        assert!(cache.core.state.lock().bytes <= 600);
        cache.core.clear();
        assert_eq!(cache.core.state.lock().bytes, 0);
    }
}
