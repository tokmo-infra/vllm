// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use fastokens::Tokenizer;
use fastokens::added_tokens::Segment;
use serde_json::Value;
use tracing::{debug, info, warn};

use self::shared::SharedPool;

mod shared;

const MIB: usize = 1 << 20;
const MAX_NAMESPACE_BYTES: usize = 128;
const MAX_CACHE_MIB: usize = 16 * 1024;
// Include a conservative allowance for the two indexes, Arc headers and allocations.
const L1_ENTRY_OVERHEAD: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CacheMode {
    FastokensWhole,
    FastokensSegments,
    HuggingFaceWhole,
}

impl CacheMode {
    fn key_tag(self) -> u8 {
        match self {
            Self::FastokensWhole => 0,
            Self::FastokensSegments => 1,
            Self::HuggingFaceWhole => 2,
        }
    }
}

pub(super) struct CacheConfig {
    l1_bytes: usize,
    shared: Option<(PathBuf, usize, String)>,
    epoch_file: Option<PathBuf>,
}

impl CacheConfig {
    pub(super) fn from_env() -> Option<Self> {
        fn budget(name: &str, default: f64, unit_bytes: usize) -> Option<usize> {
            let value = match std::env::var_os(name) {
                None => default,
                Some(value) => match value.to_str().and_then(|value| value.parse::<f64>().ok()) {
                    Some(value) => value,
                    None => {
                        warn!(name, "invalid tokenizer cache budget");
                        return None;
                    }
                },
            };
            let bytes = value * unit_bytes as f64;
            if !bytes.is_finite() || !(0.0..=(MAX_CACHE_MIB * MIB) as f64).contains(&bytes) {
                warn!(name, "invalid tokenizer cache budget");
                return None;
            }
            Some(bytes as usize)
        }

        let gib = 1024 * MIB;
        let mut l1_bytes = budget("VLLM_GLM53_TOKENIZER_CACHE_GIB", 0.0, gib)?;
        if l1_bytes == 0 {
            return None;
        }
        let epoch_file = std::env::var_os("VLLM_GLM53_TOKENIZER_CACHE_EPOCH_FILE")
            .filter(|path| !path.is_empty())
            .map(PathBuf::from);
        let shared = std::env::var_os("VLLM_GLM53_TOKENIZER_SHARED_PATH")
            .filter(|path| !path.is_empty())
            .and_then(|path| {
                l1_bytes = l1_bytes.min(budget("VLLM_GLM53_TOKENIZER_LOCAL_L1_MIB", 64.0, MIB)?);
                let budget = budget("VLLM_GLM53_TOKENIZER_SHARED_GIB", 8.0, gib)?;
                let namespace = match std::env::var("VLLM_RS_TOKENIZER_NAMESPACE") {
                    Ok(value) => value,
                    Err(std::env::VarError::NotPresent) => "glm53".into(),
                    Err(_) => return None,
                };
                if budget == 0 || namespace.is_empty() || namespace.len() > MAX_NAMESPACE_BYTES {
                    return None;
                }
                let path = PathBuf::from(path);
                let mut name = path.file_name()?.to_os_string();
                name.push(".rust-021-v1");
                Some((path.with_file_name(name), budget, namespace))
            });
        if shared.is_none()
            && std::env::var_os("VLLM_GLM53_TOKENIZER_SHARED_PATH").is_some_and(|p| !p.is_empty())
        {
            warn!("invalid shared tokenizer configuration; only L1 is available");
        }
        Some(Self {
            l1_bytes,
            shared,
            epoch_file,
        })
    }

    pub(super) fn build(self, tokenizer_json: &Value, mode: CacheMode) -> Option<EncodeCache> {
        let model = &tokenizer_json["model"];
        if model["type"] != "BPE"
            || !(model["dropout"].is_null() || model["dropout"].as_f64() == Some(0.0))
        {
            warn!("tokenizer cache requires deterministic BPE; using native encoding");
            return None;
        }
        let mode = if mode == CacheMode::FastokensWhole && supports_segment_cache(tokenizer_json) {
            CacheMode::FastokensSegments
        } else {
            mode
        };
        let shared = self.shared.and_then(|(dir, bytes, namespace)| {
            match SharedPool::open(dir, bytes, namespace, tokenizer_json) {
                Ok(pool) => Some(pool),
                Err(error) => {
                    warn!(%error, "tokenizer shared cache unavailable; using native encoding on misses");
                    None
                }
            }
        });
        info!(
            l1_bytes = self.l1_bytes,
            shared = shared.is_some(),
            ?mode,
            epoch_file = self.epoch_file.is_some(),
            "Rust tokenizer cache configured"
        );
        (self.l1_bytes > 0 || shared.is_some())
            .then(|| EncodeCache::new(self.l1_bytes, shared, self.epoch_file, mode))
    }
}

fn current_epoch(path: Option<&Path>) -> Option<String> {
    let Some(path) = path else {
        return Some(String::new());
    };
    // Epoch files are atomically replaced by the operator. Refuse missing,
    // invalid, or oversize generations; never substitute the prior one.
    let mut epoch = String::new();
    File::open(path)
        .ok()?
        .take((MAX_NAMESPACE_BYTES + 2) as u64)
        .read_to_string(&mut epoch)
        .ok()?;
    if epoch.len() > MAX_NAMESPACE_BYTES + 1 {
        return None;
    }
    let epoch = epoch.trim();
    (!epoch.is_empty() && epoch.len() <= MAX_NAMESPACE_BYTES).then(|| epoch.to_owned())
}

fn supports_segment_cache(config: &Value) -> bool {
    fn split_local(pre: &Value) -> bool {
        match pre.get("type").and_then(Value::as_str) {
            Some("Split") => pre["behavior"] == "Isolated" && pre["invert"] == false,
            Some("ByteLevel") => pre["add_prefix_space"] == false,
            Some("Sequence") => pre["pretokenizers"]
                .as_array()
                .is_some_and(|steps| steps.iter().all(split_local)),
            _ => pre.is_null(),
        }
    }

    fn preserves_ids(post: &Value) -> bool {
        match post.get("type").and_then(Value::as_str) {
            Some("ByteLevel") => true,
            Some("TemplateProcessing") => post["single"].as_array().is_some_and(|steps| {
                steps.iter().filter(|v| v.get("Sequence").is_some()).count() == 1
                    && steps
                        .iter()
                        .all(|v| v["Sequence"]["id"] == "A" || v.get("SpecialToken").is_some())
            }),
            _ => post.is_null(),
        }
    }

    let model = &config["model"];
    let normalizer = &config["normalizer"];
    model["type"] == "BPE"
        && (model["dropout"].is_null() || model["dropout"].as_f64() == Some(0.0))
        && (normalizer.is_null()
            || (normalizer["type"] == "Sequence"
                && normalizer["normalizers"].as_array().is_some_and(Vec::is_empty)))
        && config["padding"].is_null() && config["truncation"].is_null()
        && split_local(&config["pre_tokenizer"])
        && preserves_ids(&config["post_processor"])
        && config["added_tokens"].as_array().is_some_and(|tokens| {
            tokens.iter().any(|t| t["special"] == true)
                && tokens.iter().all(|t| {
                    // Mixed raw and normalized passes can give overlapping
                    // literals different priority in HF and fastokens.
                    ["lstrip", "rstrip", "single_word", "normalized"]
                        .iter()
                        .all(|flag| t[flag] != true)
                })
        })
}

pub(super) struct EncodeCache {
    l1: Mutex<LocalCache>,
    epoch_file: Option<PathBuf>,
    mode: CacheMode,
    l1_bytes: usize,
    shared: Option<SharedPool>,
    hits: AtomicU64,
    shared_hits: AtomicU64,
    misses: AtomicU64,
    reused_tokens: AtomicU64,
    encoded_tokens: AtomicU64,
    special_tokens: AtomicU64,
}

impl EncodeCache {
    fn new(
        l1_bytes: usize,
        shared: Option<SharedPool>,
        epoch_file: Option<PathBuf>,
        mode: CacheMode,
    ) -> Self {
        Self {
            l1: Mutex::default(),
            epoch_file,
            mode,
            l1_bytes,
            shared,
            hits: AtomicU64::new(0),
            shared_hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            reused_tokens: AtomicU64::new(0),
            encoded_tokens: AtomicU64::new(0),
            special_tokens: AtomicU64::new(0),
        }
    }

    pub(super) fn encode(
        &self,
        tokenizer: &Tokenizer,
        salt: Option<&str>,
        text: &str,
    ) -> Result<Vec<u32>, fastokens::Error> {
        let Some(epoch) = self.refresh_epoch() else {
            return tokenizer.encode(text);
        };
        if text.is_empty() {
            return tokenizer.encode(text);
        }
        let mut ids = Vec::new();
        // If independent splits cannot be proven exact, only reuse a whole
        // input. This still caches deterministic deployed BPE tokenizers.
        let added = (self.mode == CacheMode::FastokensSegments)
            .then(|| tokenizer.added_tokens())
            .flatten();
        if let Some(added) = added {
            let mut literal_ids = 0;
            for segment in added.split(text) {
                match segment {
                    Segment::Token(id) => { ids.push(id); literal_ids += 1; },
                    Segment::Text("") => {}
                    Segment::Text(text) => {
                        self.append_segment(&epoch, salt, text, &mut ids, || {
                            super::encode_fastokens_ordinary(tokenizer, text)
                        })?;
                    }
                }
            }
            self.special_tokens.fetch_add(literal_ids, Ordering::Relaxed);
        } else {
            self.append_segment(&epoch, salt, text, &mut ids, || tokenizer.encode(text))?;
        }
        Ok(ids)
    }

    pub(super) fn encode_whole<E>(
        &self,
        salt: Option<&str>,
        text: &str,
        encode: impl FnOnce() -> Result<Vec<u32>, E>,
    ) -> Result<Vec<u32>, E> {
        let Some(epoch) = self.refresh_epoch() else {
            return encode();
        };
        if text.is_empty() {
            return encode();
        }
        let mut ids = Vec::new();
        self.append_segment(&epoch, salt, text, &mut ids, encode)?;
        Ok(ids)
    }

    fn refresh_epoch(&self) -> Option<String> {
        if self.epoch_file.is_none() {
            return Some(String::new());
        }
        let mut cache = self.l1.lock().ok()?;
        let epoch = current_epoch(self.epoch_file.as_deref());
        if epoch != cache.epoch {
            *cache = LocalCache {
                epoch: epoch.clone(),
                ..LocalCache::default()
            };
        }
        epoch
    }

    fn append_segment<E>(
        &self,
        epoch: &str,
        salt: Option<&str>,
        text: &str,
        ids: &mut Vec<u32>,
        encode: impl FnOnce() -> Result<Vec<u32>, E>,
    ) -> Result<(), E> {
        fn record(counter: &AtomicU64, tier: &'static str) {
            let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
            if count == 1 || count % 1024 == 0 {
                debug!(tier, count, "Rust tokenizer cache observations");
            }
        }
        // Escape NUL-leading unsalted text so arbitrary strings cannot collide
        // with scoped keys. The common, unsalted path borrows the input.
        let scoped;
        let text = match salt {
            Some(s) => { scoped = format!("\0s{}:{s}{text}", s.len()); &scoped }
            None if text.starts_with('\0') => { scoped = format!("\0u{text}"); &scoped }
            None => text,
        };
        if self.l1_bytes > 0
            && let Some(hit) = self.l1.lock().ok().and_then(|mut cache| {
                (self.epoch_file.is_none() || cache.epoch.as_deref() == Some(epoch))
                    .then(|| cache.get(text))
                    .flatten()
            })
        {
            record(&self.hits, "l1_hit");
            self.reused_tokens.fetch_add(hit.len() as u64, Ordering::Relaxed);
            ids.extend_from_slice(&hit);
            return Ok(());
        }

        // Whole-input and ordinary-segment ids must never alias in L2.
        let shared_key = self
            .shared
            .as_ref()
            .and_then(|pool| pool.key(epoch, self.mode.key_tag(), text).map(|key| (pool, key)));
        let shared_hit = shared_key.as_ref().and_then(|(pool, key)| pool.get(key));
        let encoded = match shared_hit {
            Some(hit) => {
                self.reused_tokens.fetch_add(hit.len() as u64, Ordering::Relaxed);
                record(&self.shared_hits, "shared_hit");
                hit
            }
            None => {
                record(&self.misses, "miss");
                let result = encode()?;
                self.encoded_tokens.fetch_add(result.len() as u64, Ordering::Relaxed);
                if let Some((pool, key)) = shared_key {
                    pool.put(&key, &result);
                }
                result
            }
        };
        ids.extend_from_slice(&encoded);
        if let Some(size) = l1_entry_bytes(text, &encoded)
            && size <= self.l1_bytes
            && let Ok(mut cache) = self.l1.lock()
            && (self.epoch_file.is_none() || cache.epoch.as_deref() == Some(epoch))
        {
            cache.put(text, encoded, size, self.l1_bytes);
        }
        Ok(())
    }

    pub(super) fn stats(&self) -> super::EncodeCacheStats {
        let bytes = self.l1.lock().map_or(0, |cache| cache.bytes);
        super::EncodeCacheStats {
            l1_hits: self.hits.load(Ordering::Relaxed),
            shared_hits: self.shared_hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            reused_tokens: self.reused_tokens.load(Ordering::Relaxed),
            encoded_tokens: self.encoded_tokens.load(Ordering::Relaxed),
            special_tokens: self.special_tokens.load(Ordering::Relaxed),
            l1_bytes: bytes,
        }
    }
}

fn l1_entry_bytes(text: &str, ids: &[u32]) -> Option<usize> {
    text.len()
        .checked_add(std::mem::size_of_val(ids))?
        .checked_add(L1_ENTRY_OVERHEAD)
}

struct Entry {
    ids: Arc<[u32]>,
    age: u64,
    size: usize,
}

#[derive(Default)]
struct LocalCache {
    epoch: Option<String>,
    by_text: HashMap<Arc<str>, Entry>,
    oldest: BTreeMap<u64, Arc<str>>,
    age: u64,
    bytes: usize,
}

impl LocalCache {
    fn get(&mut self, text: &str) -> Option<Arc<[u32]>> {
        let hit = self.by_text.get_mut(text)?;
        let key = self.oldest.remove(&hit.age).expect("L1 age index");
        self.age += 1;
        hit.age = self.age;
        self.oldest.insert(self.age, key);
        Some(hit.ids.clone())
    }

    fn put(&mut self, text: &str, ids: Vec<u32>, size: usize, budget: usize) {
        if self.by_text.contains_key(text) {
            return;
        }
        while self.bytes > budget - size {
            let (_, key) = self.oldest.pop_first().expect("L1 byte accounting");
            self.bytes -= self.by_text.remove(&key).expect("L1 text index").size;
        }
        self.age += 1;
        let key: Arc<str> = text.into();
        self.oldest.insert(self.age, key.clone());
        self.by_text.insert(
            key,
            Entry {
                ids: ids.into(),
                age: self.age,
                size,
            },
        );
        self.bytes += size;
    }
}

#[cfg(test)]
mod tests;
