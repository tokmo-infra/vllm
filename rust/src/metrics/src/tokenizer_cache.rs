// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::registry::Registry;

use crate::{Family, U64Counter, U64Gauge};

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct TokenizerCacheSource {
    pub source: &'static str,
}

/// Process-wide accounting for bounded tokenizer segment caches.
pub struct TokenizerCacheMetrics {
    pub bytes: U64Gauge,
    pub max_bytes: U64Gauge,
    pub epoch: U64Gauge,
    pub tokens: Family<TokenizerCacheSource, U64Counter>,
}

impl TokenizerCacheMetrics {
    pub(crate) fn register(registry: &mut Registry) -> Self {
        let registry = registry.sub_registry_with_label(("api_process".into(), "0".into()));
        let bytes = U64Gauge::default();
        let max_bytes = U64Gauge::default();
        let epoch = U64Gauge::default();
        let tokens = Family::default();
        registry.register(
            "vllm:tokenizer_cache_bytes",
            "Charged retained segment-cache bytes.",
            bytes.clone(),
        );
        registry.register(
            "vllm:tokenizer_cache_max_bytes",
            "Combined segment-cache byte limits.",
            max_bytes.clone(),
        );
        registry.register(
            "vllm:tokenizer_cache_epoch",
            "Acknowledged 48-bit SHA256 of the cache epoch.",
            epoch.clone(),
        );
        registry.register(
            "vllm:tokenizer_cache_tokens",
            "Tokens reused, encoded, or classified as special.",
            tokens.clone(),
        );
        Self {
            bytes,
            max_bytes,
            epoch,
            tokens,
        }
    }
}
