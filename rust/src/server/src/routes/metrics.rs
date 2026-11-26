// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::fmt::Write as _;
use std::sync::Arc;
use axum::extract::State;
use axum::http::header::CONTENT_TYPE;
use crate::state::AppState;
use vllm_text::tokenizer::EncodeCacheStats;
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use thiserror_ext::AsReport;
use vllm_metrics::METRICS;

const OPENMETRICS_CONTENT_TYPE: &str = "application/openmetrics-text; version=1.0.0; charset=utf-8";

pub async fn scrape(State(state): State<Arc<AppState>>) -> Response {
    render(state.chat.text().request_processor().tokenizer().cache_stats())
}

// Reading a snapshot here avoids taking a global metrics lock during encode.
pub(super) fn render(stats: Option<EncodeCacheStats>) -> Response {
    match METRICS.render() {
        Ok(mut body) => {
            if let Some(stats) = stats {
                if let Some(prefix) = body.strip_suffix("# EOF\n") { body.truncate(prefix.len()); }
                body.push_str("# HELP vllm_tokenizer_cache_operations Tokenizer encode cache lookups by outcome.\n# TYPE vllm_tokenizer_cache_operations counter\n");
                for (tier, value) in [("l1_hit", stats.l1_hits), ("shared_hit", stats.shared_hits), ("miss", stats.misses)] {
                    writeln!(body, "vllm_tokenizer_cache_operations_total{{tier=\"{tier}\"}} {value}").expect("write into String");
                }
                body.push_str("# HELP vllm:tokenizer_cache_tokens Tokenizer output tokens by encode path.\n# TYPE vllm:tokenizer_cache_tokens counter\n");
                for (source, n) in [("reused_tokens", stats.reused_tokens), ("encoded_tokens", stats.encoded_tokens), ("special_tokens", stats.special_tokens)] {
                    writeln!(body, "vllm:tokenizer_cache_tokens_total{{source=\"{source}\"}} {n}").expect("write into String");
                }
                body.push_str("# EOF\n");
            }
            (
            [(
                CONTENT_TYPE,
                HeaderValue::from_static(OPENMETRICS_CONTENT_TYPE),
            )],
            body,
        )
            .into_response()
        },

        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to render metrics: {}", error.as_report()),
        )
            .into_response(),
    }
}
