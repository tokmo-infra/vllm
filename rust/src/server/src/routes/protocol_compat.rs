// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;

use serde_json::Value;

use crate::error::ApiError;

/// Mooncake publishes KV on length completion on the internal one-token P leg.
pub(super) fn is_one_token_kv_producer(
    limit: Option<u32>,
    transfer: Option<&HashMap<String, Value>>,
) -> bool {
    limit == Some(1)
        && transfer.is_some_and(|kv| {
            kv.get("do_remote_decode").and_then(Value::as_bool) == Some(true)
                && kv.get("transfer_id").and_then(Value::as_str).is_some_and(|id| !id.is_empty())
        })
}

/// Validate the wire shape only. The existing media loader enforces access and
/// decoding policy; the route never fetches or rewrites the supplied URL.
pub(super) fn validate_video_url(url: &str, param: &'static str) -> Result<(), ApiError> {
    let valid = ["https://", "http://"].iter().any(|prefix| {
        url.strip_prefix(*prefix)
            .is_some_and(|rest| !rest.split('/').next().unwrap_or_default().is_empty())
    }) || url.strip_prefix("file:///").is_some_and(|path| !path.is_empty())
        || url
            .strip_prefix("data:video/")
            .and_then(|video| video.split_once(";base64,"))
            .is_some_and(|(kind, data)| {
                !kind.is_empty()
                    && kind.bytes().all(|b| b.is_ascii_alphanumeric() || b"_.+-".contains(&b))
                    && !data.is_empty()
            });
    if !valid || url.chars().any(char::is_whitespace) {
        return Err(ApiError::invalid_request(
            "Video requires an HTTP(S), file or base64 video URL.",
            Some(param),
        ));
    }
    Ok(())
}
