// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Anthropic Messages over the native chat renderer and engine.

mod request;
mod streaming;

use std::sync::Arc;

use axum::Json;
use axum::extract::{Request, State, rejection::JsonRejection};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header::AUTHORIZATION};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use serde_json::Value;
use uuid::Uuid;
use vllm_chat::{AssistantContentBlock, FinishReason};
use vllm_engine_core_client::protocol::output::StopReason;
use vllm_llm::TokenUsage;

use self::request::{MessagesPrompt, MessagesRequest};
use crate::error::{ApiError, chat_submit_error, server_error, text_submit_error};
use crate::routes::openai::chat_completions::lower_chat_request;
use crate::state::AppState;
use crate::utils::{resolve_request_context, sse_response};

/// The Anthropic SDK sends x-api-key. Delegate verification to the shared
/// middleware and leave other routes and explicit Bearer credentials intact.
pub(super) async fn adapt_auth(mut request: Request, next: Next) -> Response {
    if !matches!(
        request.uri().path(),
        "/v1/messages" | "/v1/messages/count_tokens"
    ) {
        return next.run(request).await;
    }
    if !request.headers().contains_key(AUTHORIZATION)
        && let Some(key) = request.headers().get("x-api-key")
        && !key.is_empty()
    {
        let mut bearer = b"Bearer ".to_vec();
        bearer.extend_from_slice(key.as_bytes());
        if let Ok(mut bearer) = HeaderValue::from_bytes(&bearer) {
            bearer.set_sensitive(true);
            request.headers_mut().insert(AUTHORIZATION, bearer);
        }
    }
    let response = next.run(request).await;
    if response.status() == StatusCode::UNAUTHORIZED {
        let status = response.status();
        return (status, Json(ErrorBody::new(status, "Unauthorized".into()))).into_response();
    }
    response
}

pub(super) async fn messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Json<MessagesRequest>, JsonRejection>,
) -> Result<Response, MessagesError> {
    let Json(body) = body.map_err(MessagesError::from)?;
    let (request, producer) = body.lower()?;
    let stream_requested = request.stream;
    let context = resolve_request_context(&headers, None);
    let model = state
        .resolve_model_with_loras(request.model.as_deref())
        .await;
    let request = lower_chat_request(request, &model, context)?;
    if request.sampling_params.max_tokens.is_some_and(|n| n == 0 || n > state.chat.text().request_processor().max_model_len()) {
        return Err(ApiError::invalid_request("max_tokens must be positive and within the model window", Some("max_tokens")).into());
    }
    let id = request.request_id.clone();
    let model = model.response_model();
    let options = state.api_server_options;
    let chat = state
        .chat
        .chat(request)
        .await
        .map_err(|error| chat_submit_error("failed to submit Messages request", error))?;
    if stream_requested {
        return Ok(sse_response(
            streaming::messages_stream(chat, id, model, options.enable_prompt_tokens_details),
            options.sse_keep_alive_interval,
        ));
    }

    let result = chat
        .collect_message()
        .await
        .map_err(|error| chat_submit_error("failed to collect Messages result", error))?;
    if let FinishReason::Error = &result.finish_reason {
        return Err(server_error!("The model failed to generate a response.").into());
    }
    let mut content = Vec::new();
    let mut tool_used = false;
    let mut tool_complete = false;
    for block in &result.message.content {
        content.push(match block {
            AssistantContentBlock::Text { text } if !text.is_empty() => {
                OutputBlock::Text { text: text.clone() }
            }
            AssistantContentBlock::Reasoning { text } if !text.is_empty() => {
                OutputBlock::Thinking {
                    thinking: text.clone(),
                    signature: signature(),
                }
            }
            AssistantContentBlock::ToolCall(call) => {
                tool_used = true;
                let Ok(input) =
                    serde_json::from_str::<serde_json::Map<String, Value>>(&call.arguments)
                else {
                    if matches!(
                        result.finish_reason,
                        FinishReason::Length | FinishReason::Stop(Some(StopReason::Text(_)))
                    ) {
                        tool_complete = false;
                        continue;
                    }
                    return Err(server_error!("The model emitted invalid tool arguments.").into());
                };
                tool_complete = true;
                OutputBlock::ToolUse {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    input,
                }
            }
            _ => continue,
        });
    }
    if content.is_empty() {
        content.push(OutputBlock::Text {
            text: String::new(),
        });
    }
    let (stop_reason, stop_sequence) = stop_fields(&result.finish_reason, tool_used, tool_complete);
    Ok(Json(MessageResponse {
        id,
        model,
        content,
        stop_reason,
        stop_sequence,
        usage: Usage::from_engine(
            result.usage.engine,
            options.enable_prompt_tokens_details,
            producer,
        ),
        kv_transfer_params: result.kv_transfer_params,
        ec_transfer_params: result.ec_transfer_params,
        kind: "message",
        role: "assistant",
    })
    .into_response())
}

pub(super) async fn count_tokens(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Json<MessagesPrompt>, JsonRejection>,
) -> Result<Response, MessagesError> {
    let Json(body) = body.map_err(MessagesError::from)?;
    let request = body.lower_count()?;
    let context = resolve_request_context(&headers, None);
    let model = state
        .resolve_model_with_loras(request.model.as_deref())
        .await;
    let request = lower_chat_request(request, &model, context)?;
    // Use the same rendering and media expansion as generation but do not
    // resolve output parsers or submit a generation request.
    let text_request = state
        .chat
        .request_processor()
        .prepare_for_tokenization(request)
        .await
        .map_err(|error| chat_submit_error("Messages count_tokens failed", error))?;
    let input_tokens = state
        .chat
        .text()
        .request_processor()
        .tokenize(text_request)
        .map_err(|error| text_submit_error("Messages count_tokens failed", error))?
        .len();
    Ok(Json(CountTokensResponse {
        input_tokens,
        context_management: ContextManagement {
            original_input_tokens: input_tokens,
        },
    })
    .into_response())
}

#[derive(Serialize)]
struct CountTokensResponse {
    input_tokens: usize,
    context_management: ContextManagement,
}

#[derive(Serialize)]
struct ContextManagement {
    original_input_tokens: usize,
}

#[derive(Serialize)]
struct MessageResponse {
    id: String,
    model: String,
    #[serde(rename = "type")]
    kind: &'static str,
    role: &'static str,
    content: Vec<OutputBlock>,
    // Python omits None on the JSON route but sends explicit nulls in SSE.
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequence: Option<String>,
    usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    kv_transfer_params: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ec_transfer_params: Option<Value>,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OutputBlock {
    Text {
        text: String,
    },
    Thinking {
        thinking: String,
        signature: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Map<String, Value>,
    },
}

#[derive(Serialize)]
struct Usage {
    input_tokens: usize,
    output_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_read_input_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_creation_input_tokens: Option<usize>,
    // Only the private one-token producer response exposes these names, for
    // the PD router's shared cache accounting contract.
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_tokens_details: Option<ProducerPromptDetails>,
}

#[derive(Serialize)]
struct ProducerPromptDetails {
    cached_tokens: usize,
}

impl Usage {
    fn from_engine(usage: TokenUsage, cache_details: bool, producer: bool) -> Self {
        let cached = usage.cached_token_count;
        let created = usage.cache_creation_token_count;
        let producer_details = cache_details && producer;
        Self {
            input_tokens: if cache_details {
                usage.prompt_token_count.saturating_sub(cached).saturating_sub(created)
            } else {
                usage.prompt_token_count
            },
            output_tokens: usage.output_token_count,
            cache_read_input_tokens: cache_details.then_some(cached),
            cache_creation_input_tokens: cache_details.then_some(created),
            prompt_tokens: producer_details.then_some(usage.prompt_token_count),
            prompt_tokens_details: producer_details.then_some(ProducerPromptDetails {
                cached_tokens: cached,
            }),
        }
    }
}

fn stop_fields(
    reason: &FinishReason,
    tool_used: bool,
    tool_complete: bool,
) -> (Option<&'static str>, Option<String>) {
    match reason {
        FinishReason::Length => (Some("max_tokens"), None),
        FinishReason::Stop(_) | FinishReason::Repetition(_) => {
            if tool_complete {
                return (Some("tool_use"), None);
            }
            if let Some(StopReason::Text(stop)) = reason.as_stop_reason() {
                return (Some("stop_sequence"), Some(stop.clone()));
            }
            (Some(if tool_used { "tool_use" } else { "end_turn" }), None)
        }
        FinishReason::Abort | FinishReason::Error => (None, None),
    }
}

fn signature() -> String {
    Uuid::new_v4().simple().to_string()
}

#[derive(Serialize)]
pub(super) struct ErrorBody {
    #[serde(rename = "type")]
    kind: &'static str,
    error: ErrorDetail,
}

#[derive(Serialize)]
struct ErrorDetail {
    #[serde(rename = "type")]
    kind: &'static str,
    message: String,
}

impl ErrorBody {
    pub(super) fn new(status: StatusCode, message: String) -> Self {
        Self {
            kind: "error",
            error: ErrorDetail {
                kind: match status {
                    StatusCode::UNAUTHORIZED => "authentication_error",
                    StatusCode::FORBIDDEN => "permission_error",
                    StatusCode::NOT_FOUND => "not_found_error",
                    StatusCode::PAYLOAD_TOO_LARGE => "request_too_large",
                    StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
                    StatusCode::SERVICE_UNAVAILABLE => "overloaded_error",
                    status if status.is_server_error() => "api_error",
                    _ => "invalid_request_error",
                },
                message,
            },
        }
    }
}

pub(super) struct MessagesError {
    status: StatusCode,
    message: String,
}

impl From<ApiError> for MessagesError {
    fn from(error: ApiError) -> Self {
        Self {
            status: error.status_code(),
            message: error.to_error_response().error.message,
        }
    }
}

impl From<JsonRejection> for MessagesError {
    fn from(error: JsonRejection) -> Self {
        let status = if error.status() == StatusCode::PAYLOAD_TOO_LARGE {
            error.status()
        } else {
            StatusCode::BAD_REQUEST
        };
        Self {
            status,
            message: error.body_text(),
        }
    }
}

impl IntoResponse for MessagesError {
    fn into_response(self) -> Response {
        (self.status, Json(ErrorBody::new(self.status, self.message))).into_response()
    }
}
