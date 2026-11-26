// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::convert::Infallible;

use asynk_strim_attr::{TryYielder, try_stream};
use axum::response::sse::Event;
use futures::StreamExt as _;
use serde_json::{Value, json};
use thiserror_ext::AsReport as _;
use tracing::error;
use vllm_chat::{
    AssistantBlockKind, AssistantMessageExt as _, ChatEvent, ChatEventStreamTrait, FinishReason,
};

use super::{ErrorBody, MessagesError, Usage, signature, stop_fields};
use crate::error::{ApiError, server_error};

fn event(kind: &'static str, mut fields: Value) -> Event {
    fields["type"] = json!(kind);
    Event::default().event(kind).json_data(&fields).expect("serializable Messages event")
}

fn block_start(index: usize, block: Value) -> Event {
    event(
        "content_block_start",
        json!({"index":index, "content_block":block}),
    )
}

fn block_delta(index: usize, delta: Value) -> Event {
    event("content_block_delta", json!({"index":index, "delta":delta}))
}

fn block_stop(index: usize) -> Event {
    event("content_block_stop", json!({"index":index}))
}

#[try_stream]
pub(super) async fn messages_stream(
    mut stream: impl ChatEventStreamTrait + Unpin,
    id: String,
    model: String,
    cache_details: bool,
    mut y: TryYielder<Event, Infallible>,
) -> Result<(), Infallible> {
    let mut open = None;
    // Chat numbers tool calls independently from text blocks. Messages has one
    // shared index space for every emitted content block, in emission order.
    let mut next_index = 0;
    let mut saw_content = false;
    let failure: MessagesError = loop {
        match stream.next().await {
            Some(Ok(ChatEvent::Start {
                prompt_token_ids, ..
            })) => {
                y.yield_ok(event(
                    "message_start",
                    json!({"message": {
                        "id":id, "type":"message", "role":"assistant", "model":model,
                        "content":[], "stop_reason":null, "stop_sequence":null,
                        "usage":{"input_tokens":prompt_token_ids.len(), "output_tokens":0}
                    }}),
                ))
                .await;
            }
            Some(Ok(ChatEvent::BlockStart { kind, .. })) => {
                let block = match kind {
                    AssistantBlockKind::Reasoning => json!({"type":"thinking", "thinking":""}),
                    AssistantBlockKind::Text => json!({"type":"text", "text":""}),
                    AssistantBlockKind::ToolCall => continue,
                };
                let index = next_index;
                next_index += 1;
                open = Some((index, kind));
                saw_content = true;
                y.yield_ok(block_start(index, block)).await;
            }
            Some(Ok(ChatEvent::BlockDelta {
                kind, delta, ..
            })) => {
                let delta = match kind {
                    AssistantBlockKind::Reasoning => {
                        json!({"type":"thinking_delta", "thinking":delta})
                    }
                    AssistantBlockKind::Text => json!({"type":"text_delta", "text":delta}),
                    AssistantBlockKind::ToolCall => continue,
                };
                let Some((index, _)) = open.filter(|(_, k)| *k == kind) else {
                    break server_error!("Messages delta without its open content block.").into();
                };
                y.yield_ok(block_delta(index, delta)).await;
            }
            Some(Ok(ChatEvent::BlockEnd { block, .. })) => {
                let kind = block.kind();
                if kind == AssistantBlockKind::ToolCall {
                    continue;
                }
                let Some((index, _)) = open.filter(|(_, k)| *k == kind) else {
                    break server_error!("Messages end without its open content block.").into();
                };
                if kind == AssistantBlockKind::Reasoning {
                    y.yield_ok(block_delta(
                        index,
                        json!({"type":"signature_delta", "signature":signature()}),
                    ))
                    .await;
                }
                y.yield_ok(block_stop(index)).await;
                open = None;
            }
            Some(Ok(ChatEvent::ToolCallStart { id, name, .. })) => {
                let index = next_index;
                next_index += 1;
                open = Some((index, AssistantBlockKind::ToolCall));
                saw_content = true;
                y.yield_ok(block_start(
                    index,
                    json!({"type":"tool_use", "id":id, "name":name, "input":{}}),
                ))
                .await;
            }
            Some(Ok(ChatEvent::ToolCallArgumentsDelta { delta, .. })) => {
                let Some((index, AssistantBlockKind::ToolCall)) = open else {
                    break server_error!("Messages tool delta without a tool block.").into();
                };
                if !delta.is_empty() {
                    y.yield_ok(block_delta(
                        index,
                        json!({"type":"input_json_delta", "partial_json":delta}),
                    ))
                    .await;
                }
            }
            Some(Ok(ChatEvent::ToolCallEnd { .. })) => {
                let Some((index, AssistantBlockKind::ToolCall)) = open else {
                    break server_error!("Messages tool end without a tool block.").into();
                };
                y.yield_ok(block_stop(index)).await;
                open = None;
            }
            Some(Ok(ChatEvent::Done {
                message,
                finish_reason,
                usage,
                ..
            })) => {
                if let FinishReason::Error = &finish_reason {
                    break server_error!("The model failed to generate a response.").into();
                }
                if let Some((index, kind)) = open.take() {
                    if kind == AssistantBlockKind::Reasoning {
                        y.yield_ok(block_delta(
                            index,
                            json!({"type":"signature_delta", "signature":signature()}),
                        ))
                        .await;
                    }
                    y.yield_ok(block_stop(index)).await;
                }
                if !saw_content {
                    y.yield_ok(block_start(0, json!({"type":"text", "text":""}))).await;
                    y.yield_ok(block_stop(0)).await;
                }
                let tool = message.tool_calls().last();
                let tool_complete = tool.is_some_and(|call| {
                    serde_json::from_str::<serde_json::Map<String, Value>>(&call.arguments).is_ok()
                });
                let (reason, sequence) = stop_fields(&finish_reason, tool.is_some(), tool_complete);
                y.yield_ok(event(
                    "message_delta",
                    json!({
                        "delta":{"stop_reason":reason, "stop_sequence":sequence},
                        "usage":Usage::from_engine(usage.engine, cache_details, false),
                    }),
                ))
                .await;
                y.yield_ok(event("message_stop", json!({}))).await;
                return Ok(());
            }
            Some(Ok(ChatEvent::LogprobsDelta { .. })) => {}
            Some(Err(err)) => {
                error!(error = %err.as_report(), "Messages stream failed");
                break server_error!("The Messages stream failed before generation completed.")
                    .into();
            }
            None => {
                break server_error!("The Messages stream ended before generation completed.")
                    .into();
            }
        }
    };
    if let Some((index, kind)) = open {
        if kind == AssistantBlockKind::Reasoning {
            y.yield_ok(block_delta(
                index,
                json!({"type":"signature_delta", "signature":signature()}),
            ))
            .await;
        }
        y.yield_ok(block_stop(index)).await;
    }
    y.yield_ok(
        Event::default()
            .event("error")
            .json_data(&ErrorBody::new(failure.status, failure.message)).expect("serializable Messages error"),
    )
    .await;
    Ok(())
}
