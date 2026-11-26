// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;

use serde::Deserialize;
use serde_json::{Map, Value, json};
use thiserror_ext::AsReport as _;
use validator::Validate as _;

use crate::error::{ApiError, bail_invalid_request};
use crate::routes::openai::chat_completions::ChatCompletionRequest;
use crate::routes::openai::utils::types::{Normalizable as _, ReasoningEffort};
use crate::routes::protocol_compat::{is_one_token_kv_producer, validate_video_url};

#[derive(Deserialize)]
pub(in crate::routes) struct MessagesRequest {
    #[serde(flatten)]
    pub prompt: MessagesPrompt,
    pub max_tokens: u32,
    pub stream: Option<bool>,
    stop_sequences: Option<Vec<String>>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i64>,
    min_tokens: Option<u32>,
    watermarking: Option<bool>,
    cache_salt: Option<String>,
    kv_transfer_params: Option<HashMap<String, Value>>,
    ec_transfer_params: Option<HashMap<String, Value>>,
    vllm_xargs: Option<HashMap<String, Value>>,
}

#[derive(Deserialize)]
pub(in crate::routes) struct MessagesPrompt {
    model: String,
    messages: Vec<Message>,
    system: Option<Content<Block>>,
    tools: Option<Vec<Tool>>,
    tool_choice: Option<ToolChoice>,
    thinking: Option<Thinking>,
    output_config: Option<OutputConfig>,
    chat_template_kwargs: Option<Map<String, Value>>,
}

#[derive(Deserialize)]
struct Message {
    role: Role,
    content: Content<Block>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Role {
    User,
    Assistant,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Content<T> {
    Text(String),
    Parts(Vec<T>),
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Block {
    Text {
        text: String,
    },
    Image {
        source: MediaSource,
    },
    Video {
        source: MediaSource,
    },
    Thinking {
        thinking: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Map<String, Value>,
    },
    ToolResult {
        tool_use_id: String,
        content: Option<Content<ResultBlock>>,
    },
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ResultBlock {
    Text {
        text: String,
    },
    Image {
        source: MediaSource,
    },
    Video {
        source: MediaSource,
    },
    ToolReference {
        #[serde(alias = "tool_name")]
        name: String,
    },
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum MediaSource {
    Url { url: String },
    Base64 { media_type: String, data: String },
}

impl MediaSource {
    fn into_part(self, video: bool) -> Result<Value, ApiError> {
        let modality = if video { "video" } else { "image" };
        let url = match self {
            Self::Url { url } => url,
            Self::Base64 { media_type, data } => {
                if !media_type.split_once('/').is_some_and(|(kind, subtype)| {
                    kind == modality
                        && !subtype.is_empty()
                        && subtype
                            .bytes()
                            .all(|b| b.is_ascii_alphanumeric() || b"_.+-".contains(&b))
                }) || data.is_empty()
                {
                    bail_invalid_request!(param = "messages", "Invalid base64 {modality} source.");
                }
                format!("data:{media_type};base64,{data}")
            }
        };
        if video {
            validate_video_url(&url, "messages")?;
            Ok(json!({"type":"video_url", "video_url":{"url":url}}))
        } else {
            Ok(json!({"type":"image_url", "image_url":{"url":url}}))
        }
    }
}

#[derive(Deserialize)]
struct Tool {
    #[serde(rename = "type")]
    _kind: Option<ToolKind>,
    name: String,
    description: Option<String>,
    input_schema: Map<String, Value>,
    strict: Option<bool>,
    defer_loading: Option<bool>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum ToolKind {
    Custom,
}

#[derive(Deserialize)]
struct ToolChoice {
    #[serde(rename = "type")]
    kind: ChoiceKind,
    name: Option<String>,
    #[serde(default)]
    disable_parallel_tool_use: bool,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum ChoiceKind {
    Auto,
    Any,
    None,
    Tool,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Thinking {
    Disabled,
    BetweenTools,
    Adaptive,
    Enabled { budget_tokens: u32 },
}

#[derive(Deserialize)]
struct OutputConfig {
    effort: Option<Effort>,
    format: Option<OutputFormat>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OutputFormat {
    JsonSchema { schema: Map<String, Value> },
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum Effort {
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

impl MessagesRequest {
    pub fn lower(self) -> Result<(ChatCompletionRequest, bool), ApiError> {
        if self.max_tokens == 0 {
            bail_invalid_request!(param = "max_tokens", "max_tokens must be positive.");
        }
        let mut chat = self.prompt.chat_body(Some(self.max_tokens))?;
        let producer =
            is_one_token_kv_producer(Some(self.max_tokens), self.kv_transfer_params.as_ref());
        chat.extend(
            json!({
                "max_completion_tokens":self.max_tokens, "stream":self.stream.unwrap_or(false),
                "stop": if producer {None} else {self.stop_sequences}, "ignore_eos":producer,
                "temperature":self.temperature, "top_p":self.top_p, "top_k":self.top_k,
                "min_tokens":self.min_tokens,
                "cache_salt":self.cache_salt, "kv_transfer_params":self.kv_transfer_params,
                "ec_transfer_params":self.ec_transfer_params, "vllm_xargs":self.vllm_xargs,
            })
            .as_object_mut()
            .map(std::mem::take)
            .expect("chat options are an object"),
        );
        // Older Chat accepts a bool, not JSON null. Omission retains its native
        // default; forward an explicit false as faithfully as an explicit true.
        if let Some(enabled) = self.watermarking {
            chat.insert("watermarking".into(), json!(enabled));
        }
        Ok((decode_chat(chat)?, producer))
    }
}

impl MessagesPrompt {
    pub fn lower_count(self) -> Result<ChatCompletionRequest, ApiError> {
        decode_chat(self.chat_body(None)?)
    }

    fn chat_body(self, max_tokens: Option<u32>) -> Result<Map<String, Value>, ApiError> {
        if self.model.is_empty() {
            bail_invalid_request!(param = "model", "Model is required.");
        }
        if self.messages.is_empty() {
            bail_invalid_request!(param = "messages", "messages must not be empty.");
        }
        let mut messages = Vec::new();
        if let Some(system) = self.system {
            let system = match system {
                Content::Text(text) => text,
                Content::Parts(parts) => parts
                    .into_iter()
                    .map(|part| match part {
                        Block::Text { text } => Ok(text),
                        _ => Err(ApiError::invalid_request(
                            "system only supports text blocks.",
                            Some("system"),
                        )),
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .join(
                        "

",
                    ),
            };
            messages.push(json!({"role":"system", "content":system}));
        }
        for message in self.messages {
            lower_message(message, &mut messages)?;
        }
        let mut chat = Map::from_iter([
            ("messages".into(), json!(messages)),
            ("model".into(), json!(self.model)),
            (
                "chat_template_kwargs".into(),
                json!(self.chat_template_kwargs),
            ),
        ]);
        if let Some(tools) = self.tools {
            let tools: Vec<Value> = tools
                .into_iter()
                .map(|tool| {
                    let Tool {
                        _kind: _,
                        name,
                        description,
                        mut input_schema,
                        strict,
                        defer_loading,
                    } = tool;
                    input_schema.entry("type").or_insert(json!("object"));
                    json!({"type":"function", "function":{
                        "name":name, "description":description, "parameters":input_schema,
                        "strict":strict, "defer_loading":defer_loading
                    }})
                })
                .collect();
            chat.insert("tools".into(), json!(tools));
        }
        if let Some(choice) = self.tool_choice {
            let value = match choice.kind {
                ChoiceKind::Auto => json!("auto"),
                ChoiceKind::Any => json!("required"),
                ChoiceKind::None => json!("none"),
                ChoiceKind::Tool => {
                    let Some(name) = choice.name.filter(|name| !name.is_empty()) else {
                        bail_invalid_request!(
                            param = "tool_choice",
                            "A named tool choice requires a name."
                        );
                    };
                    json!({"type":"function", "function":{"name":name}})
                }
            };
            chat.insert("tool_choice".into(), value);
            chat.insert(
                "parallel_tool_calls".into(),
                json!(!choice.disable_parallel_tool_use),
            );
        }
        if let Some(output) = self.output_config {
            if let Some(effort) = output.effort {
                let effort = match effort {
                    Effort::Low => ReasoningEffort::Low,
                    Effort::Medium => ReasoningEffort::Medium,
                    Effort::High => ReasoningEffort::High,
                    Effort::XHigh => ReasoningEffort::XHigh,
                    Effort::Max => ReasoningEffort::Max,
                };
                chat.insert("reasoning_effort".into(), json!(effort));
            }
            if let Some(OutputFormat::JsonSchema { schema }) = output.format {
                chat.insert("response_format".into(), json!({"type":"json_schema", "json_schema":{"name":"json_schema", "schema":schema}}));
            }
        }
        match self.thinking {
            Some(Thinking::Enabled { budget_tokens }) => {
                if budget_tokens < 1024 || max_tokens.is_some_and(|max| budget_tokens >= max) {
                    bail_invalid_request!(
                        param = "thinking",
                        "budget_tokens must be at least 1024 and less than max_tokens."
                    );
                }
                chat.insert("thinking_token_budget".into(), json!(budget_tokens));
            }
            Some(Thinking::Disabled | Thinking::BetweenTools) => {
                // Honor templates that support `none`. This does not suppress
                // a reasoning block if the model nevertheless produces one.
                chat.insert("reasoning_effort".into(), json!(ReasoningEffort::None));
            }
            Some(Thinking::Adaptive) | None => {}
        }
        Ok(chat)
    }
}

fn decode_chat(chat: Map<String, Value>) -> Result<ChatCompletionRequest, ApiError> {
    let mut request: ChatCompletionRequest =
        serde_path_to_error::deserialize(Value::Object(chat)).map_err(|err| {
            ApiError::invalid_request(
                format!("Invalid Messages request: {}", err.as_report()),
                None,
            )
        })?;
    request.normalize();
    request.validate().map_err(|err| {
        ApiError::invalid_request(
            format!("Invalid Messages request: {}", err.as_report()),
            None,
        )
    })?;
    Ok(request)
}

fn lower_message(message: Message, messages: &mut Vec<Value>) -> Result<(), ApiError> {
    let is_user = matches!(message.role, Role::User);
    let role = if is_user { "user" } else { "assistant" };
    let parts = match message.content {
        Content::Text(text) => {
            messages.push(json!({"role":role, "content":text}));
            return Ok(());
        }
        Content::Parts(parts) => parts,
    };
    let mut content = Vec::new();
    let mut reasoning = String::new();
    let mut calls = Vec::new();
    let initial_len = messages.len();
    for part in parts {
        match part {
            Block::Text { text } => content.push(json!({"type":"text", "text":text})),
            Block::Image { source } if is_user => content.push(source.into_part(false)?),
            Block::Video { source } if is_user => content.push(source.into_part(true)?),
            Block::Thinking { thinking } if !is_user => reasoning.push_str(&thinking),
            Block::ToolUse { id, name, input } if !is_user => calls.push(json!({
                "type":"function", "id":id, "function":{"name":name, "arguments":Value::Object(input).to_string()}
            })),
            Block::ToolResult { tool_use_id, content: result } if is_user => {
                if !content.is_empty() {
                    messages.push(json!({"role":"user", "content":std::mem::take(&mut content)}));
                }
                lower_tool_result(tool_use_id, result, messages)?;
            }
            _ => bail_invalid_request!(param = "messages", "Content block is not supported for {role} messages."),
        }
    }
    if is_user {
        if !content.is_empty() || messages.len() == initial_len {
            messages.push(json!({"role":"user", "content":content}));
        }
    } else {
        let mut entry = json!({"role":"assistant", "content":content});
        if !reasoning.is_empty() {
            entry["reasoning"] = json!(reasoning);
        }
        if !calls.is_empty() {
            entry["tool_calls"] = json!(calls);
        }
        messages.push(entry);
    }
    Ok(())
}

fn lower_tool_result(
    id: String,
    content: Option<Content<ResultBlock>>,
    messages: &mut Vec<Value>,
) -> Result<(), ApiError> {
    let mut text = Vec::new();
    let mut media = Vec::new();
    let mut references = Vec::new();
    match content {
        Some(Content::Text(result)) => text.push(result),
        Some(Content::Parts(parts)) => {
            for part in parts {
                match part {
                    ResultBlock::Text { text: part } => text.push(part),
                    ResultBlock::Image { source } => media.push(source.into_part(false)?),
                    ResultBlock::Video { source } => media.push(source.into_part(true)?),
                    ResultBlock::ToolReference { name } => {
                        references.push(json!({"type":"tool_reference", "name":name}))
                    }
                }
            }
        }
        None => {}
    }
    let text = text.join(
        "
",
    );
    let result = if references.is_empty() {
        json!(text)
    } else {
        if !text.is_empty() {
            references.push(json!({"type":"text", "text":text}));
        }
        json!(references)
    };
    messages.push(json!({"role":"tool", "tool_call_id":id, "content":result}));
    if !media.is_empty() {
        messages.push(json!({"role":"user", "content":media}));
    }
    Ok(())
}

#[cfg(test)]
mod parity_tests {
    use super::*;

    #[test]
    fn messages_uses_chat_default_for_omitted_watermark_and_preserves_false() {
        for value in [None, Some(false), Some(true)] {
            let mut body = json!({"model":"model", "max_tokens":16, "messages":[{"role":"user","content":"hello"}]});
            if let Some(v) = value { body["watermarking"] = json!(v); }
            let (request, _) = serde_json::from_value::<MessagesRequest>(body).unwrap().lower().unwrap();
            assert_eq!(request.watermarking, value.unwrap_or(true));
        }
    }
}
