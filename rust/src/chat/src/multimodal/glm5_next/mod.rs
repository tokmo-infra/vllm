// SPDX-License-Identifier: Apache-2.0
// Based on smg-project/smg 631ce06c7 GLM-5.3-Flash preprocessing (Apache-2.0).
// Adapted to the pinned vLLM frontend API and engine tensor layout.

use std::collections::HashMap;

use serde_json::{Value, json};

mod pixels;
mod torch_resize;
use llm_multimodal::{
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    registry::{ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult},
    types::{FieldLayout, Modality, PromptReplacement, TokenId},
};

const IMAGE: &str = "<|image|>";
const VIDEO: &str = "<|video|>";
const BEGIN: &str = "<|begin_of_image|>";
const END: &str = "<|end_of_image|>";

pub(super) struct Glm53FlashSpec;

impl Glm53FlashSpec {
    fn image_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata.config_u32(&["image_token_id"]).map(|id| id as TokenId).ok_or_else(|| {
            ModelRegistryError::MissingConfigField {
                field: "image_token_id".to_string(),
            }
        })
    }

    fn encode(metadata: &ModelMetadata, text: &str) -> RegistryResult<Vec<TokenId>> {
        let ids = metadata.tokenizer.encode_text(text).ok_or_else(|| {
            ModelRegistryError::InvalidConfig {
                message: "cannot encode frame timestamp".into(),
            }
        })?;
        Ok(ids.into_iter().map(|id| id as TokenId).collect())
    }

    fn video_grid_t(input: &PreprocessedEncoderInputs) -> RegistryResult<Vec<usize>> {
        let Some(ModelSpecificValue::IntTensor { data, shape }) =
            input.model_specific.get("video_grid_thw")
        else {
            return Err(ModelRegistryError::InvalidConfig {
                message: "video_grid_thw".to_string(),
            });
        };
        if shape.len() != 2 || shape[1] != 3 || data.len() != shape[0] * 3 {
            return Err(ModelRegistryError::InvalidConfig {
                message: "video_grid_thw".to_string(),
            });
        }
        data.as_chunks::<3>()
            .0
            .iter()
            .map(|row| {
                usize::try_from(row[0]).map_err(|_| ModelRegistryError::InvalidConfig {
                    message: "video_grid_thw".to_string(),
                })
            })
            .collect()
    }

    fn unsupported(modality: Modality) -> ModelRegistryError {
        ModelRegistryError::UnsupportedModality {
            spec: "glm53_flash",
            modality,
        }
    }
}

impl ModelProcessorSpec for Glm53FlashSpec {
    fn name(&self) -> &'static str {
        "glm53_flash"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        matches!(
            metadata.config_model_type(),
            Some("glm5_next" | "glm53_flash")
        )
    }

    fn vision_processor(
        &self,
        _: &ModelMetadata,
        config: &llm_multimodal::PreProcessorConfig,
        modality: Modality,
    ) -> RegistryResult<Box<dyn llm_multimodal::VisionPreProcessor>> {
        match modality {
            Modality::Image | Modality::Video => {
                Ok(Box::new(pixels::Glm53FlashProcessor::new(config)))
            }
            _ => Err(Self::unsupported(modality)),
        }
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok(IMAGE.to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        Self::image_id(metadata)
    }

    fn placeholder_token_for(
        &self,
        _metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<String> {
        match modality {
            Modality::Image => Ok(IMAGE.to_string()),
            Modality::Video => Ok(VIDEO.to_string()),
            _ => Err(Self::unsupported(modality)),
        }
    }

    fn placeholder_token_id_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<TokenId> {
        match modality {
            Modality::Image | Modality::Video => Self::image_id(metadata),
            _ => Err(Self::unsupported(modality)),
        }
    }

    fn modality_limits(
        &self,
        metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        // Advertise only what the checkpoint can serve, so an incapable
        // derivative is rejected at validate_media_request instead of after
        // a full media fetch + preprocess. Images need the placeholder id;
        // video additionally splices <|begin_of_image|>/<|end_of_image|>
        // frame markers into the prompt.
        let mut limits = HashMap::new();
        if Self::image_id(metadata).is_ok() {
            limits.insert(Modality::Image, usize::MAX);
            if metadata.token_id(BEGIN).is_ok() && metadata.token_id(END).is_ok() {
                limits.insert(Modality::Video, 1);
            }
        }
        Ok(limits)
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({}))
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        input: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let id = Self::image_id(metadata)?;
        Ok(input
            .feature_token_counts
            .iter()
            .map(|&count| PromptReplacement::repeated(Modality::Image, IMAGE, id, count))
            .collect())
    }

    fn prompt_replacements_for(
        &self,
        metadata: &ModelMetadata,
        input: &PreprocessedEncoderInputs,
        modality: Modality,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        if modality == Modality::Image {
            return self.prompt_replacements(metadata, input);
        }
        if modality != Modality::Video {
            return Err(Self::unsupported(modality));
        }

        let grids = Self::video_grid_t(input)?;
        if grids.len() != input.feature_token_counts.len() {
            return Err(ModelRegistryError::InvalidConfig {
                message: "video_grid_thw item count".to_string(),
            });
        }
        let image_id = Self::image_id(metadata)?;
        let begin = metadata.token_id(BEGIN)?;
        let end = metadata.token_id(END)?;
        let seconds = match input.model_specific.get("glm_grid_seconds") {
            Some(ModelSpecificValue::IntTensor { data, .. })
                if grids.as_slice() == [data.len()] =>
            {
                data
            }
            _ => {
                return Err(ModelRegistryError::InvalidConfig {
                    message: "source timestamps must match video_grid_thw".into(),
                });
            }
        };

        input
            .feature_token_counts
            .iter()
            .zip(grids)
            .map(|(&count, grid_t)| {
                if grid_t == 0 || !count.is_multiple_of(grid_t) {
                    return Err(ModelRegistryError::InvalidConfig {
                        message: "video_grid_thw temporal token layout".to_string(),
                    });
                }
                let per_grid = count / grid_t;
                let mut tokens = Vec::new();
                for second in seconds {
                    tokens.push(begin);
                    tokens.extend(std::iter::repeat_n(image_id, per_grid));
                    tokens.push(end);
                    tokens.extend(Self::encode(metadata, &format!("{second}.0 seconds"))?);
                }
                Ok(PromptReplacement::sequence(Modality::Video, VIDEO, tokens))
            })
            .collect()
    }

    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        HashMap::from([
            (
                "pixel_values".to_string(),
                FieldLayout::flat("patches_per_image"),
            ),
            ("image_grid_thw".to_string(), FieldLayout::Batched),
            ("video_grid_thw".to_string(), FieldLayout::Batched),
            ("patches_per_image".to_string(), FieldLayout::Batched),
            ("patches_per_video".to_string(), FieldLayout::Batched),
            ("video_second_per_grid".to_string(), FieldLayout::Batched),
        ])
    }

    fn keep_on_cpu_keys(&self) -> Vec<String> {
        vec!["image_grid_thw".to_string(), "video_grid_thw".to_string()]
    }
}

#[cfg(test)]
mod wire_tests {
    use super::super::{MultimodalConfigFiles, MultimodalModelInfo};
    use llm_multimodal::MediaContentPart;
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};
    use std::{fs, path::PathBuf, sync::Arc};
    use vllm_engine_core_client::protocol::{
        dtype::ModelDtype, multimodal::MmKwargValue, tensor::WireArrayData,
    };
    use vllm_tokenizer::HuggingFaceTokenizer;

    #[tokio::test]
    #[ignore = "requires independent Python GLM wire fixtures; run once for each normalization mode"]
    async fn glm_typed_wire_python_oracle() {
        let fixtures = PathBuf::from(std::env::var_os("GLM_VIDEO_FIXTURES").unwrap());
        let model = PathBuf::from(std::env::var_os("GLM_VIDEO_MODEL").unwrap());
        let manifest: Value = serde_json::from_slice(&fs::read(fixtures.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest["schema"], "glm53-mm-oracle/v1");
        assert_eq!(manifest["device_do_normalize"], std::env::var("VLLM_RS_GLM_DEVICE_NORM").as_deref() == Ok("1"));
        let tok = Arc::new(HuggingFaceTokenizer::new_fastokens(&model.join("tokenizer.json")).unwrap());
        let combined=model.join("processor_config.json"); let config=model.join("config.json");
        let info=MultimodalModelInfo::from_paths("glm".into(),Some("glm5_next".into()),MultimodalConfigFiles {
            config:Some(&config), processor_config:Some(&combined), ..Default::default()
        },tok,Default::default()).unwrap().unwrap();
        for (name,r) in manifest["cases"].as_object().unwrap() {
            let data=fs::read(fixtures.join(r["inputs"]["file"].as_str().unwrap())).unwrap();
            let mime_type=r["inputs"]["media_type"].as_str().map(str::to_owned);
            let part=if r["modality"]=="image" { MediaContentPart::ImageData{data,mime_type,detail:None,uuid:None} }
                     else { MediaContentPart::VideoData{data,mime_type,uuid:None} };
            let mut ids:Vec<u32>=serde_json::from_value(r["initial_prompt_token_ids"].clone()).unwrap();
            let f=info.prepare_multimodal(vec![part], &mut ids, ModelDtype::BFloat16,Some(4096)).await.unwrap();
            assert_eq!(json!(ids),r["prompt_token_ids"],"{name} token ids");
            assert_eq!(f.len(),1); let span=&f[0].mm_position;
            assert_eq!(json!([span.offset,span.length]),json!([r["placeholder"]["offset"],r["placeholder"]["length"]]),"{name} span");
            let mask=span.is_embed.as_ref().map(|t|t.data.as_raw_view().unwrap().iter().map(|x|*x!=0).collect::<Vec<_>>()).unwrap_or_else(||vec![true;span.length]);
            assert_eq!(json!(mask),r["placeholder"]["is_embed"]);
            let fields=f[0].data.as_ref().unwrap(); assert_eq!(fields.len(),r["tensors"].as_object().unwrap().len());
            for (key,gold) in r["tensors"].as_object().unwrap() {
                let MmKwargValue::Tensor(t)=fields[key].data.as_ref().unwrap() else {panic!("missing {key}")};
                let mut shape:Vec<usize>=serde_json::from_value(gold["shape"].clone()).unwrap();
                if key.ends_with("grid_thw") {assert_eq!(shape.remove(0),1);} // engine item drops batch axis
                assert_eq!(t.shape,shape,"{key} shape"); assert_eq!(gold["dtype"], t.dtype.as_str(),"{key} dtype");
                assert_eq!(format!("{:x}",Sha256::digest(t.data.as_raw_view().unwrap())), gold["sha256"].as_str().unwrap(),"{name}:{key} full bytes");
            }
        }
    }

    #[tokio::test]
    #[ignore = "requires local GLM tokenizer+configs, ffmpeg, and independent Python lossless fixtures"]
    async fn glm_video_python_wire_goldens() {
        let fixtures = PathBuf::from(std::env::var_os("GLM_VIDEO_FIXTURES").unwrap());
        let model = PathBuf::from(std::env::var_os("GLM_VIDEO_MODEL").unwrap());
        let cases: Value =
            serde_json::from_slice(&fs::read(fixtures.join("manifest.json")).unwrap()).unwrap();
        let tok =
            Arc::new(HuggingFaceTokenizer::new_fastokens(&model.join("tokenizer.json")).unwrap());
        let combined = model.join("processor_config.json");
        let config = model.join("config.json");
        for (name, r) in cases.as_object().unwrap() {
            let mut video: Value = serde_json::from_slice::<Value>(&fs::read(&combined).unwrap())
                .unwrap()["video_processor"]
                .clone();
            if !r["cap"].is_null() {
                video["max_frames"] = r["cap"].clone();
            }
            let dir = tempfile::tempdir().unwrap();
            let vp = dir.path().join("video_preprocessor_config.json");
            fs::write(&vp, video.to_string()).unwrap();
            let info = MultimodalModelInfo::from_paths(
                "glm".into(),
                Some("glm5_next".into()),
                MultimodalConfigFiles {
                    config: Some(&config),
                    processor_config: Some(&combined),
                    video_preprocessor_config: Some(&vp),
                    ..Default::default()
                },
                tok.clone(),
                Default::default(),
            )
            .unwrap()
            .unwrap();
            let media = MediaContentPart::VideoData {
                data: fs::read(fixtures.join(format!("{name}.mp4"))).unwrap(),
                mime_type: Some("video/mp4".into()),
                uuid: None,
            };
            let mut ids: Vec<u32> = serde_json::from_value(r["initial"].clone())
                .unwrap_or_else(|_| vec![154832, 154855, 154833]);
            let features = info
                .prepare_multimodal(vec![media], &mut ids, ModelDtype::BFloat16, Some(4096))
                .await;
            if r["prompt"].is_null() {
                assert!(
                    matches!(features.unwrap_err(), crate::error::Error::InvalidMedia(_)),
                    "{name}"
                );
                continue;
            }
            let features = features.unwrap();
            assert_eq!(json!(ids), r["prompt"], "{name} token IDs");
            assert_eq!(features.len(), 1);
            let span = &features[0].mm_position;
            let expected = &r["placeholder"];
            assert_eq!(
                json!([span.offset, span.length]),
                json!([expected["offset"], expected["length"]]),
                "{name} span"
            );
            let WireArrayData::RawView(mask) = &span.is_embed.as_ref().unwrap().data else {
                panic!("non-inline mask")
            };
            assert_eq!(
                json!(mask.iter().map(|&x| x != 0).collect::<Vec<_>>()),
                expected["is_embed"],
                "{name} mask"
            );
            let fields = features[0].data.as_ref().unwrap();
            assert_eq!(fields.len(), 2);
            let MmKwargValue::Tensor(px) = fields["pixel_values_videos"].data.as_ref().unwrap()
            else {
                panic!("non-tensor pixels")
            };
            let WireArrayData::RawView(bytes) = &px.data else {
                panic!("non-inline pixels")
            };
            assert_eq!(json!(px.shape), r["pixel_shape"], "{name} shape");
            assert_eq!(
                format!("{:x}", Sha256::digest(bytes)),
                r["pixels_sha256"].as_str().unwrap(),
                "{name} entire BF16 tensor"
            );
        }
    }
}
