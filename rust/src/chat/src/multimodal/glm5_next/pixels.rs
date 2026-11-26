// SPDX-License-Identifier: Apache-2.0
// Based on smg-project/smg 631ce06c7 GLM-5.3-Flash preprocessing (Apache-2.0).
// Adapted to the pinned vLLM frontend API and engine tensor layout.

//! GLM-5.3-Flash image and video preprocessing.

use std::borrow::Cow;

use image::{DynamicImage, GenericImageView, imageops::FilterType};
use ndarray::Array2;

use llm_multimodal::{
    encoder_inputs::ModelSpecificValue,
    types::RgbFrameRef,
    vision::{
        preprocessor_config::PreProcessorConfig,
        processor::{PreprocessedEncoderInputs, VisionPreProcessor},
        transforms::{TransformError, pil_to_filter, resize_rgb_bytes, rgb_bytes},
    },
};

const PATCH_SIZE: usize = 14;
const MERGE_SIZE: usize = 2;
const TEMPORAL_PATCH_SIZE: usize = 2;
const MIN_IMAGE_TOKENS: usize = 16;
const MAX_IMAGE_TOKENS: usize = 8000;
// Mirror Glm5NextProcessingInfo's encoder profiling/runtime ceiling.
const MAX_VIDEO_TOKENS: usize = 30_000;

#[derive(Clone, Copy)]
struct Params {
    patch: usize,
    merge: usize,
    temporal: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
    mean: [f64; 3],
    std: [f64; 3],
}

#[derive(Debug, Clone, Copy)]
struct Geometry {
    target: (usize, usize),
    content: (usize, usize),
}

trait Pixel: Copy {
    fn lut(cfg: &PreProcessorConfig, params: Params) -> Result<[[Self; 256]; 3], TransformError>;
    fn inputs(tensor: Array2<Self>, tokens: Vec<usize>, sizes: Vec<(u32,u32)>) -> PreprocessedEncoderInputs;
}
impl Pixel for f32 {
    fn lut(cfg: &PreProcessorConfig, p: Params) -> Result<[[Self; 256]; 3], TransformError> { pixel_lut(cfg,p) }
    fn inputs(tensor: Array2<Self>, t:Vec<usize>, s:Vec<(u32,u32)>) -> PreprocessedEncoderInputs { PreprocessedEncoderInputs::new(tensor,t,s) }
}
impl Pixel for u8 {
    fn lut(_: &PreProcessorConfig, _: Params) -> Result<[[Self; 256]; 3], TransformError> { Ok([std::array::from_fn(|i| i as u8);3]) }
    fn inputs(tensor: Array2<Self>, t:Vec<usize>, s:Vec<(u32,u32)>) -> PreprocessedEncoderInputs { PreprocessedEncoderInputs::from_bytes(tensor,t,s) }
}
struct Patches<T> {
    values: Vec<T>,
    grid: (usize, usize, usize),
    count: usize,
    tokens: usize,
}

trait RgbSource {
    fn dimensions(&self) -> (u32, u32);
    fn rgb(&self) -> Result<Cow<'_, [u8]>, TransformError>;
}

impl RgbSource for DynamicImage {
    fn dimensions(&self) -> (u32, u32) {
        GenericImageView::dimensions(self)
    }

    fn rgb(&self) -> Result<Cow<'_, [u8]>, TransformError> {
        let (_, _, data) = rgb_bytes(self);
        Ok(data)
    }
}

impl RgbSource for RgbFrameRef<'_> {
    fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn rgb(&self) -> Result<Cow<'_, [u8]>, TransformError> {
        validate_rgb(self.width as usize, self.height as usize, self.data.len())?;
        Ok(Cow::Borrowed(self.data))
    }
}

/// Read a `usize` extra key strictly: absent falls back to `default`, but a
/// present-yet-unusable value is a config error. `get_extra` swallows type
/// mismatches into `None`, and a silent default here changes the alignment
/// factor and token geometry — the gateway's placeholder count would then
/// disagree with the serving engine with no trace. Integral JSON floats
/// (`2.0`) are accepted for interop with float-serializing toolchains.
fn extra_usize(
    config: &PreProcessorConfig,
    key: &str,
    default: usize,
) -> Result<usize, TransformError> {
    let Some(value) = config.extra.get(key) else {
        return Ok(default);
    };
    if let Some(n) = value.as_u64() {
        return Ok(n as usize);
    }
    if let Some(f) = value.as_f64() {
        if f >= 0.0 && f.fract() == 0.0 && f <= u32::MAX as f64 {
            return Ok(f as usize);
        }
    }
    Err(shape(format!(
        "GLM preprocessor key {key} must be a non-negative integer, got {value}"
    )))
}

/// `extra_usize`'s float sibling: absent -> default, wrong type -> error.
fn extra_f32(config: &PreProcessorConfig, key: &str, default: f32) -> Result<f32, TransformError> {
    let Some(value) = config.extra.get(key) else {
        return Ok(default);
    };
    value.as_f64().map(|f| f as f32).ok_or_else(|| {
        shape(format!(
            "GLM preprocessor key {key} must be a number, got {value}"
        ))
    })
}

impl Params {
    fn from_config(config: &PreProcessorConfig, video: bool) -> Result<Self, TransformError> {
        let patch = config.get_patch_size(PATCH_SIZE);
        let merge = config.merge_size.unwrap_or(MERGE_SIZE);
        let temporal = config.temporal_patch_size.unwrap_or(TEMPORAL_PATCH_SIZE);
        let expand = extra_usize(config, "patch_expand_factor", 1)?;
        if [patch, merge, temporal, expand].contains(&0) {
            return Err(shape(
                "GLM patch, merge, temporal, and expand factors must be positive",
            ));
        }
        let factor = product(&[patch, merge, expand], "GLM alignment factor")?;
        let pixels_per_token = product(&[factor, factor, temporal], "GLM pixels per token")?;
        let budget = |pixels: Option<usize>, key: &str, fallback| {
            pixels.map_or_else(
                || {
                    product(
                        &[extra_usize(config, key, fallback)?, pixels_per_token],
                        "GLM pixel budget",
                    )
                },
                Ok,
            )
        };
        // Video budgets read video-named token keys only: an image-scale
        // `max_image_tokens` must not silently become a clip's volume cap
        // when the gateway falls back to the image config for video.
        let (min_key, max_key, max_fallback) = if video {
            ("min_video_tokens", "max_video_tokens", MAX_VIDEO_TOKENS)
        } else {
            ("min_image_tokens", "max_image_tokens", MAX_IMAGE_TOKENS)
        };
        // In a dedicated video_processor section HF uses the image-token
        // spellings for the clip budget. Do not borrow them from image config.
        let dedicated_video = video && config.extra.get("video_processor_type").is_some();
        let min_key = if dedicated_video && !config.extra.contains_key(min_key) {
            "min_image_tokens"
        } else {
            min_key
        };
        let max_key = if dedicated_video && !config.extra.contains_key(max_key) {
            "max_image_tokens"
        } else {
            max_key
        };
        let min_pixels = budget(config.min_pixels, min_key, MIN_IMAGE_TOKENS)?;
        let max_pixels = budget(config.max_pixels, max_key, max_fallback)?;
        let max_pixels = if video && config.max_pixels.is_none() {
            max_pixels.min(product(
                &[MAX_VIDEO_TOKENS, pixels_per_token],
                "GLM video cap",
            )?)
        } else {
            max_pixels
        };
        if min_pixels == 0 || min_pixels > max_pixels {
            return Err(shape("GLM pixel budget must be positive with min <= max"));
        }
        let triple = |values: &Option<Vec<f64>>, default| {
            values
                .as_ref()
                .filter(|values| values.len() >= 3)
                .map(|values| [values[0], values[1], values[2]])
                .unwrap_or(default)
        };
        Ok(Self {
            patch,
            merge,
            temporal,
            factor,
            min_pixels,
            max_pixels,
            mean: triple(&config.image_mean, PreProcessorConfig::CLIP_MEAN),
            std: triple(&config.image_std, PreProcessorConfig::CLIP_STD),
        })
    }

    fn geometry(
        self,
        frames: usize,
        height: usize,
        width: usize,
    ) -> Result<Geometry, TransformError> {
        if frames == 0 || height == 0 || width == 0 {
            return Err(shape("GLM dimensions must be positive"));
        }
        // Matches the reference family's guard (see qwen smart_resize):
        // beyond 200:1 the budget search degenerates to a 1-pixel strip on a
        // (factor, factor) canvas, silently destroying the image.
        let (long, short) = (height.max(width) as f64, height.min(width) as f64);
        if long / short > 200.0 {
            return Err(shape(format!(
                "GLM aspect ratio must be below 200:1, got {height}x{width}"
            )));
        }
        let align =
            |value: usize| product(&[value.div_ceil(self.factor), self.factor], "GLM alignment");
        let aligned_frames = ((frames as f64 / self.temporal as f64).round_ties_even() as usize
            * self.temporal)
            .max(self.temporal);
        let volume = |n: usize, h: usize, w: usize| {
            (n as u128).saturating_mul(h as u128).saturating_mul(w as u128)
        };
        let mut target = (align(height)?, align(width)?);
        let mut target_volume = volume(aligned_frames, target.0, target.1);

        if target_volume < self.min_pixels as u128 {
            let scale =
                (self.min_pixels as f64 / (frames as f64 * height as f64 * width as f64)).sqrt();
            target = (
                align(((height as f64 * scale).ceil() as usize).max(1))?,
                align(((width as f64 * scale).ceil() as usize).max(1))?,
            );
            target_volume = volume(aligned_frames, target.0, target.1);
        }

        if target_volume > self.max_pixels as u128 {
            if volume(aligned_frames, self.factor, self.factor) > self.max_pixels as u128 {
                return Err(shape("GLM max_pixels is smaller than one aligned patch"));
            }
            let (mut low, mut high, mut best) = (1usize, height, (self.factor, self.factor));
            while low <= high {
                let candidate_h = low + (high - low) / 2;
                let candidate_w =
                    ((width as u128 * candidate_h as u128) / height as u128).max(1) as usize;
                let candidate = (align(candidate_h)?, align(candidate_w)?);
                if volume(aligned_frames, candidate.0, candidate.1) <= self.max_pixels as u128 {
                    best = candidate;
                    low = candidate_h + 1;
                } else {
                    high = candidate_h - 1;
                }
            }
            target = best;
        }

        let mut scale = (target.0 as f64 / height as f64).min(target.1 as f64 / width as f64);
        if volume(frames, height, width) >= self.min_pixels as u128 {
            scale = scale.min(1.0);
        }
        Ok(Geometry {
            target,
            content: (
                ((height as f64 * scale).floor() as usize).clamp(1, target.0),
                ((width as f64 * scale).floor() as usize).clamp(1, target.1),
            ),
        })
    }
}

fn pixel_lut(
    config: &PreProcessorConfig,
    params: Params,
) -> Result<[[f32; 256]; 3], TransformError> {
    let do_rescale = config.do_rescale.unwrap_or(true);
    let do_normalize = config.do_normalize.unwrap_or(true);
    let rescale = if do_rescale {
        config.rescale_factor.unwrap_or(1.0 / 255.0)
    } else {
        1.0
    };
    if !rescale.is_finite()
        || (do_normalize
            && (params.mean.iter().any(|value| !value.is_finite())
                || params.std.iter().any(|value| !value.is_finite() || *value == 0.0)))
    {
        return Err(shape("invalid GLM pixel rescale or normalization values"));
    }
    let lut = std::array::from_fn(|channel| {
        let scale = (if do_normalize {
            rescale / params.std[channel]
        } else {
            rescale
        }) as f32;
        let bias = (if do_normalize {
            -params.mean[channel] / params.std[channel]
        } else {
            0.0
        }) as f32;
        std::array::from_fn(|value| value as f32 * scale + bias)
    });
    lut.iter()
        .flatten()
        .all(|value| value.is_finite())
        .then_some(lut)
        .ok_or_else(|| shape("GLM pixel transform overflow"))
}

fn prepare<'a, F: RgbSource>(
    frame: &'a F,
    geometry: Geometry,
    filter: FilterType,
    do_resize: bool,
) -> Result<Cow<'a, [u8]>, TransformError> {
    let (source_w, source_h) = frame.dimensions();
    let raw = frame.rgb()?;
    let ((target_h, target_w), (content_h, content_w)) = (geometry.target, geometry.content);
    if !do_resize
        || (source_w as usize == target_w
            && source_h as usize == target_h
            && geometry.content == geometry.target)
    {
        return Ok(raw);
    }
    let content = if (source_w as usize, source_h as usize) == (content_w, content_h) {
        raw.into_owned()
    } else {
        if filter == FilterType::CatmullRom {
            super::torch_resize::resize(
                raw.as_ref(),
                source_w as usize,
                source_h as usize,
                content_w,
                content_h,
            )?
        } else {
            resize_rgb_bytes(
                raw.as_ref(),
                source_w,
                source_h,
                content_w as u32,
                content_h as u32,
                filter,
            )?
            .into_raw()
        }
    };
    if geometry.content == geometry.target {
        return Ok(Cow::Owned(content));
    }
    let mut canvas = vec![0; rgb_len(target_w, target_h)?];
    let (content_row, target_row) = (content_w * 3, target_w * 3);
    for row in 0..content_h {
        canvas[row * target_row..row * target_row + content_row]
            .copy_from_slice(&content[row * content_row..(row + 1) * content_row]);
    }
    Ok(Cow::Owned(canvas))
}

fn encode<T: Pixel, F: RgbSource>(
    frames: &[F],
    budget_frames: usize,
    params: Params,
    config: &PreProcessorConfig,
) -> Result<Patches<T>, TransformError> {
    let first = frames.first().ok_or(TransformError::EmptyBatch)?;
    let source_size = first.dimensions();
    if frames.iter().any(|frame| frame.dimensions() != source_size) {
        return Err(shape("GLM video frames must have identical dimensions"));
    }
    let do_resize = config.do_resize.unwrap_or(true);
    let geometry = if do_resize {
        params.geometry(
            budget_frames,
            source_size.1 as usize,
            source_size.0 as usize,
        )?
    } else {
        Geometry {
            target: (source_size.1 as usize, source_size.0 as usize),
            content: (source_size.1 as usize, source_size.0 as usize),
        }
    };
    let (height, width) = geometry.target;
    if !height.is_multiple_of(params.factor) || !width.is_multiple_of(params.factor) {
        return Err(shape("GLM canvas is not aligned to the configured factor"));
    }
    let (grid_h, grid_w) = (height / params.patch, width / params.patch);
    let grid_t = frames.len().div_ceil(params.temporal);
    let count = product(&[grid_t, grid_h, grid_w], "GLM patch count")?;
    let features = product(
        &[3, params.temporal, params.patch, params.patch],
        "GLM features",
    )?;
    let expected = product(&[count, features], "GLM patch output size")?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(expected)
        .map_err(|error| shape(format!("cannot reserve GLM patch output: {error}")))?;
    let lut = T::lut(config, params)?;
    let filter = pil_to_filter(config.resampling.or(Some(3)));
    let merged_patch = params.merge * params.patch;

    for temporal_index in 0..grid_t {
        let start = temporal_index * params.temporal;
        let end = (start + params.temporal).min(frames.len());
        let prepared = frames[start..end]
            .iter()
            .map(|frame| prepare(frame, geometry, filter, do_resize))
            .collect::<Result<Vec<_>, _>>()?;
        let last = prepared.last().ok_or(TransformError::EmptyBatch)?;
        for patch_row in 0..grid_h / params.merge {
            for patch_col in 0..grid_w / params.merge {
                let (y0, x0) = (patch_row * merged_patch, patch_col * merged_patch);
                for merge_h in 0..params.merge {
                    for merge_w in 0..params.merge {
                        for (channel, channel_lut) in lut.iter().enumerate() {
                            for temporal in 0..params.temporal {
                                let raw = prepared.get(temporal).unwrap_or(last).as_ref();
                                for patch_y in 0..params.patch {
                                    let row = (y0 + merge_h * params.patch + patch_y) * width
                                        + x0
                                        + merge_w * params.patch;
                                    let mut source = row * 3 + channel;
                                    for _ in 0..params.patch {
                                        values.push(channel_lut[raw[source] as usize]);
                                        source += 3;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    debug_assert_eq!(values.len(), expected);
    Ok(Patches {
        values,
        grid: (grid_t, grid_h, grid_w),
        count,
        tokens: count / (params.merge * params.merge),
    })
}

fn preprocess_video_frames<T: Pixel, F: RgbSource>(
    frames: &[F],
    config: &PreProcessorConfig,
) -> Result<PreprocessedEncoderInputs, TransformError> {
    let original = frames.first().ok_or(TransformError::EmptyBatch)?.dimensions();
    let params = Params::from_config(config, true)?;
    let patches = encode::<T,_>(frames, frames.len(), params, config)?;
    let fps = extra_f32(config, "fps", 2.0)?;
    if !fps.is_finite() || fps <= 0.0 {
        return Err(shape(format!("GLM video fps must be positive, got {fps}")));
    }
    let (grid_t, grid_h, grid_w) = patches.grid;
    let features = 3 * params.temporal * params.patch * params.patch;
    let tensor = Array2::from_shape_vec((patches.count, features), patches.values)
        .map_err(|error| shape(format!("failed to build GLM video patches: {error}")))?;
    Ok(
        T::inputs(tensor, vec![patches.tokens], vec![original])
            .with_extra(
                "video_grid_thw",
                ModelSpecificValue::int_2d(vec![grid_t as i64, grid_h as i64, grid_w as i64], 1, 3),
            )
            .with_extra(
                "patches_per_video",
                ModelSpecificValue::int_1d(vec![patches.count as i64]),
            )
            .with_extra(
                "patches_per_image",
                ModelSpecificValue::int_1d(vec![patches.count as i64]),
            )
            .with_extra(
                "video_second_per_grid",
                ModelSpecificValue::Tensor {
                    data: vec![params.temporal as f32 / fps],
                    shape: vec![1],
                },
            ),
    )
}

pub(super) fn rgb_len(width: usize, height: usize) -> Result<usize, TransformError> {
    product(&[width, height, 3], "GLM RGB byte length")
}

fn product(values: &[usize], label: &str) -> Result<usize, TransformError> {
    values
        .iter()
        .try_fold(1usize, |total, value| total.checked_mul(*value))
        .ok_or_else(|| shape(format!("{label} overflow")))
}

fn validate_rgb(width: usize, height: usize, actual: usize) -> Result<(), TransformError> {
    let expected = rgb_len(width, height)?;
    if actual != expected {
        return Err(TransformError::InvalidShape {
            expected: format!("{width}x{height} RGB frame ({expected} bytes)"),
            actual: vec![actual],
        });
    }
    Ok(())
}

pub(super) fn shape(message: impl Into<String>) -> TransformError {
    TransformError::ShapeError(message.into())
}

#[derive(Debug, Clone)]
pub(super) struct Glm53FlashProcessor(PreProcessorConfig, bool);

impl Glm53FlashProcessor {
    pub(super) fn new(config: &PreProcessorConfig) -> Self {
        Self(config.clone(), std::env::var("VLLM_RS_GLM_DEVICE_NORM").as_deref() == Ok("1"))
    }
    fn images<T: Pixel>(
        &self,
        images: &[DynamicImage],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let config = &self.0;
        if images.is_empty() {
            return Err(TransformError::EmptyBatch);
        }
        let params = Params::from_config(config, false)?;
        let mut values = Vec::new();
        let mut grids = Vec::with_capacity(images.len() * 3);
        let mut counts = Vec::with_capacity(images.len());
        let mut tokens = Vec::with_capacity(images.len());
        for image in images {
            let patches = encode::<T,_>(std::slice::from_ref(image), params.temporal, params, config)?;
            grids.extend([
                patches.grid.0 as i64,
                patches.grid.1 as i64,
                patches.grid.2 as i64,
            ]);
            counts.push(patches.count as i64);
            tokens.push(patches.tokens);
            values.extend(patches.values);
        }
        let features = 3 * params.temporal * params.patch * params.patch;
        let total = values.len() / features;
        let tensor = Array2::from_shape_vec((total, features), values)
            .map_err(|error| shape(format!("failed to build GLM image patches: {error}")))?;
        Ok(T::inputs(
            tensor,
            tokens,
            images.iter().map(GenericImageView::dimensions).collect(),
        )
        .with_extra(
            "image_grid_thw",
            ModelSpecificValue::int_2d(grids, images.len(), 3),
        )
        .with_extra("patches_per_image", ModelSpecificValue::int_1d(counts)))
    }


}

impl VisionPreProcessor for Glm53FlashProcessor {
    fn default_mean(&self) -> [f64; 3] {
        PreProcessorConfig::CLIP_MEAN
    }

    fn default_std(&self) -> [f64; 3] {
        PreProcessorConfig::CLIP_STD
    }

    fn preprocess(&self, images: &[DynamicImage]) -> Result<PreprocessedEncoderInputs, TransformError> {
        if self.1 { self.images::<u8>(images) } else { self.images::<f32>(images) }
    }

    fn preprocess_video(
        &self,
        frames: &[DynamicImage],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let config = &self.0;
        if self.1 { preprocess_video_frames::<u8,_>(frames, config) } else { preprocess_video_frames::<f32,_>(frames, config) }
    }

    fn preprocess_video_rgb(
        &self,
        frames: &[RgbFrameRef<'_>],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let config = &self.0;
        if self.1 { preprocess_video_frames::<u8,_>(frames, config) } else { preprocess_video_frames::<f32,_>(frames, config) }
    }

    fn calculate_num_tokens(&self, width: u32, height: u32) -> usize {
        let config = &self.0;
        let Ok(params) = Params::from_config(config, false) else {
            return 0;
        };
        let Ok(geometry) = params.geometry(params.temporal, height as usize, width as usize) else {
            return 0;
        };
        geometry.target.0 / params.patch * (geometry.target.1 / params.patch)
            / (params.merge * params.merge)
    }

    fn model_name(&self) -> &'static str {
        "glm-5.3-flash"
    }

    fn get_processed_size(&self) -> Option<(u32, u32)> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;
    use image::{Rgb, RgbImage};
    use sha2::{Digest, Sha256};

    #[test]
    fn video_token_budget_matches_old_engine_without_allocating_frames() {
        for (requested, want) in [(240_000, 30_000), (30_000, 30_000), (4_096, 4_096)] {
            let config: PreProcessorConfig = serde_json::from_value(serde_json::json!({
                "video_processor_type":"Glm5NextVideoProcessor", "max_image_tokens":requested,
                "min_image_tokens":16, "patch_size":14, "temporal_patch_size":2, "merge_size":2
            }))
            .unwrap();
            let p = Params::from_config(&config, true).unwrap();
            assert_eq!(p.max_pixels, want * 28 * 28 * 2);
        }
    }

    // Digests are from the deployed Python Glm5NextImageProcessor, converting its
    // pixel_values to the engine's little-endian BF16 dtype before hashing.
    #[test]
    fn deployed_processor_image_tensors() {
        let config: PreProcessorConfig = serde_json::from_str(r#"{"patch_size":14,"merge_size":2,"temporal_patch_size":2,"patch_expand_factor":1,"min_image_tokens":16,"max_image_tokens":8000,"image_mean":[0.48145466,0.4578275,0.40821073],"image_std":[0.26862954,0.26130258,0.27577711]}"#).unwrap();
        for (w, h, grid, want) in [
            (
                400,
                200,
                (1, 16, 30),
                "dd479398ee4400c97b52a4eff80e5583f5136d9663892cf2e0409d7653b2c39b",
            ),
            (
                1151,
                821,
                (1, 60, 84),
                "a782c098b36cbbf7c393e9d773fb87b3326bc048620858ff77a573a92e130c3b",
            ),
            (
                56,
                56,
                (1, 8, 8),
                "c4402aa88d943718f497fa4281bf46744a4adeeadc29386add79b9e8370a49e6",
            ),
            (
                119,
                73,
                (1, 8, 12),
                "2aa09150feca5546069385999f3944b2b7da0c433e565c2e0b4c61048afaddcb",
            ),
        ] {
            let image = DynamicImage::ImageRgb8(RgbImage::from_fn(w, h, |x, y| {
                Rgb([(x * 5 + y * 3) as u8, (x + y * 7) as u8, (x * 13 + y) as u8])
            }));
            let p = Params::from_config(&config, false).unwrap();
            let patches = encode::<f32,_>(&[image], p.temporal, p, &config).unwrap();
            assert_eq!(patches.grid, grid, "{w}x{h} grid");
            let mut hash = Sha256::new();
            for v in patches.values {
                hash.update(bf16::from_f32(v).to_le_bytes());
            }
            assert_eq!(
                format!("{:x}", hash.finalize()),
                want,
                "{w}x{h} BF16 patches"
            );
        }
    }
}
