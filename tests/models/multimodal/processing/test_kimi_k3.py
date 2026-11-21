# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoImageProcessor

from vllm.config import CacheConfig, DeviceConfig, ModelConfig, VllmConfig
from vllm.model_executor.layers.fusion.mm_input_norm import build_mm_input_norm
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.utils import encode_video_url
from vllm.platforms import current_platform
from vllm.renderers import ChatParams, renderer_from_config
from vllm.transformers_utils.processors.kimi_k3 import (
    get_kimi_k3_frame_resize,
    get_kimi_k3_max_video_frames,
    get_kimi_k3_max_video_tokens,
    plan_kimi_k3_video,
)
from vllm.transformers_utils.processors.kimi_k25_vision_fused import (
    KimiK25FusedVisionProcessor,
)

from ...utils import build_model_context

_MODEL_ID = "moonshotai/Kimi-K3"


@pytest.fixture(scope="module")
def image_processors():
    ctx = build_model_context(_MODEL_ID, limit_mm_per_prompt={"image": 1})
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    k25_fused = getattr(processor.info, "image_processor")  # noqa: B009
    assert isinstance(k25_fused, KimiK25FusedVisionProcessor)
    k3_hf = AutoImageProcessor.from_pretrained(_MODEL_ID, trust_remote_code=True)
    return k25_fused, k3_hf


def _noise_image(width: int, height: int) -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


@pytest.mark.parametrize(
    ("width", "height"),
    [(310, 470), (7300, 30), (13, 9)],
)
def test_fused_image_processor_matches_hf(image_processors, width: int, height: int):
    fused, hf = image_processors
    image = _noise_image(width, height)
    expected = hf.preprocess([{"type": "image", "image": image}], return_tensors="pt")
    actual = fused.preprocess([{"type": "image", "image": image}], return_tensors="pt")

    torch.testing.assert_close(actual["grid_thws"], expected["grid_thws"])
    torch.testing.assert_close(actual["pixel_values"], expected["pixel_values"])
    assert fused.media_tokens_calculator(
        {"type": "image", "image": image}
    ) == hf.media_tokens_calculator({"type": "image", "image": image})


@pytest.mark.usefixtures("default_vllm_config")
def test_mm_device_do_normalize():
    device = current_platform.device_type
    ctx = build_model_context(_MODEL_ID, limit_mm_per_prompt={"image": 2})
    assert ctx.model_config.multimodal_config.mm_device_do_normalize

    ctx.model_config.multimodal_config.mm_device_do_normalize = False
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    images = [
        Image.new("RGB", (310, 470), color=(17, 89, 231)),
        Image.new("RGB", (480, 320), color=(201, 13, 127)),
    ]
    prompt = processor.info.get_hf_config().image_placeholder * len(images)
    mm_items = processor.info.parse_mm_data({"image": images})

    normalized_inputs = processor(prompt, mm_items=mm_items)
    normalized_values = normalized_inputs["mm_kwargs"].get_data()["pixel_values"]
    assert isinstance(normalized_values, torch.Tensor)

    ctx.model_config.multimodal_config.mm_device_do_normalize = True
    raw_inputs = processor(prompt, mm_items=mm_items)
    raw_values = raw_inputs["mm_kwargs"].get_data()["pixel_values"]
    assert isinstance(raw_values, torch.Tensor)
    assert raw_values.dtype == torch.uint8

    input_norm = build_mm_input_norm(ctx.model_config).to(device)
    output = input_norm(raw_values.flatten(1).to(device), normalized_values.dtype)
    torch.testing.assert_close(
        output, normalized_values.flatten(1).to(device), rtol=1.6e-2, atol=1e-5
    )


@pytest.fixture(scope="module")
def k3_processor():
    ctx = build_model_context(_MODEL_ID, limit_mm_per_prompt={"image": 1, "video": 1})
    return MULTIMODAL_REGISTRY.create_processor(ctx.model_config)


def _video_text(processor, out) -> str:
    """The decoded prompt span of the single video placeholder."""
    [placeholder] = out["mm_placeholders"]["video"]
    start = placeholder.offset
    token_ids = out["prompt_token_ids"][start : start + placeholder.length]
    return processor.info.get_tokenizer().decode(token_ids, skip_special_tokens=False)


@pytest.mark.parametrize(
    ("width", "height", "num_frames", "fps", "expected"),
    [
        pytest.param(640, 480, 60, 30.0, 16, id="sample-fps"),
        pytest.param(320, 240, 50, 5.0, 50, id="slow-source"),
        pytest.param(1805, 1805, 400, 8.0, 36, id="token-budget"),
        pytest.param(28, 28, 1_000_000, 30.0, 768, id="frame-cap"),
    ],
)
def test_video_plan_stays_within_budget(
    k3_processor, width: int, height: int, num_frames: int, fps: float, expected: int
):
    cfg = k3_processor.info.image_processor.media_proc_cfg
    indices = plan_kimi_k3_video(num_frames, fps, width, height, cfg)
    assert indices == (
        np.linspace(0, num_frames - 1, expected).round().astype(int).tolist()
    )

    resize = get_kimi_k3_frame_resize(width, height, cfg)
    patch = cfg["patch_size"]
    frame_patches = ((resize["new_width"] + resize["pad_width"]) // patch) * (
        (resize["new_height"] + resize["pad_height"]) // patch
    )
    num_chunks = -(-expected // cfg["temporal_merge_kernel_size"])
    assert expected * frame_patches <= cfg["in_patch_limit_video"]
    assert num_chunks * resize["num_tokens"] <= get_kimi_k3_max_video_tokens(cfg)


def test_dummy_video_fills_video_budget(k3_processor):
    """Profiling must reach the per-video token budget the encoder cache uses."""
    info = k3_processor.info
    cfg = info.image_processor.media_proc_cfg
    size = info.get_dummy_video_frame_size()
    num_frames = get_kimi_k3_max_video_frames(size.width, size.height, cfg)
    num_chunks = -(-num_frames // cfg["temporal_merge_kernel_size"])
    tokens = (
        num_chunks
        * get_kimi_k3_frame_resize(size.width, size.height, cfg)["num_tokens"]
    )
    max_tokens = info.get_mm_max_tokens_per_item(8192, {"image": 1, "video": 1})
    assert tokens == max_tokens["video"] == get_kimi_k3_max_video_tokens(cfg)


def test_video_prompt_and_inputs(k3_processor):
    """A video becomes one block of timestamped 4-frame chunks."""
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, (12, 112, 84, 3), dtype=np.uint8)
    metadata = {"fps": 4.0, "total_num_frames": 12, "frames_indices": list(range(12))}
    prompt = k3_processor.dummy_inputs.get_dummy_text({"video": 1})
    out = k3_processor(
        prompt,
        mm_items=k3_processor.info.parse_mm_data({"video": [(frames, metadata)]}),
    )

    item = out["mm_kwargs"]["video"][0]
    assert item["video_grid_thws"].data.tolist() == [[4, 8, 6]] * 3
    assert item["video_chunk_timestamps"].data.tolist() == [0, 1000, 2000]
    assert item["video_pixel_values"].data.shape[0] == 12 * 8 * 6
    assert item["video_pixel_values"].data.dtype == torch.uint8
    pads = "<|media_pad|>" * 12
    assert _video_text(k3_processor, out) == (
        "<|media_begin|>video 84x112<|media_content|>"
        f"00:00:00.000{pads}00:00:01.000{pads}00:00:02.000{pads}"
        "<|media_end|>"
    )
    assert out["mm_placeholders"]["video"][0].get_num_embeds() == 36


def test_video_subsamples_decoded_frames(k3_processor):
    """Frames decoded beyond the plan (e.g. by another loader) are subsampled."""
    frames = np.zeros((30, 56, 56, 3), dtype=np.uint8)
    metadata = {"fps": 30.0, "total_num_frames": 30, "frames_indices": list(range(30))}
    prompt = k3_processor.dummy_inputs.get_dummy_text({"video": 1})
    out = k3_processor(
        prompt,
        mm_items=k3_processor.info.parse_mm_data({"video": [(frames, metadata)]}),
    )

    # 8 fps keeps 8 of 30 frames, at positions 0, 4, 8, 12, 17, 21, 25, 29.
    item = out["mm_kwargs"]["video"][0]
    assert item["video_grid_thws"].data.tolist() == [[4, 4, 4]] * 2
    assert item["video_chunk_timestamps"].data.tolist() == [0, 566]
    pads = "<|media_pad|>" * 4
    assert _video_text(k3_processor, out) == (
        "<|media_begin|>video 56x56<|media_content|>"
        f"00:00:00.000{pads}00:00:00.566{pads}"
        "<|media_end|>"
    )


def test_video_chat():
    """A `video_url` part renders K3's video block through the chat path."""
    config = VllmConfig(
        model_config=ModelConfig(
            _MODEL_ID,
            tokenizer_mode="kimi_k3",
            trust_remote_code=True,
            limit_mm_per_prompt={"video": 1},
            mm_processor_cache_gb=0,
        ),
        # Rendering needs no KV cache; keep CPU MLA and hybrid-cache defaults aligned.
        cache_config=CacheConfig(enable_prefix_caching=False),
        device_config=DeviceConfig(device="cpu"),
    )
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, (6, 56, 56, 3), dtype=np.uint8)
    content = [
        {"type": "video_url", "video_url": {"url": encode_video_url(frames)}},
        {"type": "text", "text": "Describe the video."},
    ]
    renderer = renderer_from_config(config)
    try:
        _, prompts = renderer.render_chat(
            [[{"role": "user", "content": content}]], ChatParams()
        )
        # A JPEG frame sequence carries no frame rate, so it counts as 1 fps.
        pads = "<|media_pad|>" * 4
        assert _video_text(renderer.get_mm_processor(), prompts[0]) == (
            "<|media_begin|>video 56x56<|media_content|>"
            f"00:00:00.000{pads}00:00:04.000{pads}"
            "<|media_end|>"
        )
    finally:
        renderer.shutdown()
