# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU address/dispatch contracts, not CUDA arithmetic or performance tests."""

import pytest
import regex as re

POLICY = "vllm/model_executor/layers/fused_moe/glm53_nvfp4_policy.py"
CONTRACTS = "vllm/_glm53_nvfp4/nvfp4_ep/contracts.py"
HEADER = "vllm/_glm53_nvfp4/include/flashinfer/megamoe/sm120_nvfp4_mega_moe.cuh"


@pytest.fixture(scope="module")
def policy(source_module):
    return source_module(POLICY)


@pytest.fixture(scope="module")
def contracts(source_module):
    return source_module(CONTRACTS)


@pytest.mark.parametrize("ep", [4, 8])
def test_only_glm_nvfp4_geometry_is_eligible(policy, contracts, ep):
    geometry = [4096, 2048, 288, 8, 288 // ep, ep]
    assert policy.supported_geometry(*geometry)
    contracts.validate_shape(ep, 288, 256, 8, 4096, 2048)
    for index, bad in enumerate([5120, 4096, 287, 4, 288 // ep + 1, 2]):
        changed = list(geometry)
        changed[index] = bad
        assert not policy.supported_geometry(*changed)
    with pytest.raises(ValueError):
        contracts.validate_shape(ep, 288, 257, 8, 4096, 2048)


@pytest.mark.parametrize("ep", [4, 8])
@pytest.mark.parametrize("bound", [0, 128, 129, 159, 160, 161, 191, 192, 193, 256, 257])
def test_uniform_dense_boundary_is_rank_consistent_and_p_only(policy, ep, bound):
    sizes = [bound] * ep
    for extended in (False, True):
        expected = max(bound, 1) if bound <= (192 if extended else 128) else None
        assert [
            policy.uniform_bound(sizes, rows, ep, allow_extended=extended)
            for rows in sizes
        ] == [expected] * ep


@pytest.mark.parametrize("ep", [4, 8])
def test_sparse_256_keeps_total_work_cap_and_d_remains_128(policy, ep):
    cases = [
        ([256] * (ep // 2) + [0] * (ep // 2), 256),
        ([256, 128] + [0] * (ep - 2), 256),
        ([256] * (ep // 2) + [1] + [0] * (ep // 2 - 1), None),
    ]
    for sizes, expected in cases:
        assert [
            policy.uniform_bound(sizes, rows, ep, allow_extended=True) for rows in sizes
        ] == [expected] * ep
        assert all(policy.uniform_bound(sizes, rows, ep) is None for rows in sizes)


@pytest.mark.parametrize(
    "sizes,tile",
    [
        ([0] * 8, 24),
        ([8] * 8, 24),
        ([32] * 8, 24),
        ([63] * 8, 24),
        ([64] * 8, 32),
        ([96] * 8, 32),
        ([128] * 8, 32),
        ([128] * 4 + [0] * 4, 32),
        ([60, 60, 92, 60, 76, 52, 44, 53], 24),
    ],
)
def test_decode_tile_uses_total_work_not_one_busy_rank(policy, sizes, tile):
    assert policy.tokens_per_item(sizes, 128) == tile
    assert policy.tokens_per_item(sizes, 256) == 24


@pytest.mark.parametrize(
    "sizes,rows,ep",
    [
        (None, 1, 4),
        ([1] * 3, 1, 4),
        ([True] * 4, 1, 4),
        ([1.0] * 4, 1, 4),
        ([-1] * 4, 0, 4),
        ([1] * 4, True, 4),
        ([1] * 4, 2, 4),
        ([1] * 2, 1, 2),
    ],
)
def test_invalid_cpu_metadata_falls_back_without_tensor_inspection(
    policy, sizes, rows, ep
):
    assert policy.uniform_bound(sizes, rows, ep, allow_extended=True) is None


def quad_transpose(values):
    result = [list(row) for row in values]
    for mask in (2, 1):
        for word in range(4):
            if word & mask:
                continue
            sent = [
                row[word if lane % 4 & mask else word | mask]
                for lane, row in enumerate(result)
            ]
            for lane in range(len(result)):
                target = word if lane % 4 & mask else word | mask
                result[lane][target] = sent[lane ^ mask]
    return result


def test_cuda_vector_branch_uses_the_audited_quad_mapping(repo_root):
    source = re.sub(r"\s+", "", (repo_root / HEADER).read_text())
    # Tie the CPU address model to the actual CUDA expressions, not a snapshot.
    assert "for(intm=2;m>=1;m>>=1)" in source
    assert "__shfl_xor_sync(0xffffffffu,hi?v[q]:v[q|m],m)" in source
    assert "__ldcs(reinterpret_cast<constuint4*>(row)+(k0/2+p)*4+c)" in source
    assert "wa[j][p*2+b/2][r+2*(b%2)]=v[b]" in source
    original = [[(lane, word) for word in range(4)] for lane in range(32)]
    actual = quad_transpose(original)
    assert actual == [
        [(lane // 4 * 4 + word, lane % 4) for word in range(4)] for lane in range(32)
    ]
    assert quad_transpose(actual) == original


@pytest.mark.parametrize("k", [2048, 4096])
@pytest.mark.parametrize("layout", [-1, 0, 1, 2])
def test_vector_reads_preserve_scalar_word_addresses_and_alignment(
    contracts, k, layout
):
    exchanged = quad_transpose(
        [[(lane, word) for word in range(4)] for lane in range(4)]
    )
    for row in (0, 7, 8, 15, 127, 128, 2047, 4095):
        base = contracts.weight_row(row, 4096, layout) * (k // 2)
        for k0 in range(0, k // 64, 4):
            for pair in range(2):
                for lane in range(4):
                    start = base + (k0 // 2 + pair) * 64 + lane * 16
                    assert start % 16 == 0 and start + 16 <= base + k // 2
                    for word in range(4):
                        sender, component = exchanged[lane][word]
                        vector = (
                            base + (k0 // 2 + pair) * 64 + sender * 16 + component * 4
                        )
                        scalar = (
                            base
                            + (k0 + pair * 2 + word // 2) * 32
                            + lane * 4
                            + word % 2 * 16
                        )
                        assert vector == scalar


def test_vector_transpose_keeps_low_nibble_first_without_mxfp4_conversion():
    codes = [((i * 7) ^ (i // 8)) & 15 for i in range(128)]
    packed = bytes(codes[i] | codes[i + 1] << 4 for i in range(0, 128, 2))
    vectors = [
        [
            int.from_bytes(
                packed[lane * 16 + word * 4 : lane * 16 + word * 4 + 4], "little"
            )
            for word in range(4)
        ]
        for lane in range(4)
    ]
    for lane, words in enumerate(quad_transpose(vectors)):
        for word, value in enumerate(words):
            assert [(value >> (4 * i)) & 15 for i in range(8)] == codes[
                word * 32 + lane * 8 : word * 32 + lane * 8 + 8
            ]


@pytest.mark.parametrize("k", [2048, 4096])
@pytest.mark.parametrize("layout", [0, 1])
def test_scale_words_preserve_all_four_e4m3_bytes(contracts, k, layout):
    for row in (0, 7, 31, 32, 127, 128, 4095):
        for block in range(k // 64):
            offset = contracts.scale_word(row, block, k, layout) * 4
            expected = (
                row * (k // 16) + block * 4
                if layout == 0
                else (
                    (row // 128 * (k // 64) + block) * 512
                    + row % 32 * 16
                    + (row % 128 // 32) * 4
                )
            )
            assert tuple(range(offset, offset + 4)) == tuple(
                range(expected, expected + 4)
            )
