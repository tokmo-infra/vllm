# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from vllm.tokenizers.glm53_segment_cache_core import SegmentCache
from vllm.tokenizers.glm53_shared_segment_store import SharedSegmentStore


def test_shared_hits_exact_tokens_and_epoch_isolation(tmp_path):
    backend = Tokenizer(models.BPE(unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.train_from_iterator(
        ["hello 你好 world"],
        trainers.BpeTrainer(
            vocab_size=300,
            special_tokens=["[UNK]", "<|user|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    store = SharedSegmentStore(str(tmp_path / "shared.db"), 1 << 20, "test")

    def cache():
        copy = Tokenizer.from_str(backend.to_str())
        return SegmentCache(
            SimpleNamespace(backend_tokenizer=copy),
            None,
            0,
            shared_store=store,
            generation="one",
        )

    a, b = cache(), cache()
    text = "<|user|>hello 你好 world"
    expected = backend.encode(text, add_special_tokens=False).ids
    assert a.encode(text, "tenant")[0] == expected
    ids, counts = b.encode(text, "tenant")
    assert ids == expected and counts["reused_tokens"] > 0
    b.reset(generation="two")
    ids, counts = b.encode(text, "tenant")
    assert ids == expected and counts["reused_tokens"] == 0
    store.close()


def test_reset_does_not_reinsert_inflight_old_generation(tmp_path):
    backend = Tokenizer(models.BPE(unk_token="[UNK]"))
    backend.train_from_iterator(
        ["hello"], trainers.BpeTrainer(special_tokens=["[UNK]", "<|user|>"])
    )
    store = SharedSegmentStore(str(tmp_path / "shared.db"), 1 << 20, "test")
    entered, proceed = threading.Event(), threading.Event()

    class BlockingStore:
        def get(self, *args):
            entered.set()
            assert proceed.wait(5)
            return store.get(*args)

        def put(self, *args):
            return store.put(*args)

    cache = SegmentCache(
        SimpleNamespace(backend_tokenizer=backend),
        None,
        1 << 20,
        shared_store=BlockingStore(),
        generation="one",
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(cache.encode, "hello")
        assert entered.wait(5)
        cache.reset(generation="two")
        proceed.set()
        future.result(timeout=5)
    assert cache.snapshot()["cache_entries"] == 0
    assert cache.encode("hello")[1]["encoded_tokens"] > 0
