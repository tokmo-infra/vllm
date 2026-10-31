# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU checks against real native BPE objects; no GPU/model execution."""

import multiprocessing
import tempfile
import unittest
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import TokenizersBackend

from vllm.tokenizers.glm53_segment_cache_core import SegmentCache
from vllm.tokenizers.glm53_shared_segment_store import SharedSegmentStore


def tokenizer():
    backend = Tokenizer(models.BPE(unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.train_from_iterator(
        ["hello 你好 world\n tools and code"],
        trainers.BpeTrainer(
            vocab_size=300,
            special_tokens=["[UNK]", "<|user|>", "<|assistant|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    return TokenizersBackend(tokenizer_object=backend)


def encode_child(path, raw, salt, generation, queue):
    native = TokenizersBackend(tokenizer_object=Tokenizer.from_str(raw))
    shared = SharedSegmentStore(path, 1 << 20, "test-tokenizer")
    cache = SegmentCache(native, None, 0, shared, generation)
    queue.put(cache.encode("<|user|>hello 你好 world", salt))
    shared.close()


class SharedTokenizerTests(unittest.TestCase):
    def test_process_shared_hits_preserve_ids_and_isolate_namespace(self):
        native = tokenizer()
        text = "<|user|>hello 你好 world"
        expected = native.backend_tokenizer.encode(text, add_special_tokens=False).ids
        ctx = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "shared.db")
            observed = []
            for salt, generation in [
                ("tenant", "one"),
                ("tenant", "one"),
                ("other", "one"),
                ("tenant", "two"),
            ]:
                queue = ctx.Queue()
                child = ctx.Process(
                    target=encode_child,
                    args=(
                        path,
                        native.backend_tokenizer.to_str(),
                        salt,
                        generation,
                        queue,
                    ),
                )
                child.start()
                ids, counts = queue.get(timeout=30)
                child.join(timeout=30)
                self.assertEqual(child.exitcode, 0)
                self.assertEqual(ids, expected)
                observed.append(counts)
                queue.close()
            self.assertGreater(observed[0]["encoded_tokens"], 0)
            self.assertEqual(observed[1]["encoded_tokens"], 0)
            self.assertGreater(observed[1]["reused_tokens"], 0)
            self.assertGreater(observed[2]["encoded_tokens"], 0)
            self.assertGreater(observed[3]["encoded_tokens"], 0)

    def test_private_l1_avoids_another_shared_lookup(self):
        native = tokenizer()
        with tempfile.TemporaryDirectory() as directory:
            shared = SharedSegmentStore(
                str(Path(directory) / "shared.db"), 1 << 20, "test"
            )
            cache = SegmentCache(native, None, 1 << 20, shared, "one")
            for text in ["<|user|>hello 你好", "<|assistant|>code\n world"]:
                expected = native.backend_tokenizer.encode(
                    text, add_special_tokens=False
                ).ids
                self.assertEqual(cache.encode(text)[0], expected)
                before = cache.snapshot().get("shared_misses", 0)
                self.assertEqual(cache.encode(text)[0], expected)
                self.assertEqual(cache.snapshot().get("shared_misses", 0), before)
            shared.close()

    def test_shared_budget_evicts_without_changing_returned_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            shared = SharedSegmentStore(
                str(Path(directory) / "shared.db"), 1024, "test"
            )
            for number in range(12):
                text = f"entry-{number}" + "x" * 150
                packed = number.to_bytes(4, "little") * 20
                shared.put("one", "tenant", text, packed)
                self.assertEqual(shared.get("one", "tenant", text), packed)
                self.assertLessEqual(shared.snapshot()["cache_bytes"], 1024)
            self.assertGreater(shared.snapshot()["evictions"], 0)
            shared.close()


if __name__ == "__main__":
    unittest.main()
