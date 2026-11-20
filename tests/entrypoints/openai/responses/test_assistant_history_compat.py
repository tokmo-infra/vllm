# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import unittest
from copy import deepcopy

from pydantic import ValidationError

from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.utils import construct_input_messages


class HistoryCompatibility(unittest.TestCase):
    def parse(self, *rows):
        return ResponsesRequest.model_validate({"model": "unit", "input": list(rows)})

    def test_compact_history_round_trip(self):
        parts = [{"type": "output_text", "text": x} for x in ("lotus", " 87")]
        row = {"role": "assistant", "content": parts}
        original = deepcopy(row)
        for item in (row, dict(row, type="message")):
            r = self.parse(item)
            out = construct_input_messages(request_input=r.input)
            self.assertEqual(out, [{"role": "assistant", "content": "lotus 87"}])
        self.assertEqual(row, original)

    def test_history_and_image(self):
        hist = {"role": "assistant", "content": [{"type": "output_text", "text": "A"}]}
        image = {"type": "input_image", "image_url": "https://example.invalid/test.png"}
        for overrides, expected in (({}, "auto"), ({"detail": "high"}, "high")):
            r = self.parse(hist, {"role": "user", "content": [{**image, **overrides}]})
            self.assertEqual(
                r.model_dump()["input"][1]["content"][0]["detail"], expected
            )

    def test_plain_assistant_text(self):
        r = self.parse({"role": "assistant", "content": "hello"})
        self.assertEqual(r.model_dump()["input"][0]["content"], "hello")

    def test_validation_preserved(self):
        with self.assertRaises(ValidationError):
            self.parse({"role": "assistant", "content": [{"type": "output_text"}]})
        with self.assertRaises(ValidationError):
            self.parse(
                {"role": "user", "content": [{"type": "output_text", "text": "x"}]}
            )


if __name__ == "__main__":
    unittest.main()
