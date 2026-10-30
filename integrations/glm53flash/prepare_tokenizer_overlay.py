# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Add tokenizer caching and MLA ABI compatibility to image 540155."""

import argparse
import ast
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--site-packages",
    type=Path,
    default=Path("/usr/local/lib/python3.12/dist-packages"),
)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
p = args.output
p.mkdir(parents=True, exist_ok=True)
source = args.site_packages / "vllm"
s = (source / "renderers/base.py").read_text()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a, s.count(a))
    s = s.replace(a, b)


sub(
    "from vllm.tokenizers import TokenizerLike\n",
    "from vllm.tokenizers import TokenizerLike\nfrom v"
    "llm.tokenizers.glm53_native_cache import NativeC"
    "ache, make_cache, timed_tokenize_async, track_re"
    "nder\n",
)
sub(
    "class BaseRenderer(ABC, Generic[_T]):\n",
    "class BaseRenderer(ABC, Generic[_T]):\n    _glm53"
    "_native_cache: NativeCache | None = None\n\n",
)
sub(
    "        self.tokenizer = tokenizer\n",
    "        self.tokenizer = tokenizer\n        self."
    "_glm53_native_cache = make_cache(config, tokeniz"
    "er)\n",
)
sub(
    "        self._async_tokenizer_decode = make_asyn"
    "c(self._decode, executor=self._executor)\n",
    "        if self._glm53_native_cache is not None:"
    "\n            self._tokenize_prompt_async = timed"
    "_tokenize_async(self, self._executor)\n        se"
    "lf._async_tokenizer_decode = make_async(self._de"
    "code, executor=self._executor)\n",
)
sub(
    "        prompt: TextPrompt,\n        params: Toke"
    "nizeParams,\n    ) -> TokensPrompt:",
    "        prompt: TextPrompt,\n        params: Toke"
    "nizeParams,\n        *,\n        _queued_at: float"
    " | None = None,\n    ) -> TokensPrompt:",
)
sub(
    '        encoding = tokenizer(prompt["prompt"], **kwargs)\n',
    "        if self._glm53_native_cache is not None:"
    "\n            ids = self._glm53_native_cache.try_"
    "encode(prompt, kwargs, want_offsets, _queued_at)"
    "\n            if ids is not None:\n               "
    " return self._build_tokens_prompt(ids, prompt)\n "
    '       encoding = tokenizer(prompt["prompt"], **'
    "kwargs)\n",
)
for phrase in (
    "        tok_prompts = self.tokenize_prompts(dict_prompts, tok_params)\n",
    (
        "        tok_prompts = await self.tokenize_prompt"
        "s_async(dict_prompts, tok_params)\n"
    ),
):
    i = s.rfind(phrase)
    assert i >= 0
    s = (
        s[:i] + '        if prompt_extras and "cache_salt" in pro'
        "mpt_extras:\n            self._apply_prompt_extra"
        's(dict_prompts, {"cache_salt": prompt_extras["ca'
        'che_salt"]})\n' + s[i:]
    )
sub(
    "    async def render_chat_async(\n",
    "    @track_render\n    async def render_chat_async(\n",
)
ast.parse(s)
(p / "base.py").write_text(s)
s = (source / "envs.py").read_text()
anchor = "    VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR: str | None = None\n"
assert s.count(anchor) == 1
s = s.replace(
    anchor,
    anchor + "    VLLM_GLM53_TOKENIZER_CACHE_GIB: float = 0\n  "
    "  VLLM_GLM53_TOKENIZER_CACHE_EPOCH_FILE: str | N"
    "one = None\n",
)
anchor = '    "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": lambda: os.getenv(\n'
assert s.count(anchor) == 1
s = s.replace(
    anchor,
    '    "VLLM_GLM53_TOKENIZER_CACHE_GIB": lambda: fl'
    'oat(os.getenv("VLLM_GLM53_TOKENIZER_CACHE_GIB", '
    '"0")),\n    "VLLM_GLM53_TOKENIZER_CACHE_EPOCH_FIL'
    'E": lambda: os.getenv("VLLM_GLM53_TOKENIZER_CACH'
    'E_EPOCH_FILE", None),\n' + anchor,
)
ast.parse(s)
(p / "envs.py").write_text(s)
# The pinned image's MLA caller has a newer default-false keyword, but its
# wrapper and compiled operator both expose the original six-argument ABI.
# Its native MLA caller never requests skip_scale=True. Restore that interface.
s = (source / "v1/attention/backend.py").read_text()
assert s.count("        skip_scale: bool = False,\n") == 1
assert s.count("            skip_scale=skip_scale,\n") == 1
s = s.replace("        skip_scale: bool = False,\n", "")
s = s.replace("            skip_scale=skip_scale,\n", "")
ast.parse(s)
(p / "backend.py").write_text(s)
print("Prepared native tokenizer hooks and six-argument MLA cache interface")
