# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load isolated CPU contracts without importing vLLM or initializing CUDA."""

import ast
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(scope="session")
def repo_root():
    return Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def source_module(repo_root):
    def load(relative):
        path = repo_root / relative
        name = "_glm53_cpu_" + path.stem
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {name: module}):
            spec.loader.exec_module(module)
        return module

    return load


@pytest.fixture(scope="session")
def source_tree(repo_root):
    def parse(relative):
        path = repo_root / relative
        return ast.parse(path.read_text(), filename=str(path))

    return parse


@pytest.fixture(scope="session")
def execute_ast():
    def execute(nodes, namespace):
        future = ast.ImportFrom(
            module="__future__",
            names=[ast.alias(name="annotations")],
            level=0,
        )
        module = ast.fix_missing_locations(
            ast.Module(body=[future, *nodes], type_ignores=[])
        )
        namespace = dict(namespace)
        exec(compile(module, "<glm53-cpu-contract>", "exec"), namespace)
        return namespace

    return execute
