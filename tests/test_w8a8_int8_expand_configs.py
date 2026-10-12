# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
from pathlib import Path

import pytest

import flaggems_vllm
from flaggems_vllm import runtime

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "metax",
    reason="MetaX INT8 expansion spaces",
)


def _entries():
    backend = importlib.import_module(
        f"flaggems_vllm.runtime.backend._{flaggems_vllm.vendor_name}.ops.w8a8_block_int8_matmul"
    )
    return [
        backend._w8a8_block_int8_kernel_general,
        backend._w8a8_block_int8_kernel_swap,
        backend._w8a8_block_int8_gemv_kernel_tuned,
        backend._w8a8_block_int8_kernel_splitk,
    ]


@pytest.fixture(autouse=True)
def _restore_tuner_state(monkeypatch):
    if flaggems_vllm.vendor_name != "metax":
        return
    flagtune_runtime = importlib.import_module("flaggems_vllm.runtime.flagtune")
    monkeypatch.setattr(flagtune_runtime, "_include_ops", None)
    monkeypatch.delenv("USE_FLAGTUNE", raising=False)
    monkeypatch.delenv("FLAGTUNE_INCLUDE", raising=False)
    states = [
        (
            entry,
            dict(entry.fn.__dict__),
            tuple(dict(cache) for cache in entry.kernel_cache),
        )
        for entry in _entries()
    ]
    yield
    # Configuration switching updates hashes and persistent-cache bindings too.
    # Restore all tuner state so subsequent precision tests use their own space.
    for entry, state, caches in states:
        entry.fn.__dict__.clear()
        entry.fn.__dict__.update(state)
        entry.kernel_cache = caches


def _fingerprint(config):
    return (
        tuple(sorted(config.kwargs.items())),
        config.num_warps,
        config.num_stages,
        config.num_ctas,
    )


def test_expand_loads_and_preserves_defaults():
    for entry in _entries():
        tuner = entry.fn
        name = tuner._flagtune_expand_op_name
        spec = runtime.get_expand_config(name)
        assert spec != -1, name
        assert len(spec["strategy"]) == len(tuner.keys)
        configs = runtime.ops_get_configs(name)
        assert len(configs) > len(tuner._flagtune_default_configs)
        fingerprints = {_fingerprint(config) for config in configs}
        assert len(fingerprints) == len(configs)
        assert {
            _fingerprint(config) for config in tuner._flagtune_default_configs
        } <= fingerprints
        assert len({tuple(sorted(config.kwargs)) for config in configs}) == 1
        assert all(set(config.kwargs) <= set(entry.arg_names) for config in configs)
        for config in configs:
            if "SPLITS" in config.kwargs:
                assert 1 <= config.kwargs["SPLITS"] <= 16


def test_expand_pruning_keeps_valid_candidates():
    for entry in _entries():
        tuner = entry.fn
        prune = tuner.early_config_prune
        if prune is None:
            continue
        configs = runtime.ops_get_configs(tuner._flagtune_expand_op_name)
        for m, n, k in [(1, 65, 130), (7, 129, 257), (128, 1024, 4096)]:
            # GROUP_K is supplied as a keyword at launch, not a positional arg.
            args = {"M": m, "N": n, "K": k}
            retained = prune(configs, args, GROUP_N=128, GROUP_K=128)
            assert retained
            for config in retained:
                if config.kwargs.get("VECTOR"):
                    if "SPLITS" in config.kwargs:
                        assert m == 1 and config.kwargs["SPLITS"] == 1
                    if "BLOCK_K" in config.kwargs:
                        assert config.kwargs["BLOCK_K"] == 128
                if config.kwargs.get("SCHEDULE") in (2, 3):
                    assert m <= 8


def test_expand_switch_invalidates_entry_cache(monkeypatch):
    monkeypatch.delenv("USE_FLAGTUNE", raising=False)
    monkeypatch.delenv("FLAGTUNE_INCLUDE", raising=False)
    for entry in _entries():
        entry._apply_flagtune()
        defaults = entry.fn.configs
        for cache in entry.kernel_cache:
            cache["expansion-test-canary"] = None
        monkeypatch.setenv("USE_FLAGTUNE", "1")
        entry._apply_flagtune()
        assert entry.fn._flagtune_active
        assert len(entry.fn.configs) > len(defaults)
        assert all("expansion-test-canary" not in cache for cache in entry.kernel_cache)
        monkeypatch.delenv("USE_FLAGTUNE", raising=False)
        entry._apply_flagtune()
        assert not entry.fn._flagtune_active
        assert entry.fn.configs is defaults


def test_range_pre_hook_and_config_mapping():
    entry = _entries()[0]
    name = entry.fn._flagtune_expand_op_name
    hook = lambda args: None
    configs = runtime.ops_get_configs(name, pre_hook=hook)
    assert configs and all(config.pre_hook is hook for config in configs)
    assert all(config.pre_hook is None for config in runtime.ops_get_configs(name))
    metax_yaml = (
        Path(runtime.__file__).parent
        / "backend/_metax/w8a8_block_int8_matmul_metax_expand.yaml"
    )
    if metax_yaml.exists():
        legacy = runtime.ops_get_configs(
            "w8a8_block_int8_matmul", yaml_path=str(metax_yaml)
        )
        assert legacy and all(
            set(config.kwargs) == {"BLOCK_M", "BLOCK_N", "GROUP_M"} for config in legacy
        )
