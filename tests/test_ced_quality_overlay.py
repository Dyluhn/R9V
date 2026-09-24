# SPDX-License-Identifier: Apache-2.0
"""CED quality's model overlay on the CPU: its multi-source projector reading and its int8 loading.

The overlay imports vLLM, so the functions under test are taken from its source (ast) and run
alone, as the research tests that graded it did.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file  # noqa: E402

OVERLAY = (Path(__file__).resolve().parents[1]
           / "runtimes/qwen38-flash-next-gfx1201-mtp4-v3/overlays/model_ced_quality.py")
FUNCTIONS = {"_ced_split_of", "_ced_sources_of", "_ced_quantize", "_ced_load", "_ced_apply"}
GROUP = 128
SOURCES = ["boundary_16", "block_input_3", "block_input_15"]


def overlay(precision: str) -> dict:
    tree = ast.parse(OVERLAY.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS]
    assert {n.name for n in nodes} == FUNCTIONS
    namespace = {"torch": torch, "json": json, "_CED_GROUP": GROUP, "_CED_PRECISION": precision}
    exec(compile(ast.Module(nodes, []), str(OVERLAY), "exec"), namespace)
    return namespace


def bf16_projector(path: Path, sources: list[str]) -> Path:
    """A split-16 projector whose inputs are `sources`, GROUP columns each, as fit_projector writes it."""
    torch.manual_seed(0)
    width = GROUP * len(sources)
    maps = {f"layer.{layer}": torch.randn(4, width + 1).to(torch.bfloat16) for layer in range(16, 48)}
    maps["final"] = torch.randn(8, width + 1).to(torch.bfloat16)
    save_file(maps, str(path), metadata={"split": "16", "sources": json.dumps(sources)})
    return path


def stored_int8(bf16: Path, path: Path) -> Path:
    """The same projector stored as int8, as kva/quantize_projector.py writes it."""
    from safetensors import safe_open

    quantize = overlay("int8")["_ced_quantize"]
    tensors = {}
    with safe_open(str(bf16), "pt") as source:
        metadata = source.metadata()
        for name in source.keys():
            q, scale, bias = quantize(source.get_tensor(name))
            tensors.update({name: q, f"scale.{name}": scale, f"bias.{name}": bias})
    save_file(tensors, str(path), metadata=metadata)
    return path


def test_a_stored_int8_projector_loads_exactly_as_the_graded_quantize_at_load(tmp_path):
    bf16 = bf16_projector(tmp_path / "msfa.safetensors", SOURCES)
    int8 = stored_int8(bf16, tmp_path / "msfa-int8.safetensors")
    load = overlay("int8")["_ced_load"]

    graded = load(str(bf16), torch.device("cpu"))
    shipped = load(str(int8), torch.device("cpu"))

    assert graded.keys() == shipped.keys()
    assert all(graded[k].dtype == shipped[k].dtype and torch.equal(graded[k], shipped[k]) for k in graded)


def test_a_stored_int8_projector_declares_split_16_and_its_sources(tmp_path):
    int8 = stored_int8(bf16_projector(tmp_path / "msfa.safetensors", SOURCES), tmp_path / "int8.safetensors")
    functions = overlay("int8")

    assert functions["_ced_split_of"](str(int8)) == 16
    assert functions["_ced_sources_of"](str(int8), 16) == SOURCES


def test_a_source_at_or_after_the_split_is_refused(tmp_path):
    late = bf16_projector(tmp_path / "late.safetensors", ["boundary_16", "block_input_16"])

    with pytest.raises(ValueError, match="not boundary_16 or a block input before layer 16"):
        overlay("int8")["_ced_sources_of"](str(late), 16)


def test_int8_maps_apply_as_their_dequantized_bf16_matrix(tmp_path):
    bf16 = bf16_projector(tmp_path / "msfa.safetensors", SOURCES)
    functions = overlay("int8")
    maps = functions["_ced_load"](str(stored_int8(bf16, tmp_path / "int8.safetensors")), torch.device("cpu"))
    features = torch.randn(5, GROUP * len(SOURCES)).to(torch.bfloat16)
    q, scale = maps["layer.20"], maps["scale.layer.20"]
    matrix = (q.view(q.shape[0], -1, GROUP).to(torch.bfloat16) * scale[:, :, None]).view(q.shape)

    got = functions["_ced_apply"](maps, "layer.20", features)

    assert torch.equal(got, torch.addmm(maps["bias.layer.20"], features, matrix.t()))
