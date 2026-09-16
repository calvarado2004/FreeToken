"""GLM-5.3-Flash MTP: the layer the run serves, the weights that feed it, the NVFP4 conversion.

The MTP layer continues the decoder ids (``num_layers + k``) so the MLA/DSA KV group, the
indexer slab and the expert banks address it like a decoder layer; it only exists when the run
enables it. Its block-FP8 experts are re-quantized to the NVFP4 layout every other bank uses, so
the quantizer must round-trip through exactly the dequant the banks apply.
"""

from __future__ import annotations

import importlib.util
import json
import os

import pytest
import torch

import freetoken.distributed.info as info_mod
from freetoken.distributed import DistributedInfo

_HERE = os.path.dirname(os.path.abspath(__file__))
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def _tp_module():
    spec = importlib.util.spec_from_file_location("glm_tp_shapes", os.path.join(_HERE, "test_glm5_next_tp.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _mtp_on():
    from freetoken.models.glm5_next.args import set_mtp_enabled

    set_mtp_enabled(True)
    yield
    set_mtp_enabled(False)
    info_mod._TP_INFO = DistributedInfo(0, 1)


def _mtp_config(shapes, nextn: int = 1):
    from freetoken.models.glm5_next.config import parse_config
    from freetoken.utils.hf import RawConfigShim

    raw = shapes._raw_config()
    raw["text_config"]["num_nextn_predict_layers"] = nextn
    return parse_config(RawConfigShim(raw))


def _dequant(packed, scale, global_scale):
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).flatten(-2).long()
    values = _E2M1[codes]
    return values * scale.float().repeat_interleave(16, dim=-1) * float(global_scale)


def test_nvfp4_quantizer_round_trips_through_the_bank_dequant():
    from freetoken.models.glm5_next.weight import quantize_nvfp4

    torch.manual_seed(0)
    w = torch.randn(32, 64) * 0.02
    w[3, 5] = 0.5  # one outlier row block
    packed, scale, global_scale = quantize_nvfp4(w)
    assert packed.shape == (32, 32) and packed.dtype == torch.uint8
    assert scale.shape == (32, 4) and scale.dtype == torch.float8_e4m3fn
    rebuilt = _dequant(packed, scale, global_scale)
    rel = (rebuilt - w).norm() / w.norm()
    assert rel < 0.12, f"NVFP4 round trip error {rel:.3f}"

    # the low nibble holds the even element: +6 at column 0, -3 at column 1 of one block
    exact = torch.zeros(1, 16)
    exact[0, 0], exact[0, 1] = 6.0, -3.0
    packed, _, _ = quantize_nvfp4(exact)
    assert int(packed[0, 0]) & 0xF == 7 and int(packed[0, 0]) >> 4 == 0b1101


def test_the_mtp_layer_joins_the_dsa_group_the_indexer_and_the_banks():
    shapes = _tp_module()
    plain = shapes._config()
    config = _mtp_config(shapes)
    full = next(g for g in config.attention_groups if g.name == "full")
    full_plain = next(g for g in plain.attention_groups if g.name == "full")
    assert full.layer_ids == full_plain.layer_ids + (2,)
    assert full.num_index_layers == full_plain.num_index_layers + 1
    assert config.extra_moe_layers == 1
    assert config.glm5_args.mtp_layer_ids == (2,)


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_mtp_weights_match_the_rank_model(tmp_path, tp):
    """Every MTP tensor the reader yields lands on a declared parameter with its rank shape."""
    from safetensors.torch import save_file

    from freetoken.models.glm5_next.model import Glm5NextForCausalLM
    from freetoken.models.glm5_next.weight import iter_weights

    shapes = _tp_module()
    shapes._write_checkpoint(str(tmp_path))
    with open(tmp_path / "config.json") as f:
        raw = json.load(f)
    raw["text_config"]["num_nextn_predict_layers"] = 1
    with open(tmp_path / "config.json", "w") as f:
        json.dump(raw, f)

    info_mod._TP_INFO = DistributedInfo(0, 1)
    config = _mtp_config(shapes)
    with torch.device("meta"):
        full_model = Glm5NextForCausalLM(config)
    # routed experts reach the offload banks as pieces, never through iter_weights
    mtp_full = {k: v for k, v in full_model.state_dict().items() if k.startswith("mtp_layers.") and ".mlp.experts." not in k}
    assert mtp_full, "the MTP layer was not built"

    g = torch.Generator().manual_seed(1)
    ck = {}
    for key, value in mtp_full.items():
        name = key.replace("mtp_layers.0.", "model.language_model.layers.2.")
        name = name.replace("mlp.e_score_correction_bias", "mlp.gate.e_score_correction_bias")
        ck[name] = torch.randn(*value.shape, generator=g)
    save_file(ck, str(tmp_path / "model_mtp.safetensors"))
    with open(tmp_path / "model.safetensors.index.json") as f:
        index = json.load(f)
    index["weight_map"].update({k: "model_mtp.safetensors" for k in ck})
    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    info_mod._TP_INFO = DistributedInfo(0, tp)
    with torch.device("meta"):
        model = Glm5NextForCausalLM(_mtp_config(shapes))
    declared = {k: tuple(v.shape) for k, v in model.state_dict().items() if k.startswith("mtp_layers.") and ".mlp.experts." not in k}
    loaded = {k: tuple(v.shape) for k, v in iter_weights(str(tmp_path), torch.device("cpu"), include_moe_experts=False, include_non_moe=True, include_vision=False) if k.startswith("mtp_layers.")}
    assert loaded == declared
