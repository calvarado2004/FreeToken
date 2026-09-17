from __future__ import annotations

from unittest.mock import patch

from freetoken.server.args import parse_args


class _Config:
    def to_dict(self) -> dict:
        return {"architectures": ["DeepseekV4ForCausalLM"], "torch_dtype": "bfloat16"}


def _parse(extra: list[str]):
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: _Config()):
        return parse_args(["--model", "/models/anon", *extra])[0]


def test_sampling_defaults_are_unset_unless_given():
    args = _parse([])
    assert (args.default_temperature, args.default_top_p, args.default_top_k) == (None, None, None)


def test_sampling_defaults_are_parsed():
    args = _parse(["--default-temperature", "1.0", "--default-top-p", "0.95", "--default-top-k", "40"])
    assert args.default_temperature == 1.0
    assert args.default_top_p == 0.95
    assert args.default_top_k == 40
