from pathlib import Path

import pytest

from freetoken.kernel import pynccl


def test_nccl_discovery_accepts_versioned_runtime_library(tmp_path, monkeypatch):
    lib = tmp_path / "libnccl.so.2"
    lib.touch()
    monkeypatch.setattr(pynccl, "_nccl_search_dirs", lambda: [Path(tmp_path)])
    monkeypatch.setattr(pynccl.ctypes.util, "find_library", lambda _name: None)

    flags = pynccl._nccl_link_flags()

    assert str(lib) in flags
    assert f"-Wl,-rpath,{tmp_path}" in flags


def test_nccl_discovery_fails_before_the_jit_link(monkeypatch):
    monkeypatch.setattr(pynccl, "_nccl_search_dirs", lambda: [])
    monkeypatch.setattr(pynccl.ctypes.util, "find_library", lambda _name: None)

    with pytest.raises(RuntimeError, match="libnccl-dev"):
        pynccl._nccl_link_flags()
