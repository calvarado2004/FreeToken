from __future__ import annotations

import ctypes.util
import functools
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from freetoken.env import ENV

from .utils import load_aot

if TYPE_CHECKING:
    from abc import abstractmethod

    import torch
    from tvm_ffi import Module

    class PyNCCLCommunicator:
        @abstractmethod
        def all_reduce(self, input: torch.Tensor, op: Literal["sum"]) -> None: ...
        @abstractmethod
        def all_gather(self, output: torch.Tensor, input: torch.Tensor) -> None: ...

else:
    PyNCCLCommunicator = Any


def _nccl_search_dirs() -> list[Path]:
    """Candidate NCCL library directories, explicit override first."""
    dirs: list[Path] = []
    override = os.getenv("FREETOKEN_NCCL_LIB_DIR", "").strip()
    if override:
        dirs.append(Path(override))

    try:
        import torch

        torch_root = Path(torch.__file__).resolve().parent
        dirs.extend([torch_root / "lib", torch_root.parent / "nvidia" / "nccl" / "lib"])
    except (ImportError, OSError):
        pass

    dirs.extend(
        Path(path)
        for path in (
            "/usr/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/local/cuda/lib64",
        )
    )
    # Preserve priority while avoiding duplicate linker/rpath entries.
    return list(dict.fromkeys(dirs))


def _nccl_link_flags() -> list[str]:
    """Resolve NCCL before launching TP ranks, including wheel-bundled runtimes.

    ``-lnccl`` requires the unversioned development symlink. PyTorch wheels may ship
    only ``libnccl.so.2``; passing that file directly supports the runtime-only case
    and gives the built extension a matching rpath.
    """
    checked: list[str] = []
    for directory in _nccl_search_dirs():
        checked.append(str(directory))
        unversioned = directory / "libnccl.so"
        if unversioned.is_file():
            return [f"-L{directory}", f"-Wl,-rpath,{directory}", "-lnccl"]
        versioned = sorted(directory.glob("libnccl.so.*"), reverse=True)
        if versioned:
            return [str(versioned[0]), f"-Wl,-rpath,{directory}"]

    soname = ctypes.util.find_library("nccl")
    if soname:
        if os.path.isabs(soname):
            directory = Path(soname).parent
            return [soname, f"-Wl,-rpath,{directory}"]
        # GNU ld accepts an exact soname after ``-l:`` even when the development
        # package did not provide the unversioned ``libnccl.so`` symlink.
        return [f"-l:{soname}"]

    locations = ", ".join(checked)
    raise RuntimeError(
        "PyNCCL tensor parallelism requires an NCCL library, but none was found. "
        "Install libnccl2 plus libnccl-dev (or the distribution equivalents), "
        "or set FREETOKEN_NCCL_LIB_DIR to a directory containing libnccl.so[.N]. "
        f"Checked: {locations}"
    )


@functools.cache
def _load_nccl_module() -> Module:
    return load_aot(
        "pynccl", cuda_files=["pynccl.cu"], extra_ldflags=_nccl_link_flags()
    )


@functools.cache
def _get_pynccl_wrapper_cls():
    import tvm_ffi

    @tvm_ffi.register_object("freetoken.NCCLWrapper")
    class PyNCCLImpl(tvm_ffi.Object):
        def __init__(self, *args):
            self.__ffi_init__(*args)

    return PyNCCLImpl


def init_pynccl(
    *,
    tp_rank: int,
    tp_size: int,
    tp_cpu_group: torch.distributed.ProcessGroup,
    max_size_bytes: int = 0,
) -> PyNCCLCommunicator:
    import torch

    max_size_bytes = min(max_size_bytes, ENV.PYNCCL_MAX_BUFFER_SIZE.value)

    module = _load_nccl_module()
    cls = _get_pynccl_wrapper_cls()

    if tp_rank == 0:
        id_list = [module.create_nccl_uid()]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )
    else:
        id_list = [None]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )

    nccl_id = id_list[0]
    assert not nccl_id is None, f"Failed to get NCCL unique ID on {tp_rank = }"

    # bypass type checking for the FFI object
    return cls(tp_rank, tp_size, max_size_bytes, nccl_id)  # type: ignore
