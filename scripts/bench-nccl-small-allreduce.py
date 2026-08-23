#!/usr/bin/env python3
"""Measure the tiny BF16 all-reduces used by tensor-parallel decode.

Run this with one process per GPU, for example::

    torchrun --standalone --nproc-per-node=4 \
        scripts/bench-nccl-small-allreduce.py

NCCL reads algorithm/protocol overrides before process-group initialization, so
separate invocations can compare the topology choices without loading a model::

    NCCL_ALGO=Ring NCCL_PROTO=LL torchrun ...

The CUDA-graph numbers are the relevant ones for FreeToken decode.  Eager
numbers are included to distinguish an intrinsically slow collective from a
graph-capture incompatibility.
"""

from __future__ import annotations

import argparse
import os
import statistics

import torch
import torch.distributed as dist


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="1,2,3,4,5,6")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def _measure(run, *, warmup: int, iterations: int, repeats: int) -> float:
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            run()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return float(statistics.median(samples))


def main() -> None:
    args = _parse_args()
    rows = [int(value) for value in args.rows.split(",")]
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    if dist.get_rank() == 0:
        print(
            "world_size,algo,proto,rows,bytes,eager_us,graph_us",
            flush=True,
        )

    for row_count in rows:
        tensor = torch.ones(
            (row_count, args.hidden_size),
            dtype=torch.bfloat16,
            device="cuda",
        )
        eager_ms = _measure(
            lambda: dist.all_reduce(tensor),
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )

        dist.barrier()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dist.all_reduce(tensor)
        graph_ms = _measure(
            graph.replay,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )

        if dist.get_rank() == 0:
            print(
                f"{dist.get_world_size()},"
                f"{os.environ.get('NCCL_ALGO', 'Auto')},"
                f"{os.environ.get('NCCL_PROTO', 'Auto')},"
                f"{row_count},{tensor.numel() * tensor.element_size()},"
                f"{eager_ms * 1000.0:.2f},{graph_ms * 1000.0:.2f}",
                flush=True,
            )

        # NCCL graph registrations belong to the process group.  Releasing the
        # graph before destroying that group avoids a shutdown wait on live
        # capture resources after all measurements have already completed.
        graph.reset()
        del graph
        torch.cuda.synchronize()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
