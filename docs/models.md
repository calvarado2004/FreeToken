# Supported models

FreeToken loads HF safetensors checkpoints directly (plus native GGUF for
Gemma-4). The checkpoints below are known-good — the prebuilt kernels are tuned
for them; other checkpoints of the same architectures work too.

| Model | HF checkpoints |
|---|---|
| DeepSeek-V4 | [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) |
| GLM-5.2 | [nvidia/GLM-5.2-NVFP4](https://huggingface.co/nvidia/GLM-5.2-NVFP4) |
| GLM-4.7 | [nvidia/GLM-4.7-NVFP4](https://huggingface.co/nvidia/GLM-4.7-NVFP4) |
| Qwen3.6 / Qwen3.5 MoE | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) ([-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)), [nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4), [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) ([-FP8](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8)) |
| Qwen3.6 dense | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) ([-FP8](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)), [nvidia/Qwen3.6-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4) |
| Qwen3-MoE | [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| gpt-oss | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b), [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) |
| Gemma-4 | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it), [nvidia/Gemma-4-26B-A4B-NVFP4](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4), [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it), [nvidia/Gemma-4-31B-IT-NVFP4](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4) .. |
| MiniMax-M2.5 | [nvidia/MiniMax-M2.5-NVFP4](https://huggingface.co/nvidia/MiniMax-M2.5-NVFP4) |
| Muse-Glimmer | [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B), [RedHatAI/Muse-Glimmer-30B-NVFP4](https://huggingface.co/RedHatAI/Muse-Glimmer-30B-NVFP4) |

## MoE backends

`ft serve --moe-backend {auto,fused,offload,cpu,hybrid}`:

- **fused** — experts resident on GPU (needs the VRAM); never auto-selected.
- **offload** — experts live in host RAM, an LRU cache of expert slots on GPU;
  misses stream over PCIe.
- **cpu** — misses are computed on the CPU instead of fetched.
- **hybrid** — per step, fetches some misses over PCIe and computes the rest on
  CPU, overlapped. Run `ft bench bw` once per machine to calibrate the split.
- **auto** — dense models always resolve to `fused`; MoE models resolve to
  `offload`, upgraded to `hybrid` when a cached `ft bench bw` profile
  recommends it.

## Tensor parallelism

`ft serve --tensor-parallel-size N` shards the model over N GPUs on one host. One
scheduler process runs per rank, and rank `i` uses `cuda:i`.

DeepSeek-V4 shards:

- MLA query and output heads (`wq_b` column-parallel, `wo_a` by output group,
  `wo_b` row-parallel with one all-reduce per block)
- the MoE intermediate dimension, for both the shared expert and the routed FP4
  experts, so the host expert banks divide by N instead of being replicated
- the vocabulary, for the embedding table and the output head

It keeps replicated: the MLA latent KV path (`wkv` and the paged KV pools — every
head reads the same latent KV), the compressors and the Lightning Indexer (every
rank must select the same blocks), and the router.

Constraints for DeepSeek-V4: N must divide `o_groups` (8), so N is 1, 2, 4 or 8.
The KV pool is replicated, so its cost per GPU does not fall with N; the weights
and the expert banks do.

TP also opens a second TCP listener for rank rendezvous. It defaults to the API
port plus one; use `--distributed-port` (alias `--rendezvous-port`) to select it
explicitly when several servers share a host.

## DeepSeek-V4 dSpark

`--speculative-dspark` enables the checkpoint's block drafter and exact target
verification. Sampling remains exact, but a speedup is not universal: draft work
and a wider target pass are paid before the accepted prefix is known. The adaptive
selector profiles every legal verification width and follows the paper's measured
`D + V(k)` objective; it does not assume a fixed acceptance threshold or linear
verification cost.

The favorable regime is high prefix survival with enough experts resident or served
by the hybrid CPU/GPU path. Structured code on a 4x RTX A4000 hybrid deployment has
measured roughly 28--30 tok/s at 80--82% proposal acceptance, from an approximately
18 tok/s non-dSpark baseline. Low-survival, offload-bound work can regress: open-ended
reasoning at about 37% acceptance measured 11--14 tok/s on that deployment, and
[independent community testing](https://github.com/FlashML-org/FreeToken/pull/69#issuecomment-5384247941)
on 2x RTX 6000 Ada measured 35.91 tok/s with dSpark versus 39.25 tok/s without it
at 42% acceptance.

There is therefore no portable break-even acceptance number. It depends on draft
cost, the measured width-specific verification curve, expert-cache residency, CPU
memory bandwidth, and PCIe bandwidth. Use `ft bench decode` for alternating A/B
measurements on representative prompts, and record acceptance and selected widths
alongside tokens/s before enabling dSpark by default for a deployment.

## Notes

- `ft checkpoint` conversion is optional — it pre-converts a checkpoint into
  FreeToken's fast-load format, and `ft serve --model` auto-detects the result.
- DeepSeek-V4 checkpoints must keep the `inference/config.json` subdir — the
  authoritative model args are read from there.
- Multimodal checkpoints are served text-only.
