# DeepSeek-V4-Flash: next DSpark performance milestones

This is the execution plan after establishing the coherent approximately 30 tok/s coding
baseline on Carlos Alvarado's 4 x RTX A4000 workstation. Estimates are experiment targets,
not achieved-speed claims.

## Hardware and bottleneck model

- Tensor parallelism: TP=4, with two ranks bound to each of two NUMA nodes.
- GPU memory bandwidth: 448 GB/s per RTX A4000, or 1,792 GB/s aggregate across four
  independently attached cards. The per-card figure is from NVIDIA's
  [RTX A4000 data sheet](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs21/rtx-a4000/nvidia-rtx-a4000-datasheet.pdf).
- Measured model placement: 72.6% CPU and 27.4% GPU.
- Measured overlapped expert path: 25.37 GB/s CPU compute and 9.85 GB/s PCIe fetch;
  standalone H2D was 12.3 GB/s.
- The current automatic hybrid split therefore fetches about 28% of expert misses. That
  split is derived from a single-rank bandwidth profile, while the serving workload has
  four simultaneous ranks and two ranks contending per NUMA node. It is a starting point,
  not an assumption that outranks a TP4 measurement.
- Current confirmed startup geometry: memory ratio 0.80, 2,016 MoE slots, 477 full-KV
  pages (61,056 tokens), 1.26 GiB free after target verify graph capture.
- Current measured DSpark costs: draft about 23.47 ms; target verify rows 1..6 about
  61.17, 92.24, 106.99, 133.64, 148.22 and 157.79 ms.
- With gamma=5, perfect acceptance is bounded at about 33.1 tok/s by the measured cycle
  cost. The standard coding workload's approximately 84% acceptance predicts about
  28.7 tok/s and agrees with the observed 28--30 tok/s.

The practical implication is that aggregate VRAM bandwidth helps resident tensor-parallel
work, but an expert miss still crosses a roughly 10--12 GB/s per-rank host link. The first
milestone tunes that boundary. The second removes Python/launch overhead from the drafter.

## Ranked goals

1. **TP4-aware hybrid split and cache tuning: projected 3--12%.** Measure the optimal
   PCIe-fetch/CPU-compute split under real two-ranks-per-NUMA contention, then sweep MoE
   residency without changing KV capacity or graph headroom.
2. **Drafter CUDA graph: projected 2--6% end to end.** Capture only fixed-shape drafter
   work that is graph-safe; keep context-KV maintenance and any dynamic request work eager,
   following the checkpoint's DSpark algorithm and the working implementation's ordering.
3. **Better fallback probes: projected 5--12% on difficult reasoning, neutral on code.**
   Retain observed-acceptance control; do not infer difficulty from prompt text.
4. **Confidence validation and scheduling refinements: projected 0--3% for one request,
   potentially more under concurrency.** Preserve exact sampling semantics and quality.
5. **Remaining collective/kernel tuning: projected 1--5%.** Re-profile first. On the current
   branch, `fast_index_copy_multi` is a larger CUDA item than NCCL, unlike the older profile.

The percentages are not additive promises. Interactions and an unchanged acceptance rate can
make combined gains smaller than the sum.

## Milestone 1: TP4-aware split and cache experiment

The deployment script accepts three independent experiment pins:

- `MOE_HYBRID_FETCH_FRACTION`: fraction of each step's misses fetched over PCIe.
- `MOE_CACHE_SIZE`: total GPU MoE expert slots.
- `NUM_PAGES`: full-KV page count.

Unset variables retain the confirmed automatic production behavior. A cache experiment must
pin both `MOE_CACHE_SIZE` and `NUM_PAGES`; otherwise auto-sizing can exchange KV capacity for
expert slots and confound both throughput and OOM conclusions.

### Split sweep

Hold `MOE_CACHE_SIZE=2016` and `NUM_PAGES=477`. Test fractions 0.20, 0.24, 0.28, 0.32
and 0.36. For each clean restart:

1. Confirm the exact pushed commit and startup geometry.
2. Record the target-verify curve and post-capture free VRAM.
3. Warm the five-step adaptive draft-cost calibration.
4. Run the fixed coding control and the fixed difficult-reasoning control.
5. Calculate request-local acceptance by subtracting speculative counters.
6. Reject incoherence, repetition, missing final answers, OOMs, or any coding result below
   the protected baseline beyond normal run-to-run noise.

Use fixed output budgets so a branch comparison is not confounded by different stopping
points: 6,000 tokens for the pet-store coding task, 20,000 for the difficult high-reasoning
task, and 6,000 for poem/creative-prose controls. A reasoning run that exhausts its budget
without emitting a final answer fails the quality gate even when its hidden reasoning contains
the right intermediate result.

Promote the best repeated result, not the best single window. If the optimum is materially
different across the two workloads, retain automatic 28% and treat a request-level split as a
future scheduling question rather than overfitting the default.

### Cache sweep

With the best split fixed, trade only verified free headroom for MoE slots while keeping
`NUM_PAGES=477`. Start conservatively around 2,080, 2,144 and 2,208 slots. Every point must
retain at least 768 MiB after all target and drafter graph captures. Stop increasing at the
first capture OOM or headroom failure. The 0.90-memory-ratio/2,509-slot capture OOM is already
a known failed point and need not be repeated.

## Milestone 2: drafter CUDA graph

The first graph target is the fixed TP4, one-request, gamma=5 drafter backbone/head path.
Context feature-KV updates, request metadata construction, variable-length work and host-side
acceptance stay eager until each is proven capture-safe. This avoids repeating the target-graph
failure where a prefill overlap stream waited on uncaptured work.

Implementation gates:

1. Eager and replayed draft token IDs and confidence values match under deterministic greedy
   validation.
2. Temperature sampling remains statistically and semantically unchanged; graph capture must
   not freeze RNG state or replay identical random draws.
3. Drafter context commit ordering matches the DSpark paper and the known-good vLLM fork.
4. Capture fits the 0.80 deployment with at least 768 MiB post-capture headroom.
5. The measured drafter cost falls from approximately 23.47 ms and end-to-end coding improves
   in repeated matched tests without reducing acceptance.
6. Long prefill, OpenWebUI-style requests, tool calls, four concurrent requests, coding,
   reasoning and repetition/coherence tests remain stable before promotion.

`FREETOKEN_DSPARK_DRAFT_GRAPH=0` disables only the new drafter graph while leaving the
rest of the exact experiment commit unchanged. Use it for the matched eager A/B and as the
rollback path if capture headroom or graph replay fails.

## Branch and promotion discipline

- Protected recovery: `milestone/dsv4-dspark-30tps` at `986823e`; do not move it.
- Confirmed production baseline: `prod/dsv4-dspark-enhanced-v2` at `2ce0070`.
- These experiments live on `experiment/dsv4-tp4-drafter-graph`.
- Lenovo source deployment is Git-only: commit, push to Carlos's fork, then fetch and pull
  that branch on the server. Direct source-file copies are forbidden.
- Promote each gain independently only after an A/B passes throughput, quality, stability,
  memory, long-context and tool-use gates. Preserve the last confirmed-gains branch when a
  new production candidate is created.
