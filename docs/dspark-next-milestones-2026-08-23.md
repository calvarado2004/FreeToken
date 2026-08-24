# DeepSeek-V4-Flash: next DSpark performance milestones

This is the execution plan after establishing the coherent approximately 30 tok/s coding
baseline on Carlos Alvarado's 4 x RTX A4000 workstation. Estimates are experiment targets,
not achieved-speed claims. Carlos makes the final call on what constitutes a baseline; this
document keeps measurements, caveats and recommendations separate.

## End-of-day checkpoint and immediate next steps

The Lenovo is currently serving commit `6b3de2d` from
`experiment/dsv4-tp4-drafter-graph`. The deployed systemd profile deliberately uses the more
stable eager drafter (`FREETOKEN_DSPARK_DRAFT_GRAPH=0`) while retaining DSpark, the measured
acceptance fallback, 2,048-token prefill chunks and the confirmed TP4 memory geometry. The
service is active but deliberately disabled at boot.

Pinned runtime geometry:

- TP=4; memory ratio 0.80; MoE cache 2,016 slots; 477 KV pages.
- Hybrid expert fetch fraction 0.28.
- Acceptance fallback threshold/minimum/cooldown: 0.60 / 32 proposals / 64 target steps.
- 61,056-token allocated full-KV capacity for prompt plus output, despite the checkpoint's
  architectural 1,048,576-token context limit.
- Final measured placement: 146.62 GiB of host experts and 55.24 GiB of GPU allocations,
  reported as 72.6% CPU / 27.4% GPU for counted placement. Each NUMA node owns two TP ranks.
- 1.26 GiB free GPU memory after target-verify graph capture.

The matched rich pet-store workload showed why the drafter graph remains opt-in:

| Same experiment commit and 9k limit | End-to-end | Steady mean | Steady jitter | Fallback activations |
| --- | ---: | ---: | ---: | ---: |
| Drafter graph enabled | 5,451 tokens / 217.775 s = 25.03 tok/s | 23.97 tok/s | 4.84 tok/s stddev | 12 |
| Eager drafter | 7,168 tokens / 289.292 s = 24.78 tok/s | 24.23 tok/s | 3.72 tok/s stddev | 7 |

The graph reduced the measured drafter stage to about 18.43 ms versus about 21.71 ms in the
matched eager run, but yielded only about 1% end-to-end throughput at materially higher jitter
and more fallback activations. Both responses were complete, coherent, non-repeating and judged
visually excellent. Recommendation: keep eager as the release candidate and retain the graph
behind its environment switch until its variance is understood.

Work should resume in this order:

1. **Freeze and reproduce the candidate.** Re-run one 6k pet-store coding control, one bounded
   representative 20k-cap high-reasoning control, one 6k creative control and one OpenWebUI
   long-history request from the exact systemd commit. Record request-local counters rather than
   cumulative `spec:` percentages. Do not use adversarial open-ended proof prompts as routine
   gates; reserve them for labeled stress tests.
2. **Complete the TP4-aware hybrid split sweep.** With graph disabled, pin MoE=2,016 and
   KV=477, then repeat fractions 0.20, 0.24, 0.28, 0.32 and 0.36. Compare repeated runs, not a
   single best window. Preserve 2,048-token prefill and the fallback so each point differs only
   in the split.
3. **Sweep MoE residency only after selecting the split.** Try 2,080, 2,144 and 2,208 slots
   while keeping 477 KV pages. Reject any point below 768 MiB post-capture free VRAM or showing
   an OOM, quality loss or decode regression. Do not repeat the known failed auto-sized
   2,509-slot / memory-ratio 0.90 capture.
4. **Diagnose the drafter graph, do not promote it yet.** Profile the extra variance and fallback
   entries, verify CUDA RNG/sampling semantics, and test whether graph replay changes scheduling
   or synchronization around the sequential Markov sampler. A graph result must beat eager in
   repeated end-to-end runs, not merely reduce the draft substage.
5. **Try a one-token MTP/target fallback only as an isolated later experiment.** It is most
   relevant when DSpark acceptance is persistently weak on reasoning or creative prose. Keep
   code on DSpark when its measured acceptance remains high; never classify difficulty from the
   prompt text.
6. **Run the deferred stability matrix before release.** Include tool calls, repeated-token and
   coherence checks, an OpenWebUI long-context request, a Qwen-Code-like large-instruction
   request within allocated KV capacity, and four concurrent coding/reasoning requests. Treat
   concurrency as a separate throughput/stability result because the current service pins one
   running request.
7. **Publish only confirmed gains.** Create a new protected PROD branch and a release on Carlos's
   fork only after the exact candidate passes the matrix. Keep the existing recovery and PROD
   branches immutable. PR #71 should receive independent net gains and their measurements; the
   drafter graph should not be promoted upstream from the present A/B.

The performance target remains approximately 35 tok/s for representative reasoning and 60 tok/s
for coding. Reaching it is an experiment goal, not a claim. The present measured gamma=5 cycle
cost explains the current approximately 30 tok/s coding region, so a large jump will require a
material improvement in verify/draft cycle cost, accepted tokens per cycle, or both—not only a
small launch-overhead reduction.

## Operational handoff

The full launch stream, including the ASCII art, all TP-rank loader output, NUMA distribution and
API readiness, is mirrored to journald while `/tmp/freetoken-dsv4.log` remains available to the
management script:

```bash
journalctl -u freetoken-dsv4.service -f -o cat
sudo systemctl start freetoken-dsv4.service
sudo systemctl stop freetoken-dsv4.service
systemctl status freetoken-dsv4.service
```

Do not enable the service unless Carlos explicitly requests it. The six existing vLLM services
were also confirmed disabled and inactive. Lenovo source deployment remains Git-only: commit and
push a named branch, then fetch/switch/pull on the server. Source-file copies are forbidden.

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
