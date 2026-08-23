# DeepSeek-V4-Flash DSpark performance log — 2026-08-23

This note records measured behavior on Carlos Alvarado's Lenovo workstation. It is an
experiment log, not a universal performance claim.

## Test system

- Model: DeepSeek-V4-Flash-0731
- Engine: FreeToken, tensor parallel size 4
- GPUs: 4 x NVIDIA RTX A4000 16 GiB (Ampere)
- Host: two NUMA nodes; two TP ranks and 20 assigned physical CPU cores per node
- Deployment memory ratio: 0.80
- Hybrid expert fetch: automatic 28% unless explicitly stated otherwise
- Counted model placement: 72.6% CPU / 27.4% GPU
- Free GPU memory after CUDA graph capture: 1.26 GiB

The Lenovo must deploy source by fetching a pushed Git branch. Direct source-file copies are
forbidden.

## Protected and experimental branches

| Branch | Commit | Purpose |
| --- | --- | --- |
| `milestone/dsv4-dspark-30tps` | `986823e` | Protected recovery point; do not advance casually. |
| `prod/dsv4-dspark-enhanced-v2` | `2ce0070` | Confirmed pre-graph-fix enhanced baseline. |
| `prod/dsv4-dspark-enhanced-v3` | `000b41f` | Graph-capture fixes plus 2048-token prefill default. |
| `experiment/dsv4-v2-prefill2048` | `606b7cf` | Enhanced-v2 plus the 2048-token prefill default and this measurement log. |
| `experiment/dspark-acceptance-fallback` | `7a93e04` | Opt-in measured-acceptance circuit breaker under A/B test. |
| `experiment/dsv4-prefill-profile` | `91a748f` | Opt-in bounded one-shot prefill profiler; pushed but not deployed. |

## Confirmed prefill gain

Changing the server profile's default prefill chunk from 1024 to 2048 produced the largest
confirmed gain of the day.

| Fixed 7,924-token prompt | Full-chunk throughput | Total prefill wall time |
| --- | ---: | ---: |
| 1024-token chunks | about 321.8-323.2 tok/s | 25.79 s |
| 2048-token chunks | about 625.4-627.3 tok/s | 15.31 s |

This is approximately 1.94x full-chunk throughput and 40.6% lower prefill wall time. A separate
9,139-token prompt followed by a 200-token response completed coherently without OOM. GPU memory
remained within the 16 GiB devices. The 2048 setting is therefore a confirmed Ampere workstation
gain, not merely a profiler projection.

The long-prompt measurement was made on the graph-fix lineage. The exact
`experiment/dsv4-v2-prefill2048` commit is configured for 2048 and still needs one long-prompt
confirmation before promotion.

## Decode comparison

### Graph-fix lineage (`000b41f`)

The standard coding prompt remained coherent and produced steady windows between 27.47 and
30.07 tok/s. However, an identical high-effort 25-horses reasoning test produced these eleven
steady windows:

`17.40, 17.48, 17.16, 15.88, 15.84, 18.50, 16.10, 18.35, 16.51, 15.18, 17.06 tok/s`

Mean: **16.86 tok/s**. Request-local speculative acceptance was **705/1363 = 51.7%**.

### Enhanced-v2 plus 2048 (`f1e5c3f`)

This branch differs from `2ce0070` by only the one-line 2048-token server default. After the
five-step adaptive draft-cost warm-up, the same high-effort 25-horses prompt produced:

`20.24, 17.60, 20.65, 19.47, 21.24, 18.12, 15.13, 17.67, 17.58, 16.40, 17.93 tok/s`

Mean: **18.37 tok/s**. Request-local speculative acceptance was **756/1240 = 61.0%**. The answer
correctly derived the seven-race strategy, but a 1200-token cap stopped it during the lower-bound
proof before a final response was emitted. Several windows recovered the earlier 20-21 tok/s
milestone, but the later decline means reasoning is improved rather than fully stabilized.

The standard 600-token coding control then produced coherent HTML at **29.29 and 30.08 tok/s**,
with request-local acceptance **363/445 = 81.6%**. A prior 5.36 tok/s line belonged to the short
256-token calibration request and included idle/calibration time; it is not a steady coding
measurement. A separate user-observed coding concern should still be retested with their exact
prompt before this branch is promoted.

## Why the graph-fix lineage is suspect

The two commits between enhanced-v2 and the graph-fix head are:

- `5ba7abc`: keep DSpark verify off prefill streams
- `95e3233`: make MoE overlap graph events capture-local

At startup, the relevant target-verify cost curves were:

| Rows | Graph-fix lineage | Enhanced-v2 lineage |
| ---: | ---: | ---: |
| 1 | 61.83 ms | 58.31 ms |
| 2 | 96.30 ms | 86.38 ms |
| 3 | 134.83 ms | 109.45 ms |
| 4 | 134.83 ms | 125.37 ms |
| 5 | 145.66 ms | 147.69 ms |
| 6 | 156.38 ms | 154.62 ms |

The regression is largest at widths two and three, which are important when reasoning confidence
does not justify the full speculative prefix. This is a correlation, not yet proof of which commit
causes the regression. The next proper step is to test `5ba7abc` alone on top of enhanced-v2 plus
2048, using the same warmed prompts and request-local acceptance calculation.

## Workload sensitivity and log interpretation

Two temperature-1.0 poem requests were coherent and non-repeating but decoded at roughly
12-14 tok/s. Their request-local acceptance was about 48%, versus about 82% for the standard coding
control. Creative prose has a flatter, less predictable distribution and is intrinsically harder
for the drafter than repetitive code. These poem prompts contained only nine input tokens, so they
never exercised the 2048-token prefill path and cannot be evidence that the chunk-size change hurt
decode.

The `spec: accepted/drafted` figures in FreeToken's status log are cumulative since server start.
For a single request, subtract the counters immediately before the request from the counters at
completion. The displayed cumulative percentage can otherwise conceal a severe per-request drop.

The first prefill or decode throughput line after an idle interval is also not a steady-state
measurement because the status reporter includes time since the previous report. Use consecutive
full-chunk or decode windows instead.

## Measured-acceptance fallback experiment

Commit `7a93e04` adds an opt-in, request-local circuit breaker. It does not classify the prompt.
After 32 actual DSpark proposals, a measured acceptance rate below 60% selects ordinary target
decode for exactly 64 steps and then probes DSpark again. The ordinary path already commits each
new target hidden feature into the drafter's context KV, so a later probe resumes from current
state. The fallback is disabled by default and passed 202 DSpark/hybrid-fetch tests on the exact
pushed Lenovo worktree.

The standard 600-token pet-store landing-page control did not trip the circuit breaker. It
completed in 21.11 seconds, with steady windows of **30.55 and 27.33 tok/s**, and produced coherent
HTML/CSS until the requested length cutoff. A 256-token warm-up accepted 137/163 proposals (84%);
its final 32-proposal tail happened to fall to 56.2% and tripped only as the request completed.
That late event had no effect on the completed output, but it shows that 32 proposals remains an
experimental, somewhat reactive observation window.

The identical temperature-1.0 request `tell me a story about horses and humans` is the intended
low-acceptance control:

| Run | Completion | Wall time | End-to-end completion rate |
| --- | ---: | ---: | ---: |
| No circuit breaker (`606b7cf`, inferred from server timestamps) | about 1,536 tokens | about 98 s | about 15.7 tok/s |
| Acceptance fallback (`7a93e04`) | 1,513 tokens | 88.07 s | 17.18 tok/s |

The fallback run repeatedly measured 30.3%-59.4% local acceptance. Its target-only windows reached
**17.86-19.59 tok/s**, while mixed probe windows could still fall to about 15-17 tok/s. The response
stopped naturally, was coherent end to end, and contained no token repetition. This is roughly a
10% wall-time reduction on one matched creative-prose run; it is promising evidence, not yet a
universal claim.

Bounded bridge-and-torch logic controls found the correct 17-minute strategy immediately and did
not repeat tokens. However, both `reasoning_effort=high` and `low` consumed 1,200 completion tokens
trying to formalize the optimality proof and reached the length limit without opening the final
answer channel. This resembles the earlier 1,200-token 25-horses truncation and is not introduced
by the fallback, but it remains a quality/serving caveat: reasoning-mode output budgets or
termination behavior need a separate fix before production promotion.

### Capture-budget finding

A clean restart at `--memory-ratio 0.90` auto-sized 2,509 MoE cache slots, versus the previously
successful 2,016, and left only 1.02 GiB before graph capture. DSpark verify capture OOMed. At
`0.80`, auto-sizing returned to 2,016 slots, left 2.56 GiB before capture and 1.26 GiB afterward,
and the server started normally. The Lenovo profile therefore now defaults to 0.80. This is a
separate auto-cache/capture-budget interaction; the acceptance controller itself allocates no GPU
tensors.

## Current conclusion and promotion gate

`606b7cf` remains the best fallback-disabled combined candidate: it retains approximately 30 tok/s
on the standard coding control and improves the fixed reasoning control while carrying the 2048
prefill default. `7a93e04` adds a promising opt-in low-acceptance speedup without regressing the
coding control, but it is not yet a production declaration.

Before promotion:

1. Repeat the fixed long-prompt test and confirm approximately 625 tok/s full 2048-token chunks,
   coherent output, and no OOM on `f1e5c3f`.
2. Repeat at least two bounded high-effort reasoning prompts that reach a final answer.
3. Repeat Carlos's exact coding workload and confirm sustained approximately 30 tok/s.
4. Test `5ba7abc` and `95e3233` separately to locate the reasoning regression while retaining the
   Ada graph-capture compatibility work.
5. Do not move the protected recovery branch during this experiment.
6. Replicate the matched creative-prose A/B and tune the acceptance window/cooldown before enabling
   the circuit breaker by default.
7. Add a reasoning-mode termination/budget test that requires a final answer, not merely a correct
   hidden chain.
