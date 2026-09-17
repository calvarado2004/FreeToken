#!/usr/bin/env bash
# Start / stop GLM-5.3-Flash on the 4x RTX A4000, served with HTTPS on 0.0.0.0:8081 as
# lenovo-thinkstation.levelg.io. Same commands as freetoken-dsv4.sh (start, stop, kill,
# status, logs, test), which does the work; this only sets the GLM defaults.
#
# The checkpoint is RedHatAI/GLM-5.3-Flash-NVFP4: NVFP4 routed experts (W4A16, served by the
# Triton kernel and the CPU executor, both of which run on Ampere) and bf16 everything else.

set -uo pipefail

export MODEL="${MODEL:-$HOME/models/GLM-5.3-Flash-NVFP4}"
export MODEL_LABEL="${MODEL_LABEL:-GLM-5.3-Flash NVFP4}"
export LOG="${LOG:-/tmp/freetoken-glm53.log}"
export PIDFILE="${PIDFILE:-/tmp/freetoken-glm53.pid}"
# No dSpark drafter for GLM; the fallback flags belong to it.
export SPECULATIVE_DSPARK=0
export DSPARK_FALLBACK_ACCEPTANCE=0
# MLA keeps one 512-wide latent per token, so a large KV floor costs little VRAM next to experts.
export KV_RESERVE_TOKENS="${KV_RESERVE_TOKENS:-131072}"
# Z.ai's recommended sampling (model card); the checkpoint's generation_config.json only
# carries temperature, so without these an API client that sends nothing gets top_p 1.0.
export DEFAULT_TEMPERATURE="${DEFAULT_TEMPERATURE:-1.0}"
export DEFAULT_TOP_P="${DEFAULT_TOP_P:-0.95}"

exec /usr/bin/bash "$(dirname "$(readlink -f "$0")")/freetoken-dsv4.sh" "$@"
