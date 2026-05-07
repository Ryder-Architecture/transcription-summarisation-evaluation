#!/usr/bin/env bash
# Stage 99 — Auto-driven full pipeline.
#
# Cycles through each candidate, loading it into the experiment vLLM container
# (docker-compose.experiment.yml), waiting for /v1/models, running stage 03 +
# stage 04, then swapping. Finally loads the judge model and runs stage 05/06/07.
#
# Re-runs are safe: each stage is idempotent. Models with completed results are
# skipped at the docker-load step so we don't burn ~1 min spinning vLLM up.
#
# Usage:
#   ./99_run_all.sh /path/to/transcription_dump.sql
#   CROSS_CHECK=1 ./99_run_all.sh /path/to/transcription_dump.sql
#   SKIP_PREP=1   ./99_run_all.sh                   # if 01+02 already done
#
# Env vars:
#   VLLM_PORT             127.0.0.1:11434  (matches docker-compose.experiment.yml)
#   GEN_CONCURRENCY       16
#   JUDGE_CONCURRENCY     8
#   JUDGE_MODEL           Qwen/Qwen3.6-27B-Instruct-FP8
#   JUDGE_QUANTIZATION    "" (FP8 native, no quant flag)
#   JUDGE_GPU_MEM_UTIL    0.55
#   JUDGE_MAX_NUM_SEQS    8
#   CANDIDATES            colon-separated "MODEL|QUANT|MEM_UTIL" entries
#   CROSS_CHECK           "1" to run stage 05 after stage 04
#   SKIP_PREP             "1" to skip stages 01 + 02

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

VLLM_PORT="${VLLM_PORT:-11434}"
VLLM_BASE="http://localhost:${VLLM_PORT}"
VLLM_CHAT_URL="${VLLM_BASE}/v1/chat/completions"
HEALTH_URL="${VLLM_BASE}/v1/models"
GEN_CONCURRENCY="${GEN_CONCURRENCY:-16}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-8}"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.experiment.yml"
CONTAINER_NAME="experiment-vllm"

# Candidates: "HF_MODEL_ID|QUANTIZATION|GPU_MEM_UTIL"
# Default matches the four models in the experiment plan.
CANDIDATES="${CANDIDATES:-Qwen/Qwen3.5-4B||0.18:QuantTrio/Qwen3.5-4B-AWQ|awq_marlin|0.18:Qwen/Qwen3.5-9B||0.25:QuantTrio/Qwen3.5-9B-AWQ|awq_marlin|0.18}"

JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3.6-27B-FP8}"
JUDGE_QUANTIZATION="${JUDGE_QUANTIZATION:-}"
JUDGE_GPU_MEM_UTIL="${JUDGE_GPU_MEM_UTIL:-0.55}"
JUDGE_MAX_NUM_SEQS="${JUDGE_MAX_NUM_SEQS:-8}"


log() {
    echo
    echo "============================================================================"
    echo "  $(date '+%H:%M:%S')  $*"
    echo "============================================================================"
}


stop_vllm() {
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "  Stopping ${CONTAINER_NAME}…"
        docker compose -f "${COMPOSE_FILE}" down --remove-orphans 2>&1 | sed 's/^/    /'
    fi
}


start_vllm() {
    local model="$1" quant="$2" mem_util="$3" max_seqs="${4:-16}"
    echo "  Loading vLLM: ${model} (quant=${quant:-none}, mem=${mem_util}, max_num_seqs=${max_seqs})"
    MODEL="${model}" \
        QUANTIZATION="${quant}" \
        GPU_MEMORY_UTILIZATION="${mem_util}" \
        MAX_NUM_SEQS="${max_seqs}" \
        docker compose -f "${COMPOSE_FILE}" up -d 2>&1 | sed 's/^/    /'
}


wait_for_ready() {
    local model="$1" max_wait=1800 elapsed=0
    echo "  Waiting for vLLM /v1/models (up to ${max_wait}s)…"
    while [ $elapsed -lt $max_wait ]; do
        if curl -sf "${HEALTH_URL}" 2>/dev/null | grep -q '"data"'; then
            echo "  Model ready after ${elapsed}s"
            return 0
        fi
        if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            echo "  ERROR: container exited"
            docker compose -f "${COMPOSE_FILE}" logs --tail 30 2>&1 | sed 's/^/    /'
            return 1
        fi
        sleep 10
        elapsed=$((elapsed + 10))
        [ $((elapsed % 60)) -eq 0 ] && echo "    …${elapsed}s"
    done
    echo "  ERROR: timed out waiting for ${model}"
    docker compose -f "${COMPOSE_FILE}" logs --tail 30 2>&1 | sed 's/^/    /'
    return 1
}


candidate_done() {
    # Returns 0 if model has all 30 executive summaries already
    local model="$1"
    local safe="${model//\//__}"
    local f="${SCRIPT_DIR}/results/summaries/${safe}.json"
    [ -f "$f" ] || return 1
    local n
    n=$(.venv/bin/python -c "import json; d=json.load(open('$f')); print(sum(1 for r in d if r.get('is_executive')))" 2>/dev/null || echo 0)
    [ "$n" -ge 30 ]
}


judgements_done() {
    local model="$1"
    local safe="${model//\//__}"
    local short="${JUDGE_MODEL##*/}"
    local f="${SCRIPT_DIR}/results/judgements/local-${short}_${safe}.json"
    [ -f "$f" ]
}


# ─── Stage 01 + 02 ───────────────────────────────────────────────────────────

if [ "${SKIP_PREP:-0}" != "1" ]; then
    DUMP="${1:-}"
    if [ -z "$DUMP" ]; then
        echo "Usage: $0 <pg_dump.sql>   (or SKIP_PREP=1 $0)" >&2
        exit 1
    fi
    log "Stage 01: prepare dataset"
    .venv/bin/python 01_prepare_dataset.py --dump "$DUMP"

    log "Stage 02: sample transcripts"
    .venv/bin/python 02_sample_transcripts.py
fi


# ─── Stage 03: per-candidate generation (auto model swap) ────────────────────

log "Stage 03: generate (cycle ${CANDIDATES})"

IFS=':' read -ra CAND_LIST <<<"${CANDIDATES}"
for entry in "${CAND_LIST[@]}"; do
    IFS='|' read -r model quant mem_util <<<"${entry}"
    log "Candidate: ${model}"

    if candidate_done "${model}"; then
        echo "  All 30 transcripts already complete — skipping container swap"
        continue
    fi

    stop_vllm
    start_vllm "${model}" "${quant}" "${mem_util}" 16
    if ! wait_for_ready "${model}"; then
        echo "  FAILED to start ${model}, moving to next candidate"
        continue
    fi

    .venv/bin/python 03_generate.py \
        --model "${model}" \
        --vllm-url "${VLLM_CHAT_URL}" \
        --gen-concurrency "${GEN_CONCURRENCY}"
done


# ─── Stage 04: load judge model and judge each candidate ─────────────────────

log "Stage 04: load judge ${JUDGE_MODEL} and judge all candidates"

stop_vllm
start_vllm "${JUDGE_MODEL}" "${JUDGE_QUANTIZATION}" "${JUDGE_GPU_MEM_UTIL}" "${JUDGE_MAX_NUM_SEQS}"
if ! wait_for_ready "${JUDGE_MODEL}"; then
    echo "  FAILED to start judge — aborting"
    exit 1
fi

for entry in "${CAND_LIST[@]}"; do
    IFS='|' read -r model _quant _mem <<<"${entry}"
    if judgements_done "${model}"; then
        echo "  ${model}: judgements file already exists, skipping (idempotent stage 04 will skip done items)"
    fi
    .venv/bin/python 04_judge_local.py \
        --judge-model "${JUDGE_MODEL}" \
        --candidate-model "${model}" \
        --vllm-url "${VLLM_CHAT_URL}" \
        --judge-concurrency "${JUDGE_CONCURRENCY}"
done


# ─── Stage 05: optional Anthropic Haiku cross-check ──────────────────────────

if [ "${CROSS_CHECK:-0}" == "1" ]; then
    if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
        echo "WARNING: CROSS_CHECK=1 but ANTHROPIC_API_KEY unset — skipping stage 05"
    else
        local_judge_short="local-${JUDGE_MODEL##*/}"
        for entry in "${CAND_LIST[@]}"; do
            IFS='|' read -r model _q _m <<<"${entry}"
            log "Stage 05: Haiku cross-check ${model}"
            .venv/bin/python 05_judge_crosscheck.py \
                --candidate-model "${model}" \
                --local-judge-name "${local_judge_short}"
        done
    fi
fi


# ─── Stage 06: derived metrics ───────────────────────────────────────────────

stop_vllm

for entry in "${CAND_LIST[@]}"; do
    IFS='|' read -r model _q _m <<<"${entry}"
    log "Stage 06: metrics ${model}"
    .venv/bin/python 06_metrics.py --model "${model}"
done


log "Done. See results/metrics/ for per-model outputs."
