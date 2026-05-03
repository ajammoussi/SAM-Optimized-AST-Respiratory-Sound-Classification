#!/bin/bash
# ========================================================================
# DGX W&B Sweep Agent Launcher (ICBHI AST-SAM)
# ========================================================================
# Adapted from DomainBed exp/dgx_scripts/dgx_sweep_agents.sh, simplified
# (no flock / health monitor / retries — keep it simple for a 6h run).
#
# Prereqs:
#   1. wandb login        (or set WANDB_API_KEY in .env)
#   2. wandb sweep exp/sweeps/icbhi_focused.yaml
#      → prints "Created sweep with ID: <id>"
#   3. SWEEP_ID=<entity>/<project>/<id> bash exp/dgx/run_agents.sh
#
# Optional env vars:
#   NUM_GPUS    — override GPU count (default: nvidia-smi count)
#   AGENT_ARGS  — extra flags passed to `wandb agent` (e.g. --count 6)
# ========================================================================

set -euo pipefail

if [[ -z "${SWEEP_ID:-}" ]]; then
    echo "ERROR: SWEEP_ID environment variable is required"
    echo "Usage:  SWEEP_ID=<entity/project/id> bash $0"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}"
LOG_DIR="${PROJECT_ROOT}/logs/agents"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

declare -a AGENT_PIDS=()

cleanup() {
    echo "[run_agents] cleanup: killing $((${#AGENT_PIDS[@]})) agent(s) …"
    for pid in "${AGENT_PIDS[@]:-}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup SIGINT SIGTERM EXIT

echo "========================================================================"
echo " DGX W&B Sweep Agent Launcher"
echo "========================================================================"
echo "  Sweep ID    : ${SWEEP_ID}"
echo "  GPUs        : ${NUM_GPUS}"
echo "  Project root: ${PROJECT_ROOT}"
echo "  Log dir     : ${LOG_DIR}"
echo "========================================================================"

for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    log_file="${LOG_DIR}/agent_gpu${gpu}_${TIMESTAMP}.log"
    echo "[run_agents] launching agent on GPU ${gpu}  (log: ${log_file})"
    CUDA_VISIBLE_DEVICES="${gpu}" \
        wandb agent ${AGENT_ARGS:-} "${SWEEP_ID}" >> "${log_file}" 2>&1 &
    AGENT_PIDS+=($!)
    sleep 2  # stagger to avoid HF Hub thundering herd
done

echo "[run_agents] all ${NUM_GPUS} agents launched. Streaming master log …"
echo "[run_agents] dashboard: https://wandb.ai/${SWEEP_ID%/*}"
echo "[run_agents] tail logs/agents/*.log to follow individual GPUs"

wait "${AGENT_PIDS[@]}" 2>/dev/null || true
echo "[run_agents] all agents finished"
