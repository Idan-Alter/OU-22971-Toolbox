#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/run_logs"

mkdir -p "${LOG_DIR}"

run_and_log() {
    local run_name="$1"
    shift

    local log_path="${LOG_DIR}/${run_name}_stdout.log"
    echo "==> Running ${run_name}"
    echo "    Log: ${log_path}"
    (
        cd "${WORKSPACE_ROOT}"
        "$@"
    ) 2>&1 | tee "${log_path}"
}

run_and_log \
    baseline \
    torchrun --standalone --nproc_per_node=2 3_profiler_cpu_traces/profile_manual_data_parallel.py --trace-name baseline

run_and_log \
    baseline_memory \
    torchrun --standalone --nproc_per_node=2 3_profiler_cpu_traces/profile_manual_data_parallel.py --trace-name baseline_memory --profile-memory
