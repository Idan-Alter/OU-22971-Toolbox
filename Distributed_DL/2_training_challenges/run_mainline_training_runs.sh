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
    torchrun --standalone --nproc_per_node=2 2_training_challenges/manual_data_parallel_demo.py

run_and_log \
    more_data \
    torchrun --standalone --nproc_per_node=2 2_training_challenges/manual_data_parallel_demo.py --batch-size 256

run_and_log \
    larger_network \
    torchrun --standalone --nproc_per_node=2 2_training_challenges/manual_data_parallel_demo.py --base-channels 64 --conv-blocks 5

run_and_log \
    more_communication \
    torchrun --standalone --nproc_per_node=2 2_training_challenges/manual_data_parallel_demo.py --extra-sync-mb 256

run_and_log \
    one_slow_rank \
    torchrun --standalone --nproc_per_node=2 2_training_challenges/manual_data_parallel_demo.py --slow-rank 0 --sleep-before-sync 1.0
