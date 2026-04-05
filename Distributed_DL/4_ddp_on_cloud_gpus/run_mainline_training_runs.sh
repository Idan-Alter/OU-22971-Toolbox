#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="profile_ddp_gpu.py"
LOG_DIR="$(pwd)/run_logs"

if [[ ! -f "${SCRIPT_PATH}" ]]; then
    echo "Expected ${SCRIPT_PATH} in the current working directory: $(pwd)" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"

EXTRA_ARGS=()
if [[ -n "${UNIT4_EXTRA_ARGS:-}" ]]; then
    read -r -a EXTRA_ARGS <<< "${UNIT4_EXTRA_ARGS}"
fi

run_and_log() {
    local run_name="$1"
    shift

    local log_path="${LOG_DIR}/${run_name}_stdout.log"
    echo "==> Running ${run_name}"
    echo "    Log: ${log_path}"
    "$@" 2>&1 | tee "${log_path}"
}

# These are the mainline Unit 4 training runs from the lesson markdown.
# Run this script from the directory that contains profile_ddp_gpu.py.
# By default these commands expect the documented 2-GPU setup. If you want to
# use the script's CPU smoke-test mode instead, export UNIT4_EXTRA_ARGS="--cpu".
run_and_log \
    initial \
    torchrun --standalone --nproc_per_node=2 "${SCRIPT_PATH}" "${EXTRA_ARGS[@]}" --trace-dir . --trace-name runpod_gpu_initial --num-workers 0

run_and_log \
    baseline \
    torchrun --standalone --nproc_per_node=2 "${SCRIPT_PATH}" "${EXTRA_ARGS[@]}" --trace-dir . --trace-name runpod_gpu_baseline

run_and_log \
    batch256 \
    torchrun --standalone --nproc_per_node=2 "${SCRIPT_PATH}" "${EXTRA_ARGS[@]}" --trace-dir . --trace-name runpod_gpu_batch256 --batch-size 256

run_and_log \
    resnet50 \
    torchrun --standalone --nproc_per_node=2 "${SCRIPT_PATH}" "${EXTRA_ARGS[@]}" --trace-dir . --trace-name runpod_gpu_resnet50 --model resnet50 --batch-size 128
