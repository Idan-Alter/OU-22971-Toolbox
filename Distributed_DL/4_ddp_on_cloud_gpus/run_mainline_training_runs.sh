#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="profile_ddp_gpu.py"
LOG_DIR="$(pwd)/run_logs"
TRACE_ROOT="$(pwd)/run_traces"

if [[ ! -f "${SCRIPT_PATH}" ]]; then
    echo "Expected ${SCRIPT_PATH} in the current working directory: $(pwd)" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}" "${TRACE_ROOT}"

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

run_torchrun() {
    local run_name="$1"
    local nproc_per_node="$2"
    local trace_name="$3"
    shift 3

    run_and_log \
        "${run_name}" \
        torchrun --standalone "--nproc_per_node=${nproc_per_node}" "${SCRIPT_PATH}" \
            "${EXTRA_ARGS[@]}" \
            --trace-dir "${TRACE_ROOT}/${run_name}" \
            --trace-name "${trace_name}" \
            "$@"
}

# These are the mainline Unit 4 training runs from the lesson markdown.
# Run this script from the directory that contains profile_ddp_gpu.py.
# By default the DDP commands expect the documented 2-GPU setup. If you want to
# use the script's CPU smoke-test mode instead, export UNIT4_EXTRA_ARGS="--cpu".
run_torchrun ddp_gpu_baseline 2 gpu_baseline
run_torchrun ddp_gpu_workers4 2 gpu_workers4 --num-workers 4
run_torchrun ddp_gpu_batch256 2 gpu_batch256_workers4 --batch-size 256 --num-workers 4
run_torchrun ddp_gpu_resnet50 2 gpu_resnet50_batch128_workers4 --model resnet50 --batch-size 128 --num-workers 4
run_torchrun ddp_gpu_resnet50 2 gpu_resnet50_batch256_workers4 --model resnet50 --batch-size 256 --num-workers 4
