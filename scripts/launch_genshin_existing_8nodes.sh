#!/usr/bin/env bash
# Coordinate one distributed run across existing one-node Slurm allocations.
#
# allocations.txt contains one rank:job_id:node line per node:
#   0:1234:gpu-node-01
#   1:1235:gpu-node-02
# Usage:
#   ALLOCATIONS_FILE=allocations.txt NFS_DIR=/shared/project \
#     bash scripts/launch_genshin_existing_8nodes.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
: "${NFS_DIR:?Set NFS_DIR to the shared model/cache root}"
: "${ALLOCATIONS_FILE:?Set ALLOCATIONS_FILE to rank:job_id:node entries}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/opt/dlami/nvme/${USER}/genshin-training}"
MASTER_PORT="${MASTER_PORT:-29691}"
RUN_NAME="${RUN_NAME:-genshin3_llava0162_qwen35_128k_tp4_mtp1_gbs128_$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-${LOCAL_DATA_ROOT}/checkpoints/${RUN_NAME}}"
RUN_STATE_DIR="${RUN_STATE_DIR:-${NFS_DIR}/run-state/${RUN_NAME}}"

mapfile -t ALLOCATIONS < <(
    awk 'NF && $1 !~ /^#/' "$ALLOCATIONS_FILE"
)
(( ${#ALLOCATIONS[@]} > 0 )) || {
    echo "[ERR] no allocations found in $ALLOCATIONS_FILE"
    exit 1
}
IFS=: read -r first_rank first_job first_node <<<"${ALLOCATIONS[0]}"
[[ "$first_rank" == "0" ]] || {
    echo "[ERR] first allocation must have node rank 0"
    exit 1
}
MASTER_ADDR="${MASTER_ADDR:-$(getent ahostsv4 "$first_node" | awk '{print $1; exit}')}"
[[ -n "$MASTER_ADDR" ]] || {
    echo "[ERR] cannot resolve master node $first_node; set MASTER_ADDR"
    exit 1
}

mkdir -p "$RUN_STATE_DIR" "$PROJECT_ROOT/logs"
cat > "${RUN_STATE_DIR}/existing_allocations.txt" <<EOF
run_name=$RUN_NAME
master=$MASTER_ADDR:$MASTER_PORT
checkpoint_dir=$OUTPUT_DIR
allocations=${ALLOCATIONS[*]}
EOF

export PROJECT_ROOT NFS_DIR LOCAL_DATA_ROOT MASTER_ADDR MASTER_PORT RUN_NAME OUTPUT_DIR
export NNODES="${#ALLOCATIONS[@]}" DIRECT_INNER=true WANDB_MODE="${WANDB_MODE:-online}"
export PYTHON_BIN="${PYTHON_BIN:-python}"

pids=()
cleanup() {
    for pid in "${pids[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup INT TERM

echo "[launcher] run=$RUN_NAME output=$OUTPUT_DIR master=$MASTER_ADDR:$MASTER_PORT"
for entry in "${ALLOCATIONS[@]}"; do
    IFS=: read -r rank job node <<<"$entry"
    log="${PROJECT_ROOT}/logs/${RUN_NAME}.launcher.rank${rank}.log"
    echo "[launcher] rank=$rank job=$job node=$node log=$log"
    NODE_RANK="$rank" \
        srun --jobid="$job" --overlap --nodes=1 --ntasks=1 \
        --cpus-per-task=96 --mem=0 --gres=gpu:8 --nodelist="$node" \
        bash "${PROJECT_ROOT}/scripts/46_train_genshin_llava_onevision_128k_multinode.slurm" \
        >"$log" 2>&1 &
    pids+=("$!")
done

remaining=${#pids[@]}
rc=0
while (( remaining > 0 )); do
    if wait -n "${pids[@]}"; then
        ((remaining -= 1))
    else
        rc=$?
        echo "[launcher] a rank failed rc=$rc; terminating remaining ranks" >&2
        cleanup
        break
    fi
done
wait 2>/dev/null || true
echo "[launcher] finished rc=$rc"
exit "$rc"
