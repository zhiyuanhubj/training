#!/usr/bin/env bash
# Assemble one distributed node-local MCore checkpoint for a safe restart.
#
# Run inside an allocation containing the same nodes/NVMe volumes that wrote it:
#   LOCAL_CHECKPOINT=/local/.../checkpoint-100 \
#   STAGING_ROOT=/shared/resume-staging \
#   bash scripts/assemble_local_mcore_checkpoint.sh
set -euo pipefail

: "${SLURM_NNODES:?Run this script inside the multi-node Slurm allocation}"
: "${LOCAL_CHECKPOINT:?Set LOCAL_CHECKPOINT to the node-local checkpoint directory}"
: "${STAGING_ROOT:?Set STAGING_ROOT to temporary shared storage}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CHECKPOINT_NAME="$(basename "$LOCAL_CHECKPOINT")"
STAGED_CHECKPOINT="${STAGING_ROOT%/}/${CHECKPOINT_NAME}"

mkdir -p "$STAGED_CHECKPOINT"
if compgen -G "$STAGED_CHECKPOINT/iter_*/*.distcp" >/dev/null; then
    echo "[ERR] staging destination already contains shards: $STAGED_CHECKPOINT"
    echo "[ERR] use an empty STAGING_ROOT to avoid mixing checkpoint generations"
    exit 1
fi

export LOCAL_CHECKPOINT STAGED_CHECKPOINT
srun --nodes="$SLURM_NNODES" --ntasks-per-node=1 --cpus-per-task=4 \
    bash -lc '
        set -euo pipefail
        tracker="$LOCAL_CHECKPOINT/latest_checkpointed_iteration.txt"
        [[ -s "$tracker" ]] || {
            echo "[ERR] $(hostname): missing $tracker" >&2
            exit 1
        }
        iteration="$(<"$tracker")"
        [[ "$iteration" =~ ^[0-9]+$ ]] || {
            echo "[ERR] $(hostname): invalid iteration $iteration" >&2
            exit 1
        }
        iter_name="$(printf "iter_%07d" "$iteration")"
        source_iter="$LOCAL_CHECKPOINT/$iter_name"
        destination_iter="$STAGED_CHECKPOINT/$iter_name"
        mkdir -p "$destination_iter"
        shopt -s nullglob
        shards=("$source_iter"/*.distcp)
        (( ${#shards[@]} > 0 )) || {
            echo "[ERR] $(hostname): no local .distcp shards under $source_iter" >&2
            exit 1
        }
        cp "${shards[@]}" "$destination_iter/"
        if [[ "${SLURM_PROCID}" == "0" ]]; then
            cp "$LOCAL_CHECKPOINT/latest_checkpointed_iteration.txt" \
                "$LOCAL_CHECKPOINT/args.json" "$STAGED_CHECKPOINT/"
            cp "$source_iter/.metadata" "$source_iter/common.pt" \
                "$source_iter/metadata.json" "$destination_iter/"
        fi
        echo "[gather] host=$(hostname) shards=${#shards[@]} iteration=$iteration"
    '

python "${PROJECT_ROOT}/scripts/resolve_mcore_checkpoint.py" \
    --checkpoint "$STAGED_CHECKPOINT" --require-full-state --field json
echo "[gather] resumable checkpoint: $STAGED_CHECKPOINT"
echo "[gather] pass RESUME_FROM=$STAGED_CHECKPOINT while keeping OUTPUT_DIR node-local"
