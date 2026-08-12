#!/usr/bin/env bash
# Genshin-only 128K full-parameter SFT on 8 nodes / 64 H200 GPUs.
#
# Submit with cluster-specific account and partition flags:
#   sbatch -A <account> -p <partition> \
#     --export=ALL,NFS_DIR=/shared/project,LOCAL_DATA_ROOT=/local/nvme/genshin-training \
#     scripts/47_train_genshin_only_128k_multinode.slurm
#
# The same LOCAL_DATA_ROOT path and prepared dataset must exist on every node.
#SBATCH --nodes=8
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=7-00:00:00
#SBATCH --job-name=genshin-only-128k
#SBATCH --output=logs/genshin_only_128k_%j.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
: "${NFS_DIR:?Set NFS_DIR to the shared model/cache root}"
export LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/opt/dlami/nvme/${USER}/genshin-training}"
export MIXTURE_YAML="${MIXTURE_YAML:-${LOCAL_DATA_ROOT}/genshin-only/genshin-only-128k.local.yaml}"
export MODEL_PATH="${MODEL_PATH:-${NFS_DIR}/models/Qwen3.5-9B}"
export GAME_NAME="${GAME_NAME:-genshin_only_qwen35_128k}"
export RUN_NAME="${RUN_NAME:-${GAME_NAME}_tp4_mtp1_gbs128_$(date -u +%Y%m%dT%H%M%SZ)_j${SLURM_JOB_ID}}"

# local: fastest and does not consume shared capacity, but restart requires the
# same NVMe volumes plus assemble_local_mcore_checkpoint.sh. shared: portable
# across replacement nodes and directly supports AUTO_RESUME.
export CHECKPOINT_STORAGE="${CHECKPOINT_STORAGE:-local}"
case "$CHECKPOINT_STORAGE" in
    local)
        export OUTPUT_DIR="${OUTPUT_DIR:-${LOCAL_DATA_ROOT}/checkpoints/${RUN_NAME}}"
        ;;
    shared)
        export OUTPUT_DIR="${OUTPUT_DIR:-${NFS_DIR}/outputs/${RUN_NAME}}"
        ;;
    *)
        echo "[ERR] CHECKPOINT_STORAGE must be local or shared"
        exit 1
        ;;
esac

# Explicitly pin the validated production contract.
export TP="${TP:-4}"
export PP="${PP:-1}"
export MICRO_BS="${MICRO_BS:-1}"
export GLOBAL_BS="${GLOBAL_BS:-128}"
export MAX_LEN="${MAX_LEN:-131072}"
export IMG_MAX_TOK="${IMG_MAX_TOK:-832}"
export PACKING="${PACKING:-true}"
export PACKING_LENGTH="${PACKING_LENGTH:-131072}"
export MTP_LAYERS="${MTP_LAYERS:-1}"
export RECOMPUTE="${RECOMPUTE:-full}"
export SAVE_STEPS="${SAVE_STEPS:-100}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-0}"
export EVAL_STEPS="${EVAL_STEPS:-5}"
export EVAL_ITERS="${EVAL_ITERS:-2}"

exec bash "${PROJECT_ROOT}/scripts/46_train_genshin_llava_onevision_128k_multinode.slurm"
