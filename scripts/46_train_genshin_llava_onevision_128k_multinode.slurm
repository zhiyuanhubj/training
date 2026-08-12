#!/bin/bash
# Genshin 3.0 passes + LLaVA-OneVision 0.162 passes, 128K full-parameter SFT.
# Both sources are node-local Parquet under the generated canonical union root.
#
# Production:
#   NFS_DIR=/path/to/shared/cache \
#   LOCAL_DATA_ROOT=/opt/dlami/nvme/$USER/genshin-training \
#   sbatch scripts/46_train_genshin_llava_onevision_128k_multinode.slurm
#
# Every allocated node must have the same LOCAL_DATA_ROOT contents and path.
#SBATCH --account=xgen-mm
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=8
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=4-00:00:00
#SBATCH --job-name=genshin-llava-128k
#SBATCH --output=logs/genshin_llava_128k_%j.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
: "${NFS_DIR:?Set NFS_DIR to the shared cache directory}"
export NFS_DIR

export LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/opt/dlami/nvme/${USER}/genshin-training}"
export MIXTURE_YAML="${MIXTURE_YAML:-${LOCAL_DATA_ROOT}/mixture-compact/genshin-llava-128k.local.yaml}"
[[ -f "$MIXTURE_YAML" ]] || {
    echo "[ERR] node-local mixture config not found: $MIXTURE_YAML"
    echo "[ERR] stage and validate both datasets on every allocated node before launching"
    exit 1
}
export MODEL_PATH="${MODEL_PATH:-${NFS_DIR}/models/Qwen3.5-9B}"
export GAME_NAME="${GAME_NAME:-genshin3_llava0162_qwen35_128k}"
export RESUME_FROM="${RESUME_FROM:-}"
export AUTO_RESUME="${AUTO_RESUME:-false}"
_OUTPUT_DIR_EXPLICIT="${OUTPUT_DIR+x}"
if [[ -n "$RESUME_FROM" && "$RESUME_FROM" != "auto" && -z "$_OUTPUT_DIR_EXPLICIT" ]]; then
    export OUTPUT_DIR="$(dirname "$(readlink -f "$RESUME_FROM")")"
    export RUN_NAME="${RUN_NAME:-$(basename "$OUTPUT_DIR")}"
else
    export RUN_NAME="${RUN_NAME:-${GAME_NAME}_tp4_full_$(date +%Y%m%d_%H%M%S)_j${SLURM_JOB_ID}}"
    export OUTPUT_DIR="${OUTPUT_DIR:-${NFS_DIR}/outputs/${RUN_NAME}}"
fi
if [[ "$AUTO_RESUME" == "true" || "$RESUME_FROM" == "auto" ]]; then
    [[ -n "$_OUTPUT_DIR_EXPLICIT" ]] || {
        echo "[ERR] AUTO_RESUME requires a stable, explicitly supplied OUTPUT_DIR"
        exit 1
    }
fi

# ----------------------------- data loading -----------------------------
export STREAMING="${STREAMING:-true}"
export STREAMING_SHARD_BY_DP="${STREAMING_SHARD_BY_DP:-true}"
export DATASET_SHUFFLE="${DATASET_SHUFFLE:-true}"
export TRAIN_DATALOADER_SHUFFLE="${TRAIN_DATALOADER_SHUFFLE:-true}"
export SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-64}"
export STOPPING_STRATEGY="${STOPPING_STRATEGY:-all_exhausted_without_replacement}"
export STOP_AT_DATASET_END="${STOP_AT_DATASET_END:-true}"
export PASS_AWARE_MIXTURE="${PASS_AWARE_MIXTURE:-false}"
export CHANNEL_COLUMNS_JSON="${CHANNEL_COLUMNS_JSON:-{\"conversations\":\"messages\",\"game\":\"channel\",\"data_source\":\"channel\"}}"
export ENABLE_CHANNEL_LOSS="${ENABLE_CHANNEL_LOSS:-true}"
export ADD_GENSHIN_SPECIAL_TOKENS="${ADD_GENSHIN_SPECIAL_TOKENS:-true}"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHON_BIN
export INTERLEAVE_PROB="${INTERLEAVE_PROB:-$("${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/lib_mixture.py" "${MIXTURE_YAML}" --print-interleave-prob)}"

# -------------------------- GPU / parallelism ---------------------------
export GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export TP="${TP:-4}"
export PP="${PP:-1}"
export SEQ_PARALLEL="${SEQ_PARALLEL:-true}"

# 8 nodes x 8 GPUs, TP=4, PP=1 -> DP=16. GBS=128 gives grad-accum=8.
# Qwen3.5-9B has four KV heads; TP=8 fails Megatron's query-group divisibility check.
export MICRO_BS="${MICRO_BS:-1}"
export GLOBAL_BS="${GLOBAL_BS:-128}"

# ----------------------- context / multimodality ------------------------
export MAX_LEN="${MAX_LEN:-131072}"
export IMG_MAX_TOK="${IMG_MAX_TOK:-832}"
export PACKING="${PACKING:-true}"
export PACKING_LENGTH="${PACKING_LENGTH:-131072}"
export PACKING_NUM_PROC="${PACKING_NUM_PROC:-4}"
export PACKING_INTERVAL="${PACKING_INTERVAL:-8}"
export PACKING_STRATEGY="${PACKING_STRATEGY:-binpack}"
export DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
export SWIFT_USE_MCORE_GDN="${SWIFT_USE_MCORE_GDN:-1}"
export DETERMINISTIC_MODE="${DETERMINISTIC_MODE:-false}"
export SEED="${SEED:-42}"

# ------------------------ schedule / optimizer --------------------------
# Mixture passes already define total exposure, so the trainer runs one epoch.
export EPOCHS="${EPOCHS:-1}"
export TRAIN_ITERS="${TRAIN_ITERS:-0}"
export LR="${LR:-1e-5}"
export MIN_LR="${MIN_LR:-1e-6}"
export VIT_LR="${VIT_LR:-3e-6}"
export ALIGNER_LR="${ALIGNER_LR:-1e-5}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
export WARMUP_ITERS="${WARMUP_ITERS:-100}"
export LR_DECAY_STYLE="${LR_DECAY_STYLE:-constant}"

# ---------------------- memory / fused kernels --------------------------
export RECOMPUTE="${RECOMPUTE:-full}"
export RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-1}"
export CROSS_ENTROPY_LOSS_FUSION="${CROSS_ENTROPY_LOSS_FUSION:-true}"
export CROSS_ENTROPY_FUSION_IMPL="${CROSS_ENTROPY_FUSION_IMPL:-te}"
export DISABLE_TORCH_COMPILE="${DISABLE_TORCH_COMPILE:-true}"
# MTP checkpoint TP-replica metadata is corrected by
# scripts/mcore_mtp_checkpoint_patch.py and covered by crash/resume validation.
export MTP_LAYERS="${MTP_LAYERS:-1}"
export MTP_LOSS_SCALE="${MTP_LOSS_SCALE:-0.1}"

# ----------------------- evaluation / checkpoint ------------------------
export EVAL_ITERS="${EVAL_ITERS:-2}"
export SAVE_STEPS="${SAVE_STEPS:-100}"
export EVAL_STEPS="${EVAL_STEPS:-5}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-0}"
export LOGGING_STEPS="${LOGGING_STEPS:-1}"
export SAVE_SAFETENSORS="${SAVE_SAFETENSORS:-false}"
export ADD_VERSION="${ADD_VERSION:-false}"
export NO_SAVE_RNG="${NO_SAVE_RNG:-false}"
export NO_SAVE_OPTIM="${NO_SAVE_OPTIM:-false}"
export NO_LOAD_OPTIM="${NO_LOAD_OPTIM:-false}"
export NO_LOAD_RNG="${NO_LOAD_RNG:-false}"

# Iterable datasets have no __len__. With packing, raw rows / GBS is only a
# generous safety ceiling; STOP_AT_DATASET_END ends the run at the exact final
# complete packed optimizer step without cycling the mixture.
if [[ "$STREAMING" == "true" && "$TRAIN_ITERS" -le 0 ]]; then
    _MIXTURE_TOTAL_SAMPLES="$(
        "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/lib_mixture.py" \
            "${MIXTURE_YAML}" --print-total-samples
    )"
    export TRAIN_ITERS="$(( _MIXTURE_TOTAL_SAMPLES / GLOBAL_BS ))"
    echo "[mixture] streaming train_iters=$TRAIN_ITERS from $_MIXTURE_TOTAL_SAMPLES samples / GBS=$GLOBAL_BS"
fi

# -------------------------------- WandB ---------------------------------
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-opensima}"
export WANDB_ENTITY="${WANDB_ENTITY:-OpenSIMA}"

# This file is the complete experiment configuration. 43 only provides the
# Slurm rendezvous/srun layer; 41 converts these values into Megatron flags.
if [[ "${DIRECT_INNER:-false}" == "true" ]]; then
    export DATASET_SPEC="$(
        "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/lib_mixture.py" \
            "${MIXTURE_YAML}" --print-spec
    )"
    _MIX_VAL="$(
        "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/lib_mixture.py" \
            "${MIXTURE_YAML}" --print-val
    )"
    [[ -z "$_MIX_VAL" ]] || export VAL_FILE="$_MIX_VAL"
    export ENABLE_CHANNEL_LOSS=true
    exec bash "${PROJECT_ROOT}/scripts/43_train_inner.sh"
else
    exec bash "${PROJECT_ROOT}/scripts/43_train_genshin_multinode.slurm"
fi
