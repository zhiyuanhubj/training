#!/bin/bash
# 把 genshin 3 节点训练 srun --overlap 挂到已有的占位 hold job 上（继承其 4 天时限，避免 12h 被砍）。
# 用法: HOLD_JID=16128 bash scripts/launch_gs_3node_overlap.sh
set -uo pipefail
export PROJECT_ROOT=/fsx/home/zhiyuan/game/extracted/training-main

HOLD_JID="${HOLD_JID:?need HOLD_JID}"

# ---------- 训练配置（3 节点: TP=4 PP=1 -> DP=6, global_bs=96 整除 DP=6, micro=2 -> grad_accum=8）----------
export MODEL_PATH="${PROJECT_ROOT}/models/Qwen3.5-9B"
export TRAIN_FILE="/fsx/sfr/zhiyuan/yuanshen_processed_filtered/_merged/train.jsonl"
export VAL_FILE="/fsx/sfr/zhiyuan/yuanshen_processed_filtered/_merged/val.jsonl"
export GAME_NAME="gs"

export GPUS="0,1,2,3,4,5,6,7"
export NPROC_PER_NODE=8
export TP=4
export PP=1
export MICRO_BS=2
export GLOBAL_BS=96
export RECOMPUTE=none
export MAX_LEN=40960
export IMG_MAX_TOK=1024

export EPOCHS=3
export LR=1e-5
export LR_DECAY_STYLE=constant
export SAVE_STEPS=100
export EVAL_STEPS=100

export WANDB_MODE=online
export WANDB_PROJECT=opensima
export WANDB_ENTITY=OpenSIMA
export RUN_NAME="${RUN_NAME:-bcft_gs_3n_dp6_$(date +%Y%m%d_%H%M%S)}"

# ---------- rendezvous（取 hold job 的首个节点做 master, 强制 IPv4）----------
nodes=( $(scontrol show hostnames "$(squeue -j "$HOLD_JID" -h -o %N)") )
head="${nodes[0]}"
MASTER_ADDR="$(getent ahostsv4 "$head" 2>/dev/null | awk '{print $1; exit}')"
if [[ -z "$MASTER_ADDR" || "$MASTER_ADDR" == fe80* ]]; then
    MASTER_ADDR="$(echo "$head" | sed -E 's/^ip-([0-9]+)-([0-9]+)-([0-9]+)-([0-9]+).*/\1.\2.\3.\4/')"
fi
export MASTER_ADDR
export MASTER_PORT="$(( 20000 + HOLD_JID % 20000 ))"

echo "============================================================"
echo " GS 3-NODE TRAIN onto hold job $HOLD_JID"
echo "   nodes=${nodes[*]}"
echo "   master=$MASTER_ADDR:$MASTER_PORT  TP=$TP PP=$PP DP=$(( ${#nodes[@]} * NPROC_PER_NODE / (TP*PP) ))"
echo "   global_bs=$GLOBAL_BS micro=$MICRO_BS recompute=$RECOMPUTE max_len=$MAX_LEN"
echo "   run=$RUN_NAME  data=$TRAIN_FILE"
echo "============================================================"

# 每节点 1 task; --cpus-per-task=4 = hold job 每 task 预算(否则 step creation disabled)。
srun --overlap --jobid="$HOLD_JID" --nodes=3 --ntasks-per-node=1 \
     --cpus-per-task=4 --gres=gpu:8 \
     bash "${PROJECT_ROOT}/scripts/43_train_inner.sh"
