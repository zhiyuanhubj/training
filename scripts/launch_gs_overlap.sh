#!/bin/bash
# 通用: 把 genshin 训练 srun --overlap 挂到已有占位 hold job 上(继承其 4 天时限)。
# 用法: HOLD_JID=16061 NUM_NODES=8 GLOBAL_BS=128 RUN_NAME=bcft_gs_8n_dp16_xxx \
#        setsid nohup bash scripts/launch_gs_overlap.sh > logs/launch_xxx.log 2>&1 &
# DP = NUM_NODES*8/(TP*PP); GLOBAL_BS 必须能被 DP 整除。每 RUN_NAME 一个独立 output 目录,不会覆盖。
set -uo pipefail
export PROJECT_ROOT=/fsx/home/zhiyuan/game/extracted/training-main

HOLD_JID="${HOLD_JID:?need HOLD_JID}"
NUM_NODES="${NUM_NODES:-3}"

# ---------- 训练配置 ----------
export MODEL_PATH="${PROJECT_ROOT}/models/Qwen3.5-9B"
export TRAIN_FILE="/fsx/sfr/zhiyuan/yuanshen_processed_filtered/_merged/train.jsonl"
export VAL_FILE="/fsx/sfr/zhiyuan/yuanshen_processed_filtered/_merged/val.jsonl"
export GAME_NAME="gs"

export GPUS="0,1,2,3,4,5,6,7"
export NPROC_PER_NODE=8
export TP=4
export PP=1
export MICRO_BS="${MICRO_BS:-2}"
export GLOBAL_BS="${GLOBAL_BS:-96}"     # 3n→96(DP6), 8n→128(DP16)
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
export RUN_NAME="${RUN_NAME:-bcft_gs_${NUM_NODES}n_$(date +%Y%m%d_%H%M%S)}"

# ---------- rendezvous (hold job 首节点做 master, 强制 IPv4) ----------
nodes=( $(scontrol show hostnames "$(squeue -j "$HOLD_JID" -h -o %N)") )
head="${nodes[0]}"
MASTER_ADDR="$(getent ahostsv4 "$head" 2>/dev/null | awk '{print $1; exit}')"
if [[ -z "$MASTER_ADDR" || "$MASTER_ADDR" == fe80* ]]; then
    MASTER_ADDR="$(echo "$head" | sed -E 's/^ip-([0-9]+)-([0-9]+)-([0-9]+)-([0-9]+).*/\1.\2.\3.\4/')"
fi
export MASTER_ADDR
export MASTER_PORT="$(( 20000 + HOLD_JID % 20000 ))"

DP=$(( NUM_NODES * NPROC_PER_NODE / (TP*PP) ))
echo "============================================================"
echo " GS ${NUM_NODES}-NODE TRAIN onto hold job $HOLD_JID"
echo "   nodes=${nodes[*]}"
echo "   master=$MASTER_ADDR:$MASTER_PORT  TP=$TP PP=$PP DP=$DP"
echo "   global_bs=$GLOBAL_BS micro=$MICRO_BS grad_accum=$(( GLOBAL_BS / (MICRO_BS*DP) )) recompute=$RECOMPUTE"
echo "   run=$RUN_NAME  output=outputs/$RUN_NAME"
echo "============================================================"
if (( GLOBAL_BS % DP != 0 )); then echo "[FATAL] GLOBAL_BS=$GLOBAL_BS 不能被 DP=$DP 整除"; exit 1; fi

srun --overlap --jobid="$HOLD_JID" --nodes="$NUM_NODES" --ntasks-per-node=1 \
     --cpus-per-task=4 --gres=gpu:8 \
     bash "${PROJECT_ROOT}/scripts/43_train_inner.sh"
