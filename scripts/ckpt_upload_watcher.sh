#!/usr/bin/env bash
# 训练 ckpt -> HF 上传 + 传完删本地 watcher。
# 监测 RUN 输出目录下完成的 checkpoint-N(含 safetensors 权重 + 可重启状态),
# 上传到 HF(用户 WeihaoTan), 校验远端齐全后删本地 ckpt 省空间。
#
# 用法: RUN_DIR=<outputs/RUN_NAME> HF_REPO=WeihaoTan/xxx bash ckpt_upload_watcher.sh
# 环境:
#   RUN_DIR      训练输出目录(其下有 v0-*/checkpoint-*)  [必填]
#   HF_REPO      目标 HF 仓库(model)                      [必填]
#   TOKEN_FILE   HF 写 token 文件(默认 ~/.hf_write_token)
#   POLL_SEC     轮询间隔(默认 300=5分钟)
#   SETTLE_SEC   ckpt 写完后静置多久才算稳定(默认 600)
#   KEEP_LATEST  保留最近 N 个不删(默认 1, 防止删掉正用于续训的)
set -uo pipefail
export PATH=/fsx/home/zhiyuan/miniconda3/bin:$PATH
RUN_DIR="${RUN_DIR:?need RUN_DIR}"
HF_REPO="${HF_REPO:?need HF_REPO}"
TOKEN_FILE="${TOKEN_FILE:-/fsx/home/zhiyuan/.hf_write_token}"
POLL_SEC="${POLL_SEC:-300}"
SETTLE_SEC="${SETTLE_SEC:-600}"
KEEP_LATEST="${KEEP_LATEST:-1}"
export HF_TOKEN="$(cat "$TOKEN_FILE")"
export HF_HUB_ENABLE_HF_TRANSFER=1
STATE="$RUN_DIR/.ckpt_upload_state"; mkdir -p "$STATE"
LOG="$RUN_DIR/ckpt_upload_watcher.log"

log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "watcher start RUN_DIR=$RUN_DIR HF_REPO=$HF_REPO poll=${POLL_SEC}s settle=${SETTLE_SEC}s keep=$KEEP_LATEST"

# 确保仓库存在(私有)
hf repo create "$HF_REPO" --repo-type model --private -y >/dev/null 2>&1 || true

ckpt_complete(){  # $1=ckpt dir; 完成判据: 有 index + 所有分片 + config, 且 SETTLE_SEC 内无改动
    local d="$1"
    local idx="$d/model.safetensors.index.json"
    [ -f "$idx" ] || return 1
    [ -f "$d/config.json" ] || return 1
    # 所有分片在场
    local shards; shards=$(python3 - "$idx" <<'PY' 2>/dev/null
import json,sys,os
d=os.path.dirname(sys.argv[1]); idx=json.load(open(sys.argv[1]))
files=set(idx.get("weight_map",{}).values())
print("OK" if files and all(os.path.exists(os.path.join(d,f)) for f in files) else "NO")
PY
)
    [ "$shards" = "OK" ] || return 1
    # 静置: 最近修改时间距今 > SETTLE_SEC
    local newest; newest=$(find "$d" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1)
    local now; now=$(date +%s)
    [ -n "$newest" ] && [ $((now-newest)) -ge "$SETTLE_SEC" ]
}

upload_one(){  # $1=ckpt dir, $2=name(checkpoint-N)
    local d="$1" name="$2"
    log "upload start: $name ($(du -sh "$d" 2>/dev/null|cut -f1))"
    local ok=0
    for attempt in 1 2 3 4 5; do
        if hf upload "$HF_REPO" "$d" "$name" --repo-type model >> "$LOG" 2>&1; then ok=1; break; fi
        log "  upload attempt $attempt failed; retry in 60s"; sleep 60
    done
    [ "$ok" = 1 ] || { log "upload FAILED after retries: $name"; return 1; }
    # 校验远端 safetensors 分片数 >= 本地(用 huggingface_hub.list_repo_files)
    local nloc nrem
    nloc=$(find "$d" -name "*.safetensors" | wc -l)
    nrem=$(python3 - "$HF_REPO" "$name" <<'PY' 2>/dev/null
from huggingface_hub import list_repo_files
import os,sys
repo,name=sys.argv[1],sys.argv[2]
try:
    fs=list_repo_files(repo, repo_type="model", token=os.environ["HF_TOKEN"])
    print(sum(1 for f in fs if f.startswith(name+"/") and f.endswith(".safetensors")))
except Exception:
    print(0)
PY
)
    log "  verify: local_shards=$nloc remote_safetensors=$nrem"
    if [ "$nrem" -ge "$nloc" ] && [ "$nloc" -gt 0 ]; then
        log "upload verified: $name"; return 0
    fi
    log "verify inconclusive (nrem=$nrem nloc=$nloc) — 不删本地以防丢失: $name"; return 1
}

while true; do
    # 找所有 checkpoint-*,按 step 排序
    mapfile -t cks < <(find "$RUN_DIR" -maxdepth 2 -type d -name "checkpoint-*" 2>/dev/null | sort -V)
    ntot=${#cks[@]}
    if [ "$ntot" -gt 0 ]; then
        keepfrom=$((ntot-KEEP_LATEST))
        for i in "${!cks[@]}"; do
            d="${cks[$i]}"; name=$(basename "$d")
            done_mark="$STATE/${name}.uploaded"
            del_mark="$STATE/${name}.deleted"
            [ -f "$del_mark" ] && continue
            # 上传(若未传且已完成)
            if [ ! -f "$done_mark" ]; then
                if ckpt_complete "$d"; then
                    if upload_one "$d" "$name"; then touch "$done_mark"; fi
                fi
            fi
            # 删本地(已传 且 不在保留窗口内)
            if [ -f "$done_mark" ] && [ "$i" -lt "$keepfrom" ]; then
                log "delete local (uploaded & not in keep window): $name"
                rm -rf "$d" && touch "$del_mark"
            fi
        done
    fi
    # 训练结束(无 megatron 进程 且 已无未处理)则可退出; 这里简单持续轮询
    sleep "$POLL_SEC"
done
