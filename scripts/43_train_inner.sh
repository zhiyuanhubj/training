#!/bin/bash
# 由 43_train_genshin_multinode.slurm 的 srun 在【每个节点】上调用。
# 职责: 激活 conda env + 配置跨节点 NCCL/EFA + 设置本节点分布式 rank, 再调用 41 训练脚本。
# 训练配置(MODEL_PATH/TP/GLOBAL_BS/MASTER_ADDR/...)由外层 sbatch export, srun(--export=ALL 默认)透传。
set -uo pipefail

CONDA_ENV_NAME="${CONDA_ENV_NAME:-megatron-sft}"
if [[ -z "${CONDA_BASE:-}" ]]; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
fi
[[ -n "$CONDA_BASE" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]] || {
    echo "[ERR] set CONDA_BASE to the Miniconda/Anaconda installation root"
    exit 1
}
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

# ---------- 跨节点 NCCL / EFA (p5en 有 EFA + aws-ofi-nccl 插件) ----------
# 把 EFA libfabric + ofi-nccl 网络插件加进 LD_LIBRARY_PATH, 让 NCCL 走 EFA RDMA。
# 若插件加载失败 NCCL 会回退到 socket(走下面探测到的网卡), 慢但不影响正确性。
if [[ -d /opt/amazon/efa && -d /opt/amazon/ofi-nccl ]]; then
    export LD_LIBRARY_PATH="/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH:-}"
    export FI_PROVIDER="${FI_PROVIDER:-efa}"
    export FI_EFA_USE_DEVICE_RDMA="${FI_EFA_USE_DEVICE_RDMA:-1}"
fi
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# bootstrap/GLOO 用的主网卡(取默认路由网卡)
IFACE="$(awk '$2=="00000000"{print $1; exit}' /proc/net/route 2>/dev/null)"
if [[ -z "$IFACE" ]]; then
    for n in /sys/class/net/*; do [[ -e "$n/device" ]] && { IFACE="$(basename "$n")"; break; }; done
fi
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$IFACE}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$IFACE}"

# ---------- 分布式 rank ----------
# Explicit values allow one coordinated run to span several existing one-node
# allocations. A normal multi-node sbatch still falls back to Slurm's values.
export NNODES="${NNODES:-${SLURM_NNODES}}"
export NODE_RANK="${NODE_RANK:-${SLURM_NODEID}}"

echo "[inner] host=$(hostname) NODE_RANK=${NODE_RANK}/${NNODES} MASTER=${MASTER_ADDR}:${MASTER_PORT} iface=${NCCL_SOCKET_IFNAME} env=${CONDA_DEFAULT_ENV}"

exec bash "${PROJECT_ROOT}/scripts/41_train_bc_full_megatron.sh"
