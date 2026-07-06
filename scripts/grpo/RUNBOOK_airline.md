# Airline 4-way ablation — 自跑手册 (2×5090)

## 0. 一次性设好环境变量 (每开新终端跑一次)
```bash
export WORKDIR=$HOME/agentic-rl
export PYBIN=$WORKDIR/venv/bin/python EXTRA_PATH=$WORKDIR/venv/bin
export HF_HOME=$WORKDIR/hf-cache TRITON_CACHE_DIR=$WORKDIR/.triton
export POLICY_MODEL=$WORKDIR/models/qwen25-7b-instruct TAU2_CMD=tau2
export USER_GPU=0 POLICY_GPU=1 MAXC=8
cd $WORKDIR
```

## 1. 开跑前: 确认两卡没被别人占 (共享机上只应看到少量他人桌面进程 ~500M)
```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | \
  while read l; do p=$(echo $l|cut -d, -f1|tr -d " "); echo "$l  $(ps -o user= -p $p)"; done
```
若他人占着大显存 -> 等其跑完再开 (否则会被挤崩)。

## 2. 先 smoke (~13 分钟, 验证链路通)
```bash
RUN=abl4_smoke TRAIN_TASK_IDS=2,3 EVAL_TASK_IDS=0,1 N=2 ITERS=1 EVAL_TRIALS=1 \
  bash scripts/grpo/run_grpo_ablation_4way.sh
```
终端会实时打印进度; 看到最后 `ABLATION_ANALYSIS_DONE` = 通过。

## 3. 全量 (~4-4.5h). 用 tmux 防断线 + 能随时回看
```bash
tmux new -s abl            # 进 tmux
# (在 tmux 里, 环境变量要重设一遍 -> 重跑第0步)
RUN=abl_4way_dual bash scripts/grpo/run_grpo_ablation_4way.sh
# 看着进度; 要离开按 Ctrl-b 然后 d (detach); 回来: tmux attach -t abl
```

## 4. (可选) 另开一个终端只看干净进度
```bash
tail -f $WORKDIR/outputs/abl_4way_dual/master.log | \
  grep --line-buffered -E '\[abl\]|\[grpo\]|Status:|METHOD|eval:|DONE'
```

## 5. 跑完看结果 (脚本末尾已自动出配对分析; 也可手动复算)
```bash
$PYBIN scripts/grpo/tau2_eval_analyze.py outputs/abl_4way_dual/base_eval.jsonl \
  vanilla=outputs/abl_4way_dual/vanilla_eval_final.jsonl \
  prmlata=outputs/abl_4way_dual/prmlata_eval_final.jsonl \
  prmonly=outputs/abl_4way_dual/prmonly_eval_final.jsonl \
  lataonly=outputs/abl_4way_dual/lataonly_eval_final.jsonl
```

## 6. 跑完释放显存 (脚本正常退出会自动释放; 这是兜底)
```bash
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  [ "$(ps -o user= -p $p)" = "$USER" ] && { echo kill $p; kill -9 $p; }
done   # 只杀自己的, 不碰他人进程
```

## 单卡 fallback (若只有一张卡空, 比如 GPU1 被占, 用 GPU0)
把第3步换成:
```bash
RUN=abl4_1gpu GPU=0 USER_BACKEND=selfhost AUTO_SHUTDOWN=0 \
  bash scripts/grpo/run_grpo_ablation_1gpu_4way.sh
```
(其余 export 不变; 单卡用 base+LoRA 多路复用本地 usersim, ~6.5h)

## 硬化说明 (这版脚本已带)
- usersim/policy 两端点每次 collect 前健康检查, 死了自动重启
- 0-rollout 自动重试一次; 真起不来会大声 abort (不再静默白跑)
- vLLM 日志追加(不覆盖) -> 可回溯
