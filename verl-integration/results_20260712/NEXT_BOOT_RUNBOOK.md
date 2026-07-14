# 下次开机 Runbook — 拿到干净 SFT→GRPO 曲线

机器:RTX PRO 6000 96GB(AutoDL 实例,SSH 主机/端口/密码见本地私密笔记,不入公开仓;若重开分配新端口则换)。
今天所有崩溃修复(guard/logprobs/extra_fields、USERSIM_BACKEND、DeepInfra pin)已在数据盘 `tau2_integration/` 保留,开机即在。

---

## 就绪度诚实评估

| 步 | 就绪度 | 说明 |
|---|---|---|
| A 可靠 usersim | 🟡 小改+开机验证 | 用 OpenRouter 多provider冗余(不是本地usersim,后者显存赌) |
| B 更大 val 集 | 🟢 一条命令 | tau2 airline ~50 任务,重切 val,CPU 跑 ~1min |
| C 更多步/lr | 🟢 改 env | 平凡 |

---

## 步 A — usersim 换前沿模型(核心可靠性修复,已实测)

**根因量化**(2026-07-12 本地实测 50并发×4波压测):
- `qwen-2.5-72b`(今天崩的)= 仅 2 provider → 突发下**成功率仅 62%**(75个400,Novita completions bug),p95=37s
- 全部前沿模型 = **100% 成功**(厂商自托管海量容量):

| 模型 | 压测成功率 | p50延迟 | ~$/轮(127M in/3M out) |
|---|---|---|---|
| **openai/gpt-4o**(默认,最好)| 100% | 1.61s | ~$350 |
| anthropic/claude-sonnet-4 | 100% | 1.73s | ~$430 |
| **google/gemini-2.5-flash**(省钱迭代档)| 100% | 1.12s | **~$46** |
| google/gemini-2.5-pro | 100% | 2.24s | ~$190 |

**选型(2026-07-12 全部本地压测 50并发×4波)**:
| 档 | 模型 | 成功率 | p50 | provider | $/轮 |
|---|---|---|---|---|---|
| ⭐**默认(最优)** | `meta-llama/llama-3.3-70b-instruct` | 100% | 1.93s | **10** | **$14** |
| 更强 usersim | `deepseek/deepseek-chat`(V3) | 100% | 2.43s | 3 | $28 |
| 甜点(前沿最快) | `google/gemini-2.5-flash` | 100% | 1.12s | 1(Google)| $46 |
| τ-bench 官方同款 | `openai/gpt-4o` | 100% | 1.61s | 2(Azure/OpenAI)| $350 |

- **llama-3.3-70b 是最优解**:100%、最便宜、最快、provider 冗余最强 → usersim 的活够用,可靠性彻底解决
- deepseek-chat:模型更聪明(V3),但贵 2×、provider 少(3),对"照剧本演顾客"不值
- gpt-4o:仅当要"τ-bench 官方 usersim"简历噱头时用
- `TAU2_USER_ALLOW_FALLBACKS=true` 即可,不需手配 provider

**开机验证**:val_before_train 后 `degrade` 应≈0(不再像 qwen 那样限流崩)。

## 步 B — 更大 val 集(降方差,核心)

现 test 仅 10 任务(±0.1=±1任务,今天 same-model baseline 0.4 vs 0.7 就是这噪声)。
tau2 airline 共 ~50 任务。重切:
```bash
cd /root/autodl-tmp/verl-work
# 用 data_prep_airline.py 重切成 train~30 / val~20(改脚本里的 split 比例或 task 列表)
/root/autodl-tmp/venv-verl/bin/python tau2_integration/data_prep_airline.py --val_size 20   # 若无此arg,手改split
# 校验:python -c "import pandas as pd; print(len(pd.read_parquet('data/tau2_airline/test.parquet')))"  # 应=20
```
val 20 任务把 ±0.1 噪声降到 ~±0.05,曲线能看出 <0.1 的真实提升。

## 步 C — 更多步 + lr

```
LR=4e-6            # 2e-6 今天证明太平; 4e-6 可; 也可扫 3e-6/5e-6
EPOCHS=40         # 更多步给曲线爬升空间(今天20步)
TEST_FREQ=5       # 每5步一个val点
```

---

## 一键启动(A改完 + B切完后)

```bash
cd /root/autodl-tmp/verl-work/tau2_integration
nohup env \
  GPU=0 POLICY_MODEL=/root/autodl-tmp/verl-work/models/qwen25-7b-sft-airline \
  USERSIM_BACKEND=openrouter TAU2_USER_LLM=openrouter/meta-llama/llama-3.3-70b-instruct \
  TAU2_USER_ALLOW_FALLBACKS=true \
  `# 默认 llama-3.3-70b($14,100%,最快,10 provider) —— 最优解` \
  `# 想更强 usersim: openrouter/deepseek/deepseek-chat($28,100%,3 provider)` \
  `# 想 τ-bench 官方同款(简历噱头): openrouter/openai/gpt-4o($350,100%)` \
  AGENT_WORKERS=2 TRAIN_BS=16 ROLLOUT_N=8 PPO_MINI=8 \
  GPU_UTIL=0.30 MAX_MODEL_LEN=10240 MAX_RESP=3072 MAX_PROMPT=6144 \
  LR=4e-6 EPOCHS=40 SAVE_FREQ=-1 TEST_FREQ=5 RESUME_MODE=disable PARAM_OFFLOAD=False \
  EXP_NAME=tau2_airline_7b_clean \
  bash run_tau2_grpo_7b.sh > /root/autodl-tmp/verl-work/tau2_airline_7b_clean.log 2>&1 &
```

## 预期与预算
- 每步 ~450s;val(20任务)~8min;40步 + 8个val ≈ **4-5h**,¥5.98/h ≈ ¥30
- 成功判据:val 曲线在更大 val 上稳定爬升(baseline→+0.1 以上)且 same-model 复测方差 <0.1
- 若 A 改完 degrade 仍高 → OpenRouter 该模型限流是硬约束,退路=本地 usersim(接受显存调参)或换更可用的 usersim 模型
```
