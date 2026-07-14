# 手写版 tau2-airline 定版基准(从公司机实际 eval jsonl 实测,2026-07-13)

来源:公司内网机的 `outputs/nail_down_0705/*_eval.jsonl`(每个 200 条;主机/路径见本地私密笔记,不入公开仓)

## 真实数字(mean reward over 200 sampled trajectories)

| checkpoint | mean(avg,=手写版"pass^1") | pass^10(全对) | best@10(≥1次) |
|---|---|---|---|
| base | 0.200 | 0.000 | 0.500 |
| SFT (e9_sft) | **0.405** | 0.050 | 0.600 |
| GRPO (mega_vanilla) | **0.545** | 0.400 | 0.600 |
| GRPO+PRM+LATA (line1_prmlata) | **0.550** | 0.450 | 0.600 |

**"0.20→0.55" = base → 最佳 GRPO,全是 200 条采样的平均成功率。SFT 单独到 0.405,GRPO 净 +0.14。**

## 精确 eval 协议(nail_down_0705.sh)
- **20 held-out 任务(定死)**: `0,1,5,6,10,11,15,16,20,21,25,26,30,31,35,36,40,41,45,46`(train=其余30个)
- **EVAL_TRIALS=10**(每任务10次)= 200 条
- **agent/policy temp = 0.5**;**usersim = 本地 qwen25-7b-instruct,USER_TEMP=0(确定性)**
- max_steps=40,agent_max_tokens=768,seed=900,max_model_len=32768
- 报的指标 = 200 条 mean reward(avg)

## 我们 veRL 复现的对齐清单
- ✅ 可对齐:20 个 task_id、trials=10、agent temp=0.5、avg 口径
- ⚠️ 难对齐:**usersim** —— 手写版用本地7B;veRL 训练时单卡装不下本地usersim(actor峰值54G+vLLM29G+usersim20G>96G),必须 OpenRouter(llama-70b)。这是绝对数字差异的主因
- ⚠️ 其他小差异:agent_max_tokens 768(我们 3072)

## 我们目前的数(供对比,口径尚未完全对齐)
- greedy pass@1(旧,错口径): 0.10
- 采样 avg@8(shuffle-42 val, llama usersim, temp1.0): **0.25**  ← baseline
- (对齐版待跑:exact-20任务 + temp0.5 + n=10)
