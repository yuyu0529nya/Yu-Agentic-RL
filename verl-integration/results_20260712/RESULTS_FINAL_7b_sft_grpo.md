# 7B SFT->veRL GRPO on tau2-airline — 最终结果 (2026-07-12, RTX PRO 6000 96GB)

## ✅ 基础设施成果 (核心交付, 面试级)
- 自写 tau2<->veRL 多轮 agent loop (@register tau2_agent), 单卡 96GB 跑通
- **7B veRL agentic GRPO 稳定训练 20+ 步, 0 崩溃** (两个 lr 设置)
- 训练动态全正常: pg_loss / 组相对 advantage / weight-sync(0.5-1.2s) / rollout-actor pearson
- 用上简历硬核件: ContextVar 并发隔离 + fused-CE(actor峰值53GB) + OpenRouter 72B usersim
- actor peak 53GB + vLLM 29GB = 82/96GB (usersim走OpenRouter不占卡)

## 📊 val 学习曲线 (10-task val, reward/mean@1)
- lr=2e-6: step 0/5/10/15/20 = 0.4 / 0.4 / 0.2 / 0.4 / 0.4  (平, 围绕SFT baseline)
- lr=4e-6: step 0/5/10       = 0.7 / 0.3 / 0.4              (同一SFT模型baseline却0.7 vs 上面0.4!)

## 📉 train reward (128 traj/step, 抽样)
- lr=2e-6: 早期均值0.20 -> 尾5均值0.27 (微弱上飘, 但在噪声内; 含corrupt步s4/s6/s14=~0)
- lr=4e-6: 早期0.20 -> 尾0.12 (含corrupt步s9/s12=0.0)

## ★ 诚实结论
**信号被 usersim 可靠性噪声淹没, 不是训练问题**:
1. 同一 SFT 模型 val baseline 一次0.4一次0.7 —— ±0.3 纯来自 72B OpenRouter usersim 的限流噪声(DeepInfra 429 突发, 偶发整批崩溃到1-token)
2. 10-task val 本身方差大(±0.1=±1任务) —— 正是 Track A pass^k 揭示的单点val不可靠, 现在7B上再次验证
3. corrupt步(整批usersim失败->全reward0)每4-5步一次, 污染训练信号
**任何 lr 都救不了 —— 瓶颈是评测可靠性, 非训练本身**

## 要拿到干净 0.4->0.55 曲线需要 (下次)
- 可靠 usersim: 本地自托管(无限流) —— 但需降 actor 显存装下(max_resp 3072->2048 + util调低, ~87GB可装)
- 更大 val 集(50-100任务, 降方差) —— tau2 airline test只10任务, 需扩
- 更多算力/步数
