# Roadmap

## Phase 0: Evaluation Foundation

目标：建立可复现 baseline。

- 接入官方 tau2-bench。
- 跑 airline smoke baseline。
- 汇总 `avg_reward`、`pass^k`、失败任务。
- 分析 trajectory 和 reward breakdown。

## Phase 1: SFT From Successful Trajectories

目标：让 3B/7B 模型学会多轮工具调用格式。

- 从成功 trajectory 导出 assistant-only 训练样本。
- 处理 Qwen chat template 和 tool call 格式。
- 实现多轮 loss mask，只训练 assistant 决策 token。
- 用 QLoRA 训练一个小版本。

## Phase 2: Preference Training

目标：比较成功轨迹和失败轨迹。

- 构造 chosen / rejected pair。
- 尝试 DPO 或 ORPO。
- 分析偏好学习是否提升未见任务。

## Phase 3: GRPO For Agentic RL

目标：让模型通过环境 reward 学策略。

- 每个 task 采样多条 trajectory。
- 用 final reward、格式 reward、工具合法性 reward、长度惩罚组成 rule reward。
- 先小规模实现，再迁移到 veRL。
- 重点诊断 reward 饱和、训练泄漏、长链路退化。

## Phase 4: Systems Upgrade

目标：接近高阶简历项目形态。

- vLLM rollout。
- 异步采样和训练解耦。
- ContextVar 隔离多环境并发状态。
- FSDP / 显存峰值诊断。
