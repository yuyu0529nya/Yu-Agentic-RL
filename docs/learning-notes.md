# 学习笔记: Agentic RL 第一阶段

## 我们现在到底在做什么

我们要做的是长链路工具智能体训练系统。最终目标是让一个模型在航空客服任务里学会多轮对话、查工具、改数据库，并在任务完成率上变好。

但是训练之前必须先有评测。没有稳定评测，SFT、DPO、GRPO 都是在黑箱里乱试。

## tau-bench 的任务结构

一个任务通常包含：

- 用户场景：用户是谁、想干什么、知道什么、不知道什么。
- domain policy：客服 agent 必须遵守的业务规则。
- tools：agent 可以调用的 API，比如查用户、查预订、改签、取消。
- evaluation criteria：如何给最终轨迹打分。

重要点：`actions` 是参考轨迹，不一定是唯一正确解。airline/retail/telecom 的官方 reward 主要由 `DB` 和 `COMMUNICATE` 组成。

## 为什么先跑 baseline

baseline 有三个作用：

1. 给后续 SFT/GRPO 一个对照组。
2. 产出真实 trajectory，帮我们理解失败类型。
3. 检查评测脚本、API、并发、保存格式是否可靠。

如果 baseline 都不稳，训练后提升多少没有可信度。

## 第一周目标

1. 跑 `inspect_tau2_tasks.py`，理解 airline 任务数量、split、reward_basis。
2. 用 5 个任务跑一次 smoke baseline。
3. 看 `results.json` 中的 messages、tool_calls、reward_info。
4. 总结 3 个失败样例：是工具没调、业务规则错、用户模拟异常，还是沟通信息没说。

完成这些之后，我们再进入 SFT 数据抽取。
