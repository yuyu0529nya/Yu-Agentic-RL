# Week 01: Baseline And Evaluation

这周只做一件事：把 tau2-bench airline 的评测链路跑稳。

## 为什么先做评测

后训练项目最怕一开始就训练。因为如果 baseline、任务切分、指标和失败归因都没搭好，
后面的 SFT / DPO / GRPO 分数涨了也很难解释：可能是真提升，也可能是数据泄漏、采样
不公平、reward 写错，或者用户模拟器变了。

所以第一周目标是：

- 会运行官方 `tau2 run`。
- 知道结果文件里有什么。
- 能算 `avg_reward` 和 `pass^k`。
- 能打开失败 trajectory，解释为什么失败。

## 核心概念

### trajectory

一次完整任务交互就是一条 trajectory。它通常包含：

- user simulator 发出的用户消息。
- agent 的回复或工具调用。
- environment/tool 的返回结果。
- 最终 termination reason。
- evaluator 给出的 reward。

后面 SFT 会把成功 trajectory 转成训练样本，GRPO 会把多条采样 trajectory 的 reward
拿来更新 policy。

### reward

airline / retail / telecom 的官方 reward 通常由多个部分相乘：

- 数据库最终状态是否正确。
- 是否向用户传达了必须说明的信息。
- 某些任务可能还包含动作、环境断言或自然语言断言。

乘法意味着一个关键部分为 0，最终 reward 就是 0。这对 RL 很重要，因为稀疏 reward 会让
长链路任务非常难学。

### pass^k

tau-bench 的 `pass^k` 不是简单“采 k 次只要一次成功”。官方实现使用：

```text
C(success_count, k) / C(num_trials, k)
```

也就是从同一任务的多次 trial 中抽 k 条都成功的概率。第一阶段我们先跟官方口径保持一致。

## 本周任务

1. 跑通 3 个 task 的 smoke baseline。
2. 跑通 10 个 task、1 trial 的 baseline。
3. 跑通 10 个 task、4 trial 的 baseline，用来观察 `pass^1` 到 `pass^4`。
4. 手动读 5 条失败 trajectory，写出失败原因。

## 推荐命令

```powershell
.\scripts\run_tau2_baseline.ps1 `
  -Domain airline `
  -NumTasks 3 `
  -NumTrials 1 `
  -AgentLlm openai/gpt-4.1-mini `
  -UserLlm openai/gpt-4.1-mini
```

如果模型调用成本可接受，再扩大：

```powershell
.\scripts\run_tau2_baseline.ps1 `
  -Domain airline `
  -NumTasks 10 `
  -NumTrials 4 `
  -MaxConcurrency 2 `
  -AgentLlm openai/gpt-4.1-mini `
  -UserLlm openai/gpt-4.1-mini
```

## 第一周产物

- `data/simulations/<run_name>/results.json`
- summary 输出截图或复制文本
- 失败样例笔记
- 对 airline 工具和 policy 的初步理解

下一步才是构造 SFT 数据。
