# Agentic RL Portfolio Packaging

Date: 2026-07-06

This document is the job-facing packaging master copy for the three public projects:

- tau2-agentic-rl: long-horizon tool-calling agent GRPO post-training
- search-agent-rlvr: verifiable-reward RL for multi-turn retrieval agents
- YuResearchAgent: citation-aware Deep Research multi-agent system

The positioning principle is simple: lead with outcomes, systems, evidence, and scale. Do not lead
with limitations, defensive comparisons, or implementation caveats.

## One-Line Portfolio Positioning

Chinese:

构建了一组 Agentic RL 项目，覆盖可验证奖励 RLVR、多轮工具调用 Agent 的 GRPO 后训练，以及引用增强的 Deep Research 多智能体系统；项目包含统计显著性验证、多随机种子复核和原始评测证据归档。

English:

Built a portfolio of agentic RL systems covering verifiable-reward RLVR, multi-turn tool-calling
GRPO post-training, and a citation-aware Deep Research multi-agent system, with statistically
validated gains, multi-seed confirmation, and archived raw evaluation artifacts.

## Resume Version

Use the dense version below when the resume has enough space or when applying to LLM Agent /
post-training / RLVR roles. It is intentionally more technical than the GitHub README first page.

### Agentic-GRPO-LongHorizon: 长链路多工具智能体的 GRPO 后训练系统

2026.06 - 2026.07 | 独立开发者 | 个人项目 | GitHub

- 面向 tau2-bench airline 长程客服任务，搭建 Baseline -> Teacher-SFT -> GRPO 三阶段 Agent RL pipeline，覆盖 policy server、self-hosted user simulator、tau2 tool execution、on-policy rollout collection、reward scoring、QLoRA adapter update 与 paired evaluation；held-out pass^1 从 Base 0.20 -> Teacher-SFT 0.41 -> GRPO 0.55，单次峰值 0.56。

- 自实现适配多轮工具调用 Agent 的 GRPO trainer：按 task 分组 rollout，计算 group-relative advantage `(r - mean) / (std + eps)`，加入 outcome-variance advantage gating 过滤无对比样本，支持 length-aware advantage normalization、dense process reward、advantage clipping 与 assistant-only token NLL。

- 设计 teacher-trajectory distillation warm start，将成功教师轨迹蒸馏为高技能初始策略，再接 on-policy GRPO；教师采集在 30 个 RL-train tasks 上达到 137/180 = 76% success，最终构造 114 条 success-only 去重轨迹，30/30 tasks covered，51% 含关键 write-tool actions。

- 针对 Qwen2.5 chat template 实现 Render-Twice-Diff 多轮 loss mask，通过两次渲染 diff 精确定位 assistant turn token，解决长对话中 natural language、tool-call wrapper 与 JSON payload 的 token 对齐问题，避免 assistant-only loss 漏标或错标。

- 针对 20K-token 级长上下文 trajectory 做训练显存与稳定性优化：真实 teacher set p50=5.1K、max=19.2K tokens，默认 4096 truncation 会切掉 67/114 条样本的 late write-action turns；将 max_seq_len 提到 20480，并通过 eager attention -> SDPA、移除 all-ones attention mask、4096-token chunked cross-entropy over hidden states，避免 LxL attention 与 full-sequence fp32 vocab logits OOM。

- 建立严谨评测与泄漏识别体系：N-trial pass^k evaluation、paired-by-task bootstrap CI、McNemar exact test、seen/unseen split、fixed eval seed 与 pinned-greedy user simulator；10-trial nail-down 中每个 checkpoint n=200，paired bootstrap CI 排除 0。

- 系统化做 GRPO 机制分析与消融：对比 vanilla GRPO、PRM-Lite dense process reward、LATA length-aware advantage、PRM-Lite+LATA、turn-discounted advantage 等方案，定位 group-reward saturation、reward-length coupling、train/eval regime mismatch，并确认 LATA 在同预算下显著优于 turn-discount。

Key Results:

- Base -> Teacher-SFT -> GRPO: pass^1 0.20 -> 0.41 -> 0.55
- Single-session peak: pass^1 0.56
- Long-context SFT: 114/114 trajectories trained with zero truncation, peak 29.3GB / 31.4GB on one 32GB card
- Training-seed robustness: seed 101 / 202 / 303 endpoints = 0.520 / 0.550 / 0.490
- Nail-down evaluation: n=200 per checkpoint, paired bootstrap CI excluding 0
- Evidence: eval JSONL, master logs, ablation reports, seed runs, and GitHub artifact archive

### Search-Agent-RLVR: 可验证奖励的多轮检索智能体强化学习系统

2026.06 - 2026.07 | 独立开发者 | 个人项目 | GitHub

- 构建 HotpotQA 多轮检索 Agent RLVR 环境：模型输出 `<search>` 查询，本地 BM25 retriever 在 question-specific distractor corpus 中返回 passages，模型继续多轮推理并输出 `<answer>`；exact-match / token-F1 reward 可完全自动验证，避免 user simulator 或 LLM judge 噪声。

- 自实现端到端 GRPO/RLVR 训练链路：vLLM policy serving、multi-turn rollout collection、verifiable QA reward、group-relative advantage、outcome-variance gating、QLoRA 4-bit update、adapter iteration、paired held-out evaluation 与 raw rollout recheck。

- 实现纯 stdlib BM25 检索器与可复现实验环境，固定 corpus、检索参数（k1=1.5, b=0.75）和 reward parser；训练/测试问题 disjoint，增益来自同分布跨问题泛化，而不是训练集记忆或在线搜索波动。

- 优化 GRPO update 工程效率：将逐条 rollout loss 改为 padded batch + single `F.cross_entropy`，配合 length bucketing 降低 padding 浪费，并保持 assistant-token NLL 与逐样本实现数值一致。

- 设计双卡解耦训练执行脚本：vLLM resident server 常驻 GPU0，训练进程独占 GPU1，通过 LoRA adapter iteration / hot-reload 减少每轮 serve-stop-restart 开销；Round-4 在 2x5090 环境完成 6-iteration on-policy run，并对每轮做 McNemar + bootstrap CI。

- 系统分析 binary reward 下的 reward-optimization pressure：发现模型通过缩短 answer 获得 EM 增益；随后引入 token-F1 partial credit、length-aware advantage normalization、KL-to-base anchor、dense process reward 等对照方案。

- 做三路 controlled ablation：KL-to-base、dense process reward、LATA length-aware advantage；最终 LATA 达到最强且最稳定 endpoint，HotpotQA held-out EM 从 38.7% -> 49.3%，n=2400 multi-trial evaluation 复核 p < 1e-30。

- 将同一 trainer 迁移至 GSM8K exact-match RLVR：128 train questions x 8 rollouts per iter，4 iterations chained LoRA update，full test set n=1319 上 pass@1 从 61.4% -> 67.4%，验证 trainer 跨任务可复用。

Key Results:

- HotpotQA EM: 38.7% -> 49.3%, +10.7 pts, McNemar p < 0.001
- Multi-trial confirmation: n=2400, p < 1e-30
- GSM8K pass@1: 61.4% -> 67.4%, n=1319, McNemar p < 0.001
- Ablations: binary reward -> token-F1 -> KL / process reward / LATA
- Evidence: raw rollouts, EM recheck, answer-length behavior analysis, paired statistics

### YuResearchAgent: 面向长问题的 Deep Research 多智能体系统

2025.xx - 2026.xx | 独立开发者 | 个人项目 | GitHub

- 独立开发 Deep Research 多智能体系统，将复杂 query 拆解为 DAG 子任务，并通过 Planner、9-state async Orchestrator、Researcher、Summarizer、Critic / Red-Blue review、Memory Store 与 Compressor 完成并发检索、上下文压缩、报告合成和对抗修复。

- 实现工程化异步调度内核：DAG 拓扑层级并发、Semaphore 控制、失败重规划、全局 timeout 降级、AgentPool 复用、多后端 OpenAI-compatible model router，以及 CLI / REPL / Gradio streaming Web UI 三种入口。

- 自建 ResearchBench 35 题 x 11 领域评测体系，规则指标覆盖 semantic/factual coverage、citation coverage、hallucination risk、logical consistency、comprehensiveness、efficiency；用 paired bootstrap、p-value、Cohen's d 和 paired t-test 做统计验证。

- 在 15 题 head-to-head evaluation 中，相比 single-shot LLM baseline 综合分 0.5586 -> 0.6034，相对提升 +8.0%，paired bootstrap 95% CI=[+0.0134,+0.0761]，p=0.0021，Cohen's d=0.83。

- 设计 citation-aware synthesis pipeline，将 title / author / year / URL 等结构化 source metadata 注入 synthesis prompt，强制正文 `[N]` 引用与 `参考来源` 规范表；citation quality 4/10 -> 7/10，LLM-Judge overall 6/10 -> 8/10。

- 强化生产鲁棒性：统一 JSON fallback parser 处理 markdown fences、trailing commas、line comments、balanced braces 与 noisy prefixes，畸形 LLM 输出恢复率从 json.loads baseline 1/9 提升到 fallback 9/9；加入显式 request timeout、remaining-time adversarial wrapper、memory quality filter 与 path sandbox。

- 完成 API-free 单元测试与 CI：167 个 unit tests 覆盖 JSON parsing、orchestrator timeout/replan、rule-based metrics、memory heuristics、notepad/tool logic 等核心模块，CI 覆盖 Python 3.10-3.13。

Key Results:

- ResearchBench head-to-head: 0.5586 -> 0.6034, +8.0%, p=0.0021, Cohen's d=0.83
- Citation quality: 4/10 -> 7/10
- LLM-Judge overall: 6/10 -> 8/10
- JSON robustness: 1/9 -> 9/9 malformed-output recovery
- Engineering closure: DAG async orchestration / memory / citation / eval / Web UI / CI

## Compact Resume Version

Use this when space is tight.

### Agentic-GRPO-LongHorizon: 长链路多工具智能体的强化学习后训练系统

2026.06 - 2026.07 | 独立开发者 | 个人项目 | GitHub

- 面向 tau2-bench airline 长程客服任务，搭建 Baseline -> Teacher-SFT -> GRPO 三阶段 Agent RL pipeline，覆盖多轮对话、工具调用、用户模拟器交互、on-policy rollout、QLoRA 更新与 paired evaluation；held-out pass^1 从 Base 0.20 -> SFT 0.41 -> GRPO 0.55，单次峰值 0.56。

- 设计并实现多轮工具调用 Agent 的 GRPO 训练与评测闭环：policy server + self-hosted user simulator + tau2 tool execution + rollout collection + reward scoring + adapter update，支持长 trajectory 下的多 trial paired evaluation、bootstrap CI 与 McNemar 显著性检验。

- 构建 teacher-trajectory distillation warm start 方案，将策略从低信号 sparse-reward 区间提升到可学习区域；随后使用同一 GRPO pipeline 实现稳定提升，三组独立训练种子最终达到 pass^1 0.49-0.55，均相对 Base 和 SFT 起点显著提升。

- 系统分析 GRPO 在长链路 Agent 任务中的 reward signal 与优化机制，覆盖 group-reward saturation、reward-length coupling、train/eval regime mismatch 等问题，并通过 controlled ablation 验证 length-aware advantage、dense process reward、outcome-variance gating 等机制。

- 实现 Render-Twice-Diff 多轮 loss mask，解决 Qwen2.5 chat template 下 assistant turn token 对齐问题；配合长上下文训练显存优化、seeded replicates、日志归档和 artifact 备份，形成可复现实验链路。

Key Results:

- Base -> SFT -> GRPO: pass^1 0.20 -> 0.41 -> 0.55
- 10-trial nail-down: n=200 per checkpoint, paired bootstrap CI excluding 0
- Training-seed robustness: 0.49-0.55 across three independent GRPO replicates
- Evidence: eval JSONL, master logs, experiment reports, and GitHub artifact archive

### Search-Agent-RLVR: 可验证奖励的多轮检索智能体强化学习系统

2026.06 - 2026.07 | 独立开发者 | 个人项目 | GitHub

- 构建面向 HotpotQA 的多轮检索 Agent RLVR 系统，支持 `<search>` -> BM25 retrieval -> `<answer>` 的工具调用轨迹采集，并使用 exact-match / token-F1 作为 verifiable reward 进行 GRPO 后训练。

- 自实现 rollout collection、reward scoring、GRPO update、QLoRA adapter training 与 paired evaluation；在 held-out HotpotQA 上 Exact-Match 从 38.7% 提升至 49.3%（+10.7 pts），McNemar p < 0.001，并在 n=2400 multi-trial evaluation 中复核 p < 1e-30。

- 系统分析 binary reward 下的 reward-optimization pressure，定位 answer-length pressure；通过 token-F1 reward shaping 与 length-aware advantage normalization 稳定训练行为，对比 KL-to-base、dense process reward、LATA 三类方案。

- 将同一 GRPO/RLVR 框架迁移至 GSM8K exact-match reward 场景，pass@1 从 61.4% 提升至 67.4%，验证框架在检索 QA 与数学推理任务上的可复用性。

Key Results:

- HotpotQA EM: 38.7% -> 49.3%, +10.7 pts, McNemar p < 0.001
- Multi-trial confirmation: n=2400, p < 1e-30
- GSM8K pass@1: 61.4% -> 67.4%
- Evidence: reward design, controlled ablation, paired statistics, raw rollout recheck

### YuResearchAgent: 面向长问题的 Deep Research 多智能体系统

2025.xx - 2026.xx | 独立开发者 | 个人项目 | GitHub

- 独立开发 Deep Research 多智能体系统，将复杂 query 拆解为 DAG 子任务，并通过 Planner、Orchestrator、Researcher、Summarizer、Critic 等模块完成并发检索、共享记忆、上下文压缩、报告合成与对抗审查。

- 自建 ResearchBench 35 题 x 11 领域评测体系，覆盖事实性、引用质量、幻觉风险、逻辑一致性、完备性与效率；在 15 题 head-to-head evaluation 中，相比 single-shot LLM baseline 综合分提升 +8.0%，paired bootstrap p=0.0021，Cohen's d=0.83。

- 设计 citation-aware synthesis pipeline，将 title / author / year / URL 等结构化来源元信息传入合成器，生成规范参考文献表；引用质量从 4/10 提升到 7/10，LLM-Judge overall 从 6/10 提升到 8/10。

- 工程化实现 9 状态 async Orchestrator、DAG 拓扑调度、失败重规划、全局 timeout 降级、AgentPool 复用、多后端模型路由、Gradio streaming Web UI 和 167 个 API-free 单元测试。

Key Results:

- ResearchBench head-to-head: +8.0%, p=0.0021, Cohen's d=0.83
- Citation quality: 4/10 -> 7/10
- LLM-Judge overall: 6/10 -> 8/10
- Engineering closure: CLI / REPL / Web UI / CI / evaluation / citation pipeline

## Short Resume Version

- Built `tau2-agentic-rl`, a long-horizon tool-calling Agent GRPO post-training system on tau2-bench airline, covering Baseline -> Teacher-SFT -> GRPO, on-policy rollout, QLoRA update, paired evaluation, and multi-seed confirmation; held-out pass^1 improved from 0.20 -> 0.41 -> 0.55 with three GRPO seeds ending at 0.49-0.55.

- Built `search-agent-rlvr`, a verifiable-reward RL system for multi-turn retrieval agents, with self-implemented rollout collection, reward scoring, GRPO update, QLoRA adapter training, and paired statistics; HotpotQA EM improved 38.7% -> 49.3% and was reconfirmed at n=2400, p < 1e-30.

- Built `YuResearchAgent`, a citation-aware Deep Research multi-agent system with DAG planning, async orchestration, shared memory, adversarial review, ResearchBench evaluation, and Gradio streaming UI; achieved +8.0% over single-shot LLM baseline on ResearchBench and improved citation quality from 4/10 -> 7/10.

## 30-Second Pitch

我最近做的是一组三个 Agentic RL 项目。第一个是 tau2-bench airline 上的长链路工具调用 Agent 后训练系统，跑通 Base -> Teacher-SFT -> GRPO，pass^1 从 0.20 提到 0.55，并做了三训练种子复核。第二个是 search-agent-rlvr，在 HotpotQA 这种可验证奖励环境里研究 GRPO/RLVR，EM 从 38.7% 到 49.3%，n=2400 复核 p < 1e-30。第三个是 YuResearchAgent，一个 Deep Research 多智能体系统，有 DAG 调度、并发检索、引用增强、评测体系和 Web UI。整体主线是：我不仅做 Agent demo，而是做 Agent 训练、评测、机制分析和工程闭环。

## 3-Minute Interview Pitch

我的项目主线是 Agentic RL systems，分成三个层次。

第一层是可验证奖励环境，也就是 search-agent-rlvr。我先把任务简化成一个多轮检索 Agent：模型可以发 `<search>`，系统用 BM25 在 HotpotQA corpus 里检索，然后模型输出 `<answer>`。这个环境的好处是 reward 可验证，可以用 exact-match 和 token-F1，不依赖主观 judge。我自实现了 rollout collection、reward scoring、GRPO update、QLoRA adapter training 和 paired evaluation。最终 HotpotQA held-out EM 从 38.7% 提到 49.3%，并且 n=2400 复核 p < 1e-30。这个项目主要让我把 GRPO/RLVR 的机制研究清楚，包括 binary reward 的 answer-length pressure，以及 token-F1 和 length-aware advantage 如何稳定训练。

第二层是更真实的长链路工具调用 Agent，也就是 tau2-agentic-rl。这个项目面向 tau2-bench airline，多轮客服任务里模型要和 user simulator 交互、调用工具、完成 booking/change/cancel 等长程任务。我搭了 Baseline -> Teacher-SFT -> GRPO 三阶段 pipeline，包括 policy server、自托管 user simulator、tau2 tool execution、on-policy rollout、reward scoring、QLoRA update 和 paired evaluation。最终 held-out pass^1 从 Base 0.20 到 Teacher-SFT 0.41，再到 GRPO 0.55，单次 peak 0.56，三组独立 GRPO seed 都在 0.49-0.55。这个项目的重点是把可验证环境里学到的 reward optimization 和 length-aware 思路迁移到真实 noisy agent benchmark。

第三层是完整的 Agent 产品和评测系统，也就是 YuResearchAgent。它是一个 Deep Research 多智能体系统，把复杂 query 拆成 DAG 子任务，用 Planner、Orchestrator、Researcher、Summarizer、Critic 完成并发检索、共享记忆、上下文压缩、引用增强和报告生成。我还做了 ResearchBench 35 题 x 11 领域评测，15 题 head-to-head 里相对 single-shot LLM baseline 提升 +8.0%，p=0.0021。同时做了 citation-aware synthesis，把 title、author、year、URL 等结构化来源传给合成器，引用质量从 4/10 提到 7/10。

这三个项目连起来，就是从干净 RLVR 环境，到真实长链路 Agent GRPO，再到 Deep Research 多智能体工程系统。它们共同展示的是训练闭环、评测闭环、机制分析和工程落地能力。

## Packaging Rules

Use these rules in resumes, GitHub READMEs, and interviews:

- Lead with the result number before the implementation detail.
- Say "GRPO post-training system", not "toy trainer" or "script".
- Say "mechanism analysis" or "controlled ablation", not "failure".
- Say "low-signal sparse-reward region", not "it did not work".
- Say "evidence archived", not "logs lying around".
- Mention exact statistics only when they are backed by artifacts.
- Do not mention "No veRL / No TRL" unless directly asked about implementation choices.
- Do not lead with limitations; discuss them only if the interviewer asks.

## Project Links

- tau2-agentic-rl: https://github.com/yuyu0529nya/tau2-agentic-rl
- search-agent-rlvr: https://github.com/yuyu0529nya/search-agent-rlvr
- YuResearchAgent: https://github.com/yuyu0529nya/YuResearchAgent
