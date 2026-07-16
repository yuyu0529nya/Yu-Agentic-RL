# Resume Bullets — LLM Agent RL Post-Training Project

Honest, interview-defensible. Most numbers are from paired significance-tested runs; the veRL
reproduction headline is 2-seed (mean±std ≈ 0.556±0.01), the gating A/B is single-seed (flagged inline).
GRPO described accurately (group-relative advantage replaces the critic — NOT a full
PPO clip/KL implementation).

---

## 项目名
LLM Agent 强化学习后训练系统(GRPO + 可验证奖励)

## 中文 bullets(按强弱;位置紧就留前 3 条)
1. 从零实现单卡 GRPO 训练框架:组相对优势做 baseline(免 critic)、outcome/连续方差优势门控、优势加权 assistant-token 似然的批处理更新,并支持 KL-to-base 锚与长度感知优势归一;QLoRA 4-bit + vLLM,在单张 32G 卡上交替推理/训练完成 7B 在线 RL。
2. 多轮工具 Agent(tau2 airline,主线成果)——把三次 null 翻盘成显著正结果:先诊断 GRPO-from-base 的 null 根因为梯度饥饿(base 成功率~20%→多数 rollout 组全败→组内方差为零→无梯度)+ 自采样技能盲区(成功轨迹从不调用改签类写工具);再用教师轨迹蒸馏做热启动(强模型经同一 tau2 环境采成功轨迹→行为克隆 QLoRA SFT),held-out pass^1 base 0.20→蒸馏 0.41→GRPO 0.55(10-trial nail-down n=200,三段全部显著、配对 bootstrap CI 排除 0;四种奖励配置均显著超 base 与各自 SFT 起点)。关键佐证:同一 2×2 消融从 base 起跑全 null、仅换蒸馏热启动即全部起飞,证明瓶颈是起点太弱而非算法。附机制发现:RL 上限由技能覆盖而非 SFT 地基高低决定(0.14 与 0.32 两种地基经 RL 均达 ~0.55);训练量消融显示曲线 ~5 迭代饱和于 0.55-0.56、续训至 10 迭代无过优化崩盘,Turn-Discounted 优势同预算对照持平淘汰;3 个独立训练 seed 重训终点 0.49-0.55,每个 seed 对 base 与对 SFT 起点均单独显著(训练随机性稳健)。
3. veRL 工业框架复现并超越(单卡 96G):把上面的手写 tau2-airline GRPO 移植进 veRL(FSDP + async vLLM),自建 tau2↔veRL 多轮 agent loop(ContextVar 工具态并发隔离、fused-CE kernel、单张 96G 卡跑通 7B 在线 RL);诊断复现初期不收敛根因为 lr 过小(grad_norm≈0.05、距裁剪阈差 20×、pg_clipfrac≈0.001)而非样本效率,修正后 held-out BINARY mean@4 0.275→0.5625,超过自研手写版的 0.545(跨 2 seed 复现:0.5625 / 0.55,mean±std ≈ 0.556±0.01)。并读 rollout 把"分数涨"落到"行为可解释":晚期模型查询类工具调用 ↑2–5×(先查证再行动)、超权限时带准确 summary 规范转人工、自己乱改的写操作反而更克制。
4. 对自研门控做受控 A/B,给出诚实负结果:为验证"binary-outcome 优势门控(dynamic sampling)提升样本效率"的假设,做单变量 A/B(门控开/关,同 lr 同 seed、其余一字不差)——用数据证伪:门控关 val 0.5625 vs 门控开 0.4125;机理为 ~20% 成功率下门控丢弃 54–75% 的 rollout(live_frac 0.25–0.46)→ 饿死梯度,样本量损失压倒去噪收益。门控实现经 11 项 CPU 单测验证正确(杀掉全败组的长度相关幻影梯度),结论是它在低成功率任务上不迁移。展示控制实验设计 + 因果归因 + 学术诚实。
5. 多轮检索 Agent + RLVR(可控环境的 RL 科学腿):构建"搜索→BM25 检索→作答"多轮 agent,以 token 级 F1 为可验证奖励做 on-policy GRPO,held-out EM 38.7%→49.3%(+10.7,McNemar p<0.001,n=300;多评复核 n=2400,p<1e-30)——确定性奖励环境把测量噪声隔离掉,使下面的过优化机制研究成为可能。
6. 诊断并根治 RL 过度优化:定位 held-out 崩盘为答案长度坍缩(24→7 字符)的 Goodhart 现象,系统对比三种防过优化手段——KL-to-base 锚、稠密过程奖励、长度感知优势归一;长度归一直击坍缩机制,取得最高且最稳结果(0.493,终点即最优、无需早停),KL 防崩但牺牲峰值(0.437),过程奖励保峰值且后期稳(0.470);进一步把两强杠杆组合后做调稳消融——发现调轻过程奖励(β0.3→0.15)或增大组规模(N8→12)能消除中期 dip 且保峰(EM 0.49/F1 0.61),而 KL 只减轻未根治,再次印证对症修复优于通用正则;两强变体经 8-trial 多评(n=2400)钉死,均 EM~0.49/F1~0.61、McNemar p<0.0001、彼此统计平局,且与上一轮冠军同水平(消 dip 不损峰)。
7. 单轮推理 RLVR 交叉验证:Qwen2.5-1.5B + GSM8K,pass@1 61.4%→67.4%(+6.0,McNemar p<0.001,n=1319),验证训练框架正确性。
8. 评测纪律贯穿全项目:held-out 划分 / 多 trial pass@1 / 配对 McNemar + bootstrap CI;正负结果一视同仁,所有数字可从原始轨迹 jsonl 复算。

## 两行极简版
> 自研 GRPO 后训练框架(免 critic 组相对优势 + 优势门控 + KL 锚 + 长度感知优势,QLoRA+vLLM);tau2 多轮工具客服 Agent 上以"教师蒸馏热启动 + GRPO"把三次 null 翻盘为 held-out pass^1 0.20→0.41→0.55(n=200 配对显著,并证明 null 根因是梯度饥饿);把该 GRPO 复现进工业框架 veRL、诊断真因是 lr 后 held-out 0.275→0.5625 超过手写版,并对自研门控做受控 A/B 证伪其有效性(诚实负结果);另在可控奖励环境训练多轮检索 Agent(EM 38.7%→49.3%,p<0.001,n=2400 复核),系统对比三种防过优化手段,GSM8K 交叉验证训练器正确性。

---

## English bullets
**LLM Agent RL Post-Training System (GRPO + Verifiable Rewards)**
- Built a single-GPU GRPO trainer from scratch: group-relative advantage baseline (critic-free), outcome/continuous-variance advantage gating, batched advantage-weighted assistant-token NLL, with optional KL-to-base anchor and length-aware advantage normalization; QLoRA 4-bit + vLLM, alternating serve/train to run online RL on a 7B model on one 32 GB GPU.
- Multi-turn tool agent (tau2 airline — the headline): turned three nulls into a significant win. Diagnosed the GRPO-from-base null as gradient starvation (base ~20% success → all-fail rollout groups → zero intra-group variance → no gradient) plus a self-sampling skill gap; fixed it with teacher-trajectory distillation as a warm start, then GRPO: held-out pass^1 **0.20 → 0.41 → 0.55** (10-trial nail-down, n=200, every stage's paired CI excluding 0). Key control: the identical 2×2 ablation is all-null from base and uniformly significant from the distillation floor. Mechanistic extras: the RL ceiling is set by skill coverage, not SFT-floor height; training saturates by ~iter 5 and stays collapse-free through 10 iterations; turn-discounted advantage flatlines at matched budget; and 3 independent training-seed replicates land at 0.49–0.55, each individually significant vs both base and its SFT start.
- Reproduced and surpassed it on veRL (FSDP + async-vLLM industrial framework, single 96 GB GPU): built the tau2↔veRL multi-turn agent loop (ContextVar tool-state isolation, fused-CE kernel); diagnosed the early non-convergence as an **LR problem** (grad_norm≈0.05, ~20× under the clip threshold), not sample efficiency — after fixing lr, held-out BINARY mean@4 **0.275 → 0.5625**, above my hand-written GRPO's 0.545 (reproduced across 2 seeds: 0.5625 / 0.55, mean±std ≈ 0.556±0.01). Read the rollouts to make "the number went up" explainable: the late-stage policy calls query tools 2–5× more (verify before acting), hands off to a human with an accurate summary when out of scope, and is more restrained with write actions.
- Ran a controlled A/B that **refuted my own trick** (research honesty): to test whether my binary-outcome advantage gating (dynamic sampling) improves sample efficiency, I ran a single-variable A/B (gating on/off, same lr/seed, everything else identical) — the data refuted it: gating **OFF 0.5625 vs ON 0.4125**. Mechanism: at ~20% success the gate drops **54–75%** of rollouts (live_frac 0.25–0.46) and starves the gradient. The gate's implementation is correct (11 CPU unit tests kill the phantom length-advantage on all-fail groups); it simply does not transfer to a low-success-rate task.
- Multi-turn retrieval agent + RLVR (controlled-environment RL-science leg): `<search>` → BM25 retrieval → `<answer>` loop trained with a token-F1 verifiable reward via on-policy GRPO; held-out EM 38.7% → 49.3% (+10.7, McNemar p<0.001, n=300; re-confirmed at n=2400, p<1e-30). The deterministic reward isolates measurement noise, enabling the over-optimization mechanism study below.
- Diagnosed and fixed reward over-optimization: traced the held-out collapse to answer-length collapse (24→7 chars, a Goodhart effect), then ran a head-to-head of three anti-collapse levers — KL-to-base anchor, dense process reward, and length-aware advantage. The length-aware fix targets the mechanism directly and won (0.493, highest and most stable, endpoint = best so no early stopping); KL prevents collapse but caps the peak (0.437); process reward keeps the peak and stays stable (0.470). A follow-up stabilization ablation on the *combined* lever showed that weakening the process reward (β 0.3→0.15) or enlarging the group (N 8→12) removes a residual mid-run dip while keeping the peak (EM 0.49 / F1 0.61), whereas a KL anchor only softens it — reconfirming mechanism-targeted fixes over generic regularizers. Both top variants were then nailed down at 8 trials/question (n=2400): each reaches EM ~0.49 / F1 ~0.61, McNemar p<0.0001, a statistical tie with each other and on par with the prior round's best (dip removed at no cost to the peak).
- Cross-validated the trainer on single-turn reasoning: Qwen2.5-1.5B on GSM8K, pass@1 61.4% → 67.4% (+6.0, McNemar p<0.001, n=1319).
- Evaluation discipline throughout: held-out splits, multi-trial pass@1, paired McNemar + bootstrap CI; negative and positive results held to the same recompute-from-raw-JSONL standard.

---

## 30-秒口头答案:为什么 GRPO 而非 PPO/DPO
用可验证奖励(精确匹配/F1),不需要 reward model;GRPO 用一组样本的均值替代 critic 当 baseline,省掉与策略同尺寸的价值网络,所以单张 32G 卡就能跑 7B 在线 RL。PPO 要 actor+critic 显存翻倍且工程更重;DPO 是离线偏好优化、拿不到"模型自己试错变好"的在线曲线。

## 30-秒口头答案:过度优化怎么发现、三种修法怎么选
根因:奖励被"答案变短"刷分——held-out 崩盘时答案从 24 字符坍缩到 7。我用三种手段对照修复:(1) KL-to-base 锚把策略拽回基座,防崩但太"钝",连有益的缩短也压住,峰值最低(0.437);(2) 稠密过程奖励给检索中间步加信号,保住峰值且后期稳(0.470),但我的数据显示检索召回本就高(rHit 0.75→0.83),所以它帮的是稳定而非上限;(3) 长度感知优势(优势÷√长度)直接抵消"越短优势越大"的梯度,最对症——最高分 0.493、终点即最优、无需早停。结论:对症的机制级修复 > 通用正则 > 间接信号。
