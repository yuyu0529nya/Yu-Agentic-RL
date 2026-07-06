# Resume Bullets — LLM Agent RL Post-Training Project

Honest, interview-defensible. All numbers are from paired significance-tested runs.
GRPO described accurately (group-relative advantage replaces the critic — NOT a full
PPO clip/KL implementation).

---

## 项目名
LLM Agent 强化学习后训练系统(GRPO + 可验证奖励)

## 中文 bullets(按强弱;位置紧就留前 3 条)
1. 从零实现单卡 GRPO 训练框架:组相对优势做 baseline(免 critic)、outcome/连续方差优势门控、优势加权 assistant-token 似然的批处理更新,并支持 KL-to-base 锚与长度感知优势归一;QLoRA 4-bit + vLLM,在单张 32G 卡上交替推理/训练完成 7B 在线 RL。
2. 多轮检索 Agent + RLVR(旗舰):构建"搜索→BM25 检索→作答"多轮 agent,以 token 级 F1 为可验证奖励做 on-policy GRPO,held-out EM 38.7%→49.3%(+10.7,McNemar p<0.001,n=300)。
3. 诊断并根治 RL 过度优化:定位 held-out 崩盘为答案长度坍缩(24→7 字符)的 Goodhart 现象,系统对比三种防过优化手段——KL-to-base 锚、稠密过程奖励、长度感知优势归一;长度归一直击坍缩机制,取得最高且最稳结果(0.493,终点即最优、无需早停),KL 防崩但牺牲峰值(0.437),过程奖励保峰值且后期稳(0.470);进一步把两强杠杆组合后做调稳消融——发现调轻过程奖励(β0.3→0.15)或增大组规模(N8→12)能消除中期 dip 且保峰(EM 0.49/F1 0.61),而 KL 只减轻未根治,再次印证对症修复优于通用正则;两强变体经 8-trial 多评(n=2400)钉死,均 EM~0.49/F1~0.61、McNemar p<0.0001、彼此统计平局,且与上一轮冠军同水平(消 dip 不损峰)。
4. 单轮推理 RLVR 交叉验证:Qwen2.5-1.5B + GSM8K,pass@1 61.4%→67.4%(+6.0,McNemar p<0.001,n=1319),验证训练框架正确性。
5. 多轮工具 Agent(tau2 airline)——把三次 null 翻盘成显著正结果:先诊断 GRPO-from-base 的 null 根因为梯度饥饿(base 成功率~20%→多数 rollout 组全败→组内方差为零→无梯度)+ 自采样技能盲区(成功轨迹从不调用改签类写工具);再用教师轨迹蒸馏做热启动(强模型经同一 tau2 环境采成功轨迹→行为克隆 QLoRA SFT),held-out pass^1 base 0.20→蒸馏 0.41→GRPO 0.55(10-trial nail-down n=200,三段全部显著、配对 bootstrap CI 排除 0;四种奖励配置均显著超 base 与各自 SFT 起点)。关键佐证:同一 2×2 消融从 base 起跑全 null、仅换蒸馏热启动即全部起飞,证明瓶颈是起点太弱而非算法。附机制发现:RL 上限由技能覆盖而非 SFT 地基高低决定(0.14 与 0.32 两种地基经 RL 均达 ~0.55)。全程 held-out 划分 / 多 trial pass@1 / 配对 McNemar+bootstrap CI 规范,正负结果一视同仁复算存证。

## 两行极简版
> 自研单卡 GRPO 后训练框架(免 critic 组相对优势 + 优势门控 + KL 锚 + 长度感知优势,QLoRA+vLLM);以可验证奖励训练多轮检索 Agent,held-out EM 38.7%→49.3%(+10.7,p<0.001);系统对比三种防过优化手段(长度归一最优),并在 GSM8K 交叉验证(+6.0,p<0.001)。

---

## English bullets
**LLM Agent RL Post-Training System (GRPO + Verifiable Rewards)**
- Built a single-GPU GRPO trainer from scratch: group-relative advantage baseline (critic-free), outcome/continuous-variance advantage gating, batched advantage-weighted assistant-token NLL, with optional KL-to-base anchor and length-aware advantage normalization; QLoRA 4-bit + vLLM, alternating serve/train to run online RL on a 7B model on one 32 GB GPU.
- Multi-turn retrieval agent + RLVR (flagship): `<search>` → BM25 retrieval → `<answer>` loop trained with a token-F1 verifiable reward via on-policy GRPO; held-out EM 38.7% → 49.3% (+10.7, McNemar p<0.001, n=300).
- Diagnosed and fixed reward over-optimization: traced the held-out collapse to answer-length collapse (24→7 chars, a Goodhart effect), then ran a head-to-head of three anti-collapse levers — KL-to-base anchor, dense process reward, and length-aware advantage. The length-aware fix targets the mechanism directly and won (0.493, highest and most stable, endpoint = best so no early stopping); KL prevents collapse but caps the peak (0.437); process reward keeps the peak and stays stable (0.470). A follow-up stabilization ablation on the *combined* lever showed that weakening the process reward (β 0.3→0.15) or enlarging the group (N 8→12) removes a residual mid-run dip while keeping the peak (EM 0.49 / F1 0.61), whereas a KL anchor only softens it — reconfirming mechanism-targeted fixes over generic regularizers. Both top variants were then nailed down at 8 trials/question (n=2400): each reaches EM ~0.49 / F1 ~0.61, McNemar p<0.0001, a statistical tie with each other and on par with the prior round's best (dip removed at no cost to the peak).
- Cross-validated the trainer on single-turn reasoning: Qwen2.5-1.5B on GSM8K, pass@1 61.4% → 67.4% (+6.0, McNemar p<0.001, n=1319).
- Multi-turn tool agent (tau2 airline) — turned three nulls into a significant win: diagnosed the GRPO-from-base null as gradient starvation (base ~20% success → most rollout groups all-fail → zero intra-group variance → no gradient) plus a self-sampling skill gap (the base policy's own successes never call the reservation-change write-tools). Fixed it with teacher-trajectory distillation as a warm start (a strong model plays the agent through the same tau2 harness → behavior-cloning QLoRA SFT), then GRPO: held-out pass^1 base 0.20 → distilled 0.41 → RL 0.55 (10-trial nail-down, n=200; all three stages significant with paired bootstrap CIs excluding 0; all four reward configs beat both base and their own SFT start). Key control: the identical 2×2 ablation is all-null from base and uniformly significant from the distillation floor — proving the blocker was a too-weak starting policy, not the algorithm. Plus a mechanistic finding: the RL ceiling is set by skill coverage, not SFT-floor height (a 0.14 floor and a 0.32 floor both reach ~0.55 after RL). Throughout: held-out splits, multi-trial pass@1, paired McNemar + bootstrap CI, with negative and positive results held to the same recompute-from-raw standard.

---

## 30-秒口头答案:为什么 GRPO 而非 PPO/DPO
用可验证奖励(精确匹配/F1),不需要 reward model;GRPO 用一组样本的均值替代 critic 当 baseline,省掉与策略同尺寸的价值网络,所以单张 32G 卡就能跑 7B 在线 RL。PPO 要 actor+critic 显存翻倍且工程更重;DPO 是离线偏好优化、拿不到"模型自己试错变好"的在线曲线。

## 30-秒口头答案:过度优化怎么发现、三种修法怎么选
根因:奖励被"答案变短"刷分——held-out 崩盘时答案从 24 字符坍缩到 7。我用三种手段对照修复:(1) KL-to-base 锚把策略拽回基座,防崩但太"钝",连有益的缩短也压住,峰值最低(0.437);(2) 稠密过程奖励给检索中间步加信号,保住峰值且后期稳(0.470),但我的数据显示检索召回本就高(rHit 0.75→0.83),所以它帮的是稳定而非上限;(3) 长度感知优势(优势÷√长度)直接抵消"越短优势越大"的梯度,最对症——最高分 0.493、终点即最优、无需早停。结论:对症的机制级修复 > 通用正则 > 间接信号。
