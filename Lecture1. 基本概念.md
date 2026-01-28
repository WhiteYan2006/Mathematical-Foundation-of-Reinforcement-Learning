---
date: 2026-01-28
tags:
  - 强化学习
---

![[Chapter_1.pdf]]

# Lecture 1: Basic Concepts in Reinforcement Learning

> [!info] 学习导航
> * **前置知识**: 概率论基础（条件概率、期望）。
> * **核心目标**: 理解 Agent 如何通过与 Environment 交互，最大化长期累积奖励。
> * **关联章节**: 为 [[Lecture2. 贝尔曼公式]] 奠定符号基础。

## 1. 核心直觉 (Intuition)

在深入数学符号之前，我们先用一个非 Grid World 的生活案例来建立直觉。

> [!example] 直觉案例：教狗狗握手
> 想象你在训练一只狗狗（**Agent**）：
> * **环境 (Environment)**: 客厅。
> * **状态 (State)**: 狗狗是坐着、站着，还是在乱跑。
> * **动作 (Action)**: 狗狗伸出爪子、叫一声，或者原地打滚。
> * **奖励 (Reward)**: 如果狗狗伸出爪子，你给它一块肉干（+1）；如果它乱叫，你通过不理睬或斥责给予反馈（-1 或 0）。
>
> **核心逻辑**:
> 1.  狗狗最初是乱试的（随机策略）。
> 2.  它发现“伸爪子”这个动作经常紧接着“肉干”这个奖励。
> 3.  为了吃到更多肉干（最大化 **Return**），它调整了自己的行为模式（**Policy**），在听到“握手”指令（特定状态）下，倾向于做“伸爪子”这个动作。
>
> 强化学习就是通过**试错 (Trial-and-Error)**，学习在不同状态下应该采取什么动作，从而让未来的奖励总和最大化。

---

## Example: Grid World
![[Pasted image 20260128173307.png]]


## 2. 基础组件与定义 (Basic Elements)

基于**Grid World**的例子，形式化定义 RL 的关键要素。

### 2.1 状态 (State) & 状态空间 (State Space)
* **定义**: 智能体**Agent**相对于环境**Environment**的状态**Status**描述。
* **符号**: $s$。
* **例子**: 在**Grid World**中，**位置**即状态。
* **状态空间State Space**：$\mathcal{S}$代表所有状态的集合
    $$\mathcal{S} = \{s_i\}_{i=1}^9=\{s_1, s_2, ..., s_9\}$$
   

### 2.2 动作 (Action) & 动作空间 (Action Space)
* **定义**: 智能体在某个状态下可以做出的行为。
* **符号**: $a$。
* **动作空间**: 状态 $s_i$ 下所有可用动作的集合，记为 $\mathcal{A}(s_i)$。

> [!important] 注意
> 不同的状态可能拥有不同的动作空间，所以动作空间的数学描述中会有$(s_{i})$。$\mathcal{A}$可以视为$s_{i}$的函数
* **例子**:
    $$\mathcal{A}(s_i) = \{a_1(\text{up}), a_2(\text{right}), a_3(\text{down}), a_4(\text{left}), a_5(\text{stay})\}$$

### 2.3 状态转移 (State Transition)
- 当智能体在状态 $s$ 采取动作 $a$ 后，环境会使其转变到新的状态 $s'$。
	- 例如$s_1 \xrightarrow{a_2} s_2$
- State Transition定义了一种agent和环境交互的行为
- 这一过程通常是**随机的 (Stochastic)**。


> [!math] 状态转移概率 (State Transition Probability)
> 我们使用**条件概率**来描述这一动态过程：
> $$p(s' | s, a) = \mathbb{P}(S_{t+1}=s' | S_t=s, A_t=a)$$
>
> * **解释**: 在当前状态 $s$ 采取动作 $a$ 的条件下，下一时刻状态变成 $s'$ 的概率。
> * **性质**: 对于固定的 $s$ 和 $a$，所有可能的下一状态概率之和为 1：
>     $$\sum_{s' \in \mathcal{S}} p(s' | s, a) = 1$$

### 2.4 策略 (Policy)
策略是智能体的大脑，决定了它在某个状态下如何行动。
	**Policy**会告诉Agent如果在一个State应该采取什么Action

* **符号**: $\pi$。
* **数学表示**:
    $$\pi(a | s) = \mathbb{P}(A_t=a | S_t=s)$$
- **数学含义**：$\pi$指定了任何一个状态下，任何一个Action的概率式多少
	- 例：$\pi(a_{1}|s_{1})=0$代表在$s_{1}$下采取$a_{1}$的概率为0；$\pi(a_{2}|s_{1})=1$代表在$s_{1}$下采取$a_{2}$的概率为1
	   
* **类型**:
    1.  **确定性策略 (Deterministic Policy)**: 在状态 $s$ 必定采取某个动作。即 $\pi(a|s) = 1$ (对某个 $a$)，其余为 0。
    2.  **随机性策略 (Stochastic Policy)**: 在状态 $s$ 以一定概率分布选择动作。例如 50% 往左，50% 往右。

### 2.5 奖励 (Reward)
* **定义**: 采取动作后获得的**标量**反馈信号。
* **符号**: $r$。
* **概率描述**: $p(r | s, a)$ 表示在状态 $s$ 做动作 $a$ 获得奖励 $r$ 的概率。
* **作用**: 引导智能体行为的指挥棒。**正数**代表鼓励，**负数**代表惩罚。

---

## 3. 回报与折扣 (Trajectory & Return)

这是本章数学推导的核心。我们需要量化“什么样的策略是好的”。

### 3.1 轨迹 (Trajectory)
一个**轨迹Trajectory**（或称为 **Episode**）是**状态-动作-奖励**（State-Action-Reward）的序列：
> [!example] Trajectory Example
> 这是一个Trajectory：在$s_{1}$采取$a_{2}$得到$r=0$，接着……
> $$s_1 \xrightarrow[r=0]{a_2} s_2 \xrightarrow[r=0]{a_3} s_5 \xrightarrow[r=0]{a_3} s_8 \xrightarrow[r=1]{a_2} s_9$$


### 3.2 回报 (Return)
- 我们不仅关心即时奖励，更关心**未来所有奖励的总和**。这个总和称为**Return**。
- **Return**是针对一个**Trajectory**而言的，把沿着这个Trajectory的所有Reward加起来。
	- 使用Return来在数学上刻画一个Policy的好坏
 

### 3.3 折扣回报 (Discounted Return) - Level C 推导


> [!math] 无折扣回报 (Undiscounted Return)
> $$G_t = r_{t+1} + r_{t+2} + r_{t+3} + ... + r_T$$
> 其中 $T$ 是终止时刻。
>

但是，对于**连续性任务 (Continuing Tasks)**，即没有终止状态的任务（$T = \infty$），上述求和可能发散趋于无穷大。因此，我们需要引入**折扣因子 (Discount Rate)**。

引入折扣因子**discount rate**: $\gamma \in [0, 1)$。

> [!abstract] 定义：（Discounted Return)折扣回报
> $$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$
>

**为什么引入 $\gamma$？**
1.  **数学收敛性**: 避免无穷级数发散。
2.  **经济学解释**: 现在的 100 块钱比未来的 100 块钱更有价值（不确定性、通胀等）。$\gamma \to 0$ 意味着智能体短视（只看眼前）；$\gamma \to 1$ 意味着远视。

- 通过控制$\gamma$，可以控制Agent学习到的策：减小$\gamma$使得它更加“近视“，更加注重最近的Reward；反之则更加”远视“

### 3.4 Episode
- 和Trajectory差不多，Episode就是一个**走到结束状态（Terminal State）的Trajectory**
- 一般是**有限步**的，这样的任务也被称为**episodic tasks**
	- 没有Terminal State的任务：**Continuing Tasks**

---

## 4. 马尔可夫决策过程 (MDP)

所有上述概念最终汇总于 **MDP** 框架。


### 4.1 Key elements of MDP
- Sets：
	- States：$\mathcal{S}$: 状态空间。
	- Action：$\mathcal{A}(s)$: 动作空间。
	- Reward：$\mathcal{R}(s,a)$: 奖励空间
- Probability distribution：
	- State Transition Probability：$p(s'|s,a)$: 状态转移概率
	- Reward Probability：$p(r|s,a)$: 奖励概率
- Policy：在状态$s$，采取行动$a$的概率$\pi(a|s)$

### 4.2 马尔可夫性质 (Markov Property)
RL 的核心假设是：**历史无关性假设 (Memoryless Property)**。

> [!abstract] 定义
> 下一状态 $s_{t+1}$ 和奖励 $r_{t+1}$ 仅取决于当前状态 $s_t$ 和动作 $a_t$，而与过去的历史轨迹无关。
>
> $$p(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ...) = p(s_{t+1} | s_t, a_t)$$
> $$p(r_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ...) = p(r_{t+1} | s_t, a_t)$$
>

### 4.3 Markov Decision Process VS Markov Process
- 当**Policy确定**，Markov Decision Process就变成Markov Process


---

## 5. 总结与下一步 (Summary & Next Step)

本章我们构建了强化学习的“语言系统”。我们定义了智能体（Agent）如何通过**策略 (Policy)** 在 **MDP** 中与环境交互，产生**轨迹 (Trajectory)**，并为了最大化**折扣回报 (Discounted Return)** 而优化行为。

**关键公式回顾**:
* 策略: $\pi(a|s)$
* 转移: $p(s'|s,a)$
* 回报: $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$

> [!question] 思考
> 我们知道了如何计算一条特定轨迹的回报。但是，环境是随机的（$p(s'|s,a)$），策略也是随机的（$\pi(a|s)$）。我们如何评价一个状态 $s$ 到底是好是坏呢？仅看一次随机轨迹是不够的。
>
> 这就需要引入**期望 (Expectation)** 的概念，从而引出下一章的核心——**状态价值函数 (State Value Function)** 和 **贝尔曼方程 (Bellman Equation)**。