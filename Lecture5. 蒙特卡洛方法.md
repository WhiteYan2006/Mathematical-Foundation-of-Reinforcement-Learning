---
date: 2026-02-07
tags:
  - 强化学习
---
![[Chapter_5.pdf]]

# 第五章：蒙特卡洛方法 (Monte Carlo Methods)

> [!abstract] 核心直觉
> 
> 之前的章需要已知环境的**模型 (Model)**，即状态转移概率 $p(s'|s,a)$ 和奖励函数 $r(s,a)$。
> 
> **蒙特卡洛 (Monte Carlo, MC)** 方法的核心在于**“免模型 (Model-free)”**。当环境未知时，我们无法通过计算直接求期望，但可以通过与环境交互，产生大量的**经验轨迹 (Episodes)**，利用“大数定律”将样本均值作为期望的估计值。
> 
> **一句话总结**：用大量的**采样 (Sampling)** 来逼近未知的**期望 (Expectation)**。

---

## 1. 动机案例：均值估计 (Mean Estimation)

在进入强化学习之前，我们先通过一个简单的统计学例子来理解 MC 的本质。

### 1.1 抛硬币模型

假设有一个随机变量 $X$，代表抛硬币的结果：
- 正面 (Head): $X = +1$
- 反面 (Tail): $X = -1$

我们的目标是计算 $X$ 的期望 $\mathbb{E}[X]$。

#### 方法一：基于模型 (Model-based)

如果我们已知硬币的概率分布（模型）：
$$p(X=1)=0.5, \quad p(X=-1)=0.5$$
则可以直接根据定义计算：
$$\begin{aligned} \mathbb{E}[X] &= \sum_{x} x p(x) \\ &= 1 \times 0.5 + (-1) \times 0.5 \\ &= 0 \end{aligned}$$

> [!important] 局限性
> 
> 在现实世界的强化学习任务中（如控制机器人），我们通常**无法预知**环境的精确概率分布 $p(s'|s,a)$。

#### 方法二：免模型 (Model-free) - 蒙特卡洛估计
如果我们不知道概率分布，可以通过**重复试验**来估计。
假设我们抛了 $N$ 次硬币，得到样本序列 $\{x_1, x_2, \dots, x_N\}$。
根据**大数定律 (Law of Large Numbers)**，样本均值 $\bar{x}$ 会收敛于真实期望 $\mathbb{E}[X]$：
$$\mathbb{E}[X] \approx \bar{x} = \frac{1}{N} \sum_{j=1}^{N} x_j$$
> [!math] 数学性质：无偏性与收敛
> 
> 假设样本 $\{x_j\}$ 是**独立同分布 (i.i.d.)** 的，样本均值 $\bar{x}$ 具有以下性质：
> 
> 1. **无偏估计 (Unbiased)**:
>     
>     $$\mathbb{E}[\bar{x}] = \mathbb{E}\left[ \frac{1}{N}\sum x_j \right] = \frac{1}{N}\sum \mathbb{E}[x_j] = \mathbb{E}[X]$$
>     
> 2. **方差收敛**:
>     
>     $$\text{Var}[\bar{x}] = \text{Var}\left[ \frac{1}{N}\sum x_j \right] = \frac{1}{N^2}\sum \text{Var}[x_j] = \frac{1}{N}\text{Var}[X]$$
>     
>     当 $N \to \infty$ 时，方差趋于 0，估计值越来越精准。
>     

### 1.2 为什么要用 MC 做强化学习？

在强化学习中，**状态价值 (State Value)** $v_{\pi}(s)$ 和 **动作价值 (Action Value)** $q_{\pi}(s,a)$ 的定义本质上就是**期望**：
$$v_{\pi}(s) = \mathbb{E}[G_t | S_t = s]$$
$$q_{\pi}(s,a) = \mathbb{E}[G_t | S_t = s, A_t = a]$$

其中 $G_t$ 是从 $t$ 时刻开始的**回报 (Return)**。既然是期望，且通常没有模型，我们就可以用 MC 方法，通过采样多条轨迹，计算回报 $G_t$ 的平均值来估计价值。

---

## 2. 最简单的 MC 算法：MC Basic

MC Basic 算法本质上是**策略迭代 (Policy Iteration)** 的一个“免模型”变种。

### 2.1 从策略迭代到 MC Basic

回顾第四章的策略迭代，包含两个步骤：
1. **策略评估 (Policy Evaluation)**: 计算 $v_{\pi}$。
2. **策略提升 (Policy Improvement)**: $\pi_{new} = \arg\max (r + \gamma P v_{\pi})$。

在免模型设定下，我们面临两个困难：
1. 我们无法利用贝尔曼方程解 $v_{\pi}$，因为不知道 $P$。
2. 即使有了 $v_{\pi}$，我们也无法进行策略提升，因为策略提升公式 $\arg\max_a \sum p(s'|s,a) v_{\pi}(s')$ 依然需要模型 $p$。

> [!math] 关键推导：为什么要估计 $q(s,a)$ 而不是 $v(s)$？
> 
> 策略提升需要求解：
> 
> $$\pi_{k+1}(s) = \arg\max_{a} \left[ R(s,a) + \gamma \sum_{s'} p(s'|s,a) v_{\pi_k}(s') \right]$$
> 
> 注意方括号内的项正是**动作价值** $q_{\pi_k}(s,a)$。
> 
> - 如果我们只有 $v_{\pi_k}$，想要选出最好的动作 $a$，必须利用模型 $p(s'|s,a)$ 来推演下一步。
>     
> - 如果我们直接估计了 **$q_{\pi_k}(s,a)$**，则策略提升变得非常简单，无需模型：
>     
>     $$\pi_{k+1}(s) = \arg\max_{a} q_{\pi_k}(s,a)$$
>     
>     **结论**：在 Model-free RL 中，我们需要估计 $q$ 值而不是 $v$ 值。
>     

### 2.2 MC Basic 算法流程

MC Basic 直接模拟策略迭代的过程：
1. **初始化**: 任意策略 $\pi_0$。
2. **第 $k$ 次迭代**:
    - **步骤 1：策略评估 (Policy Evaluation)**
        - 对每一个状态-动作对 $(s,a)$，利用当前策略 $\pi_k$ 生成大量轨迹 (Episodes)。
        - 计算这些轨迹的回报 $g(s,a)$。
        - 利用平均值近似 $q_{\pi_k}(s,a)$：
            $$q_{\pi_k}(s,a) \approx \frac{1}{N} \sum_{i=1}^{N} g^{(i)}(s,a)$$
    - **步骤 2：策略提升 (Policy Improvement)**
        - 直接对 $q$ 值取贪心 (Greedy)：$$a_k^*(s) = \arg\max_{a} q_{\pi_k}(s,a)$$$$\pi_{k+1}(a|s) = \begin{cases} 1 & a = a_k^*(s) \\ 0 & a \neq a_k^*(s) \end{cases}$$
> [!example] 例子：网格世界
> 
> 假设在某个状态 $s_1$，有两个动作 $a_1, a_2$。
> 
> - 生成 100 条从 $(s_1, a_1)$ 出发的轨迹，平均回报是 -5。
>     
> - 生成 100 条从 $(s_1, a_2)$ 出发的轨迹，平均回报是 +10。
>     
> - 策略提升：在 $s_1$ 状态，新的策略将 100% 概率选择 $a_2$。
>     

---

## 3. 提升数据效率：MC Exploring Starts

MC Basic 虽然理论上可行，但效率极低，甚至无法实际运行。

**痛点**：它需要在策略评估阶段，对**每一个** $(s,a)$ 对都单独生成大量轨迹。这不仅浪费，而且有些 $(s,a)$ 在当前策略下可能永远访问不到。

### 3.1 充分利用数据 (Efficient Use of Data)

一条长轨迹通常包含很多信息。例如轨迹：

$$s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_4} s_5 \xrightarrow{a_1} \dots$$

这条轨迹不仅可以用来估计 $q(s_1, a_2)$（初始访问），它经过的 $s_2 \xrightarrow{a_4} \dots$ 实际上也是一条从 $(s_2, a_4)$ 出发的有效轨迹。

> [!abstract] 访问策略 (Visit Strategies)
> 
> - **初始访问 (Initial-visit)**: 只利用轨迹的起始 $(s,a)$。MC Basic 使用此法，效率低。
>     
> - **首次访问 (First-visit)**: 在一条轨迹内，只计算某 $(s,a)$ 第一次出现时的回报。
>     
> - **每次访问 (Every-visit)**: 只要 $(s,a)$ 出现，就计算其后续回报。理论上偏差更小，数据利用率最高。
>     

### 3.2 策略更新的效率

MC Basic 等到所有 $(s,a)$ 都评估完才更新策略。我们可以**逐回合 (Episode-by-episode)** 更新。
即：生成一条轨迹 -> 评估这条轨迹涉及的 $(s,a)$ -> 立即更新相关策略。这也是**广义策略迭代 (Generalized Policy Iteration, GPI)** 的思想。

### 3.3 探索起始假设 (Exploring Starts)

为了确保所有 $(s,a)$ 都能被评估到（从而找到最优动作），我们需要保证**探索 (Exploration)**。
**Exploring Starts** 算法假设：
> 我们有能力强制智能体从任意指定的 $(s,a)$ 状态开始一局游戏。
> 这样可以保证所有状态动作对都有非零的概率被访问到。

**算法流程 (MC Exploring Starts)**:
1. **生成轨迹**: 随机选择 $(s_0, a_0)$（保证覆盖所有可能对），遵循当前策略 $\pi$ 生成轨迹。
2. **回报计算**: 从后向前计算回报 $G_t$：
    $$G_t = r_{t+1} + \gamma G_{t+1}$$
3. **更新 Q 值**: 对轨迹中出现的每个 $(s_t, a_t)$（使用首次访问法）：
    $$\text{Returns}(s_t, a_t) \leftarrow \text{Returns}(s_t, a_t) + G_t$$
    $$q(s_t, a_t) \leftarrow \text{Average}(\text{Returns}(s_t, a_t))$$
4. **更新策略**:
    $$\pi(s_t) \leftarrow \arg\max_a q(s_t, a)$$
    

---

## 4. 去除强假设：MC $\varepsilon$-Greedy

**痛点**: "Exploring Starts" 在现实中很难实现（例如自动驾驶汽车不能随机从马路中间且速度为 100km/h 的状态开始训练）。我们需要一种不需要强制指定起始状态，也能保证探索的机制。

### 4.1 软策略 (Soft Policies)

**定义**: 软策略是指对所有状态 $s$ 和所有动作 $a$，都有 $\pi(a|s) > 0$。
这样，只要轨迹足够长，智能体就有机会访问到所有状态动作对。

### 4.2 $\varepsilon$-贪心策略 ($\varepsilon$-Greedy Policies)

这是最常用的软策略。
$$\pi(a|s) = \begin{cases} 1 - \epsilon + \frac{\epsilon}{|\mathcal{A}(s)|}, & \text{if } a = a^* \text{ (Greedy action)} \\ \frac{\epsilon}{|\mathcal{A}(s)|}, & \text{if } a \neq a^* \text{ (Non-greedy action)} \end{cases}$$

其中 $|\mathcal{A}(s)|$ 是动作空间大小。

> [!math] 推导：为什么要加一项 $\frac{\epsilon}{|\mathcal{A}(s)|}$？
> 
> 简单的理解是：以 $1-\epsilon$ 的概率选贪心动作，以 $\epsilon$ 的概率**均匀随机**选所有动作（包括贪心动作）。
> 
> 贪心动作被选中的总概率为：
> 
> $$\underbrace{1 - \epsilon}_{\text{确定性选择}} + \underbrace{\epsilon \times \frac{1}{|\mathcal{A}(s)|}}_{\text{随机选中}} = 1 - \epsilon + \frac{\epsilon}{|\mathcal{A}(s)|}$$
> 
> 这保证了 $\pi(a|s) \ge \frac{\epsilon}{|\mathcal{A}(s)|} > 0$，即始终保持探索性。

### 4.3 嵌入到 MC 算法中

原本的策略提升步骤是求解：

$$\pi_{k+1} = \arg\max_{\pi \in \Pi} \sum_a \pi(a|s) q_{\pi_k}(s,a)$$

这得到的是确定性贪心策略。
现在，我们将搜索空间 $\Pi$ 限制为**所有 $\epsilon$-贪心策略的集合 $\Pi_{\epsilon}$**：

$$\pi_{k+1} = \arg\max_{\pi \in \Pi_{\epsilon}} \sum_a \pi(a|s) q_{\pi_k}(s,a)$$

答案非常直观：**基于当前 $q$ 值的 $\epsilon$-贪心策略**就是最优解。

> [!important] 探索与利用的权衡 (Exploration-Exploitation Trade-off)
> 
> - **$\epsilon \to 0$**: 趋近于贪心策略，**利用 (Exploitation)** 能力强，但可能陷入局部最优，未探索的状态无法被评估。
>     
> - **$\epsilon \to 1$**: 趋近于均匀随机策略，**探索 (Exploration)** 能力强，但策略性能差，很难利用已学到的知识。
>     
> 
> 即使 $\pi$ 只是在 $\Pi_{\epsilon}$ 集合中最优（而不是全局最优），当 $\epsilon$ 足够小时，它的表现也非常接近全局最优。

### 4.4 MC $\varepsilon$-Greedy 算法流程

1. **初始化**: 任意 $\epsilon$-贪心策略，任意 $Q(s,a)$。
2. **循环 (每条轨迹)**:
    - **生成轨迹**: 从任意点出发，遵循当前的 $\epsilon$-贪心策略 $\pi$ 生成一条轨迹。
    - **评估 (Evaluation)**: 使用轨迹的回报更新 $Q(s,a)$（通常用均值更新）。
    - **提升 (Improvement)**: 更新策略 $\pi$ 为基于新 $Q$ 的 $\epsilon$-贪心策略。
        $$a^* = \arg\max_a Q(s,a)$$
        $$\pi(a|s) \leftarrow \varepsilon\text{-greedy}(a^*)$$
> [!math] 增量式均值更新 (Incremental Mean Update)
> 
> 在代码实现中，我们不需要存储所有回报求和。可以用增量公式：
> 
> $$Q_{new}(s,a) = Q_{old}(s,a) + \frac{1}{N(s,a)} \left( G - Q_{old}(s,a) \right)$$
> 
> 这里的 $\frac{1}{N(s,a)}$ 可以看作学习率 $\alpha$。如果环境是非平稳的，可以固定 $\alpha$ 为一个小常数。

---

## 5. 总结

|**算法**|**核心特点**|**优点**|**缺点**|
|---|---|---|---|
|**MC Basic**|模仿策略迭代，评估完所有状态再更新|理论清晰|效率极低，需大量计算|
|**MC Exploring Starts**|利用每条轨迹更新；假设随机起点|数据效率高|**Exploring Starts** 假设在现实中难以满足|
|**MC $\varepsilon$-Greedy**|引入 $\varepsilon$ 软策略|**无需特殊起点假设**，保证持续探索|得到的策略不是绝对最优（含有随机性）|

> [!important] 核心逻辑链
> 
> 1. 我们想做 RL，但没有模型 $\to$ 使用 **MC 估计**。
>     
> 2. 没有模型无法做一步预测 $\to$ 必须估计 **$Q$ 值**而非 $V$ 值。
>     
> 3. 需要覆盖所有状态 $\to$ 引入 **Exploring Starts**。
>     
> 4. 无法控制起始状态 $\to$ 引入 **$\varepsilon$-Greedy** 软策略。
>