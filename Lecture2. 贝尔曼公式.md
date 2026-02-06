---
date: 2026-01-28
tags:
  - 强化学习
---
![[Chapter_2.pdf]]

# 第二章：Bellman Equation

> [!abstract] 章节概要
> 
> 本章是强化学习的基石。我们将从**回报 (Return)** 的概念出发，引出衡量策略好坏的核心指标——**状态价值 (State Value)**。
> 
> 最核心的内容是推导**贝尔曼方程 (Bellman Equation)**，它描述了状态价值之间的递归关系。我们将展示如何通过**矩阵形式**直接求解或迭代求解状态价值，并最终引入**动作价值 (Action Value, Q-value)**。

---

## 1. 核心概念回顾：回报 (Return)

在强化学习中，我们的目标是最大化长期的收益。为了数学化这个目标，我们定义了“回报”。

### 1.1 轨迹与回报定义

> [!Definition] Some notations
> 考虑一个单步的过程
> $$S_t \xrightarrow{A_t} R_{t+1}, S_{t+1} \xrightarrow{A_{t+1}} R_{t+2}, S_{t+2} \dots$$
> Note that $S_{t},A_{t},R_{t+1}$ are all **random variables（随机变量）**
> - 对于这些随机变量可以进行求期望等操作

This step is governed by the following probability distributions:
* $S_t \rightarrow A_t$ is governed by $\pi(A_t = a | S_t = s)$
* $S_t, A_t \rightarrow R_{t+1}$ is governed by $p(R_{t+1} = r | S_t = s, A_t = a)$
* $S_t, A_t \rightarrow S_{t+1}$ is governed by $p(S_{t+1} = s' | S_t = s, A_t = a)$

考虑多步的Trajectory可以得到：**折扣回报 (Discounted Return)** $G_t$ 定义为从 $t$ 时刻开始，未来所有奖励的折扣累加和：

> [!math] 公式定义
> 
> $$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$
> 
> - $R_{t+1}$: $t$ 时刻采取动作后，在 $t+1$ 时刻收到的即时奖励 (Immediate Reward)。
>     
> - $\gamma \in [0, 1)$: **折扣因子 (Discount Rate)**。它决定了我们看多远的未来。
>     

---

## 2. 状态价值 (State Value)

回报 $G_t$ 是一个**随机变量 (Random Variable)**，因为它取决于未来的动作选择（策略概率）和环境转移（状态转移概率）。为了评估一个状态“好不好”，我们需要求回报的期望。

> [!abstract] 定义：状态价值函数 (State-Value Function)
> 
> **状态价值** $v_{\pi}(s)$ 是在状态 $s$ 下，遵循策略 $\pi$ 能获得的期望回报：
> 
> $$v_{\pi}(s) = \mathbb{E}[G_t | S_t = s]$$

**关键点：**
- 它是关于**状态 $s$ 的函数**。
- 它依赖于策略 $\pi$。（也是一个**关于$\pi$的函数**，可以写成$v(s,\pi)$）策略不同，同一个状态的价值也不同。
- 代表一种价值，$v_{\pi}(s)$比较大时可以代表这个状态$s$是比较有价值的（从这个状态出发会得到更多return)
- **return 和 State-value的联系**：
	- **Return**针对**单个**Trajectory求return，**State-value**对**多个**Trajectory求return再取**平均**
	- 如果环境和策略都是确定性的 (Deterministic)，那么 $v_{\pi}(s)$ 就等于具体的 $G_t$ 值。
	    

---

## 3. 贝尔曼方程的推导 (Derivation of Bellman Equation)

这是本章的重中之重。我们要通过数学推导，找到 $v_{\pi}(s)$ 与后继状态价值 $v_{\pi}(s')$ 之间的递归关系。

### 3.1 第一步：回报的递归展开

利用 $G_t$ 的定义，我们可以将其写成递归形式：

$$\begin{aligned} G_t &= R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots \\ &= R_{t+1} + \gamma \underbrace{(R_{t+2} + \gamma R_{t+3} + \dots)}_{G_{t+1}} \\ &= R_{t+1} + \gamma G_{t+1} \end{aligned}$$

### 3.2 第二步：代入期望并利用线性性质

将上式代入状态价值的定义：

$$\begin{aligned} v_{\pi}(s) &= \mathbb{E}[G_t | S_t = s] \\ &= \mathbb{E}[R_{t+1} + \gamma G_{t+1} | S_t = s] \\ &= \underbrace{\mathbb{E}[R_{t+1} | S_t = s]}_{\text{第一项：即时奖励期望}} + \gamma \underbrace{\mathbb{E}[G_{t+1} | S_t = s]}_{\text{第二项：未来价值期望}} \end{aligned}$$

### 3.3 第三步：展开各项 (Detailed Expansion)

我们需要利用**全期望公式 (Law of Total Expectation)**，对动作 $A_t$ 和下一状态 $S_{t+1}$ 进行展开。

> [!math] 符号说明
> 
> - $\pi(a|s)$: 策略，在状态 $s$ 选择动作 $a$ 的概率。
>     
> - $p(r|s,a)$: 奖励概率，在 $s$ 做 $a$ 获得奖励 $r$ 的概率。
>     
> - $p(s'|s,a)$: 状态转移概率，在 $s$ 做 $a$ 转移到 $s'$ 的概率。
>     

#### A. 计算第一项：平均即时奖励

我们要计算 $R_{t+1}$ 的期望。首先对动作 $a$ 求和，再对奖励 $r$ 求和：

$$\begin{aligned} \mathbb{E}[R_{t+1} | S_t = s] &= \sum_{a} \pi(a|s) \mathbb{E}[R_{t+1} | S_t = s, A_t = a] \\ &= \sum_{a} \pi(a|s) \sum_{r} p(r|s,a) \cdot r \end{aligned}$$

#### B. 计算第二项：未来价值的期望

这是推导的难点。我们需要处理 $G_{t+1}$ 的期望。

$$\begin{aligned} \mathbb{E}[G_{t+1} | S_t = s] &= \sum_{s'} p(s'|s) \mathbb{E}[G_{t+1} | S_t = s, S_{t+1} = s'] \\ &\quad \text{其中 } p(s'|s) \text{ 包含了动作的选择，即 } \sum_a \pi(a|s)p(s'|s,a) \end{aligned}$$

利用**马尔可夫性质 (Markov Property)**：给定 $S_{t+1}=s'$，未来的回报 $G_{t+1}$ 与 $S_t$ 无关。因此：

$$\mathbb{E}[G_{t+1} | S_t = s, S_{t+1} = s'] = \mathbb{E}[G_{t+1} | S_{t+1} = s'] = v_{\pi}(s')$$

将 $v_{\pi}(s')$ 代回原式：

$$\begin{aligned} \mathbb{E}[G_{t+1} | S_t = s] &= \sum_{s'} \left( \sum_{a} \pi(a|s) p(s'|s,a) \right) v_{\pi}(s') \\ &= \sum_{a} \pi(a|s) \sum_{s'} p(s'|s,a) v_{\pi}(s') \end{aligned}$$

### 3.4 最终结果：贝尔曼方程

将 A 和 B 两部分合并，得到描述所有状态价值关系的方程组：

> [!important] 贝尔曼方程 (Element-wise Form)
> 
> $$v_{\pi}(s) = \sum_{a} \pi(a|s) \left[ \underbrace{\sum_{r} p(r|s,a) r}_{\text{平均即时奖励}} + \gamma \underbrace{\sum_{s'} p(s'|s,a) v_{\pi}(s')}_{\text{平均未来价值}} \right]$$

**直观理解**：一个状态的价值 = (在这个状态下采取所有动作的概率 $\times$ (该动作带来的即时奖励 + 折扣后的下一状态价值)) 的总和。这是一种 **Bootstrapping (自举)** 方法：用下一状态的估计值来更新当前状态的值。

虽然看起来是依赖其他的$s'$，但实际上会有一个方程组，最后可以联立解出

---

## 4. 贝尔曼方程的矩阵形式 (Matrix-Vector Form)

为了便于计算和理论分析，我们将上述方程写成矩阵形式。

假设状态空间 $\mathcal{S} = \{s_1, s_2, \dots, s_n\}$。

### 4.1 定义向量和矩阵

1. **状态价值向量** $v_{\pi}$:$$v_{\pi} = [v_{\pi}(s_1), v_{\pi}(s_2), \dots, v_{\pi}(s_n)]^T$$
2. **期望即时奖励向量** $r_{\pi}$:
    其中第 $i$ 个元素为 $r_{\pi}(s_i) = \sum_a \pi(a|s_i) \sum_r p(r|s_i, a) r$。$r_{\pi}=[r_{\pi}(s_1), r_{\pi}(s_2), \dots, r_{\pi}(s_n)]^T$
3. **State Transition Matrix状态转移矩阵** $P_{\pi}$:
    $P_{\pi} \in \mathbb{R}^{n \times n}$，其中第 $i$ 行第 $j$ 列的元素 $[P_{\pi}]_{ij} = p(s_j | s_i)$，表示在策略 $\pi$ 下从 $s_i$ 转移到 $s_j$ 的概率：
    $$[P_{\pi}]_{ij} = p(s_j | s_i)= \sum_a \pi(a|s_i) p(s_j | s_i, a)$$
### 4.2 矩阵方程

于是，贝尔曼方程组可以优雅地写为：

> [!math] 贝尔曼方程 (Matrix Form)
> 
> $$v_{\pi} = r_{\pi} + \gamma P_{\pi} v_{\pi}$$



### 贝尔曼方程的矩阵形式详细推导

> [!abstract] 推导目标
> 
> 我们已知针对单个状态 $s$ 的贝尔曼方程（标量形式），目标是将其转化为适用于所有状态 $s_1, \dots, s_n$ 的统一矩阵形式：$$v_{\pi} = r_{\pi} + \gamma P_{\pi} v_{\pi}$$

#### 第一步：回顾标量形式 (Scalar Form)

对于任意一个特定状态 $s_i \in \mathcal{S}$，其状态价值 $v_{\pi}(s_i)$ 由两部分组成：**即时奖励的期望** 和 **未来状态价值的折扣期望**。
根据贝尔曼方程的定义：

$$v_{\pi}(s_i) = \sum_{a} \pi(a|s_i) \sum_{r} p(r|s_i, a) r + \gamma \sum_{a} \pi(a|s_i) \sum_{s_j} p(s_j|s_i, a) v_{\pi}(s_j)$$
这个公式看起来很长，我们先定义两个辅助项来简化它。
##### 1. 定义期望即时奖励 $r_{\pi}(s_i)$
把公式的第一部分提取出来，定义为“在状态 $s_i$ 下遵循策略 $\pi$ 能获得的平均奖励”：
$$r_{\pi}(s_i) \triangleq \sum_{a} \pi(a|s_i) \sum_{r} p(r|s_i, a) r$$
##### 2. 定义状态转移概率 $p_{\pi}(s_j | s_i)$
把公式第二部分中的概率项合并。我们需要算出“在策略 $\pi$ 下，从 $s_i$ 转移到 $s_j$ 的总概率”（消掉了动作 $a$）：
$$p_{\pi}(s_j | s_i) \triangleq \sum_{a} \pi(a|s_i) p(s_j|s_i, a)$$
> [!math] 简化后的标量方程
> 
> 将上述两项代回原方程，我们得到一个更干净的形式：
> 
> $$v_{\pi}(s_i) = r_{\pi}(s_i) + \gamma \sum_{s_j \in \mathcal{S}} p_{\pi}(s_j | s_i) v_{\pi}(s_j)$$
> 
> _解读：状态 $i$ 的价值 = 状态 $i$ 的平均奖励 + $\gamma \times$ (从 $i$ 跳到所有可能的 $j$ 的加权平均价值)。_

#### 第二步：展开联立方程组 (System of Equations)

假设状态空间共有 $n$ 个状态：$\mathcal{S} = \{s_1, s_2, \dots, s_n\}$。
对于每一个状态，都存在一个上述的标量方程。我们将这 $n$ 个方程全部列出来：

$$\begin{cases} \text{状态 } s_1: \quad v_{\pi}(s_1) = r_{\pi}(s_1) + \gamma \left[ p_{\pi}(s_1|s_1)v_{\pi}(s_1) + p_{\pi}(s_2|s_1)v_{\pi}(s_2) + \dots + p_{\pi}(s_n|s_1)v_{\pi}(s_n) \right] \\ \text{状态 } s_2: \quad v_{\pi}(s_2) = r_{\pi}(s_2) + \gamma \left[ p_{\pi}(s_1|s_2)v_{\pi}(s_1) + p_{\pi}(s_2|s_2)v_{\pi}(s_2) + \dots + p_{\pi}(s_n|s_2)v_{\pi}(s_n) \right] \\ \vdots \\ \text{状态 } s_n: \quad v_{\pi}(s_n) = r_{\pi}(s_n) + \gamma \left[ p_{\pi}(s_1|s_n)v_{\pi}(s_1) + p_{\pi}(s_2|s_n)v_{\pi}(s_2) + \dots + p_{\pi}(s_n|s_n)v_{\pi}(s_n) \right] \end{cases}$$

> [!important] 观察结构
> 
> 注意方括号 `[...]` 中的内容。每一行实际上是 **概率向量** 与 **价值向量** 的 **内积 (Dot Product)**。
> 
> 例如第一行括弧内等于：$[p_{\pi}(s_1|s_1), \dots, p_{\pi}(s_n|s_1)] \cdot [v_{\pi}(s_1), \dots, v_{\pi}(s_n)]^T$。

#### 第三步：向量与矩阵化 (Matrixification)
现在我们将上述方程组转化为矩阵形式。
##### 1. 定义向量

我们将所有的 $v_{\pi}(s_i)$ 和 $r_{\pi}(s_i)$ 堆叠成列向量：
$$v_{\pi} = \begin{bmatrix} v_{\pi}(s_1) \\ v_{\pi}(s_2) \\ \vdots \\ v_{\pi}(s_n) \end{bmatrix}, \quad r_{\pi} = \begin{bmatrix} r_{\pi}(s_1) \\ r_{\pi}(s_2) \\ \vdots \\ r_{\pi}(s_n) \end{bmatrix}$$
##### 2. 定义状态转移矩阵 (State Transition Matrix) $P_{\pi}$

这是最关键的一步。我们将所有的 $p_{\pi}(s_j|s_i)$ 排列成一个 $n \times n$ 的矩阵。
**注意下标的顺序**：第 $i$ 行 第 $j$ 列表示从 $s_i$ 到 $s_j$ 的概率。
$$P_{\pi} = \begin{bmatrix} p_{\pi}(s_1|s_1) & p_{\pi}(s_2|s_1) & \dots & p_{\pi}(s_n|s_1) \\ p_{\pi}(s_1|s_2) & p_{\pi}(s_2|s_2) & \dots & p_{\pi}(s_n|s_2) \\ \vdots & \vdots & \ddots & \vdots \\ p_{\pi}(s_1|s_n) & p_{\pi}(s_2|s_n) & \dots & p_{\pi}(s_n|s_n) \end{bmatrix}$$

> [!note] 性质
> 
> 矩阵 $P_{\pi}$ 是一个 **随机矩阵 (Stochastic Matrix)**，其每一行的元素之和必须为 1（因为从状态 $s_i$ 出发，必定会转移到某个状态）。

##### 3. 组装公式

现在，让我们计算矩阵乘法 $P_{\pi} v_{\pi}$：

$$P_{\pi} v_{\pi} = \begin{bmatrix} p_{\pi}(s_1|s_1) & \dots & p_{\pi}(s_n|s_1) \\ \vdots & \ddots & \vdots \\ p_{\pi}(s_1|s_n) & \dots & p_{\pi}(s_n|s_n) \end{bmatrix} \begin{bmatrix} v_{\pi}(s_1) \\ \vdots \\ v_{\pi}(s_n) \end{bmatrix} = \begin{bmatrix} \sum_{j=1}^n p_{\pi}(s_j|s_1) v_{\pi}(s_j) \\ \vdots \\ \sum_{j=1}^n p_{\pi}(s_j|s_n) v_{\pi}(s_j) \end{bmatrix}$$

你会发现，**结果向量的第 $i$ 个元素，正是我们在前面提到方程组中方括号里的内容！**
因此，整个方程组可以写成：

$$\underbrace{\begin{bmatrix} v_{\pi}(s_1) \\ \vdots \\ v_{\pi}(s_n) \end{bmatrix}}_{v_{\pi}} = \underbrace{\begin{bmatrix} r_{\pi}(s_1) \\ \vdots \\ r_{\pi}(s_n) \end{bmatrix}}_{r_{\pi}} + \gamma \underbrace{\begin{bmatrix} \sum p(s_j|s_1)v(s_j) \\ \vdots \\ \sum p(s_j|s_n)v(s_j) \end{bmatrix}}_{P_{\pi} v_{\pi}}$$

即：
$$v_{\pi} = r_{\pi} + \gamma P_{\pi} v_{\pi}$$

## 5. 求解状态价值 (Solving State Values)

我们得到了一个关于 $v_{\pi}$ 的线性方程组。求解它被称为 **策略评估 (Policy Evaluation)**。

### 5.1 解析解 (Closed-form Solution)

通过移项求解：

$$\begin{aligned} v_{\pi} - \gamma P_{\pi} v_{\pi} &= r_{\pi} \\ (I - \gamma P_{\pi}) v_{\pi} &= r_{\pi} \\ v_{\pi} &= (I - \gamma P_{\pi})^{-1} r_{\pi} \end{aligned}$$

- **局限性**：矩阵求逆的计算复杂度是 $O(n^3)$，对于状态数量巨大的环境（如围棋），这种方法不可行。
    

### 5.2 迭代解法 (Iterative Solution)

我们可以将贝尔曼方程看作一个更新规则，从任意初始值 $v_0$ 开始迭代：

$$v_{k+1} = r_{\pi} + \gamma P_{\pi} v_k$$

> [!math] 收敛性证明 (Proof of Convergence)
> 
> 为什么这样做一定能收敛到真实值 $v_{\pi}$？
> 
> 定义误差 $\delta_k = v_k - v_{\pi}$。我们希望证明当 $k \to \infty$ 时，$\delta_k \to 0$。
> 
> 1. 写出迭代公式：$v_{k+1} = r_{\pi} + \gamma P_{\pi} v_k$
>     
> 2. 写出真实值满足的公式：$v_{\pi} = r_{\pi} + \gamma P_{\pi} v_{\pi}$
>     
> 3. 两式相减：
>     
>     $$v_{k+1} - v_{\pi} = \gamma P_{\pi} (v_k - v_{\pi})$$
>     
>     $$\delta_{k+1} = \gamma P_{\pi} \delta_k$$
>     
> 4. 递推展开：
>     
>     $$\delta_{k+1} = \gamma P_{\pi} (\gamma P_{\pi} \delta_{k-1}) = \gamma^2 P_{\pi}^2 \delta_{k-1} = \dots = \gamma^{k+1} P_{\pi}^{k+1} \delta_0$$
>     
> 5. 分析极限：
>     
>     由于 $P_{\pi}$ 是随机矩阵（行和为1），其幂 $P_{\pi}^{k+1}$ 的元素依然在 $[0,1]$ 之间。
>     
>     因为折扣因子 $\gamma < 1$，所以 $\lim_{k \to \infty} \gamma^{k+1} = 0$。
>     
>     因此，$\delta_{k+1} \to 0$，即 $v_{k+1} \to v_{\pi}$。
>     

---

## 6. 动作价值 (Action Value)

除了评估状态，我们更关心在某个状态下**采取特定动作**的好坏，这直接指导我们如何做决策。

> [!abstract] 定义：动作价值函数 (Q-function)
> 
> **动作价值** $q_{\pi}(s,a)$ 是在状态 $s$ 下，**强制采取动作 $a$**，之后遵循策略 $\pi$ 能获得的期望回报：
> 
> $$q_{\pi}(s,a) = \mathbb{E}[G_t | S_t = s, A_t = a]$$

### 6.1 状态价值与动作价值的关系

状态价值**State-Value**是动作价值**Action-Value**关于**策略概率的加权平均**：

> [!math] $v$ 与 $q$ 的关系
> 
> $$v_{\pi}(s) =  \sum_{a} \pi(a|s) \mathbb{E}[G_{t} | S_t=s,A_{t}=a] =\sum_{a} \pi(a|s) q_{\pi}(s,a)$$

**理解**：一个状态“值多少钱”，取决于在这个状态下你能做的所有动作“值多少钱”以及你选择这些动作的概率。

### 6.2 动作价值的贝尔曼方程

将 $q_{\pi}(s,a)$ 展开（类似于推导 $v_{\pi}(s)$ 的过程，但第一步动作 $a$ 是固定的）：

$$\begin{aligned} q_{\pi}(s,a) &= \mathbb{E}[R_{t+1} + \gamma G_{t+1} | S_t = s, A_t = a] \\ &= \sum_{r} p(r|s,a) r + \gamma \sum_{s'} p(s'|s,a) v_{\pi}(s') \end{aligned}$$
- 注：比较$v_{\pi}(s)=\sum_{a} \pi(a|s) q_{\pi}(s,a)$和$v_{\pi}(s) = \sum_{a} \pi(a|s) \left[ {\sum_{r} p(r|s,a) r} + \gamma \sum_{s'} p(s'|s,a) v_{\pi}(s') \right]$也可以得到
这里出现了一个有趣的结构：$q$ 的定义里用到了 $v$。如果我们把 $v_{\pi}(s')$ 再次展开成 $q$，就得到了 $q$ 自己的递归形式，但这通常在后续章节（如 SARSA, Q-learning）中讨论。

> [!Note] 公式的意义
> $$q_{\pi}(s,a) = \mathbb{E}[G_t | S_t = s, A_t = a]$$
> - 如果对于一个状态，知道了所有的Action-Value，那么求平均就能得到这个状态的State-Value
> $$q_{\pi}(s,a)= \sum_{r} p(r|s,a) r + \gamma \sum_{s'} p(s'|s,a) v_{\pi}(s')$$
> - 如果知道所有状态的State-Value，那么也能求出来所有的Action-Value



---

## 7. 总结 (Summary)

> [!check] 核心知识点回顾
> 
> 1. **回报 (Return)**: 评价轨迹收益的总和，引入 $\gamma$ 保证收敛。
>     
> 2. **状态价值 (State Value)**: 回报的期望，$v_{\pi}(s)$。
>     
> 3. **贝尔曼方程 (Bellman Equation)**:
>     
>     - 元素形式: $v_{\pi}(s) = \sum_a \pi(a|s) [ \text{Reward} + \gamma \sum_{s'} p(s'|s,a) v_{\pi}(s') ]$
>         
>     - 矩阵形式: $v = r + \gamma P v$
>         
> 4. **求解方法**:
>     
>     - 解析解: $v = (I - \gamma P)^{-1} r$
>         
>     - 迭代解: $v \leftarrow r + \gamma P v$ (利用了 $\gamma < 1$ 的收缩性质)。
>         
> 5. **动作价值 (Action Value)**: $q_{\pi}(s,a)$，它是连接状态与策略选择的桥梁。
>     

