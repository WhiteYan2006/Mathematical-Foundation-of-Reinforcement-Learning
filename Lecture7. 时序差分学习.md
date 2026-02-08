---
date: 2026-02-08
tags:
  - 强化学习
---

# 第七章：时序差分学习 (Temporal-Difference Learning)

> [!abstract] 章节概要
> 本章介绍强化学习中最著名的算法类别——**时序差分 (TD)** 学习。
>
> **核心定位**：
> - TD 学习是继蒙特卡洛 (MC) 之后的第二种 **无模型 (Model-free)** 方法
> - 它结合了 MC 的采样思想（从经验中学习）和 DP 的 **自举 (Bootstrapping)** 思想
> - 本章将展示如何利用 **随机近似 (Stochastic Approximation)** 方法（Robbins-Monro 算法）来求解贝尔曼方程
>
> **本章算法总览**：
>
> | 算法 | 估计对象 | 求解方程 | 策略类型 |
> |:---:|:---:|:---:|:---:|
> | TD(0) | 状态价值 $v_\pi(s)$ | 贝尔曼期望方程 | On-policy |
> | Sarsa | 动作价值 $q_\pi(s,a)$ | 贝尔曼期望方程 | On-policy |
> | Expected Sarsa | 动作价值 $q_\pi(s,a)$ | 贝尔曼期望方程 | On-policy |
> | n-step Sarsa | 动作价值 $q_\pi(s,a)$ | 贝尔曼期望方程 | On-policy |
> | Q-learning | **最优**动作价值 $q_*(s,a)$ | **贝尔曼最优方程** | **Off-policy** |

---

## 1. 引导性示例：从随机近似到 TD 算法

在正式介绍 TD 之前，我们需要复习 **Robbins-Monro (RM) 算法**，因为 **TD 算法本质上就是求解贝尔曼方程的 RM 算法**。

### 1.1 示例一：简单均值估计问题

**问题**：计算随机变量 $X$ 的期望 $w = \mathbb{E}[X]$。

> [!math] 推导过程
> **Step 1: 构造求根问题**
>
> 定义函数：
> $$g(w) = w - \mathbb{E}[X] = 0$$
>
> **Step 2: 定义带噪观测**
>
> 由于我们只能观测到样本 $x$（而非期望），定义观测函数：
> $$\tilde{g}(w, \eta) = w - x = (w - \mathbb{E}[X]) + (\mathbb{E}[X] - x) \doteq g(w) + \eta$$
>
> 其中 $\eta = \mathbb{E}[X] - x$ 是随机噪声。
>
> **Step 3: 应用 RM 算法**
>
> $$w_{k+1} = w_k - \alpha_k \tilde{g}(w_k, \eta_k) = w_k - \alpha_k (w_k - x_k)$$
>
> 这实际上就是**增量式求平均**的公式。

### 1.2 示例二：函数均值估计

**问题**：计算 $w = \mathbb{E}[v(X)]$，其中 $v(\cdot)$ 是已知函数。

> [!math] 推导过程
> **Step 1: 构造求根问题**
> $$g(w) = w - \mathbb{E}[v(X)] = 0$$
>
> **Step 2: 定义带噪观测**
> $$\tilde{g}(w, \eta) = w - v(x) = g(w) + \eta$$
>
> **Step 3: RM 算法**
> $$w_{k+1} = w_k - \alpha_k [w_k - v(x_k)]$$

### 1.3 示例三：TD 的雏形

**问题**：计算 $w = \mathbb{E}[R + \gamma v(X)]$，其中 $R, X$ 是随机变量，$\gamma$ 是常数，$v(\cdot)$ 是函数。

> [!math] 核心推导：从 RM 到 TD 的形式
> **Step 1: 构造求根问题**
> $$g(w) = w - \mathbb{E}[R + \gamma v(X)] = 0$$
>
> **Step 2: 定义带噪观测**
>
> 给定样本 $r$ 和 $x$：
> $$\tilde{g}(w, \eta) = w - [r + \gamma v(x)] = g(w) + \eta$$
>
> **Step 3: RM 算法**
> $$w_{k+1} = w_k - \alpha_k (w_k - [r_k + \gamma v(x_k)])$$
>
> 改写为：
> $$\boxed{w_{k+1} = w_k + \alpha_k \left[\underbrace{r_k + \gamma v(x_k)}_{\text{Target}} - w_k\right]}$$
>
> 这个形式 $w \leftarrow w + \alpha [\text{Target} - w]$ 正是 **TD 学习的基础结构**！

> [!tip] 关键洞察
> 上述三个示例逐步增加复杂度，但都可以用 RM 算法求解。TD 算法正是第三个示例的具体应用。

---

## 2. 状态价值的 TD 学习 (TD Learning of State Values)

### 2.1 算法描述

> [!note] TD 学习的双重含义
> - **广义**：TD 学习泛指一大类强化学习算法（本章所有算法都属于 TD 学习）
> - **狭义**：特指用于估计状态价值的 TD(0) 算法

给定策略 $\pi$，我们希望估计其**状态价值函数** $v_\pi(s)$。

**所需数据**：序列 $(s_0, r_1, s_1, \ldots, s_t, r_{t+1}, s_{t+1}, \ldots)$ 或 $\{(s_t, r_{t+1}, s_{t+1})\}_t$

> [!important] TD(0) 算法
> $$v_{t+1}(s_t) = v_t(s_t) - \alpha_t(s_t) \left[v_t(s_t) - (r_{t+1} + \gamma v_t(s_{t+1}))\right]$$
> $$v_{t+1}(s) = v_t(s), \quad \forall s \neq s_t$$
>
> 其中 $t = 0, 1, 2, \ldots$，$v_t(s_t)$ 是 $v_\pi(s_t)$ 在时刻 $t$ 的估计值，$\alpha_t(s_t)$ 是学习率。
>
> **注意**：第二个方程常被省略，但它在数学上是完整算法不可或缺的部分。

### 2.2 关键概念：TD Target 与 TD Error

将 TD 算法改写为更直观的形式：

$$\underbrace{v_{t+1}(s_t)}_{\text{new estimate}} = \underbrace{v_t(s_t)}_{\text{current estimate}} - \alpha_t(s_t) \underbrace{\left[v_t(s_t) - \overbrace{(r_{t+1} + \gamma v_t(s_{t+1}))}^{\text{TD target } \bar{v}_t}\right]}_{\text{TD error } \delta_t}$$

> [!important] 关键术语
> **TD 目标 (TD Target)**：
> $$\bar{v}_t \doteq r_{t+1} + \gamma v_t(s_{t+1})$$
>
> **TD 误差 (TD Error)**：
> $$\delta_t \doteq v_t(s_t) - \bar{v}_t = v_t(s_t) - [r_{t+1} + \gamma v_t(s_{t+1})]$$

#### 为什么叫 "TD Target"？

> [!math] 数学证明：$v(s_t)$ 被驱动向 $\bar{v}_t$ 靠近
> 从 TD 更新公式出发：
> $$v_{t+1}(s_t) = v_t(s_t) - \alpha_t(s_t)[v_t(s_t) - \bar{v}_t]$$
>
> 两边减去 $\bar{v}_t$：
> $$v_{t+1}(s_t) - \bar{v}_t = v_t(s_t) - \bar{v}_t - \alpha_t(s_t)[v_t(s_t) - \bar{v}_t]$$
> $$= [1 - \alpha_t(s_t)][v_t(s_t) - \bar{v}_t]$$
>
> 取绝对值：
> $$|v_{t+1}(s_t) - \bar{v}_t| = |1 - \alpha_t(s_t)| \cdot |v_t(s_t) - \bar{v}_t|$$
>
> 由于 $\alpha_t(s_t)$ 是小正数，$0 < 1 - \alpha_t(s_t) < 1$，因此：
> $$\boxed{|v_{t+1}(s_t) - \bar{v}_t| < |v_t(s_t) - \bar{v}_t|}$$
>
> **结论**：新估计值 $v_{t+1}(s_t)$ 比旧估计值 $v_t(s_t)$ 更接近 TD target $\bar{v}_t$。

#### TD Error 的含义

> [!math] TD Error 的期望特性
> 定义真实 TD error：
> $$\delta_{\pi,t} \doteq v_\pi(s_t) - [r_{t+1} + \gamma v_\pi(s_{t+1})]$$
>
> 计算其条件期望：
> $$\mathbb{E}[\delta_{\pi,t} | S_t = s_t] = v_\pi(s_t) - \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t = s_t]$$
>
> 由贝尔曼期望方程 $v_\pi(s) = \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t = s]$：
> $$\mathbb{E}[\delta_{\pi,t} | S_t = s_t] = 0$$

> [!tip] TD Error 的解释
> - **时序差分**：TD error 反映了相邻两个时间步之间估计值的差异
> - **创新 (Innovation)**：如果 $v_t = v_\pi$，则 TD error 的期望为零；反之，非零的 TD error 表示 $v_t \neq v_\pi$
> - **新信息**：TD error 可解释为从经验样本 $(s_t, r_{t+1}, s_{t+1})$ 中获得的新信息

### 2.3 数学原理：求解贝尔曼期望方程

**TD 算法在数学上做什么？答：通过随机近似求解贝尔曼期望方程。**

> [!math] 完整推导：从贝尔曼方程到 TD 算法
> **Step 1: 状态价值的定义**
> $$v_\pi(s) = \mathbb{E}[R_{t+1} + \gamma G_{t+1} | S_t = s], \quad s \in \mathcal{S}$$
>
> **Step 2: 贝尔曼期望方程**
>
> 由于 $\mathbb{E}[G_{t+1}|S_t = s] = \sum_a \pi(a|s) \sum_{s'} p(s'|s,a) v_\pi(s') = \mathbb{E}[v_\pi(S_{t+1})|S_t = s]$，得：
> $$\boxed{v_\pi(s) = \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t = s], \quad s \in \mathcal{S}}$$
>
> 这就是**贝尔曼期望方程 (Bellman Expectation Equation)**。
>
> **Step 3: 构造求根问题**
>
> 对状态 $s_t$，定义：
> $$g(v_\pi(s_t)) \doteq v_\pi(s_t) - \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t = s_t]$$
>
> 贝尔曼方程等价于 $g(v_\pi(s_t)) = 0$。
>
> **Step 4: 定义带噪观测**
>
> 给定样本 $r_{t+1}$ 和 $s_{t+1}$：
> $$\tilde{g}(v_\pi(s_t)) = v_\pi(s_t) - [r_{t+1} + \gamma v_\pi(s_{t+1})]$$
>
> 可验证：
> $$\tilde{g}(v_\pi(s_t)) = \underbrace{v_\pi(s_t) - \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1})|s_t]}_{g(v_\pi(s_t))} + \underbrace{\mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1})|s_t] - [r_{t+1} + \gamma v_\pi(s_{t+1})]}_{\eta}$$
>
> **Step 5: RM 算法**
> $$v_{k+1}(s) = v_k(s) - \alpha_k \tilde{g}(v_k(s)) = v_k(s) - \alpha_k \left(v_k(s) - [r_k + \gamma v_\pi(s'_k)]\right)$$
>
> **Step 6: 关键修改**
>
> 上述 RM 算法有两个假设需要修改：
> 1. 经验数据从 $\{(s, r, s')\}$ 改为**时序数据** $\{(s_t, r_{t+1}, s_{t+1})\}_t$
> 2. **自举**：由于 $v_\pi(s_{t+1})$ 未知，用当前估计值 $v_t(s_{t+1})$ 代替
>
> 最终得到 TD(0) 算法：
> $$\boxed{v_{t+1}(s_t) = v_t(s_t) - \alpha_t(s_t)\left(v_t(s_t) - [r_{t+1} + \gamma v_t(s_{t+1})]\right)}$$

### 2.4 收敛性定理

> [!theorem] 定理 7.1：TD 学习的收敛性
> 给定策略 $\pi$，由 TD 算法 (7.1)，当 $t \to \infty$ 时，$v_t(s)$ **几乎必然 (almost surely)** 收敛到 $v_\pi(s)$，$\forall s \in \mathcal{S}$，如果满足：
> $$\sum_t \alpha_t(s) = \infty \quad \text{且} \quad \sum_t \alpha_t^2(s) < \infty, \quad \forall s \in \mathcal{S}$$

> [!note] 关于学习率的说明
> - **条件解释**：$\sum_t \alpha_t(s) = \infty$ 要求每个状态 $s$ 被访问无限次（或足够多次）
> - **实践中**：学习率 $\alpha$ 常设为**小常数**。此时 $\sum_t \alpha_t^2(s) < \infty$ 不再成立，但可证明算法在**期望意义**下收敛
> - **收敛证明**：基于第六章的随机近似定理 (Theorem 6.3)

### 2.5 TD 与 MC 的对比

> [!comparison] TD Learning vs Monte Carlo Learning
>
> | 特性 | TD Learning | Monte Carlo |
> |:---|:---|:---|
> | **更新时机** | **在线 (Online)**：每一步都可更新 | **离线 (Offline)**：必须等到回合结束 |
> | **任务类型** | 适用于**持续性任务**和分段任务 | 仅适用于**分段任务** (Episodic) |
> | **自举** | **是**：用估计值更新估计值 | **否**：用真实回报更新估计值 |
> | **方差** | **低**：仅依赖一步随机性 $(R_{t+1}, S_{t+1})$ | **高**：依赖整个轨迹，累积多步方差 |
> | **偏差** | **有**：初始阶段因自举而有偏 | **无偏**：无偏估计（采样足够时） |
> | **初始猜测** | **需要**：自举依赖初始估计 | **不需要**：直接从回报计算 |

> [!example] 方差对比的直观理解
> - **TD (Sarsa)**：估计 $q_\pi(s_t, a_t)$ 只需三个随机变量的样本：$R_{t+1}, S_{t+1}, A_{t+1}$
> - **MC**：估计 $q_\pi(s_t, a_t)$ 需要整条轨迹 $R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots$
> - 假设每个回合长度为 $L$，每个状态有 $|\mathcal{A}|$ 个动作，则可能的轨迹数为 $|\mathcal{A}|^L$

---

## 3. 动作价值的 TD 学习：Sarsa 算法

### 3.1 为什么需要估计动作价值？

- TD(0) 只能估计**状态价值** $v_\pi(s)$
- 为了进行**控制 (Control)**（寻找最优策略），需要估计**动作价值** $q_\pi(s, a)$
- 因为策略提升需要 $\pi'(s) = \arg\max_a q_\pi(s, a)$

### 3.2 Sarsa 算法

**算法名称来源**：每次迭代需要五元组 $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$，即 **S**tate-**A**ction-**R**eward-**S**tate-**A**ction。

**所需数据**：$\{(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})\}_t$

> [!important] Sarsa 更新公式
> $$q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - \alpha_t(s_t, a_t) \left[q_t(s_t, a_t) - (r_{t+1} + \gamma q_t(s_{t+1}, a_{t+1}))\right]$$
> $$q_{t+1}(s, a) = q_t(s, a), \quad \forall (s, a) \neq (s_t, a_t)$$
>
> 其中 TD Target 为 $\bar{q}_t = r_{t+1} + \gamma q_t(s_{t+1}, a_{t+1})$

### 3.3 数学原理

> [!math] Sarsa 求解的贝尔曼方程
> Sarsa 是求解以下**动作价值形式的贝尔曼期望方程**的随机近似算法：
> $$\boxed{q_\pi(s, a) = \mathbb{E}[R + \gamma q_\pi(S', A') | s, a], \quad \forall s, a}$$
>
> **证明**：动作价值的贝尔曼方程为
> $$q_\pi(s,a) = \sum_r rp(r|s,a) + \gamma \sum_{s'} \sum_{a'} q_\pi(s',a') p(s'|s,a) \pi(a'|s')$$
>
> 由于 $p(s',a'|s,a) = p(s'|s,a)p(a'|s') \doteq p(s'|s,a)\pi(a'|s')$（条件独立性），上式等价于：
> $$q_\pi(s,a) = \mathbb{E}[R + \gamma q_\pi(S', A') | s, a]$$

### 3.4 收敛性定理

> [!theorem] 定理 7.2：Sarsa 的收敛性
> 给定策略 $\pi$，由 Sarsa 算法，当 $t \to \infty$ 时，$q_t(s, a)$ **几乎必然**收敛到 $q_\pi(s, a)$，$\forall (s, a)$，如果满足：
> $$\sum_t \alpha_t(s, a) = \infty \quad \text{且} \quad \sum_t \alpha_t^2(s, a) < \infty, \quad \forall (s, a)$$

### 3.5 Sarsa 的策略搜索：广义策略迭代

Sarsa 本身只能**评估**给定策略。要**寻找最优策略**，需要结合策略提升步骤。

> [!algorithm] 算法 7.1：Sarsa 最优策略搜索
> **初始化**：$\alpha_t(s,a) = \alpha > 0$，$\epsilon \in (0,1)$，初始 $q_0(s,a)$，$\epsilon$-greedy 策略 $\pi_0$
>
> **目标**：从初始状态 $s_0$ 找到到达目标状态的最优路径
>
> **For** each episode **do**：
> > 在 $s_0$ 处根据 $\pi_0(s_0)$ 生成 $a_0$
> >
> > **If** $s_t$ 不是目标状态 $(t = 0, 1, 2, \ldots)$ **do**：
> > > 收集经验 $(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$：根据 $\pi_t(s_t)$ 执行 $a_t$，观测 $r_{t+1}, s_{t+1}$，根据 $\pi_t(s_{t+1})$ 选择 $a_{t+1}$
> > >
> > > **更新 q 值**：
> > > $$q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - \alpha_t(s_t, a_t)[q_t(s_t, a_t) - (r_{t+1} + \gamma q_t(s_{t+1}, a_{t+1}))]$$
> > >
> > > **更新策略** ($\epsilon$-greedy)：
> > > $$\pi_{t+1}(a|s_t) = \begin{cases} 1 - \frac{\epsilon}{|\mathcal{A}(s_t)|}(|\mathcal{A}(s_t)| - 1) & \text{if } a = \arg\max_a q_{t+1}(s_t, a) \\ \frac{\epsilon}{|\mathcal{A}(s_t)|} & \text{otherwise} \end{cases}$$
> > >
> > > $s_t \leftarrow s_{t+1}$，$a_t \leftarrow a_{t+1}$

> [!tip] 算法说明
> - 基于**广义策略迭代 (GPI)** 思想：每更新一次 q 值就立即更新策略
> - 使用 **$\epsilon$-greedy** 而非 greedy 策略，确保**探索 (Exploration)**
> - Sarsa 是 **On-policy** 算法：行为策略与目标策略相同

---

## 4. 期望 Sarsa (Expected Sarsa)

### 4.1 算法描述

Expected Sarsa 是 Sarsa 的变体，通过对下一动作取期望来**减小方差**。

> [!important] Expected Sarsa 更新公式
> $$q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - \alpha_t(s_t, a_t) \left[q_t(s_t, a_t) - (r_{t+1} + \gamma \mathbb{E}[q_t(s_{t+1}, A)])\right]$$
>
> 其中：
> $$\mathbb{E}[q_t(s_{t+1}, A)] = \sum_a \pi_t(a|s_{t+1}) q_t(s_{t+1}, a) \doteq v_t(s_{t+1})$$
>
> **TD Target 对比**：
> - Sarsa：$\bar{q}_t = r_{t+1} + \gamma q_t(s_{t+1}, a_{t+1})$
> - Expected Sarsa：$\bar{q}_t = r_{t+1} + \gamma \mathbb{E}[q_t(s_{t+1}, A)]$

### 4.2 数学原理

> [!math] Expected Sarsa 求解的贝尔曼方程
> Expected Sarsa 是求解以下方程的随机近似算法：
> $$q_\pi(s, a) = \mathbb{E}\left[R_{t+1} + \gamma \mathbb{E}_{A_{t+1} \sim \pi(S_{t+1})}[q_\pi(S_{t+1}, A_{t+1})] \Big| S_t = s, A_t = a\right]$$
>
> 由于 $\mathbb{E}[q_\pi(S_{t+1}, A_{t+1})|S_{t+1}] = \sum_{A'} q_\pi(S_{t+1}, A') \pi(A'|S_{t+1}) = v_\pi(S_{t+1})$，上式等价于：
> $$q_\pi(s, a) = \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t = s, A_t = a]$$
>
> 这就是标准的贝尔曼期望方程。

### 4.3 特点

> [!comparison] Sarsa vs Expected Sarsa
>
> | 特性 | Sarsa | Expected Sarsa |
> |:---|:---|:---|
> | **所需数据** | $(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$ | $(s_t, a_t, r_{t+1}, s_{t+1})$ |
> | **计算量** | 低 | 高（需对所有动作求和） |
> | **方差** | 高（$a_{t+1}$ 带来随机性） | **低**（消除 $a_{t+1}$ 的随机性） |
> | **收敛速度** | 较慢 | 通常更快 |

---

## 5. n-步 Sarsa (n-step Sarsa)

### 5.1 动机：统一 Sarsa 与 MC

n-step Sarsa 架起了 TD(0) 和 MC 之间的桥梁。

回顾动作价值的定义：
$$q_\pi(s, a) = \mathbb{E}[G_t | S_t = s, A_t = a]$$

其中折扣回报 $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots$

> [!math] $G_t$ 的不同分解形式
> $$\begin{aligned}
> \text{Sarsa} \quad &\longleftarrow \quad G_t^{(1)} = R_{t+1} + \gamma q_\pi(S_{t+1}, A_{t+1}) \\
> & \quad\quad\quad\quad G_t^{(2)} = R_{t+1} + \gamma R_{t+2} + \gamma^2 q_\pi(S_{t+2}, A_{t+2}) \\
> & \quad\quad\quad\quad \vdots \\
> \text{n-step Sarsa} \quad &\longleftarrow \quad G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^n q_\pi(S_{t+n}, A_{t+n}) \\
> & \quad\quad\quad\quad \vdots \\
> \text{MC} \quad &\longleftarrow \quad G_t^{(\infty)} = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots
> \end{aligned}$$
>
> **注意**：$G_t = G_t^{(1)} = G_t^{(2)} = \cdots = G_t^{(n)} = G_t^{(\infty)}$，上标仅表示不同的分解结构。

### 5.2 算法描述

> [!important] n-step Sarsa 更新公式
> $$q_{t+n}(s_t, a_t) = q_{t+n-1}(s_t, a_t) - \alpha_{t+n-1}(s_t, a_t) \left[q_{t+n-1}(s_t, a_t) - (r_{t+1} + \gamma r_{t+2} + \cdots + \gamma^n q_{t+n-1}(s_{t+n}, a_{t+n}))\right]$$
>
> **所需数据**：$(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1}, \ldots, r_{t+n}, s_{t+n}, a_{t+n})$
>
> **注意**：由于需要 $(r_{t+n}, s_{t+n}, a_{t+n})$，更新 $(s_t, a_t)$ 的 q 值必须等到时刻 $t+n$。

### 5.3 求解的方程

> [!math] n-step Sarsa 求解的贝尔曼方程
> - 当 $n = 1$（Sarsa）：
>   $$q_\pi(s, a) = \mathbb{E}[G_t^{(1)}|s, a] = \mathbb{E}[R_{t+1} + \gamma q_\pi(S_{t+1}, A_{t+1}) | s, a]$$
>
> - 当 $n = \infty$（MC）：
>   $$q_\pi(s, a) = \mathbb{E}[G_t^{(\infty)}|s, a] = \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \ldots | s, a]$$
>
> - 一般 $n$：
>   $$q_\pi(s, a) = \mathbb{E}[G_t^{(n)}|s, a] = \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^n q_\pi(S_{t+n}, A_{t+n}) | s, a]$$

### 5.4 偏差-方差权衡 (Bias-Variance Trade-off)

> [!comparison] n 的选择
>
> | $n$ 值 | 接近算法 | 偏差 | 方差 | 特点 |
> |:---:|:---:|:---:|:---:|:---|
> | 小 | Sarsa | 大（受自举影响） | **小** | 对初始猜测敏感 |
> | 大 | MC | **小**（接近真实回报） | 大 | 需要更多样本 |
>
> **总结**：n-step Sarsa 的性能介于 Sarsa 和 MC 之间，$n$ 的最优选择取决于具体问题。

---

## 6. Q-learning：最优动作价值学习

### 6.1 动机

- **Sarsa** 只能估计**给定策略**的动作价值 $q_\pi(s,a)$，需要结合策略提升才能找最优策略
- **Q-learning** 可以**直接估计最优动作价值** $q_*(s,a)$，从而直接获得最优策略

### 6.2 算法描述

> [!important] Q-learning 更新公式
> $$q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - \alpha_t(s_t, a_t) \left[q_t(s_t, a_t) - \left(r_{t+1} + \gamma \max_{a \in \mathcal{A}} q_t(s_{t+1}, a)\right)\right]$$
> $$q_{t+1}(s, a) = q_t(s, a), \quad \forall (s, a) \neq (s_t, a_t)$$
>
> **TD Target 对比**：
> - Sarsa：$\bar{q}_t = r_{t+1} + \gamma q_t(s_{t+1}, a_{t+1})$ — 使用实际采样的 $a_{t+1}$
> - Q-learning：$\bar{q}_t = r_{t+1} + \gamma \max_a q_t(s_{t+1}, a)$ — 使用最优动作
>
> **所需数据**：$(s_t, a_t, r_{t+1}, s_{t+1})$ — 注意**不需要** $a_{t+1}$！

### 6.3 数学原理：求解贝尔曼最优方程

> [!math] Q-learning 求解的贝尔曼最优方程
> Q-learning 是求解以下**贝尔曼最优方程 (Bellman Optimality Equation)** 的随机近似算法：
> $$\boxed{q(s, a) = \mathbb{E}\left[R_{t+1} + \gamma \max_a q(S_{t+1}, a) \Big| S_t = s, A_t = a\right], \quad \forall s, a}$$
>
> **证明**：
> 根据期望的定义，上式可写为：
> $$q(s, a) = \sum_r p(r|s,a)r + \gamma \sum_{s'} p(s'|s,a) \max_{a \in \mathcal{A}(s')} q(s', a)$$
>
> 对两边取 $\max_a$：
> $$\max_{a \in \mathcal{A}(s)} q(s, a) = \max_{a \in \mathcal{A}(s)} \left[\sum_r p(r|s,a)r + \gamma \sum_{s'} p(s'|s,a) \max_{a' \in \mathcal{A}(s')} q(s', a')\right]$$
>
> 令 $v(s) \doteq \max_{a \in \mathcal{A}(s)} q(s, a)$，则：
> $$v(s) = \max_{a \in \mathcal{A}(s)} \left[\sum_r p(r|s,a)r + \gamma \sum_{s'} p(s'|s,a) v(s')\right]$$
>
> 这正是**第三章**介绍的贝尔曼最优方程（状态价值形式）。

### 6.4 On-policy vs Off-policy

> [!important] 核心概念：行为策略与目标策略
> 在强化学习中，存在两种策略：
> - **行为策略 (Behavior Policy)** $\pi_b$：用于**生成经验数据**的策略
> - **目标策略 (Target Policy)** $\pi_T$：算法正在**学习/优化**的策略
>
> **定义**：
> - **On-policy**：$\pi_b = \pi_T$（行为策略和目标策略相同）
> - **Off-policy**：$\pi_b \neq \pi_T$（行为策略和目标策略不同）

#### 为什么 Sarsa 是 On-policy？

> [!math] Sarsa 的 On-policy 性质分析
> **从求解的方程角度**：Sarsa 求解给定策略 $\pi$ 的贝尔曼方程 $q_\pi(s,a) = \mathbb{E}[R + \gamma q_\pi(S', A')|s, a]$
>
> **从所需样本角度**：Sarsa 需要 $(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$，其中：
> $$s_t \xrightarrow{\pi_b} a_t \xrightarrow{\text{model}} r_{t+1}, s_{t+1} \xrightarrow{\pi_b} a_{t+1}$$
>
> - $a_t$ 由行为策略 $\pi_b$ 在 $s_t$ 处生成
> - $a_{t+1}$ 由行为策略 $\pi_b$ 在 $s_{t+1}$ 处生成
> - 目标策略 $\pi_T$ 是根据估计的 q 值不断更新的策略
> - 由于 $\pi_T$ 的评估依赖于 $a_{t+1}$，而 $a_{t+1}$ 由 $\pi_b$ 生成，因此 **$\pi_b$ 必须等于 $\pi_T$**

#### 为什么 Q-learning 是 Off-policy？

> [!math] Q-learning 的 Off-policy 性质分析
> **从求解的方程角度**：Q-learning 求解**贝尔曼最优方程** $q(s,a) = \mathbb{E}[R + \gamma \max_a q(S', a)|s, a]$
>
> 这个方程**不依赖于任何特定策略**——它直接求解最优动作价值！
>
> **从所需样本角度**：Q-learning 只需要 $(s_t, a_t, r_{t+1}, s_{t+1})$，其中：
> $$s_t \xrightarrow{\pi_b} a_t \xrightarrow{\text{model}} r_{t+1}, s_{t+1}$$
>
> - **关键**：生成 $(r_{t+1}, s_{t+1})$ 的过程**不依赖任何策略**（由环境/模型决定）
> - 行为策略 $\pi_b$ 只用于在 $s_t$ 处选择 $a_t$
> - 目标策略 $\pi_T$ 是根据 $q$ 值的 greedy 策略
> - 由于 TD target $r_{t+1} + \gamma \max_a q(s_{t+1}, a)$ **不依赖** $\pi_b$
> - 因此 **$\pi_b$ 可以是任意策略**，包括人类操作、随机策略等

> [!tip] Off-policy 的优势
> 1. 可以从**其他策略生成的经验**中学习最优策略
> 2. 行为策略可以选择**高度探索性**的策略（如均匀随机），提高样本效率
> 3. 可以复用历史数据（经验回放 Experience Replay）

### 6.5 Q-learning 的两种实现

#### On-policy 版本

> [!algorithm] 算法 7.2：Q-learning (On-policy 版本)
> **初始化**：$q_0(s,a)$，$\epsilon$-greedy 策略 $\pi_0$，$\alpha_t(s,a) = \alpha > 0$
>
> **目标**：从初始状态 $s_0$ 找到到达目标状态的最优路径
>
> **For** each episode **do**：
> > **If** $s_t$ 不是目标状态 **do**：
> > > 收集经验 $(s_t, a_t, r_{t+1}, s_{t+1})$：根据 $\pi_t(s_t)$ 执行 $a_t$，观测 $r_{t+1}, s_{t+1}$
> > >
> > > **更新 q 值**：
> > > $$q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - \alpha_t(s_t, a_t)[q_t(s_t, a_t) - (r_{t+1} + \gamma \max_a q_t(s_{t+1}, a))]$$
> > >
> > > **更新策略** ($\epsilon$-greedy)：
> > > $$\pi_{t+1}(a|s_t) = \begin{cases} 1 - \frac{\epsilon}{|\mathcal{A}(s_t)|}(|\mathcal{A}(s_t)| - 1) & \text{if } a = \arg\max_a q_{t+1}(s_t, a) \\ \frac{\epsilon}{|\mathcal{A}(s_t)|} & \text{otherwise} \end{cases}$$

#### Off-policy 版本

> [!algorithm] 算法 7.3：Q-learning (Off-policy 版本)
> **初始化**：$q_0(s,a)$，行为策略 $\pi_b(a|s)$，$\alpha_t(s,a) = \alpha > 0$
>
> **目标**：从所有状态学习最优策略 $\pi_T$
>
> **For** each episode $\{s_0, a_0, r_1, s_1, a_1, r_2, \ldots\}$ generated by $\pi_b$ **do**：
> > **For** each step $t = 0, 1, 2, \ldots$ **do**：
> > > **更新 q 值**：
> > > $$q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - \alpha_t(s_t, a_t)[q_t(s_t, a_t) - (r_{t+1} + \gamma \max_a q_t(s_{t+1}, a))]$$
> > >
> > > **更新目标策略** (greedy)：
> > > $$\pi_{T,t+1}(a|s_t) = \begin{cases} 1 & \text{if } a = \arg\max_a q_{t+1}(s_t, a) \\ 0 & \text{otherwise} \end{cases}$$
>
> **注意**：
> - 行为策略 $\pi_b$ 可以是任意策略，只要足够探索性
> - 目标策略 $\pi_T$ 是 greedy 的（不需要探索，因为它不用于生成样本）
> - Off-policy 版本可以**离线**实现：先收集所有经验，再处理

> [!warning] 关于 On-policy vs Off-policy 的常见误区
> **易混淆概念**：On-policy/Off-policy vs Online/Offline
>
> - **On-policy/Off-policy**：指行为策略与目标策略是否相同
> - **Online/Offline**：指学习是否与环境交互同时进行
>
> **关系**：
> - On-policy 算法只能 **online** 实现（必须边交互边学习）
> - Off-policy 算法可以 **online 或 offline** 实现

---

## 7. 统一视角 (A Unified Point of View)

### 7.1 统一的更新公式

本章介绍的所有算法都可以写成统一形式：

$$\boxed{q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - \alpha_t(s_t, a_t)[q_t(s_t, a_t) - \bar{q}_t]}$$

其中 $\bar{q}_t$ 是 **TD Target**。不同算法的区别仅在于 TD Target 的构造：

> [!summary] 各算法的 TD Target
>
> | 算法 | TD Target $\bar{q}_t$ |
> |:---:|:---|
> | **Sarsa** | $r_{t+1} + \gamma q_t(s_{t+1}, a_{t+1})$ |
> | **n-step Sarsa** | $r_{t+1} + \gamma r_{t+2} + \cdots + \gamma^n q_t(s_{t+n}, a_{t+n})$ |
> | **Expected Sarsa** | $r_{t+1} + \gamma \sum_a \pi_t(a|s_{t+1}) q_t(s_{t+1}, a)$ |
> | **Q-learning** | $r_{t+1} + \gamma \max_a q_t(s_{t+1}, a)$ |
> | **Monte Carlo** | $r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \ldots$ |
>
> **注意**：MC 可视为特例——设 $\alpha_t(s_t, a_t) = 1$，则 $q_{t+1}(s_t, a_t) = \bar{q}_t$

### 7.2 统一的求解方程

所有算法都可视为求解某个方程 $q(s, a) = \mathbb{E}[\bar{q}_t | s, a]$ 的随机近似算法：

> [!summary] 各算法求解的方程
>
> | 算法 | 求解的方程 | 方程类型 |
> |:---|:---|:---:|
> | **Sarsa** | $q_\pi(s,a) = \mathbb{E}[R_{t+1} + \gamma q_\pi(S_{t+1}, A_{t+1}) \| S_t = s, A_t = a]$ | BE |
> | **n-step Sarsa** | $q_\pi(s,a) = \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^n q_\pi(S_{t+n}, A_{t+n}) \| S_t = s, A_t = a]$ | BE |
> | **Expected Sarsa** | $q_\pi(s,a) = \mathbb{E}[R_{t+1} + \gamma \mathbb{E}_{A_{t+1}}[q_\pi(S_{t+1}, A_{t+1})] \| S_t = s, A_t = a]$ | BE |
> | **Q-learning** | $q(s,a) = \mathbb{E}[R_{t+1} + \gamma \max_a q(S_{t+1}, a) \| S_t = s, A_t = a]$ | **BOE** |
> | **Monte Carlo** | $q_\pi(s,a) = \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \ldots \| S_t = s, A_t = a]$ | BE |
>
> *BE = Bellman Equation（贝尔曼期望方程），BOE = Bellman Optimality Equation（贝尔曼最优方程）*

### 7.3 关键区别总结

> [!important] 核心对比
>
> | 算法 | 估计对象 | 策略类型 | 核心特点 |
> |:---:|:---:|:---:|:---|
> | **Sarsa** | $q_\pi$ | On-policy | 学习当前策略的价值（保守、安全） |
> | **Q-learning** | $q_*$ | **Off-policy** | 直接学习最优价值（激进、大胆） |
> | **MC** | $q_\pi$ | On-policy | 无偏但高方差，需完整回合 |
>
> **直观理解**：
> - **Sarsa** 评估"我按当前策略走会怎样"
> - **Q-learning** 评估"最优情况下会怎样"（即使当前不是最优）

---

## 8. 章节总结

### 8.1 核心思想

1. **TD 学习**利用贝尔曼方程的结构，通过**自举 (Bootstrapping)** 实现增量更新
2. 所有 TD 算法都可视为求解贝尔曼方程（或贝尔曼最优方程）的**随机近似算法**
3. TD error 表示新信息/**创新**，驱动估计值向更准确的方向更新

### 8.2 算法选择指南

```
需要估计什么？
├── 状态价值 v(s) → TD(0)
└── 动作价值 q(s,a)
    ├── 给定策略的价值 q_π
    │   ├── 低方差优先 → Expected Sarsa
    │   ├── 偏差-方差平衡 → n-step Sarsa
    │   └── 简单实现 → Sarsa
    └── 最优价值 q_* → Q-learning
        ├── 可用其他策略的数据 → Off-policy Q-learning
        └── 仅用当前策略数据 → On-policy Q-learning
```

### 8.3 常见问题解答 (Q&A)

> [!faq] Q1: "TD" 中的 "TD" 是什么意思？
> **A**: 每个 TD 算法都有一个 TD error，表示新样本与当前估计之间的差异。由于这个差异是在**相邻时间步**之间计算的，因此称为 **temporal-difference（时序差分）**。

> [!faq] Q2: "学习" 在 TD 学习中意味着什么？
> **A**: 从数学角度，"学习"本质上是"估计"。即从样本中估计状态/动作价值，然后基于估计值获得策略。

> [!faq] Q3: Sarsa 只能估计给定策略的价值，如何用它学习最优策略？
> **A**: 需要将价值估计与**策略提升**过程结合。即每更新一次价值，就相应更新策略，这就是**广义策略迭代 (GPI)** 的思想。

> [!faq] Q4: 为什么 Sarsa 使用 ε-greedy 而不是 greedy 策略？
> **A**: 因为 Sarsa 是 on-policy 算法，策略既用于生成样本，也是要评估的策略。使用 ε-greedy 可以保证**探索**，确保有足够多的样本。

> [!faq] Q5: 为什么实践中常用常数学习率，而不是递减学习率？
> **A**: 理论上 $\alpha_t$ 应递减以保证收敛。但在最优策略搜索中，被评估的策略**不断变化**（非平稳）。使用常数 $\alpha$ 可以更快地适应策略变化。虽然估计值会有波动，但只要 $\alpha$ 足够小，波动可忽略。

> [!faq] Q6: 为什么 Q-learning 是 off-policy 而其他 TD 算法是 on-policy？
> **A**: 根本原因是 Q-learning 求解**贝尔曼最优方程**，而其他算法求解**贝尔曼期望方程**。贝尔曼最优方程不依赖任何特定策略，因此行为策略可以与目标策略不同。

> [!faq] Q7: Off-policy Q-learning 的目标策略为什么是 greedy 而不是 ε-greedy？
> **A**: 因为目标策略**不需要**用于生成样本（由行为策略负责），所以不需要探索性。目标策略只需要是基于当前 q 值的最优策略。

---

