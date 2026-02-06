---
date: 2026-02-05
tags:
  - 强化学习
---
 
# Chapter 4: 值迭代与策略迭代 (Value Iteration & Policy Iteration)

> [!abstract] 核心目标
> 
> 本章的目的是寻找 **最优策略 (Optimal Policy)** 和 **最优状态值 (Optimal State Value)**。
> 
> 在上一章（Chapter 3）中，我们引入了 **贝尔曼最优方程 (Bellman Optimality Equation, BOE)**：
> 
> $$v^* = \max_{\pi} (r_{\pi} + \gamma P_{\pi} v^*)$$
> 
> 本章将介绍三种求解该方程的动态规划算法：
> 
> 1. **值迭代 (Value Iteration, VI)**：直接迭代求解 BOE。
>     
> 2. **策略迭代 (Policy Iteration, PI)**：显式地分离策略评估和策略改进。
>     
> 3. **截断策略迭代 (Truncated Policy Iteration)**：二者的通用形式。
>     

---

## 1. 值迭代算法 (Value Iteration Algorithm)

值迭代是基于 **压缩映射定理 (Contraction Mapping Theorem)** 的直接应用。它通过不断迭代贝尔曼最优算子来逼近 $v^*$。

### 1.1 算法核心原理

根据 [[Lecture3. 贝尔曼最优公式]]，BOE 可以写成不动点形式 $v = f(v)$，其中 $f(v) = \max_{\pi}(r_{\pi} + \gamma P_{\pi}v)$。

值迭代的迭代公式为：

$$v_{k+1} = f(v_k) = \max_{\pi} (r_{\pi} + \gamma P_{\pi} v_k), \quad k=0,1,2,\dots$$

> [!important] 注意
> 
> 这里 $v_k$ **不是** 某个策略的状态值（即 $v_k \neq v_{\pi}$），它只是算法过程中的一个中间变量（数值向量）。只有当 $k \to \infty$ 时，$v_k$ 才收敛到最优状态值 $v^*$。

### 1.2 算法分解：两步走 (Two Steps)

为了理解和实现，我们可以将上述公式分解为两个步骤：

1. **策略更新 (Policy Update)**：
    给定数值 $v_k$，寻找一个贪婪策略 $\pi_{k+1}$ 来最大化右侧目标：
    $$\pi_{k+1} = \arg \max_{\pi} (r_{\pi} + \gamma P_{\pi} v_k)$$
    
> [!math] 解释
> 
> 这一步本质上是在问：如果下一时刻的状态价值是 $v_k$，我现在应该选哪个动作能获得最大的期望回报？
    
2. **值更新 (Value Update)**：
    使用刚才找到的策略 $\pi_{k+1}$ 来更新 $v_{k+1}$：$$v_{k+1} = r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_k$$
### 1.3 逐元素形式 (Elementwise Form) - 实现细节

在编程实现时，我们针对每一个状态 $s$ 进行操作。

#### Step 1: 策略更新 (计算 Q 值)

策略 $\pi_{k+1}$ 是通过最大化 $q$ 值得到的贪婪策略。

对于状态 $s$，动作 $a$ 的 **Q值 (q-value)** 计算如下：

$$q_k(s, a) = \sum_{r} p(r|s,a)r + \gamma \sum_{s'} p(s'|s,a) v_k(s')$$

_注：这里的第一项 $\sum p(r|s,a)r$ 是期望即时奖励，第二项是折扣后的未来期望价值。_

此时，贪婪策略 $\pi_{k+1}(a|s)$ 定义为：

$$\pi_{k+1}(a|s) = \begin{cases} 1, & a = a^*_k(s) \\ 0, & a \neq a^*_k(s) \end{cases}$$

其中 $a^*_k(s) = \arg \max_{a} q_k(s, a)$。

#### Step 2: 值更新 (取最大值)

将贪婪策略代入值更新公式：

$$\begin{aligned} v_{k+1}(s) &= \sum_{a} \pi_{k+1}(a|s) q_k(s, a) \\ &= \max_{a} q_k(s, a) \quad \text{(因为策略只在最大Q值处取1)} \end{aligned}$$

> [!summary] 值迭代伪代码总结
> 
> 1. 初始化 $v_0$ (任意值，如全0)。
>     
> 2. **While** $||v_{k+1} - v_k|| > \epsilon$ **do**:
>     
>     - For each state $s$:
>         
>         - For each action $a$:
>             
>             - $q_k(s, a) = \mathbb{E}[R] + \gamma \sum_{s'} p(s'|s,a) v_k(s')$
>                 
>         - $v_{k+1}(s) = \max_{a} q_k(s, a)$
>             
> 3. 输出收敛后的 $v^*$ 和对应的贪婪策略 $\pi^*$。
>     

- Elementwise Form一般用于编程实现，Matrix-Vector  Form一般用于理论分析

---

## 2. 策略迭代算法 (Policy Iteration Algorithm)

值迭代主要关注数值 $v$ 的收敛，而 **策略迭代 (Policy Iteration)** 关注策略序列 $\pi_0, \pi_1, \dots$ 的收敛。

### 2.1 算法流程

从一个初始策略 $\pi_0$ 开始，不断循环以下两步，直到策略不再改变：

1. **策略评估 (Policy Evaluation, PE)**：
    
    计算当前策略 $\pi_k$ 的真实状态值 $v_{\pi_k}$。这需要求解 **贝尔曼方程 (Bellman Equation)**：
    
    $$v_{\pi_k} = r_{\pi_k} + \gamma P_{\pi_k} v_{\pi_k}$$
    
2. **策略改进 (Policy Improvement, PI)**：
    
    根据 $v_{\pi_k}$ 寻找更好的策略 $\pi_{k+1}$：
    
    $$\pi_{k+1} = \arg \max_{\pi} (r_{\pi} + \gamma P_{\pi} v_{\pi_k})$$
    

流程图示：

$$\pi_0 \xrightarrow{PE} v_{\pi_0} \xrightarrow{PI} \pi_1 \xrightarrow{PE} v_{\pi_1} \xrightarrow{PI} \pi_2 \dots \xrightarrow{} \pi^*$$

### 2.2 关键数学推导

#### Q1: 如何进行策略评估 (PE)?

我们需要从方程 $v_{\pi_k} = r_{\pi_k} + \gamma P_{\pi_k} v_{\pi_k}$ 中解出 $v_{\pi_k}$。

- **方法一：闭式解 (Closed-form)**
    
    $$v_{\pi_k} = (I - \gamma P_{\pi_k})^{-1} r_{\pi_k}$$
    
    _缺点：矩阵求逆计算量大 ($O(n^3)$)，不适合大状态空间。_
    
- **方法二：迭代解 (Iterative Solution)**
    
    我们可以在 PE 步骤内部再使用一个迭代算法（这实际上是固定策略下的值迭代）：
    
    $$v_{\pi_k}^{(j+1)} = r_{\pi_k} + \gamma P_{\pi_k} v_{\pi_k}^{(j)}, \quad j=0,1,2,\dots$$
    
    当 $j \to \infty$ 时，$v_{\pi_k}^{(j)} \to v_{\pi_k}$。
    

#### Q2: 为什么策略会改进 (PI)?

我们需要证明 $v_{\pi_{k+1}} \ge v_{\pi_k}$。

> [!math] 策略改进引理 (Policy Improvement Lemma) 证明
> 
> 根据定义：
> 
> $\pi_{k+1} = \arg \max_{\pi} (r_{\pi} + \gamma P_{\pi} v_{\pi_k})$
> 
> 这意味着在单步更新中，$\pi_{k+1}$ 优于 $\pi_k$：
> 
> $$r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_{\pi_k} \ge r_{\pi_k} + \gamma P_{\pi_k} v_{\pi_k} = v_{\pi_k}$$
> 
> 令 $\Delta = v_{\pi_{k+1}} - v_{\pi_k}$。利用 $v_{\pi_{k+1}}$ 的贝尔曼方程展开：
> 
> $$\begin{aligned} v_{\pi_{k+1}} - v_{\pi_k} &= (r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_{\pi_{k+1}}) - v_{\pi_k} \\ &= r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_{\pi_{k+1}} \underbrace{- (r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_{\pi_k}) + (r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_{\pi_k})}_{\text{加一项减一项}} - v_{\pi_k} \\ &= \gamma P_{\pi_{k+1}} (v_{\pi_{k+1}} - v_{\pi_k}) + \underbrace{(r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_{\pi_k}) - v_{\pi_k}}_{\ge 0 \text{ (由上述贪婪性)}} \\ \end{aligned}$$
> 
> 记 $\delta = (r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_{\pi_k}) - v_{\pi_k} \ge 0$。
> 
> 此时有 $\Delta = \gamma P_{\pi_{k+1}} \Delta + \delta$。
> 
> 递归展开：$\Delta = \delta + \gamma P_{\pi_{k+1}} \delta + (\gamma P_{\pi_{k+1}})^2 \delta + \dots$
> 
> 由于 $\delta \ge 0, P \ge 0, \gamma > 0$，因此 $\Delta \ge 0$。
> 
> **证毕：** $v_{\pi_{k+1}} \ge v_{\pi_k}$。

---

## 3. 截断策略迭代 (Truncated Policy Iteration)

### 3.1 动机与联系

对比 VI 和 PI，我们会发现：

- **VI (Value Iteration)**：每步只做 **1次** 迭代来更新值 ($v_{k+1} = \max \dots$)。这相当于在 PE 阶段只迭代了一次。
- **PI (Policy Iteration)**：每步要做 **无穷次** 迭代直到算出精确的 $v_{\pi_k}$。

**截断策略迭代** 是两者的折中：在策略评估 (PE) 阶段，我们不需要完全解出 $v_{\pi_k}$，也不像 VI 那样只迭代一次，而是迭代固定次数（或直到满足某个宽松的收敛标准）。
### 3.2 算法比较表

|**算法**|**PE 阶段迭代次数**|**计算成本/步**|**收敛所需步数**|
|---|---|---|---|
|**值迭代 (VI)**|1 次|低|多|
|**策略迭代 (PI)**|$\infty$ (直至收敛)|高|少|
|**截断策略迭代**|$j_{truncate}$ (如 10 次)|中|中|

### 3.3 实现细节

算法结构如下：
1. **外层循环 ($k$)**：策略更新。
    - **内层循环 ($j$)**：策略评估 (PE)。
        - 设定 $v^{(0)} = v_{k-1}$ (继承上一步的值，利用 **Bootstrapping**)。
        - 执行 $j_{truncate}$ 次：$v^{(j+1)} = r_{\pi_k} + \gamma P_{\pi_k} v^{(j)}$。
        - 令 $v_k = v^{(j_{truncate})}$。
    - **策略改进 (PI)**：
        - $\pi_{k+1} = \arg \max_{\pi} (r_{\pi} + \gamma P_{\pi} v_k)$。

> [!example] 直觉理解
> 
> 想象你在爬山（寻找最优值）。
> 
> - **VI**：看一眼地图，迈一步，重新看地图，再迈一步。
>     
> - **PI**：看一眼地图，计算出如果沿着当前方向一直走到底能多高（计算量大），然后选个最好的新方向。
>     
> - **Truncated PI**：看一眼地图，沿着当前方向走几步（比如10步），觉得差不多了，再重新看地图选方向。这通常是最高效的。
>     

---

## 4. 总结与核心结论

1. **收敛性**：这三种算法最终都会收敛到最优状态值 $v^*$ 和最优策略 $\pi^*$。
2. **通用性**：它们都属于 **广义策略迭代 (Generalized Policy Iteration, GPI)** 的范畴——即“评估值”和“改进策略”两个过程交替进行。
3. **模型依赖**：这些算法都需要已知 $p(r|s,a)$ 和 $p(s'|s,a)$，因此它们属于 **动态规划 (Dynamic Programming)** 方法，而非这类不需要模型的强化学习（如 Q-Learning）。

> [!math] 关键公式速查
> 
> - **BOE**: $v = \max_{\pi}(r_{\pi} + \gamma P_{\pi}v)$
>     
> - **VI Update**: $v_{k+1}(s) = \max_a \left( \sum r + \gamma \sum v_k(s') \right)$
>     
> - **PI Evaluation**: $v_{\pi_k} = (I - \gamma P_{\pi_k})^{-1} r_{\pi_k}$
>     
> - **PI Improvement**: $\pi_{k+1}(s) = \arg \max_a q_{\pi_k}(s,a)$
>