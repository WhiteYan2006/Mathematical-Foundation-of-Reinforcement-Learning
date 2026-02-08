---
date: 2026-02-08
tags:
  - 强化学习
---
![[Chapter_6.pdf]]
# 第六章：随机逼近与随机梯度下降 (Stochastic Approximation & SGD)

## 1. 引言：填补蒙特卡洛与时序差分之间的空白

在强化学习的课程体系中，本章起到了承上启下的关键作用 。

- **上一章 (Chapter 5)**：我们学习了 **蒙特卡洛学习 (Monte Carlo Learning, MC)**。MC 需要等到一个 Episode 结束后，收集所有数据一次性计算平均值，这被称为 **非增量式 (Non-incremental)** 方法 。
    
- **下一章 (Chapter 7)**：我们将学习 **时序差分学习 (Temporal-Difference Learning, TD)**。TD 不需要等待 Episode 结束，每一步都在更新，这是一种 **增量式 (Incremental)** 方法 。
    

> [!abstract] 核心动机 (Motivation) 许多初学者在首次接触 TD 算法时，会觉得其更新公式（如 $V(S) \leftarrow V(S) + \alpha [R + \gamma V(S') - V(S)]$）非常反直觉 。
> 
> 本章的目的就是通过 **随机逼近 (Stochastic Approximation, SA)** 理论，为这种“增量式迭代”提供数学基础 。我们后续会发现，TD 算法本质上就是一种特殊的随机逼近算法 。

---

## 2. 均值估计 (Mean Estimation)

均值估计是理解增量式更新的最简单切入点。在 RL 中，状态价值 (State Value) 本质上就是回报 (Return) 的期望（均值）。

### 2.1 问题定义

假设我们有一组独立同分布 (i.i.d.) 的样本序列 $\{x_i\}_{i=1}^N$，我们需要估计其期望 $\mathbb{E}[X]$ 。 最直观的方法是 **蒙特卡洛估计**：

$$\bar{x} = \frac{1}{N} \sum_{i=1}^N x_i$$

这要求我们收集完所有 $N$ 个样本后才能计算，效率较低 。

### 2.2 增量式均值推导 (Incremental Formulation)

我们需要一种方法，每接收到一个新样本 $x_k$，就能立即更新当前的均值估计 $w_k$ 。

> [!math] 增量公式的严格推导
> 设 $w_k$ 为基于前 $k-1$ 个样本计算出的均值，即 $w_k = \frac{1}{k-1}\sum_{i=1}^{k-1}x_i$ 。 当获得第 $k$ 个样本 $x_k$ 时，新的均值 $w_{k+1}$ 为：
> 
> $$ \begin{aligned} w_{k+1} &= \frac{1}{k} \sum_{i=1}^{k} x_i \\ &= \frac{1}{k} \left( \sum_{i=1}^{k-1} x_i + x_k \right) \quad \text{(将求和拆分为前 } k-1 \text{ 项和第 } k \text{ 项)} \\ &= \frac{1}{k} \left( (k-1)w_k + x_k \right) \quad \text{(利用 } \sum_{i=1}^{k-1}x_i = (k-1)w_k \text{ 替换)} \\ &= \frac{k-1}{k} w_k + \frac{1}{k} x_k \\ &= \left( 1 - \frac{1}{k} \right) w_k + \frac{1}{k} x_k \\ &= w_k - \frac{1}{k} w_k + \frac{1}{k} x_k \\ &= w_k + \frac{1}{k} (x_k - w_k) \quad \text{(整理得到最终形式)} \end{aligned} $$
> 
> 最终迭代公式为：
> 
> $$w_{k+1} = w_k - \frac{1}{k}(w_k - x_k)$$

### 2.3 广义形式

我们可以将步长 $\frac{1}{k}$ 推广为一般的系数 $\alpha_k$ ：
$$w_{k+1} = w_k - \alpha_k (w_k - x_k)$$

这个公式的物理意义非常直观：

- $w_k$：当前的估计值。
- $x_k - w_k$：**误差 (Error)** 或 **残差**（观测值与估计值之差）。
- $\alpha_k$：**步长 (Step size)** 或 **学习率**。
- **更新逻辑**：新的估计 = 旧的估计 + 步长 $\times$ 误差 。
    

---

## 3. Robbins-Monro (RM) 算法

RM 算法是随机逼近领域的奠基之作，它解决了一个更通用的问题：**求根 (Root Finding)** 。
### 3.1 问题描述

我们需要找到方程 $g(w) = 0$ 的根 $w^*$ 。
- **难点**：函数 $g(w)$ 的解析式是未知的（黑盒），或者无法直接计算其导数 。
- **可用信息**：我们可以观测到带噪声的函数值。即给定一个输入 $w_k$，我们得到的观测值是 $\tilde{g}(w_k, \eta_k) = g(w_k) + \eta_k$，其中 $\eta_k$ 是观测噪声 。

### 3.2 算法公式
Robbins-Monro 提出的迭代算法如下：
$$w_{k+1} = w_k - a_k \tilde{g}(w_k, \eta_k)$$

其中 $a_k$ 是正系数序列 。

> [!example] 直觉理解 (Intuition) 为什么这个公式能找到根？假设 $g(w)$ 是单调递增函数 ：
> 
> - 如果 $w_k > w^*$（当前猜测偏大），由于单调性，通常 $g(w_k) > 0$。根据公式 $w_{k+1} = w_k - a_k \cdot (\text{正数})$，下一次猜测 $w_{k+1}$ 会变小，向 $w^*$ 靠近。
>     
> - 如果 $w_k < w^*$（当前猜测偏小），通常 $g(w_k) < 0$。根据公式 $w_{k+1} = w_k - a_k \cdot (\text{负数})$，下一次猜测 $w_{k+1}$ 会变大，同样向 $w^*$ 靠近。
>     
> 
> 即使存在噪声 $\eta_k$，只要步长 $a_k$ 逐渐衰减，噪声的影响最终会被平均掉。

### 3.3 收敛性分析 (Robbins-Monro Theorem)

为了保证 $w_k$ 以概率 1 (w.p.1) 收敛到 $w^*$，必须满足以下三个条件 ：

1. **梯度有界条件**：$0 < c_1 \le \nabla_w g(w) \le c_2$。这保证了函数单调且不会过于陡峭 。
2. **噪声条件**：$\mathbb{E}[\eta_k | [cite_start]\mathcal{H}_k] = 0$ 且方差有限。即噪声均值为 0 。
3. **步长序列条件 (关键)** ：
    - $\sum_{k=1}^{\infty} a_k = \infty$
    - $\sum_{k=1}^{\infty} a_k^2 < \infty$
        
> [!important] 步长条件的深度解析 * **$\sum a_k = \infty$ (无穷发散)**：步长不能衰减得太快。如果步长衰减太快（例如 $a_k = 1/2^k$），总的移动距离 $\sum a_k$ 是有限的。如果初始猜测 $w_1$ 距离真实值 $w^*$ 很远，算法可能永远“走不到”目的地 。 * **$\sum a_k^2 < \infty$ (平方收敛)**：步长必须最终趋于 0。这是为了抑制噪声 $\eta_k$ 的影响。随着迭代进行，我们需要步长足够小，使得噪声带来的随机抖动即使累积起来也是有限的，从而让结果稳定在 $w^*$ 附近 。
> 
> **经典步长选择**：$a_k = \frac{1}{k}$ 满足上述条件，因为调和级数发散 ($\sum \frac{1}{k} \to \infty$) 而 p-级数 ($p=2$) 收敛 ($\sum \frac{1}{k^2} = \frac{\pi^2}{6} < \infty$) 。

### 3.4 均值估计是 RM 的特例

回到均值估计问题，我们想求 $\mathbb{E}[X]$。 令 $g(w) = w - \mathbb{E}[X]$。显然，方程 $g(w)=0$ 的根就是 $w^* = \mathbb{E}[X]$ 。 我们的观测值是 $x_k$，观测函数构造为：
$$\tilde{g}(w_k, x_k) = w_k - x_k$$

我们可以把观测值写成“真值+噪声”的形式：
$$w_k - x_k = (w_k - \mathbb{E}[X]) + (\mathbb{E}[X] - x_k) = g(w_k) + \eta_k$$

其中 $\eta_k = \mathbb{E}[X] - x_k$，满足均值为 0 。 代入 RM 公式：
$$w_{k+1} = w_k - a_k (w_k - x_k)$$

这与增量式均值估计算法完全一致 。

---

## 4. 随机梯度下降 (Stochastic Gradient Descent, SGD)

SGD 是机器学习和强化学习中最常用的优化算法，它本质上也是 RM 算法的一种特例 。
### 4.1 优化问题设置

目标是最小化目标函数 $J(w)$，该函数定义为随机变量 $f(w, X)$ 的期望 ：
$$\min_w J(w) = \mathbb{E}[f(w, X)]$$

### 4.2 三种梯度下降方法的对比

|**方法**|**更新公式**|**特点**|
|---|---|---|
|**GD (Gradient Descent)**|$w_{k+1} = w_k - \alpha_k \mathbb{E}[\nabla_w f(w_k, X)]$|理论上最优，但实际中无法直接计算期望 $\mathbb{E}$ 。|
|**BGD (Batch GD)**|$w_{k+1} = w_k - \alpha_k \frac{1}{n}\sum_{i=1}^n \nabla_w f(w_k, x_i)$|用整个数据集的平均梯度近似期望。计算量大，每一步都要遍历所有数据 。|
|**SGD (Stochastic GD)**|$w_{k+1} = w_k - \alpha_k \nabla_w f(w_k, x_k)$|每次只用**一个样本** $x_k$ 的梯度来更新。计算极快，但引入了噪声 。|

### 4.3 为什么 SGD 是 RM 算法？ (Why SGD is an RM Algorithm?)

#### 1. 问题转化：从优化到求根
SGD 的目标是最小化目标函数 $J(w) = \mathbb{E}[f(w, X)]$。 这等价于求解其梯度为 0 的根 ：
$$g(w) \triangleq \nabla_w J(w) = \mathbb{E}[\nabla_w f(w, X)] = 0$$

这里，$g(w)$ 是我们要找根的函数（即真实梯度）。
#### 2. 构造观测函数 (The Noisy Observation)
在 RM 算法中，我们无法直接获得 $g(w)$，只能获得观测值 $\tilde{g}$。 在 SGD 中，我们的观测值是**随机梯度** ：
$$\tilde{g}(w, x) \triangleq \nabla_w f(w, x)$$

我们可以将这个观测值数学上拆解为“真值 + 噪声”的形式 ：
$$\tilde{g}(w, x) = \underbrace{\mathbb{E}[\nabla_w f(w, X)]}_{g(w)} + \underbrace{\left( \nabla_w f(w, x) - \mathbb{E}[\nabla_w f(w, X)] \right)}_{\eta}$$

其中，$\eta$ 是观测噪声。
#### 3. 形式统一
将上述观测值代入 RM 算法的标准迭代公式 $w_{k+1} = w_k - a_k \tilde{g}(w_k, \eta_k)$ ：
$$w_{k+1} = w_k - a_k \nabla_w f(w_k, x_k)$$

这正是 SGD 的算法公式。因此，SGD 本质上就是试图寻找 $g(w)=0$ 根的 RM 算法 。

---

### 4.4 SGD 的收敛模式 (Convergence Pattern)
为了回答“SGD 的收敛是慢还是随机？”这个问题，**相对误差**给出了答案 。
#### 1. 定义相对误差 (Relative Error)
定义 $\delta_k$ 为“随机梯度与真实梯度之差”与“真实梯度”的比值 ：
$$\delta_k \doteq \frac{| \nabla_w f(w_k, x_k) - \mathbb{E}[\nabla_w f(w_k, X)] |}{| \mathbb{E}[\nabla_w f(w_k, X)] |}$$

#### 2. 分母的推导 (均值定理)
由于 $w^*$ 是最优解，所以 $\mathbb{E}[\nabla_w f(w^*, X)] = 0$。我们可以把分母重写为 ：
$$\text{分母} = | \mathbb{E}[\nabla_w f(w_k, X)] - \mathbb{E}[\nabla_w f(w^*, X)] |$$
根据**中值定理 (Mean Value Theorem)**，存在 $\tilde{w}_k \in [w_k, w^*]$ 使得 ：
$$\text{分母} = | \mathbb{E}[\nabla_w^2 f(\tilde{w}_k, X) (w_k - w^*)] |$$

> [!note]
> 
> 这里 $\nabla_w^2 f$ 是海森矩阵 (Hessian Matrix)。

#### 3. 引入强凸假设
假设函数 $f$ 是严格凸的，即其二阶导数有下界 $c > 0$ ：
$$\nabla_w^2 f \ge c$$

那么分母可以放缩为 ：
$$| \mathbb{E}[\nabla_w^2 f(\tilde{w}_k, X)] (w_k - w^*) | \ge c | w_k - w^* |$$
#### 4. 最终不等式与结论
将分母的下界代入 $\delta_k$ 的定义，得到 ：

$$\delta_k \le \frac{| \nabla_w f(w_k, x_k) - \mathbb{E}[\nabla_w f(w_k, X)] |}{c | w_k - w^* |}$$

> [!math] 收敛模式总结 上式表明 $\delta_k$ 与距离 $|w_k - w^*|$ 成**反比**。这解释了 SGD 的两个阶段 ：
> 
> 1. **远离最优解时** ($|w_k - w^*|$ 很大)：$\delta_k$ 很小。SGD 表现得像普通梯度下降 (GD)，快速逼近。
>     
> 2. **接近最优解时** ($|w_k - w^*|$ 很小)：$\delta_k$ 变得很大（可能会趋向无穷）。此时相对误差极大，随机性占主导，导致参数在 $w^*$ 附近**震荡**。
>     

这就是为什么 SGD 在后期必须减小学习率 $\alpha_k$ 的数学原因。
## 5. BGD, MBGD 与 SGD 的确定性形式

### 5.1 确定性问题的随机化 (A Deterministic Formulation)

在实际应用中（例如训练神经网络），我们通常面对的是一个固定的、有限的数据集 $\{x_i\}_{i=1}^n$，而不是从某种未知分布中无限采样的随机变量。
**1. 问题背景：确定性优化** 我们要最小化的是所有样本损失的平均值 ：
$$J(w) = \frac{1}{n}\sum_{i=1}^n f(w, x_i)$$
这是一个确定性的问题。如果我们使用标准的梯度下降 (GD)，更新公式为 ：
$$w_{k+1} = w_k - \alpha_k \frac{1}{n}\sum_{i=1}^n \nabla_w f(w_k, x_i)$$
**2. 算法的简化** 如果 $n$ 很大，计算求和项非常耗时。我们希望每次迭代只使用一个样本 $x_k$ ：

$$w_{k+1} = w_k - \alpha_k \nabla_w f(w_k, x_k)$$
> [!question] 关键问题
> 
> 这个算法只涉及确定性的数据集，没有显式的随机变量，它还能被称为 **随机**梯度下降 (SGD) 吗？它是如何与 SGD 的期望最小化理论联系起来的？

**3. 随机变量的构造 (The Construction)** 为了将上述确定性问题纳入 SGD 的理论框架，我们需要**人为引入**一个随机变量 $X$ 。
- **定义域**：$X$ 取值于有限集合 $\{x_i\}_{i=1}^n$。
- **概率分布**：$X$ 服从**均匀分布 (Uniform Distribution)**，即每一个样本被选中的概率相等 ：
    $$P(X=x_i) = \frac{1}{n}$$
**4. 数学等价性证明**
现在，我们可以计算这个随机变量 $X$ 的函数 $f(w, X)$ 的期望：

$$\begin{aligned} \mathbb{E}[f(w, X)] &= \sum_{i=1}^n P(X=x_i) \cdot f(w, x_i) \\ &= \sum_{i=1}^n \frac{1}{n} \cdot f(w, x_i) \\ &= \frac{1}{n} \sum_{i=1}^n f(w, x_i) \\ &= J(w) \end{aligned}$$
> [!math] 结论
> $$J(w) = \frac{1}{n}\sum_{i=1}^n f(w, x_i) = \mathbb{E}[f(w, X)]$$
> 这里的等号是**严格成立 (Strict)** 的，而非近似 。

**5. 这种转化的意义** 通过上述构造，我们证明了：**最小化有限数据的平均误差，等价于最小化人为构造的随机变量的期望误差。** 因此，只要我们在每一步迭代中，**均匀且独立 (Uniformly and Independently)** 地从集合 $\{x_i\}_{i=1}^n$ 中采样 $x_k$，那么算法 $w_{k+1} = w_k - \alpha_k \nabla_w f(w_k, x_k)$ 就完全符合 SGD 的定义 。

> [!important] 注意 由于是随机采样，$x_k$ 在不同迭代步中可能会重复取到同一个样本，这是允许的，也是符合 IID 采样要求的 。

### 5.2 Mini-Batch Gradient Descent (MBGD)

介于 BGD 和 SGD 之间的方法 。

$$w_{k+1} = w_k - \alpha_k \frac{1}{m} \sum_{j \in \mathcal{I}_k} \nabla_w f(w_k, x_j)$$

其中 $m$ 是 Batch Size。

- **优点**：
    - 相比 SGD：利用了更多样本，梯度的方差更小，震荡更少，收敛更稳 。
    - 相比 BGD：不需要遍历全量数据，计算效率更高 。

> [!example] 示例对比 课件中的二维平面均值估计实验显示 ：
> 
> - **SGD ($m=1$)**：路径非常曲折，充满了随机性，但在初期下降极快。
>     
> - **MBGD ($m=5, 50$)**：路径变得平滑，越接近 BGD 的直线路径。
>     

---

## 6. 总结 (Summary)

本章为后续章节奠定了坚实的数学基础：

1. **均值估计**：$w_{k+1} = w_k - \alpha_k (w_k - x_k)$ 是所有增量学习的原型。
2. **RM 算法**：证明了在只有噪声观测的情况下，通过衰减步长可以找到方程的根。
3. **SGD**：是 RM 算法在优化问题上的应用。
    
**对强化学习的意义**：

在[[Lecture7. 时序差分学习|下一章]]中，我们将看到 **TD Learning** 的更新公式：
$$V(S) \leftarrow V(S) + \alpha [ \underbrace{R + \gamma V(S')}_{\text{Target}} - V(S) ]$$

仔细观察这个公式，它正是 **均值估计/SGD** 的形式：$w_{k+1} = w_k + \alpha (x_k - w_k)$。 TD 算法本质上就是试图通过含噪声的样本（TD Target）来逼近真实的价值函数，这完全符合本章讨论的随机逼近理论 。