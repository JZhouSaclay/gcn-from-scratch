# Phase 2: 模型架构实现 (Model Architecture)

## 当前进度 (Current Progress)

本阶段已完成以下核心文件的实现：

| 文件 | 核心功能 | 关键类 |
|------|----------|--------|
| `src/layers.py` | 图卷积层底层实现 | `GraphConvolution`, `GCNLayer` |
| `src/model.py` | 完整 GCN 模型架构 | `GCN` (2层), `DeepGCN` (可扩展) |

**已实现能力：**
- ✅ `GraphConvolution` 层：严格实现 $H = \hat{A} X W + b$（使用 `torch.matmul`）
- ✅ Kaiming Uniform 参数初始化
- ✅ 两层 GCN 模型：$X \rightarrow H^{(1)} \rightarrow Z$
- ✅ 训练流程支持：ReLU 激活 + Dropout 正则化
- ✅ 推理模式：`predict()` 方法禁用 Dropout

---

## 阶段目标 (Phase Goal)

> **基于图卷积算子构建端到端的节点分类神经网络，实现特征变换-邻居聚合的交替计算。**

---

## 核心动机与原理 (The "Why")

### 1. 为什么图卷积的公式是 $H = \hat{A} X W$？

**传统 CNN 的局限：**
标准卷积核（如 3×3）在图像上滑动，假设了数据具有**规则网格结构**（像素上下左右均匀分布）。但图结构是**不规则的**：每个节点的邻居数量不同，没有固定的"上下左右"。

**图卷积的直觉：**
借鉴 CNN 的局部性原则，将"卷积"重新定义为：

> **每个节点的特征 = 它自己和邻居特征的加权平均，然后做线性变换**

数学拆解：

$$H = \hat{A} X W$$

可以分解为两个操作：

**Step 1: 特征变换** $X' = X W$
- 每个节点独立做线性变换（就像普通全连接层）
- $X: [N, F_{in}]$，$W: [F_{in}, F_{out}]$
- 结果 $X': [N, F_{out}]$，每个节点有了 $F_{out}$ 维新特征

**Step 2: 邻居聚合** $H = \hat{A} X'$
- $\hat{A}_{ij}$ 表示节点 $i$ 对节点 $j$ 的"关注程度"
- 新特征 $H_i = \sum_j \hat{A}_{ij} X'_j$，即邻居特征的加权求和

**为什么是矩阵乘法？**
因为 $\hat{A}$ 已经包含了归一化权重，一行 $\hat{A}_i$ 就是一个**聚合核**，自动处理不同度数的邻居。

---

### 2. 为什么需要两层 GCN？

**单层 GCN 的局限：**

$$H^{(1)} = \text{ReLU}(\hat{A} X W^{(0)})$$

每个节点只能聚合**1跳邻居**（直接连接的节点）的信息。在 Cora 引用网络中：
- 单层：论文只能"看到"直接引用它的论文
- 两层：论文可以"看到"引用者的引用者（2跳邻居）

**为什么是两层而不是更深？**

Kipf & Welling 的实验发现：
- **2层**：在 Cora 上达到 ~81.5% 准确率
- **3层及以上**：准确率下降，出现**过平滑（Over-smoothing）**问题

**过平滑的直觉：**
当层数太多时，所有节点的特征趋于一致——就像把不同颜色的颜料反复混合，最终都变成灰色。图卷积本质上是"拉平"邻居差异的操作，做太多次就失去了区分性。

**两层的折中：**
- 足够捕捉"引用网络中的主题聚类"（我的引用者研究什么，我也可能研究什么）
- 不足以让特征扩散到整个图，保留节点特异性

---

### 3. 为什么要加 Dropout？

**背景问题：** 训练集只有 140 个节点，却有 1433 维特征 → **严重的过拟合风险**

**Dropout 的作用机制：**

训练时以概率 $p$（通常 0.5）随机置零隐藏层特征：

$$H^{(1)}_{\text{drop}} = \text{Mask} \odot \text{ReLU}(\hat{A} X W^{(0)})$$

**为什么有效？**

1. **集成学习视角**：每次训练相当于随机采样一个子网络，最终模型是指数级多个子网络的平均
2. **特征共适应打破**：阻止模型过度依赖某些特定特征组合
3. **图数据的特殊收益**：GCN 中，一个节点的预测依赖于邻居；Dropout 让模型学会"即使某些邻居信息缺失，也能正确预测"

**为什么放在第一层输出后？**

$$\text{Layer1} \rightarrow \text{ReLU} \rightarrow \text{Dropout} \rightarrow \text{Layer2}$$

- 不对输入层做 Dropout：原始特征 $X$ 是稀疏的 0-1 向量，再丢弃会丢失过多信息
- 不对输出层做 Dropout：分类层需要稳定输出

---

### 4. 为什么不把 LogSoftmax 放在模型里？

**我们的实现：**
```python
# model.py forward()
x = self.gc2(x, adj)  # 输出原始 logits
return x  # 没有 Softmax!
```

**训练时使用：**
```python
# train.py
loss = F.cross_entropy(logits, labels)  # 内部自动做 log_softmax + nll_loss
```

**好处：**

1. **数值稳定性**：`cross_entropy` 内部使用 log-sum-exp 技巧，避免数值溢出
2. **灵活性**：模型输出 logits 可以直接用于多种损失函数（交叉熵、标签平滑、对比学习等）
3. **调试方便**：可以直接查看原始分数，不被 Softmax 压缩到 (0,1)

---

## 实现方式 (Methodology)

### GraphConvolution 层实现

```python
def forward(self, x, adj):
    # Step 1: 特征变换
    support = torch.matmul(x, self.weight)  # [N, F_in] @ [F_in, F_out] -> [N, F_out]

    # Step 2: 邻居聚合
    output = torch.matmul(adj, support)      # [N, N] @ [N, F_out] -> [N, F_out]

    # Step 3: 加偏置
    if self.bias is not None:
        output = output + self.bias

    return output
```

**参数初始化（Kaiming Uniform）：**

```python
def reset_parameters(self):
    # 针对 ReLU 激活优化，保持前向传播方差恒定
    nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
```

### 完整模型架构图

```
输入: X [N, 1433], A_hat [N, N]

        ┌─────────────────────────────────────┐
        │         GCN Layer 1                 │
        │  ┌─────────┐    ┌─────────┐        │
        │  │  X @ W1 │ -> │ A_hat @ │ -> H1  │
        │  │[1433,16]│    │   H'    │[N,16] │
        │  └─────────┘    └─────────┘        │
        └─────────────────────────────────────┘
                      ↓
                ReLU activation
                      ↓
                Dropout(p=0.5)  ← 训练时开启，推理时关闭
                      ↓
        ┌─────────────────────────────────────┐
        │         GCN Layer 2                 │
        │  ┌─────────┐    ┌─────────┐        │
        │  │ H1 @ W2 │ -> │ A_hat @ │ -> Z  │
        │  │ [16, 7] │    │   H'    │[N, 7] │
        │  └─────────┘    └─────────┘        │
        └─────────────────────────────────────┘
                      ↓
              输出: Logits [N, 7]
                      ↓
          CrossEntropyLoss (with mask)
```

---

## 阶段展望与下一步 (Outlook & Next Steps)

### 本阶段奠定的基础

1. **图卷积算子完备**：`GraphConvolution` 可作为独立模块复用到其他图神经网络（如 GAT、GraphSAGE 的变体）
2. **模型架构确定**：两层 GCN + ReLU + Dropout 是节点分类任务的经典 baseline
3. **前向/推理双模式**：`model.train()` 和 `model.eval()` 支持训练和预测两种场景

### 引出的下一步需求

**Phase 3 必须解决的核心问题：**

> 如何利用 `train_mask` 中的 140 个标签，通过反向传播优化模型参数？

具体需求：

1. **训练循环实现**：
   - 如何计算 masked loss？（只对有标签节点计算交叉熵）
   - 选择什么优化器？（论文使用 Adam，lr=0.01）
   - 如何处理过拟合？（早停 Early Stopping）

2. **评估指标监控**：
   - 训练/验证/测试准确率如何分别计算？
   - 如何保存最佳模型？

3. **端到端整合**：
   - `main.py` 如何串联 data_loader → model → train → evaluate？

### 关键接口约定

本阶段输出的模型为下一阶段提供了标准接口：

```python
# 模型创建
model = GCN(nfeat=1433, nhid=16, nclass=7, dropout=0.5)

# 前向传播（训练模式）
model.train()
logits = model(x, adj_hat)  # [N, 7]

# 预测（推理模式）
model.eval()
preds = model.predict(x, adj_hat)  # [N] 类别索引
```

下一阶段的 `train.py` 将负责：
1. 用 `train_mask` 选择训练节点计算 `F.cross_entropy(logits[train_mask], labels[train_mask])`
2. 用 `optimizer.step()` 更新 `model.gc1.weight`, `model.gc2.weight` 等参数
3. 用 `val_mask` 监控验证准确率，触发早停

---

## 附录：与论文的对应关系

| 论文公式 | 我们的实现 | 位置 |
|----------|------------|------|
| $H^{(l+1)} = \sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l)}W^{(l)})$ | `output = torch.matmul(adj, torch.matmul(x, weight))` | `layers.py:85-94` |
| $Z = f(X, A) = \text{softmax}(\hat{A}\text{ReLU}(\hat{A}XW^{(0)})W^{(1)})$ | `GCN.forward()` 方法 | `model.py:51-63` |
| Dropout 概率 0.5 | `F.dropout(x, p=0.5, training=self.training)` | `model.py:58` |
| 隐藏层维度 16 | `nhid=16` 默认参数 | `model.py:34` |
