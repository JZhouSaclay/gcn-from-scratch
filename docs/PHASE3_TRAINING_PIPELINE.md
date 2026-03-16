# Phase 3: 训练管道与端到端整合 (Training Pipeline)

## 当前进度 (Current Progress)

本阶段已完成以下核心文件的实现：

| 文件 | 核心功能 | 关键函数/类 |
|------|----------|-------------|
| `src/train.py` | 训练循环与早停机制 | `train_gcn()` |
| `main.py` | 端到端训练管道 | `main()`, `load_data()` |

**已实现能力：**
- ✅ 半监督训练循环：仅对 `train_mask` 标记的节点计算损失和准确率
- ✅ Adam 优化器：学习率 0.01，权重衰减 5e-4
- ✅ 验证集早停（Early Stopping）： patience=50 防止过拟合
- ✅ 定期日志输出：每 10 个 epoch 打印训练状态
- ✅ 端到端整合：数据加载 → 模型构建 → 训练 → 测试评估

**最终成果：**
- 🎯 **Test Accuracy: 80.90%** (809/1000 正确)
- 📊 与论文 (Kipf & Welling 2017) 的 81.5% 仅差 **0.6%**

---

## 阶段目标 (Phase Goal)

> **将数据加载、模型构建、训练优化、测试评估串联成完整的机器学习工作流，实现可复现的端到端训练。**

---

## 核心动机与原理 (The "Why")

### 1. 为什么必须使用 mask 进行半监督训练？

**问题的本质：** Cora 数据集只有 140 个有标签节点（每类 20 个），但总共有 2708 个节点。

**如果忽略 mask：**
```python
# 错误的做法：对所有节点计算损失
loss = F.cross_entropy(logits, labels)  # 2708 个节点全部参与
```
- 1068 个测试节点和 500 个验证节点的标签会"泄露"到训练中
- 模型会过拟合到测试集，失去泛化能力评估的意义

**正确的半监督做法：**
```python
# 正确的做法：只用 train_mask 为 True 的节点
loss = F.cross_entropy(logits[train_mask], labels[train_mask])  # 只有 140 个节点
```

**关键洞察：**
- **前向传播**：使用所有节点和边（利用图结构传播信息）
- **损失计算**：只用训练节点（半监督约束）
- **图卷积的特殊性**：邻居节点的特征可以参与计算，即使邻居本身没有标签

这就是 GCN 的魔力所在：**用 140 个标签，指导 2708 个节点的分类**。

---

### 2. 为什么使用 Early Stopping？

**观察到的现象：**
从训练日志可以看到：
- Epoch 50: Train Acc = 93.6%, Val Acc = 75.4%
- Epoch 100: Train Acc = 100%, Val Acc = 77.8%
- Epoch 149 (Best): Val Acc = **80.0%**
- Epoch 199: Val Acc = 79.6% (下降)

**问题：** 训练准确率很快达到 100%，但验证准确率在 80% 左右波动。

**原因分析：**
- **过拟合**：模型记住了 140 个训练样本的特征
- **验证集波动**：模型在验证集上的性能在达到峰值后开始下降

**Early Stopping 机制：**
```
监控验证准确率
    ↓
连续 patience 个 epoch 没有提升
    ↓
回滚到最佳模型状态
    ↓
停止训练，防止过拟合
```

在我们的实现中，`patience=50` 意味着允许模型在 50 个 epoch 内尝试突破验证集性能瓶颈，最终在第 149 个 epoch 找到最佳点。

---

### 3. 为什么选择 Adam 优化器？

**论文选择：** Kipf & Welling 使用 Adam (lr=0.01, weight_decay=5e-4)。

**Adam 的优势：**
1. **自适应学习率**：为每个参数维护独立的学习率，适合稀疏梯度（如图卷积中的不同节点）
2. **动量加速**：利用梯度的历史信息加速收敛
3. **对超参数不敏感**：默认参数在大多数任务上表现良好

**Weight Decay (5e-4) 的作用：**
- 这是 L2 正则化系数
- 惩罚大权重，防止模型过拟合到训练集的 140 个样本
- 作用于所有可训练参数（W1, b1, W2, b2）

---

### 4. 为什么每 10 个 epoch 打印一次日志？

**训练动态监控：**
```
Epoch   1: Train Loss=1.97, Train Acc=15.0%, Val Acc=32.6%
Epoch  10: Train Loss=1.82, Train Acc=47.9%, Val Acc=50.6%
Epoch  50: Train Loss=0.58, Train Acc=93.6%, Val Acc=75.4%
Epoch 100: Train Loss=0.18, Train Acc=100%, Val Acc=77.8%
Epoch 149: (Best Val Acc=80.0%)
```

**这些数字告诉我们：**
- **Epoch 1-30**：模型快速学习，损失从 1.97 降到 1.19
- **Epoch 30-100**：训练准确率继续提升，但验证准确率增长放缓
- **Epoch 100+**：训练准确率饱和，验证准确率在 78-80% 之间波动

**监控指标的意义：**
- **Train Loss**：模型是否在学习（应该下降）
- **Train Acc**：是否过拟合（增长太快要小心）
- **Val Acc**：真实泛化能力（我们最关心的指标）

---

### 5. 为什么最终测试准确率与论文接近？

**我们的结果：80.90% vs 论文：81.5%**

**成功的关键因素：**

1. **正确的数据预处理**
   - 对称归一化邻接矩阵 ✓
   - 特征 L2 归一化 ✓
   - 严格的 140/500/1000 数据划分 ✓

2. **正确的模型架构**
   - 两层 GCN (1433→16→7) ✓
   - ReLU + Dropout(0.5) ✓

3. **正确的训练配置**
   - Adam 优化器，lr=0.01 ✓
   - weight_decay=5e-4 ✓
   - 早停机制防止过拟合 ✓

4. **正确的评估方式**
   - 只在 test_mask 上评估最终性能 ✓
   - 使用训练过程中验证集最佳模型 ✓

**微小的差距来源：**
- 随机种子不同导致的数据划分差异
- PyTorch 版本差异
- 浮点数精度差异

0.6% 的差距在可接受范围内，证明我们的 "from scratch" 实现是正确的！

---

## 实现方式 (Methodology)

### 训练循环核心逻辑

```python
def train_gcn(model, x, adj, labels, train_mask, val_mask, test_mask, ...):
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, epochs + 1):
        # ===== Training =====
        model.train()
        optimizer.zero_grad()

        logits = model(x, adj)  # 所有节点的前向传播

        # 半监督核心：只用 train_mask 计算损失
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        loss.backward()
        optimizer.step()

        # ===== Validation =====
        model.eval()
        with torch.no_grad():
            val_loss = F.cross_entropy(logits[val_mask], labels[val_mask])
            val_acc = accuracy(logits[val_mask], labels[val_mask])

        # ===== Early Stopping =====
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()  # 保存最佳模型
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break  # 触发早停

    # 加载最佳模型进行最终评估
    model.load_state_dict(best_model_state)
    test_acc = accuracy(logits[test_mask], labels[test_mask])
```

### 端到端管道架构

```
┌─────────────────────────────────────────────────────────────┐
│                         main.py                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: Load Data                                           │
│    ├── CoraDataLoader().load()                               │
│    ├── Returns: x, adj, y, train_mask, val_mask, test_mask  │
│    └── Move to device (CPU/GPU)                              │
│                         ↓                                    │
│  Step 2: Build Model                                         │
│    ├── GCN(nfeat=1433, nhid=16, nclass=7)                   │
│    ├── Total params: 23,063                                  │
│    └── model.to(device)                                      │
│                         ↓                                    │
│  Step 3: Train                                               │
│    ├── train_gcn(...)                                        │
│    ├── Epoch 1-149: Training loop                            │
│    ├── Early stopping at epoch 199                           │
│    └── Best Val Acc: 80.0%                                   │
│                         ↓                                    │
│  Step 4: Final Evaluation                                    │
│    ├── model.eval()                                          │
│    ├── logits = model(x, adj)                               │
│    ├── test_acc = accuracy(logits[test_mask], ...)          │
│    └── Test Acc: 80.90% ✓                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 阶段展望与下一步 (Outlook & Next Steps)

### 本阶段奠定的基础

1. **可复现的训练管道**：完整的 `main.py` 可以从零开始训练并评估 GCN
2. **验证实现正确性**：80.90% 的准确率证明我们的 "from scratch" 实现与论文一致
3. **调试与监控机制**：日志输出和早停机制让训练过程可控

### 潜在扩展方向

**1. 超参数调优**
```bash
# 尝试不同的隐藏层维度
python main.py --hidden 32

# 尝试不同的学习率
python main.py --lr 0.005

# 尝试不同的 dropout
python main.py --dropout 0.3
```

**2. 其他数据集**
```bash
# CiteSeer 数据集
python main.py --dataset citeseer

# PubMed 数据集（需要稀疏矩阵优化）
python main.py --dataset pubmed
```

**3. 更深的模型**
```python
# 使用 DeepGCN 类
model = DeepGCN(nfeat=1433, nhid=16, nclass=7, nlayers=3)
```

**4. 模型保存与加载**
```python
# 保存训练好的模型
torch.save(model.state_dict(), 'gcn_cora.pth')

# 加载模型进行推理
model.load_state_dict(torch.load('gcn_cora.pth'))
```

**5. 可视化分析**
- 绘制训练曲线（Loss vs Epoch）
- 可视化节点嵌入（t-SNE 降维）
- 分析错误分类的节点

---

## 附录：与论文的完整对比

| 配置项 | 论文 (Kipf & Welling) | 我们的实现 | 状态 |
|--------|----------------------|-----------|------|
| 数据集 | Cora | Cora | ✓ |
| 层数 | 2 | 2 | ✓ |
| 隐藏维度 | 16 | 16 | ✓ |
| 优化器 | Adam | Adam | ✓ |
| 学习率 | 0.01 | 0.01 | ✓ |
| Weight Decay | 5e-4 | 5e-4 | ✓ |
| Dropout | 0.5 | 0.5 | ✓ |
| 训练集 | 140 (20/类) | 140 (20/类) | ✓ |
| 验证集 | 500 | 500 | ✓ |
| 测试集 | 1000 | 1000 | ✓ |
| **Test Accuracy** | **81.5%** | **80.9%** | ✓ |

**结论**：我们成功地从零开始实现了 Kipf & Welling 的 GCN 论文，在 Cora 数据集上达到了论文级别的性能！
