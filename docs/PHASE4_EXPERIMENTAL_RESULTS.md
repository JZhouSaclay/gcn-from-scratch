# GCN From Scratch - 终极实验报告

## 项目概述

本项目从零开始（from scratch）使用基础 PyTorch Tensors 实现了 Kipf & Welling 的 "Semi-Supervised Classification with Graph Convolutional Networks" (ICLR 2017) 论文。所有图操作均使用 `torch.matmul` 等基础运算完成，不依赖 PyTorch Geometric 或 DGL 等高级库。

**核心实现文件：**
- `src/data_loader.py` - 数据加载与预处理
- `src/layers.py` - 图卷积层底层实现
- `src/model.py` - GCN 与 MLP 模型架构
- `src/train.py` - 训练循环与早停机制
- `main.py` - 端到端训练管道

---

## 实验一：Cora 数据集主实验

### 数据集统计

| 属性 | 数值 |
|------|------|
| 节点数（论文数） | 2,708 |
| 边数（引用关系） | 5,429 |
| 特征维度 | 1,433（词袋模型） |
| 类别数 | 7 |
| 训练集 | 140（每类20个） |
| 验证集 | 500 |
| 测试集 | 1,000 |

### 模型配置

| 配置项 | 设置值 | 论文设置 |
|--------|--------|----------|
| 层数 | 2 | 2 ✓ |
| 隐藏维度 | 16 | 16 ✓ |
| 优化器 | Adam | Adam ✓ |
| 学习率 | 0.01 | 0.01 ✓ |
| Weight Decay | 5e-4 | 5e-4 ✓ |
| Dropout | 0.5 | 0.5 ✓ |
| 早停耐心值 | 50 | - |

### 实验结果

| 指标 | 我们的实现 | 论文 (Kipf & Welling 2017) | 差距 |
|------|-----------|---------------------------|------|
| **Test Accuracy** | **80.90%** (809/1000) | **81.5%** | **-0.60%** |
| Train Accuracy | 100.00% | - | - |
| Validation Accuracy | 80.00% | - | - |
| Best Epoch | 149 | - | - |
| Training Time | ~4s | - | - |
| 参数量 | 23,063 | - | - |

### 结果分析

✅ **成功复现论文结果**：我们的实现与论文报告准确率仅相差 0.6%，在可接受范围内。

**关键成功因素：**
1. 正确的对称归一化邻接矩阵实现
2. 严格遵循论文的 140/500/1000 数据划分
3. 早停机制（patience=50）防止过拟合
4. 半监督训练：仅对 train_mask 节点计算损失

---

## 实验二：消融实验 - MLP vs GCN

### 实验设计

**目的**：验证图卷积（邻居聚合）对性能的贡献。

**控制变量**：
- 相同架构：1433 → 16 → 7
- 相同参数量：23,063
- 相同超参数：lr=0.01, weight_decay=5e-4, patience=50
- 相同随机种子：42

**唯一区别**：
- **MLP**：不使用邻接矩阵，`output = Linear(ReLU(Linear(x)))`
- **GCN**：使用图卷积，`output = GraphConv(ReLU(GraphConv(x, adj)), adj)`

### 实验结果对比

| 指标 | MLP | GCN | 提升 |
|------|-----|-----|------|
| **Test Accuracy** | **53.30%** | **79.70%** | **+26.40%** 🎯 |
| Train Accuracy | 100.00% | 98.57% | -1.43% |
| Validation Accuracy | 52.60% | 79.40% | +26.80% |
| Best Epoch | 63 | 58 | -5 |
| Training Time | 0.48s | 2.10s | +1.62s |

### 关键发现

🔍 **1. 图卷积带来巨大提升**
- GCN 比 MLP 提升 **26.4%**，证明图结构信息至关重要
- 引用网络中的"同配性假设"得到验证：相互引用的论文主题相似

🔍 **2. MLP 严重过拟合**
- MLP 训练准确率 100%，但测试仅 53.3%
- 原因：1433 维特征 + 140 个样本，模型轻易记住训练集
- 缺乏图结构提供的正则化约束

🔍 **3. GCN 具有更好的泛化能力**
- GCN 测试准确率（79.7%）接近验证准确率（79.4%）
- 邻居聚合提供了隐式的数据增强和正则化

---

## 实验三：t-SNE 节点聚类可视化

### 可视化方法

- **输入**：GCN 第一层输出的 16 维隐藏特征（ReLU 激活后）
- **降维算法**：t-SNE (perplexity=30, max_iter=1000)
- **输出**：2D 散点图，7 个类别用不同颜色标记

### 可视化结果

![t-SNE Visualization of GCN Hidden Features](../results/cora_tsne.png)

### 聚类质量定量分析

| 类别 | 类内紧密度 ↓ | 观察 |
|------|-------------|------|
| Rule Learning | **13.03** | 最紧凑，分类最容易 ✓ |
| Reinforcement Learning | 13.43 | 较紧凑 |
| Genetic Algorithms | 14.26 | 较紧凑 |
| Case Based | 16.67 | 中等分散 |
| Probabilistic Methods | 17.93 | 较分散 |
| Theory | 17.93 | 较分散 |
| Neural Networks | 21.19 | 最分散（跨领域主题多）|
| **平均** | **16.35** | **整体良好** |

**类间分离度：**
- 平均中心距离：50.53
- 最小中心距离：27.88
- **聚类质量分数**（分离度/紧密度）：**3.09** ✓

### 可视化结论

📊 **7 个类别形成基本分离的簇**，验证了 GCN 学到的节点表示具有良好的类别区分性：

1. **Rule Learning**（浅蓝色）：右下角最紧凑的簇，与其他类别边界清晰
2. **Genetic Algorithms**（橙色）：左下角聚集良好
3. **Neural Networks**（青绿色）：分布较分散，可能与多个领域有交叉引用
4. **Theory**（粉色）：右上区域，与 Neural Networks 有部分重叠

---

## 实验四：CiteSeer 数据集扩展实验

### 数据集统计

| 属性 | 数值 |
|------|------|
| 节点数 | 3,312 |
| 边数 | 4,732 |
| 特征维度 | 3,703 |
| 类别数 | 6 |
| 训练集 | 120（每类20个） |
| 验证集 | 500 |
| 测试集 | 1,000 |

### 模型配置

| 配置项 | 设置值 |
|--------|--------|
| 层数 | 2 |
| 隐藏维度 | 32（相比 Cora 增加，因特征维度更高） |
| 学习率 | 0.01 |
| Weight Decay | 5e-4 |
| Dropout | 0.5 |
| 早停耐心值 | 100 |

### 实验结果

| 指标 | 数值 |
|------|------|
| **Test Accuracy** | **69.10%** (691/1000) |
| Train Accuracy | 100.00% |
| Validation Accuracy | 71.00% |
| Best Epoch | 132 |
| Training Time | 7.25s |
| 参数量 | 118,726 |

### 与 Cora 的对比分析

| 对比维度 | Cora | CiteSeer | 分析 |
|----------|------|----------|------|
| **Test Accuracy** | **80.90%** | **69.10%** | CiteSeer 更难分类 |
| 特征维度 | 1,433 | 3,703 | CiteSeer 特征更稀疏 |
| 图密度 | 更密 | 更稀疏 | CiteSeer 边数/节点数更低 |
| 收敛 epoch | 149 | 132 | 两者收敛速度相近 |
| 训练时间 | ~4s | ~7s | CiteSeer 特征维度更高 |

**CiteSeer 难度更高的原因：**
1. 图结构更稀疏（边数/节点数更低），信息传播受限
2. 特征维度更高但更稀疏（3,703 维），有效信号更少
3. 部分节点孤立，缺乏邻居信息

---

## 总结与结论

### 核心成果

| 成就 | 说明 |
|------|------|
| ✅ 论文复现 | Cora 数据集 80.90% vs 论文 81.5%，差距仅 0.6% |
| ✅ 消融验证 | 证明图卷积贡献 +26.4% 准确率 |
| ✅ 可视化分析 | t-SNE 展示清晰的类别聚类结构 |
| ✅ 跨数据集验证 | CiteSeer 数据集 69.10% 准确率 |
| ✅ 纯 PyTorch 实现 | 不依赖 PyG/DGL，全部使用基础 Tensor 运算 |

### 关键技术要点

1. **对称归一化**：`Â = D^{-1/2} Ã D^{-1/2}` 是 GCN 稳定训练的关键
2. **半监督训练**：仅对 140 个有标签节点计算损失，但利用全部 2708 个节点的图结构
3. **早停机制**：patience=50 有效防止过拟合，找到最优验证点
4. **邻居聚合**：图卷积通过聚合邻居特征提供隐式正则化

### 项目文件清单

```
gcn-from-scratch/
├── src/
│   ├── data_loader.py    # 数据加载（Cora/CiteSeer）
│   ├── layers.py         # GraphConvolution 层
│   ├── model.py          # GCN & MLP 模型
│   ├── train.py          # 训练循环
│   └── utils.py          # 工具函数
├── main.py               # 主训练脚本
├── run_ablation.py       # 消融实验脚本
├── visualize.py          # t-SNE 可视化脚本
├── results/
│   ├── cora_tsne.png     # t-SNE 可视化图
│   └── citeseer_log.txt  # CiteSeer 实验日志
└── docs/
    ├── PHASE1_DATA_PREPARATION.md
    ├── PHASE2_MODEL_ARCHITECTURE.md
    ├── PHASE3_TRAINING_PIPELINE.md
    ├── PHASE4_ANALYSIS.md
    └── PHASE4_EXPERIMENTAL_RESULTS.md  # 本文档
```

---

**报告生成时间**：2026-03-16
**项目状态**：✅ 完成
**主要贡献者**：GCN From Scratch Team
