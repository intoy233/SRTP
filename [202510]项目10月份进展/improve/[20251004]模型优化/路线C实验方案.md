# 路线C: 单一模型优化实验方案

**决策时间**: 2025年10月4日
**背景**: 分诊-专家系统全面失败(所有阈值R²<0.5920)
**当前最佳**: Griffin Plot + 幂函数变换 + 贝叶斯岭回归 (R²=0.5920)
**目标**: R² > 0.63, 相对提升 > 6.4%

---

## 一、战略转变

### ❌ 已放弃的方案:
1. **分诊-专家系统** (高风险R²全部为负)
2. **物理混合模型** (Scruton法则R²=-5.88)
3. **数据收集** (时间成本1-2月,成功率50%)

### ✅ 新方向: 榨取现有数据的最大价值

**核心思想**:
- 保持78维幂函数特征 (已证明有效)
- 尝试更强大的非线性模型
- 集成多个模型降低方差
- 特征选择去除冗余信息

---

## 二、实验设计

### 📊 实验矩阵

```
当前基线: 贝叶斯岭回归 (线性模型)
    ↓
方向1: 集成学习 (多个弱学习器→强学习器)
  ├── Bagging: Random Forest, Extra Trees
  ├── Boosting: XGBoost, LightGBM, CatBoost
  └── Stacking: 多模型融合

方向2: 深度学习 (非线性拟合能力)
  └── MLP + Dropout + BatchNorm + Early Stopping

方向3: 特征优化 (降维去噪)
  ├── LASSO特征选择
  ├── 递归特征消除(RFE)
  └── 主成分分析(PCA)
```

---

## 三、详细实验方案

### 🔬 实验1: Bagging集成

**模型**: Random Forest, Extra Trees

**原理**:
- 训练多棵决策树,每棵树用不同的数据子集
- 预测时取所有树的平均值
- 降低方差,提升泛化能力

**超参数搜索**:
```python
Random Forest:
- n_estimators: [100, 200, 300, 500]
- max_depth: [10, 15, 20, None]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]

Extra Trees:
- n_estimators: [100, 200, 300, 500]
- max_depth: [10, 15, 20, None]
- min_samples_split: [2, 5, 10]
```

**预期性能**: R² ≈ 0.60-0.62 (相对提升 +1-5%)

**优势**:
- ✓ 对过拟合鲁棒
- ✓ 处理78维特征能力强
- ✓ 自带特征重要性分析

---

### 🚀 实验2: Boosting集成

**模型**: XGBoost, LightGBM, CatBoost

**原理**:
- 顺序训练多棵树,每棵树修正前面树的错误
- 关注难预测样本(自动加权)
- 梯度提升优化

**超参数搜索**:
```python
XGBoost:
- n_estimators: [100, 200, 300, 500]
- learning_rate: [0.01, 0.05, 0.1, 0.2]
- max_depth: [3, 5, 7, 10]
- subsample: [0.6, 0.8, 1.0]
- colsample_bytree: [0.6, 0.8, 1.0]

LightGBM:
- n_estimators: [100, 200, 300, 500]
- learning_rate: [0.01, 0.05, 0.1]
- num_leaves: [31, 50, 100]
- max_depth: [-1, 10, 20]

CatBoost:
- iterations: [100, 200, 500]
- learning_rate: [0.01, 0.05, 0.1]
- depth: [4, 6, 8, 10]
```

**预期性能**: R² ≈ 0.62-0.65 (相对提升 +5-10%) ← **重点!**

**优势**:
- ✓ 强大的非线性拟合能力
- ✓ 自动处理特征交互
- ✓ 对异常值鲁棒
- ⚠ 需要防止过拟合(early stopping)

---

### 🎯 实验3: Stacking集成

**模型**: 多层模型融合

**架构**:
```
Level 0 (基学习器):
├── Ridge回归
├── Random Forest
├── XGBoost
├── LightGBM
└── SVR (支持向量回归)

Level 1 (元学习器):
└── BayesianRidge (带不确定性量化)
```

**原理**:
- Level 0的5个模型独立预测
- 将5个预测值作为新特征
- Level 1学习如何组合这些预测

**预期性能**: R² ≈ 0.63-0.67 (相对提升 +6-13%) ← **最佳预期!**

**优势**:
- ✓ 集成多种模型的优势
- ✓ 自动学习最优权重
- ✓ 保留不确定性量化(元模型用贝叶斯岭)

---

### 🧠 实验4: 深度学习

**模型**: 多层感知机(MLP) + Dropout + BatchNorm

**架构**:
```python
Input (78维)
    ↓
Dense(256) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Dense(128) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Dense(64) + BatchNorm + ReLU + Dropout(0.2)
    ↓
Dense(32) + ReLU + Dropout(0.1)
    ↓
Output(1) - 线性激活

损失函数: MSE (均方误差)
优化器: Adam (learning_rate=0.001)
Early Stopping: 验证集20轮不提升则停止
```

**超参数搜索**:
- 隐藏层单元: [128-64-32], [256-128-64], [512-256-128]
- Dropout率: [0.1, 0.2, 0.3, 0.4]
- Learning rate: [0.0001, 0.0005, 0.001, 0.005]
- Batch size: [16, 32, 64]

**预期性能**: R² ≈ 0.60-0.64 (相对提升 +1-8%)

**优势**:
- ✓ 强大的非线性表达能力
- ✓ 自动学习特征组合
- ✓ Dropout防止过拟合

**风险**:
- ⚠ 小样本(190)可能欠采样
- ⚠ 需要仔细调参

---

### ✂️ 实验5: 特征选择

**方法1: LASSO回归 (L1正则化)**
```python
LassoCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0], cv=5)
→ 自动选择最优alpha
→ 系数为0的特征被剔除
```

**方法2: 递归特征消除(RFE)**
```python
RFE(
    estimator=XGBoost(),
    n_features_to_select=30,  # 从78→30
    step=5
)
→ 逐步剔除最不重要的特征
```

**方法3: 基于重要性筛选**
```python
Random Forest训练后:
feature_importances_ > threshold (e.g. 0.01)
→ 保留重要特征
```

**流程**:
1. 特征选择: 78维 → 30-40维
2. 重新训练最佳模型(XGBoost/Stacking)
3. 对比降维前后性能

**预期性能**: R² ≈ 0.58-0.62 (可能不提升,但模型更简洁)

---

## 四、实验流程

### 📅 时间安排(7-10天)

**Day 1-2: Bagging集成**
- 实现Random Forest
- 实现Extra Trees
- 网格搜索超参数
- 5-Fold交叉验证

**Day 3-4: Boosting集成** ← **重点!**
- 实现XGBoost
- 实现LightGBM
- 实现CatBoost
- 超参数优化(Bayesian Optimization)

**Day 5-6: Stacking集成** ← **重点!**
- 训练5个基学习器
- 训练元学习器
- 交叉验证防止数据泄露

**Day 7-8: 深度学习**
- 实现MLP架构
- 超参数调优
- Early Stopping

**Day 9: 特征选择**
- LASSO/RFE筛选特征
- 重新训练最佳模型

**Day 10: 最终对比与决策**
- 所有模型性能对比
- 选择最佳方案
- 撰写总结报告

---

## 五、评估标准

### 📊 主要指标:

1. **验证集R²** (最重要)
   - 目标: > 0.63 (相对提升 > 6.4%)
   - 优秀: > 0.65 (相对提升 > 10%)

2. **验证集RMSE**
   - 目标: < 13.0 mm
   - 优秀: < 12.0 mm

3. **5-Fold一致性**
   - R²标准差 < 0.05 (稳定性)

4. **不确定性量化** (次要)
   - 难/简样本不确定性比值 > 1.0

### 🎯 成功标准:

**最低要求**:
- 验证R² > 0.63 (超越当前0.5920至少6.4%)
- RMSE < 13.0 mm
- 5-Fold稳定(std < 0.05)

**理想目标**:
- 验证R² > 0.65 (+10%)
- RMSE < 12.0 mm
- 保留不确定性量化能力

---

## 六、技术栈

### 📦 所需库:

```python
# 基础
import numpy as np
import pandas as pd

# 机器学习
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, LassoCV
from sklearn.svm import SVR
from sklearn.feature_selection import RFE

# Boosting
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# 深度学习
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# 超参数优化
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV  # Bayesian Optimization

# 评估
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
```

---

## 七、预期结果对比

| 模型 | 预期R² | 预期RMSE | 相对提升 | 优势 | 风险 |
|------|--------|----------|----------|------|------|
| **基线(贝叶斯岭)** | 0.5920 | 13.65mm | - | 不确定性量化 | 线性模型局限 |
| Random Forest | 0.60-0.62 | 13.0-13.5mm | +1-5% | 鲁棒,易用 | 提升有限 |
| XGBoost | 0.62-0.65 | 12.0-13.0mm | +5-10% | 强大非线性 | 需调参 |
| LightGBM | 0.62-0.64 | 12.5-13.0mm | +5-8% | 快速,高效 | 小样本可能过拟合 |
| **Stacking** | **0.63-0.67** | **11.5-12.5mm** | **+6-13%** | 集成优势 | 复杂度高 |
| MLP | 0.60-0.64 | 12.0-13.5mm | +1-8% | 非线性强 | 小样本风险 |
| 特征选择+XGBoost | 0.60-0.63 | 12.5-13.0mm | +1-6% | 模型简化 | 可能损失信息 |

**最佳预期**: Stacking集成 (R²=0.63-0.67, RMSE=11.5-12.5mm)

---

## 八、风险与应对

### 风险1: 过拟合 (概率60%)
**表现**: 训练R²高,验证R²低
**应对**:
- Early Stopping (验证集监控)
- Cross-Validation (5-Fold)
- Dropout/L2正则化 (深度学习)
- 剪枝/最大深度限制 (树模型)

### 风险2: 性能提升不足 (概率30%)
**表现**: R² < 0.63
**应对**:
- 尝试更多特征工程 (对数变换,Box-Cox)
- 数据增强 (SMOTE,但要谨慎)
- 降低目标,接受当前最佳模型

### 风险3: 丢失不确定性量化 (概率80%)
**表现**: XGBoost等模型不提供std
**应对**:
- Stacking时元模型用BayesianRidge
- 使用Quantile Regression (分位数回归)
- NGBoost (Natural Gradient Boosting, 带不确定性)

---

## 九、下一步行动

### 🔥 立即行动(今天):
- [x] 设计实验方案
- [ ] 安装所需库 (`pip install xgboost lightgbm catboost tensorflow scikit-optimize`)
- [ ] 编写Bagging实验代码

### 📅 本周计划:
- [ ] Day 1-2: Bagging (Random Forest, Extra Trees)
- [ ] Day 3-4: Boosting (XGBoost, LightGBM, CatBoost)
- [ ] Day 5-6: Stacking (5个基学习器 + 贝叶斯元学习器)
- [ ] Day 7: 深度学习(MLP)

### 🎯 最终目标:
- [ ] 找到R² > 0.63的最佳模型
- [ ] 撰写最终实验报告
- [ ] 部署模型供工程应用

---

**方案撰写**: 2025年10月4日
**负责人**: 吴先生
**预计完成**: 2025年10月14日

**核心策略**:
1. ✅ 保留有效特征工程 (Griffin Plot + 幂函数)
2. ✅ 尝试更强大的非线性模型 (Boosting, Stacking)
3. ✅ 防止过拟合 (Cross-Validation, Early Stopping)
4. ✅ 保留不确定性量化能力 (Stacking元模型用贝叶斯)

**让我们开始吧!** 🚀
