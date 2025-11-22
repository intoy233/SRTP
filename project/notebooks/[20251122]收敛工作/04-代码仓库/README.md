# 04-代码仓库

**说明**: 本文件夹包含SRTP项目的核心算法实现、数据处理脚本、模型训练脚本等关键代码文件。

---

## 📂 文件夹结构

```
04-代码仓库/
├── dual_expert_framework.py          # 双专家混合模型核心框架
├── dual_expert_v2_optimized.py       # 双专家模型V2优化版本
├── step4_train_version_b.py          # Version B训练脚本
├── step5_train_version_c.py          # Version C训练脚本
├── clean_dataset_v2.py               # 数据清洗脚本V2
└── integrate_final_dataset_v2.py     # 最终数据集整合脚本V2
```

---

## 🔑 核心代码说明

### 1. 双专家混合模型 (最终方案)

#### dual_expert_framework.py
**功能**: 风险感知双专家混合框架的初始实现

**核心架构**:
```python
Stage 1: 风险分类器 (RandomForest)
  输入: 桥梁结构特征 (B, D, L, Drag, Lift, ...)
  输出: 风险等级 (Low/Medium/High)

Stage 2: 专家路由
  - 低/中风险样本 → Expert-L (Stacking集成)
  - 高风险样本 → Expert-H (Stacking集成)
```

**关键参数**:
- 风险阈值: 60mm (通过敏感性分析确定)
- 分类器: RandomForest (n_estimators=100, max_depth=10)
- Expert-L基模型: Ridge + Lasso + RandomForest + SVR
- Expert-H基模型: Ridge + Lasso + RandomForest + SVR
- 元学习器: BayesianRidge

**性能指标**:
- Overall R²: 0.76
- High-Risk R²: 0.64
- RMSE: 13.26mm
- 风险分类器F1-score: 0.88

---

#### dual_expert_v2_optimized.py
**功能**: 双专家模型的优化版本

**改进点**:
1. **训练策略优化**: Expert-L使用全部数据训练(而非仅低/中风险子集)
2. **超参数调优**: 网格搜索优化基模型参数
3. **代码结构优化**: 更清晰的模块化设计
4. **可视化增强**: 添加更多诊断图表

**使用示例**:
```python
from dual_expert_v2_optimized import DualExpertModel

# 初始化模型
model = DualExpertModel(threshold=60, random_state=42)

# 训练
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
metrics = model.evaluate(X_test, y_test)
print(f"Overall R²: {metrics['overall_r2']:.4f}")
print(f"High-Risk R²: {metrics['high_r2']:.4f}")
```

---

### 2. 数据处理脚本

#### clean_dataset_v2.py
**功能**: 系统性数据清洗与质量提升

**清洗步骤**:
1. **删除物理冲突样本** (9条)
   - 检查: Amplitude > 1000mm (异常大振幅)
   - 检查: Critical_Wind_Speed < 1 m/s (物理不合理)
   - 检查: B/D比例异常 (如B/D > 100)

2. **补全气动系数缺失值** (106条)
   - Drag系数: 使用断面形状类似桥梁的经验值
   - Lift系数: 根据Strouhal数和断面形状估算

3. **数据类型规范化**
   - 数值列转换为float64
   - 类别列转换为category
   - 日期列标准化格式

**质量提升效果**:
- 清洗前: 475样本
- 清洗后: 466样本
- 真实Vcr占比: 58%
- 缺失值: 0

---

#### integrate_final_dataset_v2.py
**功能**: 整合多批次数据源为最终数据集

**数据来源**:
- Batch 1: 原始196样本 (2024年10月)
- Batch 2-6: 文献调研新增样本 (2024年11月)
  - SCI/EI期刊论文: 58篇
  - 学位论文: 12篇
  - 现场监测记录: 若干

**整合流程**:
```
原始数据 → 去重 → 格式统一 → 质量检查 → 最终数据集
196条    → 469条 → 475条     → 466条      → dataset_clean_v2.csv
```

**特征工程**:
- 原始特征: B, D, L, Amplitude, Critical_Wind_Speed, Drag, Lift
- 派生特征: B/D比例, Strouhal数, 雷诺数
- 标签: Amplitude (振幅, mm)

---

### 3. 实验训练脚本

#### step4_train_version_b.py
**功能**: Version B (混合数据) 实验训练

**实验设置**:
- 数据: 369条 (含经验公式填充的Vcr)
- 模型: 单一Stacking集成
- 目标: 测试混合数据对高风险预测的影响

**结果**:
- Overall R²: 0.32
- High-Risk R²: 0.73 (首次突破正值!)
- 教训: 高风险性能提升,但Overall被严重拖累

---

#### step5_train_version_c.py
**功能**: Version C (数据清洗) 实验训练

**实验设置**:
- 数据: 466条 (删除污染源,补全缺失值)
- 模型: 单一Stacking集成
- 目标: 测试数据清洗对性能的影响

**结果**:
- Overall R²: 0.25
- High-Risk R²: 0.75
- 结论: 数据清洗未能解决根本矛盾 → 启发双专家方案

---

## 🎯 代码开发规范

### 代码风格
- 遵循 PEP 8 规范
- 使用 Black 格式化 (line-length=100)
- 类型提示: 关键函数使用类型注解

### 文档字符串
```python
def train_model(X: np.ndarray, y: np.ndarray, **kwargs) -> tuple:
    """
    训练双专家混合模型

    参数:
        X: 特征矩阵, shape=(n_samples, n_features)
        y: 标签向量, shape=(n_samples,)
        **kwargs: 模型超参数

    返回:
        (model, metrics): 训练好的模型和评估指标
    """
    pass
```

### 命名规范
- 类名: PascalCase (如 `DualExpertModel`)
- 函数名: snake_case (如 `train_expert_l`)
- 常量: UPPER_SNAKE_CASE (如 `DEFAULT_THRESHOLD`)

---

## 📊 实验复现指南

### 环境配置
```bash
# Python版本
python 3.8+

# 核心依赖
pip install numpy pandas scikit-learn matplotlib seaborn
```

### 复现双专家模型
```bash
# 1. 数据准备
python scripts/integrate_final_dataset_v2.py
python scripts/clean_dataset_v2.py

# 2. 训练模型
python scripts/dual_expert_v2_optimized.py

# 3. 查看结果
# 输出: notebooks/[20251122]风险分区+专家组合/
#   - 04-混淆矩阵-Fold[1-5].png
#   - 05-拟合散点图-Fold[1-5].png
#   - 06-阈值敏感性分析.png
```

---

## 🔧 代码改进历程

| 阶段 | 核心代码 | 关键改进 | 性能 |
|------|----------|---------|------|
| 十月版 | `train_baseline.py` | 基础Stacking集成 | Overall 0.63, High <0 |
| Version B | `step4_train_version_b.py` | 混合数据训练 | Overall 0.32, High 0.73 |
| Version C | `step5_train_version_c.py` | 数据清洗 | Overall 0.25, High 0.75 |
| 双专家V1 | `dual_expert_framework.py` | 风险分区+双专家 | Overall 0.76, High 0.64 [YES] |
| 双专家V2 | `dual_expert_v2_optimized.py` | 训练策略优化 | (待实验) |

---

## 📝 代码贡献统计

**代码行数**:
- Python代码: ~2000行
- 注释行数: ~500行
- 文档字符串: ~300行

**核心算法**:
- 双专家混合模型: 1个
- 数据处理脚本: 5个
- 实验训练脚本: 8个
- 可视化脚本: 3个

**开发周期**:
- 2024年10月: 基础模型开发
- 2024年11月上旬: Version A/B/C实验
- 2024年11月中旬: 数据清洗与补全
- 2024年11月下旬: 双专家框架实现

---

**整理日期**: 2025-11-22
**代码质量**: 遵循Claude Code开发规范
**可维护性**: 完整注释与文档
**可复现性**: 提供完整的复现脚本
