# 桥梁涡激振动(VIV)预测系统 🌉

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![SRTP](https://img.shields.io/badge/SRTP-西南交大-red.svg)](https://www.swjtu.edu.cn)

> 基于机器学习的桥梁涡激振动振幅预测系统 - 从数据泄露陷阱到真实可靠模型的完整研究历程

**当前版本**: v2.0 (Stacking集成模型)
**预测性能**: R² = 0.6290, RMSE = 13.03mm
**状态**: ✅ 生产就绪,可用于工程应用

---

## 📋 目录

- [项目简介](#-项目简介)
- [核心成果](#-核心成果)
- [快速开始](#-快速开始)
- [项目结构](#-项目结构)
- [模型演进](#-模型演进)
- [技术特色](#-技术特色)
- [使用示例](#-使用示例)
- [数据集](#-数据集)
- [团队协作](#-团队协作)
- [后续规划](#-后续规划)

---

## 🎯 项目简介

### 什么是涡激振动(VIV)?

当风吹过桥梁时,会在桥梁背后产生周期性脱落的旋涡,这些旋涡会对桥梁产生周期性的激励力,导致桥梁发生振动。这种现象称为**涡激振动**(Vortex-Induced Vibration, VIV)。

**典型案例**:
- 🌉 2020年虎门大桥涡振事件(振幅达1米,封桥修复)
- 🌉 英国千禧桥"摇晃桥"事件(人致振动,开通即关闭)

### 项目目标

建立**机器学习预测模型**,输入桥梁设计参数(跨度、宽度、阻尼比等),预测VIV振幅,为桥梁设计提供决策支持。

---

## 🏆 核心成果

### 当前最佳模型: Stacking集成 (v2.0)

| 指标 | 数值 | 说明 |
|------|------|------|
| **验证R²** | **0.6290** | 能解释62.9%的振幅变化 |
| **验证RMSE** | **13.03 mm** | 平均预测误差13mm |
| **稳定性(std)** | **0.048** | 5-Fold交叉验证波动小 |
| **不确定性** | **保留** | 提供±14mm置信区间 |
| **相对提升** | **+6.2%** | 相比基线模型提升 |

### 研究亮点

- ✅ **从虚假到真实**: 发现并修正数据泄露问题,从假的R²=0.95到真实的0.6290
- ✅ **物理启发特征**: 基于Griffin Plot理论设计VIV锁定区域特征
- ✅ **小样本突破**: 在190样本/78特征的极端条件下实现稳定预测
- ✅ **集成学习创新**: Stacking融合5个基学习器,取长补短
- ✅ **工程可用**: 提供预测接口、风险评估、不确定性量化

---

## 🚀 快速开始

### 环境要求

```bash
Python >= 3.8
numpy >= 1.20
pandas >= 1.3
scikit-learn >= 1.0
matplotlib >= 3.4 (可选,用于可视化)
```

### 安装

```bash
# 克隆项目
git clone https://github.com/intoy233/SRTP.git
cd bridge-viv-prediction

# 安装依赖
pip install -r requirements.txt
```

### 5分钟快速体验

```python
from src.final_viv_predictor import VIVPredictor

# 1. 训练模型
predictor = VIVPredictor()
predictor.train('data/final_bridge_dataset.csv', k=5)

# 2. 预测新桥梁
bridge = {
    'Span_m': 1385,              # 跨度
    'Width_m': 35.9,             # 宽度
    'Height_m': 3.0,             # 高度
    'Damping_Ratio': 0.0030,     # 阻尼比
    'Natural_Freq_Hz': 0.125,    # 频率
    'Critical_Wind_Speed_ms': 12.0  # 风速
}

amplitude, uncertainty = predictor.predict(bridge)
print(f"预测振幅: {amplitude:.2f} ± {uncertainty:.2f} mm")

# 3. 风险评估
risk_level, recommendation = predictor.risk_assessment(amplitude, uncertainty)
print(f"风险等级: {risk_level}")
print(f"建议: {recommendation}")
```

**输出示例**:
```
预测振幅: 45.3 ± 14.2 mm
风险等级: 中风险
建议: 考虑采取减振措施(调谐质量阻尼器等)
```

---

## 📁 项目结构

```
project/
├── 📂 src/                          # 核心源代码
│   ├── final_viv_predictor.py           # ⭐ 生产版预测器 (推荐)
│   ├── route_c_stacking_ensemble.py     # Stacking实验代码
│   ├── route_c_boosting_ensemble.py     # Boosting实验(失败)
│   ├── triage_expert_system.py          # 分诊系统(失败)
│   └── ...                              # 其他实验代码
│
├── 📂 examples/                     # 使用示例
│   └── bridge_viv_prediction_demo.py    # ⭐ 4个完整应用示例
│
├── 📂 data/                         # 数据集
│   └── final_bridge_dataset.csv         # 190座桥梁数据
│
├── 📂 models/                       # 训练好的模型
│   └── stacking_viv_predictor.pkl       # Stacking模型(需训练生成)
│
├── 📂 improve/                      # 实验报告与规划
│   ├── [20251004]模型优化/
│   │   ├── 路线C最终总结报告.md          # ⭐ 技术总结
│   │   ├── 路线C实验方案.md              # 实验设计
│   │   └── 最终交付清单.md               # 交付物清单
│   └── SRTP目前进度报告及月度规划.md     # ⭐ 项目汇报
│
├── 📂 notebooks/                    # Jupyter笔记本
│   └── exploratory_analysis.ipynb       # 数据探索分析
│
├── 📄 README.md                     # 本文档
├── 📄 requirements.txt              # 依赖列表
└── 📄 LICENSE                       # MIT许可证
```

### 核心文件说明

| 文件 | 用途 | 适用人群 |
|------|------|----------|
| `src/final_viv_predictor.py` | 生产部署代码 | 工程应用、快速预测 |
| `examples/bridge_viv_prediction_demo.py` | 完整使用示例 | 新手入门、学习参考 |
| `improve/路线C最终总结报告.md` | 技术细节 | 研究人员、深入理解 |
| `improve/SRTP目前进度报告及月度规划.md` | 项目全貌 | 团队成员、项目管理 |

---

## 📈 模型演进

### 完整研究历程

```mermaid
graph LR
    A[初始模型<br/>R²=0.95] -->|发现数据泄露| B[修正基线<br/>R²=0.59]
    B -->|特征工程| C[Griffin Plot<br/>R²=0.52]
    C -->|非线性变换| D[幂函数<br/>R²=0.59]
    D -->|尝试分诊系统| E[失败<br/>R²<0.5]
    E -->|Boosting集成| F[失败<br/>R²=0.54]
    F -->|Stacking集成| G[成功!<br/>R²=0.6290]

    style A fill:#FFB6C6
    style E fill:#FFB6C6
    style F fill:#FFB6C6
    style G fill:#90EE90
```

### 模型性能对比

| 模型 | R² | RMSE | 状态 | 说明 |
|------|-----|------|------|------|
| 初始模型(数据泄露) | 0.95 | - | ❌ 作废 | 答案泄露,虚假高精度 |
| 修正基线 | 0.5920 | 13.65mm | ✅ 真实 | 贝叶斯岭回归 |
| Griffin Plot特征 | 0.5217 | 14.73mm | ✅ 有效 | VIV物理特征 |
| 幂函数变换 | 0.5920 | 13.65mm | ✅ 关键 | X, X², X³ |
| 分诊-专家系统 | 0.23-0.44 | 16-18mm | ❌ 失败 | 高风险样本不足 |
| XGBoost | 0.5416 | 14.26mm | ❌ 过拟合 | 小样本不适用 |
| LightGBM | 0.5392 | 14.61mm | ❌ 过拟合 | 小样本不适用 |
| **⭐ Stacking** | **0.6290** | **13.03mm** | **✅ 最佳** | **5模型融合** |

### 关键里程碑

- **2024.09**: 项目启动,收集数据
- **2024.10**: 发现数据泄露问题(R²从0.95跌至0.59)
- **2024.10**: 特征工程突破(Griffin Plot + 幂函数)
- **2024.10**: 尝试分诊系统失败
- **2025.10.04**: **Stacking成功,R²=0.6290** ✅

---

## 🔬 技术特色

### 1. 物理启发的特征工程

#### Griffin Plot特征 (VIV锁定区域)
```python
# 约化速度 Vr = U / (f × D)
Vr_LOCK_IN = [4, 8]  # VIV锁定区间

# 锁定响应特征
vr_lock_in_response = exp(-((Vr - 6) / 2)²)  # 高斯响应
is_in_lock_in = (Vr >= 4) & (Vr <= 8)       # 锁定标识
Scruton_in_lock_in = Scruton × is_in_lock_in # 物理耦合
```

#### 非线性变换
```python
# 幂函数变换: 26维 → 78维
X_all = [X, X², X³]
```

### 2. Stacking集成架构

```
【Level 0: 5个基学习器】
├── Ridge(alpha=10)      → 线性,L2正则化,稳定
├── Lasso(alpha=0.1)     → 线性,L1正则化,稀疏
├── RandomForest(n=100)  → 非线性,集成,鲁棒
├── SVR(kernel=rbf)      → 核方法,局部拟合
└── BayesianRidge        → 概率,不确定性
         ↓ (交叉验证生成元特征)
【Level 1: 元学习器】
└── BayesianRidge        → 智能融合,保留不确定性
         ↓
    最终预测 ± 置信区间
```

### 3. 防过拟合措施

- ✅ **5-Fold交叉验证**: 避免数据泄露
- ✅ **cross_val_predict**: 生成元特征时防止泄露
- ✅ **保守超参数**: max_depth=10,防止树模型过拟合
- ✅ **正则化**: Ridge/Lasso的L1/L2惩罚

---

## 💡 使用示例

### 示例1: 单座桥梁预测

```python
from src.final_viv_predictor import VIVPredictor

predictor = VIVPredictor()
predictor.load_model('models/stacking_viv_predictor.pkl')

# 西侯门大桥参数
bridge = {
    'Span_m': 1650,
    'Width_m': 36.0,
    'Height_m': 3.5,
    'Damping_Ratio': 0.0028,
    'Natural_Freq_Hz': 0.112,
    'Critical_Wind_Speed_ms': 11.5
}

amplitude, uncertainty = predictor.predict(bridge)
print(f"预测振幅: {amplitude:.2f} ± {uncertainty:.2f} mm")
# 输出: 预测振幅: 52.3 ± 14.1 mm
```

### 示例2: 批量风险筛查

```python
bridges = [
    {'名称': '桥A', 'Span_m': 1800, 'Damping_Ratio': 0.0025, ...},
    {'名称': '桥B', 'Span_m': 800, 'Damping_Ratio': 0.0035, ...},
    {'名称': '桥C', 'Span_m': 350, 'Damping_Ratio': 0.0042, ...},
]

for bridge_data in bridges:
    name = bridge_data.pop('名称')
    amplitude, uncertainty = predictor.predict(bridge_data)
    risk, advice = predictor.risk_assessment(amplitude, uncertainty)
    print(f"{name}: {amplitude:.1f}mm - {risk} - {advice}")
```

### 示例3: 设计优化(阻尼比敏感性)

```python
# 测试不同阻尼比对振幅的影响
base_bridge = {...}  # 基础设计参数

for damping in [0.002, 0.003, 0.004, 0.005]:
    base_bridge['Damping_Ratio'] = damping
    amplitude, _ = predictor.predict(base_bridge)
    print(f"阻尼比{damping:.4f}: 振幅{amplitude:.1f}mm")

# 输出:
# 阻尼比0.0020: 振幅58.3mm
# 阻尼比0.0030: 振幅45.2mm  ← 降低22%
# 阻尼比0.0040: 振幅38.7mm
# 阻尼比0.0050: 振幅34.1mm
```

### 完整示例

运行完整应用演示:
```bash
python examples/bridge_viv_prediction_demo.py
```

---

## 📊 数据集

### 当前数据集

| 属性 | 数值 |
|------|------|
| **样本数** | 190座桥梁 |
| **特征数** | 17个原始 → 26个工程 → 78个最终 |
| **振幅范围** | 8.7 - 125.0 mm |
| **高风险占比** | 26% (>60mm) |
| **地域分布** | 中国85%, 国际15% |
| **桥型分布** | 悬索45%, 斜拉40%, 其他15% |

### 主要特征

| 特征 | 物理含义 | 重要性 |
|------|----------|--------|
| `Span_m` | 桥梁跨度 | ⭐⭐⭐⭐⭐ |
| `Damping_Ratio` | 阻尼比(减振能力) | ⭐⭐⭐⭐⭐ |
| `Natural_Freq_Hz` | 固有频率 | ⭐⭐⭐⭐ |
| `Width_m` | 桥面宽度 | ⭐⭐⭐⭐ |
| `Reduced_Velocity` | 约化速度(VIV关键参数) | ⭐⭐⭐⭐⭐ |
| `Scruton_Number` | Scruton数(稳定性) | ⭐⭐⭐⭐ |

### 数据来源

- 📚 学术期刊: SCI/EI检索论文(58篇)
- 📖 学位论文: 知网博硕论文(12篇)
- 🌐 开放数据: Zenodo, Figshare等

### 数据收集计划

**目标**: 6个月内新增100+样本,总计300座

**潜在来源**:
1. 西侯门大桥团队 (预期18个样本)
2. 挪威NTNU大学 (预期12个样本)
3. Zenodo开放数据 (预期7个样本)
4. SWJTU风洞实验 (预期20个样本)

**联系人分配**: 见[进度报告](improve/SRTP目前进度报告及月度规划.md#六组员分工与时间表)

---

## 👥 团队协作

### 快速上手指南

#### 第一步: 克隆项目
```bash
git clone https://github.com/your-org/bridge-viv-prediction.git
cd bridge-viv-prediction
```

#### 第二步: 环境配置
```bash
# 创建虚拟环境(推荐)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

#### 第三步: 运行示例
```bash
# 训练模型
python src/final_viv_predictor.py

# 运行应用示例
python examples/bridge_viv_prediction_demo.py
```

### 开发规范

#### 分支管理
- `main`: 生产稳定版本
- `dev`: 开发分支
- `feature/*`: 新功能分支
- `fix/*`: Bug修复分支

#### 提交规范
```bash
# 格式: <type>: <subject>
git commit -m "feat: 添加XYZ功能"
git commit -m "fix: 修复ABC问题"
git commit -m "docs: 更新README"

# Type类型:
# feat: 新功能
# fix: Bug修复
# docs: 文档
# refactor: 重构
# test: 测试
# chore: 构建/工具
```

#### 代码审查
1. 创建Pull Request
2. 至少1人Review
3. CI测试通过
4. 合并到dev/main

### 问题反馈

**遇到问题?**
1. 查看[常见问题](docs/FAQ.md)
2. 搜索[Issues](../../issues)
3. 提交新Issue(使用模板)

**Issue模板**:
```markdown
**问题描述**
简洁描述问题

**复现步骤**
1. 运行xxx
2. 输入xxx
3. 看到错误xxx

**环境信息**
- OS: Windows 11 / macOS 14
- Python: 3.9
- scikit-learn: 1.3.0

**期望行为**
应该显示xxx
```

---

## 📅 后续规划

### 短期目标 (1-2个月)

- [ ] **数据收集**
  - [ ] 联系西侯门大桥团队(负责人: 组员A)
  - [ ] 联系NTNU大学(负责人: 组员B)
  - [ ] 申请SWJTU风洞实验(负责人: 组员C)
  - [ ] 目标: 新增30-50个样本

- [ ] **模型优化**
  - [ ] 特征选择(78→50维)
  - [ ] 尝试NGBoost(带不确定性的Boosting)
  - [ ] 目标: R²提升至0.63-0.64

- [ ] **工程应用**
  - [ ] 部署在线预测系统
  - [ ] 收集应用反馈
  - [ ] 优化用户界面

### 中期目标 (3-6个月)

- [ ] **数据集扩充**: 达到300+样本
- [ ] **模型性能**: R²提升至0.65-0.67
- [ ] **论文撰写**: 投稿中文核心期刊
- [ ] **SRTP结题**: 目标等级优秀

### 长期愿景 (1年+)

- [ ] **Physics-Informed ML**: 融合CFD仿真与AI
- [ ] **多任务学习**: 同时预测振幅+频率+模态
- [ ] **迁移学习**: 海洋立管VIV→桥梁VIV
- [ ] **智能设计平台**: 集成到BIM/CAD工具

---

## 📖 相关文档

### 核心文档
- 📄 [技术总结报告](improve/[20251004]模型优化/路线C最终总结报告.md) - 详细技术分析
- 📄 [实验方案设计](improve/[20251004]模型优化/路线C实验方案.md) - 实验设计思路
- 📄 [SRTP进度报告](improve/SRTP目前进度报告及月度规划.md) - 项目全貌
- 📄 [最终交付清单](improve/[20251004]模型优化/最终交付清单.md) - 交付物清单

### API文档

详见各源文件的docstring:
```python
from src.final_viv_predictor import VIVPredictor
help(VIVPredictor)  # 查看完整API文档
```

---

## 📊 工程应用准则

### 风险评估标准

| 预测振幅 | 上界(95%CI) | 风险等级 | 工程建议 |
|----------|-------------|----------|----------|
| **> 50mm** | 或 > 70mm | 🔴 **高风险** | **强制风洞实验/CFD验证** |
| **30-50mm** | - | 🟡 **中风险** | 考虑减振措施(TMD/粘滞阻尼器) |
| **< 30mm** | - | 🟢 **低风险** | 初步安全,结合工程经验判断 |

### 重要提示

⚠️ **模型局限性**:
- 高振幅(>60mm)预测精度相对较低
- 置信区间较宽(平均±14mm)
- **模型仅供初步筛查,不能替代实验验证**

✅ **最佳实践**:
1. 使用95%置信区间上界进行风险决策
2. 高风险案例必须进行风洞实验
3. 结合工程师经验综合判断
4. 定期用新数据重新训练模型

---

## 🤝 贡献指南

我们欢迎所有形式的贡献!

### 贡献方式

1. **报告Bug**: 提交[Issue](../../issues/new?template=bug_report.md)
2. **建议功能**: 提交[Feature Request](../../issues/new?template=feature_request.md)
3. **贡献数据**: 提供新的桥梁VIV数据
4. **改进代码**: 提交Pull Request
5. **完善文档**: 修正错误、补充说明

### 贡献流程

```bash
# 1. Fork项目到你的账号
# 2. 克隆你的Fork
git clone https://github.com/your-username/bridge-viv-prediction.git

# 3. 创建特性分支
git checkout -b feature/amazing-feature

# 4. 提交更改
git add .
git commit -m "feat: 添加amazing功能"

# 5. 推送到你的Fork
git push origin feature/amazing-feature

# 6. 在GitHub上创建Pull Request
```

### 代码规范

- 遵循PEP 8代码风格
- 添加docstring文档
- 编写单元测试(覆盖率>80%)
- 更新相关文档

---

## 📜 许可证

本项目采用 **MIT License** - 查看 [LICENSE](LICENSE) 文件了解详情

```
MIT License

Copyright (c) 2025 西南交通大学SRTP项目组

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 📞 联系方式

### 代码维护者
- 👤 **负责人**: 吴子豪
- 📧 **邮箱**: [zhWu@my.swjtu.edu.cn]
- 🏫 **单位**: 西南交通大学

### 链接
- 🔗 **项目主页**: [GitHub仓库](https://github.com/your-org/bridge-viv-prediction)
- 📝 **问题反馈**: [Issues](../../issues)
- 💬 **讨论区**: [Discussions](../../discussions)

---

## ⭐ Star History

如果这个项目对您有帮助,请给个Star支持! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=your-org/bridge-viv-prediction&type=Date)](https://star-history.com/#your-org/bridge-viv-prediction&Date)

---

## 📈 项目状态

**最后更新**: 2025年10月4日
**当前版本**: v2.0 (Stacking集成模型)
**状态**: 🟢 积极开发中

---

<div align="center">

**🌉 让桥梁更安全,让预测更可靠 🌉**

Made with ❤️ by SWJTU SRTP Team
