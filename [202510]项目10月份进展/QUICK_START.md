# 快速开始指南 🚀

> 5分钟上手桥梁VIV预测系统

---

## 📋 前置要求

- ✅ Python 3.8 或更高版本
- ✅ pip 包管理器
- ✅ 10GB 可用磁盘空间

---

## 🔧 步骤1: 克隆项目

```bash
# 克隆项目到本地
git clone https://github.com/your-org/bridge-viv-prediction.git
cd bridge-viv-prediction
```

---

## 📦 步骤2: 安装依赖

### 方式1: 完整安装 (推荐)

```bash
pip install -r requirements.txt
```

### 方式2: 最小化安装 (仅核心功能)

```bash
pip install numpy pandas scikit-learn matplotlib
```

### 验证安装

```bash
python -c "import numpy, pandas, sklearn; print('安装成功!')"
```

---

## 🎯 步骤3: 运行第一个预测

### 3.1 训练模型

```bash
# 训练Stacking集成模型 (约2-3分钟)
python src/final_viv_predictor.py
```

**输出示例**:
```
================================================================================
VIV振幅预测器 - 训练Stacking模型
================================================================================

数据集: 190 座桥梁
特征维度: 78 (26基础 + 26² + 26³)
有效样本: 190

开始5-Fold交叉验证训练...
  Fold 1: R²=0.6570, RMSE=12.91mm
  Fold 2: R²=0.5373, RMSE=11.90mm
  Fold 3: R²=0.6413, RMSE=13.76mm
  Fold 4: R²=0.6757, RMSE=13.87mm
  Fold 5: R²=0.6338, RMSE=12.73mm

================================================================================
训练完成!
================================================================================
交叉验证R²: 0.6290 (±0.0481)
交叉验证RMSE: 13.03 mm
================================================================================

模型已保存至: ../models/stacking_viv_predictor.pkl
```

### 3.2 运行应用示例

```bash
# 运行完整应用演示
python examples/bridge_viv_prediction_demo.py
```

**你会看到**:
1. 单座桥梁预测
2. 批量风险筛查
3. 设计优化分析
4. 不确定性量化

---

## 💡 步骤4: 预测你自己的桥梁

创建文件 `my_prediction.py`:

```python
from src.final_viv_predictor import VIVPredictor

# 加载训练好的模型
predictor = VIVPredictor()
predictor.load_model('models/stacking_viv_predictor.pkl')

# 输入你的桥梁参数
my_bridge = {
    'Span_m': 1200,              # 主跨长度 (米)
    'Width_m': 35,               # 桥面宽度 (米)
    'Height_m': 3.0,             # 主梁高度 (米)
    'Damping_Ratio': 0.0030,     # 阻尼比
    'Natural_Freq_Hz': 0.135,    # 自振频率 (Hz)
    'Critical_Wind_Speed_ms': 12.8  # 临界风速 (m/s)
}

# 预测VIV振幅
amplitude, uncertainty = predictor.predict(my_bridge)

print(f"\n【预测结果】")
print(f"VIV振幅: {amplitude:.2f} mm")
print(f"不确定性: ±{uncertainty:.2f} mm")
print(f"95%置信区间: [{amplitude-1.96*uncertainty:.2f}, {amplitude+1.96*uncertainty:.2f}] mm")

# 风险评估
risk_level, recommendation = predictor.risk_assessment(amplitude, uncertainty)

print(f"\n【风险评估】")
print(f"风险等级: {risk_level}")
print(f"工程建议: {recommendation}")
```

运行:
```bash
python my_prediction.py
```

---

## 📊 理解输出结果

### 预测振幅
- **含义**: 桥梁在临界风速下的最大振动幅度
- **单位**: mm (毫米)
- **例子**: 45.3mm = 4.53厘米

### 不确定性 (±14mm)
- **含义**: 预测的置信区间范围
- **95%置信区间**: 有95%的概率真实振幅在 [预测值-1.96×不确定性, 预测值+1.96×不确定性] 范围内

### 风险等级
- 🟢 **低风险** (<30mm): 初步安全
- 🟡 **中风险** (30-50mm): 建议采取减振措施
- 🔴 **高风险** (>50mm 或 上界>70mm): **强制进行风洞实验**

---

## 🎓 下一步学习

### 初学者
1. 阅读 [README.md](README.md) - 了解项目全貌
2. 查看 [使用示例](examples/bridge_viv_prediction_demo.py) - 学习更多用法
3. 运行 Jupyter Notebook - 交互式探索数据

### 进阶用户
1. 阅读 [技术总结报告](improve/[20251004]模型优化/路线C最终总结报告.md) - 深入理解模型
2. 阅读 [实验方案](improve/[20251004]模型优化/路线C实验方案.md) - 了解设计思路
3. 修改超参数重新训练 - 尝试优化模型

### 研究人员
1. 阅读 [SRTP进度报告](improve/SRTP目前进度报告及月度规划.md) - 了解完整研究历程
2. 查看失败实验代码 - 学习避坑经验
3. 贡献新数据或算法 - 参与项目改进

---

## 🆘 常见问题

### Q1: 模型预测不准确怎么办?
**A**:
- 检查输入参数是否在合理范围内
- 高振幅(>60mm)预测精度相对较低,属正常
- 必须结合工程经验综合判断
- **重要**: 高风险案例必须风洞实验验证

### Q2: 缺少某个特征怎么办?
**A**:
- 必需特征: `Span_m`, `Width_m`, `Height_m`, `Damping_Ratio`, `Natural_Freq_Hz`, `Critical_Wind_Speed_ms`
- 如果缺失,可以参考类似桥梁或使用典型值估算
- 但会引入误差,不确定性会增大

### Q3: 如何提高预测精度?
**A**:
- **短期**: 调整超参数 (见技术报告)
- **中期**: 收集更多数据重新训练
- **长期**: 尝试新算法 (Physics-Informed ML等)

### Q4: 可以预测其他桥梁振动吗?
**A**:
- 当前仅支持VIV (涡激振动)
- 不支持颤振、抖振等其他风致振动
- 未来计划扩展到多种振动类型

### Q5: 模型可以商用吗?
**A**:
- 本项目采用MIT许可证,可免费商用
- 但**必须注明来源**
- **强烈建议**: 商用前进行额外验证

---

## 📞 获取帮助

遇到问题? 可以:

1. 📖 查看 [完整文档](README.md)
2. 🔍 搜索 [Issues](../../issues)
3. 💬 发起 [Discussions](../../discussions)
4. 📧 联系维护者: [your-email@swjtu.edu.cn]

---

## ✅ 检查清单

完成快速开始后,你应该能够:

- [ ] 成功安装所有依赖
- [ ] 训练Stacking模型
- [ ] 运行应用示例
- [ ] 预测自己的桥梁VIV振幅
- [ ] 理解预测结果和风险评估
- [ ] 知道如何获取进一步帮助

---

<div align="center">

**🎉 恭喜你完成快速开始! 🎉**

现在你已经掌握了基本用法,可以探索更多高级功能了!

[← 返回README](README.md) | [查看完整示例 →](examples/bridge_viv_prediction_demo.py)

</div>
