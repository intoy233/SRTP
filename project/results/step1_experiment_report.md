# 实验步骤1: 线性回归模型

**数据集**: 80座桥梁
**有效样本**: 80个
**特征数**: 9个
**训练集**: 64个样本
**测试集**: 16个样本

## 模型性能

### 振幅预测 (回归)

- **训练集 R²**: 0.1035
- **测试集 R²**: 0.0804
- **训练集 RMSE**: 16.0072mm
- **测试集 RMSE**: 17.4789mm

### 风险分类 (分类)

- **风险分类准确率**: 0.4375

## 特征重要性

1. **Width_Height_Ratio**: 29.9074
2. **Width_m**: 22.3454
3. **Height_m**: 18.3663
4. **Critical_Wind_Speed_ms**: 8.3337
5. **VIV_Wind_Speed_ms**: 7.4581
6. **First_Freq_Hz**: 2.6741
7. **Damping_Ratio**: 2.5192
8. **Span_m**: 1.0251
9. **Natural_Freq_Hz**: 0.6472

**实验时间**: 2024
**模型文件**: experiments\step1_linear_model.json
**预测样例**: results\step1_predictions_sample.json
