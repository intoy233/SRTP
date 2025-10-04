#!/usr/bin/env python3
"""
诊断Scruton定律在我们数据集中的适用性
检查 Max_Amplitude 与 1/Scruton_Number 的关系
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 加载数据
df = pd.read_csv('../data/final_bridge_dataset.csv')

# 计算Scruton Number
df['Scruton_Number'] = df['Damping_Ratio'] * (df['Width_m'] / df['Height_m']) * 100

# 移除缺失值
df_clean = df[['Max_Amplitude_mm', 'Scruton_Number']].dropna()

print("="*80)
print("Scruton定律适用性诊断")
print("="*80)

print(f"\n样本数: {len(df_clean)}")
print(f"\nScruton_Number统计:")
print(f"  范围: {df_clean['Scruton_Number'].min():.2f} - {df_clean['Scruton_Number'].max():.2f}")
print(f"  均值: {df_clean['Scruton_Number'].mean():.2f}")
print(f"  中位数: {df_clean['Scruton_Number'].median():.2f}")
print(f"  标准差: {df_clean['Scruton_Number'].std():.2f}")

print(f"\nMax_Amplitude统计:")
print(f"  范围: {df_clean['Max_Amplitude_mm'].min():.2f} - {df_clean['Max_Amplitude_mm'].max():.2f} mm")
print(f"  均值: {df_clean['Max_Amplitude_mm'].mean():.2f} mm")
print(f"  中位数: {df_clean['Max_Amplitude_mm'].median():.2f} mm")
print(f"  标准差: {df_clean['Max_Amplitude_mm'].std():.2f} mm")

# 检查 Amplitude vs 1/Scruton 的关系
df_clean['Scruton_Inv'] = 1.0 / df_clean['Scruton_Number']

# 线性回归拟合
X = df_clean[['Scruton_Inv']].values
y = df_clean['Max_Amplitude_mm'].values

lr = LinearRegression()
lr.fit(X, y)

y_pred = lr.predict(X)
r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)

print(f"\n" + "="*80)
print("Scruton定律验证: Max_Amplitude = k / Scruton_Number")
print("="*80)
print(f"\n线性拟合: Max_Amplitude = {lr.coef_[0]:.2f} * (1/Scruton_Number) + {lr.intercept_:.2f}")
print(f"R2得分: {r2:.4f}")

if r2 > 0.3:
    print("结论: Scruton定律在数据中有一定适用性 OK")
    print(f"建议k_scruton参数: {lr.coef_[0]:.2f}")
else:
    print("结论: Scruton定律在数据中几乎不适用 FAIL")
    print("原因: Max_Amplitude受多种因素影响,不仅是Scruton_Number")

# 计算不同k值下的预测误差
print(f"\n" + "="*80)
print("测试不同k_scruton参数的拟合质量")
print("="*80)

for k in [100, 200, 500, 1000, 2000, lr.coef_[0]]:
    y_phys = k / df_clean['Scruton_Number']
    mse = np.mean((y - y_phys)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y - y_phys))

    print(f"k={k:7.1f}: RMSE={rmse:7.2f} mm, MAE={mae:6.2f} mm")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 子图1: Amplitude vs Scruton_Number
axes[0].scatter(df_clean['Scruton_Number'], df_clean['Max_Amplitude_mm'], alpha=0.6, s=50)
axes[0].set_xlabel('Scruton Number', fontsize=12)
axes[0].set_ylabel('Max Amplitude (mm)', fontsize=12)
axes[0].set_title('Amplitude vs Scruton Number', fontsize=14)
axes[0].grid(True, alpha=0.3)

# 子图2: Amplitude vs 1/Scruton_Number
axes[1].scatter(df_clean['Scruton_Inv'], df_clean['Max_Amplitude_mm'], alpha=0.6, s=50, label='实际数据')
axes[1].plot(df_clean['Scruton_Inv'], y_pred, 'r-', linewidth=2, label=f'线性拟合 (R2={r2:.3f})')
axes[1].set_xlabel('1 / Scruton Number', fontsize=12)
axes[1].set_ylabel('Max Amplitude (mm)', fontsize=12)
axes[1].set_title('Amplitude vs 1/Scruton (Scruton定律)', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/scruton_law_diagnosis.png', dpi=150, bbox_inches='tight')
print(f"\n图像已保存: ../results/scruton_law_diagnosis.png")

# 检查特征之间的相关性
print(f"\n" + "="*80)
print("特征相关性分析")
print("="*80)

# 需要的其他特征
feature_cols = ['Damping_Ratio', 'Width_m', 'Height_m', 'Span_m',
                'Natural_Freq_Hz', 'Critical_Wind_Speed_ms', 'Max_Amplitude_mm']
df_features = df[feature_cols].dropna()

corr_with_amp = df_features.corr()['Max_Amplitude_mm'].sort_values(ascending=False)
print("\n与Max_Amplitude相关性最强的特征:")
for feature, corr in corr_with_amp.items():
    if feature != 'Max_Amplitude_mm':
        print(f"  {feature:30s}: {corr:+.4f}")

print("\n" + "="*80)
print("核心结论")
print("="*80)
print("1. Scruton定律(Max_Amp = k/Sc)仅能解释部分方差")
print(f"2. 单纯使用Scruton约束作为物理损失会破坏模型")
print("3. VIV振幅受多参数耦合影响,不能简化为单一定律")
print("4. 建议: 使用更弱的物理约束,或改用软性惩罚")
print("="*80)
