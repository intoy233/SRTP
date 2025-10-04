#!/usr/bin/env python3
"""检查Risk_Level数据分布"""

import pandas as pd

df = pd.read_csv('../data/final_bridge_dataset.csv')

print("="*80)
print("Risk_Level 数据分布分析")
print("="*80)

print(f"\n总样本数: {len(df)}")
print(f"Risk_Level缺失数: {df['Risk_Level'].isna().sum()} ({df['Risk_Level'].isna().sum()/len(df)*100:.1f}%)")

print(f"\nRisk_Level分布:")
risk_counts = df['Risk_Level'].value_counts()
for risk, count in risk_counts.items():
    print(f"  {risk:10s}: {count:3d} ({count/len(df.dropna(subset=['Risk_Level']))*100:.1f}%)")

print(f"\n与Max_Amplitude共存的样本数: {df.dropna(subset=['Risk_Level', 'Max_Amplitude_mm']).shape[0]}")
