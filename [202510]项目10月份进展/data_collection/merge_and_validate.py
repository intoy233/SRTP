#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并新数据到现有数据集并验证质量
"""

import pandas as pd
import numpy as np

# 读取现有数据集
existing_df = pd.read_csv('../data/enhanced_bridge_dataset.csv')
print(f"现有数据集: {len(existing_df)}座桥梁")

# 读取新收集的数据
new_df = pd.read_csv('new_bridge_data.csv')
print(f"新收集数据: {len(new_df)}座桥梁")

# 合并数据集
merged_df = pd.concat([existing_df, new_df], ignore_index=True)
print(f"合并后数据集: {len(merged_df)}座桥梁")

# 重新生成BridgeID
merged_df['BridgeID'] = [f"{i+1:03d}" for i in range(len(merged_df))]

# 数据质量检查
print("\n" + "="*60)
print("数据质量报告")
print("="*60)

# 1. 桥梁类型分布
print("\n桥梁类型分布:")
print(merged_df['BridgeType'].value_counts())

# 2. 国家分布
print("\n国家分布:")
print(merged_df['Country'].value_counts())

# 3. 断面类型分布
print("\n断面类型分布:")
print(merged_df['Structure_Type'].value_counts())

# 4. 核心字段完整性
print("\n核心字段完整性:")
core_fields = ['BridgeName', 'Span_m', 'Width_m', 'Height_m', 'Max_Amplitude_mm', 'Natural_Freq_Hz']
for field in core_fields:
    completeness = (merged_df[field].notna().sum() / len(merged_df)) * 100
    print(f"  {field}: {completeness:.1f}%")

# 5. 数据统计
print("\n数据统计:")
stats_fields = ['Span_m', 'Width_m', 'Height_m', 'Max_Amplitude_mm', 'Natural_Freq_Hz', 'Damping_Ratio']
for field in stats_fields:
    if field in merged_df.columns:
        print(f"\n{field}:")
        print(f"  范围: {merged_df[field].min():.3f} - {merged_df[field].max():.3f}")
        print(f"  均值: {merged_df[field].mean():.3f}")
        print(f"  标准差: {merged_df[field].std():.3f}")

# 6. 检查重复桥梁
duplicates = merged_df[merged_df.duplicated(subset=['BridgeName'], keep=False)]
if len(duplicates) > 0:
    print(f"\nWarning: Found {len(duplicates)} duplicate bridges:")
    print(duplicates[['BridgeID', 'BridgeName', 'Country']])
    # 去重,保留第一条
    print("\nRemoving duplicates (keeping first occurrence)...")
    merged_df = merged_df.drop_duplicates(subset=['BridgeName'], keep='first')
    print(f"After deduplication: {len(merged_df)} bridges")
else:
    print("\nOK: No duplicate bridges")

# 7. 异常值检查
print("\nOutlier check:")
validation_rules = {
    'Span_m': (50, 3000),
    'Width_m': (10, 60),
    'Height_m': (1, 10),
    'Max_Amplitude_mm': (0.1, 500),
    'Natural_Freq_Hz': (0.05, 2.0),
}

for field, (min_val, max_val) in validation_rules.items():
    if field in merged_df.columns:
        outliers = merged_df[(merged_df[field] < min_val) | (merged_df[field] > max_val)]
        if len(outliers) > 0:
            print(f"  Warning: {field}: {len(outliers)} outliers")
        else:
            print(f"  OK: {field}: no outliers")

# 重新生成BridgeID(去重后)
merged_df['BridgeID'] = [f"{i+1:03d}" for i in range(len(merged_df))]

# 保存合并后的数据集
output_file = '../data/expanded_bridge_dataset.csv'
merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\nOK: Merged dataset saved: {output_file}")
print(f"Total: {len(merged_df)} bridges")
