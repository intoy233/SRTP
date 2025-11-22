#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并最终数据集
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("="*60)
print("Final Dataset Merging")
print("="*60)

# 1. 读取现有扩展数据集
existing_df = pd.read_csv('../data/expanded_bridge_dataset.csv')
print(f"\nExisting dataset: {len(existing_df)} bridges")

# 2. 读取新提取的数据
new_df = pd.read_csv('extracted_bridge_data.csv')
print(f"New extracted data: {len(new_df)} bridges")

# 3. 合并数据集
merged_df = pd.concat([existing_df, new_df], ignore_index=True)
print(f"Merged dataset: {len(merged_df)} bridges")

# 4. 检查重复(基于桥梁名称)
duplicates = merged_df[merged_df.duplicated(subset=['BridgeName'], keep=False)]
if len(duplicates) > 0:
    print(f"\nWarning: Found {len(duplicates)} duplicate bridges")
    print(duplicates[['BridgeID', 'BridgeName', 'Country']].head(20))

    # 去重(保留第一次出现)
    print("\nRemoving duplicates (keeping first occurrence)...")
    merged_df = merged_df.drop_duplicates(subset=['BridgeName'], keep='first')
    print(f"After deduplication: {len(merged_df)} bridges")

# 5. 重新生成BridgeID
merged_df['BridgeID'] = [f"{i+1:03d}" for i in range(len(merged_df))]

# 6. 数据统计
print("\n" + "="*60)
print("Final Dataset Statistics")
print("="*60)

print(f"\nTotal bridges: {len(merged_df)}")
print(f"Countries: {merged_df['Country'].nunique()}")
print(f"Bridge types:")
print(merged_df['BridgeType'].value_counts())

print(f"\nStructure types:")
print(merged_df['Structure_Type'].value_counts())

# 核心字段完整性
print(f"\nCore field completeness:")
core_fields = ['BridgeName', 'Span_m', 'Width_m', 'Height_m', 'Max_Amplitude_mm', 'Natural_Freq_Hz']
for field in core_fields:
    if field in merged_df.columns:
        completeness = (merged_df[field].notna().sum() / len(merged_df)) * 100
        print(f"  {field}: {completeness:.1f}%")

# 数据统计
print(f"\nData statistics:")
if 'Max_Amplitude_mm' in merged_df.columns:
    print(f"  Amplitude range: {merged_df['Max_Amplitude_mm'].min():.1f}mm - {merged_df['Max_Amplitude_mm'].max():.1f}mm")
    print(f"  Mean amplitude: {merged_df['Max_Amplitude_mm'].mean():.1f}mm")

if 'Span_m' in merged_df.columns:
    print(f"  Span range: {merged_df['Span_m'].min():.1f}m - {merged_df['Span_m'].max():.1f}m")
    print(f"  Mean span: {merged_df['Span_m'].mean():.1f}m")

# 7. 保存最终数据集
output_file = '../data/final_bridge_dataset.csv'
merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n" + "="*60)
print(f"Final dataset saved: {output_file}")
print(f"Total bridges: {len(merged_df)}")
print("="*60)

# 8. 生成对比报告
print(f"\nDataset Growth Report:")
print(f"  Original (enhanced): 85 bridges")
print(f"  After first expansion: 88 bridges")
print(f"  After PDF extraction: {len(merged_df)} bridges")
print(f"  Total growth: +{len(merged_df) - 85} bridges (+{((len(merged_df) - 85) / 85 * 100):.1f}%)")

# 9. 保存数据增长报告
report = f"""
Final Bridge Dataset Report
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Dataset Evolution:
- Original dataset: 85 bridges
- After manual international bridges: 88 bridges (去重后)
- After PDF extraction: {len(merged_df)} bridges
- Total growth: +{len(merged_df) - 85} bridges

Final Dataset Statistics:
- Total bridges: {len(merged_df)}
- Countries: {merged_df['Country'].nunique()}
- Bridge types: {merged_df['BridgeType'].nunique()}

Bridge Type Distribution:
{merged_df['BridgeType'].value_counts().to_string()}

Country Distribution (Top 10):
{merged_df['Country'].value_counts().head(10).to_string()}

Core Field Completeness:
"""

for field in core_fields:
    if field in merged_df.columns:
        completeness = (merged_df[field].notna().sum() / len(merged_df)) * 100
        report += f"  {field}: {completeness:.1f}%\n"

report += f"""
Data Quality:
- Amplitude range: {merged_df['Max_Amplitude_mm'].min():.1f}mm - {merged_df['Max_Amplitude_mm'].max():.1f}mm
- Span range: {merged_df['Span_m'].min():.1f}m - {merged_df['Span_m'].max():.1f}m

Ready for Deep Learning: {'YES - sufficient data!' if len(merged_df) >= 100 else 'Not yet - need more data'}
{'='*60}
"""

with open('final_dataset_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\nDetailed report saved: final_dataset_report.txt")

# 10. 深度学习就绪评估
print(f"\n" + "="*60)
print("Deep Learning Readiness Assessment")
print("="*60)
if len(merged_df) >= 500:
    print("STATUS: READY for complex deep learning models!")
    print(f"  {len(merged_df)} bridges >= 500 (ideal threshold)")
elif len(merged_df) >= 300:
    print("STATUS: READY for moderate neural networks")
    print(f"  {len(merged_df)} bridges >= 300 (good threshold)")
elif len(merged_df) >= 150:
    print("STATUS: READY for simple neural networks")
    print(f"  {len(merged_df)} bridges >= 150 (minimum threshold)")
else:
    print("STATUS: Continue with ridge regression")
    print(f"  {len(merged_df)} bridges < 150 (need more data for DL)")
print("="*60)
