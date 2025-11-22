#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并新数据与原始数据集
执行去重检查、数据清洗、最终合并
"""

import pandas as pd
import numpy as np
from pathlib import Path


def merge_datasets(original_path: str, new_path: str, output_path: str) -> bool:
    """合并原始数据集和新收集数据"""
    print("=" * 80)
    print("Merging Original and New Bridge VIV Datasets")
    print("=" * 80)

    try:
        # 1. 读取数据
        print("Step 1: Loading datasets...")
        df_original = pd.read_csv(original_path, encoding='utf-8-sig')
        df_new = pd.read_csv(new_path, encoding='utf-8-sig')

        print(f"  Original dataset: {len(df_original)} samples")
        print(f"  New dataset: {len(df_new)} samples")
        print(f"  Expected total: {len(df_original) + len(df_new)} samples\n")

        # 2. 去重检查
        print("Step 2: Duplicate check...")
        print("  Method 1: Exact name + country match")

        # 标准化桥梁名称用于匹配
        df_original['Name_Clean'] = df_original['BridgeName'].str.strip().str.lower()
        df_new['Name_Clean'] = df_new['BridgeName'].str.strip().str.lower()

        # 查找重复
        duplicates = []
        for idx, row in df_new.iterrows():
            matches = df_original[
                (df_original['Name_Clean'] == row['Name_Clean']) &
                (df_original['Country'] == row['Country'])
            ]
            if len(matches) > 0:
                duplicates.append({
                    'new_id': row['BridgeID'],
                    'new_name': row['BridgeName'],
                    'country': row['Country'],
                    'original_id': matches.iloc[0]['BridgeID']
                })

        print(f"  Found {len(duplicates)} exact duplicates:")
        if len(duplicates) > 0:
            for dup in duplicates[:10]:  # 只显示前10个
                print(f"    - {dup['new_name']} ({dup['country']}): "
                      f"New ID={dup['new_id']}, Original ID={dup['original_id']}")
            if len(duplicates) > 10:
                print(f"    ... and {len(duplicates) - 10} more")

        # Method 2: 宽松匹配 (相似跨度和频率)
        print("\n  Method 2: Fuzzy match (Span ±5%, Freq ±5%)")
        fuzzy_duplicates = []

        for idx, row in df_new.iterrows():
            # 跳过已确认的精确重复
            if any(d['new_id'] == row['BridgeID'] for d in duplicates):
                continue

            matches = df_original[
                (np.abs(df_original['Span_m'] - row['Span_m']) / row['Span_m'] < 0.05) &
                (np.abs(df_original['Natural_Freq_Hz'] - row['Natural_Freq_Hz']) / row['Natural_Freq_Hz'] < 0.05) &
                (df_original['Country'] == row['Country'])
            ]

            if len(matches) > 0:
                fuzzy_duplicates.append({
                    'new_id': row['BridgeID'],
                    'new_name': row['BridgeName'],
                    'new_span': row['Span_m'],
                    'new_freq': row['Natural_Freq_Hz'],
                    'original_id': matches.iloc[0]['BridgeID'],
                    'original_name': matches.iloc[0]['BridgeName'],
                    'original_span': matches.iloc[0]['Span_m'],
                    'original_freq': matches.iloc[0]['Natural_Freq_Hz']
                })

        print(f"  Found {len(fuzzy_duplicates)} fuzzy duplicates:")
        if len(fuzzy_duplicates) > 0:
            for dup in fuzzy_duplicates[:5]:
                print(f"    - New: {dup['new_name']} (Span={dup['new_span']:.1f}m, Freq={dup['new_freq']:.3f}Hz)")
                print(f"      vs Original: {dup['original_name']} (Span={dup['original_span']:.1f}m, Freq={dup['original_freq']:.3f}Hz)")

        # 决策: 移除重复
        dup_ids = [d['new_id'] for d in duplicates]
        if len(dup_ids) > 0:
            print(f"\n  Removing {len(dup_ids)} exact duplicates from new dataset...")
            df_new = df_new[~df_new['BridgeID'].isin(dup_ids)]

        # 对于模糊重复,保留(可能是同一桥梁的不同工况/实验)
        if len(fuzzy_duplicates) > 0:
            print(f"\n  Keeping {len(fuzzy_duplicates)} fuzzy matches (likely different test conditions)")

        # 删除临时列
        df_original.drop(columns=['Name_Clean'], inplace=True)
        df_new.drop(columns=['Name_Clean'], inplace=True)

        # 3. BridgeID冲突处理
        print("\n\nStep 3: Resolving BridgeID conflicts...")
        max_original_id = df_original['BridgeID'].max()
        print(f"  Original max ID: {max_original_id}")
        print(f"  New dataset ID range: {df_new['BridgeID'].min()}-{df_new['BridgeID'].max()}")

        # 重新分配新数据的ID
        df_new['BridgeID'] = range(max_original_id + 1, max_original_id + 1 + len(df_new))
        print(f"  Reassigned new IDs: {df_new['BridgeID'].min()}-{df_new['BridgeID'].max()}")

        # 4. 合并数据
        print("\n\nStep 4: Merging datasets...")
        df_merged = pd.concat([df_original, df_new], ignore_index=True)
        print(f"  Merged dataset: {len(df_merged)} samples")

        # 5. 最终数据质量检查
        print("\n\nStep 5: Final data quality check...")

        # 检查必需字段
        required_cols = [
            'BridgeID', 'BridgeName', 'BridgeType', 'Country', 'Span_m',
            'Width_m', 'Height_m', 'Natural_Freq_Hz', 'Critical_Wind_Speed_ms',
            'Max_Amplitude_mm', 'Damping_Ratio'
        ]

        for col in required_cols:
            missing = df_merged[col].isnull().sum()
            missing_pct = missing / len(df_merged) * 100
            print(f"  {col}: {missing} missing ({missing_pct:.1f}%)")

        # 振幅分布
        print("\n  Amplitude distribution:")
        low = (df_merged['Max_Amplitude_mm'] <= 30).sum()
        medium = ((df_merged['Max_Amplitude_mm'] > 30) & (df_merged['Max_Amplitude_mm'] <= 60)).sum()
        high = ((df_merged['Max_Amplitude_mm'] > 60) & (df_merged['Max_Amplitude_mm'] <= 100)).sum()
        critical = (df_merged['Max_Amplitude_mm'] > 100).sum()

        print(f"    Low (<=30mm): {low} ({low/len(df_merged)*100:.1f}%)")
        print(f"    Medium (30-60mm): {medium} ({medium/len(df_merged)*100:.1f}%)")
        print(f"    High (60-100mm): {high} ({high/len(df_merged)*100:.1f}%)")
        print(f"    Critical (>100mm): {critical} ({critical/len(df_merged)*100:.1f}%)")

        high_risk_total = high + critical
        print(f"\n    High-risk total (>60mm): {high_risk_total} ({high_risk_total/len(df_merged)*100:.1f}%)")

        original_high_risk = ((df_original['Max_Amplitude_mm'] > 60)).sum()
        print(f"    Improvement: {original_high_risk} -> {high_risk_total} (+{high_risk_total - original_high_risk})")

        # 6. 保存合并数据
        print(f"\n\nStep 6: Saving merged dataset...")
        df_merged.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"  Saved to: {output_path}")

        # 7. 生成统计报告
        print("\n\n" + "=" * 80)
        print("MERGE SUMMARY")
        print("=" * 80)
        print(f"Original dataset:        {len(df_original)} samples")
        print(f"New dataset:             {len(df_new)} samples")
        print(f"Exact duplicates removed: {len(duplicates)} samples")
        print(f"Final merged dataset:    {len(df_merged)} samples")
        print(f"\nHigh-risk samples:       {original_high_risk} -> {high_risk_total} (+{high_risk_total - original_high_risk}, +{(high_risk_total - original_high_risk)/original_high_risk*100:.1f}%)")
        print(f"\nSample/Feature ratio:    {len(df_merged)}/78 = {len(df_merged)/78:.2f}")
        print(f"  Baseline: 2.51 (insufficient)")
        print(f"  Current:  {len(df_merged)/78:.2f} ({len(df_merged)/78 - 2.51:+.2f} improvement)")
        print(f"  Target:   5.00 (recommended)")

        if len(df_merged) / 78 >= 5.0:
            print("\n  STATUS: SUFFICIENT for 78-feature model training!")
        elif len(df_merged) / 78 >= 3.85:
            print("\n  STATUS: ACCEPTABLE (above minimum threshold)")
        else:
            print("\n  STATUS: STILL INSUFFICIENT (need more samples)")

        print("=" * 80)

        return True

    except Exception as e:
        print(f"ERROR: Merge failed - {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    project_root = Path(__file__).parent.parent

    original_csv = project_root / 'data' / 'final_bridge_dataset.csv'
    new_csv = project_root / 'notebooks' / '[20251118]改进实验' / '07-新收集数据-完整.csv'
    output_csv = project_root / 'data' / 'final_bridge_dataset_merged.csv'

    if not original_csv.exists():
        print(f"ERROR: Original dataset not found: {original_csv}")
        exit(1)

    if not new_csv.exists():
        print(f"ERROR: New dataset not found: {new_csv}")
        exit(1)

    success = merge_datasets(str(original_csv), str(new_csv), str(output_csv))
    exit(0 if success else 1)
