#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新数据验证脚本 - Validate New Bridge VIV Data
用于验证Gemini收集的新桥梁VIV数据是否符合模型输入要求
"""

import pandas as pd
import numpy as np
from pathlib import Path


def validate_bridge_data(csv_path: str) -> bool:
    """验证新收集的桥梁VIV数据"""
    print("=" * 80)
    print("Bridge VIV Data Validation Script")
    print("=" * 80)
    print(f"Validating file: {csv_path}\n")

    # 1. 读取数据
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        print(f"File loaded successfully: {len(df)} samples")
    except Exception as e:
        print(f"ERROR: Failed to read file: {e}")
        return False

    # 2. 检查必需字段
    required_cols = [
        'BridgeID', 'BridgeName', 'BridgeType', 'Country', 'PaperSource',
        'Year', 'Span_m', 'Width_m', 'Height_m', 'Width_Height_Ratio',
        'Total_Length_m', 'Structure_Type', 'Natural_Freq_Hz', 'First_Freq_Hz',
        'Second_Freq_Hz', 'Drag_Coefficient', 'Lift_Coefficient',
        'VIV_Wind_Speed_ms', 'Critical_Wind_Speed_ms', 'Max_Amplitude_mm',
        'Amplitude_RMS_mm', 'Damping_Ratio', 'Vibration_Suppression',
        'Suppression_Effect', 'Risk_Level', 'Notes'
    ]

    print("\n1. Checking Required Fields")
    print("-" * 80)
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        print(f"ERROR: Missing required fields: {missing_cols}")
        return False
    print(f"PASS: All {len(required_cols)} required fields present")

    # 3. 检查核心字段缺失率
    print("\n2. Checking Core Fields Missing Rate")
    print("-" * 80)
    core_fields = {
        'Damping_Ratio': 10,
        'Critical_Wind_Speed_ms': 10,
        'Max_Amplitude_mm': 0,  # 目标变量不能缺失
        'Span_m': 5,
        'Width_m': 5,
        'Natural_Freq_Hz': 10
    }

    all_pass = True
    for field, max_missing_pct in core_fields.items():
        missing_count = df[field].isnull().sum()
        missing_rate = missing_count / len(df) * 100

        if missing_rate > max_missing_pct:
            print(f"  WARNING: '{field}' missing rate {missing_rate:.1f}% "
                  f"(threshold: {max_missing_pct}%)")
            if field == 'Max_Amplitude_mm':
                print("    CRITICAL: Target variable cannot have missing values!")
                all_pass = False
        else:
            print(f"  PASS: '{field}' missing rate {missing_rate:.1f}%")

    if not all_pass:
        return False

    # 4. 检查数值范围
    print("\n3. Checking Value Ranges")
    print("-" * 80)
    range_checks = {
        'Span_m': (16, 5000, 'm'),
        'Width_m': (4, 60, 'm'),
        'Height_m': (2.3, 55, 'm'),
        'Natural_Freq_Hz': (0.049, 0.962, 'Hz'),
        'Critical_Wind_Speed_ms': (5.1, 22, 'm/s'),
        'Max_Amplitude_mm': (0, 200, 'mm'),
        'Damping_Ratio': (0.005, 0.058, '-')
    }

    for field, (min_val, max_val, unit) in range_checks.items():
        if field in df.columns:
            valid = df[field].dropna()
            if len(valid) == 0:
                continue

            out_of_range = ((valid < min_val) | (valid > max_val)).sum()
            if out_of_range > 0:
                print(f"  WARNING: '{field}' has {out_of_range} values out of range "
                      f"[{min_val}, {max_val}] {unit}")
                outliers = valid[(valid < min_val) | (valid > max_val)]
                print(f"    Outlier values: {outliers.tolist()[:5]}")  # 显示前5个
            else:
                print(f"  PASS: '{field}' range [{min_val}, {max_val}] {unit}")

    # 5. 物理一致性检查
    print("\n4. Checking Physical Consistency")
    print("-" * 80)
    inconsistencies = 0

    # 检查宽高比
    if all(c in df.columns for c in ['Width_m', 'Height_m', 'Width_Height_Ratio']):
        df['calc_ratio'] = df['Width_m'] / df['Height_m']
        valid_mask = df[['Width_m', 'Height_m', 'Width_Height_Ratio']].notna().all(axis=1)
        if valid_mask.sum() > 0:
            ratio_error = np.abs(df.loc[valid_mask, 'calc_ratio'] -
                                df.loc[valid_mask, 'Width_Height_Ratio']) / df.loc[valid_mask, 'calc_ratio']
            bad_ratio = (ratio_error > 0.05).sum()
            if bad_ratio > 0:
                inconsistencies += bad_ratio
                print(f"  WARNING: {bad_ratio} samples have Width/Height ratio error >5%")
            else:
                print("  PASS: Width/Height ratio consistency OK")

    # 检查VIV风速 vs 临界风速
    if all(c in df.columns for c in ['VIV_Wind_Speed_ms', 'Critical_Wind_Speed_ms']):
        valid_mask = df[['VIV_Wind_Speed_ms', 'Critical_Wind_Speed_ms']].notna().all(axis=1)
        if valid_mask.sum() > 0:
            invalid = (df.loc[valid_mask, 'VIV_Wind_Speed_ms'] >
                      df.loc[valid_mask, 'Critical_Wind_Speed_ms']).sum()
            if invalid > 0:
                inconsistencies += invalid
                print(f"  WARNING: {invalid} samples have VIV_Wind_Speed > Critical_Wind_Speed")
            else:
                print("  PASS: VIV wind speed vs critical wind speed OK")

    # 检查RMS振幅 vs 最大振幅
    if all(c in df.columns for c in ['Amplitude_RMS_mm', 'Max_Amplitude_mm']):
        valid_mask = df[['Amplitude_RMS_mm', 'Max_Amplitude_mm']].notna().all(axis=1)
        if valid_mask.sum() > 0:
            invalid = (df.loc[valid_mask, 'Amplitude_RMS_mm'] >
                      df.loc[valid_mask, 'Max_Amplitude_mm']).sum()
            if invalid > 0:
                inconsistencies += invalid
                print(f"  WARNING: {invalid} samples have Amplitude_RMS > Max_Amplitude")
            else:
                print("  PASS: RMS amplitude vs max amplitude OK")

    if inconsistencies == 0:
        print("  PASS: All physical consistency checks passed")

    # 6. 数据分布统计
    print("\n5. Data Distribution Statistics")
    print("-" * 80)

    # 桥梁类型分布
    if 'BridgeType' in df.columns:
        print("Bridge Type Distribution:")
        type_dist = df['BridgeType'].value_counts()
        for bridge_type, count in type_dist.items():
            print(f"  - {bridge_type}: {count} ({count/len(df)*100:.1f}%)")

    # 风险等级分布
    if 'Max_Amplitude_mm' in df.columns:
        print("\nRisk Level Distribution (by amplitude):")
        low = (df['Max_Amplitude_mm'] <= 30).sum()
        medium = ((df['Max_Amplitude_mm'] > 30) & (df['Max_Amplitude_mm'] <= 60)).sum()
        high = ((df['Max_Amplitude_mm'] > 60) & (df['Max_Amplitude_mm'] <= 100)).sum()
        critical = (df['Max_Amplitude_mm'] > 100).sum()

        print(f"  - Low (<=30mm): {low} ({low/len(df)*100:.1f}%)")
        print(f"  - Medium (30-60mm): {medium} ({medium/len(df)*100:.1f}%)")
        print(f"  - High (60-100mm): {high} ({high/len(df)*100:.1f}%)")
        print(f"  - Critical (>100mm): {critical} ({critical/len(df)*100:.1f}%)")

        high_risk_total = high + critical
        if high_risk_total < len(df) * 0.25:
            print(f"  WARNING: High-risk samples ({high_risk_total}) < 25% of total")
            print("           Recommend collecting more high-risk samples")

    # 7. 去重检查
    print("\n6. Duplicate Check")
    print("-" * 80)
    if 'BridgeName' in df.columns and 'Country' in df.columns:
        duplicates = df.duplicated(subset=['BridgeName', 'Country'], keep=False)
        dup_count = duplicates.sum()
        if dup_count > 0:
            print(f"  WARNING: Found {dup_count} potential duplicate samples")
            print("  Duplicate bridge names:")
            dup_bridges = df[duplicates][['BridgeName', 'Country']].drop_duplicates()
            for _, row in dup_bridges.iterrows():
                print(f"    - {row['BridgeName']} ({row['Country']})")
        else:
            print("  PASS: No duplicate samples found")

    # 8. 汇总报告
    print("\n" + "=" * 80)
    print("Validation Summary")
    print("=" * 80)
    print(f"Total samples: {len(df)}")
    print(f"Required fields: {len(required_cols)} / {len(required_cols)} present")

    if inconsistencies == 0:
        print("\nStatus: PASS - Data ready for model training")
        print("\nNext steps:")
        print("  1. Merge this data with existing dataset")
        print("  2. Re-train VIV prediction model")
        print("  3. Evaluate performance improvement")
        return True
    else:
        print(f"\nStatus: WARNING - Found {inconsistencies} inconsistencies")
        print("Please review and fix the issues above before using this data.")
        return False


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python validate_new_data.py <path_to_csv_file>")
        print("\nExample:")
        print("  python scripts/validate_new_data.py data/bridge_viv_data_gemini_20251119.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not Path(csv_path).exists():
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    success = validate_bridge_data(csv_path)
    sys.exit(0 if success else 1)
