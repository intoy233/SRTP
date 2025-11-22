#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1: 删除3条Critical_Wind_Speed_ms缺失样本
生成干净的合并数据集
"""

import pandas as pd
import numpy as np
from pathlib import Path


def clean_missing_critical_wind_speed(input_csv: str, output_csv: str) -> bool:
    """删除Critical_Wind_Speed_ms缺失样本"""
    print("=" * 80)
    print("Step 1: Cleaning Missing Critical_Wind_Speed_ms Samples")
    print("=" * 80)

    try:
        # 1. 读取数据
        print(f"\n1. Loading dataset: {input_csv}")
        df = pd.read_csv(input_csv, encoding='utf-8-sig')
        print(f"   Total samples before cleaning: {len(df)}")

        # 2. 检查缺失情况
        print("\n2. Checking Critical_Wind_Speed_ms missing values:")
        missing_count = df['Critical_Wind_Speed_ms'].isnull().sum()
        print(f"   Missing samples: {missing_count}")

        if missing_count > 0:
            print(f"\n   Missing samples details:")
            missing_samples = df[df['Critical_Wind_Speed_ms'].isnull()][
                ['BridgeID', 'BridgeName', 'Country', 'BridgeType', 'Span_m']
            ]
            for idx, row in missing_samples.iterrows():
                print(f"     - ID={row['BridgeID']}: {row['BridgeName']} ({row['Country']}), "
                      f"Type={row['BridgeType']}, Span={row['Span_m']:.1f}m")

        # 3. 删除缺失样本
        print(f"\n3. Removing {missing_count} samples with missing Critical_Wind_Speed_ms...")
        df_clean = df.dropna(subset=['Critical_Wind_Speed_ms'])
        print(f"   Samples after cleaning: {len(df_clean)}")
        print(f"   Removed: {len(df) - len(df_clean)} samples")

        # 4. 验证清洗结果
        print("\n4. Validation after cleaning:")
        remaining_missing = df_clean['Critical_Wind_Speed_ms'].isnull().sum()
        print(f"   Remaining missing Critical_Wind_Speed_ms: {remaining_missing}")

        if remaining_missing > 0:
            print(f"   ERROR: Still have missing values!")
            return False

        # 5. 数据完整性检查
        print("\n5. Data integrity check:")
        print(f"   Total samples: {len(df_clean)}")
        print(f"   Total features: {len(df_clean.columns)}")

        # 核心字段缺失率
        core_fields = ['Damping_Ratio', 'Max_Amplitude_mm', 'Span_m',
                       'Width_m', 'Natural_Freq_Hz', 'Critical_Wind_Speed_ms']

        print("\n   Core fields missing rates:")
        for field in core_fields:
            missing_rate = df_clean[field].isnull().sum() / len(df_clean) * 100
            print(f"     {field}: {missing_rate:.1f}%")

        # 振幅分布
        print("\n6. Amplitude distribution after cleaning:")
        low = (df_clean['Max_Amplitude_mm'] <= 30).sum()
        medium = ((df_clean['Max_Amplitude_mm'] > 30) &
                  (df_clean['Max_Amplitude_mm'] <= 60)).sum()
        high = ((df_clean['Max_Amplitude_mm'] > 60) &
                (df_clean['Max_Amplitude_mm'] <= 100)).sum()
        critical = (df_clean['Max_Amplitude_mm'] > 100).sum()

        print(f"   Low (<=30mm): {low} ({low/len(df_clean)*100:.1f}%)")
        print(f"   Medium (30-60mm): {medium} ({medium/len(df_clean)*100:.1f}%)")
        print(f"   High (60-100mm): {high} ({high/len(df_clean)*100:.1f}%)")
        print(f"   Critical (>100mm): {critical} ({critical/len(df_clean)*100:.1f}%)")

        high_risk_total = high + critical
        print(f"\n   High-risk total (>60mm): {high_risk_total} ({high_risk_total/len(df_clean)*100:.1f}%)")

        # 7. 保存清洗后数据
        print(f"\n7. Saving cleaned dataset to: {output_csv}")
        df_clean.to_csv(output_csv, index=False, encoding='utf-8-sig')

        print("\n" + "=" * 80)
        print("Step 1 Complete!")
        print("=" * 80)
        print(f"Clean dataset saved: {output_csv}")
        print(f"Total samples: {len(df_clean)} (removed {len(df) - len(df_clean)})")
        print(f"Sample/Feature ratio: {len(df_clean)/78:.2f}")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\nERROR: Cleaning failed - {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    project_root = Path(__file__).parent.parent

    input_csv = project_root / 'data' / 'final_bridge_dataset_merged.csv'
    output_csv = project_root / 'data' / 'final_bridge_dataset_clean.csv'

    if not input_csv.exists():
        print(f"ERROR: Input file not found: {input_csv}")
        exit(1)

    success = clean_missing_critical_wind_speed(str(input_csv), str(output_csv))
    exit(0 if success else 1)
