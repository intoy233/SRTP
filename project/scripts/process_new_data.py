#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理Gemini收集的新桥梁VIV数据
修复表头错误、转换格式、执行初步清洗
"""

import pandas as pd
import numpy as np
from pathlib import Path


def fix_and_convert_new_data(txt_path: str, output_csv_path: str) -> bool:
    """修复表头错误并转换为标准CSV格式"""
    print("=" * 80)
    print("Processing New Bridge VIV Data from Gemini")
    print("=" * 80)
    print(f"Input file: {txt_path}")
    print(f"Output file: {output_csv_path}\n")

    try:
        # 1. 读取TXT文件
        print("Step 1: Reading TXT file...")
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 2. 修复表头错误
        print("Step 2: Fixing header error (Second_Freq_Freq_Hz -> Second_Freq_Hz)...")
        header = lines[0].strip()
        if 'Second_Freq_Freq_Hz' in header:
            header = header.replace('Second_Freq_Freq_Hz', 'Second_Freq_Hz')
            print("  FIXED: Header corrected")
        else:
            print("  INFO: No header error found")

        # 3. 重新组装内容
        lines[0] = header + '\n'

        # 4. 保存为CSV
        print("Step 3: Saving as CSV with UTF-8-sig encoding...")
        with open(output_csv_path, 'w', encoding='utf-8-sig') as f:
            f.writelines(lines)

        # 5. 验证读取
        print("Step 4: Verifying CSV structure...")
        df = pd.read_csv(output_csv_path, encoding='utf-8-sig')
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Column names:\n{list(df.columns)}\n")

        # 6. 基本统计
        print("Step 5: Basic statistics:")
        print(f"  Bridge types: {df['BridgeType'].value_counts().to_dict()}")
        print(f"  Countries: {len(df['Country'].unique())} unique countries")
        print(f"  Year range: {df['Year'].min():.0f} - {df['Year'].max():.0f}")

        # 7. 检查目标变量
        print("\nStep 6: Checking target variable (Max_Amplitude_mm):")
        missing_target = df['Max_Amplitude_mm'].isnull().sum()
        print(f"  Missing values: {missing_target} ({missing_target/len(df)*100:.1f}%)")

        if missing_target > 0:
            print(f"  WARNING: Target variable has missing values!")
            return False

        # 8. 振幅分布统计
        print("\nStep 7: Amplitude distribution:")
        low = (df['Max_Amplitude_mm'] <= 30).sum()
        medium = ((df['Max_Amplitude_mm'] > 30) & (df['Max_Amplitude_mm'] <= 60)).sum()
        high = ((df['Max_Amplitude_mm'] > 60) & (df['Max_Amplitude_mm'] <= 100)).sum()
        critical = (df['Max_Amplitude_mm'] > 100).sum()

        print(f"  Low (<=30mm): {low} ({low/len(df)*100:.1f}%)")
        print(f"  Medium (30-60mm): {medium} ({medium/len(df)*100:.1f}%)")
        print(f"  High (60-100mm): {high} ({high/len(df)*100:.1f}%)")
        print(f"  Critical (>100mm): {critical} ({critical/len(df)*100:.1f}%)")

        high_risk_total = high + critical
        print(f"\n  High-risk total (>60mm): {high_risk_total} ({high_risk_total/len(df)*100:.1f}%)")

        # 9. 核心字段缺失率
        print("\nStep 8: Core field missing rates:")
        core_fields = ['Damping_Ratio', 'Critical_Wind_Speed_ms', 'Max_Amplitude_mm',
                       'Span_m', 'Width_m', 'Natural_Freq_Hz']

        for field in core_fields:
            missing_rate = df[field].isnull().sum() / len(df) * 100
            status = "PASS" if missing_rate <= 10 else "WARNING"
            print(f"  [{status}] {field}: {missing_rate:.1f}% missing")

        print("\n" + "=" * 80)
        print("Processing Complete!")
        print("=" * 80)
        print(f"Converted file saved to: {output_csv_path}")
        print(f"Total samples: {len(df)}")
        print(f"Ready for validation script: validate_new_data.py")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"ERROR: Processing failed - {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # 文件路径
    project_root = Path(__file__).parent.parent
    txt_path = project_root / 'notebooks' / '[20251118]改进实验' / '07-新收集数据.txt'
    csv_path = project_root / 'notebooks' / '[20251118]改进实验' / '07-新收集数据-已修正.csv'

    if not txt_path.exists():
        print(f"ERROR: Input file not found: {txt_path}")
        exit(1)

    success = fix_and_convert_new_data(str(txt_path), str(csv_path))
    exit(0 if success else 1)
