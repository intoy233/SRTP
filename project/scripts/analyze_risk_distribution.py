#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集风险等级分布分析
用于验证高风险样本数量和分布情况
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_risk_distribution():
    """分析数据集的风险等级分布"""

    data_path = Path(__file__).parent.parent / "data" / "final_bridge_dataset.csv"
    df = pd.read_csv(data_path)

    print("=" * 80)
    print("桥梁VIV数据集 - 风险等级分布分析")
    print("=" * 80)

    print(f"\n总样本数: {len(df)} 座桥梁")

    # 按不同阈值统计
    thresholds = [30, 45, 60, 80, 100]

    print("\n按振幅阈值统计:")
    print("-" * 80)
    print(f"{'阈值 (mm)':<15} {'样本数':<10} {'占比 (%)':<15} {'定义':<20}")
    print("-" * 80)

    for threshold in thresholds:
        count = len(df[df['Max_Amplitude_mm'] > threshold])
        percentage = count / len(df) * 100

        if threshold == 30:
            risk_def = "中高风险"
        elif threshold == 45:
            risk_def = "高风险（保守）"
        elif threshold == 60:
            risk_def = "高风险（标准）"
        elif threshold == 80:
            risk_def = "极高风险"
        else:
            risk_def = "极端风险"

        print(f"> {threshold:<13} {count:<10} {percentage:<15.1f} {risk_def:<20}")

    # 详细分段统计
    print("\n按振幅区间统计:")
    print("-" * 80)
    print(f"{'振幅区间 (mm)':<20} {'样本数':<10} {'占比 (%)':<15} {'风险等级':<15}")
    print("-" * 80)

    bins = [0, 30, 60, 100, float('inf')]
    labels = ['0-30 (低风险)', '30-60 (中风险)', '60-100 (高风险)', '>100 (极高风险)']
    risk_labels = ['Low', 'Medium', 'High', 'Very High']

    for i in range(len(bins)-1):
        if bins[i+1] == float('inf'):
            mask = df['Max_Amplitude_mm'] > bins[i]
            interval_str = f"> {bins[i]}"
        else:
            mask = (df['Max_Amplitude_mm'] > bins[i]) & (df['Max_Amplitude_mm'] <= bins[i+1])
            interval_str = f"{bins[i]}-{bins[i+1]}"

        count = mask.sum()
        percentage = count / len(df) * 100

        print(f"{interval_str:<20} {count:<10} {percentage:<15.1f} {risk_labels[i]:<15}")

    # 描述性统计
    print("\n振幅描述性统计:")
    print("-" * 80)
    stats = df['Max_Amplitude_mm'].describe()
    print(stats)

    # 检查类别平衡性 (60mm阈值)
    high_risk_60 = len(df[df['Max_Amplitude_mm'] > 60])
    low_risk_60 = len(df[df['Max_Amplitude_mm'] <= 60])

    print("\n类别平衡性分析 (60mm阈值):")
    print("-" * 80)
    print(f"高风险样本 (>60mm): {high_risk_60} 座 ({high_risk_60/len(df)*100:.1f}%)")
    print(f"低中风险样本 (≤60mm): {low_risk_60} 座 ({low_risk_60/len(df)*100:.1f}%)")

    imbalance_ratio = max(high_risk_60, low_risk_60) / min(high_risk_60, low_risk_60)
    print(f"\n不平衡比率: {imbalance_ratio:.2f} : 1")

    if imbalance_ratio < 2:
        print("✓ 数据基本平衡 (比率 < 2:1)")
        print("  建议: SMOTE等过采样技术可能带来有限提升")
    elif imbalance_ratio < 3:
        print("⚠ 存在轻度不平衡 (比率 2:1 - 3:1)")
        print("  建议: 可尝试SMOTE，但需谨慎评估效果")
    else:
        print("⚠⚠ 存在严重不平衡 (比率 > 3:1)")
        print("  建议: 强烈建议使用SMOTE/GAN等数据增强技术")

    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)

    return df


if __name__ == "__main__":
    df = analyze_risk_distribution()
