#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阈值分析: 分析不同高风险阈值对样本分布的影响
Threshold Analysis for Risk Classification
"""

import pandas as pd
import numpy as np

def analyze_threshold_impact():
    """分析不同阈值的影响"""

    # 读取数据
    df = pd.read_csv('../data/final_bridge_dataset.csv')

    print("="*80)
    print("振幅阈值分析 - 路线B决策支持")
    print("="*80)

    print(f"\n数据集总样本数: {len(df)}")

    print("\n振幅分布统计:")
    print(df['Max_Amplitude_mm'].describe())

    print("\n" + "="*80)
    print("不同阈值下的高风险样本分布")
    print("="*80)

    results = []

    thresholds = [45, 50, 55, 60, 65, 70]

    print(f"\n{'阈值(mm)':<12} {'高风险样本数':<15} {'占比':<12} {'样本/特征比(78D)':<20}")
    print("-"*80)

    for threshold in thresholds:
        high_risk_count = (df['Max_Amplitude_mm'] > threshold).sum()
        percentage = high_risk_count / len(df) * 100
        sample_feature_ratio = high_risk_count / 78  # 78维特征(幂函数变换)

        print(f">{threshold:<11} {high_risk_count:<15} {percentage:<11.1f}% {sample_feature_ratio:<20.2f}")

        results.append({
            'threshold': threshold,
            'high_risk_count': high_risk_count,
            'percentage': percentage,
            'sample_feature_ratio': sample_feature_ratio
        })

    # 推荐阈值
    print("\n" + "="*80)
    print("阈值选择建议")
    print("="*80)

    print("\n基准要求:")
    print("  - 样本/特征比 >= 1.5 (基本要求)")
    print("  - 样本/特征比 >= 10.0 (理想要求)")
    print("  - 高风险样本数 >= 60 (避免单个fold样本过少)")

    print("\n分析结果:")

    for res in results:
        threshold = res['threshold']
        count = res['high_risk_count']
        ratio = res['sample_feature_ratio']

        if ratio >= 10.0:
            status = "[OK][OK] Ideal"
        elif ratio >= 1.5:
            status = "[OK] Usable"
        elif ratio >= 1.0:
            status = "[!] Barely"
        else:
            status = "[X] Insufficient"

        print(f"  >{threshold}mm: {count} bridges, ratio={ratio:.2f} -> {status}")

    # 最优推荐
    print("\n" + "="*80)
    print("最终推荐")
    print("="*80)

    # 找到满足ratio>=1.5的最高阈值
    valid_options = [r for r in results if r['sample_feature_ratio'] >= 1.5]

    if valid_options:
        best = max(valid_options, key=lambda x: x['threshold'])
        print(f"\n推荐阈值: >{best['threshold']}mm")
        print(f"  高风险样本: {best['high_risk_count']}座 ({best['percentage']:.1f}%)")
        print(f"  样本/特征比: {best['sample_feature_ratio']:.2f}")
        print(f"  5-Fold每折训练集高风险样本: ~{int(best['high_risk_count']*0.8)}")
        print(f"\n理由: 这是满足基本要求(比值>=1.5)的最严格阈值,保留了'高风险'的针对性")
    else:
        print("\n警告: 所有阈值都无法满足基本要求(比值>=1.5)")
        print("建议:")
        print("  1. 降低阈值至45mm (牺牲针对性换取样本数)")
        print("  2. 收集更多数据")
        print("  3. 放弃分诊系统,使用单一模型")

    return results

if __name__ == '__main__':
    results = analyze_threshold_impact()
