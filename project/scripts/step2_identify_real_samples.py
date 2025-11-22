#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2: 识别并筛选真实临界风速样本
用于训练Version A (真实数据baseline)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def identify_real_critical_wind_speed_samples(clean_csv: str, output_csv: str) -> bool:
    """识别真实临界风速样本(非经验填充)"""
    print("=" * 80)
    print("Step 2: Identifying Real Critical_Wind_Speed_ms Samples")
    print("=" * 80)

    try:
        # 1. 读取数据
        print(f"\n1. Loading cleaned dataset: {clean_csv}")
        df = pd.read_csv(clean_csv, encoding='utf-8-sig')
        print(f"   Total samples: {len(df)}")

        # 2. 识别原始196样本
        print("\n2. Identifying original 196 samples (all real data):")
        original_mask = df['BridgeID'] <= 196
        original_count = original_mask.sum()
        print(f"   Original samples (ID <= 196): {original_count}")

        # 3. 识别新数据中的真实样本
        # 根据数据合并日志,只有3个样本有真实临界风速(不是经验填充)
        # 这些是Gemini在数据模板中提供的示例
        print("\n3. Identifying new samples with real Critical_Wind_Speed_ms:")

        # 读取原始的07-数据模板 查看哪些有真实值
        # 根据合并报告:"Missing Critical_Wind_Speed_ms: 300 (99.0%)"
        # 意味着303个新样本中只有3个有原始值

        # 策略:检查新数据(ID > 196)中Critical_Wind_Speed_ms不在裁剪边界的样本
        new_data = df[~original_mask].copy()

        # 裁剪边界是5.1和22.0,如果值恰好等于这些,可能是被裁剪的
        # 但更安全的方法是:检查值是否为"典型经验公式结果"

        # 经验公式: Vcr = f * B / St
        # 如果实际值与公式计算值相差<1%,认为是填充值

        new_data['calc_vcr'] = (new_data['Natural_Freq_Hz'] *
                                new_data['Width_m'] / 0.15)  # St=0.15

        # 计算误差
        new_data['vcr_error'] = np.abs(
            new_data['Critical_Wind_Speed_ms'] - new_data['calc_vcr']
        ) / new_data['calc_vcr']

        # 如果误差<5%,认为是经验填充;否则是真实值
        real_vcr_mask = new_data['vcr_error'] > 0.05

        # 但考虑到被裁剪到22.0的情况
        clipped_mask = (new_data['Critical_Wind_Speed_ms'] == 22.0) | \
                       (new_data['Critical_Wind_Speed_ms'] == 5.1)

        # 如果被裁剪,即使误差大也不是真实值
        real_vcr_mask = real_vcr_mask & (~clipped_mask)

        real_new_count = real_vcr_mask.sum()
        print(f"   New samples with real Critical_Wind_Speed_ms: {real_new_count}")

        # 4. 合并真实样本
        print("\n4. Creating Version A dataset (real data only):")

        # 所有原始样本 + 新数据中的真实样本
        real_mask = original_mask.copy()
        real_mask[~original_mask] = real_vcr_mask.values

        df_real = df[real_mask].copy()

        print(f"   Total real samples: {len(df_real)}")
        print(f"     - Original samples: {original_count}")
        print(f"     - New real samples: {real_new_count}")

        # 5. 数据质量检查
        print("\n5. Version A dataset quality check:")

        # 振幅分布
        low = (df_real['Max_Amplitude_mm'] <= 30).sum()
        medium = ((df_real['Max_Amplitude_mm'] > 30) &
                  (df_real['Max_Amplitude_mm'] <= 60)).sum()
        high = ((df_real['Max_Amplitude_mm'] > 60) &
                (df_real['Max_Amplitude_mm'] <= 100)).sum()
        critical = (df_real['Max_Amplitude_mm'] > 100).sum()

        print(f"\n   Amplitude distribution:")
        print(f"     Low (<=30mm): {low} ({low/len(df_real)*100:.1f}%)")
        print(f"     Medium (30-60mm): {medium} ({medium/len(df_real)*100:.1f}%)")
        print(f"     High (60-100mm): {high} ({high/len(df_real)*100:.1f}%)")
        print(f"     Critical (>100mm): {critical} ({critical/len(df_real)*100:.1f}%)")

        high_risk_total = high + critical
        print(f"\n     High-risk total (>60mm): {high_risk_total} ({high_risk_total/len(df_real)*100:.1f}%)")

        # 6. 样本/特征比
        sample_ratio = len(df_real) / 78
        print(f"\n   Sample/Feature ratio: {sample_ratio:.2f}")

        if sample_ratio >= 2.5:
            print(f"   Status: Acceptable (above minimum threshold)")
        else:
            print(f"   WARNING: Sample/Feature ratio < 2.5, may cause overfitting")

        # 7. 保存Version A数据集
        print(f"\n6. Saving Version A dataset to: {output_csv}")

        # 删除临时计算列
        df_real_clean = df_real.drop(columns=['calc_vcr', 'vcr_error'], errors='ignore')
        df_real_clean.to_csv(output_csv, index=False, encoding='utf-8-sig')

        print("\n" + "=" * 80)
        print("Step 2 Complete!")
        print("=" * 80)
        print(f"Version A dataset saved: {output_csv}")
        print(f"Total samples: {len(df_real_clean)}")
        print(f"High-risk samples: {high_risk_total}")
        print(f"Sample/Feature ratio: {sample_ratio:.2f}")
        print("\nReady for Version A model training!")
        print("=" * 80)

        # 8. 保存样本统计报告
        stats_report = {
            'total_samples': len(df_real_clean),
            'original_samples': original_count,
            'new_real_samples': real_new_count,
            'high_risk_samples': int(high_risk_total),
            'sample_feature_ratio': float(sample_ratio),
            'amplitude_distribution': {
                'low': int(low),
                'medium': int(medium),
                'high': int(high),
                'critical': int(critical)
            }
        }

        stats_path = Path(output_csv).parent / 'version_a_stats.txt'
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("Version A (Real Data Baseline) Statistics\n")
            f.write("=" * 60 + "\n\n")
            for key, value in stats_report.items():
                f.write(f"{key}: {value}\n")

        print(f"\nStatistics saved to: {stats_path}")

        return True

    except Exception as e:
        print(f"\nERROR: Failed to identify real samples - {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    project_root = Path(__file__).parent.parent

    clean_csv = project_root / 'data' / 'final_bridge_dataset_clean.csv'
    output_csv = project_root / 'data' / 'final_bridge_dataset_version_a.csv'

    if not clean_csv.exists():
        print(f"ERROR: Clean dataset not found: {clean_csv}")
        exit(1)

    success = identify_real_critical_wind_speed_samples(str(clean_csv), str(output_csv))
    exit(0 if success else 1)
