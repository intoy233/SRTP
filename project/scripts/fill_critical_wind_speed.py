#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
填充Critical_Wind_Speed_ms缺失值
使用经验公式: Vcr = f * B / St
其中:
  f = Natural_Freq_Hz (自振频率)
  B = Width_m (主梁宽度)
  St = Strouhal数 (0.12-0.20,取0.15)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def fill_critical_wind_speed(csv_path: str, output_path: str) -> bool:
    """使用经验公式填充临界风速缺失值"""
    print("=" * 80)
    print("Filling Missing Critical_Wind_Speed_ms Values")
    print("=" * 80)
    print(f"Input file: {csv_path}\n")

    try:
        # 1. 读取数据
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        print(f"Total samples: {len(df)}")
        missing_count = df['Critical_Wind_Speed_ms'].isnull().sum()
        print(f"Missing Critical_Wind_Speed_ms: {missing_count} ({missing_count/len(df)*100:.1f}%)\n")

        # 2. 经验公式填充
        print("Filling using empirical formula:")
        print("  Vcr = Natural_Freq_Hz * Width_m / Strouhal_Number")
        print("  Strouhal_Number = 0.15 (typical for steel box girders)\n")

        # Strouhal数:根据结构类型选择
        def get_strouhal_number(structure_type):
            """根据结构类型返回Strouhal数"""
            strouhal_map = {
                'Steel Box': 0.15,
                'Truss': 0.18,
                'Steel Truss': 0.18,
                'Box Girder': 0.14,
                'Streamlined Box': 0.12,
                'Composite': 0.15,
                'Composite Box': 0.14,
                'Composite Deck': 0.14,
                'Concrete Box': 0.16,
                'Hybrid Box': 0.15,
                'Orthotropic Steel Box': 0.15
            }
            return strouhal_map.get(structure_type, 0.15)  # 默认0.15

        # 创建掩码
        mask = df['Critical_Wind_Speed_ms'].isnull()

        # 计算填充值
        df.loc[mask, 'Strouhal_Temp'] = df.loc[mask, 'Structure_Type'].apply(get_strouhal_number)
        df.loc[mask, 'Critical_Wind_Speed_ms'] = (
            df.loc[mask, 'Natural_Freq_Hz'] * df.loc[mask, 'Width_m'] / df.loc[mask, 'Strouhal_Temp']
        )

        # 删除临时列
        df.drop(columns=['Strouhal_Temp'], inplace=True, errors='ignore')

        # 3. 验证填充结果
        print("Validation after filling:")
        remaining_missing = df['Critical_Wind_Speed_ms'].isnull().sum()
        print(f"  Remaining missing: {remaining_missing}")

        # 检查填充值是否合理
        filled_values = df.loc[mask, 'Critical_Wind_Speed_ms']
        print(f"\nFilled values statistics:")
        print(f"  Count: {len(filled_values)}")
        print(f"  Min: {filled_values.min():.2f} m/s")
        print(f"  Max: {filled_values.max():.2f} m/s")
        print(f"  Mean: {filled_values.mean():.2f} m/s")
        print(f"  Median: {filled_values.median():.2f} m/s")

        # 检查异常值
        out_of_range = ((filled_values < 5.1) | (filled_values > 22)).sum()
        if out_of_range > 0:
            print(f"\n  WARNING: {out_of_range} filled values out of range [5.1, 22] m/s")
            outliers = filled_values[(filled_values < 5.1) | (filled_values > 22)]
            print(f"  Outlier values: {outliers.tolist()[:5]}...")

            # 裁剪异常值
            print("\n  Clipping outliers to valid range...")
            df.loc[mask, 'Critical_Wind_Speed_ms'] = df.loc[mask, 'Critical_Wind_Speed_ms'].clip(5.1, 22)

        # 4. 保存结果
        print(f"\nSaving filled data to: {output_path}")
        df.to_csv(output_path, index=False, encoding='utf-8-sig')

        # 5. 最终验证
        print("\nFinal verification:")
        final_missing = df['Critical_Wind_Speed_ms'].isnull().sum()
        print(f"  Missing values: {final_missing}")
        print(f"  Complete samples: {len(df) - final_missing} ({(len(df)-final_missing)/len(df)*100:.1f}%)")

        print("\n" + "=" * 80)
        print("Filling Complete!")
        print("=" * 80)
        print(f"Output saved to: {output_path}")
        print(f"Ready for merging with original dataset")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"ERROR: Filling failed - {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    project_root = Path(__file__).parent.parent
    input_csv = project_root / 'notebooks' / '[20251118]改进实验' / '07-新收集数据-已修正.csv'
    output_csv = project_root / 'notebooks' / '[20251118]改进实验' / '07-新收集数据-完整.csv'

    if not input_csv.exists():
        print(f"ERROR: Input file not found: {input_csv}")
        exit(1)

    success = fill_critical_wind_speed(str(input_csv), str(output_csv))
    exit(0 if success else 1)
