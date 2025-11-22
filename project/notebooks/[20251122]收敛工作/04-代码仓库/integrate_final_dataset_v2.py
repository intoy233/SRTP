"""
整合最终数据集脚本 (简化版 - 使用真值表)
功能：将原始数据集与人工整理的真实临界风速数据进行整合
生成：project/dataset.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher
from bridge_vcr_ground_truth import BRIDGE_CRITICAL_WIND_SPEED

# ====== 配置路径 ======
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# 输入文件
CLEAN_CSV = DATA_DIR / "final_bridge_dataset_clean.csv"

# 输出文件
OUTPUT_CSV = BASE_DIR / "dataset.csv"

# ====== 模糊匹配桥梁名称 ======
def fuzzy_match_bridge_name(name1: str, name2: str, threshold: float = 0.85) -> bool:
    """
    使用模糊匹配判断两个桥梁名称是否相同

    Args:
        name1: 名称1
        name2: 名称2
        threshold: 相似度阈值（0-1）

    Returns:
        bool: 是否匹配
    """
    # 预处理：转小写，移除常见后缀
    def normalize(name):
        import re
        name = name.lower()
        name = re.sub(r'\s+(bridge|viaduct|crossing|span)$', '', name)
        name = re.sub(r'\s+\(.*?\)', '', name)  # 移除括号
        name = name.strip()
        return name

    norm1 = normalize(name1)
    norm2 = normalize(name2)

    # 精确匹配
    if norm1 == norm2:
        return True

    # 模糊匹配
    similarity = SequenceMatcher(None, norm1, norm2).ratio()
    return similarity >= threshold

# ====== 主整合逻辑 ======
def integrate_dataset():
    """
    整合数据集主函数
    """
    print("="*70)
    print("开始整合最终数据集 (使用真值表)".center(70))
    print("="*70)

    # 1. 加载原始数据
    print("\n[Step 1] 加载原始数据集...")
    df = pd.read_csv(CLEAN_CSV, encoding='utf-8-sig')
    print(f"  原始数据: {len(df)} 条样本")
    print(f"  列名: {list(df.columns)}")

    # 2. 加载真值表
    print("\n[Step 2] 加载真实临界风速真值表...")
    print(f"  真值表记录数: {len(BRIDGE_CRITICAL_WIND_SPEED)}")
    print(f"  风速范围: [{min(BRIDGE_CRITICAL_WIND_SPEED.values())}, {max(BRIDGE_CRITICAL_WIND_SPEED.values())}] m/s")

    # 3. 更新原始数据集中的 Critical_Wind_Speed_ms
    print("\n[Step 3] 更新 Critical_Wind_Speed_ms...")
    updated_count = 0
    match_log = []

    for idx, row in df.iterrows():
        bridge_name = row['BridgeName']
        current_vcr = row['Critical_Wind_Speed_ms']

        # 尝试精确匹配
        matched_name = None
        if bridge_name in BRIDGE_CRITICAL_WIND_SPEED:
            matched_name = bridge_name
        else:
            # 尝试模糊匹配
            for truth_name in BRIDGE_CRITICAL_WIND_SPEED.keys():
                if fuzzy_match_bridge_name(bridge_name, truth_name, threshold=0.85):
                    matched_name = truth_name
                    break

        if matched_name:
            new_vcr = BRIDGE_CRITICAL_WIND_SPEED[matched_name]
            # 判断是否需要更新
            # 条件1：当前值为经验填充值 (22.0 或 5.1)
            # 条件2：新值与当前值相差超过5%
            is_empirical = np.isclose(current_vcr, 22.0) or np.isclose(current_vcr, 5.1)
            is_different = abs(new_vcr - current_vcr) / current_vcr > 0.05 if current_vcr > 0 else True

            if is_empirical or is_different:
                df.at[idx, 'Critical_Wind_Speed_ms'] = new_vcr
                # 添加数据源标记
                current_notes = row['Notes'] if pd.notna(row['Notes']) else ""
                if "Real Vcr from Literature" not in current_notes:
                    df.at[idx, 'Notes'] = f"{current_notes}; Real Vcr from Literature" if current_notes else "Real Vcr from Literature"
                updated_count += 1
                match_log.append({
                    "BridgeID": row['BridgeID'],
                    "BridgeName": bridge_name,
                    "Matched_Name": matched_name,
                    "Old_Vcr": current_vcr,
                    "New_Vcr": new_vcr,
                    "Change": f"{((new_vcr - current_vcr) / current_vcr * 100):.1f}%"
                })

    print(f"  更新: {updated_count} 条样本的 Critical_Wind_Speed_ms")

    # 4. 统计更新结果
    print("\n[Step 4] 统计更新结果...")
    empirical_count = ((df['Critical_Wind_Speed_ms'] == 22.0) | (df['Critical_Wind_Speed_ms'] == 5.1)).sum()
    real_count = len(df) - empirical_count

    print(f"  真实数据: {real_count} 条 ({real_count/len(df)*100:.1f}%)")
    print(f"  经验填充: {empirical_count} 条 ({empirical_count/len(df)*100:.1f}%)")
    print(f"  Critical_Wind_Speed 范围: [{df['Critical_Wind_Speed_ms'].min():.1f}, {df['Critical_Wind_Speed_ms'].max():.1f}] m/s")
    print(f"  Critical_Wind_Speed 均值: {df['Critical_Wind_Speed_ms'].mean():.2f} m/s")
    print(f"  Critical_Wind_Speed 中位数: {df['Critical_Wind_Speed_ms'].median():.2f} m/s")

    # 5. 保存最终数据集
    print("\n[Step 5] 保存最终数据集...")
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"  保存路径: {OUTPUT_CSV}")
    print(f"  总样本数: {len(df)}")

    # 6. 保存更新日志
    log_df = pd.DataFrame(match_log)
    log_path = BASE_DIR / "dataset_update_log.csv"
    if len(log_df) > 0:
        log_df.to_csv(log_path, index=False, encoding='utf-8-sig')
        print(f"  更新日志: {log_path}")
        print(f"  记录数: {len(log_df)}")

    # 7. 显示部分更新样例
    print("\n[Step 6] 更新样例 (前20条):")
    print("-"*70)
    if len(log_df) > 0:
        print(log_df.head(20).to_string(index=False))
    else:
        print("  [无更新记录]")

    # 8. 统计未匹配的桥梁
    print("\n[Step 7] 未匹配桥梁统计...")
    unmatched = []
    for idx, row in df.iterrows():
        bridge_name = row['BridgeName']
        if bridge_name not in BRIDGE_CRITICAL_WIND_SPEED:
            # 尝试模糊匹配
            found = False
            for truth_name in BRIDGE_CRITICAL_WIND_SPEED.keys():
                if fuzzy_match_bridge_name(bridge_name, truth_name, threshold=0.85):
                    found = True
                    break
            if not found:
                unmatched.append(bridge_name)

    print(f"  未匹配桥梁数: {len(unmatched)}")
    if len(unmatched) > 0 and len(unmatched) <= 20:
        print(f"  未匹配列表: {unmatched}")

    print("\n" + "="*70)
    print("数据集整合完成！".center(70))
    print("="*70)

    return df

# ====== 执行整合 ======
if __name__ == "__main__":
    integrate_dataset()
