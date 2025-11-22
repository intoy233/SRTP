"""
整合最终数据集脚本
功能：将原始数据集与 Batch 2-6 及查漏补缺报告中的真实临界风速数据进行整合
生成：project/dataset.csv
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple
from difflib import SequenceMatcher

# ====== 配置路径 ======
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
NOTEBOOK_DIR = BASE_DIR / "notebooks" / "[20251119]数据补全"

# 输入文件
CLEAN_CSV = DATA_DIR / "final_bridge_dataset_clean.csv"
BATCH_FILES = [
    NOTEBOOK_DIR / "01-数据补全batch2.md",
    NOTEBOOK_DIR / "01-数据补全batch3.md",
    NOTEBOOK_DIR / "01-数据补全batch4.md",
    NOTEBOOK_DIR / "01-数据补全batch5.md",
    NOTEBOOK_DIR / "01-数据补全batch6.md",
    NOTEBOOK_DIR / "02-查漏补缺报告.md"
]

# 输出文件
OUTPUT_CSV = BASE_DIR / "dataset.csv"

# ====== 桥梁名称映射字典（处理同名异称问题）======
NAME_MAPPING = {
    # 中文-英文对照
    "虎门大桥": "Humen Bridge",
    "西侯门大桥": "Xihoumen Bridge",
    "润扬大桥": "Runyang Bridge",
    "苏通大桥": "Sutong Bridge",
    "江阴大桥": "Jiangyin Bridge",
    "青马大桥": "Tsing Ma Bridge",
    "明石海峡大桥": "Akashi Kaikyo Bridge",
    "大贝尔特东桥": "Great Belt East Bridge",
    "李舜臣大桥": "Yi Sun-sin Bridge",
    "昂船洲大桥": "Stonecutters Bridge",
    "多得罗大桥": "Tatara Bridge",
    "诺曼底大桥": "Normandy Bridge",
    "里里奥-安提里奥大桥": "Rio-Antirrio Bridge",
    "米约高架桥": "Millau Viaduct",
    "俄罗斯岛大桥": "Russky Bridge",
    "亨伯大桥": "Humber Bridge",
    "老塞文桥": "Severn Bridge",
    "新塞文桥": "Second Severn Crossing",
    "福斯公路大桥": "Forth Road Bridge",
    "金门大桥": "Golden Gate Bridge",
    "韦拉扎诺海峡大桥": "Verrazano-Narrows Bridge",
    "麦基诺大桥": "Mackinac Bridge",
    "老塔科马海峡大桥": "Tacoma Narrows Bridge",
    "鹦鹉洲长江大桥": "Yingwuzhou Yangtze River Bridge",
    "杨浦大桥": "Yangpu Bridge",
    "南浦大桥": "Nanpu Bridge",
    "二七长江大桥": "Erqi Yangtze River Bridge",
    "卢浦大桥": "Lupu Bridge",
    "朝天门大桥": "Chaotianmen Bridge",
    "大胜关大桥": "Dashengguan Bridge",
    "四渡河大桥": "Sidu River Bridge",
    "润扬南桥": "Runyang South Bridge",
    "润扬北桥": "Runyang North Bridge",
    "洞庭湖大桥": "Dongting Lake Bridge",
    "金沙江大桥": "Jinshajiang Bridge",
    "五峰山大桥": "Wufengshan Bridge",
    "高海岸大桥": "Höga Kusten Bridge",
    "1812宪法桥": "Puente de la Constitución de 1812",
    "鹤见翼桥": "Tsurumi Tsubasa Bridge",
    "贝永桥": "Bayonne Bridge",

    # 别名处理
    "Yi Sun-sin": "Yi Sun-sin Bridge",
    "E-San-Se Bridge": "Yi Sun-sin Bridge",  # 重复项
    "Pont de Normandie": "Normandy Bridge",
    "Rion-Antirion Bridge": "Rio-Antirrio Bridge",
    "Runyang Suspension Bridge": "Runyang Bridge",
    "Humen Pearl River Bridge": "Humen Bridge",
    "Severn Bridge (First)": "Severn Bridge",
}

# ====== 解析Markdown文件提取临界风速数据 ======
def parse_batch_file(file_path: Path) -> List[Dict]:
    """
    解析Batch Markdown文件，提取桥梁名称和临界风速

    返回格式：
    [
        {
            "name": "Humen Bridge",
            "critical_wind_speed": 8.7,
            "source": "Field Monitoring, 2020",
            "status": "Real Data"
        },
        ...
    ]
    """
    results = []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 正则表达式模式：提取桥梁信息块
    # 匹配格式：### XX. Bridge Name (中文名)
    #          - **Critical Wind Speed**: **XX.X m/s**
    #          - **Source**: ...
    #          - **Status**: ✅ **Real Data**

    # 模式1：标准格式
    pattern1 = r'###\s+\d+\.\s+([^\n(]+?)(?:\s+\([^)]+\))?\s*\n.*?Critical\s+Wind\s+Speed[^:]*:\s+\*\*([^\*]+)\*\*.*?Source[^:]*:\s+([^\n]+?)\.?\s*\n.*?Status[^:]*:\s+([^\n]+)'

    matches = re.finditer(pattern1, content, re.DOTALL | re.IGNORECASE)

    for match in matches:
        bridge_name = match.group(1).strip()
        critical_speed_text = match.group(2).strip()
        source = match.group(3).strip()
        status = match.group(4).strip()

        # 解析临界风速
        # 优先顺序：建议值 > 括号内数值 > 第一个数值

        # 1. 优先提取"建议值"（用于Stable桥梁）
        context = content[match.start():match.end()+300]
        suggest_match = re.search(r'建议值[^:]*:\s+\*\*?([\d.]+)', context)

        if suggest_match:
            speed = float(suggest_match.group(1))
        elif "Stable" in critical_speed_text or ">" in critical_speed_text:
            # 2. 处理 "Stable / >XX m/s" 格式 - 使用保守的默认值
            if ">80" in critical_speed_text or ">60" in critical_speed_text:
                speed = 50.0
            elif ">40" in critical_speed_text:
                speed = 40.0
            elif ">35" in critical_speed_text:
                speed = 35.0
            else:
                # 尝试提取数值
                stable_match = re.search(r'>\s*([\d.]+)\s*m/s', critical_speed_text)
                if stable_match and float(stable_match.group(1)) < 60:
                    speed = float(stable_match.group(1))
                else:
                    speed = 40.0  # 默认值
        else:
            # 3. 常规格式："XX.X m/s (VIV Onset)"
            speed_match = re.search(r'^([\d.]+)\s*m/s', critical_speed_text)
            if speed_match:
                speed = float(speed_match.group(1))
            else:
                continue  # 无法解析，跳过

        # 清理桥梁名称
        bridge_name = bridge_name.replace("Bridge (Second reference verification)", "Bridge")
        bridge_name = bridge_name.replace("(Second reference verification)", "").strip()
        bridge_name = bridge_name.replace("(Third reference verification)", "").strip()
        bridge_name = bridge_name.replace("(Fourth reference verification)", "").strip()
        bridge_name = bridge_name.replace("(Fifth reference verification)", "").strip()
        bridge_name = bridge_name.replace("(VIV Test 2)", "").strip()
        bridge_name = bridge_name.replace("(VIV Test 3)", "").strip()
        bridge_name = bridge_name.replace("(Revisit)", "").strip()

        # 应用名称映射
        if bridge_name in NAME_MAPPING:
            bridge_name = NAME_MAPPING[bridge_name]

        # 判断是否为真实数据
        is_real = "Real Data" in status or "✅" in status

        results.append({
            "name": bridge_name,
            "critical_wind_speed": speed,
            "source": source,
            "status": "Real" if is_real else "Inferred",
            "batch_file": file_path.name
        })

    return results

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
    print("开始整合最终数据集".center(70))
    print("="*70)

    # 1. 加载原始数据
    print("\n[Step 1] 加载原始数据集...")
    df = pd.read_csv(CLEAN_CSV, encoding='utf-8-sig')
    print(f"  原始数据: {len(df)} 条样本")
    print(f"  列名: {list(df.columns)}")

    # 2. 解析所有Batch文件
    print("\n[Step 2] 解析Batch文件...")
    all_batch_data = []
    for batch_file in BATCH_FILES:
        if batch_file.exists():
            print(f"  解析: {batch_file.name}")
            batch_data = parse_batch_file(batch_file)
            all_batch_data.extend(batch_data)
            print(f"    提取: {len(batch_data)} 条记录")
        else:
            print(f"  [警告] 文件不存在: {batch_file}")

    print(f"\n  总计提取: {len(all_batch_data)} 条真实临界风速记录")

    # 3. 去重（同一桥梁可能在多个Batch中出现）
    print("\n[Step 3] 去重真实数据...")
    unique_batch_data = {}
    for record in all_batch_data:
        name = record["name"]
        if name not in unique_batch_data:
            unique_batch_data[name] = record
        else:
            # 保留状态为"Real"的记录，或者wind speed更可靠的记录
            if record["status"] == "Real" and unique_batch_data[name]["status"] != "Real":
                unique_batch_data[name] = record

    print(f"  去重后: {len(unique_batch_data)} 条唯一桥梁记录")

    # 4. 创建桥梁名称到临界风速的映射
    bridge_vcr_map = {name: data["critical_wind_speed"] for name, data in unique_batch_data.items()}

    # 5. 更新原始数据集中的 Critical_Wind_Speed_ms
    print("\n[Step 4] 更新 Critical_Wind_Speed_ms...")
    updated_count = 0
    match_log = []

    for idx, row in df.iterrows():
        bridge_name = row['BridgeName']
        current_vcr = row['Critical_Wind_Speed_ms']

        # 尝试精确匹配
        matched_name = None
        if bridge_name in bridge_vcr_map:
            matched_name = bridge_name
        else:
            # 尝试模糊匹配
            for batch_name in bridge_vcr_map.keys():
                if fuzzy_match_bridge_name(bridge_name, batch_name, threshold=0.85):
                    matched_name = batch_name
                    break

        if matched_name:
            new_vcr = bridge_vcr_map[matched_name]
            # 判断是否需要更新
            # 条件1：当前值为经验填充值 (22.0 或 5.1)
            # 条件2：新值与当前值相差超过5%
            is_empirical = np.isclose(current_vcr, 22.0) or np.isclose(current_vcr, 5.1)
            is_different = abs(new_vcr - current_vcr) / current_vcr > 0.05 if current_vcr > 0 else True

            if is_empirical or is_different:
                df.at[idx, 'Critical_Wind_Speed_ms'] = new_vcr
                # 添加数据源标记
                df.at[idx, 'Notes'] = f"{row['Notes']}; Real Vcr from Literature" if pd.notna(row['Notes']) else "Real Vcr from Literature"
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

    # 6. 统计更新结果
    print("\n[Step 5] 统计更新结果...")
    empirical_count = ((df['Critical_Wind_Speed_ms'] == 22.0) | (df['Critical_Wind_Speed_ms'] == 5.1)).sum()
    real_count = len(df) - empirical_count

    print(f"  真实数据: {real_count} 条 ({real_count/len(df)*100:.1f}%)")
    print(f"  经验填充: {empirical_count} 条 ({empirical_count/len(df)*100:.1f}%)")
    print(f"  Critical_Wind_Speed 范围: [{df['Critical_Wind_Speed_ms'].min():.1f}, {df['Critical_Wind_Speed_ms'].max():.1f}] m/s")
    print(f"  Critical_Wind_Speed 均值: {df['Critical_Wind_Speed_ms'].mean():.2f} m/s")

    # 7. 保存最终数据集
    print("\n[Step 6] 保存最终数据集...")
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"  保存路径: {OUTPUT_CSV}")
    print(f"  总样本数: {len(df)}")

    # 8. 保存更新日志
    log_df = pd.DataFrame(match_log)
    log_path = BASE_DIR / "dataset_update_log.csv"
    if len(log_df) > 0:
        log_df.to_csv(log_path, index=False, encoding='utf-8-sig')
        print(f"  更新日志: {log_path}")
        print(f"  记录数: {len(log_df)}")

    # 9. 显示部分更新样例
    print("\n[Step 7] 更新样例 (前10条):")
    print("-"*70)
    if len(log_df) > 0:
        print(log_df.head(10).to_string(index=False))
    else:
        print("  [无更新记录]")

    print("\n" + "="*70)
    print("数据集整合完成！".center(70))
    print("="*70)

    return df

# ====== 执行整合 ======
if __name__ == "__main__":
    integrate_dataset()
