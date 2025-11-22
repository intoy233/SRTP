"""
调试Batch文件解析逻辑
"""

import re
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
BATCH_FILE = BASE_DIR / "notebooks" / "[20251119]数据补全" / "01-数据补全batch3.md"

with open(BATCH_FILE, 'r', encoding='utf-8') as f:
    content = f.read()

# 正则表达式模式
pattern1 = r'###\s+\d+\.\s+([^\n(]+?)(?:\s+\([^)]+\))?\s*\n.*?Critical\s+Wind\s+Speed[^:]*:\s+\*\*([^\*]+)\*\*.*?Source[^:]*:\s+([^\n]+?)\.?\s*\n.*?Status[^:]*:\s+([^\n]+)'

matches = list(re.finditer(pattern1, content, re.DOTALL | re.IGNORECASE))

print(f"找到 {len(matches)} 条匹配")
print("\n" + "="*70 + "\n")

for i, match in enumerate(matches[:3]):  # 只显示前3条
    bridge_name = match.group(1).strip()
    critical_speed_text = match.group(2).strip()
    source = match.group(3).strip()
    status = match.group(4).strip()

    print(f"[匹配 {i+1}]")
    print(f"  桥梁名称: {bridge_name}")
    print(f"  临界风速文本: {critical_speed_text}")
    print(f"  Source: {source}")
    print(f"  Status: {status}")

    # 解析逻辑测试
    context = content[match.start():match.end()+300]
    suggest_match = re.search(r'建议值[^:]*:\s+\*\*?([\d.]+)', context)

    print(f"\n  [解析测试]")
    if suggest_match:
        speed = float(suggest_match.group(1))
        print(f"    - 找到建议值: {speed} m/s")
    elif "Stable" in critical_speed_text or ">" in critical_speed_text:
        print(f"    - 检测到Stable格式")
        if ">80" in critical_speed_text:
            print(f"    - 使用默认值: 50.0 m/s")
        elif ">40" in critical_speed_text:
            print(f"    - 使用默认值: 40.0 m/s")
    else:
        speed_match = re.search(r'^([\d.]+)\s*m/s', critical_speed_text)
        if speed_match:
            speed = float(speed_match.group(1))
            print(f"    - 常规提取: {speed} m/s")

    print("\n" + "="*70 + "\n")
