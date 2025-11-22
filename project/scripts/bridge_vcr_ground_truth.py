"""
桥梁真实临界风速数据表 (从Batch文件人工整理)
数据来源：notebooks/[20251119]数据补全/01-数据补全batch2-6.md 和 02-查漏补缺报告.md
"""

# 真实临界风速数据（来自文献，优先使用 VIV Onset 而非 Stable 值）
BRIDGE_CRITICAL_WIND_SPEED = {
    # Batch 2: 著名国际桥梁
    "Yi Sun-sin Bridge": 9.0,
    "Great Belt East Bridge": 5.0,
    "Trans-Tokyo Bay Crossing Bridge": 16.0,
    "Akashi Kaikyo Bridge": 45.0,  # 桁架梁，高稳定性
    "Tsing Ma Bridge": 40.0,  # 桁架/箱梁混合，高稳定性
    "Jiangyin Bridge": 10.0,
    "Humen 2nd Bridge": 12.0,
    "Russky Bridge": 14.0,
    "Rio-Antirrio Bridge": 11.0,
    "Fred Hartman Bridge": 11.0,
    "Commodore Barry Bridge": 10.5,
    "Tatara Bridge": 8.0,
    "Kap Shui Mun Bridge": 40.0,  # 高稳定性
    "Second Severn Crossing": 10.0,
    "Normandy Bridge": 9.0,

    # Batch 3: VIV经典案例
    "Yingwuzhou Yangtze River Bridge": 6.0,
    "Verrazano-Narrows Bridge": 8.0,
    "Golden Gate Bridge": 16.0,  # 历史VIV/Flutter起振点
    "Mackinac Bridge": 45.0,  # 桁架梁，高稳定性
    "Deer Isle Bridge": 9.0,
    "Bronx-Whitestone Bridge": 10.0,
    "Tacoma Narrows Bridge": 8.5,  # 老塔科马
    "Humber Bridge": 14.0,
    "Yangpu Bridge": 10.0,
    "Nanpu Bridge": 11.0,
    "Erqi Yangtze River Bridge": 8.0,
    "Lupu Bridge": 13.0,  # 拱桥，拱肋VIV
    "Chaotianmen Bridge": 35.0,  # 桁架拱，高稳定性
    "Dashengguan Bridge": 40.0,  # 高铁桥，桁架，高稳定性

    # Batch 4: 中国及亚洲桥梁
    "Xinguang Bridge": 35.0,
    "Liede Bridge": 10.0,
    "Jiayue Bridge": 12.0,
    "Zhoushan Continental Bridge": 11.0,
    "Hangzhou Bay Bridge": 13.0,
    "Donghai Bridge": 12.5,
    "Pingsheng Bridge": 9.5,
    "Egongyan Bridge": 8.0,
    "Masan Bay Bridge": 10.0,
    "Incheon Bridge": 15.0,
    "Yeongjong Grand Bridge": 40.0,
    "Gwangan Bridge": 11.0,
    "Seohae Grand Bridge": 13.0,
    "Machang Bridge": 12.0,
    "Aphae-Amtae Bridge": 10.5,
    "Ulsan Bridge": 9.0,
    "Noryang Bridge": 11.0,
    "Millau Viaduct": 35.0,
    "Yavuz Sultan Selim Bridge": 40.0,
    "Osman Gazi Bridge": 12.0,
    "1915 Çanakkale Bridge": 50.0,  # 双幅分离，极高稳定性
    "Hardanger Bridge": 10.0,
    "Hålogaland Bridge": 12.0,
    "Askøy Bridge": 9.0,
    "Osteroy Bridge": 8.0,
    "Lysefjord Bridge": 10.0,
    "Bømla Bridge": 9.0,
    "Stord Bridge": 9.0,
    "Forth Road Bridge": 35.0,  # 桁架梁
    "Severn Bridge": 8.0,
    "Messina Strait Bridge": 50.0,  # 设计方案，多箱梁
    "Kao-Ping Hsi Bridge": 11.0,
    "Chi-Lu Bridge": 10.0,
    "Vasco da Gama Bridge": 12.0,
    "Sutong Bridge": 35.0,  # 高稳定性

    # Batch 6: 中国新增桥梁
    "Minpu Bridge": 8.5,
    "Xupu Bridge": 10.0,
    "Second Nanjing Yangtze Bridge": 11.5,
    "Third Nanjing Yangtze Bridge": 10.0,
    "Qingzhou Minjiang Bridge": 12.0,
    "Jingyue Yangtze River Bridge": 9.5,
    "Edong Yangtze River Bridge": 11.0,
    "Cuntan Yangtze River Bridge": 13.0,
    "Baishazhou Yangtze River Bridge": 10.0,
    "Junshan Yangtze River Bridge": 11.0,
    "My Thuan Bridge": 12.0,
    "Can Tho Bridge": 11.5,
    "Bosphorus Bridge": 10.0,
    "Fatih Sultan Mehmet Bridge": 11.0,
    "Tagus River Bridge": 35.0,  # 桁架
    "Golden Horn Bridge": 13.0,
    "Linpu Bridge": 10.5,
    "Jiaxing-Shaoxing Sea Bridge": 12.0,
    "Xiangshan Harbor Bridge": 11.0,
    "Qingdao Bay Bridge": 14.0,

    # 查漏补缺报告: 额外13座桥梁
    "Sidu River Bridge": 9.0,
    "Runyang South Bridge": 9.5,
    "Dongting Lake Bridge": 11.0,
    "E-ling Bridge": 10.0,
    "Jinshajiang Bridge": 12.0,
    "Wufengshan Bridge": 45.0,  # 桁架梁
    "Runyang North Bridge": 11.0,
    "Höga Kusten Bridge": 12.0,
    "Puente de la Constitución de 1812": 14.0,
    "Tsurumi Tsubasa Bridge": 13.0,
    "Bayonne Bridge": 35.0,  # 拱桥桁架
    "E-470 Cable-Stayed Bridge": 15.0,
    "Sartell Bridge": 14.0,

    # 别名处理
    "Pont de Normandie": 9.0,  # = Normandy Bridge
    "Rion-Antirion Bridge": 11.0,  # = Rio-Antirrio Bridge
    "Runyang Bridge": 9.5,  # = Runyang South Bridge
    "Runyang Suspension Bridge": 9.5,
    "Humen Bridge": 8.0,  # = Humen Pearl River Bridge
    "Humen Pearl River Bridge": 8.0,
    "Xihoumen Bridge": 6.0,
    "Stonecutters Bridge": 25.0,
    "E-San-Se Bridge": 9.0,  # = Yi Sun-sin Bridge
    "Jiangyin Yangtze River Bridge": 10.0,  # = Jiangyin Bridge
    "Cheonsa Bridge": 10.5,  # = Aphae-Amtae Bridge
    "Severn Bridge (First)": 8.0,
    "Second Bosphorus Bridge": 11.0,  # = Fatih Sultan Mehmet Bridge
    "25 de Abril Bridge": 35.0,  # = Tagus River Bridge
    "Qingdao Haiwan Bridge": 14.0,  # = Qingdao Bay Bridge (斜拉桥部分)
}

# 数据来源说明
DATA_SOURCE = {
    "Batch 2": "01-数据补全batch2.md - 15座著名国际桥梁",
    "Batch 3": "01-数据补全batch3.md - 14座VIV经典案例",
    "Batch 4": "01-数据补全batch4.md - 47座中国及亚洲桥梁",
    "Batch 5": "01-数据补全batch5.md - 重复验证项（已合并）",
    "Batch 6": "01-数据补全batch6.md - 20座中国新增桥梁",
    "补缺报告": "02-查漏补缺报告.md - 13座补充桥梁",
}

# 统计信息
STATISTICS = {
    "总桥梁数": len(set(BRIDGE_CRITICAL_WIND_SPEED.values())),
    "唯一桥梁数": len(set(BRIDGE_CRITICAL_WIND_SPEED.keys())),
    "风速范围": f"[{min(BRIDGE_CRITICAL_WIND_SPEED.values())}, {max(BRIDGE_CRITICAL_WIND_SPEED.values())}] m/s",
    "平均风速": f"{sum(BRIDGE_CRITICAL_WIND_SPEED.values()) / len(BRIDGE_CRITICAL_WIND_SPEED):.2f} m/s",
}

if __name__ == "__main__":
    print("="*70)
    print("桥梁真实临界风速数据表".center(70))
    print("="*70)
    print(f"\n总记录数: {len(BRIDGE_CRITICAL_WIND_SPEED)}")
    print(f"唯一桥梁数: {len(set(BRIDGE_CRITICAL_WIND_SPEED.keys()))}")
    print(f"风速范围: [{min(BRIDGE_CRITICAL_WIND_SPEED.values())}, {max(BRIDGE_CRITICAL_WIND_SPEED.values())}] m/s")
    print(f"平均风速: {sum(BRIDGE_CRITICAL_WIND_SPEED.values()) / len(BRIDGE_CRITICAL_WIND_SPEED):.2f} m/s")
    print("\n数据来源:")
    for key, value in DATA_SOURCE.items():
        print(f"  - {key}: {value}")
