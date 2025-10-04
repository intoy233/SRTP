#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据收集配置文件
包含搜索关键词、字段映射、数据验证规则等配置
"""

# 搜索关键词配置
SEARCH_KEYWORDS = [
    # 英文关键词
    "bridge vortex-induced vibration",
    "bridge VIV wind tunnel test",
    "long-span bridge VIV",
    "cable-stayed bridge vortex shedding",
    "suspension bridge wind-induced vibration",

    # 中文关键词
    "桥梁涡振 风洞试验",
    "大跨度桥梁 涡激振动",
    "斜拉桥 涡振",
    "悬索桥 风致振动",
]

# CSV字段配置(必须与现有数据集完全一致)
CSV_FIELDS = [
    'BridgeID',
    'BridgeName',
    'BridgeType',
    'Country',
    'PaperSource',
    'Year',
    'Span_m',
    'Width_m',
    'Height_m',
    'Width_Height_Ratio',
    'Total_Length_m',
    'Structure_Type',
    'Natural_Freq_Hz',
    'First_Freq_Hz',
    'Second_Freq_Hz',
    'Drag_Coefficient',
    'Lift_Coefficient',
    'VIV_Wind_Speed_ms',
    'Critical_Wind_Speed_ms',
    'Max_Amplitude_mm',
    'Amplitude_RMS_mm',
    'Damping_Ratio',
    'Vibration_Suppression',
    'Suppression_Effect',
    'Risk_Level',
    'Notes'
]

# 核心字段(必填)
REQUIRED_FIELDS = [
    'BridgeName',
    'Max_Amplitude_mm',  # 目标变量,必须有
]

# 重要字段(优先提取)
IMPORTANT_FIELDS = [
    'Span_m',
    'Width_m',
    'Height_m',
    'Natural_Freq_Hz',
    'Damping_Ratio',
    'Country',
]

# 数据验证规则(合理值范围)
VALIDATION_RULES = {
    'Span_m': (50, 3000),           # 主跨50m-3000m
    'Width_m': (10, 60),             # 宽度10m-60m
    'Height_m': (1, 10),             # 梁高1m-10m
    'Width_Height_Ratio': (3, 20),   # 宽高比3-20
    'Natural_Freq_Hz': (0.05, 2.0),  # 频率0.05-2.0Hz
    'First_Freq_Hz': (0.05, 2.0),
    'Second_Freq_Hz': (0.1, 3.0),
    'Damping_Ratio': (0.001, 0.05),  # 阻尼比0.1%-5%
    'VIV_Wind_Speed_ms': (3, 30),    # 风速3-30m/s
    'Critical_Wind_Speed_ms': (5, 50),
    'Max_Amplitude_mm': (0.1, 500),  # 振幅0.1-500mm
    'Amplitude_RMS_mm': (0.1, 400),
    'Drag_Coefficient': (0.1, 2.0),
    'Lift_Coefficient': (0.01, 1.0),
}

# 单位转换映射
UNIT_CONVERSIONS = {
    # 长度单位
    'm': 1.0,
    'cm': 0.01,
    'mm': 0.001,
    'km': 1000.0,

    # 频率单位
    'Hz': 1.0,
    'rad/s': 1/(2*3.14159),

    # 速度单位
    'm/s': 1.0,
    'km/h': 1/3.6,
    'mph': 0.44704,
}

# 桥梁类型映射
BRIDGE_TYPE_MAPPING = {
    'suspension': 'Suspension',
    'cable-stayed': 'Cable-Stayed',
    'arch': 'Arch',
    'beam': 'Beam',
    'truss': 'Truss',
    '悬索桥': 'Suspension',
    '斜拉桥': 'Cable-Stayed',
    '拱桥': 'Arch',
    '梁桥': 'Beam',
    '桁架桥': 'Truss',
}

# 断面类型映射
STRUCTURE_TYPE_MAPPING = {
    'steel box': 'Steel Box',
    'concrete box': 'Concrete Box',
    'composite': 'Composite',
    'truss': 'Steel Truss',
    '钢箱梁': 'Steel Box',
    '混凝土箱梁': 'Concrete Box',
    '组合梁': 'Composite',
    '桁架': 'Steel Truss',
}

# 国家映射
COUNTRY_MAPPING = {
    'china': 'China',
    'usa': 'USA',
    'japan': 'Japan',
    'uk': 'UK',
    'france': 'France',
    'germany': 'Germany',
    'italy': 'Italy',
    'spain': 'Spain',
    'korea': 'South Korea',
    '中国': 'China',
    '美国': 'USA',
    '日本': 'Japan',
    '英国': 'UK',
    '法国': 'France',
    '德国': 'Germany',
}

# 正则表达式模式
REGEX_PATTERNS = {
    # 桥梁名称
    'bridge_name': r'(?:Bridge|桥)[:\s]*([A-Za-z\u4e00-\u9fa5\s]+)',

    # 跨度
    'span': r'(?:span|跨度)[:\s]*([0-9.]+)\s*(?:m|米)',

    # 宽度
    'width': r'(?:width|宽度|deck width)[:\s]*([0-9.]+)\s*(?:m|米)',

    # 高度
    'height': r'(?:height|高度|depth|梁高)[:\s]*([0-9.]+)\s*(?:m|米)',

    # 频率
    'frequency': r'(?:frequency|频率|natural frequency)[:\s]*([0-9.]+)\s*(?:Hz|赫兹)',

    # 振幅
    'amplitude': r'(?:amplitude|振幅|max amplitude)[:\s]*([0-9.]+)\s*(?:mm|毫米)',

    # 阻尼比
    'damping': r'(?:damping ratio|阻尼比)[:\s]*([0-9.]+)\s*(?:%|percent)?',

    # 风速
    'wind_speed': r'(?:wind speed|风速)[:\s]*([0-9.]+)\s*(?:m/s|米/秒)',
}

# 输出配置
OUTPUT_FILE = 'new_bridge_data.csv'
LOG_FILE = 'data_collection.log'
ERROR_FILE = 'collection_errors.log'

# 爬虫配置
CRAWLER_CONFIG = {
    'max_papers': 100,          # 最多下载论文数
    'timeout': 30,              # 请求超时时间(秒)
    'retry_times': 3,           # 重试次数
    'delay': 2,                 # 请求间隔(秒)
    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
}
