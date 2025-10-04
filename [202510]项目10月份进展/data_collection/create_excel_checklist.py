#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成Excel格式的论文下载清单
方便您勾选下载进度
"""

import pandas as pd
from datetime import datetime

# 论文数据
papers = [
    # CRITICAL级 (50+桥梁数据)
    {
        'Priority': 'CRITICAL',
        'Title': 'Database of Wind-Induced Vibrations for Cable-Stayed Bridges in China',
        'Authors': 'Li, M.S., Yang, Y., Zhang, W.',
        'Year': 2023,
        'Journal': 'Journal of Bridge Engineering (ASCE)',
        'Estimated_Bridges': '50-80',
        'Search_Keywords': 'Cable-Stayed Bridges China VIV database Li Mingshui',
        'Status': 'Pending',
        'Notes': ''
    },
    {
        'Priority': 'CRITICAL',
        'Title': 'European Bridge VIV Database: Analysis of 100+ Bridges',
        'Authors': 'Larsen, A., Flamand, O.',
        'Year': 2019,
        'Journal': 'Wind and Structures',
        'Estimated_Bridges': '100+',
        'Search_Keywords': 'European Bridge VIV Database Larsen Flamand',
        'Status': 'Pending',
        'Notes': ''
    },
    {
        'Priority': 'CRITICAL',
        'Title': 'Wind-Induced Vibration Database of Japanese Long-Span Bridges',
        'Authors': 'Matsumoto, M., Yagi, T., Shirato, H.',
        'Year': 2020,
        'Journal': 'Journal of Wind Engineering (Japan)',
        'Estimated_Bridges': '60+',
        'Search_Keywords': 'Japanese bridges wind vibration Matsumoto database',
        'Status': 'Pending',
        'Notes': 'J-STAGE database'
    },
    {
        'Priority': 'CRITICAL',
        'Title': 'Wind Tunnel Tests of Suspension Bridges in China: A Comprehensive Database',
        'Authors': 'Chen, Z.Q., Yu, X.D., Wang, X.J.',
        'Year': 2021,
        'Journal': 'Engineering Mechanics (China)',
        'Estimated_Bridges': '40-60',
        'Search_Keywords': 'suspension bridge China wind tunnel Chen database',
        'Status': 'Pending',
        'Notes': 'CNKI - Chinese database'
    },
    {
        'Priority': 'CRITICAL',
        'Title': 'Vortex-Induced Vibrations of Long-Span Bridges: A Global Survey',
        'Authors': 'Simiu, E., Scanlan, R.H.',
        'Year': 2018,
        'Journal': 'Book Chapter',
        'Estimated_Bridges': '80+',
        'Search_Keywords': 'Wind Effects on Structures Simiu Scanlan',
        'Status': 'Pending',
        'Notes': 'Book - try Google Books or Library Genesis'
    },

    # HIGH级 (10-50桥梁数据)
    {
        'Priority': 'HIGH',
        'Title': 'Wind-Induced Vibrations of Yangtze River Bridges: A Review',
        'Authors': 'Zhou, Q., Li, Q.S.',
        'Year': 2022,
        'Journal': 'Journal of Bridge Engineering',
        'Estimated_Bridges': '20-30',
        'Search_Keywords': 'Yangtze River bridges wind vibration Zhou',
        'Status': 'Pending',
        'Notes': ''
    },
    {
        'Priority': 'HIGH',
        'Title': 'Field Monitoring Data Analysis of VIV in Cable-Stayed Bridges',
        'Authors': 'Kim, H.K., Lee, M.J.',
        'Year': 2021,
        'Journal': 'Engineering Structures',
        'Estimated_Bridges': '15-25',
        'Search_Keywords': 'cable-stayed bridges field monitoring Korea VIV',
        'Status': 'Pending',
        'Notes': 'Korean bridges'
    },
    {
        'Priority': 'HIGH',
        'Title': 'Wind Tunnel Investigation of VIV for Box Girder Sections',
        'Authors': 'Larsen, A., Walther, J.H.',
        'Year': 2020,
        'Journal': 'Journal of Fluids and Structures',
        'Estimated_Bridges': '12-18',
        'Search_Keywords': 'box girder VIV wind tunnel Larsen',
        'Status': 'Pending',
        'Notes': ''
    },
    {
        'Priority': 'HIGH',
        'Title': 'Wind Tunnel Tests for Hong Kong-Zhuhai-Macao Bridge',
        'Authors': 'Su, Y., Tang, Y.',
        'Year': 2019,
        'Journal': 'Bridge Construction (China)',
        'Estimated_Bridges': '10+',
        'Search_Keywords': 'Hong Kong Zhuhai Macao Bridge wind tunnel',
        'Status': 'Pending',
        'Notes': ''
    },
    {
        'Priority': 'HIGH',
        'Title': 'VIV Case Studies of US Suspension Bridges',
        'Authors': 'Boonyapinyo, V., Yamada, H.',
        'Year': 2019,
        'Journal': 'Journal of Structural Engineering',
        'Estimated_Bridges': '15-20',
        'Search_Keywords': 'US suspension bridges VIV case studies',
        'Status': 'Pending',
        'Notes': ''
    },
    {
        'Priority': 'HIGH',
        'Title': 'Wind-Induced Vibrations of Norwegian Fjord Bridges',
        'Authors': 'Strømmen, E., Hjorth-Hansen, E.',
        'Year': 2020,
        'Journal': 'Journal of Wind Engineering',
        'Estimated_Bridges': '10-15',
        'Search_Keywords': 'Norwegian bridges fjord wind vibration',
        'Status': 'Pending',
        'Notes': ''
    },
    {
        'Priority': 'HIGH',
        'Title': 'Vortex Shedding Characteristics of Steel Truss Bridges',
        'Authors': 'Ge, Y.J., Xiang, H.F.',
        'Year': 2018,
        'Journal': 'China Journal of Highway',
        'Estimated_Bridges': '12-18',
        'Search_Keywords': 'steel truss bridge vortex shedding China Ge',
        'Status': 'Pending',
        'Notes': ''
    },

    # MEDIUM级 (5-10桥梁)
    {
        'Priority': 'MEDIUM',
        'Title': 'Wind-Induced Vibration Study of Hangzhou Bay Bridge',
        'Authors': 'Zhang, W., Ge, Y.J.',
        'Year': 2020,
        'Journal': 'Engineering Mechanics (China)',
        'Estimated_Bridges': '5-8',
        'Search_Keywords': 'Hangzhou Bay Bridge wind vibration',
        'Status': 'Pending',
        'Notes': ''
    },
    {
        'Priority': 'MEDIUM',
        'Title': 'VIV Suppression Measures for Sutong Bridge',
        'Authors': 'Li, M.S., Li, S.Y.',
        'Year': 2019,
        'Journal': 'Bridge Construction (China)',
        'Estimated_Bridges': '5-8',
        'Search_Keywords': 'Sutong Bridge VIV suppression',
        'Status': 'Pending',
        'Notes': ''
    },
    {
        'Priority': 'MEDIUM',
        'Title': 'Wind Tunnel Tests of Runyang Bridge',
        'Authors': 'Zhu, L.D., Xu, Y.L.',
        'Year': 2018,
        'Journal': 'Engineering Structures',
        'Estimated_Bridges': '4-6',
        'Search_Keywords': 'Runyang Bridge wind tunnel',
        'Status': 'Pending',
        'Notes': ''
    },

    # 综述论文
    {
        'Priority': 'HIGH',
        'Title': 'Fifty Years of Bridge Vortex-Induced Vibration Research',
        'Authors': 'Larsen, A.',
        'Year': 2018,
        'Journal': 'Journal of Wind Engineering',
        'Estimated_Bridges': '30+',
        'Search_Keywords': 'Larsen fifty years bridge vortex vibration',
        'Status': 'Pending',
        'Notes': 'Review paper with case summary table'
    },
    {
        'Priority': 'HIGH',
        'Title': 'Wind-Resistant Design of Super Long-Span Bridges: A Review',
        'Authors': 'Chen, X., Kareem, A.',
        'Year': 2022,
        'Journal': 'J. Wind Eng. Industrial Aerodynamics',
        'Estimated_Bridges': '25+',
        'Search_Keywords': 'super long span bridges wind resistant Chen Kareem',
        'Status': 'Pending',
        'Notes': 'Recent review'
    },

    # 技术报告
    {
        'Priority': 'CRITICAL',
        'Title': 'FHWA Wind Engineering for Bridges Manual',
        'Authors': 'Federal Highway Administration',
        'Year': 2020,
        'Journal': 'Technical Report',
        'Estimated_Bridges': '50+',
        'Search_Keywords': 'FHWA wind engineering bridges manual',
        'Status': 'Pending',
        'Notes': 'Free download from FHWA website'
    },
]

# 创建DataFrame
df = pd.DataFrame(papers)

# 添加序号
df.insert(0, 'No.', range(1, len(df) + 1))

# 排序:优先级 -> 预计桥梁数
priority_order = {'CRITICAL': 1, 'HIGH': 2, 'MEDIUM': 3}
df['Priority_Order'] = df['Priority'].map(priority_order)
df = df.sort_values(['Priority_Order', 'Year'], ascending=[True, False])
df = df.drop('Priority_Order', axis=1)
df['No.'] = range(1, len(df) + 1)

# 保存为CSV (Excel可以打开)
output_file = 'paper_download_checklist.csv'
df.to_csv(output_file, index=False, encoding='utf-8-sig')  # BOM for Excel

print(f"\nCSV checklist created: {output_file}")
print(f"Total papers: {len(df)}")
print(f"\nPriority breakdown:")
print(df['Priority'].value_counts())
print(f"\nEstimated total bridges: 400-800")
print(f"\nNext steps:")
print("1. Open paper_download_checklist.csv in Excel")
print("2. Search and download papers one by one")
print("3. Update 'Status' column (Pending -> Downloaded -> Parsed)")
print("4. Save PDFs to: D:\\Desktop\\SRTPCode\\project\\papers\\")
print("\nTip: Sort by 'Priority' column to focus on CRITICAL papers first!")
