# 桥梁涡激振动(VIV)预测模型 - 数据收集需求清单

**文档版本**: v1.0
**创建日期**: 2025-11-19
**目标用途**: 供Gemini进行学术文献调研,收集更多桥梁VIV实验数据
**最低数据需求**: **200-300条新样本** (使总样本数达到400-500条)

---

## 📋 一、数据收集目标概述

### 1.1 当前数据集现状
- **现有样本数**: 196座桥梁
- **核心问题**: 样本量不足,导致模型性能瓶颈(R²=0.64)
- **高风险样本**: 仅51个(振幅>60mm),无法有效训练

### 1.2 数据扩充目标
- **最低目标**: 新增200条数据 → 总计400条 (提升R²至0.70+)
- **理想目标**: 新增300条数据 → 总计500条 (提升R²至0.75+)
- **优先级**: **高风险样本(振幅>60mm)** 需要新增至少100条

### 1.3 数据来源类型
优先级从高到低:
1. ⭐⭐⭐ **实桥监测数据** (Field Monitoring) - 最高质量,最符合实际
2. ⭐⭐ **风洞试验数据** (Wind Tunnel Test) - 可控条件,数据完整
3. ⭐ **数值仿真数据** (CFD/FEM Simulation) - 需验证准确性
4. **文献报告数据** (Published Papers) - 数据可能不完整

---

## 📊 二、数据格式详细规范

### 2.1 必需字段 (Required Fields) - 缺一不可

以下26个字段是当前模型的**完整输入**,新数据必须包含这些字段:

| 字段名 | 数据类型 | 单位 | 取值范围 | 说明 | 示例值 |
|--------|---------|------|----------|------|--------|
| **BridgeID** | 整数 | - | >0 | 唯一桥梁编号 | 197 |
| **BridgeName** | 文本 | - | - | 桥梁名称(中英文均可) | "Golden Gate Bridge" |
| **BridgeType** | 文本 | - | 见2.2节 | 桥梁类型 | "Suspension" |
| **Country** | 文本 | - | - | 所在国家 | "USA" |
| **PaperSource** | 文本 | - | - | 数据来源文献/报告 | "Journal of Wind Engineering 2023" |
| **Year** | 整数 | 年 | 1990-2025 | 数据获取年份 | 2023 |
| **Span_m** | 浮点数 | 米 | 16-5000 | 主跨跨度 | 1280 |
| **Width_m** | 浮点数 | 米 | 4-60 | 主梁宽度 | 32.5 |
| **Height_m** | 浮点数 | 米 | 2.3-55 | 主梁高度 | 3.8 |
| **Width_Height_Ratio** | 浮点数 | - | >1 | 宽高比 | 8.55 |
| **Total_Length_m** | 浮点数 | 米 | >0 | 桥梁总长度 | 2737 |
| **Structure_Type** | 文本 | - | 见2.3节 | 主梁结构类型 | "Steel Box" |
| **Natural_Freq_Hz** | 浮点数 | Hz | 0.049-0.962 | 一阶自振频率 | 0.182 |
| **First_Freq_Hz** | 浮点数 | Hz | 0.049-0.962 | 第一阶竖向频率 | 0.15 |
| **Second_Freq_Hz** | 浮点数 | Hz | >0 | 第二阶竖向频率 | 0.45 |
| **Drag_Coefficient** | 浮点数 | - | 0-3 | 阻力系数 | 0.8 |
| **Lift_Coefficient** | 浮点数 | - | -2 至 +2 | 升力系数 | 0.17 |
| **VIV_Wind_Speed_ms** | 浮点数 | m/s | >0 | VIV发生时的风速 | 6.5 |
| **Critical_Wind_Speed_ms** | 浮点数 | m/s | 5.1-22 | 临界风速 | 8.7 |
| **Max_Amplitude_mm** | 浮点数 | mm | >0 | **[目标变量]** 最大振幅 | 23.1 |
| **Amplitude_RMS_mm** | 浮点数 | mm | >0 | 振幅均方根 | 18.5 |
| **Damping_Ratio** | 浮点数 | - | 0.005-0.058 | 阻尼比 | 0.02 |
| **Vibration_Suppression** | 文本 | - | 见2.4节 | 是否有减振措施 | "Yes" / "No" |
| **Suppression_Effect** | 文本 | - | - | 减振措施说明 | "TMD installed" |
| **Risk_Level** | 文本 | - | 见2.5节 | 风险等级(自动计算) | "Medium" |
| **Notes** | 文本 | - | - | 备注信息 | "Wind Tunnel Test" |

### 2.2 BridgeType 桥梁类型枚举

当前数据集中已有的桥梁类型:
- `"Suspension"` - 悬索桥 (优先)
- `"Cable-Stayed"` - 斜拉桥 (优先)
- `"Arch"` - 拱桥
- `"Beam"` - 梁桥
- `"Truss"` - 桁架桥
- `"Girder"` - 板梁桥

**⚠️ 重要**: 优先收集**悬索桥**和**斜拉桥**数据,因为这两类桥梁最易发生VIV。

### 2.3 Structure_Type 主梁结构类型枚举

- `"Steel Box"` - 钢箱梁 (最常见,优先)
- `"Truss"` - 桁架结构
- `"Plate Girder"` - 板梁
- `"Streamlined Box"` - 流线型箱梁
- `"Closed Box"` - 闭口箱梁
- `"Open Box"` - 开口箱梁
- `"Composite"` - 组合梁

### 2.4 Vibration_Suppression 减振措施枚举

- `"Yes"` - 有减振措施
- `"No"` - 无减振措施
- `nan` 或 留空 - 未知

如果选择"Yes",请在`Suppression_Effect`字段中详细说明措施类型:
- "TMD" (Tuned Mass Damper 调谐质量阻尼器)
- "Guide Vanes" (导流板)
- "Stabilizer Plates" (稳定板)
- "Fairings" (导流罩)
- "Other: [具体说明]"

### 2.5 Risk_Level 风险等级(可自动计算)

基于`Max_Amplitude_mm`自动分类:
- `"Low"` - 振幅 ≤ 30mm
- `"Medium"` - 30mm < 振幅 ≤ 60mm
- `"High"` - 60mm < 振幅 ≤ 100mm
- `"Critical"` - 振幅 > 100mm

**⚠️ 新数据可留空此字段,模型会自动计算。**

---

## 🎯 三、关键参数优先级与质量要求

### 3.1 核心参数 (Top Priority) - 必须准确

以下参数对模型性能影响最大,**绝对不能缺失或估算**:

| 参数 | 重要性 | 影响模型R²贡献 | 数据来源建议 |
|------|--------|----------------|-------------|
| **Damping_Ratio** (阻尼比) | ⭐⭐⭐⭐⭐ | 15.09% | 实验测量(必须) |
| **Critical_Wind_Speed_ms** (临界风速) | ⭐⭐⭐⭐⭐ | 12.73% | 风洞试验/实桥监测 |
| **Max_Amplitude_mm** (最大振幅) | ⭐⭐⭐⭐⭐ | - | **[目标变量]** 实测/风洞试验 |
| **Span_m** (主跨) | ⭐⭐⭐⭐ | 7.31% | 设计图纸/文献 |
| **Width_m** (主梁宽度) | ⭐⭐⭐ | 4.66% | 设计图纸 |
| **Natural_Freq_Hz** (自振频率) | ⭐⭐⭐⭐ | - | 实测/有限元计算 |

**⚠️ 数据质量红线:**
- `Damping_Ratio`误差 < 5% (例如: 0.020 ± 0.001)
- `Max_Amplitude_mm`误差 < 10% (例如: 50mm ± 5mm)
- `Critical_Wind_Speed_ms`误差 < 5% (例如: 10 m/s ± 0.5 m/s)

### 3.2 次要参数 (Medium Priority) - 可适度估算

以下参数重要性较低,如果文献中未明确提供,可使用经验公式估算:

| 参数 | 估算公式 | 示例 |
|------|---------|------|
| **Width_Height_Ratio** | `Width_m / Height_m` | 32m / 3.8m = 8.42 |
| **Drag_Coefficient** | 钢箱梁: 0.8-1.2; 流线型: 0.4-0.6 | 0.8 (钢箱梁默认值) |
| **Lift_Coefficient** | 钢箱梁: 0.1-0.3; 流线型: -0.1-0.2 | 0.17 (钢箱梁默认值) |
| **First_Freq_Hz** | 通常等于`Natural_Freq_Hz` | 0.182 |
| **Second_Freq_Hz** | 约为`First_Freq_Hz`的2.5-3.5倍 | 0.182 × 3 = 0.546 |

### 3.3 可选参数 (Low Priority) - 可缺失

以下字段缺失率在当前数据集中超过40%,新数据可以不提供:
- `Total_Length_m` (缺失率56.6%) - 非关键特征
- `Year` (缺失率46.9%) - 仅用于文献追溯
- `Amplitude_RMS_mm` (缺失率44.9%) - 可由`Max_Amplitude_mm`估算

---

## 🔍 四、数据来源与检索策略

### 4.1 推荐学术数据库

请Gemini在以下数据库中检索:

1. **Web of Science / SCI期刊** (优先)
   - 关键词: `"Vortex-Induced Vibration"` + `"Bridge"` + `"Wind Tunnel"` / `"Field Monitoring"`
   - 时间范围: 2015-2025 (优先最新数据)
   - 期刊筛选: Q1/Q2区期刊

2. **Elsevier ScienceDirect**
   - 重点期刊:
     - *Journal of Wind Engineering and Industrial Aerodynamics*
     - *Engineering Structures*
     - *Journal of Bridge Engineering*
     - *Wind and Structures*

3. **中国知网 (CNKI)** (中文文献)
   - 关键词: "桥梁涡激振动" / "大跨度桥梁" / "风致振动"
   - 优先: 国家自然科学基金资助项目的论文

4. **会议论文集**
   - International Conference on Wind Engineering (ICWE)
   - International Symposium on Bridge and Structural Engineering (ISBSE)

### 4.2 数据提取优先级

从文献中提取数据时,按以下优先级:

**优先级1**: 文中**表格数据** (Table)
- 示例: "Table 3: Wind Tunnel Test Results"
- 优势: 结构化,字段完整

**优先级2**: **实验结果图表** (Figure)
- 示例: "Figure 5: Amplitude vs Wind Speed"
- 需要: 使用工具提取数值点

**优先级3**: **正文描述** (Text)
- 示例: "The maximum amplitude was measured as 45mm at a wind speed of 12 m/s."
- 缺点: 信息分散,需人工整合

### 4.3 桥梁案例优先级

**高优先级桥梁** (全球知名VIV案例):
- 中国:
  - ✅ 虎门大桥 (Humen Bridge) - 已收录,寻找更多工况
  - 西堠门大桥 (Xihoumen Bridge)
  - 泰州长江大桥 (Taizhou Yangtze River Bridge)
  - 港珠澳大桥 (Hong Kong-Zhuhai-Macao Bridge)
  - 苏通大桥 (Sutong Bridge)
  - 润扬长江大桥 (Runyang Bridge)

- 日本:
  - 明石海峡大桥 (Akashi Kaikyo Bridge)
  - 濑户大桥 (Seto Ohashi Bridge)
  - 来岛海峡大桥 (Kurushima-Kaikyo Bridge)

- 欧美:
  - ✅ 金门大桥 (Golden Gate Bridge) - 已收录
  - Tacoma Narrows Bridge (USA)
  - Great Belt Bridge (Denmark)
  - Stonecutters Bridge (Hong Kong)

**中优先级**: 跨度500-1500m的中大型桥梁

**低优先级**: 跨度<500m的中小型桥梁

---

## 📐 五、数据完整性与一致性检查

### 5.1 数据完整性检查清单

对于每一条新数据,请Gemini执行以下检查:

- [ ] **必需字段检查**: 26个必需字段是否全部填写?
- [ ] **数值范围检查**: 是否在合理范围内? (见第二节表格)
- [ ] **单位一致性**: 是否使用国际单位制(SI)?
  - ✅ 正确: 跨度 = 1280 (单位: 米)
  - ❌ 错误: 跨度 = 4200 (单位: 英尺,需转换)
- [ ] **物理一致性**: 是否违反物理规律?
  - 检查: `Width_Height_Ratio ≈ Width_m / Height_m` (误差<5%)
  - 检查: `VIV_Wind_Speed_ms ≤ Critical_Wind_Speed_ms` (涡激风速通常低于临界风速)
  - 检查: `Amplitude_RMS_mm ≤ Max_Amplitude_mm` (均方根≤最大值)

### 5.2 数据质量等级分类

根据数据来源和完整性,对新数据进行分级:

| 等级 | 标准 | 处理方式 |
|------|------|----------|
| **A级** (优质) | 实桥监测或风洞试验,26个字段完整,核心参数误差<5% | 直接采纳 |
| **B级** (可用) | 文献数据,核心字段完整,次要字段估算 | 标注来源后采纳 |
| **C级** (存疑) | 仿真数据或缺失>5个核心字段 | 需人工复核 |
| **D级** (拒绝) | 缺失目标变量`Max_Amplitude_mm`或核心参数 | 不采纳 |

### 5.3 数据去重检查

避免重复收集已有桥梁数据:

**检查方法**:
1. 比对`BridgeName` + `Country` (桥梁名称+国家)
2. 比对`Span_m` + `Natural_Freq_Hz` (跨度+频率,容差±5%)

**如果发现重复**:
- 保留数据质量更高的版本
- 如果是同一桥梁的不同工况(如不同风速),则作为新样本保留

---

## 📂 六、数据提交格式

### 6.1 CSV文件格式

新数据请以**CSV格式**提交,严格遵循以下规范:

```csv
BridgeID,BridgeName,BridgeType,Country,PaperSource,Year,Span_m,Width_m,Height_m,Width_Height_Ratio,Total_Length_m,Structure_Type,Natural_Freq_Hz,First_Freq_Hz,Second_Freq_Hz,Drag_Coefficient,Lift_Coefficient,VIV_Wind_Speed_ms,Critical_Wind_Speed_ms,Max_Amplitude_mm,Amplitude_RMS_mm,Damping_Ratio,Vibration_Suppression,Suppression_Effect,Risk_Level,Notes
197,Xihoumen Bridge,Suspension,China,Journal of Bridge Engineering 2023,2023,1650,36.0,3.5,10.29,2589,Steel Box,0.156,0.156,0.468,0.85,0.19,7.2,9.8,68.5,54.2,0.018,Yes,Guide Vanes,High,Wind Tunnel Test
198,Akashi Kaikyo Bridge,Suspension,Japan,Wind Engineering 2022,2022,1991,35.5,14.0,2.54,3911,Truss,0.082,0.082,0.287,1.20,0.35,8.5,11.2,42.3,33.8,0.012,No,,Medium,Field Monitoring
...
```

**⚠️ 格式要求:**
- 编码: **UTF-8 with BOM** (确保中文不乱码)
- 分隔符: 逗号 `,`
- 文本字段: 如果包含逗号,需用双引号包裹 (例如: `"Wind Tunnel Test, 2023"`)
- 缺失值: 留空或使用`nan`
- 小数点: 使用`.`而非`,` (例如: `3.14`而非`3,14`)

### 6.2 数据命名规范

文件命名格式:
```
bridge_viv_data_[来源]_[日期].csv
```

示例:
- `bridge_viv_data_gemini_research_20251119.csv`
- `bridge_viv_data_jweia_journal_20251120.csv`

### 6.3 数据提交包结构

每批次数据提交时,请包含以下文件:

```
提交包/
├── bridge_viv_data_[来源]_[日期].csv     # 新数据CSV文件
├── data_sources.md                       # 数据来源文献列表
├── data_quality_report.md                # 数据质量报告
└── extraction_notes.txt                  # 提取过程备注
```

**data_sources.md 示例**:
```markdown
# 数据来源文献列表

## 样本197: Xihoumen Bridge
- **文献标题**: Wind-Induced Vibration Analysis of Xihoumen Bridge
- **作者**: Zhang et al.
- **期刊**: Journal of Bridge Engineering, Vol. 28, No. 5, 2023
- **DOI**: 10.1061/(ASCE)BE.1943-5592.0001985
- **数据位置**: Table 4, Page 12
- **数据质量**: A级 (风洞试验数据)

## 样本198: Akashi Kaikyo Bridge
...
```

---

## ✅ 七、数据验证脚本

为确保新数据可以直接导入模型,我们提供了**自动验证脚本**。

### 7.1 验证脚本使用方法

在提交数据前,请使用以下Python脚本验证数据格式:

```python
# 保存为: scripts/validate_new_data.py

import pandas as pd
import numpy as np

def validate_bridge_data(csv_path):
    """验证新收集的桥梁VIV数据"""
    print(f"正在验证文件: {csv_path}")

    # 1. 读取数据
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except Exception as e:
        print(f"❌ 文件读取失败: {e}")
        return False

    # 2. 检查必需字段
    required_cols = [
        'BridgeID', 'BridgeName', 'BridgeType', 'Country', 'PaperSource',
        'Year', 'Span_m', 'Width_m', 'Height_m', 'Width_Height_Ratio',
        'Total_Length_m', 'Structure_Type', 'Natural_Freq_Hz', 'First_Freq_Hz',
        'Second_Freq_Hz', 'Drag_Coefficient', 'Lift_Coefficient',
        'VIV_Wind_Speed_ms', 'Critical_Wind_Speed_ms', 'Max_Amplitude_mm',
        'Amplitude_RMS_mm', 'Damping_Ratio', 'Vibration_Suppression',
        'Suppression_Effect', 'Risk_Level', 'Notes'
    ]

    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        print(f"❌ 缺失必需字段: {missing_cols}")
        return False
    print(f"✅ 所有必需字段完整 ({len(required_cols)}个)")

    # 3. 检查核心字段缺失率
    core_fields = ['Damping_Ratio', 'Critical_Wind_Speed_ms', 'Max_Amplitude_mm',
                   'Span_m', 'Width_m', 'Natural_Freq_Hz']

    for field in core_fields:
        missing_rate = df[field].isnull().sum() / len(df) * 100
        if missing_rate > 10:
            print(f"⚠️  核心字段 '{field}' 缺失率 {missing_rate:.1f}% (建议<10%)")
        else:
            print(f"✅ 核心字段 '{field}' 缺失率 {missing_rate:.1f}%")

    # 4. 检查数值范围
    range_checks = {
        'Span_m': (16, 5000),
        'Width_m': (4, 60),
        'Height_m': (2.3, 55),
        'Natural_Freq_Hz': (0.049, 0.962),
        'Critical_Wind_Speed_ms': (5.1, 22),
        'Max_Amplitude_mm': (0, 200),
        'Damping_Ratio': (0.005, 0.058)
    }

    for field, (min_val, max_val) in range_checks.items():
        if field in df.columns:
            valid = df[field].dropna()
            out_of_range = ((valid < min_val) | (valid > max_val)).sum()
            if out_of_range > 0:
                print(f"⚠️  字段 '{field}' 有 {out_of_range} 个值超出范围 [{min_val}, {max_val}]")
            else:
                print(f"✅ 字段 '{field}' 数值范围正常")

    # 5. 物理一致性检查
    inconsistencies = 0

    # 检查宽高比
    if all(c in df.columns for c in ['Width_m', 'Height_m', 'Width_Height_Ratio']):
        df['calc_ratio'] = df['Width_m'] / df['Height_m']
        ratio_error = np.abs(df['calc_ratio'] - df['Width_Height_Ratio']) / df['calc_ratio']
        if (ratio_error > 0.05).any():
            inconsistencies += (ratio_error > 0.05).sum()
            print(f"⚠️  有 {(ratio_error > 0.05).sum()} 个样本的宽高比计算不一致(误差>5%)")

    # 检查VIV风速 vs 临界风速
    if all(c in df.columns for c in ['VIV_Wind_Speed_ms', 'Critical_Wind_Speed_ms']):
        invalid = (df['VIV_Wind_Speed_ms'] > df['Critical_Wind_Speed_ms']).sum()
        if invalid > 0:
            inconsistencies += invalid
            print(f"⚠️  有 {invalid} 个样本的VIV风速大于临界风速(物理不合理)")

    if inconsistencies == 0:
        print("✅ 物理一致性检查通过")

    # 6. 汇总
    print(f"\n{'='*60}")
    print(f"验证完成! 数据集包含 {len(df)} 条样本")
    print(f"{'='*60}")

    return True

# 使用示例
if __name__ == '__main__':
    validate_bridge_data('path/to/your/new_data.csv')
```

**运行命令**:
```bash
python scripts/validate_new_data.py
```

### 7.2 验证通过标准

新数据应满足以下标准才能提交:
- ✅ 所有26个必需字段存在
- ✅ 核心字段(`Damping_Ratio`, `Max_Amplitude_mm`等)缺失率<10%
- ✅ 数值范围在合理区间内
- ✅ 物理一致性检查通过
- ✅ 无重复样本

---

## 📈 八、数据收集进度跟踪

### 8.1 收集进度表

| 阶段 | 目标样本数 | 当前进度 | 截止日期 | 负责人 |
|------|-----------|---------|----------|--------|
| 第一批 | 100条 | 0/100 | 待定 | Gemini |
| 第二批 | 100条 | 0/100 | 待定 | Gemini |
| 第三批 | 100条 | 0/100 | 待定 | Gemini |
| **总计** | **300条** | **0/300** | - | - |

### 8.2 高风险样本进度表

| 振幅区间 | 当前数量 | 目标数量 | 缺口 | 优先级 |
|---------|---------|---------|------|--------|
| 60-100mm | 47 | 100 | 53 | ⭐⭐⭐ |
| >100mm | 4 | 50 | 46 | ⭐⭐⭐⭐⭐ |
| **高风险合计** | **51** | **150** | **99** | - |

**⚠️ 关键**: 新增数据中**至少1/3应为高风险样本**(振幅>60mm)!

---

## 🚀 九、数据集成与模型重训练计划

### 9.1 数据集成流程

1. **数据验证**: 使用第7节验证脚本检查新数据
2. **数据清洗**: 处理缺失值、异常值
3. **数据合并**: 将新数据追加到`final_bridge_dataset.csv`
4. **去重检查**: 确保无重复样本
5. **数据集切分**: 重新划分训练集/测试集

### 9.2 模型性能预期

根据统计学习理论,样本量与模型性能的关系:

| 总样本数 | 样本/特征比 | 预期整体R² | 预期高风险R² | 备注 |
|---------|------------|-----------|-------------|------|
| 196 (当前) | 2.51 | 0.64 | -1.48 | 当前状态 |
| 300 | 3.85 | 0.68-0.72 | 0.20-0.40 | 最低目标 |
| 400 | 5.13 | 0.72-0.76 | 0.45-0.55 | 理想目标 |
| 500 | 6.41 | 0.75-0.80 | 0.55-0.65 | 最优状态 |

**⚠️ 关键结论**:
- 样本数达到400时,样本/特征比>5,可以有效训练78维特征模型
- 高风险R²预期从-1.48提升至0.45+,是质的飞跃!

---

## 📞 十、联系与支持

### 10.1 数据提交方式

- **方式1**: 发送至邮箱 [待填写]
- **方式2**: 上传至云盘 [待填写]
- **方式3**: GitHub Pull Request [待填写]

### 10.2 问题反馈

如果在数据收集过程中遇到以下问题,请及时反馈:
- 文献中数据格式不清晰
- 某些参数定义存在歧义
- 单位制不统一
- 数据质量存疑

### 10.3 致谢

感谢Gemini协助进行文献调研和数据收集!您的贡献将直接提升桥梁VIV预测模型的性能,为桥梁工程安全保驾护航!

---

**文档末尾**

**吴先生审阅签名**: __________________
**日期**: __________________

**Gemini接收确认**: __________________
**日期**: __________________
