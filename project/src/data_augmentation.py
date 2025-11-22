#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据扩增模块 - 针对桥梁VIV小数据集问题
基于物理原理和现有数据生成合理的合成样本
"""

import numpy as np
import pandas as pd
import random
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class BridgeVIVDataAugmenter:
    """桥梁VIV数据扩增器"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)

        # 基于工程经验的参数范围
        self.parameter_ranges = {
            'Span_m': (200, 3000),           # 桥梁跨度范围
            'Width_m': (15, 60),             # 桥面宽度范围
            'Height_m': (1.5, 8.0),          # 梁高范围
            'Natural_Freq_Hz': (0.05, 1.0),  # 自振频率范围
            'VIV_Wind_Speed_ms': (4, 15),    # VIV风速范围
            'Critical_Wind_Speed_ms': (8, 25), # 临界风速范围
            'Damping_Ratio': (0.002, 0.030), # 阻尼比范围
        }

        # 桥梁类型分布权重
        self.bridge_types = {
            'Suspension': 0.15,      # 悬索桥
            'Cable-Stayed': 0.35,    # 斜拉桥
            'Girder': 0.40,          # 梁桥
            'Arch': 0.10            # 拱桥
        }

        # 结构形式分布
        self.structure_types = {
            'Steel Box': 0.4,
            'Concrete Box': 0.3,
            'Steel Truss': 0.15,
            'Composite': 0.15
        }

    def analyze_existing_data(self, df: pd.DataFrame) -> Dict:
        """分析现有数据的分布特征"""
        logger.info("分析现有数据分布...")

        analysis = {
            'correlations': {},
            'distributions': {},
            'outliers': {},
            'patterns': {}
        }

        # 分析数值特征的分布
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in df.columns:
                values = df[col].dropna()
                analysis['distributions'][col] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'median': float(values.median()),
                    'q25': float(values.quantile(0.25)),
                    'q75': float(values.quantile(0.75))
                }

        # 分析重要的物理关系
        if all(col in df.columns for col in ['Span_m', 'Natural_Freq_Hz']):
            # 跨度与频率的关系
            correlation = df['Span_m'].corr(df['Natural_Freq_Hz'])
            analysis['correlations']['span_frequency'] = float(correlation)

        if all(col in df.columns for col in ['Width_m', 'Height_m']):
            # 宽高比分析
            analysis['patterns']['width_height_ratio'] = {
                'mean': float((df['Width_m'] / df['Height_m']).mean()),
                'std': float((df['Width_m'] / df['Height_m']).std())
            }

        if all(col in df.columns for col in ['VIV_Wind_Speed_ms', 'Max_Amplitude_mm']):
            # 风速与振幅关系
            correlation = df['VIV_Wind_Speed_ms'].corr(df['Max_Amplitude_mm'])
            analysis['correlations']['wind_amplitude'] = float(correlation)

        logger.info(f"数据分析完成: {len(analysis['distributions'])}个特征分布")
        return analysis

    def generate_synthetic_bridge(self, base_data_stats: Dict) -> Dict:
        """基于物理原理生成一座合成桥梁"""

        # 1. 生成基础几何参数
        bridge_type = self._sample_bridge_type()
        structure_type = self._sample_structure_type()

        # 根据桥型调整参数范围
        span_range = self._adjust_span_range_by_type(bridge_type)
        span = random.uniform(*span_range)

        # 2. 基于跨度生成其他几何参数
        width = self._generate_width_from_span(span, bridge_type)
        height = self._generate_height_from_span_width(span, width, structure_type)

        # 3. 生成动力学参数
        natural_freq = self._generate_frequency_from_span(span, structure_type)
        first_freq = natural_freq * random.uniform(0.8, 0.95)  # 一阶频率略低于自振频率
        second_freq = natural_freq * random.uniform(2.0, 3.5)  # 二阶频率

        # 4. 生成阻尼参数
        damping_ratio = self._generate_damping_ratio(structure_type)

        # 5. 生成气动参数
        drag_coeff, lift_coeff = self._generate_aerodynamic_coeffs(width/height, structure_type)

        # 6. 生成风速参数
        viv_wind_speed = self._generate_viv_wind_speed(natural_freq, height)
        critical_wind_speed = viv_wind_speed * random.uniform(1.3, 2.0)

        # 7. 基于物理模型计算振幅
        amplitude, amplitude_rms = self._calculate_viv_amplitude(
            span, width, height, natural_freq, damping_ratio,
            viv_wind_speed, drag_coeff, lift_coeff
        )

        # 8. 确定风险等级
        risk_level = self._classify_risk_level(amplitude)

        # 9. 生成抑振措施
        vibration_suppression, suppression_effect = self._generate_suppression_measures(
            amplitude, bridge_type
        )

        # 构建合成桥梁数据
        synthetic_bridge = {
            'BridgeID': f'SYN_{random.randint(1000, 9999)}',
            'BridgeName': f'Synthetic Bridge {random.randint(100, 999)}',
            'BridgeType': bridge_type,
            'PaperSource': 'Synthetic Data',
            'Year': str(random.randint(2015, 2024)),
            'Span_m': round(span, 1),
            'Width_m': round(width, 1),
            'Height_m': round(height, 2),
            'Width_Height_Ratio': round(width/height, 2),
            'Total_Length_m': round(span * random.uniform(1.2, 2.5), 0),
            'Structure_Type': structure_type,
            'Natural_Freq_Hz': round(natural_freq, 3),
            'First_Freq_Hz': round(first_freq, 3),
            'Second_Freq_Hz': round(second_freq, 3),
            'Drag_Coefficient': round(drag_coeff, 2),
            'Lift_Coefficient': round(lift_coeff, 2),
            'VIV_Wind_Speed_ms': round(viv_wind_speed, 1),
            'Critical_Wind_Speed_ms': round(critical_wind_speed, 1),
            'Max_Amplitude_mm': round(amplitude, 1),
            'Amplitude_RMS_mm': round(amplitude_rms, 1),
            'Damping_Ratio': round(damping_ratio, 4),
            'Vibration_Suppression': vibration_suppression,
            'Suppression_Effect': suppression_effect,
            'Risk_Level': risk_level,
            'Notes': 'Generated by physics-based synthesis'
        }

        return synthetic_bridge

    def _sample_bridge_type(self) -> str:
        """按权重采样桥梁类型"""
        types, weights = zip(*self.bridge_types.items())
        return random.choices(types, weights=weights)[0]

    def _sample_structure_type(self) -> str:
        """按权重采样结构形式"""
        types, weights = zip(*self.structure_types.items())
        return random.choices(types, weights=weights)[0]

    def _adjust_span_range_by_type(self, bridge_type: str) -> Tuple[float, float]:
        """根据桥型调整跨度范围"""
        base_min, base_max = self.parameter_ranges['Span_m']

        if bridge_type == 'Suspension':
            return (800, 3000)      # 悬索桥大跨度
        elif bridge_type == 'Cable-Stayed':
            return (400, 1500)      # 斜拉桥中大跨度
        elif bridge_type == 'Girder':
            return (200, 800)       # 梁桥中小跨度
        elif bridge_type == 'Arch':
            return (300, 1200)      # 拱桥中等跨度
        else:
            return (base_min, base_max)

    def _generate_width_from_span(self, span: float, bridge_type: str) -> float:
        """基于跨度和桥型生成桥面宽度"""
        # 跨度越大，桥面相对越宽
        base_width = 20 + span * 0.01  # 基础关系

        # 桥型修正
        if bridge_type == 'Suspension':
            multiplier = random.uniform(1.2, 1.8)
        elif bridge_type == 'Cable-Stayed':
            multiplier = random.uniform(1.0, 1.5)
        elif bridge_type == 'Girder':
            multiplier = random.uniform(0.8, 1.2)
        else:  # Arch
            multiplier = random.uniform(0.9, 1.3)

        width = base_width * multiplier
        return max(15, min(60, width))  # 限制在合理范围

    def _generate_height_from_span_width(self, span: float, width: float, structure_type: str) -> float:
        """基于跨度、宽度和结构形式生成梁高"""
        # 基于跨度的梁高估算 (跨度/高度比通常在30-60之间)
        target_ratio = random.uniform(30, 60)
        base_height = span / target_ratio

        # 结构形式修正
        if structure_type == 'Steel Box':
            multiplier = random.uniform(0.8, 1.2)
        elif structure_type == 'Concrete Box':
            multiplier = random.uniform(1.0, 1.4)
        elif structure_type == 'Steel Truss':
            multiplier = random.uniform(1.2, 1.8)
        else:  # Composite
            multiplier = random.uniform(0.9, 1.3)

        height = base_height * multiplier
        return max(1.5, min(8.0, height))

    def _generate_frequency_from_span(self, span: float, structure_type: str) -> float:
        """基于跨度和结构形式生成自振频率"""
        # 经验公式: f ≈ C / L^2 (C是常数，L是跨度)
        if structure_type == 'Steel Box':
            C = random.uniform(800000, 1200000)
        elif structure_type == 'Concrete Box':
            C = random.uniform(600000, 1000000)
        elif structure_type == 'Steel Truss':
            C = random.uniform(500000, 900000)
        else:  # Composite
            C = random.uniform(700000, 1100000)

        frequency = C / (span ** 2)
        return max(0.05, min(1.0, frequency))

    def _generate_damping_ratio(self, structure_type: str) -> float:
        """基于结构形式生成阻尼比"""
        if structure_type == 'Steel Box':
            mean_damping = 0.008
            std_damping = 0.003
        elif structure_type == 'Concrete Box':
            mean_damping = 0.015
            std_damping = 0.005
        elif structure_type == 'Steel Truss':
            mean_damping = 0.012
            std_damping = 0.004
        else:  # Composite
            mean_damping = 0.010
            std_damping = 0.003

        damping = np.random.normal(mean_damping, std_damping)
        return max(0.002, min(0.030, damping))

    def _generate_aerodynamic_coeffs(self, width_height_ratio: float, structure_type: str) -> Tuple[float, float]:
        """基于宽高比和结构形式生成气动力系数"""
        # 宽高比对气动力系数的影响
        if width_height_ratio < 3:
            # 窄高断面
            drag_base = 1.2
            lift_base = 0.3
        elif width_height_ratio < 8:
            # 中等宽高比
            drag_base = 0.9
            lift_base = 0.2
        else:
            # 宽扁断面
            drag_base = 0.7
            lift_base = 0.1

        # 结构形式修正
        if structure_type == 'Steel Box':
            drag_coeff = drag_base * random.uniform(0.8, 1.2)
            lift_coeff = lift_base * random.uniform(0.7, 1.3)
        elif structure_type == 'Concrete Box':
            drag_coeff = drag_base * random.uniform(0.9, 1.3)
            lift_coeff = lift_base * random.uniform(0.8, 1.4)
        elif structure_type == 'Steel Truss':
            drag_coeff = drag_base * random.uniform(1.5, 2.0)  # 桁架阻力更大
            lift_coeff = lift_base * random.uniform(0.5, 1.0)
        else:  # Composite
            drag_coeff = drag_base * random.uniform(0.85, 1.15)
            lift_coeff = lift_base * random.uniform(0.75, 1.25)

        return max(0.3, min(2.5, drag_coeff)), max(0.05, min(0.8, lift_coeff))

    def _generate_viv_wind_speed(self, natural_freq: float, height: float) -> float:
        """基于频率和尺寸生成VIV风速"""
        # 基于Strouhal数关系: St = f*D/U ≈ 0.2
        strouhal = random.uniform(0.15, 0.25)
        viv_speed = natural_freq * height / strouhal

        # 添加随机扰动
        viv_speed *= random.uniform(0.8, 1.2)

        return max(4, min(15, viv_speed))

    def _calculate_viv_amplitude(self, span: float, width: float, height: float,
                               natural_freq: float, damping_ratio: float,
                               viv_wind_speed: float, drag_coeff: float, lift_coeff: float) -> Tuple[float, float]:
        """基于物理模型计算VIV振幅"""

        # 简化的VIV振幅估算模型
        # 基于无量纲参数

        # Reynolds数影响
        reynolds = viv_wind_speed * height / 1.5e-5  # 假设运动粘度
        reynolds_factor = min(2.0, math.log10(reynolds / 1e5) + 1)

        # 质量阻尼参数 (简化)
        mass_damping = damping_ratio * 1000  # 假设密度比例

        # Scruton数影响 (St = 2*m*ζ/ρ*D^2)
        scruton_number = mass_damping
        scruton_factor = 1 / (1 + scruton_number / 10)

        # 宽高比影响
        aspect_ratio_factor = min(1.5, width / height / 5)

        # 频率参数影响
        freq_factor = 1 / (1 + natural_freq * 2)

        # 基础振幅计算
        base_amplitude = height * 0.1  # 基础振幅约为梁高的10%

        # 综合修正
        amplitude = (base_amplitude *
                    reynolds_factor *
                    scruton_factor *
                    aspect_ratio_factor *
                    freq_factor *
                    random.uniform(0.7, 1.3))  # 随机因子

        # RMS振幅 (通常为最大振幅的70-85%)
        amplitude_rms = amplitude * random.uniform(0.7, 0.85)

        return max(5, min(80, amplitude)), max(3, min(60, amplitude_rms))

    def _classify_risk_level(self, amplitude: float) -> str:
        """基于振幅分类风险等级"""
        if amplitude < 20:
            return 'Low'
        elif amplitude < 40:
            return 'Medium'
        else:
            return 'High'

    def _generate_suppression_measures(self, amplitude: float, bridge_type: str) -> Tuple[str, str]:
        """生成抑振措施和效果"""
        if amplitude < 20:
            # 低风险，通常不需要措施
            return 'None', 'None'
        elif amplitude < 40:
            # 中风险，可能有简单措施
            if random.random() < 0.4:
                measures = ['Fairings', 'Guide Vanes', 'Stabilizers']
                measure = random.choice(measures)
                effect = f'Reduce {random.randint(15, 35)}%'
                return measure, effect
            else:
                return 'None', 'None'
        else:
            # 高风险，通常有抑振措施
            if random.random() < 0.8:
                measures = ['Fairings', 'Guide Vanes', 'Tuned Mass Damper', 'Active Control']
                measure = random.choice(measures)
                effect = f'Reduce {random.randint(25, 60)}%'
                return measure, effect
            else:
                return 'Under Design', 'TBD'

    def augment_dataset(self, original_df: pd.DataFrame, target_size: int = 200,
                       noise_level: float = 0.05) -> pd.DataFrame:
        """数据扩增主函数"""
        logger.info(f"开始数据扩增: {len(original_df)} -> {target_size}")

        # 分析原始数据
        data_analysis = self.analyze_existing_data(original_df)

        # 计算需要生成的样本数
        current_size = len(original_df)
        samples_needed = max(0, target_size - current_size)

        if samples_needed == 0:
            logger.info("数据集已达到目标大小，无需扩增")
            return original_df.copy()

        logger.info(f"需要生成 {samples_needed} 个合成样本")

        # 生成合成数据
        synthetic_samples = []
        for i in range(samples_needed):
            if i % 50 == 0:
                logger.info(f"生成进度: {i}/{samples_needed}")

            synthetic_bridge = self.generate_synthetic_bridge(data_analysis)
            synthetic_samples.append(synthetic_bridge)

        # 转换为DataFrame
        synthetic_df = pd.DataFrame(synthetic_samples)

        # 添加噪声扰动到原始数据（数据增强）
        original_augmented = self._add_noise_to_original(original_df, noise_level)

        # 合并数据
        combined_df = pd.concat([original_df, original_augmented, synthetic_df],
                               ignore_index=True)

        logger.info(f"数据扩增完成: 最终大小 {len(combined_df)}")
        return combined_df

    def _add_noise_to_original(self, df: pd.DataFrame, noise_level: float) -> pd.DataFrame:
        """向原始数据添加小量噪声生成变体"""
        augmented_samples = []

        # 为每个原始样本生成1-2个变体
        for _, row in df.iterrows():
            if random.random() < 0.6:  # 60%的原始样本会生成变体

                augmented_row = row.copy()

                # 对数值特征添加小量噪声
                numeric_cols = ['Span_m', 'Width_m', 'Height_m', 'Natural_Freq_Hz',
                              'VIV_Wind_Speed_ms', 'Critical_Wind_Speed_ms', 'Damping_Ratio']

                for col in numeric_cols:
                    if col in augmented_row and pd.notna(augmented_row[col]):
                        original_value = float(augmented_row[col])
                        noise = np.random.normal(0, noise_level * original_value)
                        augmented_row[col] = original_value + noise

                # 重新计算衍生特征
                if 'Width_m' in augmented_row and 'Height_m' in augmented_row:
                    augmented_row['Width_Height_Ratio'] = augmented_row['Width_m'] / augmented_row['Height_m']

                # 重新计算振幅（基于修改后的参数）
                new_amplitude = self._recalculate_amplitude_with_noise(augmented_row, noise_level)
                augmented_row['Max_Amplitude_mm'] = new_amplitude
                augmented_row['Amplitude_RMS_mm'] = new_amplitude * random.uniform(0.7, 0.85)
                augmented_row['Risk_Level'] = self._classify_risk_level(new_amplitude)

                # 更新ID和备注
                original_id = augmented_row.get('BridgeID', 'Unknown')
                augmented_row['BridgeID'] = f"{original_id}_AUG"
                augmented_row['Notes'] = 'Augmented variant'

                augmented_samples.append(augmented_row)

        return pd.DataFrame(augmented_samples)

    def _recalculate_amplitude_with_noise(self, row: pd.Series, noise_level: float) -> float:
        """基于噪声扰动后的参数重新计算振幅"""
        try:
            original_amplitude = float(row.get('Max_Amplitude_mm', 30))

            # 基于修改后的参数调整振幅
            damping_factor = 1 / (row.get('Damping_Ratio', 0.01) * 100 + 1)
            freq_factor = 1 / (row.get('Natural_Freq_Hz', 0.2) * 2 + 1)
            wind_factor = row.get('VIV_Wind_Speed_ms', 8) / 10

            adjustment = damping_factor * freq_factor * wind_factor
            new_amplitude = original_amplitude * adjustment * random.uniform(0.8, 1.2)

            return max(5, min(80, new_amplitude))
        except:
            return float(row.get('Max_Amplitude_mm', 30)) * random.uniform(0.9, 1.1)

def main():
    """测试数据扩增功能"""
    print("测试桥梁VIV数据扩增器")
    print("=" * 50)

    # 加载原始数据
    data_path = Path("../../bridge_dataset_fixed.csv")
    if not data_path.exists():
        print("ERROR: 原始数据文件不存在")
        return

    try:
        original_df = pd.read_csv(data_path, encoding='utf-8-sig')
        print(f"加载原始数据: {len(original_df)} 座桥梁")

        # 创建数据扩增器
        augmenter = BridgeVIVDataAugmenter(random_state=42)

        # 执行数据扩增
        augmented_df = augmenter.augment_dataset(
            original_df,
            target_size=200,  # 目标200座桥梁
            noise_level=0.03  # 3%噪声水平
        )

        print(f"扩增后数据: {len(augmented_df)} 座桥梁")

        # 保存扩增后的数据
        output_path = Path("../../bridge_dataset_augmented.csv")
        augmented_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"扩增数据已保存: {output_path}")

        # 生成统计报告
        report_path = Path("results") / "data_augmentation_report.md"
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 数据扩增报告\n\n")
            f.write(f"**原始数据**: {len(original_df)} 座桥梁\n")
            f.write(f"**扩增后数据**: {len(augmented_df)} 座桥梁\n")
            f.write(f"**新增样本**: {len(augmented_df) - len(original_df)} 个\n\n")

            f.write("## 数据分布对比\n\n")

            # 风险等级分布
            original_risk = original_df['Risk_Level'].value_counts()
            augmented_risk = augmented_df['Risk_Level'].value_counts()

            f.write("### 风险等级分布\n\n")
            f.write("| 风险等级 | 原始数据 | 扩增后数据 |\n")
            f.write("|----------|----------|------------|\n")
            for risk in ['Low', 'Medium', 'High']:
                orig_count = original_risk.get(risk, 0)
                aug_count = augmented_risk.get(risk, 0)
                f.write(f"| {risk} | {orig_count} | {aug_count} |\n")

            f.write(f"\n**生成时间**: 2024\n")

        print(f"扩增报告已保存: {report_path}")

    except Exception as e:
        print(f"ERROR: 数据扩增失败 - {e}")

if __name__ == "__main__":
    main()