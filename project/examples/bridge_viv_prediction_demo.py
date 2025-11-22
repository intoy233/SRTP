#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
桥梁VIV振幅预测 - 工程应用示例
Bridge VIV Amplitude Prediction - Engineering Application Demo

演示如何使用训练好的Stacking模型进行桥梁VIV风险预测
"""

import sys
sys.path.append('../src')

from final_viv_predictor import VIVPredictor
import pandas as pd


def example_1_single_bridge():
    """示例1: 单座桥梁预测"""
    print("="*80)
    print("示例1: 单座桥梁VIV振幅预测")
    print("="*80)

    # 加载已训练模型
    predictor = VIVPredictor()

    # 如果模型文件不存在,需要先训练
    try:
        predictor.load_model('../models/stacking_viv_predictor.pkl')
        print("✓ 已加载训练好的模型")
    except FileNotFoundError:
        print("模型文件不存在,开始训练...")
        predictor.train('../data/final_bridge_dataset.csv', k=5)
        predictor.save_model('../models/stacking_viv_predictor.pkl')

    # 输入桥梁参数 (以西候门大桥为例)
    bridge = {
        'Span_m': 1650.0,          # 主跨长度 (m)
        'Width_m': 36.0,           # 桥面宽度 (m)
        'Height_m': 3.5,           # 主梁高度 (m)
        'Damping_Ratio': 0.0028,   # 阻尼比
        'Natural_Freq_Hz': 0.112,  # 自振频率 (Hz)
        'Critical_Wind_Speed_ms': 11.5  # 临界风速 (m/s)
    }

    print("\n【输入参数】")
    print(f"  桥梁名称: 西侯门大桥 (示例)")
    print(f"  主跨: {bridge['Span_m']} m")
    print(f"  宽度: {bridge['Width_m']} m")
    print(f"  高度: {bridge['Height_m']} m")
    print(f"  阻尼比: {bridge['Damping_Ratio']:.4f}")
    print(f"  自振频率: {bridge['Natural_Freq_Hz']:.3f} Hz")
    print(f"  临界风速: {bridge['Critical_Wind_Speed_ms']:.1f} m/s")

    # 预测
    amplitude, uncertainty = predictor.predict(bridge)

    print("\n【预测结果】")
    print(f"  VIV振幅: {amplitude:.2f} mm")
    print(f"  不确定性: ±{uncertainty:.2f} mm")
    print(f"  95%置信区间: [{amplitude-1.96*uncertainty:.2f}, {amplitude+1.96*uncertainty:.2f}] mm")

    # 风险评估
    risk_level, recommendation = predictor.risk_assessment(amplitude, uncertainty)

    print("\n【风险评估】")
    print(f"  风险等级: {risk_level}")
    print(f"  工程建议: {recommendation}")

    print("\n" + "="*80)


def example_2_batch_prediction():
    """示例2: 批量桥梁预测"""
    print("\n" + "="*80)
    print("示例2: 批量桥梁VIV风险筛查")
    print("="*80)

    # 加载模型
    predictor = VIVPredictor()
    try:
        predictor.load_model('../models/stacking_viv_predictor.pkl')
    except FileNotFoundError:
        predictor.train('../data/final_bridge_dataset.csv', k=5)
        predictor.save_model('../models/stacking_viv_predictor.pkl')

    # 待评估的多座桥梁
    bridges = [
        {'名称': '大跨径悬索桥A', 'Span_m': 1800, 'Width_m': 38, 'Height_m': 3.2,
         'Damping_Ratio': 0.0025, 'Natural_Freq_Hz': 0.095, 'Critical_Wind_Speed_ms': 10.5},

        {'名称': '中等跨径斜拉桥B', 'Span_m': 800, 'Width_m': 32, 'Height_m': 2.8,
         'Damping_Ratio': 0.0035, 'Natural_Freq_Hz': 0.185, 'Critical_Wind_Speed_ms': 14.2},

        {'名称': '小跨径钢箱梁C', 'Span_m': 350, 'Width_m': 28, 'Height_m': 2.5,
         'Damping_Ratio': 0.0042, 'Natural_Freq_Hz': 0.285, 'Critical_Wind_Speed_ms': 18.5},
    ]

    print("\n【批量预测结果】\n")
    print(f"{'桥梁名称':<15} {'预测振幅(mm)':<15} {'不确定性':<12} {'风险等级':<10} {'建议'}")
    print("-"*80)

    for bridge_data in bridges:
        name = bridge_data.pop('名称')
        amplitude, uncertainty = predictor.predict(bridge_data)
        risk_level, recommendation = predictor.risk_assessment(amplitude, uncertainty)

        # 简化建议显示
        if '强烈' in recommendation:
            advice = '风洞实验'
        elif '减振' in recommendation:
            advice = '减振措施'
        else:
            advice = '初步安全'

        print(f"{name:<15} {amplitude:>6.2f} ± {uncertainty:>5.2f}    ±{uncertainty:<7.2f} {risk_level:<10} {advice}")

    print("\n" + "="*80)


def example_3_design_optimization():
    """示例3: 设计方案优化 - 对比不同阻尼比"""
    print("\n" + "="*80)
    print("示例3: 设计优化 - 阻尼比对VIV振幅的影响")
    print("="*80)

    # 加载模型
    predictor = VIVPredictor()
    try:
        predictor.load_model('../models/stacking_viv_predictor.pkl')
    except FileNotFoundError:
        predictor.train('../data/final_bridge_dataset.csv', k=5)
        predictor.save_model('../models/stacking_viv_predictor.pkl')

    # 基础桥梁参数
    base_bridge = {
        'Span_m': 1200,
        'Width_m': 35,
        'Height_m': 3.0,
        'Natural_Freq_Hz': 0.135,
        'Critical_Wind_Speed_ms': 12.8
    }

    # 测试不同阻尼比方案
    damping_ratios = [0.0020, 0.0030, 0.0040, 0.0050, 0.0060]

    print("\n【参数】")
    print(f"  主跨: {base_bridge['Span_m']} m")
    print(f"  宽度: {base_bridge['Width_m']} m")
    print(f"  临界风速: {base_bridge['Critical_Wind_Speed_ms']} m/s")

    print("\n【不同阻尼比方案对比】\n")
    print(f"{'阻尼比':<10} {'预测振幅(mm)':<15} {'降幅(%)':<12} {'风险等级':<10}")
    print("-"*60)

    baseline_amplitude = None
    for damping in damping_ratios:
        bridge = base_bridge.copy()
        bridge['Damping_Ratio'] = damping

        amplitude, uncertainty = predictor.predict(bridge)
        risk_level, _ = predictor.risk_assessment(amplitude, uncertainty)

        if baseline_amplitude is None:
            baseline_amplitude = amplitude
            reduction = 0.0
        else:
            reduction = (baseline_amplitude - amplitude) / baseline_amplitude * 100

        print(f"{damping:<10.4f} {amplitude:>6.2f} ± {uncertainty:>5.2f}    {reduction:>6.1f}%      {risk_level:<10}")

    print("\n【结论】增加阻尼比可有效降低VIV振幅")
    print("  建议: 如预测振幅过高,考虑增设调谐质量阻尼器(TMD)或粘滞阻尼器")

    print("\n" + "="*80)


def example_4_uncertainty_analysis():
    """示例4: 不确定性分析"""
    print("\n" + "="*80)
    print("示例4: 不确定性量化与风险管理")
    print("="*80)

    # 加载模型
    predictor = VIVPredictor()
    try:
        predictor.load_model('../models/stacking_viv_predictor.pkl')
    except FileNotFoundError:
        predictor.train('../data/final_bridge_dataset.csv', k=5)
        predictor.save_model('../models/stacking_viv_predictor.pkl')

    # 高风险案例
    high_risk_bridge = {
        'Span_m': 1950,
        'Width_m': 40,
        'Height_m': 3.8,
        'Damping_Ratio': 0.0022,
        'Natural_Freq_Hz': 0.089,
        'Critical_Wind_Speed_ms': 9.8
    }

    amplitude, uncertainty = predictor.predict(high_risk_bridge)

    print("\n【桥梁参数】")
    print(f"  主跨: {high_risk_bridge['Span_m']} m (超大跨)")
    print(f"  阻尼比: {high_risk_bridge['Damping_Ratio']:.4f} (偏低)")

    print("\n【预测结果】")
    print(f"  点估计: {amplitude:.2f} mm")
    print(f"  不确定性: ±{uncertainty:.2f} mm")

    print("\n【置信区间分析】")
    print(f"  68%置信区间: [{amplitude-uncertainty:.2f}, {amplitude+uncertainty:.2f}] mm")
    print(f"  95%置信区间: [{amplitude-1.96*uncertainty:.2f}, {amplitude+1.96*uncertainty:.2f}] mm")

    # 保守估计
    conservative_estimate = amplitude + 1.96 * uncertainty
    print(f"\n  保守估计(上界95%): {conservative_estimate:.2f} mm")

    # 风险决策
    print("\n【风险管理决策】")
    if conservative_estimate > 70:
        print("  ⚠ 保守估计超过70mm!")
        print("  决策: 必须进行风洞实验验证,禁止仅依赖模型预测")
    elif conservative_estimate > 50:
        print("  ⚠ 保守估计超过50mm")
        print("  决策: 强烈建议风洞实验,或设计预留减振装置接口")
    else:
        print("  ✓ 保守估计在可接受范围内")
        print("  决策: 初步安全,但建议CFD数值模拟验证")

    print("\n" + "="*80)


def main():
    """主函数 - 运行所有示例"""
    print("\n" + "="*80)
    print("桥梁VIV振幅预测 - 工程应用演示")
    print("基于Stacking集成模型 (R²=0.6290, RMSE=13.03mm)")
    print("="*80)

    try:
        # 示例1: 单座桥梁预测
        example_1_single_bridge()

        # 示例2: 批量预测
        example_2_batch_prediction()

        # 示例3: 设计优化
        example_3_design_optimization()

        # 示例4: 不确定性分析
        example_4_uncertainty_analysis()

        print("\n" + "="*80)
        print("所有示例运行完成!")
        print("="*80)
        print("\n【使用建议】")
        print("1. 预测振幅 > 50mm 或 上界 > 70mm → 必须风洞实验")
        print("2. 预测振幅 30-50mm → 考虑减振措施(TMD/粘滞阻尼器)")
        print("3. 预测振幅 < 30mm → 初步安全,结合工程经验判断")
        print("4. 模型仅供初步筛查,不能替代实验验证!")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n错误: {e}")
        print("请确保:")
        print("  1. 数据文件存在: ../data/final_bridge_dataset.csv")
        print("  2. models目录已创建: ../models/")
        print("  3. 已安装依赖: numpy, pandas, scikit-learn")


if __name__ == '__main__':
    main()
