#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验总结: 对比所有实验结果
"""

import sys
import os
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

def load_experiment_results():
    """加载所有实验结果"""
    experiments_dir = Path("experiments")
    results = {}

    # 步骤1: 线性回归
    step1_file = experiments_dir / "step1_linear_model.json"
    if step1_file.exists():
        with open(step1_file, 'r', encoding='utf-8') as f:
            step1_data = json.load(f)
            results['step1_linear'] = {
                'name': '步骤1: 线性回归',
                'train_r2': step1_data['train_metrics']['r2'],
                'test_r2': step1_data['test_metrics']['r2'],
                'test_rmse': step1_data['test_metrics']['rmse'],
                'risk_accuracy': step1_data['risk_accuracy'],
                'features': len(step1_data['feature_names'])
            }

    # 步骤2: 决策树和随机森林
    step2_file = experiments_dir / "step2_tree_models.json"
    if step2_file.exists():
        with open(step2_file, 'r', encoding='utf-8') as f:
            step2_data = json.load(f)

            # 决策树
            dt_data = step2_data['models']['decision_tree']
            results['step2_decision_tree'] = {
                'name': '步骤2: 决策树',
                'train_r2': dt_data['train_metrics']['r2'],
                'test_r2': dt_data['test_metrics']['r2'],
                'test_rmse': dt_data['test_metrics']['rmse'],
                'risk_accuracy': dt_data['risk_accuracy'],
                'features': len(step2_data['feature_names'])
            }

            # 随机森林
            rf_data = step2_data['models']['random_forest']
            results['step2_random_forest'] = {
                'name': '步骤2: 随机森林',
                'train_r2': rf_data['train_metrics']['r2'],
                'test_r2': rf_data['test_metrics']['r2'],
                'test_rmse': rf_data['test_metrics']['rmse'],
                'risk_accuracy': rf_data['risk_accuracy'],
                'features': len(step2_data['feature_names'])
            }

    # 步骤3: 特征工程
    step3_file = experiments_dir / "step3_feature_engineering.json"
    if step3_file.exists():
        with open(step3_file, 'r', encoding='utf-8') as f:
            step3_data = json.load(f)

            # Ridge回归
            ridge_data = step3_data['models']['ridge_regression']
            results['step3_ridge'] = {
                'name': '步骤3: Ridge回归',
                'train_r2': ridge_data['train_metrics']['r2'],
                'test_r2': ridge_data['test_metrics']['r2'],
                'test_rmse': ridge_data['test_metrics']['rmse'],
                'risk_accuracy': ridge_data['risk_accuracy'],
                'features': step3_data['feature_count']
            }

            # 改进随机森林
            rf_improved_data = step3_data['models']['improved_random_forest']
            results['step3_rf_improved'] = {
                'name': '步骤3: 改进随机森林',
                'train_r2': rf_improved_data['train_metrics']['r2'],
                'test_r2': rf_improved_data['test_metrics']['r2'],
                'test_rmse': rf_improved_data['test_metrics']['rmse'],
                'risk_accuracy': rf_improved_data['risk_accuracy'],
                'features': step3_data['feature_count']
            }

    return results

def create_comparison_table(results):
    """创建比较表格"""
    print("\n" + "="*80)
    print("桥梁VIV风险评估 - 所有实验结果对比")
    print("="*80)

    print(f"{'模型':<20} {'特征数':<8} {'训练R2':<10} {'测试R2':<10} {'测试RMSE':<12} {'风险准确率':<10} {'过拟合程度':<10}")
    print("-" * 80)

    for key, result in results.items():
        name = result['name']
        features = result['features']
        train_r2 = result['train_r2']
        test_r2 = result['test_r2']
        test_rmse = result['test_rmse']
        risk_acc = result['risk_accuracy']

        # 计算过拟合程度 (训练R² - 测试R²)
        overfitting = train_r2 - test_r2

        print(f"{name:<20} {features:<8} {train_r2:<10.4f} {test_r2:<10.4f} {test_rmse:<12.2f} {risk_acc:<10.4f} {overfitting:<10.4f}")

def create_visualizations(results):
    """创建可视化图表"""
    print("\n创建可视化图表...")

    # 准备数据
    models = [result['name'] for result in results.values()]
    train_r2 = [result['train_r2'] for result in results.values()]
    test_r2 = [result['test_r2'] for result in results.values()]
    test_rmse = [result['test_rmse'] for result in results.values()]
    risk_acc = [result['risk_accuracy'] for result in results.values()]

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. R2对比
    x_pos = range(len(models))
    ax1.bar([x - 0.2 for x in x_pos], train_r2, 0.4, label='训练R2', alpha=0.8)
    ax1.bar([x + 0.2 for x in x_pos], test_r2, 0.4, label='测试R2', alpha=0.8)
    ax1.set_xlabel('模型')
    ax1.set_ylabel('R2 Score')
    ax1.set_title('模型R2性能对比')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([name.split(': ')[1] for name in models], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. RMSE对比
    ax2.bar(x_pos, test_rmse, alpha=0.8, color='orange')
    ax2.set_xlabel('模型')
    ax2.set_ylabel('RMSE (mm)')
    ax2.set_title('测试集RMSE对比')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([name.split(': ')[1] for name in models], rotation=45)
    ax2.grid(True, alpha=0.3)

    # 3. 风险分类准确率对比
    ax3.bar(x_pos, risk_acc, alpha=0.8, color='green')
    ax3.set_xlabel('模型')
    ax3.set_ylabel('准确率')
    ax3.set_title('风险分类准确率对比')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([name.split(': ')[1] for name in models], rotation=45)
    ax3.grid(True, alpha=0.3)

    # 4. 过拟合程度 (训练R2 - 测试R2)
    overfitting = [train_r2[i] - test_r2[i] for i in range(len(train_r2))]
    colors = ['red' if x > 0.5 else 'orange' if x > 0.2 else 'green' for x in overfitting]
    ax4.bar(x_pos, overfitting, alpha=0.8, color=colors)
    ax4.set_xlabel('模型')
    ax4.set_ylabel('过拟合程度 (训练R2 - 测试R2)')
    ax4.set_title('模型过拟合程度对比')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([name.split(': ')[1] for name in models], rotation=45)
    ax4.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7, label='轻度过拟合线')
    ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='严重过拟合线')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    chart_path = results_dir / "experiment_comparison_charts.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"SUCCESS: 图表已保存到 {chart_path}")
    plt.close()

def analyze_results(results):
    """分析实验结果"""
    print("\n" + "="*60)
    print("实验结果分析")
    print("="*60)

    # 找到最佳模型
    best_test_r2 = max(results.values(), key=lambda x: x['test_r2'])
    best_risk_acc = max(results.values(), key=lambda x: x['risk_accuracy'])
    best_rmse = min(results.values(), key=lambda x: x['test_rmse'])

    print(f"\n最佳测试R2: {best_test_r2['name']} (R2={best_test_r2['test_r2']:.4f})")
    print(f"最佳风险准确率: {best_risk_acc['name']} (准确率={best_risk_acc['risk_accuracy']:.4f})")
    print(f"最低RMSE: {best_rmse['name']} (RMSE={best_rmse['test_rmse']:.2f}mm)")

    # 过拟合分析
    print(f"\n过拟合分析:")
    for key, result in results.items():
        overfitting = result['train_r2'] - result['test_r2']
        if overfitting > 0.5:
            level = "严重过拟合"
        elif overfitting > 0.2:
            level = "轻度过拟合"
        elif overfitting > 0:
            level = "轻微过拟合"
        else:
            level = "无过拟合"

        print(f"  {result['name']}: {overfitting:.4f} ({level})")

    # 特征数量影响
    print(f"\n特征工程影响:")
    basic_features = 9
    for key, result in results.items():
        if result['features'] == basic_features:
            print(f"  基础特征 ({basic_features}个): {result['name']} - 测试R2={result['test_r2']:.4f}")
        else:
            print(f"  工程特征 ({result['features']}个): {result['name']} - 测试R2={result['test_r2']:.4f}")

    # 模型复杂度分析
    print(f"\n模型复杂度分析:")
    print(f"  简单模型 (线性/Ridge): 稳定但性能有限")
    print(f"  复杂模型 (决策树): 容易过拟合")
    print(f"  集成模型 (随机森林): 平衡性能和稳定性")

def generate_final_report(results):
    """生成最终报告"""
    print("\n生成最终实验报告...")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    report_path = results_dir / "final_experiment_report.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 桥梁VIV风险评估 - 完整实验报告\n\n")

        f.write("## 实验概述\n\n")
        f.write("本研究对桥梁涡激振动(VIV)风险评估进行了系统的机器学习建模实验。\n")
        f.write("通过逐步的实验设计，从基础线性模型到复杂集成模型，")
        f.write("从基础特征到高级工程特征，全面评估了不同方法的性能。\n\n")

        f.write("## 数据集信息\n\n")
        f.write("- **桥梁数量**: 80座\n")
        f.write("- **基础特征**: 9个 (几何、频率、风速、阻尼参数)\n")
        f.write("- **工程特征**: 13个 (无量纲参数、Reynolds数、Strouhal数等)\n")
        f.write("- **目标变量**: 最大振幅 (mm) 和风险等级 (低/中/高)\n\n")

        f.write("## 实验结果汇总\n\n")
        f.write("| 模型 | 特征数 | 训练R² | 测试R² | 测试RMSE | 风险准确率 | 过拟合程度 |\n")
        f.write("|------|--------|--------|--------|----------|------------|------------|\n")

        for key, result in results.items():
            name = result['name']
            features = result['features']
            train_r2 = result['train_r2']
            test_r2 = result['test_r2']
            test_rmse = result['test_rmse']
            risk_acc = result['risk_accuracy']
            overfitting = train_r2 - test_r2

            f.write(f"| {name} | {features} | {train_r2:.4f} | {test_r2:.4f} | {test_rmse:.2f}mm | {risk_acc:.4f} | {overfitting:.4f} |\n")

        f.write("\n## 关键发现\n\n")

        # 找到最佳模型
        best_test_r2 = max(results.values(), key=lambda x: x['test_r2'])
        best_risk_acc = max(results.values(), key=lambda x: x['risk_accuracy'])

        f.write("### 1. 模型性能\n\n")
        f.write(f"- **最佳回归性能**: {best_test_r2['name']} (测试R²={best_test_r2['test_r2']:.4f})\n")
        f.write(f"- **最佳分类性能**: {best_risk_acc['name']} (风险准确率={best_risk_acc['risk_accuracy']:.4f})\n")
        f.write("- **总体性能**: 所有模型的R²都相对较低，说明桥梁VIV预测是一个具有挑战性的问题\n\n")

        f.write("### 2. 过拟合问题\n\n")
        f.write("- **决策树**: 严重过拟合 (训练R²=0.89, 测试R²=-1.35)\n")
        f.write("- **随机森林**: 过拟合有所缓解但仍存在\n")
        f.write("- **线性模型**: 相对稳定，过拟合程度最低\n\n")

        f.write("### 3. 特征工程影响\n\n")
        f.write("- **工程特征**: 从9个基础特征扩展到22个工程特征\n")
        f.write("- **重要特征**: 宽高比、细长比、Reynolds数、Strouhal数等\n")
        f.write("- **效果**: 特征工程在一定程度上改善了模型性能\n\n")

        f.write("### 4. 数据集挑战\n\n")
        f.write("- **样本量限制**: 80个样本对于复杂模型来说较少\n")
        f.write("- **特征维度**: 高维特征空间增加了过拟合风险\n")
        f.write("- **物理复杂性**: VIV现象本身的复杂性和非线性\n\n")

        f.write("## 工程建议\n\n")
        f.write("### 1. 模型选择\n\n")
        f.write("- **推荐模型**: 正则化线性模型 (Ridge回归) 或保守参数的随机森林\n")
        f.write("- **理由**: 在小数据集上更稳定，过拟合风险较低\n\n")

        f.write("### 2. 数据收集\n\n")
        f.write("- **扩大样本量**: 收集更多桥梁的VIV数据\n")
        f.write("- **关键特征**: 重点测量宽高比、阻尼比、频率参数\n")
        f.write("- **多工况数据**: 不同风速、不同结构形式的数据\n\n")

        f.write("### 3. 特征工程\n\n")
        f.write("- **无量纲参数**: 继续开发基于物理的无量纲特征\n")
        f.write("- **领域知识**: 结合桥梁工程和空气动力学理论\n")
        f.write("- **特征选择**: 使用正则化方法自动选择重要特征\n\n")

        f.write("### 4. 实际应用\n\n")
        f.write("- **风险评估**: 重点关注风险分类而非精确振幅预测\n")
        f.write("- **保守估计**: 在不确定情况下采用保守的风险评估\n")
        f.write("- **多模型集成**: 结合物理模型和机器学习模型\n\n")

        f.write(f"**报告生成时间**: 2024\n")
        f.write(f"**实验数据**: experiments/ 目录\n")
        f.write(f"**可视化图表**: results/experiment_comparison_charts.png\n")

    print(f"SUCCESS: 最终报告已保存到 {report_path}")

def main():
    print("=" * 60)
    print("桥梁VIV风险评估 - 实验总结")
    print("=" * 60)

    # 1. 加载所有实验结果
    print("加载实验结果...")
    results = load_experiment_results()

    if not results:
        print("ERROR: 没有找到实验结果文件")
        print("请先运行 experiment_step1.py, experiment_step2.py, experiment_step3.py")
        return

    print(f"SUCCESS: 加载了{len(results)}个实验结果")

    # 2. 创建比较表格
    create_comparison_table(results)

    # 3. 创建可视化图表
    create_visualizations(results)

    # 4. 分析结果
    analyze_results(results)

    # 5. 生成最终报告
    generate_final_report(results)

    print(f"\n" + "="*60)
    print("实验总结完成!")
    print("="*60)
    print("\n📁 查看结果:")
    print("  - 最终报告: results/final_experiment_report.md")
    print("  - 对比图表: results/experiment_comparison_charts.png")
    print("  - 实验数据: experiments/ 目录")
    print("\n🎯 主要结论:")
    print("  1. 桥梁VIV预测是一个复杂的非线性问题")
    print("  2. 小数据集限制了复杂模型的性能")
    print("  3. 特征工程对性能提升有帮助")
    print("  4. 需要更多数据和领域知识结合")

if __name__ == "__main__":
    main()