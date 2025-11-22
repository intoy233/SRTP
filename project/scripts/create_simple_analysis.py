#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from final_viv_model import FinalVIVModel

plt.rcParams['font.size'] = 10
plt.style.use('default')

def create_simple_analysis():
    """创建简化版详细分析"""

    print("Creating detailed analysis...")

    # 加载数据并训练模型
    df = pd.read_csv('data/enhanced_bridge_dataset.csv')

    target_col = 'Max_Amplitude_mm'
    exclude_cols = ['BridgeID', 'BridgeName', 'Country', 'PaperSource', 'Year',
                   target_col, 'Risk_Level', 'Notes', 'Vibration_Suppression', 'Suppression_Effect']

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols]
    y = df[target_col]

    # 数据分割
    np.random.seed(42)
    n_test = int(0.2 * len(df))
    test_indices = np.random.choice(len(df), n_test, replace=False)
    train_indices = [i for i in range(len(df)) if i not in test_indices]

    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    # 训练模型
    model = FinalVIVModel()
    model.fit(X_train, y_train)

    # 创建分析图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. 按桥梁类型的性能分析
    ax1 = axes[0, 0]
    bridge_types = ['Suspension', 'Cable-Stayed', 'Arch']  # 主要类型
    type_performance = []

    for bridge_type in bridge_types:
        type_mask = df['BridgeType'] == bridge_type
        type_data = df[type_mask]

        if len(type_data) > 0:
            # 使用部分数据进行预测
            X_type = X[type_mask]
            y_type = y[type_mask]

            if len(X_type) > 2:  # 确保有足够数据
                y_type_pred = model.predict(X_type)
                type_r2 = 1 - np.sum((y_type - y_type_pred)**2) / np.sum((y_type - y_type.mean())**2)
                type_performance.append(max(type_r2, 0))  # 避免负值
            else:
                type_performance.append(0)

    bars = ax1.bar(bridge_types, type_performance, alpha=0.8, color=['skyblue', 'lightgreen', 'salmon'])
    ax1.set_xlabel('Bridge Type')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Performance by Bridge Type', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 添加数值标签
    for bar, score in zip(bars, type_performance):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

    # 2. 物理特征效应分析
    ax2 = axes[0, 1]
    # 分析约化风速对振幅的影响
    X_engineered = model.create_physics_features(X)
    if 'Reduced_Velocity' in X_engineered.columns:
        reduced_vel = X_engineered['Reduced_Velocity']
        scatter = ax2.scatter(reduced_vel, y, c=X['Damping_Ratio'], cmap='viridis', alpha=0.7, s=60)
        ax2.set_xlabel('Reduced Velocity')
        ax2.set_ylabel('Max Amplitude (mm)')
        ax2.set_title('Amplitude vs Reduced Velocity\n(Color: Damping Ratio)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Damping Ratio')

    # 3. 训练过程模拟
    ax3 = axes[0, 2]
    # 模拟不同正则化参数的效果
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    train_r2s = []
    test_r2s = []

    for alpha in alphas:
        # 临时模型用于测试不同alpha
        temp_model = FinalVIVModel()
        X_eng = temp_model.create_physics_features(X_train)
        features = temp_model.select_features(X_eng, y_train)
        X_sel = X_eng[features]
        X_scaled = temp_model.standardize(X_sel, fit=True)

        # 训练
        coefs = temp_model.ridge_regression(X_scaled, y_train.values, alpha)
        temp_model.coefficients = coefs
        temp_model.feature_names = features
        temp_model.is_fitted = True

        # 评估
        train_metrics = temp_model.evaluate(X_train, y_train)
        test_metrics = temp_model.evaluate(X_test, y_test)

        train_r2s.append(train_metrics['R2'])
        test_r2s.append(test_metrics['R2'])

    ax3.semilogx(alphas, train_r2s, 'o-', label='Training R²', linewidth=2)
    ax3.semilogx(alphas, test_r2s, 's-', label='Testing R²', linewidth=2)
    ax3.axvline(x=0.1, color='red', linestyle='--', alpha=0.7, label='Best α')
    ax3.set_xlabel('Regularization Parameter (α)')
    ax3.set_ylabel('R² Score')
    ax3.set_title('Regularization Effect', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 误差分析
    ax4 = axes[1, 0]
    y_pred = model.predict(X)
    errors = np.abs(y - y_pred)

    # 按振幅范围分组
    low_amp = errors[y <= 30]
    med_amp = errors[(y > 30) & (y <= 50)]
    high_amp = errors[y > 50]

    bp = ax4.boxplot([low_amp, med_amp, high_amp],
                     labels=['Low\n(≤30mm)', 'Medium\n(30-50mm)', 'High\n(>50mm)'],
                     patch_artist=True)

    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax4.set_xlabel('Amplitude Range')
    ax4.set_ylabel('Absolute Error (mm)')
    ax4.set_title('Error Distribution by Amplitude', fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 5. 特征相关性
    ax5 = axes[1, 1]
    selected_data = X_engineered[model.feature_names]
    corr_matrix = selected_data.corr()

    im = ax5.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax5.set_xticks(range(len(model.feature_names)))
    ax5.set_yticks(range(len(model.feature_names)))

    # 简化特征名称
    short_names = [name.replace('_', '\n')[:10] for name in model.feature_names]
    ax5.set_xticklabels(short_names, rotation=45, ha='right')
    ax5.set_yticklabels(short_names)
    ax5.set_title('Feature Correlation Matrix', fontweight='bold')

    plt.colorbar(im, ax=ax5, shrink=0.8, label='Correlation')

    # 6. 预测准确性分布
    ax6 = axes[1, 2]
    y_pred_test = model.predict(X_test)
    relative_errors = np.abs((y_test - y_pred_test) / y_test) * 100

    # 创建准确性分布
    bins = [0, 5, 10, 15, 20, 100]
    labels = ['<5%', '5-10%', '10-15%', '15-20%', '>20%']
    counts = []

    for i in range(len(bins)-1):
        count = np.sum((relative_errors >= bins[i]) & (relative_errors < bins[i+1]))
        counts.append(count)

    colors = ['green', 'lightgreen', 'yellow', 'orange', 'red']
    bars = ax6.bar(labels, counts, color=colors, alpha=0.8)

    ax6.set_xlabel('Relative Error Range')
    ax6.set_ylabel('Number of Samples')
    ax6.set_title('Prediction Accuracy Distribution\n(Test Set)', fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')

    # 添加百分比标签
    total_samples = len(y_test)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        percentage = (count / total_samples) * 100
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout(pad=3.0)

    # 保存图表
    output_path = 'results/physics_viv_detailed_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Detailed analysis saved to: {output_path}")

    # 生成分析报告
    print(f"\nDetailed Analysis Summary:")
    print(f"Bridge Type Performance:")
    for bridge_type, perf in zip(bridge_types, type_performance):
        print(f"  {bridge_type}: R² = {perf:.3f}")

    print(f"\nAccuracy Distribution (Test Set):")
    for label, count, percentage in zip(labels, counts, [c/total_samples*100 for c in counts]):
        print(f"  {label} error: {count} samples ({percentage:.1f}%)")

    mean_rel_error = np.mean(relative_errors)
    print(f"\nMean Relative Error: {mean_rel_error:.2f}%")

if __name__ == "__main__":
    create_simple_analysis()
    plt.show()