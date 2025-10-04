#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from final_viv_model import FinalVIVModel
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图样式
plt.style.use('default')
sns.set_palette("husl")

def create_comprehensive_performance_charts():
    """创建物理VIV模型的综合性能图表"""

    print("Creating Physics VIV Model Performance Charts...")

    # 1. 加载数据并训练模型
    df = pd.read_csv('data/enhanced_bridge_dataset.csv')

    target_col = 'Max_Amplitude_mm'
    exclude_cols = ['BridgeID', 'BridgeName', 'Country', 'PaperSource', 'Year',
                   target_col, 'Risk_Level', 'Notes', 'Vibration_Suppression', 'Suppression_Effect']

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols]
    y = df[target_col]

    # 数据分割 (使用相同的随机种子确保一致性)
    np.random.seed(42)
    n_test = int(0.2 * len(df))
    test_indices = np.random.choice(len(df), n_test, replace=False)
    train_indices = [i for i in range(len(df)) if i not in test_indices]

    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    # 训练模型
    model = FinalVIVModel()
    model.fit(X_train, y_train)

    # 获取预测结果
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 计算性能指标
    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)

    # 创建图表
    fig = plt.figure(figsize=(20, 16))

    # 1. 预测 vs 实际值散点图 (训练集)
    ax1 = plt.subplot(2, 4, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.7, color='blue', s=60)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    plt.xlabel('Actual Max Amplitude (mm)', fontsize=12)
    plt.ylabel('Predicted Max Amplitude (mm)', fontsize=12)
    plt.title(f'Training Set\nR² = {train_metrics["R2"]:.4f}, RMSE = {train_metrics["RMSE"]:.3f}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # 添加完美预测线
    min_val = min(y_train.min(), y_train_pred.min())
    max_val = max(y_train.max(), y_train_pred.max())
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    # 2. 预测 vs 实际值散点图 (测试集)
    ax2 = plt.subplot(2, 4, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.7, color='green', s=60)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Max Amplitude (mm)', fontsize=12)
    plt.ylabel('Predicted Max Amplitude (mm)', fontsize=12)
    plt.title(f'Testing Set\nR² = {test_metrics["R2"]:.4f}, RMSE = {test_metrics["RMSE"]:.3f}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    # 3. 残差分析 (训练集)
    ax3 = plt.subplot(2, 4, 3)
    residuals_train = y_train - y_train_pred
    plt.scatter(y_train_pred, residuals_train, alpha=0.7, color='blue', s=60)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Values (mm)', fontsize=12)
    plt.ylabel('Residuals (mm)', fontsize=12)
    plt.title('Training Residuals Analysis', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # 4. 残差分析 (测试集)
    ax4 = plt.subplot(2, 4, 4)
    residuals_test = y_test - y_test_pred
    plt.scatter(y_test_pred, residuals_test, alpha=0.7, color='green', s=60)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Values (mm)', fontsize=12)
    plt.ylabel('Residuals (mm)', fontsize=12)
    plt.title('Testing Residuals Analysis', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # 5. 特征重要性
    ax5 = plt.subplot(2, 4, 5)
    feature_names = model.feature_names
    feature_coefs = model.coefficients['coef']
    feature_importance = np.abs(feature_coefs)

    # 排序特征重要性
    sorted_idx = np.argsort(feature_importance)[::-1]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_importance = feature_importance[sorted_idx]

    # 创建特征重要性柱状图
    bars = plt.barh(range(len(sorted_names)), sorted_importance, alpha=0.8)
    plt.yticks(range(len(sorted_names)), [name.replace('_', '\n') for name in sorted_names], fontsize=10)
    plt.xlabel('Absolute Coefficient Value', fontsize=12)
    plt.title('Feature Importance\n(Ridge Regression Coefficients)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')

    # 为柱状图添加颜色渐变
    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # 6. 模型性能对比
    ax6 = plt.subplot(2, 4, 6)
    models = ['Original\nRidge\n(80 samples)', 'Extended\nLinear\n(950 samples)', 'SOTA\nDeep Learning', 'Hybrid\nSOTA', 'Physics\nVIV Model']
    r2_scores = [0.938, 0.038, -0.348, -1.443, test_metrics['R2']]
    rmse_scores = [4.22, 0.223, 0.255, 0.331, test_metrics['RMSE']]

    x = np.arange(len(models))
    width = 0.35

    bars1 = plt.bar(x - width/2, r2_scores, width, label='R² Score', alpha=0.8, color='skyblue')

    # 为当前模型高亮
    bars1[-1].set_color('gold')
    bars1[-1].set_edgecolor('red')
    bars1[-1].set_linewidth(2)

    plt.xlabel('Model Type', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.title('Model Performance Comparison\n(R² Score)', fontsize=14, fontweight='bold')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()

    # 添加数值标签
    for i, (bar, score) in enumerate(zip(bars1, r2_scores)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (0.05 if height >= 0 else -0.05),
                f'{score:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
                fontweight='bold' if i == len(bars1)-1 else 'normal')

    # 7. 预测误差分布
    ax7 = plt.subplot(2, 4, 7)
    errors_train = np.abs(y_train - y_train_pred)
    errors_test = np.abs(y_test - y_test_pred)

    plt.hist(errors_train, bins=15, alpha=0.7, label=f'Training (MAE: {train_metrics["MAE"]:.3f})', color='blue')
    plt.hist(errors_test, bins=10, alpha=0.7, label=f'Testing (MAE: {test_metrics["MAE"]:.3f})', color='green')
    plt.xlabel('Absolute Error (mm)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Prediction Error Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 8. 样本预测对比 (测试集前15个样本)
    ax8 = plt.subplot(2, 4, 8)
    n_samples = min(15, len(y_test))
    sample_indices = range(n_samples)

    x_pos = np.arange(n_samples)
    width = 0.35

    bars1 = plt.bar(x_pos - width/2, y_test.iloc[:n_samples], width, label='Actual', alpha=0.8, color='red')
    bars2 = plt.bar(x_pos + width/2, y_test_pred[:n_samples], width, label='Predicted', alpha=0.8, color='blue')

    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Max Amplitude (mm)', fontsize=12)
    plt.title(f'Sample Predictions\n(First {n_samples} Test Samples)', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, [f'S{i+1}' for i in range(n_samples)], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    # 调整布局
    plt.tight_layout(pad=3.0)

    # 添加总标题
    fig.suptitle('Physics-Based VIV Model Performance Analysis\n'
                f'Dataset: {len(df)} bridges | Training: {len(X_train)} | Testing: {len(X_test)} | '
                f'Selected Features: {len(model.feature_names)} | Best α: {model.coefficients["alpha"]}',
                fontsize=16, fontweight='bold', y=0.98)

    # 保存图表
    output_path = 'results/physics_viv_model_performance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Performance charts saved to: {output_path}")

    # 显示统计信息
    print(f"\nModel Performance Summary:")
    print(f"Training - R²: {train_metrics['R2']:.4f}, RMSE: {train_metrics['RMSE']:.3f}, MAE: {train_metrics['MAE']:.3f}")
    print(f"Testing  - R²: {test_metrics['R2']:.4f}, RMSE: {test_metrics['RMSE']:.3f}, MAE: {test_metrics['MAE']:.3f}")

    print(f"\nSelected Features ({len(model.feature_names)}):")
    for feature, coef in zip(model.feature_names, model.coefficients['coef']):
        print(f"  {feature}: {coef:.4f}")

    return model, train_metrics, test_metrics

def create_detailed_analysis():
    """创建详细的模型分析图表"""

    print("\nCreating detailed analysis charts...")

    # 加载数据
    df = pd.read_csv('data/enhanced_bridge_dataset.csv')

    target_col = 'Max_Amplitude_mm'
    exclude_cols = ['BridgeID', 'BridgeName', 'Country', 'PaperSource', 'Year',
                   target_col, 'Risk_Level', 'Notes', 'Vibration_Suppression', 'Suppression_Effect']

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols]
    y = df[target_col]

    # 训练模型
    np.random.seed(42)
    n_test = int(0.2 * len(df))
    test_indices = np.random.choice(len(df), n_test, replace=False)
    train_indices = [i for i in range(len(df)) if i not in test_indices]

    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    model = FinalVIVModel()
    model.fit(X_train, y_train)

    # 创建详细分析图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. 按桥梁类型的性能分析
    ax1 = axes[0, 0]
    bridge_types = df['BridgeType'].unique()
    type_performance = {}

    for bridge_type in bridge_types:
        type_mask = df['BridgeType'] == bridge_type
        type_indices = df[type_mask].index

        # 分离训练和测试数据
        type_train_mask = type_indices.intersection(X_train.index)
        type_test_mask = type_indices.intersection(X_test.index)

        if len(type_test_mask) > 0:
            X_type_test = X.loc[type_test_mask]
            y_type_test = y.loc[type_test_mask]
            y_type_pred = model.predict(X_type_test)

            type_r2 = 1 - np.sum((y_type_test - y_type_pred)**2) / np.sum((y_type_test - y_type_test.mean())**2)
            type_performance[bridge_type] = type_r2

    types = list(type_performance.keys())
    r2_scores = list(type_performance.values())

    bars = ax1.bar(types, r2_scores, alpha=0.8, color=['skyblue', 'lightgreen', 'salmon', 'gold', 'plum'])
    ax1.set_xlabel('Bridge Type', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('Performance by Bridge Type', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, score in zip(bars, r2_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

    # 2. 残差 vs 特征关系
    ax2 = axes[0, 1]
    y_test_pred = model.predict(X_test)
    residuals = y_test - y_test_pred

    # 选择一个重要特征进行分析
    important_feature = model.feature_names[np.argmax(np.abs(model.coefficients['coef']))]
    X_test_engineered = model.create_physics_features(X_test)
    feature_values = X_test_engineered[important_feature]

    ax2.scatter(feature_values, residuals, alpha=0.7, color='red', s=60)
    ax2.axhline(y=0, color='blue', linestyle='--', linewidth=2)
    ax2.set_xlabel(f'{important_feature}', fontsize=12)
    ax2.set_ylabel('Residuals (mm)', fontsize=12)
    ax2.set_title(f'Residuals vs {important_feature.replace("_", " ")}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. 学习曲线模拟
    ax3 = axes[0, 2]
    train_sizes = np.linspace(0.3, 1.0, 8)
    train_r2_scores = []
    val_r2_scores = []

    for size in train_sizes:
        n_samples = int(size * len(X_train))
        sample_indices = np.random.choice(len(X_train), n_samples, replace=False)

        X_sample = X_train.iloc[sample_indices]
        y_sample = y_train.iloc[sample_indices]

        # 简单模拟 (实际应该重新训练)
        temp_model = FinalVIVModel()
        temp_model.fit(X_sample, y_sample)

        train_pred = temp_model.predict(X_sample)
        val_pred = temp_model.predict(X_test)

        train_r2 = 1 - np.sum((y_sample - train_pred)**2) / np.sum((y_sample - y_sample.mean())**2)
        val_r2 = 1 - np.sum((y_test - val_pred)**2) / np.sum((y_test - y_test.mean())**2)

        train_r2_scores.append(train_r2)
        val_r2_scores.append(val_r2)

    ax3.plot(train_sizes * len(X_train), train_r2_scores, 'o-', label='Training R²', color='blue', linewidth=2)
    ax3.plot(train_sizes * len(X_train), val_r2_scores, 's-', label='Validation R²', color='red', linewidth=2)
    ax3.set_xlabel('Training Set Size', fontsize=12)
    ax3.set_ylabel('R² Score', fontsize=12)
    ax3.set_title('Learning Curve', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 特征相关性热图
    ax4 = axes[1, 0]
    X_engineered = model.create_physics_features(X)
    selected_features_data = X_engineered[model.feature_names]
    corr_matrix = selected_features_data.corr()

    im = ax4.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax4.set_xticks(range(len(model.feature_names)))
    ax4.set_yticks(range(len(model.feature_names)))
    ax4.set_xticklabels([name.replace('_', '\n') for name in model.feature_names], rotation=45, ha='right')
    ax4.set_yticklabels([name.replace('_', '\n') for name in model.feature_names])
    ax4.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
    cbar.set_label('Correlation Coefficient', fontsize=10)

    # 5. 误差分析箱线图
    ax5 = axes[1, 1]
    y_pred_all = model.predict(X)
    errors_all = np.abs(y - y_pred_all)

    # 按振幅大小分组
    amplitude_bins = pd.cut(y, bins=4, labels=['Low', 'Medium', 'High', 'Very High'])
    error_groups = [errors_all[amplitude_bins == label] for label in ['Low', 'Medium', 'High', 'Very High']]

    bp = ax5.boxplot(error_groups, labels=['Low', 'Medium', 'High', 'Very High'], patch_artist=True)

    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax5.set_xlabel('Amplitude Range', fontsize=12)
    ax5.set_ylabel('Absolute Error (mm)', fontsize=12)
    ax5.set_title('Error Distribution by Amplitude Range', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # 6. 物理参数效应分析
    ax6 = axes[1, 2]

    # 分析阻尼比对预测精度的影响
    damping_values = X['Damping_Ratio']
    y_pred_all = model.predict(X)
    prediction_errors = np.abs(y - y_pred_all)

    # 创建散点图
    scatter = ax6.scatter(damping_values, prediction_errors, c=y, cmap='viridis', alpha=0.7, s=60)
    ax6.set_xlabel('Damping Ratio', fontsize=12)
    ax6.set_ylabel('Prediction Error (mm)', fontsize=12)
    ax6.set_title('Prediction Error vs Damping Ratio\n(Color: Actual Amplitude)', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax6, shrink=0.8)
    cbar.set_label('Actual Amplitude (mm)', fontsize=10)

    plt.tight_layout(pad=3.0)

    # 保存详细分析图
    output_path = 'results/physics_viv_detailed_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Detailed analysis charts saved to: {output_path}")

if __name__ == "__main__":
    # 创建综合性能图表
    model, train_metrics, test_metrics = create_comprehensive_performance_charts()

    # 创建详细分析图表
    create_detailed_analysis()

    plt.show()