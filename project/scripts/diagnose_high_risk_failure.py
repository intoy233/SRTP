#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断高风险样本预测失败的根本原因
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import r2_score, mean_squared_error

from src.final_viv_predictor import VIVPredictor


def diagnose_high_risk_prediction():
    """诊断高风险样本预测失败原因"""

    # 加载数据
    data_path = Path('data/final_bridge_dataset.csv')
    df = pd.read_csv(data_path)

    # 创建特征
    predictor = VIVPredictor()
    X, valid_idx = predictor._create_features(df)
    y = df.loc[valid_idx, 'Max_Amplitude_mm'].values

    print("="*80)
    print("高风险样本预测失败诊断报告")
    print("="*80)

    # 1. 样本分布分析
    print("\n1. 样本分布分析")
    print("-"*80)
    high_risk_mask = y > 60
    low_mid_risk_mask = ~high_risk_mask

    print(f"总样本数: {len(y)}")
    print(f"高风险样本 (>60mm): {np.sum(high_risk_mask)} ({np.sum(high_risk_mask)/len(y)*100:.1f}%)")
    print(f"低中风险样本 (<=60mm): {np.sum(low_mid_risk_mask)} ({np.sum(low_mid_risk_mask)/len(y)*100:.1f}%)")

    print(f"\n高风险振幅范围: {y[high_risk_mask].min():.2f} - {y[high_risk_mask].max():.2f} mm")
    print(f"低中风险振幅范围: {y[low_mid_risk_mask].min():.2f} - {y[low_mid_risk_mask].max():.2f} mm")
    print(f"高风险均值: {y[high_risk_mask].mean():.2f} mm")
    print(f"低中风险均值: {y[low_mid_risk_mask].mean():.2f} mm")
    print(f"高风险标准差: {y[high_risk_mask].std():.2f} mm")
    print(f"低中风险标准差: {y[low_mid_risk_mask].std():.2f} mm")

    # 2. 特征分析
    print("\n2. 特征维度分析")
    print("-"*80)
    print(f"特征维度: {X.shape[1]}")
    print(f"样本数/特征数比: {X.shape[0] / X.shape[1]:.2f}")
    print(f"高风险样本数/特征数比: {np.sum(high_risk_mask) / X.shape[1]:.2f}")

    if X.shape[0] / X.shape[1] < 5:
        print("WARNING: Sample/Feature ratio < 5, severe curse of dimensionality!")
    if np.sum(high_risk_mask) / X.shape[1] < 1:
        print("WARNING: High-risk samples < features, cannot learn effectively!")

    # 3. 特征相关性分析
    print("\n3. 特征与目标变量相关性分析")
    print("-"*80)
    correlations = []
    for i in range(X.shape[1]):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        if not np.isnan(corr):
            correlations.append(abs(corr))

    print(f"平均相关系数: {np.mean(correlations):.4f}")
    print(f"最大相关系数: {np.max(correlations):.4f}")
    print(f"相关系数 > 0.3 的特征数: {np.sum(np.array(correlations) > 0.3)}")
    print(f"相关系数 > 0.5 的特征数: {np.sum(np.array(correlations) > 0.5)}")

    # 4. 交叉验证分析(分高低风险)
    print("\n4. 5-Fold交叉验证详细分析")
    print("-"*80)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()

    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 标准化
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 训练简单模型
        model = BayesianRidge(max_iter=300, tol=1e-3)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # 整体指标
        overall_r2 = r2_score(y_test, y_pred)
        overall_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # 高风险指标
        high_risk_test_mask = y_test > 60
        if np.sum(high_risk_test_mask) > 0:
            y_test_high = y_test[high_risk_test_mask]
            y_pred_high = y_pred[high_risk_test_mask]
            high_risk_r2 = r2_score(y_test_high, y_pred_high)
            high_risk_rmse = np.sqrt(mean_squared_error(y_test_high, y_pred_high))
            high_risk_count = np.sum(high_risk_test_mask)
        else:
            high_risk_r2 = np.nan
            high_risk_rmse = np.nan
            high_risk_count = 0

        # 低中风险指标
        low_mid_test_mask = y_test <= 60
        if np.sum(low_mid_test_mask) > 0:
            y_test_low = y_test[low_mid_test_mask]
            y_pred_low = y_pred[low_mid_test_mask]
            low_mid_r2 = r2_score(y_test_low, y_pred_low)
            low_mid_rmse = np.sqrt(mean_squared_error(y_test_low, y_pred_low))
        else:
            low_mid_r2 = np.nan
            low_mid_rmse = np.nan

        fold_results.append({
            'fold': fold_idx,
            'test_size': len(y_test),
            'high_risk_count': high_risk_count,
            'overall_r2': overall_r2,
            'overall_rmse': overall_rmse,
            'high_risk_r2': high_risk_r2,
            'high_risk_rmse': high_risk_rmse,
            'low_mid_r2': low_mid_r2,
            'low_mid_rmse': low_mid_rmse
        })

        print(f"\nFold {fold_idx}:")
        print(f"  Test size: {len(y_test)} (High-risk: {high_risk_count})")
        print(f"  Overall R2: {overall_r2:.4f}, RMSE: {overall_rmse:.2f}")
        print(f"  High-risk R2: {high_risk_r2:.4f}, RMSE: {high_risk_rmse:.2f}")
        print(f"  Low-mid R2: {low_mid_r2:.4f}, RMSE: {low_mid_rmse:.2f}")

    # 5. 结论
    print("\n5. 诊断结论")
    print("-"*80)

    df_results = pd.DataFrame(fold_results)
    avg_high_risk_r2 = df_results['high_risk_r2'].mean()
    avg_low_mid_r2 = df_results['low_mid_r2'].mean()

    print(f"Average High-risk R2: {avg_high_risk_r2:.4f}")
    print(f"Average Low-mid R2: {avg_low_mid_r2:.4f}")
    print(f"R2 gap: {avg_low_mid_r2 - avg_high_risk_r2:.4f}")

    print("\nPossible causes:")
    if avg_high_risk_r2 < 0:
        print("  X High-risk R2 is negative -> Model prediction worse than mean!")
        print("  Possible reasons: 1) Too few high-risk samples(50); 2) Large variance; 3) Features have no discriminative power")

    if np.sum(high_risk_mask) / X.shape[1] < 1:
        print("  X High-risk samples(50) < Features(78) -> Severe overfitting")

    if X.shape[0] / X.shape[1] < 5:
        print("  X Sample/Feature ratio < 5 -> Curse of dimensionality")

    # 6. 保存结果
    output_dir = Path('notebooks/[20251118]改进实验')
    output_dir.mkdir(parents=True, exist_ok=True)

    df_results.to_csv(output_dir / '诊断-交叉验证详细结果.csv', index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存到: {output_dir / '诊断-交叉验证详细结果.csv'}")


if __name__ == '__main__':
    diagnose_high_risk_prediction()
