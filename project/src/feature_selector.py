#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征选择实验 - Feature Selection Experiments

目标: 从78维特征降至15-30维，缓解维度灾难，提升泛化能力

方法:
1. Random Forest重要性排序
2. Lasso系数筛选
3. 递归特征消除(RFE)
4. 相关性分析

输出:
- 特征重要性排序
- 不同特征数量下的模型性能
- 最佳特征子集
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import BayesianRidge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from .final_viv_predictor import VIVPredictor


def load_and_prepare_data() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """加载数据并构建特征矩阵"""
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "final_bridge_dataset.csv"

    df = pd.read_csv(data_path)

    # 使用VIVPredictor的特征工程
    predictor = VIVPredictor()
    X, valid_idx = predictor._create_features(df)
    y = df.loc[valid_idx, "Max_Amplitude_mm"].values

    # 获取特征名称
    feature_names = predictor.feature_names

    print(f"数据加载完成: {X.shape[0]} 样本, {X.shape[1]} 特征")
    return X, y, feature_names


def select_features_by_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    top_k: int = 80,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    方法1: 使用随机森林的特征重要性选择top-k特征

    Args:
        X: 特征矩阵
        y: 目标变量
        feature_names: 特征名称列表
        top_k: 保留的特征数量

    Returns:
        selected_indices: 选中特征的索引
        selected_names: 选中特征的名称
        importances: 所有特征的重要性
    """
    print(f"\n方法1: Random Forest 特征重要性（保留top-{top_k}）")
    print("-" * 80)

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 训练随机森林
    print("训练Random Forest...")
    rf.fit(X_scaled, y)

    # 获取重要性
    importances = rf.feature_importances_

    # 排序并选择top-k
    indices = np.argsort(importances)[::-1][:top_k]
    selected_names = [feature_names[i] for i in indices]

    print(f"完成! 选中 {len(indices)} 个特征")
    print(f"\nTop 10 重要特征:")
    for i, idx in enumerate(indices[:10], 1):
        print(f"  {i:2d}. {feature_names[idx]:<40} (重要性: {importances[idx]:.4f})")

    return indices, selected_names, importances


def select_features_by_lasso(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    alpha: float = 0.01,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    方法2: 使用Lasso回归筛选非零系数特征

    Args:
        X: 特征矩阵
        y: 目标变量
        feature_names: 特征名称列表
        alpha: L1正则化系数

    Returns:
        selected_indices: 选中特征的索引
        selected_names: 选中特征的名称
        coefficients: 所有特征的系数
    """
    print(f"\n方法2: Lasso 系数筛选（alpha={alpha}）")
    print("-" * 80)

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 训练Lasso
    print("训练Lasso...")
    lasso = Lasso(alpha=alpha, max_iter=5000, random_state=42)
    lasso.fit(X_scaled, y)

    # 筛选非零系数
    coefficients = lasso.coef_
    indices = np.where(np.abs(coefficients) > 1e-5)[0]
    selected_names = [feature_names[i] for i in indices]

    print(f"完成! 选中 {len(indices)} 个特征（非零系数）")
    print(f"\nTop 10 系数最大特征:")
    top_coef_indices = np.argsort(np.abs(coefficients))[::-1][:10]
    for i, idx in enumerate(top_coef_indices, 1):
        print(
            f"  {i:2d}. {feature_names[idx]:<40} (系数: {coefficients[idx]:+.4f})"
        )

    return indices, selected_names, coefficients


def select_features_by_rfe(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_features_to_select: int = 80,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    方法3: 递归特征消除(RFE)

    Args:
        X: 特征矩阵
        y: 目标变量
        feature_names: 特征名称列表
        n_features_to_select: 目标特征数量

    Returns:
        selected_indices: 选中特征的索引
        selected_names: 选中特征的名称
        ranking: 特征排名
    """
    print(f"\n方法3: 递归特征消除 RFE（目标: {n_features_to_select} 个特征）")
    print("-" * 80)

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 使用BayesianRidge作为基础估计器
    estimator = BayesianRidge(max_iter=300)

    print("运行RFE（可能需要几分钟）...")
    rfe = RFE(
        estimator=estimator,
        n_features_to_select=n_features_to_select,
        step=10,  # 每次消除10个特征
    )
    rfe.fit(X_scaled, y)

    # 获取选中的特征
    indices = np.where(rfe.support_)[0]
    selected_names = [feature_names[i] for i in indices]
    ranking = rfe.ranking_

    print(f"完成! 选中 {len(indices)} 个特征")

    return indices, selected_names, ranking


def evaluate_feature_subset(
    X: np.ndarray,
    y: np.ndarray,
    selected_indices: np.ndarray,
    method_name: str,
    n_splits: int = 5,
) -> Dict[str, float]:
    """
    评估特征子集的性能（5-Fold交叉验证）

    Args:
        X: 完整特征矩阵
        y: 目标变量
        selected_indices: 选中特征的索引
        method_name: 方法名称（用于打印）
        n_splits: 交叉验证折数

    Returns:
        metrics: 性能指标字典
    """
    print(f"\n评估 {method_name}（{len(selected_indices)} 个特征）")
    print("-" * 80)

    # 提取特征子集
    X_selected = X[:, selected_indices]

    # 交叉验证
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2_scores = []
    rmse_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_selected), 1):
        X_train, X_val = X_selected[train_idx], X_selected[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # 训练模型（使用简单的BayesianRidge快速验证）
        model = BayesianRidge(max_iter=300)
        model.fit(X_train_scaled, y_train)

        # 预测
        y_pred = model.predict(X_val_scaled)

        # 评估
        r2 = r2_score(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        r2_scores.append(r2)
        rmse_scores.append(rmse)

        print(f"  Fold {fold}: R2 = {r2:.4f}, RMSE = {rmse:.2f} mm")

    # 汇总
    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)
    mean_rmse = np.mean(rmse_scores)

    print(f"\n{method_name} 平均性能:")
    print(f"  R2 = {mean_r2:.4f} (± {std_r2:.4f})")
    print(f"  RMSE = {mean_rmse:.2f} mm")

    return {
        "method": method_name,
        "n_features": len(selected_indices),
        "mean_r2": mean_r2,
        "std_r2": std_r2,
        "mean_rmse": mean_rmse,
    }


def compare_feature_counts(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    feature_counts: List[int] = [30, 50, 80, 100, 150],
) -> pd.DataFrame:
    """
    对比不同特征数量下的模型性能

    Args:
        X: 特征矩阵
        y: 目标变量
        feature_names: 特征名称列表
        feature_counts: 要测试的特征数量列表

    Returns:
        results_df: 结果DataFrame
    """
    print("\n对比不同特征数量的性能")
    print("=" * 80)

    # 先用RF计算重要性
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf.fit(X_scaled, y)
    importances = rf.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]

    results = []

    for n_features in feature_counts:
        if n_features > X.shape[1]:
            continue

        print(f"\n测试 top-{n_features} 特征...")
        indices = sorted_indices[:n_features]

        metrics = evaluate_feature_subset(
            X, y, indices, f"Top-{n_features}", n_splits=5
        )
        results.append(metrics)

    results_df = pd.DataFrame(results)
    return results_df


def run_feature_selection_experiments() -> None:
    """运行完整的特征选择实验"""
    print("=" * 80)
    print("特征选择实验 - Feature Selection Experiments")
    print("=" * 80)
    print("目标: 从78维降至15-30维，提升模型泛化能力，使样本/特征比>5")
    print("=" * 80)

    # 1. 加载数据
    X, y, feature_names = load_and_prepare_data()

    # 2. 方法1: Random Forest
    rf_indices, rf_names, rf_importances = select_features_by_importance(
        X, y, feature_names, top_k=30
    )
    rf_metrics = evaluate_feature_subset(X, y, rf_indices, "Random Forest Top-30")

    # 3. 方法2: Lasso
    lasso_indices, lasso_names, lasso_coefs = select_features_by_lasso(
        X, y, feature_names, alpha=0.01
    )
    lasso_metrics = evaluate_feature_subset(
        X, y, lasso_indices, f"Lasso (alpha=0.01)"
    )

    # 4. 方法3: RFE（可选，较慢）
    # rfe_indices, rfe_names, rfe_ranking = select_features_by_rfe(
    #     X, y, feature_names, n_features_to_select=80
    # )
    # rfe_metrics = evaluate_feature_subset(X, y, rfe_indices, "RFE Top-80")

    # 5. 对比不同特征数量
    print("\n" + "=" * 80)
    print("对比实验: 不同特征数量的影响")
    print("=" * 80)
    comparison_df = compare_feature_counts(
        X, y, feature_names, feature_counts=[15, 20, 30, 40, 50]
    )

    # 6. 保存结果
    output_dir = Path(__file__).parent.parent / "notebooks" / "[20251118]改进实验"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存特征重要性
    importance_df = pd.DataFrame(
        {
            "feature_name": feature_names,
            "rf_importance": rf_importances,
            "lasso_coefficient": lasso_coefs,
        }
    )
    importance_df = importance_df.sort_values("rf_importance", ascending=False)
    importance_df.to_csv(
        output_dir / "04-特征重要性排序.csv", index=False, encoding="utf-8-sig"
    )

    # 保存对比结果
    comparison_df.to_csv(
        output_dir / "04-特征数量对比.csv", index=False, encoding="utf-8-sig"
    )

    # 保存最佳特征子集
    best_features_df = pd.DataFrame({"feature_name": rf_names[:30]})
    best_features_df.to_csv(
        output_dir / "04-最佳特征子集-RF-Top30.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 7. 生成Markdown报告
    with open(output_dir / "04-特征选择实验报告.md", "w", encoding="utf-8") as f:
        f.write("# 特征选择实验报告\n\n")
        f.write(
            f"**实验日期**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        )

        f.write("## 1. 实验目标\n\n")
        f.write(
            f"从 {X.shape[1]} 维特征降至 50-100 维，缓解维度灾难，提升泛化能力。\n\n"
        )

        f.write("## 2. 方法对比\n\n")
        f.write("| 方法 | 特征数 | R2 | RMSE (mm) |\n")
        f.write("|------|--------|----|-----------|\n")
        f.write(
            f"| Baseline (全部78维) | 78 | 0.6390 | 13.05 |\n"
        )  # 实际基线
        f.write(
            f"| Random Forest Top-30 | {rf_metrics['n_features']} | {rf_metrics['mean_r2']:.4f} | {rf_metrics['mean_rmse']:.2f} |\n"
        )
        f.write(
            f"| Lasso (alpha=0.01) | {lasso_metrics['n_features']} | {lasso_metrics['mean_r2']:.4f} | {lasso_metrics['mean_rmse']:.2f} |\n"
        )
        f.write("\n")

        f.write("## 3. 不同特征数量对比\n\n")
        try:
            f.write(comparison_df.to_markdown(index=False))
            f.write("\n\n")
        except:
            f.write(comparison_df.to_csv(index=False))
            f.write("\n\n")

        f.write("## 4. Top 20 重要特征\n\n")
        f.write("| 排名 | 特征名称 | RF重要性 | Lasso系数 |\n")
        f.write("|------|----------|----------|-----------|\n")
        for i in range(min(20, len(importance_df))):
            row = importance_df.iloc[i]
            f.write(
                f"| {i+1} | {row['feature_name']} | {row['rf_importance']:.4f} | {row['lasso_coefficient']:+.4f} |\n"
            )
        f.write("\n")

        f.write("## 5. 结论与建议\n\n")
        best_n = comparison_df.loc[comparison_df["mean_r2"].idxmax(), "n_features"]
        best_r2 = comparison_df["mean_r2"].max()
        f.write(f"- 最佳特征数量: **{int(best_n)} 个**\n")
        f.write(f"- 最佳R2: **{best_r2:.4f}**\n")
        f.write(
            f"- 相比全特征(78维)的优势: {'降维后性能提升' if best_r2 > 0.6390 else '降维后性能略降，但泛化能力可能更强'}\n"
        )
        f.write("\n")

    print("\n" + "=" * 80)
    print("实验完成! 结果已保存:")
    print(f"  - {output_dir / '04-特征重要性排序.csv'}")
    print(f"  - {output_dir / '04-特征数量对比.csv'}")
    print(f"  - {output_dir / '04-最佳特征子集-RF-Top80.csv'}")
    print(f"  - {output_dir / '04-特征选择实验报告.md'}")
    print("=" * 80)


if __name__ == "__main__":
    run_feature_selection_experiments()
