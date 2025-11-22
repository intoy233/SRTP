#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
加权损失Stacking实验 - Weighted Loss Stacking Experiments

目标: 通过对高风险样本赋予更高权重，提升模型对关键区域的预测精度

核心思想:
- 当前模型对高风险样本(>60mm)的R²仅0.40
- 通过加权损失函数，让模型更关注这部分样本
- Trade-off: 高风险精度↑，整体精度可能略↓（可接受）

实验设计:
- 测试不同的权重比例: 1:1.5, 1:2, 1:3, 1:5
- 对比高风险样本的R²提升
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .final_viv_predictor import VIVPredictor


def load_dataset(data_path: Path) -> pd.DataFrame:
    """加载数据集"""
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    return pd.read_csv(data_path)


def build_feature_matrix(data_frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """构建特征矩阵"""
    predictor = VIVPredictor()
    features_matrix, valid_index = predictor._create_features(data_frame)
    targets = data_frame.loc[valid_index, "Max_Amplitude_mm"].values
    return features_matrix, targets


def fit_weighted_stacking_model(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    sample_weights: np.ndarray,
) -> Tuple[StandardScaler, list, BayesianRidge]:
    """
    训练加权Stacking模型

    Args:
        train_features: 训练特征
        train_targets: 训练目标
        sample_weights: 样本权重（高风险样本权重更高）

    Returns:
        scaler: 标准化器
        base_models: 基学习器列表
        meta_model: 元学习器
    """
    scaler = StandardScaler()
    scaled_train_features = scaler.fit_transform(train_features)

    # Level 0: 基学习器
    base_models = [
        ("Ridge", Ridge(alpha=10.0)),
        ("Lasso", Lasso(alpha=0.1, max_iter=5000)),
        (
            "RandomForest",
            RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
        ),
        ("SVR_RBF", SVR(kernel="rbf", C=10, gamma="scale")),
        ("BayesianRidge", BayesianRidge(max_iter=300, tol=1e-3)),
    ]

    # 训练基学习器，生成元特征（使用加权交叉验证）
    cross_validator = KFold(n_splits=3, shuffle=True, random_state=42)
    meta_features = np.zeros((len(scaled_train_features), len(base_models)))

    for model_index, (model_name, model) in enumerate(base_models):
        fold_predictions = np.zeros(len(scaled_train_features))

        for train_index, validation_index in cross_validator.split(
            scaled_train_features
        ):
            inner_train_features = scaled_train_features[train_index]
            inner_validation_features = scaled_train_features[validation_index]
            inner_train_targets = train_targets[train_index]
            inner_sample_weights = sample_weights[train_index]

            # 检查模型是否支持sample_weight
            if model_name in ["Ridge", "Lasso", "RandomForest", "BayesianRidge"]:
                # 这些模型支持sample_weight
                model.fit(
                    inner_train_features,
                    inner_train_targets,
                    sample_weight=inner_sample_weights,
                )
            else:
                # SVR等模型不直接支持sample_weight，使用加权样本复制策略
                # （简化处理：仅用原始权重训练）
                model.fit(inner_train_features, inner_train_targets)

            fold_predictions[validation_index] = model.predict(
                inner_validation_features
            )

        meta_features[:, model_index] = fold_predictions

        # 在全部训练集上重新训练
        if model_name in ["Ridge", "Lasso", "RandomForest", "BayesianRidge"]:
            model.fit(
                scaled_train_features, train_targets, sample_weight=sample_weights
            )
        else:
            model.fit(scaled_train_features, train_targets)

    # Level 1: 元学习器（使用加权训练）
    meta_model = BayesianRidge(max_iter=300, tol=1e-3)
    meta_model.fit(meta_features, train_targets, sample_weight=sample_weights)

    return scaler, base_models, meta_model


def predict_stacking(
    test_features: np.ndarray,
    scaler: StandardScaler,
    base_models: list,
    meta_model: BayesianRidge,
) -> np.ndarray:
    """使用Stacking模型预测"""
    scaled_test_features = scaler.transform(test_features)
    meta_features = np.zeros((len(scaled_test_features), len(base_models)))

    for model_index, (_, model) in enumerate(base_models):
        meta_features[:, model_index] = model.predict(scaled_test_features)

    predictions, _ = meta_model.predict(meta_features, return_std=True)
    return predictions


def evaluate_weighted_stacking(
    features_matrix: np.ndarray,
    targets: np.ndarray,
    weight_ratio: float = 2.0,
    high_risk_threshold: float = 60.0,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    评估加权Stacking模型

    Args:
        features_matrix: 特征矩阵
        targets: 目标变量
        weight_ratio: 高风险样本的权重比例（相对于低风险样本）
        high_risk_threshold: 高风险阈值（mm）
        n_splits: 交叉验证折数
        random_state: 随机种子

    Returns:
        metrics: 评估指标字典
    """
    cross_validator = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    all_true_targets: List[np.ndarray] = []
    all_predictions: List[np.ndarray] = []

    print(f"\n评估加权Stacking（权重比例 = 1:{weight_ratio}）")
    print("=" * 80)

    for fold, (train_index, test_index) in enumerate(
        cross_validator.split(features_matrix), 1
    ):
        print(f"\nFold {fold}/{n_splits}")
        print("-" * 80)

        train_features = features_matrix[train_index]
        test_features = features_matrix[test_index]
        train_targets = targets[train_index]
        test_targets = targets[test_index]

        # 计算样本权重
        sample_weights = np.where(
            train_targets > high_risk_threshold, weight_ratio, 1.0
        )

        high_risk_count = np.sum(train_targets > high_risk_threshold)
        low_risk_count = len(train_targets) - high_risk_count
        print(
            f"训练集: {len(train_targets)} 样本 (高风险: {high_risk_count}, 低风险: {low_risk_count})"
        )
        print(f"高风险样本权重: {weight_ratio:.1f}x")

        # 训练加权模型
        scaler, base_models, meta_model = fit_weighted_stacking_model(
            train_features, train_targets, sample_weights
        )

        # 预测
        fold_predictions = predict_stacking(test_features, scaler, base_models, meta_model)

        # 保存结果
        all_true_targets.append(test_targets)
        all_predictions.append(fold_predictions)

        # Fold级别的评估
        fold_r2 = r2_score(test_targets, fold_predictions)
        fold_rmse = np.sqrt(mean_squared_error(test_targets, fold_predictions))

        # 高风险子集评估
        high_risk_mask = test_targets > high_risk_threshold
        if np.any(high_risk_mask):
            high_risk_r2 = r2_score(
                test_targets[high_risk_mask], fold_predictions[high_risk_mask]
            )
            high_risk_rmse = np.sqrt(
                mean_squared_error(
                    test_targets[high_risk_mask], fold_predictions[high_risk_mask]
                )
            )
            print(
                f"Fold {fold} - 整体: R²={fold_r2:.4f}, RMSE={fold_rmse:.2f}mm"
            )
            print(
                f"Fold {fold} - 高风险: R²={high_risk_r2:.4f}, RMSE={high_risk_rmse:.2f}mm"
            )
        else:
            print(
                f"Fold {fold} - 整体: R²={fold_r2:.4f}, RMSE={fold_rmse:.2f}mm"
            )
            print(f"Fold {fold} - 高风险: （无高风险样本）")

    # 汇总所有fold的结果
    concatenated_true = np.concatenate(all_true_targets)
    concatenated_predictions = np.concatenate(all_predictions)

    # 整体指标
    overall_r2 = r2_score(concatenated_true, concatenated_predictions)
    overall_rmse = float(
        np.sqrt(mean_squared_error(concatenated_true, concatenated_predictions))
    )

    # 高风险子集指标
    high_risk_mask = concatenated_true > high_risk_threshold
    if np.any(high_risk_mask):
        high_risk_true = concatenated_true[high_risk_mask]
        high_risk_pred = concatenated_predictions[high_risk_mask]
        high_risk_r2 = r2_score(high_risk_true, high_risk_pred)
        high_risk_rmse = float(np.sqrt(mean_squared_error(high_risk_true, high_risk_pred)))
        high_risk_count = np.sum(high_risk_mask)
    else:
        high_risk_r2 = float("nan")
        high_risk_rmse = float("nan")
        high_risk_count = 0

    print("\n" + "=" * 80)
    print("汇总结果")
    print("=" * 80)
    print(f"整体性能:")
    print(f"  R² = {overall_r2:.4f}")
    print(f"  RMSE = {overall_rmse:.2f} mm")
    print(f"\n高风险子集 (>{high_risk_threshold}mm, n={high_risk_count}):")
    print(f"  R² = {high_risk_r2:.4f}")
    print(f"  RMSE = {high_risk_rmse:.2f} mm")

    return {
        "weight_ratio": weight_ratio,
        "overall_r2": float(overall_r2),
        "overall_rmse": overall_rmse,
        "high_risk_r2": float(high_risk_r2),
        "high_risk_rmse": high_risk_rmse,
        "high_risk_count": int(high_risk_count),
    }


def run_weighted_stacking_experiments() -> None:
    """运行完整的加权Stacking实验"""
    print("=" * 80)
    print("加权损失Stacking实验")
    print("=" * 80)
    print("目标: 提升高风险样本(>60mm)的预测精度")
    print("=" * 80)

    # 加载数据
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "final_bridge_dataset.csv"

    print("\n加载数据集...")
    data_frame = load_dataset(data_path)
    print(f"数据集: {len(data_frame)} 座桥梁")

    print("\n构建特征矩阵...")
    features_matrix, targets = build_feature_matrix(data_frame)
    print(f"特征矩阵: {features_matrix.shape}")

    # 数据分布统计
    high_risk_count = np.sum(targets > 60)
    low_risk_count = len(targets) - high_risk_count
    print(f"\n数据分布:")
    print(f"  高风险 (>60mm): {high_risk_count} 座 ({high_risk_count/len(targets)*100:.1f}%)")
    print(f"  低中风险 (≤60mm): {low_risk_count} 座 ({low_risk_count/len(targets)*100:.1f}%)")

    # 实验：测试不同的权重比例
    weight_ratios = [1.0, 1.5, 2.0, 3.0, 5.0]
    results = []

    for weight_ratio in weight_ratios:
        print(f"\n{'='*80}")
        print(f"实验: 权重比例 = 1:{weight_ratio}")
        print(f"{'='*80}")

        if weight_ratio == 1.0:
            print("(这是基线实验，等权重)")

        metrics = evaluate_weighted_stacking(
            features_matrix,
            targets,
            weight_ratio=weight_ratio,
            high_risk_threshold=60.0,
            n_splits=5,
        )

        results.append(metrics)

    # 保存结果
    output_dir = project_root / "notebooks" / "[20251118]改进实验"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)

    # 保存CSV
    csv_path = output_dir / "05-加权Stacking结果.csv"
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 保存Markdown报告
    md_path = output_dir / "05-加权Stacking实验报告.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 加权损失Stacking实验报告\n\n")
        f.write(f"**实验日期**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        f.write("## 1. 实验目标\n\n")
        f.write("通过对高风险样本（振幅>60mm）赋予更高权重，提升模型对关键区域的预测精度。\n\n")

        f.write("## 2. 实验设计\n\n")
        f.write("- **高风险阈值**: 60mm\n")
        f.write(f"- **高风险样本数**: {high_risk_count} 座 ({high_risk_count/len(targets)*100:.1f}%)\n")
        f.write(f"- **低中风险样本数**: {low_risk_count} 座 ({low_risk_count/len(targets)*100:.1f}%)\n")
        f.write("- **测试权重比例**: 1:1.0, 1:1.5, 1:2.0, 1:3.0, 1:5.0\n")
        f.write("- **交叉验证**: 5-Fold\n\n")

        f.write("## 3. 实验结果\n\n")
        try:
            f.write(results_df.to_markdown(index=False))
            f.write("\n\n")
        except:
            f.write("```\n")
            f.write(results_df.to_string(index=False))
            f.write("\n```\n\n")

        f.write("## 4. 结果分析\n\n")

        # 找到高风险R²最高的权重比例
        best_idx = results_df["high_risk_r2"].idxmax()
        best_ratio = results_df.loc[best_idx, "weight_ratio"]
        best_high_risk_r2 = results_df.loc[best_idx, "high_risk_r2"]
        baseline_high_risk_r2 = results_df.loc[0, "high_risk_r2"]  # weight_ratio=1.0

        improvement = (
            (best_high_risk_r2 - baseline_high_risk_r2) / baseline_high_risk_r2 * 100
        )

        f.write(f"- **最佳权重比例**: 1:{best_ratio:.1f}\n")
        f.write(f"- **最佳高风险R²**: {best_high_risk_r2:.4f}\n")
        f.write(f"- **相比基线提升**: {improvement:+.1f}%\n\n")

        f.write("### 4.1 高风险R²变化趋势\n\n")
        for _, row in results_df.iterrows():
            f.write(
                f"- 权重1:{row['weight_ratio']:.1f} → 高风险R² = {row['high_risk_r2']:.4f}\n"
            )
        f.write("\n")

        f.write("### 4.2 整体性能与高风险性能的Trade-off\n\n")
        baseline_overall_r2 = results_df.loc[0, "overall_r2"]
        best_overall_r2 = results_df.loc[best_idx, "overall_r2"]
        overall_change = (
            (best_overall_r2 - baseline_overall_r2) / baseline_overall_r2 * 100
        )

        f.write(
            f"- 最佳权重下，整体R²变化: {baseline_overall_r2:.4f} → {best_overall_r2:.4f} ({overall_change:+.1f}%)\n"
        )

        if overall_change < -2:
            f.write(
                "- ⚠️ 整体性能下降超过2%，需权衡是否值得为提升高风险精度而牺牲整体性能\n"
            )
        elif overall_change < 0:
            f.write(
                "- ✓ 整体性能略微下降（可接受），换取高风险区域性能提升\n"
            )
        else:
            f.write(
                "- ✓✓ 整体性能未降反升，加权策略双赢！\n"
            )
        f.write("\n")

        f.write("## 5. 结论与建议\n\n")

        if improvement > 10:
            f.write(
                f"- ✓ **显著提升**：高风险R²提升{improvement:.1f}%，加权损失策略有效\n"
            )
            f.write(f"- **建议权重比例**: 1:{best_ratio:.1f}\n")
            f.write(
                "- **后续行动**: 将加权Stacking作为高风险样本预测的首选模型\n"
            )
        elif improvement > 5:
            f.write(
                f"- ⚠️ **中等提升**：高风险R²提升{improvement:.1f}%，有一定效果\n"
            )
            f.write(
                "- **建议**: 结合其他方法（如SMOTE、特征工程）进一步提升\n"
            )
        else:
            f.write(
                f"- ⚠️ **提升有限**：高风险R²仅提升{improvement:.1f}%\n"
            )
            f.write(
                "- **建议**: 优先尝试其他方法（特征选择、GAN、分段建模）\n"
            )
        f.write("\n")

    print("\n" + "=" * 80)
    print("实验完成! 结果已保存:")
    print(f"  - CSV: {csv_path}")
    print(f"  - 报告: {md_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_weighted_stacking_experiments()
