#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Imbalance experiments for bridge VIV prediction.

This module provides:
1. A unified baseline evaluation for the current Stacking model.
2. A SMOTE-style oversampling experiment focusing on high-risk bridges.

It reuses the feature engineering from src.final_viv_predictor.VIVPredictor
and reports both overall metrics and metrics on the high-risk subset
(Max_Amplitude_mm > 60mm by default).
"""

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

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
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    return pd.read_csv(data_path)


def build_feature_matrix(data_frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    predictor = VIVPredictor()
    features_matrix, valid_index = predictor._create_features(data_frame)  # type: ignore[attr-defined]
    targets = data_frame.loc[valid_index, "Max_Amplitude_mm"].values
    return features_matrix, targets


def fit_stacking_model(
    train_features: np.ndarray,
    train_targets: np.ndarray,
) -> Tuple[StandardScaler, list, BayesianRidge]:
    scaler = StandardScaler()
    scaled_train_features = scaler.fit_transform(train_features)

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
    meta_model = BayesianRidge(max_iter=300, tol=1e-3)

    cross_validator = KFold(n_splits=3, shuffle=True, random_state=42)
    meta_features = np.zeros((len(scaled_train_features), len(base_models)))

    for model_index, (_, model) in enumerate(base_models):
        fold_predictions = np.zeros(len(scaled_train_features))
        for train_index, validation_index in cross_validator.split(scaled_train_features):
            inner_train_features = scaled_train_features[train_index]
            inner_validation_features = scaled_train_features[validation_index]
            inner_train_targets = train_targets[train_index]

            model.fit(inner_train_features, inner_train_targets)
            fold_predictions[validation_index] = model.predict(inner_validation_features)

        meta_features[:, model_index] = fold_predictions
        model.fit(scaled_train_features, train_targets)

    meta_model.fit(meta_features, train_targets)
    return scaler, base_models, meta_model


def predict_stacking(
    test_features: np.ndarray,
    scaler: StandardScaler,
    base_models: list,
    meta_model: BayesianRidge,
) -> np.ndarray:
    scaled_test_features = scaler.transform(test_features)
    meta_features = np.zeros((len(scaled_test_features), len(base_models)))
    for model_index, (_, model) in enumerate(base_models):
        meta_features[:, model_index] = model.predict(scaled_test_features)
    predictions, _ = meta_model.predict(meta_features, return_std=True)
    return predictions


def smote_high_risk_oversampler(
    threshold_mm: float = 60.0,
    oversampling_factor: float = 2.0,
    random_state: int = 42,
) -> Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    random_generator = np.random.RandomState(random_state)

    def oversample(
        train_features: np.ndarray,
        train_targets: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        high_risk_mask = train_targets > threshold_mm
        high_risk_features = train_features[high_risk_mask]
        high_risk_targets = train_targets[high_risk_mask]

        if len(high_risk_features) < 2:
            return train_features, train_targets

        target_high_risk_count = int(len(high_risk_features) * oversampling_factor)
        additional_count = max(0, target_high_risk_count - len(high_risk_features))

        if additional_count == 0:
            return train_features, train_targets

        synthetic_features = []
        synthetic_targets = []

        for _ in range(additional_count):
            index_a = random_generator.randint(0, len(high_risk_features))
            index_b = random_generator.randint(0, len(high_risk_features))
            feature_a = high_risk_features[index_a]
            feature_b = high_risk_features[index_b]
            target_a = high_risk_targets[index_a]
            target_b = high_risk_targets[index_b]

            interpolation_weight = random_generator.rand()
            new_feature = feature_a + interpolation_weight * (feature_b - feature_a)
            new_target = target_a + interpolation_weight * (target_b - target_a)

            synthetic_features.append(new_feature)
            synthetic_targets.append(new_target)

        if synthetic_features:
            synthetic_features_array = np.stack(synthetic_features, axis=0)
            synthetic_targets_array = np.array(synthetic_targets)

            augmented_features = np.concatenate(
                [train_features, synthetic_features_array], axis=0
            )
            augmented_targets = np.concatenate(
                [train_targets, synthetic_targets_array], axis=0
            )
            return augmented_features, augmented_targets

        return train_features, train_targets

    return oversample


def evaluate_with_cross_validation(
    features_matrix: np.ndarray,
    targets: np.ndarray,
    high_risk_threshold_mm: float = 60.0,
    n_splits: int = 5,
    random_state: int = 42,
    oversampler: Optional[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
) -> Dict[str, float]:
    cross_validator = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    all_true_targets: list = []
    all_predictions: list = []

    for train_index, test_index in cross_validator.split(features_matrix):
        train_features = features_matrix[train_index]
        test_features = features_matrix[test_index]
        train_targets = targets[train_index]
        test_targets = targets[test_index]

        if oversampler is not None:
            train_features, train_targets = oversampler(train_features, train_targets)

        scaler, base_models, meta_model = fit_stacking_model(train_features, train_targets)
        fold_predictions = predict_stacking(test_features, scaler, base_models, meta_model)

        all_true_targets.append(test_targets)
        all_predictions.append(fold_predictions)

    concatenated_true = np.concatenate(all_true_targets)
    concatenated_predictions = np.concatenate(all_predictions)

    overall_r2 = r2_score(concatenated_true, concatenated_predictions)
    overall_rmse = float(np.sqrt(mean_squared_error(concatenated_true, concatenated_predictions)))

    high_risk_mask = concatenated_true > high_risk_threshold_mm
    if np.any(high_risk_mask):
        high_risk_true = concatenated_true[high_risk_mask]
        high_risk_pred = concatenated_predictions[high_risk_mask]
        high_risk_r2 = r2_score(high_risk_true, high_risk_pred)
        high_risk_rmse = float(np.sqrt(mean_squared_error(high_risk_true, high_risk_pred)))
    else:
        high_risk_r2 = float("nan")
        high_risk_rmse = float("nan")

    return {
        "overall_r2": float(overall_r2),
        "overall_rmse": overall_rmse,
        "high_risk_r2": float(high_risk_r2),
        "high_risk_rmse": high_risk_rmse,
    }


def run_baseline_and_smote_experiments() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "final_bridge_dataset.csv"

    results = []

    print("Loading dataset...")
    data_frame = load_dataset(data_path)
    print(f"Loaded {len(data_frame)} bridges from {data_path}")

    print("Building feature matrix using VIVPredictor feature engineering...")
    features_matrix, targets = build_feature_matrix(data_frame)
    print(f"Feature matrix shape: {features_matrix.shape}")

    print("\nEvaluating baseline Stacking model (no oversampling)...")
    baseline_metrics = evaluate_with_cross_validation(features_matrix, targets)
    print("Baseline metrics:")
    print(f"  Overall R2:      {baseline_metrics['overall_r2']:.4f}")
    print(f"  Overall RMSE:    {baseline_metrics['overall_rmse']:.2f} mm")
    print(f"  High-risk R2:    {baseline_metrics['high_risk_r2']:.4f}")
    print(f"  High-risk RMSE:  {baseline_metrics['high_risk_rmse']:.2f} mm")

    results.append(
        {
            "experiment": "baseline",
            "oversampling_factor": 1.0,
            **baseline_metrics,
        }
    )

    print("\nEvaluating VIV-SMOTE experiments (high-risk oversampling)...")
    for factor in [1.5, 2.0, 3.0]:
        print(f"\n  Oversampling factor = {factor:.1f}")
        oversampler = smote_high_risk_oversampler(oversampling_factor=factor)
        smote_metrics = evaluate_with_cross_validation(
            features_matrix,
            targets,
            oversampler=oversampler,
        )
        print("  VIV-SMOTE metrics:")
        print(f"    Overall R2:      {smote_metrics['overall_r2']:.4f}")
        print(f"    Overall RMSE:    {smote_metrics['overall_rmse']:.2f} mm")
        print(f"    High-risk R2:    {smote_metrics['high_risk_r2']:.4f}")
        print(f"    High-risk RMSE:  {smote_metrics['high_risk_rmse']:.2f} mm")

        results.append(
            {
                "experiment": "viv_smote",
                "oversampling_factor": factor,
                **smote_metrics,
            }
        )

    output_dir = project_root / "notebooks" / "[20251118]改进实验"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_frame = pd.DataFrame(results)
    csv_path = output_dir / "02-基线与SMOTE结果.csv"
    md_path = output_dir / "02-基线与SMOTE结果.md"

    results_frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with open(md_path, "w", encoding="utf-8") as file:
        file.write("# 基线与 VIV-SMOTE 实验结果\n\n")
        file.write("本文件由 `src/imbalance_experiments.py` 自动生成。\n\n")
        file.write("## 指标说明\n\n")
        file.write("- `overall_r2` / `overall_rmse`: 全体样本 (190 座桥梁 + 合成样本) 的指标。\n")
        file.write("- `high_risk_r2` / `high_risk_rmse`: 高风险子集 (振幅 > 60mm) 的指标。\n")
        file.write("- `oversampling_factor`: 高风险样本在特征空间线性插值的扩增倍数。\n\n")

        file.write("## 实验结果表\n\n")
        try:
            table_markdown = results_frame.to_markdown(index=False)
            file.write(table_markdown)
            file.write("\n")
        except Exception:
            file.write(results_frame.to_csv(index=False))


if __name__ == "__main__":
    run_baseline_and_smote_experiments()
