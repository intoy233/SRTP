#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2: 训练Version A (真实数据baseline)
使用Stacking模型,5折交叉验证
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import json
from datetime import datetime


def load_and_prepare_data(csv_path: str):
    """加载并准备数据"""
    print("\n1. Loading and preparing data...")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f"   Total samples: {len(df)}")

    # 特征工程 (简化版,直接使用原始特征)
    feature_cols = [
        'Span_m', 'Width_m', 'Height_m', 'Width_Height_Ratio',
        'Natural_Freq_Hz', 'First_Freq_Hz', 'Damping_Ratio',
        'Critical_Wind_Speed_ms', 'Drag_Coefficient', 'Lift_Coefficient'
    ]

    # 检查缺失
    available_features = [col for col in feature_cols if col in df.columns]
    print(f"   Available features: {len(available_features)}")

    X = df[available_features].copy()
    y = df['Max_Amplitude_mm'].copy()

    # 处理缺失值
    X = X.fillna(X.median())

    print(f"   Feature shape: {X.shape}")
    print(f"   Target shape: {y.shape}")

    return X, y, available_features


def create_stacking_model():
    """创建Stacking模型"""
    print("\n2. Creating Stacking model...")

    # 基学习器
    base_learners = [
        ('ridge', Ridge(alpha=10.0, random_state=42)),
        ('lasso', Lasso(alpha=1.0, random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=10,
                                     random_state=42, n_jobs=-1)),
        ('svr', SVR(kernel='rbf', C=10.0, gamma='scale'))
    ]

    # 元学习器
    meta_learner = BayesianRidge()

    # Stacking模型
    model = StackingRegressor(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )

    print("   Base learners: Ridge, Lasso, RandomForest, SVR")
    print("   Meta learner: BayesianRidge")
    print("   CV folds: 5")

    return model


def evaluate_with_cross_validation(model, X, y, n_folds=5):
    """使用交叉验证评估模型"""
    print(f"\n3. Evaluating with {n_folds}-fold cross-validation...")

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5折交叉验证
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # R2评分
    r2_scores = cross_val_score(model, X_scaled, y, cv=kfold,
                                scoring='r2', n_jobs=-1)

    # RMSE评分
    neg_rmse_scores = cross_val_score(model, X_scaled, y, cv=kfold,
                                      scoring='neg_root_mean_squared_error',
                                      n_jobs=-1)
    rmse_scores = -neg_rmse_scores

    # MAE评分
    neg_mae_scores = cross_val_score(model, X_scaled, y, cv=kfold,
                                     scoring='neg_mean_absolute_error',
                                     n_jobs=-1)
    mae_scores = -neg_mae_scores

    print(f"\n   Cross-Validation Results ({n_folds} folds):")
    print(f"   =" * 60)
    print(f"   R2 Scores:")
    for i, score in enumerate(r2_scores, 1):
        print(f"     Fold {i}: {score:.4f}")
    print(f"   Mean R2: {r2_scores.mean():.4f} (±{r2_scores.std():.4f})")

    print(f"\n   RMSE Scores:")
    for i, score in enumerate(rmse_scores, 1):
        print(f"     Fold {i}: {score:.2f} mm")
    print(f"   Mean RMSE: {rmse_scores.mean():.2f} (±{rmse_scores.std():.2f}) mm")

    print(f"\n   MAE Scores:")
    for i, score in enumerate(mae_scores, 1):
        print(f"     Fold {i}: {score:.2f} mm")
    print(f"   Mean MAE: {mae_scores.mean():.2f} (±{mae_scores.std():.2f}) mm")

    return {
        'r2_mean': float(r2_scores.mean()),
        'r2_std': float(r2_scores.std()),
        'r2_scores': r2_scores.tolist(),
        'rmse_mean': float(rmse_scores.mean()),
        'rmse_std': float(rmse_scores.std()),
        'rmse_scores': rmse_scores.tolist(),
        'mae_mean': float(mae_scores.mean()),
        'mae_std': float(mae_scores.std()),
        'mae_scores': mae_scores.tolist()
    }


def evaluate_high_risk_performance(model, X, y, n_folds=5):
    """评估高风险样本性能 (>60mm)"""
    print("\n4. Evaluating HIGH-RISK sample performance...")

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5折交叉验证
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    high_risk_r2_scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_scaled), 1):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 训练模型
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_test)

        # 筛选高风险样本 (>60mm)
        high_risk_mask = y_test > 60
        high_risk_count = high_risk_mask.sum()

        if high_risk_count > 0:
            y_test_high = y_test[high_risk_mask]
            y_pred_high = y_pred[high_risk_mask]

            # 计算R2
            r2_high = r2_score(y_test_high, y_pred_high)
            high_risk_r2_scores.append(r2_high)

            print(f"   Fold {fold_idx}: {high_risk_count} high-risk samples, R2 = {r2_high:.4f}")
        else:
            print(f"   Fold {fold_idx}: No high-risk samples in test set")

    if len(high_risk_r2_scores) > 0:
        print(f"\n   Mean High-Risk R2: {np.mean(high_risk_r2_scores):.4f} "
              f"(±{np.std(high_risk_r2_scores):.4f})")
        return {
            'high_risk_r2_mean': float(np.mean(high_risk_r2_scores)),
            'high_risk_r2_std': float(np.std(high_risk_r2_scores)),
            'high_risk_r2_scores': [float(s) for s in high_risk_r2_scores]
        }
    else:
        print(f"\n   WARNING: No high-risk samples found in any fold")
        return {
            'high_risk_r2_mean': None,
            'high_risk_r2_std': None,
            'high_risk_r2_scores': []
        }


def train_final_model(model, X, y):
    """训练最终模型并保存"""
    print("\n5. Training final model on full dataset...")

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 训练
    model.fit(X_scaled, y)

    # 全数据集评估
    y_pred = model.predict(X_scaled)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    print(f"   Full dataset performance:")
    print(f"     R2: {r2:.4f}")
    print(f"     RMSE: {rmse:.2f} mm")
    print(f"     MAE: {mae:.2f} mm")

    # 高风险样本评估
    high_risk_mask = y > 60
    if high_risk_mask.sum() > 0:
        y_high = y[high_risk_mask]
        y_pred_high = y_pred[high_risk_mask]
        r2_high = r2_score(y_high, y_pred_high)
        rmse_high = np.sqrt(mean_squared_error(y_high, y_pred_high))

        print(f"\n   High-risk samples ({high_risk_mask.sum()}):")
        print(f"     R2: {r2_high:.4f}")
        print(f"     RMSE: {rmse_high:.2f} mm")

    return model, scaler, {
        'full_r2': float(r2),
        'full_rmse': float(rmse),
        'full_mae': float(mae),
        'high_risk_r2': float(r2_high) if high_risk_mask.sum() > 0 else None,
        'high_risk_rmse': float(rmse_high) if high_risk_mask.sum() > 0 else None
    }


def save_results(model, scaler, cv_results, high_risk_results, full_results,
                feature_names, output_dir):
    """保存模型和结果"""
    print("\n6. Saving results...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 保存模型
    model_path = output_path / 'version_a_stacking_model.pkl'
    joblib.dump(model, model_path)
    print(f"   Model saved: {model_path}")

    # 保存scaler
    scaler_path = output_path / 'version_a_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"   Scaler saved: {scaler_path}")

    # 保存结果
    results = {
        'version': 'A (Real Data Baseline)',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_info': {
            'n_samples': int(len(feature_names)),
            'n_features': int(len(feature_names))
        },
        'cross_validation': cv_results,
        'high_risk_performance': high_risk_results,
        'full_dataset': full_results,
        'feature_names': feature_names
    }

    results_path = output_path / 'version_a_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"   Results saved: {results_path}")


def main():
    """主函数"""
    print("=" * 80)
    print("Version A (Real Data Baseline) - Model Training")
    print("=" * 80)

    # 路径配置
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data' / 'final_bridge_dataset_version_a.csv'
    output_dir = project_root / 'experiments' / 'version_a'

    if not data_path.exists():
        print(f"ERROR: Dataset not found: {data_path}")
        return False

    try:
        # 1. 加载数据
        X, y, feature_names = load_and_prepare_data(str(data_path))

        # 2. 创建模型
        model = create_stacking_model()

        # 3. 交叉验证
        cv_results = evaluate_with_cross_validation(model, X, y, n_folds=5)

        # 4. 高风险样本评估
        high_risk_results = evaluate_high_risk_performance(model, X, y, n_folds=5)

        # 5. 训练最终模型
        final_model, scaler, full_results = train_final_model(model, X, y)

        # 6. 保存结果
        save_results(final_model, scaler, cv_results, high_risk_results,
                    full_results, feature_names, str(output_dir))

        print("\n" + "=" * 80)
        print("Version A Training Complete!")
        print("=" * 80)
        print(f"\nKey Results:")
        print(f"  Overall R2: {cv_results['r2_mean']:.4f} (±{cv_results['r2_std']:.4f})")
        print(f"  Overall RMSE: {cv_results['rmse_mean']:.2f} (±{cv_results['rmse_std']:.2f}) mm")

        if high_risk_results['high_risk_r2_mean'] is not None:
            print(f"  High-Risk R2: {high_risk_results['high_risk_r2_mean']:.4f} "
                  f"(±{high_risk_results['high_risk_r2_std']:.4f})")

        print(f"\nResults saved to: {output_dir}")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\nERROR: Training failed - {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
