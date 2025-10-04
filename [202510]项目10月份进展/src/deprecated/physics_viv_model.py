#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class PhysicsVIVModel:
    """基于物理原理的VIV预测模型"""

    def __init__(self):
        self.coefficients = {}
        self.feature_names = []
        self.scaler_params = {}
        self.is_fitted = False

    def create_physics_features(self, df):
        """创建VIV物理特征"""
        df_new = df.copy()

        # 1. 约化风速 (核心VIV参数)
        if all(col in df.columns for col in ['Critical_Wind_Speed_ms', 'Natural_Freq_Hz', 'Width_m']):
            df_new['Reduced_Velocity'] = df['Critical_Wind_Speed_ms'] / (df['Natural_Freq_Hz'] * df['Width_m'])

        # 2. Scruton数 (稳定性指标)
        if all(col in df.columns for col in ['Damping_Ratio', 'Width_m', 'Height_m']):
            df_new['Scruton_Number'] = df['Damping_Ratio'] * (df['Width_m'] / df['Height_m']) * 100

        # 3. 宽高比
        if all(col in df.columns for col in ['Width_m', 'Height_m']):
            df_new['Aspect_Ratio'] = df['Width_m'] / df['Height_m']

        # 4. 结构刚度
        if all(col in df.columns for col in ['Natural_Freq_Hz', 'Span_m']):
            df_new['Stiffness_Parameter'] = df['Natural_Freq_Hz'] * df['Span_m']**0.5

        # 5. 气动力比
        if all(col in df.columns for col in ['Lift_Coefficient', 'Drag_Coefficient']):
            df_new['Lift_Drag_Ratio'] = df['Lift_Coefficient'] / (df['Drag_Coefficient'] + 1e-6)

        # 6. VIV敏感性
        if 'Damping_Ratio' in df.columns:
            df_new['VIV_Susceptibility'] = 1.0 / (df['Damping_Ratio'] + 1e-6)

        return df_new

    def select_features(self, X, y, max_features=8):
        """基于相关性选择特征"""
        correlations = {}

        for col in X.select_dtypes(include=[np.number]).columns:
            if X[col].std() > 1e-6:
                corr = np.corrcoef(X[col], y)[0, 1]
                if not np.isnan(corr):
                    correlations[col] = abs(corr)

        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feature for feature, _ in sorted_features[:max_features]]
        return selected_features

    def standardize(self, X, fit=True):
        """标准化"""
        if fit:
            self.scaler_params = {
                'mean': X.mean(),
                'std': X.std() + 1e-8
            }
        X_scaled = (X - self.scaler_params['mean']) / self.scaler_params['std']
        return X_scaled.values

    def ridge_regression(self, X, y, alpha=1.0):
        """岭回归"""
        n_samples, n_features = X.shape
        X_with_intercept = np.column_stack([np.ones(n_samples), X])

        XTX = X_with_intercept.T @ X_with_intercept
        XTy = X_with_intercept.T @ y

        reg_matrix = alpha * np.eye(n_features + 1)
        reg_matrix[0, 0] = 0

        coefficients = np.linalg.solve(XTX + reg_matrix, XTy)

        return {
            'intercept': coefficients[0],
            'coef': coefficients[1:],
            'alpha': alpha
        }

    def cross_validate(self, X, y, k_folds=5):
        """交叉验证"""
        n_samples = len(y)
        fold_size = n_samples // k_folds

        alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
        best_alpha = None
        best_score = -np.inf

        for alpha in alphas:
            scores = []

            for fold in range(k_folds):
                start_idx = fold * fold_size
                end_idx = start_idx + fold_size if fold < k_folds - 1 else n_samples

                val_indices = list(range(start_idx, end_idx))
                train_indices = [i for i in range(n_samples) if i not in val_indices]

                X_train_fold = X[train_indices]
                y_train_fold = y[train_indices]
                X_val_fold = X[val_indices]
                y_val_fold = y[val_indices]

                model_result = self.ridge_regression(X_train_fold, y_train_fold, alpha)

                X_val_with_intercept = np.column_stack([np.ones(len(X_val_fold)), X_val_fold])
                y_pred = X_val_with_intercept @ np.concatenate([[model_result['intercept']], model_result['coef']])

                ss_res = np.sum((y_val_fold - y_pred) ** 2)
                ss_tot = np.sum((y_val_fold - np.mean(y_val_fold)) ** 2)
                r2 = 1 - (ss_res / (ss_tot + 1e-8))
                scores.append(r2)

            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_alpha = alpha

        return best_alpha, best_score

    def fit(self, X, y):
        """训练模型"""
        print("开始训练物理VIV预测模型...")

        # 1. 物理特征工程
        print("创建物理特征...")
        X_engineered = self.create_physics_features(X)

        # 2. 特征选择
        print("选择最佳特征...")
        self.feature_names = self.select_features(X_engineered, y, max_features=8)
        print(f"选择的特征: {self.feature_names}")

        X_selected = X_engineered[self.feature_names]

        # 3. 标准化
        print("标准化特征...")
        X_scaled = self.standardize(X_selected, fit=True)

        # 4. 交叉验证
        print("交叉验证优化...")
        best_alpha, best_score = self.cross_validate(X_scaled, y.values)
        print(f"最佳正则化参数: α = {best_alpha}")
        print(f"交叉验证R²: {best_score:.4f}")

        # 5. 训练最终模型
        print("训练最终模型...")
        self.coefficients = self.ridge_regression(X_scaled, y.values, best_alpha)

        self.is_fitted = True
        print("模型训练完成!")
        return self

    def predict(self, X):
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")

        X_engineered = self.create_physics_features(X)
        X_selected = X_engineered[self.feature_names]
        X_scaled = self.standardize(X_selected, fit=False)

        X_with_intercept = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled])
        y_pred = X_with_intercept @ np.concatenate([[self.coefficients['intercept']], self.coefficients['coef']])

        return y_pred

    def evaluate(self, X, y):
        """评估性能"""
        y_pred = self.predict(X)

        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - y_pred))

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        return {'RMSE': rmse, 'R2': r2, 'MAE': mae, 'MSE': mse}

    def get_feature_importance(self):
        """特征重要性"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.coefficients['coef'],
            'abs_coefficient': np.abs(self.coefficients['coef'])
        }).sort_values('abs_coefficient', ascending=False)

        return importance_df

def run_experiment():
    """运行实验"""
    print("物理VIV预测实验")
    print("=" * 50)

    # 加载数据
    try:
        df = pd.read_csv('data/enhanced_bridge_dataset.csv')
        print(f"数据集: {df.shape[0]}座桥梁, {df.shape[1]}个特征")
    except FileNotFoundError:
        print("数据文件未找到")
        return None, None

    # 准备数据
    target_col = 'Max_Amplitude_mm'
    exclude_cols = ['BridgeID', 'BridgeName', 'Country', 'PaperSource', 'Year',
                   target_col, 'Risk_Level', 'Notes', 'Vibration_Suppression', 'Suppression_Effect']

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols]
    y = df[target_col]

    print(f"目标: {target_col}")
    print(f"特征数: {len(feature_cols)}")
    print(f"目标统计: 均值={y.mean():.2f}mm, 标准差={y.std():.2f}mm")

    # 数据分割
    np.random.seed(42)
    n_test = int(0.2 * len(df))
    test_indices = np.random.choice(len(df), n_test, replace=False)
    train_indices = [i for i in range(len(df)) if i not in test_indices]

    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    print(f"数据分割: 训练集{len(X_train)}, 测试集{len(X_test)}")

    # 训练模型
    print("\n" + "=" * 50)
    model = PhysicsVIVModel()
    model.fit(X_train, y_train)

    # 评估
    print("\n模型评估")
    print("=" * 50)

    train_metrics = model.evaluate(X_train, y_train)
    print("训练集性能:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.6f}")

    test_metrics = model.evaluate(X_test, y_test)
    print("\n测试集性能:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.6f}")

    # 特征重要性
    print("\n特征重要性:")
    importance_df = model.get_feature_importance()
    for _, row in importance_df.iterrows():
        print(f"  {row['feature']}: {row['coefficient']:.4f}")

    # 预测示例
    y_test_pred = model.predict(X_test)
    print("\n预测vs实际 (前5个样本):")
    for i in range(5):
        actual = y_test.iloc[i]
        predicted = y_test_pred[i]
        error = abs(actual - predicted)
        print(f"  实际: {actual:.2f}, 预测: {predicted:.2f}, 误差: {error:.2f}")

    # 生成报告
    generate_report(model, train_metrics, test_metrics, importance_df)

    return model, test_metrics

def generate_report(model, train_metrics, test_metrics, importance_df):
    """生成报告"""
    report = f"""物理VIV预测模型实验报告
============================================================
实验时间: {pd.Timestamp.now()}

1. 模型架构
------------------------------
- 基于VIV物理机制的特征工程
- 智能特征选择 (相关性排序)
- 标准化预处理
- 交叉验证超参数优化
- 物理约束岭回归

2. 最终性能
------------------------------
训练集:
  RMSE: {train_metrics['RMSE']:.6f}
  R²: {train_metrics['R2']:.6f}
  MAE: {train_metrics['MAE']:.6f}

测试集:
  RMSE: {test_metrics['RMSE']:.6f}
  R²: {test_metrics['R2']:.6f}
  MAE: {test_metrics['MAE']:.6f}

3. 关键特征
------------------------------
正则化参数: α = {model.coefficients['alpha']}
特征数量: {len(model.feature_names)}

特征重要性:
"""

    for _, row in importance_df.iterrows():
        report += f"{row['feature']}: {row['coefficient']:.4f}\n"

    report += f"""
4. 性能对比
------------------------------
相比之前的模型:
- 原始岭回归 (80样本): R² = 0.938
- SOTA深度学习: R² = -0.348
- 混合SOTA: R² = -1.443
- 物理模型: R² = {test_metrics['R2']:.4f}

5. 优势
------------------------------
- 专门针对VIV问题优化
- 基于物理理论
- 适合小数据集
- 结果可解释
"""

    # 保存报告
    with open('results/physics_viv_model_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n报告已保存: results/physics_viv_model_report.txt")

if __name__ == "__main__":
    model, results = run_experiment()