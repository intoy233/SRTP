#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路线C - 实验: Boosting集成优化
Route C - Boosting Ensemble Optimization

目标: 通过XGBoost/LightGBM/CatBoost超越当前最佳模型(R2=0.5920)
策略: 梯度提升 + 超参数优化 + 交叉验证
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')


class BoostingEnsemble:
    """Boosting集成模型"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None

    def load_and_prepare_data(self):
        """加载并准备数据(复用幂函数特征)"""
        print("="*80)
        print("路线C: Boosting集成优化")
        print("="*80)
        print("目标: R2 > 0.63 (超越当前最佳0.5920)")
        print("="*80)

        self.df = pd.read_csv(self.data_path)
        print(f"\n数据集: {len(self.df)} 座桥梁")

        # 排除列
        exclude_cols = [
            'BridgeName', 'Country', 'BridgeType', 'PaperSource', 'Year',
            'Max_Amplitude_mm',
            'Amplitude_RMS_mm', 'VIV_Wind_Speed_ms',
            'Risk_Level', 'Notes', 'Vibration_Suppression', 'Suppression_Effect',
            'Total_Length_m', 'First_Freq_Hz', 'Second_Freq_Hz',
            'Drag_Coefficient', 'Lift_Coefficient',
            'BridgeID', 'Structure_Type'
        ]

        actual_exclude = [col for col in exclude_cols if col in self.df.columns]
        feature_cols = [col for col in self.df.columns if col not in actual_exclude]

        # 创建基础+Griffin特征
        df_features = self._create_all_features(feature_cols)
        df_features = df_features.dropna()

        # 目标变量
        self.y = self.df.loc[df_features.index, 'Max_Amplitude_mm'].values

        # 幂函数变换 (26 -> 78维)
        X_base = df_features.values
        X_squared = X_base ** 2
        X_cubed = X_base ** 3
        self.X = np.hstack([X_base, X_squared, X_cubed])

        self.feature_names = (
            list(df_features.columns) +
            [f"{col}_sq" for col in df_features.columns] +
            [f"{col}_cu" for col in df_features.columns]
        )

        print(f"\n特征维度: {len(self.feature_names)} (含幂函数变换)")
        print(f"样本数: {len(self.X)}")
        print(f"目标变量范围: {self.y.min():.1f} - {self.y.max():.1f} mm")

        return self.X, self.y

    def _create_all_features(self, feature_cols):
        """创建基础+交互+Griffin特征(26维)"""
        df_features = self.df[feature_cols].copy()

        # 基础物理特征
        if all(col in self.df.columns for col in ['Damping_Ratio', 'Width_m', 'Height_m']):
            df_features['Scruton_Number'] = (
                self.df['Damping_Ratio'] * (self.df['Width_m'] / self.df['Height_m']) * 100
            )

        if all(col in self.df.columns for col in ['Width_m', 'Height_m']):
            df_features['Aspect_Ratio'] = self.df['Width_m'] / self.df['Height_m']

        if 'Damping_Ratio' in self.df.columns:
            df_features['VIV_Susceptibility'] = 1.0 / (self.df['Damping_Ratio'] + 1e-6)

        if all(col in self.df.columns for col in ['Critical_Wind_Speed_ms', 'Natural_Freq_Hz', 'Width_m']):
            df_features['Reduced_Velocity'] = (
                self.df['Critical_Wind_Speed_ms'] / (self.df['Natural_Freq_Hz'] * self.df['Width_m'])
            )
            median_vr = df_features['Reduced_Velocity'].median()
            df_features['Reduced_Velocity'].fillna(median_vr, inplace=True)

        if all(col in self.df.columns for col in ['Natural_Freq_Hz', 'Span_m']):
            df_features['Stiffness_Parameter'] = self.df['Natural_Freq_Hz'] * np.sqrt(self.df['Span_m'])

        # 交互特征
        if all(col in df_features.columns for col in ['Damping_Ratio', 'Span_m']):
            df_features['Damping_x_Span'] = df_features['Damping_Ratio'] * df_features['Span_m']

        if all(col in df_features.columns for col in ['Natural_Freq_Hz', 'Width_m']):
            df_features['Freq_x_Width'] = df_features['Natural_Freq_Hz'] * df_features['Width_m']

        if all(col in df_features.columns for col in ['Scruton_Number', 'Reduced_Velocity']):
            df_features['Scruton_x_ReVel'] = df_features['Scruton_Number'] * df_features['Reduced_Velocity']

        if all(col in df_features.columns for col in ['Damping_Ratio', 'Critical_Wind_Speed_ms']):
            df_features['Damping_x_WindSpeed'] = df_features['Damping_Ratio'] * df_features['Critical_Wind_Speed_ms']

        if 'Damping_Ratio' in df_features.columns:
            df_features['Damping_squared'] = df_features['Damping_Ratio'] ** 2

        if 'Span_m' in df_features.columns:
            df_features['Span_sqrt'] = np.sqrt(df_features['Span_m'])

        if 'Aspect_Ratio' in df_features.columns:
            df_features['Aspect_Ratio_squared'] = df_features['Aspect_Ratio'] ** 2

        if all(col in df_features.columns for col in ['Stiffness_Parameter', 'Damping_Ratio']):
            df_features['Stiffness_Damping_Ratio'] = df_features['Stiffness_Parameter'] / (df_features['Damping_Ratio'] + 1e-6)

        # Griffin Plot特征
        if 'Reduced_Velocity' in df_features.columns:
            Vr = df_features['Reduced_Velocity']
            VR_LOCK_IN_START = 4.0
            VR_LOCK_IN_END = 8.0
            VR_LOCK_IN_CENTER = 6.0

            df_features['is_in_lock_in_region'] = (
                (Vr >= VR_LOCK_IN_START) & (Vr <= VR_LOCK_IN_END)
            ).astype(float)

            df_features['distance_to_lock_in_center'] = np.abs(Vr - VR_LOCK_IN_CENTER)

            sigma = 2.0
            df_features['vr_lock_in_response'] = np.exp(
                -((Vr - VR_LOCK_IN_CENTER) / sigma) ** 2
            )

            if 'Scruton_Number' in df_features.columns:
                df_features['Scruton_in_lock_in'] = (
                    df_features['Scruton_Number'] * df_features['is_in_lock_in_region']
                )

            def viv_branch(vr):
                if vr < VR_LOCK_IN_START:
                    return 0
                elif vr <= VR_LOCK_IN_END:
                    return 1
                else:
                    return 2

            df_features['viv_branch'] = Vr.apply(viv_branch)

            df_features['lock_in_depth'] = np.where(
                df_features['is_in_lock_in_region'] == 1,
                1.0 - df_features['distance_to_lock_in_center'] / (VR_LOCK_IN_END - VR_LOCK_IN_START),
                0.0
            )

        return df_features

    def evaluate_xgboost(self, k=5):
        """实验1: XGBoost回归"""
        print("\n" + "="*80)
        print("实验1: XGBoost回归")
        print("="*80)

        # 超参数网格(保守策略,防止过拟合)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_alpha': [0, 0.1, 1.0],  # L1正则化
            'reg_lambda': [1, 5, 10]  # L2正则化
        }

        print(f"\n超参数搜索空间: {len(param_grid['n_estimators']) * len(param_grid['learning_rate']) * len(param_grid['max_depth']) * len(param_grid['subsample']) * len(param_grid['colsample_bytree']) * len(param_grid['reg_alpha']) * len(param_grid['reg_lambda'])} 组合")
        print("使用RandomizedSearchCV加速搜索...")

        # 使用RandomizedSearchCV加速
        from sklearn.model_selection import RandomizedSearchCV

        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )

        random_search = RandomizedSearchCV(
            xgb_model,
            param_distributions=param_grid,
            n_iter=50,  # 随机采样50组
            cv=5,
            scoring='r2',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )

        random_search.fit(self.X, self.y)

        print(f"\n最佳超参数:")
        for param, value in random_search.best_params_.items():
            print(f"  {param}: {value}")

        # 5-Fold交叉验证评估
        best_model = random_search.best_estimator_
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X), 1):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            model = xgb.XGBRegressor(**random_search.best_params_, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            r2 = r2_score(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))

            results.append({'r2': r2, 'rmse': rmse})
            print(f"Fold {fold}: R2={r2:.4f}, RMSE={rmse:.2f}mm")

        avg_r2 = np.mean([r['r2'] for r in results])
        avg_rmse = np.mean([r['rmse'] for r in results])
        std_r2 = np.std([r['r2'] for r in results])

        print(f"\n汇总: R2={avg_r2:.4f} (+/- {std_r2:.4f}), RMSE={avg_rmse:.2f}mm")

        # 与基线对比
        baseline_r2 = 0.5920
        improvement = (avg_r2 - baseline_r2) / baseline_r2 * 100

        if avg_r2 > baseline_r2:
            print(f"[OK] 超越基线! 相对提升: {improvement:.1f}%")
        else:
            print(f"[!] 未超越基线 (差距: {(baseline_r2 - avg_r2):.4f})")

        return results, random_search.best_params_

    def evaluate_lightgbm(self, k=5):
        """实验2: LightGBM回归"""
        print("\n" + "="*80)
        print("实验2: LightGBM回归")
        print("="*80)

        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 50, 100],
            'max_depth': [-1, 10, 20],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_alpha': [0, 0.1, 1.0],
            'reg_lambda': [1, 5, 10]
        }

        from sklearn.model_selection import RandomizedSearchCV

        lgb_model = lgb.LGBMRegressor(
            objective='regression',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

        random_search = RandomizedSearchCV(
            lgb_model,
            param_distributions=param_grid,
            n_iter=50,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )

        random_search.fit(self.X, self.y)

        print(f"\n最佳超参数:")
        for param, value in random_search.best_params_.items():
            print(f"  {param}: {value}")

        # 5-Fold交叉验证
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X), 1):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            model = lgb.LGBMRegressor(**random_search.best_params_, random_state=42, n_jobs=-1, verbose=-1)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            r2 = r2_score(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))

            results.append({'r2': r2, 'rmse': rmse})
            print(f"Fold {fold}: R2={r2:.4f}, RMSE={rmse:.2f}mm")

        avg_r2 = np.mean([r['r2'] for r in results])
        avg_rmse = np.mean([r['rmse'] for r in results])
        std_r2 = np.std([r['r2'] for r in results])

        print(f"\n汇总: R2={avg_r2:.4f} (+/- {std_r2:.4f}), RMSE={avg_rmse:.2f}mm")

        baseline_r2 = 0.5920
        improvement = (avg_r2 - baseline_r2) / baseline_r2 * 100

        if avg_r2 > baseline_r2:
            print(f"[OK] 超越基线! 相对提升: {improvement:.1f}%")
        else:
            print(f"[!] 未超越基线 (差距: {(baseline_r2 - avg_r2):.4f})")

        return results, random_search.best_params_

    def evaluate_catboost(self, k=5):
        """实验3: CatBoost回归"""
        print("\n" + "="*80)
        print("实验3: CatBoost回归")
        print("="*80)

        param_grid = {
            'iterations': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6, 8],
            'l2_leaf_reg': [1, 3, 5, 10],
            'subsample': [0.8, 1.0]
        }

        from sklearn.model_selection import RandomizedSearchCV

        cat_model = CatBoostRegressor(
            loss_function='RMSE',
            random_state=42,
            verbose=0
        )

        random_search = RandomizedSearchCV(
            cat_model,
            param_distributions=param_grid,
            n_iter=30,  # CatBoost慢,减少搜索次数
            cv=5,
            scoring='r2',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )

        random_search.fit(self.X, self.y)

        print(f"\n最佳超参数:")
        for param, value in random_search.best_params_.items():
            print(f"  {param}: {value}")

        # 5-Fold交叉验证
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X), 1):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            model = CatBoostRegressor(**random_search.best_params_, random_state=42, verbose=0)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            r2 = r2_score(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))

            results.append({'r2': r2, 'rmse': rmse})
            print(f"Fold {fold}: R2={r2:.4f}, RMSE={rmse:.2f}mm")

        avg_r2 = np.mean([r['r2'] for r in results])
        avg_rmse = np.mean([r['rmse'] for r in results])
        std_r2 = np.std([r['r2'] for r in results])

        print(f"\n汇总: R2={avg_r2:.4f} (+/- {std_r2:.4f}), RMSE={avg_rmse:.2f}mm")

        baseline_r2 = 0.5920
        improvement = (avg_r2 - baseline_r2) / baseline_r2 * 100

        if avg_r2 > baseline_r2:
            print(f"[OK] 超越基线! 相对提升: {improvement:.1f}%")
        else:
            print(f"[!] 未超越基线 (差距: {(baseline_r2 - avg_r2):.4f})")

        return results, random_search.best_params_


def main():
    print("="*80)
    print("路线C: Boosting集成优化实验")
    print("="*80)
    print("当前最佳: Griffin Plot + 幂函数变换 + 贝叶斯岭回归 (R2=0.5920)")
    print("目标: R2 > 0.63 (+6.4%)")
    print("="*80)

    # 初始化
    ensemble = BoostingEnsemble('../data/final_bridge_dataset.csv')
    X, y = ensemble.load_and_prepare_data()

    # 实验1: XGBoost
    print("\n" + "#"*80)
    print("# 实验1: XGBoost")
    print("#"*80)
    xgb_results, xgb_params = ensemble.evaluate_xgboost(k=5)

    # 实验2: LightGBM
    print("\n" + "#"*80)
    print("# 实验2: LightGBM")
    print("#"*80)
    lgb_results, lgb_params = ensemble.evaluate_lightgbm(k=5)

    # 实验3: CatBoost
    print("\n" + "#"*80)
    print("# 实验3: CatBoost")
    print("#"*80)
    cat_results, cat_params = ensemble.evaluate_catboost(k=5)

    # 最终对比
    print("\n" + "="*80)
    print("Boosting集成性能对比")
    print("="*80)

    models = [
        ('Baseline (Bayesian Ridge)', {'r2': 0.5920, 'rmse': 13.65}),
        ('XGBoost', {'r2': np.mean([r['r2'] for r in xgb_results]), 'rmse': np.mean([r['rmse'] for r in xgb_results])}),
        ('LightGBM', {'r2': np.mean([r['r2'] for r in lgb_results]), 'rmse': np.mean([r['rmse'] for r in lgb_results])}),
        ('CatBoost', {'r2': np.mean([r['r2'] for r in cat_results]), 'rmse': np.mean([r['rmse'] for r in cat_results])})
    ]

    print(f"\n{'Model':<25} {'R2':<12} {'RMSE':<12} {'vs Baseline':<15}")
    print("-"*80)

    for name, result in models:
        r2 = result['r2']
        rmse = result['rmse']
        improvement = (r2 - 0.5920) / 0.5920 * 100

        status = "[OK]" if r2 > 0.5920 else "[-]"
        print(f"{name:<25} {r2:.4f}      {rmse:.2f} mm    {status} {improvement:+.1f}%")

    # 选择最佳
    best_model = max(models[1:], key=lambda x: x[1]['r2'])
    print(f"\n最佳模型: {best_model[0]} (R2={best_model[1]['r2']:.4f})")

    if best_model[1]['r2'] > 0.63:
        print("[OK][OK] 达成目标! R2 > 0.63")
    elif best_model[1]['r2'] > 0.5920:
        print("[OK] 超越基线,但未达目标 (0.5920 < R2 < 0.63)")
    else:
        print("[X] 未超越基线,需要尝试其他方法 (Stacking/深度学习)")

    print("\n" + "="*80)
    print("实验完成!")
    print("="*80)

    return xgb_results, lgb_results, cat_results


if __name__ == '__main__':
    xgb_res, lgb_res, cat_res = main()
