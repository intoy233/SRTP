#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路线C - 最终实验: Stacking集成
Route C - Final Experiment: Stacking Ensemble

策略: 多模型融合,取长补短
架构: 5个基学习器 + 1个元学习器(贝叶斯岭,保留不确定性)
目标: R² > 0.63
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class StackingEnsemble:
    """Stacking集成模型"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None

        # Level 0 基学习器
        self.base_models = None
        # Level 1 元学习器
        self.meta_model = None
        self.scaler = StandardScaler()

    def load_and_prepare_data(self):
        """加载并准备数据"""
        print("="*80)
        print("路线C - 最终实验: Stacking集成")
        print("="*80)
        print("架构: Ridge + Lasso + RandomForest + SVR + BayesianRidge (基学习器)")
        print("      -> BayesianRidge (元学习器,保留不确定性)")
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

        # 创建特征
        df_features = self._create_all_features(feature_cols)
        df_features = df_features.dropna()

        self.y = self.df.loc[df_features.index, 'Max_Amplitude_mm'].values

        # 幂函数变换
        X_base = df_features.values
        X_squared = X_base ** 2
        X_cubed = X_base ** 3
        self.X = np.hstack([X_base, X_squared, X_cubed])

        self.feature_names = (
            list(df_features.columns) +
            [f"{col}_sq" for col in df_features.columns] +
            [f"{col}_cu" for col in df_features.columns]
        )

        print(f"\n特征维度: {len(self.feature_names)}")
        print(f"样本数: {len(self.X)}")

        return self.X, self.y

    def _create_all_features(self, feature_cols):
        """创建基础+交互+Griffin特征"""
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

    def build_stacking_model(self):
        """构建Stacking模型"""
        print("\n构建Stacking架构:")

        # Level 0: 5个基学习器 (多样性)
        self.base_models = [
            ('Ridge', Ridge(alpha=10.0)),
            ('Lasso', Lasso(alpha=0.1, max_iter=5000)),
            ('RandomForest', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
            ('SVR_RBF', SVR(kernel='rbf', C=10, gamma='scale')),
            ('BayesianRidge', BayesianRidge(max_iter=300, tol=1e-3))
        ]

        # Level 1: 元学习器 (贝叶斯岭,带不确定性)
        self.meta_model = BayesianRidge(max_iter=300, tol=1e-3)

        print("  Level 0 (基学习器):")
        for name, model in self.base_models:
            print(f"    - {name}")

        print("  Level 1 (元学习器):")
        print(f"    - BayesianRidge (保留不确定性量化)")

        return self.base_models, self.meta_model

    def evaluate_stacking(self, k=5):
        """5-Fold交叉验证评估Stacking"""
        print("\n" + "="*80)
        print(f"K-Fold交叉验证 (k={k})")
        print("="*80)

        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X), 1):
            print(f"\nFold {fold}/{k}")
            print("-"*80)

            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            # 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Step 1: 训练基学习器,生成元特征
            print("训练基学习器...")
            meta_features_train = np.zeros((len(X_train), len(self.base_models)))
            meta_features_val = np.zeros((len(X_val), len(self.base_models)))

            for i, (name, model) in enumerate(self.base_models):
                # 使用交叉验证生成训练集的元特征(避免数据泄露)
                kf_inner = KFold(n_splits=3, shuffle=True, random_state=42)
                meta_features_train[:, i] = cross_val_predict(
                    model, X_train_scaled, y_train, cv=kf_inner
                )

                # 训练完整模型并预测验证集
                model.fit(X_train_scaled, y_train)
                meta_features_val[:, i] = model.predict(X_val_scaled)

                # 基学习器性能
                train_pred = meta_features_train[:, i]
                train_r2 = r2_score(y_train, train_pred)
                val_pred = meta_features_val[:, i]
                val_r2 = r2_score(y_val, val_pred)

                print(f"  {name:<15}: Train R2={train_r2:.4f}, Val R2={val_r2:.4f}")

            # Step 2: 训练元学习器
            print("\n训练元学习器...")
            self.meta_model.fit(meta_features_train, y_train)

            # Step 3: 预测
            y_pred, y_std = self.meta_model.predict(meta_features_val, return_std=True)

            # 评估
            r2 = r2_score(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mean_std = np.mean(y_std)

            results.append({'r2': r2, 'rmse': rmse, 'mean_std': mean_std})

            print(f"\nStacking性能: R2={r2:.4f}, RMSE={rmse:.2f}mm, 平均不确定性={mean_std:.2f}mm")

        # 汇总
        print("\n" + "="*80)
        print("Stacking性能汇总")
        print("="*80)

        avg_r2 = np.mean([r['r2'] for r in results])
        avg_rmse = np.mean([r['rmse'] for r in results])
        std_r2 = np.std([r['r2'] for r in results])
        avg_uncertainty = np.mean([r['mean_std'] for r in results])

        print(f"\n验证集R2: {avg_r2:.4f} (+/- {std_r2:.4f})")
        print(f"验证集RMSE: {avg_rmse:.2f} mm")
        print(f"平均不确定性: {avg_uncertainty:.2f} mm")

        # 与基线对比
        baseline_r2 = 0.5920
        baseline_rmse = 13.65
        improvement = (avg_r2 - baseline_r2) / baseline_r2 * 100

        print(f"\n当前最佳基线: R2={baseline_r2:.4f}, RMSE={baseline_rmse:.2f}mm")

        if avg_r2 > baseline_r2:
            print(f"[OK] Stacking超越基线! 相对提升: {improvement:.1f}%")
            if avg_r2 > 0.63:
                print(f"[OK][OK] 达成目标! R2={avg_r2:.4f} > 0.63")
            else:
                print(f"[!] 超越基线但未达目标 ({avg_r2:.4f} < 0.63)")
        else:
            print(f"[X] Stacking未超越基线 (差距: {(baseline_r2 - avg_r2):.4f})")

        return results

    def compare_all_models(self, stacking_results):
        """对比所有路线C的模型"""
        print("\n" + "="*80)
        print("路线C所有模型性能对比")
        print("="*80)

        models = [
            ('Baseline (Bayesian Ridge)', 0.5920, 13.65),
            ('XGBoost', 0.5416, 14.26),
            ('LightGBM', 0.5392, 14.61),
            ('CatBoost', 0.5544, 14.21),
            ('Stacking', np.mean([r['r2'] for r in stacking_results]), np.mean([r['rmse'] for r in stacking_results]))
        ]

        print(f"\n{'Model':<30} {'R2':<12} {'RMSE':<12} {'vs Baseline':<15}")
        print("-"*80)

        for name, r2, rmse in models:
            improvement = (r2 - 0.5920) / 0.5920 * 100
            status = "[OK]" if r2 > 0.5920 else "[-]"
            print(f"{name:<30} {r2:.4f}      {rmse:.2f} mm    {status} {improvement:+.1f}%")

        # 最佳模型
        best_model = max(models, key=lambda x: x[1])
        print(f"\n最佳模型: {best_model[0]} (R2={best_model[1]:.4f}, RMSE={best_model[2]:.2f}mm)")

        return models


def main():
    print("="*80)
    print("路线C - 最终实验: Stacking集成")
    print("="*80)
    print("这是路线C的最后一搏!")
    print("如果Stacking失败,则R2=0.5920就是数据的极限")
    print("="*80)

    # 初始化
    ensemble = StackingEnsemble('../data/final_bridge_dataset.csv')
    X, y = ensemble.load_and_prepare_data()

    # 构建Stacking
    ensemble.build_stacking_model()

    # 评估
    stacking_results = ensemble.evaluate_stacking(k=5)

    # 对比所有模型
    all_models = ensemble.compare_all_models(stacking_results)

    # 最终结论
    print("\n" + "="*80)
    print("路线C实验结论")
    print("="*80)

    stacking_r2 = np.mean([r['r2'] for r in stacking_results])

    if stacking_r2 > 0.63:
        print("\n[OK][OK] 成功! Stacking达成目标 (R2 > 0.63)")
        print("建议: 采用Stacking作为最终模型")
    elif stacking_r2 > 0.5920:
        print("\n[OK] Stacking超越基线,但未达目标")
        print(f"R2={stacking_r2:.4f} (基线0.5920, 目标0.63)")
        print("建议: 根据提升幅度决定是否采用Stacking")
    else:
        print("\n[X] 路线C失败! 所有方法均未超越基线")
        print(f"Stacking R2={stacking_r2:.4f} < 基线0.5920")
        print("\n结论: R2=0.5920 (幂函数变换+贝叶斯岭) 已是数据的极限")
        print("原因: 样本量不足(190) + 特征维度高(78) → 无法支撑更复杂模型")
        print("\n最终建议:")
        print("  1. 接受当前最佳模型 (R2=0.5920)")
        print("  2. 收集更多数据 (目标300+样本)")
        print("  3. 特征选择降维 (78→30-40维)")

    print("\n" + "="*80)
    print("实验完成!")
    print("="*80)

    return stacking_results, all_models


if __name__ == '__main__':
    stacking_res, all_models = main()
