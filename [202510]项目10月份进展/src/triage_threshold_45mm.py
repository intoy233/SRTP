#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路线B实验: 降低高风险阈值至45mm
Triage-Expert System with Lowered Threshold (45mm)

目标: 通过降低阈值增加高风险样本数,改善专家模型性能
当前: >60mm (51座, 26.0%) → 新阈值: >45mm (102座, 52.0%)
样本/特征比: 0.65 → 1.31 (+101%)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')


class TriageSystem45mm:
    """阈值=45mm的分诊-专家系统"""

    def __init__(self, data_path, threshold=45.0):
        self.data_path = data_path
        self.HIGH_RISK_THRESHOLD = threshold
        self.df = None
        self.X = None
        self.y_amplitude = None
        self.y_risk_labels = None
        self.feature_names = None

    def load_and_prepare_data(self):
        """加载并准备数据"""
        print("="*80)
        print(f"路线B: 降低高风险阈值实验 (阈值={self.HIGH_RISK_THRESHOLD}mm)")
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

        # 目标变量
        self.y_amplitude = self.df.loc[df_features.index, 'Max_Amplitude_mm'].values

        # 幂函数变换
        X_base = df_features.values
        X_squared = X_base ** 2
        X_cubed = X_base ** 3
        self.X = np.hstack([X_base, X_squared, X_cubed])

        self.feature_names = (
            list(df_features.columns) +
            [f"{col}_squared" for col in df_features.columns] +
            [f"{col}_cubed" for col in df_features.columns]
        )

        # 风险标签
        self.y_risk_labels = np.where(
            self.y_amplitude > self.HIGH_RISK_THRESHOLD,
            'high',
            'normal'
        )

        print(f"\n特征维度: {len(self.feature_names)} (含幂函数变换)")
        print(f"样本数: {len(self.X)}")

        # 风险分布
        high_count = (self.y_risk_labels == 'high').sum()
        normal_count = (self.y_risk_labels == 'normal').sum()

        print(f"\n风险分布 (阈值={self.HIGH_RISK_THRESHOLD}mm):")
        print(f"  高风险 (>{self.HIGH_RISK_THRESHOLD}mm): {high_count} ({high_count/len(self.X)*100:.1f}%)")
        print(f"  常规风险 (<={self.HIGH_RISK_THRESHOLD}mm): {normal_count} ({normal_count/len(self.X)*100:.1f}%)")
        print(f"  样本/特征比 (高风险): {high_count}/{len(self.feature_names)} = {high_count/len(self.feature_names):.2f}")

        return self.X, self.y_amplitude, self.y_risk_labels

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

    def evaluate_triage_expert(self, k=5):
        """5-Fold交叉验证评估分诊-专家系统"""
        print("\n" + "="*80)
        print(f"K-Fold交叉验证 (k={k}) - 阈值={self.HIGH_RISK_THRESHOLD}mm")
        print("="*80)

        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        results = {'overall': [], 'high_risk': [], 'triage': []}

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X), 1):
            print(f"\n{'='*80}")
            print(f"Fold {fold}/{k}")
            print(f"{'='*80}")

            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_amp_train, y_amp_val = self.y_amplitude[train_idx], self.y_amplitude[val_idx]
            y_risk_train, y_risk_val = self.y_risk_labels[train_idx], self.y_risk_labels[val_idx]

            # 步骤1: 训练分诊分类器
            scaler_triage = StandardScaler()
            X_train_scaled = scaler_triage.fit_transform(X_train)
            X_val_scaled = scaler_triage.transform(X_val)

            triage_clf = GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42
            )
            triage_clf.fit(X_train_scaled, y_risk_train)
            risk_pred = triage_clf.predict(X_val_scaled)
            triage_acc = np.mean(risk_pred == y_risk_val)

            print(f"分诊准确率: {triage_acc*100:.1f}%")

            # 步骤2: 训练专家回归模型
            expert_models = {}
            expert_scalers = {}

            for risk_level in ['normal', 'high']:
                mask = (y_risk_train == risk_level)
                X_expert = X_train[mask]
                y_expert = y_amp_train[mask]

                if len(X_expert) == 0:
                    continue

                print(f"训练'{risk_level}'专家: {len(X_expert)}个样本, 振幅范围{y_expert.min():.1f}-{y_expert.max():.1f}mm")

                scaler_expert = StandardScaler()
                X_expert_scaled = scaler_expert.fit_transform(X_expert)

                model = BayesianRidge(n_iter=300, tol=1e-3)
                model.fit(X_expert_scaled, y_expert)

                expert_models[risk_level] = model
                expert_scalers[risk_level] = scaler_expert

            # 步骤3: 预测
            y_pred = np.zeros(len(X_val))
            for i, risk_level in enumerate(risk_pred):
                if risk_level in expert_models:
                    X_sample_scaled = expert_scalers[risk_level].transform(X_val[i:i+1])
                    y_pred[i] = expert_models[risk_level].predict(X_sample_scaled)[0]
                else:
                    y_pred[i] = y_amp_train.mean()

            # 评估
            overall_r2 = 1 - np.sum((y_amp_val - y_pred)**2) / np.sum((y_amp_val - np.mean(y_amp_val))**2)
            overall_rmse = np.sqrt(np.mean((y_amp_val - y_pred)**2))

            high_risk_mask = y_amp_val > self.HIGH_RISK_THRESHOLD
            if high_risk_mask.sum() > 0:
                high_r2 = 1 - np.sum((y_amp_val[high_risk_mask] - y_pred[high_risk_mask])**2) / \
                          np.sum((y_amp_val[high_risk_mask] - np.mean(y_amp_val[high_risk_mask]))**2)
                high_rmse = np.sqrt(np.mean((y_amp_val[high_risk_mask] - y_pred[high_risk_mask])**2))
            else:
                high_r2 = high_rmse = np.nan

            results['overall'].append({'r2': overall_r2, 'rmse': overall_rmse})
            results['high_risk'].append({'r2': high_r2, 'rmse': high_rmse})
            results['triage'].append({'accuracy': triage_acc})

            print(f"整体R2={overall_r2:.4f}, RMSE={overall_rmse:.2f}mm", end="")
            if not np.isnan(high_r2):
                print(f", 高风险R2={high_r2:.4f}, RMSE={high_rmse:.2f}mm")
            else:
                print()

        # 汇总
        print("\n" + "="*80)
        print("性能汇总")
        print("="*80)

        triage_acc = np.mean([r['accuracy'] for r in results['triage']])
        overall_r2 = np.mean([r['r2'] for r in results['overall']])
        overall_rmse = np.mean([r['rmse'] for r in results['overall']])
        high_r2_valid = [r['r2'] for r in results['high_risk'] if not np.isnan(r['r2'])]
        high_rmse_valid = [r['rmse'] for r in results['high_risk'] if not np.isnan(r['rmse'])]

        print(f"\n分诊准确率: {triage_acc*100:.1f}%")
        print(f"整体回归: R2={overall_r2:.4f}, RMSE={overall_rmse:.2f}mm")
        if high_r2_valid:
            print(f"高风险样本(>{self.HIGH_RISK_THRESHOLD}mm): R2={np.mean(high_r2_valid):.4f}, RMSE={np.mean(high_rmse_valid):.2f}mm")

        return results


def compare_thresholds():
    """对比不同阈值的性能"""
    print("="*80)
    print("路线B: 阈值对比实验")
    print("="*80)

    thresholds_to_test = [45, 50, 55, 60]
    all_results = {}

    for threshold in thresholds_to_test:
        print(f"\n{'#'*80}")
        print(f"# 测试阈值: {threshold}mm")
        print(f"{'#'*80}")

        system = TriageSystem45mm('../data/final_bridge_dataset.csv', threshold=threshold)
        system.load_and_prepare_data()
        results = system.evaluate_triage_expert(k=5)
        all_results[threshold] = results

    # 最终对比
    print("\n" + "="*80)
    print("阈值对比总结")
    print("="*80)

    print(f"\n{'阈值(mm)':<12} {'整体R2':<12} {'整体RMSE':<12} {'高风险R2':<12} {'高风险RMSE':<12}")
    print("-"*80)

    for threshold, results in all_results.items():
        overall_r2 = np.mean([r['r2'] for r in results['overall']])
        overall_rmse = np.mean([r['rmse'] for r in results['overall']])
        high_r2_valid = [r['r2'] for r in results['high_risk'] if not np.isnan(r['r2'])]
        high_rmse_valid = [r['rmse'] for r in results['high_risk'] if not np.isnan(r['rmse'])]

        if high_r2_valid:
            high_r2 = np.mean(high_r2_valid)
            high_rmse = np.mean(high_rmse_valid)
            print(f">{threshold:<11} {overall_r2:.4f}      {overall_rmse:.2f} mm    {high_r2:.4f}      {high_rmse:.2f} mm")
        else:
            print(f">{threshold:<11} {overall_r2:.4f}      {overall_rmse:.2f} mm    N/A          N/A")

    # 与基线对比
    print("\n" + "="*80)
    print("与当前最佳模型对比")
    print("="*80)
    print("\n当前最佳模型(幂函数变换+贝叶斯岭):")
    print("  整体R2: 0.5920, RMSE: 13.65mm")

    print("\n结论:")
    best_threshold = max(all_results.keys(), key=lambda t: np.mean([r['r2'] for r in all_results[t]['overall']]))
    best_r2 = np.mean([r['r2'] for r in all_results[best_threshold]['overall']])

    if best_r2 > 0.5920:
        print(f"  [OK] 阈值{best_threshold}mm性能超越基线! (R2={best_r2:.4f} > 0.5920)")
    else:
        print(f"  [X] 所有阈值均未超越基线 (最佳R2={best_r2:.4f} < 0.5920)")
        print(f"  [!] 建议: 放弃分诊系统,继续使用当前最佳模型")

    return all_results


if __name__ == '__main__':
    results = compare_thresholds()
