#!/usr/bin/env python3
"""
改进方案: B(简化特征分诊) + C(加权回归)
Improved Solutions: Simplified Triage + Weighted Regression

方案B: 简化特征分诊系统
- 使用26维基础特征(不含幂函数变换)
- 避免高维导致的过拟合
- 样本/特征比更健康: 40/26 ≈ 1.5

方案C: 加权回归
- 单一模型,但对高风险样本赋予更高权重
- sample_weight: 高风险×3, 常规×1
- 预期: 高风险预测提升,整体R2略降(可接受)

对比实验:
1. 基线: Griffin(26特征) + 贝叶斯岭回归
2. 方案B: 简化特征分诊系统
3. 方案C: 加权回归
4. 当前最佳: Griffin+幂函数(78特征) + 贝叶斯岭回归
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')


class ImprovedVIVPredictor:
    """改进的VIV预测器"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_base = None  # 26维基础特征
        self.X_power = None  # 78维幂函数特征
        self.y = None
        self.feature_names = None

    def load_and_prepare_data(self):
        """加载并准备数据"""
        print("="*80)
        print("改进方案对比实验: B(简化特征分诊) + C(加权回归)")
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

        # 创建基础特征
        df_features = self._create_all_features(feature_cols)
        df_features = df_features.dropna()

        # 目标变量
        self.y = self.df.loc[df_features.index, 'Max_Amplitude_mm'].values

        # 26维基础特征
        self.X_base = df_features.values
        self.feature_names = df_features.columns.tolist()

        # 78维幂函数特征
        X_squared = self.X_base ** 2
        X_cubed = self.X_base ** 3
        self.X_power = np.hstack([self.X_base, X_squared, X_cubed])

        print(f"\n基础特征: {self.X_base.shape[1]} 维, {len(self.X_base)} 个样本")
        print(f"幂函数特征: {self.X_power.shape[1]} 维")
        print(f"目标变量范围: {self.y.min():.1f} - {self.y.max():.1f} mm")

        return self.X_base, self.X_power, self.y

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

    def evaluate_baseline(self, k=5):
        """基线: Griffin(26特征) + 贝叶斯岭回归"""
        print("\n" + "="*80)
        print("基线模型: Griffin(26特征) + 贝叶斯岭回归")
        print("="*80)

        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        results = {'overall': [], 'high_risk': []}

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_base), 1):
            X_train, X_val = self.X_base[train_idx], self.X_base[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            # 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # 贝叶斯岭回归
            model = BayesianRidge(n_iter=300, tol=1e-3)
            model.fit(X_train_scaled, y_train)

            # 预测
            y_pred, y_std = model.predict(X_val_scaled, return_std=True)

            # 整体性能
            overall_r2 = 1 - np.sum((y_val - y_pred)**2) / np.sum((y_val - np.mean(y_val))**2)
            overall_rmse = np.sqrt(np.mean((y_val - y_pred)**2))

            # 高风险样本性能
            high_risk_mask = y_val > 60
            if high_risk_mask.sum() > 0:
                high_r2 = 1 - np.sum((y_val[high_risk_mask] - y_pred[high_risk_mask])**2) / \
                          np.sum((y_val[high_risk_mask] - np.mean(y_val[high_risk_mask]))**2)
                high_rmse = np.sqrt(np.mean((y_val[high_risk_mask] - y_pred[high_risk_mask])**2))
            else:
                high_r2 = high_rmse = np.nan

            results['overall'].append({'r2': overall_r2, 'rmse': overall_rmse})
            results['high_risk'].append({'r2': high_r2, 'rmse': high_rmse})

            print(f"Fold {fold}: 整体R2={overall_r2:.4f}, RMSE={overall_rmse:.2f}mm", end="")
            if not np.isnan(high_r2):
                print(f", 高风险R2={high_r2:.4f}, RMSE={high_rmse:.2f}mm")
            else:
                print()

        # 汇总
        overall_r2 = np.mean([r['r2'] for r in results['overall']])
        overall_rmse = np.mean([r['rmse'] for r in results['overall']])
        high_r2_valid = [r['r2'] for r in results['high_risk'] if not np.isnan(r['r2'])]
        high_rmse_valid = [r['rmse'] for r in results['high_risk'] if not np.isnan(r['rmse'])]

        print(f"\n汇总: 整体R2={overall_r2:.4f}, RMSE={overall_rmse:.2f}mm")
        if high_r2_valid:
            print(f"      高风险R2={np.mean(high_r2_valid):.4f}, RMSE={np.mean(high_rmse_valid):.2f}mm")

        return results

    def evaluate_simplified_triage(self, k=5):
        """方案B: 简化特征(26维)分诊-专家系统"""
        print("\n" + "="*80)
        print("方案B: 简化特征分诊-专家系统(26维,二元)")
        print("="*80)

        HIGH_RISK_THRESHOLD = 60.0
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        results = {'overall': [], 'high_risk': [], 'triage': []}

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_base), 1):
            X_train, X_val = self.X_base[train_idx], self.X_base[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            # 创建风险标签
            y_risk_train = np.where(y_train > HIGH_RISK_THRESHOLD, 'high', 'normal')
            y_risk_val = np.where(y_val > HIGH_RISK_THRESHOLD, 'high', 'normal')

            # 步骤1: 训练分诊分类器
            scaler_triage = StandardScaler()
            X_train_scaled = scaler_triage.fit_transform(X_train)
            X_val_scaled = scaler_triage.transform(X_val)

            triage_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
            triage_clf.fit(X_train_scaled, y_risk_train)

            risk_pred = triage_clf.predict(X_val_scaled)
            triage_acc = np.mean(risk_pred == y_risk_val)

            # 步骤2: 训练专家回归模型
            expert_models = {}
            expert_scalers = {}

            for risk_level in ['normal', 'high']:
                mask = (y_risk_train == risk_level)
                X_expert = X_train[mask]
                y_expert = y_train[mask]

                if len(X_expert) == 0:
                    continue

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
                    y_pred[i] = y_train.mean()

            # 评估
            overall_r2 = 1 - np.sum((y_val - y_pred)**2) / np.sum((y_val - np.mean(y_val))**2)
            overall_rmse = np.sqrt(np.mean((y_val - y_pred)**2))

            high_risk_mask = y_val > HIGH_RISK_THRESHOLD
            if high_risk_mask.sum() > 0:
                high_r2 = 1 - np.sum((y_val[high_risk_mask] - y_pred[high_risk_mask])**2) / \
                          np.sum((y_val[high_risk_mask] - np.mean(y_val[high_risk_mask]))**2)
                high_rmse = np.sqrt(np.mean((y_val[high_risk_mask] - y_pred[high_risk_mask])**2))
            else:
                high_r2 = high_rmse = np.nan

            results['overall'].append({'r2': overall_r2, 'rmse': overall_rmse})
            results['high_risk'].append({'r2': high_r2, 'rmse': high_rmse})
            results['triage'].append({'accuracy': triage_acc})

            print(f"Fold {fold}: 分诊={triage_acc*100:.1f}%, 整体R2={overall_r2:.4f}, RMSE={overall_rmse:.2f}mm", end="")
            if not np.isnan(high_r2):
                print(f", 高风险R2={high_r2:.4f}, RMSE={high_rmse:.2f}mm")
            else:
                print()

        # 汇总
        triage_acc = np.mean([r['accuracy'] for r in results['triage']])
        overall_r2 = np.mean([r['r2'] for r in results['overall']])
        overall_rmse = np.mean([r['rmse'] for r in results['overall']])
        high_r2_valid = [r['r2'] for r in results['high_risk'] if not np.isnan(r['r2'])]
        high_rmse_valid = [r['rmse'] for r in results['high_risk'] if not np.isnan(r['rmse'])]

        print(f"\n汇总: 分诊={triage_acc*100:.1f}%, 整体R2={overall_r2:.4f}, RMSE={overall_rmse:.2f}mm")
        if high_r2_valid:
            print(f"      高风险R2={np.mean(high_r2_valid):.4f}, RMSE={np.mean(high_rmse_valid):.2f}mm")

        return results

    def evaluate_weighted_regression(self, k=5, high_weight=3.0):
        """方案C: 加权回归(高风险样本权重×3)"""
        print("\n" + "="*80)
        print(f"方案C: 加权回归(高风险样本权重×{high_weight})")
        print("="*80)

        HIGH_RISK_THRESHOLD = 60.0
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        results = {'overall': [], 'high_risk': []}

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_power), 1):
            X_train, X_val = self.X_power[train_idx], self.X_power[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            # 计算样本权重
            weights = np.where(y_train > HIGH_RISK_THRESHOLD, high_weight, 1.0)

            # 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # 加权贝叶斯岭回归
            model = BayesianRidge(n_iter=300, tol=1e-3)

            # 使用样本权重(通过重复样本模拟)
            # sklearn的BayesianRidge不直接支持sample_weight,需要手动处理
            # 简化方案: 重复高风险样本
            high_mask = (y_train > HIGH_RISK_THRESHOLD)
            n_high = high_mask.sum()

            if n_high > 0:
                # 重复高风险样本(weight-1)次
                X_train_repeated = np.vstack([
                    X_train_scaled,
                    np.repeat(X_train_scaled[high_mask], int(high_weight-1), axis=0)
                ])
                y_train_repeated = np.concatenate([
                    y_train,
                    np.repeat(y_train[high_mask], int(high_weight-1))
                ])

                model.fit(X_train_repeated, y_train_repeated)
            else:
                model.fit(X_train_scaled, y_train)

            # 预测
            y_pred, y_std = model.predict(X_val_scaled, return_std=True)

            # 整体性能
            overall_r2 = 1 - np.sum((y_val - y_pred)**2) / np.sum((y_val - np.mean(y_val))**2)
            overall_rmse = np.sqrt(np.mean((y_val - y_pred)**2))

            # 高风险样本性能
            high_risk_mask = y_val > HIGH_RISK_THRESHOLD
            if high_risk_mask.sum() > 0:
                high_r2 = 1 - np.sum((y_val[high_risk_mask] - y_pred[high_risk_mask])**2) / \
                          np.sum((y_val[high_risk_mask] - np.mean(y_val[high_risk_mask]))**2)
                high_rmse = np.sqrt(np.mean((y_val[high_risk_mask] - y_pred[high_risk_mask])**2))
            else:
                high_r2 = high_rmse = np.nan

            results['overall'].append({'r2': overall_r2, 'rmse': overall_rmse})
            results['high_risk'].append({'r2': high_r2, 'rmse': high_rmse})

            print(f"Fold {fold}: 整体R2={overall_r2:.4f}, RMSE={overall_rmse:.2f}mm", end="")
            if not np.isnan(high_r2):
                print(f", 高风险R2={high_r2:.4f}, RMSE={high_rmse:.2f}mm")
            else:
                print()

        # 汇总
        overall_r2 = np.mean([r['r2'] for r in results['overall']])
        overall_rmse = np.mean([r['rmse'] for r in results['overall']])
        high_r2_valid = [r['r2'] for r in results['high_risk'] if not np.isnan(r['r2'])]
        high_rmse_valid = [r['rmse'] for r in results['high_risk'] if not np.isnan(r['rmse'])]

        print(f"\n汇总: 整体R2={overall_r2:.4f}, RMSE={overall_rmse:.2f}mm")
        if high_r2_valid:
            print(f"      高风险R2={np.mean(high_r2_valid):.4f}, RMSE={np.mean(high_rmse_valid):.2f}mm")

        return results


def main():
    print("="*80)
    print("改进方案全面对比实验")
    print("="*80)
    print("方案B: 简化特征(26维)分诊-专家系统")
    print("方案C: 加权回归(高风险样本权重×3)")
    print("="*80)

    # 初始化
    predictor = ImprovedVIVPredictor('../data/final_bridge_dataset.csv')
    X_base, X_power, y = predictor.load_and_prepare_data()

    # 基线
    print("\n" + "#"*80)
    print("# 实验1: 基线模型")
    print("#"*80)
    baseline_results = predictor.evaluate_baseline(k=5)

    # 方案B
    print("\n" + "#"*80)
    print("# 实验2: 方案B - 简化特征分诊")
    print("#"*80)
    planB_results = predictor.evaluate_simplified_triage(k=5)

    # 方案C
    print("\n" + "#"*80)
    print("# 实验3: 方案C - 加权回归")
    print("#"*80)
    planC_results = predictor.evaluate_weighted_regression(k=5, high_weight=3.0)

    # 最终对比
    print("\n" + "="*80)
    print("最终性能对比总结")
    print("="*80)

    models = [
        ('基线(26特征)', baseline_results),
        ('方案B(简化分诊)', planB_results),
        ('方案C(加权回归)', planC_results)
    ]

    print(f"\n{'模型':<20} {'整体R2':<12} {'整体RMSE':<12} {'高风险R2':<12} {'高风险RMSE':<12}")
    print("-"*80)

    for name, results in models:
        overall_r2 = np.mean([r['r2'] for r in results['overall']])
        overall_rmse = np.mean([r['rmse'] for r in results['overall']])
        high_r2_valid = [r['r2'] for r in results['high_risk'] if not np.isnan(r['r2'])]
        high_rmse_valid = [r['rmse'] for r in results['high_risk'] if not np.isnan(r['rmse'])]

        if high_r2_valid:
            high_r2 = np.mean(high_r2_valid)
            high_rmse = np.mean(high_rmse_valid)
            print(f"{name:<20} {overall_r2:.4f}      {overall_rmse:.2f} mm    {high_r2:.4f}      {high_rmse:.2f} mm")
        else:
            print(f"{name:<20} {overall_r2:.4f}      {overall_rmse:.2f} mm    N/A          N/A")

    print("\n" + "="*80)
    print("实验完成!")
    print("="*80)

    return baseline_results, planB_results, planC_results


if __name__ == '__main__':
    baseline, planB, planC = main()
