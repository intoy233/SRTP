#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIV振幅预测器 - 生产部署版本
VIV Amplitude Predictor - Production Deployment

最终模型: Stacking集成 (R²=0.6290, RMSE=13.03mm)
基学习器: Ridge + Lasso + RandomForest + SVR + BayesianRidge
元学习器: BayesianRidge (带不确定性量化)

使用方法:
    predictor = VIVPredictor()
    predictor.train('path/to/data.csv')
    amplitude, uncertainty = predictor.predict(bridge_params)
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class VIVPredictor:
    """VIV振幅预测器 - Stacking集成模型"""

    def __init__(self):
        self.base_models = None
        self.meta_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False

    def _create_features(self, df):
        """创建完整特征集 (基础+交互+Griffin Plot+幂函数)"""
        # 排除非特征列
        exclude_cols = [
            'BridgeName', 'Country', 'BridgeType', 'PaperSource', 'Year',
            'Max_Amplitude_mm', 'Amplitude_RMS_mm', 'VIV_Wind_Speed_ms',
            'Risk_Level', 'Notes', 'Vibration_Suppression', 'Suppression_Effect',
            'Total_Length_m', 'First_Freq_Hz', 'Second_Freq_Hz',
            'Drag_Coefficient', 'Lift_Coefficient', 'BridgeID', 'Structure_Type'
        ]

        actual_exclude = [col for col in exclude_cols if col in df.columns]
        feature_cols = [col for col in df.columns if col not in actual_exclude]
        df_features = df[feature_cols].copy()

        # 基础物理特征
        if all(col in df.columns for col in ['Damping_Ratio', 'Width_m', 'Height_m']):
            df_features['Scruton_Number'] = (
                df['Damping_Ratio'] * (df['Width_m'] / df['Height_m']) * 100
            )

        if all(col in df.columns for col in ['Width_m', 'Height_m']):
            df_features['Aspect_Ratio'] = df['Width_m'] / df['Height_m']

        if 'Damping_Ratio' in df.columns:
            df_features['VIV_Susceptibility'] = 1.0 / (df['Damping_Ratio'] + 1e-6)

        if all(col in df.columns for col in ['Critical_Wind_Speed_ms', 'Natural_Freq_Hz', 'Width_m']):
            df_features['Reduced_Velocity'] = (
                df['Critical_Wind_Speed_ms'] / (df['Natural_Freq_Hz'] * df['Width_m'])
            )
            median_vr = df_features['Reduced_Velocity'].median()
            df_features['Reduced_Velocity'].fillna(median_vr, inplace=True)

        if all(col in df.columns for col in ['Natural_Freq_Hz', 'Span_m']):
            df_features['Stiffness_Parameter'] = df['Natural_Freq_Hz'] * np.sqrt(df['Span_m'])

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

        # Griffin Plot特征 (VIV锁定区域特征)
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

        df_features = df_features.dropna()

        # 幂函数变换 (X, X², X³)
        X_base = df_features.values
        X_squared = X_base ** 2
        X_cubed = X_base ** 3
        X_all = np.hstack([X_base, X_squared, X_cubed])

        # 保存特征名称
        self.feature_names = (
            list(df_features.columns) +
            [f"{col}_sq" for col in df_features.columns] +
            [f"{col}_cu" for col in df_features.columns]
        )

        return X_all, df_features.index

    def train(self, data_path, k=5):
        """训练Stacking模型"""
        print("="*80)
        print("VIV振幅预测器 - 训练Stacking模型")
        print("="*80)

        # 加载数据
        df = pd.read_csv(data_path)
        print(f"\n数据集: {len(df)} 座桥梁")

        # 创建特征
        X, valid_idx = self._create_features(df)
        y = df.loc[valid_idx, 'Max_Amplitude_mm'].values

        print(f"特征维度: {X.shape[1]} (26基础 + 26² + 26³)")
        print(f"有效样本: {len(X)}")

        # 初始化模型
        self.base_models = [
            ('Ridge', Ridge(alpha=10.0)),
            ('Lasso', Lasso(alpha=0.1, max_iter=5000)),
            ('RandomForest', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
            ('SVR_RBF', SVR(kernel='rbf', C=10, gamma='scale')),
            ('BayesianRidge', BayesianRidge(max_iter=300, tol=1e-3))
        ]
        self.meta_model = BayesianRidge(max_iter=300, tol=1e-3)

        print("\nStacking架构:")
        print("  Level 0 (基学习器): Ridge, Lasso, RandomForest, SVR, BayesianRidge")
        print("  Level 1 (元学习器): BayesianRidge (带不确定性)")

        # K-Fold交叉验证训练
        print(f"\n开始{k}-Fold交叉验证训练...")
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # 训练基学习器
            meta_features_train = np.zeros((len(X_train), len(self.base_models)))
            meta_features_val = np.zeros((len(X_val), len(self.base_models)))

            for i, (name, model) in enumerate(self.base_models):
                kf_inner = KFold(n_splits=3, shuffle=True, random_state=42)
                meta_features_train[:, i] = cross_val_predict(
                    model, X_train_scaled, y_train, cv=kf_inner
                )
                model.fit(X_train_scaled, y_train)
                meta_features_val[:, i] = model.predict(X_val_scaled)

            # 训练元学习器
            self.meta_model.fit(meta_features_train, y_train)
            y_pred, _ = self.meta_model.predict(meta_features_val, return_std=True)

            r2 = r2_score(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            fold_scores.append({'r2': r2, 'rmse': rmse})

            print(f"  Fold {fold}: R²={r2:.4f}, RMSE={rmse:.2f}mm")

        # 最终在全部数据上训练
        print("\n在全部数据上训练最终模型...")
        X_scaled = self.scaler.fit_transform(X)

        # 训练基学习器
        meta_features = np.zeros((len(X), len(self.base_models)))
        for i, (name, model) in enumerate(self.base_models):
            kf_inner = KFold(n_splits=3, shuffle=True, random_state=42)
            meta_features[:, i] = cross_val_predict(model, X_scaled, y, cv=kf_inner)
            model.fit(X_scaled, y)

        # 训练元学习器
        self.meta_model.fit(meta_features, y)

        # 汇总性能
        avg_r2 = np.mean([s['r2'] for s in fold_scores])
        avg_rmse = np.mean([s['rmse'] for s in fold_scores])
        std_r2 = np.std([s['r2'] for s in fold_scores])

        print("\n" + "="*80)
        print("训练完成!")
        print("="*80)
        print(f"交叉验证R²: {avg_r2:.4f} (±{std_r2:.4f})")
        print(f"交叉验证RMSE: {avg_rmse:.2f} mm")
        print("="*80)

        self.is_trained = True
        return avg_r2, avg_rmse

    def predict(self, bridge_params):
        """
        预测VIV振幅

        参数:
            bridge_params: dict或DataFrame, 包含桥梁参数

        返回:
            amplitude: 预测振幅 (mm)
            uncertainty: 不确定性 (mm, 1倍标准差)
        """
        if not self.is_trained:
            raise RuntimeError("模型未训练! 请先调用train()方法")

        # 转换为DataFrame
        if isinstance(bridge_params, dict):
            df_input = pd.DataFrame([bridge_params])
        else:
            df_input = bridge_params.copy()

        # 创建特征 (复用训练时的特征工程)
        X_input, _ = self._create_features(df_input)
        X_scaled = self.scaler.transform(X_input)

        # 基学习器预测
        meta_features = np.zeros((len(X_input), len(self.base_models)))
        for i, (name, model) in enumerate(self.base_models):
            meta_features[:, i] = model.predict(X_scaled)

        # 元学习器预测 (带不确定性)
        y_pred, y_std = self.meta_model.predict(meta_features, return_std=True)

        if len(y_pred) == 1:
            return y_pred[0], y_std[0]
        else:
            return y_pred, y_std

    def risk_assessment(self, amplitude, uncertainty):
        """
        风险评估

        参数:
            amplitude: 预测振幅 (mm)
            uncertainty: 不确定性 (mm)

        返回:
            risk_level: 风险等级 (低/中/高)
            recommendation: 工程建议
        """
        upper_bound = amplitude + uncertainty  # 上界 (保守估计)

        if amplitude > 50 or upper_bound > 70:
            risk_level = "高风险"
            recommendation = "【强烈建议】必须进行风洞实验或CFD验证!"
        elif amplitude > 30:
            risk_level = "中风险"
            recommendation = "考虑采取减振措施 (调谐质量阻尼器等)"
        else:
            risk_level = "低风险"
            recommendation = "初步评估安全,但建议结合工程师经验综合判断"

        return risk_level, recommendation

    def save_model(self, filepath):
        """保存模型"""
        if not self.is_trained:
            raise RuntimeError("模型未训练,无法保存!")

        model_data = {
            'base_models': self.base_models,
            'meta_model': self.meta_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"模型已保存至: {filepath}")

    def load_model(self, filepath):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.base_models = model_data['base_models']
        self.meta_model = model_data['meta_model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_trained = True

        print(f"模型已加载: {filepath}")


def demo_usage():
    """使用示例"""
    print("="*80)
    print("VIV预测器 - 使用示例")
    print("="*80)

    # 1. 训练模型
    predictor = VIVPredictor()
    avg_r2, avg_rmse = predictor.train('../data/final_bridge_dataset.csv', k=5)

    # 2. 保存模型
    predictor.save_model('../models/stacking_viv_predictor.pkl')

    # 3. 预测示例
    print("\n" + "="*80)
    print("预测示例")
    print("="*80)

    # 示例桥梁参数
    bridge_example = {
        'Span_m': 1385.0,
        'Width_m': 35.9,
        'Height_m': 3.0,
        'Damping_Ratio': 0.0030,
        'Natural_Freq_Hz': 0.125,
        'Critical_Wind_Speed_ms': 12.0
    }

    print("\n输入参数:")
    for key, value in bridge_example.items():
        print(f"  {key}: {value}")

    amplitude, uncertainty = predictor.predict(bridge_example)

    print(f"\n预测结果:")
    print(f"  振幅: {amplitude:.2f} ± {uncertainty:.2f} mm")
    print(f"  置信区间: [{amplitude-uncertainty:.2f}, {amplitude+uncertainty:.2f}] mm")

    # 4. 风险评估
    risk_level, recommendation = predictor.risk_assessment(amplitude, uncertainty)
    print(f"\n风险评估:")
    print(f"  风险等级: {risk_level}")
    print(f"  工程建议: {recommendation}")

    print("\n" + "="*80)
    print("示例完成!")
    print("="*80)


if __name__ == '__main__':
    demo_usage()
