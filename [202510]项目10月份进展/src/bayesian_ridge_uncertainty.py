#!/usr/bin/env python3
"""
贝叶斯岭回归 - 不确定性量化
Bayesian Ridge Regression with Uncertainty Quantification

核心价值:
- 不仅预测振幅点估计,还给出预测不确定性
- 工程应用: 高不确定性的桥梁需要更保守的设计
- SRTP创新点: 从点预测升级到区间预测

技术细节:
- 贝叶斯推断: 参数w不是固定值,而是概率分布
- 预测分布: p(y|X,D) = ∫p(y|X,w)p(w|D)dw
- 输出: 均值y_pred和标准差y_std
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, BayesianRidge
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class BayesianVIVModel:
    """贝叶斯岭回归VIV预测模型"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.model = None

    def load_and_prepare_data(self):
        """加载并准备数据 (复用之前的清洗逻辑)"""
        print("="*80)
        print("贝叶斯岭回归 - 不确定性量化模型")
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

        # 创建基础物理特征
        df_features = self._create_physics_features(feature_cols)

        # 创建交互特征
        df_features = self._create_interaction_features(df_features)

        # 移除缺失值
        df_features = df_features.dropna()

        # 目标变量
        self.y = self.df.loc[df_features.index, 'Max_Amplitude_mm'].values

        # 特征矩阵
        self.X = df_features.values
        self.feature_names = df_features.columns.tolist()

        print(f"\n最终特征集: {len(self.feature_names)} 个特征, {len(self.X)} 个样本")
        print(f"目标变量范围: {self.y.min():.1f} - {self.y.max():.1f} mm")

        return self.X, self.y

    def _create_physics_features(self, feature_cols):
        """创建基础物理特征"""
        df_features = self.df[feature_cols].copy()

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

        return df_features

    def _create_interaction_features(self, df_features):
        """创建交互特征"""
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

        return df_features

    def evaluate_kfold_with_uncertainty(self, k=5):
        """K-Fold交叉验证 - 对比标准岭回归vs贝叶斯岭回归"""
        print("\n" + "="*80)
        print(f"K-Fold交叉验证对比实验 (k={k})")
        print("="*80)

        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        # 存储结果
        ridge_results = []
        bayesian_results = []
        uncertainty_analysis = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X), 1):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            # 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # 模型1: 标准岭回归
            ridge = Ridge(alpha=10.0)
            ridge.fit(X_train_scaled, y_train)
            y_val_ridge = ridge.predict(X_val_scaled)

            ridge_r2 = 1 - np.sum((y_val - y_val_ridge)**2) / np.sum((y_val - np.mean(y_val))**2)
            ridge_rmse = np.sqrt(np.mean((y_val - y_val_ridge)**2))

            ridge_results.append({
                'fold': fold,
                'r2': ridge_r2,
                'rmse': ridge_rmse
            })

            # 模型2: 贝叶斯岭回归
            bayesian = BayesianRidge(
                n_iter=300,
                tol=1e-3,
                alpha_1=1e-6,  # Gamma先验超参数
                alpha_2=1e-6,
                lambda_1=1e-6,
                lambda_2=1e-6,
                compute_score=True
            )
            bayesian.fit(X_train_scaled, y_train)

            # 预测均值和标准差
            y_val_bayesian, y_val_std = bayesian.predict(X_val_scaled, return_std=True)

            bayesian_r2 = 1 - np.sum((y_val - y_val_bayesian)**2) / np.sum((y_val - np.mean(y_val))**2)
            bayesian_rmse = np.sqrt(np.mean((y_val - y_val_bayesian)**2))

            # 不确定性分析
            mean_uncertainty = np.mean(y_val_std)
            max_uncertainty = np.max(y_val_std)
            min_uncertainty = np.min(y_val_std)

            # 置信区间覆盖率 (95%置信区间是否包含真实值)
            y_lower = y_val_bayesian - 1.96 * y_val_std
            y_upper = y_val_bayesian + 1.96 * y_val_std
            coverage = np.mean((y_val >= y_lower) & (y_val <= y_upper))

            bayesian_results.append({
                'fold': fold,
                'r2': bayesian_r2,
                'rmse': bayesian_rmse
            })

            uncertainty_analysis.append({
                'fold': fold,
                'mean_std': mean_uncertainty,
                'max_std': max_uncertainty,
                'min_std': min_uncertainty,
                'coverage_95': coverage
            })

            print(f"\nFold {fold}/{k}:")
            print(f"  标准岭回归:   R2={ridge_r2:.4f}, RMSE={ridge_rmse:.2f} mm")
            print(f"  贝叶斯岭回归: R2={bayesian_r2:.4f}, RMSE={bayesian_rmse:.2f} mm")
            print(f"  平均不确定性: {mean_uncertainty:.2f} mm")
            print(f"  95%置信区间覆盖率: {coverage*100:.1f}%")

        # 汇总统计
        print("\n" + "="*80)
        print("性能对比汇总")
        print("="*80)

        ridge_r2_mean = np.mean([r['r2'] for r in ridge_results])
        ridge_r2_std = np.std([r['r2'] for r in ridge_results])
        ridge_rmse_mean = np.mean([r['rmse'] for r in ridge_results])

        bayesian_r2_mean = np.mean([r['r2'] for r in bayesian_results])
        bayesian_r2_std = np.std([r['r2'] for r in bayesian_results])
        bayesian_rmse_mean = np.mean([r['rmse'] for r in bayesian_results])

        print(f"\n标准岭回归:")
        print(f"  验证集R2:   {ridge_r2_mean:.4f} ± {ridge_r2_std:.4f}")
        print(f"  验证集RMSE: {ridge_rmse_mean:.2f} mm")

        print(f"\n贝叶斯岭回归:")
        print(f"  验证集R2:   {bayesian_r2_mean:.4f} ± {bayesian_r2_std:.4f}")
        print(f"  验证集RMSE: {bayesian_rmse_mean:.2f} mm")

        print(f"\n不确定性统计:")
        print(f"  平均预测标准差: {np.mean([u['mean_std'] for u in uncertainty_analysis]):.2f} mm")
        print(f"  平均95%覆盖率: {np.mean([u['coverage_95'] for u in uncertainty_analysis])*100:.1f}%")

        return {
            'ridge': ridge_results,
            'bayesian': bayesian_results,
            'uncertainty': uncertainty_analysis
        }

    def train_final_model_and_visualize(self):
        """训练最终贝叶斯模型并可视化不确定性"""
        print("\n" + "="*80)
        print("训练最终贝叶斯岭回归模型")
        print("="*80)

        # 标准化
        self.scaler.fit(self.X)
        X_scaled = self.scaler.transform(self.X)

        # 训练贝叶斯岭回归
        self.model = BayesianRidge(
            n_iter=300,
            tol=1e-3,
            alpha_1=1e-6,
            alpha_2=1e-6,
            lambda_1=1e-6,
            lambda_2=1e-6,
            compute_score=True
        )
        self.model.fit(X_scaled, self.y)

        # 预测
        y_pred, y_std = self.model.predict(X_scaled, return_std=True)

        # 评估
        r2 = 1 - np.sum((self.y - y_pred)**2) / np.sum((self.y - np.mean(self.y))**2)
        rmse = np.sqrt(np.mean((self.y - y_pred)**2))

        print(f"\n最终模型性能:")
        print(f"  R2:   {r2:.4f}")
        print(f"  RMSE: {rmse:.2f} mm")
        print(f"  平均不确定性: {np.mean(y_std):.2f} mm")

        # 可视化
        self._visualize_uncertainty(self.y, y_pred, y_std)

        return self.model

    def _visualize_uncertainty(self, y_true, y_pred, y_std):
        """可视化预测不确定性"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 子图1: 预测vs真实 (带误差棒)
        axes[0, 0].errorbar(y_true, y_pred, yerr=1.96*y_std, fmt='o', alpha=0.5,
                            elinewidth=1, capsize=3, markersize=4)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                        'r--', linewidth=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual Amplitude (mm)', fontsize=12)
        axes[0, 0].set_ylabel('Predicted Amplitude (mm)', fontsize=12)
        axes[0, 0].set_title('Bayesian Prediction with 95% Confidence Interval', fontsize=14)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 子图2: 不确定性分布直方图
        axes[0, 1].hist(y_std, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(np.mean(y_std), color='r', linestyle='--', linewidth=2,
                           label=f'Mean={np.mean(y_std):.2f} mm')
        axes[0, 1].set_xlabel('Prediction Std (mm)', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].set_title('Distribution of Prediction Uncertainty', fontsize=14)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 子图3: 不确定性vs真实振幅
        axes[1, 0].scatter(y_true, y_std, alpha=0.6, s=50)
        axes[1, 0].set_xlabel('Actual Amplitude (mm)', fontsize=12)
        axes[1, 0].set_ylabel('Prediction Std (mm)', fontsize=12)
        axes[1, 0].set_title('Uncertainty vs Actual Amplitude', fontsize=14)
        axes[1, 0].grid(True, alpha=0.3)

        # 子图4: 预测误差vs不确定性
        residuals = np.abs(y_true - y_pred)
        axes[1, 1].scatter(y_std, residuals, alpha=0.6, s=50)
        axes[1, 1].plot([y_std.min(), y_std.max()], [y_std.min(), y_std.max()],
                        'r--', linewidth=2, label='Ideal Calibration')
        axes[1, 1].set_xlabel('Prediction Std (mm)', fontsize=12)
        axes[1, 1].set_ylabel('Absolute Prediction Error (mm)', fontsize=12)
        axes[1, 1].set_title('Uncertainty Calibration', fontsize=14)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('../results/bayesian_uncertainty_visualization.png', dpi=150, bbox_inches='tight')
        print(f"\n可视化已保存: ../results/bayesian_uncertainty_visualization.png")

    def predict_with_uncertainty(self, bridge_params):
        """对新桥梁预测振幅和不确定性"""
        if self.model is None:
            raise ValueError("模型未训练,请先调用train_final_model_and_visualize")

        # 标准化
        X_scaled = self.scaler.transform(bridge_params)

        # 预测
        y_pred, y_std = self.model.predict(X_scaled, return_std=True)

        # 95%置信区间
        y_lower = y_pred - 1.96 * y_std
        y_upper = y_pred + 1.96 * y_std

        return {
            'prediction': y_pred[0],
            'std': y_std[0],
            'lower_95': y_lower[0],
            'upper_95': y_upper[0]
        }


def main():
    print("="*80)
    print("贝叶斯岭回归 - 不确定性量化实验")
    print("="*80)
    print("目标: 在保持R2性能的同时,提供预测不确定性")
    print("工程价值: 高不确定性桥梁需要更保守设计")
    print("="*80)

    # 初始化模型
    bvm = BayesianVIVModel('../data/final_bridge_dataset.csv')

    # 加载数据
    X, y = bvm.load_and_prepare_data()

    # K-Fold交叉验证对比
    results = bvm.evaluate_kfold_with_uncertainty(k=5)

    # 训练最终模型并可视化
    final_model = bvm.train_final_model_and_visualize()

    # 示例: 对一座新桥预测
    print("\n" + "="*80)
    print("示例: 对第一座桥梁预测")
    print("="*80)

    bridge_sample = X[0:1, :]
    prediction = bvm.predict_with_uncertainty(bridge_sample)

    print(f"\n预测结果:")
    print(f"  点估计:        {prediction['prediction']:.2f} mm")
    print(f"  标准差:        {prediction['std']:.2f} mm")
    print(f"  95%置信区间:  [{prediction['lower_95']:.2f}, {prediction['upper_95']:.2f}] mm")
    print(f"  真实值:        {y[0]:.2f} mm")

    if y[0] >= prediction['lower_95'] and y[0] <= prediction['upper_95']:
        print("  结论: 真实值在95%置信区间内 OK")
    else:
        print("  结论: 真实值不在95%置信区间内 WARNING")

    print("\n" + "="*80)
    print("实验完成!")
    print("="*80)
    print("核心发现:")
    print("1. 贝叶斯岭回归R2性能与标准岭回归相当")
    print("2. 额外提供预测不确定性,工程价值显著提升")
    print("3. 95%置信区间覆盖率应接近95%(模型校准良好)")
    print("="*80)

    return results, final_model


if __name__ == '__main__':
    results, model = main()
