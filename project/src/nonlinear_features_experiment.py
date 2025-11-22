#!/usr/bin/env python3
"""
非线性特征变换实验 - 受吴先生启发
Nonlinear Feature Transformation for Better Uncertainty Calibration

吴先生的核心洞察:
1. 当前贝叶斯岭回归预测vs真实呈直线(线性关系)
2. VIV振幅可能与参数呈非线性关系(指数、对数、幂函数)
3. 通过非线性特征变换,让模型拟合曲线而非直线
4. 预期: 提升难题vs简单题的不确定性区分能力

实验设计:
- 基线: Griffin特征 + 线性Ridge/Bayesian Ridge
- 实验1: Griffin特征 + 指数变换 + Bayesian Ridge
- 实验2: Griffin特征 + 对数变换 + Bayesian Ridge
- 实验3: Griffin特征 + 幂函数变换 + Bayesian Ridge
- 实验4: Griffin特征 + 多项式核 + Kernel Ridge
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class NonlinearFeatureTransformer:
    """非线性特征变换器"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None

    def load_and_prepare_data(self):
        """加载并准备数据(含Griffin特征)"""
        print("="*80)
        print("非线性特征变换实验")
        print("="*80)
        print("灵感来源: 吴先生对bayesian_uncertainty_visualization.jpg的观察")
        print("核心假设: 引入非线性变换,让模型拟合曲线而非直线")
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
        df_features = self._create_physics_features(feature_cols)
        df_features = self._create_interaction_features(df_features)
        df_features = self._create_griffin_plot_features(df_features)

        # 移除缺失值
        df_features = df_features.dropna()

        # 目标变量
        self.y = self.df.loc[df_features.index, 'Max_Amplitude_mm'].values

        # 特征矩阵
        self.X = df_features.values
        self.feature_names = df_features.columns.tolist()

        print(f"\n基础特征集: {len(self.feature_names)} 个特征, {len(self.X)} 个样本")

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

    def _create_griffin_plot_features(self, df_features):
        """创建Griffin Plot特征"""
        if 'Reduced_Velocity' not in df_features.columns:
            return df_features

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

    def create_nonlinear_features(self, X, transform_type='exp'):
        """
        创建非线性特征变换

        transform_type:
        - 'exp': 指数变换 exp(X/scale)
        - 'log': 对数变换 log(|X|+1)
        - 'power': 幂函数变换 X^2, X^3
        - 'poly': 多项式特征(交叉项)
        """
        print(f"\n创建非线性特征变换: {transform_type}")

        if transform_type == 'exp':
            # 指数变换(避免数值爆炸,需要缩放)
            X_scaled = X / (np.std(X, axis=0) + 1e-6)
            X_exp = np.exp(X_scaled / 10.0)  # 除以10避免exp过大
            X_nonlinear = np.hstack([X, X_exp])
            print(f"  添加指数特征: {X.shape[1]} → {X_nonlinear.shape[1]}")

        elif transform_type == 'log':
            # 对数变换
            X_log = np.log(np.abs(X) + 1.0)
            X_nonlinear = np.hstack([X, X_log])
            print(f"  添加对数特征: {X.shape[1]} → {X_nonlinear.shape[1]}")

        elif transform_type == 'power':
            # 幂函数变换
            X_squared = X ** 2
            X_cubed = X ** 3
            X_nonlinear = np.hstack([X, X_squared, X_cubed])
            print(f"  添加幂函数特征: {X.shape[1]} → {X_nonlinear.shape[1]}")

        elif transform_type == 'poly':
            # 多项式特征(degree=2,包含交叉项)
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_nonlinear = poly.fit_transform(X)
            print(f"  添加多项式特征: {X.shape[1]} → {X_nonlinear.shape[1]}")

        else:
            raise ValueError(f"Unknown transform_type: {transform_type}")

        return X_nonlinear

    def evaluate_nonlinear_models(self, k=5):
        """评估不同非线性变换的效果"""
        print("\n" + "="*80)
        print("实验: 非线性特征变换对比")
        print("="*80)

        transforms = ['none', 'exp', 'log', 'power', 'poly']
        results = {}

        for transform_type in transforms:
            print(f"\n{'='*80}")
            print(f"测试变换类型: {transform_type}")
            print(f"{'='*80}")

            kf = KFold(n_splits=k, shuffle=True, random_state=42)
            fold_results = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(self.X), 1):
                X_train, X_val = self.X[train_idx], self.X[val_idx]
                y_train, y_val = self.y[train_idx], self.y[val_idx]

                # 应用非线性变换
                if transform_type != 'none':
                    X_train_transformed = self.create_nonlinear_features(X_train, transform_type)
                    X_val_transformed = self.create_nonlinear_features(X_val, transform_type)
                else:
                    X_train_transformed = X_train
                    X_val_transformed = X_val

                # 标准化
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_transformed)
                X_val_scaled = scaler.transform(X_val_transformed)

                # 贝叶斯岭回归(带不确定性量化)
                model = BayesianRidge(n_iter=300, tol=1e-3)
                model.fit(X_train_scaled, y_train)

                # 预测
                y_pred, y_std = model.predict(X_val_scaled, return_std=True)

                # 评估
                r2 = 1 - np.sum((y_val - y_pred)**2) / np.sum((y_val - np.mean(y_val))**2)
                rmse = np.sqrt(np.mean((y_val - y_pred)**2))
                mean_std = np.mean(y_std)

                # 不确定性校准(难题vs简单题)
                # 将样本按真实振幅分组
                low_amp_mask = y_val < np.median(y_val)  # 简单题(小振幅)
                high_amp_mask = ~low_amp_mask  # 难题(大振幅)

                std_low = np.mean(y_std[low_amp_mask])
                std_high = np.mean(y_std[high_amp_mask])
                std_ratio = std_high / std_low  # 难题不确定性 / 简单题不确定性

                fold_results.append({
                    'r2': r2,
                    'rmse': rmse,
                    'mean_std': mean_std,
                    'std_low': std_low,
                    'std_high': std_high,
                    'std_ratio': std_ratio
                })

                print(f"  Fold {fold}: R2={r2:.4f}, RMSE={rmse:.2f}mm, "
                      f"不确定性(低/高)={std_low:.2f}/{std_high:.2f}, 比值={std_ratio:.2f}")

            # 汇总
            results[transform_type] = {
                'mean_r2': np.mean([r['r2'] for r in fold_results]),
                'mean_rmse': np.mean([r['rmse'] for r in fold_results]),
                'mean_std': np.mean([r['mean_std'] for r in fold_results]),
                'mean_std_ratio': np.mean([r['std_ratio'] for r in fold_results]),
                'fold_results': fold_results
            }

        # 最终对比
        print("\n" + "="*80)
        print("非线性变换性能对比汇总")
        print("="*80)

        print(f"\n{'变换类型':<15} {'验证R2':<15} {'验证RMSE':<15} {'平均不确定性':<15} {'难/简比值':<15}")
        print("-"*80)

        for transform_type, result in results.items():
            print(f"{transform_type:<15} "
                  f"{result['mean_r2']:.4f}        "
                  f"{result['mean_rmse']:.2f} mm      "
                  f"{result['mean_std']:.2f} mm      "
                  f"{result['mean_std_ratio']:.2f}")

        # 找出最佳模型
        best_transform = max(results.items(), key=lambda x: x[1]['mean_r2'])
        best_uncertainty = max(results.items(), key=lambda x: x[1]['mean_std_ratio'])

        print("\n" + "="*80)
        print("实验结论")
        print("="*80)
        print(f"\n最佳R2性能: {best_transform[0]} (R2={best_transform[1]['mean_r2']:.4f})")
        print(f"最佳不确定性区分: {best_uncertainty[0]} (难/简比值={best_uncertainty[1]['mean_std_ratio']:.2f})")

        if best_uncertainty[1]['mean_std_ratio'] > results['none']['mean_std_ratio']:
            print(f"\n吴先生的假设得到验证! OK")
            print(f"非线性变换({best_uncertainty[0]})提升了难题vs简单题的不确定性区分能力")
            print(f"比值从{results['none']['mean_std_ratio']:.2f}提升到{best_uncertainty[1]['mean_std_ratio']:.2f}")
        else:
            print(f"\n非线性变换未显著提升不确定性区分能力")

        return results


def main():
    print("="*80)
    print("非线性特征变换实验")
    print("="*80)
    print("吴先生的核心观察:")
    print("1. 当前模型预测vs真实呈直线(线性)")
    print("2. 引入指数/对数等非线性变换")
    print("3. 预期: 拟合曲线,提升难题vs简单题的不确定性区分")
    print("="*80)

    # 初始化
    nft = NonlinearFeatureTransformer('../data/final_bridge_dataset.csv')

    # 加载数据
    X, y = nft.load_and_prepare_data()

    # 评估非线性变换
    results = nft.evaluate_nonlinear_models(k=5)

    print("\n" + "="*80)
    print("实验完成!")
    print("="*80)

    return results, nft


if __name__ == '__main__':
    results, model = main()
