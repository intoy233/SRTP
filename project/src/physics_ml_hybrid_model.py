#!/usr/bin/env python3
"""
物理-机器学习混合模型 - 受吴先生启发
Physics-ML Hybrid Model (Residual Learning)

核心思想:
1. 建立基础物理模型作为粗略基准: y_phys = k / Scruton_Number
2. ML模型不预测振幅本身,而是预测残差: residual = y_true - y_phys
3. 最终预测: y_final = y_phys + y_ml_residual

优势:
- 物理模型捕捉基本趋势(虽然R2仅0.0768)
- ML模型专注学习复杂的非线性偏差
- 分解任务,降低ML学习难度
- 在小数据集上更鲁棒

参考文献:
- 吴先生在改进1.md中的"物理-机器学习混合模型"思路
- 残差学习(Residual Learning)在ResNet中的成功应用
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')


class PhysicsMLHybridModel:
    """物理-机器学习混合模型"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.y_phys = None  # 物理模型预测
        self.y_residual = None  # 残差(真实值-物理预测)
        self.feature_names = None
        self.k_scruton = None  # Scruton定律系数

    def load_and_prepare_data(self):
        """加载并准备数据"""
        print("="*80)
        print("物理-机器学习混合模型")
        print("="*80)
        print("灵感来源: 吴先生在改进1.md中的残差学习思路")
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

        # 创建特征(含Griffin Plot特征)
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

    def fit_physics_baseline(self):
        """
        训练物理基线模型: y_phys = k / Scruton_Number

        使用线性回归拟合最优k值
        """
        print("\n" + "="*80)
        print("步骤1: 训练物理基线模型")
        print("="*80)

        # 找到Scruton_Number列
        if 'Scruton_Number' not in self.feature_names:
            raise ValueError("缺少Scruton_Number特征")

        scruton_idx = self.feature_names.index('Scruton_Number')
        Sc = self.X[:, scruton_idx]

        # 使用线性回归拟合: y = k/Sc
        # 转换为线性形式: y = k * (1/Sc)
        Sc_inv = 1.0 / (Sc + 1e-6)

        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(Sc_inv.reshape(-1, 1), self.y)

        self.k_scruton = lr.coef_[0]

        # 物理模型预测
        self.y_phys = self.k_scruton / (Sc + 1e-6)

        # 计算物理模型性能
        phys_r2 = 1 - np.sum((self.y - self.y_phys)**2) / np.sum((self.y - np.mean(self.y))**2)
        phys_rmse = np.sqrt(np.mean((self.y - self.y_phys)**2))

        print(f"\n物理模型: y_phys = {self.k_scruton:.2f} / Scruton_Number")
        print(f"  R2:   {phys_r2:.4f}")
        print(f"  RMSE: {phys_rmse:.2f} mm")

        # 计算残差
        self.y_residual = self.y - self.y_phys

        print(f"\n残差统计:")
        print(f"  范围: {self.y_residual.min():.2f} - {self.y_residual.max():.2f} mm")
        print(f"  均值: {self.y_residual.mean():.2f} mm (应接近0)")
        print(f"  标准差: {self.y_residual.std():.2f} mm")

        return self.y_phys, self.y_residual

    def evaluate_hybrid_model(self, k=5):
        """
        评估混合模型性能

        对比三种策略:
        1. 纯ML: 直接预测y
        2. 纯物理: y_phys = k/Sc
        3. 混合: y = y_phys + ML(residual)
        """
        print("\n" + "="*80)
        print("步骤2: K-Fold交叉验证对比")
        print("="*80)

        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        # 找到Scruton_Number索引
        scruton_idx = self.feature_names.index('Scruton_Number')

        # 存储结果
        pure_ml_results = []
        pure_physics_results = []
        hybrid_results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X), 1):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            # 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # 策略1: 纯ML(岭回归)
            pure_ml = Ridge(alpha=10.0)
            pure_ml.fit(X_train_scaled, y_train)
            y_pred_pure_ml = pure_ml.predict(X_val_scaled)

            pure_ml_r2 = 1 - np.sum((y_val - y_pred_pure_ml)**2) / np.sum((y_val - np.mean(y_val))**2)
            pure_ml_rmse = np.sqrt(np.mean((y_val - y_pred_pure_ml)**2))

            # 策略2: 纯物理模型
            Sc_train = X_train[:, scruton_idx]
            Sc_val = X_val[:, scruton_idx]

            Sc_inv_train = 1.0 / (Sc_train + 1e-6)
            Sc_inv_val = 1.0 / (Sc_val + 1e-6)

            from sklearn.linear_model import LinearRegression
            phys_model = LinearRegression()
            phys_model.fit(Sc_inv_train.reshape(-1, 1), y_train)

            k_fold = phys_model.coef_[0]
            y_pred_phys = k_fold / (Sc_val + 1e-6)

            pure_phys_r2 = 1 - np.sum((y_val - y_pred_phys)**2) / np.sum((y_val - np.mean(y_val))**2)
            pure_phys_rmse = np.sqrt(np.mean((y_val - y_pred_phys)**2))

            # 策略3: 混合模型(物理+ML残差)
            # 训练集上计算残差
            y_train_phys = k_fold / (Sc_train + 1e-6)
            residual_train = y_train - y_train_phys

            # ML模型预测残差
            residual_ml = Ridge(alpha=10.0)
            residual_ml.fit(X_train_scaled, residual_train)
            residual_pred = residual_ml.predict(X_val_scaled)

            # 最终预测 = 物理基线 + ML修正
            y_pred_hybrid = y_pred_phys + residual_pred

            hybrid_r2 = 1 - np.sum((y_val - y_pred_hybrid)**2) / np.sum((y_val - np.mean(y_val))**2)
            hybrid_rmse = np.sqrt(np.mean((y_val - y_pred_hybrid)**2))

            pure_ml_results.append({'r2': pure_ml_r2, 'rmse': pure_ml_rmse})
            pure_physics_results.append({'r2': pure_phys_r2, 'rmse': pure_phys_rmse})
            hybrid_results.append({'r2': hybrid_r2, 'rmse': hybrid_rmse})

            print(f"\nFold {fold}/{k}:")
            print(f"  纯ML模型:     R2={pure_ml_r2:.4f}, RMSE={pure_ml_rmse:.2f} mm")
            print(f"  纯物理模型:   R2={pure_phys_r2:.4f}, RMSE={pure_phys_rmse:.2f} mm")
            print(f"  混合模型:     R2={hybrid_r2:.4f}, RMSE={hybrid_rmse:.2f} mm")

        # 汇总
        print("\n" + "="*80)
        print("性能对比汇总")
        print("="*80)

        pure_ml_r2 = np.mean([r['r2'] for r in pure_ml_results])
        pure_ml_rmse = np.mean([r['rmse'] for r in pure_ml_results])

        pure_phys_r2 = np.mean([r['r2'] for r in pure_physics_results])
        pure_phys_rmse = np.mean([r['rmse'] for r in pure_physics_results])

        hybrid_r2 = np.mean([r['r2'] for r in hybrid_results])
        hybrid_rmse = np.mean([r['rmse'] for r in hybrid_results])

        print(f"\n纯ML模型(基线):")
        print(f"  验证集R2:   {pure_ml_r2:.4f}")
        print(f"  验证集RMSE: {pure_ml_rmse:.2f} mm")

        print(f"\n纯物理模型:")
        print(f"  验证集R2:   {pure_phys_r2:.4f}")
        print(f"  验证集RMSE: {pure_phys_rmse:.2f} mm")

        print(f"\n物理-ML混合模型:")
        print(f"  验证集R2:   {hybrid_r2:.4f}")
        print(f"  验证集RMSE: {hybrid_rmse:.2f} mm")

        print(f"\n性能提升(混合 vs 纯ML):")
        r2_improve = hybrid_r2 - pure_ml_r2
        rmse_improve = hybrid_rmse - pure_ml_rmse

        print(f"  ΔR2:   {r2_improve:+.4f} ({r2_improve/pure_ml_r2*100:+.2f}%)")
        print(f"  ΔRMSE: {rmse_improve:+.2f} mm ({rmse_improve/pure_ml_rmse*100:+.2f}%)")

        if r2_improve > 0:
            print(f"\n结论: 混合模型有效提升性能! OK")
        else:
            print(f"\n结论: 混合模型未带来提升,纯ML已经足够好")

        return {
            'pure_ml': pure_ml_results,
            'pure_physics': pure_physics_results,
            'hybrid': hybrid_results,
            'improvement': {
                'r2': r2_improve,
                'rmse': rmse_improve
            }
        }


def main():
    print("="*80)
    print("物理-机器学习混合模型实验")
    print("="*80)
    print("灵感来源: 吴先生在改进1.md中的残差学习思路")
    print("目标: 用ML修正物理模型的偏差")
    print("="*80)

    # 初始化
    pmlhm = PhysicsMLHybridModel('../data/final_bridge_dataset.csv')

    # 加载数据
    X, y = pmlhm.load_and_prepare_data()

    # 训练物理基线
    y_phys, y_residual = pmlhm.fit_physics_baseline()

    # 评估混合模型
    results = pmlhm.evaluate_hybrid_model(k=5)

    print("\n" + "="*80)
    print("实验完成!")
    print("="*80)

    return results, pmlhm


if __name__ == '__main__':
    results, model = main()
