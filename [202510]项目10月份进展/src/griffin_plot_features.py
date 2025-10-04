#!/usr/bin/env python3
"""
格里芬图(Griffin Plot)特征工程 - 受吴先生启发
VIV Lock-in Region Feature Engineering

物理背景:
- Griffin Plot是VIV领域的经验规律,描述振幅-约化速度关系
- VIV锁定区(Lock-in Region): 4 < Vr < 8
- 在锁定区内,振幅急剧增长;区外振幅较小

核心思想:
1. 创建二元特征: is_in_lock_in_region (是否在锁定区)
2. 创建连续特征: distance_to_lock_in_center (距离锁定区中心的距离)
3. 创建非线性特征: vr_lock_in_response (锁定区响应强度)

预期收益:
- 帮助模型区分"有VIV风险"和"无VIV风险"工况
- 在锁定区内进行更精确的预测
- 减少大振幅样本的预测不确定性
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')


class GriffinPlotFeatureEngineer:
    """基于格里芬图的VIV特征工程"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None

        # Griffin Plot参数(基于VIV文献)
        self.VR_LOCK_IN_START = 4.0   # 锁定区起始
        self.VR_LOCK_IN_END = 8.0     # 锁定区结束
        self.VR_LOCK_IN_CENTER = 6.0  # 锁定区中心(振幅最大点)

    def load_and_prepare_data(self):
        """加载并准备数据"""
        print("="*80)
        print("格里芬图(Griffin Plot)特征工程")
        print("="*80)
        print("灵感来源: 吴先生在改进1.md中的建议")
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

        # **核心创新: 创建Griffin Plot特征**
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

        # 统计锁定区样本分布
        if 'is_in_lock_in_region' in df_features.columns:
            lock_in_count = df_features['is_in_lock_in_region'].sum()
            print(f"\n锁定区样本统计:")
            print(f"  锁定区内(4<Vr<8): {int(lock_in_count)} 个 ({lock_in_count/len(df_features)*100:.1f}%)")
            print(f"  锁定区外: {len(df_features)-int(lock_in_count)} 个 ({(1-lock_in_count/len(df_features))*100:.1f}%)")

            # 锁定区内外的平均振幅对比
            lock_in_mask = (df_features['is_in_lock_in_region'] == 1).values
            amp_in_lock = self.y[lock_in_mask]
            amp_out_lock = self.y[~lock_in_mask]

            print(f"\n振幅统计对比:")
            print(f"  锁定区内平均振幅: {amp_in_lock.mean():.2f} mm")
            print(f"  锁定区外平均振幅: {amp_out_lock.mean():.2f} mm")
            print(f"  振幅提升倍数: {amp_in_lock.mean()/amp_out_lock.mean():.2f}x")

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

        # Reduced Velocity (核心参数,用于Griffin Plot)
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
        """
        创建基于Griffin Plot的特征 - 核心创新

        Griffin Plot理论:
        - VIV响应在约化速度Vr=4-8范围内最强(锁定区Lock-in)
        - 锁定区中心约在Vr≈6,振幅达到峰值
        - Vr<4或Vr>8时,VIV响应减弱或消失
        """
        print("\n创建Griffin Plot特征:")

        if 'Reduced_Velocity' not in df_features.columns:
            print("WARNING 缺少Reduced_Velocity,无法创建Griffin特征")
            return df_features

        Vr = df_features['Reduced_Velocity']

        # 特征1: 二元特征 - 是否在锁定区内
        df_features['is_in_lock_in_region'] = (
            (Vr >= self.VR_LOCK_IN_START) & (Vr <= self.VR_LOCK_IN_END)
        ).astype(float)
        print(f"  OK 创建 is_in_lock_in_region (锁定区标识)")

        # 特征2: 连续特征 - 距离锁定区中心的距离
        df_features['distance_to_lock_in_center'] = np.abs(Vr - self.VR_LOCK_IN_CENTER)
        print(f"  OK 创建 distance_to_lock_in_center (距锁定中心距离)")

        # 特征3: 锁定区响应强度 (基于高斯函数)
        # 物理意义: 越接近Vr=6,响应越强;远离时响应衰减
        # 使用高斯核: exp(-((Vr-6)/sigma)^2)
        sigma = 2.0  # 控制锁定区宽度
        df_features['vr_lock_in_response'] = np.exp(
            -((Vr - self.VR_LOCK_IN_CENTER) / sigma) ** 2
        )
        print(f"  OK 创建 vr_lock_in_response (锁定响应强度,高斯核)")

        # 特征4: 锁定区内的Scruton交互
        # 物理意义: 锁定区内,Scruton数的影响可能更显著
        if 'Scruton_Number' in df_features.columns:
            df_features['Scruton_in_lock_in'] = (
                df_features['Scruton_Number'] * df_features['is_in_lock_in_region']
            )
            print(f"  OK 创建 Scruton_in_lock_in (锁定区内Scruton交互)")

        # 特征5: 分段线性特征 - VIV发展阶段
        # Vr<4: 初始分支(Initial Branch)
        # 4<Vr<8: 上分支(Upper Branch,锁定区)
        # Vr>8: 下分支(Lower Branch)
        def viv_branch(vr):
            if vr < self.VR_LOCK_IN_START:
                return 0  # 初始分支
            elif vr <= self.VR_LOCK_IN_END:
                return 1  # 上分支(锁定区)
            else:
                return 2  # 下分支

        df_features['viv_branch'] = Vr.apply(viv_branch)
        print(f"  OK 创建 viv_branch (VIV发展阶段: 0=初始,1=锁定,2=下分支)")

        # 特征6: 锁定区深度 (进入锁定区的程度)
        # 物理意义: 在锁定区边缘(Vr≈4或8)时,VIV刚开始或即将结束
        #          在锁定区中心(Vr≈6)时,VIV最强
        df_features['lock_in_depth'] = np.where(
            df_features['is_in_lock_in_region'] == 1,
            1.0 - df_features['distance_to_lock_in_center'] / (self.VR_LOCK_IN_END - self.VR_LOCK_IN_START),
            0.0
        )
        print(f"  OK 创建 lock_in_depth (锁定区深度,0=区外,1=中心)")

        return df_features

    def evaluate_griffin_features(self, k=5):
        """评估Griffin Plot特征的性能提升"""
        print("\n" + "="*80)
        print("实验: Griffin Plot特征 vs 基线特征")
        print("="*80)

        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        # 准备两个特征集
        # 基线: 不包含Griffin特征
        griffin_feature_names = [
            'is_in_lock_in_region', 'distance_to_lock_in_center',
            'vr_lock_in_response', 'Scruton_in_lock_in',
            'viv_branch', 'lock_in_depth'
        ]

        griffin_feature_idx = [
            i for i, name in enumerate(self.feature_names)
            if name in griffin_feature_names
        ]

        baseline_feature_idx = [
            i for i, name in enumerate(self.feature_names)
            if name not in griffin_feature_names
        ]

        print(f"\n特征统计:")
        print(f"  基线特征数: {len(baseline_feature_idx)}")
        print(f"  Griffin特征数: {len(griffin_feature_idx)}")
        print(f"  总特征数: {len(self.feature_names)}")

        # 存储结果
        baseline_results = []
        griffin_results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X), 1):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            # 基线特征集
            X_train_baseline = X_train[:, baseline_feature_idx]
            X_val_baseline = X_val[:, baseline_feature_idx]

            # 完整特征集(含Griffin)
            X_train_full = X_train
            X_val_full = X_val

            # 标准化
            scaler_baseline = StandardScaler()
            X_train_baseline_scaled = scaler_baseline.fit_transform(X_train_baseline)
            X_val_baseline_scaled = scaler_baseline.transform(X_val_baseline)

            scaler_full = StandardScaler()
            X_train_full_scaled = scaler_full.fit_transform(X_train_full)
            X_val_full_scaled = scaler_full.transform(X_val_full)

            # 模型: 岭回归
            baseline_model = Ridge(alpha=10.0)
            baseline_model.fit(X_train_baseline_scaled, y_train)
            y_pred_baseline = baseline_model.predict(X_val_baseline_scaled)

            griffin_model = Ridge(alpha=10.0)
            griffin_model.fit(X_train_full_scaled, y_train)
            y_pred_griffin = griffin_model.predict(X_val_full_scaled)

            # 评估
            baseline_r2 = 1 - np.sum((y_val - y_pred_baseline)**2) / np.sum((y_val - np.mean(y_val))**2)
            baseline_rmse = np.sqrt(np.mean((y_val - y_pred_baseline)**2))

            griffin_r2 = 1 - np.sum((y_val - y_pred_griffin)**2) / np.sum((y_val - np.mean(y_val))**2)
            griffin_rmse = np.sqrt(np.mean((y_val - y_pred_griffin)**2))

            baseline_results.append({'r2': baseline_r2, 'rmse': baseline_rmse})
            griffin_results.append({'r2': griffin_r2, 'rmse': griffin_rmse})

            print(f"\nFold {fold}/{k}:")
            print(f"  基线特征:    R2={baseline_r2:.4f}, RMSE={baseline_rmse:.2f} mm")
            print(f"  Griffin特征: R2={griffin_r2:.4f}, RMSE={griffin_rmse:.2f} mm")
            print(f"  提升:        ΔR2={griffin_r2-baseline_r2:+.4f}, ΔRMSE={griffin_rmse-baseline_rmse:+.2f} mm")

        # 汇总
        baseline_r2_mean = np.mean([r['r2'] for r in baseline_results])
        baseline_rmse_mean = np.mean([r['rmse'] for r in baseline_results])

        griffin_r2_mean = np.mean([r['r2'] for r in griffin_results])
        griffin_rmse_mean = np.mean([r['rmse'] for r in griffin_results])

        print("\n" + "="*80)
        print("性能对比汇总")
        print("="*80)
        print(f"\n基线特征(无Griffin):")
        print(f"  验证集R2:   {baseline_r2_mean:.4f}")
        print(f"  验证集RMSE: {baseline_rmse_mean:.2f} mm")

        print(f"\nGriffin Plot特征:")
        print(f"  验证集R2:   {griffin_r2_mean:.4f}")
        print(f"  验证集RMSE: {griffin_rmse_mean:.2f} mm")

        print(f"\n性能提升:")
        r2_improve = griffin_r2_mean - baseline_r2_mean
        r2_improve_pct = (r2_improve / baseline_r2_mean) * 100
        rmse_improve = griffin_rmse_mean - baseline_rmse_mean
        rmse_improve_pct = (rmse_improve / baseline_rmse_mean) * 100

        print(f"  ΔR2:   {r2_improve:+.4f} ({r2_improve_pct:+.2f}%)")
        print(f"  ΔRMSE: {rmse_improve:+.2f} mm ({rmse_improve_pct:+.2f}%)")

        if r2_improve > 0:
            print(f"\n结论: Griffin Plot特征有效提升性能! OK")
        else:
            print(f"\n结论: Griffin Plot特征未带来提升,需要调整参数")

        return {
            'baseline': baseline_results,
            'griffin': griffin_results,
            'improvement': {
                'r2': r2_improve,
                'rmse': rmse_improve
            }
        }


def main():
    print("="*80)
    print("格里芬图(Griffin Plot)特征工程实验")
    print("="*80)
    print("灵感来源: 吴先生在改进1.md中的VIV锁定区思路")
    print("目标: 利用VIV物理规律提升预测性能")
    print("="*80)

    # 初始化
    gpfe = GriffinPlotFeatureEngineer('../data/final_bridge_dataset.csv')

    # 加载数据并创建Griffin特征
    X, y = gpfe.load_and_prepare_data()

    # 评估Griffin特征的效果
    results = gpfe.evaluate_griffin_features(k=5)

    print("\n" + "="*80)
    print("实验完成!")
    print("="*80)

    return results, gpfe


if __name__ == '__main__':
    results, model = main()
