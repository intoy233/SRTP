#!/usr/bin/env python3
"""
85座高质量数据集 - 完整特征实验
严格排除数据泄露,利用First_Freq, Drag/Lift等完整特征
对比: 数据质量(85座完整) vs 数据数量(196座部分缺失)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')


class FullFeatureVIVModel:
    """利用完整特征的VIV模型 (85座数据)"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None

    def load_data(self):
        """加载85座数据"""
        print("="*80)
        print("步骤1: 加载85座高质量数据集")
        print("="*80)

        self.df = pd.read_csv(self.data_path)
        print(f"\n数据集: {len(self.df)} 座桥梁")

        # 严格排除数据泄露和不可用特征
        exclude_cols = [
            # 标识列
            'BridgeName', 'Country', 'BridgeType', 'PaperSource', 'Year', 'BridgeID',
            # 目标变量
            'Max_Amplitude_mm',
            # 数据泄露特征 (关键!)
            'Amplitude_RMS_mm',  # 与Max_Amplitude相关系数0.99
            'VIV_Wind_Speed_ms',  # VIV发生时的风速,是结果
            # 辅助列
            'Risk_Level', 'Notes', 'Vibration_Suppression', 'Suppression_Effect',
            'Structure_Type'
        ]

        actual_exclude = [col for col in exclude_cols if col in self.df.columns]
        feature_cols = [col for col in self.df.columns if col not in actual_exclude]

        print(f"\n排除的列 ({len(actual_exclude)}):")
        for col in actual_exclude:
            if col in ['Amplitude_RMS_mm', 'VIV_Wind_Speed_ms']:
                print(f"  - {col:<30} *** 数据泄露,必须排除 ***")
            else:
                print(f"  - {col}")

        print(f"\n保留的原始特征 ({len(feature_cols)}):")
        for col in feature_cols:
            print(f"  - {col}")

        return feature_cols

    def create_full_physics_features(self, feature_cols):
        """创建完整物理特征集 (利用85座的完整辅助特征)"""
        print("\n" + "="*80)
        print("步骤2: 完整物理特征工程 (利用First_Freq, Drag/Lift等)")
        print("="*80)

        df_features = self.df[feature_cols].copy()

        # ===== 基础派生特征 =====

        # Scruton Number (核心VIV参数)
        if all(col in self.df.columns for col in ['Damping_Ratio', 'Width_m', 'Height_m']):
            df_features['Scruton_Number'] = (
                self.df['Damping_Ratio'] * (self.df['Width_m'] / self.df['Height_m']) * 100
            )
            print("OK 创建 Scruton_Number")

        # Aspect Ratio
        if all(col in self.df.columns for col in ['Width_m', 'Height_m']):
            df_features['Aspect_Ratio'] = self.df['Width_m'] / self.df['Height_m']
            print("OK 创建 Aspect_Ratio")

        # VIV Susceptibility
        if 'Damping_Ratio' in self.df.columns:
            df_features['VIV_Susceptibility'] = 1.0 / (self.df['Damping_Ratio'] + 1e-6)
            print("OK 创建 VIV_Susceptibility")

        # Reduced Velocity
        if all(col in self.df.columns for col in ['Critical_Wind_Speed_ms', 'Natural_Freq_Hz', 'Width_m']):
            df_features['Reduced_Velocity'] = (
                self.df['Critical_Wind_Speed_ms'] / (self.df['Natural_Freq_Hz'] * self.df['Width_m'])
            )
            print("OK 创建 Reduced_Velocity")

        # ===== 新增: 利用完整特征 (196座数据没有的!) =====

        # 模态特征1: 多阶频率比
        if all(col in self.df.columns for col in ['First_Freq_Hz', 'Second_Freq_Hz']):
            df_features['Freq_Ratio_2nd_1st'] = (
                self.df['Second_Freq_Hz'] / (self.df['First_Freq_Hz'] + 1e-6)
            )
            print("OK 创建 Freq_Ratio_2nd_1st (模态频率比) *** 新特征 ***")

        # 模态特征2: 自振频率与一阶频率差异
        if all(col in self.df.columns for col in ['Natural_Freq_Hz', 'First_Freq_Hz']):
            df_features['Freq_Diff_Natural_First'] = (
                self.df['Natural_Freq_Hz'] - self.df['First_Freq_Hz']
            )
            print("OK 创建 Freq_Diff_Natural_First (频率差) *** 新特征 ***")

        # 刚度参数 (利用一阶频率)
        if all(col in self.df.columns for col in ['First_Freq_Hz', 'Span_m']):
            df_features['Stiffness_Parameter_1st'] = (
                self.df['First_Freq_Hz'] * np.sqrt(self.df['Span_m'])
            )
            print("OK 创建 Stiffness_Parameter_1st (基于一阶频率) *** 新特征 ***")

        # 气动力特征1: 升阻比
        if all(col in self.df.columns for col in ['Lift_Coefficient', 'Drag_Coefficient']):
            df_features['Lift_Drag_Ratio'] = (
                self.df['Lift_Coefficient'] / (np.abs(self.df['Drag_Coefficient']) + 1e-6)
            )
            print("OK 创建 Lift_Drag_Ratio (升阻比) *** 新特征 ***")

        # 气动力特征2: 总气动力
        if all(col in self.df.columns for col in ['Lift_Coefficient', 'Drag_Coefficient']):
            df_features['Total_Aero_Force'] = np.sqrt(
                self.df['Lift_Coefficient']**2 + self.df['Drag_Coefficient']**2
            )
            print("OK 创建 Total_Aero_Force (总气动力) *** 新特征 ***")

        # 尺度参数: 跨宽比
        if all(col in self.df.columns for col in ['Span_m', 'Width_m']):
            df_features['Span_Width_Ratio'] = self.df['Span_m'] / self.df['Width_m']
            print("OK 创建 Span_Width_Ratio")

        # ===== 交互特征 =====

        # 阻尼-跨度交互
        if all(col in df_features.columns for col in ['Damping_Ratio', 'Span_m']):
            df_features['Damping_x_Span'] = df_features['Damping_Ratio'] * df_features['Span_m']
            print("OK 创建 Damping_x_Span")

        # 频率-宽度交互
        if all(col in df_features.columns for col in ['Natural_Freq_Hz', 'Width_m']):
            df_features['Freq_x_Width'] = df_features['Natural_Freq_Hz'] * df_features['Width_m']
            print("OK 创建 Freq_x_Width")

        # Scruton-约化风速交互
        if all(col in df_features.columns for col in ['Scruton_Number', 'Reduced_Velocity']):
            df_features['Scruton_x_ReVel'] = df_features['Scruton_Number'] * df_features['Reduced_Velocity']
            print("OK 创建 Scruton_x_ReVel")

        # 阻尼-风速交互
        if all(col in df_features.columns for col in ['Damping_Ratio', 'Critical_Wind_Speed_ms']):
            df_features['Damping_x_WindSpeed'] = df_features['Damping_Ratio'] * df_features['Critical_Wind_Speed_ms']
            print("OK 创建 Damping_x_WindSpeed")

        # 气动力-约化风速交互 (新)
        if all(col in df_features.columns for col in ['Total_Aero_Force', 'Reduced_Velocity']):
            df_features['Aero_x_ReVel'] = df_features['Total_Aero_Force'] * df_features['Reduced_Velocity']
            print("OK 创建 Aero_x_ReVel (气动力-风速耦合) *** 新特征 ***")

        # 升阻比-Scruton交互 (新)
        if all(col in df_features.columns for col in ['Lift_Drag_Ratio', 'Scruton_Number']):
            df_features['LiftDrag_x_Scruton'] = df_features['Lift_Drag_Ratio'] * df_features['Scruton_Number']
            print("OK 创建 LiftDrag_x_Scruton (升阻比-Scruton耦合) *** 新特征 ***")

        # ===== 非线性特征 =====

        # 阻尼平方
        if 'Damping_Ratio' in df_features.columns:
            df_features['Damping_squared'] = df_features['Damping_Ratio'] ** 2
            print("OK 创建 Damping_squared")

        # 跨度平方根
        if 'Span_m' in df_features.columns:
            df_features['Span_sqrt'] = np.sqrt(df_features['Span_m'])
            print("OK 创建 Span_sqrt")

        # 宽高比平方
        if 'Aspect_Ratio' in df_features.columns:
            df_features['Aspect_Ratio_squared'] = df_features['Aspect_Ratio'] ** 2
            print("OK 创建 Aspect_Ratio_squared")

        return df_features

    def prepare_data(self, df_features):
        """准备训练数据"""
        # 移除缺失值
        before_drop = len(df_features)
        df_features = df_features.dropna()
        after_drop = len(df_features)

        if before_drop > after_drop:
            print(f"\nWARNING 移除 {before_drop - after_drop} 行含缺失值的样本")

        # 目标变量
        self.y = self.df.loc[df_features.index, 'Max_Amplitude_mm'].values.reshape(-1, 1)

        # 特征矩阵
        self.X = df_features.values
        self.feature_names = df_features.columns.tolist()

        print(f"\n最终特征集: {len(self.feature_names)} 个特征, {len(self.X)} 个样本")
        print(f"\n前10个特征:")
        for i, name in enumerate(self.feature_names[:10], 1):
            print(f"  {i}. {name}")
        if len(self.feature_names) > 10:
            print(f"  ... (共{len(self.feature_names)}个)")

        print(f"\n目标变量 Max_Amplitude_mm:")
        print(f"  范围: {self.y.min():.1f} - {self.y.max():.1f} mm")
        print(f"  均值: {self.y.mean():.2f} mm")
        print(f"  标准差: {self.y.std():.2f} mm")

        return self.X, self.y

    def evaluate_model_kfold(self, model, model_name, k=5):
        """K-Fold交叉验证评估"""
        print("\n" + "="*80)
        print(f"评估模型: {model_name}")
        print("="*80)

        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X), 1):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            # 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # 训练
            model.fit(X_train_scaled, y_train.ravel())

            # 预测
            y_val_pred = model.predict(X_val_scaled).reshape(-1, 1)

            # 评估
            val_r2 = 1 - np.sum((y_val - y_val_pred)**2) / np.sum((y_val - np.mean(y_val))**2)
            val_rmse = np.sqrt(np.mean((y_val - y_val_pred)**2))
            val_mae = np.mean(np.abs(y_val - y_val_pred))

            fold_results.append({
                'fold': fold,
                'val_r2': val_r2,
                'val_rmse': val_rmse,
                'val_mae': val_mae
            })

            print(f"Fold {fold}/{k}: R2={val_r2:.4f}, RMSE={val_rmse:.2f} mm")

        # 汇总
        val_r2_scores = [r['val_r2'] for r in fold_results]
        val_rmse_scores = [r['val_rmse'] for r in fold_results]
        val_mae_scores = [r['val_mae'] for r in fold_results]

        print(f"\n{model_name} 性能汇总:")
        print(f"  验证集R2:   {np.mean(val_r2_scores):.4f} ± {np.std(val_r2_scores):.4f}")
        print(f"  验证集RMSE: {np.mean(val_rmse_scores):.2f} ± {np.std(val_rmse_scores):.2f} mm")
        print(f"  验证集MAE:  {np.mean(val_mae_scores):.2f} ± {np.std(val_mae_scores):.2f} mm")

        return {
            'model_name': model_name,
            'mean_r2': np.mean(val_r2_scores),
            'std_r2': np.std(val_r2_scores),
            'mean_rmse': np.mean(val_rmse_scores),
            'std_rmse': np.std(val_rmse_scores),
            'mean_mae': np.mean(val_mae_scores),
            'std_mae': np.std(val_mae_scores)
        }


def main():
    print("="*80)
    print("85座高质量数据 vs 196座扩充数据 - 对比实验")
    print("="*80)
    print("核心问题: 数据质量(完整特征) vs 数据数量(样本更多)")
    print("="*80)

    # ===== 实验A: 85座完整特征 =====
    print("\n" + "#"*80)
    print("# 实验A: 85座高质量数据 (完整特征集)")
    print("#"*80)

    model_85 = FullFeatureVIVModel('../data/enhanced_bridge_dataset.csv')
    feature_cols_85 = model_85.load_data()
    df_features_85 = model_85.create_full_physics_features(feature_cols_85)
    X_85, y_85 = model_85.prepare_data(df_features_85)

    # 测试两种模型
    ridge_85 = Ridge(alpha=10.0)
    result_ridge_85 = model_85.evaluate_model_kfold(ridge_85, "岭回归(85座完整特征)", k=5)

    gbr_85 = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=10,
        random_state=42
    )
    result_gbr_85 = model_85.evaluate_model_kfold(gbr_85, "Gradient Boosting(85座完整特征)", k=5)

    # ===== 实验B: 196座部分特征 (对比基准) =====
    print("\n" + "#"*80)
    print("# 实验B: 196座扩充数据 (部分特征,作为对比)")
    print("#"*80)
    print("(从之前的实验结果中引用)")
    print("\n岭回归(196座增强特征):")
    print("  验证集R2:   0.5092 ± 0.0932")
    print("  验证集RMSE: 14.94 ± 1.34 mm")

    print("\nGradient Boosting(196座增强特征):")
    print("  验证集R2:   0.5296 ± 0.1645")
    print("  验证集RMSE: 14.63 ± 2.88 mm")

    result_ridge_196 = {
        'model_name': '岭回归(196座增强特征)',
        'mean_r2': 0.5092,
        'std_r2': 0.0932,
        'mean_rmse': 14.94,
        'std_rmse': 1.34
    }

    result_gbr_196 = {
        'model_name': 'Gradient Boosting(196座增强特征)',
        'mean_r2': 0.5296,
        'std_r2': 0.1645,
        'mean_rmse': 14.63,
        'std_rmse': 2.88
    }

    # ===== 最终对比 =====
    print("\n" + "="*80)
    print("数据质量 vs 数据数量 - 最终对比")
    print("="*80)

    results = [
        ('85座完整特征', result_ridge_85, result_gbr_85),
        ('196座部分特征', result_ridge_196, result_gbr_196)
    ]

    print(f"\n{'数据集':<20} {'模型':<30} {'R2':<20} {'RMSE':<20}")
    print("-"*90)

    for dataset_name, ridge_result, gbr_result in results:
        print(f"{dataset_name:<20} {'岭回归':<30} "
              f"{ridge_result['mean_r2']:.4f}±{ridge_result['std_r2']:.4f}    "
              f"{ridge_result['mean_rmse']:.2f}±{ridge_result['std_rmse']:.2f}mm")
        print(f"{'':<20} {'Gradient Boosting':<30} "
              f"{gbr_result['mean_r2']:.4f}±{gbr_result['std_r2']:.4f}    "
              f"{gbr_result['mean_rmse']:.2f}±{gbr_result['std_rmse']:.2f}mm")
        print("-"*90)

    # 分析结论
    print("\n" + "="*80)
    print("关键发现")
    print("="*80)

    if result_ridge_85['mean_r2'] > result_ridge_196['mean_r2']:
        diff = result_ridge_85['mean_r2'] - result_ridge_196['mean_r2']
        pct = diff / result_ridge_196['mean_r2'] * 100
        print(f"\n1. 85座完整特征 > 196座部分特征")
        print(f"   岭回归R2提升: {diff:.4f} ({pct:+.1f}%)")
        print(f"   结论: 数据质量(完整特征) > 数据数量")
    else:
        diff = result_ridge_196['mean_r2'] - result_ridge_85['mean_r2']
        pct = diff / result_ridge_85['mean_r2'] * 100
        print(f"\n1. 196座部分特征 > 85座完整特征")
        print(f"   岭回归R2提升: {diff:.4f} ({pct:+.1f}%)")
        print(f"   结论: 数据数量 > 数据质量(样本量优势)")

    print(f"\n2. 特征数对比:")
    print(f"   85座: {len(model_85.feature_names)} 个特征 (包含First_Freq, Drag/Lift等)")
    print(f"   196座: 20 个特征 (缺失First_Freq, Drag/Lift等)")

    print(f"\n3. 样本数对比:")
    print(f"   85座: {len(X_85)} 个样本")
    print(f"   196座: 190 个样本 (2.24倍)")

    print("\n" + "="*80)
    print("实验完成!")
    print("="*80)

    return result_ridge_85, result_gbr_85, result_ridge_196, result_gbr_196


if __name__ == '__main__':
    results = main()
