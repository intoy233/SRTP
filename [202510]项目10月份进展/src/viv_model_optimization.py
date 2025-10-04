#!/usr/bin/env python3
"""
VIV模型优化 - 务实的性能提升方案
目标: R2从0.46提升到0.55-0.60
方法: 交互特征 + 集成学习
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')


class VIVModelOptimizer:
    """VIV模型优化器"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None

    def load_and_clean_data(self):
        """加载并清洗数据"""
        print("="*80)
        print("步骤1: 加载数据")
        print("="*80)

        self.df = pd.read_csv(self.data_path)
        print(f"\n原始数据集: {len(self.df)} 座桥梁")

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

        print(f"\n保留的原始特征: {len(feature_cols)} 个")

        return feature_cols

    def create_base_physics_features(self, feature_cols):
        """创建基础物理特征"""
        print("\n" + "="*80)
        print("步骤2: 基础物理特征工程")
        print("="*80)

        df_features = self.df[feature_cols].copy()

        # Scruton Number
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
            median_vr = df_features['Reduced_Velocity'].median()
            df_features['Reduced_Velocity'].fillna(median_vr, inplace=True)
            print(f"OK 创建 Reduced_Velocity (缺失值填充中位数 {median_vr:.2f})")

        # Stiffness Parameter
        if all(col in self.df.columns for col in ['Natural_Freq_Hz', 'Span_m']):
            df_features['Stiffness_Parameter'] = self.df['Natural_Freq_Hz'] * np.sqrt(self.df['Span_m'])
            print("OK 创建 Stiffness_Parameter")

        return df_features

    def create_interaction_features(self, df_features):
        """创建交互特征 - 性能提升关键"""
        print("\n" + "="*80)
        print("步骤3: 交互特征工程 (核心优化)")
        print("="*80)

        # 交互特征1: 阻尼-跨度交互
        if all(col in df_features.columns for col in ['Damping_Ratio', 'Span_m']):
            df_features['Damping_x_Span'] = df_features['Damping_Ratio'] * df_features['Span_m']
            print("OK 创建 Damping_x_Span (阻尼随跨度的效应)")

        # 交互特征2: 频率-宽度交互
        if all(col in df_features.columns for col in ['Natural_Freq_Hz', 'Width_m']):
            df_features['Freq_x_Width'] = df_features['Natural_Freq_Hz'] * df_features['Width_m']
            print("OK 创建 Freq_x_Width (频率-宽度耦合)")

        # 交互特征3: Scruton-约化风速交互
        if all(col in df_features.columns for col in ['Scruton_Number', 'Reduced_Velocity']):
            df_features['Scruton_x_ReVel'] = df_features['Scruton_Number'] * df_features['Reduced_Velocity']
            print("OK 创建 Scruton_x_ReVel (核心参数交互)")

        # 交互特征4: 阻尼-风速交互
        if all(col in df_features.columns for col in ['Damping_Ratio', 'Critical_Wind_Speed_ms']):
            df_features['Damping_x_WindSpeed'] = df_features['Damping_Ratio'] * df_features['Critical_Wind_Speed_ms']
            print("OK 创建 Damping_x_WindSpeed (阻尼-风速耦合)")

        # 非线性特征1: 阻尼平方
        if 'Damping_Ratio' in df_features.columns:
            df_features['Damping_squared'] = df_features['Damping_Ratio'] ** 2
            print("OK 创建 Damping_squared (非线性阻尼效应)")

        # 非线性特征2: 跨度平方根
        if 'Span_m' in df_features.columns:
            df_features['Span_sqrt'] = np.sqrt(df_features['Span_m'])
            print("OK 创建 Span_sqrt (跨度影响递减)")

        # 非线性特征3: 宽高比平方
        if 'Aspect_Ratio' in df_features.columns:
            df_features['Aspect_Ratio_squared'] = df_features['Aspect_Ratio'] ** 2
            print("OK 创建 Aspect_Ratio_squared (宽高比非线性)")

        # 比例特征: 刚度与阻尼比
        if all(col in df_features.columns for col in ['Stiffness_Parameter', 'Damping_Ratio']):
            df_features['Stiffness_Damping_Ratio'] = df_features['Stiffness_Parameter'] / (df_features['Damping_Ratio'] + 1e-6)
            print("OK 创建 Stiffness_Damping_Ratio (刚度阻尼比)")

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
            y_train_pred = model.predict(X_train_scaled).reshape(-1, 1)
            y_val_pred = model.predict(X_val_scaled).reshape(-1, 1)

            # 评估
            train_r2 = 1 - np.sum((y_train - y_train_pred)**2) / np.sum((y_train - np.mean(y_train))**2)
            val_r2 = 1 - np.sum((y_val - y_val_pred)**2) / np.sum((y_val - np.mean(y_val))**2)

            train_rmse = np.sqrt(np.mean((y_train - y_train_pred)**2))
            val_rmse = np.sqrt(np.mean((y_val - y_val_pred)**2))

            train_mae = np.mean(np.abs(y_train - y_train_pred))
            val_mae = np.mean(np.abs(y_val - y_val_pred))

            fold_results.append({
                'fold': fold,
                'train_r2': train_r2,
                'val_r2': val_r2,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'train_mae': train_mae,
                'val_mae': val_mae
            })

            print(f"Fold {fold}/{k}: 验证集 R2={val_r2:.4f}, RMSE={val_rmse:.2f} mm")

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
            'std_mae': np.std(val_mae_scores),
            'fold_results': fold_results
        }


def main():
    print("="*80)
    print("VIV模型优化实验 - 务实的性能提升方案")
    print("="*80)
    print("目标: R2从0.46提升到0.55-0.60")
    print("方法: 交互特征 + 集成学习")
    print("="*80)

    # 初始化
    optimizer = VIVModelOptimizer('../data/final_bridge_dataset.csv')

    # 加载数据
    feature_cols = optimizer.load_and_clean_data()

    # 创建基础物理特征
    df_base = optimizer.create_base_physics_features(feature_cols)

    print("\n" + "#"*80)
    print("# 实验1: 基线模型 (仅基础物理特征)")
    print("#"*80)

    X_base, y_base = optimizer.prepare_data(df_base)

    baseline_model = Ridge(alpha=10.0)
    baseline_result = optimizer.evaluate_model_kfold(baseline_model, "基线岭回归 (基础特征)", k=5)

    # 创建交互特征
    df_enhanced = optimizer.create_interaction_features(df_base.copy())

    print("\n" + "#"*80)
    print("# 实验2: 增强特征 (基础 + 交互特征)")
    print("#"*80)

    X_enhanced, y_enhanced = optimizer.prepare_data(df_enhanced)

    enhanced_ridge = Ridge(alpha=10.0)
    enhanced_result = optimizer.evaluate_model_kfold(enhanced_ridge, "岭回归 (增强特征)", k=5)

    print("\n" + "#"*80)
    print("# 实验3: Bagging Ridge (集成学习)")
    print("#"*80)

    bagging_model = BaggingRegressor(
        estimator=Ridge(alpha=10.0),
        n_estimators=50,
        max_samples=0.8,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    )

    bagging_result = optimizer.evaluate_model_kfold(bagging_model, "Bagging Ridge (50棵树)", k=5)

    print("\n" + "#"*80)
    print("# 实验4: Gradient Boosting")
    print("#"*80)

    gbr_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    )

    gbr_result = optimizer.evaluate_model_kfold(gbr_model, "Gradient Boosting (100棵树)", k=5)

    # 最终对比
    print("\n" + "="*80)
    print("性能提升对比总结")
    print("="*80)

    results = [baseline_result, enhanced_result, bagging_result, gbr_result]

    print(f"\n{'模型':<30} {'验证R2':<15} {'验证RMSE':<15} {'vs基线':<15}")
    print("-"*80)

    baseline_r2 = baseline_result['mean_r2']

    for result in results:
        r2_improve = result['mean_r2'] - baseline_r2
        r2_improve_pct = (r2_improve / baseline_r2) * 100 if baseline_r2 > 0 else 0

        print(f"{result['model_name']:<30} "
              f"{result['mean_r2']:.4f}±{result['std_r2']:.4f}  "
              f"{result['mean_rmse']:.2f}±{result['std_rmse']:.2f}mm  "
              f"{r2_improve:+.4f} ({r2_improve_pct:+.1f}%)")

    # 找出最佳模型
    best_result = max(results, key=lambda x: x['mean_r2'])

    print("\n" + "="*80)
    print("最佳模型")
    print("="*80)
    print(f"模型: {best_result['model_name']}")
    print(f"验证集R2: {best_result['mean_r2']:.4f} ± {best_result['std_r2']:.4f}")
    print(f"验证集RMSE: {best_result['mean_rmse']:.2f} ± {best_result['std_rmse']:.2f} mm")
    print(f"相比基线提升: {(best_result['mean_r2'] - baseline_r2):.4f} ({((best_result['mean_r2'] - baseline_r2)/baseline_r2*100):+.1f}%)")

    print("\n" + "="*80)
    print("实验完成!")
    print("="*80)

    return results, best_result


if __name__ == '__main__':
    results, best_result = main()
