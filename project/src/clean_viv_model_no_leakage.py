#!/usr/bin/env python3
"""
干净的VIV预测模型 - 无数据泄露版本
排除Amplitude_RMS_mm等泄露特征,使用K-Fold交叉验证
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


class CleanVIVModel:
    """无数据泄露的VIV预测模型"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None

    def load_and_clean_data(self):
        """加载并清洗数据,排除泄露特征"""
        print("="*80)
        print("步骤1: 加载数据并排除数据泄露特征")
        print("="*80)

        self.df = pd.read_csv(self.data_path)
        print(f"\n原始数据集: {len(self.df)} 座桥梁, {self.df.shape[1]} 列")

        # 定义排除的列
        exclude_cols = [
            # 标识列
            'BridgeName', 'Country', 'BridgeType', 'PaperSource', 'Year',
            # 目标变量
            'Max_Amplitude_mm',
            # 数据泄露特征 (与Max_Amplitude高度相关)
            'Amplitude_RMS_mm',  # 相关系数0.99
            'VIV_Wind_Speed_ms',  # VIV发生时的风速,是结果不是输入
            # 辅助列
            'Risk_Level', 'Notes', 'Vibration_Suppression', 'Suppression_Effect',
            # 高缺失列 (缺失率>50%)
            'Total_Length_m', 'First_Freq_Hz', 'Second_Freq_Hz',
            'Drag_Coefficient', 'Lift_Coefficient',
            # ID列
            'BridgeID', 'Structure_Type'
        ]

        # 检查哪些排除列实际存在
        actual_exclude = [col for col in exclude_cols if col in self.df.columns]
        print(f"\n排除的列 ({len(actual_exclude)}):")
        for col in actual_exclude:
            if col in self.df.columns:
                missing_pct = self.df[col].isna().sum() / len(self.df) * 100
                print(f"  - {col:<30} (缺失率: {missing_pct:>5.1f}%)")

        # 保留的特征列
        feature_cols = [col for col in self.df.columns if col not in actual_exclude]
        print(f"\n保留的原始特征 ({len(feature_cols)}):")
        for col in feature_cols:
            print(f"  - {col}")

        return feature_cols

    def create_physics_features(self, feature_cols):
        """创建物理派生特征(无数据泄露)"""
        print("\n" + "="*80)
        print("步骤2: 物理特征工程")
        print("="*80)

        df_features = self.df[feature_cols].copy()

        # Scruton Number - VIV最重要的无量纲参数
        if all(col in self.df.columns for col in ['Damping_Ratio', 'Width_m', 'Height_m']):
            df_features['Scruton_Number'] = (
                self.df['Damping_Ratio'] * (self.df['Width_m'] / self.df['Height_m']) * 100
            )
            print("OK 创建 Scruton_Number = Damping * (Width/Height) * 100")

        # Aspect Ratio - 宽高比
        if all(col in self.df.columns for col in ['Width_m', 'Height_m']):
            df_features['Aspect_Ratio'] = self.df['Width_m'] / self.df['Height_m']
            print("OK 创建 Aspect_Ratio = Width / Height")

        # VIV Susceptibility - VIV敏感性
        if 'Damping_Ratio' in self.df.columns:
            df_features['VIV_Susceptibility'] = 1.0 / (self.df['Damping_Ratio'] + 1e-6)
            print("OK 创建 VIV_Susceptibility = 1 / Damping")

        # Reduced Velocity - 折减风速 (如果有风速数据)
        if all(col in self.df.columns for col in ['Critical_Wind_Speed_ms', 'Natural_Freq_Hz', 'Width_m']):
            df_features['Reduced_Velocity'] = (
                self.df['Critical_Wind_Speed_ms'] / (self.df['Natural_Freq_Hz'] * self.df['Width_m'])
            )
            # 处理缺失值
            median_vr = df_features['Reduced_Velocity'].median()
            df_features['Reduced_Velocity'].fillna(median_vr, inplace=True)
            print(f"OK 创建 Reduced_Velocity = Vcr / (f * B), 缺失值填充中位数 {median_vr:.2f}")

        # Stiffness Parameter - 刚度参数
        if all(col in self.df.columns for col in ['Natural_Freq_Hz', 'Span_m']):
            df_features['Stiffness_Parameter'] = self.df['Natural_Freq_Hz'] * np.sqrt(self.df['Span_m'])
            print("OK 创建 Stiffness_Parameter = Freq * sqrt(Span)")

        # 移除仍然存在的缺失值行
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
        print(f"特征列表:")
        for i, name in enumerate(self.feature_names, 1):
            print(f"  {i}. {name}")

        print(f"\n目标变量 Max_Amplitude_mm:")
        print(f"  范围: {self.y.min():.1f} - {self.y.max():.1f} mm")
        print(f"  均值: {self.y.mean():.2f} mm")
        print(f"  标准差: {self.y.std():.2f} mm")

        return self.X, self.y

    def k_fold_cross_validation(self, k=5, alpha=10.0):
        """K-Fold交叉验证"""
        print("\n" + "="*80)
        print(f"步骤3: {k}-Fold 交叉验证 (alpha={alpha})")
        print("="*80)

        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        fold_results = []

        print(f"\n开始训练 {k} 折...")
        print("-"*80)

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X), 1):
            # 划分数据
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            # 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # 训练岭回归
            model = Ridge(alpha=alpha)
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
                'val_mae': val_mae,
                'model': model,
                'scaler': scaler
            })

            print(f"Fold {fold}/{k}:")
            print(f"  训练集: R2={train_r2:.4f}, RMSE={train_rmse:.2f} mm, MAE={train_mae:.2f} mm")
            print(f"  验证集: R2={val_r2:.4f}, RMSE={val_rmse:.2f} mm, MAE={val_mae:.2f} mm")

        # 汇总统计
        print("\n" + "="*80)
        print("K-Fold交叉验证结果汇总")
        print("="*80)

        val_r2_scores = [r['val_r2'] for r in fold_results]
        val_rmse_scores = [r['val_rmse'] for r in fold_results]
        val_mae_scores = [r['val_mae'] for r in fold_results]

        print(f"\n验证集性能 (Mean ± Std):")
        print(f"  R2   = {np.mean(val_r2_scores):.4f} ± {np.std(val_r2_scores):.4f}")
        print(f"  RMSE = {np.mean(val_rmse_scores):.2f} ± {np.std(val_rmse_scores):.2f} mm")
        print(f"  MAE  = {np.mean(val_mae_scores):.2f} ± {np.std(val_mae_scores):.2f} mm")

        print(f"\nR2 各折详情:")
        for r in fold_results:
            print(f"  Fold {r['fold']}: {r['val_r2']:.4f}")

        return fold_results

    def feature_importance_analysis(self, fold_results):
        """特征重要性分析"""
        print("\n" + "="*80)
        print("步骤4: 特征重要性分析")
        print("="*80)

        # 计算所有折的平均系数
        all_coefs = np.array([r['model'].coef_ for r in fold_results])
        mean_coefs = np.mean(all_coefs, axis=0)
        std_coefs = np.std(all_coefs, axis=0)

        # 创建特征重要性DataFrame
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': mean_coefs,
            'Std': std_coefs,
            'Abs_Coefficient': np.abs(mean_coefs)
        }).sort_values('Abs_Coefficient', ascending=False)

        print(f"\n特征重要性排序 (按|系数|降序):")
        print("-"*80)
        print(f"{'排名':<5} {'特征名':<30} {'系数':<12} {'±标准差':<12}")
        print("-"*80)
        for i, row in enumerate(feature_importance.iterrows(), 1):
            _, data = row
            print(f"{i:<5} {data['Feature']:<30} {data['Coefficient']:>10.4f}  ±{data['Std']:>8.4f}")

        return feature_importance


def main():
    print("="*80)
    print("干净VIV预测模型 - 无数据泄露, K-Fold交叉验证")
    print("="*80)
    print(f"运行时间: 2025-10-04")
    print(f"目标: 评估真实预测性能,排除Amplitude_RMS等泄露特征")
    print("="*80)

    # 测试两个数据集
    datasets = [
        {
            'name': '85座高质量数据集',
            'path': '../data/enhanced_bridge_dataset.csv'
        },
        {
            'name': '196座扩充数据集',
            'path': '../data/final_bridge_dataset.csv'
        }
    ]

    all_results = {}

    for dataset in datasets:
        print("\n\n" + "#"*80)
        print(f"# 数据集: {dataset['name']}")
        print("#"*80)

        model = CleanVIVModel(dataset['path'])

        # 步骤1: 加载清洗
        feature_cols = model.load_and_clean_data()

        # 步骤2: 特征工程
        X, y = model.create_physics_features(feature_cols)

        # 步骤3: K-Fold交叉验证
        fold_results = model.k_fold_cross_validation(k=5, alpha=10.0)

        # 步骤4: 特征重要性
        feature_importance = model.feature_importance_analysis(fold_results)

        all_results[dataset['name']] = {
            'fold_results': fold_results,
            'feature_importance': feature_importance,
            'n_samples': len(X),
            'n_features': len(model.feature_names)
        }

    # 最终对比
    print("\n\n" + "="*80)
    print("最终性能对比: 85座 vs 196座")
    print("="*80)

    for dataset_name, results in all_results.items():
        val_r2_scores = [r['val_r2'] for r in results['fold_results']]
        val_rmse_scores = [r['val_rmse'] for r in results['fold_results']]

        print(f"\n{dataset_name}:")
        print(f"  样本数: {results['n_samples']}")
        print(f"  特征数: {results['n_features']}")
        print(f"  验证集R2:   {np.mean(val_r2_scores):.4f} ± {np.std(val_r2_scores):.4f}")
        print(f"  验证集RMSE: {np.mean(val_rmse_scores):.2f} ± {np.std(val_rmse_scores):.2f} mm")

    print("\n" + "="*80)
    print("实验完成! 这是无数据泄露的真实预测性能!")
    print("="*80)


if __name__ == '__main__':
    main()
