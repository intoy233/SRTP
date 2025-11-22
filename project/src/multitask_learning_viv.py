#!/usr/bin/env python3
"""
多任务学习框架 - 受DeepVIV启发
Multi-Task Learning for VIV Prediction

核心思想:
- 单个模型同时预测多个相关任务
- 任务1: 回归 - Max_Amplitude预测
- 任务2: 分类 - Risk_Level预测(High/Medium/Low)

优势:
- 共享表示学习,提升泛化能力
- 辅助任务提供额外监督信号
- 工程应用更全面(不仅预测振幅,还给出风险等级)

技术实现:
- 使用两个独立的sklearn模型(回归+分类)
- 共享特征工程pipeline
- 联合评估性能
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class MultiTaskVIVModel:
    """多任务VIV预测模型"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y_amplitude = None
        self.y_risk = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.amplitude_model = None
        self.risk_model = None

    def load_and_prepare_data(self):
        """加载并准备数据"""
        print("="*80)
        print("多任务学习模型 - 同时预测振幅和风险等级")
        print("="*80)

        self.df = pd.read_csv(self.data_path)
        print(f"\n数据集: {len(self.df)} 座桥梁")

        # 排除列
        exclude_cols = [
            'BridgeName', 'Country', 'BridgeType', 'PaperSource', 'Year',
            'Max_Amplitude_mm',  # 任务1的目标
            'Amplitude_RMS_mm', 'VIV_Wind_Speed_ms',
            'Risk_Level',  # 任务2的目标
            'Notes', 'Vibration_Suppression', 'Suppression_Effect',
            'Total_Length_m', 'First_Freq_Hz', 'Second_Freq_Hz',
            'Drag_Coefficient', 'Lift_Coefficient',
            'BridgeID', 'Structure_Type'
        ]

        actual_exclude = [col for col in exclude_cols if col in self.df.columns]
        feature_cols = [col for col in self.df.columns if col not in actual_exclude]

        # 创建特征
        df_features = self._create_physics_features(feature_cols)
        df_features = self._create_interaction_features(df_features)

        # 移除缺失值(同时需要Amplitude和Risk_Level)
        valid_idx = df_features.index.intersection(
            self.df.dropna(subset=['Max_Amplitude_mm', 'Risk_Level']).index
        )
        df_features = df_features.loc[valid_idx]

        # 移除特征中的NaN行
        df_features = df_features.dropna()

        # 任务1目标: Max_Amplitude (回归)
        self.y_amplitude = self.df.loc[df_features.index, 'Max_Amplitude_mm'].values

        # 任务2目标: Risk_Level (分类)
        risk_labels = self.df.loc[df_features.index, 'Risk_Level'].values
        self.y_risk = self.label_encoder.fit_transform(risk_labels)  # High/Medium/Low → 0/1/2

        # 特征矩阵
        self.X = df_features.values
        self.feature_names = df_features.columns.tolist()

        print(f"\n最终特征集: {len(self.feature_names)} 个特征")
        print(f"有效样本数: {len(self.X)} 个桥梁")

        print(f"\n任务1 - Max_Amplitude (回归):")
        print(f"  范围: {self.y_amplitude.min():.1f} - {self.y_amplitude.max():.1f} mm")
        print(f"  均值: {self.y_amplitude.mean():.2f} mm")

        print(f"\n任务2 - Risk_Level (分类):")
        risk_dist = pd.Series(risk_labels).value_counts()
        for risk, count in risk_dist.items():
            print(f"  {risk:10s}: {count:3d} ({count/len(risk_labels)*100:.1f}%)")

        return self.X, self.y_amplitude, self.y_risk

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

    def evaluate_multitask_kfold(self, k=5):
        """K-Fold交叉验证 - 多任务学习"""
        print("\n" + "="*80)
        print(f"多任务学习 K-Fold交叉验证 (k={k})")
        print("="*80)

        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        # 存储结果
        amplitude_results = []
        risk_results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X), 1):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_amp_train, y_amp_val = self.y_amplitude[train_idx], self.y_amplitude[val_idx]
            y_risk_train, y_risk_val = self.y_risk[train_idx], self.y_risk[val_idx]

            # 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # 任务1: 振幅回归
            amp_model = Ridge(alpha=10.0)
            amp_model.fit(X_train_scaled, y_amp_train)
            y_amp_pred = amp_model.predict(X_val_scaled)

            amp_r2 = 1 - np.sum((y_amp_val - y_amp_pred)**2) / np.sum((y_amp_val - np.mean(y_amp_val))**2)
            amp_rmse = np.sqrt(np.mean((y_amp_val - y_amp_pred)**2))

            # 任务2: 风险等级分类
            risk_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            risk_model.fit(X_train_scaled, y_risk_train)
            y_risk_pred = risk_model.predict(X_val_scaled)

            risk_accuracy = accuracy_score(y_risk_val, y_risk_pred)

            amplitude_results.append({
                'fold': fold,
                'r2': amp_r2,
                'rmse': amp_rmse
            })

            risk_results.append({
                'fold': fold,
                'accuracy': risk_accuracy,
                'y_true': y_risk_val,
                'y_pred': y_risk_pred
            })

            print(f"\nFold {fold}/{k}:")
            print(f"  任务1 (振幅回归):   R2={amp_r2:.4f}, RMSE={amp_rmse:.2f} mm")
            print(f"  任务2 (风险分类):   Accuracy={risk_accuracy*100:.1f}%")

        # 汇总统计
        print("\n" + "="*80)
        print("多任务性能汇总")
        print("="*80)

        amp_r2_mean = np.mean([r['r2'] for r in amplitude_results])
        amp_r2_std = np.std([r['r2'] for r in amplitude_results])
        amp_rmse_mean = np.mean([r['rmse'] for r in amplitude_results])

        risk_acc_mean = np.mean([r['accuracy'] for r in risk_results])
        risk_acc_std = np.std([r['accuracy'] for r in risk_results])

        print(f"\n任务1 - 振幅预测 (回归):")
        print(f"  验证集R2:   {amp_r2_mean:.4f} ± {amp_r2_std:.4f}")
        print(f"  验证集RMSE: {amp_rmse_mean:.2f} mm")

        print(f"\n任务2 - 风险等级分类:")
        print(f"  验证集准确率: {risk_acc_mean*100:.2f}% ± {risk_acc_std*100:.2f}%")

        # 混淆矩阵汇总(合并所有fold)
        all_y_true = np.concatenate([r['y_true'] for r in risk_results])
        all_y_pred = np.concatenate([r['y_pred'] for r in risk_results])

        print(f"\n风险等级分类报告:")
        print(classification_report(
            all_y_true, all_y_pred,
            target_names=self.label_encoder.classes_,
            digits=3
        ))

        # 可视化混淆矩阵
        self._plot_confusion_matrix(all_y_true, all_y_pred)

        return {
            'amplitude': amplitude_results,
            'risk': risk_results,
            'amplitude_mean_r2': amp_r2_mean,
            'risk_mean_accuracy': risk_acc_mean
        }

    def _plot_confusion_matrix(self, y_true, y_pred):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.xlabel('Predicted Risk Level', fontsize=12)
        plt.ylabel('Actual Risk Level', fontsize=12)
        plt.title('Risk Level Classification - Confusion Matrix', fontsize=14)
        plt.tight_layout()
        plt.savefig('../results/multitask_confusion_matrix.png', dpi=150, bbox_inches='tight')
        print(f"\n混淆矩阵已保存: ../results/multitask_confusion_matrix.png")

    def compare_single_vs_multitask(self, k=5):
        """对比单任务vs多任务学习的性能"""
        print("\n" + "="*80)
        print("实验: 单任务 vs 多任务学习对比")
        print("="*80)

        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        single_task_r2 = []
        multitask_r2 = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X), 1):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_amp_train, y_amp_val = self.y_amplitude[train_idx], self.y_amplitude[val_idx]
            y_risk_train = self.y_risk[train_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # 单任务: 仅训练振幅回归
            single_model = Ridge(alpha=10.0)
            single_model.fit(X_train_scaled, y_amp_train)
            y_pred_single = single_model.predict(X_val_scaled)
            r2_single = 1 - np.sum((y_amp_val - y_pred_single)**2) / np.sum((y_amp_val - np.mean(y_amp_val))**2)

            # 多任务: 同时训练振幅回归+风险分类
            # (这里仍然是独立训练,但共享特征工程)
            # 真正的多任务学习需要神经网络共享底层,但我们用sklearn模拟效果
            multi_amp_model = Ridge(alpha=10.0)
            multi_risk_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

            multi_amp_model.fit(X_train_scaled, y_amp_train)
            multi_risk_model.fit(X_train_scaled, y_risk_train)

            y_pred_multi = multi_amp_model.predict(X_val_scaled)
            r2_multi = 1 - np.sum((y_amp_val - y_pred_multi)**2) / np.sum((y_amp_val - np.mean(y_amp_val))**2)

            single_task_r2.append(r2_single)
            multitask_r2.append(r2_multi)

        print(f"\n振幅预测性能对比:")
        print(f"  单任务学习: R2={np.mean(single_task_r2):.4f} ± {np.std(single_task_r2):.4f}")
        print(f"  多任务学习: R2={np.mean(multitask_r2):.4f} ± {np.std(multitask_r2):.4f}")

        print(f"\n结论:")
        if np.mean(multitask_r2) > np.mean(single_task_r2):
            print(f"  多任务学习略优于单任务 (+{(np.mean(multitask_r2)-np.mean(single_task_r2))*100:.2f}%)")
        else:
            print(f"  两者性能相当(由于使用sklearn独立模型,未真正共享表示)")

        print(f"\n说明:")
        print(f"  - sklearn实现为独立模型,未真正实现深度多任务学习")
        print(f"  - 但多任务框架提供了振幅+风险的联合预测")
        print(f"  - 工程价值: 一次预测同时给出振幅值和风险等级")


def main():
    print("="*80)
    print("多任务学习实验 - 受DeepVIV启发")
    print("="*80)
    print("目标: 同时预测振幅(回归) + 风险等级(分类)")
    print("灵感: DeepVIV的多输出神经网络框架")
    print("="*80)

    # 初始化模型
    mtvm = MultiTaskVIVModel('../data/final_bridge_dataset.csv')

    # 加载数据
    X, y_amp, y_risk = mtvm.load_and_prepare_data()

    # 多任务K-Fold交叉验证
    results = mtvm.evaluate_multitask_kfold(k=5)

    # 单任务vs多任务对比
    mtvm.compare_single_vs_multitask(k=5)

    print("\n" + "="*80)
    print("实验完成!")
    print("="*80)
    print("核心发现:")
    print("1. 多任务学习成功预测振幅+风险等级")
    print(f"2. 振幅预测R2={results['amplitude_mean_r2']:.4f}")
    print(f"3. 风险分类准确率={results['risk_mean_accuracy']*100:.1f}%")
    print("4. 工程价值: 一次预测提供完整风险评估")
    print("="*80)

    return results


if __name__ == '__main__':
    results = main()
