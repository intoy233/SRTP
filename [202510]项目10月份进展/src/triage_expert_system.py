#!/usr/bin/env python3
"""
分诊-专家混合预测系统 - 受吴先生启发
Triage-Expert Hybrid Prediction System

核心思想:
1. 风险分诊: 二元/三元分类器识别高风险样本(Max_Amplitude > 60mm)
2. 专家回归: 针对不同风险区间训练专门的贝叶斯回归模型
3. 端到端流程: 分类→专家选择→回归预测→输出(风险+振幅±不确定性)

优势:
- 解决单一模型在高风险样本上预测不足的问题
- 专家模型专注于特定振幅区间,性能更优
- 提供完整的风险评估报告

灵感来源: 吴先生在改进2.md中的"分诊-专家"系统设计
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')


class TriageExpertSystem:
    """分诊-专家混合预测系统"""

    def __init__(self, data_path, triage_mode='binary'):
        """
        triage_mode:
        - 'binary': 二元分类(高风险>60mm vs 常规≤60mm)
        - 'ternary': 三元分类(高>60mm, 中40-60mm, 低<40mm)
        """
        self.data_path = data_path
        self.triage_mode = triage_mode
        self.df = None
        self.X = None
        self.y_amplitude = None
        self.y_risk_labels = None
        self.feature_names = None

        # 分诊分类器
        self.triage_classifier = None
        self.scaler_triage = StandardScaler()

        # 专家回归模型
        self.expert_models = {}  # {risk_level: BayesianRidge}
        self.scaler_experts = {}  # {risk_level: StandardScaler}

        # 风险阈值
        if triage_mode == 'binary':
            self.HIGH_RISK_THRESHOLD = 60.0  # >60mm为高风险
            self.risk_levels = ['normal', 'high']
        elif triage_mode == 'ternary':
            self.LOW_THRESHOLD = 40.0
            self.HIGH_THRESHOLD = 60.0
            self.risk_levels = ['low', 'medium', 'high']
        else:
            raise ValueError(f"Unknown triage_mode: {triage_mode}")

    def load_and_prepare_data(self):
        """加载并准备数据(含幂函数变换)"""
        print("="*80)
        print("分诊-专家混合预测系统")
        print("="*80)
        print(f"模式: {self.triage_mode} 分类")
        print("灵感来源: 吴先生在改进2.md中的设计")
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
        df_features = self._create_all_features(feature_cols)

        # 移除缺失值
        df_features = df_features.dropna()

        # 目标变量: 振幅
        self.y_amplitude = self.df.loc[df_features.index, 'Max_Amplitude_mm'].values

        # 特征矩阵(基础26特征)
        X_base = df_features.values

        # 幂函数变换(X, X^2, X^3)
        X_squared = X_base ** 2
        X_cubed = X_base ** 3
        self.X = np.hstack([X_base, X_squared, X_cubed])

        self.feature_names = (
            list(df_features.columns) +
            [f"{col}_squared" for col in df_features.columns] +
            [f"{col}_cubed" for col in df_features.columns]
        )

        # 创建风险标签
        self.y_risk_labels = self._create_risk_labels(self.y_amplitude)

        print(f"\n最终特征集: {len(self.feature_names)} 个特征(含幂函数变换), {len(self.X)} 个样本")
        print(f"目标变量范围: {self.y_amplitude.min():.1f} - {self.y_amplitude.max():.1f} mm")

        # 风险分布统计
        print(f"\n风险分布:")
        unique, counts = np.unique(self.y_risk_labels, return_counts=True)
        for level, count in zip(unique, counts):
            print(f"  {level:10s}: {count:3d} ({count/len(self.y_risk_labels)*100:.1f}%)")

        return self.X, self.y_amplitude, self.y_risk_labels

    def _create_all_features(self, feature_cols):
        """创建基础+交互+Griffin特征"""
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

    def _create_risk_labels(self, y_amplitude):
        """根据振幅创建风险标签"""
        if self.triage_mode == 'binary':
            # 二元分类: >60mm为高风险
            return np.where(y_amplitude > self.HIGH_RISK_THRESHOLD, 'high', 'normal')

        elif self.triage_mode == 'ternary':
            # 三元分类: <40mm低, 40-60mm中, >60mm高
            labels = np.empty(len(y_amplitude), dtype=object)
            labels[y_amplitude < self.LOW_THRESHOLD] = 'low'
            labels[(y_amplitude >= self.LOW_THRESHOLD) & (y_amplitude <= self.HIGH_THRESHOLD)] = 'medium'
            labels[y_amplitude > self.HIGH_THRESHOLD] = 'high'
            return labels

    def train_triage_classifier(self, X_train, y_risk_train):
        """训练风险分诊分类器"""
        print("\n" + "="*80)
        print("步骤1: 训练风险分诊分类器")
        print("="*80)

        # 标准化
        X_train_scaled = self.scaler_triage.fit_transform(X_train)

        # 使用Gradient Boosting分类器(性能优于Logistic Regression)
        self.triage_classifier = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            min_samples_split=10,
            random_state=42
        )

        self.triage_classifier.fit(X_train_scaled, y_risk_train)

        # 训练集性能
        y_train_pred = self.triage_classifier.predict(X_train_scaled)
        train_acc = accuracy_score(y_risk_train, y_train_pred)

        print(f"\n分诊分类器训练完成:")
        print(f"  模型: Gradient Boosting")
        print(f"  训练集准确率: {train_acc*100:.2f}%")

        return self.triage_classifier

    def train_expert_regressors(self, X_train, y_amplitude_train, y_risk_train):
        """训练专家回归模型(每个风险级别一个)"""
        print("\n" + "="*80)
        print("步骤2: 训练专家回归模型")
        print("="*80)

        for risk_level in self.risk_levels:
            # 筛选当前风险级别的样本
            mask = (y_risk_train == risk_level)
            X_expert = X_train[mask]
            y_expert = y_amplitude_train[mask]

            if len(X_expert) == 0:
                print(f"\nWARNING 风险级别 '{risk_level}' 无训练样本,跳过")
                continue

            print(f"\n训练 '{risk_level}' 风险专家:")
            print(f"  训练样本数: {len(X_expert)}")
            print(f"  振幅范围: {y_expert.min():.1f} - {y_expert.max():.1f} mm")

            # 标准化
            scaler = StandardScaler()
            X_expert_scaled = scaler.fit_transform(X_expert)

            # 贝叶斯岭回归
            model = BayesianRidge(n_iter=300, tol=1e-3)
            model.fit(X_expert_scaled, y_expert)

            # 保存模型和scaler
            self.expert_models[risk_level] = model
            self.scaler_experts[risk_level] = scaler

            # 训练集性能
            y_pred = model.predict(X_expert_scaled)
            r2 = 1 - np.sum((y_expert - y_pred)**2) / np.sum((y_expert - np.mean(y_expert))**2)
            rmse = np.sqrt(np.mean((y_expert - y_pred)**2))

            print(f"  训练集R2: {r2:.4f}")
            print(f"  训练集RMSE: {rmse:.2f} mm")

        print(f"\n专家模型训练完成,共{len(self.expert_models)}个专家")

        return self.expert_models

    def predict(self, X, return_uncertainty=True):
        """
        端到端预测流程

        返回:
        - risk_pred: 风险等级预测
        - amplitude_pred: 振幅预测
        - amplitude_std: 不确定性(如果return_uncertainty=True)
        """
        # 步骤1: 风险分诊
        X_scaled = self.scaler_triage.transform(X)
        risk_pred = self.triage_classifier.predict(X_scaled)

        # 步骤2: 专家预测
        amplitude_pred = np.zeros(len(X))
        amplitude_std = np.zeros(len(X)) if return_uncertainty else None

        for i, risk_level in enumerate(risk_pred):
            if risk_level not in self.expert_models:
                # 如果该风险级别没有专家模型,使用全局平均值
                amplitude_pred[i] = self.y_amplitude.mean()
                if return_uncertainty:
                    amplitude_std[i] = self.y_amplitude.std()
                continue

            # 使用对应专家模型预测
            expert_model = self.expert_models[risk_level]
            expert_scaler = self.scaler_experts[risk_level]

            X_expert_scaled = expert_scaler.transform(X[i:i+1])

            if return_uncertainty:
                amp_pred, amp_std = expert_model.predict(X_expert_scaled, return_std=True)
                amplitude_pred[i] = amp_pred[0]
                amplitude_std[i] = amp_std[0]
            else:
                amplitude_pred[i] = expert_model.predict(X_expert_scaled)[0]

        if return_uncertainty:
            return risk_pred, amplitude_pred, amplitude_std
        else:
            return risk_pred, amplitude_pred

    def evaluate_kfold(self, k=5):
        """K-Fold交叉验证评估整个系统"""
        print("\n" + "="*80)
        print(f"K-Fold交叉验证 (k={k})")
        print("="*80)

        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        # 存储结果
        overall_results = []  # 整体性能
        high_risk_results = []  # 高风险样本性能
        triage_results = []  # 分诊性能

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X), 1):
            print(f"\n{'='*80}")
            print(f"Fold {fold}/{k}")
            print(f"{'='*80}")

            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_amp_train, y_amp_val = self.y_amplitude[train_idx], self.y_amplitude[val_idx]
            y_risk_train, y_risk_val = self.y_risk_labels[train_idx], self.y_risk_labels[val_idx]

            # 训练分诊分类器
            self.train_triage_classifier(X_train, y_risk_train)

            # 训练专家回归模型
            self.train_expert_regressors(X_train, y_amp_train, y_risk_train)

            # 预测
            risk_pred, amp_pred, amp_std = self.predict(X_val, return_uncertainty=True)

            # 评估分诊性能
            triage_acc = accuracy_score(y_risk_val, risk_pred)
            triage_f1 = f1_score(y_risk_val, risk_pred, average='weighted')

            # 评估整体回归性能
            overall_r2 = 1 - np.sum((y_amp_val - amp_pred)**2) / np.sum((y_amp_val - np.mean(y_amp_val))**2)
            overall_rmse = np.sqrt(np.mean((y_amp_val - amp_pred)**2))
            overall_mae = np.mean(np.abs(y_amp_val - amp_pred))

            # 评估高风险样本性能(核心指标!)
            if self.triage_mode == 'binary':
                high_risk_mask = y_amp_val > self.HIGH_RISK_THRESHOLD
            else:
                high_risk_mask = y_amp_val > self.HIGH_THRESHOLD

            if high_risk_mask.sum() > 0:
                high_r2 = 1 - np.sum((y_amp_val[high_risk_mask] - amp_pred[high_risk_mask])**2) / \
                          np.sum((y_amp_val[high_risk_mask] - np.mean(y_amp_val[high_risk_mask]))**2)
                high_rmse = np.sqrt(np.mean((y_amp_val[high_risk_mask] - amp_pred[high_risk_mask])**2))
                high_mae = np.mean(np.abs(y_amp_val[high_risk_mask] - amp_pred[high_risk_mask]))
            else:
                high_r2 = high_rmse = high_mae = np.nan

            overall_results.append({'r2': overall_r2, 'rmse': overall_rmse, 'mae': overall_mae})
            high_risk_results.append({'r2': high_r2, 'rmse': high_rmse, 'mae': high_mae})
            triage_results.append({'accuracy': triage_acc, 'f1': triage_f1})

            print(f"\n分诊性能: 准确率={triage_acc*100:.1f}%, F1={triage_f1:.3f}")
            print(f"整体回归: R2={overall_r2:.4f}, RMSE={overall_rmse:.2f} mm")
            if not np.isnan(high_r2):
                print(f"高风险样本({high_risk_mask.sum()}个): R2={high_r2:.4f}, RMSE={high_rmse:.2f} mm")

        # 汇总
        print("\n" + "="*80)
        print("性能汇总")
        print("="*80)

        print(f"\n分诊分类器:")
        print(f"  准确率: {np.mean([r['accuracy'] for r in triage_results])*100:.2f}%")
        print(f"  F1得分: {np.mean([r['f1'] for r in triage_results]):.3f}")

        print(f"\n整体回归性能:")
        print(f"  验证集R2:   {np.mean([r['r2'] for r in overall_results]):.4f}")
        print(f"  验证集RMSE: {np.mean([r['rmse'] for r in overall_results]):.2f} mm")

        high_r2_valid = [r['r2'] for r in high_risk_results if not np.isnan(r['r2'])]
        high_rmse_valid = [r['rmse'] for r in high_risk_results if not np.isnan(r['rmse'])]

        if high_r2_valid:
            print(f"\n高风险样本性能(核心指标!):")
            print(f"  验证集R2:   {np.mean(high_r2_valid):.4f}")
            print(f"  验证集RMSE: {np.mean(high_rmse_valid):.2f} mm")

        return {
            'overall': overall_results,
            'high_risk': high_risk_results,
            'triage': triage_results
        }


def compare_binary_vs_ternary():
    """对比二元vs三元分类器"""
    print("="*80)
    print("实验: 二元分类 vs 三元分类对比")
    print("="*80)

    results_comparison = {}

    for mode in ['binary', 'ternary']:
        print(f"\n{'#'*80}")
        print(f"# 测试模式: {mode.upper()}")
        print(f"{'#'*80}")

        system = TriageExpertSystem('../data/final_bridge_dataset.csv', triage_mode=mode)
        system.load_and_prepare_data()
        results = system.evaluate_kfold(k=5)

        results_comparison[mode] = results

    # 最终对比
    print("\n" + "="*80)
    print("二元 vs 三元分类对比总结")
    print("="*80)

    for mode, results in results_comparison.items():
        overall_r2 = np.mean([r['r2'] for r in results['overall']])
        overall_rmse = np.mean([r['rmse'] for r in results['overall']])

        high_r2_valid = [r['r2'] for r in results['high_risk'] if not np.isnan(r['r2'])]
        high_rmse_valid = [r['rmse'] for r in results['high_risk'] if not np.isnan(r['rmse'])]

        triage_acc = np.mean([r['accuracy'] for r in results['triage']])

        print(f"\n{mode.upper()}:")
        print(f"  分诊准确率: {triage_acc*100:.2f}%")
        print(f"  整体R2: {overall_r2:.4f}, RMSE: {overall_rmse:.2f} mm")
        if high_r2_valid:
            print(f"  高风险R2: {np.mean(high_r2_valid):.4f}, RMSE: {np.mean(high_rmse_valid):.2f} mm")

    # 选择最佳
    binary_high_r2 = np.mean([r['r2'] for r in results_comparison['binary']['high_risk'] if not np.isnan(r['r2'])])
    ternary_high_r2 = np.mean([r['r2'] for r in results_comparison['ternary']['high_risk'] if not np.isnan(r['r2'])])

    print("\n" + "="*80)
    print("最终建议")
    print("="*80)

    if binary_high_r2 > ternary_high_r2:
        print("推荐使用: BINARY (二元分类)")
        print(f"理由: 高风险样本R2更高 ({binary_high_r2:.4f} vs {ternary_high_r2:.4f})")
    else:
        print("推荐使用: TERNARY (三元分类)")
        print(f"理由: 高风险样本R2更高 ({ternary_high_r2:.4f} vs {binary_high_r2:.4f})")

    return results_comparison


if __name__ == '__main__':
    results = compare_binary_vs_ternary()
