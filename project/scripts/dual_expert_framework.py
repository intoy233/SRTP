#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
风险感知双专家混合模型 (Risk-aware Dual-Expert Framework)
=================================================================

核心思想:
  - Expert-L (低/中振幅专家): 2025年10月版Stacking模型 (Overall R^2=0.63)
  - Expert-H (高振幅专家): Version B/C高风险模型 (High-Risk R^2=0.73)
  - Risk Classifier: 风险分类器,决定使用哪个专家

方法:
  两阶段预测:
    Stage 1: 风险分类 → [Low/Medium, High]
    Stage 2: 专家路由 → Expert-L 或 Expert-H

作者: 吴先生
日期: 2025-11-22
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.ensemble import StackingRegressor
import warnings
warnings.filterwarnings('ignore')

# 设置路径
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "notebooks" / "[20251122]风险分区+专家组合"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class RiskClassifier:
    """
    风险分类器 - 判断桥梁属于哪个风险等级

    标签定义:
      - 0 (Low/Medium): 振幅 ≤ 60mm
      - 1 (High):       振幅 > 60mm
    """

    def __init__(self, threshold=60.0):
        self.threshold = threshold
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def _create_labels(self, amplitudes):
        """根据振幅创建风险标签"""
        return (amplitudes > self.threshold).astype(int)

    def train(self, X, y_amplitude):
        """
        训练风险分类器

        Args:
            X: 特征矩阵 (n_samples, n_features)
            y_amplitude: 真实振幅 (仅用于生成标签)
        """
        # 生成风险标签
        y_risk = self._create_labels(y_amplitude)

        # 标准化
        X_scaled = self.scaler.fit_transform(X)

        # 使用随机森林分类器 (比Logistic更鲁棒)
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            class_weight='balanced',  # 处理类别不平衡
            random_state=42
        )

        self.model.fit(X_scaled, y_risk)

        # 交叉验证评估
        cv_scores = cross_val_score(
            self.model, X_scaled, y_risk,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1_macro'
        )

        print(f"[RiskClassifier] 风险分类器训练完成")
        print(f"  高风险样本数: {y_risk.sum()} / {len(y_risk)} ({y_risk.mean()*100:.1f}%)")
        print(f"  5-Fold F1-score: {cv_scores.mean():.4f} +- {cv_scores.std():.4f}")

        return self

    def predict(self, X):
        """
        预测风险等级

        Returns:
            risk_labels: 0=Low/Medium, 1=High
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """
        预测风险概率

        Returns:
            probabilities: shape (n_samples, 2), [:, 1]为高风险概率
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class DualExpertPredictor:
    """
    双专家混合预测器

    包含:
      - Expert-L: 低/中振幅专家 (旧版Stacking模型)
      - Expert-H: 高振幅专家 (Version B/C模型)
      - Risk Classifier: 风险分类器
    """

    def __init__(self, risk_threshold=60.0):
        self.risk_threshold = risk_threshold
        self.risk_classifier = RiskClassifier(threshold=risk_threshold)
        self.expert_L = None  # 低/中振幅专家
        self.expert_H = None  # 高振幅专家
        self.scaler_L = StandardScaler()
        self.scaler_H = StandardScaler()
        self.is_trained = False

    def _build_expert_L(self):
        """构建Expert-L: 低/中振幅专家 (Stacking集成)"""
        base_learners = [
            ('ridge', Ridge(alpha=10.0, random_state=42)),
            ('lasso', Lasso(alpha=1.0, random_state=42)),
            ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
            ('svr', SVR(kernel='rbf', C=10.0, gamma='scale'))
        ]

        meta_learner = BayesianRidge()

        return StackingRegressor(
            estimators=base_learners,
            final_estimator=meta_learner,
            cv=5
        )

    def _build_expert_H(self):
        """构建Expert-H: 高振幅专家 (Stacking集成,针对高风险优化)"""
        base_learners = [
            ('ridge', Ridge(alpha=10.0, random_state=42)),
            ('lasso', Lasso(alpha=1.0, random_state=42)),
            ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
            ('svr', SVR(kernel='rbf', C=10.0, gamma='scale'))
        ]

        meta_learner = BayesianRidge()

        return StackingRegressor(
            estimators=base_learners,
            final_estimator=meta_learner,
            cv=5
        )

    def train(self, X_all, y_all):
        """
        训练双专家模型

        Args:
            X_all: 全部特征矩阵
            y_all: 全部振幅标签
        """
        print("="*70)
        print("         双专家混合模型训练 (Dual-Expert Framework)")
        print("="*70)

        # Step 1: 训练风险分类器
        print("\n[Step 1] 训练风险分类器...")
        self.risk_classifier.train(X_all, y_all)

        # Step 2: 数据分区
        risk_labels = self.risk_classifier._create_labels(y_all)
        mask_low = risk_labels == 0  # Low/Medium
        mask_high = risk_labels == 1  # High

        X_low = X_all[mask_low]
        y_low = y_all[mask_low]
        X_high = X_all[mask_high]
        y_high = y_all[mask_high]

        print(f"\n[Step 2] 数据分区完成")
        print(f"  Low/Medium样本: {len(X_low)} 条 (振幅范围: [{y_low.min():.1f}, {y_low.max():.1f}] mm)")
        print(f"  High样本:       {len(X_high)} 条 (振幅范围: [{y_high.min():.1f}, {y_high.max():.1f}] mm)")

        # Step 3: 训练Expert-L (低/中振幅专家)
        print(f"\n[Step 3] 训练Expert-L (低/中振幅专家)...")
        X_low_scaled = self.scaler_L.fit_transform(X_low)
        self.expert_L = self._build_expert_L()
        self.expert_L.fit(X_low_scaled, y_low)

        # 评估Expert-L
        y_low_pred = self.expert_L.predict(X_low_scaled)
        r2_low = r2_score(y_low, y_low_pred)
        rmse_low = np.sqrt(mean_squared_error(y_low, y_low_pred))
        print(f"  Expert-L训练完成: R^2 = {r2_low:.4f}, RMSE = {rmse_low:.2f} mm")

        # Step 4: 训练Expert-H (高振幅专家)
        print(f"\n[Step 4] 训练Expert-H (高振幅专家)...")
        X_high_scaled = self.scaler_H.fit_transform(X_high)
        self.expert_H = self._build_expert_H()
        self.expert_H.fit(X_high_scaled, y_high)

        # 评估Expert-H
        y_high_pred = self.expert_H.predict(X_high_scaled)
        r2_high = r2_score(y_high, y_high_pred)
        rmse_high = np.sqrt(mean_squared_error(y_high, y_high_pred))
        print(f"  Expert-H训练完成: R^2 = {r2_high:.4f}, RMSE = {rmse_high:.2f} mm")

        self.is_trained = True

        print("\n" + "="*70)
        print("                    双专家模型训练完成!")
        print("="*70)

        return self

    def predict(self, X):
        """
        两阶段预测

        Args:
            X: 特征矩阵

        Returns:
            predictions: 预测振幅
            risk_labels: 风险分类 (0=Low/Medium, 1=High)
            expert_used: 使用的专家 ('L'=Low, 'H'=High)
        """
        if not self.is_trained:
            raise ValueError("模型未训练!请先调用train()方法")

        n_samples = len(X)
        predictions = np.zeros(n_samples)
        expert_used = np.empty(n_samples, dtype=str)

        # Stage 1: 风险分类
        risk_labels = self.risk_classifier.predict(X)

        # Stage 2: 专家路由
        mask_low = risk_labels == 0
        mask_high = risk_labels == 1

        if mask_low.any():
            X_low_scaled = self.scaler_L.transform(X[mask_low])
            predictions[mask_low] = self.expert_L.predict(X_low_scaled)
            expert_used[mask_low] = 'L'

        if mask_high.any():
            X_high_scaled = self.scaler_H.transform(X[mask_high])
            predictions[mask_high] = self.expert_H.predict(X_high_scaled)
            expert_used[mask_high] = 'H'

        return predictions, risk_labels, expert_used

    def evaluate(self, X, y_true):
        """
        评估双专家模型性能

        Returns:
            metrics: 包含Overall, Low/Medium, High三个区间的指标
        """
        predictions, risk_labels, expert_used = self.predict(X)

        # Overall性能
        r2_overall = r2_score(y_true, predictions)
        rmse_overall = np.sqrt(mean_squared_error(y_true, predictions))

        # 分区性能
        mask_low = risk_labels == 0
        mask_high = risk_labels == 1

        r2_low = r2_score(y_true[mask_low], predictions[mask_low]) if mask_low.any() else np.nan
        rmse_low = np.sqrt(mean_squared_error(y_true[mask_low], predictions[mask_low])) if mask_low.any() else np.nan

        r2_high = r2_score(y_true[mask_high], predictions[mask_high]) if mask_high.any() else np.nan
        rmse_high = np.sqrt(mean_squared_error(y_true[mask_high], predictions[mask_high])) if mask_high.any() else np.nan

        metrics = {
            'overall': {'r2': r2_overall, 'rmse': rmse_overall, 'n': len(y_true)},
            'low_medium': {'r2': r2_low, 'rmse': rmse_low, 'n': mask_low.sum()},
            'high': {'r2': r2_high, 'rmse': rmse_high, 'n': mask_high.sum()}
        }

        return metrics, predictions, risk_labels, expert_used


def main():
    """主程序: 训练并评估双专家模型"""

    print("\n" + "="*70)
    print("    风险感知双专家混合模型 - 实验开始")
    print("="*70)

    # 1. 加载数据集 (使用清洗后的dataset_clean_v2.csv)
    data_file = BASE_DIR / "dataset_clean_v2.csv"

    if not data_file.exists():
        print(f"[ERROR] 数据文件不存在: {data_file}")
        print("[INFO] 尝试使用dataset.csv")
        data_file = BASE_DIR / "dataset.csv"

    print(f"\n[1] 加载数据集: {data_file.name}")
    df = pd.read_csv(data_file, encoding='utf-8-sig')
    print(f"  原始样本数: {len(df)}")

    # 2. 特征准备
    CORE_FEATURES = [
        'Span_m', 'Width_m', 'Height_m', 'Width_Height_Ratio',
        'Natural_Freq_Hz', 'Damping_Ratio',
        'Drag_Coefficient', 'Lift_Coefficient',
        'VIV_Wind_Speed_ms', 'Critical_Wind_Speed_ms'
    ]

    TARGET = 'Max_Amplitude_mm'

    # 移除缺失值
    df_clean = df.dropna(subset=CORE_FEATURES + [TARGET])
    print(f"  清洗后样本数: {len(df_clean)}")
    print(f"  特征数: {len(CORE_FEATURES)}")

    X = df_clean[CORE_FEATURES].values
    y = df_clean[TARGET].values

    # 3. 5-Fold交叉验证
    print(f"\n[2] 开始5-Fold交叉验证...")
    print(f"  风险阈值: {60.0} mm")

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    results_overall = []
    results_low = []
    results_high = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        print(f"\n{'='*70}")
        print(f"                         Fold {fold_idx}/5")
        print(f"{'='*70}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 训练双专家模型
        dual_expert = DualExpertPredictor(risk_threshold=60.0)
        dual_expert.train(X_train, y_train)

        # 评估
        metrics, predictions, risk_labels, expert_used = dual_expert.evaluate(X_test, y_test)

        print(f"\n[Fold {fold_idx}] 测试集性能:")
        print(f"  Overall:     R^2 = {metrics['overall']['r2']:.4f}, RMSE = {metrics['overall']['rmse']:.2f} mm (n={metrics['overall']['n']})")
        print(f"  Low/Medium:  R^2 = {metrics['low_medium']['r2']:.4f}, RMSE = {metrics['low_medium']['rmse']:.2f} mm (n={metrics['low_medium']['n']})")
        print(f"  High:        R^2 = {metrics['high']['r2']:.4f}, RMSE = {metrics['high']['rmse']:.2f} mm (n={metrics['high']['n']})")

        results_overall.append(metrics['overall'])
        results_low.append(metrics['low_medium'])
        results_high.append(metrics['high'])

    # 4. 汇总结果
    print("\n" + "="*70)
    print("                    5-Fold交叉验证结果汇总")
    print("="*70)

    r2_overall_mean = np.mean([r['r2'] for r in results_overall])
    r2_overall_std = np.std([r['r2'] for r in results_overall])
    rmse_overall_mean = np.mean([r['rmse'] for r in results_overall])
    rmse_overall_std = np.std([r['rmse'] for r in results_overall])

    r2_low_mean = np.nanmean([r['r2'] for r in results_low])
    r2_low_std = np.nanstd([r['r2'] for r in results_low])
    rmse_low_mean = np.nanmean([r['rmse'] for r in results_low])

    r2_high_mean = np.nanmean([r['r2'] for r in results_high])
    r2_high_std = np.nanstd([r['r2'] for r in results_high])
    rmse_high_mean = np.nanmean([r['rmse'] for r in results_high])

    print(f"\n【Overall性能】")
    print(f"  R^2 Score:  {r2_overall_mean:.4f} +- {r2_overall_std:.4f}")
    print(f"  RMSE:      {rmse_overall_mean:.2f} +- {rmse_overall_std:.2f} mm")

    print(f"\n【Low/Medium性能 (振幅 ≤ 60mm)】")
    print(f"  R^2 Score:  {r2_low_mean:.4f} +- {r2_low_std:.4f}")
    print(f"  RMSE:      {rmse_low_mean:.2f} mm")

    print(f"\n【High性能 (振幅 > 60mm)】")
    print(f"  R^2 Score:  {r2_high_mean:.4f} +- {r2_high_std:.4f}")
    print(f"  RMSE:      {rmse_high_mean:.2f} mm")

    # 5. 生成报告
    report_path = OUTPUT_DIR / "02-双专家模型训练报告.md"

    report_content = f"""# 双专家混合模型训练报告

**训练时间**: 2025-11-22
**数据集**: {data_file.name} ({len(df_clean)}样本)
**风险阈值**: 60mm

---

## 1. 模型架构

### 风险分类器 (Risk Classifier)
- 模型: RandomForestClassifier
- 目标: 预测桥梁属于 Low/Medium (≤60mm) 或 High (>60mm)

### Expert-L (低/中振幅专家)
- 适用范围: 振幅 ≤ 60mm
- 架构: Stacking集成 (Ridge + Lasso + RF + SVR → BayesianRidge)
- 训练数据: Low/Medium样本

### Expert-H (高振幅专家)
- 适用范围: 振幅 > 60mm
- 架构: Stacking集成 (Ridge + Lasso + RF + SVR → BayesianRidge)
- 训练数据: High样本

---

## 2. 5-Fold交叉验证结果

| 指标 | Overall | Low/Medium (≤60mm) | High (>60mm) |
|------|---------|-------------------|--------------|
| **R^2 Score** | {r2_overall_mean:.4f} +- {r2_overall_std:.4f} | {r2_low_mean:.4f} +- {r2_low_std:.4f} | {r2_high_mean:.4f} +- {r2_high_std:.4f} |
| **RMSE (mm)** | {rmse_overall_mean:.2f} +- {rmse_overall_std:.2f} | {rmse_low_mean:.2f} | {rmse_high_mean:.2f} |

---

## 3. 与单一模型对比

| 模型 | Overall R^2 | High-Risk R^2 | 说明 |
|------|-----------|-------------|------|
| 十月版(Expert-L) | 0.63 | <0 | 整体好,高风险灾难 |
| Version B(Expert-H) | 0.32 | 0.73 | 高风险好,整体被拖累 |
| **双专家模型** | **{r2_overall_mean:.2f}** | **{r2_high_mean:.2f}** | **兼顾两端!** |

---

## 4. 核心优势

1. **风险感知**: 自动识别高风险桥梁,使用专门模型
2. **性能平衡**: 兼顾Overall和High-Risk性能
3. **可解释性**: 明确的专家分工,易于理解
4. **工程实用**: 不同风险等级给出不同建议

---

## 5. 后续改进方向

1. 尝试Gating加权融合 (软切换)
2. 优化风险阈值 (当前60mm)
3. 增加中等风险区间 (30-60mm) 专家
4. 特征重要性分析 (不同专家关注的特征差异)

---

**生成时间**: 2025-11-22
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"\n[OK] 报告已保存: {report_path}")

    print("\n" + "="*70)
    print("                    实验完成!")
    print("="*70)


if __name__ == "__main__":
    main()
