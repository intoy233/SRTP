"""
Step 4: 训练 Version B (混合样本版)
数据集: dataset.csv (475样本, 53.9%真实数据 + 46.1%经验填充)
目标: 建立性能上限基准，验证数据补充带来的真实提升

关键验证指标:
1. Overall R^2 & RMSE
2. High-Risk R^2 (振幅 > 60mm)
3. 学习曲线 (samples vs score)
4. 特征重要性变化
5. 与Version A对比分析

预期: High-Risk R^2 从 -1.48 → 0.45+
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, BayesianRidge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ====== 配置路径 ======
BASE_DIR = Path(__file__).parent.parent
DATA_FILE = BASE_DIR / "dataset.csv"
OUTPUT_DIR = BASE_DIR / "notebooks" / "[20251119]数据补全"
OUTPUT_DIR.mkdir(exist_ok=True)

# Version A 结果（用于对比）
VERSION_A_RESULTS = {
    "overall_r2": 0.5333,
    "overall_r2_std": 0.2570,
    "overall_rmse": 14.90,
    "overall_rmse_std": 1.85,
    "high_risk_r2": -1.9169,
    "high_risk_r2_std": 3.6808,
    "high_risk_rmse": 38.16,
    "high_risk_rmse_std": 8.77,
    "n_samples": 216,
    "n_high_risk": 64
}

# ====== 核心特征（8个，移除高缺失率的Drag/Lift Coefficient）======
# 原因：Drag_Coefficient和Lift_Coefficient缺失106条（22.3%），严重影响可用样本量
CORE_FEATURES = [
    'Span_m',
    'Width_m',
    'Height_m',
    'Width_Height_Ratio',
    'Natural_Freq_Hz',
    'VIV_Wind_Speed_ms',
    'Critical_Wind_Speed_ms',
    'Damping_Ratio'
]

TARGET = 'Max_Amplitude_mm'

# ====== 数据加载与预处理 ======
def load_and_preprocess_data():
    """
    加载dataset.csv并进行预处理
    """
    print("="*70)
    print("Step 4: 训练 Version B (混合样本版)".center(70))
    print("="*70)

    # 1. 加载数据
    print("\n[1] 加载数据...")
    df = pd.read_csv(DATA_FILE, encoding='utf-8-sig')
    print(f"  总样本数: {len(df)}")

    # 2. 检查核心特征
    print("\n[2] 检查特征完整性...")
    missing = df[CORE_FEATURES + [TARGET]].isnull().sum()
    if missing.sum() > 0:
        print(f"  警告: 存在缺失值!")
        print(missing[missing > 0])
        # 删除缺失值
        df = df.dropna(subset=CORE_FEATURES + [TARGET])
        print(f"  删除后样本数: {len(df)}")
    else:
        print(f"  [OK] 所有核心特征完整")

    # 3. 数据质量统计
    print("\n[3] 数据质量统计...")
    empirical_mask = (df['Critical_Wind_Speed_ms'] == 22.0) | (df['Critical_Wind_Speed_ms'] == 5.1)
    n_empirical = empirical_mask.sum()
    n_real = len(df) - n_empirical
    print(f"  真实数据: {n_real} 条 ({n_real/len(df)*100:.1f}%)")
    print(f"  经验填充: {n_empirical} 条 ({n_empirical/len(df)*100:.1f}%)")
    print(f"  Critical_Wind_Speed 范围: [{df['Critical_Wind_Speed_ms'].min():.1f}, {df['Critical_Wind_Speed_ms'].max():.1f}] m/s")
    print(f"  Critical_Wind_Speed 均值: {df['Critical_Wind_Speed_ms'].mean():.2f} m/s")

    # 4. 高风险样本统计
    high_risk_mask = df[TARGET] > 60
    n_high_risk = high_risk_mask.sum()
    print(f"\n  高风险样本 (Amp>60mm): {n_high_risk} 条 ({n_high_risk/len(df)*100:.1f}%)")
    print(f"  高风险中真实Vcr: {(~empirical_mask & high_risk_mask).sum()} 条")
    print(f"  高风险中经验Vcr: {(empirical_mask & high_risk_mask).sum()} 条")

    # 5. 提取特征和目标
    X = df[CORE_FEATURES].values
    y = df[TARGET].values

    # 6. 高风险索引
    high_risk_idx = np.where(df[TARGET].values > 60)[0]

    return X, y, high_risk_idx, df

# ====== 构建Stacking模型（与Version A一致）======
def build_model():
    """
    构建Stacking集成模型
    """
    base_learners = [
        ('ridge', Ridge(alpha=10.0, random_state=42)),
        ('lasso', Lasso(alpha=1.0, random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
        ('svr', SVR(kernel='rbf', C=10.0, gamma='scale'))
    ]

    meta_learner = BayesianRidge()

    model = StackingRegressor(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5
    )

    return model

# ====== 1. Overall性能评估 ======
def evaluate_overall_performance(X, y):
    """
    评估整体性能（5-Fold交叉验证）
    """
    print("\n" + "="*70)
    print("[评估1] Overall 性能 (5-Fold CV)".center(70))
    print("="*70)

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 构建模型
    model = build_model()

    # 交叉验证 - R^2
    cv_r2 = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')

    # 交叉验证 - RMSE (负MSE的平方根)
    cv_neg_mse = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_neg_mse)

    # 结果
    results = {
        "r2_mean": cv_r2.mean(),
        "r2_std": cv_r2.std(),
        "rmse_mean": cv_rmse.mean(),
        "rmse_std": cv_rmse.std(),
        "r2_folds": cv_r2,
        "rmse_folds": cv_rmse
    }

    print(f"\n  R^2 Score: {results['r2_mean']:.4f} +- {results['r2_std']:.4f}")
    print(f"  RMSE:     {results['rmse_mean']:.2f} +- {results['rmse_std']:.2f} mm")
    print(f"\n  各Fold详细:")
    for i, (r2, rmse) in enumerate(zip(cv_r2, cv_rmse), 1):
        print(f"    Fold {i}: R^2 = {r2:.4f}, RMSE = {rmse:.2f} mm")

    # 与Version A对比
    print(f"\n  【对比 Version A】")
    print(f"    R^2 提升: {results['r2_mean'] - VERSION_A_RESULTS['overall_r2']:.4f} ({(results['r2_mean'] / VERSION_A_RESULTS['overall_r2'] - 1) * 100:+.1f}%)")
    print(f"    RMSE 变化: {results['rmse_mean'] - VERSION_A_RESULTS['overall_rmse']:.2f} mm ({(results['rmse_mean'] / VERSION_A_RESULTS['overall_rmse'] - 1) * 100:+.1f}%)")

    return results

# ====== 2. High-Risk性能评估 ======
def evaluate_high_risk_performance(X, y, high_risk_idx):
    """
    评估高风险样本性能（振幅 > 60mm）
    """
    print("\n" + "="*70)
    print("[评估2] High-Risk 性能 (Amp > 60mm, 5-Fold CV)".center(70))
    print("="*70)

    # 提取高风险样本
    X_hr = X[high_risk_idx]
    y_hr = y[high_risk_idx]

    print(f"\n  高风险样本数: {len(X_hr)}")
    print(f"  振幅范围: [{y_hr.min():.1f}, {y_hr.max():.1f}] mm")
    print(f"  振幅均值: {y_hr.mean():.1f} mm")

    # 标准化
    scaler = StandardScaler()
    X_hr_scaled = scaler.fit_transform(X_hr)

    # 构建模型
    model = build_model()

    # 交叉验证 - R^2
    cv_r2 = cross_val_score(model, X_hr_scaled, y_hr, cv=5, scoring='r2')

    # 交叉验证 - RMSE
    cv_neg_mse = cross_val_score(model, X_hr_scaled, y_hr, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_neg_mse)

    # 结果
    results = {
        "r2_mean": cv_r2.mean(),
        "r2_std": cv_r2.std(),
        "rmse_mean": cv_rmse.mean(),
        "rmse_std": cv_rmse.std(),
        "r2_folds": cv_r2,
        "rmse_folds": cv_rmse,
        "n_samples": len(X_hr)
    }

    print(f"\n  R^2 Score: {results['r2_mean']:.4f} +- {results['r2_std']:.4f}")
    print(f"  RMSE:     {results['rmse_mean']:.2f} +- {results['rmse_std']:.2f} mm")
    print(f"\n  各Fold详细:")
    for i, (r2, rmse) in enumerate(zip(cv_r2, cv_rmse), 1):
        print(f"    Fold {i}: R^2 = {r2:.4f}, RMSE = {rmse:.2f} mm")

    # 与Version A对比
    print(f"\n  【对比 Version A】")
    improvement = results['r2_mean'] - VERSION_A_RESULTS['high_risk_r2']
    print(f"    R^2 提升: {improvement:.4f} (从 {VERSION_A_RESULTS['high_risk_r2']:.2f} → {results['r2_mean']:.4f})")
    if VERSION_A_RESULTS['high_risk_r2'] < 0 and results['r2_mean'] > 0:
        print(f"    [OK] 关键突破: 从负值转为正值！")
    print(f"    RMSE 变化: {results['rmse_mean'] - VERSION_A_RESULTS['high_risk_rmse']:.2f} mm ({(results['rmse_mean'] / VERSION_A_RESULTS['high_risk_rmse'] - 1) * 100:+.1f}%)")
    print(f"    样本数增加: {results['n_samples'] - VERSION_A_RESULTS['n_high_risk']} 条 ({(results['n_samples'] / VERSION_A_RESULTS['n_high_risk'] - 1) * 100:+.1f}%)")

    return results

# ====== 3. 学习曲线 ======
def plot_learning_curve(X, y):
    """
    绘制学习曲线 (samples vs score)
    验证样本量增加是否带来性能提升
    """
    print("\n" + "="*70)
    print("[评估3] 学习曲线分析 (Samples vs Score)".center(70))
    print("="*70)

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 构建模型
    model = build_model()

    # 计算学习曲线
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X_scaled, y,
        train_sizes=train_sizes,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        random_state=42
    )

    # 统计
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    print(f"\n  训练样本量范围: {train_sizes_abs.min()} - {train_sizes_abs.max()}")
    print(f"  最终验证R^2: {val_mean[-1]:.4f} +- {val_std[-1]:.4f}")
    print(f"  训练R^2: {train_mean[-1]:.4f} +- {train_std[-1]:.4f}")
    print(f"  过拟合程度: {train_mean[-1] - val_mean[-1]:.4f}")

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))

    # 训练曲线
    ax.plot(train_sizes_abs, train_mean, 'o-', color='#2E86C1',
            label=f'训练集 (最终: {train_mean[-1]:.3f})', linewidth=2)
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                     alpha=0.2, color='#2E86C1')

    # 验证曲线
    ax.plot(train_sizes_abs, val_mean, 'o-', color='#E74C3C',
            label=f'验证集 (最终: {val_mean[-1]:.3f})', linewidth=2)
    ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                     alpha=0.2, color='#E74C3C')

    # Version A基线
    ax.axhline(y=VERSION_A_RESULTS['overall_r2'], color='gray', linestyle='--',
               label=f'Version A Baseline ({VERSION_A_RESULTS["overall_r2"]:.3f}, n={VERSION_A_RESULTS["n_samples"]})')

    ax.set_xlabel('训练样本数', fontsize=12, fontweight='bold')
    ax.set_ylabel('R^2 Score', fontsize=12, fontweight='bold')
    ax.set_title('Version B 学习曲线：样本量 vs 性能', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([max(val_mean.min() - 0.1, -0.5), min(train_mean.max() + 0.1, 1.0)])

    plt.tight_layout()
    save_path = OUTPUT_DIR / "04-Version_B-学习曲线.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n  [OK] 学习曲线已保存: {save_path}")
    plt.close()

    return {
        "train_sizes": train_sizes_abs,
        "train_scores_mean": train_mean,
        "val_scores_mean": val_mean,
        "final_val_r2": val_mean[-1],
        "overfitting": train_mean[-1] - val_mean[-1]
    }

# ====== 4. 特征重要性分析 ======
def analyze_feature_importance(X, y, df):
    """
    分析特征重要性变化（使用RandomForest）
    """
    print("\n" + "="*70)
    print("[评估4] 特征重要性分析".center(70))
    print("="*70)

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 使用RandomForest计算特征重要性
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    rf.fit(X_scaled, y)

    # 特征重要性
    importance = rf.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': CORE_FEATURES,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    print("\n  特征重要性排名:")
    for i, row in importance_df.iterrows():
        print(f"    {row['Feature']:25s}: {row['Importance']:.4f}")

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#E74C3C' if feat == 'Critical_Wind_Speed_ms' else '#3498DB'
              for feat in importance_df['Feature']]

    ax.barh(range(len(importance_df)), importance_df['Importance'], color=colors, alpha=0.8)
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['Feature'])
    ax.set_xlabel('特征重要性', fontsize=12, fontweight='bold')
    ax.set_title('Version B 特征重要性排名 (Random Forest)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # 标注Critical_Wind_Speed_ms
    crit_idx = list(importance_df['Feature']).index('Critical_Wind_Speed_ms')
    crit_imp = importance_df.iloc[crit_idx]['Importance']
    ax.text(crit_imp + 0.005, crit_idx, f'{crit_imp:.4f}',
            va='center', fontweight='bold', color='#E74C3C')

    plt.tight_layout()
    save_path = OUTPUT_DIR / "04-Version_B-特征重要性.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n  [OK] 特征重要性图已保存: {save_path}")
    plt.close()

    return importance_df

# ====== 5. 生成综合对比报告 ======
def generate_comparison_report(overall_res, high_risk_res, lc_res, importance_df):
    """
    生成Version A vs Version B 综合对比报告
    """
    print("\n" + "="*70)
    print("[最终总结] Version A vs Version B 对比".center(70))
    print("="*70)

    report = f"""
# Version B 训练结果报告

**训练时间**: 2025-11-19
**数据集**: dataset.csv (475样本, 53.9%真实数据)
**模型**: Stacking (Ridge + Lasso + RandomForest + SVR → BayesianRidge)

---

## 1. Overall 性能对比

| 指标 | Version A | Version B | 提升 |
|------|-----------|-----------|------|
| **样本数** | 216 | **475** | **+119.9%** |
| **R^2 Score** | 0.5333 +- 0.257 | **{overall_res['r2_mean']:.4f} +- {overall_res['r2_std']:.4f}** | **{overall_res['r2_mean'] - VERSION_A_RESULTS['overall_r2']:+.4f}** ({(overall_res['r2_mean'] / VERSION_A_RESULTS['overall_r2'] - 1) * 100:+.1f}%) |
| **RMSE (mm)** | 14.90 +- 1.85 | **{overall_res['rmse_mean']:.2f} +- {overall_res['rmse_std']:.2f}** | **{overall_res['rmse_mean'] - VERSION_A_RESULTS['overall_rmse']:+.2f}** ({(overall_res['rmse_mean'] / VERSION_A_RESULTS['overall_rmse'] - 1) * 100:+.1f}%) |

### 分析
- **样本量增加**: 从216→475 (+259条, +119.9%)
- **R^2表现**: {'提升' if overall_res['r2_mean'] > VERSION_A_RESULTS['overall_r2'] else '下降'} {abs(overall_res['r2_mean'] - VERSION_A_RESULTS['overall_r2']):.4f}
- **RMSE表现**: {'改善' if overall_res['rmse_mean'] < VERSION_A_RESULTS['overall_rmse'] else '恶化'} {abs(overall_res['rmse_mean'] - VERSION_A_RESULTS['overall_rmse']):.2f} mm

---

## 2. High-Risk 性能对比 [*][*][*]

| 指标 | Version A | Version B | 提升 |
|------|-----------|-----------|------|
| **高风险样本数** | 64 | **{high_risk_res['n_samples']}** | **+{high_risk_res['n_samples'] - VERSION_A_RESULTS['n_high_risk']}** ({(high_risk_res['n_samples'] / VERSION_A_RESULTS['n_high_risk'] - 1) * 100:+.1f}%) |
| **R^2 Score** | **-1.9169** +- 3.68 | **{high_risk_res['r2_mean']:.4f} +- {high_risk_res['r2_std']:.4f}** | **{high_risk_res['r2_mean'] - VERSION_A_RESULTS['high_risk_r2']:+.4f}** |
| **RMSE (mm)** | 38.16 +- 8.77 | **{high_risk_res['rmse_mean']:.2f} +- {high_risk_res['rmse_std']:.2f}** | **{high_risk_res['rmse_mean'] - VERSION_A_RESULTS['high_risk_rmse']:+.2f}** ({(high_risk_res['rmse_mean'] / VERSION_A_RESULTS['high_risk_rmse'] - 1) * 100:+.1f}%) |

### 关键发现
{'[YES] **重大突破**: High-Risk R^2 从负值转为正值！' if VERSION_A_RESULTS['high_risk_r2'] < 0 and high_risk_res['r2_mean'] > 0 else ''}
- **样本量增加**: {high_risk_res['n_samples'] - VERSION_A_RESULTS['n_high_risk']} 条高风险样本 ({(high_risk_res['n_samples'] / VERSION_A_RESULTS['n_high_risk'] - 1) * 100:+.1f}%)
- **R^2改善**: {high_risk_res['r2_mean'] - VERSION_A_RESULTS['high_risk_r2']:.4f} ({'从灾难性预测恢复到可用水平' if high_risk_res['r2_mean'] > 0 else '仍需改进'})
- **RMSE变化**: {high_risk_res['rmse_mean'] - VERSION_A_RESULTS['high_risk_rmse']:+.2f} mm

{'**验证成功**: Step 3的数据补充策略有效！高风险段预测性能显著改善。' if high_risk_res['r2_mean'] > 0.3 else '**需要继续补充**: 高风险R^2仍未达到目标(0.45+)，建议优先补充高振幅桥梁的真实Vcr数据。'}

---

## 3. 学习曲线分析

| 指标 | 数值 |
|------|------|
| **最终验证R^2** | {lc_res['final_val_r2']:.4f} |
| **最终训练R^2** | {lc_res['train_scores_mean'][-1]:.4f} |
| **过拟合程度** | {lc_res['overfitting']:.4f} |
| **收敛状态** | {'已收敛' if lc_res['overfitting'] < 0.15 else '轻度过拟合' if lc_res['overfitting'] < 0.25 else '明显过拟合'} |

### 趋势判断
- **样本量效应**: {'学习曲线仍在上升，增加样本可能带来进一步提升' if lc_res['val_scores_mean'][-1] > lc_res['val_scores_mean'][-2] else '学习曲线已趋于平稳，需要质量更高的数据'}
- **模型稳定性**: {'稳定' if lc_res['overfitting'] < 0.20 else '需要正则化'}

---

## 4. 特征重要性 Top 5

| 排名 | 特征 | 重要性 |
|------|------|--------|
| 1 | {importance_df.iloc[0]['Feature']} | {importance_df.iloc[0]['Importance']:.4f} |
| 2 | {importance_df.iloc[1]['Feature']} | {importance_df.iloc[1]['Importance']:.4f} |
| 3 | {importance_df.iloc[2]['Feature']} | {importance_df.iloc[2]['Importance']:.4f} |
| 4 | {importance_df.iloc[3]['Feature']} | {importance_df.iloc[3]['Importance']:.4f} |
| 5 | {importance_df.iloc[4]['Feature']} | {importance_df.iloc[4]['Importance']:.4f} |

### Critical_Wind_Speed_ms 地位
- **重要性排名**: 第 {list(importance_df['Feature']).index('Critical_Wind_Speed_ms') + 1} 位
- **重要性数值**: {importance_df[importance_df['Feature'] == 'Critical_Wind_Speed_ms']['Importance'].values[0]:.4f}
- **结论**: {'核心特征，数据质量提升带来显著影响' if importance_df[importance_df['Feature'] == 'Critical_Wind_Speed_ms']['Importance'].values[0] > 0.10 else '重要性中等，可能仍需更多真实数据'}

---

## 5. 结论与建议

### 5.1 数据补充验证结果

{'[YES] **成功**: 数据补充策略有效' if high_risk_res['r2_mean'] > VERSION_A_RESULTS['high_risk_r2'] + 1.0 else '[WARN] **部分成功**: 有改善但未达预期'}

- 样本量从216→475，高风险样本从64→{high_risk_res['n_samples']}
- 真实Vcr数据占比从45.1%→53.9%
- High-Risk R^2 {'从 {:.2f} 提升至 {:.4f}'.format(VERSION_A_RESULTS['high_risk_r2'], high_risk_res['r2_mean']) if high_risk_res['r2_mean'] > VERSION_A_RESULTS['high_risk_r2'] else '改善有限'}

### 5.2 Version C 策略建议

| 优先级 | 行动项 | 目标 | 预期影响 |
|-------|--------|------|---------|
| **P0** | 补充高风险桥梁真实Vcr | 80-100座 | High-Risk R^2 → 0.50+ |
| **P1** | 将真实数据占比提升至70% | 再增100座 | Overall R^2 → 0.75+ |
| **P2** | 清洗/剔除质量差的样本 | 删除~50座 | 降低噪声 |

### 5.3 当前性能评级

| 维度 | 评分 | 说明 |
|------|------|------|
| **Overall性能** | {'A' if overall_res['r2_mean'] > 0.70 else 'B+' if overall_res['r2_mean'] > 0.60 else 'B' if overall_res['r2_mean'] > 0.50 else 'C'} | R^2 = {overall_res['r2_mean']:.4f} |
| **High-Risk性能** | {'A' if high_risk_res['r2_mean'] > 0.50 else 'B' if high_risk_res['r2_mean'] > 0.30 else 'C' if high_risk_res['r2_mean'] > 0.00 else 'D'} | R^2 = {high_risk_res['r2_mean']:.4f} |
| **数据质量** | B+ | 53.9%真实数据 |
| **模型稳定性** | {'A' if lc_res['overfitting'] < 0.15 else 'B' if lc_res['overfitting'] < 0.25 else 'C'} | 过拟合 = {lc_res['overfitting']:.4f} |

**综合评级**: **{'A-' if overall_res['r2_mean'] > 0.65 and high_risk_res['r2_mean'] > 0.35 else 'B+' if overall_res['r2_mean'] > 0.55 else 'B'}**

---

**生成时间**: 2025-11-19
**下一步**: {'立即训练Version C' if high_risk_res['r2_mean'] > 0.45 else '继续补充数据后再训练Version C'}
"""

    # 保存报告
    report_path = OUTPUT_DIR / "04-Version_B-训练报告.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    print(f"\n[OK] 完整报告已保存: {report_path}")

    return report

# ====== 主执行流程 ======
def main():
    """
    主函数：执行Version B完整训练流程
    """
    # 1. 加载数据
    X, y, high_risk_idx, df = load_and_preprocess_data()

    # 2. Overall性能评估
    overall_results = evaluate_overall_performance(X, y)

    # 3. High-Risk性能评估
    high_risk_results = evaluate_high_risk_performance(X, y, high_risk_idx)

    # 4. 学习曲线
    lc_results = plot_learning_curve(X, y)

    # 5. 特征重要性
    importance_df = analyze_feature_importance(X, y, df)

    # 6. 生成综合报告
    generate_comparison_report(overall_results, high_risk_results, lc_results, importance_df)

    print("\n" + "="*70)
    print("Version B 训练完成！".center(70))
    print("="*70)
    print(f"\n核心结论:")
    print(f"  1. Overall R^2: {overall_results['r2_mean']:.4f} (vs Version A: {VERSION_A_RESULTS['overall_r2']:.4f})")
    print(f"  2. High-Risk R^2: {high_risk_results['r2_mean']:.4f} (vs Version A: {VERSION_A_RESULTS['high_risk_r2']:.4f})")
    print(f"  3. 高风险预测{'成功改善' if high_risk_results['r2_mean'] > VERSION_A_RESULTS['high_risk_r2'] + 1.0 else '仍需提升'}")
    print(f"\n输出文件:")
    print(f"  - {OUTPUT_DIR / '04-Version_B-学习曲线.png'}")
    print(f"  - {OUTPUT_DIR / '04-Version_B-特征重要性.png'}")
    print(f"  - {OUTPUT_DIR / '04-Version_B-训练报告.md'}")

if __name__ == "__main__":
    main()
