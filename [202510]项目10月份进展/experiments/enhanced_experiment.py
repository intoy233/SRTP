#!/usr/bin/env python3
"""
增强版桥梁VIV风险评估实验
测试改进后的特征工程和模型性能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class EnhancedBridgeVIVExperiment:
    """增强版桥梁VIV实验类"""

    def __init__(self):
        self.data = None
        self.X_original = None
        self.X_enhanced = None
        self.y = None
        self.feature_names_original = None
        self.feature_names_enhanced = None
        self.results = {}

    def load_data(self, file_path):
        """加载桥梁数据"""
        try:
            self.data = pd.read_csv(file_path, encoding='utf-8')
            print(f"成功加载数据: {self.data.shape}")
        except:
            try:
                self.data = pd.read_csv(file_path, encoding='gbk')
                print(f"成功加载数据: {self.data.shape}")
            except Exception as e:
                print(f"数据加载失败: {e}")
                return False

        print(f"数据列名: {list(self.data.columns)}")
        print(f"数据基本信息:")
        print(self.data.info())
        return True

    def create_enhanced_features(self, data):
        """创建增强版特征工程"""
        df = data.copy()

        print("开始增强特征工程...")

        # === 基础几何特征 ===
        if 'Span_m' in df.columns and 'Width_m' in df.columns:
            df['Aspect_Ratio'] = df['Span_m'] / df['Width_m']

        if 'Width_m' in df.columns and 'Height_m' in df.columns:
            df['Width_Height_Ratio'] = df['Width_m'] / df['Height_m']

        if 'Span_m' in df.columns and 'Height_m' in df.columns:
            df['Slenderness_Ratio'] = df['Span_m'] / df['Height_m']

        # === 高级几何特征 ===
        if 'Width_m' in df.columns and 'Height_m' in df.columns:
            df['Section_Area'] = df['Width_m'] * df['Height_m']
            df['Section_Perimeter'] = 2 * (df['Width_m'] + df['Height_m'])
            df['Hydraulic_Diameter'] = 4 * df['Section_Area'] / df['Section_Perimeter']
            df['Compactness_Factor'] = df['Section_Area'] / (df['Section_Perimeter']**2)

        # === 频率相关特征 ===
        if 'Natural_Freq_Hz' in df.columns and 'First_Freq_Hz' in df.columns:
            df['Freq_Ratio'] = df['Natural_Freq_Hz'] / (df['First_Freq_Hz'] + 1e-8)

        if 'First_Freq_Hz' in df.columns and 'Second_Freq_Hz' in df.columns:
            df['Higher_Freq_Ratio'] = df['Second_Freq_Hz'] / (df['First_Freq_Hz'] + 1e-8)

        # === 核心无量纲参数 ===
        # 1. 约化风速 (Reduced Wind Speed)
        if all(col in df.columns for col in ['VIV_Wind_Speed_ms', 'Natural_Freq_Hz', 'Width_m']):
            df['Reduced_Wind_Speed'] = df['VIV_Wind_Speed_ms'] / (df['Natural_Freq_Hz'] * df['Width_m'])

        if all(col in df.columns for col in ['Critical_Wind_Speed_ms', 'Natural_Freq_Hz', 'Width_m']):
            df['Critical_Reduced_Wind_Speed'] = df['Critical_Wind_Speed_ms'] / (df['Natural_Freq_Hz'] * df['Width_m'])

        # 2. Reynolds数
        if 'VIV_Wind_Speed_ms' in df.columns and 'Width_m' in df.columns:
            nu = 1.5e-5  # 运动粘度
            df['Reynolds_Number'] = (df['VIV_Wind_Speed_ms'] * df['Width_m']) / nu
            df['Log_Reynolds'] = np.log10(df['Reynolds_Number'] + 1)

        # 3. Strouhal数
        if all(col in df.columns for col in ['Natural_Freq_Hz', 'VIV_Wind_Speed_ms', 'Width_m']):
            df['Strouhal_Number'] = (df['Natural_Freq_Hz'] * df['Width_m']) / (df['VIV_Wind_Speed_ms'] + 1e-8)

        # 4. 质量阻尼参数
        if 'Damping_Ratio' in df.columns:
            typical_mass_ratio = 10.0
            df['Mass_Damping_Parameter'] = typical_mass_ratio * df['Damping_Ratio']
            df['Scruton_Number'] = 2 * typical_mass_ratio * df['Damping_Ratio']
            df['Log_Damping'] = np.log(df['Damping_Ratio'] + 1e-8)

        # 5. 高级无量纲参数
        if all(col in df.columns for col in ['VIV_Wind_Speed_ms', 'Damping_Ratio']):
            df['Wind_Resistance_Ratio'] = df['VIV_Wind_Speed_ms'] / (df['Damping_Ratio'] + 1e-8)

        if all(col in df.columns for col in ['Natural_Freq_Hz', 'Span_m']):
            df['Stiffness_Parameter'] = (df['Natural_Freq_Hz'] * df['Span_m'])**2

        if all(col in df.columns for col in ['VIV_Wind_Speed_ms', 'Natural_Freq_Hz', 'Span_m']):
            df['Aeroelastic_Parameter'] = df['VIV_Wind_Speed_ms'] / (df['Natural_Freq_Hz'] * df['Span_m'])

        # === 工程实用参数 ===
        if all(col in df.columns for col in ['Reduced_Wind_Speed', 'Mass_Damping_Parameter']):
            df['VIV_Risk_Index'] = df['Reduced_Wind_Speed'] / (df['Mass_Damping_Parameter'] + 1e-8)

        if all(col in df.columns for col in ['Width_Height_Ratio', 'Damping_Ratio']):
            df['Vibration_Sensitivity'] = df['Width_Height_Ratio'] / (df['Damping_Ratio'] + 1e-8)

        # === 物理约束检查 ===
        for col in df.columns:
            if 'Ratio' in col or 'Parameter' in col or 'Number' in col:
                if df[col].dtype in ['float64', 'float32']:
                    # 处理无穷值和NaN
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    # 用中位数填充NaN
                    df[col] = df[col].fillna(df[col].median())
                    # 限制极端值
                    if 'Reynolds' not in col:  # Reynolds数可以很大
                        df[col] = np.clip(df[col], -1000, 1000)

        return df

    def prepare_datasets(self):
        """准备原始和增强数据集"""
        if self.data is None:
            print("请先加载数据!")
            return False

        # 目标变量
        target_col = 'Max_Amplitude_mm'
        if target_col not in self.data.columns:
            print(f"目标变量 {target_col} 不存在!")
            return False

        self.y = self.data[target_col].values

        # 原始特征 (只选择数值特征)
        exclude_cols = [target_col, 'Bridge_Name', 'Bridge_ID']  # 排除非特征列
        original_features = []
        for col in self.data.columns:
            if col not in exclude_cols and self.data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                original_features.append(col)

        self.feature_names_original = original_features
        self.X_original = self.data[original_features].values

        # 增强特征
        enhanced_data = self.create_enhanced_features(self.data)

        # 选择增强后的数值特征
        enhanced_features = []
        for col in enhanced_data.columns:
            if col not in exclude_cols and enhanced_data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                enhanced_features.append(col)

        self.feature_names_enhanced = enhanced_features
        self.X_enhanced = enhanced_data[enhanced_features].values

        print(f"原始特征数: {len(self.feature_names_original)}")
        print(f"增强特征数: {len(self.feature_names_enhanced)}")
        print(f"新增特征数: {len(self.feature_names_enhanced) - len(self.feature_names_original)}")

        return True

    def evaluate_model(self, model, X, y, cv=5):
        """评估模型性能"""
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 训练模型
        model.fit(X_train_scaled, y_train)

        # 预测
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        # 交叉验证
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv,
                                  scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)

        # 计算指标
        metrics = {
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'test_mse': mean_squared_error(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'cv_rmse_mean': np.mean(cv_rmse),
            'cv_rmse_std': np.std(cv_rmse),
            'cv_stability': 1 / (1 + np.std(cv_rmse) / np.mean(cv_rmse))  # 稳定性分数
        }

        return metrics, y_test, y_test_pred

    def run_comparison_experiment(self):
        """运行对比实验"""
        print("\n=== 开始模型对比实验 ===")

        # 推荐的稳定模型配置
        models = {
            'Ridge': Ridge(alpha=10.0),  # 强正则化
            'Conservative_RF': RandomForestRegressor(
                n_estimators=20, max_depth=5, min_samples_split=10,
                min_samples_leaf=5, max_features='sqrt', random_state=42
            ),
            'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000),
            'Linear': LinearRegression()
        }

        self.results = {}

        for model_name, model in models.items():
            print(f"\n--- 测试模型: {model_name} ---")

            # 原始特征性能
            print("使用原始特征...")
            original_metrics, y_test_orig, y_pred_orig = self.evaluate_model(
                model, self.X_original, self.y
            )

            # 增强特征性能
            print("使用增强特征...")
            enhanced_metrics, y_test_enh, y_pred_enh = self.evaluate_model(
                model, self.X_enhanced, self.y
            )

            self.results[model_name] = {
                'original': original_metrics,
                'enhanced': enhanced_metrics,
                'predictions': {
                    'original': (y_test_orig, y_pred_orig),
                    'enhanced': (y_test_enh, y_pred_enh)
                }
            }

            # 打印结果
            print(f"原始特征 - 测试RMSE: {original_metrics['test_rmse']:.2f}, R²: {original_metrics['test_r2']:.3f}")
            print(f"增强特征 - 测试RMSE: {enhanced_metrics['test_rmse']:.2f}, R²: {enhanced_metrics['test_r2']:.3f}")
            print(f"改进幅度 - RMSE: {((original_metrics['test_rmse'] - enhanced_metrics['test_rmse']) / original_metrics['test_rmse'] * 100):.1f}%")
            print(f"稳定性改进: {enhanced_metrics['cv_stability'] - original_metrics['cv_stability']:.3f}")

    def generate_performance_report(self):
        """生成性能报告"""
        print("\n" + "="*60)
        print("               模型性能对比报告")
        print("="*60)

        # 创建对比表
        comparison_data = []
        for model_name, results in self.results.items():
            orig = results['original']
            enh = results['enhanced']

            improvement_rmse = ((orig['test_rmse'] - enh['test_rmse']) / orig['test_rmse']) * 100
            improvement_r2 = enh['test_r2'] - orig['test_r2']
            improvement_stability = enh['cv_stability'] - orig['cv_stability']

            comparison_data.append({
                '模型': model_name,
                '原始RMSE': f"{orig['test_rmse']:.2f}",
                '增强RMSE': f"{enh['test_rmse']:.2f}",
                'RMSE改进%': f"{improvement_rmse:.1f}%",
                '原始R²': f"{orig['test_r2']:.3f}",
                '增强R²': f"{enh['test_r2']:.3f}",
                'R²改进': f"{improvement_r2:.3f}",
                '稳定性改进': f"{improvement_stability:.3f}"
            })

        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))

        # 找出最佳模型
        best_model = None
        best_score = -float('inf')
        for model_name, results in self.results.items():
            # 综合评分：稳定性(40%) + R²(40%) + RMSE改进(20%)
            enh = results['enhanced']
            orig = results['original']
            rmse_improvement = ((orig['test_rmse'] - enh['test_rmse']) / orig['test_rmse'])

            score = (0.4 * enh['cv_stability'] +
                    0.4 * enh['test_r2'] +
                    0.2 * rmse_improvement)

            if score > best_score:
                best_score = score
                best_model = model_name

        print(f"\n推荐最佳模型: {best_model}")
        print(f"综合评分: {best_score:.3f}")

        return comparison_df

    def create_visualizations(self):
        """创建可视化图表"""
        print("\n生成可视化图表...")

        # 设置图表样式
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('桥梁VIV风险评估模型性能对比', fontsize=16, fontweight='bold')

        # 1. RMSE对比
        ax1 = axes[0, 0]
        models = list(self.results.keys())
        original_rmse = [self.results[m]['original']['test_rmse'] for m in models]
        enhanced_rmse = [self.results[m]['enhanced']['test_rmse'] for m in models]

        x = np.arange(len(models))
        width = 0.35

        ax1.bar(x - width/2, original_rmse, width, label='原始特征', alpha=0.8, color='skyblue')
        ax1.bar(x + width/2, enhanced_rmse, width, label='增强特征', alpha=0.8, color='lightcoral')

        ax1.set_xlabel('模型')
        ax1.set_ylabel('RMSE')
        ax1.set_title('测试集RMSE对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. R²对比
        ax2 = axes[0, 1]
        original_r2 = [self.results[m]['original']['test_r2'] for m in models]
        enhanced_r2 = [self.results[m]['enhanced']['test_r2'] for m in models]

        ax2.bar(x - width/2, original_r2, width, label='原始特征', alpha=0.8, color='skyblue')
        ax2.bar(x + width/2, enhanced_r2, width, label='增强特征', alpha=0.8, color='lightcoral')

        ax2.set_xlabel('模型')
        ax2.set_ylabel('R²')
        ax2.set_title('测试集R²对比')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 稳定性对比
        ax3 = axes[1, 0]
        original_stability = [self.results[m]['original']['cv_stability'] for m in models]
        enhanced_stability = [self.results[m]['enhanced']['cv_stability'] for m in models]

        ax3.bar(x - width/2, original_stability, width, label='原始特征', alpha=0.8, color='skyblue')
        ax3.bar(x + width/2, enhanced_stability, width, label='增强特征', alpha=0.8, color='lightcoral')

        ax3.set_xlabel('模型')
        ax3.set_ylabel('稳定性分数')
        ax3.set_title('交叉验证稳定性对比')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 预测vs实际 (最佳模型)
        ax4 = axes[1, 1]
        best_model = 'Ridge'  # 通常Ridge是最稳定的
        if best_model in self.results:
            y_test, y_pred = self.results[best_model]['predictions']['enhanced']
            ax4.scatter(y_test, y_pred, alpha=0.6, color='green')

            # 添加对角线
            min_val = min(min(y_test), min(y_pred))
            max_val = max(max(y_test), max(y_pred))
            ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)

            ax4.set_xlabel('实际振幅 (mm)')
            ax4.set_ylabel('预测振幅 (mm)')
            ax4.set_title(f'{best_model} 预测效果 (增强特征)')
            ax4.grid(True, alpha=0.3)

            # 添加R²信息
            r2 = self.results[best_model]['enhanced']['test_r2']
            ax4.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax4.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig('enhanced_model_performance.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_results(self):
        """保存实验结果"""
        # 保存详细结果到CSV
        detailed_results = []
        for model_name, results in self.results.items():
            for version in ['original', 'enhanced']:
                metrics = results[version]
                detailed_results.append({
                    'Model': model_name,
                    'Version': version,
                    'Test_RMSE': metrics['test_rmse'],
                    'Test_R2': metrics['test_r2'],
                    'Test_MAE': metrics['test_mae'],
                    'CV_RMSE_Mean': metrics['cv_rmse_mean'],
                    'CV_RMSE_Std': metrics['cv_rmse_std'],
                    'CV_Stability': metrics['cv_stability']
                })

        results_df = pd.DataFrame(detailed_results)
        results_df.to_csv('enhanced_experiment_results.csv', index=False)
        print(f"\n实验结果已保存到: enhanced_experiment_results.csv")


def main():
    """主函数"""
    print("=== 增强版桥梁VIV风险评估实验 ===\n")

    # 创建实验对象
    experiment = EnhancedBridgeVIVExperiment()

    # 尝试加载数据
    data_paths = [
        "../bridge_dataset_fixed.csv",
        "../../bridge_dataset_fixed.csv",
        "../bridge_dataset_augmented.csv",
        "../../bridge_dataset_augmented.csv"
    ]

    data_loaded = False
    for path in data_paths:
        if experiment.load_data(path):
            data_loaded = True
            break

    if not data_loaded:
        print("无法加载数据文件，生成模拟数据进行演示...")
        # 生成模拟数据
        np.random.seed(42)
        n_samples = 60

        simulated_data = pd.DataFrame({
            'Span_m': np.random.uniform(100, 2000, n_samples),
            'Width_m': np.random.uniform(20, 60, n_samples),
            'Height_m': np.random.uniform(3, 10, n_samples),
            'Natural_Freq_Hz': np.random.uniform(0.1, 2.0, n_samples),
            'First_Freq_Hz': np.random.uniform(0.08, 1.8, n_samples),
            'Second_Freq_Hz': np.random.uniform(0.2, 2.5, n_samples),
            'VIV_Wind_Speed_ms': np.random.uniform(8, 25, n_samples),
            'Critical_Wind_Speed_ms': np.random.uniform(12, 30, n_samples),
            'Damping_Ratio': np.random.uniform(0.01, 0.1, n_samples)
        })

        # 生成目标变量（基于物理关系）
        vr = simulated_data['VIV_Wind_Speed_ms'] / (simulated_data['Natural_Freq_Hz'] * simulated_data['Width_m'])
        damping_effect = 50 / simulated_data['Damping_Ratio']
        noise = np.random.normal(0, 5, n_samples)
        simulated_data['Max_Amplitude_mm'] = np.clip(vr * damping_effect + noise, 1, 200)

        experiment.data = simulated_data
        print(f"生成模拟数据: {experiment.data.shape}")

    # 准备数据集
    if not experiment.prepare_datasets():
        return

    # 运行对比实验
    experiment.run_comparison_experiment()

    # 生成报告
    comparison_df = experiment.generate_performance_report()

    # 创建可视化
    experiment.create_visualizations()

    # 保存结果
    experiment.save_results()

    print(f"\n实验完成！检查生成的文件:")
    print(f"- enhanced_experiment_results.csv")
    print(f"- enhanced_model_performance.png")


if __name__ == "__main__":
    main()