#!/usr/bin/env python3
"""
验证和分析扩展桥梁VIV数据集
检查数据质量并生成详细报告
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

class DatasetValidator:
    def __init__(self, csv_path):
        """初始化数据集验证器"""
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path, encoding='utf-8-sig')
        self.validation_results = {}

    def basic_statistics(self):
        """基础统计分析"""
        print("=== 数据集基础统计信息 ===")
        print(f"数据集形状: {self.df.shape}")
        print(f"列数: {self.df.shape[1]}")
        print(f"行数: {self.df.shape[0]}")

        # 检查缺失值
        missing_values = self.df.isnull().sum()
        print(f"\n缺失值统计:")
        if missing_values.sum() == 0:
            print("  [+] 无缺失值")
        else:
            print(missing_values[missing_values > 0])

        # 数据类型
        print(f"\n数据类型:")
        print(self.df.dtypes.value_counts())

        # 桥梁类型分布
        print(f"\n桥梁类型分布:")
        bridge_type_dist = self.df['bridge_type'].value_counts()
        print(bridge_type_dist)

        # 断面类型分布
        print(f"\n断面类型分布:")
        section_type_dist = self.df['section_type'].value_counts()
        print(section_type_dist)

        self.validation_results['basic_stats'] = {
            'shape': self.df.shape,
            'missing_values': missing_values.sum(),
            'bridge_types': bridge_type_dist.to_dict(),
            'section_types': section_type_dist.to_dict()
        }

        return self.validation_results['basic_stats']

    def physical_validation(self):
        """物理参数合理性验证"""
        print("\n=== 物理参数合理性验证 ===")

        validation_rules = {
            'span_length': (10, 3000, '主跨长度'),
            'deck_width': (5, 100, '桥面宽度'),
            'tower_height': (10, 500, '塔高'),
            'frequency_1st': (0.01, 2.0, '一阶频率'),
            'damping_ratio': (0.001, 0.1, '阻尼比'),
            'wind_speed_critical': (2, 50, '临界风速'),
            'drag_coefficient': (0.3, 2.5, '阻力系数'),
            'strouhal_number': (0.05, 0.25, '斯特劳哈尔数'),
            'viv_amplitude': (0.001, 3.0, 'VIV幅度'),
            'scruton_number': (0.1, 100, '斯克鲁顿数'),
            'reduced_velocity': (1, 20, '减缩速度')
        }

        validation_summary = {}

        for param, (min_val, max_val, desc) in validation_rules.items():
            if param in self.df.columns:
                values = self.df[param]
                out_of_range = ((values < min_val) | (values > max_val)).sum()
                total_count = len(values)
                valid_percentage = (total_count - out_of_range) / total_count * 100

                print(f"{desc} ({param}):")
                print(f"  范围: [{min_val}, {max_val}]")
                print(f"  实际范围: [{values.min():.3f}, {values.max():.3f}]")
                print(f"  异常值数量: {out_of_range}/{total_count} ({100-valid_percentage:.1f}%)")
                print(f"  平均值: {values.mean():.3f}")

                validation_summary[param] = {
                    'expected_range': [min_val, max_val],
                    'actual_range': [values.min(), values.max()],
                    'outliers': out_of_range,
                    'valid_percentage': valid_percentage,
                    'mean': values.mean()
                }

        self.validation_results['physical_validation'] = validation_summary
        return validation_summary

    def engineering_relationships(self):
        """工程关系验证"""
        print("\n=== 工程关系验证 ===")

        # 1. 频率与跨度关系
        correlation_freq_span = self.df['frequency_1st'].corr(1/np.sqrt(self.df['span_length']))
        print(f"频率与跨度关系 (f ∝ 1/√L): 相关系数 = {correlation_freq_span:.3f}")

        # 2. 斯克鲁顿数与VIV幅度关系
        correlation_scruton_viv = self.df['scruton_number'].corr(-np.log(self.df['viv_amplitude'] + 0.01))
        print(f"斯克鲁顿数与VIV幅度关系: 相关系数 = {correlation_scruton_viv:.3f}")

        # 3. 塔高跨度比合理性
        height_span_ratio = self.df['height_to_span_ratio']
        print(f"塔高跨度比分布:")
        print(f"  平均值: {height_span_ratio.mean():.3f}")
        print(f"  标准差: {height_span_ratio.std():.3f}")
        print(f"  范围: [{height_span_ratio.min():.3f}, {height_span_ratio.max():.3f}]")

        # 4. 不同桥型的特征差异
        print(f"\n按桥梁类型的特征统计:")
        bridge_type_stats = self.df.groupby('bridge_type')[['span_length', 'frequency_1st', 'viv_amplitude']].mean()
        print(bridge_type_stats.round(3))

        self.validation_results['engineering_relationships'] = {
            'freq_span_correlation': correlation_freq_span,
            'scruton_viv_correlation': correlation_scruton_viv,
            'height_span_ratio_stats': {
                'mean': height_span_ratio.mean(),
                'std': height_span_ratio.std(),
                'range': [height_span_ratio.min(), height_span_ratio.max()]
            },
            'bridge_type_means': bridge_type_stats.to_dict()
        }

        return self.validation_results['engineering_relationships']

    def statistical_tests(self):
        """统计检验"""
        print("\n=== 统计检验 ===")

        # 正态性检验（选择几个关键变量）
        key_variables = ['viv_amplitude', 'damping_ratio', 'frequency_1st', 'wind_speed_critical']

        normality_results = {}
        for var in key_variables:
            if var in self.df.columns:
                # Shapiro-Wilk检验（样本量大时使用Kolmogorov-Smirnov）
                if len(self.df) > 5000:
                    statistic, p_value = stats.kstest(self.df[var], 'norm')
                    test_name = 'Kolmogorov-Smirnov'
                else:
                    statistic, p_value = stats.shapiro(self.df[var][:5000])  # 限制样本量
                    test_name = 'Shapiro-Wilk'

                is_normal = p_value > 0.05
                print(f"{var} 正态性检验 ({test_name}):")
                print(f"  统计量: {statistic:.6f}")
                print(f"  p值: {p_value:.6f}")
                print(f"  结果: {'正态分布' if is_normal else '非正态分布'}")

                normality_results[var] = {
                    'test': test_name,
                    'statistic': statistic,
                    'p_value': p_value,
                    'is_normal': is_normal
                }

        # 不同桥型间差异检验（ANOVA）
        bridge_types = self.df['bridge_type'].unique()
        if len(bridge_types) > 2:
            groups = [self.df[self.df['bridge_type'] == bt]['viv_amplitude'].values for bt in bridge_types]
            f_stat, p_val_anova = stats.f_oneway(*groups)
            print(f"\n桥梁类型间VIV幅度差异 (ANOVA):")
            print(f"  F统计量: {f_stat:.3f}")
            print(f"  p值: {p_val_anova:.6f}")
            print(f"  结果: {'存在显著差异' if p_val_anova < 0.05 else '无显著差异'}")

            normality_results['anova_bridge_types'] = {
                'f_statistic': f_stat,
                'p_value': p_val_anova,
                'significant': p_val_anova < 0.05
            }

        self.validation_results['statistical_tests'] = normality_results
        return normality_results

    def create_validation_plots(self):
        """创建验证图表"""
        print("\n=== 生成验证图表 ===")

        fig = plt.figure(figsize=(20, 16))

        # 1. 主要变量分布图
        plt.subplot(3, 4, 1)
        plt.hist(self.df['viv_amplitude'], bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('VIV幅度')
        plt.ylabel('频数')
        plt.title('VIV幅度分布')
        plt.grid(True, alpha=0.3)

        # 2. 桥梁类型分布
        plt.subplot(3, 4, 2)
        bridge_counts = self.df['bridge_type'].value_counts()
        plt.pie(bridge_counts.values, labels=bridge_counts.index, autopct='%1.1f%%')
        plt.title('桥梁类型分布')

        # 3. 断面类型分布
        plt.subplot(3, 4, 3)
        section_counts = self.df['section_type'].value_counts()
        plt.bar(range(len(section_counts)), section_counts.values)
        plt.xticks(range(len(section_counts)), section_counts.index, rotation=45)
        plt.ylabel('数量')
        plt.title('断面类型分布')

        # 4. 频率-跨度关系
        plt.subplot(3, 4, 4)
        plt.scatter(self.df['span_length'], self.df['frequency_1st'], alpha=0.6, s=20)
        plt.xlabel('主跨长度 (m)')
        plt.ylabel('一阶频率 (Hz)')
        plt.title('频率-跨度关系')
        plt.grid(True, alpha=0.3)

        # 5. 斯克鲁顿数-VIV幅度关系
        plt.subplot(3, 4, 5)
        plt.scatter(self.df['scruton_number'], self.df['viv_amplitude'], alpha=0.6, s=20)
        plt.xlabel('斯克鲁顿数')
        plt.ylabel('VIV幅度')
        plt.title('斯克鲁顿数-VIV幅度关系')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)

        # 6. 按桥型的VIV幅度箱线图
        plt.subplot(3, 4, 6)
        bridge_types = self.df['bridge_type'].unique()
        viv_by_type = [self.df[self.df['bridge_type'] == bt]['viv_amplitude'] for bt in bridge_types]
        plt.boxplot(viv_by_type, labels=bridge_types)
        plt.xticks(rotation=45)
        plt.ylabel('VIV幅度')
        plt.title('按桥型的VIV幅度分布')

        # 7. 阻尼比分布
        plt.subplot(3, 4, 7)
        plt.hist(self.df['damping_ratio'], bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('阻尼比')
        plt.ylabel('频数')
        plt.title('阻尼比分布')
        plt.grid(True, alpha=0.3)

        # 8. 临界风速分布
        plt.subplot(3, 4, 8)
        plt.hist(self.df['wind_speed_critical'], bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('临界风速 (m/s)')
        plt.ylabel('频数')
        plt.title('临界风速分布')
        plt.grid(True, alpha=0.3)

        # 9. 塔高跨度比分布
        plt.subplot(3, 4, 9)
        plt.hist(self.df['height_to_span_ratio'], bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('塔高跨度比')
        plt.ylabel('频数')
        plt.title('塔高跨度比分布')
        plt.grid(True, alpha=0.3)

        # 10. 建造年份分布
        plt.subplot(3, 4, 10)
        plt.hist(self.df['construction_year'], bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('建造年份')
        plt.ylabel('频数')
        plt.title('建造年份分布')
        plt.grid(True, alpha=0.3)

        # 11. 相关性热力图
        plt.subplot(3, 4, 11)
        numeric_cols = ['span_length', 'frequency_1st', 'damping_ratio', 'viv_amplitude',
                       'wind_speed_critical', 'scruton_number']
        correlation_matrix = self.df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('特征相关性矩阵')

        # 12. 真实桥梁 vs 合成桥梁对比
        plt.subplot(3, 4, 12)
        real_bridges = self.df[self.df['bridge_id'].str.contains('REAL')]
        synth_bridges = self.df[self.df['bridge_id'].str.contains('SYNTH')]

        plt.scatter(real_bridges['span_length'], real_bridges['viv_amplitude'],
                   alpha=0.7, label='真实桥梁', s=30)
        plt.scatter(synth_bridges['span_length'], synth_bridges['viv_amplitude'],
                   alpha=0.7, label='合成桥梁', s=30)
        plt.xlabel('主跨长度 (m)')
        plt.ylabel('VIV幅度')
        plt.title('真实 vs 合成桥梁对比')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('D:\Desktop\SRTPCode\project\dataset_validation_report.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        print("验证图表已保存: dataset_validation_report.png")

    def generate_report(self):
        """生成完整的验证报告"""
        print("\n=== 生成验证报告 ===")

        # 执行所有验证
        basic_stats = self.basic_statistics()
        physical_validation = self.physical_validation()
        engineering_relationships = self.engineering_relationships()
        statistical_tests = self.statistical_tests()
        self.create_validation_plots()

        # 生成文本报告
        report_path = 'D:\Desktop\SRTPCode\project\dataset_validation_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("扩展桥梁VIV数据集验证报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 基础统计
            f.write("1. 基础统计信息\n")
            f.write("-" * 30 + "\n")
            f.write(f"数据集大小: {basic_stats['shape']}\n")
            f.write(f"缺失值数量: {basic_stats['missing_values']}\n")
            f.write(f"桥梁类型分布:\n")
            for bridge_type, count in basic_stats['bridge_types'].items():
                f.write(f"  {bridge_type}: {count}\n")
            f.write(f"断面类型分布:\n")
            for section_type, count in basic_stats['section_types'].items():
                f.write(f"  {section_type}: {count}\n")

            # 物理验证
            f.write(f"\n2. 物理参数验证\n")
            f.write("-" * 30 + "\n")
            for param, results in physical_validation.items():
                f.write(f"{param}:\n")
                f.write(f"  有效性: {results['valid_percentage']:.1f}%\n")
                f.write(f"  预期范围: {results['expected_range']}\n")
                f.write(f"  实际范围: [{results['actual_range'][0]:.3f}, {results['actual_range'][1]:.3f}]\n")

            # 工程关系
            f.write(f"\n3. 工程关系验证\n")
            f.write("-" * 30 + "\n")
            f.write(f"频率-跨度相关性: {engineering_relationships['freq_span_correlation']:.3f}\n")
            f.write(f"斯克鲁顿数-VIV相关性: {engineering_relationships['scruton_viv_correlation']:.3f}\n")

            # 统计检验
            f.write(f"\n4. 统计检验结果\n")
            f.write("-" * 30 + "\n")
            for var, results in statistical_tests.items():
                if var != 'anova_bridge_types':
                    f.write(f"{var}: {'正态分布' if results['is_normal'] else '非正态分布'} (p={results['p_value']:.6f})\n")

            # 总结
            f.write(f"\n5. 数据质量总结\n")
            f.write("-" * 30 + "\n")
            if basic_stats['missing_values'] == 0:
                f.write("[+] 无缺失值\n")
            else:
                f.write("[-] 存在缺失值\n")

            valid_params = sum(1 for results in physical_validation.values() if results['valid_percentage'] > 95)
            total_params = len(physical_validation)
            f.write(f"[+] {valid_params}/{total_params} 个参数通过物理合理性检验 (>95%)\n")

            if abs(engineering_relationships['freq_span_correlation']) > 0.3:
                f.write("[+] 频率-跨度关系符合工程规律\n")
            else:
                f.write("[-] 频率-跨度关系需要改进\n")

        print(f"验证报告已保存: {report_path}")
        return self.validation_results

def main():
    """主函数"""
    print("=== 扩展桥梁VIV数据集验证器 ===")

    # 验证数据集
    csv_path = 'D:\Desktop\SRTPCode\project\expanded_bridge_viv_dataset.csv'
    validator = DatasetValidator(csv_path)

    # 生成完整报告
    results = validator.generate_report()

    print(f"\n[完成] 数据集验证完成！")
    print("生成的文件:")
    print("- dataset_validation_report.png (验证图表)")
    print("- dataset_validation_report.txt (验证报告)")

if __name__ == "__main__":
    main()