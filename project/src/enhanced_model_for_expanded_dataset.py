#!/usr/bin/env python3
"""
增强桥梁VIV风险评估模型 - 适配扩展数据集
处理1000个样本，包含桥梁类型和断面类型特征
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

class EnhancedBridgeVIVModel:
    def __init__(self, data_path):
        """初始化增强桥梁VIV模型"""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.results = {}

    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        print("=== 加载和预处理扩展数据集 ===")

        # 加载数据
        self.df = pd.read_csv(self.data_path, encoding='utf-8-sig')
        print(f"原始数据形状: {self.df.shape}")

        # 选择特征
        feature_columns = [
            # 基础几何特征
            'span_length', 'deck_width', 'tower_height', 'height_to_span_ratio',

            # 动力学特征
            'frequency_1st', 'damping_ratio', 'mass_per_length', 'stiffness',

            # 空气动力学特征
            'wind_speed_critical', 'drag_coefficient', 'strouhal_number',
            'aspect_ratio', 'reynolds_number',

            # VIV相关特征
            'scruton_number', 'reduced_velocity', 'lock_in_range',

            # 类别特征（编码后）
            'bridge_type_code', 'section_type_code', 'construction_year'
        ]

        # 检查特征是否存在
        available_features = []
        for feature in feature_columns:
            if feature in self.df.columns:
                available_features.append(feature)
            else:
                print(f"警告: 特征 {feature} 不存在于数据集中")

        print(f"可用特征数量: {len(available_features)}")

        # 创建特征矩阵
        X = self.df[available_features].copy()

        # 目标变量
        y = self.df['viv_amplitude'].copy()

        # 数据清洗
        # 移除极端异常值
        for col in X.select_dtypes(include=[np.number]).columns:
            Q1 = X[col].quantile(0.01)
            Q3 = X[col].quantile(0.99)
            mask = (X[col] >= Q1) & (X[col] <= Q3)
            X = X[mask]
            y = y[mask]

        print(f"清洗后数据形状: {X.shape}")

        # 分割数据
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=self.df.loc[X.index, 'bridge_type']
        )

        # 标准化特征
        numeric_features = self.X_train.select_dtypes(include=[np.number]).columns
        self.X_train_scaled = self.X_train.copy()
        self.X_test_scaled = self.X_test.copy()

        self.X_train_scaled[numeric_features] = self.scaler.fit_transform(self.X_train[numeric_features])
        self.X_test_scaled[numeric_features] = self.scaler.transform(self.X_test[numeric_features])

        print(f"训练集形状: {self.X_train.shape}")
        print(f"测试集形状: {self.X_test.shape}")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def create_physics_based_features(self):
        """创建基于物理原理的特征"""
        print("\n=== 创建物理基础特征 ===")

        def add_physics_features(df):
            df_new = df.copy()

            # 维度1：基于风工程的无量纲参数
            if 'wind_speed_critical' in df.columns and 'deck_width' in df.columns and 'frequency_1st' in df.columns:
                df_new['reduced_velocity_enhanced'] = df['wind_speed_critical'] / (df['frequency_1st'] * df['deck_width'])

            # 维度2：结构动力学参数
            if 'mass_per_length' in df.columns and 'stiffness' in df.columns:
                df_new['natural_frequency_calc'] = np.sqrt(df['stiffness'] / df['mass_per_length']) / (2 * np.pi)

            # 维度3：几何比例参数
            if 'span_length' in df.columns and 'deck_width' in df.columns:
                df_new['slenderness_ratio'] = df['span_length'] / df['deck_width']

            if 'tower_height' in df.columns and 'deck_width' in df.columns:
                df_new['tower_width_ratio'] = df['tower_height'] / df['deck_width']

            # 维度4：空气动力学参数组合
            if 'strouhal_number' in df.columns and 'reynolds_number' in df.columns:
                df_new['strouhal_reynolds'] = df['strouhal_number'] * np.log(df['reynolds_number'] + 1)

            # 维度5：VIV敏感性指标
            if 'scruton_number' in df.columns and 'damping_ratio' in df.columns:
                df_new['viv_susceptibility'] = 1 / (df['scruton_number'] * df['damping_ratio'] + 0.001)

            # 维度6：桥梁类型相关特征
            if 'bridge_type_code' in df.columns and 'span_length' in df.columns:
                # 不同桥型的跨度敏感性
                df_new['type_span_interaction'] = df['bridge_type_code'] * np.log(df['span_length'] + 1)

            if 'section_type_code' in df.columns and 'aspect_ratio' in df.columns:
                # 断面类型与宽厚比的交互
                df_new['section_aspect_interaction'] = df['section_type_code'] * df['aspect_ratio']

            return df_new

        # 为训练集和测试集添加物理特征
        self.X_train_physics = add_physics_features(self.X_train_scaled)
        self.X_test_physics = add_physics_features(self.X_test_scaled)

        print(f"添加物理特征后训练集形状: {self.X_train_physics.shape}")
        print(f"新增特征数量: {self.X_train_physics.shape[1] - self.X_train_scaled.shape[1]}")

        return self.X_train_physics, self.X_test_physics

    def train_multiple_models(self):
        """训练多个机器学习模型"""
        print("\n=== 训练多个机器学习模型 ===")

        # 创建物理特征
        X_train_final, X_test_final = self.create_physics_based_features()

        # 定义模型
        models_config = {
            'Ridge': {
                'model': Ridge(),
                'params': {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]}
            },
            'ElasticNet': {
                'model': ElasticNet(),
                'params': {
                    'alpha': [0.1, 0.5, 1.0],
                    'l1_ratio': [0.1, 0.5, 0.9]
                }
            },
            'RandomForest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 15, 20],
                    'min_samples_split': [5, 10]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [5, 7, 10]
                }
            },
            'SVR': {
                'model': SVR(),
                'params': {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto'],
                    'kernel': ['rbf', 'linear']
                }
            }
        }

        # 训练和评估每个模型
        for model_name, config in models_config.items():
            print(f"\n训练 {model_name} 模型...")

            # 网格搜索最佳参数
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )

            grid_search.fit(X_train_final, self.y_train)

            # 获取最佳模型
            best_model = grid_search.best_estimator_

            # 预测
            train_pred = best_model.predict(X_train_final)
            test_pred = best_model.predict(X_test_final)

            # 评估指标
            train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
            train_r2 = r2_score(self.y_train, train_pred)
            test_r2 = r2_score(self.y_test, test_pred)
            train_mae = mean_absolute_error(self.y_train, train_pred)
            test_mae = mean_absolute_error(self.y_test, test_pred)

            # 交叉验证分数
            cv_scores = cross_val_score(best_model, X_train_final, self.y_train,
                                      cv=5, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
            cv_std = np.sqrt(cv_scores.std())

            # 存储结果
            self.models[model_name] = best_model
            self.results[model_name] = {
                'best_params': grid_search.best_params_,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'cv_rmse': cv_rmse,
                'cv_std': cv_std,
                'train_pred': train_pred,
                'test_pred': test_pred
            }

            print(f"  最佳参数: {grid_search.best_params_}")
            print(f"  测试RMSE: {test_rmse:.4f}")
            print(f"  测试R²: {test_r2:.4f}")
            print(f"  交叉验证RMSE: {cv_rmse:.4f}±{cv_std:.4f}")

        return self.results

    def feature_importance_analysis(self):
        """特征重要性分析"""
        print("\n=== 特征重要性分析 ===")

        # 使用随机森林进行特征重要性分析
        if 'RandomForest' in self.models:
            rf_model = self.models['RandomForest']
            feature_names = self.X_train_physics.columns
            importances = rf_model.feature_importances_

            # 创建特征重要性DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

            print("前15个最重要特征:")
            print(importance_df.head(15))

            # 绘制特征重要性图
            plt.figure(figsize=(12, 8))
            plt.barh(range(15), importance_df.head(15)['importance'])
            plt.yticks(range(15), importance_df.head(15)['feature'])
            plt.xlabel('特征重要性')
            plt.title('随机森林特征重要性排序（前15个）')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('D:\Desktop\SRTPCode\project\feature_importance_expanded.png',
                       dpi=300, bbox_inches='tight')
            plt.close()

            return importance_df

    def model_comparison_visualization(self):
        """模型比较可视化"""
        print("\n=== 生成模型比较图表 ===")

        # 性能对比图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. RMSE对比
        model_names = list(self.results.keys())
        train_rmse = [self.results[name]['train_rmse'] for name in model_names]
        test_rmse = [self.results[name]['test_rmse'] for name in model_names]

        x = np.arange(len(model_names))
        width = 0.35

        axes[0, 0].bar(x - width/2, train_rmse, width, label='训练RMSE', alpha=0.8)
        axes[0, 0].bar(x + width/2, test_rmse, width, label='测试RMSE', alpha=0.8)
        axes[0, 0].set_xlabel('模型')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].set_title('模型RMSE性能对比')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. R²对比
        train_r2 = [self.results[name]['train_r2'] for name in model_names]
        test_r2 = [self.results[name]['test_r2'] for name in model_names]

        axes[0, 1].bar(x - width/2, train_r2, width, label='训练$R^2$', alpha=0.8)
        axes[0, 1].bar(x + width/2, test_r2, width, label='测试$R^2$', alpha=0.8)
        axes[0, 1].set_xlabel('模型')
        axes[0, 1].set_ylabel('$R^2$ Score')
        axes[0, 1].set_title('模型$R^2$性能对比')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(model_names, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 预测vs实际散点图（最佳模型）
        best_model_name = min(model_names, key=lambda x: self.results[x]['test_rmse'])
        best_results = self.results[best_model_name]

        axes[1, 0].scatter(self.y_test, best_results['test_pred'], alpha=0.6)
        axes[1, 0].plot([self.y_test.min(), self.y_test.max()],
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('实际VIV幅度')
        axes[1, 0].set_ylabel('预测VIV幅度')
        axes[1, 0].set_title(f'最佳模型预测效果 ({best_model_name})')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 交叉验证RMSE对比
        cv_rmse = [self.results[name]['cv_rmse'] for name in model_names]
        cv_std = [self.results[name]['cv_std'] for name in model_names]

        axes[1, 1].bar(model_names, cv_rmse, yerr=cv_std, capsize=5, alpha=0.8)
        axes[1, 1].set_xlabel('模型')
        axes[1, 1].set_ylabel('交叉验证RMSE')
        axes[1, 1].set_title('交叉验证性能对比')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('D:\Desktop\SRTPCode\project\model_comparison_expanded.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        print("模型比较图表已保存: model_comparison_expanded.png")

    def bridge_type_analysis(self):
        """按桥梁类型分析模型性能"""
        print("\n=== 按桥梁类型分析模型性能 ===")

        # 获取最佳模型
        best_model_name = min(self.results.keys(), key=lambda x: self.results[x]['test_rmse'])
        best_predictions = self.results[best_model_name]['test_pred']

        # 获取测试集的桥梁类型
        test_bridge_types = self.df.loc[self.X_test.index, 'bridge_type']

        # 按桥梁类型计算性能
        bridge_performance = {}
        for bridge_type in test_bridge_types.unique():
            mask = test_bridge_types == bridge_type
            type_y_true = self.y_test[mask]
            type_y_pred = best_predictions[mask]

            if len(type_y_true) > 0:
                bridge_performance[bridge_type] = {
                    'rmse': np.sqrt(mean_squared_error(type_y_true, type_y_pred)),
                    'r2': r2_score(type_y_true, type_y_pred),
                    'mae': mean_absolute_error(type_y_true, type_y_pred),
                    'count': len(type_y_true),
                    'mean_actual': type_y_true.mean(),
                    'mean_predicted': type_y_pred.mean()
                }

        # 显示结果
        print(f"使用最佳模型 ({best_model_name}) 的按桥型性能分析:")
        for bridge_type, perf in bridge_performance.items():
            print(f"\n{bridge_type}:")
            print(f"  样本数: {perf['count']}")
            print(f"  RMSE: {perf['rmse']:.4f}")
            print(f"  R²: {perf['r2']:.4f}")
            print(f"  MAE: {perf['mae']:.4f}")
            print(f"  实际平均VIV: {perf['mean_actual']:.4f}")
            print(f"  预测平均VIV: {perf['mean_predicted']:.4f}")

        return bridge_performance

    def generate_final_report(self):
        """生成最终报告"""
        print("\n=== 生成最终模型报告 ===")

        # 确定最佳模型
        best_model_name = min(self.results.keys(), key=lambda x: self.results[x]['test_rmse'])
        best_results = self.results[best_model_name]

        report_path = 'D:\Desktop\SRTPCode\project\enhanced_model_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("增强桥梁VIV风险评估模型报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据集大小: {self.df.shape}\n")
            f.write(f"训练集大小: {self.X_train.shape}\n")
            f.write(f"测试集大小: {self.X_test.shape}\n\n")

            f.write("模型性能对比:\n")
            f.write("-" * 40 + "\n")
            for model_name, results in self.results.items():
                f.write(f"{model_name}:\n")
                f.write(f"  测试RMSE: {results['test_rmse']:.4f}\n")
                f.write(f"  测试R²: {results['test_r2']:.4f}\n")
                f.write(f"  测试MAE: {results['test_mae']:.4f}\n")
                f.write(f"  交叉验证RMSE: {results['cv_rmse']:.4f}±{results['cv_std']:.4f}\n\n")

            f.write(f"最佳模型: {best_model_name}\n")
            f.write("-" * 40 + "\n")
            f.write(f"最佳参数: {best_results['best_params']}\n")
            f.write(f"测试集性能:\n")
            f.write(f"  RMSE: {best_results['test_rmse']:.4f}\n")
            f.write(f"  R²: {best_results['test_r2']:.4f}\n")
            f.write(f"  MAE: {best_results['test_mae']:.4f}\n")

            # 与之前80样本模型的比较
            f.write(f"\n与小样本模型对比:\n")
            f.write("-" * 40 + "\n")
            f.write(f"数据集扩充: 80 → {self.df.shape[0]} 样本\n")
            f.write(f"特征数扩充: 25 → {self.X_train_physics.shape[1]} 特征\n")
            f.write(f"新增特征: 桥梁类型、断面类型、物理组合特征\n")

        print(f"最终报告已保存: {report_path}")
        return best_model_name, best_results

def main():
    """主函数"""
    print("=== 增强桥梁VIV风险评估模型 ===")

    # 初始化模型
    data_path = 'D:\Desktop\SRTPCode\project\expanded_bridge_viv_dataset.csv'
    model = EnhancedBridgeVIVModel(data_path)

    # 加载和预处理数据
    model.load_and_preprocess_data()

    # 训练多个模型
    results = model.train_multiple_models()

    # 特征重要性分析
    importance_df = model.feature_importance_analysis()

    # 模型比较可视化
    model.model_comparison_visualization()

    # 按桥梁类型分析
    bridge_performance = model.bridge_type_analysis()

    # 生成最终报告
    best_model_name, best_results = model.generate_final_report()

    print(f"\n[完成] 增强模型训练完成！")
    print(f"最佳模型: {best_model_name}")
    print(f"测试集RMSE: {best_results['test_rmse']:.4f}")
    print(f"测试集R²: {best_results['test_r2']:.4f}")

    print("\n生成的文件:")
    print("- feature_importance_expanded.png (特征重要性)")
    print("- model_comparison_expanded.png (模型比较)")
    print("- enhanced_model_report.txt (详细报告)")

if __name__ == "__main__":
    main()