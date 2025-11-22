"""
集成实验模块
结合高级特征工程和深度学习模型，实现完整的建模流程
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import joblib
import json
import os
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from advanced_feature_engineering import AdvancedBridgeFeatureEngineering
from deep_learning import PhysicsInformedMLP, AdaptiveNeuralNetwork


class IntegratedBridgeVIVModel:
    """集成的桥梁VIV建模系统"""

    def __init__(self, use_advanced_features: bool = True, model_type: str = 'adaptive'):
        self.use_advanced_features = use_advanced_features
        self.model_type = model_type
        self.feature_engineer = None
        self.model = None
        self.is_fitted = False

    def fit(self, data: pd.DataFrame, target_col: str = 'Max_Amplitude_mm'):
        """训练完整的建模流程"""
        print("=== 集成建模系统训练 ===")

        # 1. 特征工程
        if self.use_advanced_features:
            print("应用高级特征工程...")
            self.feature_engineer = AdvancedBridgeFeatureEngineering()
            X, y = self.feature_engineer.fit_transform(
                data,
                target_col=target_col,
                use_polynomial=True,
                use_interactions=True,
                feature_selection_method='mutual_info',
                k_features=25
            )
        else:
            print("使用基础特征...")
            # 只使用数值特征
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in numeric_cols:
                numeric_cols.remove(target_col)
            X = data[numeric_cols].values
            y = data[target_col].values

        # 2. 训练深度学习模型
        print(f"训练{self.model_type}模型...")
        if self.model_type == 'adaptive':
            self.model = AdaptiveNeuralNetwork(min_samples_for_deep=50)
        elif self.model_type == 'physics_informed':
            self.model = PhysicsInformedMLP(
                hidden_layer_sizes=(100, 50, 25),
                alpha=0.001,
                max_iter=500
            )

        self.model.fit(X, y)
        self.is_fitted = True

        print("集成模型训练完成")

    def predict(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")

        # 特征工程
        if self.use_advanced_features:
            X = self.feature_engineer.transform(data)
        else:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            X = data[numeric_cols].values

        # 模型预测
        predictions = self.model.predict(X)
        return predictions

    def evaluate(self, data: pd.DataFrame, target_col: str = 'Max_Amplitude_mm') -> Dict[str, float]:
        """评估模型"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")

        # 准备目标变量
        y_true = data[target_col].values

        # 预测
        predictions = self.predict(data)
        y_pred = predictions['amplitude']

        # 计算指标
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # 风险分类评估
        risk_true = self._prepare_risk_labels(y_true)
        risk_pred = predictions['risk_classes']
        risk_accuracy = accuracy_score(risk_true, risk_pred)

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'risk_accuracy': risk_accuracy
        }

    def _prepare_risk_labels(self, amplitudes: np.ndarray) -> np.ndarray:
        """准备风险标签"""
        risk_labels = np.zeros(len(amplitudes))
        risk_labels[amplitudes >= 20] = 1  # 中风险
        risk_labels[amplitudes >= 40] = 2  # 高风险
        return risk_labels

    def save_model(self, filepath: str):
        """保存模型"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")

        model_data = {
            'feature_engineer': self.feature_engineer,
            'model': self.model,
            'use_advanced_features': self.use_advanced_features,
            'model_type': self.model_type
        }

        joblib.dump(model_data, filepath)
        print(f"集成模型已保存到: {filepath}")

    def load_model(self, filepath: str):
        """加载模型"""
        model_data = joblib.load(filepath)

        self.feature_engineer = model_data['feature_engineer']
        self.model = model_data['model']
        self.use_advanced_features = model_data['use_advanced_features']
        self.model_type = model_data['model_type']
        self.is_fitted = True

        print(f"集成模型已从 {filepath} 加载")


def run_integrated_experiment(data_path: str = "../data/bridge_dataset_fixed.csv",
                             output_dir: str = "../experiments/") -> Dict:
    """运行集成实验"""
    print("=== 集成建模实验 ===")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    try:
        data = pd.read_csv(data_path)
        print(f"加载数据: {data.shape}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return {}

    # 训练测试分割
    train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)

    # 实验配置
    experiments = {
        'baseline_adaptive': {
            'use_advanced_features': False,
            'model_type': 'adaptive'
        },
        'baseline_physics': {
            'use_advanced_features': False,
            'model_type': 'physics_informed'
        },
        'advanced_adaptive': {
            'use_advanced_features': True,
            'model_type': 'adaptive'
        },
        'advanced_physics': {
            'use_advanced_features': True,
            'model_type': 'physics_informed'
        }
    }

    results = {}

    for exp_name, config in experiments.items():
        print(f"\n--- 运行 {exp_name} 实验 ---")

        try:
            # 创建模型
            model = IntegratedBridgeVIVModel(
                use_advanced_features=config['use_advanced_features'],
                model_type=config['model_type']
            )

            # 训练
            model.fit(train_data)

            # 评估
            train_metrics = model.evaluate(train_data)
            test_metrics = model.evaluate(test_data)

            # 保存模型
            model_path = os.path.join(output_dir, f"integrated_{exp_name}.pkl")
            model.save_model(model_path)

            # 记录结果
            results[exp_name] = {
                'config': config,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics
            }

            # 计算过拟合程度
            overfitting = train_metrics['r2'] - test_metrics['r2']

            print(f"{exp_name} 结果:")
            print(f"  训练 R2: {train_metrics['r2']:.4f}")
            print(f"  测试 R2: {test_metrics['r2']:.4f}")
            print(f"  测试 RMSE: {test_metrics['rmse']:.2f} mm")
            print(f"  风险准确率: {test_metrics['risk_accuracy']:.4f}")
            print(f"  过拟合程度: {overfitting:.4f}")

        except Exception as e:
            print(f"{exp_name} 实验失败: {e}")
            continue

    # 保存实验结果
    if results:
        result_file = os.path.join(output_dir, "integrated_experiment_results.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n实验结果已保存到: {result_file}")

        # 性能对比
        print("\n=== 集成实验性能对比 ===")
        best_r2 = 0
        best_model = ""

        for exp_name, result in results.items():
            test_r2 = result['test_metrics']['r2']
            test_rmse = result['test_metrics']['rmse']
            risk_acc = result['test_metrics']['risk_accuracy']

            print(f"{exp_name}:")
            print(f"  R2: {test_r2:.4f}")
            print(f"  RMSE: {test_rmse:.2f}mm")
            print(f"  风险准确率: {risk_acc:.4f}")

            if test_r2 > best_r2:
                best_r2 = test_r2
                best_model = exp_name

        print(f"\n最佳模型: {best_model} (R2={best_r2:.4f})")

        # 特征工程影响分析
        print("\n=== 特征工程影响分析 ===")
        for model_type in ['adaptive', 'physics']:
            baseline_key = f'baseline_{model_type}'
            advanced_key = f'advanced_{model_type}'

            if baseline_key in results and advanced_key in results:
                baseline_r2 = results[baseline_key]['test_metrics']['r2']
                advanced_r2 = results[advanced_key]['test_metrics']['r2']

                if baseline_r2 > 0:
                    improvement = ((advanced_r2 - baseline_r2) / baseline_r2) * 100
                    print(f"{model_type}模型:")
                    print(f"  基础特征 R2: {baseline_r2:.4f}")
                    print(f"  高级特征 R2: {advanced_r2:.4f}")
                    print(f"  提升幅度: {improvement:.1f}%")

    return results


def create_prediction_analysis(model_path: str, test_data_path: str, output_dir: str = "../results/"):
    """创建预测分析报告"""
    print("=== 预测分析报告 ===")

    # 加载模型和数据
    model = IntegratedBridgeVIVModel()
    model.load_model(model_path)

    test_data = pd.read_csv(test_data_path)
    y_true = test_data['Max_Amplitude_mm'].values

    # 预测
    predictions = model.predict(test_data)
    y_pred = predictions['amplitude']
    risk_pred = predictions['risk_classes']

    # 计算残差
    residuals = y_true - y_pred

    # 创建分析报告
    analysis = {
        'prediction_summary': {
            'total_samples': len(y_true),
            'mean_true': float(y_true.mean()),
            'mean_pred': float(y_pred.mean()),
            'std_true': float(y_true.std()),
            'std_pred': float(y_pred.std())
        },
        'error_analysis': {
            'mean_absolute_error': float(np.mean(np.abs(residuals))),
            'root_mean_square_error': float(np.sqrt(np.mean(residuals**2))),
            'max_positive_error': float(residuals.max()),
            'max_negative_error': float(residuals.min()),
            'error_std': float(residuals.std())
        },
        'risk_classification': {
            'total_low_risk': int(np.sum(risk_pred == 0)),
            'total_medium_risk': int(np.sum(risk_pred == 1)),
            'total_high_risk': int(np.sum(risk_pred == 2)),
        }
    }

    # 保存分析结果
    os.makedirs(output_dir, exist_ok=True)
    analysis_file = os.path.join(output_dir, "prediction_analysis.json")

    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    print(f"预测分析报告已保存到: {analysis_file}")
    return analysis


if __name__ == "__main__":
    # 运行集成实验
    results = run_integrated_experiment()

    # 如果实验成功，创建预测分析
    if results:
        # 使用最佳模型进行预测分析
        best_model_key = max(results.keys(), key=lambda k: results[k]['test_metrics']['r2'])
        model_path = f"../experiments/integrated_{best_model_key}.pkl"

        if os.path.exists(model_path):
            create_prediction_analysis(
                model_path,
                "../data/bridge_dataset_fixed.csv",
                "../results/"
            )