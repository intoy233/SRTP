"""
深度学习模型模块
基于实验结果的工程建议，开发深度神经网络来捕获桥梁VIV的非线性关系
使用sklearn的MLPRegressor作为深度学习的替代方案
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.ensemble import VotingRegressor, VotingClassifier
from sklearn.multioutput import MultiOutputRegressor
import joblib
import json
import os
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class PhysicsInformedMLP:
    """物理信息多层感知器 - 结合物理约束的神经网络模型"""

    def __init__(self, hidden_layer_sizes=(100, 50, 25), alpha=0.001,
                 learning_rate_init=0.001, max_iter=500):
        """
        初始化物理信息MLP模型

        Args:
            hidden_layer_sizes: 隐藏层大小元组
            alpha: L2正则化参数
            learning_rate_init: 初始学习率
            max_iter: 最大迭代次数
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter

        # 振幅回归模型
        self.amplitude_model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            solver='adam'
        )

        # 风险分类模型
        self.risk_model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            solver='adam'
        )

        self.scaler = StandardScaler()
        self.target_scaler = RobustScaler()
        self.is_fitted = False

    def _apply_physics_constraints(self, predictions: np.ndarray, features: np.ndarray) -> np.ndarray:
        """应用物理约束到预测结果"""
        constrained_predictions = predictions.copy()

        # 约束1: 振幅不应为负值
        constrained_predictions = np.maximum(constrained_predictions, 0)

        # 约束2: 振幅与阻尼比的反比关系
        if features.shape[1] > 8:  # 假设阻尼比是最后一个特征
            damping_ratio = features[:, -1]
            # 高阻尼对应低振幅的软约束
            damping_factor = 1 / (1 + 10 * damping_ratio)
            constrained_predictions = constrained_predictions * (0.5 + 0.5 * damping_factor)

        # 约束3: 过高振幅的限制 (>150mm被认为不现实)
        constrained_predictions = np.minimum(constrained_predictions, 150)

        return constrained_predictions

    def _prepare_risk_labels(self, amplitudes: np.ndarray) -> np.ndarray:
        """准备风险标签"""
        risk_labels = np.zeros(len(amplitudes))
        risk_labels[amplitudes >= 20] = 1  # 中风险
        risk_labels[amplitudes >= 40] = 2  # 高风险
        return risk_labels

    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练模型"""
        # 数据预处理
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).ravel()

        # 准备风险标签
        risk_labels = self._prepare_risk_labels(y)

        # 训练振幅回归模型
        self.amplitude_model.fit(X_scaled, y_scaled)

        # 训练风险分类模型
        self.risk_model.fit(X_scaled, risk_labels)

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")

        X_scaled = self.scaler.transform(X)

        # 振幅预测
        amplitude_scaled = self.amplitude_model.predict(X_scaled)
        amplitude = self.target_scaler.inverse_transform(amplitude_scaled.reshape(-1, 1)).ravel()

        # 应用物理约束
        amplitude = self._apply_physics_constraints(amplitude, X)

        # 风险预测
        risk_classes = self.risk_model.predict(X_scaled)
        risk_probabilities = self.risk_model.predict_proba(X_scaled)

        return {
            'amplitude': amplitude,
            'risk_classes': risk_classes,
            'risk_probabilities': risk_probabilities
        }

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """评估模型"""
        predictions = self.predict(X)
        risk_labels = self._prepare_risk_labels(y)

        # 回归指标
        mse = mean_squared_error(y, predictions['amplitude'])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, predictions['amplitude'])
        r2 = r2_score(y, predictions['amplitude'])

        # 分类指标
        risk_accuracy = accuracy_score(risk_labels, predictions['risk_classes'])

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'risk_accuracy': risk_accuracy
        }


class DeepEnsembleRegressor:
    """深度集成回归器 - 多个神经网络的集成"""

    def __init__(self, n_models=3):
        """
        初始化深度集成模型

        Args:
            n_models: 集成模型的数量
        """
        self.n_models = n_models
        self.models = []

        # 创建不同架构的模型
        architectures = [
            (100, 50),           # 浅层网络
            (150, 100, 50),      # 中层网络
            (200, 150, 100, 50), # 深层网络
        ]

        for i in range(n_models):
            arch = architectures[i % len(architectures)]
            model = PhysicsInformedMLP(
                hidden_layer_sizes=arch,
                alpha=0.001 * (1 + i * 0.5),  # 不同的正则化强度
                learning_rate_init=0.001 / (1 + i * 0.2),  # 不同的学习率
                max_iter=300 + i * 100  # 不同的训练轮数
            )
            self.models.append(model)

        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练所有模型"""
        for i, model in enumerate(self.models):
            print(f"训练集成模型 {i+1}/{self.n_models}")
            model.fit(X, y)

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """集成预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")

        amplitude_predictions = []
        risk_predictions = []
        risk_probabilities = []

        for model in self.models:
            pred = model.predict(X)
            amplitude_predictions.append(pred['amplitude'])
            risk_predictions.append(pred['risk_classes'])
            risk_probabilities.append(pred['risk_probabilities'])

        # 振幅预测使用平均值
        ensemble_amplitude = np.mean(amplitude_predictions, axis=0)

        # 风险分类使用投票
        risk_votes = np.array(risk_predictions).T
        ensemble_risk = np.array([np.bincount(votes).argmax() for votes in risk_votes])

        # 风险概率使用平均
        ensemble_risk_probs = np.mean(risk_probabilities, axis=0)

        return {
            'amplitude': ensemble_amplitude,
            'risk_classes': ensemble_risk,
            'risk_probabilities': ensemble_risk_probs
        }

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """评估集成模型"""
        predictions = self.predict(X)
        risk_labels = PhysicsInformedMLP(max_iter=1)._prepare_risk_labels(y)

        # 回归指标
        mse = mean_squared_error(y, predictions['amplitude'])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, predictions['amplitude'])
        r2 = r2_score(y, predictions['amplitude'])

        # 分类指标
        risk_accuracy = accuracy_score(risk_labels, predictions['risk_classes'])

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'risk_accuracy': risk_accuracy
        }


class AdaptiveNeuralNetwork:
    """自适应神经网络 - 根据数据规模调整网络结构"""

    def __init__(self, min_samples_for_deep=100):
        self.min_samples_for_deep = min_samples_for_deep
        self.model = None
        self.is_fitted = False

    def _select_architecture(self, n_samples: int, n_features: int) -> tuple:
        """根据样本数和特征数选择网络架构"""
        if n_samples < self.min_samples_for_deep:
            # 小数据集使用简单架构
            return (min(50, n_features * 2), min(25, n_features))
        elif n_samples < 500:
            # 中等数据集使用中等架构
            return (min(100, n_features * 3), min(50, n_features * 2), min(25, n_features))
        else:
            # 大数据集使用复杂架构
            return (min(200, n_features * 4), min(100, n_features * 3), min(50, n_features * 2))

    def fit(self, X: np.ndarray, y: np.ndarray):
        """自适应训练"""
        n_samples, n_features = X.shape

        # 选择合适的架构
        architecture = self._select_architecture(n_samples, n_features)

        # 选择合适的正则化强度
        alpha = 0.01 if n_samples < 100 else 0.001

        print(f"自适应架构: {architecture}, 正则化强度: {alpha}")

        # 创建模型
        if n_samples >= self.min_samples_for_deep * 2:
            # 使用集成模型
            self.model = DeepEnsembleRegressor(n_models=3)
        else:
            # 使用单个物理信息模型
            self.model = PhysicsInformedMLP(
                hidden_layer_sizes=architecture,
                alpha=alpha,
                max_iter=min(1000, max(200, n_samples * 5))
            )

        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """评估"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        return self.model.evaluate(X, y)


def run_deep_learning_experiment(data_path: str = "../data/bridge_dataset_fixed.csv",
                                augmented_data_path: str = "../data/bridge_viv_augmented.csv",
                                output_dir: str = "../experiments/") -> Dict:
    """运行深度学习实验"""

    print("=== 桥梁VIV深度学习实验 ===")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    try:
        # 优先使用扩增数据
        if os.path.exists(augmented_data_path):
            data = pd.read_csv(augmented_data_path)
            print(f"使用扩增数据集: {len(data)} 个样本")
        else:
            data = pd.read_csv(data_path)
            print(f"使用原始数据集: {len(data)} 个样本")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return {}

    # 准备特征和目标 - 只选择数值特征
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if 'Max_Amplitude_mm' in numeric_cols:
        numeric_cols.remove('Max_Amplitude_mm')

    print(f"数值特征列: {numeric_cols}")

    X = data[numeric_cols].values
    y = data['Max_Amplitude_mm'].values

    print(f"特征维度: {X.shape}")
    print(f"目标变量范围: [{y.min():.2f}, {y.max():.2f}] mm")

    # 训练测试分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # 实验配置
    model_configs = {
        'physics_informed': {
            'model_class': PhysicsInformedMLP,
            'params': {
                'hidden_layer_sizes': (100, 50, 25),
                'alpha': 0.001,
                'max_iter': 500
            }
        },
        'deep_ensemble': {
            'model_class': DeepEnsembleRegressor,
            'params': {
                'n_models': 3
            }
        },
        'adaptive_network': {
            'model_class': AdaptiveNeuralNetwork,
            'params': {
                'min_samples_for_deep': 100
            }
        }
    }

    results = {}

    for model_name, config in model_configs.items():
        print(f"\n--- 训练 {model_name} 模型 ---")

        try:
            # 创建模型
            model_class = config['model_class']
            model = model_class(**config['params'])

            # 训练
            model.fit(X_train, y_train)

            # 评估
            train_metrics = model.evaluate(X_train, y_train)
            test_metrics = model.evaluate(X_test, y_test)

            # 保存模型
            model_path = os.path.join(output_dir, f"deep_learning_{model_name}.pkl")
            joblib.dump(model, model_path)

            # 记录结果
            results[model_name] = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'config': config['params']
            }

            # 计算过拟合程度
            overfitting = train_metrics['r2'] - test_metrics['r2']

            # 打印结果
            print(f"{model_name} 训练完成:")
            print(f"  训练 R2: {train_metrics['r2']:.4f}")
            print(f"  测试 R2: {test_metrics['r2']:.4f}")
            print(f"  测试 RMSE: {test_metrics['rmse']:.2f} mm")
            print(f"  风险分类准确率: {test_metrics['risk_accuracy']:.4f}")
            print(f"  过拟合程度: {overfitting:.4f}")

        except Exception as e:
            print(f"{model_name} 模型训练失败: {e}")
            continue

    # 保存实验结果
    experiment_file = os.path.join(output_dir, "deep_learning_experiment.json")

    with open(experiment_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n实验结果已保存到: {experiment_file}")

    # 与之前的最佳结果比较
    if results:
        print("\n=== 深度学习vs传统模型对比 ===")
        best_traditional_r2 = 0.0804  # 步骤1线性回归的结果

        best_dl_r2 = max([result['test_metrics']['r2'] for result in results.values()])
        best_dl_model = max(results.keys(), key=lambda k: results[k]['test_metrics']['r2'])

        print(f"最佳传统模型 R2: {best_traditional_r2:.4f}")
        print(f"最佳深度学习模型 R2: {best_dl_r2:.4f} ({best_dl_model})")

        if best_dl_r2 > best_traditional_r2:
            improvement = ((best_dl_r2 - best_traditional_r2) / best_traditional_r2) * 100
            print(f"深度学习模型提升: {improvement:.1f}%")
        else:
            print("传统模型仍然表现更好")

    return results


def hyperparameter_optimization(X: np.ndarray, y: np.ndarray) -> Dict:
    """超参数优化实验"""
    print("\n=== 神经网络超参数优化 ===")

    # 定义超参数搜索空间
    param_grid = {
        'hidden_layer_sizes': [
            (50,), (100,), (150,),
            (50, 25), (100, 50), (150, 100),
            (100, 50, 25), (150, 100, 50)
        ],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'learning_rate_init': [0.0001, 0.001, 0.01]
    }

    # 创建基础模型
    base_model = MLPRegressor(
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        solver='adam'
    )

    # 网格搜索
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )

    print("开始超参数优化...")
    grid_search.fit(X, y)

    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳CV R2: {grid_search.best_score_:.4f}")

    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_model': grid_search.best_estimator_
    }


if __name__ == "__main__":
    # 运行深度学习实验
    results = run_deep_learning_experiment()

    if results:
        print("\n=== 深度学习实验总结 ===")
        for model_name, result in results.items():
            test_metrics = result['test_metrics']
            train_metrics = result['train_metrics']
            overfitting = train_metrics['r2'] - test_metrics['r2']

            print(f"{model_name}:")
            print(f"  测试 R2: {test_metrics['r2']:.4f}")
            print(f"  测试 RMSE: {test_metrics['rmse']:.2f}mm")
            print(f"  风险准确率: {test_metrics['risk_accuracy']:.4f}")
            print(f"  过拟合程度: {overfitting:.4f}")
            print()

        # 如果有扩增数据，运行超参数优化
        try:
            augmented_data_path = "../data/bridge_viv_augmented.csv"
            if os.path.exists(augmented_data_path):
                data = pd.read_csv(augmented_data_path)
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if 'Max_Amplitude_mm' in numeric_cols:
                    numeric_cols.remove('Max_Amplitude_mm')
                X = data[numeric_cols].values
                y = data['Max_Amplitude_mm'].values

                optimization_results = hyperparameter_optimization(X, y)
                print(f"超参数优化完成，最佳R2: {optimization_results['best_score']:.4f}")
        except Exception as e:
            print(f"超参数优化失败: {e}")