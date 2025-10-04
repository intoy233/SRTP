#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
桥梁VIV振幅预测 - 神经网络模型
对比简单NN vs 中等NN vs 岭回归基线(R2=0.962)
"""

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class VIVNeuralNetwork:
    """纯NumPy实现的神经网络 - 不依赖外部深度学习库"""

    def __init__(self, layers, learning_rate=0.01, epochs=1000, random_seed=42):
        """
        初始化神经网络

        Parameters:
        -----------
        layers : list
            网络结构,如[10, 20, 10, 1]表示输入10维,两个隐藏层20和10,输出1维
        learning_rate : float
            学习率
        epochs : int
            训练轮数
        random_seed : int
            随机种子
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_seed = random_seed

        # 初始化权重和偏置
        np.random.seed(random_seed)
        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            # Xavier初始化
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)

        self.train_loss_history = []
        self.val_loss_history = []

    def relu(self, x):
        """ReLU激活函数"""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """ReLU导数"""
        return (x > 0).astype(float)

    def forward(self, X):
        """前向传播"""
        self.activations = [X]
        self.z_values = []

        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)

            # 最后一层不用激活函数(线性输出)
            if i == len(self.weights) - 1:
                a = z
            else:
                a = self.relu(z)

            self.activations.append(a)

        return self.activations[-1]

    def backward(self, X, y):
        """反向传播"""
        m = X.shape[0]

        # 输出层误差
        delta = self.activations[-1] - y

        # 存储梯度
        weight_gradients = []
        bias_gradients = []

        # 反向传播
        for i in range(len(self.weights) - 1, -1, -1):
            # 计算梯度
            dw = np.dot(self.activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m

            weight_gradients.insert(0, dw)
            bias_gradients.insert(0, db)

            if i > 0:
                # 传播到前一层
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.z_values[i-1])

        # 更新权重和偏置
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]

    def compute_loss(self, y_true, y_pred):
        """计算MSE损失"""
        return np.mean((y_true - y_pred) ** 2)

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """训练模型"""
        for epoch in range(self.epochs):
            # 前向传播
            y_pred = self.forward(X_train)

            # 计算训练损失
            train_loss = self.compute_loss(y_train, y_pred)
            self.train_loss_history.append(train_loss)

            # 反向传播 - 必须在验证集forward之前进行
            self.backward(X_train, y_train)

            # 计算验证损失
            if X_val is not None and y_val is not None:
                y_val_pred = self.forward(X_val)
                val_loss = self.compute_loss(y_val, y_val_pred)
                self.val_loss_history.append(val_loss)

            # 打印进度
            if verbose and (epoch + 1) % 100 == 0:
                if X_val is not None:
                    print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
                else:
                    print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}')

    def predict(self, X):
        """预测"""
        return self.forward(X)


class VIVModelComparison:
    """VIV模型对比系统 - NN vs 岭回归"""

    def __init__(self, data_path='../data/enhanced_bridge_dataset.csv'):
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.scaler_mean = None
        self.scaler_std = None

    def load_and_prepare_data(self):
        """加载和准备数据"""
        print("="*60)
        print("加载数据...")
        print("="*60)

        self.df = pd.read_csv(self.data_path)
        print(f"数据集大小: {len(self.df)} 座桥梁")

        # 创建物理特征
        df_features = pd.DataFrame()

        # 基础特征
        df_features['Span_m'] = self.df['Span_m']
        df_features['Width_m'] = self.df['Width_m']
        df_features['Height_m'] = self.df['Height_m']
        df_features['Natural_Freq_Hz'] = self.df['Natural_Freq_Hz']
        df_features['Damping_Ratio'] = self.df['Damping_Ratio']

        # 物理衍生特征
        df_features['Width_Height_Ratio'] = self.df['Width_m'] / self.df['Height_m']
        df_features['Scruton_Number'] = self.df['Damping_Ratio'] * (self.df['Width_m'] / self.df['Height_m']) * 100
        df_features['VIV_Susceptibility'] = 1.0 / (self.df['Damping_Ratio'] + 1e-6)

        # 添加Reduced Velocity(如果有风速数据)
        if 'Critical_Wind_Speed_ms' in self.df.columns:
            df_features['Reduced_Velocity'] = self.df['Critical_Wind_Speed_ms'] / (self.df['Natural_Freq_Hz'] * self.df['Width_m'])
            df_features['Reduced_Velocity'].fillna(df_features['Reduced_Velocity'].median(), inplace=True)

        # 添加跨宽比
        df_features['Span_Width_Ratio'] = self.df['Span_m'] / self.df['Width_m']

        # 目标变量
        self.y = self.df['Max_Amplitude_mm'].values.reshape(-1, 1)

        # 特征矩阵
        self.X = df_features.values
        self.feature_names = df_features.columns.tolist()

        print(f"\n特征数量: {self.X.shape[1]}")
        print(f"特征列表: {self.feature_names}")
        print(f"目标变量: Max_Amplitude_mm")
        print(f"目标范围: {self.y.min():.1f} - {self.y.max():.1f} mm")

        return self.X, self.y

    def normalize_features(self, X_train, X_test):
        """标准化特征"""
        self.scaler_mean = np.mean(X_train, axis=0)
        self.scaler_std = np.std(X_train, axis=0) + 1e-8

        X_train_norm = (X_train - self.scaler_mean) / self.scaler_std
        X_test_norm = (X_test - self.scaler_mean) / self.scaler_std

        return X_train_norm, X_test_norm

    def train_test_split(self, test_size=0.2, random_seed=42):
        """划分训练集和测试集"""
        np.random.seed(random_seed)
        n_samples = len(self.X)
        indices = np.random.permutation(n_samples)

        n_test = int(n_samples * test_size)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        X_train = self.X[train_indices]
        X_test = self.X[test_indices]
        y_train = self.y[train_indices]
        y_test = self.y[test_indices]

        print(f"\n数据划分:")
        print(f"  训练集: {len(X_train)} 样本")
        print(f"  测试集: {len(X_test)} 样本")

        return X_train, X_test, y_train, y_test

    def ridge_regression_baseline(self, X_train, y_train, X_test, y_test, alpha=0.1):
        """岭回归基线模型"""
        print(f"\n{'='*60}")
        print("岭回归基线模型 (alpha={})".format(alpha))
        print("="*60)

        # 标准化
        X_train_norm, X_test_norm = self.normalize_features(X_train, X_test)

        # 添加截距项
        X_train_with_bias = np.column_stack([np.ones(len(X_train_norm)), X_train_norm])
        X_test_with_bias = np.column_stack([np.ones(len(X_test_norm)), X_test_norm])

        # 岭回归求解
        n_features = X_train_with_bias.shape[1]
        identity = np.eye(n_features)
        identity[0, 0] = 0  # 不惩罚截距

        XTX = np.dot(X_train_with_bias.T, X_train_with_bias)
        XTy = np.dot(X_train_with_bias.T, y_train)

        coefficients = np.linalg.solve(XTX + alpha * identity, XTy)

        # 预测
        y_train_pred = np.dot(X_train_with_bias, coefficients)
        y_test_pred = np.dot(X_test_with_bias, coefficients)

        # 评估
        train_r2 = 1 - np.sum((y_train - y_train_pred)**2) / np.sum((y_train - np.mean(y_train))**2)
        test_r2 = 1 - np.sum((y_test - y_test_pred)**2) / np.sum((y_test - np.mean(y_test))**2)

        train_rmse = np.sqrt(np.mean((y_train - y_train_pred)**2))
        test_rmse = np.sqrt(np.mean((y_test - y_test_pred)**2))

        train_mae = np.mean(np.abs(y_train - y_train_pred))
        test_mae = np.mean(np.abs(y_test - y_test_pred))

        print(f"\n训练集性能:")
        print(f"  R2 = {train_r2:.4f}")
        print(f"  RMSE = {train_rmse:.2f} mm")
        print(f"  MAE = {train_mae:.2f} mm")

        print(f"\n测试集性能:")
        print(f"  R2 = {test_r2:.4f}")
        print(f"  RMSE = {test_rmse:.2f} mm")
        print(f"  MAE = {test_mae:.2f} mm")

        return {
            'name': 'Ridge Regression',
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'coefficients': coefficients
        }

    def simple_neural_network(self, X_train, y_train, X_test, y_test):
        """简单神经网络 (2-3层)"""
        print(f"\n{'='*60}")
        print("简单神经网络模型 (2隐藏层)")
        print("="*60)

        # 标准化
        X_train_norm, X_test_norm = self.normalize_features(X_train, X_test)

        # 网络结构: 输入层 -> 16 -> 8 -> 输出层
        n_features = X_train.shape[1]
        layers = [n_features, 16, 8, 1]

        print(f"\n网络结构: {layers}")
        print(f"总参数数: {sum([layers[i]*layers[i+1] + layers[i+1] for i in range(len(layers)-1)])}")

        # 创建和训练模型
        model = VIVNeuralNetwork(layers, learning_rate=0.01, epochs=1500, random_seed=42)

        print(f"\n开始训练...")
        model.fit(X_train_norm, y_train, X_test_norm, y_test, verbose=True)

        # 预测
        y_train_pred = model.predict(X_train_norm)
        y_test_pred = model.predict(X_test_norm)

        # 评估
        train_r2 = 1 - np.sum((y_train - y_train_pred)**2) / np.sum((y_train - np.mean(y_train))**2)
        test_r2 = 1 - np.sum((y_test - y_test_pred)**2) / np.sum((y_test - np.mean(y_test))**2)

        train_rmse = np.sqrt(np.mean((y_train - y_train_pred)**2))
        test_rmse = np.sqrt(np.mean((y_test - y_test_pred)**2))

        train_mae = np.mean(np.abs(y_train - y_train_pred))
        test_mae = np.mean(np.abs(y_test - y_test_pred))

        print(f"\n训练集性能:")
        print(f"  R2 = {train_r2:.4f}")
        print(f"  RMSE = {train_rmse:.2f} mm")
        print(f"  MAE = {train_mae:.2f} mm")

        print(f"\n测试集性能:")
        print(f"  R2 = {test_r2:.4f}")
        print(f"  RMSE = {test_rmse:.2f} mm")
        print(f"  MAE = {test_mae:.2f} mm")

        return {
            'name': 'Simple NN (2 layers)',
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'model': model,
            'layers': layers
        }

    def medium_neural_network(self, X_train, y_train, X_test, y_test):
        """中等神经网络 (4-5层)"""
        print(f"\n{'='*60}")
        print("中等神经网络模型 (4隐藏层)")
        print("="*60)

        # 标准化
        X_train_norm, X_test_norm = self.normalize_features(X_train, X_test)

        # 网络结构: 输入层 -> 32 -> 64 -> 32 -> 16 -> 输出层
        n_features = X_train.shape[1]
        layers = [n_features, 32, 64, 32, 16, 1]

        print(f"\n网络结构: {layers}")
        print(f"总参数数: {sum([layers[i]*layers[i+1] + layers[i+1] for i in range(len(layers)-1)])}")

        # 创建和训练模型 - 降低学习率避免梯度爆炸
        model = VIVNeuralNetwork(layers, learning_rate=0.001, epochs=2000, random_seed=42)

        print(f"\n开始训练...")
        model.fit(X_train_norm, y_train, X_test_norm, y_test, verbose=True)

        # 预测
        y_train_pred = model.predict(X_train_norm)
        y_test_pred = model.predict(X_test_norm)

        # 评估
        train_r2 = 1 - np.sum((y_train - y_train_pred)**2) / np.sum((y_train - np.mean(y_train))**2)
        test_r2 = 1 - np.sum((y_test - y_test_pred)**2) / np.sum((y_test - np.mean(y_test))**2)

        train_rmse = np.sqrt(np.mean((y_train - y_train_pred)**2))
        test_rmse = np.sqrt(np.mean((y_test - y_test_pred)**2))

        train_mae = np.mean(np.abs(y_train - y_train_pred))
        test_mae = np.mean(np.abs(y_test - y_test_pred))

        print(f"\n训练集性能:")
        print(f"  R2 = {train_r2:.4f}")
        print(f"  RMSE = {train_rmse:.2f} mm")
        print(f"  MAE = {train_mae:.2f} mm")

        print(f"\n测试集性能:")
        print(f"  R2 = {test_r2:.4f}")
        print(f"  RMSE = {test_rmse:.2f} mm")
        print(f"  MAE = {test_mae:.2f} mm")

        return {
            'name': 'Medium NN (4 layers)',
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'model': model,
            'layers': layers
        }

    def compare_models(self):
        """对比所有模型"""
        # 加载数据
        self.load_and_prepare_data()

        # 划分数据
        X_train, X_test, y_train, y_test = self.train_test_split(test_size=0.2, random_seed=42)

        # 训练和评估所有模型
        results = []

        # 1. 岭回归基线
        ridge_result = self.ridge_regression_baseline(X_train, y_train, X_test, y_test, alpha=0.1)
        results.append(ridge_result)

        # 2. 简单神经网络
        simple_nn_result = self.simple_neural_network(X_train, y_train, X_test, y_test)
        results.append(simple_nn_result)

        # 3. 中等神经网络
        medium_nn_result = self.medium_neural_network(X_train, y_train, X_test, y_test)
        results.append(medium_nn_result)

        # 生成对比报告
        self.generate_comparison_report(results)

        return results

    def generate_comparison_report(self, results):
        """生成对比报告"""
        print(f"\n{'='*60}")
        print("模型性能对比总结")
        print("="*60)

        print(f"\n{'模型':<25} {'测试R2':<12} {'测试RMSE':<12} {'测试MAE':<12}")
        print("-" * 60)

        for result in results:
            print(f"{result['name']:<25} {result['test_r2']:<12.4f} {result['test_rmse']:<12.2f} {result['test_mae']:<12.2f}")

        # 找出最佳模型
        best_model = max(results, key=lambda x: x['test_r2'])
        print(f"\n最佳模型: {best_model['name']}")
        print(f"  测试集 R2 = {best_model['test_r2']:.4f}")
        print(f"  测试集 RMSE = {best_model['test_rmse']:.2f} mm")

        # 保存报告
        report = f"""
桥梁VIV振幅预测 - 神经网络 vs 岭回归对比报告
{'='*60}
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
数据集: 196座桥梁

模型性能对比:
{'-'*60}
"""
        for result in results:
            report += f"\n{result['name']}:\n"
            report += f"  训练集 R2 = {result['train_r2']:.4f}, RMSE = {result['train_rmse']:.2f} mm\n"
            report += f"  测试集 R2 = {result['test_r2']:.4f}, RMSE = {result['test_rmse']:.2f} mm\n"

        report += f"\n{'='*60}\n"
        report += f"最佳模型: {best_model['name']}\n"
        report += f"测试集 R2 = {best_model['test_r2']:.4f}\n"
        report += f"{'='*60}\n"

        with open('viv_nn_comparison_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n报告已保存至: viv_nn_comparison_report.txt")


def main():
    print("="*60)
    print("桥梁VIV振幅预测 - 神经网络建模")
    print("="*60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 创建对比系统
    comparison = VIVModelComparison()

    # 运行对比实验
    results = comparison.compare_models()

    print(f"\n{'='*60}")
    print("实验完成!")
    print("="*60)


if __name__ == "__main__":
    main()
