"""
桥梁涡振风险评估神经网络模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BridgeVortexNN(nn.Module):
    """桥梁涡振风险评估神经网络"""
    
    def __init__(self, input_dim, hidden_dims=[64, 32, 16], output_dim=1, dropout_rate=0.2):
        """
        初始化神经网络
        
        Args:
            input_dim (int): 输入特征维度
            hidden_dims (list): 隐藏层维度列表
            output_dim (int): 输出维度（1为回归，3为分类）
            dropout_rate (float): Dropout比率
        """
        super(BridgeVortexNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        # 隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        return self.network(x)

class BridgeVortexMultiTaskNN(nn.Module):
    """多任务桥梁涡振风险评估神经网络"""
    
    def __init__(self, input_dim, shared_hidden_dims=[64, 32], 
                 amplitude_hidden_dims=[16], classification_hidden_dims=[16],
                 num_risk_classes=3, dropout_rate=0.2):
        """
        初始化多任务神经网络
        
        Args:
            input_dim (int): 输入特征维度
            shared_hidden_dims (list): 共享层维度
            amplitude_hidden_dims (list): 振幅预测分支隐藏层维度
            classification_hidden_dims (list): 分类分支隐藏层维度
            num_risk_classes (int): 风险等级类别数
            dropout_rate (float): Dropout比率
        """
        super(BridgeVortexMultiTaskNN, self).__init__()
        
        # 共享特征提取层
        shared_layers = []
        prev_dim = input_dim
        
        for hidden_dim in shared_hidden_dims:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.BatchNorm1d(hidden_dim))
            shared_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*shared_layers)
        
        # 振幅预测分支（回归任务）
        amplitude_layers = []
        amplitude_prev_dim = prev_dim
        
        for hidden_dim in amplitude_hidden_dims:
            amplitude_layers.append(nn.Linear(amplitude_prev_dim, hidden_dim))
            amplitude_layers.append(nn.ReLU())
            amplitude_layers.append(nn.Dropout(dropout_rate))
            amplitude_prev_dim = hidden_dim
        
        amplitude_layers.append(nn.Linear(amplitude_prev_dim, 1))
        self.amplitude_branch = nn.Sequential(*amplitude_layers)
        
        # 涡振发生预测分支（二分类任务）
        occurrence_layers = []
        occurrence_prev_dim = prev_dim
        
        for hidden_dim in classification_hidden_dims:
            occurrence_layers.append(nn.Linear(occurrence_prev_dim, hidden_dim))
            occurrence_layers.append(nn.ReLU())
            occurrence_layers.append(nn.Dropout(dropout_rate))
            occurrence_prev_dim = hidden_dim
        
        occurrence_layers.append(nn.Linear(occurrence_prev_dim, 2))
        self.occurrence_branch = nn.Sequential(*occurrence_layers)
        
        # 风险等级预测分支（多分类任务）
        risk_layers = []
        risk_prev_dim = prev_dim
        
        for hidden_dim in classification_hidden_dims:
            risk_layers.append(nn.Linear(risk_prev_dim, hidden_dim))
            risk_layers.append(nn.ReLU())
            risk_layers.append(nn.Dropout(dropout_rate))
            risk_prev_dim = hidden_dim
        
        risk_layers.append(nn.Linear(risk_prev_dim, num_risk_classes))
        self.risk_branch = nn.Sequential(*risk_layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        # 共享特征提取
        shared_features = self.shared_layers(x)
        
        # 各分支预测
        amplitude_pred = self.amplitude_branch(shared_features)
        occurrence_pred = self.occurrence_branch(shared_features)
        risk_pred = self.risk_branch(shared_features)
        
        return {
            'amplitude': amplitude_pred,
            'occurrence': occurrence_pred,
            'risk': risk_pred
        }

class RiskAssessmentModel:
    """风险评估模型包装器"""
    
    def __init__(self, model, device='cpu'):
        """
        初始化风险评估模型
        
        Args:
            model: 训练好的神经网络模型
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # 风险等级阈值
        self.risk_thresholds = {
            'low': 1.0,      # 振幅 < 1cm
            'medium': 10.0,  # 1cm <= 振幅 < 10cm
            'high': 40.0     # 振幅 >= 40cm
        }
    
    def predict_amplitude(self, features):
        """预测振幅"""
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device)
            if isinstance(self.model, BridgeVortexMultiTaskNN):
                predictions = self.model(features_tensor)
                amplitude_pred = predictions['amplitude']
            else:
                amplitude_pred = self.model(features_tensor)
            
            return amplitude_pred.cpu().numpy()
    
    def assess_risk(self, features):
        """综合风险评估"""
        amplitude_pred = self.predict_amplitude(features)
        
        risk_scores = []
        risk_levels = []
        
        for amp in amplitude_pred:
            amp_value = float(amp)
            
            if amp_value < self.risk_thresholds['low']:
                risk_level = '低风险'
                risk_score = 0.1 + 0.3 * (amp_value / self.risk_thresholds['low'])
            elif amp_value < self.risk_thresholds['medium']:
                risk_level = '中风险'
                risk_score = 0.4 + 0.4 * ((amp_value - self.risk_thresholds['low']) / 
                                         (self.risk_thresholds['medium'] - self.risk_thresholds['low']))
            else:
                risk_level = '高风险'
                risk_score = 0.8 + 0.2 * min(1.0, (amp_value - self.risk_thresholds['medium']) / 
                                            (self.risk_thresholds['high'] - self.risk_thresholds['medium']))
            
            risk_scores.append(risk_score)
            risk_levels.append(risk_level)
        
        return {
            'amplitude_prediction': amplitude_pred,
            'risk_scores': np.array(risk_scores),
            'risk_levels': risk_levels
        }
    
    def generate_report(self, features, bridge_info=None):
        """生成风险评估报告"""
        assessment = self.assess_risk(features)
        
        report = {
            'bridge_info': bridge_info,
            'predicted_amplitude': float(assessment['amplitude_prediction'][0]),
            'risk_score': float(assessment['risk_scores'][0]),
            'risk_level': assessment['risk_levels'][0],
            'recommendations': self._get_recommendations(assessment['risk_levels'][0])
        }
        
        return report
    
    def _get_recommendations(self, risk_level):
        """根据风险等级给出建议"""
        recommendations = {
            '低风险': [
                "桥梁涡振风险较低，建议定期监测",
                "可考虑在关键部位安装传感器进行长期监测",
                "建议每年进行一次结构健康检查"
            ],
            '中风险': [
                "桥梁存在一定涡振风险，建议加强监测",
                "考虑安装风速监测设备",
                "建议评估是否需要安装抑振措施",
                "增加检查频率至每半年一次"
            ],
            '高风险': [
                "桥梁涡振风险较高，需要立即采取措施",
                "强烈建议安装抑振装置（如调谐质量阻尼器）",
                "建立实时监测系统",
                "考虑限制通行或降低设计风速",
                "建议每季度进行详细检查"
            ]
        }
        
        return recommendations.get(risk_level, ["请咨询专业工程师"])

# 使用示例
if __name__ == "__main__":
    # 创建简单回归模型
    input_dim = 10  # 假设有10个输入特征
    model = BridgeVortexNN(input_dim=input_dim, hidden_dims=[64, 32, 16])
    
    # 创建多任务模型
    multi_task_model = BridgeVortexMultiTaskNN(input_dim=input_dim)
    
    # 测试前向传播
    test_input = torch.randn(5, input_dim)  # 5个样本
    
    print("简单模型输出:", model(test_input).shape)
    print("多任务模型输出:", {k: v.shape for k, v in multi_task_model(test_input).items()})
    
    print("模型创建成功！")