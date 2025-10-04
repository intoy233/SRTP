"""
桥梁涡振数据加载和预处理模块
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

class BridgeDataLoader:
    """桥梁涡振数据加载器"""
    
    def __init__(self, data_path):
        """
        初始化数据加载器
        
        Args:
            data_path (str): 数据文件路径
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.features = None
        self.targets = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        """加载原始数据"""
        try:
            if self.data_path.endswith('.csv'):
                self.raw_data = pd.read_csv(self.data_path, encoding='utf-8')
            elif self.data_path.endswith('.xlsx'):
                self.raw_data = pd.read_excel(self.data_path)
            else:
                raise ValueError("不支持的文件格式，请使用CSV或Excel文件")
            
            print(f"成功加载数据，共{len(self.raw_data)}条记录")
            return self.raw_data
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None
    
    def clean_data(self):
        """数据清洗"""
        if self.raw_data is None:
            print("请先加载数据")
            return None
        
        # 复制数据
        data = self.raw_data.copy()
        
        # 处理缺失值
        # 数值型字段用中位数填充
        numeric_columns = ['宽高比', '跨度_m', '长度_m', '自振频率_Hz', 
                          '一阶频率_Hz', '二阶频率_Hz', '涡振风速_m_s', 
                          '振幅_cm', '阻力比', '措施后振幅_cm']
        
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                data[col].fillna(data[col].median(), inplace=True)
        
        # 分类型字段用众数填充
        categorical_columns = ['结构形式', '自证措施', '涡振发生', '风险等级']
        for col in categorical_columns:
            if col in data.columns:
                data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else '未知', inplace=True)
        
        # 删除重复行
        data.drop_duplicates(inplace=True)
        
        self.processed_data = data
        print(f"数据清洗完成，剩余{len(data)}条记录")
        return data
    
    def feature_engineering(self):
        """特征工程"""
        if self.processed_data is None:
            print("请先进行数据清洗")
            return None
        
        data = self.processed_data.copy()
        
        # 定义输入特征
        feature_columns = [
            '宽高比', '跨度_m', '长度_m', '自振频率_Hz', 
            '一阶频率_Hz', '二阶频率_Hz', '涡振风速_m_s', '阻力比'
        ]
        
        # 处理分类特征
        if '结构形式' in data.columns:
            # 对结构形式进行独热编码
            structure_dummies = pd.get_dummies(data['结构形式'], prefix='结构')
            data = pd.concat([data, structure_dummies], axis=1)
            feature_columns.extend(structure_dummies.columns.tolist())
        
        if '自证措施' in data.columns:
            # 将自证措施转换为二进制特征
            data['有自证措施'] = (data['自证措施'] != '无').astype(int)
            feature_columns.append('有自证措施')
        
        # 创建新特征
        if '跨度_m' in data.columns and '长度_m' in data.columns:
            data['跨长比'] = data['跨度_m'] / data['长度_m']
            feature_columns.append('跨长比')
        
        if '一阶频率_Hz' in data.columns and '二阶频率_Hz' in data.columns:
            data['频率比'] = data['二阶频率_Hz'] / data['一阶频率_Hz']
            feature_columns.append('频率比')
        
        # 提取特征和目标变量
        self.features = data[feature_columns].fillna(0)
        
        # 目标变量：振幅预测（回归）和涡振发生预测（分类）
        if '振幅_cm' in data.columns:
            self.targets = {
                'amplitude': data['振幅_cm'],
                'vortex_occurrence': data['涡振发生'] if '涡振发生' in data.columns else None,
                'risk_level': data['风险等级'] if '风险等级' in data.columns else None
            }
        
        print(f"特征工程完成，共{len(feature_columns)}个特征")
        return self.features, self.targets
    
    def normalize_features(self):
        """特征标准化"""
        if self.features is None:
            print("请先进行特征工程")
            return None
        
        self.features_normalized = pd.DataFrame(
            self.scaler.fit_transform(self.features),
            columns=self.features.columns,
            index=self.features.index
        )
        
        print("特征标准化完成")
        return self.features_normalized
    
    def split_data(self, test_size=0.25, random_state=42):
        """数据集划分"""
        if self.features is None or self.targets is None:
            print("请先完成特征工程")
            return None
        
        # 使用标准化后的特征
        features = self.features_normalized if hasattr(self, 'features_normalized') else self.features
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            features, self.targets['amplitude'], 
            test_size=test_size, random_state=random_state
        )
        
        print(f"数据集划分完成：训练集{len(X_train)}条，测试集{len(X_test)}条")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
    
    def save_processed_data(self, output_path):
        """保存处理后的数据"""
        if self.processed_data is None:
            print("没有处理后的数据可保存")
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.processed_data.to_csv(output_path, index=False, encoding='utf-8')
        print(f"处理后的数据已保存到: {output_path}")

# 使用示例
if __name__ == "__main__":
    # 示例用法
    data_path = "../../data/templates/bridge_vortex_data_template.csv"
    
    loader = BridgeDataLoader(data_path)
    
    # 加载数据
    raw_data = loader.load_data()
    
    if raw_data is not None:
        # 数据清洗
        clean_data = loader.clean_data()
        
        # 特征工程
        features, targets = loader.feature_engineering()
        
        # 特征标准化
        normalized_features = loader.normalize_features()
        
        # 数据集划分
        data_splits = loader.split_data()
        
        print("数据预处理完成！")