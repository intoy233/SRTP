"""
改进的特征工程模块
基于实验结果和工程建议，开发高级特征工程方法
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class AdvancedBridgeFeatureEngineering:
    """高级桥梁VIV特征工程类"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.pca = None
        self.poly_features = None
        self.feature_names = []
        self.is_fitted = False

    def create_physics_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """创建基于物理原理的工程特征 - 增强版"""
        df = data.copy()

        # === 基础几何特征 ===
        if 'Span_m' in df.columns and 'Width_m' in df.columns:
            df['Aspect_Ratio'] = df['Span_m'] / df['Width_m']

        if 'Width_m' in df.columns and 'Height_m' in df.columns:
            df['Width_Height_Ratio'] = df['Width_m'] / df['Height_m']

        if 'Span_m' in df.columns and 'Height_m' in df.columns:
            df['Slenderness_Ratio'] = df['Span_m'] / df['Height_m']

        # === 高级几何特征 ===
        # 截面积比和周长比
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

        # 频率密度参数
        if all(col in df.columns for col in ['Natural_Freq_Hz', 'Span_m']):
            df['Freq_Length_Product'] = df['Natural_Freq_Hz'] * df['Span_m']

        # === 风速相关特征 ===
        if 'VIV_Wind_Speed_ms' in df.columns and 'Critical_Wind_Speed_ms' in df.columns:
            df['Wind_Speed_Ratio'] = df['VIV_Wind_Speed_ms'] / (df['Critical_Wind_Speed_ms'] + 1e-8)

        # === 核心无量纲参数 ===
        # 1. 约化风速 (Reduced Wind Speed) - 关键VIV参数
        if all(col in df.columns for col in ['VIV_Wind_Speed_ms', 'Natural_Freq_Hz', 'Width_m']):
            df['Reduced_Wind_Speed'] = df['VIV_Wind_Speed_ms'] / (df['Natural_Freq_Hz'] * df['Width_m'])

        if all(col in df.columns for col in ['Critical_Wind_Speed_ms', 'Natural_Freq_Hz', 'Width_m']):
            df['Critical_Reduced_Wind_Speed'] = df['Critical_Wind_Speed_ms'] / (df['Natural_Freq_Hz'] * df['Width_m'])

        # 2. Reynolds数 - 流体特性
        if 'VIV_Wind_Speed_ms' in df.columns and 'Width_m' in df.columns:
            rho = 1.225  # 空气密度 kg/m³
            nu = 1.5e-5  # 运动粘度 m²/s
            df['Reynolds_Number'] = (df['VIV_Wind_Speed_ms'] * df['Width_m']) / nu
            df['Log_Reynolds'] = np.log10(df['Reynolds_Number'] + 1)

        # 3. Strouhal数 - 涡脱频率特性
        if all(col in df.columns for col in ['Natural_Freq_Hz', 'VIV_Wind_Speed_ms', 'Width_m']):
            df['Strouhal_Number'] = (df['Natural_Freq_Hz'] * df['Width_m']) / (df['VIV_Wind_Speed_ms'] + 1e-8)

        # 4. 质量阻尼参数 - VIV响应关键参数
        if 'Damping_Ratio' in df.columns:
            # 假设典型桥梁质量密度
            typical_mass_ratio = 10.0  # 典型质量比
            df['Mass_Damping_Parameter'] = typical_mass_ratio * df['Damping_Ratio']
            df['Log_Damping'] = np.log(df['Damping_Ratio'] + 1e-8)
            df['Inv_Damping'] = 1 / (df['Damping_Ratio'] + 1e-8)

        # 5. Scruton数 - 关键稳定性参数
        if 'Damping_Ratio' in df.columns:
            df['Scruton_Number'] = 2 * typical_mass_ratio * df['Damping_Ratio']

        # === 高级无量纲参数 ===
        # 6. 风阻比 (Wind-to-Resistance Ratio)
        if all(col in df.columns for col in ['VIV_Wind_Speed_ms', 'Damping_Ratio']):
            df['Wind_Resistance_Ratio'] = df['VIV_Wind_Speed_ms'] / (df['Damping_Ratio'] + 1e-8)

        # 7. 结构刚度参数
        if all(col in df.columns for col in ['Natural_Freq_Hz', 'Span_m']):
            df['Stiffness_Parameter'] = (df['Natural_Freq_Hz'] * df['Span_m'])**2

        # 8. 气动弹性参数
        if all(col in df.columns for col in ['VIV_Wind_Speed_ms', 'Natural_Freq_Hz', 'Span_m']):
            df['Aeroelastic_Parameter'] = df['VIV_Wind_Speed_ms'] / (df['Natural_Freq_Hz'] * df['Span_m'])

        # 9. 振动敏感性参数
        if all(col in df.columns for col in ['Width_Height_Ratio', 'Damping_Ratio']):
            df['Vibration_Sensitivity'] = df['Width_Height_Ratio'] / (df['Damping_Ratio'] + 1e-8)

        # === 组合稳定性参数 ===
        # 10. 综合稳定性指标
        if all(col in df.columns for col in ['Damping_Ratio', 'Wind_Speed_Ratio']):
            df['Stability_Parameter'] = df['Damping_Ratio'] / (df['Wind_Speed_Ratio'] + 1e-8)

        # 11. 气动-结构耦合参数
        if all(col in df.columns for col in ['Reynolds_Number', 'Reduced_Wind_Speed']):
            df['Aero_Structural_Coupling'] = df['Reynolds_Number'] / (df['Reduced_Wind_Speed'] + 1e-8)

        # === 工程实用参数 ===
        # 12. VIV风险指数
        if all(col in df.columns for col in ['Reduced_Wind_Speed', 'Mass_Damping_Parameter']):
            df['VIV_Risk_Index'] = df['Reduced_Wind_Speed'] / (df['Mass_Damping_Parameter'] + 1e-8)

        # 13. 临界性参数
        if all(col in df.columns for col in ['VIV_Wind_Speed_ms', 'Critical_Wind_Speed_ms', 'Damping_Ratio']):
            df['Criticality_Parameter'] = (df['VIV_Wind_Speed_ms'] / df['Critical_Wind_Speed_ms']) / (df['Damping_Ratio'] + 1e-8)

        # === 几何-动力学组合特征 ===
        if all(col in df.columns for col in ['Width_Height_Ratio', 'Natural_Freq_Hz']):
            df['Geo_Dynamic_Factor'] = df['Width_Height_Ratio'] * df['Natural_Freq_Hz']

        if all(col in df.columns for col in ['Aspect_Ratio', 'Freq_Ratio']):
            df['Geo_Freq_Coupling'] = df['Aspect_Ratio'] * df['Freq_Ratio']

        # === 尺度效应参数 ===
        # 14. 相对尺寸参数
        if 'Span_m' in df.columns:
            df['Span_Scale'] = df['Span_m'] / 1000.0  # 标准化到km
            df['Log_Span'] = np.log10(df['Span_m'])

        if 'Width_m' in df.columns:
            df['Width_Scale'] = df['Width_m'] / 50.0  # 标准化到典型桥宽
            df['Log_Width'] = np.log10(df['Width_m'])

        # === 物理约束检查和修正 ===
        # 确保物理合理性
        for col in df.columns:
            if 'Ratio' in col or 'Parameter' in col:
                # 处理极端值
                if df[col].dtype in ['float64', 'float32']:
                    df[col] = np.clip(df[col], -1000, 1000)
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    df[col] = df[col].fillna(df[col].median())

        return df

    def create_interaction_features(self, data: pd.DataFrame, max_interactions: int = 20) -> pd.DataFrame:
        """创建交互特征"""
        df = data.copy()

        # 选择数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # 重要特征组合
        important_pairs = [
            ('Width_Height_Ratio', 'Natural_Freq_Hz'),
            ('Damping_Ratio', 'VIV_Wind_Speed_ms'),
            ('Span_m', 'Width_m'),
            ('Reynolds_Number', 'Strouhal_Number'),
            ('Reduced_Wind_Speed', 'Damping_Ratio')
        ]

        interaction_count = 0
        for col1, col2 in important_pairs:
            if col1 in numeric_cols and col2 in numeric_cols and interaction_count < max_interactions:
                # 乘积交互
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                interaction_count += 1

                if interaction_count < max_interactions:
                    # 比值交互
                    df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
                    interaction_count += 1

        return df

    def create_polynomial_features(self, data: pd.DataFrame, degree: int = 2,
                                 selected_features: list = None) -> pd.DataFrame:
        """创建多项式特征"""
        df = data.copy()

        if selected_features is None:
            # 选择重要的特征进行多项式展开
            selected_features = ['Width_Height_Ratio', 'Damping_Ratio', 'VIV_Wind_Speed_ms',
                               'Natural_Freq_Hz', 'Reduced_Wind_Speed']

        available_features = [f for f in selected_features if f in df.columns]

        if len(available_features) > 0:
            # 只对选定特征进行多项式展开
            subset_data = df[available_features]

            if not hasattr(self, 'poly_features') or self.poly_features is None:
                self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
                poly_data = self.poly_features.fit_transform(subset_data)
            else:
                poly_data = self.poly_features.transform(subset_data)

            # 创建多项式特征名称
            poly_feature_names = self.poly_features.get_feature_names_out(available_features)

            # 添加多项式特征到原数据
            for i, name in enumerate(poly_feature_names):
                if name not in available_features:  # 避免重复原始特征
                    df[f'poly_{name}'] = poly_data[:, i]

        return df

    def create_statistical_features(self, data: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
        """创建统计特征（基于相似桥梁）"""
        df = data.copy()

        # 按桥梁类型分组的统计特征（如果有类型信息）
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in ['VIV_Wind_Speed_ms', 'Natural_Freq_Hz', 'Damping_Ratio']:
            if col in numeric_cols:
                # 滚动统计
                df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size, min_periods=1).mean()
                df[f'{col}_rolling_std'] = df[col].rolling(window=window_size, min_periods=1).std().fillna(0)

                # 与全局均值的偏差
                global_mean = df[col].mean()
                df[f'{col}_deviation_from_mean'] = df[col] - global_mean

                # 百分位数特征
                df[f'{col}_percentile_rank'] = df[col].rank(pct=True)

        return df

    def select_features(self, X: np.ndarray, y: np.ndarray, method: str = 'regularized_selection',
                       k: int = 'auto') -> np.ndarray:
        """改进的特征选择 - 支持正则化方法"""
        if k == 'auto':
            k = min(50, X.shape[1] // 2)  # 自动选择特征数

        if method == 'f_regression':
            self.feature_selector = SelectKBest(score_func=f_regression, k=k)
            X_selected = self.feature_selector.fit_transform(X, y)

        elif method == 'mutual_info':
            self.feature_selector = SelectKBest(score_func=mutual_info_regression, k=k)
            X_selected = self.feature_selector.fit_transform(X, y)

        elif method == 'random_forest':
            # 使用随机森林进行特征重要性选择
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            feature_importance = rf.feature_importances_

            # 选择top k个特征
            top_indices = np.argsort(feature_importance)[-k:]
            self.selected_feature_indices = top_indices
            return X[:, top_indices]

        elif method == 'lasso_selection':
            # 使用Lasso进行特征选择
            from sklearn.linear_model import LassoCV
            from sklearn.feature_selection import SelectFromModel

            # 使用交叉验证选择最佳alpha
            lasso_cv = LassoCV(cv=5, random_state=42, max_iter=1000)
            lasso_cv.fit(X, y)

            # 使用选定的alpha创建特征选择器
            lasso = LassoCV(alphas=[lasso_cv.alpha_], cv=5, random_state=42, max_iter=1000)
            self.feature_selector = SelectFromModel(lasso, prefit=False)
            X_selected = self.feature_selector.fit_transform(X, y)

        elif method == 'elastic_net_selection':
            # 使用ElasticNet进行特征选择
            from sklearn.linear_model import ElasticNetCV
            from sklearn.feature_selection import SelectFromModel

            # 使用交叉验证选择最佳参数
            elastic_net_cv = ElasticNetCV(cv=5, random_state=42, max_iter=1000)
            elastic_net_cv.fit(X, y)

            # 创建特征选择器
            elastic_net = ElasticNetCV(
                alphas=[elastic_net_cv.alpha_],
                l1_ratio=[elastic_net_cv.l1_ratio_],
                cv=5, random_state=42, max_iter=1000
            )
            self.feature_selector = SelectFromModel(elastic_net, prefit=False)
            X_selected = self.feature_selector.fit_transform(X, y)

        elif method == 'regularized_selection':
            # 多种正则化方法组合选择
            selected_features_sets = []

            # 1. Lasso选择
            try:
                from sklearn.linear_model import LassoCV
                from sklearn.feature_selection import SelectFromModel

                lasso_cv = LassoCV(cv=3, random_state=42, max_iter=1000)
                lasso_cv.fit(X, y)
                lasso_selector = SelectFromModel(lasso_cv, prefit=True)
                lasso_features = lasso_selector.get_support(indices=True)
                selected_features_sets.append(set(lasso_features))
            except:
                pass

            # 2. Random Forest选择
            try:
                rf = RandomForestRegressor(n_estimators=50, random_state=42)
                rf.fit(X, y)
                rf_selector = SelectFromModel(rf, prefit=True)
                rf_features = rf_selector.get_support(indices=True)
                selected_features_sets.append(set(rf_features))
            except:
                pass

            # 3. 互信息选择
            try:
                mi_selector = SelectKBest(score_func=mutual_info_regression, k=min(30, X.shape[1]//3))
                mi_selector.fit(X, y)
                mi_features = mi_selector.get_support(indices=True)
                selected_features_sets.append(set(mi_features))
            except:
                pass

            # 特征交集或并集策略
            if len(selected_features_sets) > 1:
                # 取至少在两个方法中被选中的特征
                feature_votes = {}
                for feature_set in selected_features_sets:
                    for feature_idx in feature_set:
                        feature_votes[feature_idx] = feature_votes.get(feature_idx, 0) + 1

                # 选择投票数量>=2的特征，或者最高投票的前k个特征
                high_vote_features = [f for f, v in feature_votes.items() if v >= 2]

                if len(high_vote_features) == 0:
                    # 如果没有高投票特征，选择投票最多的前k个
                    sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
                    selected_indices = [f for f, v in sorted_features[:k]]
                elif len(high_vote_features) > k:
                    # 如果高投票特征过多，选择投票最多的前k个
                    high_vote_sorted = sorted([(f, feature_votes[f]) for f in high_vote_features],
                                            key=lambda x: x[1], reverse=True)
                    selected_indices = [f for f, v in high_vote_sorted[:k]]
                else:
                    selected_indices = high_vote_features
            else:
                # 只有一个方法有效，使用该方法的结果
                selected_indices = list(selected_features_sets[0]) if selected_features_sets else list(range(min(k, X.shape[1])))

            self.selected_feature_indices = np.array(selected_indices)
            X_selected = X[:, self.selected_feature_indices]

        elif method == 'stability_selection':
            # 稳定性选择 - 适合小样本
            from sklearn.utils import resample

            n_iterations = 50
            subsample_ratio = 0.8
            selection_threshold = 0.6  # 特征被选择的最小频率

            feature_selection_freq = np.zeros(X.shape[1])

            for i in range(n_iterations):
                # 子采样数据
                X_subsample, y_subsample = resample(X, y,
                                                  n_samples=int(len(X) * subsample_ratio),
                                                  random_state=i)

                # 使用Lasso进行特征选择
                from sklearn.linear_model import LassoCV
                from sklearn.feature_selection import SelectFromModel

                try:
                    lasso = LassoCV(cv=3, random_state=i, max_iter=1000)
                    lasso.fit(X_subsample, y_subsample)
                    selector = SelectFromModel(lasso, prefit=True)
                    selected = selector.get_support()
                    feature_selection_freq += selected.astype(int)
                except:
                    # 如果Lasso失败，使用随机森林
                    rf = RandomForestRegressor(n_estimators=50, random_state=i)
                    rf.fit(X_subsample, y_subsample)
                    selector = SelectFromModel(rf, prefit=True)
                    selected = selector.get_support()
                    feature_selection_freq += selected.astype(int)

            # 计算选择频率
            selection_probs = feature_selection_freq / n_iterations

            # 选择高频特征
            stable_features = np.where(selection_probs >= selection_threshold)[0]

            if len(stable_features) == 0:
                # 如果没有稳定特征，选择频率最高的前k个
                stable_features = np.argsort(selection_probs)[-k:]
            elif len(stable_features) > k:
                # 如果稳定特征过多，选择频率最高的前k个
                stable_features_sorted = np.argsort(selection_probs[stable_features])[-k:]
                stable_features = stable_features[stable_features_sorted]

            self.selected_feature_indices = stable_features
            X_selected = X[:, stable_features]

        else:
            # 默认使用互信息
            self.feature_selector = SelectKBest(score_func=mutual_info_regression, k=k)
            X_selected = self.feature_selector.fit_transform(X, y)

        return X_selected

    def apply_pca(self, X: np.ndarray, n_components: float = 0.95) -> np.ndarray:
        """应用主成分分析"""
        if self.pca is None:
            self.pca = PCA(n_components=n_components, random_state=42)
            X_pca = self.pca.fit_transform(X)
        else:
            X_pca = self.pca.transform(X)

        return X_pca

    def detect_outliers(self, data: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """异常值检测和处理"""
        df = data.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # 限制异常值而不是删除
                df[col] = np.clip(df[col], lower_bound, upper_bound)

            elif method == 'z_score':
                z_scores = np.abs(stats.zscore(df[col]))
                threshold = 3
                mean_val = df[col].mean()
                std_val = df[col].std()

                # 将异常值替换为边界值
                df.loc[z_scores > threshold, col] = mean_val + threshold * std_val * np.sign(df.loc[z_scores > threshold, col] - mean_val)

        return df

    def fit_transform(self, data: pd.DataFrame, target_col: str = 'Max_Amplitude_mm',
                     use_polynomial: bool = True, use_interactions: bool = True,
                     use_pca: bool = False, feature_selection_method: str = 'mutual_info',
                     k_features: int = 'auto') -> tuple:
        """完整的特征工程管道"""

        # 1. 异常值检测和处理
        df = self.detect_outliers(data)

        # 2. 创建物理特征
        df = self.create_physics_features(df)

        # 3. 创建交互特征
        if use_interactions:
            df = self.create_interaction_features(df)

        # 4. 创建多项式特征
        if use_polynomial:
            df = self.create_polynomial_features(df)

        # 5. 创建统计特征
        df = self.create_statistical_features(df)

        # 6. 准备特征和目标
        if target_col in df.columns:
            y = df[target_col].values
            df = df.drop(columns=[target_col])
        else:
            y = None

        # 只保留数值特征
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        X = df[numeric_cols].values

        # 7. 标准化特征
        X_scaled = self.scaler.fit_transform(X)

        # 8. 特征选择
        if feature_selection_method and y is not None:
            X_selected = self.select_features(X_scaled, y, feature_selection_method, k_features)
        else:
            X_selected = X_scaled

        # 9. PCA降维
        if use_pca:
            X_final = self.apply_pca(X_selected)
        else:
            X_final = X_selected

        # 保存特征名称
        self.feature_names = numeric_cols
        self.is_fitted = True

        print(f"特征工程完成:")
        print(f"  原始特征数: {data.shape[1]}")
        print(f"  工程特征数: {len(numeric_cols)}")
        print(f"  最终特征数: {X_final.shape[1]}")

        if y is not None:
            return X_final, y
        else:
            return X_final

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """对新数据应用相同的特征工程"""
        if not self.is_fitted:
            raise ValueError("特征工程器尚未拟合")

        # 应用相同的特征工程步骤
        df = self.detect_outliers(data)
        df = self.create_physics_features(df)
        df = self.create_interaction_features(df)

        if self.poly_features is not None:
            df = self.create_polynomial_features(df)

        df = self.create_statistical_features(df)

        # 只保留训练时的特征
        available_features = [f for f in self.feature_names if f in df.columns]
        missing_features = [f for f in self.feature_names if f not in df.columns]

        if missing_features:
            print(f"警告: 缺少特征 {missing_features}")

        X = df[available_features].values

        # 标准化
        X_scaled = self.scaler.transform(X)

        # 特征选择
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_scaled)
        elif hasattr(self, 'selected_feature_indices'):
            X_selected = X_scaled[:, self.selected_feature_indices]
        else:
            X_selected = X_scaled

        # PCA
        if self.pca is not None:
            X_final = self.pca.transform(X_selected)
        else:
            X_final = X_selected

        return X_final

    def get_feature_importance(self, X: np.ndarray, y: np.ndarray) -> dict:
        """获取特征重要性"""
        if not self.is_fitted:
            raise ValueError("特征工程器尚未拟合")

        # 使用随机森林获取特征重要性
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)

        importance_dict = {}
        if self.feature_selector is not None:
            selected_features = self.feature_selector.get_support(indices=True)
            for i, importance in enumerate(rf.feature_importances_):
                feature_idx = selected_features[i]
                if feature_idx < len(self.feature_names):
                    importance_dict[self.feature_names[feature_idx]] = importance
        else:
            for i, importance in enumerate(rf.feature_importances_):
                if i < len(self.feature_names):
                    importance_dict[self.feature_names[i]] = importance

        # 按重要性排序
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_importance)


def test_advanced_feature_engineering():
    """测试高级特征工程"""
    print("=== 高级特征工程测试 ===")

    # 加载数据
    try:
        data = pd.read_csv("../bridge_dataset_fixed.csv")
        print(f"加载数据: {data.shape}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        try:
            data = pd.read_csv("../../bridge_dataset_fixed.csv")
            print(f"加载数据: {data.shape}")
        except Exception as e2:
            print(f"数据加载失败: {e2}")
            return

    # 创建特征工程器
    feature_engineer = AdvancedBridgeFeatureEngineering()

    # 应用特征工程
    X, y = feature_engineer.fit_transform(
        data,
        target_col='Max_Amplitude_mm',
        use_polynomial=True,
        use_interactions=True,
        use_pca=False,
        feature_selection_method='mutual_info',
        k_features=30
    )

    print(f"最终特征矩阵形状: {X.shape}")
    print(f"目标变量范围: [{y.min():.2f}, {y.max():.2f}]")

    # 获取特征重要性
    importance = feature_engineer.get_feature_importance(X, y)
    print("\n前10个重要特征:")
    for i, (feature, score) in enumerate(list(importance.items())[:10]):
        print(f"  {i+1}. {feature}: {score:.4f}")

    return X, y, feature_engineer


class ConservativeRiskAssessment:
    """保守的风险评估器 - 结合物理模型和机器学习"""

    def __init__(self):
        self.risk_thresholds = {
            'low': 20,      # 低风险: < 20mm
            'medium': 40,   # 中风险: 20-40mm
            'high': float('inf')  # 高风险: > 40mm
        }
        self.confidence_factors = {
            'data_quality': 1.0,
            'model_agreement': 1.0,
            'physics_consistency': 1.0
        }

    def physics_based_risk_estimate(self, data: pd.DataFrame) -> pd.DataFrame:
        """基于物理公式的保守风险估计"""
        df = data.copy()

        # 基于经验公式的振幅估计
        if all(col in df.columns for col in ['Reduced_Wind_Speed', 'Mass_Damping_Parameter']):
            # 使用Scruton数和约化风速的经验关系
            vr = df['Reduced_Wind_Speed']
            scruton = df.get('Scruton_Number', df.get('Mass_Damping_Parameter', 10))

            # 保守的振幅估计公式 (基于Ehsan Gad等人的研究)
            amplitude_estimate = np.zeros_like(vr)

            # VIV锁定区间 (约化风速4-8)
            lock_in_mask = (vr >= 4) & (vr <= 8)
            if lock_in_mask.any():
                # 在锁定区间内，振幅与Scruton数反比
                amplitude_estimate[lock_in_mask] = (200 * vr[lock_in_mask]) / (scruton[lock_in_mask] + 1)

            # 锁定区间外，振幅较小
            non_lock_mask = ~lock_in_mask
            amplitude_estimate[non_lock_mask] = (50 * vr[non_lock_mask]) / (scruton[non_lock_mask] + 5)

            df['Physics_Amplitude_Estimate'] = amplitude_estimate

        # 基于几何参数的风险因子
        if 'Width_Height_Ratio' in df.columns:
            # 宽高比越大，涡振风险越高
            df['Geometry_Risk_Factor'] = np.clip(df['Width_Height_Ratio'] / 5.0, 0.5, 2.0)

        # 基于Reynolds数的修正
        if 'Reynolds_Number' in df.columns:
            # 超临界Reynolds数区域风险更高
            re = df['Reynolds_Number']
            df['Reynolds_Risk_Factor'] = np.where(
                re > 3e5,  # 超临界区
                1.5,       # 风险增加50%
                1.0        # 正常风险
            )

        return df

    def assess_data_quality(self, data: pd.DataFrame) -> float:
        """评估数据质量"""
        quality_score = 1.0

        # 检查缺失值
        missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        quality_score *= (1 - missing_ratio)

        # 检查数据范围合理性
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if data[col].std() == 0:  # 常数列
                quality_score *= 0.8
            if (data[col] < 0).any() and col in ['Width_m', 'Height_m', 'Span_m']:  # 负值几何参数
                quality_score *= 0.5

        return max(quality_score, 0.3)  # 最低质量分数

    def assess_model_agreement(self, predictions: dict) -> float:
        """评估多模型一致性"""
        if len(predictions) < 2:
            return 0.7  # 单模型默认置信度

        pred_values = list(predictions.values())
        pred_array = np.array(pred_values)

        # 计算变异系数
        mean_pred = np.mean(pred_array, axis=0)
        std_pred = np.std(pred_array, axis=0)
        cv = std_pred / (mean_pred + 1e-8)

        # 一致性得分 (变异系数越小，一致性越高)
        agreement_score = np.exp(-np.mean(cv))
        return np.clip(agreement_score, 0.3, 1.0)

    def assess_physics_consistency(self, data: pd.DataFrame, predictions: np.ndarray) -> float:
        """评估预测与物理规律的一致性"""
        consistency_score = 1.0

        # 检查预测值的物理合理性
        if np.any(predictions < 0):  # 负振幅不合理
            consistency_score *= 0.5

        if np.any(predictions > 500):  # 过大振幅不合理 (>50cm)
            consistency_score *= 0.3

        # 检查与已知物理规律的一致性
        if 'Damping_Ratio' in data.columns:
            # 高阻尼应对应低振幅
            high_damping = data['Damping_Ratio'] > 0.05
            high_amplitude = predictions > 50
            if np.any(high_damping & high_amplitude):
                consistency_score *= 0.7

        return max(consistency_score, 0.3)

    def conservative_risk_classification(self, amplitude_predictions: np.ndarray,
                                       confidence_scores: np.ndarray) -> tuple:
        """保守的风险分类"""
        risk_levels = np.zeros(len(amplitude_predictions), dtype=int)
        adjusted_predictions = amplitude_predictions.copy()

        # 根据置信度调整预测值 (低置信度时采用更保守估计)
        for i, (pred, conf) in enumerate(zip(amplitude_predictions, confidence_scores)):
            if conf < 0.5:
                # 低置信度时，增加安全系数
                safety_factor = 2.0 / (conf + 0.5)
                adjusted_predictions[i] = pred * safety_factor
            elif conf < 0.7:
                # 中等置信度时，适度增加安全系数
                safety_factor = 1.5 / (conf + 0.3)
                adjusted_predictions[i] = pred * safety_factor

        # 分类
        for i, adj_pred in enumerate(adjusted_predictions):
            if adj_pred < self.risk_thresholds['low']:
                risk_levels[i] = 0  # 低风险
            elif adj_pred < self.risk_thresholds['medium']:
                risk_levels[i] = 1  # 中风险
            else:
                risk_levels[i] = 2  # 高风险

        return risk_levels, adjusted_predictions

    def multi_model_ensemble_predict(self, models: dict, X: np.ndarray,
                                   method: str = 'weighted_average') -> tuple:
        """多模型集成预测"""
        predictions = {}

        # 获取各模型预测
        for name, model in models.items():
            try:
                pred = model.predict(X)
                predictions[name] = pred
            except Exception as e:
                print(f"模型 {name} 预测失败: {e}")
                continue

        if not predictions:
            raise ValueError("所有模型预测失败")

        # 模型权重 (基于历史性能)
        model_weights = {
            'ridge': 0.3,      # 稳定性高
            'random_forest': 0.25,  # 泛化能力好
            'xgboost': 0.2,    # 精度高但可能过拟合
            'linear': 0.15,    # 简单稳定
            'neural_network': 0.1  # 复杂度高，小数据集权重低
        }

        if method == 'weighted_average':
            # 加权平均
            weighted_pred = np.zeros(len(X))
            total_weight = 0

            for name, pred in predictions.items():
                weight = model_weights.get(name, 0.1)
                weighted_pred += weight * pred
                total_weight += weight

            ensemble_pred = weighted_pred / total_weight

        elif method == 'conservative_voting':
            # 保守投票：选择更高的预测值
            pred_array = np.array(list(predictions.values()))
            ensemble_pred = np.percentile(pred_array, 75, axis=0)  # 75分位数

        elif method == 'median':
            # 中位数集成
            pred_array = np.array(list(predictions.values()))
            ensemble_pred = np.median(pred_array, axis=0)

        else:
            # 简单平均
            pred_array = np.array(list(predictions.values()))
            ensemble_pred = np.mean(pred_array, axis=0)

        return ensemble_pred, predictions

    def comprehensive_risk_assessment(self, data: pd.DataFrame, models: dict,
                                    X: np.ndarray) -> dict:
        """综合风险评估"""
        results = {}

        # 1. 物理模型估计
        data_with_physics = self.physics_based_risk_estimate(data)

        # 2. 多模型机器学习预测
        ml_prediction, individual_predictions = self.multi_model_ensemble_predict(
            models, X, method='conservative_voting'
        )

        # 3. 评估各种置信度因子
        data_quality = self.assess_data_quality(data)
        model_agreement = self.assess_model_agreement(individual_predictions)

        # 合并物理和ML预测
        if 'Physics_Amplitude_Estimate' in data_with_physics.columns:
            physics_pred = data_with_physics['Physics_Amplitude_Estimate'].values
            # 物理模型和ML模型的加权组合
            combined_prediction = 0.4 * physics_pred + 0.6 * ml_prediction
        else:
            combined_prediction = ml_prediction

        physics_consistency = self.assess_physics_consistency(data, combined_prediction)

        # 4. 综合置信度
        overall_confidence = (data_quality * model_agreement * physics_consistency) ** 0.5
        confidence_array = np.full(len(combined_prediction), overall_confidence)

        # 5. 保守风险分类
        risk_levels, conservative_predictions = self.conservative_risk_classification(
            combined_prediction, confidence_array
        )

        # 6. 整理结果
        results = {
            'original_ml_prediction': ml_prediction,
            'physics_prediction': data_with_physics.get('Physics_Amplitude_Estimate', np.zeros_like(ml_prediction)),
            'combined_prediction': combined_prediction,
            'conservative_prediction': conservative_predictions,
            'risk_levels': risk_levels,
            'confidence_scores': confidence_array,
            'individual_model_predictions': individual_predictions,
            'quality_factors': {
                'data_quality': data_quality,
                'model_agreement': model_agreement,
                'physics_consistency': physics_consistency,
                'overall_confidence': overall_confidence
            }
        }

        return results


class StableModelSelector:
    """稳定的模型选择器 - 专为小数据集优化"""

    def __init__(self):
        self.recommended_models = {}
        self.model_configs = {}
        self.evaluation_results = {}

    def get_conservative_model_configs(self):
        """获取保守的模型配置"""
        configs = {
            'ridge': {
                'model_class': 'Ridge',
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0],  # 较强的正则化
                    'fit_intercept': [True],
                    'solver': ['auto']
                },
                'stability_score': 0.9,  # 最稳定
                'small_data_suitability': 0.95,
                'description': 'Ridge回归 - 最稳定，适合小数据集'
            },
            'conservative_random_forest': {
                'model_class': 'RandomForestRegressor',
                'params': {
                    'n_estimators': [20, 50],  # 较少的树，减少过拟合
                    'max_depth': [3, 5, 7],    # 限制深度
                    'min_samples_split': [10, 20],  # 增加最小分割样本
                    'min_samples_leaf': [5, 10],    # 增加叶子节点最小样本
                    'max_features': ['sqrt', 0.5],  # 限制特征数
                    'bootstrap': [True],
                    'random_state': [42]
                },
                'stability_score': 0.8,
                'small_data_suitability': 0.7,
                'description': '保守随机森林 - 参数保守，避免过拟合'
            },
            'elastic_net': {
                'model_class': 'ElasticNet',
                'params': {
                    'alpha': [0.1, 1.0, 10.0],
                    'l1_ratio': [0.1, 0.5, 0.9],  # L1和L2正则化平衡
                    'fit_intercept': [True],
                    'max_iter': [1000]
                },
                'stability_score': 0.85,
                'small_data_suitability': 0.9,
                'description': 'ElasticNet - 结合L1和L2正则化'
            },
            'linear_regression': {
                'model_class': 'LinearRegression',
                'params': {
                    'fit_intercept': [True]
                },
                'stability_score': 0.7,
                'small_data_suitability': 0.8,
                'description': '线性回归 - 最简单，解释性强'
            },
            'conservative_xgboost': {
                'model_class': 'XGBRegressor',
                'params': {
                    'n_estimators': [20, 50],  # 较少的估计器
                    'max_depth': [3, 4],       # 限制深度
                    'learning_rate': [0.01, 0.05, 0.1],  # 较低学习率
                    'subsample': [0.8, 0.9],   # 子采样避免过拟合
                    'colsample_bytree': [0.8, 0.9],
                    'reg_alpha': [1, 10],      # L1正则化
                    'reg_lambda': [1, 10],     # L2正则化
                    'random_state': [42]
                },
                'stability_score': 0.6,
                'small_data_suitability': 0.5,
                'description': '保守XGBoost - 强正则化参数'
            }
        }
        return configs

    def evaluate_model_stability(self, model, X, y, cv_folds=3, n_repeats=5):
        """评估模型稳定性"""
        from sklearn.model_selection import RepeatedKFold, cross_val_score
        from sklearn.metrics import mean_squared_error

        # 重复交叉验证
        rkf = RepeatedKFold(n_splits=cv_folds, n_repeats=n_repeats, random_state=42)

        try:
            scores = cross_val_score(model, X, y, cv=rkf, scoring='neg_mean_squared_error')
            rmse_scores = np.sqrt(-scores)

            stability_metrics = {
                'mean_rmse': np.mean(rmse_scores),
                'std_rmse': np.std(rmse_scores),
                'cv_rmse': np.std(rmse_scores) / np.mean(rmse_scores),  # 变异系数
                'stability_score': 1 / (1 + np.std(rmse_scores) / np.mean(rmse_scores))  # 稳定性分数
            }
        except Exception as e:
            print(f"模型评估失败: {e}")
            stability_metrics = {
                'mean_rmse': float('inf'),
                'std_rmse': float('inf'),
                'cv_rmse': float('inf'),
                'stability_score': 0.0
            }

        return stability_metrics

    def recommend_models_for_small_data(self, X, y, max_models=3):
        """为小数据集推荐模型"""
        from sklearn.linear_model import Ridge, ElasticNet, LinearRegression
        from sklearn.ensemble import RandomForestRegressor

        configs = self.get_conservative_model_configs()
        model_evaluations = []

        print(f"为小数据集 (n={len(X)}, p={X.shape[1]}) 评估模型...")

        # 评估每个推荐模型
        for model_name, config in configs.items():
            print(f"评估 {model_name}...")

            try:
                # 创建模型实例
                if config['model_class'] == 'Ridge':
                    # 对Ridge使用最保守参数
                    alpha = 10.0 if len(X) < 50 else 1.0
                    model = Ridge(alpha=alpha, fit_intercept=True)

                elif config['model_class'] == 'RandomForestRegressor':
                    # 保守的随机森林参数
                    n_est = min(20, len(X) // 3)  # 基于样本数调整
                    model = RandomForestRegressor(
                        n_estimators=max(10, n_est),
                        max_depth=min(5, len(X) // 10 + 2),
                        min_samples_split=max(10, len(X) // 8),
                        min_samples_leaf=max(5, len(X) // 15),
                        max_features='sqrt',
                        bootstrap=True,
                        random_state=42
                    )

                elif config['model_class'] == 'ElasticNet':
                    model = ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True, max_iter=1000)

                elif config['model_class'] == 'LinearRegression':
                    model = LinearRegression(fit_intercept=True)

                elif config['model_class'] == 'XGBRegressor':
                    try:
                        from xgboost import XGBRegressor
                        model = XGBRegressor(
                            n_estimators=min(20, len(X) // 2),
                            max_depth=3,
                            learning_rate=0.05,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            reg_alpha=10,
                            reg_lambda=10,
                            random_state=42
                        )
                    except ImportError:
                        print(f"XGBoost不可用，跳过 {model_name}")
                        continue
                else:
                    continue

                # 评估稳定性
                stability_metrics = self.evaluate_model_stability(model, X, y)

                # 计算综合评分
                # 小数据集权重: 稳定性(50%) + 小数据适用性(30%) + 性能(20%)
                performance_score = 1 / (1 + stability_metrics['mean_rmse'] / np.std(y))
                comprehensive_score = (
                    0.5 * stability_metrics['stability_score'] +
                    0.3 * config['small_data_suitability'] +
                    0.2 * performance_score
                )

                evaluation = {
                    'model_name': model_name,
                    'model': model,
                    'config': config,
                    'stability_metrics': stability_metrics,
                    'comprehensive_score': comprehensive_score,
                    'performance_score': performance_score
                }

                model_evaluations.append(evaluation)

            except Exception as e:
                print(f"模型 {model_name} 评估失败: {e}")
                continue

        # 按综合评分排序
        model_evaluations.sort(key=lambda x: x['comprehensive_score'], reverse=True)

        # 选择前max_models个模型
        recommended = model_evaluations[:max_models]

        # 保存评估结果
        self.evaluation_results = {
            'all_evaluations': model_evaluations,
            'recommended_models': recommended,
            'data_characteristics': {
                'n_samples': len(X),
                'n_features': X.shape[1],
                'target_std': np.std(y),
                'target_range': np.max(y) - np.min(y)
            }
        }

        return recommended

    def get_model_ensemble(self, recommended_models, X, y):
        """创建推荐模型的集成"""
        ensemble_models = {}

        for eval_result in recommended_models:
            model_name = eval_result['model_name']
            model = eval_result['model']

            try:
                # 训练模型
                model.fit(X, y)
                ensemble_models[model_name] = model
            except Exception as e:
                print(f"模型 {model_name} 训练失败: {e}")

        return ensemble_models

    def print_recommendations(self, recommended_models):
        """打印推荐结果"""
        print("\n=== 模型推荐结果 ===")

        for i, eval_result in enumerate(recommended_models, 1):
            model_name = eval_result['model_name']
            config = eval_result['config']
            stability = eval_result['stability_metrics']
            score = eval_result['comprehensive_score']

            print(f"\n{i}. {model_name}")
            print(f"   描述: {config['description']}")
            print(f"   综合评分: {score:.3f}")
            print(f"   稳定性分数: {stability['stability_score']:.3f}")
            print(f"   RMSE变异系数: {stability['cv_rmse']:.3f}")
            print(f"   小数据适用性: {config['small_data_suitability']:.2f}")

        print(f"\n推荐使用前{min(2, len(recommended_models))}个模型进行集成预测")


def test_conservative_risk_assessment():
    """测试保守风险评估"""
    print("=== 保守风险评估测试 ===")

    # 模拟数据和模型
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    import pandas as pd

    # 创建模拟数据
    np.random.seed(42)
    n_samples = 20
    data = pd.DataFrame({
        'Reduced_Wind_Speed': np.random.uniform(3, 10, n_samples),
        'Mass_Damping_Parameter': np.random.uniform(5, 20, n_samples),
        'Width_Height_Ratio': np.random.uniform(3, 8, n_samples),
        'Reynolds_Number': np.random.uniform(1e5, 1e6, n_samples),
        'Damping_Ratio': np.random.uniform(0.01, 0.1, n_samples)
    })

    X = data.values

    # 创建模拟模型
    models = {
        'ridge': Ridge().fit(X, np.random.uniform(10, 60, n_samples)),
        'random_forest': RandomForestRegressor(n_estimators=10, random_state=42).fit(X, np.random.uniform(15, 50, n_samples))
    }

    # 创建风险评估器
    risk_assessor = ConservativeRiskAssessment()

    # 进行综合风险评估
    results = risk_assessor.comprehensive_risk_assessment(data, models, X)

    print(f"ML预测范围: [{results['original_ml_prediction'].min():.1f}, {results['original_ml_prediction'].max():.1f}]")
    print(f"保守预测范围: [{results['conservative_prediction'].min():.1f}, {results['conservative_prediction'].max():.1f}]")
    print(f"风险等级分布: {np.bincount(results['risk_levels'])}")
    print(f"整体置信度: {results['quality_factors']['overall_confidence']:.3f}")

    return results


def test_stable_model_selector():
    """测试稳定模型选择器"""
    print("=== 稳定模型选择器测试 ===")

    # 创建模拟小数据集
    np.random.seed(42)
    n_samples = 30
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] * 2 + X[:, 1] * -1 + X[:, 2] * 0.5 +
         np.random.randn(n_samples) * 0.5 + 30)

    # 创建模型选择器
    selector = StableModelSelector()

    # 获取推荐模型
    recommended = selector.recommend_models_for_small_data(X, y, max_models=3)

    # 打印推荐结果
    selector.print_recommendations(recommended)

    # 创建模型集成
    ensemble_models = selector.get_model_ensemble(recommended, X, y)

    print(f"\n创建了 {len(ensemble_models)} 个模型的集成")

    return recommended, ensemble_models


if __name__ == "__main__":
    print("=== 高级特征工程与保守风险评估系统测试 ===\n")

    # 1. 测试高级特征工程
    test_advanced_feature_engineering()
    print("\n" + "="*50 + "\n")

    # 2. 测试保守风险评估
    test_conservative_risk_assessment()
    print("\n" + "="*50 + "\n")

    # 3. 测试稳定模型选择器
    test_stable_model_selector()