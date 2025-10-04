#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Engineering Module for Bridge VIV Risk Assessment
Provides advanced feature engineering capabilities with domain knowledge
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, mutual_info_regression, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures as SklearnPolynomialFeatures
import logging

logger = logging.getLogger(__name__)

class BridgeFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Comprehensive feature engineering for bridge VIV data
    Combines domain knowledge with statistical feature generation
    """

    def __init__(self,
                 include_domain_features: bool = True,
                 include_polynomial: bool = False,
                 polynomial_degree: int = 2,
                 include_interactions: bool = False,
                 include_ratios: bool = True,
                 include_dimensionless: bool = True,
                 include_statistical: bool = False,
                 max_features: Optional[int] = None):
        """
        Initialize feature engineer

        Args:
            include_domain_features: Include domain-specific features
            include_polynomial: Include polynomial features
            polynomial_degree: Degree for polynomial features
            include_interactions: Include interaction features
            include_ratios: Include ratio features
            include_dimensionless: Include dimensionless parameters
            include_statistical: Include statistical features
            max_features: Maximum number of features to keep
        """
        self.include_domain_features = include_domain_features
        self.include_polynomial = include_polynomial
        self.polynomial_degree = polynomial_degree
        self.include_interactions = include_interactions
        self.include_ratios = include_ratios
        self.include_dimensionless = include_dimensionless
        self.include_statistical = include_statistical
        self.max_features = max_features

        self.feature_names_ = []
        self.poly_transformer = None
        self.feature_selector = None
        self.fitted_ = False

    def _create_domain_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific features based on bridge engineering knowledge"""
        X_domain = X.copy()

        # Geometric ratios
        if 'Width_m' in X.columns and 'Height_m' in X.columns:
            X_domain['Width_Height_Ratio'] = X['Width_m'] / X['Height_m']

        if 'Span_m' in X.columns and 'Height_m' in X.columns:
            X_domain['Slenderness_Ratio'] = X['Span_m'] / X['Height_m']

        if 'Span_m' in X.columns and 'Width_m' in X.columns:
            X_domain['Span_Width_Ratio'] = X['Span_m'] / X['Width_m']

        # Dynamic characteristics
        if 'First_Freq_Hz' in X.columns and 'Natural_Freq_Hz' in X.columns:
            X_domain['Frequency_Ratio'] = X['First_Freq_Hz'] / X['Natural_Freq_Hz']

        if 'Second_Freq_Hz' in X.columns and 'First_Freq_Hz' in X.columns:
            X_domain['Freq_Second_First_Ratio'] = X['Second_Freq_Hz'] / X['First_Freq_Hz']

        # Aerodynamic parameters
        if 'VIV_Wind_Speed_ms' in X.columns and 'Critical_Wind_Speed_ms' in X.columns:
            X_domain['Wind_Speed_Ratio'] = X['VIV_Wind_Speed_ms'] / X['Critical_Wind_Speed_ms']

        if 'Drag_Coefficient' in X.columns and 'Lift_Coefficient' in X.columns:
            X_domain['Drag_Lift_Ratio'] = X['Drag_Coefficient'] / (X['Lift_Coefficient'] + 1e-8)

        # Vibration characteristics
        if 'Max_Amplitude_mm' in X.columns and 'Height_m' in X.columns:
            X_domain['Amplitude_Height_Ratio'] = X['Max_Amplitude_mm'] / (X['Height_m'] * 1000)

        if 'Amplitude_RMS_mm' in X.columns and 'Max_Amplitude_mm' in X.columns:
            X_domain['RMS_Max_Amplitude_Ratio'] = X['Amplitude_RMS_mm'] / (X['Max_Amplitude_mm'] + 1e-8)

        return X_domain

    def _create_dimensionless_parameters(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create dimensionless parameters relevant to VIV"""
        X_dim = X.copy()

        # Strouhal number (St = f*D/U)
        if all(col in X.columns for col in ['Natural_Freq_Hz', 'Width_m', 'VIV_Wind_Speed_ms']):
            X_dim['Strouhal_Number'] = (X['Natural_Freq_Hz'] * X['Width_m']) / (X['VIV_Wind_Speed_ms'] + 1e-8)

        # Reynolds number approximation (Re = U*D/ν, assuming ν ≈ 1.5e-5 for air)
        if 'VIV_Wind_Speed_ms' in X.columns and 'Width_m' in X.columns:
            nu = 1.5e-5  # Kinematic viscosity of air
            X_dim['Reynolds_Number'] = (X['VIV_Wind_Speed_ms'] * X['Width_m']) / nu

        # Reduced velocity (Ur = U/(f*D))
        if all(col in X.columns for col in ['VIV_Wind_Speed_ms', 'Natural_Freq_Hz', 'Width_m']):
            X_dim['Reduced_Velocity'] = X['VIV_Wind_Speed_ms'] / (X['Natural_Freq_Hz'] * X['Width_m'] + 1e-8)

        # Mass ratio (approximation based on structural characteristics)
        if all(col in X.columns for col in ['Damping_Ratio', 'Natural_Freq_Hz']):
            # Simplified mass ratio indicator
            X_dim['Mass_Ratio_Indicator'] = X['Damping_Ratio'] / (X['Natural_Freq_Hz'] + 1e-8)

        # Aspect ratio
        if 'Total_Length_m' in X.columns and 'Width_m' in X.columns:
            X_dim['Aspect_Ratio'] = X['Total_Length_m'] / X['Width_m']

        # Blockage ratio approximation (simplified)
        if 'Width_m' in X.columns and 'Height_m' in X.columns:
            # Assuming a reference dimension
            reference_width = 50.0  # Reference width for normalization
            X_dim['Blockage_Ratio'] = (X['Width_m'] * X['Height_m']) / (reference_width ** 2)

        return X_dim

    def _create_ratio_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create ratio features between all numeric columns"""
        X_ratios = X.copy()

        if not self.include_ratios:
            return X_ratios

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        # Limit to avoid feature explosion
        key_cols = [col for col in numeric_cols if any(keyword in col.lower() for keyword in
                   ['span', 'width', 'height', 'freq', 'speed', 'amplitude', 'damping', 'drag', 'lift'])]

        if len(key_cols) > 10:
            key_cols = key_cols[:10]  # Limit to most important columns

        for i, col1 in enumerate(key_cols):
            for col2 in key_cols[i+1:]:
                if col1 != col2 and col1 in X.columns and col2 in X.columns:
                    # Avoid division by zero
                    denominator = X[col2].replace(0, 1e-8)
                    ratio_name = f"{col1}_{col2}_ratio"
                    X_ratios[ratio_name] = X[col1] / denominator

        return X_ratios

    def _create_polynomial_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial features"""
        if not self.include_polynomial:
            return X

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        # Limit columns to avoid feature explosion
        key_cols = [col for col in numeric_cols if not col.endswith('_ratio') and not col.endswith('_Number')][:8]

        if len(key_cols) == 0:
            return X

        try:
            self.poly_transformer = SklearnPolynomialFeatures(
                degree=self.polynomial_degree,
                include_bias=False,
                interaction_only=not self.include_interactions
            )

            X_poly_array = self.poly_transformer.fit_transform(X[key_cols])
            poly_feature_names = self.poly_transformer.get_feature_names_out(key_cols)

            # Create polynomial DataFrame
            X_poly = pd.DataFrame(X_poly_array, columns=poly_feature_names, index=X.index)

            # Combine with original features
            X_combined = pd.concat([X, X_poly.iloc[:, len(key_cols):]], axis=1)  # Skip original features

            return X_combined

        except Exception as e:
            logger.warning(f"Failed to create polynomial features: {e}")
            return X

    def _create_statistical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features from related columns"""
        if not self.include_statistical:
            return X

        X_stats = X.copy()

        # Group related features
        feature_groups = {
            'geometric': ['Span_m', 'Width_m', 'Height_m', 'Total_Length_m'],
            'frequency': ['Natural_Freq_Hz', 'First_Freq_Hz', 'Second_Freq_Hz'],
            'aerodynamic': ['Drag_Coefficient', 'Lift_Coefficient', 'VIV_Wind_Speed_ms', 'Critical_Wind_Speed_ms'],
            'amplitude': ['Max_Amplitude_mm', 'Amplitude_RMS_mm']
        }

        for group_name, cols in feature_groups.items():
            available_cols = [col for col in cols if col in X.columns]

            if len(available_cols) >= 2:
                group_data = X[available_cols]

                # Statistical features
                X_stats[f'{group_name}_mean'] = group_data.mean(axis=1)
                X_stats[f'{group_name}_std'] = group_data.std(axis=1)
                X_stats[f'{group_name}_min'] = group_data.min(axis=1)
                X_stats[f'{group_name}_max'] = group_data.max(axis=1)
                X_stats[f'{group_name}_range'] = X_stats[f'{group_name}_max'] - X_stats[f'{group_name}_min']

                # Coefficient of variation
                X_stats[f'{group_name}_cv'] = X_stats[f'{group_name}_std'] / (X_stats[f'{group_name}_mean'] + 1e-8)

        return X_stats

    def _select_features(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Select most important features"""
        if self.max_features is None or X.shape[1] <= self.max_features:
            return X

        if y is None:
            # Use variance-based selection
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold()
            X_selected = selector.fit_transform(X)
            selected_features = X.columns[selector.get_support()]
            return X[selected_features[:self.max_features]]

        # Use statistical tests for feature selection
        try:
            if y.dtype == 'object' or len(y.unique()) <= 10:
                # Classification
                score_func = f_classif
            else:
                # Regression
                score_func = f_regression

            self.feature_selector = SelectKBest(score_func=score_func, k=self.max_features)
            X_selected = self.feature_selector.fit_transform(X, y)
            selected_features = X.columns[self.feature_selector.get_support()]

            logger.info(f"Selected {len(selected_features)} features out of {X.shape[1]}")
            return X[selected_features]

        except Exception as e:
            logger.warning(f"Feature selection failed: {e}")
            return X.iloc[:, :self.max_features]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the feature engineer"""
        # Store original feature names
        self.original_features_ = X.columns.tolist()

        # Create features step by step
        X_features = X.copy()

        if self.include_domain_features:
            X_features = self._create_domain_features(X_features)

        if self.include_dimensionless:
            X_features = self._create_dimensionless_parameters(X_features)

        if self.include_ratios:
            X_features = self._create_ratio_features(X_features)

        if self.include_statistical:
            X_features = self._create_statistical_features(X_features)

        if self.include_polynomial:
            X_features = self._create_polynomial_features(X_features)

        # Feature selection
        X_features = self._select_features(X_features, y)

        self.feature_names_ = X_features.columns.tolist()
        self.fitted_ = True

        logger.info(f"Feature engineering fit completed: {len(self.original_features_)} -> {len(self.feature_names_)} features")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data"""
        if not self.fitted_:
            raise ValueError("FeatureEngineer must be fitted before transform")

        # Apply the same transformations as in fit
        X_features = X.copy()

        if self.include_domain_features:
            X_features = self._create_domain_features(X_features)

        if self.include_dimensionless:
            X_features = self._create_dimensionless_parameters(X_features)

        if self.include_ratios:
            X_features = self._create_ratio_features(X_features)

        if self.include_statistical:
            X_features = self._create_statistical_features(X_features)

        if self.include_polynomial and self.poly_transformer is not None:
            try:
                numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                key_cols = [col for col in numeric_cols if not col.endswith('_ratio') and not col.endswith('_Number')][:8]

                if len(key_cols) > 0:
                    X_poly_array = self.poly_transformer.transform(X_features[key_cols])
                    poly_feature_names = self.poly_transformer.get_feature_names_out(key_cols)

                    X_poly = pd.DataFrame(X_poly_array, columns=poly_feature_names, index=X_features.index)
                    X_features = pd.concat([X_features, X_poly.iloc[:, len(key_cols):]], axis=1)
            except Exception as e:
                logger.warning(f"Failed to transform polynomial features: {e}")

        # Apply feature selection
        if self.feature_selector is not None:
            try:
                available_features = [col for col in self.feature_names_ if col in X_features.columns]
                if len(available_features) != len(self.feature_names_):
                    logger.warning(f"Some features missing during transform: {len(available_features)}/{len(self.feature_names_)}")

                X_features = X_features[available_features]
            except Exception as e:
                logger.warning(f"Feature selection transform failed: {e}")
                # Fallback to available features
                available_features = [col for col in self.feature_names_ if col in X_features.columns]
                X_features = X_features[available_features]
        else:
            # Select features by name
            available_features = [col for col in self.feature_names_ if col in X_features.columns]
            X_features = X_features[available_features]

        return X_features

    def get_feature_importance_names(self) -> Dict[str, str]:
        """Get feature names with descriptions for interpretability"""
        descriptions = {
            'Width_Height_Ratio': 'Bridge deck width to height ratio (geometric parameter)',
            'Slenderness_Ratio': 'Span to height ratio (structural slenderness)',
            'Frequency_Ratio': 'First to natural frequency ratio (dynamic characteristic)',
            'Wind_Speed_Ratio': 'VIV to critical wind speed ratio (aerodynamic parameter)',
            'Amplitude_Height_Ratio': 'Amplitude to deck height ratio (vibration severity)',
            'Strouhal_Number': 'Dimensionless frequency parameter (St = fD/U)',
            'Reynolds_Number': 'Dimensionless flow parameter (Re = UD/ν)',
            'Reduced_Velocity': 'Dimensionless velocity parameter (Ur = U/fD)',
            'Drag_Lift_Ratio': 'Aerodynamic force ratio (drag/lift)',
            'Aspect_Ratio': 'Bridge length to width ratio (planform geometry)'
        }

        feature_descriptions = {}
        for feature in self.feature_names_:
            if feature in descriptions:
                feature_descriptions[feature] = descriptions[feature]
            elif any(keyword in feature.lower() for keyword in ['ratio', 'number', 'velocity']):
                feature_descriptions[feature] = f'Engineered feature: {feature}'
            else:
                feature_descriptions[feature] = f'Original or derived feature: {feature}'

        return feature_descriptions

class FeaturePipeline:
    """Pipeline for combining multiple feature engineering steps"""

    def __init__(self, steps: List[Tuple[str, BaseEstimator]]):
        """
        Initialize feature pipeline

        Args:
            steps: List of (name, transformer) tuples
        """
        self.steps = steps
        self.fitted_steps_ = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit all steps in the pipeline"""
        X_current = X.copy()

        for name, transformer in self.steps:
            logger.info(f"Fitting {name}...")
            transformer.fit(X_current, y)
            X_current = transformer.transform(X_current)
            self.fitted_steps_.append((name, transformer))

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data through all steps"""
        X_current = X.copy()

        for name, transformer in self.fitted_steps_:
            X_current = transformer.transform(X_current)

        return X_current

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform data"""
        return self.fit(X, y).transform(X)

def create_feature_pipeline(config: Dict[str, Any]) -> FeaturePipeline:
    """
    Create a feature engineering pipeline based on configuration

    Args:
        config: Configuration dictionary

    Returns:
        FeaturePipeline object
    """
    steps = []

    # Main feature engineer
    feature_engineer = BridgeFeatureEngineer(
        include_domain_features=config.get('include_domain_features', True),
        include_polynomial=config.get('include_polynomial', False),
        polynomial_degree=config.get('polynomial_degree', 2),
        include_interactions=config.get('include_interactions', False),
        include_ratios=config.get('include_ratios', True),
        include_dimensionless=config.get('include_dimensionless', True),
        include_statistical=config.get('include_statistical', False),
        max_features=config.get('max_features', None)
    )

    steps.append(('feature_engineer', feature_engineer))

    # Optional PCA for dimensionality reduction
    if config.get('apply_pca', False):
        n_components = config.get('pca_components', 0.95)
        pca = PCA(n_components=n_components, random_state=42)
        steps.append(('pca', pca))

    return FeaturePipeline(steps)

def main():
    """Example usage"""
    # Create sample data
    data = {
        'Span_m': [1000, 800, 1200],
        'Width_m': [30, 25, 35],
        'Height_m': [3, 2.5, 3.5],
        'Natural_Freq_Hz': [0.15, 0.20, 0.12],
        'VIV_Wind_Speed_ms': [8, 7, 9],
        'Max_Amplitude_mm': [25, 15, 40]
    }
    df = pd.DataFrame(data)

    # Feature engineering
    feature_engineer = BridgeFeatureEngineer(
        include_domain_features=True,
        include_dimensionless=True,
        include_polynomial=True
    )

    X_features = feature_engineer.fit_transform(df)
    print(f"Original features: {df.shape[1]}")
    print(f"Engineered features: {X_features.shape[1]}")
    print(f"Feature names: {feature_engineer.feature_names_}")

if __name__ == "__main__":
    main()