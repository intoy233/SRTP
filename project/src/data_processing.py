#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Bridge VIV Data Processing Module
Extends the original data processing with advanced features and pipeline support
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from scipy import stats
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings
import joblib
from pathlib import Path

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    """Box-Cox transformation for skewed features"""

    def __init__(self, columns: Optional[List[str]] = None):
        self.columns = columns
        self.lambdas_ = {}
        self.fitted_columns_ = []

    def fit(self, X: pd.DataFrame, y=None):
        if self.columns is None:
            # Auto-detect skewed columns (|skewness| > 0.5)
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            skewness = X[numeric_cols].skew()
            self.columns = skewness[abs(skewness) > 0.5].index.tolist()

        for col in self.columns:
            if col in X.columns:
                # Ensure positive values for Box-Cox
                if (X[col] <= 0).any():
                    logger.warning(f"Column {col} has non-positive values, skipping Box-Cox")
                    continue

                try:
                    _, fitted_lambda = boxcox(X[col])
                    self.lambdas_[col] = fitted_lambda
                    self.fitted_columns_.append(col)
                except Exception as e:
                    logger.warning(f"Failed to fit Box-Cox for column {col}: {e}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = X.copy()

        for col in self.fitted_columns_:
            if col in X_transformed.columns:
                try:
                    X_transformed[col] = boxcox(X_transformed[col], lmbda=self.lambdas_[col])
                except Exception as e:
                    logger.warning(f"Failed to transform column {col}: {e}")

        return X_transformed

class PolynomialFeatures(BaseEstimator, TransformerMixin):
    """Custom polynomial features generator"""

    def __init__(self, degree: int = 2, include_bias: bool = False,
                 interaction_only: bool = False, columns: Optional[List[str]] = None):
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.columns = columns
        self.feature_names_ = []

    def fit(self, X: pd.DataFrame, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include=[np.number]).columns.tolist()

        self.feature_names_ = []

        # Original features
        self.feature_names_.extend(X.columns.tolist())

        # Polynomial features
        if self.degree >= 2:
            # Squared terms
            for col in self.columns:
                if col in X.columns:
                    self.feature_names_.append(f"{col}^2")

            # Interaction terms
            if not self.interaction_only and len(self.columns) > 1:
                for i, col1 in enumerate(self.columns):
                    for col2 in self.columns[i+1:]:
                        if col1 in X.columns and col2 in X.columns:
                            self.feature_names_.append(f"{col1}*{col2}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = X.copy()

        # Add squared terms
        if self.degree >= 2:
            for col in self.columns:
                if col in X_transformed.columns:
                    X_transformed[f"{col}^2"] = X_transformed[col] ** 2

            # Add interaction terms
            if not self.interaction_only and len(self.columns) > 1:
                for i, col1 in enumerate(self.columns):
                    for col2 in self.columns[i+1:]:
                        if col1 in X_transformed.columns and col2 in X_transformed.columns:
                            X_transformed[f"{col1}*{col2}"] = X_transformed[col1] * X_transformed[col2]

        return X_transformed

class BridgeVIVDataProcessor:
    """Enhanced Bridge VIV Data Processor with pipeline support"""

    def __init__(self, data_path: Union[str, Path], random_state: int = 42):
        """
        Initialize data processor

        Args:
            data_path: Path to CSV data file
            random_state: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.random_state = random_state
        self.raw_data = None
        self.processed_data = None

        # Transformers
        self.scaler = None
        self.label_encoders = {}
        self.boxcox_transformer = None
        self.poly_transformer = None

        # Feature engineering components
        self.feature_pipeline = None
        self.preprocessing_pipeline = None

        # Set random seeds for reproducibility
        np.random.seed(self.random_state)

        logger.info(f"Initialized BridgeVIVDataProcessor with random_state={self.random_state}")

    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file"""
        try:
            # Try different encodings
            for encoding in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
                try:
                    self.raw_data = pd.read_csv(self.data_path, encoding=encoding)
                    logger.info(f"Successfully loaded data with encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue

            if self.raw_data is None:
                raise ValueError("Failed to load data with any encoding")

            logger.info(f"Loaded {len(self.raw_data)} records with {len(self.raw_data.columns)} columns")
            return self.raw_data

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def explore_data(self) -> Dict[str, Any]:
        """Data exploration and analysis"""
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        exploration_results = {}

        logger.info("=== Data Exploration ===")

        # Basic info
        exploration_results['shape'] = self.raw_data.shape
        exploration_results['columns'] = self.raw_data.columns.tolist()
        exploration_results['dtypes'] = self.raw_data.dtypes.to_dict()

        # Missing values
        missing_values = self.raw_data.isnull().sum()
        exploration_results['missing_values'] = missing_values[missing_values > 0].to_dict()

        # Numeric columns statistics
        numeric_cols = self.raw_data.select_dtypes(include=[np.number]).columns
        exploration_results['numeric_stats'] = self.raw_data[numeric_cols].describe().to_dict()

        # Target variable distribution
        target_cols = ['Max_Amplitude_mm', 'Risk_Level']
        for col in target_cols:
            if col in self.raw_data.columns:
                if self.raw_data[col].dtype == 'object':
                    exploration_results[f'{col}_distribution'] = self.raw_data[col].value_counts().to_dict()
                else:
                    exploration_results[f'{col}_stats'] = {
                        'mean': self.raw_data[col].mean(),
                        'std': self.raw_data[col].std(),
                        'min': self.raw_data[col].min(),
                        'max': self.raw_data[col].max(),
                        'skewness': self.raw_data[col].skew()
                    }

        logger.info(f"Data shape: {exploration_results['shape']}")
        logger.info(f"Missing values: {len(exploration_results['missing_values'])} columns")

        return exploration_results

    def clean_data(self, outlier_method: str = 'iqr',
                   outlier_threshold: float = 1.5) -> pd.DataFrame:
        """
        Clean data with outlier detection and missing value imputation

        Args:
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'none')
            outlier_threshold: Threshold for outlier detection
        """
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        data_cleaned = self.raw_data.copy()

        logger.info("Starting data cleaning...")

        # Handle missing values
        numeric_cols = data_cleaned.select_dtypes(include=[np.number]).columns
        categorical_cols = data_cleaned.select_dtypes(include=['object']).columns

        # Impute numeric columns with median
        for col in numeric_cols:
            if data_cleaned[col].isnull().sum() > 0:
                median_val = data_cleaned[col].median()
                data_cleaned[col].fillna(median_val, inplace=True)
                logger.info(f"Filled {col} missing values with median: {median_val:.3f}")

        # Impute categorical columns with mode
        for col in categorical_cols:
            if data_cleaned[col].isnull().sum() > 0:
                mode_val = data_cleaned[col].mode().iloc[0] if not data_cleaned[col].mode().empty else 'Unknown'
                data_cleaned[col].fillna(mode_val, inplace=True)
                logger.info(f"Filled {col} missing values with mode: {mode_val}")

        # Handle outliers
        if outlier_method == 'iqr':
            for col in numeric_cols:
                if col not in ['BridgeID', 'Year']:  # Skip ID and year columns
                    Q1 = data_cleaned[col].quantile(0.25)
                    Q3 = data_cleaned[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - outlier_threshold * IQR
                    upper_bound = Q3 + outlier_threshold * IQR

                    outliers = ((data_cleaned[col] < lower_bound) |
                               (data_cleaned[col] > upper_bound)).sum()

                    if outliers > 0:
                        logger.info(f"Column '{col}': found {outliers} outliers")
                        data_cleaned[col] = np.clip(data_cleaned[col], lower_bound, upper_bound)

        elif outlier_method == 'zscore':
            for col in numeric_cols:
                if col not in ['BridgeID', 'Year']:
                    z_scores = np.abs(stats.zscore(data_cleaned[col]))
                    outliers = (z_scores > outlier_threshold).sum()

                    if outliers > 0:
                        logger.info(f"Column '{col}': found {outliers} outliers (z-score)")
                        # Replace outliers with median
                        median_val = data_cleaned[col].median()
                        data_cleaned.loc[z_scores > outlier_threshold, col] = median_val

        self.processed_data = data_cleaned
        logger.info("Data cleaning completed")

        return data_cleaned

    def feature_engineering(self,
                          include_polynomial: bool = False,
                          include_interactions: bool = False,
                          apply_boxcox: bool = False,
                          polynomial_degree: int = 2) -> pd.DataFrame:
        """
        Advanced feature engineering with configurable options

        Args:
            include_polynomial: Whether to include polynomial features
            include_interactions: Whether to include interaction features
            apply_boxcox: Whether to apply Box-Cox transformation
            polynomial_degree: Degree for polynomial features
        """
        if self.processed_data is None:
            raise ValueError("Data not cleaned. Call clean_data() first.")

        data_fe = self.processed_data.copy()

        logger.info("Starting feature engineering...")

        # Create basic derived features
        if 'Width_m' in data_fe.columns and 'Height_m' in data_fe.columns:
            data_fe['Width_Height_Ratio'] = data_fe['Width_m'] / data_fe['Height_m']

        if 'Span_m' in data_fe.columns and 'Height_m' in data_fe.columns:
            data_fe['Slenderness_Ratio'] = data_fe['Span_m'] / data_fe['Height_m']

        if 'Max_Amplitude_mm' in data_fe.columns and 'Height_m' in data_fe.columns:
            data_fe['Amplitude_Height_Ratio'] = data_fe['Max_Amplitude_mm'] / (data_fe['Height_m'] * 1000)

        if 'First_Freq_Hz' in data_fe.columns and 'Natural_Freq_Hz' in data_fe.columns:
            data_fe['Frequency_Ratio'] = data_fe['First_Freq_Hz'] / data_fe['Natural_Freq_Hz']

        if 'VIV_Wind_Speed_ms' in data_fe.columns and 'Critical_Wind_Speed_ms' in data_fe.columns:
            data_fe['Wind_Speed_Ratio'] = data_fe['VIV_Wind_Speed_ms'] / data_fe['Critical_Wind_Speed_ms']

        # Strouhal number approximation
        if all(col in data_fe.columns for col in ['VIV_Wind_Speed_ms', 'Natural_Freq_Hz', 'Width_m']):
            data_fe['Strouhal_Number'] = (data_fe['Natural_Freq_Hz'] * data_fe['Width_m']) / data_fe['VIV_Wind_Speed_ms']

        # Reynolds number approximation (simplified)
        if all(col in data_fe.columns for col in ['VIV_Wind_Speed_ms', 'Width_m']):
            # Assuming air density and kinematic viscosity at standard conditions
            data_fe['Reynolds_Number'] = (data_fe['VIV_Wind_Speed_ms'] * data_fe['Width_m']) / 1.5e-5

        # Encode categorical variables
        categorical_cols = ['BridgeType', 'Structure_Type', 'Vibration_Suppression', 'Risk_Level']
        for col in categorical_cols:
            if col in data_fe.columns:
                le = LabelEncoder()
                data_fe[f'{col}_encoded'] = le.fit_transform(data_fe[col].astype(str))
                self.label_encoders[col] = le

        # Apply Box-Cox transformation for skewed features
        if apply_boxcox:
            numeric_cols = data_fe.select_dtypes(include=[np.number]).columns
            skewed_cols = []
            for col in numeric_cols:
                if abs(data_fe[col].skew()) > 0.5 and (data_fe[col] > 0).all():
                    skewed_cols.append(col)

            if skewed_cols:
                self.boxcox_transformer = BoxCoxTransformer(columns=skewed_cols)
                data_fe = self.boxcox_transformer.fit_transform(data_fe)
                logger.info(f"Applied Box-Cox transformation to {len(skewed_cols)} columns")

        # Add polynomial features
        if include_polynomial:
            numeric_cols = [col for col in data_fe.select_dtypes(include=[np.number]).columns
                          if col not in ['BridgeID', 'Year'] and not col.endswith('_encoded')]

            self.poly_transformer = PolynomialFeatures(
                degree=polynomial_degree,
                interaction_only=not include_interactions,
                columns=numeric_cols[:5]  # Limit to avoid feature explosion
            )
            data_fe = self.poly_transformer.fit_transform(data_fe)
            logger.info(f"Added polynomial features (degree={polynomial_degree})")

        self.processed_data = data_fe
        logger.info("Feature engineering completed")

        return data_fe

    def prepare_ml_data(self, target_tasks: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, pd.Series], List[str]]:
        """
        Prepare data for machine learning

        Args:
            target_tasks: List of target tasks ('amplitude', 'risk_class', 'viv_occurrence', 'all')
        """
        if self.processed_data is None:
            raise ValueError("Data not processed. Call feature_engineering() first.")

        if target_tasks is None:
            target_tasks = ['amplitude', 'risk_class', 'viv_occurrence']

        # Select feature columns
        feature_cols = []
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns

        # Core features
        core_features = [
            'Span_m', 'Width_Height_Ratio', 'Slenderness_Ratio', 'Natural_Freq_Hz',
            'First_Freq_Hz', 'Second_Freq_Hz', 'Drag_Coefficient', 'Lift_Coefficient',
            'VIV_Wind_Speed_ms', 'Damping_Ratio', 'Frequency_Ratio', 'Wind_Speed_Ratio'
        ]

        # Add engineered features
        engineered_features = [
            'Amplitude_Height_Ratio', 'Strouhal_Number', 'Reynolds_Number'
        ]

        # Add encoded categorical features
        encoded_features = [col for col in self.processed_data.columns if col.endswith('_encoded')]

        # Combine all features
        all_potential_features = core_features + engineered_features + encoded_features

        # Filter existing features
        feature_cols = [col for col in all_potential_features if col in self.processed_data.columns]

        # Add polynomial features if available
        poly_features = [col for col in self.processed_data.columns
                        if '^2' in col or '*' in col]
        feature_cols.extend(poly_features)

        X = self.processed_data[feature_cols]

        # Prepare targets
        targets = {}

        if 'amplitude' in target_tasks or 'all' in target_tasks:
            if 'Max_Amplitude_mm' in self.processed_data.columns:
                targets['amplitude'] = self.processed_data['Max_Amplitude_mm']

        if 'risk_class' in target_tasks or 'all' in target_tasks:
            if 'Risk_Level_encoded' in self.processed_data.columns:
                targets['risk_class'] = self.processed_data['Risk_Level_encoded']

        if 'viv_occurrence' in target_tasks or 'all' in target_tasks:
            if 'Max_Amplitude_mm' in self.processed_data.columns:
                # Binary classification: VIV occurrence (amplitude > 20mm)
                targets['viv_occurrence'] = (self.processed_data['Max_Amplitude_mm'] > 20).astype(int)

        logger.info(f"Prepared ML data: {X.shape[1]} features, {len(targets)} target tasks")

        return X, targets, feature_cols

    def split_and_scale_data(self,
                           test_size: float = 0.2,
                           val_size: float = 0.2,
                           scaler_type: str = 'standard',
                           apply_smote: bool = False,
                           target_tasks: List[str] = None) -> Dict[str, Dict]:
        """
        Split data and apply scaling with optional SMOTE

        Args:
            test_size: Test set proportion
            val_size: Validation set proportion (from remaining data)
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
            apply_smote: Whether to apply SMOTE for imbalanced classification
            target_tasks: Target tasks to prepare
        """
        X, targets, feature_names = self.prepare_ml_data(target_tasks)

        # Initialize scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_names, index=X.index)

        datasets = {}

        for task_name, y in targets.items():
            # Stratified split for classification tasks
            stratify = y if task_name in ['risk_class', 'viv_occurrence'] else None

            # Train/test split
            X_temp, X_test, y_temp, y_test = train_test_split(
                X_scaled, y, test_size=test_size,
                random_state=self.random_state, stratify=stratify
            )

            # Train/validation split
            val_size_adjusted = val_size / (1 - test_size)
            stratify_temp = y_temp if task_name in ['risk_class', 'viv_occurrence'] else None

            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted,
                random_state=self.random_state, stratify=stratify_temp
            )

            # Apply SMOTE for imbalanced classification
            if apply_smote and task_name in ['risk_class', 'viv_occurrence']:
                try:
                    smote = SMOTE(random_state=self.random_state)
                    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
                    X_train = pd.DataFrame(X_train_smote, columns=feature_names)
                    y_train = pd.Series(y_train_smote)
                    logger.info(f"Applied SMOTE to {task_name}: {len(X_train)} samples after oversampling")
                except Exception as e:
                    logger.warning(f"Failed to apply SMOTE to {task_name}: {e}")

            datasets[task_name] = {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test
            }

        logger.info("Data splitting and scaling completed")
        for task_name, data in datasets.items():
            logger.info(f"{task_name}: Train={len(data['X_train'])}, Val={len(data['X_val'])}, Test={len(data['X_test'])}")

        return datasets, feature_names

    def get_cv_folds(self, n_splits: int = 5, task_name: str = 'amplitude') -> StratifiedKFold:
        """Get cross-validation folds"""
        if task_name in ['risk_class', 'viv_occurrence']:
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        else:
            from sklearn.model_selection import KFold
            return KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

    def save_processor(self, path: Union[str, Path]) -> None:
        """Save the processor state"""
        save_dict = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'boxcox_transformer': self.boxcox_transformer,
            'poly_transformer': self.poly_transformer,
            'random_state': self.random_state
        }
        joblib.dump(save_dict, path)
        logger.info(f"Processor saved to {path}")

    def load_processor(self, path: Union[str, Path]) -> None:
        """Load the processor state"""
        save_dict = joblib.load(path)
        self.scaler = save_dict['scaler']
        self.label_encoders = save_dict['label_encoders']
        self.boxcox_transformer = save_dict['boxcox_transformer']
        self.poly_transformer = save_dict['poly_transformer']
        self.random_state = save_dict['random_state']
        logger.info(f"Processor loaded from {path}")

def main():
    """Example usage"""
    processor = BridgeVIVDataProcessor('data/bridge_dataset_fixed.csv')

    # Data processing pipeline
    processor.load_data()
    processor.explore_data()
    processor.clean_data()
    processor.feature_engineering(include_polynomial=True, apply_boxcox=True)

    # Prepare ML data
    datasets, features = processor.split_and_scale_data(
        apply_smote=True,
        target_tasks=['amplitude', 'risk_class', 'viv_occurrence']
    )

    return processor, datasets, features

if __name__ == "__main__":
    processor, datasets, features = main()