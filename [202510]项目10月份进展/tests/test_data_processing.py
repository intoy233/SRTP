#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for data processing module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_processing import BridgeVIVDataProcessor, BoxCoxTransformer, PolynomialFeatures

class TestBridgeVIVDataProcessor:
    """Test cases for BridgeVIVDataProcessor"""

    @pytest.fixture
    def sample_data(self):
        """Create sample bridge data"""
        data = {
            'BridgeID': ['001', '002', '003', '004', '005'],
            'BridgeName': ['Bridge A', 'Bridge B', 'Bridge C', 'Bridge D', 'Bridge E'],
            'BridgeType': ['Suspension', 'Cable-Stayed', 'Girder', 'Suspension', 'Cable-Stayed'],
            'Span_m': [1000, 800, 600, 1200, 900],
            'Width_m': [30, 25, 20, 35, 28],
            'Height_m': [3.0, 2.5, 2.0, 3.5, 2.8],
            'Natural_Freq_Hz': [0.15, 0.20, 0.25, 0.12, 0.18],
            'First_Freq_Hz': [0.13, 0.18, 0.23, 0.10, 0.16],
            'Second_Freq_Hz': [0.30, 0.40, 0.50, 0.25, 0.35],
            'VIV_Wind_Speed_ms': [8.0, 7.5, 9.0, 8.5, 7.8],
            'Critical_Wind_Speed_ms': [12.0, 11.0, 13.5, 12.8, 11.5],
            'Max_Amplitude_mm': [25, 15, 35, 45, 20],
            'Damping_Ratio': [0.01, 0.015, 0.008, 0.005, 0.012],
            'Risk_Level': ['Medium', 'Low', 'Medium', 'High', 'Low']
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def temp_csv_file(self, sample_data):
        """Create temporary CSV file with sample data"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            return f.name

    def test_initialization(self, temp_csv_file):
        """Test processor initialization"""
        processor = BridgeVIVDataProcessor(temp_csv_file)
        assert processor.data_path == Path(temp_csv_file)
        assert processor.random_state == 42
        assert processor.raw_data is None

    def test_load_data(self, temp_csv_file):
        """Test data loading"""
        processor = BridgeVIVDataProcessor(temp_csv_file)
        data = processor.load_data()

        assert data is not None
        assert len(data) == 5
        assert 'BridgeID' in data.columns
        assert 'Max_Amplitude_mm' in data.columns

    def test_explore_data(self, temp_csv_file):
        """Test data exploration"""
        processor = BridgeVIVDataProcessor(temp_csv_file)
        processor.load_data()

        results = processor.explore_data()

        assert 'shape' in results
        assert 'columns' in results
        assert 'missing_values' in results
        assert results['shape'] == (5, 14)

    def test_clean_data(self, temp_csv_file):
        """Test data cleaning"""
        processor = BridgeVIVDataProcessor(temp_csv_file)
        processor.load_data()

        cleaned_data = processor.clean_data()

        assert cleaned_data is not None
        assert len(cleaned_data) == 5
        # Check no missing values
        assert cleaned_data.isnull().sum().sum() == 0

    def test_feature_engineering(self, temp_csv_file):
        """Test feature engineering"""
        processor = BridgeVIVDataProcessor(temp_csv_file)
        processor.load_data()
        processor.clean_data()

        fe_data = processor.feature_engineering()

        assert fe_data is not None
        # Check new features are created
        assert 'Width_Height_Ratio' in fe_data.columns
        assert 'Slenderness_Ratio' in fe_data.columns

    def test_prepare_ml_data(self, temp_csv_file):
        """Test ML data preparation"""
        processor = BridgeVIVDataProcessor(temp_csv_file)
        processor.load_data()
        processor.clean_data()
        processor.feature_engineering()

        X, targets, feature_names = processor.prepare_ml_data()

        assert X is not None
        assert isinstance(targets, dict)
        assert len(feature_names) > 0
        assert 'amplitude' in targets or 'risk_class' in targets

    def test_split_and_scale_data(self, temp_csv_file):
        """Test data splitting and scaling"""
        processor = BridgeVIVDataProcessor(temp_csv_file)
        processor.load_data()
        processor.clean_data()
        processor.feature_engineering()

        datasets, feature_names = processor.split_and_scale_data(test_size=0.2)

        assert isinstance(datasets, dict)
        assert len(feature_names) > 0

        for task_name, data in datasets.items():
            assert 'X_train' in data
            assert 'X_test' in data
            assert 'y_train' in data
            assert 'y_test' in data

class TestBoxCoxTransformer:
    """Test cases for BoxCoxTransformer"""

    def test_boxcox_transform(self):
        """Test Box-Cox transformation"""
        # Create positive data for Box-Cox
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'feature3': [-1, 0, 1, 2, 3]  # This should be skipped
        })

        transformer = BoxCoxTransformer()
        transformed_data = transformer.fit_transform(data)

        assert transformed_data is not None
        assert len(transformer.fitted_columns_) <= 2  # feature3 should be skipped

class TestPolynomialFeatures:
    """Test cases for PolynomialFeatures"""

    def test_polynomial_features(self):
        """Test polynomial feature generation"""
        data = pd.DataFrame({
            'x1': [1, 2, 3],
            'x2': [2, 3, 4]
        })

        transformer = PolynomialFeatures(degree=2, include_interactions=True)
        transformed_data = transformer.fit_transform(data)

        assert transformed_data is not None
        assert transformed_data.shape[1] > data.shape[1]  # More features created
        assert 'x1^2' in transformed_data.columns or 'x1^2' in transformer.feature_names_

def test_integration():
    """Integration test for complete data processing pipeline"""
    # Create sample data
    data = {
        'Span_m': [1000, 800, 1200],
        'Width_m': [30, 25, 35],
        'Height_m': [3.0, 2.5, 3.5],
        'Natural_Freq_Hz': [0.15, 0.20, 0.12],
        'VIV_Wind_Speed_ms': [8.0, 7.5, 8.5],
        'Max_Amplitude_mm': [25, 15, 45],
        'Risk_Level': ['Medium', 'Low', 'High']
    }
    df = pd.DataFrame(data)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_file = f.name

    try:
        # Run complete pipeline
        processor = BridgeVIVDataProcessor(temp_file)
        processor.load_data()
        processor.clean_data()
        processor.feature_engineering()
        datasets, features = processor.split_and_scale_data()

        assert len(datasets) > 0
        assert len(features) > 0

    finally:
        # Cleanup
        Path(temp_file).unlink()

if __name__ == "__main__":
    pytest.main([__file__])