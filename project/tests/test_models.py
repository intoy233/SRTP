#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for models module
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models import (
    ModelFactory, LinearModel, RandomForestModel, XGBoostModel,
    LightGBMModel, EnsembleModel
)

class TestModelFactory:
    """Test cases for ModelFactory"""

    def test_create_linear_model(self):
        """Test creating linear model"""
        model = ModelFactory.create_model('linear', 'regression')
        assert model is not None
        assert isinstance(model, LinearModel)

    def test_create_random_forest_model(self):
        """Test creating random forest model"""
        model = ModelFactory.create_model('random_forest', 'regression', n_estimators=10)
        assert model is not None
        assert isinstance(model, RandomForestModel)

    def test_create_xgboost_model(self):
        """Test creating XGBoost model"""
        model = ModelFactory.create_model('xgboost', 'regression', n_estimators=10)
        assert model is not None
        assert isinstance(model, XGBoostModel)

    def test_create_lightgbm_model(self):
        """Test creating LightGBM model"""
        model = ModelFactory.create_model('lightgbm', 'regression', n_estimators=10)
        assert model is not None
        assert isinstance(model, LightGBMModel)

    def test_get_baseline_models(self):
        """Test getting baseline models"""
        models = ModelFactory.get_baseline_models('regression')
        assert isinstance(models, dict)
        assert len(models) > 0
        assert 'linear' in models

    def test_invalid_model_type(self):
        """Test invalid model type"""
        with pytest.raises(ValueError):
            ModelFactory.create_model('invalid_model', 'regression')

class TestLinearModel:
    """Test cases for LinearModel"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y_reg = np.random.randn(100)
        y_clf = np.random.randint(0, 2, 100)
        return X, y_reg, y_clf

    def test_linear_regression(self, sample_data):
        """Test linear regression"""
        X, y_reg, _ = sample_data

        model = LinearModel('regression', 'linear')
        model.fit(X, y_reg)

        assert model.is_fitted
        assert model.training_time > 0

        predictions = model.predict(X)
        assert len(predictions) == len(y_reg)

    def test_logistic_regression(self, sample_data):
        """Test logistic regression"""
        X, _, y_clf = sample_data

        model = LinearModel('binary_classification', 'logistic')
        model.fit(X, y_clf)

        assert model.is_fitted

        predictions = model.predict(X)
        assert len(predictions) == len(y_clf)

        probabilities = model.predict_proba(X)
        assert probabilities.shape == (len(y_clf), 2)

    def test_save_load_model(self, sample_data):
        """Test model saving and loading"""
        X, y_reg, _ = sample_data

        model = LinearModel('regression', 'linear')
        model.fit(X, y_reg)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model.save_model(f.name)

            # Create new model and load
            new_model = LinearModel('regression', 'linear')
            new_model.load_model(f.name)

            assert new_model.is_fitted

            # Test predictions are same
            pred1 = model.predict(X[:10])
            pred2 = new_model.predict(X[:10])
            np.testing.assert_array_almost_equal(pred1, pred2)

        # Cleanup
        Path(f.name).unlink()

class TestRandomForestModel:
    """Test cases for RandomForestModel"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y_reg = np.random.randn(50)
        y_clf = np.random.randint(0, 2, 50)
        return X, y_reg, y_clf

    def test_random_forest_regression(self, sample_data):
        """Test random forest regression"""
        X, y_reg, _ = sample_data

        model = RandomForestModel('regression', n_estimators=5)
        model.fit(X, y_reg)

        assert model.is_fitted

        predictions = model.predict(X)
        assert len(predictions) == len(y_reg)

        # Test feature importance
        importance = model.get_feature_importance()
        assert len(importance) == X.shape[1]

    def test_random_forest_classification(self, sample_data):
        """Test random forest classification"""
        X, _, y_clf = sample_data

        model = RandomForestModel('binary_classification', n_estimators=5)
        model.fit(X, y_clf)

        assert model.is_fitted

        predictions = model.predict(X)
        assert len(predictions) == len(y_clf)

        probabilities = model.predict_proba(X)
        assert probabilities.shape == (len(y_clf), 2)

class TestXGBoostModel:
    """Test cases for XGBoostModel"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y_reg = np.random.randn(50)
        y_clf = np.random.randint(0, 2, 50)
        return X, y_reg, y_clf

    def test_xgboost_regression(self, sample_data):
        """Test XGBoost regression"""
        X, y_reg, _ = sample_data

        model = XGBoostModel('regression', n_estimators=5)
        model.fit(X, y_reg)

        assert model.is_fitted

        predictions = model.predict(X)
        assert len(predictions) == len(y_reg)

        # Test feature importance
        importance = model.get_feature_importance()
        assert len(importance) == X.shape[1]

    def test_xgboost_with_validation(self, sample_data):
        """Test XGBoost with validation data"""
        X, y_reg, _ = sample_data

        # Split data
        X_train, X_val = X[:30], X[30:]
        y_train, y_val = y_reg[:30], y_reg[30:]

        model = XGBoostModel('regression', n_estimators=10)
        model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=5)

        assert model.is_fitted

class TestEnsembleModel:
    """Test cases for EnsembleModel"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y_reg = np.random.randn(50)
        y_clf = np.random.randint(0, 2, 50)
        return X, y_reg, y_clf

    def test_ensemble_regression(self, sample_data):
        """Test ensemble for regression"""
        X, y_reg, _ = sample_data

        # Create base models
        base_models = [
            LinearModel('regression', 'linear'),
            RandomForestModel('regression', n_estimators=5),
            XGBoostModel('regression', n_estimators=5)
        ]

        ensemble = EnsembleModel('regression', base_models)
        ensemble.fit(X, y_reg)

        assert ensemble.is_fitted

        predictions = ensemble.predict(X)
        assert len(predictions) == len(y_reg)

    def test_ensemble_classification(self, sample_data):
        """Test ensemble for classification"""
        X, _, y_clf = sample_data

        # Create base models
        base_models = [
            LinearModel('binary_classification', 'logistic'),
            RandomForestModel('binary_classification', n_estimators=5)
        ]

        ensemble = EnsembleModel('binary_classification', base_models, voting='hard')
        ensemble.fit(X, y_clf)

        assert ensemble.is_fitted

        predictions = ensemble.predict(X)
        assert len(predictions) == len(y_clf)

def test_model_integration():
    """Integration test for complete model pipeline"""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randn(100)

    # Test different models
    models_to_test = ['linear', 'random_forest', 'xgboost', 'lightgbm']

    for model_name in models_to_test:
        try:
            model = ModelFactory.create_model(
                model_name, 'regression',
                n_estimators=5 if 'forest' in model_name or 'boost' in model_name else None
            )
            model.fit(X, y)

            predictions = model.predict(X[:10])
            assert len(predictions) == 10

        except ImportError:
            # Skip if library not available
            continue

if __name__ == "__main__":
    pytest.main([__file__])