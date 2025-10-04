#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for metrics module
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from metrics import BridgeVIVMetrics, RegressionMetrics, ClassificationMetrics

class TestRegressionMetrics:
    """Test cases for RegressionMetrics"""

    @pytest.fixture
    def sample_predictions(self):
        """Create sample regression predictions"""
        np.random.seed(42)
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        return y_true, y_pred

    def test_calculate_metrics(self, sample_predictions):
        """Test regression metrics calculation"""
        y_true, y_pred = sample_predictions

        metrics = RegressionMetrics.calculate_metrics(y_true, y_pred)

        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'mape' in metrics

        # Check values are reasonable
        assert metrics['mse'] > 0
        assert metrics['rmse'] > 0
        assert metrics['mae'] > 0
        assert -1 <= metrics['r2'] <= 1
        assert metrics['mape'] >= 0

    def test_perfect_predictions(self):
        """Test with perfect predictions"""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()

        metrics = RegressionMetrics.calculate_metrics(y_true, y_pred)

        assert metrics['mse'] == 0
        assert metrics['rmse'] == 0
        assert metrics['mae'] == 0
        assert metrics['r2'] == 1.0
        assert metrics['mape'] == 0

class TestClassificationMetrics:
    """Test cases for ClassificationMetrics"""

    @pytest.fixture
    def sample_binary_predictions(self):
        """Create sample binary classification predictions"""
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.2, 0.4, 0.3, 0.7, 0.85, 0.15, 0.6])
        return y_true, y_pred, y_prob

    @pytest.fixture
    def sample_multiclass_predictions(self):
        """Create sample multiclass predictions"""
        y_true = np.array([0, 1, 2, 1, 0, 2, 1, 0, 2, 1])
        y_pred = np.array([0, 1, 2, 1, 0, 1, 1, 0, 2, 2])
        y_prob = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.3, 0.6, 0.1],
            [0.9, 0.05, 0.05],
            [0.2, 0.5, 0.3],
            [0.1, 0.8, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.1, 0.8],
            [0.2, 0.3, 0.5]
        ])
        return y_true, y_pred, y_prob

    def test_binary_classification_metrics(self, sample_binary_predictions):
        """Test binary classification metrics"""
        y_true, y_pred, y_prob = sample_binary_predictions

        metrics = ClassificationMetrics.calculate_metrics(y_true, y_pred, y_prob)

        # Check all required metrics are present
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'auc_roc' in metrics
        assert 'auc_pr' in metrics

        # Check values are in valid ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
        assert 0 <= metrics['auc_roc'] <= 1
        assert 0 <= metrics['auc_pr'] <= 1

    def test_multiclass_classification_metrics(self, sample_multiclass_predictions):
        """Test multiclass classification metrics"""
        y_true, y_pred, y_prob = sample_multiclass_predictions

        metrics = ClassificationMetrics.calculate_metrics(y_true, y_pred, y_prob)

        # Check required metrics
        assert 'accuracy' in metrics
        assert 'precision_macro' in metrics
        assert 'recall_macro' in metrics
        assert 'f1_macro' in metrics

        # Check values are valid
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision_macro'] <= 1
        assert 0 <= metrics['recall_macro'] <= 1
        assert 0 <= metrics['f1_macro'] <= 1

    def test_perfect_classification(self):
        """Test with perfect classification"""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = y_true.copy()
        y_prob = np.array([0.0, 1.0, 0.0, 1.0, 0.0])

        metrics = ClassificationMetrics.calculate_metrics(y_true, y_pred, y_prob)

        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0

class TestBridgeVIVMetrics:
    """Test cases for BridgeVIVMetrics"""

    @pytest.fixture
    def sample_data(self):
        """Create sample bridge VIV evaluation data"""
        predictions = {
            'amplitude': np.array([15.0, 25.0, 35.0, 45.0, 20.0]),
            'risk_class': np.array([0, 1, 1, 2, 0]),  # Low, Medium, Medium, High, Low
            'viv_occurrence': np.array([0, 1, 1, 1, 0])
        }

        targets = {
            'amplitude': np.array([18.0, 22.0, 38.0, 42.0, 19.0]),
            'risk_class': np.array([0, 1, 2, 2, 0]),
            'viv_occurrence': np.array([0, 1, 1, 1, 0])
        }

        probabilities = {
            'risk_class': np.array([
                [0.9, 0.08, 0.02],
                [0.2, 0.7, 0.1],
                [0.1, 0.6, 0.3],
                [0.05, 0.15, 0.8],
                [0.85, 0.1, 0.05]
            ]),
            'viv_occurrence': np.array([0.1, 0.8, 0.9, 0.85, 0.2])
        }

        return predictions, targets, probabilities

    def test_evaluate_all_tasks(self, sample_data):
        """Test comprehensive evaluation of all tasks"""
        predictions, targets, probabilities = sample_data

        results = BridgeVIVMetrics.evaluate_all_tasks(
            predictions, targets, probabilities
        )

        # Check structure
        assert 'amplitude' in results
        assert 'risk_class' in results
        assert 'viv_occurrence' in results
        assert 'summary' in results

        # Check amplitude metrics (regression)
        amplitude_metrics = results['amplitude']
        assert 'mse' in amplitude_metrics
        assert 'rmse' in amplitude_metrics
        assert 'mae' in amplitude_metrics
        assert 'r2' in amplitude_metrics

        # Check risk_class metrics (multiclass classification)
        risk_metrics = results['risk_class']
        assert 'accuracy' in risk_metrics
        assert 'precision_macro' in risk_metrics
        assert 'recall_macro' in risk_metrics
        assert 'f1_macro' in risk_metrics

        # Check viv_occurrence metrics (binary classification)
        viv_metrics = results['viv_occurrence']
        assert 'accuracy' in viv_metrics
        assert 'precision' in viv_metrics
        assert 'recall' in viv_metrics
        assert 'f1' in viv_metrics
        assert 'auc_roc' in viv_metrics

    def test_create_evaluation_report(self, sample_data):
        """Test evaluation report creation"""
        predictions, targets, probabilities = sample_data

        results = BridgeVIVMetrics.evaluate_all_tasks(
            predictions, targets, probabilities
        )

        report = BridgeVIVMetrics.create_evaluation_report(results)

        assert isinstance(report, str)
        assert len(report) > 0
        assert 'Bridge VIV Model Evaluation Report' in report
        assert 'Amplitude Prediction (Regression)' in report
        assert 'Risk Classification' in report
        assert 'VIV Occurrence Prediction' in report

    def test_compare_models(self, sample_data):
        """Test model comparison functionality"""
        predictions, targets, probabilities = sample_data

        # Create two sets of results
        results1 = BridgeVIVMetrics.evaluate_all_tasks(
            predictions, targets, probabilities
        )

        # Slightly different predictions for model 2
        predictions2 = predictions.copy()
        predictions2['amplitude'] = predictions2['amplitude'] + 2.0

        results2 = BridgeVIVMetrics.evaluate_all_tasks(
            predictions2, targets, probabilities
        )

        comparison = BridgeVIVMetrics.compare_models({
            'Model_1': results1,
            'Model_2': results2
        })

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) > 0
        assert 'Model_1' in comparison.columns
        assert 'Model_2' in comparison.columns

    def test_calculate_risk_stratified_metrics(self, sample_data):
        """Test risk-stratified metrics calculation"""
        predictions, targets, _ = sample_data

        # Create amplitude predictions and true values
        y_pred_amplitude = predictions['amplitude']
        y_true_amplitude = targets['amplitude']

        stratified_metrics = BridgeVIVMetrics.calculate_risk_stratified_metrics(
            y_true_amplitude, y_pred_amplitude
        )

        assert isinstance(stratified_metrics, dict)
        # Should have metrics for different risk levels
        assert len(stratified_metrics) > 0

def test_integration():
    """Integration test for complete metrics pipeline"""
    # Create comprehensive test data
    np.random.seed(42)
    n_samples = 50

    # Generate realistic bridge VIV predictions
    amplitude_true = np.random.exponential(25, n_samples)
    amplitude_pred = amplitude_true + np.random.normal(0, 3, n_samples)

    # Risk classes based on amplitude
    risk_true = np.where(amplitude_true < 20, 0, np.where(amplitude_true < 40, 1, 2))
    risk_pred = np.where(amplitude_pred < 20, 0, np.where(amplitude_pred < 40, 1, 2))

    # VIV occurrence based on amplitude
    viv_true = (amplitude_true > 15).astype(int)
    viv_pred = (amplitude_pred > 15).astype(int)

    # Create probability arrays
    risk_prob = np.random.dirichlet([1, 1, 1], n_samples)
    viv_prob = np.random.beta(2, 2, n_samples)

    predictions = {
        'amplitude': amplitude_pred,
        'risk_class': risk_pred,
        'viv_occurrence': viv_pred
    }

    targets = {
        'amplitude': amplitude_true,
        'risk_class': risk_true,
        'viv_occurrence': viv_true
    }

    probabilities = {
        'risk_class': risk_prob,
        'viv_occurrence': viv_prob
    }

    # Run complete evaluation
    results = BridgeVIVMetrics.evaluate_all_tasks(
        predictions, targets, probabilities
    )

    # Generate report
    report = BridgeVIVMetrics.create_evaluation_report(results)

    assert len(results) == 4  # 3 tasks + summary
    assert len(report) > 100  # Substantial report content

if __name__ == "__main__":
    pytest.main([__file__])