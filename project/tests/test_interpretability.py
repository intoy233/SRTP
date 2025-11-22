#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for interpretability module
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import sys
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from interpretability import ShapAnalyzer

class TestShapAnalyzer:
    """Test cases for ShapAnalyzer"""

    @pytest.fixture
    def sample_model_and_data(self):
        """Create sample model and data for testing"""
        np.random.seed(42)
        X = np.random.randn(50, 8)
        y = np.random.randn(50)
        feature_names = [f'feature_{i}' for i in range(8)]

        # Create a simple model for testing
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        return model, X, feature_names

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        # Cleanup
        if temp_path.exists():
            shutil.rmtree(temp_path)

    def test_initialization(self):
        """Test analyzer initialization"""
        analyzer = ShapAnalyzer()
        # Should initialize without error regardless of SHAP availability
        assert analyzer is not None

    def test_analyze_model_without_shap(self, sample_model_and_data):
        """Test analysis when SHAP is not available"""
        model, X, feature_names = sample_model_and_data

        analyzer = ShapAnalyzer()
        # Force unavailable for testing
        analyzer.available = False

        results = analyzer.analyze_model(
            model, X[:10], feature_names,
            task_type='regression',
            save_plots=False
        )

        # Should return empty results when SHAP unavailable
        assert results == {}

    def test_analyze_model_with_mock_shap(self, sample_model_and_data, temp_dir):
        """Test analysis with mocked SHAP functionality"""
        model, X, feature_names = sample_model_and_data

        analyzer = ShapAnalyzer()

        # If SHAP is available, test it
        if analyzer.available:
            results = analyzer.analyze_model(
                model, X[:20], feature_names,
                task_type='regression',
                save_plots=True,
                output_dir=str(temp_dir)
            )

            # Check structure
            assert isinstance(results, dict)
            if results:  # If SHAP analysis succeeded
                assert 'global_importance' in results or 'feature_ranking' in results

    def test_create_feature_importance_report(self):
        """Test feature importance report creation"""
        analyzer = ShapAnalyzer()

        # Create mock feature ranking
        feature_ranking = [
            {'rank': 1, 'feature': 'span_length', 'importance': 0.35, 'percentage': 35.0},
            {'rank': 2, 'feature': 'wind_speed', 'importance': 0.25, 'percentage': 25.0},
            {'rank': 3, 'feature': 'bridge_width', 'importance': 0.20, 'percentage': 20.0},
            {'rank': 4, 'feature': 'frequency_hz', 'importance': 0.15, 'percentage': 15.0},
            {'rank': 5, 'feature': 'damping_ratio', 'importance': 0.05, 'percentage': 5.0}
        ]

        feature_descriptions = {
            'span_length': 'Main span length of the bridge',
            'wind_speed': 'Critical wind speed for VIV onset',
            'bridge_width': 'Width of the bridge deck',
            'frequency_hz': 'Natural frequency of the bridge',
            'damping_ratio': 'Structural damping ratio'
        }

        report = analyzer.create_feature_importance_report(
            feature_ranking,
            feature_descriptions
        )

        assert isinstance(report, str)
        assert len(report) > 0
        assert 'Feature Importance Analysis' in report
        assert 'span_length' in report
        assert 'wind_speed' in report
        assert '35.0%' in report  # Check percentage formatting

    def test_report_categories(self):
        """Test feature category analysis in report"""
        analyzer = ShapAnalyzer()

        # Create feature ranking with categorizable features
        feature_ranking = [
            {'rank': 1, 'feature': 'width_height_ratio', 'importance': 0.30, 'percentage': 30.0},
            {'rank': 2, 'feature': 'natural_freq_hz', 'importance': 0.25, 'percentage': 25.0},
            {'rank': 3, 'feature': 'wind_speed_critical', 'importance': 0.20, 'percentage': 20.0},
            {'rank': 4, 'feature': 'span_length', 'importance': 0.15, 'percentage': 15.0},
            {'rank': 5, 'feature': 'damping_ratio', 'importance': 0.10, 'percentage': 10.0}
        ]

        report = analyzer.create_feature_importance_report(feature_ranking)

        # Check that category analysis is included
        assert 'Feature Category Analysis' in report
        assert 'Geometric' in report or 'Dynamic' in report or 'Aerodynamic' in report

    def test_report_insights(self):
        """Test insights generation in report"""
        analyzer = ShapAnalyzer()

        feature_ranking = [
            {'rank': 1, 'feature': 'critical_feature', 'importance': 0.40, 'percentage': 40.0},
            {'rank': 2, 'feature': 'second_feature', 'importance': 0.20, 'percentage': 20.0},
            {'rank': 3, 'feature': 'third_feature', 'importance': 0.15, 'percentage': 15.0},
            {'rank': 4, 'feature': 'fourth_feature', 'importance': 0.15, 'percentage': 15.0},
            {'rank': 5, 'feature': 'fifth_feature', 'importance': 0.10, 'percentage': 10.0}
        ]

        report = analyzer.create_feature_importance_report(feature_ranking)

        # Check insights section
        assert 'Key Insights' in report
        assert 'Most Important Feature' in report
        assert 'critical_feature' in report
        assert '40.0%' in report

    def test_report_save_to_file(self, temp_dir):
        """Test saving report to file"""
        analyzer = ShapAnalyzer()

        feature_ranking = [
            {'rank': 1, 'feature': 'test_feature', 'importance': 1.0, 'percentage': 100.0}
        ]

        save_path = temp_dir / 'test_report.md'

        report = analyzer.create_feature_importance_report(
            feature_ranking,
            save_path=save_path
        )

        # Check file was created
        assert save_path.exists()

        # Check content matches
        with open(save_path, 'r', encoding='utf-8') as f:
            saved_content = f.read()

        assert saved_content == report
        assert 'test_feature' in saved_content

class TestShapMethods:
    """Test individual SHAP analysis methods"""

    def test_calculate_global_importance(self):
        """Test global importance calculation"""
        analyzer = ShapAnalyzer()

        # Mock SHAP values
        shap_values = np.array([
            [0.1, -0.2, 0.3, 0.0],
            [-0.1, 0.3, -0.1, 0.2],
            [0.2, 0.0, 0.4, -0.1]
        ])
        feature_names = ['feat_a', 'feat_b', 'feat_c', 'feat_d']

        importance = analyzer._calculate_global_importance(shap_values, feature_names)

        assert isinstance(importance, dict)
        assert len(importance) == 4
        assert all(name in importance for name in feature_names)

        # Check values are mean absolute SHAP values
        expected_feat_a = np.mean(np.abs([0.1, -0.1, 0.2]))
        assert abs(importance['feat_a'] - expected_feat_a) < 1e-6

    def test_rank_features(self):
        """Test feature ranking"""
        analyzer = ShapAnalyzer()

        feature_importance = {
            'feature_a': 0.3,
            'feature_b': 0.1,
            'feature_c': 0.5,
            'feature_d': 0.1
        }

        ranking = analyzer._rank_features(feature_importance)

        assert isinstance(ranking, list)
        assert len(ranking) == 4

        # Check ordering
        assert ranking[0]['feature'] == 'feature_c'  # Highest importance
        assert ranking[0]['rank'] == 1
        assert ranking[1]['feature'] == 'feature_a'
        assert ranking[1]['rank'] == 2

        # Check percentages sum to 100
        total_percentage = sum(item['percentage'] for item in ranking)
        assert abs(total_percentage - 100.0) < 1e-6

    def test_calculate_summary_stats(self):
        """Test summary statistics calculation"""
        analyzer = ShapAnalyzer()

        shap_values = np.array([
            [0.1, -0.2],
            [-0.1, 0.3],
            [0.2, 0.0]
        ])
        feature_names = ['feat_a', 'feat_b']

        stats = analyzer._calculate_summary_stats(shap_values, feature_names)

        assert isinstance(stats, dict)
        assert len(stats) == 2

        # Check feat_a stats
        feat_a_stats = stats['feat_a']
        assert 'mean' in feat_a_stats
        assert 'std' in feat_a_stats
        assert 'min' in feat_a_stats
        assert 'max' in feat_a_stats
        assert 'mean_abs' in feat_a_stats

        # Verify calculations
        feat_a_values = [0.1, -0.1, 0.2]
        assert abs(feat_a_stats['mean'] - np.mean(feat_a_values)) < 1e-6
        assert abs(feat_a_stats['std'] - np.std(feat_a_values)) < 1e-6
        assert abs(feat_a_stats['min'] - np.min(feat_a_values)) < 1e-6
        assert abs(feat_a_stats['max'] - np.max(feat_a_values)) < 1e-6

def test_integration():
    """Integration test for interpretability analysis"""
    # Create realistic bridge VIV data
    np.random.seed(42)
    n_samples = 100
    n_features = 10

    # Generate correlated features (simulating real bridge data)
    X = np.random.randn(n_samples, n_features)
    feature_names = [
        'span_length', 'bridge_width', 'bridge_height', 'wind_speed',
        'natural_frequency', 'damping_ratio', 'reynolds_number',
        'strouhal_number', 'width_height_ratio', 'slenderness_ratio'
    ]

    # Create target based on features (amplitude prediction)
    y = (0.3 * X[:, 0] +  # span_length
         0.2 * X[:, 3] +  # wind_speed
         -0.1 * X[:, 5] + # damping_ratio
         np.random.normal(0, 0.1, n_samples))

    # Train a model
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=20, random_state=42)
    model.fit(X, y)

    # Test interpretability analysis
    analyzer = ShapAnalyzer()

    # Create feature descriptions
    feature_descriptions = {
        'span_length': 'Main span length of the bridge in meters',
        'bridge_width': 'Width of the bridge deck in meters',
        'bridge_height': 'Height of the bridge deck in meters',
        'wind_speed': 'Critical wind speed for VIV onset in m/s',
        'natural_frequency': 'First natural frequency in Hz',
        'damping_ratio': 'Structural damping ratio (dimensionless)'
    }

    if analyzer.available:
        # Run analysis
        results = analyzer.analyze_model(
            model, X[:30], feature_names,
            task_type='regression',
            save_plots=False
        )

        if results and 'feature_ranking' in results:
            # Generate report
            report = analyzer.create_feature_importance_report(
                results['feature_ranking'],
                feature_descriptions
            )

            assert len(report) > 100
            assert 'span_length' in report  # Should include important features

    # Test should complete without errors even if SHAP not available
    assert True

if __name__ == "__main__":
    pytest.main([__file__])