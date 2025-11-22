#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for the complete Bridge VIV assessment system
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
import sys
import shutil
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_processing import BridgeVIVDataProcessor
from models import ModelFactory
from train import BridgeVIVTrainer
from predict import BridgeVIVPredictor
from utils import load_config, save_config

class TestSystemIntegration:
    """Integration tests for the complete system"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        # Cleanup
        if temp_path.exists():
            shutil.rmtree(temp_path)

    @pytest.fixture
    def sample_bridge_data(self, temp_dir):
        """Create sample bridge dataset"""
        np.random.seed(42)
        n_samples = 100

        # Generate realistic bridge data
        data = {
            'BridgeID': [f'BR_{i:03d}' for i in range(n_samples)],
            'BridgeName': [f'Bridge_{i}' for i in range(n_samples)],
            'BridgeType': np.random.choice(['Suspension', 'Cable-Stayed', 'Girder'], n_samples),
            'Span_m': np.random.uniform(500, 2000, n_samples),
            'Width_m': np.random.uniform(20, 50, n_samples),
            'Height_m': np.random.uniform(2, 8, n_samples),
            'Natural_Freq_Hz': np.random.uniform(0.1, 0.5, n_samples),
            'First_Freq_Hz': np.random.uniform(0.08, 0.45, n_samples),
            'Second_Freq_Hz': np.random.uniform(0.2, 1.0, n_samples),
            'VIV_Wind_Speed_ms': np.random.uniform(6, 12, n_samples),
            'Critical_Wind_Speed_ms': np.random.uniform(10, 20, n_samples),
            'Damping_Ratio': np.random.uniform(0.005, 0.02, n_samples),
        }

        # Generate amplitude based on physical relationships
        amplitude = (
            5 + 0.02 * data['Span_m'] +
            0.5 * data['VIV_Wind_Speed_ms'] +
            20 * (1 / data['Damping_Ratio'] - 50) +
            np.random.normal(0, 5, n_samples)
        )
        amplitude = np.clip(amplitude, 5, 80)

        data['Max_Amplitude_mm'] = amplitude

        # Generate risk levels based on amplitude
        risk_level = []
        for amp in amplitude:
            if amp < 20:
                risk_level.append('Low')
            elif amp < 40:
                risk_level.append('Medium')
            else:
                risk_level.append('High')

        data['Risk_Level'] = risk_level

        # Create DataFrame and save
        df = pd.DataFrame(data)
        csv_path = temp_dir / 'bridge_data.csv'
        df.to_csv(csv_path, index=False)

        return csv_path, df

    @pytest.fixture
    def test_config(self, temp_dir):
        """Create test configuration"""
        config = {
            'data': {
                'file_path': str(temp_dir / 'bridge_data.csv'),
                'target_column': 'Max_Amplitude_mm',
                'risk_column': 'Risk_Level',
                'exclude_columns': ['BridgeID', 'BridgeName']
            },
            'tasks': {
                'target_tasks': ['amplitude', 'risk_class'],
                'primary_task': 'amplitude'
            },
            'preprocessing': {
                'test_size': 0.2,
                'random_state': 42,
                'apply_scaling': True,
                'include_polynomial': False,
                'apply_boxcox': False
            },
            'models': {
                'baseline_models': ['linear', 'random_forest'],
                'advanced_models': []
            },
            'training': {
                'cv_folds': 3,
                'random_state': 42,
                'n_jobs': 1
            },
            'tracking': {
                'backend': 'local',
                'experiment_dir': str(temp_dir / 'experiments')
            }
        }

        config_path = temp_dir / 'config.yaml'
        save_config(config, config_path)

        return config_path, config

    def test_end_to_end_pipeline(self, sample_bridge_data, test_config, temp_dir):
        """Test complete end-to-end pipeline"""
        csv_path, df = sample_bridge_data
        config_path, config = test_config

        # 1. Data Processing
        processor = BridgeVIVDataProcessor(csv_path)
        processor.load_data()

        assert processor.raw_data is not None
        assert len(processor.raw_data) == 100

        # Clean and process data
        processor.clean_data()
        processor.feature_engineering()

        # Prepare ML data
        X, targets, feature_names = processor.prepare_ml_data()

        assert X is not None
        assert 'amplitude' in targets
        assert 'risk_class' in targets
        assert len(feature_names) > 0

        # Split and scale data
        datasets, feature_names = processor.split_and_scale_data(test_size=0.2)

        assert 'amplitude' in datasets
        assert 'risk_class' in datasets

        for task_name, data in datasets.items():
            assert 'X_train' in data
            assert 'X_test' in data
            assert 'y_train' in data
            assert 'y_test' in data

        # 2. Model Training
        trainer = BridgeVIVTrainer(config_path)

        # Train models
        results = trainer.train_multiple_models(['linear', 'random_forest'])

        assert isinstance(results, dict)
        assert len(results) > 0

        # Check that models were trained for each task
        for model_results in results.values():
            assert 'models' in model_results
            assert 'amplitude' in model_results['models']

        # 3. Model Prediction
        # Save a model for prediction testing
        best_model_info = None
        for model_name, model_results in results.items():
            if 'amplitude' in model_results['models']:
                model_obj = model_results['models']['amplitude']
                model_path = temp_dir / f'{model_name}_amplitude_model.pkl'

                # Save model
                import joblib
                joblib.dump(model_obj, model_path)

                best_model_info = {
                    'path': model_path,
                    'model': model_obj
                }
                break

        assert best_model_info is not None

        # Test prediction
        predictor = BridgeVIVPredictor(best_model_info['path'])

        # Create test data
        test_data = df.iloc[:5].copy()
        predictions = predictor.predict(test_data)

        assert len(predictions) == 5
        assert all(pred > 0 for pred in predictions)  # Amplitude should be positive

        # Test prediction with confidence
        pred_results = predictor.predict_with_confidence(test_data)

        assert 'predictions' in pred_results
        assert len(pred_results['predictions']) == 5

    def test_data_processing_pipeline(self, sample_bridge_data, temp_dir):
        """Test complete data processing pipeline"""
        csv_path, df = sample_bridge_data

        processor = BridgeVIVDataProcessor(csv_path)

        # Test complete pipeline
        processor.load_data()
        exploration_results = processor.explore_data()

        assert 'shape' in exploration_results
        assert 'columns' in exploration_results
        assert exploration_results['shape'][0] == 100

        # Test cleaning
        cleaned_data = processor.clean_data()
        assert cleaned_data is not None

        # Test feature engineering
        fe_data = processor.feature_engineering(
            include_polynomial=True,
            include_interactions=False,
            apply_boxcox=False
        )

        assert fe_data is not None
        assert fe_data.shape[1] > cleaned_data.shape[1]  # More features

        # Test ML preparation
        X, targets, feature_names = processor.prepare_ml_data()

        assert X.shape[0] == len(df)
        assert 'amplitude' in targets
        assert 'risk_class' in targets

        # Test split and scale
        datasets, feature_names = processor.split_and_scale_data()

        for task_name, data in datasets.items():
            train_size = len(data['y_train'])
            test_size = len(data['y_test'])
            total_size = train_size + test_size

            # Check split ratio is approximately correct
            assert 0.75 <= train_size / total_size <= 0.85

    def test_model_training_pipeline(self, sample_bridge_data, test_config, temp_dir):
        """Test model training pipeline"""
        csv_path, df = sample_bridge_data
        config_path, config = test_config

        trainer = BridgeVIVTrainer(config_path)

        # Test single model training
        model_type = 'linear'
        results = trainer.train_single_model(model_type)

        assert model_type in results
        assert 'models' in results[model_type]
        assert 'metrics' in results[model_type]

        # Check that models were trained for each task
        models = results[model_type]['models']
        metrics = results[model_type]['metrics']

        assert 'amplitude' in models
        assert 'amplitude' in metrics

        # Check model has required methods
        amplitude_model = models['amplitude']
        assert hasattr(amplitude_model, 'predict')
        assert hasattr(amplitude_model, 'fit')

        # Check metrics structure
        amplitude_metrics = metrics['amplitude']
        assert 'train' in amplitude_metrics
        assert 'test' in amplitude_metrics

        train_metrics = amplitude_metrics['train']
        test_metrics = amplitude_metrics['test']

        # For regression, should have MSE, RMSE, MAE, R2
        for metric_set in [train_metrics, test_metrics]:
            assert 'mse' in metric_set
            assert 'rmse' in metric_set
            assert 'mae' in metric_set
            assert 'r2' in metric_set

    def test_configuration_handling(self, temp_dir):
        """Test configuration loading and validation"""
        # Create minimal config
        config = {
            'data': {
                'file_path': 'test.csv'
            },
            'tasks': {
                'target_tasks': ['amplitude']
            }
        }

        config_path = temp_dir / 'test_config.yaml'
        save_config(config, config_path)

        # Test loading
        loaded_config = load_config(config_path)

        assert loaded_config['data']['file_path'] == 'test.csv'
        assert loaded_config['tasks']['target_tasks'] == ['amplitude']

    def test_model_factory_integration(self):
        """Test model factory with all model types"""
        # Test regression models
        regression_models = ['linear', 'random_forest']

        for model_type in regression_models:
            try:
                model = ModelFactory.create_model(model_type, 'regression')
                assert model is not None
                assert hasattr(model, 'fit')
                assert hasattr(model, 'predict')
            except ImportError:
                # Skip if dependencies not available
                continue

        # Test classification models
        classification_models = ['linear', 'random_forest']

        for model_type in classification_models:
            try:
                model = ModelFactory.create_model(model_type, 'binary_classification')
                assert model is not None
                assert hasattr(model, 'fit')
                assert hasattr(model, 'predict')
            except ImportError:
                # Skip if dependencies not available
                continue

    def test_error_handling(self, temp_dir):
        """Test error handling throughout the system"""
        # Test invalid data file
        with pytest.raises(FileNotFoundError):
            processor = BridgeVIVDataProcessor('nonexistent.csv')
            processor.load_data()

        # Test invalid model type
        with pytest.raises(ValueError):
            ModelFactory.create_model('invalid_model', 'regression')

        # Test invalid model path for predictor
        with pytest.raises(FileNotFoundError):
            BridgeVIVPredictor('nonexistent_model.pkl')

    def test_memory_efficiency(self, sample_bridge_data, test_config, temp_dir):
        """Test that the system handles data efficiently"""
        csv_path, df = sample_bridge_data
        config_path, config = test_config

        # Process data in chunks to test memory efficiency
        processor = BridgeVIVDataProcessor(csv_path)
        processor.load_data()

        # Test that data processing doesn't create excessive copies
        initial_columns = len(processor.raw_data.columns)

        processor.clean_data()
        processor.feature_engineering()

        # Should not exponentially increase memory usage
        final_data = processor.processed_data
        assert final_data is not None

        # Memory usage should be reasonable (not more than 10x original)
        assert len(final_data.columns) < initial_columns * 10

def test_system_robustness():
    """Test system robustness with edge cases"""
    # Test with minimal data
    minimal_data = {
        'Span_m': [1000, 1200],
        'Width_m': [30, 35],
        'Height_m': [3.0, 3.5],
        'Max_Amplitude_mm': [25, 45],
        'Risk_Level': ['Medium', 'High']
    }

    df = pd.DataFrame(minimal_data)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_file = f.name

    try:
        processor = BridgeVIVDataProcessor(temp_file)
        processor.load_data()

        # Should handle minimal data gracefully
        assert len(processor.raw_data) == 2

        # Try processing - should not crash
        processor.clean_data()
        processor.feature_engineering()

        X, targets, feature_names = processor.prepare_ml_data()

        # Should produce valid output even with minimal data
        assert X is not None
        assert len(feature_names) > 0

    finally:
        Path(temp_file).unlink()

if __name__ == "__main__":
    pytest.main([__file__])