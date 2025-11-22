#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for tracking module
"""

import pytest
import tempfile
import json
from pathlib import Path
import sys
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tracking import ExperimentTracker, LocalTracker, WandBTracker, MLflowTracker

class TestLocalTracker:
    """Test cases for LocalTracker"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        # Cleanup
        if temp_path.exists():
            shutil.rmtree(temp_path)

    def test_initialization(self, temp_dir):
        """Test tracker initialization"""
        tracker = LocalTracker(str(temp_dir / 'experiments'))
        assert tracker.experiment_dir.exists()

    def test_start_experiment(self, temp_dir):
        """Test starting an experiment"""
        tracker = LocalTracker(str(temp_dir / 'experiments'))

        config = {
            'model_type': 'random_forest',
            'task_type': 'regression',
            'n_estimators': 100
        }

        experiment_id = tracker.start_experiment('test_experiment', config)

        assert experiment_id is not None
        assert len(experiment_id) > 0

        # Check experiment directory was created
        exp_path = tracker.experiment_dir / experiment_id
        assert exp_path.exists()

        # Check config was saved
        config_file = exp_path / 'config.json'
        assert config_file.exists()

        with open(config_file, 'r') as f:
            saved_config = json.load(f)

        assert saved_config['model_type'] == config['model_type']

    def test_log_metrics(self, temp_dir):
        """Test logging metrics"""
        tracker = LocalTracker(str(temp_dir / 'experiments'))

        experiment_id = tracker.start_experiment('test_experiment', {})

        metrics = {
            'mse': 0.1,
            'rmse': 0.316,
            'r2': 0.85
        }

        tracker.log_metrics(metrics, step=1)

        # Check metrics were saved
        metrics_file = tracker.experiment_dir / experiment_id / 'metrics.json'
        assert metrics_file.exists()

        with open(metrics_file, 'r') as f:
            saved_metrics = json.load(f)

        assert 'step_1' in saved_metrics
        assert saved_metrics['step_1']['mse'] == metrics['mse']

    def test_log_parameters(self, temp_dir):
        """Test logging parameters"""
        tracker = LocalTracker(str(temp_dir / 'experiments'))

        experiment_id = tracker.start_experiment('test_experiment', {})

        params = {
            'learning_rate': 0.01,
            'batch_size': 32,
            'model_name': 'test_model'
        }

        tracker.log_parameters(params)

        # Check parameters were saved
        params_file = tracker.experiment_dir / experiment_id / 'parameters.json'
        assert params_file.exists()

        with open(params_file, 'r') as f:
            saved_params = json.load(f)

        assert saved_params['learning_rate'] == params['learning_rate']
        assert saved_params['model_name'] == params['model_name']

    def test_save_model(self, temp_dir):
        """Test saving model artifacts"""
        tracker = LocalTracker(str(temp_dir / 'experiments'))

        experiment_id = tracker.start_experiment('test_experiment', {})

        # Create a dummy model file
        model_content = "dummy model content"
        model_file = temp_dir / 'test_model.pkl'
        with open(model_file, 'w') as f:
            f.write(model_content)

        tracker.save_model(str(model_file), 'test_model.pkl')

        # Check model was copied
        saved_model = tracker.experiment_dir / experiment_id / 'test_model.pkl'
        assert saved_model.exists()

        with open(saved_model, 'r') as f:
            content = f.read()

        assert content == model_content

    def test_end_experiment(self, temp_dir):
        """Test ending an experiment"""
        tracker = LocalTracker(str(temp_dir / 'experiments'))

        experiment_id = tracker.start_experiment('test_experiment', {})

        summary = {
            'status': 'completed',
            'best_metric': 0.95,
            'total_time': 120.5
        }

        tracker.end_experiment(summary)

        # Check summary was saved
        summary_file = tracker.experiment_dir / experiment_id / 'summary.json'
        assert summary_file.exists()

        with open(summary_file, 'r') as f:
            saved_summary = json.load(f)

        assert saved_summary['status'] == summary['status']
        assert saved_summary['best_metric'] == summary['best_metric']

class TestWandBTracker:
    """Test cases for WandBTracker"""

    def test_initialization_no_wandb(self):
        """Test initialization when wandb is not available"""
        # This will create a mock tracker
        tracker = WandBTracker()
        assert not tracker.available

    def test_mock_operations(self):
        """Test that mock operations don't crash"""
        tracker = WandBTracker()

        # These should not raise errors even without wandb
        experiment_id = tracker.start_experiment('test', {})
        tracker.log_metrics({'test': 1.0})
        tracker.log_parameters({'param': 'value'})
        tracker.save_model('dummy_path', 'model.pkl')
        tracker.end_experiment({'status': 'completed'})

        assert experiment_id is not None

class TestMLflowTracker:
    """Test cases for MLflowTracker"""

    def test_initialization_no_mlflow(self):
        """Test initialization when mlflow is not available"""
        tracker = MLflowTracker()
        assert not tracker.available

    def test_mock_operations(self):
        """Test that mock operations don't crash"""
        tracker = MLflowTracker()

        # These should not raise errors even without mlflow
        experiment_id = tracker.start_experiment('test', {})
        tracker.log_metrics({'test': 1.0})
        tracker.log_parameters({'param': 'value'})
        tracker.save_model('dummy_path', 'model.pkl')
        tracker.end_experiment({'status': 'completed'})

        assert experiment_id is not None

class TestExperimentTracker:
    """Test cases for ExperimentTracker factory"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        # Cleanup
        if temp_path.exists():
            shutil.rmtree(temp_path)

    def test_create_local_tracker(self, temp_dir):
        """Test creating local tracker"""
        config = {
            'tracking': {
                'backend': 'local',
                'experiment_dir': str(temp_dir / 'experiments')
            }
        }

        tracker = ExperimentTracker.create_tracker(config)
        assert isinstance(tracker, LocalTracker)

    def test_create_wandb_tracker(self):
        """Test creating wandb tracker"""
        config = {
            'tracking': {
                'backend': 'wandb',
                'project': 'test_project'
            }
        }

        tracker = ExperimentTracker.create_tracker(config)
        assert isinstance(tracker, WandBTracker)

    def test_create_mlflow_tracker(self):
        """Test creating mlflow tracker"""
        config = {
            'tracking': {
                'backend': 'mlflow',
                'tracking_uri': 'sqlite:///test.db'
            }
        }

        tracker = ExperimentTracker.create_tracker(config)
        assert isinstance(tracker, MLflowTracker)

    def test_invalid_backend(self):
        """Test invalid backend"""
        config = {
            'tracking': {
                'backend': 'invalid_backend'
            }
        }

        with pytest.raises(ValueError):
            ExperimentTracker.create_tracker(config)

    def test_default_config(self, temp_dir):
        """Test with default configuration"""
        tracker = ExperimentTracker.create_tracker({})
        assert isinstance(tracker, LocalTracker)

def test_integration(tmp_path):
    """Integration test for complete tracking workflow"""
    # Setup tracker
    config = {
        'tracking': {
            'backend': 'local',
            'experiment_dir': str(tmp_path / 'experiments')
        }
    }

    tracker = ExperimentTracker.create_tracker(config)

    # Start experiment
    exp_config = {
        'model_type': 'random_forest',
        'task_type': 'regression',
        'dataset': 'bridge_viv_test'
    }

    experiment_id = tracker.start_experiment('integration_test', exp_config)

    # Log parameters
    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    }
    tracker.log_parameters(params)

    # Log metrics over multiple steps
    for step in range(5):
        metrics = {
            'train_mse': 0.1 / (step + 1),
            'val_mse': 0.15 / (step + 1),
            'train_r2': 0.8 + step * 0.02,
            'val_r2': 0.75 + step * 0.02
        }
        tracker.log_metrics(metrics, step=step)

    # Save a dummy model
    model_file = tmp_path / 'test_model.pkl'
    with open(model_file, 'w') as f:
        f.write("dummy model for testing")

    tracker.save_model(str(model_file), 'final_model.pkl')

    # End experiment
    summary = {
        'status': 'completed',
        'best_val_r2': 0.83,
        'total_time': 45.2,
        'final_train_mse': 0.02,
        'final_val_mse': 0.03
    }

    tracker.end_experiment(summary)

    # Verify files were created
    exp_dir = tracker.experiment_dir / experiment_id
    assert exp_dir.exists()
    assert (exp_dir / 'config.json').exists()
    assert (exp_dir / 'parameters.json').exists()
    assert (exp_dir / 'metrics.json').exists()
    assert (exp_dir / 'final_model.pkl').exists()
    assert (exp_dir / 'summary.json').exists()

    # Verify content
    with open(exp_dir / 'metrics.json', 'r') as f:
        metrics_data = json.load(f)

    assert len(metrics_data) == 5  # 5 steps
    assert 'step_0' in metrics_data
    assert 'step_4' in metrics_data
    assert metrics_data['step_4']['val_r2'] > metrics_data['step_0']['val_r2']

if __name__ == "__main__":
    pytest.main([__file__])