#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment Tracking for Bridge VIV Risk Assessment
Supports MLflow and Weights & Biases
"""

import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import json
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseTracker(ABC):
    """Base class for experiment trackers"""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.current_run = None

    @abstractmethod
    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start a new experiment run"""
        pass

    @abstractmethod
    def end_run(self) -> None:
        """End the current experiment run"""
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters"""
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics"""
        pass

    @abstractmethod
    def log_artifact(self, artifact_path: Union[str, Path]) -> None:
        """Log artifact file"""
        pass

class MLflowTracker(BaseTracker):
    """MLflow experiment tracker"""

    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        super().__init__(experiment_name)

        try:
            import mlflow
            self.mlflow = mlflow
            self.available = True
        except ImportError:
            logger.warning("MLflow not available. Install with: pip install mlflow")
            self.available = False
            return

        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created MLflow experiment: {experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})")

            mlflow.set_experiment(experiment_name)
            self.experiment_id = experiment_id

        except Exception as e:
            logger.error(f"Failed to setup MLflow experiment: {e}")
            self.available = False

    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start a new MLflow run"""
        if not self.available:
            return

        try:
            self.current_run = self.mlflow.start_run(run_name=run_name)
            logger.info(f"Started MLflow run: {run_name or self.current_run.info.run_id}")
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")

    def end_run(self) -> None:
        """End the current MLflow run"""
        if not self.available or self.current_run is None:
            return

        try:
            self.mlflow.end_run()
            logger.info("Ended MLflow run")
            self.current_run = None
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {e}")

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow"""
        if not self.available or self.current_run is None:
            return

        try:
            # Convert values to strings (MLflow requirement)
            str_params = {k: str(v) for k, v in params.items()}
            self.mlflow.log_params(str_params)
        except Exception as e:
            logger.error(f"Failed to log params to MLflow: {e}")

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics to MLflow"""
        if not self.available or self.current_run is None:
            return

        try:
            # Filter out non-numeric values
            numeric_metrics = {k: v for k, v in metrics.items()
                             if isinstance(v, (int, float)) and not (isinstance(v, float) and
                             (v != v or v == float('inf') or v == float('-inf')))}  # Check for NaN/inf

            if numeric_metrics:
                self.mlflow.log_metrics(numeric_metrics)
        except Exception as e:
            logger.error(f"Failed to log metrics to MLflow: {e}")

    def log_artifact(self, artifact_path: Union[str, Path]) -> None:
        """Log artifact to MLflow"""
        if not self.available or self.current_run is None:
            return

        try:
            self.mlflow.log_artifact(str(artifact_path))
        except Exception as e:
            logger.error(f"Failed to log artifact to MLflow: {e}")

    def log_model(self, model, artifact_path: str = "model") -> None:
        """Log model to MLflow"""
        if not self.available or self.current_run is None:
            return

        try:
            # Try to log as sklearn model first
            if hasattr(model, 'fit') and hasattr(model, 'predict'):
                self.mlflow.sklearn.log_model(model, artifact_path)
            else:
                # Fallback to pickle
                import pickle
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                    pickle.dump(model, f)
                    self.mlflow.log_artifact(f.name, artifact_path)
        except Exception as e:
            logger.error(f"Failed to log model to MLflow: {e}")

class WandBTracker(BaseTracker):
    """Weights & Biases experiment tracker"""

    def __init__(self, experiment_name: str, project: str, entity: Optional[str] = None):
        super().__init__(experiment_name)
        self.project = project
        self.entity = entity

        try:
            import wandb
            self.wandb = wandb
            self.available = True
        except ImportError:
            logger.warning("Weights & Biases not available. Install with: pip install wandb")
            self.available = False

    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start a new W&B run"""
        if not self.available:
            return

        try:
            self.current_run = self.wandb.init(
                project=self.project,
                entity=self.entity,
                name=run_name,
                reinit=True
            )
            logger.info(f"Started W&B run: {run_name or 'unnamed'}")
        except Exception as e:
            logger.error(f"Failed to start W&B run: {e}")

    def end_run(self) -> None:
        """End the current W&B run"""
        if not self.available or self.current_run is None:
            return

        try:
            self.wandb.finish()
            logger.info("Ended W&B run")
            self.current_run = None
        except Exception as e:
            logger.error(f"Failed to end W&B run: {e}")

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to W&B"""
        if not self.available or self.current_run is None:
            return

        try:
            self.wandb.config.update(params)
        except Exception as e:
            logger.error(f"Failed to log params to W&B: {e}")

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics to W&B"""
        if not self.available or self.current_run is None:
            return

        try:
            # Filter out non-numeric values
            numeric_metrics = {k: v for k, v in metrics.items()
                             if isinstance(v, (int, float)) and not (isinstance(v, float) and
                             (v != v or v == float('inf') or v == float('-inf')))}

            if numeric_metrics:
                self.wandb.log(numeric_metrics)
        except Exception as e:
            logger.error(f"Failed to log metrics to W&B: {e}")

    def log_artifact(self, artifact_path: Union[str, Path]) -> None:
        """Log artifact to W&B"""
        if not self.available or self.current_run is None:
            return

        try:
            artifact = self.wandb.Artifact(name=Path(artifact_path).stem, type='result')
            artifact.add_file(str(artifact_path))
            self.wandb.log_artifact(artifact)
        except Exception as e:
            logger.error(f"Failed to log artifact to W&B: {e}")

class LocalTracker(BaseTracker):
    """Local file-based experiment tracker"""

    def __init__(self, experiment_name: str, output_dir: str = "experiments"):
        super().__init__(experiment_name)
        self.output_dir = Path(output_dir)
        self.experiment_dir = self.output_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.available = True

        self.run_data = {}
        self.run_dir = None

    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start a new local run"""
        if run_name is None:
            run_name = f"run_{int(time.time())}"

        self.run_dir = self.experiment_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.run_data = {
            'run_name': run_name,
            'start_time': time.time(),
            'params': {},
            'metrics': {},
            'artifacts': []
        }

        logger.info(f"Started local run: {run_name} at {self.run_dir}")

    def end_run(self) -> None:
        """End the current local run"""
        if self.run_dir is None:
            return

        try:
            self.run_data['end_time'] = time.time()
            self.run_data['duration'] = self.run_data['end_time'] - self.run_data['start_time']

            # Save run data
            run_file = self.run_dir / 'run_data.json'
            with open(run_file, 'w') as f:
                json.dump(self.run_data, f, indent=2, default=str)

            logger.info(f"Ended local run: {self.run_data['run_name']}")
            self.run_dir = None

        except Exception as e:
            logger.error(f"Failed to end local run: {e}")

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters locally"""
        if self.run_dir is None:
            return

        self.run_data['params'].update(params)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics locally"""
        if self.run_dir is None:
            return

        # Filter numeric metrics
        numeric_metrics = {k: v for k, v in metrics.items()
                         if isinstance(v, (int, float)) and not (isinstance(v, float) and
                         (v != v or v == float('inf') or v == float('-inf')))}

        self.run_data['metrics'].update(numeric_metrics)

    def log_artifact(self, artifact_path: Union[str, Path]) -> None:
        """Log artifact locally (copy to run directory)"""
        if self.run_dir is None:
            return

        try:
            import shutil
            artifact_path = Path(artifact_path)

            if artifact_path.exists():
                dest_path = self.run_dir / artifact_path.name
                shutil.copy2(artifact_path, dest_path)
                self.run_data['artifacts'].append(str(dest_path))
        except Exception as e:
            logger.error(f"Failed to log artifact locally: {e}")

class ExperimentTracker:
    """Main experiment tracker that can use multiple backends"""

    def __init__(self,
                 backend: str = 'mlflow',
                 experiment_name: str = 'bridge_viv_experiment',
                 tracking_uri: Optional[str] = None,
                 project: Optional[str] = None,
                 entity: Optional[str] = None,
                 output_dir: str = "experiments"):
        """
        Initialize experiment tracker

        Args:
            backend: Tracking backend ('mlflow', 'wandb', 'local')
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking URI
            project: W&B project name
            entity: W&B entity name
            output_dir: Local output directory
        """
        self.backend = backend
        self.trackers: List[BaseTracker] = []

        # Initialize trackers based on backend
        if backend == 'mlflow' or backend == 'all':
            mlflow_tracker = MLflowTracker(experiment_name, tracking_uri)
            if mlflow_tracker.available:
                self.trackers.append(mlflow_tracker)

        if backend == 'wandb' or backend == 'all':
            if project:
                wandb_tracker = WandBTracker(experiment_name, project, entity)
                if wandb_tracker.available:
                    self.trackers.append(wandb_tracker)

        if backend == 'local' or backend == 'all' or not self.trackers:
            # Always have local tracker as fallback
            local_tracker = LocalTracker(experiment_name, output_dir)
            self.trackers.append(local_tracker)

        logger.info(f"Initialized experiment tracking with {len(self.trackers)} trackers")

    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start experiment run on all trackers"""
        for tracker in self.trackers:
            tracker.start_run(run_name)

    def end_run(self) -> None:
        """End experiment run on all trackers"""
        for tracker in self.trackers:
            tracker.end_run()

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to all trackers"""
        for tracker in self.trackers:
            tracker.log_params(params)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics to all trackers"""
        for tracker in self.trackers:
            tracker.log_metrics(metrics)

    def log_artifact(self, artifact_path: Union[str, Path]) -> None:
        """Log artifact to all trackers"""
        for tracker in self.trackers:
            tracker.log_artifact(artifact_path)

    def log_model(self, model, artifact_path: str = "model") -> None:
        """Log model to supported trackers"""
        for tracker in self.trackers:
            if hasattr(tracker, 'log_model'):
                tracker.log_model(model, artifact_path)

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.end_run()

class ExperimentComparison:
    """Compare experiments across runs"""

    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri

    def get_runs_dataframe(self) -> Optional['pd.DataFrame']:
        """Get all runs as a DataFrame (MLflow only)"""
        try:
            import mlflow
            import pandas as pd

            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)

            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                logger.warning(f"Experiment {self.experiment_name} not found")
                return None

            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            return runs

        except ImportError:
            logger.warning("MLflow not available for experiment comparison")
            return None
        except Exception as e:
            logger.error(f"Failed to get runs DataFrame: {e}")
            return None

    def get_best_run(self, metric_name: str, mode: str = 'max') -> Optional[Dict[str, Any]]:
        """Get the best run based on a metric"""
        runs_df = self.get_runs_dataframe()

        if runs_df is None or len(runs_df) == 0:
            return None

        metric_col = f"metrics.{metric_name}"
        if metric_col not in runs_df.columns:
            logger.warning(f"Metric {metric_name} not found in runs")
            return None

        if mode == 'max':
            best_run = runs_df.loc[runs_df[metric_col].idxmax()]
        else:
            best_run = runs_df.loc[runs_df[metric_col].idxmin()]

        return best_run.to_dict()

    def compare_runs(self, metric_names: List[str]) -> Optional['pd.DataFrame']:
        """Compare runs on specified metrics"""
        runs_df = self.get_runs_dataframe()

        if runs_df is None:
            return None

        # Select relevant columns
        columns = ['run_id', 'start_time', 'status']

        # Add metric columns
        for metric in metric_names:
            metric_col = f"metrics.{metric}"
            if metric_col in runs_df.columns:
                columns.append(metric_col)

        # Add some parameter columns
        param_cols = [col for col in runs_df.columns if col.startswith('params.')]
        columns.extend(param_cols[:5])  # Limit to first 5 params

        comparison_df = runs_df[columns].copy()
        comparison_df = comparison_df.sort_values('start_time', ascending=False)

        return comparison_df

def main():
    """Test experiment tracking"""
    # Test local tracker
    tracker = ExperimentTracker(backend='local', experiment_name='test_experiment')

    tracker.start_run('test_run')

    # Log some params and metrics
    tracker.log_params({
        'model': 'test_model',
        'learning_rate': 0.01,
        'batch_size': 32
    })

    tracker.log_metrics({
        'accuracy': 0.95,
        'loss': 0.05,
        'f1_score': 0.93
    })

    tracker.end_run()

    print("Experiment tracking test completed!")

if __name__ == "__main__":
    main()