#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Script for Bridge VIV Risk Assessment
Supports multiple models, tasks, and evaluation metrics
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings

import pandas as pd
import numpy as np
import yaml
import logging
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import make_scorer
import joblib

# Import our modules
from data_processing import BridgeVIVDataProcessor
from features import BridgeFeatureEngineer
from models import ModelFactory, BaseModel
from utils import setup_logging, load_config, create_output_directories
from metrics import get_scorer, evaluate_model, plot_results
from tracking import ExperimentTracker

warnings.filterwarnings('ignore')

class BridgeVIVTrainer:
    """Main trainer class for Bridge VIV models"""

    def __init__(self, config_path: str):
        """
        Initialize trainer

        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.setup_logging()
        self.create_directories()

        # Initialize components
        self.data_processor = None
        self.feature_engineer = None
        self.tracker = None
        self.results = {}

        logger.info(f"BridgeVIVTrainer initialized with config: {config_path}")

    def setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})
        setup_logging(
            level=log_config.get('level', 'INFO'),
            log_file=log_config.get('log_file'),
            console=log_config.get('console', True)
        )

    def create_directories(self):
        """Create output directories"""
        output_config = self.config.get('output', {})
        create_output_directories([
            output_config.get('results_dir', 'results'),
            output_config.get('experiments_dir', 'experiments'),
            'logs'
        ])

    def load_and_process_data(self) -> Tuple[Dict[str, Dict], List[str]]:
        """
        Load and process data

        Returns:
            Tuple of (datasets, feature_names)
        """
        logger.info("Loading and processing data...")

        # Initialize data processor
        data_config = self.config['data']
        preprocessing_config = self.config['preprocessing']

        self.data_processor = BridgeVIVDataProcessor(
            data_path=data_config['dataset_path'],
            random_state=data_config['random_state']
        )

        # Load data
        self.data_processor.load_data()

        # Explore data
        exploration_results = self.data_processor.explore_data()
        logger.info(f"Dataset shape: {exploration_results['shape']}")

        # Clean data
        self.data_processor.clean_data(
            outlier_method=preprocessing_config.get('outlier_method', 'iqr'),
            outlier_threshold=preprocessing_config.get('outlier_threshold', 1.5)
        )

        # Feature engineering
        feature_config = self.config.get('feature_engineering', {})
        self.data_processor.feature_engineering(
            include_polynomial=feature_config.get('include_polynomial', False),
            include_interactions=feature_config.get('include_interactions', False),
            apply_boxcox=feature_config.get('apply_boxcox', False),
            polynomial_degree=feature_config.get('polynomial_degree', 2)
        )

        # Split and scale data
        tasks_config = self.config['tasks']
        datasets, feature_names = self.data_processor.split_and_scale_data(
            test_size=preprocessing_config.get('test_size', 0.2),
            val_size=preprocessing_config.get('val_size', 0.2),
            scaler_type=preprocessing_config.get('scaler_type', 'standard'),
            apply_smote=preprocessing_config.get('apply_smote', True),
            target_tasks=tasks_config.get('target_tasks', ['amplitude'])
        )

        logger.info(f"Data processing completed. Features: {len(feature_names)}")
        return datasets, feature_names

    def setup_experiment_tracking(self):
        """Setup experiment tracking"""
        tracking_config = self.config.get('tracking', {})

        if tracking_config.get('enabled', True):
            self.tracker = ExperimentTracker(
                backend=tracking_config.get('backend', 'mlflow'),
                experiment_name=tracking_config.get('experiment_name', 'bridge_viv'),
                tracking_uri=tracking_config.get('mlflow_uri', 'file:./experiments/mlflow')
            )

    def train_single_model(self,
                          model: BaseModel,
                          model_name: str,
                          task_name: str,
                          datasets: Dict[str, Dict],
                          feature_names: List[str]) -> Dict[str, Any]:
        """
        Train a single model for a specific task

        Args:
            model: Model instance
            model_name: Name of the model
            task_name: Name of the task
            datasets: Dataset splits
            feature_names: List of feature names

        Returns:
            Dictionary with training results
        """
        logger.info(f"Training {model_name} for {task_name}...")

        task_data = datasets[task_name]
        X_train = task_data['X_train']
        y_train = task_data['y_train']
        X_val = task_data['X_val']
        y_val = task_data['y_val']
        X_test = task_data['X_test']
        y_test = task_data['y_test']

        # Start experiment run
        if self.tracker:
            run_name = f"{model_name}_{task_name}"
            self.tracker.start_run(run_name)

        # Train model
        start_time = time.time()

        if hasattr(model, 'fit') and 'X_val' in model.fit.__code__.co_varnames:
            # Model supports validation data (e.g., XGBoost, LightGBM)
            model.fit(X_train.values, y_train.values, X_val.values, y_val.values)
        else:
            model.fit(X_train.values, y_train.values)

        training_time = time.time() - start_time

        # Make predictions
        y_train_pred = model.predict(X_train.values)
        y_val_pred = model.predict(X_val.values)
        y_test_pred = model.predict(X_test.values)

        # Evaluate model
        task_type = self._get_task_type(task_name)
        train_metrics = evaluate_model(y_train.values, y_train_pred, task_type)
        val_metrics = evaluate_model(y_val.values, y_val_pred, task_type)
        test_metrics = evaluate_model(y_test.values, y_test_pred, task_type)

        # Cross-validation
        cv_scores = self._perform_cross_validation(
            model, X_train.values, y_train.values, task_type
        )

        # Collect results
        results = {
            'model_name': model_name,
            'task_name': task_name,
            'task_type': task_type,
            'training_time': training_time,
            'prediction_time': getattr(model, 'prediction_time', 0.0),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'cv_scores': cv_scores,
            'n_features': len(feature_names),
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'predictions': {
                'y_test_true': y_test.values,
                'y_test_pred': y_test_pred
            }
        }

        # Log to experiment tracker
        if self.tracker:
            self._log_results_to_tracker(results, model, feature_names)

        # Save model
        output_config = self.config.get('output', {})
        if output_config.get('save_models', True):
            model_path = Path(output_config.get('experiments_dir', 'experiments')) / f"{model_name}_{task_name}_model.pkl"
            model.save_model(model_path)
            results['model_path'] = str(model_path)

        # Generate plots
        if output_config.get('save_plots', True):
            self._generate_plots(results, model, feature_names)

        if self.tracker:
            self.tracker.end_run()

        logger.info(f"Completed training {model_name} for {task_name}")
        return results

    def train_models(self,
                    datasets: Dict[str, Dict],
                    feature_names: List[str],
                    model_names: Optional[List[str]] = None,
                    task_names: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Train multiple models for multiple tasks

        Args:
            datasets: Dataset splits
            feature_names: List of feature names
            model_names: List of model names to train
            task_names: List of task names to train

        Returns:
            Dictionary with all training results
        """
        if model_names is None:
            model_names = self.config['models']['baseline_models']

        if task_names is None:
            task_names = list(datasets.keys())

        all_results = {}

        for task_name in task_names:
            if task_name not in datasets:
                logger.warning(f"Task {task_name} not found in datasets")
                continue

            all_results[task_name] = {}
            task_type = self._get_task_type(task_name)

            for model_name in model_names:
                try:
                    # Get model parameters
                    model_params = self.config['models'].get('model_params', {}).get(model_name, {})

                    # Add input dimension for neural networks
                    if model_name == 'mlp':
                        model_params['input_dim'] = len(feature_names)

                    # Create model
                    model = ModelFactory.create_model(
                        model_type=model_name,
                        task_type=task_type,
                        random_state=self.config['data']['random_state'],
                        **model_params
                    )

                    # Train model
                    results = self.train_single_model(
                        model, model_name, task_name, datasets, feature_names
                    )

                    all_results[task_name][model_name] = results

                except Exception as e:
                    logger.error(f"Failed to train {model_name} for {task_name}: {e}")
                    continue

        return all_results

    def _get_task_type(self, task_name: str) -> str:
        """Get task type from task name"""
        if task_name == 'amplitude':
            return 'regression'
        elif task_name == 'viv_occurrence':
            return 'binary_classification'
        elif task_name == 'risk_class':
            return 'multiclass_classification'
        else:
            return 'regression'  # default

    def _perform_cross_validation(self,
                                 model: BaseModel,
                                 X: np.ndarray,
                                 y: np.ndarray,
                                 task_type: str) -> Dict[str, float]:
        """Perform cross-validation"""
        cv_config = self.config.get('cross_validation', {})
        cv_folds = self.data_processor.get_cv_folds(
            n_splits=cv_config.get('n_splits', 5),
            task_name=task_type
        )

        # Get appropriate scorer
        if task_type == 'regression':
            scorers = {
                'neg_mean_absolute_error': 'neg_mean_absolute_error',
                'neg_root_mean_squared_error': 'neg_root_mean_squared_error',
                'r2': 'r2'
            }
        else:
            scorers = {
                'accuracy': 'accuracy',
                'f1': 'f1_macro',
                'precision': 'precision_macro',
                'recall': 'recall_macro'
            }

        try:
            cv_results = cross_validate(
                model.model,  # Use the underlying sklearn/xgb model
                X, y,
                cv=cv_folds,
                scoring=scorers,
                return_train_score=True,
                n_jobs=1  # Avoid nested parallelism
            )

            # Process results
            cv_scores = {}
            for metric in scorers.keys():
                cv_scores[f'{metric}_mean'] = cv_results[f'test_{metric}'].mean()
                cv_scores[f'{metric}_std'] = cv_results[f'test_{metric}'].std()

            return cv_scores

        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            return {}

    def _log_results_to_tracker(self,
                               results: Dict[str, Any],
                               model: BaseModel,
                               feature_names: List[str]):
        """Log results to experiment tracker"""
        if not self.tracker:
            return

        # Log parameters
        params = {
            'model_name': results['model_name'],
            'task_name': results['task_name'],
            'task_type': results['task_type'],
            'n_features': results['n_features'],
            'train_size': results['train_size'],
            'random_state': self.config['data']['random_state']
        }

        # Add model-specific parameters
        model_params = self.config['models'].get('model_params', {}).get(results['model_name'], {})
        params.update({f"model_{k}": v for k, v in model_params.items()})

        self.tracker.log_params(params)

        # Log metrics
        metrics = {}
        for split in ['train', 'val', 'test']:
            split_metrics = results[f'{split}_metrics']
            for metric, value in split_metrics.items():
                metrics[f'{split}_{metric}'] = value

        # Add CV metrics
        for metric, value in results['cv_scores'].items():
            metrics[f'cv_{metric}'] = value

        # Add timing metrics
        metrics['training_time'] = results['training_time']
        metrics['prediction_time'] = results['prediction_time']

        self.tracker.log_metrics(metrics)

    def _generate_plots(self,
                       results: Dict[str, Any],
                       model: BaseModel,
                       feature_names: List[str]):
        """Generate and save plots"""
        output_config = self.config.get('output', {})
        results_dir = Path(output_config.get('results_dir', 'results'))
        plots_dir = results_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)

        model_name = results['model_name']
        task_name = results['task_name']
        task_type = results['task_type']

        try:
            # Plot results
            plot_results(
                y_true=results['predictions']['y_test_true'],
                y_pred=results['predictions']['y_test_pred'],
                task_type=task_type,
                save_path=plots_dir / f"{model_name}_{task_name}_results.png",
                title=f"{model_name} - {task_name}"
            )

            # Feature importance
            if hasattr(model, 'get_feature_importance'):
                importance = model.get_feature_importance()
                self._plot_feature_importance(
                    importance, feature_names,
                    plots_dir / f"{model_name}_{task_name}_importance.png"
                )

        except Exception as e:
            logger.warning(f"Failed to generate plots for {model_name}_{task_name}: {e}")

    def _plot_feature_importance(self,
                                importance: np.ndarray,
                                feature_names: List[str],
                                save_path: Path):
        """Plot feature importance"""
        import matplotlib.pyplot as plt

        # Sort features by importance
        indices = np.argsort(importance)[::-1]
        top_n = min(20, len(feature_names))  # Show top 20 features

        plt.figure(figsize=(10, 8))
        plt.title("Feature Importance")
        plt.bar(range(top_n), importance[indices[:top_n]])
        plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_results(self, results: Dict[str, Dict]):
        """Save results to files"""
        output_config = self.config.get('output', {})
        results_dir = Path(output_config.get('results_dir', 'results'))

        # Save detailed results
        if output_config.get('save_metrics', True):
            results_file = results_dir / 'training_results.pkl'
            joblib.dump(results, results_file)
            logger.info(f"Results saved to {results_file}")

        # Create summary
        summary = self._create_results_summary(results)

        # Save summary as CSV
        summary_df = pd.DataFrame(summary)
        summary_file = results_dir / 'results_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Summary saved to {summary_file}")

        # Generate markdown report
        if output_config.get('generate_report', True):
            self._generate_markdown_report(summary_df, results_dir / 'results_summary.md')

    def _create_results_summary(self, results: Dict[str, Dict]) -> List[Dict]:
        """Create a summary of results"""
        summary = []

        for task_name, task_results in results.items():
            for model_name, model_results in task_results.items():
                summary_row = {
                    'task': task_name,
                    'model': model_name,
                    'task_type': model_results['task_type'],
                    'training_time': model_results['training_time'],
                    'prediction_time_ms': model_results['prediction_time'],
                    'n_features': model_results['n_features'],
                    'train_size': model_results['train_size'],
                    'test_size': model_results['test_size']
                }

                # Add test metrics
                for metric, value in model_results['test_metrics'].items():
                    summary_row[f'test_{metric}'] = value

                # Add CV metrics (mean only)
                for metric, value in model_results['cv_scores'].items():
                    if metric.endswith('_mean'):
                        summary_row[f'cv_{metric[:-5]}'] = value

                summary.append(summary_row)

        return summary

    def _generate_markdown_report(self, summary_df: pd.DataFrame, save_path: Path):
        """Generate markdown report"""
        with open(save_path, 'w') as f:
            f.write("# Bridge VIV Risk Assessment - Training Results\n\n")
            f.write(f"Generated at: {pd.Timestamp.now()}\n\n")

            # Overall summary
            f.write("## Overall Summary\n\n")
            f.write(f"- Total models trained: {len(summary_df)}\n")
            f.write(f"- Tasks: {', '.join(summary_df['task'].unique())}\n")
            f.write(f"- Models: {', '.join(summary_df['model'].unique())}\n\n")

            # Best models per task
            f.write("## Best Models per Task\n\n")
            for task in summary_df['task'].unique():
                task_data = summary_df[summary_df['task'] == task]

                if task == 'amplitude':
                    best_model = task_data.loc[task_data['test_rmse'].idxmin()]
                    metric = 'RMSE'
                    value = best_model['test_rmse']
                else:
                    best_model = task_data.loc[task_data['test_f1'].idxmax()]
                    metric = 'F1'
                    value = best_model['test_f1']

                f.write(f"### {task.title()}\n")
                f.write(f"- **Best Model**: {best_model['model']}\n")
                f.write(f"- **{metric}**: {value:.4f}\n")
                f.write(f"- **Training Time**: {best_model['training_time']:.2f}s\n\n")

            # Detailed results
            f.write("## Detailed Results\n\n")
            for task in summary_df['task'].unique():
                f.write(f"### {task.title()}\n\n")
                task_data = summary_df[summary_df['task'] == task]

                f.write("| Model | ")
                if task == 'amplitude':
                    f.write("RMSE | MAE | R² |")
                else:
                    f.write("Accuracy | F1 | Precision | Recall |")
                f.write(" Training Time |\n")

                f.write("|-------|")
                if task == 'amplitude':
                    f.write("------|-----|-------|")
                else:
                    f.write("----------|----|-----------|---------")
                f.write("---------------|\n")

                for _, row in task_data.iterrows():
                    f.write(f"| {row['model']} | ")
                    if task == 'amplitude':
                        f.write(f"{row['test_rmse']:.4f} | {row['test_mae']:.4f} | {row['test_r2']:.4f} |")
                    else:
                        f.write(f"{row['test_accuracy']:.4f} | {row['test_f1']:.4f} | {row['test_precision']:.4f} | {row['test_recall']:.4f} |")
                    f.write(f" {row['training_time']:.2f}s |\n")

                f.write("\n")

        logger.info(f"Markdown report saved to {save_path}")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Bridge VIV Risk Assessment Models")
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--task', type=str, choices=['amplitude', 'risk_class', 'viv_occurrence', 'all'],
                       help='Specific task to train (overrides config)')
    parser.add_argument('--model', type=str,
                       help='Specific model to train (overrides config)')
    parser.add_argument('--mode', type=str, choices=['baseline', 'full'], default='baseline',
                       help='Training mode: baseline or full experiments')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate configuration and data without training')

    args = parser.parse_args()

    try:
        # Initialize trainer
        trainer = BridgeVIVTrainer(args.config)

        # Override config with command line arguments
        if args.verbose:
            trainer.config['development']['verbose'] = True
        if args.dry_run:
            trainer.config['development']['dry_run'] = True

        # Setup experiment tracking
        trainer.setup_experiment_tracking()

        # Load and process data
        datasets, feature_names = trainer.load_and_process_data()

        if args.dry_run:
            logger.info("Dry run completed successfully. Configuration and data are valid.")
            return

        # Determine models and tasks to train
        if args.model:
            model_names = [args.model]
        else:
            model_names = trainer.config['models']['baseline_models']

        if args.task:
            if args.task == 'all':
                task_names = list(datasets.keys())
            else:
                task_names = [args.task]
        else:
            task_names = list(datasets.keys())

        # Train models
        logger.info(f"Starting training: {len(model_names)} models × {len(task_names)} tasks")
        results = trainer.train_models(datasets, feature_names, model_names, task_names)

        # Save results
        trainer.save_results(results)

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()