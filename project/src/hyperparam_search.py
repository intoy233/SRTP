#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperparameter Search for Bridge VIV Risk Assessment
Supports Optuna, Grid Search, and Random Search
"""

import argparse
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import json

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import cross_val_score, ParameterGrid
from sklearn.metrics import make_scorer

# Import our modules
from data_processing import BridgeVIVDataProcessor
from models import ModelFactory, BaseModel
from metrics import get_scorer, evaluate_model
from tracking import ExperimentTracker
from utils import load_config, setup_logging, create_output_directories

logger = logging.getLogger(__name__)

class HyperparameterSearcher:
    """Hyperparameter optimization for bridge VIV models"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize hyperparameter searcher

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.hyperopt_config = config.get('hyperopt', {})
        self.data_processor = None
        self.tracker = None
        self.best_params = {}
        self.best_scores = {}
        self.search_history = []

    def setup_data(self) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Setup data for hyperparameter search

        Returns:
            Tuple of (X, y, task_type)
        """
        # Initialize data processor
        data_config = self.config['data']
        preprocessing_config = self.config['preprocessing']

        self.data_processor = BridgeVIVDataProcessor(
            data_path=data_config['dataset_path'],
            random_state=data_config['random_state']
        )

        # Load and process data
        self.data_processor.load_data()
        self.data_processor.clean_data()
        self.data_processor.feature_engineering()

        # Get training data
        tasks_config = self.config['tasks']
        datasets, _ = self.data_processor.split_and_scale_data(
            target_tasks=[tasks_config.get('primary_task', 'amplitude')]
        )

        primary_task = tasks_config.get('primary_task', 'amplitude')
        task_data = datasets[primary_task]

        # Combine train and validation for hyperparameter search
        X_train = task_data['X_train']
        X_val = task_data['X_val']
        y_train = task_data['y_train']
        y_val = task_data['y_val']

        X = pd.concat([X_train, X_val]).values
        y = pd.concat([y_train, y_val]).values

        # Determine task type
        if primary_task == 'amplitude':
            task_type = 'regression'
        elif primary_task == 'viv_occurrence':
            task_type = 'binary_classification'
        else:
            task_type = 'multiclass_classification'

        return X, y, task_type

    def setup_tracking(self):
        """Setup experiment tracking"""
        tracking_config = self.config.get('tracking', {})

        if tracking_config.get('enabled', True):
            self.tracker = ExperimentTracker(
                backend=tracking_config.get('backend', 'mlflow'),
                experiment_name=f"{tracking_config.get('experiment_name', 'bridge_viv')}_hyperopt",
                tracking_uri=tracking_config.get('mlflow_uri', 'file:./experiments/mlflow')
            )

    def objective_function(self,
                          trial: Any,
                          model_name: str,
                          X: np.ndarray,
                          y: np.ndarray,
                          task_type: str) -> float:
        """
        Objective function for hyperparameter optimization

        Args:
            trial: Optimization trial object
            model_name: Name of the model
            X: Feature matrix
            y: Target values
            task_type: Type of task

        Returns:
            Score to optimize
        """
        # Get search space for the model
        search_spaces = self.hyperopt_config.get('search_spaces', {})
        if model_name not in search_spaces:
            raise ValueError(f"No search space defined for model: {model_name}")

        search_space = search_spaces[model_name]

        # Suggest hyperparameters
        params = {}
        for param_name, param_range in search_space.items():
            if isinstance(param_range, list) and len(param_range) == 2:
                if isinstance(param_range[0], int):
                    params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                elif isinstance(param_range[0], float):
                    params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
            elif isinstance(param_range, list):
                params[param_name] = trial.suggest_categorical(param_name, param_range)

        # Add additional parameters
        if model_name == 'mlp':
            params['input_dim'] = X.shape[1]

        params['random_state'] = self.config['data']['random_state']

        # Create and train model
        try:
            model = ModelFactory.create_model(model_name, task_type, **params)

            # Get scorer
            metric_name = self.hyperopt_config.get('metric', 'rmse')
            scorer = get_scorer(metric_name, task_type)

            # Cross-validation
            cv_config = self.config.get('cross_validation', {})
            cv_folds = self.data_processor.get_cv_folds(
                n_splits=cv_config.get('n_splits', 5),
                task_name=task_type
            )

            scores = cross_val_score(model.model, X, y, cv=cv_folds, scoring=scorer, n_jobs=1)
            score = scores.mean()

            # For minimization metrics, return negative score
            if metric_name in ['mae', 'mse', 'rmse']:
                score = -score

            # Log trial
            if self.tracker:
                self.tracker.start_run(f"trial_{trial.number}")
                self.tracker.log_params(params)
                self.tracker.log_metrics({
                    'cv_score': score,
                    'cv_std': scores.std(),
                    'trial_number': trial.number
                })
                self.tracker.end_run()

            # Store in history
            trial_info = {
                'trial_number': trial.number,
                'params': params,
                'score': score,
                'std': scores.std()
            }
            self.search_history.append(trial_info)

            return score

        except Exception as e:
            logger.error(f"Trial failed: {e}")
            # Return worst possible score
            return float('-inf') if self.hyperopt_config.get('direction', 'maximize') == 'maximize' else float('inf')

    def search_optuna(self,
                     model_name: str,
                     X: np.ndarray,
                     y: np.ndarray,
                     task_type: str) -> Dict[str, Any]:
        """
        Hyperparameter search using Optuna

        Args:
            model_name: Name of the model
            X: Feature matrix
            y: Target values
            task_type: Type of task

        Returns:
            Best parameters and results
        """
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna not installed. Install with: pip install optuna")

        # Create study
        direction = self.hyperopt_config.get('direction', 'maximize')
        study = optuna.create_study(direction=direction)

        # Objective function wrapper
        def objective(trial):
            return self.objective_function(trial, model_name, X, y, task_type)

        # Run optimization
        n_trials = self.hyperopt_config.get('n_trials', 50)
        timeout = self.hyperopt_config.get('timeout')

        logger.info(f"Starting Optuna optimization: {n_trials} trials, timeout={timeout}s")

        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        # Get results
        best_params = study.best_params
        best_score = study.best_value

        logger.info(f"Optuna optimization completed. Best score: {best_score:.4f}")

        return {
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': len(study.trials),
            'study': study
        }

    def search_grid(self,
                   model_name: str,
                   X: np.ndarray,
                   y: np.ndarray,
                   task_type: str) -> Dict[str, Any]:
        """
        Grid search for hyperparameters

        Args:
            model_name: Name of the model
            X: Feature matrix
            y: Target values
            task_type: Type of task

        Returns:
            Best parameters and results
        """
        # Get search space
        search_spaces = self.hyperopt_config.get('search_spaces', {})
        if model_name not in search_spaces:
            raise ValueError(f"No search space defined for model: {model_name}")

        search_space = search_spaces[model_name]

        # Convert ranges to lists for grid search
        param_grid = {}
        for param_name, param_range in search_space.items():
            if isinstance(param_range, list) and len(param_range) == 2:
                if isinstance(param_range[0], int):
                    # Create integer range
                    param_grid[param_name] = list(range(param_range[0], param_range[1] + 1,
                                                       max(1, (param_range[1] - param_range[0]) // 5)))
                elif isinstance(param_range[0], float):
                    # Create float range
                    param_grid[param_name] = np.linspace(param_range[0], param_range[1], 5).tolist()
            else:
                param_grid[param_name] = param_range

        # Generate parameter combinations
        param_combinations = list(ParameterGrid(param_grid))

        logger.info(f"Starting grid search: {len(param_combinations)} combinations")

        best_score = float('-inf') if self.hyperopt_config.get('direction', 'maximize') == 'maximize' else float('inf')
        best_params = None
        direction = self.hyperopt_config.get('direction', 'maximize')

        # Evaluate each combination
        for i, params in enumerate(param_combinations):
            try:
                # Add additional parameters
                if model_name == 'mlp':
                    params['input_dim'] = X.shape[1]
                params['random_state'] = self.config['data']['random_state']

                # Create model
                model = ModelFactory.create_model(model_name, task_type, **params)

                # Get scorer
                metric_name = self.hyperopt_config.get('metric', 'rmse')
                scorer = get_scorer(metric_name, task_type)

                # Cross-validation
                cv_config = self.config.get('cross_validation', {})
                cv_folds = self.data_processor.get_cv_folds(
                    n_splits=cv_config.get('n_splits', 5),
                    task_name=task_type
                )

                scores = cross_val_score(model.model, X, y, cv=cv_folds, scoring=scorer, n_jobs=1)
                score = scores.mean()

                # For minimization metrics
                if metric_name in ['mae', 'mse', 'rmse']:
                    score = -score

                # Check if best
                if (direction == 'maximize' and score > best_score) or \
                   (direction == 'minimize' and score < best_score):
                    best_score = score
                    best_params = params.copy()

                # Log trial
                if self.tracker:
                    self.tracker.start_run(f"grid_trial_{i}")
                    self.tracker.log_params(params)
                    self.tracker.log_metrics({
                        'cv_score': score,
                        'cv_std': scores.std(),
                        'trial_number': i
                    })
                    self.tracker.end_run()

                # Store in history
                trial_info = {
                    'trial_number': i,
                    'params': params,
                    'score': score,
                    'std': scores.std()
                }
                self.search_history.append(trial_info)

                if i % 10 == 0:
                    logger.info(f"Grid search progress: {i+1}/{len(param_combinations)}")

            except Exception as e:
                logger.error(f"Grid search trial {i} failed: {e}")
                continue

        logger.info(f"Grid search completed. Best score: {best_score:.4f}")

        return {
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': len(param_combinations)
        }

    def search_random(self,
                     model_name: str,
                     X: np.ndarray,
                     y: np.ndarray,
                     task_type: str) -> Dict[str, Any]:
        """
        Random search for hyperparameters

        Args:
            model_name: Name of the model
            X: Feature matrix
            y: Target values
            task_type: Type of task

        Returns:
            Best parameters and results
        """
        # Get search space
        search_spaces = self.hyperopt_config.get('search_spaces', {})
        if model_name not in search_spaces:
            raise ValueError(f"No search space defined for model: {model_name}")

        search_space = search_spaces[model_name]
        n_trials = self.hyperopt_config.get('n_trials', 50)

        logger.info(f"Starting random search: {n_trials} trials")

        best_score = float('-inf') if self.hyperopt_config.get('direction', 'maximize') == 'maximize' else float('inf')
        best_params = None
        direction = self.hyperopt_config.get('direction', 'maximize')

        # Random search
        for i in range(n_trials):
            try:
                # Sample parameters
                params = {}
                for param_name, param_range in search_space.items():
                    if isinstance(param_range, list) and len(param_range) == 2:
                        if isinstance(param_range[0], int):
                            params[param_name] = np.random.randint(param_range[0], param_range[1] + 1)
                        elif isinstance(param_range[0], float):
                            params[param_name] = np.random.uniform(param_range[0], param_range[1])
                    elif isinstance(param_range, list):
                        params[param_name] = np.random.choice(param_range)

                # Add additional parameters
                if model_name == 'mlp':
                    params['input_dim'] = X.shape[1]
                params['random_state'] = self.config['data']['random_state']

                # Create model
                model = ModelFactory.create_model(model_name, task_type, **params)

                # Get scorer
                metric_name = self.hyperopt_config.get('metric', 'rmse')
                scorer = get_scorer(metric_name, task_type)

                # Cross-validation
                cv_config = self.config.get('cross_validation', {})
                cv_folds = self.data_processor.get_cv_folds(
                    n_splits=cv_config.get('n_splits', 5),
                    task_name=task_type
                )

                scores = cross_val_score(model.model, X, y, cv=cv_folds, scoring=scorer, n_jobs=1)
                score = scores.mean()

                # For minimization metrics
                if metric_name in ['mae', 'mse', 'rmse']:
                    score = -score

                # Check if best
                if (direction == 'maximize' and score > best_score) or \
                   (direction == 'minimize' and score < best_score):
                    best_score = score
                    best_params = params.copy()

                # Log trial
                if self.tracker:
                    self.tracker.start_run(f"random_trial_{i}")
                    self.tracker.log_params(params)
                    self.tracker.log_metrics({
                        'cv_score': score,
                        'cv_std': scores.std(),
                        'trial_number': i
                    })
                    self.tracker.end_run()

                # Store in history
                trial_info = {
                    'trial_number': i,
                    'params': params,
                    'score': score,
                    'std': scores.std()
                }
                self.search_history.append(trial_info)

                if i % 10 == 0:
                    logger.info(f"Random search progress: {i+1}/{n_trials}")

            except Exception as e:
                logger.error(f"Random search trial {i} failed: {e}")
                continue

        logger.info(f"Random search completed. Best score: {best_score:.4f}")

        return {
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': n_trials
        }

    def search_hyperparameters(self, model_name: str) -> Dict[str, Any]:
        """
        Main hyperparameter search function

        Args:
            model_name: Name of the model to optimize

        Returns:
            Search results
        """
        # Setup data
        X, y, task_type = self.setup_data()

        # Setup tracking
        self.setup_tracking()

        logger.info(f"Starting hyperparameter search for {model_name}")
        logger.info(f"Data shape: {X.shape}, Task type: {task_type}")

        # Choose search method
        framework = self.hyperopt_config.get('framework', 'optuna')

        start_time = time.time()

        if framework == 'optuna':
            results = self.search_optuna(model_name, X, y, task_type)
        elif framework == 'grid_search':
            results = self.search_grid(model_name, X, y, task_type)
        elif framework == 'random_search':
            results = self.search_random(model_name, X, y, task_type)
        else:
            raise ValueError(f"Unknown framework: {framework}")

        search_time = time.time() - start_time

        # Add metadata to results
        results.update({
            'model_name': model_name,
            'task_type': task_type,
            'framework': framework,
            'search_time': search_time,
            'data_shape': X.shape,
            'history': self.search_history
        })

        # Store results
        self.best_params[model_name] = results['best_params']
        self.best_scores[model_name] = results['best_score']

        return results

    def save_results(self, results: Dict[str, Any], output_dir: str = "experiments"):
        """
        Save search results

        Args:
            results: Search results
            output_dir: Output directory
        """
        output_path = Path(output_dir) / "hyperopt_results"
        output_path.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        model_name = results['model_name']
        results_file = output_path / f"{model_name}_hyperopt_results.json"

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Hyperparameter search results saved to {results_file}")

        # Save summary
        summary = {
            'model_name': model_name,
            'best_params': results['best_params'],
            'best_score': results['best_score'],
            'n_trials': results['n_trials'],
            'search_time': results['search_time'],
            'framework': results['framework']
        }

        summary_file = output_path / f"{model_name}_hyperopt_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Hyperparameter search summary saved to {summary_file}")

def main():
    """Main hyperparameter search function"""
    parser = argparse.ArgumentParser(description="Hyperparameter Search for Bridge VIV Models")
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True,
                       help='Model to optimize')
    parser.add_argument('--trials', type=int,
                       help='Number of trials (overrides config)')
    parser.add_argument('--framework', type=str, choices=['optuna', 'grid_search', 'random_search'],
                       help='Search framework (overrides config)')
    parser.add_argument('--metric', type=str,
                       help='Metric to optimize (overrides config)')
    parser.add_argument('--output', type=str, default='experiments',
                       help='Output directory')

    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config(args.config)

        # Override with command line arguments
        if args.trials:
            config['hyperopt']['n_trials'] = args.trials
        if args.framework:
            config['hyperopt']['framework'] = args.framework
        if args.metric:
            config['hyperopt']['metric'] = args.metric

        # Validate hyperopt is enabled
        if not config.get('hyperopt', {}).get('enabled', False):
            logger.warning("Hyperparameter optimization is disabled in config. Enabling for this run.")
            config['hyperopt']['enabled'] = True

        # Setup logging
        log_config = config.get('logging', {})
        setup_logging(
            level=log_config.get('level', 'INFO'),
            log_file=log_config.get('log_file'),
            console=log_config.get('console', True)
        )

        # Initialize searcher
        searcher = HyperparameterSearcher(config)

        # Run search
        results = searcher.search_hyperparameters(args.model)

        # Save results
        searcher.save_results(results, args.output)

        # Print summary
        print(f"\nHyperparameter Search Results for {args.model}:")
        print(f"Best Score: {results['best_score']:.4f}")
        print(f"Search Time: {results['search_time']:.2f}s")
        print(f"Trials: {results['n_trials']}")
        print(f"Best Parameters:")
        for param, value in results['best_params'].items():
            print(f"  {param}: {value}")

        logger.info("Hyperparameter search completed successfully!")

    except Exception as e:
        logger.error(f"Hyperparameter search failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()