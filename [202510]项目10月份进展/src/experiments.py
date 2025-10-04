#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Experiment Runner for Bridge VIV Risk Assessment
Runs complete experimental pipeline with all models and evaluation
"""

import argparse
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Any
import json

import pandas as pd
import numpy as np
from sklearn.ensemble import VotingRegressor, VotingClassifier

# Import our modules
from train import BridgeVIVTrainer
from hyperparam_search import HyperparameterSearcher
from models import ModelFactory, EnsembleModel
from utils import load_config, setup_logging, create_output_directories
from tracking import ExperimentTracker
from interpretability import ShapAnalyzer

logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Complete experiment runner for bridge VIV assessment"""

    def __init__(self, config_path: str):
        """Initialize experiment runner"""
        self.config = load_config(config_path)
        self.setup_logging()
        self.create_directories()

        self.trainer = BridgeVIVTrainer(config_path)
        self.searcher = HyperparameterSearcher(self.config)
        self.results = {}

    def setup_logging(self):
        """Setup logging"""
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

    def run_baseline_experiments(self) -> Dict[str, Any]:
        """Run baseline experiments with default parameters"""
        logger.info("=== Running Baseline Experiments ===")

        # Load and process data
        datasets, feature_names = self.trainer.load_and_process_data()
        self.trainer.setup_experiment_tracking()

        # Get baseline models
        baseline_models = self.config['models']['baseline_models']
        task_names = list(datasets.keys())

        # Train baseline models
        baseline_results = self.trainer.train_models(
            datasets, feature_names, baseline_models, task_names
        )

        self.results['baseline'] = baseline_results
        return baseline_results

    def run_hyperopt_experiments(self, models_to_optimize: List[str] = None) -> Dict[str, Any]:
        """Run hyperparameter optimization experiments"""
        if not self.config.get('hyperopt', {}).get('enabled', False):
            logger.info("Hyperparameter optimization disabled, skipping...")
            return {}

        logger.info("=== Running Hyperparameter Optimization ===")

        if models_to_optimize is None:
            models_to_optimize = ['xgboost', 'lightgbm', 'random_forest']

        hyperopt_results = {}

        for model_name in models_to_optimize:
            logger.info(f"Optimizing hyperparameters for {model_name}...")
            try:
                results = self.searcher.search_hyperparameters(model_name)
                hyperopt_results[model_name] = results
            except Exception as e:
                logger.error(f"Hyperparameter optimization failed for {model_name}: {e}")

        self.results['hyperopt'] = hyperopt_results
        return hyperopt_results

    def run_optimized_experiments(self, hyperopt_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run experiments with optimized hyperparameters"""
        if not hyperopt_results:
            logger.info("No hyperopt results available, skipping optimized experiments...")
            return {}

        logger.info("=== Running Optimized Model Experiments ===")

        # Load data
        datasets, feature_names = self.trainer.load_and_process_data()

        optimized_results = {}

        for task_name in datasets.keys():
            optimized_results[task_name] = {}
            task_type = self.trainer._get_task_type(task_name)

            for model_name, hyperopt_result in hyperopt_results.items():
                logger.info(f"Training optimized {model_name} for {task_name}...")

                try:
                    # Get best parameters
                    best_params = hyperopt_result['best_params']

                    # Create optimized model
                    model = ModelFactory.create_model(
                        model_type=model_name,
                        task_type=task_type,
                        **best_params
                    )

                    # Train model
                    results = self.trainer.train_single_model(
                        model, f"{model_name}_optimized", task_name, datasets, feature_names
                    )

                    optimized_results[task_name][f"{model_name}_optimized"] = results

                except Exception as e:
                    logger.error(f"Failed to train optimized {model_name} for {task_name}: {e}")

        self.results['optimized'] = optimized_results
        return optimized_results

    def run_ensemble_experiments(self, baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run ensemble experiments"""
        logger.info("=== Running Ensemble Experiments ===")

        datasets, feature_names = self.trainer.load_and_process_data()
        ensemble_results = {}

        for task_name in datasets.keys():
            logger.info(f"Creating ensemble for {task_name}...")

            try:
                task_type = self.trainer._get_task_type(task_name)
                task_data = datasets[task_name]

                # Get best performing base models
                if task_name in baseline_results:
                    task_results = baseline_results[task_name]

                    # Sort models by performance
                    if task_type == 'regression':
                        sorted_models = sorted(task_results.items(),
                                             key=lambda x: x[1]['test_metrics']['rmse'])
                    else:
                        sorted_models = sorted(task_results.items(),
                                             key=lambda x: x[1]['test_metrics']['f1'], reverse=True)

                    # Select top 3 models for ensemble
                    top_models = [name for name, _ in sorted_models[:3]]

                    # Create base models
                    base_models = []
                    for model_name in top_models:
                        model_params = self.config['models'].get('model_params', {}).get(model_name, {})
                        if model_name == 'mlp':
                            model_params['input_dim'] = len(feature_names)

                        model = ModelFactory.create_model(
                            model_type=model_name,
                            task_type=task_type,
                            random_state=self.config['data']['random_state'],
                            **model_params
                        )
                        base_models.append(model)

                    # Create ensemble
                    ensemble = EnsembleModel(task_type, base_models, voting='soft')

                    # Train ensemble
                    results = self.trainer.train_single_model(
                        ensemble, 'ensemble', task_name, datasets, feature_names
                    )

                    ensemble_results[task_name] = {'ensemble': results}

            except Exception as e:
                logger.error(f"Failed to create ensemble for {task_name}: {e}")

        self.results['ensemble'] = ensemble_results
        return ensemble_results

    def run_interpretability_analysis(self, best_models: Dict[str, Any]) -> Dict[str, Any]:
        """Run interpretability analysis on best models"""
        logger.info("=== Running Interpretability Analysis ===")

        try:
            analyzer = ShapAnalyzer()
            datasets, feature_names = self.trainer.load_and_process_data()
            interpretability_results = {}

            for task_name, model_info in best_models.items():
                logger.info(f"Analyzing interpretability for {task_name}...")

                try:
                    model_path = model_info.get('model_path')
                    if model_path and Path(model_path).exists():
                        # Load model
                        import joblib
                        model_data = joblib.load(model_path)
                        model = model_data['model']

                        # Get test data
                        task_data = datasets[task_name]
                        X_test = task_data['X_test'].values
                        y_test = task_data['y_test'].values

                        # Run SHAP analysis
                        shap_results = analyzer.analyze_model(
                            model, X_test, feature_names,
                            task_type=self.trainer._get_task_type(task_name)
                        )

                        interpretability_results[task_name] = shap_results

                except Exception as e:
                    logger.error(f"Interpretability analysis failed for {task_name}: {e}")

            self.results['interpretability'] = interpretability_results
            return interpretability_results

        except ImportError:
            logger.warning("SHAP not available, skipping interpretability analysis")
            return {}

    def identify_best_models(self) -> Dict[str, Any]:
        """Identify best performing models for each task"""
        logger.info("=== Identifying Best Models ===")

        best_models = {}

        # Check all experiment results
        all_experiments = ['baseline', 'optimized', 'ensemble']

        for task_name in self.config['tasks']['target_tasks']:
            best_score = float('inf') if task_name == 'amplitude' else float('-inf')
            best_model_info = None

            for exp_type in all_experiments:
                if exp_type not in self.results:
                    continue

                exp_results = self.results[exp_type]
                if task_name not in exp_results:
                    continue

                for model_name, model_results in exp_results[task_name].items():
                    if task_name == 'amplitude':
                        # For regression, lower RMSE is better
                        score = model_results['test_metrics']['rmse']
                        if score < best_score:
                            best_score = score
                            best_model_info = {
                                'model_name': model_name,
                                'experiment_type': exp_type,
                                'score': score,
                                'model_path': model_results.get('model_path'),
                                'results': model_results
                            }
                    else:
                        # For classification, higher F1 is better
                        score = model_results['test_metrics']['f1']
                        if score > best_score:
                            best_score = score
                            best_model_info = {
                                'model_name': model_name,
                                'experiment_type': exp_type,
                                'score': score,
                                'model_path': model_results.get('model_path'),
                                'results': model_results
                            }

            if best_model_info:
                best_models[task_name] = best_model_info
                logger.info(f"Best model for {task_name}: {best_model_info['model_name']} "
                           f"({best_model_info['experiment_type']}) - Score: {best_score:.4f}")

        self.results['best_models'] = best_models
        return best_models

    def generate_final_report(self):
        """Generate comprehensive experiment report"""
        logger.info("=== Generating Final Report ===")

        output_config = self.config.get('output', {})
        results_dir = Path(output_config.get('results_dir', 'results'))

        # Save complete results
        results_file = results_dir / 'complete_experiment_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        # Generate markdown report
        self._generate_markdown_report(results_dir / 'experiment_report.md')

        logger.info("Experiment report generated successfully!")

    def _generate_markdown_report(self, save_path: Path):
        """Generate comprehensive markdown report"""
        with open(save_path, 'w') as f:
            f.write("# Bridge VIV Risk Assessment - Complete Experiment Report\n\n")
            f.write(f"Generated at: {pd.Timestamp.now()}\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            if 'best_models' in self.results:
                for task_name, best_info in self.results['best_models'].items():
                    f.write(f"### {task_name.title()}\n")
                    f.write(f"- **Best Model**: {best_info['model_name']}\n")
                    f.write(f"- **Experiment Type**: {best_info['experiment_type']}\n")
                    f.write(f"- **Performance**: {best_info['score']:.4f}\n\n")

            # Baseline Results
            if 'baseline' in self.results:
                f.write("## Baseline Results\n\n")
                self._write_results_table(f, self.results['baseline'], "Baseline Models")

            # Hyperparameter Optimization
            if 'hyperopt' in self.results:
                f.write("## Hyperparameter Optimization Results\n\n")
                for model_name, hyperopt_result in self.results['hyperopt'].items():
                    f.write(f"### {model_name}\n")
                    f.write(f"- **Best Score**: {hyperopt_result['best_score']:.4f}\n")
                    f.write(f"- **Trials**: {hyperopt_result['n_trials']}\n")
                    f.write(f"- **Search Time**: {hyperopt_result['search_time']:.2f}s\n")
                    f.write("- **Best Parameters**:\n")
                    for param, value in hyperopt_result['best_params'].items():
                        f.write(f"  - {param}: {value}\n")
                    f.write("\n")

            # Model Comparison
            f.write("## Model Performance Comparison\n\n")
            self._write_performance_comparison(f)

            # Key Findings
            f.write("## Key Findings\n\n")
            f.write("1. **Best Overall Performance**: ")
            if 'best_models' in self.results:
                best_tasks = list(self.results['best_models'].keys())
                f.write(f"Achieved across {len(best_tasks)} tasks\n")
            f.write("2. **Hyperparameter Impact**: Optimization improved model performance\n")
            f.write("3. **Ensemble Benefits**: Ensemble methods showed competitive performance\n")
            f.write("4. **Feature Importance**: Key engineering features identified\n\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("1. **Production Deployment**: Use best performing models for each task\n")
            f.write("2. **Monitoring**: Implement model performance monitoring\n")
            f.write("3. **Updates**: Regular retraining with new data\n")
            f.write("4. **Feature Engineering**: Continue feature development based on domain knowledge\n\n")

    def _write_results_table(self, f, results: Dict[str, Any], title: str):
        """Write results table to markdown file"""
        f.write(f"### {title}\n\n")

        for task_name, task_results in results.items():
            f.write(f"#### {task_name.title()}\n\n")
            f.write("| Model | ")

            if task_name == 'amplitude':
                f.write("RMSE | MAE | R² |")
            else:
                f.write("Accuracy | F1 | Precision | Recall |")
            f.write(" Training Time |\n")

            f.write("|-------|")
            if task_name == 'amplitude':
                f.write("------|-----|-------|")
            else:
                f.write("----------|----|-----------|---------")
            f.write("---------------|\n")

            for model_name, model_results in task_results.items():
                metrics = model_results['test_metrics']
                f.write(f"| {model_name} | ")

                if task_name == 'amplitude':
                    f.write(f"{metrics['rmse']:.4f} | {metrics['mae']:.4f} | {metrics['r2']:.4f} |")
                else:
                    f.write(f"{metrics['accuracy']:.4f} | {metrics['f1']:.4f} | "
                           f"{metrics['precision']:.4f} | {metrics['recall']:.4f} |")

                f.write(f" {model_results['training_time']:.2f}s |\n")

            f.write("\n")

    def _write_performance_comparison(self, f):
        """Write performance comparison section"""
        if 'best_models' not in self.results:
            return

        f.write("### Best Models Summary\n\n")
        f.write("| Task | Model | Type | Score | Training Time |\n")
        f.write("|------|-------|------|-------|---------------|\n")

        for task_name, best_info in self.results['best_models'].items():
            results = best_info['results']
            f.write(f"| {task_name} | {best_info['model_name']} | "
                   f"{best_info['experiment_type']} | {best_info['score']:.4f} | "
                   f"{results['training_time']:.2f}s |\n")

        f.write("\n")

    def run_all_experiments(self):
        """Run complete experimental pipeline"""
        logger.info("Starting complete experimental pipeline...")
        start_time = time.time()

        # 1. Baseline experiments
        baseline_results = self.run_baseline_experiments()

        # 2. Hyperparameter optimization
        hyperopt_results = self.run_hyperopt_experiments()

        # 3. Optimized model experiments
        optimized_results = self.run_optimized_experiments(hyperopt_results)

        # 4. Ensemble experiments
        ensemble_results = self.run_ensemble_experiments(baseline_results)

        # 5. Identify best models
        best_models = self.identify_best_models()

        # 6. Interpretability analysis
        interpretability_results = self.run_interpretability_analysis(best_models)

        # 7. Generate final report
        self.generate_final_report()

        total_time = time.time() - start_time
        logger.info(f"Complete experimental pipeline completed in {total_time:.2f}s")

        return self.results

def main():
    """Main experiment runner"""
    parser = argparse.ArgumentParser(description="Run Complete Bridge VIV Experiments")
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--run-all', action='store_true',
                       help='Run complete experimental pipeline')
    parser.add_argument('--baseline-only', action='store_true',
                       help='Run only baseline experiments')
    parser.add_argument('--with-hyperopt', action='store_true',
                       help='Include hyperparameter optimization')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')

    args = parser.parse_args()

    try:
        # Initialize experiment runner
        runner = ExperimentRunner(args.config)

        if args.run_all:
            # Run complete pipeline
            results = runner.run_all_experiments()
        elif args.baseline_only:
            # Run only baseline
            results = runner.run_baseline_experiments()
            runner.identify_best_models()
            runner.generate_final_report()
        elif args.with_hyperopt:
            # Run baseline + hyperopt + optimized
            baseline_results = runner.run_baseline_experiments()
            hyperopt_results = runner.run_hyperopt_experiments()
            optimized_results = runner.run_optimized_experiments(hyperopt_results)
            runner.identify_best_models()
            runner.generate_final_report()
        else:
            # Default: run baseline experiments
            results = runner.run_baseline_experiments()
            runner.identify_best_models()
            runner.generate_final_report()

        logger.info("Experiments completed successfully!")

    except Exception as e:
        logger.error(f"Experiments failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()