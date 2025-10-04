#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediction Script for Bridge VIV Risk Assessment
Load trained models and make predictions on new data
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import pandas as pd
import numpy as np
import joblib

from utils import load_config, setup_logging, load_data
from data_processing import BridgeVIVDataProcessor

logger = logging.getLogger(__name__)

class BridgeVIVPredictor:
    """Bridge VIV prediction system"""

    def __init__(self, model_path: Union[str, Path], processor_path: Optional[Union[str, Path]] = None):
        """
        Initialize predictor

        Args:
            model_path: Path to trained model
            processor_path: Path to data processor (optional)
        """
        self.model_path = Path(model_path)
        self.processor_path = Path(processor_path) if processor_path else None

        self.model = None
        self.processor = None
        self.model_info = {}

        self.load_model()
        if self.processor_path:
            self.load_processor()

    def load_model(self):
        """Load trained model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            model_data = joblib.load(self.model_path)

            if isinstance(model_data, dict):
                self.model = model_data['model']
                self.model_info = {
                    'task_type': model_data.get('task_type', 'unknown'),
                    'is_fitted': model_data.get('is_fitted', False),
                    'training_time': model_data.get('training_time', 0.0)
                }
            else:
                self.model = model_data
                self.model_info = {'task_type': 'unknown'}

            logger.info(f"Model loaded successfully from {self.model_path}")

        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")

    def load_processor(self):
        """Load data processor"""
        if not self.processor_path.exists():
            logger.warning(f"Processor file not found: {self.processor_path}")
            return

        try:
            self.processor = BridgeVIVDataProcessor('')
            self.processor.load_processor(self.processor_path)
            logger.info(f"Data processor loaded from {self.processor_path}")
        except Exception as e:
            logger.warning(f"Failed to load processor: {e}")

    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess input data

        Args:
            data: Input DataFrame

        Returns:
            Preprocessed feature array
        """
        if self.processor:
            # Use trained processor
            try:
                # Apply same transformations as training
                processed_data = data.copy()

                # Apply feature engineering if available
                if hasattr(self.processor, 'boxcox_transformer') and self.processor.boxcox_transformer:
                    processed_data = self.processor.boxcox_transformer.transform(processed_data)

                if hasattr(self.processor, 'poly_transformer') and self.processor.poly_transformer:
                    processed_data = self.processor.poly_transformer.transform(processed_data)

                # Apply scaling
                if self.processor.scaler:
                    # Select only the features that were used in training
                    available_features = [col for col in processed_data.columns
                                        if col in processed_data.select_dtypes(include=[np.number]).columns]

                    feature_matrix = processed_data[available_features].values
                    scaled_features = self.processor.scaler.transform(feature_matrix)

                    return scaled_features
                else:
                    return processed_data.select_dtypes(include=[np.number]).values

            except Exception as e:
                logger.error(f"Preprocessing with trained processor failed: {e}")
                return self._basic_preprocessing(data)
        else:
            return self._basic_preprocessing(data)

    def _basic_preprocessing(self, data: pd.DataFrame) -> np.ndarray:
        """
        Basic preprocessing without trained processor

        Args:
            data: Input DataFrame

        Returns:
            Basic preprocessed features
        """
        logger.warning("Using basic preprocessing - results may be suboptimal")

        # Select numeric columns
        numeric_data = data.select_dtypes(include=[np.number])

        # Fill missing values
        numeric_data = numeric_data.fillna(numeric_data.median())

        # Basic feature engineering
        if 'Width_m' in numeric_data.columns and 'Height_m' in numeric_data.columns:
            numeric_data['Width_Height_Ratio'] = numeric_data['Width_m'] / numeric_data['Height_m']

        if 'Span_m' in numeric_data.columns and 'Height_m' in numeric_data.columns:
            numeric_data['Slenderness_Ratio'] = numeric_data['Span_m'] / numeric_data['Height_m']

        return numeric_data.values

    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions

        Args:
            data: Input data (DataFrame or array)

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        # Preprocess data
        if isinstance(data, pd.DataFrame):
            X = self.preprocess_data(data)
        else:
            X = data

        # Make predictions
        try:
            if hasattr(self.model, 'predict'):
                predictions = self.model.predict(X)
            else:
                raise ValueError("Model does not have predict method")

            return predictions

        except Exception as e:
            raise ValueError(f"Prediction failed: {e}")

    def predict_proba(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict probabilities (for classification models)

        Args:
            data: Input data

        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        # Preprocess data
        if isinstance(data, pd.DataFrame):
            X = self.preprocess_data(data)
        else:
            X = data

        # Predict probabilities
        try:
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)
            else:
                raise ValueError("Model does not support probability prediction")

            return probabilities

        except Exception as e:
            raise ValueError(f"Probability prediction failed: {e}")

    def predict_with_confidence(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Make predictions with confidence scores

        Args:
            data: Input data

        Returns:
            Dictionary with predictions and confidence scores
        """
        predictions = self.predict(data)

        result = {'predictions': predictions}

        # For classification, add probabilities and confidence
        if self.model_info.get('task_type') in ['binary_classification', 'multiclass_classification']:
            try:
                probabilities = self.predict_proba(data)
                result['probabilities'] = probabilities

                # Confidence as max probability
                result['confidence'] = np.max(probabilities, axis=1)

            except Exception:
                logger.warning("Could not compute confidence scores")

        return result

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return self.model_info.copy()

def load_best_models(experiments_dir: str = "experiments") -> Dict[str, str]:
    """
    Load paths to best models for each task

    Args:
        experiments_dir: Experiments directory

    Returns:
        Dictionary mapping task names to model paths
    """
    experiments_path = Path(experiments_dir)
    results_file = experiments_path / "complete_experiment_results.json"

    if not results_file.exists():
        logger.warning(f"Experiment results not found at {results_file}")
        return {}

    try:
        import json
        with open(results_file, 'r') as f:
            results = json.load(f)

        best_models = {}
        if 'best_models' in results:
            for task_name, model_info in results['best_models'].items():
                model_path = model_info.get('model_path')
                if model_path and Path(model_path).exists():
                    best_models[task_name] = model_path

        return best_models

    except Exception as e:
        logger.error(f"Failed to load best models: {e}")
        return {}

def predict_risk_assessment(data: pd.DataFrame, models_dir: str = "experiments") -> Dict[str, Any]:
    """
    Comprehensive risk assessment prediction

    Args:
        data: Input bridge data
        models_dir: Directory containing trained models

    Returns:
        Risk assessment results
    """
    best_models = load_best_models(models_dir)

    if not best_models:
        raise ValueError("No trained models found")

    results = {}

    # Predict amplitude
    if 'amplitude' in best_models:
        try:
            predictor = BridgeVIVPredictor(best_models['amplitude'])
            amplitude_pred = predictor.predict(data)
            results['amplitude_prediction'] = amplitude_pred
            results['max_amplitude'] = float(np.max(amplitude_pred))
            results['mean_amplitude'] = float(np.mean(amplitude_pred))
        except Exception as e:
            logger.error(f"Amplitude prediction failed: {e}")

    # Predict VIV occurrence
    if 'viv_occurrence' in best_models:
        try:
            predictor = BridgeVIVPredictor(best_models['viv_occurrence'])
            viv_pred = predictor.predict_with_confidence(data)
            results['viv_occurrence'] = viv_pred['predictions']
            if 'confidence' in viv_pred:
                results['viv_confidence'] = viv_pred['confidence']
        except Exception as e:
            logger.error(f"VIV occurrence prediction failed: {e}")

    # Predict risk class
    if 'risk_class' in best_models:
        try:
            predictor = BridgeVIVPredictor(best_models['risk_class'])
            risk_pred = predictor.predict_with_confidence(data)
            results['risk_class'] = risk_pred['predictions']
            if 'confidence' in risk_pred:
                results['risk_confidence'] = risk_pred['confidence']
        except Exception as e:
            logger.error(f"Risk class prediction failed: {e}")

    # Overall risk assessment
    if 'amplitude_prediction' in results:
        amplitude = results['max_amplitude']
        if amplitude < 20:
            overall_risk = "Low"
        elif amplitude < 40:
            overall_risk = "Medium"
        else:
            overall_risk = "High"

        results['overall_risk'] = overall_risk

    return results

def main():
    """Main prediction function"""
    parser = argparse.ArgumentParser(description="Bridge VIV Risk Assessment Prediction")
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input CSV file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output CSV file')
    parser.add_argument('--processor', type=str,
                       help='Path to data processor file')
    parser.add_argument('--task', type=str, choices=['amplitude', 'risk_class', 'viv_occurrence'],
                       help='Prediction task')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run comprehensive risk assessment with all models')

    args = parser.parse_args()

    try:
        # Setup logging
        setup_logging(level='INFO', console=True)

        # Load input data
        input_data = load_data(args.input)
        logger.info(f"Loaded input data: {input_data.shape}")

        if args.comprehensive:
            # Comprehensive risk assessment
            results = predict_risk_assessment(input_data)

            # Create output DataFrame
            output_data = input_data.copy()
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    output_data[key] = value
                else:
                    output_data[key] = value

        else:
            # Single model prediction
            predictor = BridgeVIVPredictor(args.model, args.processor)

            # Make predictions
            predictions = predictor.predict(input_data)

            # Try to get probabilities for classification
            model_info = predictor.get_model_info()
            if model_info.get('task_type') in ['binary_classification', 'multiclass_classification']:
                try:
                    probabilities = predictor.predict_proba(input_data)
                    confidence = np.max(probabilities, axis=1)
                except:
                    probabilities = None
                    confidence = None
            else:
                probabilities = None
                confidence = None

            # Create output DataFrame
            output_data = input_data.copy()

            if args.task:
                output_data[f'{args.task}_prediction'] = predictions
                if confidence is not None:
                    output_data[f'{args.task}_confidence'] = confidence
            else:
                output_data['predictions'] = predictions
                if confidence is not None:
                    output_data['confidence'] = confidence

        # Save results
        output_data.to_csv(args.output, index=False)
        logger.info(f"Predictions saved to {args.output}")

        # Print summary
        if args.comprehensive:
            print("\nRisk Assessment Summary:")
            if 'overall_risk' in results:
                print(f"Overall Risk Level: {results['overall_risk']}")
            if 'max_amplitude' in results:
                print(f"Maximum Predicted Amplitude: {results['max_amplitude']:.2f} mm")
        else:
            print(f"\nPrediction Summary:")
            print(f"Number of predictions: {len(predictions)}")
            if args.task == 'amplitude':
                print(f"Mean predicted amplitude: {np.mean(predictions):.2f} mm")
                print(f"Max predicted amplitude: {np.max(predictions):.2f} mm")
            elif confidence is not None:
                print(f"Mean confidence: {np.mean(confidence):.3f}")

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()