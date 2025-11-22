#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Library for Bridge VIV Risk Assessment
Comprehensive collection of traditional ML and deep learning models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Type
from abc import ABC, abstractmethod
import logging
import joblib
from pathlib import Path
import time

# Traditional ML models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingRegressor, VotingClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

# Gradient boosting models
import xgboost as xgb
import lightgbm as lgb
try:
    import catboost as cb
except ImportError:
    cb = None

# Neural networks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Base class for all models"""

    def __init__(self, task_type: str, random_state: int = 42):
        """
        Initialize base model

        Args:
            task_type: Type of task ('regression', 'binary_classification', 'multiclass_classification')
            random_state: Random seed for reproducibility
        """
        self.task_type = task_type
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        self.training_time = 0.0
        self.prediction_time = 0.0

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseModel':
        """Fit the model"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (for classification models)"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError("Model does not support probability prediction")

    def save_model(self, path: Union[str, Path]) -> None:
        """Save the model"""
        joblib.dump({
            'model': self.model,
            'task_type': self.task_type,
            'is_fitted': self.is_fitted,
            'training_time': self.training_time
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: Union[str, Path]) -> None:
        """Load the model"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.task_type = model_data['task_type']
        self.is_fitted = model_data['is_fitted']
        self.training_time = model_data.get('training_time', 0.0)
        logger.info(f"Model loaded from {path}")

class LinearModel(BaseModel):
    """Linear models (Linear/Ridge/Lasso Regression, Logistic Regression)"""

    def __init__(self, task_type: str, model_type: str = 'linear',
                 alpha: float = 1.0, random_state: int = 42):
        """
        Initialize linear model

        Args:
            task_type: Type of task
            model_type: Type of linear model ('linear', 'ridge', 'lasso', 'logistic')
            alpha: Regularization strength
            random_state: Random seed
        """
        super().__init__(task_type, random_state)
        self.model_type = model_type
        self.alpha = alpha

        if task_type == 'regression':
            if model_type == 'linear':
                self.model = LinearRegression()
            elif model_type == 'ridge':
                self.model = Ridge(alpha=alpha, random_state=random_state)
            elif model_type == 'lasso':
                self.model = Lasso(alpha=alpha, random_state=random_state, max_iter=2000)
        else:  # classification
            self.model = LogisticRegression(
                C=1/alpha if alpha > 0 else 1.0,
                random_state=random_state,
                max_iter=2000
            )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'LinearModel':
        """Fit the linear model"""
        start_time = time.time()
        self.model.fit(X, y)
        self.training_time = time.time() - start_time
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        start_time = time.time()
        predictions = self.model.predict(X)
        self.prediction_time = (time.time() - start_time) / len(X) * 1000  # ms per sample
        return predictions

class RandomForestModel(BaseModel):
    """Random Forest model"""

    def __init__(self, task_type: str, n_estimators: int = 100,
                 max_depth: Optional[int] = None, random_state: int = 42):
        """
        Initialize Random Forest model

        Args:
            task_type: Type of task
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            random_state: Random seed
        """
        super().__init__(task_type, random_state)

        if task_type == 'regression':
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
        else:  # classification
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'RandomForestModel':
        """Fit the Random Forest model"""
        start_time = time.time()
        self.model.fit(X, y)
        self.training_time = time.time() - start_time
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        start_time = time.time()
        predictions = self.model.predict(X)
        self.prediction_time = (time.time() - start_time) / len(X) * 1000
        return predictions

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance"""
        if self.is_fitted:
            return self.model.feature_importances_
        else:
            raise ValueError("Model must be fitted before getting feature importance")

class SVMModel(BaseModel):
    """Support Vector Machine model"""

    def __init__(self, task_type: str, kernel: str = 'rbf',
                 C: float = 1.0, gamma: str = 'scale', random_state: int = 42):
        """
        Initialize SVM model

        Args:
            task_type: Type of task
            kernel: Kernel type
            C: Regularization parameter
            gamma: Kernel coefficient
            random_state: Random seed
        """
        super().__init__(task_type, random_state)

        if task_type == 'regression':
            self.model = SVR(kernel=kernel, C=C, gamma=gamma)
        else:  # classification
            self.model = SVC(
                kernel=kernel, C=C, gamma=gamma,
                random_state=random_state, probability=True
            )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'SVMModel':
        """Fit the SVM model"""
        start_time = time.time()
        self.model.fit(X, y)
        self.training_time = time.time() - start_time
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        start_time = time.time()
        predictions = self.model.predict(X)
        self.prediction_time = (time.time() - start_time) / len(X) * 1000
        return predictions

class XGBoostModel(BaseModel):
    """XGBoost model"""

    def __init__(self, task_type: str, n_estimators: int = 100,
                 max_depth: int = 6, learning_rate: float = 0.1,
                 random_state: int = 42):
        """Initialize XGBoost model"""
        super().__init__(task_type, random_state)

        if task_type == 'regression':
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                verbosity=0
            )
        else:  # classification
            objective = 'binary:logistic' if task_type == 'binary_classification' else 'multi:softprob'
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                objective=objective,
                random_state=random_state,
                verbosity=0
            )

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            early_stopping_rounds: int = 10, **kwargs) -> 'XGBoostModel':
        """Fit the XGBoost model with optional early stopping"""
        start_time = time.time()

        if X_val is not None and y_val is not None:
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )
        else:
            self.model.fit(X, y)

        self.training_time = time.time() - start_time
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        start_time = time.time()
        predictions = self.model.predict(X)
        self.prediction_time = (time.time() - start_time) / len(X) * 1000
        return predictions

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance"""
        if self.is_fitted:
            return self.model.feature_importances_
        else:
            raise ValueError("Model must be fitted before getting feature importance")

class LightGBMModel(BaseModel):
    """LightGBM model"""

    def __init__(self, task_type: str, n_estimators: int = 100,
                 max_depth: int = -1, learning_rate: float = 0.1,
                 random_state: int = 42):
        """Initialize LightGBM model"""
        super().__init__(task_type, random_state)

        if task_type == 'regression':
            self.model = lgb.LGBMRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                verbosity=-1
            )
        else:  # classification
            objective = 'binary' if task_type == 'binary_classification' else 'multiclass'
            self.model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                objective=objective,
                random_state=random_state,
                verbosity=-1
            )

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            early_stopping_rounds: int = 10, **kwargs) -> 'LightGBMModel':
        """Fit the LightGBM model with optional early stopping"""
        start_time = time.time()

        if X_val is not None and y_val is not None:
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )
        else:
            self.model.fit(X, y)

        self.training_time = time.time() - start_time
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        start_time = time.time()
        predictions = self.model.predict(X)
        self.prediction_time = (time.time() - start_time) / len(X) * 1000
        return predictions

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance"""
        if self.is_fitted:
            return self.model.feature_importances_
        else:
            raise ValueError("Model must be fitted before getting feature importance")

class CatBoostModel(BaseModel):
    """CatBoost model"""

    def __init__(self, task_type: str, n_estimators: int = 100,
                 max_depth: int = 6, learning_rate: float = 0.1,
                 random_state: int = 42):
        """Initialize CatBoost model"""
        super().__init__(task_type, random_state)

        if cb is None:
            raise ImportError("CatBoost is not installed")

        if task_type == 'regression':
            self.model = cb.CatBoostRegressor(
                iterations=n_estimators,
                depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                verbose=False
            )
        else:  # classification
            self.model = cb.CatBoostClassifier(
                iterations=n_estimators,
                depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                verbose=False
            )

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            early_stopping_rounds: int = 10, **kwargs) -> 'CatBoostModel':
        """Fit the CatBoost model with optional early stopping"""
        start_time = time.time()

        if X_val is not None and y_val is not None:
            self.model.fit(
                X, y,
                eval_set=(X_val, y_val),
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )
        else:
            self.model.fit(X, y)

        self.training_time = time.time() - start_time
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        start_time = time.time()
        predictions = self.model.predict(X)
        self.prediction_time = (time.time() - start_time) / len(X) * 1000
        return predictions

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance"""
        if self.is_fitted:
            return self.model.feature_importances_
        else:
            raise ValueError("Model must be fitted before getting feature importance")

# Neural Network Models (PyTorch)
if TORCH_AVAILABLE:
    class MLPModel(BaseModel):
        """Multi-Layer Perceptron using PyTorch"""

        def __init__(self, task_type: str, input_dim: int, hidden_dims: List[int] = [64, 32],
                     dropout_rate: float = 0.3, learning_rate: float = 0.001,
                     random_state: int = 42):
            """Initialize MLP model"""
            super().__init__(task_type, random_state)

            torch.manual_seed(random_state)
            np.random.seed(random_state)

            self.input_dim = input_dim
            self.hidden_dims = hidden_dims
            self.dropout_rate = dropout_rate
            self.learning_rate = learning_rate

            # Determine output dimension
            if task_type == 'regression':
                output_dim = 1
            elif task_type == 'binary_classification':
                output_dim = 1
            else:  # multiclass
                output_dim = None  # Will be set during fit

            self.output_dim = output_dim
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        def _build_model(self, output_dim: int):
            """Build the neural network"""
            layers = []
            prev_dim = self.input_dim

            # Hidden layers
            for hidden_dim in self.hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate)
                ])
                prev_dim = hidden_dim

            # Output layer
            layers.append(nn.Linear(prev_dim, output_dim))

            # Add activation for classification
            if self.task_type == 'binary_classification':
                layers.append(nn.Sigmoid())
            elif self.task_type == 'multiclass_classification':
                layers.append(nn.Softmax(dim=1))

            return nn.Sequential(*layers)

        def fit(self, X: np.ndarray, y: np.ndarray,
                X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                epochs: int = 100, batch_size: int = 32,
                early_stopping_patience: int = 10, **kwargs) -> 'MLPModel':
            """Fit the MLP model"""
            start_time = time.time()

            # Determine output dimension for multiclass
            if self.task_type == 'multiclass_classification':
                self.output_dim = len(np.unique(y))

            # Build model
            self.model = self._build_model(self.output_dim).to(self.device)

            # Convert to tensors
            X_tensor = torch.FloatTensor(X).to(self.device)

            if self.task_type == 'regression':
                y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(self.device)
                criterion = nn.MSELoss()
            elif self.task_type == 'binary_classification':
                y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(self.device)
                criterion = nn.BCELoss()
            else:  # multiclass
                y_tensor = torch.LongTensor(y).to(self.device)
                criterion = nn.CrossEntropyLoss()

            # Optimizer
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

            # Training loop
            train_dataset = TensorDataset(X_tensor, y_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(epochs):
                self.model.train()
                train_loss = 0.0

                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)

                    if self.task_type == 'multiclass_classification':
                        outputs = outputs.squeeze()
                        loss = criterion(outputs, batch_y)
                    else:
                        loss = criterion(outputs, batch_y)

                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                # Validation
                if X_val is not None and y_val is not None:
                    self.model.eval()
                    with torch.no_grad():
                        X_val_tensor = torch.FloatTensor(X_val).to(self.device)

                        if self.task_type == 'regression':
                            y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1)).to(self.device)
                        elif self.task_type == 'binary_classification':
                            y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1)).to(self.device)
                        else:
                            y_val_tensor = torch.LongTensor(y_val).to(self.device)

                        val_outputs = self.model(X_val_tensor)
                        val_loss = criterion(val_outputs, y_val_tensor).item()

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1

                        if patience_counter >= early_stopping_patience:
                            logger.info(f"Early stopping at epoch {epoch}")
                            break

            self.training_time = time.time() - start_time
            self.is_fitted = True
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            """Make predictions"""
            if not self.is_fitted:
                raise ValueError("Model must be fitted before prediction")

            start_time = time.time()
            self.model.eval()

            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                outputs = self.model(X_tensor)

                if self.task_type == 'regression':
                    predictions = outputs.cpu().numpy().flatten()
                elif self.task_type == 'binary_classification':
                    predictions = (outputs.cpu().numpy().flatten() > 0.5).astype(int)
                else:  # multiclass
                    predictions = torch.argmax(outputs, dim=1).cpu().numpy()

            self.prediction_time = (time.time() - start_time) / len(X) * 1000
            return predictions

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            """Predict probabilities"""
            if not self.is_fitted:
                raise ValueError("Model must be fitted before prediction")

            if self.task_type == 'regression':
                raise ValueError("Regression models don't support probability prediction")

            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                outputs = self.model(X_tensor)

                if self.task_type == 'binary_classification':
                    proba = outputs.cpu().numpy()
                    return np.column_stack([1 - proba, proba])
                else:  # multiclass
                    return outputs.cpu().numpy()

class EnsembleModel(BaseModel):
    """Ensemble model combining multiple base models"""

    def __init__(self, task_type: str, base_models: List[BaseModel],
                 voting: str = 'soft', weights: Optional[List[float]] = None):
        """
        Initialize ensemble model

        Args:
            task_type: Type of task
            base_models: List of base models
            voting: Voting strategy ('hard' or 'soft')
            weights: Weights for each model
        """
        super().__init__(task_type)
        self.base_models = base_models
        self.voting = voting
        self.weights = weights

        if task_type == 'regression':
            # For regression, use simple averaging
            self.ensemble_method = 'average'
        else:
            # For classification, use sklearn VotingClassifier
            self.ensemble_method = 'voting'

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'EnsembleModel':
        """Fit all base models"""
        start_time = time.time()

        for i, model in enumerate(self.base_models):
            logger.info(f"Training base model {i+1}/{len(self.base_models)}")
            model.fit(X, y, **kwargs)

        self.training_time = time.time() - start_time
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions"""
        start_time = time.time()

        if self.task_type == 'regression':
            # Average predictions for regression
            predictions = []
            for model in self.base_models:
                pred = model.predict(X)
                predictions.append(pred)

            predictions = np.array(predictions)

            if self.weights is not None:
                weights = np.array(self.weights).reshape(-1, 1)
                ensemble_pred = np.average(predictions, axis=0, weights=weights.flatten())
            else:
                ensemble_pred = np.mean(predictions, axis=0)

        else:
            # Voting for classification
            if self.voting == 'hard':
                predictions = []
                for model in self.base_models:
                    pred = model.predict(X)
                    predictions.append(pred)

                predictions = np.array(predictions)
                # Majority voting
                ensemble_pred = []
                for i in range(len(X)):
                    votes = predictions[:, i]
                    ensemble_pred.append(np.bincount(votes.astype(int)).argmax())
                ensemble_pred = np.array(ensemble_pred)

            else:  # soft voting
                probabilities = []
                for model in self.base_models:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X)
                        probabilities.append(proba)
                    else:
                        # Fallback to hard predictions
                        pred = model.predict(X)
                        if self.task_type == 'binary_classification':
                            proba = np.column_stack([1 - pred, pred])
                        else:
                            # One-hot encode for multiclass
                            n_classes = len(np.unique(pred))
                            proba = np.eye(n_classes)[pred]
                        probabilities.append(proba)

                probabilities = np.array(probabilities)

                if self.weights is not None:
                    weights = np.array(self.weights).reshape(-1, 1, 1)
                    avg_proba = np.average(probabilities, axis=0, weights=weights.flatten())
                else:
                    avg_proba = np.mean(probabilities, axis=0)

                ensemble_pred = np.argmax(avg_proba, axis=1)

        self.prediction_time = (time.time() - start_time) / len(X) * 1000
        return ensemble_pred

class ModelFactory:
    """Factory for creating models"""

    @staticmethod
    def create_model(model_type: str, task_type: str, **kwargs) -> BaseModel:
        """
        Create a model instance

        Args:
            model_type: Type of model
            task_type: Type of task
            **kwargs: Additional arguments for model initialization

        Returns:
            Model instance
        """
        model_registry = {
            'linear': LinearModel,
            'ridge': lambda task_type, **kwargs: LinearModel(task_type, 'ridge', **kwargs),
            'lasso': lambda task_type, **kwargs: LinearModel(task_type, 'lasso', **kwargs),
            'logistic': lambda task_type, **kwargs: LinearModel(task_type, 'logistic', **kwargs),
            'random_forest': RandomForestModel,
            'svm': SVMModel,
            'xgboost': XGBoostModel,
            'lightgbm': LightGBMModel,
            'catboost': CatBoostModel,
        }

        if TORCH_AVAILABLE:
            model_registry['mlp'] = MLPModel

        if model_type not in model_registry:
            raise ValueError(f"Unknown model type: {model_type}")

        return model_registry[model_type](task_type, **kwargs)

    @staticmethod
    def get_baseline_models(task_type: str, **kwargs) -> Dict[str, BaseModel]:
        """Get a set of baseline models for comparison"""
        models = {}

        if task_type == 'regression':
            models['linear'] = LinearModel(task_type, 'linear', **kwargs)
            models['ridge'] = LinearModel(task_type, 'ridge', **kwargs)
            models['random_forest'] = RandomForestModel(task_type, **kwargs)
            models['xgboost'] = XGBoostModel(task_type, **kwargs)
            models['lightgbm'] = LightGBMModel(task_type, **kwargs)

            if cb is not None:
                models['catboost'] = CatBoostModel(task_type, **kwargs)

        else:  # classification
            models['logistic'] = LinearModel(task_type, 'logistic', **kwargs)
            models['random_forest'] = RandomForestModel(task_type, **kwargs)
            models['svm'] = SVMModel(task_type, **kwargs)
            models['xgboost'] = XGBoostModel(task_type, **kwargs)
            models['lightgbm'] = LightGBMModel(task_type, **kwargs)

            if cb is not None:
                models['catboost'] = CatBoostModel(task_type, **kwargs)

        if TORCH_AVAILABLE and kwargs.get('input_dim') is not None:
            models['mlp'] = MLPModel(task_type, **kwargs)

        return models

def main():
    """Example usage"""
    # Create sample data
    X = np.random.randn(100, 10)
    y_reg = np.random.randn(100)
    y_clf = np.random.randint(0, 2, 100)

    # Test regression models
    reg_models = ModelFactory.get_baseline_models('regression')
    for name, model in reg_models.items():
        print(f"Training {name} for regression...")
        model.fit(X, y_reg)
        pred = model.predict(X[:10])
        print(f"  Predictions shape: {pred.shape}")

    # Test classification models
    clf_models = ModelFactory.get_baseline_models('binary_classification', input_dim=10)
    for name, model in clf_models.items():
        print(f"Training {name} for classification...")
        model.fit(X, y_clf)
        pred = model.predict(X[:10])
        print(f"  Predictions shape: {pred.shape}")

if __name__ == "__main__":
    main()