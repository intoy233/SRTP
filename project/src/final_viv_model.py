#!/usr/bin/env python3

import numpy as np
import pandas as pd

class FinalVIVModel:
    def __init__(self):
        self.coefficients = {}
        self.feature_names = []
        self.scaler_params = {}
        self.is_fitted = False

    def create_physics_features(self, df):
        df_new = df.copy()

        # Key VIV physics features
        if all(col in df.columns for col in ['Critical_Wind_Speed_ms', 'Natural_Freq_Hz', 'Width_m']):
            df_new['Reduced_Velocity'] = df['Critical_Wind_Speed_ms'] / (df['Natural_Freq_Hz'] * df['Width_m'])

        if all(col in df.columns for col in ['Damping_Ratio', 'Width_m', 'Height_m']):
            df_new['Scruton_Number'] = df['Damping_Ratio'] * (df['Width_m'] / df['Height_m']) * 100

        if all(col in df.columns for col in ['Width_m', 'Height_m']):
            df_new['Aspect_Ratio'] = df['Width_m'] / df['Height_m']

        if all(col in df.columns for col in ['Natural_Freq_Hz', 'Span_m']):
            df_new['Stiffness_Parameter'] = df['Natural_Freq_Hz'] * df['Span_m']**0.5

        if 'Damping_Ratio' in df.columns:
            df_new['VIV_Susceptibility'] = 1.0 / (df['Damping_Ratio'] + 1e-6)

        return df_new

    def select_features(self, X, y, max_features=8):
        correlations = {}
        for col in X.select_dtypes(include=[np.number]).columns:
            if X[col].std() > 1e-6:
                corr = np.corrcoef(X[col], y)[0, 1]
                if not np.isnan(corr):
                    correlations[col] = abs(corr)

        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        return [feature for feature, _ in sorted_features[:max_features]]

    def standardize(self, X, fit=True):
        if fit:
            self.scaler_params = {'mean': X.mean(), 'std': X.std() + 1e-8}
        return ((X - self.scaler_params['mean']) / self.scaler_params['std']).values

    def ridge_regression(self, X, y, alpha=1.0):
        n_samples, n_features = X.shape
        X_with_intercept = np.column_stack([np.ones(n_samples), X])

        XTX = X_with_intercept.T @ X_with_intercept
        XTy = X_with_intercept.T @ y

        reg_matrix = alpha * np.eye(n_features + 1)
        reg_matrix[0, 0] = 0

        coefficients = np.linalg.solve(XTX + reg_matrix, XTy)
        return {'intercept': coefficients[0], 'coef': coefficients[1:], 'alpha': alpha}

    def cross_validate(self, X, y, k_folds=5):
        n_samples = len(y)
        fold_size = n_samples // k_folds
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
        best_alpha, best_score = None, -np.inf

        for alpha in alphas:
            scores = []
            for fold in range(k_folds):
                start_idx = fold * fold_size
                end_idx = start_idx + fold_size if fold < k_folds - 1 else n_samples

                val_indices = list(range(start_idx, end_idx))
                train_indices = [i for i in range(n_samples) if i not in val_indices]

                X_train_fold = X[train_indices]
                y_train_fold = y[train_indices]
                X_val_fold = X[val_indices]
                y_val_fold = y[val_indices]

                model_result = self.ridge_regression(X_train_fold, y_train_fold, alpha)

                X_val_with_intercept = np.column_stack([np.ones(len(X_val_fold)), X_val_fold])
                y_pred = X_val_with_intercept @ np.concatenate([[model_result['intercept']], model_result['coef']])

                ss_res = np.sum((y_val_fold - y_pred) ** 2)
                ss_tot = np.sum((y_val_fold - np.mean(y_val_fold)) ** 2)
                r2 = 1 - (ss_res / (ss_tot + 1e-8))
                scores.append(r2)

            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_alpha = alpha

        return best_alpha, best_score

    def fit(self, X, y):
        print("Training Physics VIV Model...")

        X_engineered = self.create_physics_features(X)
        self.feature_names = self.select_features(X_engineered, y, max_features=8)
        print(f"Selected features: {len(self.feature_names)}")

        X_selected = X_engineered[self.feature_names]
        X_scaled = self.standardize(X_selected, fit=True)

        best_alpha, best_score = self.cross_validate(X_scaled, y.values)
        print(f"Best alpha: {best_alpha}, CV R2: {best_score:.4f}")

        self.coefficients = self.ridge_regression(X_scaled, y.values, best_alpha)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X_engineered = self.create_physics_features(X)
        X_selected = X_engineered[self.feature_names]
        X_scaled = self.standardize(X_selected, fit=False)

        X_with_intercept = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled])
        return X_with_intercept @ np.concatenate([[self.coefficients['intercept']], self.coefficients['coef']])

    def evaluate(self, X, y):
        y_pred = self.predict(X)

        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - y_pred))

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        return {'RMSE': rmse, 'R2': r2, 'MAE': mae}

def main():
    print("Physics-Based VIV Prediction Experiment")
    print("=" * 50)

    # Load data
    df = pd.read_csv('data/enhanced_bridge_dataset.csv')
    print(f"Dataset: {df.shape[0]} bridges, {df.shape[1]} features")

    # Prepare data
    target_col = 'Max_Amplitude_mm'
    exclude_cols = ['BridgeID', 'BridgeName', 'Country', 'PaperSource', 'Year',
                   target_col, 'Risk_Level', 'Notes', 'Vibration_Suppression', 'Suppression_Effect']

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols]
    y = df[target_col]

    print(f"Target: {target_col}")
    print(f"Features: {len(feature_cols)}")
    print(f"Target stats: mean={y.mean():.2f}, std={y.std():.2f}")

    # Split data
    np.random.seed(42)
    n_test = int(0.2 * len(df))
    test_indices = np.random.choice(len(df), n_test, replace=False)
    train_indices = [i for i in range(len(df)) if i not in test_indices]

    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    print(f"Split: train={len(X_train)}, test={len(X_test)}")

    # Train model
    print("\n" + "=" * 50)
    model = FinalVIVModel()
    model.fit(X_train, y_train)

    # Evaluate
    print("\nModel Evaluation")
    print("=" * 50)

    train_metrics = model.evaluate(X_train, y_train)
    print("Training Performance:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.6f}")

    test_metrics = model.evaluate(X_test, y_test)
    print("\nTesting Performance:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.6f}")

    # Feature importance
    print("\nFeature Importance:")
    for i, (feature, coef) in enumerate(zip(model.feature_names, model.coefficients['coef'])):
        print(f"  {feature}: {coef:.4f}")

    # Generate comparison
    y_pred = model.predict(X_test)
    print("\nPrediction vs Actual (first 10 samples):")
    for i in range(min(10, len(y_test))):
        actual = y_test.iloc[i]
        predicted = y_pred[i]
        error = abs(actual - predicted)
        print(f"  Actual: {actual:.2f}, Predicted: {predicted:.2f}, Error: {error:.2f}")

    # Save report
    report = f"""Physics-Based VIV Model Report
============================================================
Experiment Time: {pd.Timestamp.now()}

Performance:
Training - RMSE: {train_metrics['RMSE']:.6f}, R2: {train_metrics['R2']:.6f}
Testing  - RMSE: {test_metrics['RMSE']:.6f}, R2: {test_metrics['R2']:.6f}

Model Comparison:
- Original Ridge (80 samples): R2 = 0.938
- SOTA Deep Learning: R2 = -0.348
- Hybrid SOTA: R2 = -1.443
- Physics Model: R2 = {test_metrics['R2']:.4f}

Selected Features ({len(model.feature_names)}):
"""

    for feature, coef in zip(model.feature_names, model.coefficients['coef']):
        report += f"{feature}: {coef:.4f}\n"

    with open('results/final_physics_model_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nReport saved: results/final_physics_model_report.txt")
    return model, test_metrics

if __name__ == "__main__":
    model, results = main()