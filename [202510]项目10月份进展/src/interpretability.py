#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Interpretability Analysis for Bridge VIV Risk Assessment
SHAP-based feature importance and explanation
"""

import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class ShapAnalyzer:
    """SHAP-based model interpretability analyzer"""

    def __init__(self):
        """Initialize SHAP analyzer"""
        try:
            import shap
            self.shap = shap
            self.available = True
            logger.info("SHAP available for interpretability analysis")
        except ImportError:
            logger.warning("SHAP not available. Install with: pip install shap")
            self.available = False

    def analyze_model(self,
                     model: Any,
                     X: np.ndarray,
                     feature_names: List[str],
                     task_type: str = 'regression',
                     max_display: int = 20,
                     save_plots: bool = True,
                     output_dir: str = 'results/interpretability') -> Dict[str, Any]:
        """
        Comprehensive SHAP analysis of a model

        Args:
            model: Trained model
            X: Feature matrix
            feature_names: Names of features
            task_type: Type of task
            max_display: Maximum features to display in plots
            save_plots: Whether to save plots
            output_dir: Output directory for plots

        Returns:
            Dictionary with analysis results
        """
        if not self.available:
            logger.warning("SHAP not available, returning empty results")
            return {}

        # Create output directory
        if save_plots:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        results = {}

        try:
            # Create explainer
            explainer = self._create_explainer(model, X, task_type)

            # Calculate SHAP values
            logger.info("Calculating SHAP values...")
            shap_values = explainer.shap_values(X)

            # Handle different output formats
            if isinstance(shap_values, list):
                # Multi-class classification
                shap_values_main = shap_values[0]  # Use first class for analysis
                results['shap_values_all_classes'] = shap_values
            else:
                shap_values_main = shap_values
                results['shap_values'] = shap_values

            # Global feature importance
            feature_importance = self._calculate_global_importance(shap_values_main, feature_names)
            results['global_importance'] = feature_importance

            # Generate plots
            if save_plots:
                self._generate_plots(
                    explainer, shap_values, X, feature_names,
                    task_type, max_display, output_path
                )

            # Feature importance ranking
            results['feature_ranking'] = self._rank_features(feature_importance)

            # Summary statistics
            results['summary_stats'] = self._calculate_summary_stats(shap_values_main, feature_names)

            logger.info("SHAP analysis completed successfully")

        except Exception as e:
            logger.error(f"SHAP analysis failed: {e}")

        return results

    def _create_explainer(self, model: Any, X: np.ndarray, task_type: str):
        """Create appropriate SHAP explainer"""
        try:
            # Try TreeExplainer first (for tree-based models)
            explainer = self.shap.TreeExplainer(model)
            logger.info("Using TreeExplainer")
            return explainer
        except:
            pass

        try:
            # Try LinearExplainer (for linear models)
            explainer = self.shap.LinearExplainer(model, X)
            logger.info("Using LinearExplainer")
            return explainer
        except:
            pass

        try:
            # Fallback to KernelExplainer (model-agnostic)
            background = self.shap.kmeans(X, min(100, len(X)))
            explainer = self.shap.KernelExplainer(model.predict, background)
            logger.info("Using KernelExplainer")
            return explainer
        except Exception as e:
            raise ValueError(f"Could not create SHAP explainer: {e}")

    def _calculate_global_importance(self, shap_values: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Calculate global feature importance"""
        # Mean absolute SHAP values
        importance = np.mean(np.abs(shap_values), axis=0)

        # Create dictionary
        feature_importance = {}
        for i, name in enumerate(feature_names[:len(importance)]):
            feature_importance[name] = float(importance[i])

        return feature_importance

    def _rank_features(self, feature_importance: Dict[str, float]) -> List[Dict[str, Any]]:
        """Rank features by importance"""
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        ranking = []
        for rank, (feature, importance) in enumerate(sorted_features, 1):
            ranking.append({
                'rank': rank,
                'feature': feature,
                'importance': importance,
                'percentage': importance / sum(feature_importance.values()) * 100
            })

        return ranking

    def _calculate_summary_stats(self, shap_values: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Calculate summary statistics for SHAP values"""
        stats = {}

        for i, name in enumerate(feature_names[:shap_values.shape[1]]):
            feature_shap = shap_values[:, i]
            stats[name] = {
                'mean': float(np.mean(feature_shap)),
                'std': float(np.std(feature_shap)),
                'min': float(np.min(feature_shap)),
                'max': float(np.max(feature_shap)),
                'mean_abs': float(np.mean(np.abs(feature_shap)))
            }

        return stats

    def _generate_plots(self,
                       explainer: Any,
                       shap_values: Union[np.ndarray, List[np.ndarray]],
                       X: np.ndarray,
                       feature_names: List[str],
                       task_type: str,
                       max_display: int,
                       output_path: Path):
        """Generate SHAP plots"""

        # Use main SHAP values for plotting
        if isinstance(shap_values, list):
            shap_values_plot = shap_values[0]
        else:
            shap_values_plot = shap_values

        # Limit features for display
        n_features = min(max_display, len(feature_names), shap_values_plot.shape[1])

        try:
            # 1. Summary plot
            plt.figure(figsize=(10, 8))
            self.shap.summary_plot(
                shap_values_plot[:, :n_features],
                X[:, :n_features],
                feature_names=feature_names[:n_features],
                show=False
            )
            plt.tight_layout()
            plt.savefig(output_path / 'shap_summary.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 2. Bar plot
            plt.figure(figsize=(10, 8))
            self.shap.summary_plot(
                shap_values_plot[:, :n_features],
                X[:, :n_features],
                feature_names=feature_names[:n_features],
                plot_type="bar",
                show=False
            )
            plt.tight_layout()
            plt.savefig(output_path / 'shap_bar.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 3. Waterfall plot for first sample
            if hasattr(self.shap, 'waterfall_plot'):
                plt.figure(figsize=(10, 8))
                if hasattr(explainer, 'expected_value'):
                    expected_value = explainer.expected_value
                    if isinstance(expected_value, np.ndarray):
                        expected_value = expected_value[0]
                else:
                    expected_value = 0

                try:
                    # Create explanation object for waterfall plot
                    explanation = self.shap.Explanation(
                        values=shap_values_plot[0, :n_features],
                        base_values=expected_value,
                        data=X[0, :n_features],
                        feature_names=feature_names[:n_features]
                    )
                    self.shap.waterfall_plot(explanation, show=False)
                    plt.tight_layout()
                    plt.savefig(output_path / 'shap_waterfall.png', dpi=300, bbox_inches='tight')
                    plt.close()
                except:
                    logger.warning("Could not create waterfall plot")

            # 4. Force plot (save as HTML)
            try:
                if hasattr(explainer, 'expected_value'):
                    expected_value = explainer.expected_value
                    if isinstance(expected_value, np.ndarray):
                        expected_value = expected_value[0]

                    force_plot = self.shap.force_plot(
                        expected_value,
                        shap_values_plot[0, :n_features],
                        X[0, :n_features],
                        feature_names=feature_names[:n_features],
                        show=False
                    )

                    self.shap.save_html(str(output_path / 'shap_force.html'), force_plot)
            except:
                logger.warning("Could not create force plot")

            logger.info(f"SHAP plots saved to {output_path}")

        except Exception as e:
            logger.error(f"Failed to generate SHAP plots: {e}")

    def create_feature_importance_report(self,
                                       feature_ranking: List[Dict[str, Any]],
                                       feature_descriptions: Optional[Dict[str, str]] = None,
                                       save_path: Optional[Path] = None) -> str:
        """
        Create feature importance report

        Args:
            feature_ranking: Ranked features with importance
            feature_descriptions: Optional descriptions for features
            save_path: Path to save report

        Returns:
            Report as markdown string
        """
        report = []
        report.append("# Feature Importance Analysis\n")
        report.append("## Bridge VIV Risk Assessment Model Interpretability\n")

        # Top features summary
        report.append("### Top 10 Most Important Features\n")
        report.append("| Rank | Feature | Importance | Percentage | Description |\n")
        report.append("|------|---------|------------|------------|-------------|\n")

        for feature_info in feature_ranking[:10]:
            rank = feature_info['rank']
            feature = feature_info['feature']
            importance = feature_info['importance']
            percentage = feature_info['percentage']

            description = "Engineering feature"
            if feature_descriptions and feature in feature_descriptions:
                description = feature_descriptions[feature]

            report.append(f"| {rank} | {feature} | {importance:.4f} | {percentage:.1f}% | {description} |\n")

        report.append("\n")

        # Feature categories analysis
        report.append("### Feature Category Analysis\n")

        categories = {
            'Geometric': ['width', 'height', 'span', 'ratio', 'slenderness'],
            'Dynamic': ['freq', 'frequency', 'damping'],
            'Aerodynamic': ['drag', 'lift', 'wind', 'strouhal', 'reynolds'],
            'Structural': ['structure', 'material', 'type'],
            'Engineering': ['ratio', 'number', 'derived']
        }

        category_importance = {}
        for category, keywords in categories.items():
            importance = 0
            count = 0
            for feature_info in feature_ranking:
                feature = feature_info['feature'].lower()
                if any(keyword in feature for keyword in keywords):
                    importance += feature_info['importance']
                    count += 1
            category_importance[category] = {'importance': importance, 'count': count}

        report.append("| Category | Total Importance | Feature Count | Avg Importance |\n")
        report.append("|----------|------------------|---------------|----------------|\n")

        for category, info in sorted(category_importance.items(),
                                   key=lambda x: x[1]['importance'], reverse=True):
            total_imp = info['importance']
            count = info['count']
            avg_imp = total_imp / count if count > 0 else 0

            report.append(f"| {category} | {total_imp:.4f} | {count} | {avg_imp:.4f} |\n")

        report.append("\n")

        # Key insights
        report.append("### Key Insights\n")

        # Most important feature
        top_feature = feature_ranking[0]
        report.append(f"1. **Most Important Feature**: {top_feature['feature']} "
                     f"({top_feature['percentage']:.1f}% of total importance)\n")

        # Top 5 contribution
        top_5_contribution = sum(f['percentage'] for f in feature_ranking[:5])
        report.append(f"2. **Top 5 Features**: Account for {top_5_contribution:.1f}% of model decisions\n")

        # Dominant category
        top_category = max(category_importance.items(), key=lambda x: x[1]['importance'])
        report.append(f"3. **Dominant Category**: {top_category[0]} features have highest total importance\n")

        report.append("\n")

        # Recommendations
        report.append("### Engineering Recommendations\n")
        report.append("1. **Data Collection**: Focus on high-importance features for future data collection\n")
        report.append("2. **Sensor Placement**: Prioritize monitoring of critical parameters\n")
        report.append("3. **Feature Engineering**: Consider creating more features in high-importance categories\n")
        report.append("4. **Model Simplification**: Low-importance features could be removed for simpler models\n")

        report_text = "".join(report)

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Feature importance report saved to {save_path}")

        return report_text

def main():
    """Test interpretability analysis"""
    # Create sample data for testing
    np.random.seed(42)
    X = np.random.randn(100, 10)
    feature_names = [f'feature_{i}' for i in range(10)]

    # Simple model for testing
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    y = np.random.randn(100)
    model.fit(X, y)

    # Test SHAP analysis
    analyzer = ShapAnalyzer()
    if analyzer.available:
        results = analyzer.analyze_model(
            model, X[:20], feature_names,
            task_type='regression',
            save_plots=True,
            output_dir='test_interpretability'
        )

        if 'feature_ranking' in results:
            # Generate report
            report = analyzer.create_feature_importance_report(
                results['feature_ranking'],
                save_path=Path('test_interpretability/feature_report.md')
            )

            print("Top 5 features:")
            for feature_info in results['feature_ranking'][:5]:
                print(f"  {feature_info['rank']}. {feature_info['feature']}: "
                      f"{feature_info['importance']:.4f}")

        print("Interpretability analysis test completed!")
    else:
        print("SHAP not available for testing")

if __name__ == "__main__":
    main()