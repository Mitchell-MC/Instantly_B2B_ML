"""
Advanced Drift Detection Module
Implements PSI, K-S tests, and concept drift detection for comprehensive model monitoring.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedDriftDetector:
    """
    Advanced drift detection using industry-standard metrics.
    
    Implements:
    - Population Stability Index (PSI)
    - Kolmogorov-Smirnov (K-S) tests
    - Concept drift detection
    - Feature importance drift
    - Automated drift reporting
    """
    
    def __init__(self, config=None):
        """Initialize the drift detector."""
        # Default configuration
        default_config = {
            'psi_critical': 0.25,      # PSI > 0.25 = major drift
            'psi_warning': 0.10,       # PSI > 0.10 = minor drift
            'ks_significance': 0.05,   # K-S test significance level
            'concept_drift_threshold': 0.15,  # Concept drift threshold
            'bins': 10,                # Number of bins for PSI calculation
            'min_sample_size': 100     # Minimum samples for reliable drift detection
        }
        
        # Load configuration from file if not provided
        if config is None:
            try:
                import yaml
                import os
                config_path = os.path.join('config', 'main_config.yaml')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        full_config = yaml.safe_load(f)
                    if 'drift_detection' in full_config:
                        # Merge with defaults
                        drift_config = full_config['drift_detection']
                        for key, value in drift_config.items():
                            if key in default_config:
                                default_config[key] = value
                        logger.info("✅ Loaded drift detection configuration from config file")
                    else:
                        logger.warning("⚠️ No drift_detection section found in config, using defaults")
                else:
                    logger.warning("⚠️ Config file not found, using default drift detection settings")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load config file: {e}, using default settings")
        
        # Use provided config or merged defaults
        if config and isinstance(config, dict):
            if 'drift_detection' in config:
                drift_config = config['drift_detection']
                for key, value in drift_config.items():
                    if key in default_config:
                        default_config[key] = value
            else:
                # Direct drift config provided
                for key, value in config.items():
                    if key in default_config:
                        default_config[key] = value
        
        self.config = default_config
        
        self.drift_history = []
        logger.info("🔍 Advanced drift detector initialized")
    
    def calculate_psi(self, expected, actual, feature_name="unknown", bins=None):
        """
        Calculate Population Stability Index (PSI) for a feature.
        
        PSI Interpretation:
        - PSI < 0.1: No significant shift
        - 0.1 <= PSI < 0.25: Minor shift (monitor closely)
        - PSI >= 0.25: Major shift (retraining alert)
        
        Args:
            expected: Expected (training) distribution
            actual: Actual (new) distribution
            feature_name: Name of the feature for logging
            bins: Number of bins for discretization
            
        Returns:
            dict: PSI calculation results
        """
        bins = bins or self.config['bins']
        
        try:
            # Handle different data types
            if pd.api.types.is_numeric_dtype(expected) and pd.api.types.is_numeric_dtype(actual):
                # Numerical feature - create bins
                combined = pd.concat([expected, actual])
                bin_edges = np.linspace(combined.min(), combined.max(), bins + 1)
                
                # Ensure bin edges are unique
                bin_edges = np.unique(bin_edges)
                if len(bin_edges) < 2:
                    bin_edges = np.array([combined.min(), combined.max()])
                
                expected_binned = pd.cut(expected, bins=bin_edges, include_lowest=True)
                actual_binned = pd.cut(actual, bins=bin_edges, include_lowest=True)
                
            elif pd.api.types.is_categorical_dtype(expected) or pd.api.types.is_object_dtype(expected):
                # Categorical feature - use value counts
                expected_binned = expected.value_counts()
                actual_binned = actual.value_counts()
                
                # Align categories
                all_categories = expected_binned.index.union(actual_binned.index)
                expected_binned = expected_binned.reindex(all_categories, fill_value=0)
                actual_binned = actual_binned.reindex(all_categories, fill_value=0)
                
                # Convert to percentages
                expected_pct = expected_binned / expected_binned.sum()
                actual_pct = actual_binned / actual_binned.sum()
                
                # Calculate PSI
                psi = self._calculate_psi_from_pct(expected_pct, actual_pct)
                
                return {
                    'feature_name': feature_name,
                    'psi': psi,
                    'drift_level': self._classify_psi_drift(psi),
                    'expected_distribution': expected_pct.to_dict(),
                    'actual_distribution': actual_pct.to_dict(),
                    'bins': 'categorical',
                    'calculation_method': 'categorical_psi'
                }
            
            else:
                # Mixed or unsupported types
                logger.warning(f"⚠️ Unsupported data type for PSI calculation: {feature_name}")
                return {
                    'feature_name': feature_name,
                    'psi': None,
                    'drift_level': 'unsupported_type',
                    'error': 'Unsupported data type for PSI calculation'
                }
            
            # Calculate bin counts
            expected_counts = expected_binned.value_counts().sort_index()
            actual_counts = actual_binned.value_counts().sort_index()
            
            # Convert to percentages
            expected_pct = expected_counts / expected_counts.sum()
            actual_pct = actual_counts / actual_counts.sum()
            
            # Calculate PSI
            psi = self._calculate_psi_from_pct(expected_pct, actual_pct)
            
            return {
                'feature_name': feature_name,
                'psi': psi,
                'drift_level': self._classify_psi_drift(psi),
                'expected_distribution': expected_pct.to_dict(),
                'actual_distribution': actual_pct.to_dict(),
                'bins': len(bin_edges) - 1 if 'bin_edges' in locals() else bins,
                'calculation_method': 'binned_psi'
            }
            
        except Exception as e:
            logger.error(f"❌ PSI calculation failed for {feature_name}: {e}")
            return {
                'feature_name': feature_name,
                'psi': None,
                'drift_level': 'calculation_error',
                'error': str(e)
            }
    
    def _calculate_psi_from_pct(self, expected_pct, actual_pct):
        """Calculate PSI from percentage distributions."""
        try:
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            expected_pct = expected_pct + epsilon
            actual_pct = actual_pct + epsilon
            
            # Calculate PSI
            psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
            return float(psi)
            
        except Exception as e:
            logger.error(f"❌ PSI calculation error: {e}")
            return None
    
    def _classify_psi_drift(self, psi):
        """Classify PSI drift level."""
        if psi is None:
            return 'calculation_error'
        elif psi < self.config['psi_warning']:
            return 'no_drift'
        elif psi < self.config['psi_critical']:
            return 'minor_drift'
        else:
            return 'major_drift'
    
    def perform_ks_test(self, expected, actual, feature_name="unknown"):
        """
        Perform Kolmogorov-Smirnov test for numerical features.
        
        Args:
            expected: Expected (training) distribution
            actual: Actual (new) distribution
            feature_name: Name of the feature for logging
            
        Returns:
            dict: K-S test results
        """
        try:
            # Ensure numerical data
            if not (pd.api.types.is_numeric_dtype(expected) and pd.api.types.is_numeric_dtype(actual)):
                return {
                    'feature_name': feature_name,
                    'ks_statistic': None,
                    'ks_pvalue': None,
                    'drift_detected': False,
                    'error': 'Non-numerical data for K-S test'
                }
            
            # Remove NaN values
            expected_clean = expected.dropna()
            actual_clean = actual.dropna()
            
            if len(expected_clean) < 10 or len(actual_clean) < 10:
                return {
                    'feature_name': feature_name,
                    'ks_statistic': None,
                    'ks_pvalue': None,
                    'drift_detected': False,
                    'error': 'Insufficient data for K-S test'
                }
            
            # Perform K-S test
            ks_statistic, ks_pvalue = stats.ks_2samp(expected_clean, actual_clean)
            
            # Determine if drift is detected
            drift_detected = ks_pvalue < self.config['ks_significance']
            
            return {
                'feature_name': feature_name,
                'ks_statistic': float(ks_statistic),
                'ks_pvalue': float(ks_pvalue),
                'drift_detected': drift_detected,
                'expected_samples': len(expected_clean),
                'actual_samples': len(actual_clean),
                'significance_level': self.config['ks_significance']
            }
            
        except Exception as e:
            logger.error(f"❌ K-S test failed for {feature_name}: {e}")
            return {
                'feature_name': feature_name,
                'ks_statistic': None,
                'ks_pvalue': None,
                'drift_detected': False,
                'error': str(e)
            }
    
    def detect_concept_drift(self, X_expected, y_expected, X_actual, y_actual, 
                           feature_names=None, method='mutual_info'):
        """
        Detect concept drift (changes in feature-target relationships).
        
        Args:
            X_expected: Expected features
            y_expected: Expected targets
            X_actual: Actual features
            y_actual: Actual targets
            feature_names: List of feature names
            method: Detection method ('mutual_info', 'correlation', 'model_performance')
            
        Returns:
            dict: Concept drift detection results
        """
        try:
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X_expected.shape[1])]
            
            concept_drift_results = {
                'method': method,
                'detection_time': datetime.now().isoformat(),
                'feature_drift': {},
                'overall_drift_score': 0.0,
                'drift_detected': False
            }
            
            if method == 'mutual_info':
                drift_scores = self._detect_concept_drift_mutual_info(
                    X_expected, y_expected, X_actual, y_actual, feature_names
                )
            elif method == 'correlation':
                drift_scores = self._detect_concept_drift_correlation(
                    X_expected, y_expected, X_actual, y_actual, feature_names
                )
            elif method == 'model_performance':
                drift_scores = self._detect_concept_drift_model_performance(
                    X_expected, y_expected, X_actual, y_actual, feature_names
                )
            else:
                raise ValueError(f"Unknown concept drift detection method: {method}")
            
            # Calculate overall drift score
            valid_scores = [score for score in drift_scores.values() if score is not None]
            if valid_scores:
                overall_drift_score = np.mean(valid_scores)
                concept_drift_results['overall_drift_score'] = float(overall_drift_score)
                concept_drift_results['drift_detected'] = overall_drift_score > self.config['concept_drift_threshold']
            
            concept_drift_results['feature_drift'] = drift_scores
            
            logger.info(f"🔍 Concept drift detection complete:")
            logger.info(f"  Method: {method}")
            logger.info(f"  Overall drift score: {concept_drift_results['overall_drift_score']:.4f}")
            logger.info(f"  Drift detected: {'⚠️ YES' if concept_drift_results['drift_detected'] else '✅ NO'}")
            
            return concept_drift_results
            
        except Exception as e:
            logger.error(f"❌ Concept drift detection failed: {e}")
            return {
                'method': method,
                'detection_time': datetime.now().isoformat(),
                'error': str(e),
                'drift_detected': False
            }
    
    def _detect_concept_drift_mutual_info(self, X_expected, y_expected, X_actual, y_actual, feature_names):
        """Detect concept drift using mutual information changes."""
        drift_scores = {}
        
        for i, feature_name in enumerate(feature_names):
            try:
                # Calculate mutual information for expected data
                mi_expected = mutual_info_score(X_expected[:, i], y_expected)
                
                # Calculate mutual information for actual data
                mi_actual = mutual_info_score(X_actual[:, i], y_actual)
                
                # Calculate drift as relative change
                if mi_expected > 0:
                    drift_score = abs(mi_actual - mi_expected) / mi_expected
                else:
                    drift_score = 0.0
                
                drift_scores[feature_name] = float(drift_score)
                
            except Exception as e:
                logger.warning(f"⚠️ Mutual info calculation failed for {feature_name}: {e}")
                drift_scores[feature_name] = None
        
        return drift_scores
    
    def _detect_concept_drift_correlation(self, X_expected, y_expected, X_actual, y_actual, feature_names):
        """Detect concept drift using correlation changes."""
        drift_scores = {}
        
        for i, feature_name in enumerate(feature_names):
            try:
                # Calculate correlation for expected data
                corr_expected = np.corrcoef(X_expected[:, i], y_expected)[0, 1]
                
                # Calculate correlation for actual data
                corr_actual = np.corrcoef(X_actual[:, i], y_actual)[0, 1]
                
                # Handle NaN correlations
                if np.isnan(corr_expected) or np.isnan(corr_actual):
                    drift_scores[feature_name] = None
                    continue
                
                # Calculate drift as absolute difference
                drift_score = abs(corr_actual - corr_expected)
                drift_scores[feature_name] = float(drift_score)
                
            except Exception as e:
                logger.warning(f"⚠️ Correlation calculation failed for {feature_name}: {e}")
                drift_scores[feature_name] = None
        
        return drift_scores
    
    def _detect_concept_drift_model_performance(self, X_expected, y_expected, X_actual, y_actual, feature_names):
        """Detect concept drift using model performance changes."""
        try:
            # Train a simple model on expected data
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X_expected, y_expected)
            
            # Evaluate on both datasets
            score_expected = model.score(X_expected, y_expected)
            score_actual = model.score(X_actual, y_actual)
            
            # Calculate drift as performance degradation
            drift_score = max(0, score_expected - score_actual)
            
            # Return drift score for each feature (same value for all)
            return {feature_name: float(drift_score) for feature_name in feature_names}
            
        except Exception as e:
            logger.warning(f"⚠️ Model performance drift detection failed: {e}")
            return {feature_name: None for feature_name in feature_names}
    
    def comprehensive_drift_analysis(self, training_data, new_data, target_column=None, 
                                  numerical_features=None, categorical_features=None):
        """
        Perform comprehensive drift analysis using multiple methods.
        
        Args:
            training_data: Training dataset
            new_data: New dataset to compare against
            target_column: Target variable column name
            numerical_features: List of numerical feature names
            categorical_features: List of categorical feature names
            
        Returns:
            dict: Comprehensive drift analysis results
        """
        logger.info("🔍 Starting comprehensive drift analysis...")
        
        # Auto-detect feature types if not provided
        if numerical_features is None:
            numerical_features = training_data.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in numerical_features:
                numerical_features.remove(target_column)
        
        if categorical_features is None:
            categorical_features = training_data.select_dtypes(include=['object', 'category']).columns.tolist()
            if target_column in categorical_features:
                categorical_features.remove(target_column)
        
        # Initialize results
        drift_results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'training_samples': len(training_data),
            'new_samples': len(new_data),
            'psi_results': {},
            'ks_results': {},
            'concept_drift_results': None,
            'overall_drift_assessment': 'no_drift',
            'drift_summary': {
                'features_with_major_drift': 0,
                'features_with_minor_drift': 0,
                'features_with_no_drift': 0,
                'total_features_analyzed': 0
            }
        }
        
        # PSI Analysis for all features
        logger.info("📊 Calculating PSI for all features...")
        for feature in numerical_features + categorical_features:
            if feature in training_data.columns and feature in new_data.columns:
                psi_result = self.calculate_psi(
                    training_data[feature], 
                    new_data[feature], 
                    feature
                )
                drift_results['psi_results'][feature] = psi_result
        
        # K-S Test for numerical features
        logger.info("📊 Performing K-S tests for numerical features...")
        for feature in numerical_features:
            if feature in training_data.columns and feature in new_data.columns:
                ks_result = self.perform_ks_test(
                    training_data[feature], 
                    new_data[feature], 
                    feature
                )
                drift_results['ks_results'][feature] = ks_result
        
        # Concept drift detection (if target available)
        if target_column and target_column in training_data.columns and target_column in new_data.columns:
            logger.info("📊 Detecting concept drift...")
            
            # Prepare feature matrices
            X_train = training_data.drop(columns=[target_column])
            y_train = training_data[target_column]
            X_new = new_data.drop(columns=[target_column])
            y_new = new_data[target_column]
            
            # Ensure same features
            common_features = X_train.columns.intersection(X_new.columns)
            X_train = X_train[common_features]
            X_new = X_new[common_features]
            
            concept_drift_results = self.detect_concept_drift(
                X_train.values, y_train.values, 
                X_new.values, y_new.values, 
                list(common_features)
            )
            drift_results['concept_drift_results'] = concept_drift_results
        
        # Compile drift summary
        drift_results = self._compile_drift_summary(drift_results)
        
        # Determine overall drift assessment
        drift_results['overall_drift_assessment'] = self._determine_overall_drift(drift_results)
        
        logger.info(f"✅ Comprehensive drift analysis complete!")
        logger.info(f"  Overall assessment: {drift_results['overall_drift_assessment']}")
        logger.info(f"  Features analyzed: {drift_results['drift_summary']['total_features_analyzed']}")
        
        return drift_results
    
    def _compile_drift_summary(self, drift_results):
        """Compile summary statistics from drift analysis."""
        summary = drift_results['drift_summary']
        
        # Count PSI drift levels
        for feature, psi_result in drift_results['psi_results'].items():
            if psi_result and 'drift_level' in psi_result:
                drift_level = psi_result['drift_level']
                if drift_level == 'major_drift':
                    summary['features_with_major_drift'] += 1
                elif drift_level == 'minor_drift':
                    summary['features_with_minor_drift'] += 1
                elif drift_level == 'no_drift':
                    summary['features_with_no_drift'] += 1
        
        summary['total_features_analyzed'] = (
            summary['features_with_major_drift'] + 
            summary['features_with_minor_drift'] + 
            summary['features_with_no_drift']
        )
        
        return drift_results
    
    def _determine_overall_drift(self, drift_results):
        """Determine overall drift assessment."""
        summary = drift_results['drift_summary']
        
        if summary['features_with_major_drift'] > 0:
            return 'major_drift'
        elif summary['features_with_minor_drift'] > 0:
            return 'minor_drift'
        else:
            return 'no_drift'
    
    def generate_drift_report(self, drift_results, output_path=None):
        """Generate a comprehensive drift report."""
        try:
            report = {
                'drift_analysis_report': {
                    'timestamp': datetime.now().isoformat(),
                    'overall_assessment': drift_results['overall_drift_assessment'],
                    'summary': drift_results['drift_summary'],
                    'recommendations': self._generate_drift_recommendations(drift_results)
                },
                'detailed_results': self._serialize_for_json(drift_results)
            }
            
            if output_path:
                import json
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"📄 Drift report saved to: {output_path}")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Failed to generate drift report: {e}")
            return None
    
    def _serialize_for_json(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {str(k): self._serialize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_for_json(item) for item in obj]
        elif hasattr(obj, 'dtype') and hasattr(obj, 'tolist'):
            # Handle pandas/numpy objects
            return obj.tolist()
        elif hasattr(obj, '__str__'):
            # Handle other objects by converting to string
            return str(obj)
        else:
            return obj
    
    def _generate_drift_recommendations(self, drift_results):
        """Generate recommendations based on drift analysis."""
        recommendations = []
        
        summary = drift_results['drift_summary']
        overall_assessment = drift_results['overall_drift_assessment']
        
        if overall_assessment == 'major_drift':
            recommendations.append("🚨 MAJOR DRIFT DETECTED - Immediate retraining required")
            recommendations.append("   - Review feature engineering pipeline")
            recommendations.append("   - Check data quality and preprocessing")
            recommendations.append("   - Consider model architecture changes")
        elif overall_assessment == 'minor_drift':
            recommendations.append("⚠️ MINOR DRIFT DETECTED - Monitor closely")
            recommendations.append("   - Schedule retraining within 1-2 weeks")
            recommendations.append("   - Investigate specific drifting features")
            recommendations.append("   - Update feature monitoring thresholds")
        else:
            recommendations.append("✅ NO DRIFT DETECTED - Continue monitoring")
            recommendations.append("   - Maintain current retraining schedule")
            recommendations.append("   - Continue monitoring key metrics")
        
        # Add specific feature recommendations
        if summary['features_with_major_drift'] > 0:
            recommendations.append(f"   - {summary['features_with_major_drift']} features show major drift")
        
        if summary['features_with_minor_drift'] > 0:
            recommendations.append(f"   - {summary['features_with_minor_drift']} features show minor drift")
        
        return recommendations
    
    def plot_drift_visualization(self, drift_results, output_path=None):
        """Create drift visualization plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Comprehensive Drift Analysis Dashboard', fontsize=16)
            
            # PSI Drift Summary
            psi_drift_counts = {}
            for feature, result in drift_results['psi_results'].items():
                if result and 'drift_level' in result:
                    level = result['drift_level']
                    psi_drift_counts[level] = psi_drift_counts.get(level, 0) + 1
            
            if psi_drift_counts:
                axes[0, 0].pie(psi_drift_counts.values(), labels=psi_drift_counts.keys(), autopct='%1.1f%%')
                axes[0, 0].set_title('PSI Drift Distribution')
            
            # Top Drifting Features (PSI)
            if drift_results['psi_results']:
                psi_values = [(f, r['psi']) for f, r in drift_results['psi_results'].items() 
                            if r and r['psi'] is not None]
                if psi_values:
                    psi_values.sort(key=lambda x: x[1], reverse=True)
                    top_features = psi_values[:10]
                    
                    features, psi_scores = zip(*top_features)
                    axes[0, 1].barh(range(len(features)), psi_scores)
                    axes[0, 1].set_yticks(range(len(features)))
                    axes[0, 1].set_yticklabels(features)
                    axes[0, 1].set_title('Top 10 Features by PSI Score')
                    axes[0, 1].set_xlabel('PSI Score')
            
            # K-S Test Results
            if drift_results['ks_results']:
                ks_pvalues = [(f, r['ks_pvalue']) for f, r in drift_results['ks_results'].items() 
                             if r and r['ks_pvalue'] is not None]
                if ks_pvalues:
                    features, pvalues = zip(*ks_pvalues)
                    axes[1, 0].scatter(range(len(features)), pvalues, alpha=0.7)
                    axes[1, 0].axhline(y=self.config['ks_significance'], 
                                      color='red', linestyle='--', label='Significance Threshold')
                    axes[1, 0].set_title('K-S Test P-values')
                    axes[1, 0].set_ylabel('P-value')
                    axes[1, 0].legend()
            
            # Concept Drift (if available)
            if drift_results.get('concept_drift_results'):
                concept_drift = drift_results['concept_drift_results']
                if 'feature_drift' in concept_drift:
                    drift_scores = [(f, s) for f, s in concept_drift['feature_drift'].items() if s is not None]
                    if drift_scores:
                        features, scores = zip(*drift_scores)
                        axes[1, 1].bar(range(len(features)), scores)
                        axes[1, 1].set_title('Concept Drift Scores')
                        axes[1, 1].set_ylabel('Drift Score')
                        axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"📊 Drift visualization saved to: {output_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"❌ Failed to create drift visualization: {e}")

def main():
    """Demo of the advanced drift detection module."""
    print("🔍 Advanced Drift Detection Module Demo")
    print("="*60)
    
    # Initialize detector
    detector = AdvancedDriftDetector()
    
    print("💡 Available Methods:")
    print("  1. PSI calculation: detector.calculate_psi(expected, actual, feature_name)")
    print("  2. K-S testing: detector.perform_ks_test(expected, actual, feature_name)")
    print("  3. Concept drift: detector.detect_concept_drift(X_train, y_train, X_new, y_new)")
    print("  4. Comprehensive analysis: detector.comprehensive_drift_analysis(train_data, new_data)")
    
    return detector

if __name__ == "__main__":
    detector = main()
