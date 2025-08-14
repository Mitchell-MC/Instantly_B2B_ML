"""
Model Performance Tracker
Comprehensive tracking and monitoring of model performance metrics over time.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, mean_squared_error, mean_absolute_error,
    r2_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelPerformanceTracker:
    """
    Comprehensive model performance tracking and monitoring.
    
    Features:
    - Real-time performance metrics calculation
    - Performance degradation detection
    - Historical performance tracking
    - Automated alerting for performance issues
    - Performance visualization and reporting
    - A/B testing support for model comparison
    """
    
    def __init__(self, config_path="config/main_config.yaml"):
        """Initialize the performance tracker."""
        self.config = self._load_config(config_path)
        self.performance_history = []
        self.alert_history = []
        self.baseline_metrics = {}
        self.degradation_thresholds = self.config.get('degradation_thresholds', {})
        
        # Create performance storage directory
        os.makedirs("data/performance", exist_ok=True)
        
        # Load existing performance history
        self._load_performance_history()
        
        logger.info("📊 Model performance tracker initialized")
    
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        default_config = {
            'degradation_thresholds': {
                'accuracy_drop': 0.05,      # 5% accuracy drop
                'auc_drop': 0.03,           # 3% AUC drop
                'f1_drop': 0.05,            # 5% F1 drop
                'precision_drop': 0.05,     # 5% precision drop
                'recall_drop': 0.05,        # 5% recall drop
                'mse_increase': 0.10,       # 10% MSE increase
                'mae_increase': 0.10        # 10% MAE increase
            },
            'performance_window_days': 30,  # Days to consider for trend analysis
            'min_samples_for_trend': 10,    # Minimum samples for reliable trend
            'alert_cooldown_hours': 24,     # Hours between similar alerts
            'auto_retraining_threshold': 0.15  # 15% degradation triggers retraining
        }
        
        try:
            import yaml
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    full_config = yaml.safe_load(f)
                if 'model_performance' in full_config:
                    # Merge with defaults
                    perf_config = full_config['model_performance']
                    for key, value in perf_config.items():
                        if key in default_config:
                            default_config[key] = value
                    logger.info("✅ Loaded model performance configuration from config file")
                else:
                    logger.warning("⚠️ No model_performance section found in config, using defaults")
            else:
                logger.warning("⚠️ Config file not found, using default performance tracking settings")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load config file: {e}, using default settings")
        
        return default_config
    
    def track_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: Optional[np.ndarray] = None,
                         sample_weights: Optional[np.ndarray] = None,
                         metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Track model performance for a new batch of predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for classification)
            sample_weights: Sample weights for weighted metrics
            metadata: Additional metadata (model_version, timestamp, etc.)
            
        Returns:
            dict: Performance metrics and analysis
        """
        try:
            # Calculate comprehensive metrics
            metrics = self._calculate_performance_metrics(
                y_true, y_pred, y_pred_proba, sample_weights
            )
            
            # Add metadata
            timestamp = datetime.now()
            performance_record = {
                'timestamp': timestamp.isoformat(),
                'metrics': metrics,
                'metadata': metadata or {},
                'sample_count': len(y_true),
                'performance_score': self._calculate_overall_performance_score(metrics)
            }
            
            # Detect performance degradation
            degradation_analysis = self._analyze_performance_degradation(performance_record)
            performance_record['degradation_analysis'] = degradation_analysis
            
            # Store performance record
            self.performance_history.append(performance_record)
            
            # Check for alerts
            alerts = self._check_performance_alerts(performance_record)
            if alerts:
                self._process_performance_alerts(alerts, performance_record)
            
            # Save updated history
            self._save_performance_history()
            
            logger.info(f"✅ Performance tracked: {performance_record['performance_score']:.3f} "
                       f"({len(y_true)} samples)")
            
            return performance_record
            
        except Exception as e:
            logger.error(f"❌ Performance tracking failed: {e}")
            return None
    
    def _calculate_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     y_pred_proba: Optional[np.ndarray] = None,
                                     sample_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        metrics = {}
        
        try:
            # Determine if this is classification or regression
            is_classification = len(np.unique(y_true)) <= 10 and y_pred.dtype in ['object', 'int64', 'int32']
            
            if is_classification:
                # Classification metrics
                metrics['accuracy'] = accuracy_score(y_true, y_pred, sample_weight=sample_weights)
                metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', 
                                                          sample_weight=sample_weights, zero_division=0)
                metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', 
                                                    sample_weight=sample_weights, zero_division=0)
                metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', 
                                             sample_weight=sample_weights, zero_division=0)
                
                # Binary classification specific metrics
                if len(np.unique(y_true)) == 2:
                    metrics['precision_binary'] = precision_score(y_true, y_pred, 
                                                               sample_weight=sample_weights, zero_division=0)
                    metrics['recall_binary'] = recall_score(y_true, y_pred, 
                                                         sample_weight=sample_weights, zero_division=0)
                    metrics['f1_binary'] = f1_score(y_true, y_pred, 
                                                  sample_weight=sample_weights, zero_division=0)
                
                # Probability-based metrics
                if y_pred_proba is not None:
                    try:
                        metrics['log_loss'] = log_loss(y_true, y_pred_proba, sample_weight=sample_weights)
                        if len(np.unique(y_true)) == 2:
                            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba)
                        else:
                            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
                    except Exception as e:
                        logger.warning(f"⚠️ Probability-based metrics failed: {e}")
                        metrics['log_loss'] = None
                        metrics['roc_auc'] = None
                
                # Confusion matrix
                try:
                    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weights)
                    metrics['confusion_matrix'] = cm.tolist()
                except Exception as e:
                    logger.warning(f"⚠️ Confusion matrix failed: {e}")
                    metrics['confusion_matrix'] = None
                
            else:
                # Regression metrics
                metrics['mse'] = mean_squared_error(y_true, y_pred, sample_weight=sample_weights)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['mae'] = mean_absolute_error(y_true, y_pred, sample_weight=sample_weights)
                metrics['r2'] = r2_score(y_true, y_pred, sample_weight=sample_weights)
                
                # Additional regression metrics
                metrics['mape'] = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
                metrics['smape'] = 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
            
            # Common metrics
            metrics['sample_count'] = len(y_true)
            metrics['prediction_variance'] = np.var(y_pred)
            
        except Exception as e:
            logger.error(f"❌ Metric calculation failed: {e}")
            metrics = {'error': str(e)}
        
        return metrics
    
    def _calculate_overall_performance_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall performance score from individual metrics."""
        try:
            score = 0.0
            count = 0
            
            # Classification metrics (higher is better)
            for metric in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc']:
                if metric in metrics and metrics[metric] is not None:
                    score += metrics[metric]
                    count += 1
            
            # Regression metrics (lower is better, so invert)
            for metric in ['mse', 'rmse', 'mae']:
                if metric in metrics and metrics[metric] is not None:
                    # Normalize to 0-1 scale (assuming reasonable ranges)
                    normalized_score = max(0, 1 - metrics[metric] / 100)  # Adjust divisor as needed
                    score += normalized_score
                    count += 1
            
            # R² score (higher is better)
            if 'r2' in metrics and metrics['r2'] is not None:
                score += max(0, metrics['r2'])  # R² can be negative
                count += 1
            
            return score / count if count > 0 else 0.0
            
        except Exception as e:
            logger.error(f"❌ Performance score calculation failed: {e}")
            return 0.0
    
    def _analyze_performance_degradation(self, performance_record: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance degradation compared to baseline."""
        try:
            if not self.baseline_metrics:
                # Set baseline if this is the first record
                self.baseline_metrics = performance_record['metrics'].copy()
                return {
                    'degradation_detected': False,
                    'baseline_set': True,
                    'message': 'Baseline metrics established'
                }
            
            degradation_analysis = {
                'degradation_detected': False,
                'degraded_metrics': [],
                'degradation_severity': 'none',
                'overall_degradation_score': 0.0
            }
            
            degradation_scores = []
            
            # Compare each metric with baseline
            for metric_name, current_value in performance_record['metrics'].items():
                if (metric_name in self.baseline_metrics and 
                    current_value is not None and 
                    self.baseline_metrics[metric_name] is not None):
                    
                    baseline_value = self.baseline_metrics[metric_name]
                    
                    # Skip non-numeric metrics
                    if not isinstance(current_value, (int, float)) or not isinstance(baseline_value, (int, float)):
                        continue
                    
                    # Calculate degradation
                    if metric_name in ['mse', 'rmse', 'mae', 'log_loss']:
                        # For error metrics, higher is worse
                        if baseline_value > 0:
                            degradation = (current_value - baseline_value) / baseline_value
                        else:
                            degradation = 0
                    else:
                        # For performance metrics, lower is worse
                        if baseline_value > 0:
                            degradation = (baseline_value - current_value) / baseline_value
                        else:
                            degradation = 0
                    
                    # Check if degradation exceeds threshold
                    threshold_key = f"{metric_name}_drop" if metric_name not in ['mse', 'rmse', 'mae', 'log_loss'] else f"{metric_name}_increase"
                    threshold = self.degradation_thresholds.get(threshold_key, 0.05)
                    
                    if degradation > threshold:
                        degradation_analysis['degraded_metrics'].append({
                            'metric': metric_name,
                            'baseline': baseline_value,
                            'current': current_value,
                            'degradation': degradation,
                            'threshold': threshold
                        })
                        degradation_scores.append(degradation)
            
            # Determine overall degradation
            if degradation_scores:
                degradation_analysis['degradation_detected'] = True
                degradation_analysis['overall_degradation_score'] = np.mean(degradation_scores)
                
                # Classify severity
                if degradation_analysis['overall_degradation_score'] > 0.20:
                    degradation_analysis['degradation_severity'] = 'critical'
                elif degradation_analysis['overall_degradation_score'] > 0.10:
                    degradation_analysis['degradation_severity'] = 'moderate'
                else:
                    degradation_analysis['degradation_severity'] = 'minor'
            
            return degradation_analysis
            
        except Exception as e:
            logger.error(f"❌ Performance degradation analysis failed: {e}")
            return {
                'degradation_detected': False,
                'error': str(e)
            }
    
    def _check_performance_alerts(self, performance_record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for performance alerts based on degradation analysis."""
        alerts = []
        
        try:
            degradation_analysis = performance_record.get('degradation_analysis', {})
            
            if degradation_analysis.get('degradation_detected', False):
                severity = degradation_analysis.get('degradation_severity', 'unknown')
                overall_score = degradation_analysis.get('overall_degradation_score', 0.0)
                
                # Create alert
                alert = {
                    'timestamp': performance_record['timestamp'],
                    'alert_type': 'performance_degradation',
                    'severity': severity,
                    'message': f"Model performance degradation detected: {severity} (score: {overall_score:.3f})",
                    'degraded_metrics': degradation_analysis.get('degraded_metrics', []),
                    'overall_degradation_score': overall_score,
                    'performance_score': performance_record.get('performance_score', 0.0)
                }
                
                # Check if this is a retraining trigger
                if overall_score > self.config.get('auto_retraining_threshold', 0.15):
                    alert['retraining_triggered'] = True
                    alert['message'] += " - RETRAINING TRIGGERED"
                
                alerts.append(alert)
            
            # Check for other alert conditions
            performance_score = performance_record.get('performance_score', 0.0)
            if performance_score < 0.5:  # Very low performance
                alerts.append({
                    'timestamp': performance_record['timestamp'],
                    'alert_type': 'low_performance',
                    'severity': 'high',
                    'message': f"Very low model performance detected: {performance_score:.3f}",
                    'performance_score': performance_score
                })
            
        except Exception as e:
            logger.error(f"❌ Performance alert checking failed: {e}")
        
        return alerts
    
    def _process_performance_alerts(self, alerts: List[Dict[str, Any]], 
                                  performance_record: Dict[str, Any]):
        """Process and store performance alerts."""
        try:
            for alert in alerts:
                # Add to alert history
                self.alert_history.append(alert)
                
                # Log alert
                logger.warning(f"🚨 PERFORMANCE ALERT: {alert['message']}")
                
                # Save alerts to file
                self._save_alerts()
                
        except Exception as e:
            logger.error(f"❌ Alert processing failed: {e}")
    
    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get performance summary for the specified time period."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Filter recent performance records
            recent_records = [
                record for record in self.performance_history
                if datetime.fromisoformat(record['timestamp']) > cutoff_date
            ]
            
            if not recent_records:
                return {
                    'period_days': days,
                    'total_records': 0,
                    'message': f"No performance data available for the last {days} days"
                }
            
            # Calculate summary statistics
            performance_scores = [record.get('performance_score', 0) for record in recent_records]
            sample_counts = [record.get('sample_count', 0) for record in recent_records]
            
            summary = {
                'period_days': days,
                'total_records': len(recent_records),
                'total_samples': sum(sample_counts),
                'performance_trend': {
                    'mean_score': np.mean(performance_scores),
                    'std_score': np.std(performance_scores),
                    'min_score': np.min(performance_scores),
                    'max_score': np.max(performance_scores),
                    'trend_direction': self._calculate_trend_direction(performance_scores)
                },
                'degradation_summary': {
                    'total_alerts': len([r for r in recent_records if r.get('degradation_analysis', {}).get('degradation_detected', False)]),
                    'critical_degradations': len([r for r in recent_records 
                                               if r.get('degradation_analysis', {}).get('degradation_severity') == 'critical']),
                    'moderate_degradations': len([r for r in recent_records 
                                               if r.get('degradation_analysis', {}).get('degradation_severity') == 'moderate'])
                },
                'recommendations': self._generate_performance_recommendations(recent_records)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Performance summary generation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_trend_direction(self, scores: List[float]) -> str:
        """Calculate the trend direction of performance scores."""
        try:
            if len(scores) < 2:
                return 'insufficient_data'
            
            # Simple linear trend calculation
            x = np.arange(len(scores))
            slope = np.polyfit(x, scores, 1)[0]
            
            if slope > 0.01:
                return 'improving'
            elif slope < -0.01:
                return 'degrading'
            else:
                return 'stable'
                
        except Exception as e:
            logger.error(f"❌ Trend calculation failed: {e}")
            return 'unknown'
    
    def _generate_performance_recommendations(self, recent_records: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on performance analysis."""
        recommendations = []
        
        try:
            # Check for degradation patterns
            degradation_count = len([r for r in recent_records 
                                  if r.get('degradation_analysis', {}).get('degradation_detected', False)])
            
            if degradation_count > len(recent_records) * 0.3:  # More than 30% show degradation
                recommendations.append("🚨 Frequent performance degradation detected - consider immediate retraining")
            
            # Check performance score trends
            performance_scores = [r.get('performance_score', 0) for r in recent_records]
            if len(performance_scores) >= 5:
                recent_avg = np.mean(performance_scores[-5:])
                overall_avg = np.mean(performance_scores)
                if recent_avg < overall_avg * 0.9:  # Recent performance 10% below average
                    recommendations.append("⚠️ Recent performance below historical average - monitor closely")
            
            # Check for low performance
            low_performance_count = len([r for r in recent_records 
                                       if r.get('performance_score', 0) < 0.6])
            if low_performance_count > 0:
                recommendations.append(f"⚠️ {low_performance_count} recent performance records below acceptable threshold")
            
            if not recommendations:
                recommendations.append("✅ Performance is stable and within acceptable ranges")
            
        except Exception as e:
            logger.error(f"❌ Recommendation generation failed: {e}")
            recommendations = ["❌ Error generating recommendations"]
        
        return recommendations
    
    def generate_performance_report(self, output_path: Optional[str] = None, 
                                  days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        try:
            summary = self.get_performance_summary(days)
            
            report = {
                'performance_report': {
                    'generated_at': datetime.now().isoformat(),
                    'summary': summary,
                    'recent_performance': self.performance_history[-10:] if self.performance_history else [],
                    'recent_alerts': self.alert_history[-10:] if self.alert_history else []
                }
            }
            
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                logger.info(f"📄 Performance report saved to: {output_path}")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Performance report generation failed: {e}")
            return None
    
    def plot_performance_trends(self, output_path: Optional[str] = None, 
                               days: int = 30) -> None:
        """Create performance trend visualization."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Filter recent records
            recent_records = [
                record for record in self.performance_history
                if datetime.fromisoformat(record['timestamp']) > cutoff_date
            ]
            
            if not recent_records:
                logger.warning(f"⚠️ No performance data available for the last {days} days")
                return
            
            # Prepare data for plotting
            timestamps = [datetime.fromisoformat(record['timestamp']) for record in recent_records]
            performance_scores = [record.get('performance_score', 0) for record in recent_records]
            
            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Model Performance Trends (Last {days} Days)', fontsize=16)
            
            # Performance Score Trend
            axes[0, 0].plot(timestamps, performance_scores, 'b-o', linewidth=2, markersize=6)
            axes[0, 0].set_title('Performance Score Over Time')
            axes[0, 0].set_ylabel('Performance Score')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Performance Score Distribution
            axes[0, 1].hist(performance_scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 1].set_title('Performance Score Distribution')
            axes[0, 1].set_xlabel('Performance Score')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Sample Count Trend
            sample_counts = [record.get('sample_count', 0) for record in recent_records]
            axes[1, 0].plot(timestamps, sample_counts, 'g-s', linewidth=2, markersize=6)
            axes[1, 0].set_title('Sample Count Over Time')
            axes[1, 0].set_ylabel('Sample Count')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Degradation Analysis
            degradation_detected = [1 if record.get('degradation_analysis', {}).get('degradation_detected', False) else 0 
                                  for record in recent_records]
            axes[1, 1].scatter(timestamps, degradation_detected, c='red', s=100, alpha=0.7)
            axes[1, 1].set_title('Performance Degradation Detection')
            axes[1, 1].set_ylabel('Degradation Detected (1=Yes, 0=No)')
            axes[1, 1].set_ylim(-0.1, 1.1)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"📊 Performance trends plot saved to: {output_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"❌ Performance trends plotting failed: {e}")
    
    def _load_performance_history(self):
        """Load performance history from file."""
        try:
            history_path = "data/performance/performance_history.json"
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    self.performance_history = json.load(f)
                logger.info(f"📚 Loaded performance history: {len(self.performance_history)} records")
            else:
                logger.info("📚 No existing performance history found")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load performance history: {e}")
            self.performance_history = []
    
    def _save_performance_history(self):
        """Save performance history to file."""
        try:
            history_path = "data/performance/performance_history.json"
            os.makedirs(os.path.dirname(history_path), exist_ok=True)
            with open(history_path, 'w') as f:
                json.dump(self.performance_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"❌ Failed to save performance history: {e}")
    
    def _save_alerts(self):
        """Save alerts to file."""
        try:
            alerts_path = "data/performance/alerts.json"
            os.makedirs(os.path.dirname(alerts_path), exist_ok=True)
            with open(alerts_path, 'w') as f:
                json.dump(self.alert_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"❌ Failed to save alerts: {e}")
    
    def get_performance_status(self) -> Dict[str, Any]:
        """Get current performance tracking status."""
        return {
            'total_performance_records': len(self.performance_history),
            'total_alerts': len(self.alert_history),
            'baseline_established': bool(self.baseline_metrics),
            'last_performance_check': self.performance_history[-1]['timestamp'] if self.performance_history else None,
            'recent_alerts': self.alert_history[-5:] if self.alert_history else []
        }

def main():
    """Demo of the model performance tracker."""
    print("📊 Model Performance Tracker Demo")
    print("="*60)
    
    # Initialize tracker
    tracker = ModelPerformanceTracker()
    
    print("💡 Available Methods:")
    print("  1. Track performance: tracker.track_performance(y_true, y_pred, y_pred_proba)")
    print("  2. Get summary: tracker.get_performance_summary(days=30)")
    print("  3. Generate report: tracker.generate_performance_report(output_path)")
    print("  4. Plot trends: tracker.plot_performance_trends(output_path)")
    print("  5. Get status: tracker.get_performance_status()")
    
    return tracker

if __name__ == "__main__":
    tracker = main()
