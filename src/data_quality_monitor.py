"""
Data Quality Monitor
Comprehensive monitoring of data quality metrics and anomalies.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataQualityMonitor:
    """
    Comprehensive data quality monitoring and validation.
    
    Features:
    - Data completeness monitoring
    - Data type validation
    - Range and distribution analysis
    - Anomaly detection
    - Data drift monitoring
    - Quality score calculation
    - Automated alerting
    """
    
    def __init__(self, config_path="config/main_config.yaml"):
        """Initialize the data quality monitor."""
        self.config = self._load_config(config_path)
        self.quality_history = []
        self.alert_history = []
        self.baseline_quality = {}
        self.quality_thresholds = self.config.get('quality_thresholds', {})
        
        # Create quality storage directory
        os.makedirs("data/quality", exist_ok=True)
        
        # Load existing quality history
        self._load_quality_history()
        
        logger.info("🔍 Data quality monitor initialized")
    
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        default_config = {
            'quality_thresholds': {
                'completeness_min': 0.95,      # 95% completeness required
                'accuracy_min': 0.90,          # 90% accuracy required
                'consistency_min': 0.85,       # 85% consistency required
                'validity_min': 0.90,          # 90% validity required
                'uniqueness_min': 0.95,        # 95% uniqueness required
                'timeliness_hours': 24,        # Data should be within 24 hours
                'anomaly_threshold': 0.10      # 10% anomaly threshold
            },
            'monitoring_window_days': 30,      # Days to consider for trend analysis
            'min_samples_for_analysis': 100,   # Minimum samples for reliable analysis
            'alert_cooldown_hours': 12,        # Hours between similar alerts
            'auto_retraining_threshold': 0.20  # 20% quality degradation triggers retraining
        }
        
        try:
            import yaml
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    full_config = yaml.safe_load(f)
                if 'data_quality' in full_config:
                    # Merge with defaults
                    quality_config = full_config['data_quality']
                    for key, value in quality_config.items():
                        if key in default_config:
                            default_config[key] = value
                    logger.info("✅ Loaded data quality configuration from config file")
                else:
                    logger.warning("⚠️ No data_quality section found in config, using defaults")
            else:
                logger.warning("⚠️ Config file not found, using default data quality settings")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load config file: {e}, using default settings")
        
        return default_config
    
    def monitor_data_quality(self, data: pd.DataFrame, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Monitor data quality for a new dataset.
        
        Args:
            data: Input DataFrame to monitor
            metadata: Additional metadata (source, timestamp, etc.)
            
        Returns:
            dict: Quality metrics and analysis
        """
        try:
            # Calculate comprehensive quality metrics
            quality_metrics = self._calculate_quality_metrics(data)
            
            # Add metadata
            timestamp = datetime.now()
            quality_record = {
                'timestamp': timestamp.isoformat(),
                'metrics': quality_metrics,
                'metadata': metadata or {},
                'sample_count': len(data),
                'columns_count': len(data.columns),
                'quality_score': self._calculate_overall_quality_score(quality_metrics)
            }
            
            # Detect quality degradation
            degradation_analysis = self._analyze_quality_degradation(quality_record)
            quality_record['degradation_analysis'] = degradation_analysis
            
            # Store quality record
            self.quality_history.append(quality_record)
            
            # Check for alerts
            alerts = self._check_quality_alerts(quality_record)
            if alerts:
                self._process_quality_alerts(alerts, quality_record)
            
            # Save updated history
            self._save_quality_history()
            
            logger.info(f"✅ Data quality monitored: {quality_record['quality_score']:.3f} "
                       f"({len(data)} samples, {len(data.columns)} columns)")
            
            return quality_record
            
        except Exception as e:
            logger.error(f"❌ Data quality monitoring failed: {e}")
            return None
    
    def _calculate_quality_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive data quality metrics."""
        metrics = {}
        
        try:
            # Completeness metrics
            metrics['completeness'] = 1 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
            
            # Data type consistency
            type_consistency = 0
            for col in data.columns:
                if data[col].dtype in ['object', 'string']:
                    # Check if string columns have consistent formatting
                    unique_ratio = data[col].nunique() / len(data)
                    type_consistency += (1 - unique_ratio) if unique_ratio < 0.9 else 1
                else:
                    type_consistency += 1
            metrics['type_consistency'] = type_consistency / len(data.columns)
            
            # Range validation (for numeric columns)
            range_validity = 0
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                for col in numeric_cols:
                    col_data = data[col].dropna()
                    if len(col_data) > 0:
                        # Check for reasonable ranges (within 3 standard deviations)
                        mean_val = col_data.mean()
                        std_val = col_data.std()
                        if std_val > 0:
                            within_range = ((col_data >= mean_val - 3*std_val) & 
                                          (col_data <= mean_val + 3*std_val)).mean()
                            range_validity += within_range
                metrics['range_validity'] = range_validity / len(numeric_cols)
            else:
                metrics['range_validity'] = 1.0
            
            # Uniqueness metrics
            uniqueness_scores = []
            for col in data.columns:
                if data[col].dtype in ['object', 'string']:
                    # For categorical data, check if values are reasonable
                    unique_ratio = data[col].nunique() / len(data)
                    if unique_ratio < 0.5:  # Not too many unique values
                        uniqueness_scores.append(1.0)
                    else:
                        uniqueness_scores.append(0.5)
                else:
                    # For numeric data, check for reasonable distribution
                    unique_ratio = data[col].nunique() / len(data)
                    if unique_ratio > 0.1:  # Not too few unique values
                        uniqueness_scores.append(1.0)
                    else:
                        uniqueness_scores.append(0.5)
            metrics['uniqueness'] = np.mean(uniqueness_scores)
            
            # Data freshness (if timestamp column exists)
            freshness_score = 1.0
            timestamp_cols = [col for col in data.columns if 'time' in col.lower() or 'date' in col.lower()]
            if timestamp_cols:
                try:
                    # Try to parse the first timestamp column
                    time_col = timestamp_cols[0]
                    if data[time_col].dtype == 'object':
                        # Try to convert to datetime
                        pd.to_datetime(data[time_col], errors='coerce')
                    # Check if data is recent (within threshold)
                    max_time = pd.to_datetime(data[time_col].max())
                    current_time = pd.Timestamp.now()
                    hours_diff = (current_time - max_time).total_seconds() / 3600
                    if hours_diff <= self.config.get('quality_thresholds', {}).get('timeliness_hours', 24):
                        freshness_score = 1.0
                    else:
                        freshness_score = max(0, 1 - (hours_diff / 168))  # Decay over a week
                except:
                    freshness_score = 0.5  # Default if parsing fails
            metrics['freshness'] = freshness_score
            
            # Anomaly detection
            anomaly_score = self._detect_anomalies(data)
            metrics['anomaly_free'] = 1 - anomaly_score
            
            # Overall accuracy (combination of all metrics)
            metrics['overall_accuracy'] = np.mean([
                metrics['completeness'],
                metrics['type_consistency'],
                metrics['range_validity'],
                metrics['uniqueness'],
                metrics['freshness'],
                metrics['anomaly_free']
            ])
            
        except Exception as e:
            logger.error(f"❌ Quality metric calculation failed: {e}")
            metrics = {'error': str(e)}
        
        return metrics
    
    def _detect_anomalies(self, data: pd.DataFrame) -> float:
        """Detect anomalies in the dataset."""
        try:
            anomaly_scores = []
            
            for col in data.select_dtypes(include=[np.number]).columns:
                col_data = data[col].dropna()
                if len(col_data) > 10:
                    # Use IQR method for anomaly detection
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    anomalies = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
                    anomaly_ratio = anomalies / len(col_data)
                    anomaly_scores.append(anomaly_ratio)
            
            return np.mean(anomaly_scores) if anomaly_scores else 0.0
            
        except Exception as e:
            logger.warning(f"⚠️ Anomaly detection failed: {e}")
            return 0.0
    
    def _calculate_overall_quality_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score from individual metrics."""
        try:
            score = 0.0
            count = 0
            
            # Weighted scoring for different quality aspects
            weights = {
                'completeness': 0.25,
                'type_consistency': 0.20,
                'range_validity': 0.20,
                'uniqueness': 0.15,
                'freshness': 0.10,
                'anomaly_free': 0.10
            }
            
            for metric, weight in weights.items():
                if metric in metrics and metrics[metric] is not None:
                    score += metrics[metric] * weight
                    count += weight
            
            return score if count > 0 else 0.0
            
        except Exception as e:
            logger.error(f"❌ Quality score calculation failed: {e}")
            return 0.0
    
    def _analyze_quality_degradation(self, quality_record: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze quality degradation compared to baseline."""
        try:
            if not self.baseline_quality:
                # Set baseline if this is the first record
                self.baseline_quality = quality_record['metrics'].copy()
                return {
                    'degradation_detected': False,
                    'baseline_set': True,
                    'message': 'Baseline quality metrics established'
                }
            
            degradation_analysis = {
                'degradation_detected': False,
                'degraded_metrics': [],
                'degradation_severity': 'none',
                'overall_degradation_score': 0.0
            }
            
            degradation_scores = []
            
            # Compare each metric with baseline
            for metric_name, current_value in quality_record['metrics'].items():
                if (metric_name in self.baseline_quality and 
                    current_value is not None and 
                    self.baseline_quality[metric_name] is not None):
                    
                    baseline_value = self.baseline_quality[metric_name]
                    
                    # Skip non-numeric metrics
                    if not isinstance(current_value, (int, float)) or not isinstance(baseline_value, (int, float)):
                        continue
                    
                    # Calculate degradation (lower quality is worse)
                    if baseline_value > 0:
                        degradation = (baseline_value - current_value) / baseline_value
                    else:
                        degradation = 0
                    
                    # Check if degradation exceeds threshold
                    threshold = self.quality_thresholds.get(f"{metric_name}_min", 0.05)
                    
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
                if degradation_analysis['overall_degradation_score'] > 0.30:
                    degradation_analysis['degradation_severity'] = 'critical'
                elif degradation_analysis['overall_degradation_score'] > 0.15:
                    degradation_analysis['degradation_severity'] = 'moderate'
                else:
                    degradation_analysis['degradation_severity'] = 'minor'
            
            return degradation_analysis
            
        except Exception as e:
            logger.error(f"❌ Quality degradation analysis failed: {e}")
            return {
                'degradation_detected': False,
                'error': str(e)
            }
    
    def _check_quality_alerts(self, quality_record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for quality alerts based on degradation analysis."""
        alerts = []
        
        try:
            degradation_analysis = quality_record.get('degradation_analysis', {})
            
            if degradation_analysis.get('degradation_detected', False):
                severity = degradation_analysis.get('degradation_severity', 'unknown')
                overall_score = degradation_analysis.get('overall_degradation_score', 0.0)
                
                # Create alert
                alert = {
                    'timestamp': quality_record['timestamp'],
                    'alert_type': 'quality_degradation',
                    'severity': severity,
                    'message': f"Data quality degradation detected: {severity} (score: {overall_score:.3f})",
                    'degraded_metrics': degradation_analysis.get('degraded_metrics', []),
                    'overall_degradation_score': overall_score,
                    'quality_score': quality_record.get('quality_score', 0.0)
                }
                
                # Check if this is a retraining trigger
                if overall_score > self.config.get('auto_retraining_threshold', 0.20):
                    alert['retraining_triggered'] = True
                    alert['message'] += " - RETRAINING TRIGGERED"
                
                alerts.append(alert)
            
            # Check for other alert conditions
            quality_score = quality_record.get('quality_score', 0.0)
            if quality_score < 0.7:  # Very low quality
                alerts.append({
                    'timestamp': quality_record['timestamp'],
                    'alert_type': 'low_quality',
                    'severity': 'high',
                    'message': f"Very low data quality detected: {quality_score:.3f}",
                    'quality_score': quality_score
                })
            
            # Check completeness
            completeness = quality_record.get('metrics', {}).get('completeness', 1.0)
            if completeness < 0.8:  # Very low completeness
                alerts.append({
                    'timestamp': quality_record['timestamp'],
                    'alert_type': 'low_completeness',
                    'severity': 'medium',
                    'message': f"Low data completeness detected: {completeness:.3f}",
                    'completeness': completeness
                })
            
        except Exception as e:
            logger.error(f"❌ Quality alert checking failed: {e}")
        
        return alerts
    
    def _process_quality_alerts(self, alerts: List[Dict[str, Any]], 
                               quality_record: Dict[str, Any]):
        """Process and store quality alerts."""
        try:
            for alert in alerts:
                # Add to alert history
                self.alert_history.append(alert)
                
                # Log alert
                logger.warning(f"🚨 QUALITY ALERT: {alert['message']}")
                
                # Save alerts to file
                self._save_alerts()
                
        except Exception as e:
            logger.error(f"❌ Alert processing failed: {e}")
    
    def get_quality_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get quality summary for the specified time period."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Filter recent quality records
            recent_records = [
                record for record in self.quality_history
                if datetime.fromisoformat(record['timestamp']) > cutoff_date
            ]
            
            if not recent_records:
                return {
                    'period_days': days,
                    'total_records': 0,
                    'message': f"No quality data available for the last {days} days"
                }
            
            # Calculate summary statistics
            quality_scores = [record.get('quality_score', 0) for record in recent_records]
            sample_counts = [record.get('sample_count', 0) for record in recent_records]
            
            summary = {
                'period_days': days,
                'total_records': len(recent_records),
                'total_samples': sum(sample_counts),
                'quality_trend': {
                    'mean_score': np.mean(quality_scores),
                    'std_score': np.std(quality_scores),
                    'min_score': np.min(quality_scores),
                    'max_score': np.max(quality_scores),
                    'trend_direction': self._calculate_trend_direction(quality_scores)
                },
                'degradation_summary': {
                    'total_alerts': len([r for r in recent_records if r.get('degradation_analysis', {}).get('degradation_detected', False)]),
                    'critical_degradations': len([r for r in recent_records 
                                               if r.get('degradation_analysis', {}).get('degradation_severity') == 'critical']),
                    'moderate_degradations': len([r for r in recent_records 
                                               if r.get('degradation_analysis', {}).get('degradation_severity') == 'moderate'])
                },
                'recommendations': self._generate_quality_recommendations(recent_records)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Quality summary generation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_trend_direction(self, scores: List[float]) -> str:
        """Calculate the trend direction of quality scores."""
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
    
    def _generate_quality_recommendations(self, recent_records: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on quality analysis."""
        recommendations = []
        
        try:
            # Check for degradation patterns
            degradation_count = len([r for r in recent_records 
                                  if r.get('degradation_analysis', {}).get('degradation_detected', False)])
            
            if degradation_count > len(recent_records) * 0.3:  # More than 30% show degradation
                recommendations.append("🚨 Frequent quality degradation detected - investigate data sources")
            
            # Check quality score trends
            quality_scores = [r.get('quality_score', 0) for r in recent_records]
            if len(quality_scores) >= 5:
                recent_avg = np.mean(quality_scores[-5:])
                overall_avg = np.mean(quality_scores)
                if recent_avg < overall_avg * 0.9:  # Recent quality 10% below average
                    recommendations.append("⚠️ Recent quality below historical average - monitor closely")
            
            # Check for low quality
            low_quality_count = len([r for r in recent_records 
                                   if r.get('quality_score', 0) < 0.7])
            if low_quality_count > 0:
                recommendations.append(f"⚠️ {low_quality_count} recent quality records below acceptable threshold")
            
            if not recommendations:
                recommendations.append("✅ Data quality is stable and within acceptable ranges")
            
        except Exception as e:
            logger.error(f"❌ Recommendation generation failed: {e}")
            recommendations = ["❌ Error generating recommendations"]
        
        return recommendations
    
    def generate_quality_report(self, output_path: Optional[str] = None, 
                               days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        try:
            summary = self.get_quality_summary(days)
            
            report = {
                'quality_report': {
                    'generated_at': datetime.now().isoformat(),
                    'summary': summary,
                    'recent_quality': self.quality_history[-10:] if self.quality_history else [],
                    'recent_alerts': self.alert_history[-10:] if self.alert_history else []
                }
            }
            
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                logger.info(f"📄 Quality report saved to: {output_path}")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Quality report generation failed: {e}")
            return None
    
    def _load_quality_history(self):
        """Load quality history from file."""
        try:
            history_path = "data/quality/quality_history.json"
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    self.quality_history = json.load(f)
                logger.info(f"📚 Loaded quality history: {len(self.quality_history)} records")
            else:
                logger.info("📚 No existing quality history found")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load quality history: {e}")
            self.quality_history = []
    
    def _save_quality_history(self):
        """Save quality history to file."""
        try:
            history_path = "data/quality/quality_history.json"
            os.makedirs(os.path.dirname(history_path), exist_ok=True)
            with open(history_path, 'w') as f:
                json.dump(self.quality_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"❌ Failed to save quality history: {e}")
    
    def _save_alerts(self):
        """Save alerts to file."""
        try:
            alerts_path = "data/quality/alerts.json"
            os.makedirs(os.path.dirname(alerts_path), exist_ok=True)
            with open(alerts_path, 'w') as f:
                json.dump(self.alert_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"❌ Failed to save alerts: {e}")
    
    def get_quality_status(self) -> Dict[str, Any]:
        """Get current quality monitoring status."""
        return {
            'total_quality_records': len(self.quality_history),
            'total_alerts': len(self.alert_history),
            'baseline_established': bool(self.baseline_quality),
            'last_quality_check': self.quality_history[-1]['timestamp'] if self.quality_history else None,
            'recent_alerts': self.alert_history[-5:] if self.alert_history else []
        }

def main():
    """Demo of the data quality monitor."""
    print("🔍 Data Quality Monitor Demo")
    print("="*60)
    
    # Initialize monitor
    monitor = DataQualityMonitor()
    
    print("💡 Available Methods:")
    print("  1. Monitor quality: monitor.monitor_data_quality(dataframe)")
    print("  2. Get summary: monitor.get_quality_summary(days=30)")
    print("  3. Generate report: monitor.generate_quality_report(output_path)")
    print("  4. Get status: monitor.get_quality_status()")
    
    return monitor

if __name__ == "__main__":
    monitor = main()
