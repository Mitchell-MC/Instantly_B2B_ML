"""
Production Monitoring Dashboard
Real-time monitoring, automated alerting, and dashboard for MLOps pipeline.
"""

import os
import json
import time
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import yaml
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from advanced_drift_detection import AdvancedDriftDetector
from deploy import ChampionChallengerDeployment
from auto_retrain import AutomatedRetraining
from model_performance_tracker import ModelPerformanceTracker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionMonitor:
    """
    Production monitoring system with real-time alerts and dashboard.
    
    Implements:
    - Continuous monitoring loop
    - Automated alerting
    - Performance tracking
    - Drift monitoring
    - Health checks
    - Dashboard generation
    """
    
    def __init__(self, config_path="config/main_config.yaml"):
        """Initialize the production monitor."""
        self.config = self._load_config(config_path)
        self.drift_detector = AdvancedDriftDetector(self.config.get('drift_detection', {}))
        self.deployer = ChampionChallengerDeployment(config_path)
        self.retrainer = AutomatedRetraining(config_path)
        self.performance_tracker = ModelPerformanceTracker(config_path)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.monitoring_interval = self.config['monitoring'].get('monitoring_interval', 3600)  # 1 hour
        self.last_monitoring_run = None
        
        # Alert thresholds
        self.alert_thresholds = self.config['monitoring'].get('alert_thresholds', {})
        
        # Performance history
        self.performance_history = []
        self.alert_history = []
        
        # Load monitoring history
        self._load_monitoring_history()
        
        logger.info("📊 Production monitor initialized")
    
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Add default monitoring configuration if not present
            if 'monitoring' not in config:
                config['monitoring'] = {
                    'monitoring_interval': 3600,        # 1 hour
                    'alert_thresholds': {
                        'drift_critical': 0.25,         # PSI > 0.25
                        'drift_warning': 0.10,          # PSI > 0.10
                        'performance_degradation': 0.05, # 5% degradation
                        'model_health_score': 0.7       # Minimum health score
                    },
                    'alert_channels': ['log', 'file'],  # Alert output channels
                    'dashboard_refresh_interval': 300,   # 5 minutes
                    'max_history_days': 30              # Keep 30 days of history
                }
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            # Default configuration
            return {
                'monitoring': {
                    'monitoring_interval': 3600,
                    'alert_thresholds': {
                        'drift_critical': 0.25,
                        'drift_warning': 0.10,
                        'performance_degradation': 0.05,
                        'model_health_score': 0.7
                    },
                    'alert_channels': ['log', 'file'],
                    'dashboard_refresh_interval': 300,
                    'max_history_days': 30
                }
            }
    
    def _load_monitoring_history(self):
        """Load monitoring history from file."""
        history_file = "data/monitoring_history.json"
        try:
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.performance_history = data.get('performance_history', [])
                    self.alert_history = data.get('alert_history', [])
                
                logger.info(f"📚 Loaded monitoring history: {len(self.performance_history)} performance records, {len(self.alert_history)} alerts")
        except Exception as e:
            logger.warning(f"⚠️ Could not load monitoring history: {e}")
    
    def _save_monitoring_history(self):
        """Save monitoring history to file."""
        try:
            history_file = "data/monitoring_history.json"
            os.makedirs(os.path.dirname(history_file), exist_ok=True)
            
            data = {
                'performance_history': self.performance_history,
                'alert_history': self.alert_history,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Failed to save monitoring history: {e}")
    
    def start_monitoring(self):
        """Start continuous monitoring in background thread."""
        if self.is_monitoring:
            logger.warning("⚠️ Monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("🚀 Production monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("🛑 Production monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Run monitoring cycle
                self._run_monitoring_cycle()
                
                # Wait for next cycle
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"❌ Monitoring cycle failed: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _run_monitoring_cycle(self):
        """Execute one monitoring cycle."""
        logger.info("🔍 Running monitoring cycle...")
        
        try:
            # Get current monitoring data
            monitoring_data = self._collect_monitoring_data()
            
            # Analyze for alerts
            alerts = self._analyze_for_alerts(monitoring_data)
            
            # Process alerts
            if alerts:
                self._process_alerts(alerts)
            
            # Update performance history
            self._update_performance_history(monitoring_data)
            
            # Generate dashboard
            self._generate_dashboard(monitoring_data)
            
            # Update last run time
            self.last_monitoring_run = datetime.now()
            
            logger.info(f"✅ Monitoring cycle complete: {len(alerts)} alerts generated")
            
        except Exception as e:
            logger.error(f"❌ Monitoring cycle failed: {e}")
    
    def _collect_monitoring_data(self):
        """Collect current monitoring data."""
        monitoring_data = {
            'timestamp': datetime.now().isoformat(),
            'system_health': self._check_system_health(),
            'model_performance': self._get_model_performance(),
            'data_drift': self._check_data_drift(),
            'deployment_status': self.deployer.get_deployment_status(),
            'retraining_status': self.retrainer.get_retraining_status()
        }
        
        return monitoring_data
    
    def _check_system_health(self):
        """Check overall system health."""
        try:
            health_score = 1.0
            health_issues = []
            
            # Check if model files exist
            model_path = self.config.get('paths', {}).get('model_artifact', 'models/email_open_predictor_v1.0.joblib')
            if not os.path.exists(model_path):
                health_score -= 0.3
                health_issues.append("Model file missing")
            
            # Check data directory
            data_dir = self.config.get('paths', {}).get('data_dir', 'data')
            if not os.path.exists(data_dir):
                health_score -= 0.2
                health_issues.append("Data directory missing")
            
            # Check logs directory
            logs_dir = self.config.get('logging', {}).get('file', 'logs/pipeline.log')
            logs_dir = os.path.dirname(logs_dir)
            if not os.path.exists(logs_dir):
                health_score -= 0.1
                health_issues.append("Logs directory missing")
            
            return {
                'overall_score': max(0.0, health_score),
                'issues': health_issues,
                'status': 'healthy' if health_score >= 0.8 else 'warning' if health_score >= 0.6 else 'critical'
            }
            
        except Exception as e:
            logger.error(f"❌ System health check failed: {e}")
            return {
                'overall_score': 0.0,
                'issues': [f"Health check failed: {str(e)}"],
                'status': 'critical'
            }
    
    def _get_model_performance(self):
        """Get current model performance metrics."""
        try:
            # This would typically come from real-time predictions
            # For now, we'll use placeholder data
            performance = {
                'accuracy': 0.78,  # Placeholder
                'auc': 0.83,       # Placeholder
                'f1_score': 0.76,  # Placeholder
                'prediction_count': 0,  # Would be updated in real-time
                'last_updated': datetime.now().isoformat()
            }
            
            return performance
            
        except Exception as e:
            logger.error(f"❌ Model performance check failed: {e}")
            return {
                'accuracy': 0.0,
                'auc': 0.0,
                'f1_score': 0.0,
                'error': str(e)
            }
    
    def _check_data_drift(self):
        """Check for data drift in recent data."""
        try:
            # This would typically compare recent data with training data
            # For now, we'll return placeholder data
            drift_status = {
                'overall_assessment': 'no_drift',
                'last_check': datetime.now().isoformat(),
                'features_checked': 0,
                'drift_detected': False
            }
            
            return drift_status
            
        except Exception as e:
            logger.error(f"❌ Data drift check failed: {e}")
            return {
                'overall_assessment': 'check_failed',
                'error': str(e),
                'drift_detected': False
            }
    
    def _analyze_for_alerts(self, monitoring_data):
        """Analyze monitoring data for alert conditions."""
        alerts = []
        
        try:
            # Check system health
            system_health = monitoring_data['system_health']
            if system_health['overall_score'] < self.alert_thresholds['model_health_score']:
                alerts.append({
                    'level': 'critical',
                    'category': 'system_health',
                    'message': f"System health critical: {system_health['overall_score']:.2f}",
                    'timestamp': datetime.now().isoformat(),
                    'details': system_health
                })
            
            # Check model performance
            model_performance = monitoring_data['model_performance']
            if 'accuracy' in model_performance and model_performance['accuracy'] < 0.7:
                alerts.append({
                    'level': 'warning',
                    'category': 'model_performance',
                    'message': f"Model accuracy below threshold: {model_performance['accuracy']:.3f}",
                    'timestamp': datetime.now().isoformat(),
                    'details': model_performance
                })
            
            # Check data drift
            data_drift = monitoring_data['data_drift']
            if data_drift.get('drift_detected', False):
                alerts.append({
                    'level': 'warning',
                    'category': 'data_drift',
                    'message': "Data drift detected",
                    'timestamp': datetime.now().isoformat(),
                    'details': data_drift
                })
            
            # Check deployment status
            deployment_status = monitoring_data['deployment_status']
            if deployment_status.get('total_rollbacks', 0) > 0:
                alerts.append({
                    'level': 'info',
                    'category': 'deployment',
                    'message': f"Model rollbacks detected: {deployment_status['total_rollbacks']}",
                    'timestamp': datetime.now().isoformat(),
                    'details': deployment_status
                })
            
        except Exception as e:
            logger.error(f"❌ Alert analysis failed: {e}")
            alerts.append({
                'level': 'critical',
                'category': 'monitoring_error',
                'message': f"Monitoring analysis failed: {str(e)}",
                'timestamp': datetime.now().isoformat(),
                'details': {'error': str(e)}
            })
        
        return alerts
    
    def _process_alerts(self, alerts):
        """Process and log alerts."""
        for alert in alerts:
            # Add to alert history
            self.alert_history.append(alert)
            
            # Log alert
            log_message = f"🚨 {alert['level'].upper()}: {alert['message']}"
            if alert['level'] == 'critical':
                logger.critical(log_message)
            elif alert['level'] == 'warning':
                logger.warning(log_message)
            else:
                logger.info(log_message)
            
            # Save to alert file if configured
            if 'file' in self.config['monitoring']['alert_channels']:
                self._save_alert_to_file(alert)
        
        # Keep only recent alerts
        max_alerts = self.config['monitoring']['max_history_days'] * 24  # Approximate daily alerts
        if len(self.alert_history) > max_alerts:
            self.alert_history = self.alert_history[-max_alerts:]
    
    def _save_alert_to_file(self, alert):
        """Save alert to file."""
        try:
            alert_file = "data/alerts.jsonl"
            os.makedirs(os.path.dirname(alert_file), exist_ok=True)
            
            with open(alert_file, 'a') as f:
                f.write(json.dumps(alert) + '\n')
                
        except Exception as e:
            logger.error(f"❌ Failed to save alert to file: {e}")
    
    def _update_performance_history(self, monitoring_data):
        """Update performance history."""
        # Add to history
        self.performance_history.append(monitoring_data)
        
        # Keep only recent history
        max_history = self.config['monitoring']['max_history_days'] * 24  # Approximate daily records
        if len(self.performance_history) > max_history:
            self.performance_history = self.performance_history[-max_history:]
        
        # Save history
        self._save_monitoring_history()
    
    def _generate_dashboard(self, monitoring_data):
        """Generate monitoring dashboard."""
        try:
            dashboard = {
                'timestamp': datetime.now().isoformat(),
                'system_status': {
                    'monitoring_active': self.is_monitoring,
                    'last_monitoring_run': self.last_monitoring_run.isoformat() if self.last_monitoring_run else None,
                    'monitoring_interval': self.monitoring_interval
                },
                'current_metrics': monitoring_data,
                'summary': {
                    'total_alerts': len(self.alert_history),
                    'critical_alerts': len([a for a in self.alert_history if a['level'] == 'critical']),
                    'warning_alerts': len([a for a in self.alert_history if a['level'] == 'warning']),
                    'performance_records': len(self.performance_history)
                }
            }
            
            # Save dashboard
            dashboard_file = "data/monitoring_dashboard.json"
            os.makedirs(os.path.dirname(dashboard_file), exist_ok=True)
            
            with open(dashboard_file, 'w') as f:
                json.dump(dashboard, f, indent=2)
            
            logger.debug("📊 Dashboard generated and saved")
            
        except Exception as e:
            logger.error(f"❌ Dashboard generation failed: {e}")
    
    def get_monitoring_status(self):
        """Get current monitoring status."""
        return {
            'is_monitoring': self.is_monitoring,
            'monitoring_interval': self.monitoring_interval,
            'last_monitoring_run': self.last_monitoring_run.isoformat() if self.last_monitoring_run else None,
            'total_performance_records': len(self.performance_history),
            'total_alerts': len(self.alert_history),
            'recent_alerts': self.alert_history[-5:] if self.alert_history else []
        }
    
    def track_model_performance(self, y_true, y_pred, y_pred_proba=None, 
                               sample_weights=None, metadata=None):
        """
        Track model performance using the performance tracker.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for classification)
            sample_weights: Sample weights for weighted metrics
            metadata: Additional metadata (model_version, timestamp, etc.)
            
        Returns:
            dict: Performance tracking results
        """
        try:
            performance_result = self.performance_tracker.track_performance(
                y_true, y_pred, y_pred_proba, sample_weights, metadata
            )
            
            if performance_result:
                logger.info(f"✅ Model performance tracked successfully")
                return performance_result
            else:
                logger.error("❌ Failed to track model performance")
                return None
                
        except Exception as e:
            logger.error(f"❌ Model performance tracking failed: {e}")
            return None
    
    def get_performance_summary(self, days=30):
        """Get performance summary from the performance tracker."""
        try:
            return self.performance_tracker.get_performance_summary(days)
        except Exception as e:
            logger.error(f"❌ Failed to get performance summary: {e}")
            return None
    
    def generate_performance_report(self, output_path=None, days=30):
        """Generate performance report from the performance tracker."""
        try:
            return self.performance_tracker.generate_performance_report(output_path, days)
        except Exception as e:
            logger.error(f"❌ Failed to generate performance report: {e}")
            return None
    
    def run_manual_monitoring_cycle(self):
        """Run a manual monitoring cycle."""
        logger.info("🔍 Running manual monitoring cycle...")
        self._run_monitoring_cycle()
        return self.get_monitoring_status()
    
    def get_alert_summary(self, days=7):
        """Get alert summary for the last N days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            recent_alerts = [
                alert for alert in self.alert_history
                if datetime.fromisoformat(alert['timestamp']) >= cutoff_date
            ]
            
            # Group by level and category
            summary = {
                'total_alerts': len(recent_alerts),
                'by_level': {},
                'by_category': {},
                'recent_alerts': recent_alerts[-10:]  # Last 10 alerts
            }
            
            for alert in recent_alerts:
                level = alert['level']
                category = alert['category']
                
                summary['by_level'][level] = summary['by_level'].get(level, 0) + 1
                summary['by_category'][category] = summary['by_category'].get(category, 0) + 1
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Alert summary generation failed: {e}")
            return {'error': str(e)}
    
    def generate_monitoring_report(self, output_path=None):
        """Generate comprehensive monitoring report."""
        try:
            # Get current status
            status = self.get_monitoring_status()
            alert_summary = self.get_alert_summary(30)  # Last 30 days
            
            report = {
                'monitoring_report': {
                    'timestamp': datetime.now().isoformat(),
                    'monitoring_status': status,
                    'alert_summary': alert_summary,
                    'recommendations': self._generate_monitoring_recommendations(status, alert_summary)
                }
            }
            
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"📄 Monitoring report saved to: {output_path}")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Failed to generate monitoring report: {e}")
            return None
    
    def _generate_monitoring_recommendations(self, status, alert_summary):
        """Generate recommendations based on monitoring data."""
        recommendations = []
        
        # Check monitoring status
        if not status['is_monitoring']:
            recommendations.append("🚨 Start continuous monitoring for production safety")
        
        # Check alert levels
        critical_alerts = alert_summary.get('by_level', {}).get('critical', 0)
        if critical_alerts > 0:
            recommendations.append(f"🚨 Address {critical_alerts} critical alerts immediately")
        
        warning_alerts = alert_summary.get('by_level', {}).get('warning', 0)
        if warning_alerts > 5:
            recommendations.append(f"⚠️ High number of warnings ({warning_alerts}) - investigate patterns")
        
        # Check performance records
        if status['total_performance_records'] < 10:
            recommendations.append("📊 Limited performance history - continue monitoring to establish baselines")
        
        # Check monitoring frequency
        if status['monitoring_interval'] > 7200:  # More than 2 hours
            recommendations.append("⏰ Consider increasing monitoring frequency for better responsiveness")
        
        if not recommendations:
            recommendations.append("✅ Monitoring system healthy - continue current practices")
        
        return recommendations
    
    def cleanup_old_data(self):
        """Clean up old monitoring data."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config['monitoring']['max_history_days'])
            
            # Clean performance history
            original_count = len(self.performance_history)
            self.performance_history = [
                record for record in self.performance_history
                if datetime.fromisoformat(record['timestamp']) >= cutoff_date
            ]
            removed_performance = original_count - len(self.performance_history)
            
            # Clean alert history
            original_count = len(self.alert_history)
            self.alert_history = [
                alert for alert in self.alert_history
                if datetime.fromisoformat(alert['timestamp']) >= cutoff_date
            ]
            removed_alerts = original_count - len(self.alert_history)
            
            if removed_performance > 0 or removed_alerts > 0:
                logger.info(f"🧹 Cleaned up {removed_performance} performance records and {removed_alerts} alerts")
                self._save_monitoring_history()
            
        except Exception as e:
            logger.error(f"❌ Data cleanup failed: {e}")

def main():
    """Demo of the production monitoring system."""
    print("📊 Production Monitoring System Demo")
    print("="*60)
    
    # Initialize monitor
    monitor = ProductionMonitor()
    
    print("💡 Available Methods:")
    print("  1. Start monitoring: monitor.start_monitoring()")
    print("  2. Stop monitoring: monitor.stop_monitoring()")
    print("  3. Manual cycle: monitor.run_manual_monitoring_cycle()")
    print("  4. Get status: monitor.get_monitoring_status()")
    print("  5. Generate report: monitor.generate_monitoring_report()")
    
    # Show current status
    status = monitor.get_monitoring_status()
    print(f"\n📊 Current Status:")
    print(f"  Monitoring Active: {'✅ YES' if status['is_monitoring'] else '❌ NO'}")
    print(f"  Total Alerts: {status['total_alerts']}")
    print(f"  Performance Records: {status['total_performance_records']}")
    
    return monitor

if __name__ == "__main__":
    monitor = main()
