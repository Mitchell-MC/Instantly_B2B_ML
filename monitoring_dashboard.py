#!/usr/bin/env python3
"""
Monitoring Dashboard for ML Lead Scoring System
Provides metrics, alerts, and system health monitoring
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
from flask import Flask, render_template, jsonify
import threading
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
api_requests_total = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method', 'status'])
data_processing_duration = Histogram('data_processing_duration_seconds', 'Data processing duration')
model_prediction_duration = Histogram('model_prediction_duration_seconds', 'Model prediction duration')
active_leads_gauge = Gauge('active_leads_total', 'Total number of active leads')
apollo_credits_remaining = Gauge('apollo_credits_remaining', 'Remaining Apollo API credits')
data_quality_score = Gauge('data_quality_score', 'Overall data quality score')
model_performance_gauge = Gauge('model_performance_score', 'Current model performance score')

class MonitoringService:
    def __init__(self, db_config: Dict[str, str]):
        """Initialize monitoring service"""
        self.db_config = db_config
        self.app = Flask(__name__)
        self.alerts_config = self.load_alerts_config()
        self.setup_routes()
        
    def load_alerts_config(self) -> Dict:
        """Load alerting configuration"""
        return {
            'email': {
                'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
                'smtp_port': int(os.getenv('SMTP_PORT', '587')),
                'sender_email': os.getenv('SENDER_EMAIL', ''),
                'sender_password': os.getenv('SENDER_PASSWORD', ''),
                'recipients': os.getenv('ALERT_RECIPIENTS', '').split(',')
            },
            'thresholds': {
                'apollo_credits_warning': 2000,  # Warn when < 2000 credits
                'apollo_credits_critical': 500,   # Critical when < 500 credits
                'data_quality_warning': 0.7,     # Warn when quality < 0.7
                'data_quality_critical': 0.5,    # Critical when quality < 0.5
                'model_performance_warning': 0.6, # Warn when performance < 0.6
                'model_performance_critical': 0.4, # Critical when performance < 0.4
                'pipeline_failure_count': 3       # Alert after 3 consecutive failures
            }
        }
    
    def get_db_connection(self):
        """Create database connection"""
        return psycopg2.connect(**self.db_config)
    
    def setup_routes(self):
        """Setup Flask routes for monitoring dashboard"""
        
        @self.app.route('/')
        def dashboard():
            """Main monitoring dashboard"""
            metrics = self.get_system_metrics()
            return render_template('dashboard.html', metrics=metrics)
        
        @self.app.route('/api/metrics')
        def api_metrics():
            """API endpoint for metrics"""
            return jsonify(self.get_system_metrics())
        
        @self.app.route('/api/alerts')
        def api_alerts():
            """API endpoint for active alerts"""
            return jsonify(self.get_active_alerts())
        
        @self.app.route('/api/data-quality')
        def api_data_quality():
            """API endpoint for data quality metrics"""
            return jsonify(self.get_data_quality_metrics())
        
        @self.app.route('/api/model-performance')
        def api_model_performance():
            """API endpoint for model performance"""
            return jsonify(self.get_model_performance_metrics())
        
        @self.app.route('/metrics')
        def prometheus_metrics():
            """Prometheus metrics endpoint"""
            return generate_latest()
        
        @self.app.route('/health')
        def health_check():
            """Health check endpoint"""
            try:
                # Check database connection
                conn = self.get_db_connection()
                cursor = conn.cursor()
                cursor.execute('SELECT 1')
                cursor.close()
                conn.close()
                
                return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})
            except Exception as e:
                return jsonify({'status': 'unhealthy', 'error': str(e)}), 500
    
    def get_system_metrics(self) -> Dict:
        """Get comprehensive system metrics"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Lead counts by status
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_leads,
                    COUNT(CASE WHEN is_qualified_lead = true THEN 1 END) as qualified_leads,
                    AVG(engagement_score) as avg_engagement_score
                FROM ml_lead_scoring.silver_ml_features
                WHERE updated_timestamp >= NOW() - INTERVAL '30 days'
            """)
            lead_metrics = cursor.fetchone()
            
            # Data ingestion metrics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_ingested_today,
                    MAX(ingestion_timestamp) as last_ingestion
                FROM ml_lead_scoring.bronze_instantly_leads
                WHERE ingestion_timestamp >= CURRENT_DATE
            """)
            ingestion_metrics = cursor.fetchone()
            
            # Apollo usage metrics
            cursor.execute("""
                SELECT 
                    COALESCE(SUM(credits_used), 0) as monthly_usage,
                    COUNT(*) as api_calls_today
                FROM ml_lead_scoring.api_usage_log
                WHERE api_source = 'apollo' 
                AND request_timestamp >= date_trunc('month', CURRENT_DATE)
            """)
            apollo_metrics = cursor.fetchone()
            
            # Model performance metrics
            cursor.execute("""
                SELECT 
                    model_name,
                    version,
                    performance_metrics,
                    created_timestamp
                FROM ml_lead_scoring.model_versions
                WHERE is_active = true
                ORDER BY created_timestamp DESC
                LIMIT 1
            """)
            model_metrics = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            # Update Prometheus metrics
            if lead_metrics:
                active_leads_gauge.set(lead_metrics.get('total_leads', 0))
            
            if apollo_metrics:
                remaining_credits = 20000 - apollo_metrics.get('monthly_usage', 0)
                apollo_credits_remaining.set(remaining_credits)
            
            # Compile metrics
            metrics = {
                'leads': {
                    'total': lead_metrics.get('total_leads', 0) if lead_metrics else 0,
                    'qualified': lead_metrics.get('qualified_leads', 0) if lead_metrics else 0,
                    'avg_engagement': float(lead_metrics.get('avg_engagement_score', 0)) if lead_metrics else 0
                },
                'ingestion': {
                    'today_count': ingestion_metrics.get('total_ingested_today', 0) if ingestion_metrics else 0,
                    'last_ingestion': ingestion_metrics.get('last_ingestion').isoformat() if ingestion_metrics and ingestion_metrics.get('last_ingestion') else None
                },
                'apollo': {
                    'monthly_usage': apollo_metrics.get('monthly_usage', 0) if apollo_metrics else 0,
                    'remaining_credits': 20000 - (apollo_metrics.get('monthly_usage', 0) if apollo_metrics else 0),
                    'calls_today': apollo_metrics.get('api_calls_today', 0) if apollo_metrics else 0
                },
                'model': {
                    'active_model': model_metrics.get('model_name') if model_metrics else None,
                    'version': model_metrics.get('version') if model_metrics else None,
                    'performance': model_metrics.get('performance_metrics') if model_metrics else {},
                    'last_trained': model_metrics.get('created_timestamp').isoformat() if model_metrics and model_metrics.get('created_timestamp') else None
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {str(e)}")
            return {'error': str(e)}
    
    def get_data_quality_metrics(self) -> Dict:
        """Calculate data quality metrics"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Data completeness metrics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(CASE WHEN email IS NOT NULL AND email != '' THEN 1 END) * 100.0 / COUNT(*) as email_completeness,
                    COUNT(CASE WHEN company IS NOT NULL AND company != '' THEN 1 END) * 100.0 / COUNT(*) as company_completeness,
                    COUNT(CASE WHEN first_name IS NOT NULL AND first_name != '' THEN 1 END) * 100.0 / COUNT(*) as name_completeness
                FROM ml_lead_scoring.bronze_instantly_leads
                WHERE ingestion_timestamp >= NOW() - INTERVAL '7 days'
            """)
            completeness_metrics = cursor.fetchone()
            
            # Data freshness
            cursor.execute("""
                SELECT 
                    EXTRACT(HOURS FROM (NOW() - MAX(ingestion_timestamp))) as hours_since_last_update,
                    COUNT(CASE WHEN ingestion_timestamp >= NOW() - INTERVAL '24 hours' THEN 1 END) as records_last_24h
                FROM ml_lead_scoring.bronze_instantly_leads
            """)
            freshness_metrics = cursor.fetchone()
            
            # Apollo enrichment coverage
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT i.lead_id) as total_leads,
                    COUNT(DISTINCT a.lead_id) as enriched_leads
                FROM ml_lead_scoring.bronze_instantly_leads i
                LEFT JOIN ml_lead_scoring.bronze_apollo_enrichment a ON i.lead_id = a.lead_id
                WHERE i.ingestion_timestamp >= NOW() - INTERVAL '30 days'
            """)
            enrichment_metrics = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            # Calculate overall quality score
            completeness_score = (
                (completeness_metrics.get('email_completeness', 0) +
                 completeness_metrics.get('company_completeness', 0) +
                 completeness_metrics.get('name_completeness', 0)) / 300  # Average of three metrics
            )
            
            freshness_score = min(1.0, 24 / max(1, freshness_metrics.get('hours_since_last_update', 24)))
            
            enrichment_coverage = (
                enrichment_metrics.get('enriched_leads', 0) / 
                max(1, enrichment_metrics.get('total_leads', 1))
            )
            
            overall_quality = (completeness_score * 0.4 + freshness_score * 0.3 + enrichment_coverage * 0.3)
            
            # Update Prometheus metric
            data_quality_score.set(overall_quality)
            
            quality_metrics = {
                'overall_score': round(overall_quality, 3),
                'completeness': {
                    'email': round(completeness_metrics.get('email_completeness', 0), 2),
                    'company': round(completeness_metrics.get('company_completeness', 0), 2),
                    'name': round(completeness_metrics.get('name_completeness', 0), 2)
                },
                'freshness': {
                    'hours_since_update': completeness_metrics.get('hours_since_last_update', 0),
                    'records_24h': freshness_metrics.get('records_last_24h', 0)
                },
                'enrichment_coverage': round(enrichment_coverage * 100, 2)
            }
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error calculating data quality: {str(e)}")
            return {'error': str(e)}
    
    def get_model_performance_metrics(self) -> Dict:
        """Get model performance metrics"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get latest model performance
            cursor.execute("""
                SELECT 
                    model_name,
                    version,
                    performance_metrics,
                    training_data_size,
                    created_timestamp
                FROM ml_lead_scoring.model_versions
                WHERE is_active = true
                ORDER BY created_timestamp DESC
                LIMIT 1
            """)
            current_model = cursor.fetchone()
            
            # Get prediction statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_predictions,
                    AVG(predicted_score) as avg_predicted_score,
                    AVG(confidence_level) as avg_confidence,
                    COUNT(CASE WHEN predicted_score > 0.7 THEN 1 END) as high_score_predictions
                FROM ml_lead_scoring.lead_predictions
                WHERE prediction_timestamp >= NOW() - INTERVAL '7 days'
            """)
            prediction_stats = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            performance_metrics = {}
            
            if current_model:
                perf_data = current_model.get('performance_metrics', {})
                if isinstance(perf_data, str):
                    perf_data = json.loads(perf_data)
                
                f1_score = perf_data.get('f1_score', 0)
                model_performance_gauge.set(f1_score)
                
                performance_metrics = {
                    'model_info': {
                        'name': current_model.get('model_name'),
                        'version': current_model.get('version'),
                        'training_size': current_model.get('training_data_size'),
                        'trained_at': current_model.get('created_timestamp').isoformat() if current_model.get('created_timestamp') else None
                    },
                    'performance': perf_data,
                    'predictions': {
                        'total_week': prediction_stats.get('total_predictions', 0) if prediction_stats else 0,
                        'avg_score': round(prediction_stats.get('avg_predicted_score', 0), 3) if prediction_stats else 0,
                        'avg_confidence': round(prediction_stats.get('avg_confidence', 0), 3) if prediction_stats else 0,
                        'high_score_count': prediction_stats.get('high_score_predictions', 0) if prediction_stats else 0
                    }
                }
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"Error getting model performance: {str(e)}")
            return {'error': str(e)}
    
    def get_active_alerts(self) -> List[Dict]:
        """Get active system alerts"""
        alerts = []
        
        try:
            metrics = self.get_system_metrics()
            quality_metrics = self.get_data_quality_metrics()
            performance_metrics = self.get_model_performance_metrics()
            
            # Check Apollo credits
            remaining_credits = metrics.get('apollo', {}).get('remaining_credits', 0)
            if remaining_credits < self.alerts_config['thresholds']['apollo_credits_critical']:
                alerts.append({
                    'type': 'critical',
                    'category': 'apollo_credits',
                    'message': f'Apollo credits critically low: {remaining_credits} remaining',
                    'timestamp': datetime.now().isoformat()
                })
            elif remaining_credits < self.alerts_config['thresholds']['apollo_credits_warning']:
                alerts.append({
                    'type': 'warning',
                    'category': 'apollo_credits',
                    'message': f'Apollo credits running low: {remaining_credits} remaining',
                    'timestamp': datetime.now().isoformat()
                })
            
            # Check data quality
            quality_score = quality_metrics.get('overall_score', 1.0)
            if quality_score < self.alerts_config['thresholds']['data_quality_critical']:
                alerts.append({
                    'type': 'critical',
                    'category': 'data_quality',
                    'message': f'Data quality critically low: {quality_score:.2f}',
                    'timestamp': datetime.now().isoformat()
                })
            elif quality_score < self.alerts_config['thresholds']['data_quality_warning']:
                alerts.append({
                    'type': 'warning',
                    'category': 'data_quality',
                    'message': f'Data quality below threshold: {quality_score:.2f}',
                    'timestamp': datetime.now().isoformat()
                })
            
            # Check model performance
            if 'performance' in performance_metrics:
                f1_score = performance_metrics['performance'].get('f1_score', 1.0)
                if f1_score < self.alerts_config['thresholds']['model_performance_critical']:
                    alerts.append({
                        'type': 'critical',
                        'category': 'model_performance',
                        'message': f'Model performance critically low: F1={f1_score:.3f}',
                        'timestamp': datetime.now().isoformat()
                    })
                elif f1_score < self.alerts_config['thresholds']['model_performance_warning']:
                    alerts.append({
                        'type': 'warning',
                        'category': 'model_performance',
                        'message': f'Model performance below threshold: F1={f1_score:.3f}',
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Check data freshness
            hours_since_update = quality_metrics.get('freshness', {}).get('hours_since_update', 0)
            if hours_since_update > 48:  # No data for 2 days
                alerts.append({
                    'type': 'critical',
                    'category': 'data_freshness',
                    'message': f'No data ingestion for {hours_since_update:.1f} hours',
                    'timestamp': datetime.now().isoformat()
                })
            elif hours_since_update > 24:  # No data for 1 day
                alerts.append({
                    'type': 'warning',
                    'category': 'data_freshness',
                    'message': f'Data ingestion delayed: {hours_since_update:.1f} hours since last update',
                    'timestamp': datetime.now().isoformat()
                })
            
        except Exception as e:
            logger.error(f"Error checking alerts: {str(e)}")
            alerts.append({
                'type': 'error',
                'category': 'system',
                'message': f'Error checking system alerts: {str(e)}',
                'timestamp': datetime.now().isoformat()
            })
        
        return alerts
    
    def send_alert_notification(self, alert: Dict):
        """Send alert notification via email"""
        try:
            if not self.alerts_config['email']['sender_email'] or not self.alerts_config['email']['recipients']:
                logger.warning("Email configuration incomplete, skipping notification")
                return
            
            msg = MIMEMultipart()
            msg['From'] = self.alerts_config['email']['sender_email']
            msg['To'] = ', '.join(self.alerts_config['email']['recipients'])
            msg['Subject'] = f"ML Lead Scoring Alert - {alert['type'].upper()}: {alert['category']}"
            
            body = f"""
            Alert Type: {alert['type'].upper()}
            Category: {alert['category']}
            Message: {alert['message']}
            Timestamp: {alert['timestamp']}
            
            Please check the monitoring dashboard for more details.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.alerts_config['email']['smtp_server'], self.alerts_config['email']['smtp_port'])
            server.starttls()
            server.login(self.alerts_config['email']['sender_email'], self.alerts_config['email']['sender_password'])
            
            text = msg.as_string()
            server.sendmail(self.alerts_config['email']['sender_email'], self.alerts_config['email']['recipients'], text)
            server.quit()
            
            logger.info(f"Alert notification sent: {alert['category']}")
            
        except Exception as e:
            logger.error(f"Error sending alert notification: {str(e)}")
    
    def run_monitoring_loop(self):
        """Run continuous monitoring loop"""
        last_alert_check = {}
        
        while True:
            try:
                # Check for alerts
                alerts = self.get_active_alerts()
                
                # Send notifications for new critical alerts
                for alert in alerts:
                    if alert['type'] == 'critical':
                        alert_key = f"{alert['category']}_{alert['type']}"
                        
                        # Avoid spamming - send alert max once per hour
                        if alert_key not in last_alert_check or \
                           (datetime.now() - last_alert_check[alert_key]).seconds > 3600:
                            self.send_alert_notification(alert)
                            last_alert_check[alert_key] = datetime.now()
                
                # Update metrics
                self.get_system_metrics()
                self.get_data_quality_metrics()
                self.get_model_performance_metrics()
                
                # Sleep for 5 minutes
                time.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(60)  # Sleep for 1 minute on error
    
    def run_service(self, host='0.0.0.0', port=5002):
        """Run the monitoring service"""
        logger.info(f"Starting monitoring service on {host}:{port}")
        
        # Start monitoring loop in background
        monitoring_thread = threading.Thread(target=self.run_monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        # Start Flask app
        self.app.run(host=host, port=port, debug=False, threaded=True)

if __name__ == "__main__":
    # Database configuration
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME', 'ml_lead_scoring'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'password')
    }
    
    # Initialize and run monitoring service
    monitoring_service = MonitoringService(db_config)
    monitoring_service.run_service()
