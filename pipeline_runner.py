#!/usr/bin/env python3
"""
Production Pipeline Runner for Bronze to Silver ETL
Orchestrates the complete data transformation pipeline with monitoring and error handling
"""

import sys
import os
import argparse
import schedule
import time
import signal
from datetime import datetime, timedelta
from pathlib import Path
import logging
import yaml
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from bronze_to_silver_pipeline import BronzeToSilverPipeline

class PipelineOrchestrator:
    """
    Production orchestrator for the bronze to silver ETL pipeline
    Handles scheduling, monitoring, error recovery, and alerting
    """
    
    def __init__(self, config_path: str = "config/silver_layer_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.pipeline = None
        self.is_running = False
        self.setup_logging()
        self.setup_signal_handlers()
        
    def _load_config(self) -> dict:
        """Load configuration with environment variable substitution"""
        try:
            with open(self.config_path, 'r') as file:
                config_content = file.read()
                
            # Simple environment variable substitution
            import re
            import os
            
            def replace_env_vars(match):
                var_name = match.group(1)
                default_value = match.group(2) if match.group(2) else ""
                return os.getenv(var_name, default_value)
            
            # Replace ${VAR:default} patterns
            config_content = re.sub(r'\$\{([^:}]+):([^}]*)\}', replace_env_vars, config_content)
            # Replace ${VAR} patterns
            config_content = re.sub(r'\$\{([^}]+)\}', lambda m: os.getenv(m.group(1), ""), config_content)
            
            return yaml.safe_load(config_content)
        except Exception as e:
            print(f"Failed to load configuration: {e}")
            sys.exit(1)
    
    def setup_logging(self):
        """Configure comprehensive logging"""
        log_config = self.config.get('logging', {})
        
        # Create logs directory
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            handlers=[]
        )
        
        logger = logging.getLogger()
        
        # File handler
        if log_config.get('handlers', {}).get('file', {}).get('enabled', True):
            file_handler = logging.FileHandler(
                log_config.get('handlers', {}).get('file', {}).get('path', 'logs/pipeline_orchestrator.log')
            )
            file_handler.setFormatter(
                logging.Formatter(log_config.get('format'))
            )
            logger.addHandler(file_handler)
        
        # Console handler
        if log_config.get('handlers', {}).get('console', {}).get('enabled', True):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(
                getattr(logging, log_config.get('handlers', {}).get('console', {}).get('level', 'INFO'))
            )
            console_handler.setFormatter(
                logging.Formatter(log_config.get('format'))
            )
            logger.addHandler(console_handler)
        
        self.logger = logging.getLogger(__name__)
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}. Shutting down gracefully...")
            self.is_running = False
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def initialize_pipeline(self):
        """Initialize the ETL pipeline"""
        try:
            self.pipeline = BronzeToSilverPipeline(self.config_path)
            self.logger.info("Pipeline initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def run_incremental_pipeline(self):
        """Run incremental pipeline processing"""
        try:
            self.logger.info("Starting incremental pipeline run...")
            start_time = datetime.now()
            
            if not self.pipeline:
                self.initialize_pipeline()
            
            processing_config = self.config.get('processing', {})
            lookback_days = processing_config.get('default_lookback_days', 7)
            
            result = self.pipeline.run_pipeline(
                incremental=True, 
                lookback_days=lookback_days
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if result is not None:
                record_count = len(result)
                self.logger.info(f"Incremental pipeline completed: {record_count} records in {duration:.2f}s")
                self._log_pipeline_metrics('incremental', record_count, duration, 'success')
            else:
                self.logger.info("Incremental pipeline completed: No data to process")
                self._log_pipeline_metrics('incremental', 0, duration, 'success')
                
        except Exception as e:
            self.logger.error(f"Incremental pipeline failed: {e}")
            self._log_pipeline_metrics('incremental', 0, 0, 'failed', str(e))
            self._handle_pipeline_failure('incremental', e)
    
    def run_full_refresh(self):
        """Run full refresh pipeline processing"""
        try:
            self.logger.info("Starting full refresh pipeline run...")
            start_time = datetime.now()
            
            if not self.pipeline:
                self.initialize_pipeline()
            
            processing_config = self.config.get('processing', {})
            lookback_days = processing_config.get('max_lookback_days', 30)
            
            result = self.pipeline.run_pipeline(
                incremental=False, 
                lookback_days=lookback_days
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            record_count = len(result) if result is not None else 0
            self.logger.info(f"Full refresh completed: {record_count} records in {duration:.2f}s")
            self._log_pipeline_metrics('full_refresh', record_count, duration, 'success')
            
        except Exception as e:
            self.logger.error(f"Full refresh pipeline failed: {e}")
            self._log_pipeline_metrics('full_refresh', 0, 0, 'failed', str(e))
            self._handle_pipeline_failure('full_refresh', e)
    
    def run_monitoring_check(self):
        """Run monitoring and health checks"""
        try:
            self.logger.debug("Running monitoring checks...")
            
            if not self.pipeline:
                self.initialize_pipeline()
            
            # Check database connectivity
            self._check_database_health()
            
            # Check data freshness
            self._check_data_freshness()
            
            # Check data quality metrics
            self._check_data_quality()
            
            self.logger.debug("Monitoring checks completed successfully")
            
        except Exception as e:
            self.logger.error(f"Monitoring checks failed: {e}")
            self._handle_monitoring_failure(e)
    
    def _check_database_health(self):
        """Check database connectivity and performance"""
        try:
            # Simple connectivity test
            with self.pipeline.engine.connect() as conn:
                result = conn.execute("SELECT 1").fetchone()
                if result[0] != 1:
                    raise Exception("Database connectivity test failed")
        except Exception as e:
            raise Exception(f"Database health check failed: {e}")
    
    def _check_data_freshness(self):
        """Check if data is fresh enough"""
        try:
            freshness_query = """
            SELECT MAX(processed_timestamp) as latest_processed
            FROM ml_lead_scoring.silver_ml_features
            """
            
            with self.pipeline.engine.connect() as conn:
                result = conn.execute(freshness_query).fetchone()
                
            if result and result[0]:
                latest_processed = result[0]
                hours_since_update = (datetime.now() - latest_processed).total_seconds() / 3600
                
                max_staleness = self.config.get('data_quality', {}).get('thresholds', {}).get('timeliness_max_hours', 24)
                
                if hours_since_update > max_staleness:
                    raise Exception(f"Data is stale: {hours_since_update:.1f} hours since last update")
            else:
                raise Exception("No processed data found in silver layer")
                
        except Exception as e:
            raise Exception(f"Data freshness check failed: {e}")
    
    def _check_data_quality(self):
        """Check data quality metrics"""
        try:
            quality_query = """
            SELECT 
                AVG(data_quality_score) as avg_quality,
                COUNT(*) as total_records,
                COUNT(*) FILTER (WHERE data_quality_score < 0.5) as low_quality_records
            FROM ml_lead_scoring.silver_ml_features
            WHERE processed_timestamp > NOW() - INTERVAL '24 hours'
            """
            
            with self.pipeline.engine.connect() as conn:
                result = conn.execute(quality_query).fetchone()
            
            if result:
                avg_quality = result[0] or 0
                total_records = result[1] or 0
                low_quality_records = result[2] or 0
                
                min_quality = self.config.get('data_quality', {}).get('thresholds', {}).get('accuracy_min', 0.9)
                
                if avg_quality < min_quality:
                    raise Exception(f"Data quality below threshold: {avg_quality:.3f} < {min_quality}")
                
                if total_records > 0:
                    low_quality_rate = low_quality_records / total_records
                    if low_quality_rate > 0.1:  # More than 10% low quality
                        raise Exception(f"High rate of low-quality records: {low_quality_rate:.1%}")
                        
        except Exception as e:
            raise Exception(f"Data quality check failed: {e}")
    
    def _log_pipeline_metrics(self, pipeline_type: str, record_count: int, duration: float, 
                             status: str, error_message: str = None):
        """Log pipeline execution metrics"""
        try:
            metrics = {
                'timestamp': datetime.now(),
                'pipeline_type': pipeline_type,
                'record_count': record_count,
                'duration_seconds': duration,
                'status': status,
                'error_message': error_message
            }
            
            # Log to database if configured
            if self.config.get('logging', {}).get('handlers', {}).get('database', {}).get('enabled', False):
                self._write_metrics_to_database(metrics)
            
            # Log performance metrics
            if self.config.get('logging', {}).get('log_performance_metrics', True):
                self.logger.info(f"Pipeline metrics: {metrics}")
                
        except Exception as e:
            self.logger.error(f"Failed to log pipeline metrics: {e}")
    
    def _write_metrics_to_database(self, metrics: dict):
        """Write metrics to database table"""
        try:
            # This would write to a metrics table - implement based on your needs
            pass
        except Exception as e:
            self.logger.error(f"Failed to write metrics to database: {e}")
    
    def _handle_pipeline_failure(self, pipeline_type: str, error: Exception):
        """Handle pipeline failures with retries and alerting"""
        error_config = self.config.get('error_handling', {})
        
        if error_config.get('notify_on_critical_errors', True):
            self._send_alert(f"Pipeline failure: {pipeline_type}", str(error))
        
        # Implement retry logic if needed
        retry_attempts = error_config.get('retry_attempts', 0)
        if retry_attempts > 0:
            self.logger.info(f"Will retry {pipeline_type} pipeline {retry_attempts} times")
    
    def _handle_monitoring_failure(self, error: Exception):
        """Handle monitoring check failures"""
        monitoring_config = self.config.get('monitoring', {})
        
        if monitoring_config.get('alerts', {}).get('database', {}).get('enabled', True):
            self._send_alert("Monitoring check failed", str(error))
    
    def _send_alert(self, subject: str, message: str):
        """Send alert notification"""
        try:
            # Implement based on your alerting preferences
            self.logger.error(f"ALERT: {subject} - {message}")
            
            # Could implement email, Slack, or other notification methods here
            
        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")
    
    def setup_scheduler(self):
        """Setup scheduled pipeline runs"""
        scheduling_config = self.config.get('scheduling', {})
        
        # Schedule incremental runs
        incremental_schedule = scheduling_config.get('incremental_schedule', '0 */6 * * *')
        if incremental_schedule:
            # Convert cron to schedule format (simplified)
            schedule.every(6).hours.do(self.run_incremental_pipeline)
            self.logger.info(f"Scheduled incremental runs every 6 hours")
        
        # Schedule full refresh
        full_refresh_schedule = scheduling_config.get('full_refresh_schedule', '0 2 * * 0')
        if full_refresh_schedule:
            schedule.every().sunday.at("02:00").do(self.run_full_refresh)
            self.logger.info("Scheduled full refresh weekly on Sunday at 2 AM")
        
        # Schedule monitoring checks
        monitoring_schedule = scheduling_config.get('monitoring_schedule', '*/15 * * * *')
        if monitoring_schedule:
            schedule.every(15).minutes.do(self.run_monitoring_check)
            self.logger.info("Scheduled monitoring checks every 15 minutes")
    
    def run_scheduler(self):
        """Run the pipeline scheduler"""
        self.logger.info("Starting pipeline scheduler...")
        self.is_running = True
        
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                self.logger.info("Received interrupt signal")
                break
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                time.sleep(60)
        
        self.logger.info("Pipeline scheduler stopped")

def main():
    """Main execution function with CLI interface"""
    parser = argparse.ArgumentParser(description='Bronze to Silver ETL Pipeline Runner')
    parser.add_argument('--config', default='config/silver_layer_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--mode', choices=['incremental', 'full-refresh', 'monitor', 'scheduler'],
                       default='incremental', help='Execution mode')
    parser.add_argument('--lookback-days', type=int, default=None,
                       help='Days to look back for incremental processing')
    
    args = parser.parse_args()
    
    try:
        # Initialize orchestrator
        orchestrator = PipelineOrchestrator(args.config)
        
        if args.mode == 'incremental':
            orchestrator.run_incremental_pipeline()
        elif args.mode == 'full-refresh':
            orchestrator.run_full_refresh()
        elif args.mode == 'monitor':
            orchestrator.run_monitoring_check()
        elif args.mode == 'scheduler':
            orchestrator.setup_scheduler()
            orchestrator.run_scheduler()
        
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
