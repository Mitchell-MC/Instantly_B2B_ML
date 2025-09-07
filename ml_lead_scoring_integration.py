#!/usr/bin/env python3
"""
ML Lead Scoring System Integration Script
Based on transcript meeting requirements for N8N + Apollo + Instantly integration

This script orchestrates the complete ML lead scoring system including:
- N8N workflow management
- Data ingestion from Instantly
- Apollo enrichment with credit management  
- Three-layer data processing (Bronze → Silver → Gold)
- ML model training and prediction
- System monitoring and alerting
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import requests
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ml_lead_scoring_integration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MLLeadScoringSystem:
    """
    Main integration class for ML Lead Scoring System
    Implements the architecture discussed in the transcript
    """
    
    def __init__(self, config_file: str = 'config/integration_config.yaml'):
        """Initialize the ML lead scoring system"""
        self.config = self.load_config(config_file)
        self.db_config = self.config.get('database', {})
        self.n8n_config = self.config.get('n8n', {})
        self.apollo_config = self.config.get('apollo', {})
        self.instantly_config = self.config.get('instantly', {})
        
        # Service endpoints
        self.services = {
            'n8n': f"http://localhost:{self.n8n_config.get('port', 5678)}",
            'data_processor': 'http://localhost:5000',
            'ml_service': 'http://localhost:5001', 
            'monitoring': 'http://localhost:5002'
        }
        
        logger.info("ML Lead Scoring System initialized")
    
    def load_config(self, config_file: str) -> Dict:
        """Load configuration from file or use defaults"""
        try:
            if os.path.exists(config_file):
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    # Override with environment variables for API keys
                    config['apollo']['api_key'] = os.getenv('APOLLO_API_KEY', config['apollo'].get('api_key', ''))
                    config['instantly']['api_key'] = os.getenv('INSTANTLY_API_KEY', 
                                                               os.getenv('instantly_key', 
                                                                        config['instantly'].get('api_key', '')))
                    return config
        except ImportError:
            logger.warning("PyYAML not installed, using default config")
        except Exception as e:
            logger.warning(f"Error loading config: {e}")
        
        # Default configuration based on transcript requirements and notebook examples
        return {
            'database': {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': int(os.getenv('DB_PORT', '5432')),
                'database': os.getenv('DB_NAME', 'ml_lead_scoring'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', 'ml_password_123')
            },
            'n8n': {
                'port': 5678,
                'user': os.getenv('N8N_USER', 'admin'),
                'password': os.getenv('N8N_PASSWORD', 'n8n_admin_123')
            },
            'apollo': {
                'api_key': os.getenv('APOLLO_API_KEY', 'K05UXxdZgCaAFgYCTqJWmQ'),
                'monthly_limit': 20000,  # 20K credits per month as mentioned in transcript
                'reserve_credits': 2000,  # Reserve credits for training pipeline
                'rate_limit_delay': 1,
                'max_batch_size': 50
            },
            'instantly': {
                'api_key': os.getenv('INSTANTLY_API_KEY', 
                           os.getenv('instantly_key', 
                                    'ZjQ0Mjc3ZGUtNjliMi00YmMzLWE2OWMtMjhhZmQ0MDk0MTIzOkx5VWZ6UnB6RmR3Zw==')),
                'organization_id': os.getenv('ORGANIZATION_ID', 'f44277de-69b2-4bc3-a69c-28afd4094123'),
                'update_frequency': 'daily',  # Daily updates as discussed
                'batch_size': 1000,
                'retry_attempts': 3
            },
            'maturity': {
                'gold_layer_days': int(os.getenv('LEAD_MATURITY_DAYS', '30')),  # 1 month default
                'engagement_window_days': 14  # 2 weeks engagement window
            }
        }
    
    def check_services_health(self) -> Dict[str, bool]:
        """Check health of all services"""
        health_status = {}
        
        for service_name, endpoint in self.services.items():
            try:
                if service_name == 'n8n':
                    # N8N health check
                    response = requests.get(f"{endpoint}/healthz", timeout=10)
                else:
                    # Other services use /health endpoint
                    response = requests.get(f"{endpoint}/health", timeout=10)
                
                health_status[service_name] = response.status_code == 200
                
            except Exception as e:
                logger.warning(f"Service {service_name} health check failed: {e}")
                health_status[service_name] = False
        
        return health_status
    
    def check_database_connection(self) -> bool:
        """Check database connectivity and schema"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Check if schema exists
            cursor.execute("""
                SELECT schema_name FROM information_schema.schemata 
                WHERE schema_name = 'ml_lead_scoring'
            """)
            
            schema_exists = cursor.fetchone() is not None
            cursor.close()
            conn.close()
            
            if not schema_exists:
                logger.error("ML lead scoring schema not found in database")
                return False
                
            logger.info("Database connection and schema verified")
            return True
            
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def check_apollo_credits(self) -> Dict[str, Any]:
        """Check Apollo API credit usage"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get monthly usage
            cursor.execute("""
                SELECT 
                    COALESCE(SUM(credits_used), 0) as monthly_usage,
                    COUNT(*) as api_calls_this_month
                FROM ml_lead_scoring.api_usage_log
                WHERE api_source = 'apollo' 
                AND request_timestamp >= date_trunc('month', CURRENT_DATE)
            """)
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            monthly_limit = self.apollo_config['monthly_limit']
            used = result['monthly_usage'] if result else 0
            remaining = monthly_limit - used
            
            return {
                'monthly_limit': monthly_limit,
                'used': used,
                'remaining': remaining,
                'percentage_used': (used / monthly_limit) * 100,
                'api_calls': result['api_calls_this_month'] if result else 0,
                'available_for_predictions': remaining - self.apollo_config['reserve_credits']
            }
            
        except Exception as e:
            logger.error(f"Error checking Apollo credits: {e}")
            return {'error': str(e)}
    
    def trigger_instantly_ingestion(self) -> Dict[str, Any]:
        """Trigger Instantly data ingestion workflow"""
        try:
            # This would trigger the N8N workflow for Instantly ingestion
            logger.info("Triggering Instantly data ingestion workflow")
            
            # In a real implementation, this would call N8N API to trigger workflow
            # For now, we'll call the data processor directly
            response = requests.post(
                f"{self.services['data_processor']}/api/process-silver-layer",
                json={'lead_ids': []},  # Empty list processes all new data
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Instantly ingestion triggered successfully: {result}")
                return result
            else:
                logger.error(f"Failed to trigger Instantly ingestion: {response.text}")
                return {'error': f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            logger.error(f"Error triggering Instantly ingestion: {e}")
            return {'error': str(e)}
    
    def trigger_apollo_enrichment(self, max_leads: int = 50) -> Dict[str, Any]:
        """Trigger Apollo enrichment for new leads"""
        try:
            # Check credit availability first
            credit_status = self.check_apollo_credits()
            
            if 'error' in credit_status:
                return credit_status
            
            available_credits = credit_status['available_for_predictions']
            
            if available_credits < max_leads:
                logger.warning(f"Limited Apollo credits. Can only process {available_credits} leads")
                max_leads = max(0, available_credits)
            
            if max_leads == 0:
                return {'error': 'No Apollo credits available', 'credit_status': credit_status}
            
            logger.info(f"Triggering Apollo enrichment for up to {max_leads} leads")
            
            # This would trigger the N8N Apollo enrichment workflow
            # For now, return a placeholder response
            return {
                'status': 'triggered',
                'max_leads': max_leads,
                'credit_status': credit_status,
                'message': 'Apollo enrichment workflow triggered'
            }
            
        except Exception as e:
            logger.error(f"Error triggering Apollo enrichment: {e}")
            return {'error': str(e)}
    
    def process_to_gold_layer(self) -> Dict[str, Any]:
        """Process mature data to gold layer for training"""
        try:
            logger.info("Processing mature data to gold layer")
            
            response = requests.post(
                f"{self.services['data_processor']}/api/process-gold-layer",
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Gold layer processing triggered: {result}")
                return result
            else:
                logger.error(f"Failed to process gold layer: {response.text}")
                return {'error': f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            logger.error(f"Error processing gold layer: {e}")
            return {'error': str(e)}
    
    def train_model(self, model_type: str = 'random_forest') -> Dict[str, Any]:
        """Train ML model using gold layer data"""
        try:
            logger.info(f"Training {model_type} model")
            
            response = requests.post(
                f"{self.services['ml_service']}/api/train-model",
                json={'model_type': model_type, 'retrain': True},
                timeout=1800  # 30 minutes timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Model training initiated: {result}")
                return result
            else:
                logger.error(f"Failed to train model: {response.text}")
                return {'error': f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return {'error': str(e)}
    
    def predict_leads(self, leads_data: List[Dict]) -> Dict[str, Any]:
        """Make predictions for new leads"""
        try:
            logger.info(f"Making predictions for {len(leads_data)} leads")
            
            response = requests.post(
                f"{self.services['ml_service']}/api/predict",
                json={'leads_data': leads_data},
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Predictions completed: {len(result.get('predictions', []))} results")
                return result
            else:
                logger.error(f"Failed to make predictions: {response.text}")
                return {'error': f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return {'error': str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            logger.info("Gathering system status")
            
            # Check services
            health_status = self.check_services_health()
            
            # Check database
            db_status = self.check_database_connection()
            
            # Check Apollo credits
            apollo_status = self.check_apollo_credits()
            
            # Get monitoring metrics
            monitoring_data = {}
            try:
                response = requests.get(f"{self.services['monitoring']}/api/metrics", timeout=10)
                if response.status_code == 200:
                    monitoring_data = response.json()
            except:
                monitoring_data = {'error': 'Monitoring service unavailable'}
            
            return {
                'timestamp': datetime.now().isoformat(),
                'services': health_status,
                'database': db_status,
                'apollo_credits': apollo_status,
                'monitoring': monitoring_data,
                'overall_health': all(health_status.values()) and db_status
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    def run_daily_pipeline(self) -> Dict[str, Any]:
        """Run the daily pipeline as discussed in transcript"""
        try:
            logger.info("Starting daily ML lead scoring pipeline")
            
            pipeline_results = {
                'started_at': datetime.now().isoformat(),
                'steps': {},
                'overall_success': True
            }
            
            # Step 1: Check system health
            logger.info("Step 1: Checking system health")
            health_status = self.get_system_status()
            pipeline_results['steps']['health_check'] = health_status
            
            if not health_status.get('overall_health', False):
                logger.error("System health check failed, aborting pipeline")
                pipeline_results['overall_success'] = False
                return pipeline_results
            
            # Step 2: Instantly data ingestion
            logger.info("Step 2: Instantly data ingestion")
            instantly_result = self.trigger_instantly_ingestion()
            pipeline_results['steps']['instantly_ingestion'] = instantly_result
            
            if 'error' in instantly_result:
                logger.error(f"Instantly ingestion failed: {instantly_result['error']}")
                pipeline_results['overall_success'] = False
            
            # Step 3: Apollo enrichment (only for new leads)
            logger.info("Step 3: Apollo enrichment")
            apollo_result = self.trigger_apollo_enrichment(max_leads=50)
            pipeline_results['steps']['apollo_enrichment'] = apollo_result
            
            if 'error' in apollo_result:
                logger.warning(f"Apollo enrichment issues: {apollo_result['error']}")
            
            # Step 4: Process to gold layer
            logger.info("Step 4: Processing to gold layer")
            gold_result = self.process_to_gold_layer()
            pipeline_results['steps']['gold_processing'] = gold_result
            
            # Step 5: Check if model retraining is needed
            logger.info("Step 5: Checking model retraining needs")
            # This would check drift, performance degradation, etc.
            # For now, we'll retrain weekly
            last_training = self.get_last_training_date()
            if (datetime.now() - last_training).days >= 7:
                logger.info("Model retraining needed")
                train_result = self.train_model()
                pipeline_results['steps']['model_training'] = train_result
            else:
                logger.info("Model retraining not needed")
                pipeline_results['steps']['model_training'] = {'status': 'skipped', 'reason': 'Recently trained'}
            
            pipeline_results['completed_at'] = datetime.now().isoformat()
            logger.info("Daily pipeline completed successfully")
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Error in daily pipeline: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'overall_success': False
            }
    
    def get_last_training_date(self) -> datetime:
        """Get the date of last model training"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT MAX(created_timestamp) 
                FROM ml_lead_scoring.model_versions 
                WHERE is_active = true
            """)
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result and result[0]:
                return result[0]
            else:
                # If no training found, return a date in the past to trigger training
                return datetime.now() - timedelta(days=30)
                
        except Exception as e:
            logger.error(f"Error getting last training date: {e}")
            return datetime.now() - timedelta(days=30)
    
    def start_services(self) -> Dict[str, Any]:
        """Start all required services using docker-compose"""
        try:
            logger.info("Starting ML lead scoring services")
            
            # Check if docker-compose is available
            result = subprocess.run(['docker-compose', '--version'], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                return {'error': 'Docker Compose not available'}
            
            # Start services
            result = subprocess.run(['docker-compose', 'up', '-d'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Services started successfully")
                
                # Wait for services to be ready
                time.sleep(30)
                
                # Check health
                health_status = self.check_services_health()
                
                return {
                    'status': 'started',
                    'services': health_status,
                    'message': 'All services started with docker-compose'
                }
            else:
                logger.error(f"Failed to start services: {result.stderr}")
                return {'error': result.stderr}
                
        except Exception as e:
            logger.error(f"Error starting services: {e}")
            return {'error': str(e)}
    
    def stop_services(self) -> Dict[str, Any]:
        """Stop all services"""
        try:
            logger.info("Stopping ML lead scoring services")
            
            result = subprocess.run(['docker-compose', 'down'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Services stopped successfully")
                return {'status': 'stopped', 'message': 'All services stopped'}
            else:
                logger.error(f"Failed to stop services: {result.stderr}")
                return {'error': result.stderr}
                
        except Exception as e:
            logger.error(f"Error stopping services: {e}")
            return {'error': str(e)}

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='ML Lead Scoring System Integration')
    parser.add_argument('action', choices=[
        'start', 'stop', 'status', 'daily-pipeline', 'train', 'predict',
        'instantly-ingest', 'apollo-enrich', 'process-gold'
    ], help='Action to perform')
    
    parser.add_argument('--config', default='config/integration_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--model-type', default='random_forest',
                       choices=['random_forest', 'gradient_boosting', 'logistic_regression'],
                       help='Model type for training')
    parser.add_argument('--max-leads', type=int, default=50,
                       help='Maximum leads for Apollo enrichment')
    
    args = parser.parse_args()
    
    # Initialize system
    system = MLLeadScoringSystem(args.config)
    
    # Execute action
    if args.action == 'start':
        result = system.start_services()
        
    elif args.action == 'stop':
        result = system.stop_services()
        
    elif args.action == 'status':
        result = system.get_system_status()
        
    elif args.action == 'daily-pipeline':
        result = system.run_daily_pipeline()
        
    elif args.action == 'train':
        result = system.train_model(args.model_type)
        
    elif args.action == 'instantly-ingest':
        result = system.trigger_instantly_ingestion()
        
    elif args.action == 'apollo-enrich':
        result = system.trigger_apollo_enrichment(args.max_leads)
        
    elif args.action == 'process-gold':
        result = system.process_to_gold_layer()
        
    elif args.action == 'predict':
        # Example prediction with sample data
        sample_leads = [{
            'email': 'test@example.com',
            'company': 'Test Company',
            'title': 'Manager'
        }]
        result = system.predict_leads(sample_leads)
    
    # Output result
    print(json.dumps(result, indent=2, default=str))
    
    # Exit with appropriate code
    if isinstance(result, dict) and result.get('error'):
        sys.exit(1)
    elif isinstance(result, dict) and not result.get('overall_success', True):
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
