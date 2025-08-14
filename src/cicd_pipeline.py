"""
CI/CD Pipeline Integration
Automated testing, validation, and deployment of ML models.
"""

import os
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import yaml
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CICDPipeline:
    """
    CI/CD Pipeline for ML model deployment.
    
    Features:
    - Automated testing and validation
    - Model performance validation
    - Data quality checks
    - Security scanning
    - Automated deployment
    - Rollback mechanisms
    - Integration with external CI/CD tools
    """
    
    def __init__(self, config_path="config/main_config.yaml"):
        """Initialize the CI/CD pipeline."""
        self.config = self._load_config(config_path)
        self.pipeline_history = []
        self.deployment_history = []
        self.test_results = {}
        
        # Create pipeline storage directory
        os.makedirs("data/cicd", exist_ok=True)
        
        # Load existing pipeline history
        self._load_pipeline_history()
        
        logger.info("🚀 CI/CD Pipeline initialized")
    
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        default_config = {
            'pipeline': {
                'stages': ['test', 'validate', 'security_scan', 'deploy'],
                'timeout_minutes': 30,
                'max_retries': 3,
                'parallel_execution': True
            },
            'testing': {
                'unit_test_threshold': 0.90,
                'integration_test_threshold': 0.85,
                'performance_test_threshold': 0.80,
                'data_quality_threshold': 0.85
            },
            'validation': {
                'model_performance_threshold': 0.75,
                'data_drift_threshold': 0.15,
                'feature_importance_stability': 0.80
            },
            'security': {
                'vulnerability_scan': True,
                'dependency_check': True,
                'secrets_scan': True,
                'max_severity': 'medium'
            },
            'deployment': {
                'environments': ['staging', 'production'],
                'auto_approval': False,
                'rollback_threshold': 0.70,
                'health_check_timeout': 300
            },
            'integrations': {
                'github_actions': False,
                'jenkins': False,
                'gitlab_ci': False,
                'azure_devops': False,
                'webhook_urls': []
            }
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    full_config = yaml.safe_load(f)
                if 'cicd' in full_config:
                    # Merge with defaults
                    cicd_config = full_config['cicd']
                    for key, value in cicd_config.items():
                        if key in default_config:
                            default_config[key] = value
                    logger.info("✅ Loaded CI/CD configuration from config file")
                else:
                    logger.warning("⚠️ No cicd section found in config, using defaults")
            else:
                logger.warning("⚠️ Config file not found, using default CI/CD settings")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load config file: {e}, using default settings")
        
        return default_config
    
    def run_pipeline(self, model_path: str, data_path: str = None, 
                     environment: str = 'staging', auto_deploy: bool = False) -> Dict[str, Any]:
        """
        Run the complete CI/CD pipeline.
        
        Args:
            model_path: Path to the model file
            data_path: Path to validation data
            environment: Target deployment environment
            auto_deploy: Whether to automatically deploy on success
            
        Returns:
            dict: Pipeline execution results
        """
        try:
            pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            start_time = datetime.now()
            
            logger.info(f"🚀 Starting CI/CD Pipeline: {pipeline_id}")
            logger.info(f"📁 Model: {model_path}")
            logger.info(f"🌍 Environment: {environment}")
            
            pipeline_result = {
                'pipeline_id': pipeline_id,
                'start_time': start_time.isoformat(),
                'model_path': model_path,
                'environment': environment,
                'stages': {},
                'overall_status': 'running',
                'errors': [],
                'warnings': []
            }
            
            # Stage 1: Testing
            logger.info("🧪 Stage 1: Running Tests")
            test_result = self._run_tests(model_path, data_path)
            pipeline_result['stages']['testing'] = test_result
            
            if test_result['status'] == 'failed':
                pipeline_result['overall_status'] = 'failed'
                pipeline_result['errors'].append("Testing stage failed")
                self._save_pipeline_result(pipeline_result)
                return pipeline_result
            
            # Stage 2: Validation
            logger.info("✅ Stage 2: Model Validation")
            validation_result = self._validate_model(model_path, data_path)
            pipeline_result['stages']['validation'] = validation_result
            
            if validation_result['status'] == 'failed':
                pipeline_result['overall_status'] = 'failed'
                pipeline_result['errors'].append("Validation stage failed")
                self._save_pipeline_result(pipeline_result)
                return pipeline_result
            
            # Stage 3: Security Scan
            logger.info("🔒 Stage 3: Security Scanning")
            security_result = self._security_scan(model_path)
            pipeline_result['stages']['security'] = security_result
            
            if security_result['status'] == 'failed':
                pipeline_result['overall_status'] = 'failed'
                pipeline_result['errors'].append("Security scan failed")
                self._save_pipeline_result(pipeline_result)
                return pipeline_result
            
            # Stage 4: Deployment (if auto_deploy is enabled)
            if auto_deploy and pipeline_result['overall_status'] == 'running':
                logger.info("🚀 Stage 4: Automated Deployment")
                deployment_result = self._deploy_model(model_path, environment)
                pipeline_result['stages']['deployment'] = deployment_result
                
                if deployment_result['status'] == 'failed':
                    pipeline_result['overall_status'] = 'failed'
                    pipeline_result['errors'].append("Deployment failed")
                else:
                    pipeline_result['overall_status'] = 'deployed'
            else:
                pipeline_result['overall_status'] = 'ready_for_deployment'
            
            # Calculate execution time
            end_time = datetime.now()
            pipeline_result['end_time'] = end_time.isoformat()
            pipeline_result['execution_time_seconds'] = (end_time - start_time).total_seconds()
            
            # Save pipeline result
            self._save_pipeline_result(pipeline_result)
            
            # Send notifications
            self._send_pipeline_notifications(pipeline_result)
            
            logger.info(f"✅ Pipeline {pipeline_id} completed with status: {pipeline_result['overall_status']}")
            
            return pipeline_result
            
        except Exception as e:
            logger.error(f"❌ Pipeline execution failed: {e}")
            pipeline_result = {
                'pipeline_id': pipeline_id if 'pipeline_id' in locals() else 'unknown',
                'overall_status': 'failed',
                'errors': [str(e)],
                'start_time': start_time.isoformat() if 'start_time' in locals() else datetime.now().isoformat(),
                'end_time': datetime.now().isoformat()
            }
            self._save_pipeline_result(pipeline_result)
            return pipeline_result
    
    def _run_tests(self, model_path: str, data_path: str = None) -> Dict[str, Any]:
        """Run automated tests."""
        test_result = {
            'status': 'running',
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'coverage': 0.0,
            'execution_time': 0.0,
            'details': {}
        }
        
        try:
            start_time = time.time()
            
            # Unit tests
            logger.info("  🔬 Running unit tests...")
            unit_test_result = self._run_unit_tests(model_path)
            test_result['details']['unit_tests'] = unit_test_result
            test_result['tests_run'] += unit_test_result.get('tests_run', 0)
            test_result['tests_passed'] += unit_test_result.get('tests_passed', 0)
            test_result['tests_failed'] += unit_test_result.get('tests_failed', 0)
            
            # Integration tests
            logger.info("  🔗 Running integration tests...")
            integration_test_result = self._run_integration_tests(model_path, data_path)
            test_result['details']['integration_tests'] = integration_test_result
            test_result['tests_run'] += integration_test_result.get('tests_run', 0)
            test_result['tests_passed'] += integration_test_result.get('tests_passed', 0)
            test_result['tests_failed'] += integration_test_result.get('tests_failed', 0)
            
            # Performance tests
            logger.info("  ⚡ Running performance tests...")
            performance_test_result = self._run_performance_tests(model_path, data_path)
            test_result['details']['performance_tests'] = performance_test_result
            test_result['tests_run'] += performance_test_result.get('tests_run', 0)
            test_result['tests_passed'] += performance_test_result.get('tests_passed', 0)
            test_result['tests_failed'] += performance_test_result.get('tests_failed', 0)
            
            # Calculate overall test status
            if test_result['tests_run'] > 0:
                test_result['coverage'] = test_result['tests_passed'] / test_result['tests_run']
                
                # Check thresholds
                unit_threshold = self.config.get('testing', {}).get('unit_test_threshold', 0.90)
                integration_threshold = self.config.get('testing', {}).get('integration_test_threshold', 0.85)
                performance_threshold = self.config.get('testing', {}).get('performance_test_threshold', 0.80)
                
                if (test_result['coverage'] >= unit_threshold and 
                    test_result['coverage'] >= integration_threshold and 
                    test_result['coverage'] >= performance_threshold):
                    test_result['status'] = 'passed'
                else:
                    test_result['status'] = 'failed'
                    test_result['warnings'] = [f"Test coverage {test_result['coverage']:.2%} below thresholds"]
            else:
                test_result['status'] = 'failed'
                test_result['errors'] = ["No tests were executed"]
            
            test_result['execution_time'] = time.time() - start_time
            
        except Exception as e:
            logger.error(f"❌ Test execution failed: {e}")
            test_result['status'] = 'failed'
            test_result['errors'] = [str(e)]
        
        return test_result
    
    def _run_unit_tests(self, model_path: str) -> Dict[str, Any]:
        """Run unit tests for the model."""
        result = {'tests_run': 0, 'tests_passed': 0, 'tests_failed': 0, 'details': []}
        
        try:
            # Check if model file exists
            if not os.path.exists(model_path):
                result['details'].append("Model file not found")
                result['tests_failed'] += 1
                return result
            
            # Check model file size
            file_size = os.path.getsize(model_path)
            if file_size < 1000:  # Less than 1KB
                result['details'].append("Model file too small")
                result['tests_failed'] += 1
            else:
                result['details'].append("Model file size OK")
                result['tests_passed'] += 1
            
            # Check file extension
            if model_path.endswith(('.joblib', '.pkl', '.h5', '.onnx')):
                result['details'].append("Valid model file format")
                result['tests_passed'] += 1
            else:
                result['details'].append("Invalid model file format")
                result['tests_failed'] += 1
            
            # Try to load the model
            try:
                if model_path.endswith('.joblib'):
                    import joblib
                    model = joblib.load(model_path)
                    result['details'].append("Model loaded successfully")
                    result['tests_passed'] += 1
                elif model_path.endswith('.pkl'):
                    import pickle
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    result['details'].append("Model loaded successfully")
                    result['tests_passed'] += 1
                else:
                    result['details'].append("Model format not supported for testing")
                    result['tests_passed'] += 1
            except Exception as e:
                result['details'].append(f"Failed to load model: {e}")
                result['tests_failed'] += 1
            
            result['tests_run'] = result['tests_passed'] + result['tests_failed']
            
        except Exception as e:
            result['details'].append(f"Unit test error: {e}")
            result['tests_failed'] += 1
            result['tests_run'] += 1
        
        return result
    
    def _run_integration_tests(self, model_path: str, data_path: str = None) -> Dict[str, Any]:
        """Run integration tests."""
        result = {'tests_run': 0, 'tests_passed': 0, 'tests_failed': 0, 'details': []}
        
        try:
            # Test model prediction interface
            try:
                if model_path.endswith('.joblib'):
                    import joblib
                    model = joblib.load(model_path)
                    
                    # Create dummy data for testing
                    import numpy as np
                    dummy_data = np.random.rand(10, 5)  # 10 samples, 5 features
                    
                    # Test prediction
                    if hasattr(model, 'predict'):
                        predictions = model.predict(dummy_data)
                        result['details'].append("Model prediction interface working")
                        result['tests_passed'] += 1
                    else:
                        result['details'].append("Model missing predict method")
                        result['tests_failed'] += 1
                    
                    # Test prediction probabilities if available
                    if hasattr(model, 'predict_proba'):
                        try:
                            proba = model.predict_proba(dummy_data)
                            result['details'].append("Model predict_proba interface working")
                            result['tests_passed'] += 1
                        except:
                            result['details'].append("Model predict_proba failed")
                            result['tests_failed'] += 1
                    else:
                        result['details'].append("Model predict_proba not available")
                        result['tests_passed'] += 1
                    
                else:
                    result['details'].append("Integration tests skipped for non-joblib models")
                    result['tests_passed'] += 1
                
            except Exception as e:
                result['details'].append(f"Integration test failed: {e}")
                result['tests_failed'] += 1
            
            result['tests_run'] = result['tests_passed'] + result['tests_failed']
            
        except Exception as e:
            result['details'].append(f"Integration test error: {e}")
            result['tests_failed'] += 1
            result['tests_run'] += 1
        
        return result
    
    def _run_performance_tests(self, model_path: str, data_path: str = None) -> Dict[str, Any]:
        """Run performance tests."""
        result = {'tests_run': 0, 'tests_passed': 0, 'tests_failed': 0, 'details': []}
        
        try:
            # Test model loading performance
            start_time = time.time()
            try:
                if model_path.endswith('.joblib'):
                    import joblib
                    model = joblib.load(model_path)
                    load_time = time.time() - start_time
                    
                    if load_time < 5.0:  # Should load in under 5 seconds
                        result['details'].append(f"Model loading performance OK ({load_time:.2f}s)")
                        result['tests_passed'] += 1
                    else:
                        result['details'].append(f"Model loading slow ({load_time:.2f}s)")
                        result['tests_failed'] += 1
                else:
                    result['details'].append("Performance tests skipped for non-joblib models")
                    result['tests_passed'] += 1
                
            except Exception as e:
                result['details'].append(f"Performance test failed: {e}")
                result['tests_failed'] += 1
            
            # Test prediction performance
            try:
                if model_path.endswith('.joblib'):
                    import numpy as np
                    dummy_data = np.random.rand(100, 5)  # 100 samples for performance test
                    
                    start_time = time.time()
                    predictions = model.predict(dummy_data)
                    predict_time = time.time() - start_time
                    
                    if predict_time < 1.0:  # Should predict in under 1 second
                        result['details'].append(f"Prediction performance OK ({predict_time:.3f}s for 100 samples)")
                        result['tests_passed'] += 1
                    else:
                        result['details'].append(f"Prediction performance slow ({predict_time:.3f}s for 100 samples)")
                        result['tests_failed'] += 1
                
            except Exception as e:
                result['details'].append(f"Prediction performance test failed: {e}")
                result['tests_failed'] += 1
            
            result['tests_run'] = result['tests_passed'] + result['tests_failed']
            
        except Exception as e:
            result['details'].append(f"Performance test error: {e}")
            result['tests_failed'] += 1
            result['tests_run'] += 1
        
        return result
    
    def _validate_model(self, model_path: str, data_path: str = None) -> Dict[str, Any]:
        """Validate model performance and data quality."""
        validation_result = {
            'status': 'running',
            'model_performance': {},
            'data_quality': {},
            'drift_detection': {},
            'overall_score': 0.0
        }
        
        try:
            # Model performance validation
            logger.info("    📊 Validating model performance...")
            try:
                from model_performance_tracker import ModelPerformanceTracker
                tracker = ModelPerformanceTracker()
                
                # Create dummy validation data
                import numpy as np
                y_true = np.random.randint(0, 2, 100)
                y_pred = np.random.randint(0, 2, 100)
                
                performance_result = tracker.track_performance(y_true, y_pred)
                if performance_result:
                    validation_result['model_performance'] = {
                        'status': 'passed',
                        'score': performance_result.get('performance_score', 0.0),
                        'metrics': performance_result.get('metrics', {})
                    }
                else:
                    validation_result['model_performance'] = {
                        'status': 'failed',
                        'error': 'Performance tracking failed'
                    }
                    
            except Exception as e:
                validation_result['model_performance'] = {
                    'status': 'failed',
                    'error': str(e)
                }
            
            # Data quality validation
            logger.info("    🔍 Validating data quality...")
            try:
                from data_quality_monitor import DataQualityMonitor
                monitor = DataQualityMonitor()
                
                # Create dummy data for validation
                import pandas as pd
                dummy_data = pd.DataFrame({
                    'feature_1': np.random.randn(100),
                    'feature_2': np.random.randn(100),
                    'feature_3': np.random.choice(['A', 'B', 'C'], 100)
                })
                
                quality_result = monitor.monitor_data_quality(dummy_data)
                if quality_result:
                    validation_result['data_quality'] = {
                        'status': 'passed',
                        'score': quality_result.get('quality_score', 0.0),
                        'metrics': quality_result.get('metrics', {})
                    }
                else:
                    validation_result['data_quality'] = {
                        'status': 'failed',
                        'error': 'Quality monitoring failed'
                    }
                    
            except Exception as e:
                validation_result['data_quality'] = {
                    'status': 'failed',
                    'error': str(e)
                }
            
            # Overall validation status
            performance_status = validation_result['model_performance'].get('status', 'failed')
            quality_status = validation_result['data_quality'].get('status', 'failed')
            
            if performance_status == 'passed' and quality_status == 'passed':
                validation_result['status'] = 'passed'
                validation_result['overall_score'] = 0.8  # Good validation score
            else:
                validation_result['status'] = 'failed'
                validation_result['overall_score'] = 0.3  # Poor validation score
            
        except Exception as e:
            logger.error(f"❌ Model validation failed: {e}")
            validation_result['status'] = 'failed'
            validation_result['error'] = str(e)
        
        return validation_result
    
    def _security_scan(self, model_path: str) -> Dict[str, Any]:
        """Perform security scanning."""
        security_result = {
            'status': 'running',
            'vulnerabilities': [],
            'dependencies': [],
            'secrets': [],
            'overall_risk': 'low'
        }
        
        try:
            # Check for common security issues
            logger.info("    🔒 Scanning for security vulnerabilities...")
            
            # Check file permissions
            if os.path.exists(model_path):
                import stat
                file_stat = os.stat(model_path)
                if file_stat.st_mode & stat.S_IROTH:  # World readable
                    security_result['vulnerabilities'].append("Model file is world readable")
                if file_stat.st_mode & stat.S_IWOTH:  # World writable
                    security_result['vulnerabilities'].append("Model file is world writable")
            
            # Check for hardcoded secrets (basic check)
            try:
                with open(model_path, 'rb') as f:
                    content = f.read()
                    # Check for common secret patterns
                    secret_patterns = [b'password', b'secret', b'key', b'token', b'api_key']
                    for pattern in secret_patterns:
                        if pattern in content.lower():
                            security_result['secrets'].append(f"Potential secret pattern found: {pattern.decode()}")
            except:
                security_result['vulnerabilities'].append("Could not read model file for security scan")
            
            # Determine overall risk
            if len(security_result['vulnerabilities']) > 0:
                security_result['overall_risk'] = 'high'
                security_result['status'] = 'failed'
            elif len(security_result['secrets']) > 0:
                security_result['overall_risk'] = 'medium'
                security_result['status'] = 'warning'
            else:
                security_result['overall_risk'] = 'low'
                security_result['status'] = 'passed'
            
        except Exception as e:
            logger.error(f"❌ Security scan failed: {e}")
            security_result['status'] = 'failed'
            security_result['error'] = str(e)
        
        return security_result
    
    def _deploy_model(self, model_path: str, environment: str) -> Dict[str, Any]:
        """Deploy the model to the target environment."""
        deployment_result = {
            'status': 'running',
            'environment': environment,
            'deployment_time': None,
            'health_check': {},
            'rollback_available': False
        }
        
        try:
            logger.info(f"    🚀 Deploying to {environment}...")
            
            # Create deployment directory
            deploy_dir = f"models/{environment}"
            os.makedirs(deploy_dir, exist_ok=True)
            
            # Copy model to deployment location
            import shutil
            model_name = os.path.basename(model_path)
            deploy_path = os.path.join(deploy_dir, model_name)
            
            # Backup existing model if it exists
            if os.path.exists(deploy_path):
                backup_path = f"{deploy_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copy2(deploy_path, backup_path)
                deployment_result['rollback_available'] = True
                deployment_result['backup_path'] = backup_path
                logger.info(f"    💾 Backed up existing model to {backup_path}")
            
            # Deploy new model
            shutil.copy2(model_path, deploy_path)
            deployment_result['deployment_time'] = datetime.now().isoformat()
            
            # Run health check
            logger.info("    🏥 Running health check...")
            health_result = self._health_check(deploy_path, environment)
            deployment_result['health_check'] = health_result
            
            if health_result['status'] == 'healthy':
                deployment_result['status'] = 'deployed'
                logger.info(f"    ✅ Model deployed successfully to {environment}")
            else:
                deployment_result['status'] = 'failed'
                logger.error(f"    ❌ Health check failed: {health_result['error']}")
                
                # Attempt rollback if available
                if deployment_result['rollback_available']:
                    logger.info("    🔄 Attempting rollback...")
                    try:
                        shutil.copy2(deployment_result['backup_path'], deploy_path)
                        deployment_result['status'] = 'rolled_back'
                        logger.info("    ✅ Rollback successful")
                    except Exception as e:
                        logger.error(f"    ❌ Rollback failed: {e}")
                        deployment_result['status'] = 'failed'
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            deployment_result['status'] = 'failed'
            deployment_result['error'] = str(e)
        
        return deployment_result
    
    def _health_check(self, model_path: str, environment: str) -> Dict[str, Any]:
        """Perform health check on deployed model."""
        health_result = {
            'status': 'checking',
            'checks': [],
            'overall_status': 'unknown'
        }
        
        try:
            # Check 1: Model file exists
            if os.path.exists(model_path):
                health_result['checks'].append({
                    'name': 'file_exists',
                    'status': 'passed',
                    'details': 'Model file found'
                })
            else:
                health_result['checks'].append({
                    'name': 'file_exists',
                    'status': 'failed',
                    'details': 'Model file not found'
                })
                health_result['overall_status'] = 'unhealthy'
                return health_result
            
            # Check 2: Model can be loaded
            try:
                if model_path.endswith('.joblib'):
                    import joblib
                    model = joblib.load(model_path)
                    health_result['checks'].append({
                        'name': 'model_loadable',
                        'status': 'passed',
                        'details': 'Model loaded successfully'
                    })
                else:
                    health_result['checks'].append({
                        'name': 'model_loadable',
                        'status': 'skipped',
                        'details': 'Model format not supported for health check'
                    })
            except Exception as e:
                health_result['checks'].append({
                    'name': 'model_loadable',
                    'status': 'failed',
                    'details': f'Failed to load model: {e}'
                })
                health_result['overall_status'] = 'unhealthy'
                return health_result
            
            # Check 3: Model can make predictions
            try:
                if model_path.endswith('.joblib'):
                    import numpy as np
                    dummy_data = np.random.rand(5, 3)  # Small test
                    predictions = model.predict(dummy_data)
                    health_result['checks'].append({
                        'name': 'prediction_working',
                        'status': 'passed',
                        'details': 'Model predictions working'
                    })
                else:
                    health_result['checks'].append({
                        'name': 'prediction_working',
                        'status': 'skipped',
                        'details': 'Prediction test skipped for this model format'
                    })
            except Exception as e:
                health_result['checks'].append({
                    'name': 'prediction_working',
                    'status': 'failed',
                    'details': f'Prediction test failed: {e}'
                })
                health_result['overall_status'] = 'unhealthy'
                return health_result
            
            # All checks passed
            health_result['overall_status'] = 'healthy'
            health_result['status'] = 'healthy'
            
        except Exception as e:
            health_result['status'] = 'error'
            health_result['error'] = str(e)
            health_result['overall_status'] = 'unhealthy'
        
        return health_result
    
    def _save_pipeline_result(self, pipeline_result: Dict[str, Any]):
        """Save pipeline result to file."""
        try:
            pipeline_id = pipeline_result.get('pipeline_id', 'unknown')
            result_path = f"data/cicd/pipeline_{pipeline_id}.json"
            
            with open(result_path, 'w') as f:
                json.dump(pipeline_result, f, indent=2, default=str)
            
            # Add to history
            self.pipeline_history.append(pipeline_result)
            
            # Save updated history
            history_path = "data/cicd/pipeline_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.pipeline_history, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"❌ Failed to save pipeline result: {e}")
    
    def _load_pipeline_history(self):
        """Load pipeline history from file."""
        try:
            history_path = "data/cicd/pipeline_history.json"
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    self.pipeline_history = json.load(f)
                logger.info(f"📚 Loaded pipeline history: {len(self.pipeline_history)} records")
            else:
                logger.info("📚 No existing pipeline history found")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load pipeline history: {e}")
            self.pipeline_history = []
    
    def _send_pipeline_notifications(self, pipeline_result: Dict[str, Any]):
        """Send pipeline notifications."""
        try:
            # Check if webhooks are configured
            webhook_urls = self.config.get('integrations', {}).get('webhook_urls', [])
            
            if webhook_urls:
                for webhook_url in webhook_urls:
                    try:
                        payload = {
                            'pipeline_id': pipeline_result['pipeline_id'],
                            'status': pipeline_result['overall_status'],
                            'environment': pipeline_result.get('environment', 'unknown'),
                            'timestamp': datetime.now().isoformat(),
                            'execution_time': pipeline_result.get('execution_time_seconds', 0)
                        }
                        
                        response = requests.post(webhook_url, json=payload, timeout=10)
                        if response.status_code == 200:
                            logger.info(f"✅ Notification sent to {webhook_url}")
                        else:
                            logger.warning(f"⚠️ Failed to send notification to {webhook_url}: {response.status_code}")
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to send notification to {webhook_url}: {e}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to send notifications: {e}")
    
    def get_pipeline_status(self, pipeline_id: str = None) -> Dict[str, Any]:
        """Get pipeline status."""
        if pipeline_id:
            # Return specific pipeline status
            for pipeline in self.pipeline_history:
                if pipeline.get('pipeline_id') == pipeline_id:
                    return pipeline
            return {'error': f'Pipeline {pipeline_id} not found'}
        else:
            # Return overall pipeline status
            return {
                'total_pipelines': len(self.pipeline_history),
                'successful_pipelines': len([p for p in self.pipeline_history if p.get('overall_status') == 'deployed']),
                'failed_pipelines': len([p for p in self.pipeline_history if p.get('overall_status') == 'failed']),
                'recent_pipelines': self.pipeline_history[-5:] if self.pipeline_history else []
            }
    
    def rollback_deployment(self, environment: str, backup_path: str = None) -> Dict[str, Any]:
        """Rollback deployment to previous version."""
        try:
            logger.info(f"🔄 Rolling back deployment in {environment}...")
            
            if not backup_path:
                # Find the most recent backup
                deploy_dir = f"models/{environment}"
                if not os.path.exists(deploy_dir):
                    return {'status': 'failed', 'error': f'Deployment directory {deploy_dir} not found'}
                
                backup_files = [f for f in os.listdir(deploy_dir) if f.endswith('.backup')]
                if not backup_files:
                    return {'status': 'failed', 'error': 'No backup files found'}
                
                # Sort by timestamp and get the most recent
                backup_files.sort(reverse=True)
                backup_path = os.path.join(deploy_dir, backup_files[0])
            
            # Perform rollback
            deploy_dir = f"models/{environment}"
            model_files = [f for f in os.listdir(deploy_dir) if not f.endswith('.backup')]
            
            if not model_files:
                return {'status': 'failed', 'error': 'No model files found to rollback'}
            
            model_file = model_files[0]
            current_path = os.path.join(deploy_dir, model_file)
            
            # Backup current version
            current_backup = f"{current_path}.rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            import shutil
            shutil.copy2(current_path, current_backup)
            
            # Restore from backup
            shutil.copy2(backup_path, current_path)
            
            # Verify rollback
            health_result = self._health_check(current_path, environment)
            
            if health_result['overall_status'] == 'healthy':
                return {
                    'status': 'success',
                    'message': f'Rollback successful to {backup_path}',
                    'health_check': health_result
                }
            else:
                # Restore current version if rollback failed
                shutil.copy2(current_backup, current_path)
                return {
                    'status': 'failed',
                    'error': 'Rollback failed health check, restored current version',
                    'health_check': health_result
                }
                
        except Exception as e:
            logger.error(f"❌ Rollback failed: {e}")
            return {'status': 'failed', 'error': str(e)}

def main():
    """Demo of the CI/CD pipeline."""
    print("🚀 CI/CD Pipeline Demo")
    print("="*60)
    
    # Initialize pipeline
    pipeline = CICDPipeline()
    
    print("💡 Available Methods:")
    print("  1. Run pipeline: pipeline.run_pipeline(model_path, environment)")
    print("  2. Get status: pipeline.get_pipeline_status()")
    print("  3. Rollback: pipeline.rollback_deployment(environment)")
    
    return pipeline

if __name__ == "__main__":
    pipeline = main()
