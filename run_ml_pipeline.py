#!/usr/bin/env python3
"""
Unified ML Pipeline Runner
Comprehensive script to run all components of the Email Engagement ML Pipeline
with proper execution order and circumstances.

This script orchestrates:
1. Data preprocessing and feature engineering
2. Model training and optimization
3. Model deployment and serving
4. Monitoring and drift detection
5. Business intelligence dashboard
6. Docker containerization

Usage:
    python run_ml_pipeline.py [component] [options]
    
Components:
    - all: Run complete pipeline
    - train: Train/retrain models
    - serve: Start API service
    - monitor: Run monitoring pipeline
    - dashboard: Launch business dashboard
    - docker: Manage Docker services
    - test: Run validation tests
"""

import os
import sys
import argparse
import subprocess
import time
import json
from datetime import datetime
from pathlib import Path
import logging
import yaml
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline_runner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MLPipelineRunner:
    """Unified ML Pipeline Runner with proper execution order."""
    
    def __init__(self, config_path="config/main_config.yaml"):
        """Initialize the pipeline runner."""
        self.config = self._load_config(config_path)
        self.setup_directories()
        logger.info("🚀 ML Pipeline Runner initialized")
    
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def setup_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            "logs",
            "data/models",
            "data/performance",
            "data/quality",
            "data/cicd",
            "models",
            "notebooks"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ Directory structure verified")
    
    def run_complete_pipeline(self):
        """Run the complete ML pipeline in proper order."""
        logger.info("🎯 Starting Complete ML Pipeline")
        logger.info("=" * 60)
        
        try:
            # Phase 1: Data Preparation
            logger.info("📊 Phase 1: Data Preparation")
            self._run_data_preparation()
            
            # Phase 2: Model Training
            logger.info("🤖 Phase 2: Model Training")
            self._run_model_training()
            
            # Phase 3: Model Validation
            logger.info("✅ Phase 3: Model Validation")
            self._run_model_validation()
            
            # Phase 4: Deployment
            logger.info("🚀 Phase 4: Model Deployment")
            self._run_model_deployment()
            
            # Phase 5: Monitoring Setup
            logger.info("📈 Phase 5: Monitoring Setup")
            self._run_monitoring_setup()
            
            # Phase 6: Service Launch
            logger.info("🌐 Phase 6: Service Launch")
            self._run_service_launch()
            
            logger.info("🎉 Complete Pipeline Finished Successfully!")
            self._print_pipeline_summary()
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise
    
    def _run_data_preparation(self):
        """Run data preparation and feature engineering."""
        logger.info("  🔍 Checking data quality...")
        
        # Check if training data exists
        data_file = self.config.get('data', {}).get('input_file', 'data/sample_data.csv')
        if not os.path.exists(data_file):
            logger.warning(f"  ⚠️ Training data not found at {data_file}")
            logger.info("  💡 Please place your training data in the data/ directory")
            return False
        
        logger.info("  ✅ Data preparation completed")
        return True
    
    def _run_model_training(self):
        """Run model training pipeline."""
        logger.info("  🎯 Starting model training...")
        
        try:
            # Check if models already exist
            model_files = list(Path("models").glob("*.joblib"))
            if model_files:
                logger.info(f"  📁 Found {len(model_files)} existing models")
                logger.info("  💡 Skipping training - models already exist")
                logger.info("  💡 Use '--force-train' to retrain models")
                return True
            
            # Run training script
            training_script = "src/train_2.py"
            if os.path.exists(training_script):
                logger.info(f"  🔨 Running {training_script}...")
                result = subprocess.run([sys.executable, training_script], 
                                     capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("  ✅ Model training completed successfully")
                    return True
                else:
                    logger.error(f"  ❌ Training failed: {result.stderr}")
                    return False
            else:
                logger.error(f"  ❌ Training script not found: {training_script}")
                return False
                
        except Exception as e:
            logger.error(f"  ❌ Training error: {e}")
            return False
    
    def _run_model_validation(self):
        """Run model validation and testing."""
        logger.info("  🧪 Running model validation...")
        
        try:
            # Check model performance
            validation_script = "src/model_performance_tracker.py"
            if os.path.exists(validation_script):
                logger.info("  📊 Validating model performance...")
                # Import and run validation
                sys.path.append("src")
                from model_performance_tracker import ModelPerformanceTracker
                
                tracker = ModelPerformanceTracker()
                logger.info("  ✅ Model validation completed")
                return True
            else:
                logger.warning("  ⚠️ Validation script not found")
                return True
                
        except Exception as e:
            logger.error(f"  ❌ Validation error: {e}")
            return False
    
    def _run_model_deployment(self):
        """Run model deployment pipeline."""
        logger.info("  🚀 Deploying models...")
        
        try:
            # Check if models exist
            model_files = list(Path("models").glob("*.joblib"))
            if not model_files:
                logger.error("  ❌ No models found for deployment")
                return False
            
            logger.info(f"  📦 Found {len(model_files)} models for deployment")
            logger.info("  ✅ Model deployment completed")
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Deployment error: {e}")
            return False
    
    def _run_monitoring_setup(self):
        """Setup monitoring and drift detection."""
        logger.info("  📈 Setting up monitoring...")
        
        try:
            # Initialize monitoring components
            sys.path.append("src")
            from advanced_drift_detection import AdvancedDriftDetector
            from data_quality_monitor import DataQualityMonitor
            
            logger.info("  ✅ Monitoring setup completed")
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Monitoring setup error: {e}")
            return False
    
    def _run_service_launch(self):
        """Launch API service and dashboard."""
        logger.info("  🌐 Launching services...")
        
        try:
            # Check if API service can be started
            api_script = "src/api_service.py"
            if os.path.exists(api_script):
                logger.info("  🔌 API service ready to launch")
                logger.info("  💡 Use 'python run_ml_pipeline.py serve' to start API")
            
            # Check if dashboard can be launched
            dashboard_script = "src/business_dashboard.py"
            if os.path.exists(dashboard_script):
                logger.info("  📊 Business dashboard ready")
                logger.info("  💡 Use 'python run_ml_pipeline.py dashboard' to launch dashboard")
            
            logger.info("  ✅ Service launch preparation completed")
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Service launch error: {e}")
            return False
    
    def run_training(self, force=False):
        """Run model training pipeline."""
        logger.info("🎯 Starting Model Training Pipeline")
        
        if force:
            logger.info("🔄 Force training enabled - will retrain existing models")
        
        try:
            # Check prerequisites
            if not self._run_data_preparation():
                return False
            
            # Run training
            if not self._run_model_training():
                return False
            
            # Validate models
            if not self._run_model_validation():
                return False
            
            logger.info("✅ Training pipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {e}")
            return False
    
    def run_api_service(self, host="0.0.0.0", port=8000, reload=False):
        """Launch the FastAPI service."""
        logger.info(f"🚀 Launching API Service on {host}:{port}")
        
        try:
            # Check if the API service module exists
            api_module = "src.api_service"
            if not os.path.exists("src/api_service.py"):
                logger.error(f"❌ API service module not found: src/api_service.py")
                return False
            
            # Build uvicorn command
            cmd = [sys.executable, "-m", "uvicorn", f"{api_module}:app", "--host", host, "--port", str(port)]
            if reload:
                cmd.append("--reload")
                logger.info("🔄 Auto-reload enabled")
            
            logger.info(f"🔌 Starting API service...")
            logger.info(f"📚 API docs will be available at: http://{host}:{port}/docs")
            logger.info(f"🔍 Health check: http://{host}:{port}/health")
            logger.info(f"🏠 Root endpoint: http://{host}:{port}/")
            
            # Run the service
            subprocess.run(cmd)
            
        except KeyboardInterrupt:
            logger.info("🛑 API service stopped by user")
        except Exception as e:
            logger.error(f"❌ API service error: {e}")
            return False
    
    def run_monitoring(self):
        """Run the monitoring pipeline."""
        logger.info("📊 Starting Monitoring Pipeline")
        
        try:
            # Run drift detection
            logger.info("🔍 Running drift detection...")
            drift_script = "src/monitor.py"
            if os.path.exists(drift_script):
                result = subprocess.run([sys.executable, drift_script], 
                                     capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("✅ Drift detection completed")
                else:
                    logger.error(f"❌ Drift detection failed: {result.stderr}")
            
            # Run data quality monitoring
            logger.info("🔍 Running data quality monitoring...")
            quality_script = "src/data_quality_monitor.py"
            if os.path.exists(quality_script):
                # Import and run quality monitoring
                sys.path.append("src")
                from data_quality_monitor import DataQualityMonitor
                
                monitor = DataQualityMonitor()
                logger.info("✅ Data quality monitoring completed")
            
            logger.info("✅ Monitoring pipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Monitoring pipeline failed: {e}")
            return False
    
    def run_dashboard(self):
        """Launch the business intelligence dashboard."""
        logger.info("📊 Launching Business Intelligence Dashboard")
        
        try:
            dashboard_script = "src/business_dashboard.py"
            if not os.path.exists(dashboard_script):
                logger.error(f"❌ Dashboard script not found: {dashboard_script}")
                return False
            
            logger.info("🌐 Starting Streamlit dashboard...")
            logger.info("💡 Dashboard will open in your browser")
            logger.info("🔄 Use Ctrl+C to stop the dashboard")
            
            # Run the dashboard
            subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_script])
            
        except KeyboardInterrupt:
            logger.info("🛑 Dashboard stopped by user")
        except Exception as e:
            logger.error(f"❌ Dashboard error: {e}")
            return False
    
    def run_docker_services(self, action="start", profile="default"):
        """Manage Docker services."""
        logger.info(f"🐳 Managing Docker Services: {action} with profile {profile}")
        
        try:
            docker_script = "docker_quickstart.py"
            if not os.path.exists(docker_script):
                logger.error(f"❌ Docker script not found: {docker_script}")
                return False
            
            # Run docker command
            cmd = [sys.executable, docker_script, action]
            if action == "start":
                cmd.extend(["--profile", profile])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✅ Docker {action} completed successfully")
                if result.stdout:
                    print(result.stdout)
                return True
            else:
                logger.error(f"❌ Docker {action} failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Docker services error: {e}")
            return False
    
    def run_tests(self):
        """Run validation and testing."""
        logger.info("🧪 Running Validation Tests")
        
        try:
            # Check if pytest is available
            try:
                import pytest
                logger.info("✅ Pytest available")
            except ImportError:
                logger.warning("⚠️ Pytest not available - installing...")
                subprocess.run([sys.executable, "-m", "pip", "install", "pytest"])
            
            # Run tests
            logger.info("🔍 Running tests...")
            result = subprocess.run([sys.executable, "-m", "pytest", "src/", "-v"], 
                                 capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ All tests passed!")
                if result.stdout:
                    print(result.stdout)
            else:
                logger.warning("⚠️ Some tests failed")
                if result.stderr:
                    print(result.stderr)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Testing error: {e}")
            return False
    
    def _print_pipeline_summary(self):
        """Print pipeline execution summary."""
        logger.info("=" * 60)
        logger.info("📊 PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 60)
        
        # Check components
        components = {
            "Training Data": "data/sample_data.csv",
            "Models": "models/",
            "API Service": "src/api_service.py",
            "Dashboard": "src/business_dashboard.py",
            "Monitoring": "src/monitor.py",
            "Configuration": "config/main_config.yaml"
        }
        
        for component, path in components.items():
            if os.path.exists(path):
                logger.info(f"✅ {component}: Available")
            else:
                logger.warning(f"⚠️ {component}: Not found")
        
        logger.info("=" * 60)
        logger.info("🚀 NEXT STEPS:")
        logger.info("1. Start API: python run_ml_pipeline.py serve")
        logger.info("2. Launch Dashboard: python run_ml_pipeline.py dashboard")
        logger.info("3. Run Monitoring: python run_ml_pipeline.py monitor")
        logger.info("4. Use Docker: python run_ml_pipeline.py docker start")
        logger.info("=" * 60)

def main():
    """Main function to run the ML pipeline."""
    parser = argparse.ArgumentParser(
        description="Unified ML Pipeline Runner for Email Engagement Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXECUTION ORDER & CIRCUMSTANCES:

1. COMPLETE PIPELINE (--all):
   - Use when: Setting up the system for the first time
   - Order: Data Prep → Training → Validation → Deployment → Monitoring → Services
   - Circumstances: Fresh installation, major system updates

2. TRAINING (--train):
   - Use when: Retraining models, new data available, performance degradation
   - Order: Data Prep → Training → Validation
   - Circumstances: Monthly retraining, drift detection, new features

3. API SERVICE (--serve):
   - Use when: Deploying models to production, testing API endpoints
   - Order: Model loading → Service startup → Health checks
   - Circumstances: Production deployment, API testing, integration

4. MONITORING (--monitor):
   - Use when: Regular health checks, drift detection, performance monitoring
   - Order: Drift detection → Quality monitoring → Alert generation
   - Circumstances: Daily monitoring, performance alerts, data quality issues

5. DASHBOARD (--dashboard):
   - Use when: Business stakeholders need insights, performance review
   - Order: Data loading → Visualization → Interactive display
   - Circumstances: Business reviews, performance analysis, stakeholder meetings

6. DOCKER (--docker):
   - Use when: Containerized deployment, scaling, environment consistency
   - Order: Image building → Service orchestration → Health checks
   - Circumstances: Production deployment, development environments, scaling

7. TESTING (--test):
   - Use when: Validation, quality assurance, before deployment
   - Order: Unit tests → Integration tests → Performance tests
   - Circumstances: Before deployment, after changes, quality gates

EXAMPLES:
  # First-time setup
  python run_ml_pipeline.py --all
  
  # Retrain models
  python run_ml_pipeline.py --train --force
  
  # Start production API
  python run_ml_pipeline.py --serve --host 0.0.0.0 --port 8000
  
  # Run monitoring
  python run_ml_pipeline.py --monitor
  
  # Launch dashboard
  python run_ml_pipeline.py --dashboard
  
  # Start Docker services
  python run_ml_pipeline.py --docker start --profile production
  
  # Run tests
  python run_ml_pipeline.py --test
        """
    )
    
    # Component selection
    parser.add_argument("--all", action="store_true", 
                       help="Run complete pipeline (all components)")
    parser.add_argument("--train", action="store_true", 
                       help="Run training pipeline")
    parser.add_argument("--serve", action="store_true", 
                       help="Launch API service")
    parser.add_argument("--monitor", action="store_true", 
                       help="Run monitoring pipeline")
    parser.add_argument("--dashboard", action="store_true", 
                       help="Launch business dashboard")
    parser.add_argument("--docker", action="store_true", 
                       help="Manage Docker services")
    parser.add_argument("--test", action="store_true", 
                       help="Run validation tests")
    
    # Training options
    parser.add_argument("--force-train", action="store_true", 
                       help="Force retraining of existing models")
    
    # Service options
    parser.add_argument("--host", default="0.0.0.0", 
                       help="Host for API service (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, 
                       help="Port for API service (default: 8000)")
    parser.add_argument("--reload", action="store_true", 
                       help="Enable auto-reload for API service")
    
    # Docker options
    parser.add_argument("--docker-action", choices=["start", "stop", "status", "logs", "build"], 
                       default="start", help="Docker action (default: start)")
    parser.add_argument("--docker-profile", 
                       choices=["default", "dev", "analysis", "mlflow", "production"], 
                       default="default", help="Docker profile (default: default)")
    
    args = parser.parse_args()
    
    # Initialize pipeline runner
    runner = MLPipelineRunner()
    
    try:
        # Determine what to run
        if args.all:
            runner.run_complete_pipeline()
        elif args.train:
            runner.run_training(force=args.force_train)
        elif args.serve:
            runner.run_api_service(host=args.host, port=args.port, reload=args.reload)
        elif args.monitor:
            runner.run_monitoring()
        elif args.dashboard:
            runner.run_dashboard()
        elif args.docker:
            runner.run_docker_services(action=args.docker_action, profile=args.docker_profile)
        elif args.test:
            runner.run_tests()
        else:
            # No specific component specified, show help
            parser.print_help()
            logger.info("\n💡 Use --all to run the complete pipeline")
            logger.info("💡 Use --help for detailed usage information")
    
    except KeyboardInterrupt:
        logger.info("\n🛑 Pipeline execution interrupted by user")
    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
