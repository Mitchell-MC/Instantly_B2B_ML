#!/usr/bin/env python3
"""
MLflow Dashboard Launcher
Launches MLflow UI for experiment tracking and model registry visualization.
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path

def load_mlflow_config():
    """Load MLflow configuration from config file."""
    config_path = Path("config/main_config.yaml")
    
    if not config_path.exists():
        print("❌ Configuration file not found. Please run setup first.")
        return None
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        if 'mlflow' not in config:
            print("❌ MLflow configuration not found in config file.")
            return None
        
        return config['mlflow']
    
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return None

def check_mlflow_installation():
    """Check if MLflow is properly installed."""
    try:
        import mlflow
        print(f"✅ MLflow {mlflow.__version__} is installed")
        return True
    except ImportError:
        print("❌ MLflow is not installed. Please install it first:")
        print("   pip install mlflow mlflow-sklearn mlflow-xgboost")
        return False

def launch_mlflow_ui(config):
    """Launch MLflow UI with the configured tracking URI."""
    tracking_uri = config.get('tracking_uri', 'sqlite:///mlflow.db')
    
    print(f"🚀 Launching MLflow UI...")
    print(f"📊 Tracking URI: {tracking_uri}")
    print(f"🔬 Experiment: {config.get('experiment_name', 'email_engagement_prediction')}")
    print(f"📁 Model Registry: {config.get('registry_uri', 'sqlite:///mlflow.db')}")
    print(f"🌐 UI will be available at: http://localhost:5000")
    print(f"📝 Press Ctrl+C to stop the server")
    print("="*60)
    
    try:
        # Launch MLflow UI
        subprocess.run([
            sys.executable, "-m", "mlflow", "ui",
            "--backend-store-uri", tracking_uri,
            "--default-artifact-root", "./mlruns",
            "--host", "0.0.0.0",
            "--port", "5000"
        ])
    except KeyboardInterrupt:
        print("\n🛑 MLflow UI stopped by user")
    except Exception as e:
        print(f"❌ Failed to launch MLflow UI: {e}")

def setup_mlflow_database():
    """Set up MLflow SQLite database if it doesn't exist."""
    db_path = Path("mlflow.db")
    
    if not db_path.exists():
        print("🔧 Setting up MLflow SQLite database...")
        try:
            import mlflow
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow.set_experiment("email_engagement_prediction")
            
            # Create a dummy run to initialize the database
            with mlflow.start_run(run_name="setup_run") as run:
                mlflow.log_param("setup", True)
                mlflow.log_metric("setup_complete", 1.0)
            
            print("✅ MLflow database initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize MLflow database: {e}")
            return False
    
    return True

def show_mlflow_info():
    """Display MLflow information and usage instructions."""
    print("="*60)
    print("🎯 MLflow Dashboard for Email Engagement Prediction")
    print("="*60)
    print("📊 Features:")
    print("  • Experiment tracking and comparison")
    print("  • Model versioning and registry")
    print("  • Performance metrics visualization")
    print("  • Feature importance tracking")
    print("  • Model deployment management")
    print("")
    print("🔧 Setup:")
    print("  • Install MLflow: pip install mlflow mlflow-sklearn mlflow-xgboost")
    print("  • Run training scripts to populate experiments")
    print("  • Use this script to launch the dashboard")
    print("")
    print("📖 Usage:")
    print("  • View experiments: http://localhost:5000/#/experiments")
    print("  • Model registry: http://localhost:5000/#/models")
    print("  • Compare runs: http://localhost:5000/#/compare")
    print("="*60)

def main():
    """Main function to launch MLflow dashboard."""
    print("🎯 MLflow Dashboard Launcher")
    print("="*40)
    
    # Check MLflow installation
    if not check_mlflow_installation():
        return
    
    # Load configuration
    config = load_mlflow_config()
    if not config:
        return
    
    # Show information
    show_mlflow_info()
    
    # Setup database if needed
    if not setup_mlflow_database():
        return
    
    # Launch UI
    launch_mlflow_ui(config)

if __name__ == "__main__":
    main()
