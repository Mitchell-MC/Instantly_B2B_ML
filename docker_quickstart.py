#!/usr/bin/env python3
"""
Docker Quick Start Script
Helps users quickly deploy and manage the ML pipeline using Docker.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"🚀 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def check_docker():
    """Check if Docker is installed and running."""
    print("🔍 Checking Docker installation...")
    
    # Check Docker installation
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker found: {result.stdout.strip()}")
        else:
            print("❌ Docker not found. Please install Docker first.")
            return False
    except FileNotFoundError:
        print("❌ Docker not found. Please install Docker first.")
        return False
    
    # Check Docker daemon
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker daemon is running")
            return True
        else:
            print("❌ Docker daemon is not running. Please start Docker.")
            return False
    except Exception as e:
        print(f"❌ Error checking Docker daemon: {e}")
        return False

def check_docker_compose():
    """Check if Docker Compose is available."""
    print("🔍 Checking Docker Compose...")
    
    try:
        result = subprocess.run(["docker-compose", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker Compose found: {result.stdout.strip()}")
            return True
        else:
            print("❌ Docker Compose not found. Please install Docker Compose.")
            return False
    except FileNotFoundError:
        print("❌ Docker Compose not found. Please install Docker Compose.")
        return False

def build_images():
    """Build Docker images."""
    print("🔨 Building Docker images...")
    
    # Build base image
    if not run_command("docker build -t email-engagement-ml:base .", "Building base image"):
        return False
    
    # Build production image
    if not run_command("docker build --target production -t email-engagement-ml:production .", "Building production image"):
        return False
    
    # Build development image
    if not run_command("docker build --target development -t email-engagement-ml:development .", "Building development image"):
        return False
    
    # Build Jupyter image
    if not run_command("docker build --target jupyter -t email-engagement-ml:jupyter .", "Building Jupyter image"):
        return False
    
    print("✅ All Docker images built successfully")
    return True

def start_services(profile="default"):
    """Start Docker services."""
    print(f"🚀 Starting services with profile: {profile}")
    
    if profile == "default":
        command = "docker-compose up -d"
    else:
        command = f"docker-compose --profile {profile} up -d"
    
    if not run_command(command, f"Starting services with profile {profile}"):
        return False
    
    print("✅ Services started successfully")
    return True

def stop_services():
    """Stop Docker services."""
    print("🛑 Stopping services...")
    
    if not run_command("docker-compose down", "Stopping services"):
        return False
    
    print("✅ Services stopped successfully")
    return False

def show_status():
    """Show service status."""
    print("📊 Service Status:")
    
    run_command("docker-compose ps", "Checking service status")
    
    print("\n📊 Container Status:")
    run_command("docker ps -a", "Checking container status")
    
    print("\n📊 Resource Usage:")
    run_command("docker stats --no-stream", "Checking resource usage")

def show_logs(service=None):
    """Show service logs."""
    if service:
        print(f"📋 Logs for {service}:")
        run_command(f"docker-compose logs {service}", f"Showing logs for {service}")
    else:
        print("📋 All service logs:")
        run_command("docker-compose logs", "Showing all logs")

def create_notebooks_directory():
    """Create notebooks directory for Jupyter."""
    notebooks_dir = Path("notebooks")
    notebooks_dir.mkdir(exist_ok=True)
    
    # Create sample notebook
    sample_notebook = notebooks_dir / "sample_analysis.ipynb"
    if not sample_notebook.exists():
        sample_content = '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Email Engagement ML Pipeline Analysis\\n",
    "\\n",
    "This notebook provides examples of how to use the ML pipeline components.\\n",
    "\\n",
    "## Available Components\\n",
    "- Feature Engineering\\n",
    "- Model Training\\n",
    "- Prediction\\n",
    "- Monitoring\\n",
    "- Drift Detection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import pipeline components\\n",
    "import sys\\n",
    "sys.path.append('../src')\\n",
    "\\n",
    "from feature_engineering import create_xgboost_optimized_features\\n",
    "from advanced_drift_detection import AdvancedDriftDetector\\n",
    "\\n",
    "print('Pipeline components imported successfully!')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}'''
        sample_notebook.write_text(sample_content)
        print("✅ Created sample notebook in notebooks/ directory")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Docker Quick Start for ML Pipeline")
    parser.add_argument("action", choices=["start", "stop", "status", "logs", "build", "setup"],
                       help="Action to perform")
    parser.add_argument("--profile", default="default",
                       choices=["default", "dev", "analysis", "mlflow", "cache", "database", "production"],
                       help="Docker Compose profile to use")
    parser.add_argument("--service", help="Service name for logs command")
    
    args = parser.parse_args()
    
    print("🐳 Email Engagement ML Pipeline - Docker Quick Start")
    print("=" * 60)
    
    # Check prerequisites
    if not check_docker():
        sys.exit(1)
    
    if not check_docker_compose():
        sys.exit(1)
    
    # Perform requested action
    if args.action == "setup":
        print("🔧 Setting up development environment...")
        create_notebooks_directory()
        print("✅ Setup completed")
        
    elif args.action == "build":
        if not build_images():
            sys.exit(1)
        
    elif args.action == "start":
        if not start_services(args.profile):
            sys.exit(1)
        
        print(f"\n🎉 Services started successfully!")
        print(f"📊 API Service: http://localhost:8000")
        print(f"📚 API Docs: http://localhost:8000/docs")
        
        if args.profile == "analysis":
            print(f"🔬 Jupyter Lab: http://localhost:8888")
        
        if args.profile == "mlflow":
            print(f"📈 MLflow UI: http://localhost:5000")
        
        print(f"\n💡 Use 'python docker_quickstart.py status' to check service status")
        print(f"💡 Use 'python docker_quickstart.py logs' to view logs")
        
    elif args.action == "stop":
        if not stop_services():
            sys.exit(1)
        
    elif args.action == "status":
        show_status()
        
    elif args.action == "logs":
        show_logs(args.service)
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
