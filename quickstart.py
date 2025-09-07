#!/usr/bin/env python3
"""
ML Lead Scoring System - Quick Start Script
Based on transcript meeting requirements

This script provides a simple way to get the ML lead scoring system up and running.
It handles initial setup, configuration validation, and service deployment.
"""

import os
import sys
import json
import subprocess
import time
import requests
from pathlib import Path

def print_banner():
    """Print system banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                   ML Lead Scoring System                            ║
║              N8N + Apollo + Instantly Integration                   ║
║                                                                      ║
║                            ║
║  • Two main pipelines (Instantly + Apollo)                         ║
║  • Three-layer data architecture                                    ║
║  • Apollo credit management (20K/month)                            ║
║  • 1-month lead maturity for training                              ║
╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_prerequisites():
    """Check if all prerequisites are installed"""
    print("🔍 Checking prerequisites...")
    
    required_tools = {
        'docker': ['docker', '--version'],
        'docker-compose': ['docker-compose', '--version'],
        'python': ['python', '--version']
    }
    
    missing_tools = []
    
    for tool, command in required_tools.items():
        try:
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  ✅ {tool}: {result.stdout.strip()}")
            else:
                missing_tools.append(tool)
                print(f"  ❌ {tool}: Not found")
        except FileNotFoundError:
            missing_tools.append(tool)
            print(f"  ❌ {tool}: Not found")
    
    if missing_tools:
        print(f"\n❌ Missing required tools: {', '.join(missing_tools)}")
        print("Please install these tools before continuing.")
        return False
    
    print("✅ All prerequisites satisfied!")
    return True

def setup_environment():
    """Setup environment variables and configuration"""
    print("\n🔧 Setting up environment...")
    
    # Create .env file if it doesn't exist
    env_file = Path('.env')
    if not env_file.exists():
        env_content = """# ML Lead Scoring Environment Configuration

# Database Configuration
DB_PASSWORD=ml_password_123
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ml_lead_scoring
DB_USER=postgres

# N8N Configuration  
N8N_USER=admin
N8N_PASSWORD=n8n_admin_123
N8N_HOST=localhost

# API Keys (Please update with your actual keys)
INSTANTLY_API_KEY=your_instantly_api_key_here
APOLLO_API_KEY=your_apollo_api_key_here

# Monitoring & Alerts
GRAFANA_PASSWORD=grafana_admin_123
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your_email@domain.com
SENDER_PASSWORD=your_email_password
ALERT_RECIPIENTS=recipient@domain.com

# Lead Maturity Configuration
LEAD_MATURITY_DAYS=30
ENVIRONMENT=production
"""
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("  ✅ Created .env file")
        print("  ⚠️  Please update .env with your actual API keys!")
    else:
        print("  ✅ .env file already exists")
    
    # Create necessary directories
    directories = ['logs', 'models', 'data', 'grafana/dashboards', 'prometheus']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print(f"  ✅ Created directories: {', '.join(directories)}")

def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("  ✅ Dependencies installed successfully")
        else:
            print(f"  ❌ Failed to install dependencies: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ❌ Error installing dependencies: {e}")
        return False
    
    return True

def validate_configuration():
    """Validate configuration files"""
    print("\n✅ Validating configuration...")
    
    required_files = [
        'ml_lead_scoring_schema.sql',
        'docker-compose.yml',
        'n8n_instantly_ingestion_workflow.json',
        'n8n_apollo_enrichment_workflow.json', 
        'n8n_prediction_workflow.json',
        'config/integration_config.yaml'
    ]
    
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"  ❌ {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing configuration files: {missing_files}")
        return False
    
    print("✅ All configuration files present!")
    return True

def start_services():
    """Start all services using docker-compose"""
    print("\n🚀 Starting services...")
    
    try:
        # Start services
        result = subprocess.run([
            'docker-compose', 'up', '-d'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("  ✅ Services started successfully")
        else:
            print(f"  ❌ Failed to start services: {result.stderr}")
            return False
        
        # Wait for services to be ready
        print("  ⏳ Waiting for services to be ready...")
        time.sleep(30)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error starting services: {e}")
        return False

def check_service_health():
    """Check health of all services"""
    print("\n🏥 Checking service health...")
    
    services = {
        'PostgreSQL': 'http://localhost:5432',  # Will fail but that's expected
        'N8N': 'http://localhost:5678/healthz',
        'Data Processor': 'http://localhost:5000/health',
        'ML Service': 'http://localhost:5001/health',
        'Monitoring': 'http://localhost:5002/health',
        'Grafana': 'http://localhost:3000'
    }
    
    healthy_services = []
    unhealthy_services = []
    
    for service_name, endpoint in services.items():
        try:
            if service_name == 'PostgreSQL':
                # Skip direct PostgreSQL check as it requires proper client
                print(f"  ⏭️  {service_name}: Skipped (database)")
                continue
                
            response = requests.get(endpoint, timeout=10)
            if response.status_code == 200:
                healthy_services.append(service_name)
                print(f"  ✅ {service_name}: Healthy")
            else:
                unhealthy_services.append(service_name)
                print(f"  ⚠️  {service_name}: Responding but not healthy (HTTP {response.status_code})")
        except Exception as e:
            unhealthy_services.append(service_name)
            print(f"  ❌ {service_name}: Not responding ({str(e)[:50]})")
    
    if len(healthy_services) >= 3:  # Most services healthy
        print(f"\n✅ System is mostly healthy ({len(healthy_services)}/{len(services)-1} services)")
        return True
    else:
        print(f"\n⚠️  System needs attention ({len(healthy_services)}/{len(services)-1} services healthy)")
        return False

def show_access_info():
    """Show access information for all services"""
    print("""
🌐 Service Access Information:

📊 N8N Workflow Designer:
   URL: http://localhost:5678
   Username: admin
   Password: n8n_admin_123

📈 Monitoring Dashboard:
   URL: http://localhost:5002
   Real-time system metrics and alerts

📊 Grafana (Advanced Monitoring):
   URL: http://localhost:3000
   Username: admin
   Password: grafana_admin_123

🔧 API Services:
   Data Processor: http://localhost:5000
   ML Service: http://localhost:5001
   API Documentation: http://localhost:5001/docs

🎯 Quick Commands:

   # Check system status
   python ml_lead_scoring_integration.py status

   # Run daily pipeline  
   python ml_lead_scoring_integration.py daily-pipeline

   # Train new model
   python ml_lead_scoring_integration.py train

   # Stop all services
   docker-compose down

📚 Next Steps:

1. Configure API keys in .env file
2. Import N8N workflows from 'N8N automation' folder
3. Set up Apollo and Instantly API credentials in N8N
4. Run the daily pipeline to start data ingestion
5. Monitor system health via the dashboard

✅ Your ML Lead Scoring System is ready!
    """)

def main():
    """Main quickstart function"""
    print_banner()
    
    # Step 1: Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Step 2: Setup environment
    setup_environment()
    
    # Step 3: Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Step 4: Validate configuration
    if not validate_configuration():
        sys.exit(1)
    
    # Step 5: Start services
    if not start_services():
        sys.exit(1)
    
    # Step 6: Check service health
    services_healthy = check_service_health()
    
    # Step 7: Show access information
    show_access_info()
    
    if services_healthy:
        print("🎉 Quickstart completed successfully!")
        sys.exit(0)
    else:
        print("⚠️  Quickstart completed with warnings. Some services may need attention.")
        sys.exit(0)

if __name__ == "__main__":
    main()
