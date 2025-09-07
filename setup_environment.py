#!/usr/bin/env python3
"""
Environment Setup Script for ML Lead Scoring System
Creates .env file and validates configuration
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """Create .env file with proper API keys and configuration"""
    
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

# API Keys - From successful Jupyter notebook examples
INSTANTLY_API_KEY=ZjQ0Mjc3ZGUtNjliMi00YmMzLWE2OWMtMjhhZmQ0MDk0MTIzOkx5VWZ6UnB6RmR3Zw==
instantly_key=ZjQ0Mjc3ZGUtNjliMi00YmMzLWE2OWMtMjhhZmQ0MDk0MTIzOkx5VWZ6UnB6RmR3Zw==
APOLLO_API_KEY=K05UXxdZgCaAFgYCTqJWmQ
apollo_key=K05UXxdZgCaAFgYCTqJWmQ

# Monitoring & Alerts
GRAFANA_PASSWORD=grafana_admin_123
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your_email@domain.com
SENDER_PASSWORD=your_email_password
ALERT_RECIPIENTS=recipient@domain.com

# Lead Maturity Configuration (from transcript: 1 month default)
LEAD_MATURITY_DAYS=30
ENVIRONMENT=production

# Apollo Credit Management (from transcript: 20K/month limit)
APOLLO_MONTHLY_LIMIT=20000
APOLLO_RESERVE_CREDITS=2000

# Organization ID (from notebook examples)
ORGANIZATION_ID=f44277de-69b2-4bc3-a69c-28afd4094123
"""
    
    env_file = Path('.env')
    
    if env_file.exists():
        print("⚠️  .env file already exists")
        response = input("Do you want to overwrite it? (y/N): ")
        if response.lower() != 'y':
            print("❌ Keeping existing .env file")
            return False
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ Created .env file successfully")
        return True
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

def validate_configuration():
    """Validate that all required configuration is present"""
    print("\n🔍 Validating Configuration...")
    
    required_vars = [
        'INSTANTLY_API_KEY',
        'instantly_key',
        'APOLLO_API_KEY',
        'apollo_key', 
        'ORGANIZATION_ID',
        'DB_PASSWORD',
        'N8N_USER',
        'N8N_PASSWORD'
    ]
    
    # Load from .env if it exists
    env_file = Path('.env')
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv()
    
    missing_vars = []
    present_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            present_vars.append(var)
            # Don't print full API keys for security
            if 'API_KEY' in var or 'key' in var.lower():
                print(f"   ✅ {var}: {value[:10]}...")
            else:
                print(f"   ✅ {var}: {value}")
        else:
            missing_vars.append(var)
            print(f"   ❌ {var}: Missing")
    
    if missing_vars:
        print(f"\n⚠️  Missing variables: {missing_vars}")
        return False
    else:
        print(f"\n✅ All {len(required_vars)} required variables present")
        return True

def create_directories():
    """Create necessary directories for the project"""
    print("\n📁 Creating Project Directories...")
    
    directories = [
        'logs',
        'models', 
        'data',
        'config',
        'grafana/dashboards',
        'prometheus',
        'nginx'
    ]
    
    created_count = 0
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"   ✅ Created: {directory}")
                created_count += 1
            except Exception as e:
                print(f"   ❌ Failed to create {directory}: {e}")
        else:
            print(f"   ℹ️  Already exists: {directory}")
    
    print(f"\n✅ Created {created_count} new directories")

def validate_api_keys():
    """Test API keys with simple validation calls"""
    print("\n🔐 Validating API Keys...")
    
    try:
        # Test API connections
        from test_api_connections import APIConnectionTester
        tester = APIConnectionTester()
        
        # Run quick validation tests
        instantly_result = tester.test_instantly_campaigns()
        apollo_result = tester.check_apollo_credits()
        
        if instantly_result['success']:
            print("   ✅ Instantly API: Working")
        else:
            print("   ❌ Instantly API: Failed")
            
        if apollo_result['success']:
            print("   ✅ Apollo API: Working")
        else:
            print("   ❌ Apollo API: Failed")
            
        return instantly_result['success'] and apollo_result['success']
        
    except Exception as e:
        print(f"   ⚠️  Could not validate API keys: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 ML Lead Scoring System - Environment Setup")
    print("=" * 60)
    
    # Step 1: Create .env file
    print("\n1️⃣ Setting up environment file...")
    env_created = create_env_file()
    
    # Step 2: Validate configuration
    print("\n2️⃣ Validating configuration...")
    config_valid = validate_configuration()
    
    # Step 3: Create directories
    print("\n3️⃣ Creating project directories...")
    create_directories()
    
    # Step 4: Test API keys (optional)
    print("\n4️⃣ Testing API connections...")
    try:
        api_valid = validate_api_keys()
    except ImportError:
        print("   ⚠️  API validation requires 'requests' and 'pandas' packages")
        api_valid = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SETUP SUMMARY")
    print("=" * 60)
    
    if env_created:
        print("✅ Environment file: Created")
    else:
        print("⚠️  Environment file: Skipped or failed")
        
    if config_valid:
        print("✅ Configuration: Valid")
    else:
        print("❌ Configuration: Invalid - check missing variables")
        
    print("✅ Directories: Created")
    
    if api_valid:
        print("✅ API connections: Working")
    else:
        print("⚠️  API connections: Need verification")
    
    # Next steps
    print("\n📋 NEXT STEPS:")
    print("1. Review and update .env file if needed")
    print("2. Run: python test_api_connections.py")
    print("3. Run: python quickstart.py")
    print("4. Access N8N at: http://localhost:5678")
    
    if config_valid:
        print("\n🎉 Environment setup completed successfully!")
        sys.exit(0)
    else:
        print("\n⚠️  Environment setup completed with warnings")
        sys.exit(1)

if __name__ == "__main__":
    main()
