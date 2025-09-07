#!/usr/bin/env python3
"""
Apollo API Setup Verification
Verifies that the Apollo API is configured correctly using notebook patterns
"""

import os
from dotenv import load_dotenv

def verify_apollo_setup():
    """Verify Apollo environment setup"""
    print("🔍 Apollo API Setup Verification")
    print("=" * 40)
    
    # Load environment variables
    load_dotenv()
    
    # Check both environment variable patterns
    apollo_api_key = os.getenv('APOLLO_API_KEY')
    apollo_key = os.getenv('apollo_key')
    
    print(f"Environment Variables:")
    print(f"  APOLLO_API_KEY: {'✅ Set' if apollo_api_key else '❌ Not set'}")
    print(f"  apollo_key: {'✅ Set' if apollo_key else '❌ Not set'}")
    
    if apollo_api_key:
        print(f"  APOLLO_API_KEY value: {apollo_api_key[:10]}...")
    if apollo_key:
        print(f"  apollo_key value: {apollo_key[:10]}...")
    
    # Determine which key to use (prioritizing APOLLO_API_KEY)
    final_key = apollo_api_key or apollo_key
    
    if final_key:
        print(f"\n✅ Apollo API key available: {final_key[:10]}...")
        
        # Show the correct header format from notebook
        print(f"\n📋 Correct Apollo API Usage (from notebook):")
        print(f"  URL: https://api.apollo.io/api/v1/people/match")
        print(f"  Headers:")
        print(f"    'accept': 'application/json'")
        print(f"    'Cache-Control': 'no-cache'")
        print(f"    'Content-Type': 'application/json'")
        print(f"    'x-api-key': '{final_key[:10]}...'  # Lowercase x-api-key!")
        
        return True
    else:
        print(f"\n❌ No Apollo API key found")
        print(f"Please set either APOLLO_API_KEY or apollo_key in your .env file")
        return False

def create_env_with_apollo():
    """Create/update .env file with correct apollo_key"""
    print(f"\n🔧 Updating .env file with apollo_key...")
    
    env_lines = []
    apollo_key_found = False
    
    # Read existing .env file if it exists
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('apollo_key='):
                    apollo_key_found = True
                    env_lines.append('apollo_key=K05UXxdZgCaAFgYCTqJWmQ')
                else:
                    env_lines.append(line)
    
    # Add apollo_key if not found
    if not apollo_key_found:
        env_lines.append('apollo_key=K05UXxdZgCaAFgYCTqJWmQ')
    
    # Write updated .env file
    with open('.env', 'w') as f:
        for line in env_lines:
            f.write(line + '\n')
    
    print("✅ Updated .env file with apollo_key")

if __name__ == "__main__":
    if not verify_apollo_setup():
        create_env_with_apollo()
        print("\n🔄 Re-checking after update...")
        verify_apollo_setup()
    
    print(f"\n🚀 Next step: Run 'python test_api_connections.py' to test Apollo API")
