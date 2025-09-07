#!/usr/bin/env python3
"""
Test API Connections
Based on successful Jupyter notebook examples

This script validates that the Instantly and Apollo APIs are working correctly
using the same patterns from your successful notebook implementations.
"""

import os
import sys
import requests
import pandas as pd
from dotenv import load_dotenv
import json
import time
from typing import Dict, List, Optional

# Load environment variables
load_dotenv()

class APIConnectionTester:
    """Test API connections for Instantly and Apollo"""
    
    def __init__(self):
        """Initialize with API keys from environment or defaults"""
        # Instantly API - from notebook examples
        self.instantly_key = (
            os.getenv('INSTANTLY_API_KEY') or 
            os.getenv('instantly_key') or 
            'ZjQ0Mjc3ZGUtNjliMi00YmMzLWE2OWMtMjhhZmQ0MDk0MTIzOkx5VWZ6UnB6RmR3Zw=='
        )
        
        # Apollo API - from environment variables (following notebook pattern)
        self.apollo_key = (
            os.getenv('APOLLO_API_KEY') or 
            os.getenv('apollo_key') or  # Notebook uses this variable name
            'K05UXxdZgCaAFgYCTqJWmQ'
        )
        
        # Organization ID from notebook examples
        self.organization_id = (
            os.getenv('ORGANIZATION_ID') or 
            'f44277de-69b2-4bc3-a69c-28afd4094123'
        )
        
        print("🔧 API Connection Tester Initialized")
        print(f"   Instantly Key: {self.instantly_key[:20]}...")
        print(f"   Apollo Key: {self.apollo_key[:10]}...")
        print(f"   Organization ID: {self.organization_id}")
    
    def test_instantly_campaigns(self) -> Dict:
        """Test Instantly campaigns API - from notebook example"""
        print("\n📊 Testing Instantly Campaigns API...")
        
        url = "https://api.instantly.ai/api/v2/campaigns"
        headers = {"Authorization": f"Bearer {self.instantly_key}"}
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            campaigns = data.get('items', [])
            
            print(f"✅ Instantly Campaigns: {len(campaigns)} campaigns retrieved")
            
            if campaigns:
                # Show sample campaign data
                sample = campaigns[0]
                print(f"   Sample Campaign: {sample.get('name', 'Unknown')}")
                print(f"   Campaign ID: {sample.get('id', 'Unknown')}")
                print(f"   Status: {sample.get('status', 'Unknown')}")
            
            return {
                'success': True,
                'count': len(campaigns),
                'data': campaigns[:3],  # First 3 for verification
                'message': f"Retrieved {len(campaigns)} campaigns"
            }
            
        except Exception as e:
            print(f"❌ Instantly Campaigns Error: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to retrieve campaigns'
            }
    
    def test_instantly_leads(self, limit: int = 50) -> Dict:
        """Test Instantly leads API - from notebook example"""
        print(f"\n👥 Testing Instantly Leads API (limit: {limit})...")
        
        url = "https://api.instantly.ai/api/v2/leads/list"
        headers = {
            "Authorization": f"Bearer {self.instantly_key}",
            "Content-Type": "application/json"
        }
        
        payload = {"limit": limit}
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            leads = data.get('items', [])
            
            print(f"✅ Instantly Leads: {len(leads)} leads retrieved")
            
            if leads:
                # Show sample lead data
                sample = leads[0]
                print(f"   Sample Lead: {sample.get('email', 'Unknown')}")
                print(f"   Company: {sample.get('company_name', 'Unknown')}")
                print(f"   Status: {sample.get('status', 'Unknown')}")
                print(f"   Opens: {sample.get('email_open_count', 0)}")
                print(f"   Replies: {sample.get('email_reply_count', 0)}")
            
            return {
                'success': True,
                'count': len(leads),
                'data': leads[:3],  # First 3 for verification
                'message': f"Retrieved {len(leads)} leads"
            }
            
        except Exception as e:
            print(f"❌ Instantly Leads Error: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to retrieve leads'
            }
    
    def test_apollo_people_search(self) -> Dict:
        """Test Apollo people search API - try both header formats"""
        print("\n🔍 Testing Apollo People Search API...")
        
        url = "https://api.apollo.io/api/v1/mixed_people/search"
        
        # Try notebook pattern first (lowercase x-api-key)
        headers_v1 = {
            'accept': 'application/json',
            'Cache-Control': 'no-cache',
            'Content-Type': 'application/json',
            'x-api-key': self.apollo_key
        }
        
        # Try documentation pattern (X-Api-Key) as backup
        headers_v2 = {
            'Cache-Control': 'no-cache',
            'Content-Type': 'application/json',
            'X-Api-Key': self.apollo_key
        }
        
        # Simple test search
        payload = {
            "page": 1,
            "per_page": 5,
            "organization_locations": ["United States"],
            "person_titles": ["CEO", "CTO"]
        }
        
        # Try notebook pattern first
        try:
            print("   Trying notebook pattern (x-api-key)...")
            response = requests.post(url, headers=headers_v1, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            people = data.get('people', [])
            pagination = data.get('pagination', {})
            
            print(f"✅ Apollo People Search: {len(people)} people retrieved (notebook pattern)")
            print(f"   Total Available: {pagination.get('total_entries', 'Unknown')}")
            
            if people:
                sample = people[0]
                print(f"   Sample Person: {sample.get('name', 'Unknown')}")
                print(f"   Title: {sample.get('title', 'Unknown')}")
                print(f"   Company: {sample.get('organization', {}).get('name', 'Unknown')}")
            
            return {
                'success': True,
                'count': len(people),
                'pagination': pagination,
                'data': people[:2],
                'message': f"Retrieved {len(people)} people (notebook pattern)"
            }
            
        except Exception as e1:
            print(f"   Notebook pattern failed: {e1}")
            
            # Try documentation pattern as backup
            try:
                print("   Trying documentation pattern (X-Api-Key)...")
                response = requests.post(url, headers=headers_v2, json=payload, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                people = data.get('people', [])
                pagination = data.get('pagination', {})
                
                print(f"✅ Apollo People Search: {len(people)} people retrieved (docs pattern)")
                
                return {
                    'success': True,
                    'count': len(people),
                    'pagination': pagination,
                    'data': people[:2],
                    'message': f"Retrieved {len(people)} people (docs pattern)"
                }
                
            except Exception as e2:
                print(f"   Documentation pattern also failed: {e2}")
                return {
                    'success': False,
                    'error': f"Both patterns failed: {e1} | {e2}",
                    'message': 'Failed to search Apollo people'
                }
    
    def test_apollo_enrichment(self, linkedin_url: str = "https://www.linkedin.com/in/sameh-suleiman-asaad-22a8b359") -> Dict:
        """Test Apollo LinkedIn enrichment API - using real LinkedIn from notebook"""
        print(f"\n🔍 Testing Apollo LinkedIn Enrichment API...")
        
        url = "https://api.apollo.io/api/v1/people/match"
        
        # Try notebook pattern first (this is what works in your notebook)
        headers_v1 = {
            'accept': 'application/json',
            'Cache-Control': 'no-cache',
            'Content-Type': 'application/json',
            'x-api-key': self.apollo_key
        }
        
        # Try documentation pattern as backup
        headers_v2 = {
            'Cache-Control': 'no-cache',
            'Content-Type': 'application/json',
            'X-Api-Key': self.apollo_key
        }
        
        # Use LinkedIn URL pattern from notebook (using real URL from your data)
        payload = {
            "linkedin_url": linkedin_url,
            "reveal_personal_emails": False,
            "reveal_phone_number": False
        }
        
        # Try notebook pattern first
        try:
            print("   Trying notebook pattern (x-api-key)...")
            response = requests.post(url, headers=headers_v1, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            person = data.get('person', {})
            
            if person:
                print(f"✅ Apollo Enrichment: Successfully enriched LinkedIn profile (notebook pattern)")
                print(f"   Name: {person.get('name', 'Unknown')}")
                print(f"   Title: {person.get('title', 'Unknown')}")
                print(f"   Company: {person.get('organization', {}).get('name', 'Unknown')}")
                print(f"   Email: {person.get('email', 'Not available')}")
                
                return {
                    'success': True,
                    'data': person,
                    'message': f"Successfully enriched LinkedIn profile (notebook pattern)"
                }
            else:
                print(f"⚠️ Apollo Enrichment: No data found for LinkedIn profile")
                return {
                    'success': True,
                    'data': None,
                    'message': f"No enrichment data found for LinkedIn profile"
                }
            
        except Exception as e1:
            print(f"   Notebook pattern failed: {e1}")
            
            # Try documentation pattern as backup
            try:
                print("   Trying documentation pattern (X-Api-Key)...")
                response = requests.post(url, headers=headers_v2, json=payload, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                person = data.get('person', {})
                
                if person:
                    print(f"✅ Apollo Enrichment: Successfully enriched (docs pattern)")
                    print(f"   Name: {person.get('name', 'Unknown')}")
                    print(f"   Title: {person.get('title', 'Unknown')}")
                    print(f"   Company: {person.get('organization', {}).get('name', 'Unknown')}")
                    
                    return {
                        'success': True,
                        'data': person,
                        'message': f"Successfully enriched LinkedIn profile (docs pattern)"
                    }
                else:
                    return {
                        'success': True,
                        'data': None,
                        'message': f"No enrichment data found"
                    }
                    
            except Exception as e2:
                print(f"   Documentation pattern also failed: {e2}")
                return {
                    'success': False,
                    'error': f"Both patterns failed: {e1} | {e2}",
                    'message': 'Failed to enrich LinkedIn profile'
                }
    
    def check_apollo_credits(self) -> Dict:
        """Check Apollo API credit usage"""
        print("\n💳 Checking Apollo Credit Usage...")
        
        url = "https://api.apollo.io/v1/auth/health"
        headers = {
            'Cache-Control': 'no-cache',
            'Content-Type': 'application/json',
            'X-Api-Key': self.apollo_key
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Try to get credit information from headers or response
            remaining_credits = response.headers.get('X-Ratelimit-Remaining')
            monthly_limit = response.headers.get('X-Ratelimit-Limit')
            
            print(f"✅ Apollo Health Check: API is working")
            
            if remaining_credits:
                print(f"   Remaining Credits: {remaining_credits}")
            if monthly_limit:
                print(f"   Monthly Limit: {monthly_limit}")
            
            return {
                'success': True,
                'remaining_credits': remaining_credits,
                'monthly_limit': monthly_limit,
                'health_data': data,
                'message': 'Apollo API is healthy'
            }
            
        except Exception as e:
            print(f"❌ Apollo Health Check Error: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to check Apollo health'
            }
    
    def run_comprehensive_test(self) -> Dict:
        """Run comprehensive test of all API endpoints"""
        print("🚀 Running Comprehensive API Tests")
        print("=" * 50)
        
        results = {
            'test_timestamp': pd.Timestamp.now().isoformat(),
            'tests': {}
        }
        
        # Test Instantly Campaigns
        results['tests']['instantly_campaigns'] = self.test_instantly_campaigns()
        time.sleep(1)  # Rate limiting
        
        # Test Instantly Leads
        results['tests']['instantly_leads'] = self.test_instantly_leads(limit=25)
        time.sleep(1)  # Rate limiting
        
        # Test Apollo Health
        results['tests']['apollo_health'] = self.check_apollo_credits()
        time.sleep(1)  # Rate limiting
        
        # Test Apollo People Search
        results['tests']['apollo_people'] = self.test_apollo_people_search()
        time.sleep(1)  # Rate limiting
        
        # Test Apollo Enrichment (with a real LinkedIn URL from notebook data)
        results['tests']['apollo_enrichment'] = self.test_apollo_enrichment("https://www.linkedin.com/in/sameh-suleiman-asaad-22a8b359")
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 50)
        
        successful_tests = []
        failed_tests = []
        
        for test_name, test_result in results['tests'].items():
            if test_result.get('success', False):
                successful_tests.append(test_name)
                print(f"✅ {test_name}: {test_result.get('message', 'Success')}")
            else:
                failed_tests.append(test_name)
                print(f"❌ {test_name}: {test_result.get('message', 'Failed')}")
        
        results['summary'] = {
            'total_tests': len(results['tests']),
            'successful_tests': len(successful_tests),
            'failed_tests': len(failed_tests),
            'success_rate': len(successful_tests) / len(results['tests']) * 100,
            'successful_test_names': successful_tests,
            'failed_test_names': failed_tests
        }
        
        print(f"\n🎯 Overall Success Rate: {results['summary']['success_rate']:.1f}%")
        print(f"   Successful: {len(successful_tests)}/{len(results['tests'])}")
        
        if len(successful_tests) >= 3:  # At least 3 out of 5 tests should pass
            print("🎉 API connections are working well!")
        else:
            print("⚠️ Some API connections need attention")
        
        return results

def main():
    """Main testing function"""
    print("🔌 ML Lead Scoring API Connection Tests")
    print("Based on successful Jupyter notebook examples")
    print("=" * 60)
    
    # Initialize tester
    tester = APIConnectionTester()
    
    # Run comprehensive tests
    results = tester.run_comprehensive_test()
    
    # Save results to file
    results_file = f"api_test_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Exit with appropriate code
    if results['summary']['success_rate'] >= 60:  # 60% success rate minimum
        print("✅ API tests completed successfully!")
        sys.exit(0)
    else:
        print("❌ API tests completed with issues")
        sys.exit(1)

if __name__ == "__main__":
    main()
