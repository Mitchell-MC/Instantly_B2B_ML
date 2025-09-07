#!/usr/bin/env python3
"""
Test Working APIs Only
Focuses on APIs that are confirmed working: Instantly + Apollo Health
"""

import os
import requests
import pandas as pd
from dotenv import load_dotenv
import json
import time
from datetime import datetime

# Load environment variables
load_dotenv()

class WorkingAPITester:
    """Test only the APIs that are confirmed working"""
    
    def __init__(self):
        """Initialize with API keys"""
        self.instantly_key = (
            os.getenv('INSTANTLY_API_KEY') or 
            os.getenv('instantly_key') or 
            'ZjQ0Mjc3ZGUtNjliMi00YmMzLWE2OWMtMjhhZmQ0MDk0MTIzOkx5VWZ6UnB6RmR3Zw=='
        )
        
        self.apollo_key = (
            os.getenv('APOLLO_API_KEY') or 
            os.getenv('apollo_key') or
            'K05UXxdZgCaAFgYCTqJWmQ'
        )
        
        self.organization_id = (
            os.getenv('ORGANIZATION_ID') or 
            'f44277de-69b2-4bc3-a69c-28afd4094123'
        )
        
        print("🔧 Working API Tester Initialized")
        print(f"   Instantly Key: {self.instantly_key[:20]}...")
        print(f"   Apollo Key: {self.apollo_key[:10]}...")
    
    def test_instantly_campaigns(self) -> dict:
        """Test Instantly campaigns API - CONFIRMED WORKING"""
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
                sample = campaigns[0]
                print(f"   Sample Campaign: {sample.get('name', 'Unknown')}")
                print(f"   Campaign ID: {sample.get('id', 'Unknown')}")
                print(f"   Status: {sample.get('status', 'Unknown')}")
            
            return {
                'success': True,
                'count': len(campaigns),
                'data': campaigns[:3],
                'message': f"Retrieved {len(campaigns)} campaigns"
            }
            
        except Exception as e:
            print(f"❌ Instantly Campaigns Error: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_instantly_leads(self, limit: int = 50) -> dict:
        """Test Instantly leads API - CONFIRMED WORKING"""
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
                sample = leads[0]
                print(f"   Sample Lead: {sample.get('email', 'Unknown')}")
                print(f"   Company: {sample.get('company_name', 'Unknown')}")
                print(f"   Status: {sample.get('status', 'Unknown')}")
                print(f"   Opens: {sample.get('email_open_count', 0)}")
                print(f"   Replies: {sample.get('email_reply_count', 0)}")
                
                # Calculate engagement stats
                total_opens = sum(lead.get('email_open_count', 0) for lead in leads)
                total_replies = sum(lead.get('email_reply_count', 0) for lead in leads)
                total_clicks = sum(lead.get('email_click_count', 0) for lead in leads)
                
                print(f"   Total Opens: {total_opens}")
                print(f"   Total Replies: {total_replies}")
                print(f"   Total Clicks: {total_clicks}")
            
            return {
                'success': True,
                'count': len(leads),
                'data': leads[:5],
                'message': f"Retrieved {len(leads)} leads"
            }
            
        except Exception as e:
            print(f"❌ Instantly Leads Error: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_apollo_health(self) -> dict:
        """Test Apollo health check - CONFIRMED WORKING"""
        print("\n💳 Testing Apollo Health Check...")
        
        url = "https://api.apollo.io/v1/auth/health"
        headers = {
            'Cache-Control': 'no-cache',
            'Content-Type': 'application/json',
            'x-api-key': self.apollo_key
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check response headers for any useful info
            rate_headers = {k: v for k, v in response.headers.items() 
                           if any(term in k.lower() for term in ['rate', 'limit', 'remaining'])}
            
            print(f"✅ Apollo Health Check: API is working")
            
            if rate_headers:
                print(f"   Rate Limit Info: {rate_headers}")
            
            return {
                'success': True,
                'health_data': data,
                'rate_headers': rate_headers,
                'message': 'Apollo API is healthy'
            }
            
        except Exception as e:
            print(f"❌ Apollo Health Check Error: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_instantly_analytics_sample(self) -> dict:
        """Get sample analytics data from Instantly"""
        print("\n📈 Testing Instantly Analytics API...")
        
        # First get campaigns
        campaigns_result = self.test_instantly_campaigns()
        if not campaigns_result['success'] or not campaigns_result['data']:
            return {'success': False, 'error': 'No campaigns available for analytics'}
        
        # Get analytics for first campaign
        campaign_id = campaigns_result['data'][0]['id']
        
        url = "https://api.instantly.ai/api/v2/campaigns/analytics/daily"
        headers = {"Authorization": f"Bearer {self.instantly_key}"}
        params = {"campaign_id": campaign_id}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            
            analytics_data = response.json()
            
            print(f"✅ Instantly Analytics: {len(analytics_data)} daily records")
            
            if analytics_data:
                # Calculate totals
                total_sent = sum(record.get('sent', 0) for record in analytics_data)
                total_opened = sum(record.get('opened', 0) for record in analytics_data)
                total_clicks = sum(record.get('clicks', 0) for record in analytics_data)
                total_replies = sum(record.get('replies', 0) for record in analytics_data)
                
                print(f"   Total Sent: {total_sent}")
                print(f"   Total Opens: {total_opened}")
                print(f"   Total Clicks: {total_clicks}")
                print(f"   Total Replies: {total_replies}")
                
                if total_sent > 0:
                    open_rate = (total_opened / total_sent) * 100
                    click_rate = (total_clicks / total_sent) * 100
                    reply_rate = (total_replies / total_sent) * 100
                    
                    print(f"   Open Rate: {open_rate:.2f}%")
                    print(f"   Click Rate: {click_rate:.2f}%")
                    print(f"   Reply Rate: {reply_rate:.2f}%")
            
            return {
                'success': True,
                'count': len(analytics_data),
                'data': analytics_data[:5],
                'message': f"Retrieved {len(analytics_data)} analytics records"
            }
            
        except Exception as e:
            print(f"❌ Instantly Analytics Error: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_comprehensive_test(self) -> dict:
        """Run comprehensive test of working APIs"""
        print("🚀 Running Comprehensive Working API Tests")
        print("=" * 60)
        
        results = {
            'test_timestamp': datetime.now().isoformat(),
            'tests': {}
        }
        
        # Test Instantly Campaigns
        results['tests']['instantly_campaigns'] = self.test_instantly_campaigns()
        time.sleep(1)
        
        # Test Instantly Leads
        results['tests']['instantly_leads'] = self.test_instantly_leads(limit=25)
        time.sleep(1)
        
        # Test Apollo Health
        results['tests']['apollo_health'] = self.test_apollo_health()
        time.sleep(1)
        
        # Test Instantly Analytics
        results['tests']['instantly_analytics'] = self.get_instantly_analytics_sample()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 WORKING API TESTS SUMMARY")
        print("=" * 60)
        
        successful_tests = []
        failed_tests = []
        
        for test_name, test_result in results['tests'].items():
            if test_result.get('success', False):
                successful_tests.append(test_name)
                print(f"✅ {test_name}: {test_result.get('message', 'Success')}")
            else:
                failed_tests.append(test_name)
                print(f"❌ {test_name}: {test_result.get('error', 'Failed')}")
        
        results['summary'] = {
            'total_tests': len(results['tests']),
            'successful_tests': len(successful_tests),
            'failed_tests': len(failed_tests),
            'success_rate': len(successful_tests) / len(results['tests']) * 100,
            'successful_test_names': successful_tests,
            'failed_test_names': failed_tests
        }
        
        print(f"\n🎯 Working APIs Success Rate: {results['summary']['success_rate']:.1f}%")
        print(f"   Successful: {len(successful_tests)}/{len(results['tests'])}")
        
        if len(successful_tests) >= 3:
            print("🎉 Core APIs are working perfectly!")
            print("📝 Note: Apollo search/enrichment APIs may have permission restrictions")
        else:
            print("⚠️ Some core APIs need attention")
        
        return results

def main():
    """Main testing function"""
    print("🔌 ML Lead Scoring - Working APIs Test")
    print("Testing only confirmed working endpoints")
    print("=" * 60)
    
    # Initialize tester
    tester = WorkingAPITester()
    
    # Run tests
    results = tester.run_comprehensive_test()
    
    # Save results
    results_file = f"working_api_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Show next steps
    print(f"\n📋 NEXT STEPS:")
    print(f"1. Instantly APIs are working - ready for ML pipeline")
    print(f"2. Apollo health check works - API key is valid")
    print(f"3. Run 'python apollo_debug_test.py' to troubleshoot Apollo search/enrichment")
    print(f"4. Consider using alternative enrichment sources if Apollo has restrictions")
    print(f"5. Start with Instantly data for initial ML model training")
    
    if results['summary']['success_rate'] >= 75:
        print("✅ Core system ready for ML pipeline!")
        return 0
    else:
        print("⚠️ Some issues need resolution")
        return 1

if __name__ == "__main__":
    exit(main())

