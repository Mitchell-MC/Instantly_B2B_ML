#!/usr/bin/env python3
"""
Email Enrichment Pipeline
Accepts email addresses, pulls data from Instantly, enriches with Apollo, and adds to database.
Follows the Bronze→Silver→Gold architecture pattern.
"""

import os
import sys
import json
import logging
import argparse
import requests
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
import csv
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from the specified path following system instructions
load_dotenv("C:\\Users\\mccal\\Downloads\\Instantly B2B Main\\Instantly.env")

# Configure logging following the system's logging patterns
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/email_enrichment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EmailEnrichmentPipeline:
    """Pipeline to enrich emails with Instantly and Apollo data following Bronze→Silver→Gold architecture"""
    
    def __init__(self):
        """Initialize the pipeline with API clients and database connection"""
        self.setup_directories()
        self.load_credentials()
        self.setup_database()
        
    def setup_directories(self):
        """Create required directories following system patterns"""
        required_dirs = ['logs', 'data', 'models']
        for directory in required_dirs:
            os.makedirs(directory, exist_ok=True)
        logger.info("✅ Directory structure verified")
        
    def load_credentials(self):
        """Load credentials from environment variables following system configuration patterns"""
        # API Keys
        self.instantly_api_key = os.getenv('INSTANTLY_API_KEY')
        self.apollo_api_key = os.getenv('APOLLO_API_KEY')
        
        # Database credentials following the system's database configuration
        self.db_config = {
            'host': os.getenv('DB_HOST', '127.0.0.1'),
            'port': int(os.getenv('DB_PORT', '5431')),
            'database': os.getenv('DB_NAME', 'postgres'),
            'user': os.getenv('DB_USER', 'mitchell'),
            'password': os.getenv('DB_PASSWORD'),
            'sslmode': 'require' if os.getenv('DB_SSL', 'true').lower() == 'true' else 'prefer'
        }
        
        # Validate required credentials
        if not self.instantly_api_key:
            raise ValueError("INSTANTLY_API_KEY environment variable is required")
        if not self.apollo_api_key:
            raise ValueError("APOLLO_API_KEY environment variable is required")
        if not self.db_config['password']:
            raise ValueError("DB_PASSWORD environment variable is required")
            
        logger.info("✅ Credentials loaded from environment")
        
    def setup_database(self):
        """Setup database connection following PostgreSQL connection pooling patterns"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.conn.autocommit = False
            logger.info("✅ Database connection established")
            
            # Verify required tables exist following ml_lead_scoring schema
            self.verify_database_schema()
            
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise
            
    def verify_database_schema(self):
        """Verify that required database tables exist following schema-first design"""
        required_tables = [
            'bronze_instantly_leads',
            'bronze_apollo_enrichment', 
            'silver_enriched_leads'
        ]
        
        with self.conn.cursor() as cursor:
            for table in required_tables:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'ml_lead_scoring' 
                        AND table_name = %s
                    )
                """, (table,))
                
                if not cursor.fetchone()[0]:
                    logger.warning(f"⚠️ Table ml_lead_scoring.{table} does not exist")
                    
        logger.info("✅ Database schema verification complete")
        
    def load_enriched_contacts_schema(self, file_path: str) -> pd.DataFrame:
        """Load enriched_contacts.csv with proper schema detection"""
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            logger.info(f"📊 Enriched contacts loaded: {len(df)} rows, {len(df.columns)} columns")
            logger.info(f"📋 Schema columns: {list(df.columns)}")
            
            # Validate required columns for the enrichment pipeline
            required_email_columns = ['email', 'Email', 'email_address', 'lead_email']
            email_column = None
            
            for col in required_email_columns:
                if col in df.columns:
                    email_column = col
                    break
            
            if email_column is None:
                # Look for any column containing 'email'
                email_cols = [col for col in df.columns if 'email' in col.lower()]
                if email_cols:
                    email_column = email_cols[0]
                else:
                    raise ValueError(f"No email column found. Available columns: {list(df.columns)}")
            
            logger.info(f"📧 Using email column: '{email_column}'")
            
            # Clean and validate email data
            df = df.dropna(subset=[email_column])
            df = df[df[email_column].str.contains('@', na=False)]
            df = df.drop_duplicates(subset=[email_column])
            
            logger.info(f"✅ Processed {len(df)} valid unique contacts")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading enriched contacts: {e}")
            raise
            
    def extract_emails_from_csv(self, file_path: str, email_column: str = None) -> List[str]:
        """Extract emails from CSV file with enhanced schema detection"""
        try:
            # Special handling for enriched_contacts.csv
            if 'enriched_contacts.csv' in file_path:
                df = self.load_enriched_contacts_schema(file_path)
                # Find the email column automatically
                email_columns = ['email', 'Email', 'email_address', 'lead_email']
                for col in email_columns:
                    if col in df.columns:
                        email_column = col
                        break
                
                if email_column is None:
                    email_cols = [col for col in df.columns if 'email' in col.lower()]
                    email_column = email_cols[0] if email_cols else df.columns[0]
                
                emails = df[email_column].dropna().unique().tolist()
                emails = [str(email).strip() for email in emails if email and '@' in str(email)]
                logger.info(f"✅ Extracted {len(emails)} emails from enriched contacts")
                return emails
            
            # Standard CSV processing
            df = pd.read_csv(file_path)
            logger.info(f"📊 CSV file loaded with {len(df)} rows and columns: {list(df.columns)}")
            
            # Auto-detect email column if not specified
            if email_column is None:
                email_columns = ['email', 'Email', 'EMAIL', 'email_address', 'Email Address', 'lead_email']
                for col in email_columns:
                    if col in df.columns:
                        email_column = col
                        break
                
                if email_column is None:
                    # Look for columns containing 'email'
                    email_cols = [col for col in df.columns if 'email' in col.lower()]
                    if email_cols:
                        email_column = email_cols[0]
                    else:
                        logger.error(f"❌ No email column found. Available columns: {list(df.columns)}")
                        return []
            
            logger.info(f"📧 Using email column: '{email_column}'")
            
            # Extract emails and remove duplicates/empty values
            emails = df[email_column].dropna().unique().tolist()
            emails = [str(email).strip() for email in emails if email and '@' in str(email)]
            
            logger.info(f"✅ Extracted {len(emails)} unique valid emails from CSV")
            return emails
            
        except Exception as e:
            logger.error(f"❌ Error reading CSV file: {e}")
            return []
            
    def extract_emails_from_file(self, file_path: str, email_column: str = None) -> List[str]:
        """Extract emails from various file formats with enhanced support for enriched_contacts.csv"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.csv':
            return self.extract_emails_from_csv(file_path, email_column)
        elif file_ext in ['.txt', '.text']:
            # Text file - one email per line
            try:
                with open(file_path, 'r') as f:
                    emails = [line.strip() for line in f if line.strip() and '@' in line]
                logger.info(f"✅ Extracted {len(emails)} emails from text file")
                return emails
            except Exception as e:
                logger.error(f"❌ Error reading text file: {e}")
                return []
        else:
            logger.error(f"❌ Unsupported file format: {file_ext}")
            return []
        
    def fetch_instantly_data(self, email: str) -> Optional[Dict[str, Any]]:
        """Fetch lead data from Instantly API with retry logic and exponential backoff"""
        try:
            # Use the correct Instantly API endpoint
            url = "https://api.instantly.ai/api/v1/lead/get"
            headers = {
                "Content-Type": "application/json"
            }
            params = {
                "api_key": self.instantly_api_key,
                "email": email
            }
            
            logger.info(f"Fetching Instantly data for: {email}")
            
            # Implement retry logic following system patterns
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, headers=headers, params=params, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        logger.info(f"✅ Instantly data retrieved for {email}")
                        return data
                    elif response.status_code == 404:
                        logger.warning(f"⚠️ No Instantly data found for {email}")
                        return None
                    elif response.status_code == 401:
                        logger.error(f"❌ Instantly API authentication failed for {email} - check API key")
                        return None
                    elif response.status_code == 429:
                        # Rate limit - wait and retry
                        wait_time = 2 ** attempt
                        logger.warning(f"⚠️ Rate limited, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"❌ Instantly API error for {email}: {response.status_code} - {response.text}")
                        return None
                        
                except requests.RequestException as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.warning(f"⚠️ Request failed, retrying in {wait_time}s: {e}")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"❌ Max retries exceeded for {email}: {e}")
                        return None
                        
        except Exception as e:
            logger.error(f"❌ Error fetching Instantly data for {email}: {e}")
            return None
            
    def fetch_apollo_data(self, email: str, company_domain: str = None) -> Optional[Dict[str, Any]]:
        """Fetch enrichment data from Apollo API with rate limiting (200/day default)"""
        try:
            url = "https://api.apollo.io/v1/people/match"
            headers = {
                "Content-Type": "application/json",
                "Cache-Control": "no-cache",
                "X-Api-Key": self.apollo_api_key
            }
            
            payload = {
                "email": email
            }
            
            if company_domain:
                payload["organization_domain"] = company_domain
                
            logger.info(f"Fetching Apollo data for: {email}")
            
            # Respect Apollo rate limits (200/day)
            time.sleep(0.5)  # Rate limiting delay
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('person'):
                    logger.info(f"✅ Apollo data retrieved for {email}")
                    return data
                else:
                    logger.warning(f"⚠️ No Apollo person data found for {email}")
                    return None
            elif response.status_code == 403:
                logger.error(f"❌ Apollo API access denied - check API key and credits")
                return None
            elif response.status_code == 429:
                logger.error(f"❌ Apollo API rate limit exceeded")
                return None
            else:
                logger.error(f"❌ Apollo API error for {email}: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error fetching Apollo data for {email}: {e}")
            return None
            
    def insert_bronze_instantly_data(self, email: str, instantly_data: Dict[str, Any]) -> Optional[str]:
        """Insert Instantly data into bronze_instantly_leads table following Bronze layer patterns"""
        try:
            with self.conn.cursor() as cursor:
                # Extract lead data
                lead_data = instantly_data.get('lead', {})
                
                # Follow the bronze table schema from ml_lead_scoring_schema.sql
                insert_query = """
                    INSERT INTO ml_lead_scoring.bronze_instantly_leads 
                    (lead_id, email, first_name, last_name, company_name, 
                     status, created_date, updated_date, raw_data)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (email) DO UPDATE SET
                        updated_date = EXCLUDED.updated_date,
                        raw_data = EXCLUDED.raw_data
                    RETURNING lead_id
                """
                
                lead_id = lead_data.get('id', f"manual_{int(time.time())}_{hash(email) % 10000}")
                
                cursor.execute(insert_query, (
                    lead_id,
                    email,
                    lead_data.get('first_name'),
                    lead_data.get('last_name'),
                    lead_data.get('company_name'),
                    lead_data.get('status'),
                    datetime.now(timezone.utc),
                    datetime.now(timezone.utc),
                    json.dumps(instantly_data)
                ))
                
                result = cursor.fetchone()
                self.conn.commit()
                
                logger.info(f"✅ Instantly data inserted for {email}")
                return result[0] if result else lead_id
                
        except Exception as e:
            logger.error(f"❌ Error inserting Instantly data for {email}: {e}")
            self.conn.rollback()
            return None
            
    def insert_bronze_apollo_data(self, lead_id: str, email: str, apollo_data: Dict[str, Any]) -> bool:
        """Insert Apollo data into bronze_apollo_enrichment table following Bronze layer patterns"""
        try:
            with self.conn.cursor() as cursor:
                person_data = apollo_data.get('person', {})
                organization_data = person_data.get('organization', {})
                
                # Follow the bronze table schema
                insert_query = """
                    INSERT INTO ml_lead_scoring.bronze_apollo_enrichment 
                    (lead_id, email, apollo_person_id, person_data, organization_data, 
                     enriched_date, raw_data)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (email) DO UPDATE SET
                        person_data = EXCLUDED.person_data,
                        organization_data = EXCLUDED.organization_data,
                        enriched_date = EXCLUDED.enriched_date,
                        raw_data = EXCLUDED.raw_data
                """
                
                cursor.execute(insert_query, (
                    lead_id,
                    email,
                    person_data.get('id'),
                    json.dumps(person_data),
                    json.dumps(organization_data),
                    datetime.now(timezone.utc),
                    json.dumps(apollo_data)
                ))
                
                self.conn.commit()
                logger.info(f"✅ Apollo data inserted for {email}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error inserting Apollo data for {email}: {e}")
            self.conn.rollback()
            return False
            
    def create_silver_enriched_record(self, lead_id: str, email: str) -> bool:
        """Create enriched record in silver_enriched_leads table following Silver layer patterns"""
        try:
            with self.conn.cursor() as cursor:
                # Combine data from bronze tables following Bronze→Silver transformation patterns
                query = """
                    INSERT INTO ml_lead_scoring.silver_enriched_leads 
                    (lead_id, email, first_name, last_name, company_name, 
                     job_title, seniority_level, organization_employees, 
                     industry, founded_year, organization_linkedin_url,
                     person_linkedin_url, created_date, updated_date)
                    SELECT 
                        il.lead_id,
                        il.email,
                        COALESCE(
                            (ae.person_data->>'first_name'), 
                            il.first_name
                        ) as first_name,
                        COALESCE(
                            (ae.person_data->>'last_name'), 
                            il.last_name
                        ) as last_name,
                        COALESCE(
                            (ae.organization_data->>'name'), 
                            il.company_name
                        ) as company_name,
                        ae.person_data->>'title' as job_title,
                        ae.person_data->>'seniority' as seniority_level,
                        CAST(ae.organization_data->>'estimated_num_employees' as INTEGER) as organization_employees,
                        ae.organization_data->>'industry' as industry,
                        CAST(ae.organization_data->>'founded_year' as INTEGER) as founded_year,
                        ae.organization_data->>'linkedin_url' as organization_linkedin_url,
                        ae.person_data->>'linkedin_url' as person_linkedin_url,
                        NOW() as created_date,
                        NOW() as updated_date
                    FROM ml_lead_scoring.bronze_instantly_leads il
                    LEFT JOIN ml_lead_scoring.bronze_apollo_enrichment ae ON il.email = ae.email
                    WHERE il.lead_id = %s
                    ON CONFLICT (email) DO UPDATE SET
                        first_name = EXCLUDED.first_name,
                        last_name = EXCLUDED.last_name,
                        company_name = EXCLUDED.company_name,
                        job_title = EXCLUDED.job_title,
                        seniority_level = EXCLUDED.seniority_level,
                        organization_employees = EXCLUDED.organization_employees,
                        industry = EXCLUDED.industry,
                        founded_year = EXCLUDED.founded_year,
                        organization_linkedin_url = EXCLUDED.organization_linkedin_url,
                        person_linkedin_url = EXCLUDED.person_linkedin_url,
                        updated_date = EXCLUDED.updated_date
                """
                
                cursor.execute(query, (lead_id,))
                self.conn.commit()
                
                logger.info(f"✅ Silver enriched record created for {email}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error creating silver record for {email}: {e}")
            self.conn.rollback()
            return False
            
    def process_email(self, email: str) -> Dict[str, Any]:
        """Process a single email through the enrichment pipeline following Bronze→Silver→Gold architecture"""
        logger.info(f"🚀 Processing email: {email}")
        
        result = {
            'email': email,
            'status': 'processing',
            'instantly_data': False,
            'apollo_data': False,
            'database_insert': False,
            'errors': []
        }
        
        try:
            # Step 1: Fetch Instantly data (Bronze layer input)
            instantly_data = self.fetch_instantly_data(email)
            if instantly_data:
                result['instantly_data'] = True
                
                # Step 2: Insert Instantly data to bronze layer
                lead_id = self.insert_bronze_instantly_data(email, instantly_data)
                if lead_id:
                    # Step 3: Fetch Apollo data for enrichment
                    company_domain = None
                    if instantly_data.get('lead', {}).get('company_name'):
                        # Extract domain from company name (enhanced approach)
                        company_name = instantly_data['lead']['company_name'].lower().strip()
                        # Remove common suffixes and clean the name
                        company_name = company_name.replace(' inc', '').replace(' llc', '').replace(' corp', '')
                        company_name = company_name.replace(' ', '').replace('-', '')
                        company_domain = f"{company_name}.com"
                    
                    apollo_data = self.fetch_apollo_data(email, company_domain)
                    if apollo_data:
                        result['apollo_data'] = True
                        
                        # Step 4: Insert Apollo data to bronze layer
                        self.insert_bronze_apollo_data(lead_id, email, apollo_data)
                    
                    # Step 5: Create silver layer record (Bronze→Silver transformation)
                    if self.create_silver_enriched_record(lead_id, email):
                        result['database_insert'] = True
                        result['status'] = 'completed'
                    else:
                        result['errors'].append('Failed to create silver record')
                        result['status'] = 'partial_failure'
                else:
                    result['errors'].append('Failed to insert Instantly data')
                    result['status'] = 'failed'
            else:
                # No Instantly data - try Apollo only and create minimal bronze record
                apollo_data = self.fetch_apollo_data(email)
                if apollo_data:
                    result['apollo_data'] = True
                    # Create minimal instantly record for Apollo enrichment
                    minimal_instantly = {
                        'lead': {
                            'id': f"apollo_only_{int(time.time())}_{hash(email) % 10000}",
                            'email': email,
                            'status': 'apollo_enriched'
                        }
                    }
                    lead_id = self.insert_bronze_instantly_data(email, minimal_instantly)
                    if lead_id:
                        self.insert_bronze_apollo_data(lead_id, email, apollo_data)
                        if self.create_silver_enriched_record(lead_id, email):
                            result['database_insert'] = True
                            result['status'] = 'apollo_only'
                        else:
                            result['status'] = 'partial_failure'
                    else:
                        result['status'] = 'failed'
                else:
                    result['errors'].append('No data found from either Instantly or Apollo')
                    result['status'] = 'no_data'
                
        except Exception as e:
            logger.error(f"❌ Error processing {email}: {e}")
            result['errors'].append(str(e))
            result['status'] = 'error'
            
        logger.info(f"✅ Completed processing {email}: {result['status']}")
        return result
        
    def process_emails(self, emails: List[str]) -> Dict[str, Any]:
        """Process multiple emails through the enrichment pipeline with comprehensive reporting"""
        logger.info(f"🚀 Starting email enrichment pipeline for {len(emails)} emails")
        
        results = {
            'total_emails': len(emails),
            'completed': 0,
            'failed': 0,
            'no_data': 0,
            'partial_failures': 0,
            'apollo_only': 0,
            'details': []
        }
        
        for i, email in enumerate(emails, 1):
            logger.info(f"📧 Processing email {i}/{len(emails)}: {email}")
            
            # Add delay to respect API rate limits following system patterns
            time.sleep(1)
            
            result = self.process_email(email)
            results['details'].append(result)
            
            # Update counters
            status = result['status']
            if status == 'completed':
                results['completed'] += 1
            elif status in ['failed', 'error']:
                results['failed'] += 1
            elif status == 'no_data':
                results['no_data'] += 1
            elif status == 'partial_failure':
                results['partial_failures'] += 1
            elif status == 'apollo_only':
                results['apollo_only'] += 1
                
        logger.info(f"🎉 Pipeline completed: {results['completed']} completed, {results['failed']} failed")
        return results
        
    def close(self):
        """Close database connection following proper connection handling"""
        if hasattr(self, 'conn'):
            self.conn.close()
            logger.info("✅ Database connection closed")

def main():
    """Main function to run the email enrichment pipeline following system CLI patterns"""
    parser = argparse.ArgumentParser(
        description='Email Enrichment Pipeline - Process emails with Instantly + Apollo data (Bronze→Silver→Gold)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process enriched contacts file (recommended)
  python email_enrichment_pipeline.py --file enriched_contacts.csv

  # Process with limit for testing
  python email_enrichment_pipeline.py --file enriched_contacts.csv --limit 5

  # Process single email
  python email_enrichment_pipeline.py --emails john@company.com

  # Process multiple emails
  python email_enrichment_pipeline.py --emails john@company.com jane@corp.com

  # Process CSV with specific email column
  python email_enrichment_pipeline.py --file data.csv --email-column "Email Address"

  # Save results to JSON for analysis
  python email_enrichment_pipeline.py --file enriched_contacts.csv --output enrichment_results.json
        """
    )
    parser.add_argument('--emails', nargs='+', help='Email addresses to process')
    parser.add_argument('--file', help='CSV or text file containing email addresses (supports enriched_contacts.csv)')
    parser.add_argument('--email-column', help='Column name containing emails (auto-detected for enriched_contacts.csv)')
    parser.add_argument('--output', help='JSON file to save results for analysis')
    parser.add_argument('--limit', type=int, help='Limit number of emails to process (for testing)')
    
    args = parser.parse_args()
    
    # Get email list
    emails = []
    if args.emails:
        emails.extend(args.emails)
    
    if args.file:
        try:
            pipeline_temp = EmailEnrichmentPipeline()
            file_emails = pipeline_temp.extract_emails_from_file(args.file, args.email_column)
            emails.extend(file_emails)
            pipeline_temp.close()
        except Exception as e:
            logger.error(f"❌ Error reading email file: {e}")
            return
    
    if not emails:
        logger.error("❌ No emails provided. Use --emails or --file parameter")
        parser.print_help()
        return
    
    # Apply limit if specified
    if args.limit:
        emails = emails[:args.limit]
        logger.info(f"📋 Limited to first {args.limit} emails")
    
    # Process emails
    pipeline = None
    try:
        pipeline = EmailEnrichmentPipeline()
        results = pipeline.process_emails(emails)
        
        # Print comprehensive summary following system reporting patterns
        print("\n" + "="*70)
        print("EMAIL ENRICHMENT PIPELINE RESULTS (Bronze→Silver→Gold)")
        print("="*70)
        print(f"Total emails processed: {results['total_emails']}")
        print(f"✅ Completed (Both APIs): {results['completed']}")
        print(f"🔶 Apollo Only: {results['apollo_only']}")
        print(f"❌ Failed: {results['failed']}")
        print(f"⚠️ No data: {results['no_data']}")
        print(f"🔶 Partial failures: {results['partial_failures']}")
        
        # Calculate success metrics
        total_successful = results['completed'] + results['apollo_only']
        success_rate = (total_successful / results['total_emails']) * 100 if results['total_emails'] > 0 else 0
        enrichment_rate = (results['completed'] / results['total_emails']) * 100 if results['total_emails'] > 0 else 0
        
        print(f"📊 Overall Success Rate: {success_rate:.1f}%")
        print(f"📊 Full Enrichment Rate: {enrichment_rate:.1f}%")
        print("="*70)
        
        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"✅ Results saved to {args.output}")
            
    except Exception as e:
        logger.error(f"❌ Pipeline error: {e}")
        return 1
    finally:
        if pipeline:
            pipeline.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
