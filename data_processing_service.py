#!/usr/bin/env python3
"""
Data Processing Service for ML Lead Scoring
Handles transformation from Bronze → Silver → Gold layers
"""

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from flask import Flask, request, jsonify
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, db_config: Dict[str, str]):
        """Initialize data processor with database configuration"""
        self.db_config = db_config
        self.app = Flask(__name__)
        self.setup_routes()
        
        # API configurations from environment or defaults (from notebook examples)
        self.instantly_config = {
            'api_key': (
                os.getenv('INSTANTLY_API_KEY') or 
                os.getenv('instantly_key') or 
                'ZjQ0Mjc3ZGUtNjliMi00YmMzLWE2OWMtMjhhZmQ0MDk0MTIzOkx5VWZ6UnB6RmR3Zw=='
            ),
            'organization_id': (
                os.getenv('ORGANIZATION_ID') or 
                'f44277de-69b2-4bc3-a69c-28afd4094123'
            ),
            'base_url': 'https://api.instantly.ai/api/v2'
        }
        
        self.apollo_config = {
            'api_key': (
                os.getenv('APOLLO_API_KEY') or 
                'K05UXxdZgCaAFgYCTqJWmQ'
            ),
            'monthly_limit': int(os.getenv('APOLLO_MONTHLY_LIMIT', '20000')),
            'reserve_credits': int(os.getenv('APOLLO_RESERVE_CREDITS', '2000')),
            'base_url': 'https://api.apollo.io/v1'
        }
        
    def get_db_connection(self):
        """Create database connection"""
        return psycopg2.connect(**self.db_config)
    
    def setup_routes(self):
        """Setup Flask API routes for triggering processing"""
        
        @self.app.route('/api/process-silver-layer', methods=['POST'])
        def process_silver_layer_endpoint():
            try:
                data = request.get_json()
                lead_ids = data.get('lead_ids', [])
                
                if not lead_ids:
                    return jsonify({'error': 'No lead IDs provided'}), 400
                
                # Process silver layer in background
                threading.Thread(
                    target=self.process_bronze_to_silver,
                    args=(lead_ids,)
                ).start()
                
                return jsonify({
                    'status': 'processing',
                    'message': f'Started silver layer processing for {len(lead_ids)} leads'
                })
                
            except Exception as e:
                logger.error(f"Error in silver layer endpoint: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/process-gold-layer', methods=['POST'])
        def process_gold_layer_endpoint():
            try:
                # Process gold layer (maturity filter)
                threading.Thread(target=self.process_silver_to_gold).start()
                
                return jsonify({
                    'status': 'processing',
                    'message': 'Started gold layer processing'
                })
                
            except Exception as e:
                logger.error(f"Error in gold layer endpoint: {str(e)}")
                return jsonify({'error': str(e)}), 500
    
    def process_bronze_to_silver(self, lead_ids: List[str] = None):
        """
        Transform bronze layer data to silver layer with ML features
        """
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Query bronze data with Apollo enrichment
            if lead_ids:
                placeholders = ','.join(['%s'] * len(lead_ids))
                query = f"""
                SELECT 
                    i.*,
                    a.company_size,
                    a.company_revenue,
                    a.company_industry,
                    a.company_location,
                    a.employee_count,
                    a.technologies,
                    a.social_media_profiles,
                    a.company_description,
                    a.funding_info
                FROM ml_lead_scoring.bronze_instantly_leads i
                LEFT JOIN ml_lead_scoring.bronze_apollo_enrichment a 
                    ON i.lead_id = a.lead_id
                WHERE i.lead_id IN ({placeholders})
                """
                cursor.execute(query, lead_ids)
            else:
                query = """
                SELECT 
                    i.*,
                    a.company_size,
                    a.company_revenue,
                    a.company_industry,
                    a.company_location,
                    a.employee_count,
                    a.technologies,
                    a.social_media_profiles,
                    a.company_description,
                    a.funding_info
                FROM ml_lead_scoring.bronze_instantly_leads i
                LEFT JOIN ml_lead_scoring.bronze_apollo_enrichment a 
                    ON i.lead_id = a.lead_id
                WHERE i.updated_timestamp >= NOW() - INTERVAL '1 day'
                """
                cursor.execute(query)
            
            bronze_data = cursor.fetchall()
            logger.info(f"Processing {len(bronze_data)} leads for silver layer")
            
            # Process each lead
            silver_records = []
            for lead in bronze_data:
                features = self.extract_ml_features(lead)
                silver_records.append(features)
            
            # Bulk upsert to silver layer
            if silver_records:
                self.bulk_upsert_silver(silver_records, conn)
                logger.info(f"Successfully processed {len(silver_records)} leads to silver layer")
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error processing bronze to silver: {str(e)}")
            if 'conn' in locals():
                conn.rollback()
                conn.close()
    
    def extract_ml_features(self, lead_data: Dict) -> Dict:
        """
        Extract ML features from bronze layer data
        """
        lead_id = lead_data['lead_id']
        
        # Calculate engagement features
        open_rate = float(lead_data.get('open_rate', 0))
        click_rate = float(lead_data.get('click_rate', 0))
        reply_rate = float(lead_data.get('reply_rate', 0))
        bounce_rate = float(lead_data.get('bounce_rate', 0))
        
        # Calculate days since last activity
        last_activity = lead_data.get('last_activity_date')
        days_since_activity = 0
        if last_activity:
            try:
                last_activity_dt = pd.to_datetime(last_activity)
                days_since_activity = (datetime.now() - last_activity_dt).days
            except:
                days_since_activity = 999  # Default for invalid dates
        
        # Calculate lead age
        created_date = lead_data.get('created_date')
        lead_age_days = 0
        if created_date:
            try:
                created_dt = pd.to_datetime(created_date)
                lead_age_days = (datetime.now() - created_dt).days
            except:
                lead_age_days = 0
        
        # Process company features from Apollo
        company_size_category = self.categorize_company_size(
            lead_data.get('company_size'), 
            lead_data.get('employee_count')
        )
        
        revenue_category = self.categorize_revenue(lead_data.get('company_revenue'))
        industry_category = self.categorize_industry(lead_data.get('company_industry'))
        
        # Technology stack analysis
        technologies = lead_data.get('technologies')
        has_tech_stack = False
        if technologies:
            try:
                tech_data = json.loads(technologies) if isinstance(technologies, str) else technologies
                has_tech_stack = len(tech_data) > 0 if isinstance(tech_data, list) else bool(tech_data)
            except:
                has_tech_stack = False
        
        # Social media presence
        social_profiles = lead_data.get('social_media_profiles')
        social_presence_score = 0.0
        if social_profiles:
            try:
                social_data = json.loads(social_profiles) if isinstance(social_profiles, str) else social_profiles
                if isinstance(social_data, dict):
                    social_presence_score = min(len([v for v in social_data.values() if v]), 3) / 3.0
            except:
                social_presence_score = 0.0
        
        # Calculate composite engagement score
        engagement_score = self.calculate_engagement_score(
            open_rate, click_rate, reply_rate, bounce_rate, days_since_activity
        )
        
        # Determine engagement trend (simplified)
        engagement_trend = 'stable'  # Would need historical data for accurate trending
        if engagement_score > 0.7:
            engagement_trend = 'increasing'
        elif engagement_score < 0.3:
            engagement_trend = 'decreasing'
        
        # Create feature vector for ML
        feature_vector = {
            'open_rate': open_rate,
            'click_rate': click_rate,
            'reply_rate': reply_rate,
            'bounce_rate': bounce_rate,
            'days_since_activity': min(days_since_activity, 365),  # Cap at 1 year
            'lead_age_days': min(lead_age_days, 365),
            'engagement_score': engagement_score,
            'company_size_encoded': self.encode_company_size(company_size_category),
            'revenue_encoded': self.encode_revenue(revenue_category),
            'industry_encoded': self.encode_industry(industry_category),
            'has_tech_stack': int(has_tech_stack),
            'social_presence_score': social_presence_score
        }
        
        # Determine if lead is qualified (simplified logic)
        is_qualified = self.determine_lead_qualification(feature_vector)
        lead_quality_score = engagement_score * 0.6 + social_presence_score * 0.2 + (1 - bounce_rate) * 0.2
        
        return {
            'lead_id': lead_id,
            'avg_open_rate': open_rate,
            'avg_click_rate': click_rate,
            'avg_reply_rate': reply_rate,
            'engagement_trend': engagement_trend,
            'days_since_last_activity': days_since_activity,
            'total_emails_sent': 1,  # Would need to aggregate from campaign data
            'company_size_category': company_size_category,
            'revenue_category': revenue_category,
            'industry_category': industry_category,
            'employee_count_bucket': self.bucket_employee_count(lead_data.get('employee_count')),
            'has_technology_stack': has_tech_stack,
            'social_media_presence_score': social_presence_score,
            'lead_age_days': lead_age_days,
            'response_velocity': reply_rate / max(lead_age_days, 1) if lead_age_days > 0 else 0,
            'email_frequency': 1 / max(lead_age_days, 1) if lead_age_days > 0 else 0,
            'engagement_score': engagement_score,
            'is_qualified_lead': is_qualified,
            'lead_quality_score': lead_quality_score,
            'feature_vector': json.dumps(feature_vector)
        }
    
    def categorize_company_size(self, size_str: str, employee_count: int) -> str:
        """Categorize company size"""
        if employee_count:
            if employee_count < 10:
                return 'startup'
            elif employee_count < 50:
                return 'small'
            elif employee_count < 250:
                return 'medium'
            elif employee_count < 1000:
                return 'large'
            else:
                return 'enterprise'
        
        if size_str:
            size_lower = size_str.lower()
            if 'startup' in size_lower or '1-10' in size_lower:
                return 'startup'
            elif 'small' in size_lower or '11-50' in size_lower:
                return 'small'
            elif 'medium' in size_lower or '51-250' in size_lower:
                return 'medium'
            elif 'large' in size_lower or '251-1000' in size_lower:
                return 'large'
            elif 'enterprise' in size_lower or '1000+' in size_lower:
                return 'enterprise'
        
        return 'unknown'
    
    def categorize_revenue(self, revenue_str: str) -> str:
        """Categorize company revenue"""
        if not revenue_str:
            return 'unknown'
        
        revenue_lower = revenue_str.lower()
        if 'million' in revenue_lower:
            # Extract number
            import re
            numbers = re.findall(r'\d+', revenue_lower)
            if numbers:
                revenue_mil = int(numbers[0])
                if revenue_mil < 1:
                    return 'sub_1m'
                elif revenue_mil < 10:
                    return '1m_10m'
                elif revenue_mil < 100:
                    return '10m_100m'
                else:
                    return 'over_100m'
        
        return 'unknown'
    
    def categorize_industry(self, industry_str: str) -> str:
        """Categorize industry"""
        if not industry_str:
            return 'unknown'
        
        industry_lower = industry_str.lower()
        
        # Technology
        if any(term in industry_lower for term in ['software', 'technology', 'tech', 'saas', 'it']):
            return 'technology'
        
        # Finance
        elif any(term in industry_lower for term in ['finance', 'financial', 'banking', 'investment']):
            return 'finance'
        
        # Healthcare
        elif any(term in industry_lower for term in ['healthcare', 'medical', 'health', 'pharmaceutical']):
            return 'healthcare'
        
        # Manufacturing
        elif any(term in industry_lower for term in ['manufacturing', 'industrial', 'automotive']):
            return 'manufacturing'
        
        # Services
        elif any(term in industry_lower for term in ['consulting', 'services', 'professional']):
            return 'services'
        
        # Retail
        elif any(term in industry_lower for term in ['retail', 'ecommerce', 'commerce']):
            return 'retail'
        
        else:
            return 'other'
    
    def bucket_employee_count(self, count: int) -> str:
        """Bucket employee count"""
        if not count:
            return 'unknown'
        
        if count < 10:
            return '1-10'
        elif count < 50:
            return '11-50'
        elif count < 250:
            return '51-250'
        elif count < 1000:
            return '251-1000'
        else:
            return '1000+'
    
    def encode_company_size(self, category: str) -> int:
        """Encode company size for ML"""
        encoding = {
            'startup': 1, 'small': 2, 'medium': 3, 
            'large': 4, 'enterprise': 5, 'unknown': 0
        }
        return encoding.get(category, 0)
    
    def encode_revenue(self, category: str) -> int:
        """Encode revenue for ML"""
        encoding = {
            'sub_1m': 1, '1m_10m': 2, '10m_100m': 3, 
            'over_100m': 4, 'unknown': 0
        }
        return encoding.get(category, 0)
    
    def encode_industry(self, category: str) -> int:
        """Encode industry for ML"""
        encoding = {
            'technology': 1, 'finance': 2, 'healthcare': 3,
            'manufacturing': 4, 'services': 5, 'retail': 6,
            'other': 7, 'unknown': 0
        }
        return encoding.get(category, 0)
    
    def calculate_engagement_score(self, open_rate: float, click_rate: float, 
                                 reply_rate: float, bounce_rate: float, 
                                 days_since_activity: int) -> float:
        """Calculate composite engagement score"""
        # Weight the different engagement metrics
        engagement = (
            open_rate * 0.3 +
            click_rate * 0.25 +
            reply_rate * 0.35 +
            (1 - bounce_rate) * 0.1
        )
        
        # Apply recency penalty
        if days_since_activity > 30:
            recency_factor = max(0.1, 1 - (days_since_activity - 30) / 365)
            engagement *= recency_factor
        
        return min(max(engagement, 0), 1)  # Clamp between 0 and 1
    
    def determine_lead_qualification(self, features: Dict) -> bool:
        """Simple lead qualification logic"""
        return (
            features['engagement_score'] > 0.5 and
            features['bounce_rate'] < 0.3 and
            features['days_since_activity'] < 60
        )
    
    def bulk_upsert_silver(self, records: List[Dict], conn):
        """Bulk upsert records to silver layer"""
        cursor = conn.cursor()
        
        # Prepare the upsert query
        columns = [
            'lead_id', 'avg_open_rate', 'avg_click_rate', 'avg_reply_rate',
            'engagement_trend', 'days_since_last_activity', 'total_emails_sent',
            'company_size_category', 'revenue_category', 'industry_category',
            'employee_count_bucket', 'has_technology_stack', 'social_media_presence_score',
            'lead_age_days', 'response_velocity', 'email_frequency', 'engagement_score',
            'is_qualified_lead', 'lead_quality_score', 'feature_vector'
        ]
        
        placeholders = ', '.join(['%s'] * len(columns))
        column_names = ', '.join(columns)
        
        # Create conflict resolution for upsert
        update_columns = [f"{col} = EXCLUDED.{col}" for col in columns if col != 'lead_id']
        
        query = f"""
        INSERT INTO ml_lead_scoring.silver_ml_features ({column_names})
        VALUES ({placeholders})
        ON CONFLICT (lead_id) 
        DO UPDATE SET {', '.join(update_columns)}, updated_timestamp = CURRENT_TIMESTAMP
        """
        
        # Prepare data tuples
        data_tuples = []
        for record in records:
            data_tuple = tuple(record.get(col) for col in columns)
            data_tuples.append(data_tuple)
        
        # Execute bulk insert
        cursor.executemany(query, data_tuples)
        cursor.close()
    
    def process_silver_to_gold(self):
        """
        Move mature data (1+ month old) from silver to gold layer
        """
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Find mature leads (1+ month old, not already in gold)
            query = """
            SELECT s.*, 
                   EXTRACT(DAYS FROM (CURRENT_TIMESTAMP - s.created_timestamp)) as data_age_days
            FROM ml_lead_scoring.silver_ml_features s
            LEFT JOIN ml_lead_scoring.gold_training_data g ON s.lead_id = g.lead_id
            WHERE s.created_timestamp <= CURRENT_TIMESTAMP - INTERVAL '30 days'
              AND g.lead_id IS NULL
              AND s.is_qualified_lead IS NOT NULL
            """
            
            cursor.execute(query)
            mature_leads = cursor.fetchall()
            
            logger.info(f"Found {len(mature_leads)} mature leads for gold layer")
            
            # Process leads for gold layer
            gold_records = []
            for lead in mature_leads:
                gold_record = self.prepare_gold_record(lead)
                gold_records.append(gold_record)
            
            # Insert into gold layer
            if gold_records:
                self.bulk_insert_gold(gold_records, conn)
                logger.info(f"Moved {len(gold_records)} leads to gold layer")
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error processing silver to gold: {str(e)}")
            if 'conn' in locals():
                conn.rollback()
                conn.close()
    
    def prepare_gold_record(self, lead_data: Dict) -> Dict:
        """Prepare record for gold layer"""
        # Determine final outcome based on engagement patterns
        engagement_score = lead_data.get('engagement_score', 0)
        days_since_activity = lead_data.get('days_since_last_activity', 999)
        is_qualified = lead_data.get('is_qualified_lead', False)
        
        # Simple outcome classification
        if engagement_score > 0.7 and is_qualified:
            outcome = 'converted'
            target_label = 2  # High-value
        elif engagement_score > 0.4 and is_qualified:
            outcome = 'ongoing'
            target_label = 1  # Qualified
        elif days_since_activity > 60:
            outcome = 'cold'
            target_label = 0  # Not qualified
        else:
            outcome = 'disqualified'
            target_label = 0  # Not qualified
        
        return {
            'lead_id': lead_data['lead_id'],
            'feature_vector': lead_data.get('feature_vector'),
            'target_label': target_label,
            'lead_outcome': outcome,
            'final_engagement_score': engagement_score,
            'conversion_timestamp': None,  # Would be set based on actual conversion data
            'training_eligible': True,
            'data_maturity_days': lead_data.get('data_age_days', 30)
        }
    
    def bulk_insert_gold(self, records: List[Dict], conn):
        """Bulk insert records to gold layer"""
        cursor = conn.cursor()
        
        columns = [
            'lead_id', 'feature_vector', 'target_label', 'lead_outcome',
            'final_engagement_score', 'conversion_timestamp', 'training_eligible',
            'data_maturity_days'
        ]
        
        placeholders = ', '.join(['%s'] * len(columns))
        column_names = ', '.join(columns)
        
        query = f"""
        INSERT INTO ml_lead_scoring.gold_training_data ({column_names})
        VALUES ({placeholders})
        """
        
        data_tuples = []
        for record in records:
            data_tuple = tuple(record.get(col) for col in columns)
            data_tuples.append(data_tuple)
        
        cursor.executemany(query, data_tuples)
        cursor.close()
    
    def run_service(self, host='0.0.0.0', port=5000):
        """Run the Flask service"""
        logger.info(f"Starting data processing service on {host}:{port}")
        self.app.run(host=host, port=port, debug=False, threaded=True)

if __name__ == "__main__":
    # Database configuration
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME', 'ml_lead_scoring'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'password')
    }
    
    # Initialize and run service
    processor = DataProcessor(db_config)
    
    # Run background job for gold layer processing every 6 hours
    def background_gold_processing():
        while True:
            time.sleep(6 * 3600)  # 6 hours
            processor.process_silver_to_gold()
    
    threading.Thread(target=background_gold_processing, daemon=True).start()
    
    # Start the Flask service
    processor.run_service()
