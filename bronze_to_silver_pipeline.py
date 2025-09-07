"""
Bronze to Silver Layer ETL Pipeline for B2B Lead Scoring
Transforms raw bronze data into ML-ready silver layer features

Architecture: PostgreSQL Bronze → Feature Engineering → PostgreSQL Silver
Author: Senior Data Engineering Team
"""

import sys
import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine, text
from pathlib import Path
import yaml
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/bronze_to_silver_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BronzeToSilverPipeline:
    """
    Production-grade ETL pipeline for bronze to silver layer transformation
    """
    
    def __init__(self, config_path: str = "config/silver_layer_config.yaml"):
        """Initialize pipeline with configuration"""
        self.config = self._load_config(config_path)
        self.engine = self._create_db_connection()
        self.batch_size = self.config.get('processing', {}).get('batch_size', 1000)
        
        # Feature engineering configuration based on ACTUAL schema (no prefixes)
        self.target_variable = "engagement_level"
        
        # Text columns for feature engineering (actual column names)
        self.text_columns = [
            'campaign', 'personalization', 'payload', 'status_summary',
            'headline', 'title', 'apollo_name'  # Apollo text fields
        ]
        
        # Timestamp columns (actual column names)
        self.timestamp_columns = [
            'timestamp_created', 'timestamp_updated', 'timestamp_last_contact',
            'timestamp_last_touch', 'timestamp_last_open', 'timestamp_last_click',
            'timestamp_last_reply', 'timestamp_last_interest_change', 'retrieval_timestamp',
            'inserted_at', 'enriched_at'
        ]
        
        # Categorical columns (actual column names)
        self.categorical_columns = [
            # Core categorical features
            'organization', 'status', 'upload_method', 'assigned_to', 'verification_status',
            'enrichment_status', 'api_status',
            # Apollo categorical features
            'city', 'state', 'country', 'seniority', 'departments', 'functions'
        ]
        
        # Apollo enrichment columns (valuable for B2B scoring)
        self.apollo_columns = [
            'apollo_id', 'apollo_name', 'title', 'headline', 'city', 'state', 'country',
            'organization_name', 'organization_website', 'organization_phone',
            'organization_industry', 'organization_employees', 'organization_founded_year',
            'seniority', 'departments', 'functions', 'employment_history',
            'organization_data', 'account_data', 'api_response_raw',
            'credits_consumed', 'enriched_at', 'api_status'
        ]
        
        # JSONB columns for enrichment features (original pipeline compatibility)
        self.jsonb_columns = [
            'employment_history', 'organization_data', 'account_data', 'api_response_raw'
        ]
        
        # Columns to exclude from features (leakage prevention)
        self.leakage_columns = [
            # Direct engagement metrics (these create the target)
            'email_reply_count', 'email_click_count', 'email_open_count',
            # Engagement details that would leak future information
            'email_opened_variant', 'email_opened_step', 'email_clicked_variant', 
            'email_clicked_step', 'email_replied_variant', 'email_replied_step',
            # Future timestamps
            'timestamp_last_open', 'timestamp_last_reply', 'timestamp_last_click',
            'timestamp_last_touch', 'timestamp_last_interest_change',
            # Interest status (potential leakage)
            'it_interest_status'
        ]
        
        # PII columns to exclude from features
        self.pii_columns = [
            'email', 'first_name', 'last_name', 'company_name', 'phone', 'website',
            'company_domain', 'linkedin_url', 'organization_website', 'organization_phone'
        ]
        
        # High-cardinality columns that need special handling
        self.high_cardinality_columns = [
            'list_id', 'assigned_to', 'uploaded_by_user', 'apollo_id',
            'organization_name', 'employment_history', 'organization_data',
            'account_data', 'api_response_raw'
        ]
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def _create_db_connection(self):
        """Create database connection using SQLAlchemy"""
        try:
            db_config = self.config['database']
            connection_string = (
                f"postgresql://{db_config['user']}:{db_config['password']}"
                f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            )
            # Add connection parameters to set default schema
            engine = create_engine(
                connection_string,
                connect_args={
                    "options": "-csearch_path=leads"  # Set default schema to leads
                }
            )
            logger.info("Database connection established with leads schema")
            return engine
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def extract_bronze_data(self, incremental: bool = True, lookback_days: int = 7) -> pd.DataFrame:
        """
        Extract data from bronze layer with optional incremental processing
        
        Args:
            incremental: If True, only process recent data
            lookback_days: Days to look back for incremental processing
        """
        try:
            if incremental:
                cutoff_date = datetime.now() - timedelta(days=lookback_days)
                query = text("""
                SELECT * FROM leads.enriched_contacts 
                WHERE timestamp_created >= :cutoff_date1 
                   OR timestamp_updated >= :cutoff_date2
                ORDER BY timestamp_created DESC
                """)
                df = pd.read_sql(query, self.engine, params={
                    'cutoff_date1': cutoff_date, 
                    'cutoff_date2': cutoff_date
                })
                logger.info(f"Extracted {len(df)} incremental records from bronze layer")
            else:
                query = text("SELECT * FROM leads.enriched_contacts")
                df = pd.read_sql(query, self.engine)
                logger.info(f"Extracted {len(df)} total records from bronze layer")
            
            return df
        except Exception as e:
            logger.error(f"Bronze data extraction failed: {e}")
            raise
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engagement level target variable
        
        Engagement Levels:
        0: No engagement
        1: Email opened only
        2: Clicked or replied (high engagement)
        """
        try:
            # Ensure numeric columns (actual column names - no mapping needed)
            engagement_cols = ['email_open_count', 'email_click_count', 'email_reply_count']
            for col in engagement_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Create engagement level
            conditions = [
                ((df['email_click_count'] > 0) | (df['email_reply_count'] > 0)),  # High engagement
                (df['email_open_count'] > 0)  # Medium engagement
            ]
            choices = [2, 1]
            df[self.target_variable] = np.select(conditions, choices, default=0)
            
            # Log target distribution
            target_dist = df[self.target_variable].value_counts().sort_index()
            logger.info(f"Target variable distribution: {dict(target_dist)}")
            
            return df
        except Exception as e:
            logger.error(f"Target variable creation failed: {e}")
            raise
    
    def engineer_timestamp_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced timestamp feature engineering"""
        try:
            logger.info("Engineering timestamp features...")
            
            # Convert timestamp columns
            for col in self.timestamp_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
            
            # Focus on timestamp_created as primary reference
            if 'timestamp_created' in df.columns:
                ref_col = df['timestamp_created']
                
                # Basic datetime features
                df['created_day_of_week'] = ref_col.dt.dayofweek
                df['created_month'] = ref_col.dt.month
                df['created_hour'] = ref_col.dt.hour
                df['created_quarter'] = ref_col.dt.quarter
                
                # Business-relevant features (original pipeline compatibility)
                df['created_is_weekend'] = (ref_col.dt.dayofweek >= 5).astype(int)
                df['created_is_business_hours'] = (
                    (ref_col.dt.hour >= 9) & (ref_col.dt.hour <= 17)
                ).astype(int)
                df['created_is_morning'] = (ref_col.dt.hour <= 12).astype(int)
                df['created_season'] = ref_col.dt.month % 12 // 3
                
                # Recency features
                current_time = pd.Timestamp.now(tz='UTC')
                df['days_since_creation'] = (current_time - ref_col).dt.days
                df['weeks_since_creation'] = df['days_since_creation'] // 7
                
                # Time differences between events
                for col in self.timestamp_columns:
                    if col in df.columns and col != 'timestamp_created':
                        # Clean up feature names
                        clean_name = col.replace('timestamp_', '').replace('_at', '')
                        feature_name = f"days_between_{clean_name}_and_created"
                        df[feature_name] = (df[col] - ref_col).dt.days
                        df[f"has_{clean_name}"] = df[col].notna().astype(int)
                        
                        # Absolute time differences (original pipeline compatibility)
                        df[f"{feature_name}_abs"] = df[feature_name].abs()
                        
                        # Additional aliases for exact original compatibility
                        if col == 'enriched_at':
                            df['days_between_enriched_at_and_created'] = df[feature_name]
                            df['days_between_enriched_at_and_created_abs'] = df[f"{feature_name}_abs"]
                            df['has_enriched_at'] = df[f"has_{clean_name}"]
                        elif col == 'inserted_at':
                            df['days_between_inserted_at_and_created'] = df[feature_name]
                            df['days_between_inserted_at_and_created_abs'] = df[f"{feature_name}_abs"]
                            df['has_inserted_at'] = df[f"has_{clean_name}"]
                        elif col == 'retrieval_timestamp':
                            df['days_between_retrieval_timestamp_and_created'] = df[feature_name]
                            df['days_between_retrieval_timestamp_and_created_abs'] = df[f"{feature_name}_abs"]
                            df['has_retrieval_timestamp'] = df[f"has_{clean_name}"]
            
            logger.info("Timestamp features engineered successfully")
            return df
        except Exception as e:
            logger.error(f"Timestamp feature engineering failed: {e}")
            raise
    
    def engineer_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced text feature engineering"""
        try:
            logger.info("Engineering text features...")
            
            # Clean and combine text columns
            combined_text = []
            for col in self.text_columns:
                if col in df.columns:
                    # Basic text cleaning
                    clean_text = (df[col].astype(str)
                                 .str.lower()
                                 .str.replace(r'[^\w\s]', '', regex=True)
                                 .str.replace(r'\s+', ' ', regex=True)
                                 .str.strip())
                    combined_text.append(clean_text.fillna(''))
            
            # Create combined text feature
            if combined_text:
                df['combined_text'] = ''
                for text_series in combined_text:
                    df['combined_text'] += text_series.fillna('') + ' '
            else:
                df['combined_text'] = ''
            
            # Text-based features
            if 'combined_text' in df.columns:
                df['text_length'] = df['combined_text'].str.len()
                df['text_word_count'] = df['combined_text'].str.split().str.len()
                df['text_char_count'] = df['combined_text'].str.len()
                df['text_avg_word_length'] = df['text_length'] / df['text_word_count'].replace(0, 1)
                df['text_uppercase_ratio'] = df['combined_text'].str.count(r'[A-Z]') / df['text_length'].replace(0, 1)
                df['text_digit_ratio'] = df['combined_text'].str.count(r'\d') / df['text_length'].replace(0, 1)
                df['text_punctuation_ratio'] = df['combined_text'].str.count(r'[^\w\s]') / df['text_length'].replace(0, 1)
                
                # Original pipeline compatibility features
                df['combined_text_length'] = df['text_length']  # Alias for compatibility
                df['combined_text_word_count'] = df['text_word_count']  # Alias for compatibility
                df['has_numbers_in_text'] = df['combined_text'].str.contains(r'\d', regex=True).astype(int)
                df['has_email_in_text'] = df['combined_text'].str.contains(r'@', regex=True).astype(int)
                df['has_url_in_text'] = df['combined_text'].str.contains(r'http', regex=True).astype(int)
                df['text_quality_score'] = (
                    (df['text_length'] > 10).astype(int) +
                    (df['text_word_count'] > 3).astype(int) +
                    (~df['combined_text'].str.contains('nan', regex=True)).astype(int)
                )
            
            logger.info("Text features engineered successfully")
            return df
        except Exception as e:
            logger.error(f"Text feature engineering failed: {e}")
            raise
    
    def engineer_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced categorical feature engineering"""
        try:
            logger.info("Engineering categorical features...")
            
            # Core interaction features for high-value B2B combinations
            if 'organization' in df.columns and 'title' in df.columns:
                df['org_title_interaction'] = (
                    df['organization'].astype(str).fillna('Unknown') + '_' + 
                    df['title'].fillna('Unknown')
                )
            
            if 'status' in df.columns and 'upload_method' in df.columns:
                df['status_method_interaction'] = (
                    df['status'].astype(str).fillna('Unknown') + '_' + 
                    df['upload_method'].fillna('Unknown')
                )
            
            # Apollo-enhanced B2B interactions
            if 'seniority' in df.columns and 'organization_industry' in df.columns:
                df['apollo_seniority_industry'] = (
                    df['seniority'].fillna('Unknown') + '_' + 
                    df['organization_industry'].fillna('Unknown')
                )
            
            if 'departments' in df.columns and 'functions' in df.columns:
                # Handle array columns properly
                dept_str = df['departments'].apply(lambda x: str(x) if x is not None else 'Unknown')
                func_str = df['functions'].apply(lambda x: str(x) if x is not None else 'Unknown')
                df['apollo_dept_function'] = dept_str + '_' + func_str
            
            # Geographic-industry interactions
            if 'country' in df.columns and 'organization_industry' in df.columns:
                df['apollo_geo_industry'] = (
                    df['country'].fillna('Unknown') + '_' + 
                    df['organization_industry'].fillna('Unknown')
                )
                # Original pipeline compatibility
                df['geo_industry_interaction'] = df['apollo_geo_industry']
            
            # Title-industry interaction (original pipeline compatibility)
            if 'title' in df.columns and 'organization_industry' in df.columns:
                df['title_industry_interaction'] = (
                    df['title'].fillna('Unknown') + '_' + 
                    df['organization_industry'].fillna('Unknown')
                )
            
            # Industry-seniority interaction (original pipeline compatibility)
            if 'seniority' in df.columns and 'organization_industry' in df.columns:
                df['industry_seniority_interaction'] = df['apollo_seniority_industry']
            
            # Categorical encoding for high-cardinality columns
            for col in self.categorical_columns:
                if col in df.columns:
                    # Cap high cardinality by grouping rare categories
                    value_counts = df[col].value_counts()
                    top_categories = value_counts.head(20).index.tolist()
                    df[f'{col}_grouped'] = df[col].apply(
                        lambda x: x if x in top_categories else 'Other'
                    )
            
            logger.info("Categorical features engineered successfully")
            return df
        except Exception as e:
            logger.error(f"Categorical feature engineering failed: {e}")
            raise
    
    def engineer_apollo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from Apollo enrichment data"""
        try:
            logger.info("Engineering Apollo enrichment features...")
            
            # Apollo data availability indicators
            df['has_apollo_enrichment'] = df['apollo_id'].notna().astype(int)
            df['apollo_api_success'] = (df['api_status'] == 'success').astype(int)
            
            # Company size indicators
            if 'organization_employees' in df.columns:
                df['organization_employees'] = pd.to_numeric(df['organization_employees'], errors='coerce')
                df['company_size_category'] = pd.cut(
                    df['organization_employees'],
                    bins=[0, 10, 50, 200, 1000, float('inf')],
                    labels=['Startup', 'Small', 'Medium', 'Large', 'Enterprise'],
                    include_lowest=True
                ).astype(str)
                
                # Log transformation for numerical features
                df['company_size_log'] = np.log1p(df['organization_employees'].fillna(0))
            
            # Company age features
            if 'organization_founded_year' in df.columns:
                current_year = pd.Timestamp.now().year
                df['organization_founded_year'] = pd.to_numeric(df['organization_founded_year'], errors='coerce')
                df['company_age_years'] = current_year - df['organization_founded_year']
                df['company_age_category'] = pd.cut(
                    df['company_age_years'],
                    bins=[0, 5, 15, 30, float('inf')],
                    labels=['New', 'Growing', 'Established', 'Mature'],
                    include_lowest=True
                ).astype(str)
            
            # Apollo credits and enrichment quality
            if 'credits_consumed' in df.columns:
                df['credits_consumed'] = pd.to_numeric(df['credits_consumed'], errors='coerce').fillna(0)
                df['apollo_enrichment_cost'] = df['credits_consumed']
            
            # Enrichment recency
            if 'enriched_at' in df.columns:
                df['enriched_at'] = pd.to_datetime(df['enriched_at'], errors='coerce', utc=True)
                current_time = pd.Timestamp.now(tz='UTC')
                df['days_since_apollo_enrichment'] = (current_time - df['enriched_at']).dt.days
                df['apollo_data_freshness'] = pd.cut(
                    df['days_since_apollo_enrichment'],
                    bins=[0, 30, 90, 180, float('inf')],
                    labels=['Fresh', 'Recent', 'Stale', 'Old'],
                    include_lowest=True
                ).astype(str)
            
            # B2B quality indicators
            apollo_quality_indicators = [
                'title', 'organization_name', 'organization_industry',
                'seniority', 'departments', 'functions'
            ]
            apollo_completeness = []
            for col in apollo_quality_indicators:
                if col in df.columns:
                    apollo_completeness.append(df[col].notna().astype(int))
            
            if apollo_completeness:
                df['apollo_data_completeness'] = pd.concat(apollo_completeness, axis=1).sum(axis=1)
                df['apollo_data_completeness_pct'] = df['apollo_data_completeness'] / len(apollo_completeness)
            else:
                df['apollo_data_completeness'] = 0
                df['apollo_data_completeness_pct'] = 0
            
            # High-value prospect indicators
            high_value_titles = [
                'ceo', 'cto', 'cfo', 'president', 'director', 'vp', 'vice president',
                'head of', 'chief', 'founder', 'owner', 'manager'
            ]
            if 'title' in df.columns:
                df['is_high_value_title'] = df['title'].fillna('').str.lower().str.contains(
                    '|'.join(high_value_titles), regex=True
                ).astype(int)
            
            high_value_seniority = ['c_level', 'vp', 'director', 'head']
            if 'seniority' in df.columns:
                df['is_high_value_seniority'] = df['seniority'].fillna('').str.lower().str.contains(
                    '|'.join(high_value_seniority), regex=True
                ).astype(int)
            
            # Technology/innovation indicators
            tech_departments = ['engineering', 'technology', 'it', 'data', 'digital', 'innovation']
            if 'departments' in df.columns:
                # Handle array column properly
                df['is_tech_department'] = df['departments'].apply(
                    lambda x: 1 if x and any(dept.lower() in str(x).lower() for dept in tech_departments) else 0
                )
            
            logger.info("Apollo enrichment features engineered successfully")
            return df
        except Exception as e:
            logger.error(f"Apollo feature engineering failed: {e}")
            raise
    
    def engineer_jsonb_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from JSONB columns (original pipeline compatibility)"""
        try:
            logger.info("Engineering JSONB enrichment features...")
            
            # JSONB presence indicators
            for col in self.jsonb_columns:
                if col in df.columns:
                    df[f'has_{col}'] = df[col].notna().astype(int)
            
            # Enrichment completeness score
            enrichment_cols = [f'has_{col}' for col in self.jsonb_columns if f'has_{col}' in df.columns]
            if enrichment_cols:
                df['enrichment_completeness'] = df[enrichment_cols].sum(axis=1)
                df['enrichment_completeness_pct'] = df['enrichment_completeness'] / len(enrichment_cols)
            
            logger.info("JSONB enrichment features engineered successfully")
            return df
        except Exception as e:
            logger.error(f"JSONB feature engineering failed: {e}")
            raise
    
    def engineer_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engagement-related features (non-leakage)"""
        try:
            logger.info("Engineering engagement features...")
            
            # Campaign-level aggregations (historical, non-leakage)
            if 'campaign' in df.columns:
                campaign_stats = df.groupby('campaign').agg({
                    'id': 'count',  # Campaign size
                    'timestamp_created': ['min', 'max']  # Campaign duration
                }).reset_index()
                
                campaign_stats.columns = ['campaign', 'campaign_size', 'campaign_start', 'campaign_end']
                campaign_stats['campaign_duration_days'] = (
                    campaign_stats['campaign_end'] - campaign_stats['campaign_start']
                ).dt.days
                
                # Merge back to main dataframe
                df = df.merge(campaign_stats[['campaign', 'campaign_size', 'campaign_duration_days']], 
                             on='campaign', how='left')
            
            # Time-based engagement patterns (using creation time only)
            if 'timestamp_created' in df.columns:
                df['created_hour_category'] = pd.cut(
                    df['timestamp_created'].dt.hour,
                    bins=[0, 6, 12, 18, 24],
                    labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                    include_lowest=True
                )
            
            logger.info("Engagement features engineered successfully")
            return df
        except Exception as e:
            logger.error(f"Engagement feature engineering failed: {e}")
            raise
    
    def apply_data_quality_checks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive data quality validation and cleaning"""
        try:
            logger.info("Applying data quality checks...")
            initial_rows = len(df)
            
            # Remove duplicates based on email + campaign combination
            if 'email' in df.columns and 'campaign' in df.columns:
                df_deduped = df.drop_duplicates(subset=['email', 'campaign'], keep='last')
                duplicates_removed = initial_rows - len(df_deduped)
                if duplicates_removed > 0:
                    logger.warning(f"Removed {duplicates_removed} duplicate records")
                df = df_deduped
            
            # Handle extreme outliers in numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col.startswith('days_') or col.endswith('_count'):
                    q99 = df[col].quantile(0.99)
                    q01 = df[col].quantile(0.01)
                    if q99 > q01:
                        df[col] = df[col].clip(lower=q01, upper=q99)
            
            # Data quality scoring
            df['data_quality_score'] = self._calculate_quality_score(df)
            
            logger.info(f"Data quality checks completed. Final rows: {len(df)}")
            return df
        except Exception as e:
            logger.error(f"Data quality checks failed: {e}")
            raise
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate data quality score for each record"""
        quality_components = []
        
        # Completeness score (non-null essential fields)
        essential_fields = ['email', 'campaign', 'timestamp_created']
        completeness = df[essential_fields].notna().sum(axis=1) / len(essential_fields)
        quality_components.append(completeness)
        
        # Text quality score
        if 'text_quality_score' in df.columns:
            text_quality = df['text_quality_score'] / 3.0  # Normalize to 0-1
            quality_components.append(text_quality)
        
        # Engagement data availability
        engagement_fields = ['email_open_count', 'email_click_count', 'email_reply_count']
        engagement_completeness = df[engagement_fields].notna().sum(axis=1) / len(engagement_fields)
        quality_components.append(engagement_completeness)
        
        # Calculate average quality score
        quality_df = pd.DataFrame(quality_components).T
        return quality_df.mean(axis=1)
    
    def prepare_silver_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare final feature set for silver layer"""
        try:
            logger.info("Preparing silver layer features...")
            
            # Create a copy for silver layer
            silver_df = df.copy()
            
            # Add metadata columns
            silver_df['processed_timestamp'] = pd.Timestamp.now(tz='UTC')
            silver_df['pipeline_version'] = '1.0'
            silver_df['feature_engineering_date'] = pd.Timestamp.now(tz='UTC').date()
            
            # Define feature columns to keep
            feature_columns = [
                # Identifiers (for tracking, not for ML)
                'id', 'email', 'campaign',
                
                # Target variable
                self.target_variable,
                
                # Engineered timestamp features
                'created_day_of_week', 'created_month', 'created_hour', 'created_quarter',
                'created_is_weekend', 'created_is_business_hours',
                'days_since_creation', 'weeks_since_creation',
                
                # Text features
                'text_length', 'text_word_count', 'has_numbers_in_text', 'text_quality_score',
                
                # Categorical interaction features
                'org_title_interaction', 'status_method_interaction',
                'apollo_seniority_industry', 'apollo_dept_function', 'apollo_geo_industry',
                
                # Apollo enrichment features
                'has_apollo_enrichment', 'apollo_api_success', 'company_size_category',
                'company_size_log', 'company_age_years', 'company_age_category',
                'apollo_enrichment_cost', 'days_since_apollo_enrichment', 'apollo_data_freshness',
                'apollo_data_completeness', 'apollo_data_completeness_pct',
                'is_high_value_title', 'is_high_value_seniority', 'is_tech_department',
                
                # Engagement features
                'campaign_size', 'campaign_duration_days', 'created_hour_category',
                
                # Quality metrics
                'data_quality_score',
                
                # Metadata
                'processed_timestamp', 'pipeline_version', 'feature_engineering_date'
            ]
            
            # Add dynamic columns that exist
            dynamic_cols = [col for col in silver_df.columns 
                          if (col.startswith('days_between_') or 
                              col.startswith('has_') or 
                              col.endswith('_grouped')) and col not in feature_columns]
            feature_columns.extend(dynamic_cols)
            
            # Select only existing columns
            existing_features = [col for col in feature_columns if col in silver_df.columns]
            silver_df = silver_df[existing_features]
            
            logger.info(f"Silver layer prepared with {len(existing_features)} features")
            return silver_df
        except Exception as e:
            logger.error(f"Silver layer preparation failed: {e}")
            raise
    
    def write_to_silver_layer(self, df: pd.DataFrame, table_name: str = 'silver_ml_features'):
        """Write processed data to silver layer"""
        try:
            logger.info(f"Writing {len(df)} records to silver layer...")
            
            # Write to PostgreSQL silver layer in existing leads schema
            df.to_sql(
                table_name,
                self.engine,
                schema='leads',  # Use existing leads schema
                if_exists='append',
                index=False,
                method='multi',
                chunksize=self.batch_size
            )
            
            logger.info(f"Successfully wrote {len(df)} records to {table_name}")
        except Exception as e:
            logger.error(f"Failed to write to silver layer: {e}")
            raise
    
    def run_pipeline(self, incremental: bool = True, lookback_days: int = 7):
        """Execute the complete bronze to silver ETL pipeline"""
        try:
            logger.info("Starting Bronze to Silver ETL Pipeline...")
            start_time = datetime.now()
            
            # Step 1: Extract bronze data
            df = self.extract_bronze_data(incremental=incremental, lookback_days=lookback_days)
            
            if df.empty:
                logger.info("No data to process. Pipeline completed.")
                return
            
            # Step 2: Create target variable
            df = self.create_target_variable(df)
            
            # Step 3: Feature engineering
            df = self.engineer_timestamp_features(df)
            df = self.engineer_text_features(df)
            df = self.engineer_categorical_features(df)
            df = self.engineer_apollo_features(df)  # Apollo enrichment features
            df = self.engineer_jsonb_features(df)  # JSONB enrichment features (compatibility)
            df = self.engineer_engagement_features(df)
            
            # Step 4: Data quality checks
            df = self.apply_data_quality_checks(df)
            
            # Step 5: Prepare silver layer features
            silver_df = self.prepare_silver_features(df)
            
            # Step 6: Write to silver layer
            self.write_to_silver_layer(silver_df)
            
            # Pipeline completion
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Pipeline completed successfully in {duration:.2f} seconds")
            
            return silver_df
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

def main():
    """Main execution function"""
    try:
        # Initialize pipeline
        pipeline = BronzeToSilverPipeline()
        
        # Run incremental pipeline (default)
        result = pipeline.run_pipeline(incremental=True, lookback_days=7)
        
        if result is not None:
            print(f"Pipeline completed successfully. Processed {len(result)} records.")
            print(f"Silver layer features: {list(result.columns)}")
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
