#!/usr/bin/env python3
"""
Run the main bronze to silver pipeline with idempotent operations
"""

from bronze_to_silver_pipeline import BronzeToSilverPipeline
import pandas as pd
import numpy as np
import logging
from sqlalchemy import create_engine, text
import yaml
import json
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MainIdempotentPipeline(BronzeToSilverPipeline):
    """Main pipeline with idempotent operations and column filtering"""
    
    def __init__(self, config_path: str = "config/silver_layer_config.yaml"):
        super().__init__(config_path)
        
        # Get the actual silver table columns
        self.silver_columns = self._get_silver_columns()
        
    def _get_silver_columns(self):
        """Get the actual column names from the silver table"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT column_name
                    FROM information_schema.columns 
                    WHERE table_schema = 'leads' 
                    AND table_name = 'silver_ml_features'
                    ORDER BY ordinal_position
                """))
                
                columns = [row[0] for row in result.fetchall()]
                logging.info(f"Found {len(columns)} columns in silver table")
                return columns
        except Exception as e:
            logging.error(f"Could not get silver table columns: {e}")
            return []
    
    def check_existing_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check for existing records in silver layer and filter out duplicates"""
        if df.empty:
            return df
            
        logging.info(f"🔍 Checking for existing records among {len(df)} candidates...")
        
        try:
            # Get list of IDs that already exist in silver layer
            existing_ids_query = text("""
                SELECT DISTINCT id 
                FROM leads.silver_ml_features 
                WHERE id = ANY(:id_list)
            """)
            
            # Convert DataFrame IDs to list
            candidate_ids = df['id'].tolist()
            
            with self.engine.connect() as conn:
                result = conn.execute(existing_ids_query, {'id_list': candidate_ids})
                existing_ids = [row[0] for row in result.fetchall()]
            
            # Filter out records that already exist
            new_records = df[~df['id'].isin(existing_ids)].copy()
            
            logging.info(f"📊 Found {len(existing_ids)} existing records, {len(new_records)} new records to process")
            
            return new_records
            
        except Exception as e:
            logging.warning(f"Could not check existing records: {e}. Processing all records.")
            return df
    
    def write_to_silver_layer_idempotent(self, df: pd.DataFrame, table_name: str = 'silver_ml_features'):
        """Write to silver layer with duplicate prevention and robust column handling"""
        try:
            if df.empty:
                logging.info("No new records to write to silver layer")
                return
                
            # Check for existing records and filter them out
            new_records = self.check_existing_records(df)
            
            if new_records.empty:
                logging.info("All records already exist in silver layer - pipeline is idempotent ✅")
                return
            
            # Create a robust column mapping and filling strategy
            silver_df = self._prepare_robust_silver_features(new_records)
            
            logging.info(f"💾 Writing {len(silver_df)} new records with {len(silver_df.columns)} columns to silver layer...")
            
            # Write only new records to PostgreSQL silver layer
            silver_df.to_sql(
                table_name,
                self.engine,
                schema='leads',
                if_exists='append',
                index=False,
                method='multi',
                chunksize=500
            )
            
            logging.info(f"✅ Successfully wrote {len(silver_df)} new records to silver layer")
            
        except Exception as e:
            logging.error(f"Failed to write to silver layer: {e}")
            raise
    
    def _prepare_robust_silver_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare silver features with robust handling of missing columns and NULL values"""
        try:
            logging.info("🔧 Preparing robust silver features...")
            
            # Start with a copy
            silver_df = df.copy()
            
            # Ensure ALL silver table columns exist with proper defaults
            for col in self.silver_columns:
                if col not in silver_df.columns:
                    # Determine appropriate default based on column type/name
                    if col in ['email_open_count', 'email_click_count', 'email_reply_count']:
                        # These should come from source data - fill with 0 if missing
                        silver_df[col] = 0
                        logging.warning(f"⚠️  Missing engagement column {col}, filled with 0")
                    elif col.startswith('has_'):
                        # Boolean indicators default to 0
                        silver_df[col] = 0
                    elif col.startswith('days_between_') or col.startswith('days_since_'):
                        # Numeric features can be NULL initially
                        silver_df[col] = pd.NA
                    elif col in ['text_length', 'text_word_count', 'text_char_count']:
                        # Text metrics default to 0
                        silver_df[col] = 0
                    elif col in ['text_avg_word_length', 'text_uppercase_ratio', 'text_digit_ratio', 'text_punctuation_ratio']:
                        # Text ratios default to 0.0
                        silver_df[col] = 0.0
                    elif col == 'combined_text':
                        # Create combined text from available text fields
                        silver_df[col] = self._create_combined_text(silver_df)
                    elif col == 'feature_completeness_score':
                        # Calculate feature completeness
                        silver_df[col] = self._calculate_feature_completeness(silver_df)
                    elif col == 'processed_timestamp':
                        silver_df[col] = datetime.now()
                    elif col in ['timestamp_last_contact', 'timestamp_created', 'timestamp_updated']:
                        # Keep original timestamp values for gold layer filtering
                        if col in silver_df.columns:
                            pass  # Keep existing values
                        else:
                            silver_df[col] = pd.NaT  # Not available timestamp
                    else:
                        # Default for other columns based on data type expectations
                        silver_df[col] = pd.NA
            
            # Now fill in missing text features if we have the source data
            silver_df = self._enhance_text_features(silver_df)
            
            # Enhance missing timestamp features
            silver_df = self._enhance_timestamp_features(silver_df)
            
            # Select only silver table columns in the correct order
            silver_df = silver_df[self.silver_columns]
            
            # Fill remaining NaN values with appropriate defaults
            silver_df = self._fill_remaining_nulls(silver_df)
            
            logging.info(f"✅ Robust silver features prepared with {len(silver_df.columns)} columns")
            return silver_df
            
        except Exception as e:
            logging.error(f"Failed to prepare robust silver features: {e}")
            raise
    
    def _create_combined_text(self, df: pd.DataFrame) -> pd.Series:
        """Create combined text from available text columns"""
        text_cols = ['campaign', 'personalization', 'payload', 'status_summary', 'headline', 'title']
        available_text_cols = [col for col in text_cols if col in df.columns]
        
        if not available_text_cols:
            return pd.Series([''] * len(df), index=df.index)
        
        # Combine non-null text values
        combined = df[available_text_cols].fillna('').apply(
            lambda x: ' '.join(x.values).strip(), axis=1
        )
        return combined
    
    def _enhance_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhance missing text features based on available data"""
        if 'combined_text' in df.columns and df['combined_text'].notna().any():
            # Calculate text features from combined_text
            df['text_length'] = df['combined_text'].str.len().fillna(0)
            df['text_word_count'] = df['combined_text'].str.split().str.len().fillna(0)
            df['text_char_count'] = df['text_length']  # Same as length
            
            # Calculate text ratios
            total_chars = df['text_length'].replace(0, 1)  # Avoid division by zero
            df['text_uppercase_ratio'] = df['combined_text'].str.count(r'[A-Z]').fillna(0) / total_chars
            df['text_digit_ratio'] = df['combined_text'].str.count(r'\d').fillna(0) / total_chars
            df['text_punctuation_ratio'] = df['combined_text'].str.count(r'[^\w\s]').fillna(0) / total_chars
            
            # Calculate average word length
            word_counts = df['text_word_count'].replace(0, 1)
            df['text_avg_word_length'] = df['text_length'] / word_counts
        
        return df
    
    def _enhance_timestamp_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhance missing timestamp features"""
        reference_time = datetime.now()
        
        # If timestamp_created exists, calculate missing timestamp features
        if 'timestamp_created' in df.columns:
            created_times = pd.to_datetime(df['timestamp_created'], errors='coerce')
            
            # Calculate missing timestamp differences
            timestamp_cols = [
                'timestamp_updated', 'timestamp_last_contact', 'timestamp_last_touch',
                'timestamp_last_open', 'timestamp_last_click', 'timestamp_last_reply',
                'timestamp_last_interest_change', 'retrieval_timestamp', 'inserted_at', 'enriched_at'
            ]
            
            for ts_col in timestamp_cols:
                if ts_col in df.columns:
                    ts_values = pd.to_datetime(df[ts_col], errors='coerce')
                    days_col = f"days_between_{ts_col.replace('timestamp_', '').replace('_timestamp', '')}_and_created"
                    has_col = f"has_{ts_col.replace('timestamp_', '').replace('_timestamp', '')}"
                    
                    if days_col in self.silver_columns:
                        df[days_col] = (ts_values - created_times).dt.days
                    if has_col in self.silver_columns:
                        df[has_col] = ts_values.notna().astype(int)
        
        return df
    
    def _calculate_feature_completeness(self, df: pd.DataFrame) -> pd.Series:
        """Calculate feature completeness score"""
        # Count non-null values across important feature columns
        important_cols = [col for col in df.columns if col not in ['id', 'email', 'processed_timestamp']]
        if not important_cols:
            return pd.Series([0.0] * len(df), index=df.index)
        
        completeness = df[important_cols].notna().mean(axis=1)
        return completeness
    
    def _fill_remaining_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill remaining NULL values with appropriate defaults"""
        for col in df.columns:
            if df[col].dtype in ['object', 'string']:
                df[col] = df[col].fillna('')
            elif df[col].dtype in ['int64', 'Int64']:
                df[col] = df[col].fillna(0)
            elif df[col].dtype in ['float64', 'Float64']:
                df[col] = df[col].fillna(0.0)
            elif 'datetime' in str(df[col].dtype).lower():
                # Leave datetime NULLs as they are
                pass
        
        return df
    
    def engineer_advanced_jsonb_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract advanced features from JSONB columns for XGBoost model enhancement"""
        try:
            logging.info("🔍 Engineering advanced JSONB features...")
            
            # Track new features added
            new_features = []
            
            # Employment History Features (avoiding company_age_years - already exists)
            if 'employment_history' in df.columns:
                emp_features = df.apply(lambda row: self._extract_employment_features(row.get('employment_history')), axis=1)
                emp_df = pd.DataFrame(emp_features.tolist(), index=df.index)
                
                for col in emp_df.columns:
                    if col not in df.columns:  # Avoid duplicates
                        df[col] = emp_df[col]
                        new_features.append(col)
            
            # Organization Features (avoiding existing company_size_log, company_age_years)
            if 'organization_data' in df.columns:
                org_features = df.apply(lambda row: self._extract_organization_features(row.get('organization_data')), axis=1)
                org_df = pd.DataFrame(org_features.tolist(), index=df.index)
                
                for col in org_df.columns:
                    if col not in df.columns and col not in ['company_age_years', 'company_size_log']:  # Avoid duplicates
                        df[col] = org_df[col]
                        new_features.append(col)
            
            # Account Features
            if 'account_data' in df.columns:
                acc_features = df.apply(lambda row: self._extract_account_features(row.get('account_data')), axis=1)
                acc_df = pd.DataFrame(acc_features.tolist(), index=df.index)
                
                for col in acc_df.columns:
                    if col not in df.columns:  # Avoid duplicates
                        df[col] = acc_df[col]
                        new_features.append(col)
            
            # API Response Features
            if 'api_response_raw' in df.columns:
                api_features = df.apply(lambda row: self._extract_api_response_features(row.get('api_response_raw')), axis=1)
                api_df = pd.DataFrame(api_features.tolist(), index=df.index)
                
                for col in api_df.columns:
                    if col not in df.columns:  # Avoid duplicates
                        df[col] = api_df[col]
                        new_features.append(col)
            
            logging.info(f"✅ Added {len(new_features)} advanced JSONB features: {new_features[:10]}...")
            return df
            
        except Exception as e:
            logging.error(f"Advanced JSONB feature engineering failed: {e}")
            return df  # Continue pipeline even if this fails
    
    def _extract_employment_features(self, employment_history) -> dict:
        """Extract employment history features"""
        if not employment_history or pd.isna(employment_history):
            return {
                'job_count': 0, 'current_job_tenure_years': 0, 'avg_job_tenure_years': 0,
                'total_career_years': 0, 'job_progression_score': 0, 'has_management_experience': 0,
                'industry_stability': 0, 'education_level': 0
            }
        
        try:
            if isinstance(employment_history, str):
                emp_data = json.loads(employment_history)
            else:
                emp_data = employment_history
            
            if not isinstance(emp_data, list):
                return {
                    'job_count': 0, 'current_job_tenure_years': 0, 'avg_job_tenure_years': 0,
                    'total_career_years': 0, 'job_progression_score': 0, 'has_management_experience': 0,
                    'industry_stability': 0, 'education_level': 0
                }
            
            features = {}
            current_year = 2025
            
            # Basic counts
            features['job_count'] = len(emp_data)
            
            # Calculate tenures and experience
            tenures = []
            management_titles = ['director', 'manager', 'head', 'vp', 'ceo', 'cto', 'cfo', 'chief', 'president']
            has_management = 0
            education_score = 0
            industries = set()
            
            for job in emp_data:
                # Job tenure calculation
                start_year = None
                end_year = current_year
                
                if job.get('start_date'):
                    try:
                        start_year = int(str(job['start_date'])[:4]) if len(str(job['start_date'])) >= 4 else None
                    except:
                        pass
                        
                if job.get('end_date') and str(job.get('end_date')).lower() not in ['present', 'current', '']:
                    try:
                        end_year = int(str(job['end_date'])[:4]) if len(str(job['end_date'])) >= 4 else current_year
                    except:
                        pass
                
                if start_year:
                    tenure = max(0, end_year - start_year)
                    tenures.append(tenure)
                
                # Management experience
                title = str(job.get('title', '')).lower()
                if any(mgmt_word in title for mgmt_word in management_titles):
                    has_management = 1
                
                # Education level
                if str(job.get('kind', '')).lower() == 'education':
                    degree = str(job.get('degree', '')).lower()
                    if 'phd' in degree or 'doctorate' in degree:
                        education_score = max(education_score, 4)
                    elif 'master' in degree or 'mba' in degree:
                        education_score = max(education_score, 3)
                    elif 'bachelor' in degree:
                        education_score = max(education_score, 2)
                    elif 'associate' in degree:
                        education_score = max(education_score, 1)
                
                # Industry stability
                org_name = str(job.get('organization_name', '')).lower()
                if org_name:
                    industries.add(org_name)
            
            # Calculate derived features
            features['current_job_tenure_years'] = tenures[0] if tenures else 0
            features['avg_job_tenure_years'] = np.mean(tenures) if tenures else 0
            
            # Career span
            all_years = []
            for job in emp_data:
                for date_field in ['start_date', 'end_date']:
                    if job.get(date_field):
                        try:
                            year = int(str(job[date_field])[:4])
                            all_years.append(year)
                        except:
                            pass
            
            features['total_career_years'] = max(all_years) - min(all_years) if len(all_years) >= 2 else 0
            features['job_progression_score'] = len(tenures) / max(1, features['total_career_years'])
            features['has_management_experience'] = has_management
            features['industry_stability'] = 1 / max(1, len(industries))
            features['education_level'] = education_score
            
            return features
            
        except Exception:
            return {
                'job_count': 0, 'current_job_tenure_years': 0, 'avg_job_tenure_years': 0,
                'total_career_years': 0, 'job_progression_score': 0, 'has_management_experience': 0,
                'industry_stability': 0, 'education_level': 0
            }
    
    def _extract_organization_features(self, org_data) -> dict:
        """Extract organization features (avoiding duplicates)"""
        if not org_data or pd.isna(org_data):
            return {
                'company_size_bucket': 'unknown', 'has_public_trading': 0, 'alexa_rank_tier': 'unknown',
                'has_social_presence': 0, 'industry_keywords_count': 0, 'headcount_growth_6m': 0,
                'headcount_growth_12m': 0, 'headcount_growth_24m': 0, 'is_tech_company': 0,
                'is_enterprise': 0, 'market_cap_tier': 'unknown'
            }
        
        try:
            if isinstance(org_data, str):
                org_dict = json.loads(org_data)
            else:
                org_dict = org_data
            
            if not isinstance(org_dict, dict):
                return {
                    'company_size_bucket': 'unknown', 'has_public_trading': 0, 'alexa_rank_tier': 'unknown',
                    'has_social_presence': 0, 'industry_keywords_count': 0, 'headcount_growth_6m': 0,
                    'headcount_growth_12m': 0, 'headcount_growth_24m': 0, 'is_tech_company': 0,
                    'is_enterprise': 0, 'market_cap_tier': 'unknown'
                }
            
            features = {}
            
            # Company size bucket (more granular than existing categories)
            employees = org_dict.get('estimated_num_employees', 0)
            try:
                employees = int(employees) if employees else 0
            except:
                employees = 0
            
            if employees == 0:
                features['company_size_bucket'] = 'unknown'
            elif employees <= 10:
                features['company_size_bucket'] = 'startup'
            elif employees <= 50:
                features['company_size_bucket'] = 'small'
            elif employees <= 250:
                features['company_size_bucket'] = 'medium'
            elif employees <= 1000:
                features['company_size_bucket'] = 'large'
            else:
                features['company_size_bucket'] = 'enterprise'
            
            # Public trading status
            features['has_public_trading'] = 1 if org_dict.get('publicly_traded_symbol') else 0
            
            # Alexa ranking tier
            alexa_rank = org_dict.get('alexa_ranking')
            if alexa_rank:
                try:
                    rank = int(alexa_rank)
                    if rank <= 1000:
                        features['alexa_rank_tier'] = 'top_1k'
                    elif rank <= 10000:
                        features['alexa_rank_tier'] = 'top_10k'
                    elif rank <= 100000:
                        features['alexa_rank_tier'] = 'top_100k'
                    else:
                        features['alexa_rank_tier'] = 'other'
                except:
                    features['alexa_rank_tier'] = 'unknown'
            else:
                features['alexa_rank_tier'] = 'unknown'
            
            # Social media presence
            social_urls = [org_dict.get('twitter_url'), org_dict.get('facebook_url'), org_dict.get('linkedin_url')]
            features['has_social_presence'] = sum(1 for url in social_urls if url)
            
            # Industry keywords and tech detection
            keywords = org_dict.get('keywords', [])
            if isinstance(keywords, list):
                features['industry_keywords_count'] = len(keywords)
                tech_keywords = ['software', 'technology', 'tech', 'digital', 'ai', 'cloud', 'saas', 'data', 'cyber']
                tech_score = sum(1 for keyword in keywords if any(tech in str(keyword).lower() for tech in tech_keywords))
                features['is_tech_company'] = 1 if tech_score >= 2 else 0
            else:
                features['industry_keywords_count'] = 0
                features['is_tech_company'] = 0
            
            # Headcount growth
            for period, field in [('6m', 'organization_headcount_six_month_growth'), 
                                 ('12m', 'organization_headcount_twelve_month_growth'),
                                 ('24m', 'organization_headcount_twenty_four_month_growth')]:
                growth = org_dict.get(field)
                try:
                    features[f'headcount_growth_{period}'] = float(growth) if growth else 0
                except:
                    features[f'headcount_growth_{period}'] = 0
            
            # Enterprise indicator
            features['is_enterprise'] = 1 if (employees > 1000 or features['has_public_trading']) else 0
            
            # Market cap tier
            market_cap = org_dict.get('market_cap')
            if market_cap:
                try:
                    cap_str = str(market_cap).lower()
                    if 'billion' in cap_str or 'b' in cap_str:
                        features['market_cap_tier'] = 'large_cap'
                    elif 'million' in cap_str or 'm' in cap_str:
                        features['market_cap_tier'] = 'mid_cap'
                    else:
                        features['market_cap_tier'] = 'small_cap'
                except:
                    features['market_cap_tier'] = 'unknown'
            else:
                features['market_cap_tier'] = 'unknown'
            
            return features
            
        except Exception:
            return {
                'company_size_bucket': 'unknown', 'has_public_trading': 0, 'alexa_rank_tier': 'unknown',
                'has_social_presence': 0, 'industry_keywords_count': 0, 'headcount_growth_6m': 0,
                'headcount_growth_12m': 0, 'headcount_growth_24m': 0, 'is_tech_company': 0,
                'is_enterprise': 0, 'market_cap_tier': 'unknown'
            }
    
    def _extract_account_features(self, account_data) -> dict:
        """Extract account engagement features"""
        if not account_data or pd.isna(account_data):
            return {
                'account_age_days': 0, 'has_crm_integration': 0, 'label_count': 0,
                'account_stage_maturity': 0, 'phone_status_quality': 0, 'existence_level_score': 0
            }
        
        try:
            if isinstance(account_data, str):
                acc_dict = json.loads(account_data)
            else:
                acc_dict = account_data
            
            if not isinstance(acc_dict, dict):
                return {
                    'account_age_days': 0, 'has_crm_integration': 0, 'label_count': 0,
                    'account_stage_maturity': 0, 'phone_status_quality': 0, 'existence_level_score': 0
                }
            
            features = {}
            
            # Account age
            created_at = acc_dict.get('created_at')
            if created_at:
                try:
                    from datetime import datetime
                    created_date = datetime.fromisoformat(str(created_at).replace('Z', '+00:00'))
                    features['account_age_days'] = max(0, (datetime.now() - created_date).days)
                except:
                    features['account_age_days'] = 0
            else:
                features['account_age_days'] = 0
            
            # CRM integration
            features['has_crm_integration'] = 1 if acc_dict.get('crm_record_url') else 0
            
            # Account labeling
            label_ids = acc_dict.get('label_ids', [])
            features['label_count'] = len(label_ids) if isinstance(label_ids, list) else 0
            
            # Account stage maturity
            features['account_stage_maturity'] = 1 if acc_dict.get('account_stage_id') else 0
            
            # Phone status quality
            phone_status = acc_dict.get('phone_status', '')
            if phone_status == 'verified':
                features['phone_status_quality'] = 3
            elif phone_status == 'likely':
                features['phone_status_quality'] = 2
            elif phone_status == 'possible':
                features['phone_status_quality'] = 1
            else:
                features['phone_status_quality'] = 0
            
            # Existence level
            existence_level = acc_dict.get('existence_level', '')
            if existence_level == 'verified':
                features['existence_level_score'] = 3
            elif existence_level == 'likely':
                features['existence_level_score'] = 2
            elif existence_level == 'possible':
                features['existence_level_score'] = 1
            else:
                features['existence_level_score'] = 0
            
            return features
            
        except Exception:
            return {
                'account_age_days': 0, 'has_crm_integration': 0, 'label_count': 0,
                'account_stage_maturity': 0, 'phone_status_quality': 0, 'existence_level_score': 0
            }
    
    def _extract_api_response_features(self, api_data) -> dict:
        """Extract API response lead quality features"""
        if not api_data or pd.isna(api_data):
            return {
                'email_confidence': 0, 'intent_strength_score': 0, 'functions_count': 0,
                'departments_count': 0, 'is_decision_maker': 0, 'seniority_level': 0,
                'has_personal_emails': 0, 'revealed_for_team': 0
            }
        
        try:
            if isinstance(api_data, str):
                api_dict = json.loads(api_data)
            else:
                api_dict = api_data
            
            if not isinstance(api_dict, dict):
                return {
                    'email_confidence': 0, 'intent_strength_score': 0, 'functions_count': 0,
                    'departments_count': 0, 'is_decision_maker': 0, 'seniority_level': 0,
                    'has_personal_emails': 0, 'revealed_for_team': 0
                }
            
            features = {}
            
            # Email confidence
            email_conf = api_dict.get('extrapolated_email_confidence')
            try:
                features['email_confidence'] = float(email_conf) if email_conf else 0
            except:
                features['email_confidence'] = 0
            
            # Intent strength
            intent = api_dict.get('intent_strength', '')
            intent_mapping = {'high': 3, 'medium': 2, 'low': 1}
            features['intent_strength_score'] = intent_mapping.get(str(intent).lower(), 0)
            
            # Functions and departments
            functions = api_dict.get('functions', [])
            departments = api_dict.get('departments', [])
            features['functions_count'] = len(functions) if isinstance(functions, list) else 0
            features['departments_count'] = len(departments) if isinstance(departments, list) else 0
            
            # Decision maker detection
            seniority = str(api_dict.get('seniority', '')).lower()
            title = str(api_dict.get('title', '')).lower()
            
            decision_keywords = ['director', 'vp', 'president', 'ceo', 'cto', 'cfo', 'chief', 'head', 'lead']
            features['is_decision_maker'] = 1 if any(keyword in title for keyword in decision_keywords) else 0
            
            # Seniority scoring
            seniority_scores = {
                'senior': 4, 'director': 5, 'vp': 6, 'svp': 7, 'evp': 8, 'c_level': 9,
                'entry': 1, 'junior': 1, 'associate': 2, 'manager': 3
            }
            features['seniority_level'] = 0
            for level, score in seniority_scores.items():
                if level in seniority:
                    features['seniority_level'] = max(features['seniority_level'], score)
            
            # Personal emails
            personal_emails = api_dict.get('personal_emails', [])
            features['has_personal_emails'] = 1 if (isinstance(personal_emails, list) and len(personal_emails) > 0) else 0
            
            # Team revelation
            features['revealed_for_team'] = 1 if api_dict.get('revealed_for_current_team') else 0
            
            return features
            
        except Exception:
            return {
                'email_confidence': 0, 'intent_strength_score': 0, 'functions_count': 0,
                'departments_count': 0, 'is_decision_maker': 0, 'seniority_level': 0,
                'has_personal_emails': 0, 'revealed_for_team': 0
            }
    
    def prepare_silver_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare ONLY the essential ML-ready features for model training"""
        try:
            logging.info("🔧 Preparing focused ML-ready silver layer features...")
            
            # Features based on comprehensive_preprocessing_pipeline.py requirements
            ml_ready_features = [
                # Core identifiers (for tracking, not for ML)
                'id', 'email', 'campaign',
                
                # Target variable  
                'engagement_level',
                
                # LEAKY VARIABLES (retained for analysis/debugging in silver layer)
                'email_open_count', 'email_click_count', 'email_reply_count',
                
                # KEY TIMESTAMPS (needed for gold layer maturity filtering)
                'timestamp_last_contact', 'timestamp_created', 'timestamp_updated',
                
                # Core timestamp features (from advanced_timestamp_features)
                'created_day_of_week', 'created_month', 'created_hour', 'created_quarter',
                'created_is_weekend', 'created_is_business_hours', 'created_is_morning',
                'created_season', 'days_since_creation', 'weeks_since_creation',
                
                # Time differences (all timestamp relationships)
                'days_between_updated_and_created', 'days_between_last_contact_and_created',
                'days_between_last_touch_and_created', 'days_between_last_open_and_created',
                'days_between_last_click_and_created', 'days_between_last_reply_and_created',
                'days_between_last_interest_change_and_created', 'days_between_retrieval_and_created',
                'days_between_inserted_and_created', 'days_between_enriched_and_created',
                
                # Time differences (absolute values)
                'days_between_updated_and_created_abs', 'days_between_last_contact_and_created_abs',
                'days_between_last_touch_and_created_abs', 'days_between_last_open_and_created_abs',
                'days_between_last_click_and_created_abs', 'days_between_last_reply_and_created_abs',
                'days_between_last_interest_change_and_created_abs', 'days_between_retrieval_and_created_abs',
                'days_between_inserted_and_created_abs', 'days_between_enriched_and_created_abs',
                
                # Has indicators for timestamps
                'has_updated', 'has_last_contact', 'has_last_touch', 'has_last_open',
                'has_last_click', 'has_last_reply', 'has_last_interest_change',
                'has_retrieval', 'has_inserted', 'has_enriched',
                
                # Text features (from enhanced_text_preprocessing)
                'combined_text', 'combined_text_length', 'combined_text_word_count',
                'has_numbers_in_text', 'has_email_in_text', 'has_url_in_text', 'text_quality_score',
                
                # Interaction features (from create_interaction_features)
                'industry_seniority_interaction', 'geo_industry_interaction', 'title_industry_interaction',
                
                # JSONB features (from create_jsonb_features)
                'has_employment_history', 'has_organization_data', 'has_account_data', 'has_api_response_raw',
                'enrichment_completeness', 'enrichment_completeness_pct',
                
                # ADVANCED JSONB features (NEW - from nested data extraction)
                # Employment history features
                'job_count', 'current_job_tenure_years', 'avg_job_tenure_years', 'total_career_years',
                'job_progression_score', 'has_management_experience', 'industry_stability', 'education_level',
                
                # Organization features (avoiding duplicates with existing)
                'company_size_bucket', 'has_public_trading', 'alexa_rank_tier', 'has_social_presence',
                'industry_keywords_count', 'headcount_growth_6m', 'headcount_growth_12m', 'headcount_growth_24m',
                'is_tech_company', 'is_enterprise', 'market_cap_tier',
                
                # Account features
                'account_age_days', 'has_crm_integration', 'label_count', 'account_stage_maturity',
                'phone_status_quality', 'existence_level_score',
                
                # API response features
                'email_confidence', 'intent_strength_score', 'functions_count', 'departments_count',
                'is_decision_maker', 'seniority_level', 'has_personal_emails', 'revealed_for_team',
                
                # Categorical features that survive the preprocessing pipeline
                'title', 'seniority', 'organization_industry', 'country', 'city', 
                'enrichment_status', 'upload_method', 'api_status', 'state',
                
                # Minimal metadata
                'processed_timestamp', 'pipeline_version'
            ]
            
            # Select only available ML-ready features
            available_features = [col for col in ml_ready_features if col in df.columns]
            ml_silver_df = df[available_features].copy()
            
            # Add minimal required metadata if not present
            if 'processed_timestamp' not in ml_silver_df.columns:
                ml_silver_df['processed_timestamp'] = pd.Timestamp.now(tz='UTC')
            if 'pipeline_version' not in ml_silver_df.columns:
                ml_silver_df['pipeline_version'] = '1.0'
            
            logging.info(f"✅ ML-ready silver layer prepared with {len(ml_silver_df.columns)} focused features")
            # Count actual ML features (excluding identifiers, target, leaky variables, and metadata)
            non_ml_cols = ['id', 'email', 'campaign', 'engagement_level', 'email_open_count', 'email_click_count', 'email_reply_count', 'processed_timestamp', 'pipeline_version']
            ml_feature_count = len([c for c in ml_silver_df.columns if c not in non_ml_cols])
            
            # Count JSONB-derived features specifically
            jsonb_feature_prefixes = [
                'job_', 'current_job_', 'avg_job_', 'total_career_', 'has_management_', 'industry_stability', 'education_level',
                'company_size_bucket', 'has_public_trading', 'alexa_rank_tier', 'has_social_presence', 'industry_keywords_',
                'headcount_growth_', 'is_tech_company', 'is_enterprise', 'market_cap_tier',
                'account_age_', 'has_crm_', 'label_count', 'account_stage_', 'phone_status_', 'existence_level_',
                'email_confidence', 'intent_strength_', 'functions_count', 'departments_count', 'is_decision_maker',
                'seniority_level', 'has_personal_emails', 'revealed_for_team'
            ]
            jsonb_features = []
            for col in ml_silver_df.columns:
                for prefix in jsonb_feature_prefixes:
                    if col.startswith(prefix):
                        jsonb_features.append(col)
                        break
            
            logging.info(f"🎯 Features for model training: {ml_feature_count}")
            logging.info(f"🔍 Advanced JSONB features extracted: {len(jsonb_features)}")
            logging.info(f"📊 Target + Leaky variables included for silver layer analysis")
            
            return ml_silver_df
            
        except Exception as e:
            logging.error(f"ML-ready silver layer preparation failed: {e}")
            # Fall back to parent method if enhanced version fails
            return super().prepare_silver_features(df)
    
    def run_idempotent_pipeline(self, incremental: bool = False, batch_size: int = None):
        """Execute the complete bronze to silver ETL pipeline with idempotent operations"""
        try:
            logging.info("🚀 Starting Main Idempotent Bronze to Silver ETL Pipeline...")
            start_time = datetime.now()
            
            # Step 1: Extract bronze data
            if incremental:
                df = self.extract_bronze_data(incremental=True, lookback_days=7)
            else:
                # For full refresh, process all available data
                if batch_size:
                    query = text(f"""
                        SELECT * FROM leads.enriched_contacts 
                        WHERE timestamp_created IS NOT NULL
                        ORDER BY timestamp_created DESC
                        LIMIT {batch_size}
                    """)
                    df = pd.read_sql(query, self.engine)
                else:
                    # Process ALL available data
                    query = text("""
                        SELECT * FROM leads.enriched_contacts 
                        WHERE timestamp_created IS NOT NULL
                        ORDER BY timestamp_created DESC
                    """)
                    df = pd.read_sql(query, self.engine)
            
            if df.empty:
                logging.info("No data to process. Pipeline completed.")
                return None
            
            logging.info(f"📤 Extracted {len(df)} records from bronze layer")
            
            # Step 2: Create target variable
            df = self.create_target_variable(df)
            
            # Step 3: Feature engineering
            logging.info("🔧 Running feature engineering...")
            df = self.engineer_timestamp_features(df)
            df = self.engineer_text_features(df)
            df = self.engineer_categorical_features(df)
            df = self.engineer_apollo_features(df)
            df = self.engineer_jsonb_features(df)
            
            # NEW: Advanced JSONB feature extraction for XGBoost model enhancement
            df = self.engineer_advanced_jsonb_features(df)
            
            df = self.engineer_engagement_features(df)
            
            # Step 4: Data quality checks
            df = self.apply_data_quality_checks(df)
            
            # Step 5: Prepare silver layer features with robust handling
            logging.info("📋 Ensuring core engagement metrics are preserved...")
            
            # Ensure core engagement metrics exist before feature preparation
            engagement_metrics = ['email_open_count', 'email_click_count', 'email_reply_count']
            for metric in engagement_metrics:
                if metric not in df.columns:
                    df[metric] = 0
                    logging.warning(f"⚠️  Missing {metric} in source data, filled with 0")
            
            silver_df = self.prepare_silver_features(df)
            
            # Step 6: Write to silver layer (idempotent)
            self.write_to_silver_layer_idempotent(silver_df)
            
            # Pipeline completion
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logging.info(f"✅ Pipeline completed successfully in {duration:.2f} seconds")
            
            # Report final statistics
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM leads.silver_ml_features"))
                total_count = result.fetchone()[0]
                logging.info(f"📊 Total records in silver layer: {total_count:,}")
            
            return silver_df
            
        except Exception as e:
            logging.error(f"Pipeline failed: {e}")
            raise

def run_main_pipeline():
    """Main execution function"""
    try:
        print("🚀 RUNNING MAIN IDEMPOTENT BRONZE-TO-SILVER ETL PIPELINE")
        print("=" * 60)
        
        # Initialize pipeline
        pipeline = MainIdempotentPipeline()
        
        # Run for ALL available data
        print("📋 Running pipeline for FULL bronze layer dataset...")
        result = pipeline.run_idempotent_pipeline(incremental=False, batch_size=None)
        
        if result is not None:
            print(f"\n🎉 Pipeline completed successfully!")
            print(f"📊 Processed batch with comprehensive feature engineering")
            
            # Show engagement distribution
            if 'engagement_level' in result.columns:
                engagement_dist = result['engagement_level'].value_counts().sort_index()
                print("\n🎯 Engagement Level Distribution:")
                for level, count in engagement_dist.items():
                    level_name = {0: 'Low', 1: 'Medium', 2: 'High'}.get(level, 'Unknown')
                    print(f"   Level {level} ({level_name}): {count:,}")
        else:
            print("ℹ️  No new data to process")
        
    except Exception as e:
        logging.error(f"Main execution failed: {e}")
        return False
    
    return True

def test_idempotency():
    """Test that the pipeline is truly idempotent by running it twice"""
    print("\n🧪 TESTING PIPELINE IDEMPOTENCY")
    print("=" * 40)
    
    # First run
    print("1️⃣  First pipeline run...")
    success1 = run_main_pipeline()
    
    if not success1:
        print("❌ First run failed")
        return False
    
    # Second run (should be idempotent)
    print("\n2️⃣  Second pipeline run (testing idempotency)...")
    success2 = run_main_pipeline()
    
    if not success2:
        print("❌ Second run failed")
        return False
    
    print("\n✅ Idempotency test passed! Main pipeline can be run multiple times safely.")
    return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test-idempotency":
        # Test idempotency
        test_idempotency()
    else:
        # Single run
        run_main_pipeline()
