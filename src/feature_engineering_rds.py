"""
RDS Feature Engineering Module for Email Engagement Prediction
Production-ready feature engineering optimized for RDS database data.
"""

import pandas as pd
import numpy as np
import psycopg2
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
import json

# --- Database Configuration ---
DB_CONFIG = {
    'host': 'localhost',  # Use localhost since we're connecting through SSH tunnel
    'database': 'postgres',
    'user': 'mitchell',
    'password': 'CTej3Ba8uBrx6o',
    'port': 5431  # Local port forwarded through SSH tunnel
}

def connect_to_database():
    """Establish connection to RDS database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("✅ Successfully connected to RDS database")
        return conn
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        raise

def load_sample_data_from_rds(sample_size=10000):
    """Load sample data from RDS for feature engineering development."""
    print("📊 Loading sample data from RDS for feature engineering...")
    
    conn = connect_to_database()
    
    try:
        query = f"""
        SELECT * FROM leads.enriched_contacts
        WHERE email_open_count IS NOT NULL
        AND email_open_count >= 0
        LIMIT {sample_size}
        """
        
        df = pd.read_sql(query, conn)
        print(f"✅ Sample data loaded. Shape: {df.shape}")
        
        conn.close()
        return df
        
    except Exception as e:
        print(f"❌ Error loading sample data: {e}")
        conn.close()
        raise

def enhanced_text_preprocessing(df):
    """Advanced text preprocessing with quality scoring and combined features."""
    print("🔧 Enhanced text preprocessing...")
    
    # Text columns for feature engineering
    text_cols = ['campaign_id', 'email_subjects', 'email_bodies']
    
    # Clean and normalize text columns
    for col in text_cols:
        if col in df.columns:
            # Basic cleaning
            df[col] = df[col].astype(str).str.lower()
            df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True)
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
            df[col] = df[col].str.strip()
    
    # Create combined text column
    df['combined_text'] = ""
    for col in text_cols:
        if col in df.columns:
            df['combined_text'] += df[col].fillna('') + ' '
    
    # Create text-based features
    df['combined_text_length'] = df['combined_text'].str.len()
    df['combined_text_word_count'] = df['combined_text'].str.split().str.len()
    df['has_numbers_in_text'] = df['combined_text'].str.contains(r'\d', regex=True).astype(int)
    df['has_email_in_text'] = df['combined_text'].str.contains(r'@', regex=True).astype(int)
    df['has_url_in_text'] = df['combined_text'].str.contains(r'http', regex=True).astype(int)
    
    # Text quality indicators
    df['text_quality_score'] = (
        (df['combined_text_length'] > 10).astype(int) +
        (df['combined_text_word_count'] > 3).astype(int) +
        (~df['combined_text'].str.contains('nan', regex=True)).astype(int)
    )
    
    print(f"Text features created. Combined text length range: {df['combined_text_length'].min()}-{df['combined_text_length'].max()}")
    
    return df

def advanced_timestamp_features(df):
    """Enhanced timestamp feature engineering with business insights."""
    print("🔧 Advanced timestamp feature engineering...")
    
    timestamp_cols = ['timestamp_created', 'timestamp_last_contact', 'retrieval_timestamp', 
                     'enriched_at', 'inserted_at', 'last_contacted_from']
    
    if 'timestamp_created' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp_created']):
        # Basic datetime features
        df['created_day_of_week'] = df['timestamp_created'].dt.dayofweek
        df['created_month'] = df['timestamp_created'].dt.month
        df['created_hour'] = df['timestamp_created'].dt.hour
        df['created_quarter'] = df['timestamp_created'].dt.quarter
        
        # Business-relevant features
        df['created_is_weekend'] = (df['timestamp_created'].dt.dayofweek >= 5).astype(int)
        df['created_is_business_hours'] = (
            (df['timestamp_created'].dt.hour >= 9) & 
            (df['timestamp_created'].dt.hour <= 17)
        ).astype(int)
        df['created_is_morning'] = (df['timestamp_created'].dt.hour <= 12).astype(int)
        
        # Seasonal features
        df['created_season'] = df['timestamp_created'].dt.month % 12 // 3
        
        # Time differences relative to creation
        ref_date = df['timestamp_created']
        for col in timestamp_cols:
            if col in df.columns and col != 'timestamp_created' and pd.api.types.is_datetime64_any_dtype(df[col]):
                feature_name = f"days_between_{col.replace('timestamp_', '')}_and_created"
                
                # Handle timezone differences
                try:
                    # Ensure both columns have the same timezone awareness
                    col_data = df[col]
                    ref_data = ref_date
                    
                    # If one is tz-aware and the other isn't, localize the naive one
                    if col_data.dt.tz is not None and ref_data.dt.tz is None:
                        ref_data = ref_data.dt.tz_localize('UTC')
                    elif col_data.dt.tz is None and ref_data.dt.tz is not None:
                        col_data = col_data.dt.tz_localize('UTC')
                    elif col_data.dt.tz is not None and ref_data.dt.tz is not None:
                        # Both are tz-aware, convert to same timezone
                        col_data = col_data.dt.tz_convert('UTC')
                        ref_data = ref_data.dt.tz_convert('UTC')
                    
                    df[feature_name] = (col_data - ref_data).dt.days
                    
                    # Additional temporal features
                    df[f"{feature_name}_abs"] = df[feature_name].abs()
                    df[f"has_{col.replace('timestamp_', '')}"] = df[col].notna().astype(int)
                    
                except Exception as e:
                    print(f"Warning: Could not calculate time difference for {col}: {e}")
                    # Create a simple has_value feature instead
                    df[f"has_{col.replace('timestamp_', '')}"] = df[col].notna().astype(int)
        
        # Recency from current time
        try:
            current_time = pd.Timestamp.now(tz='UTC')
            if df['timestamp_created'].dt.tz is None:
                df['days_since_creation'] = (current_time - df['timestamp_created'].dt.tz_localize('UTC')).dt.days
            else:
                df['days_since_creation'] = (current_time - df['timestamp_created'].dt.tz_convert('UTC')).dt.days
        except Exception as e:
            print(f"Warning: Could not calculate days since creation: {e}")
            df['days_since_creation'] = 0
    
    print("Timestamp features created successfully")
    return df

def create_interaction_features(df):
    """Create interaction-based features from engagement data."""
    print("🔧 Creating interaction features...")
    
    # Engagement ratios
    if 'email_open_count' in df.columns and 'email_click_count' in df.columns:
        df['open_to_click_ratio'] = np.where(
            df['email_open_count'] > 0,
            df['email_click_count'] / df['email_open_count'],
            0
        )
    
    if 'email_open_count' in df.columns and 'email_reply_count' in df.columns:
        df['open_to_reply_ratio'] = np.where(
            df['email_open_count'] > 0,
            df['email_reply_count'] / df['email_open_count'],
            0
        )
    
    # Total engagement score
    engagement_cols = ['email_open_count', 'email_click_count', 'email_reply_count']
    available_engagement_cols = [col for col in engagement_cols if col in df.columns]
    
    if available_engagement_cols:
        df['total_engagement_score'] = df[available_engagement_cols].sum(axis=1)
    
    # Engagement diversity (how many different types of engagement)
    df['engagement_diversity'] = 0
    if 'email_open_count' in df.columns:
        df['engagement_diversity'] += (df['email_open_count'] > 0).astype(int)
    if 'email_click_count' in df.columns:
        df['engagement_diversity'] += (df['email_click_count'] > 0).astype(int)
    if 'email_reply_count' in df.columns:
        df['engagement_diversity'] += (df['email_reply_count'] > 0).astype(int)
    
    print("Interaction features created successfully")
    return df

def create_jsonb_features(df):
    """Extract features from JSONB columns."""
    print("🔧 Creating JSONB features...")
    
    jsonb_cols = ['employment_history', 'organization_data', 'account_data', 'api_response_raw']
    
    for col in jsonb_cols:
        if col in df.columns:
            # Extract basic JSON features
            df[f'{col}_has_data'] = df[col].notna().astype(int)
            df[f'{col}_length'] = df[col].astype(str).str.len()

            # Parse once per row
            def _safe_parse(obj):
                try:
                    if pd.notna(obj) and obj != 'null':
                        return json.loads(obj) if isinstance(obj, str) else obj
                except Exception:
                    return None
                return None

            parsed = df[col].apply(_safe_parse)

            try:
                if col == 'organization_data':
                    df[f'{col}_employee_count'] = parsed.apply(
                        lambda d: d.get('employee_count') if isinstance(d, dict) else None
                    )
                    df[f'{col}_founded_year'] = parsed.apply(
                        lambda d: d.get('founded_year') if isinstance(d, dict) else None
                    )
                    df[f'{col}_industry'] = parsed.apply(
                        lambda d: d.get('industry') if isinstance(d, dict) else None
                    )
                elif col == 'employment_history':
                    df[f'{col}_entries_count'] = parsed.apply(
                        lambda lst: len(lst) if isinstance(lst, list) else 0
                    )
            except Exception as e:
                print(f"Warning: Could not parse JSONB column {col}: {e}")
    
    print("JSONB features created successfully")
    return df

def extract_json_field(json_str, field_name):
    """Helper function to extract fields from JSON strings."""
    try:
        if pd.isna(json_str) or json_str == 'null':
            return None
        data = json.loads(json_str) if isinstance(json_str, str) else json_str
        return data.get(field_name)
    except:
        return None

def handle_outliers(df):
    """Handle outliers in numeric columns using IQR method."""
    print("🔧 Handling outliers...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing them
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    print("Outliers handled successfully")
    return df

def create_organization_features(df):
    """Create features from organization-related data."""
    print("🔧 Creating organization features...")
    
    # Company size features
    if 'organization_employee_count' in df.columns:
        df['company_size_category'] = pd.cut(
            df['organization_employee_count'],
            bins=[0, 10, 50, 200, 1000, float('inf')],
            labels=['Startup', 'Small', 'Medium', 'Large', 'Enterprise'],
            include_lowest=True
        )
    
    # Industry features
    if 'organization_industry' in df.columns:
        # Create industry categories
        tech_industries = ['technology', 'software', 'internet', 'computer', 'ai', 'machine learning']
        df['is_tech_company'] = df['organization_industry'].str.lower().str.contains(
            '|'.join(tech_industries), regex=True, na=False
        ).astype(int)
    
    # Location features
    location_cols = ['country', 'state', 'city']
    for col in location_cols:
        if col in df.columns:
            df[f'{col}_encoded'] = LabelEncoder().fit_transform(df[col].fillna('Unknown'))
    
    print("Organization features created successfully")
    return df

def create_advanced_engagement_features(df):
    """Create advanced engagement-based features."""
    print("🔧 Creating advanced engagement features...")
    
    # Engagement velocity (engagement per time unit)
    if 'email_open_count' in df.columns and 'days_since_creation' in df.columns:
        df['open_velocity'] = np.where(
            df['days_since_creation'] > 0,
            df['email_open_count'] / df['days_since_creation'],
            0
        )
    
    # Engagement consistency (variance in engagement)
    engagement_cols = ['email_open_count', 'email_click_count', 'email_reply_count']
    available_engagement_cols = [col for col in engagement_cols if col in df.columns]
    
    if len(available_engagement_cols) > 1:
        df['engagement_consistency'] = df[available_engagement_cols].std(axis=1)
    
    # Engagement momentum (trend in engagement)
    if 'email_open_count' in df.columns:
        df['high_engagement'] = (df['email_open_count'] >= 3).astype(int)
        df['moderate_engagement'] = ((df['email_open_count'] >= 1) & (df['email_open_count'] <= 2)).astype(int)
        df['no_engagement'] = (df['email_open_count'] == 0).astype(int)
    
    # Multi-channel engagement
    df['multi_channel_engaged'] = 0
    if 'email_open_count' in df.columns and 'email_click_count' in df.columns:
        df['multi_channel_engaged'] = (
            (df['email_open_count'] > 0) & (df['email_click_count'] > 0)
        ).astype(int)
    
    print("Advanced engagement features created successfully")
    return df

def encode_categorical_features(df, max_categories=100):
    """Encode categorical features with frequency-based encoding, handling unhashable types."""
    print("🔧 Encoding categorical features...")
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        if col in df.columns:
            try:
                # Check if column contains unhashable types (like lists from JSONB)
                sample_values = df[col].dropna().head(100)
                if any(isinstance(val, (list, dict, set)) for val in sample_values):
                    print(f"⚠️ Skipping {col} - contains unhashable types (likely JSONB data)")
                    continue
                
                # Get value counts
                value_counts = df[col].value_counts()
                
                # If too many categories, keep only the most frequent ones
                if len(value_counts) > max_categories:
                    top_categories = value_counts.head(max_categories).index
                    df[col] = df[col].apply(lambda x: x if x in top_categories else 'Other')
                
                # Frequency encoding
                freq_encoding = df[col].value_counts(normalize=True)
                df[f'{col}_freq_encoded'] = df[col].map(freq_encoding).fillna(0)
                
                # Label encoding
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df[f'{col}_label_encoded'] = le.fit_transform(df[col].fillna('Unknown'))
                
                print(f"  ✅ Encoded {col} ({len(value_counts)} categories)")
                
            except (TypeError, ValueError) as e:
                print(f"⚠️ Skipping {col} - encoding failed: {e}")
                continue
    
    print("Categorical features encoded successfully")
    return df

def prepare_features_for_model(df, target_variable='engagement_level', cols_to_drop=None):
    """Prepare features for modeling with comprehensive preprocessing."""
    print("🔧 Preparing features for modeling...")
    
    # Apply all feature engineering steps
    df = enhanced_text_preprocessing(df)
    df = advanced_timestamp_features(df)
    df = create_interaction_features(df)
    df = create_jsonb_features(df)
    df = handle_outliers(df)
    df = create_organization_features(df)
    df = create_advanced_engagement_features(df)
    df = encode_categorical_features(df)
    
    # Drop specified columns
    if cols_to_drop:
        available_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        df = df.drop(columns=available_cols_to_drop)
    
    # Ensure target variable is not in features
    if target_variable in df.columns:
        df = df.drop(columns=[target_variable])
    
    print(f"Final feature count: {df.shape[1]}")
    return df

def apply_rds_feature_engineering(df):
    """Main function to apply all RDS-optimized feature engineering."""
    print("🚀 Applying RDS Feature Engineering Pipeline")
    print("=" * 50)
    
    # Apply comprehensive feature engineering
    df = prepare_features_for_model(df)
    
    print("✅ RDS Feature Engineering completed successfully!")
    return df

def test_rds_feature_engineering():
    """Test the RDS feature engineering pipeline with sample data."""
    print("🧪 Testing RDS Feature Engineering Pipeline")
    print("=" * 50)
    
    try:
        # Load sample data
        df = load_sample_data_from_rds(sample_size=1000)
        
        # Apply feature engineering
        df_engineered = apply_rds_feature_engineering(df)
        
        print(f"Original shape: {df.shape}")
        print(f"Engineered shape: {df_engineered.shape}")
        print(f"Features added: {df_engineered.shape[1] - df.shape[1]}")
        
        # Show some sample features
        print("\nSample engineered features:")
        new_features = [col for col in df_engineered.columns if col not in df.columns]
        print(new_features[:10])
        
        return df_engineered
        
    except Exception as e:
        print(f"❌ Error in feature engineering test: {e}")
        raise

if __name__ == "__main__":
    # Test the feature engineering pipeline
    test_rds_feature_engineering()
