"""
RDS 2-Bucket XGBoost Model: Opening vs Not Opening

This script creates a binary classification model that predicts whether
an email will be opened (engagement_level >= 1) or not opened (engagement_level = 0).
Connects directly to RDS database instead of using CSV files.
"""

import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# --- Database Configuration ---
DB_CONFIG = {
    'host': 'localhost',  # Use localhost since we're connecting through SSH tunnel
    'database': 'postgres',
    'user': 'mitchell',
    'password': 'CTej3Ba8uBrx6o',
    'port': 5431  # Local port forwarded through SSH tunnel
}

# Columns to drop (leakage, identifiers, empty columns)
COLS_TO_DROP = [
    # Personal identifiers
    'id', 'email', 'first_name', 'last_name', 'company_name', 'linkedin_url',
    'website', 'headline', 'company_domain', 'phone', 'apollo_id',
    'apollo_name', 'organization', 'photo_url', 'organization_name',
    'organization_website', 'organization_phone',
    
    # DIRECT LEAKAGE COLUMNS - These directly encode the target outcome
    'email_open_count', 'email_reply_count', 'email_click_count',
    'email_opened_variant', 'email_opened_step', 'email_clicked_variant', 
    'email_clicked_step', 'email_replied_variant', 'email_replied_step',
    'timestamp_last_open', 'timestamp_last_reply', 'timestamp_last_click', 
    'timestamp_last_touch', 'timestamp_last_interest_change', 'timestamp_last_contact',
    'status_summary', 'status_x', 'status_y',
    
    # ENGINEERED LEAKAGE FEATURES - Derived from outcome columns
    'email_open_count_log', 'email_click_count_log', 'email_reply_count_log',
    'email_open_count_sma', 'email_open_count_ema', 'email_open_count_binned',
    'email_click_count_sma', 'email_click_count_ema', 'email_click_count_binned',
    'email_reply_count_sma', 'email_reply_count_ema', 'email_reply_count_binned',
    'status_x_sma', 'status_x_ema', 'status_x_binned',
    
    # HAS_VALUE LEAKAGE FEATURES - These encode whether outcome occurred
    'email_opened_step_has_value', 'email_opened_variant_has_value',
    'email_clicked_step_has_value', 'email_clicked_variant_has_value',
    'email_replied_step_has_value', 'email_replied_variant_has_value',
    
    # Metadata and campaign tracking (safe to drop)
    'personalization', 'payload', 'list_id', 'assigned_to', 'campaign', 
    'uploaded_by_user', 'auto_variant_select', 'verification_status',
    'campaign_id', 'email_subjects', 'email_bodies', 'campaign_schedule_end_date',
    'campaign_schedule_start_date', 'campaign_schedule_schedules', 'daily_limit',
    'email_gap', 'email_list', 'email_tag_list', 'insert_unsubscribe_header',
    'link_tracking', 'name', 'open_tracking', 'prioritize_new_leads', 'sequences',
    'stop_for_company', 'stop_on_auto_reply', 'stop_on_reply', 'text_only',
    'upload_method', 'esp_code', 'page_retrieved', 'retrieval_timestamp',
    'last_contacted_from', 'enrichment_status', 'inserted_at', 'enriched_at',
    'api_response_raw', 'credits_consumed', 'api_status', 'auto_variant_select_trigger',
    'email_gap_has_value', 'daily_limit_log',
    
    # Timestamp columns (can cause issues and may leak timing info)
    'timestamp_created_x', 'timestamp_created_y', 'timestamp_updated_x', 'timestamp_updated_y',
    'timestamp_created_x_day_of_week', 'timestamp_created_x_hour', 'timestamp_created_x_month',
    'timestamp_updated_x_day_of_week', 'timestamp_updated_x_hour', 'timestamp_updated_x_month',
    'timestamp_last_contact_day_of_week', 'timestamp_last_contact_hour', 'timestamp_last_contact_month',
    'retrieval_timestamp',
    
    # Organization duplicates and complex fields
    'organization_x', 'organization_y', 'organization_data', 'account_data',
    'employment_history', 'departments', 'functions'
]

# Text columns for feature engineering
TEXT_COLS = ['campaign_id', 'email_subjects', 'email_bodies']

# Categorical columns for encoding
CATEGORICAL_COLS = ['title', 'seniority', 'organization_industry', 'country', 'city', 'state', 'verification_status', 'enrichment_status', 'upload_method', 'api_status']

# JSONB columns for feature engineering
JSONB_COLS = ['employment_history', 'organization_data', 'account_data', 'api_response_raw']

def connect_to_database():
    """Establish connection to RDS database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("✅ Successfully connected to RDS database")
        return conn
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        sys.exit(1)

def load_data_from_rds():
    """Load data directly from RDS database."""
    print("=== LOADING DATA FROM RDS DATABASE ===")
    
    conn = connect_to_database()
    
    try:
        # Try to load data from the enriched_contacts table
        query = """
        SELECT * FROM leads.enriched_contacts
        WHERE email_open_count IS NOT NULL
        LIMIT 50000  -- Limit for testing, remove for full dataset
        """
        
        print("Executing query to load data from leads.enriched_contacts...")
        df = pd.read_sql_query(query, conn)
        print(f"Data loaded from RDS. Shape: {df.shape}")
        
        if len(df) == 0:
            print("No data found in leads.enriched_contacts. Trying other tables...")
            
            # Try other potential tables
            alternative_tables = [
                "leads.instantly_enriched_contacts",
                "public.instantly_leads", 
                "public.b2b_leads_raw",
                "leads.instantly_contacts_raw"
            ]
            
            for table in alternative_tables:
                try:
                    test_query = f"SELECT COUNT(*) as count FROM {table}"
                    count_df = pd.read_sql_query(test_query, conn)
                    count = count_df['count'].iloc[0]
                    print(f"  {table}: {count:,} rows")
                    
                    if count > 0:
                        print(f"  Found data in {table}. Loading sample...")
                        sample_query = f"SELECT * FROM {table} LIMIT 1000"
                        sample_df = pd.read_sql_query(sample_query, conn)
                        print(f"  Sample columns: {list(sample_df.columns)}")
                        
                except Exception as e:
                    print(f"  {table}: Error - {str(e)[:100]}...")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading data from database: {e}")
        print("Available tables:")
        try:
            tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            tables_df = pd.read_sql_query(tables_query, conn)
            print(tables_df['table_name'].tolist())
        except:
            pass
        sys.exit(1)
    finally:
        conn.close()

def load_and_preprocess_data():
    """Load data from RDS and create binary target variable with enhanced feature engineering."""
    print("=== LOADING AND PREPROCESSING DATA FROM RDS ===")
    
    # Load data from RDS
    df = load_data_from_rds()
    
    # If df is None, it means we were just listing tables
    if df is None:
        print("Cannot proceed with model training - no data available.")
        print("Please import your data into the database first.")
        sys.exit(1)
    
    # Create binary target: 1 if opened (engagement_level >= 1), 0 if not opened
    if 'email_open_count' not in df.columns:
        print("Error: 'email_open_count' column not found.")
        sys.exit(1)
    
    df['opened'] = (df['email_open_count'] > 0).astype(int)
    
    # Check target distribution
    target_dist = df['opened'].value_counts().sort_index()
    print(f"\nBinary target distribution:")
    print(f"  Not opened (0): {target_dist[0]:,} ({target_dist[0]/len(df)*100:.1f}%)")
    print(f"  Opened (1): {target_dist[1]:,} ({target_dist[1]/len(df)*100:.1f}%)")
    
    # Drop unnecessary columns
    df = df.drop(columns=[col for col in COLS_TO_DROP if col in df.columns], errors='ignore')
    
    # Enhanced text preprocessing (matching older script)
    df['combined_text'] = ""
    for col in TEXT_COLS:
        if col in df.columns:
            df['combined_text'] += df[col].fillna('') + ' '
    
    # Text-based features
    df['text_length'] = df['combined_text'].str.len()
    df['text_word_count'] = df['combined_text'].str.split().str.len()
    df['has_numbers'] = df['combined_text'].str.contains(r'\d', regex=True).astype(int)
    df['has_email'] = df['combined_text'].str.contains(r'@', regex=True).astype(int)
    
    # Engineer JSONB presence features (from older script)
    for col in JSONB_COLS:
        if col in df.columns:
            df[f'has_{col}'] = df[col].notna().astype(int)
    
    # Timestamp features (if available)
    if 'timestamp_created' in df.columns:
        df['timestamp_created'] = pd.to_datetime(df['timestamp_created'], errors='coerce')
        df['created_day_of_week'] = df['timestamp_created'].dt.dayofweek
        df['created_hour'] = df['timestamp_created'].dt.hour
        df['created_is_weekend'] = (df['timestamp_created'].dt.dayofweek >= 5).astype(int)
        df['created_is_business_hours'] = (
            (df['timestamp_created'].dt.hour >= 9) & 
            (df['timestamp_created'].dt.hour <= 17)
        ).astype(int)
    
    # Drop all timestamp columns to avoid string conversion issues
    timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower()]
    df = df.drop(columns=timestamp_cols, errors='ignore')
    
    # Handle categorical columns
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)
    
    # Convert categorical columns to numeric using label encoding
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # Separate features and target
    y = df['opened'].copy()
    X = df.drop(columns=['opened', 'combined_text'] + TEXT_COLS, errors='ignore')
    
    # Ensure all features are numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            # Convert to numeric, replacing non-numeric with NaN
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Drop any remaining non-numeric columns
    X = X.select_dtypes(include=[np.number])
    
    print(f"\nFeature engineering complete. Final shape: {X.shape}")
    print(f"Features include: {list(X.columns[:10])}...")
    
    return X, y

def main():
    """Main function for binary classification with RDS connection."""
    print("=== RDS 2-BUCKET XGBOOST: OPENING PREDICTION ===")
    
    # 1. Load and preprocess data from RDS
    X, y = load_and_preprocess_data()
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"\nData split:")
    print(f"  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    
    # 3. Enhanced preprocessing
    print("\n=== APPLYING ENHANCED PREPROCESSING ===")
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Feature selection - use more features for better performance
    selector = SelectKBest(f_classif, k=min(50, X_train_scaled.shape[1]))
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    # Get selected feature names
    selected_mask = selector.get_support()
    original_features = X_train.columns.tolist()
    selected_features = []
    for i, is_selected in enumerate(selected_mask):
        if is_selected and i < len(original_features):
            selected_features.append(original_features[i])
    
    print(f"Preprocessed data shape:")
    print(f"  Training: {X_train_selected.shape}")
    print(f"  Test: {X_test_selected.shape}")
    print(f"  Selected features: {selected_features}")
    
    # 4. Train improved XGBoost model
    print("\n=== TRAINING IMPROVED XGBOOST MODEL ===")
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_estimators=200,  # More trees
        max_depth=8,        # Slightly deeper
        learning_rate=0.1,
        subsample=0.8,      # Add regularization
        colsample_bytree=0.8,
        n_jobs=-1
    )
    
    model.fit(X_train_selected, y_train)
    
    # 5. Evaluate model
    print("\n=== MODEL EVALUATION ===")
    y_pred = model.predict(X_test_selected)
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Opened', 'Opened']))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # 6. Feature importance
    print("\n=== FEATURE IMPORTANCE ===")
    feature_importance = model.feature_importances_
    
    # Create feature importance dataframe
    importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=True)
    
    print(f"\nTop 10 most important features:")
    for i, (_, row) in enumerate(importance_df.tail(10).iterrows(), 1):
        print(f"  {i}. {row['Feature']}: {row['Importance']:.4f}")
    
    # 7. Create feature importance plot
    plt.figure(figsize=(12, 10))
    plt.barh(range(len(importance_df)), importance_df['Importance'])
    plt.yticks(range(len(importance_df)), importance_df['Feature'])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importances - RDS 2-Bucket XGBoost\n(Opening vs Not Opening Prediction)')
    plt.tight_layout()
    plt.savefig('RDS_2_bucket_xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"\nFeature importance plot saved as 'RDS_2_bucket_xgboost_feature_importance.png'")
    plt.show()
    
    # 8. Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Opened', 'Opened'],
                yticklabels=['Not Opened', 'Opened'])
    plt.title('Confusion Matrix - RDS 2-Bucket XGBoost\nOpening Prediction')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('RDS_2_bucket_xgboost_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved as 'RDS_2_bucket_xgboost_confusion_matrix.png'")
    plt.show()
    
    # 9. Final summary
    print(f"\n" + "="*60)
    print("FINAL SUMMARY:")
    print("="*60)
    print(f"✅ Connected to RDS database")
    print(f"✅ Binary classification: Opening vs Not Opening")
    print(f"✅ Enhanced preprocessing (better feature engineering)")
    print(f"✅ More features selected ({len(selected_features)} features)")
    print(f"✅ Improved XGBoost parameters")
    print(f"")
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Training samples: {X_train.shape[0]:,}")
    print(f"Test samples: {X_test.shape[0]:,}")
    print(f"Final features: {X_train_selected.shape[1]}")
    print("="*60)
    
    return model

if __name__ == "__main__":
    best_model = main() 