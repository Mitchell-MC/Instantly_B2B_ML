"""
Advanced RDS Training Pipeline for Email Engagement Prediction
Enhanced ML techniques with direct RDS database connection
"""

import pandas as pd
import numpy as np
import yaml
import joblib
import json
import psycopg2
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Import advanced feature engineering
from feature_engineering_rds import apply_rds_feature_engineering

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

def load_config():
    """Load configuration from YAML file."""
    config_path = Path("config/main_config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def connect_to_database():
    """Establish connection to RDS database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("✅ Successfully connected to RDS database")
        return conn
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        raise

def load_data_from_rds():
    """Load data directly from RDS database."""
    print("📊 Loading data from RDS database...")
    
    conn = connect_to_database()
    
    try:
        # Load data from the enriched_contacts table
        query = """
        SELECT * FROM leads.enriched_contacts
        WHERE email_open_count IS NOT NULL
        AND email_open_count >= 0
        LIMIT 100000  -- Adjust based on your needs
        """
        
        df = pd.read_sql(query, conn)
        print(f"✅ Data loaded from RDS. Shape: {df.shape}")
        
        conn.close()
        return df
        
    except Exception as e:
        print(f"❌ Error loading data from database: {e}")
        conn.close()
        raise

def validate_data(df):
    """Comprehensive data validation."""
    print("🔍 Validating data quality...")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Missing values analysis
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing_Count': missing_data,
        'Missing_Percent': missing_percent
    }).sort_values('Missing_Percent', ascending=False)
    
    print("Top 10 columns with missing values:")
    print(missing_df.head(10))
    
    # Handle duplicate analysis for columns with unhashable types
    print("🔍 Checking for problematic columns...")
    
    # Identify columns with unhashable types (lists, dicts, etc.)
    unhashable_cols = []
    for col in df.columns:
        try:
            # Try to hash a sample of the column
            sample = df[col].dropna().head(100)
            if len(sample) > 0:
                hash(sample.iloc[0])
        except (TypeError, ValueError):
            unhashable_cols.append(col)
    
    if unhashable_cols:
        print(f"⚠️  Found {len(unhashable_cols)} columns with unhashable types:")
        for col in unhashable_cols:
            print(f"   - {col}: {type(df[col].iloc[0]) if not df[col].isna().all() else 'all NaN'}")
        
        # Create a copy for duplicate checking, excluding unhashable columns
        df_for_duplicates = df.drop(columns=unhashable_cols)
        duplicates = df_for_duplicates.duplicated().sum()
        print(f"📊 Duplicate rows (excluding unhashable columns): {duplicates}")
        
        # Show sample of unhashable column data
        print("\n📋 Sample data from unhashable columns:")
        for col in unhashable_cols[:3]:  # Show first 3 columns
            if not df[col].isna().all():
                sample_val = df[col].dropna().iloc[0]
                print(f"   {col}: {str(sample_val)[:100]}...")
    else:
        # No unhashable columns, proceed normally
        duplicates = df.duplicated().sum()
        print(f"📊 Duplicate rows: {duplicates}")
    
    # Data types
    print("\n📊 Data types:")
    print(df.dtypes.value_counts())
    
    return df

def create_target_variable(df):
    """
    Create target variable using new 3-bucket classification:
    0: No opens
    1: 1-2 opens (no clicks/replies)
    2: 3+ opens OR any opens + click OR any opens + reply
    """
    print("🎯 Creating target variable...")
    
    # Initialize target column
    df['engagement_level'] = 0
    
    # Bucket 1: No opens
    # (already initialized as 0)
    
    # Bucket 2: 1-2 opens (no clicks/replies)
    mask_bucket2 = (
        (df['email_open_count'] >= 1) & 
        (df['email_open_count'] <= 2) & 
        (df['email_click_count'] == 0) & 
        (df['email_reply_count'] == 0)
    )
    df.loc[mask_bucket2, 'engagement_level'] = 1
    
    # Bucket 3: 3+ opens OR any opens + click OR any opens + reply
    mask_bucket3 = (
        (df['email_open_count'] >= 3) |
        (df['email_open_count'] >= 1) & (df['email_click_count'] >= 1) |
        (df['email_open_count'] >= 1) & (df['email_reply_count'] >= 1)
    )
    df.loc[mask_bucket3, 'engagement_level'] = 2
    
    # Print distribution
    print("Target variable distribution:")
    print(df['engagement_level'].value_counts().sort_index())
    print(f"Total samples: {len(df)}")
    
    return df

def prepare_features_for_model(df):
    """Prepare features for modeling with advanced preprocessing."""
    print("🔧 Preparing features for modeling...")
    
    # Apply enhanced feature engineering
    df = apply_rds_feature_engineering(df)
    
    # Drop leakage columns
    available_cols_to_drop = [col for col in COLS_TO_DROP if col in df.columns]
    df = df.drop(columns=available_cols_to_drop)
    
    print(f"Features after dropping leakage columns: {df.shape[1]}")
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Impute numeric columns
    if len(numeric_cols) > 0:
        imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # Impute categorical columns
    if len(categorical_cols) > 0:
        imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = imputer.fit_transform(df[categorical_cols])
    
    # Encode categorical variables
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes
    
    print(f"Final feature count: {df.shape[1]}")
    return df

def perform_advanced_feature_selection(X, y):
    """Perform advanced feature selection using multiple methods."""
    print("🔍 Performing advanced feature selection...")
    
    # Method 1: Correlation-based selection
    correlation_threshold = 0.95
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]
    
    # Method 2: Statistical tests
    selector = SelectKBest(score_func=f_classif, k=min(100, X.shape[1]))
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    # Method 3: Mutual information
    mi_selector = SelectKBest(score_func=mutual_info_classif, k=min(100, X.shape[1]))
    mi_selector.fit(X, y)
    mi_selected_features = X.columns[mi_selector.get_support()].tolist()
    
    # Combine all methods
    all_selected = list(set(selected_features + mi_selected_features))
    final_features = [f for f in all_selected if f not in high_corr_features]
    
    print(f"Selected {len(final_features)} features out of {X.shape[1]} original features")
    return final_features

def create_advanced_ensemble_model(X_train, y_train, config):
    """Create an advanced ensemble model with multiple algorithms."""
    print("🤖 Creating advanced ensemble model...")
    
    # Base models
    models = {
        'rf': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ),
        'xgb': xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),
        'gbm': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        ),
        'lr': LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
    }
    
    # Train base models
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    # Create voting classifier
    ensemble = VotingClassifier(
        estimators=[(name, model) for name, model in trained_models.items()],
        voting='soft'
    )
    
    return ensemble, trained_models

def perform_advanced_cross_validation(model, X, y, config):
    """Perform advanced cross-validation with multiple metrics."""
    print("📊 Performing advanced cross-validation...")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return cv_scores

def evaluate_model_advanced(model, X_test, y_test, cv_results):
    """Comprehensive model evaluation."""
    print("📈 Evaluating model performance...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")
    
    # ROC AUC (for binary classification)
    if len(np.unique(y_test)) == 2:
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        print(f"ROC AUC: {auc:.4f}")
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'confusion_matrix': cm
    }

def save_advanced_model_artifacts(model, feature_names, performance_metrics, config):
    """Save model artifacts and performance metrics."""
    print("💾 Saving model artifacts...")
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = models_dir / "email_engagement_predictor_rds_v1.0.joblib"
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")
    
    # Save feature names
    features_path = models_dir / "feature_names_rds_v1.0.json"
    with open(features_path, 'w') as f:
        json.dump(feature_names, f)
    print(f"✅ Feature names saved to {features_path}")
    
    # Save performance metrics
    metrics_path = models_dir / "performance_metrics_rds_v1.0.json"
    with open(metrics_path, 'w') as f:
        json.dump(performance_metrics, f)
    print(f"✅ Performance metrics saved to {metrics_path}")

def main():
    """Main training pipeline."""
    print("🚀 Starting Advanced RDS Training Pipeline")
    print("=" * 50)
    
    try:
        # Load configuration
        config = load_config()
        
        # Load data from RDS
        df = load_data_from_rds()
        
        # Validate data
        df = validate_data(df)
        
        # Create target variable
        df = create_target_variable(df)
        
        # Prepare features
        df = prepare_features_for_model(df)
        
        # Split data
        X = df.drop('engagement_level', axis=1)
        y = df['engagement_level']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Feature selection
        selected_features = perform_advanced_feature_selection(X_train, y_train)
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
        # Create and train model
        ensemble_model, base_models = create_advanced_ensemble_model(
            X_train_selected, y_train, config
        )
        
        # Cross-validation
        cv_results = perform_advanced_cross_validation(
            ensemble_model, X_train_selected, y_train, config
        )
        
        # Evaluate model
        performance_metrics = evaluate_model_advanced(
            ensemble_model, X_test_selected, y_test, cv_results
        )
        
        # Save artifacts
        save_advanced_model_artifacts(
            ensemble_model, selected_features, performance_metrics, config
        )
        
        print("\n🎉 Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in training pipeline: {e}")
        raise

if __name__ == "__main__":
    main()
