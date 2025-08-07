"""
Production ML Pipeline - Prediction Module
Loads trained model and makes predictions on new data.
"""

import pandas as pd
import numpy as np
import yaml
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import (
    enhanced_text_preprocessing, advanced_timestamp_features, 
    create_interaction_features, create_jsonb_features, handle_outliers,
    create_xgboost_optimized_features, encode_categorical_features, 
    prepare_features_for_model
)

def load_config():
    """Load configuration from YAML file."""
    config_path = Path("config/main_config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

def load_model_artifacts():
    """Load trained model artifacts."""
    print("🔧 Loading model artifacts...")
    
    # Try to load the optimized model first
    model_path = Path("models/email_open_predictor_optimized_v1.0.joblib")
    if model_path.exists():
        artifacts = joblib.load(model_path)
        print(f"✅ Loaded optimized model from: {model_path}")
    else:
        # Fallback to regular model
        model_path = Path("models/email_open_predictor_v1.0.joblib")
        if model_path.exists():
            artifacts = joblib.load(model_path)
            print(f"✅ Loaded regular model from: {model_path}")
        else:
            raise FileNotFoundError("No trained model found. Please run training first.")
    
    return artifacts

def load_apollo_data(config):
    """Load Apollo contacts data."""
    print("📊 Loading Apollo contacts data...")
    
    data_path = Path("apollo-contacts-export.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Apollo data file not found: {data_path}")
    
    # Load data with proper error handling
    try:
        df = pd.read_csv(data_path, on_bad_lines='warn', low_memory=False)
        print(f"✅ Apollo data loaded. Shape: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise
    
    # Standardize column names to match training data format
    column_mapping = {
        'First Name': 'first_name',
        'Last Name': 'last_name',
        'Title': 'title',
        'Company': 'company_name',
        'Company Name for Emails': 'company_name_for_emails',
        'Email': 'email',
        'Email Status': 'email_status',
        'Primary Email Source': 'primary_email_source',
        'Email Confidence': 'email_confidence',
        'Primary Email Catch-all Status': 'primary_email_catch_all_status',
        'Primary Email Last Verified At': 'primary_email_last_verified_at',
        'Seniority': 'seniority',
        'Departments': 'departments',
        'Contact Owner': 'contact_owner',
        'Work Direct Phone': 'work_direct_phone',
        'Home Phone': 'home_phone',
        'Mobile Phone': 'mobile_phone',
        'Corporate Phone': 'corporate_phone',
        'Other Phone': 'other_phone',
        'Stage': 'stage',
        'Lists': 'lists',
        'Last Contacted': 'last_contacted',
        'Account Owner': 'account_owner',
        '# Employees': 'organization_employees',
        'Industry': 'organization_industry',
        'Keywords': 'keywords',
        'Person Linkedin Url': 'person_linkedin_url',
        'Website': 'website',
        'Company Linkedin Url': 'company_linkedin_url',
        'Facebook Url': 'facebook_url',
        'Twitter Url': 'twitter_url',
        'City': 'city',
        'State': 'state',
        'Country': 'country',
        'Company Address': 'company_address',
        'Company City': 'company_city',
        'Company State': 'company_state',
        'Company Country': 'company_country',
        'Company Phone': 'company_phone',
        'Technologies': 'technologies',
        'Annual Revenue': 'annual_revenue',
        'Total Funding': 'total_funding',
        'Latest Funding': 'latest_funding',
        'Latest Funding Amount': 'latest_funding_amount',
        'Last Raised At': 'last_raised_at',
        'Subsidiary of': 'subsidiary_of',
        'Email Sent': 'email_sent',
        'Email Open': 'email_open',
        'Email Bounced': 'email_bounced',
        'Replied': 'replied',
        'Demoed': 'demoed',
        'Number of Retail Locations': 'number_of_retail_locations',
        'Apollo Contact Id': 'apollo_contact_id',
        'Apollo Account Id': 'apollo_account_id',
        'Secondary Email': 'secondary_email',
        'Secondary Email Source': 'secondary_email_source',
        'Tertiary Email': 'tertiary_email',
        'Tertiary Email Source': 'tertiary_email_source',
        'Primary Intent Topic': 'primary_intent_topic',
        'Primary Intent Score': 'primary_intent_score',
        'Secondary Intent Topic': 'secondary_intent_topic',
        'Secondary Intent Score': 'secondary_intent_score'
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Add missing columns that might be expected by the model
    expected_columns = [
        'status_x', 'status_summary', 'payload', 'esp_code', 'page_retrieved',
        'lt_interest_status', 'verification_status', 'enrichment_status',
        'organization_founded_year', 'credits_consumed', 'auto_variant_select',
        'daily_limit', 'email_gap', 'status_y'
    ]
    
    for col in expected_columns:
        if col not in df.columns:
            df[col] = None
    
    print(f"✅ Data prepared. Shape: {df.shape}")
    return df

def prepare_new_data(df, artifacts):
    """Prepare new data for prediction using the same preprocessing as training."""
    print("🔧 Preparing new data for prediction...")
    
    # Apply the same feature engineering as training
    df = enhanced_text_preprocessing(df)
    df = advanced_timestamp_features(df)
    df = create_interaction_features(df)
    df = create_jsonb_features(df)
    df = create_xgboost_optimized_features(df)
    df = handle_outliers(df)
    
    # Get the feature names used in training
    feature_names = artifacts.get('feature_names', [])
    
    # Prepare features for model (same as training)
    X, _, _ = prepare_features_for_model(
        df, 
        target_variable='engagement_level',  # This won't exist in new data
        cols_to_drop=artifacts.get('config', {}).get('features', {}).get('cols_to_drop', [])
    )
    
    # Select only the features that were used in training
    available_features = [col for col in feature_names if col in X.columns]
    missing_features = [col for col in feature_names if col not in X.columns]
    
    if missing_features:
        print(f"⚠️ Warning: {len(missing_features)} features missing from new data")
        print(f"Missing features: {missing_features[:5]}...")  # Show first 5
    
    # Add missing features with default values
    for col in missing_features:
        X[col] = 0
    
    # Select only the features used in training
    X = X[feature_names]
    
    print(f"✅ Features prepared. Shape: {X.shape}")
    return X

def predict_new_leads(X, artifacts):
    """Make predictions on new data."""
    print("🤖 Making predictions...")
    
    model = artifacts['model']
    
    # Make predictions
    predictions = model.predict(X)
    prediction_probas = model.predict_proba(X)
    
    print(f"✅ Predictions made for {len(predictions)} leads")
    return predictions, prediction_probas

def add_prediction_columns(df, predictions, prediction_probas, artifacts):
    """Add prediction columns to the dataframe."""
    print("📊 Adding prediction columns...")
    
    # Create a copy to avoid modifying original
    df_with_predictions = df.copy()
    
    # Add prediction columns
    df_with_predictions['predicted_engagement_level'] = predictions
    
    # Add probability columns
    if len(artifacts['model'].classes_) == 2:
        # Binary classification
        df_with_predictions['prediction_confidence'] = prediction_probas[:, 1]
        df_with_predictions['prediction_probability_class_0'] = prediction_probas[:, 0]
        df_with_predictions['prediction_probability_class_1'] = prediction_probas[:, 1]
    else:
        # Multi-class classification
        for i, class_label in enumerate(artifacts['model'].classes_):
            df_with_predictions[f'prediction_probability_class_{class_label}'] = prediction_probas[:, i]
        df_with_predictions['prediction_confidence'] = np.max(prediction_probas, axis=1)
    
    # Add prediction metadata
    df_with_predictions['model_version'] = artifacts.get('model_version', 'unknown')
    df_with_predictions['prediction_timestamp'] = pd.Timestamp.now()
    
    # Add engagement level mapping
    engagement_mapping = {
        0: 'No Engagement',
        1: 'Opened',
        2: 'Clicked/Replied'
    }
    df_with_predictions['predicted_engagement_label'] = df_with_predictions['predicted_engagement_level'].map(engagement_mapping)
    
    print(f"✅ Prediction columns added")
    return df_with_predictions

def handle_missing_features(X, feature_names):
    """Handle missing features by adding them with default values."""
    missing_features = [col for col in feature_names if col not in X.columns]
    
    if missing_features:
        print(f"⚠️ Adding {len(missing_features)} missing features with default values")
        for col in missing_features:
            X[col] = 0
    
    return X

def save_predictions(df_with_predictions, config):
    """Save predictions to file."""
    print("💾 Saving predictions...")
    
    # Create timestamped filename
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"apollo_predictions_{timestamp}.csv"
    
    # Save to CSV
    df_with_predictions.to_csv(output_filename, index=False)
    print(f"✅ Predictions saved to: {output_filename}")
    
    # Print summary
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    print(f"📊 Total leads processed: {len(df_with_predictions)}")
    
    if 'predicted_engagement_level' in df_with_predictions.columns:
        engagement_counts = df_with_predictions['predicted_engagement_level'].value_counts().sort_index()
        print(f"🎯 Predicted engagement distribution:")
        for level, count in engagement_counts.items():
            percentage = count / len(df_with_predictions) * 100
            print(f"  Level {level}: {count} ({percentage:.1f}%)")
    
    if 'prediction_confidence' in df_with_predictions.columns:
        avg_confidence = df_with_predictions['prediction_confidence'].mean()
        print(f"📈 Average prediction confidence: {avg_confidence:.3f}")
    
    return output_filename

def main():
    """Main prediction function."""
    print("🚀 Starting Apollo Contacts Prediction Pipeline")
    print("="*60)
    
    try:
        # 1. Load configuration
        config = load_config()
        
        # 2. Load model artifacts
        artifacts = load_model_artifacts()
        
        # 3. Load Apollo data
        df = load_apollo_data(config)
        
        # 4. Prepare new data for prediction
        X = prepare_new_data(df, artifacts)
        
        # 5. Make predictions
        predictions, prediction_probas = predict_new_leads(X, artifacts)
        
        # 6. Add prediction columns
        df_with_predictions = add_prediction_columns(df, predictions, prediction_probas, artifacts)
        
        # 7. Save predictions
        output_file = save_predictions(df_with_predictions, config)
        
        print(f"\n🎉 Prediction pipeline completed successfully!")
        print(f"📁 Results saved to: {output_file}")
        
        # Show sample predictions
        print(f"\n📋 Sample predictions (first 5 rows):")
        sample_cols = ['first_name', 'last_name', 'email', 'company_name', 
                      'predicted_engagement_level', 'predicted_engagement_label', 'prediction_confidence']
        available_cols = [col for col in sample_cols if col in df_with_predictions.columns]
        print(df_with_predictions[available_cols].head())
        
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 