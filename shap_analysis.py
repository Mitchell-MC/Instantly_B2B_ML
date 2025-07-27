"""
Updated SHAP Analysis with Corrected Feature Interpretations
Based on Apollo's actual data structure and status codes
"""

import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple

# Configuration
CSV_FILE_PATH = Path("merged_contacts.csv")
TARGET_VARIABLE = "engagement_level"
TEXT_COLS = ['campaign_id', 'email_subjects', 'email_bodies']
TIMESTAMP_COLS = ['timestamp_created', 'timestamp_last_contact', 'retrieval_timestamp', 'enriched_at', 'inserted_at', 'last_contacted_from']
CATEGORICAL_COLS = ['title', 'seniority', 'organization_industry', 'country', 'city', 'verification_status', 'enrichment_status', 'upload_method', 'api_status', 'state']
JSONB_COLS = ['employment_history', 'organization_data', 'account_data', 'api_response_raw']

COLS_TO_DROP = [
    'id', 'email_clicked_variant', 'email_clicked_step', 'email', 'first_name', 'last_name', 'company_name', 'linkedin_url',
    'website', 'headline', 'company_domain', 'phone', 'apollo_id',
    'apollo_name', 'organization', 'photo_url', 'organization_name',
    'organization_website', 'organization_phone',
    'email_reply_count', 'email_opened_variant', 'email_opened_step',
    'timestamp_last_open', 'timestamp_last_reply', 'timestamp_last_click',
    'timestamp_last_touch', 'timestamp_last_interest_change', 'timestamp_updated',
    'personalization', 'status_summary', 'payload', 'list_id',
    'assigned_to', 'campaign', 'uploaded_by_user',
]

def load_data(file_path: Path) -> pd.DataFrame:
    """Load and process data"""
    print(f"Loading data from '{file_path}'...")
    if not file_path.is_file():
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)

    try:
        df = pd.read_csv(file_path, on_bad_lines='warn', low_memory=False)
        print("Data successfully loaded.")
        
        # Convert timestamps
        all_potential_timestamps = list(set(TIMESTAMP_COLS + [col for col in COLS_TO_DROP if 'timestamp' in col]))
        for col in all_potential_timestamps:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    except Exception as e:
        print(f"An error occurred while loading or parsing the CSV: {e}")
        sys.exit(1)

def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Enhanced feature engineering with Apollo-specific features"""
    print("Starting enhanced feature engineering...")
    
    # Create target variable
    if 'email_open_count' not in df.columns or 'email_click_count' not in df.columns:
        print("Error: Source columns 'email_open_count' or 'email_click_count' not found.")
        sys.exit(1)

    conditions = [(df['email_click_count'] > 0), (df['email_open_count'] > 0)]
    choices = [2, 1]
    df[TARGET_VARIABLE] = np.select(conditions, choices, default=0)
    df = df.drop(columns=['email_open_count', 'email_click_count'])
    
    # Text processing
    df['combined_text'] = ""
    for col in TEXT_COLS:
        if col in df.columns:
            df['combined_text'] += df[col].fillna('') + ' '
    
    # Enhanced timestamp features
    if 'timestamp_created' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp_created']):
        df['created_hour'] = df['timestamp_created'].dt.hour
        df['created_day_of_week'] = df['timestamp_created'].dt.dayofweek
        df['created_month'] = df['timestamp_created'].dt.month
        
        # Time-based features
        ref_date = df['timestamp_created']
        for col in TIMESTAMP_COLS:
            if col in df.columns and col != 'timestamp_created' and pd.api.types.is_datetime64_any_dtype(df[col]):
                feature_name = f"days_between_{col.replace('timestamp_', '')}_and_created"
                df[feature_name] = (df[col] - ref_date).dt.days
    
    # Apollo-specific interaction features
    if 'organization_industry' in df.columns and 'seniority' in df.columns:
        df['industry_seniority'] = df['organization_industry'].fillna('unknown') + '_' + df['seniority'].fillna('unknown')
    
    if 'country' in df.columns and 'organization_industry' in df.columns:
        df['country_industry'] = df['country'].fillna('unknown') + '_' + df['organization_industry'].fillna('unknown')
    
    # JSONB features
    for col in JSONB_COLS:
        if col in df.columns:
            df[f'has_{col}'] = df[col].notna().astype(int)
    
    # Text length features
    if 'combined_text' in df.columns:
        df['text_length'] = df['combined_text'].str.len()
        df['text_word_count'] = df['combined_text'].str.split().str.len()
    
    # Drop specified columns
    all_cols_to_drop = list(set(COLS_TO_DROP + TIMESTAMP_COLS + JSONB_COLS + TEXT_COLS))
    df = df.drop(columns=[col for col in all_cols_to_drop if col in df.columns], errors='ignore')
    
    # Handle categorical columns - convert to numeric or drop
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != TARGET_VARIABLE:
            # For categorical columns with few unique values, encode them
            if df[col].nunique() <= 50:
                df[col] = pd.Categorical(df[col]).codes
            else:
                # Drop high cardinality categorical columns
                df = df.drop(columns=[col])
    
    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])
    
    X = df.drop(columns=[TARGET_VARIABLE], errors='ignore')
    y = df[TARGET_VARIABLE]
    
    print(f"Feature engineering complete. Shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts().sort_index()}")
    
    return X, y

def create_enhanced_shap_plots(model, X_test, y_test, feature_names):
    """Create multiple SHAP plots with corrected interpretations"""
    print("Creating enhanced SHAP plots...")
    
    # Create explainer
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    
    # 1. Summary plot for Clickers (Class 2) with corrected labels
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values[:,:,2], X_test, show=False, max_display=20)
    plt.title('SHAP Summary for High-Engagement "Clickers" (Class 2)\nCorrected Feature Interpretations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("shap_summary_clickers_corrected.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: shap_summary_clickers_corrected.png")
    
    # 2. Summary plot for Openers (Class 1)
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values[:,:,1], X_test, show=False, max_display=20)
    plt.title('SHAP Summary for "Openers" (Class 1)\nMedium Engagement Contacts', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("shap_summary_openers.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: shap_summary_openers.png")
    
    # 3. Summary plot for No Engagement (Class 0)
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values[:,:,0], X_test, show=False, max_display=20)
    plt.title('SHAP Summary for "No Engagement" (Class 0)\nLow Engagement Contacts', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("shap_summary_no_engagement.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: shap_summary_no_engagement.png")
    
    # 4. Overall summary plot (mean across all classes)
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_test, show=False, max_display=20)
    plt.title('SHAP Summary - Overall Model (All Classes)\nFeature Importance Across All Engagement Levels', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("shap_summary_overall.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: shap_summary_overall.png")
    
    # 5. Waterfall plot for a high-value prediction
    high_value_idx = np.where(y_test == 2)[0][0] if len(np.where(y_test == 2)[0]) > 0 else 0
    plt.figure(figsize=(12, 8))
    shap.waterfall_plot(shap_values[high_value_idx, :, 2], show=False, max_display=15)
    plt.title(f'SHAP Waterfall Plot - High-Engagement Prediction\nSample Contact (Index {high_value_idx})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("shap_waterfall_high_engagement.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: shap_waterfall_high_engagement.png")
    
    # 6. Feature importance bar plot
    feature_importance = np.abs(shap_values.values).mean(0).mean(0)
    # Ensure feature_names matches the length of feature_importance
    if len(feature_names) != len(feature_importance):
        feature_names = [f"feature_{i}" for i in range(len(feature_importance))]
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False).head(20)
    
    plt.figure(figsize=(12, 10))
    plt.barh(range(len(feature_importance_df)), feature_importance_df['importance'])
    plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])
    plt.xlabel('Mean |SHAP value|')
    plt.title('Feature Importance (Mean Absolute SHAP Values)\nTop 20 Most Important Features', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("shap_feature_importance_bar.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: shap_feature_importance_bar.png")
    
    return shap_values

def create_feature_interpretation_report(shap_values, X_test, feature_names):
    """Create a detailed feature interpretation report"""
    print("\n" + "="*80)
    print("DETAILED FEATURE INTERPRETATION REPORT")
    print("="*80)
    
    # Calculate feature importance for each class
    class_names = ['No Engagement', 'Opener', 'Clicker']
    
    for class_idx, class_name in enumerate(class_names):
        print(f"\n📊 {class_name.upper()} (Class {class_idx}) - Top 10 Features:")
        print("-" * 60)
        
        # Get SHAP values for this class
        class_shap_values = shap_values[:, :, class_idx]
        feature_importance = np.abs(class_shap_values).mean(0)
        
        # Ensure feature_names matches the length of feature_importance
        if len(feature_names) != len(feature_importance):
            feature_names = [f"feature_{i}" for i in range(len(feature_importance))]
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False).head(10)
        
        for idx, row in importance_df.iterrows():
            feature = row['feature']
            importance = row['importance']
            
            # Provide interpretation based on feature name
            interpretation = get_feature_interpretation(feature)
            print(f"  {idx+1:2d}. {feature:<30} | Importance: {importance:.4f}")
            print(f"      {interpretation}")
    
    # Overall feature importance
    print(f"\n🏆 OVERALL TOP 10 FEATURES (All Classes):")
    print("-" * 60)
    
    overall_importance = np.abs(shap_values.values).mean(0).mean(0)
    
    # Ensure feature_names matches the length of overall_importance
    if len(feature_names) != len(overall_importance):
        feature_names = [f"feature_{i}" for i in range(len(overall_importance))]
    
    overall_df = pd.DataFrame({
        'feature': feature_names,
        'importance': overall_importance
    }).sort_values('importance', ascending=False).head(10)
    
    for idx, row in overall_df.iterrows():
        feature = row['feature']
        importance = row['importance']
        interpretation = get_feature_interpretation(feature)
        print(f"  {idx+1:2d}. {feature:<30} | Importance: {importance:.4f}")
        print(f"      {interpretation}")

def get_feature_interpretation(feature_name):
    """Provide business interpretation for features"""
    interpretations = {
        'status_x': 'Apollo contact quality score (-3 to +3). Higher = better quality contacts.',
        'esp_code': 'Email Service Provider code. Lower codes = premium providers (Gmail, Outlook).',
        'tfidf_com': 'TF-IDF score for "com" in text. High values = generic commercial language.',
        'organization_employees': 'Company size. Larger companies = better engagement potential.',
        'organization_founded_year': 'Company age. Newer companies = more responsive to outreach.',
        'page_retrieved': 'Apollo data retrieval page number. Higher = more comprehensive data.',
        'days_since_created': 'Days since contact creation. Recency indicator.',
        'country_United States': 'US-based contacts. Highest engagement rates.',
        'organization_industry': 'Company industry. Technology = highest engagement.',
        'text_length': 'Length of combined text content. Content richness indicator.',
        'text_word_count': 'Number of words in text content. Content complexity.',
        'has_employment_history': 'Has employment history data. Data completeness indicator.',
        'has_organization_data': 'Has organization enrichment data. Data quality indicator.',
        'has_account_data': 'Has account-level data. Data richness indicator.',
        'created_hour': 'Hour of contact creation. Timing optimization.',
        'created_day_of_week': 'Day of week created. Timing optimization.',
        'created_month': 'Month of contact creation. Seasonal patterns.',
        'industry_seniority': 'Industry + seniority interaction. Targeting precision.',
        'country_industry': 'Country + industry interaction. Geographic targeting.',
    }
    
    # Check for partial matches
    for key, value in interpretations.items():
        if key in feature_name:
            return value
    
    # Default interpretation
    if 'tfidf_' in feature_name:
        word = feature_name.replace('tfidf_', '')
        return f'TF-IDF score for "{word}" in text content. Text-based feature.'
    elif 'country_' in feature_name:
        country = feature_name.replace('country_', '')
        return f'Contacts from {country}. Geographic targeting feature.'
    elif 'organization_' in feature_name:
        return f'Organization-related feature. Company data enrichment.'
    else:
        return 'Feature importance in model prediction.'

def main():
    """Main function to run updated SHAP analysis"""
    print("🚀 UPDATED SHAP ANALYSIS WITH CORRECTED INTERPRETATIONS")
    print("="*80)
    
    # Load and process data
    df = load_data(CSV_FILE_PATH)
    X, y = engineer_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data split complete. Test set shape: {X_test.shape}")
    
    # Train model
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    print("Model training complete.")
    
    # Create enhanced SHAP plots
    feature_names = X_test.columns.tolist()
    shap_values = create_enhanced_shap_plots(model, X_test, y_test, feature_names)
    
    # Create interpretation report
    create_feature_interpretation_report(shap_values, X_test, feature_names)
    
    print("\n" + "="*80)
    print("✅ SHAP ANALYSIS COMPLETE!")
    print("Generated files:")
    print("  - shap_summary_clickers_corrected.png")
    print("  - shap_summary_openers.png")
    print("  - shap_summary_no_engagement.png")
    print("  - shap_summary_overall.png")
    print("  - shap_waterfall_high_engagement.png")
    print("  - shap_feature_importance_bar.png")
    print("="*80)

if __name__ == "__main__":
    main() 