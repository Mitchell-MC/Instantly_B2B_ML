"""
Enhanced RandomForest Binary Classification (Simplified Version)
Implements senior data scientist recommendations without SMOTE to avoid environment issues.

Target: Improve from 69.9% accuracy / 78.0% AUC to 75-78% accuracy / 82-85% AUC
"""

import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Configuration
CSV_FILE_PATH = Path("merged_contacts.csv")
TARGET_VARIABLE = "opened"

# SIGNIFICANTLY REDUCED COLS_TO_DROP - Keep more predictive features
COLS_TO_DROP = [
    # Personal identifiers only
    'id', 'email', 'first_name', 'last_name', 'company_name', 'linkedin_url',
    'website', 'headline', 'company_domain', 'phone', 'apollo_id',
    'apollo_name', 'organization', 'photo_url', 'organization_name',
    'organization_website', 'organization_phone',
    
    # DIRECT LEAKAGE COLUMNS - These directly encode the target outcome
    'email_open_count', 'email_reply_count', 'email_click_count',
    'email_opened_variant', 'email_opened_step', 'email_clicked_variant', 
    'email_clicked_step', 'email_replied_variant', 'email_replied_step',
    'timestamp_last_open', 'timestamp_last_reply', 'timestamp_last_click', 
    'timestamp_last_touch', 'timestamp_last_interest_change',
    
    # Keep most other features for better performance
]

# Expanded categorical columns for richer feature set
CATEGORICAL_COLS = ['title', 'seniority', 'organization_industry', 'country', 'city', 'state', 
                   'verification_status', 'enrichment_status', 'upload_method', 'api_status',
                   'esp_code', 'campaign_id', 'email_list']

# Text columns for enhanced text processing
TEXT_COLS = ['email_subjects', 'email_bodies', 'personalization']

# JSONB columns for feature engineering
JSONB_COLS = ['employment_history', 'organization_data', 'account_data', 'api_response_raw']

def create_advanced_features(df):
    """Create advanced feature engineering based on domain insights"""
    print("🔧 Creating advanced features...")
    
    # 1. Daily Limit Engineering (based on daily limit analysis insights)
    if 'daily_limit' in df.columns:
        # Create daily limit buckets based on performance analysis
        def categorize_daily_limit_performance(limit):
            if pd.isna(limit):
                return 0  # Unknown/poor performance
            elif limit <= 100:
                return 4  # High performance (3.958 open rate)
            elif limit <= 250:
                return 2  # Low-medium performance
            elif limit <= 350:
                return 3  # Medium performance
            elif limit <= 450:
                return 2  # Medium-low performance  
            elif limit <= 650:
                return 3  # Good performance
            else:
                return 4  # Very high performance (2.723 open rate)
        
        df['daily_limit_performance_score'] = df['daily_limit'].apply(categorize_daily_limit_performance)
        df['daily_limit_log'] = np.log1p(df['daily_limit'].fillna(0))
        df['daily_limit_squared'] = df['daily_limit'].fillna(0) ** 2
        df['is_high_daily_limit'] = (df['daily_limit'] >= 500).astype(int)
    
    # 2. Company Size Engineering
    if 'organization_employees' in df.columns:
        df['organization_employees_log'] = np.log1p(df['organization_employees'].fillna(0))
        df['company_size_category'] = pd.cut(df['organization_employees'].fillna(0), 
                                           bins=[0, 10, 50, 200, 1000, float('inf')], 
                                           labels=[1, 2, 3, 4, 5])
        df['is_enterprise'] = (df['organization_employees'] >= 1000).astype(int)
        df['is_small_company'] = (df['organization_employees'] <= 50).astype(int)
    
    # 3. Industry-Seniority Interactions
    if 'organization_industry' in df.columns and 'seniority' in df.columns:
        # Create high-value combinations
        df['is_tech_executive'] = ((df['organization_industry'].str.contains('Technology|Software|Tech', na=False)) & 
                                  (df['seniority'].str.contains('C-level|VP|Director', na=False))).astype(int)
        df['is_sales_professional'] = ((df['organization_industry'].str.contains('Sales|Marketing', na=False)) & 
                                      (df['seniority'].str.contains('Manager|Director', na=False))).astype(int)
    
    # 4. Geographic Features
    if 'country' in df.columns:
        # High-engagement countries (you can adjust based on your data)
        high_engagement_countries = ['United States', 'Canada', 'United Kingdom', 'Australia', 'Germany']
        df['is_high_engagement_country'] = df['country'].isin(high_engagement_countries).astype(int)
    
    # 5. ESP Code Performance (based on ESP analysis)
    if 'esp_code' in df.columns:
        # Based on the ESP analysis showing ESP 8.0 and 11.0 perform well
        high_performance_esp = [8.0, 11.0, 2.0]  # Top performing ESPs
        df['is_high_performance_esp'] = df['esp_code'].isin(high_performance_esp).astype(int)
        df['esp_code_performance_score'] = df['esp_code'].map({
            8.0: 5, 11.0: 4, 2.0: 3, 3.0: 2, 1.0: 2, 999.0: 1, 4.0: 1, 6.0: 1
        }).fillna(0)
    
    # 6. Temporal Features (enhanced)
    timestamp_cols = ['timestamp_created_x', 'timestamp_created_y', 'timestamp_last_contact']
    for col in timestamp_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[f'{col}_hour'] = df[col].dt.hour
            df[f'{col}_day_of_week'] = df[col].dt.dayofweek
            df[f'{col}_is_weekend'] = (df[col].dt.dayofweek >= 5).astype(int)
            df[f'{col}_is_business_hours'] = ((df[col].dt.hour >= 9) & (df[col].dt.hour <= 17)).astype(int)
            df[f'{col}_quarter'] = df[col].dt.quarter
    
    # 7. Text Feature Engineering
    df['combined_text'] = ""
    for col in TEXT_COLS:
        if col in df.columns:
            df['combined_text'] += df[col].fillna('') + ' '
    
    # Enhanced text features
    df['text_length'] = df['combined_text'].str.len()
    df['text_word_count'] = df['combined_text'].str.split().str.len()
    df['has_numbers'] = df['combined_text'].str.contains(r'\d', regex=True).astype(int)
    df['has_email'] = df['combined_text'].str.contains(r'@', regex=True).astype(int)
    df['has_phone'] = df['combined_text'].str.contains(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', regex=True).astype(int)
    df['has_url'] = df['combined_text'].str.contains(r'http|www\.', regex=True).astype(int)
    df['exclamation_count'] = df['combined_text'].str.count('!').fillna(0)
    df['question_count'] = df['combined_text'].str.count('\?').fillna(0)
    df['caps_ratio'] = (df['combined_text'].str.count(r'[A-Z]') / (df['text_length'] + 1)).fillna(0)
    
    # 8. Campaign and Technical Features
    technical_features = ['link_tracking', 'open_tracking', 'daily_limit', 'email_gap']
    for col in technical_features:
        if col in df.columns:
            df[f'has_{col}'] = df[col].notna().astype(int)
    
    # 9. JSONB presence features
    for col in JSONB_COLS:
        if col in df.columns:
            df[f'has_{col}'] = df[col].notna().astype(int)
    
    print(f"✅ Advanced feature engineering complete. New shape: {df.shape}")
    return df

def create_interaction_features(X, top_features=None):
    """Create polynomial and interaction features for top predictors"""
    print("🔄 Creating interaction features...")
    
    if top_features is None:
        # Select top numerical features for interactions
        numerical_cols = X.select_dtypes(include=[np.number]).columns[:8]  # Top 8 numerical features
    else:
        numerical_cols = [col for col in top_features[:8] if col in X.columns]
    
    # Create polynomial features (degree 2) for selected features
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    
    if len(numerical_cols) > 0:
        X_interactions = poly.fit_transform(X[numerical_cols])
        feature_names = poly.get_feature_names_out(numerical_cols)
        
        # Create DataFrame with interaction features
        X_poly = pd.DataFrame(X_interactions, columns=feature_names, index=X.index)
        
        # Combine with original features
        X_enhanced = pd.concat([X, X_poly.iloc[:, len(numerical_cols):]], axis=1)  # Exclude original features
        print(f"✅ Added {X_poly.shape[1] - len(numerical_cols)} interaction features")
        return X_enhanced
    
    return X

def load_and_preprocess_data():
    """Enhanced data loading and preprocessing"""
    print("=== ENHANCED DATA LOADING AND PREPROCESSING ===")
    
    # Load data from CSV
    if not CSV_FILE_PATH.exists():
        print(f"❌ Error: The file '{CSV_FILE_PATH}' was not found.")
        sys.exit(1)
    
    try:
        df = pd.read_csv(CSV_FILE_PATH, low_memory=False)
        print(f"✅ Data loaded from CSV. Shape: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        sys.exit(1)
    
    # Create binary target
    if 'email_open_count' not in df.columns:
        print("❌ Error: 'email_open_count' column not found.")
        sys.exit(1)
    
    df[TARGET_VARIABLE] = (df['email_open_count'] > 0).astype(int)
    
    # Check target distribution
    target_dist = df[TARGET_VARIABLE].value_counts().sort_index()
    print(f"\n📊 Binary target distribution:")
    print(f"  Not opened (0): {target_dist[0]:,} ({target_dist[0]/len(df)*100:.1f}%)")
    print(f"  Opened (1): {target_dist[1]:,} ({target_dist[1]/len(df)*100:.1f}%)")
    
    # Advanced feature engineering BEFORE dropping columns
    df = create_advanced_features(df)
    
    # Drop only essential leakage columns (keeping more predictive features)
    df = df.drop(columns=[col for col in COLS_TO_DROP if col in df.columns], errors='ignore')
    
    # Handle categorical columns with enhanced encoding
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)
            # For high-cardinality categoricals, use frequency encoding for some
            if df[col].nunique() > 50:
                freq_encoding = df[col].value_counts().to_dict()
                df[f'{col}_frequency'] = df[col].map(freq_encoding)
    
    # Label encoding for categorical columns
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # Drop timestamp columns to avoid conversion issues
    timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower() and col != TARGET_VARIABLE]
    df = df.drop(columns=timestamp_cols, errors='ignore')
    
    # Separate features and target
    y = df[TARGET_VARIABLE].copy()
    X = df.drop(columns=[TARGET_VARIABLE, 'combined_text'] + 
                        [col for col in TEXT_COLS if col in df.columns], errors='ignore')
    
    # Ensure all features are numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Drop any remaining non-numeric columns
    X = X.select_dtypes(include=[np.number])
    
    print(f"\n✅ Enhanced preprocessing complete. Final shape: {X.shape}")
    print(f"📋 Sample features: {list(X.columns[:15])}...")
    
    return X, y

def main():
    """Enhanced main function with all improvements (without SMOTE)"""
    print("=== ENHANCED RANDOMFOREST: EMAIL OPENING PREDICTION (SIMPLIFIED) ===")
    
    # 1. Load and preprocess data with advanced features
    X, y = load_and_preprocess_data()
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"\n📊 Data split:")
    print(f"  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    
    # 3. Enhanced preprocessing pipeline
    print("\n=== ENHANCED PREPROCESSING PIPELINE ===")
    
    # Handle any remaining NaN columns before imputation
    # Drop columns that are all NaN
    nan_cols = X_train.columns[X_train.isna().all()].tolist()
    if nan_cols:
        print(f"⚠️  Dropping {len(nan_cols)} all-NaN columns: {nan_cols[:5]}...")
        X_train = X_train.drop(columns=nan_cols)
        X_test = X_test.drop(columns=nan_cols)
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imputed_array = imputer.fit_transform(X_train)
    X_test_imputed_array = imputer.transform(X_test)
    
    # Create DataFrames with correct column names
    X_train_imputed = pd.DataFrame(
        X_train_imputed_array,
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_imputed = pd.DataFrame(
        X_test_imputed_array,
        columns=X_test.columns,
        index=X_test.index
    )
    
    # Create interaction features
    X_train_enhanced = create_interaction_features(X_train_imputed)
    X_test_enhanced = create_interaction_features(X_test_imputed)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enhanced)
    X_test_scaled = scaler.transform(X_test_enhanced)
    
    # Enhanced feature selection - use more features (40-50 instead of 13)
    target_features = min(50, X_train_scaled.shape[1])
    selector = SelectKBest(f_classif, k=target_features)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    # Get selected feature names
    selected_mask = selector.get_support()
    selected_features = [X_train_enhanced.columns[i] for i, selected in enumerate(selected_mask) if selected]
    
    print(f"✅ Enhanced preprocessing complete:")
    print(f"  Original features: {X_train.shape[1]}")
    print(f"  With interactions: {X_train_enhanced.shape[1]}")
    print(f"  Selected features: {len(selected_features)}")
    
    # 4. Create enhanced RandomForest model with optimal parameters
    print("\n🎯 Creating enhanced RandomForest model...")
    
    rf_model = RandomForestClassifier(
        n_estimators=350,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=False,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # Handle class imbalance
    )
    
    print("🔄 Fitting enhanced model...")
    rf_model.fit(X_train_selected, y_train)
    
    # 5. Evaluate model
    print("\n=== ENHANCED MODEL EVALUATION ===")
    y_pred = rf_model.predict(X_test_selected)
    y_pred_proba = rf_model.predict_proba(X_test_selected)[:, 1]
    
    print(f"\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Opened', 'Opened']))
    print(f"🎯 Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"📈 ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # 6. Feature importance
    print("\n=== ENHANCED FEATURE IMPORTANCE ===")
    feature_importance = rf_model.feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=True)
    
    print(f"\n🔝 Top 15 most important features:")
    for i, (_, row) in enumerate(importance_df.tail(15).iterrows(), 1):
        print(f"  {i:2d}. {row['Feature']:<35}: {row['Importance']:.4f}")
    
    # 7. Create enhanced visualizations
    plt.figure(figsize=(14, 10))
    top_features = importance_df.tail(20)
    plt.barh(range(len(top_features)), top_features['Importance'])
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Feature Importances - Enhanced RandomForest\n(Opening vs Not Opening Prediction)')
    plt.tight_layout()
    plt.savefig('enhanced_randomforest_simplified_feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Enhanced feature importance plot saved")
    # plt.show()  # Commented out to avoid GUI issues
    
    # 8. Enhanced confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center", color="black")
    
    plt.title('Enhanced Confusion Matrix - RandomForest\nOpening Prediction')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.xticks([0, 1], ['Not Opened', 'Opened'])
    plt.yticks([0, 1], ['Not Opened', 'Opened'])
    plt.tight_layout()
    plt.savefig('enhanced_randomforest_simplified_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"💾 Enhanced confusion matrix saved")
    # plt.show()  # Commented out to avoid GUI issues
    
    # 9. Performance comparison and insights
    baseline_accuracy = 0.6986
    baseline_auc = 0.7800
    improvement_accuracy = accuracy_score(y_test, y_pred) - baseline_accuracy
    improvement_auc = roc_auc_score(y_test, y_pred_proba) - baseline_auc
    
    print(f"\n" + "="*80)
    print("🚀 ENHANCED MODEL PERFORMANCE SUMMARY:")
    print("="*80)
    print(f"✅ Advanced feature engineering implemented")
    print(f"✅ Enhanced RandomForest with optimal parameters")
    print(f"✅ Expanded to {len(selected_features)} features (vs 13 baseline)")
    print(f"✅ Interaction features created")
    print(f"✅ Class-balanced training")
    print(f"")
    print(f"📊 PERFORMANCE METRICS:")
    print(f"  🎯 Test accuracy: {accuracy_score(y_test, y_pred):.4f} (+{improvement_accuracy:+.4f})")
    print(f"  📈 Test ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f} (+{improvement_auc:+.4f})")
    print(f"  📋 Training samples: {X_train.shape[0]:,}")
    print(f"  📋 Test samples: {X_test.shape[0]:,}")
    print(f"  🔧 Final features: {len(selected_features)}")
    print(f"")
    if improvement_auc > 0.02:
        print(f"🎉 SIGNIFICANT IMPROVEMENT ACHIEVED!")
        print(f"  Target was 82-85% AUC (vs baseline 78%)")
        print(f"  Achieved: {roc_auc_score(y_test, y_pred_proba)*100:.1f}% AUC")
    else:
        print(f"📈 PROGRESS MADE:")
        print(f"  Baseline: {baseline_auc*100:.1f}% AUC")
        print(f"  Enhanced: {roc_auc_score(y_test, y_pred_proba)*100:.1f}% AUC")
        print(f"  Next steps: Add ensemble methods + SMOTE for further gains")
    print("="*80)
    
    return rf_model

if __name__ == "__main__":
    enhanced_model = main()