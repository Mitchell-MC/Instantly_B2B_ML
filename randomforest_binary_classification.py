"""
RandomForest Binary Classification with Grid Search Optimization
Predicts email opening behavior using ensemble methods with comprehensive hyperparameter tuning.

Based on RDS XGBoost template but uses:
- Merged CSV data instead of RDS
- RandomForest instead of XGBoost
- Two-phase grid search methodology from shap_analysis.py
"""

import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
# import seaborn as sns  # Removed - not installed in environment
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Configuration
CSV_FILE_PATH = Path("merged_contacts.csv")
TARGET_VARIABLE = "opened"

# Columns to drop (leakage, identifiers, empty columns) - adapted from RDS template
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

def load_and_preprocess_data():
    """Load data from CSV and create binary target variable with enhanced feature engineering."""
    print("=== LOADING AND PREPROCESSING DATA FROM CSV ===")
    
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
    
    # Create binary target: 1 if opened (email_open_count > 0), 0 if not opened
    if 'email_open_count' not in df.columns:
        print("❌ Error: 'email_open_count' column not found.")
        sys.exit(1)
    
    df[TARGET_VARIABLE] = (df['email_open_count'] > 0).astype(int)
    
    # Check target distribution
    target_dist = df[TARGET_VARIABLE].value_counts().sort_index()
    print(f"\n📊 Binary target distribution:")
    print(f"  Not opened (0): {target_dist[0]:,} ({target_dist[0]/len(df)*100:.1f}%)")
    print(f"  Opened (1): {target_dist[1]:,} ({target_dist[1]/len(df)*100:.1f}%)")
    
    # Drop unnecessary columns
    df = df.drop(columns=[col for col in COLS_TO_DROP if col in df.columns], errors='ignore')
    
    # Enhanced text preprocessing
    df['combined_text'] = ""
    for col in TEXT_COLS:
        if col in df.columns:
            df['combined_text'] += df[col].fillna('') + ' '
    
    # Text-based features
    df['text_length'] = df['combined_text'].str.len()
    df['text_word_count'] = df['combined_text'].str.split().str.len()
    df['has_numbers'] = df['combined_text'].str.contains(r'\d', regex=True).astype(int)
    df['has_email'] = df['combined_text'].str.contains(r'@', regex=True).astype(int)
    
    # Engineer JSONB presence features
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
    y = df[TARGET_VARIABLE].copy()
    X = df.drop(columns=[TARGET_VARIABLE, 'combined_text'] + TEXT_COLS, errors='ignore')
    
    # Ensure all features are numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            # Convert to numeric, replacing non-numeric with NaN
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Drop any remaining non-numeric columns
    X = X.select_dtypes(include=[np.number])
    
    print(f"\n✅ Feature engineering complete. Final shape: {X.shape}")
    print(f"📋 Features include: {list(X.columns[:10])}...")
    
    return X, y

def perform_grid_search(X_train, y_train):
    """Perform two-phase grid search for optimal RandomForest hyperparameters (adapted from shap_analysis.py)"""
    print("\n🔍 Starting Grid Search CV for RandomForest Hyperparameter Optimization...")
    print("="*80)
    
    # Use StratifiedKFold for consistent validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Phase 1: Broad search
    print("\n📊 Phase 1: Broad Grid Search")
    print("-" * 50)
    
    broad_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.3]
    }
    
    rf_broad = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # Handle class imbalance
    )
    
    grid_broad = GridSearchCV(
        estimator=rf_broad,
        param_grid=broad_params,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    
    print("🔄 Fitting broad grid search...")
    grid_broad.fit(X_train, y_train)
    
    print(f"✅ Best broad score: {grid_broad.best_score_:.4f}")
    print(f"📋 Best broad params: {grid_broad.best_params_}")
    
    # Phase 2: Detailed search around best parameters
    print("\n🎯 Phase 2: Detailed Grid Search")
    print("-" * 50)
    
    best_broad = grid_broad.best_params_
    
    # Create detailed parameter grid around best broad parameters
    detailed_params = {}
    
    # n_estimators refinement
    base_estimators = best_broad['n_estimators']
    detailed_params['n_estimators'] = [
        max(50, base_estimators - 50), 
        base_estimators, 
        base_estimators + 50
    ]
    
    # max_depth refinement
    if best_broad['max_depth'] is not None:
        base_depth = best_broad['max_depth']
        detailed_params['max_depth'] = [
            max(5, base_depth - 5),
            base_depth,
            base_depth + 5,
            None
        ]
    else:
        detailed_params['max_depth'] = [15, 20, 25, None]
    
    # min_samples_split refinement
    base_split = best_broad['min_samples_split']
    detailed_params['min_samples_split'] = [
        max(2, base_split - 2),
        base_split,
        base_split + 2
    ]
    
    # min_samples_leaf refinement
    base_leaf = best_broad['min_samples_leaf']
    detailed_params['min_samples_leaf'] = [
        max(1, base_leaf - 1),
        base_leaf,
        base_leaf + 1
    ]
    
    # max_features refinement
    detailed_params['max_features'] = [
        best_broad['max_features'],
        'sqrt', 
        'log2'
    ]
    
    # Add bootstrap and criterion for fine-tuning
    detailed_params['bootstrap'] = [True, False]
    detailed_params['criterion'] = ['gini', 'entropy']
    
    rf_detailed = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    grid_detailed = GridSearchCV(
        estimator=rf_detailed,
        param_grid=detailed_params,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    
    print("🔄 Fitting detailed grid search...")
    grid_detailed.fit(X_train, y_train)
    
    print(f"✅ Best detailed score: {grid_detailed.best_score_:.4f}")
    print(f"📋 Best detailed params: {grid_detailed.best_params_}")
    
    # Final model with optimal parameters
    print("\n🏆 OPTIMAL HYPERPARAMETERS FOUND:")
    print("="*50)
    optimal_params = grid_detailed.best_params_
    for param, value in optimal_params.items():
        print(f"{param:<20}: {value}")
    
    print(f"\n⭐ Optimal CV Score: {grid_detailed.best_score_:.4f}")
    
    return grid_detailed.best_estimator_, optimal_params

def main():
    """Main function for RandomForest binary classification with grid search optimization."""
    print("=== RANDOMFOREST BINARY CLASSIFICATION: EMAIL OPENING PREDICTION ===")
    
    # 1. Load and preprocess data from CSV
    X, y = load_and_preprocess_data()
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"\n📊 Data split:")
    print(f"  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    
    # 3. Enhanced preprocessing
    print("\n=== APPLYING ENHANCED PREPROCESSING ===")
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Scale features (important for some RandomForest implementations)
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
    
    print(f"✅ Preprocessed data shape:")
    print(f"  Training: {X_train_selected.shape}")
    print(f"  Test: {X_test_selected.shape}")
    print(f"  Selected features: {len(selected_features)}")
    
    # 4. Perform two-phase grid search
    optimal_model, optimal_params = perform_grid_search(X_train_selected, y_train)
    
    # 5. Evaluate model
    print("\n=== MODEL EVALUATION ===")
    y_pred = optimal_model.predict(X_test_selected)
    y_pred_proba = optimal_model.predict_proba(X_test_selected)[:, 1]
    
    print(f"\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Opened', 'Opened']))
    print(f"🎯 Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"📈 ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # 6. Feature importance
    print("\n=== FEATURE IMPORTANCE ===")
    feature_importance = optimal_model.feature_importances_
    
    # Create feature importance dataframe
    importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=True)
    
    print(f"\n🔝 Top 10 most important features:")
    for i, (_, row) in enumerate(importance_df.tail(10).iterrows(), 1):
        print(f"  {i:2d}. {row['Feature']:<30}: {row['Importance']:.4f}")
    
    # 7. Create feature importance plot
    plt.figure(figsize=(12, 10))
    top_features = importance_df.tail(20)  # Show top 20 features
    plt.barh(range(len(top_features)), top_features['Importance'])
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Feature Importances - RandomForest Binary Classification\n(Opening vs Not Opening Prediction)')
    plt.tight_layout()
    plt.savefig('randomforest_binary_feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Feature importance plot saved as 'randomforest_binary_feature_importance.png'")
    plt.show()
    
    # 8. Create confusion matrix (using matplotlib instead of seaborn)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    
    # Create heatmap manually using matplotlib
    im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im)
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center", color="black")
    
    plt.title('Confusion Matrix - RandomForest Binary Classification\nOpening Prediction')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.xticks([0, 1], ['Not Opened', 'Opened'])
    plt.yticks([0, 1], ['Not Opened', 'Opened'])
    plt.tight_layout()
    plt.savefig('randomforest_binary_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"💾 Confusion matrix saved as 'randomforest_binary_confusion_matrix.png'")
    plt.show()
    
    # 9. Final summary
    print(f"\n" + "="*80)
    print("🎉 FINAL SUMMARY:")
    print("="*80)
    print(f"✅ Loaded data from merged CSV")
    print(f"✅ Binary classification: Opening vs Not Opening")
    print(f"✅ Two-phase grid search optimization completed")
    print(f"✅ Enhanced preprocessing with feature selection")
    print(f"✅ RandomForest with optimal hyperparameters")
    print(f"")
    print(f"🎯 Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"📈 Test ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"📊 Training samples: {X_train.shape[0]:,}")
    print(f"📊 Test samples: {X_test.shape[0]:,}")
    print(f"🔧 Final features: {X_train_selected.shape[1]}")
    print(f"⚙️  Optimal parameters: {len(optimal_params)} hyperparameters optimized")
    print("="*80)
    
    return optimal_model

if __name__ == "__main__":
    best_model = main()