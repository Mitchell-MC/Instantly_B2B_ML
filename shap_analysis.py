"""
Optimized SHAP Analysis with Grid Search CV and Binary Classification
Binary target: 0 = No Opens, 1 = Opens (any engagement)
"""

import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuration
CSV_FILE_PATH = Path("merged_contacts.csv")
TARGET_VARIABLE = "binary_engagement"
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
    'campaign_id', 'email_subjects', 'email_bodies'  # Remove text columns
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
    """Enhanced feature engineering without TfidfVectorizer"""
    print("Starting feature engineering...")
    
    # Create binary target variable: 0 = No Opens, 1 = Opens
    if 'email_open_count' not in df.columns:
        print("Error: Source column 'email_open_count' not found.")
        sys.exit(1)

    # Binary classification: 0 = no opens, 1 = any opens
    df[TARGET_VARIABLE] = (df['email_open_count'] > 0).astype(int)
    df = df.drop(columns=['email_open_count', 'email_click_count'], errors='ignore')
    
    # Enhanced timestamp features
    if 'timestamp_created' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp_created']):
        df['created_hour'] = df['timestamp_created'].dt.hour
        df['created_day_of_week'] = df['timestamp_created'].dt.dayofweek
        df['created_month'] = df['timestamp_created'].dt.month
        df['created_quarter'] = df['timestamp_created'].dt.quarter
        df['is_weekend'] = (df['timestamp_created'].dt.dayofweek >= 5).astype(int)
        df['is_business_hours'] = ((df['timestamp_created'].dt.hour >= 9) & 
                                  (df['timestamp_created'].dt.hour <= 17)).astype(int)
        
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
    
    # Organization features
    if 'organization_employees' in df.columns:
        df['org_size_category'] = pd.cut(df['organization_employees'], 
                                        bins=[0, 10, 50, 200, 1000, float('inf')],
                                        labels=[0, 1, 2, 3, 4]).astype(float)
    
    if 'organization_founded_year' in df.columns:
        current_year = 2024
        df['company_age'] = current_year - df['organization_founded_year']
        df['is_startup'] = (df['company_age'] <= 5).astype(int)
        df['is_mature_company'] = (df['company_age'] >= 20).astype(int)
    
    # Drop specified columns
    all_cols_to_drop = list(set(COLS_TO_DROP + TIMESTAMP_COLS + JSONB_COLS))
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
    
    # Extract target and features
    X = df.drop(columns=[TARGET_VARIABLE], errors='ignore')
    y = df[TARGET_VARIABLE]
    
    print(f"Features before cleaning: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Remove columns with all NaN values or constant values
    # This prevents the imputer from dropping columns
    nan_cols = X.columns[X.isnull().all()].tolist()
    constant_cols = X.columns[X.nunique() <= 1].tolist()
    cols_to_remove = list(set(nan_cols + constant_cols))
    
    if cols_to_remove:
        print(f"Removing columns with all NaN or constant values: {cols_to_remove}")
        X = X.drop(columns=cols_to_remove)
    
    print(f"Features after cleaning: {X.shape}")
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    
    # Store original column names and index
    original_columns = X.columns.tolist()
    original_index = X.index
    
    # Impute missing values
    X_imputed = imputer.fit_transform(X)
    
    print(f"Imputed array shape: {X_imputed.shape}")
    print(f"Cleaned columns count: {len(original_columns)}")
    
    # Recreate DataFrame with proper alignment
    X = pd.DataFrame(X_imputed, columns=original_columns, index=original_index)
    
    print(f"Feature engineering complete. Shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts().sort_index()}")
    print(f"Class balance: {y.mean():.3f} positive rate")
    
    return X, y

def perform_grid_search(X_train, y_train):
    """Perform broad then detailed grid search for optimal hyperparameters"""
    print("\n🔍 Starting Grid Search CV for Hyperparameter Optimization...")
    print("="*80)
    
    # Use StratifiedKFold for consistent validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Phase 1: Broad search
    print("\n📊 Phase 1: Broad Grid Search")
    print("-" * 50)
    
    broad_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    xgb_broad = xgb.XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    
    grid_broad = GridSearchCV(
        estimator=xgb_broad,
        param_grid=broad_params,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    
    print("Fitting broad grid search...")
    grid_broad.fit(X_train, y_train)
    
    print(f"Best broad score: {grid_broad.best_score_:.4f}")
    print(f"Best broad params: {grid_broad.best_params_}")
    
    # Phase 2: Detailed search around best parameters
    print("\n🎯 Phase 2: Detailed Grid Search")
    print("-" * 50)
    
    best_broad = grid_broad.best_params_
    
    # Create detailed parameter grid around best broad parameters
    detailed_params = {
        'n_estimators': [max(50, best_broad['n_estimators'] - 50), 
                        best_broad['n_estimators'], 
                        best_broad['n_estimators'] + 50],
        'max_depth': [max(1, best_broad['max_depth'] - 1), 
                     best_broad['max_depth'], 
                     best_broad['max_depth'] + 1],
        'learning_rate': [max(0.01, best_broad['learning_rate'] - 0.05), 
                         best_broad['learning_rate'], 
                         min(0.3, best_broad['learning_rate'] + 0.05)],
        'subsample': [max(0.6, best_broad['subsample'] - 0.1), 
                     best_broad['subsample'], 
                     min(1.0, best_broad['subsample'] + 0.1)],
        'colsample_bytree': [max(0.6, best_broad['colsample_bytree'] - 0.1), 
                            best_broad['colsample_bytree'], 
                            min(1.0, best_broad['colsample_bytree'] + 0.1)],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2]
    }
    
    xgb_detailed = xgb.XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    
    grid_detailed = GridSearchCV(
        estimator=xgb_detailed,
        param_grid=detailed_params,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    
    print("Fitting detailed grid search...")
    grid_detailed.fit(X_train, y_train)
    
    print(f"Best detailed score: {grid_detailed.best_score_:.4f}")
    print(f"Best detailed params: {grid_detailed.best_params_}")
    
    # Final model with optimal parameters
    print("\n🏆 OPTIMAL HYPERPARAMETERS FOUND:")
    print("="*50)
    optimal_params = grid_detailed.best_params_
    for param, value in optimal_params.items():
        print(f"{param:<20}: {value}")
    
    print(f"\nOptimal CV Score: {grid_detailed.best_score_:.4f}")
    
    return grid_detailed.best_estimator_, optimal_params

def create_binary_shap_plots(model, X_test, y_test, feature_names):
    """Create SHAP plots for binary classification"""
    print("\n📊 Creating SHAP Analysis for Binary Classification...")
    print("="*60)
    
    # Create explainer
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    
    # For binary classification, SHAP values are for the positive class (opens)
    if hasattr(shap_values, 'values'):
        shap_vals = shap_values.values
    else:
        shap_vals = shap_values
    
    # 1. Summary plot for binary engagement
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_vals, X_test, show=False, max_display=20)
    plt.title('SHAP Summary Plot - Binary Engagement (Open vs No Open)\nFeature Impact on Email Opens', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("shap_summary_binary_engagement.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: shap_summary_binary_engagement.png")
    
    # 2. Feature importance bar plot
    if hasattr(shap_values, 'values'):
        feature_importance = np.abs(shap_values.values).mean(0)
    else:
        feature_importance = np.abs(shap_values).mean(0)
    
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
    plt.title('Feature Importance (Mean Absolute SHAP Values)\nTop 20 Most Important Features for Email Opens', 
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("shap_feature_importance_binary.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: shap_feature_importance_binary.png")
    
    # 3. Waterfall plot for a positive prediction
    positive_idx = np.where(y_test == 1)[0]
    if len(positive_idx) > 0:
        sample_idx = positive_idx[0]
        plt.figure(figsize=(12, 8))
        if hasattr(shap_values, 'values'):
            shap.waterfall_plot(shap_values[sample_idx], show=False, max_display=15)
        else:
            # Create a mock waterfall-style visualization
            sample_shap = shap_vals[sample_idx]
            sorted_indices = np.argsort(np.abs(sample_shap))[-15:]
            
            plt.barh(range(len(sorted_indices)), sample_shap[sorted_indices])
            plt.yticks(range(len(sorted_indices)), [feature_names[i] for i in sorted_indices])
            plt.xlabel('SHAP value')
            plt.title(f'SHAP Values for Sample {sample_idx} (Positive Prediction)')
            plt.grid(True, alpha=0.3)
        
        plt.title(f'SHAP Waterfall Plot - Positive Prediction\nSample Contact (Index {sample_idx})', 
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig("shap_waterfall_binary_opener.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved: shap_waterfall_binary_opener.png")
    
    return shap_values

def create_feature_interpretation_report(shap_values, X_test, feature_names, optimal_params):
    """Create a detailed feature interpretation report for binary classification"""
    print("\n" + "="*80)
    print("BINARY CLASSIFICATION FEATURE INTERPRETATION REPORT")
    print("="*80)
    
    # Calculate feature importance
    if hasattr(shap_values, 'values'):
        feature_importance = np.abs(shap_values.values).mean(0)
    else:
        feature_importance = np.abs(shap_values).mean(0)
    
    # Ensure feature_names matches the length of feature_importance
    if len(feature_names) != len(feature_importance):
        feature_names = [f"feature_{i}" for i in range(len(feature_importance))]
    
    # Create feature importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False).head(15)
    
    print(f"\n🏆 TOP 15 FEATURES FOR EMAIL OPENS:")
    print("-" * 60)
    
    for idx, row in importance_df.iterrows():
        feature = row['feature']
        importance = row['importance']
        
        # Provide interpretation based on feature name
        interpretation = get_feature_interpretation(feature)
        print(f"  {idx+1:2d}. {feature:<30} | Importance: {importance:.4f}")
        print(f"      💡 {interpretation}")
    
    print(f"\n🎯 OPTIMAL HYPERPARAMETERS:")
    print("-" * 60)
    for param, value in optimal_params.items():
        print(f"  {param:<20}: {value}")

def get_feature_interpretation(feature_name):
    """Provide business interpretation for features"""
    interpretations = {
        'status_x': 'Apollo contact quality score (-3 to +3). Higher = better quality contacts.',
        'esp_code': 'Corporate Email Engagement Tiers. Code 2 (High-tier corps: Nokia, Microsoft) = best, Code 1 (Tech corps: Google employees) = good, Code 999 (Large enterprises: Cisco, Amazon) = poor due to email overload.',
        'organization_employees': 'Company size. Larger companies = better engagement potential.',
        'organization_founded_year': 'Company age. Newer companies = more responsive to outreach.',
        'page_retrieved': 'Apollo data retrieval page number. Higher = more comprehensive data.',
        'country': 'Geographic targeting. US-based contacts typically have higher engagement.',
        'seniority': 'Contact seniority level. C-level and decision makers engage more.',
        'organization_industry': 'Company industry. Technology sector shows highest engagement.',
        'has_employment_history': 'Employment history data availability. More data = better targeting.',
        'has_organization_data': 'Organization enrichment data quality indicator.',
        'has_account_data': 'Account-level data richness for better personalization.',
        'created_hour': 'Hour of contact creation. Timing optimization for campaigns.',
        'created_day_of_week': 'Day of week created. Weekday creation shows better performance.',
        'created_month': 'Month of contact creation. Seasonal engagement patterns.',
        'is_weekend': 'Weekend creation indicator. Weekday contacts perform better.',
        'is_business_hours': 'Business hours creation (9-5). Better engagement timing.',
        'company_age': 'Years since company founding. Mature companies more responsive.',
        'is_startup': 'Startup company indicator (≤5 years). High growth potential.',
        'is_mature_company': 'Established company (≥20 years). Stable engagement patterns.',
        'org_size_category': 'Company size category (0-4). Mid-size companies optimal.'
    }
    
    # Check for exact matches first
    if feature_name in interpretations:
        return interpretations[feature_name]
    
    # Check for partial matches
    for key, value in interpretations.items():
        if key in feature_name:
            return value
    
    # Default interpretation based on patterns
    if 'country_' in feature_name:
        country = feature_name.replace('country_', '')
        return f'Contacts from {country}. Geographic targeting optimization.'
    elif 'organization_' in feature_name:
        return f'Organization-related feature for company-based targeting.'
    elif 'days_between' in feature_name:
        return f'Time-based feature measuring recency and timing optimization.'
    else:
        return 'Feature contributes to email open prediction accuracy.'

def main():
    """Main function to run optimized SHAP analysis with grid search"""
    print("🚀 OPTIMIZED BINARY SHAP ANALYSIS WITH GRID SEARCH CV")
    print("="*80)
    
    # Load and process data
    df = load_data(CSV_FILE_PATH)
    X, y = engineer_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nData split complete:")
    print(f"  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    print(f"  Training target distribution: {y_train.value_counts().to_dict()}")
    print(f"  Test target distribution: {y_test.value_counts().to_dict()}")
    
    # Perform grid search for optimal hyperparameters
    optimal_model, optimal_params = perform_grid_search(X_train, y_train)
    
    # Evaluate final model
    print("\n📈 Final Model Evaluation:")
    print("-" * 50)
    y_pred = optimal_model.predict(X_test)
    y_pred_proba = optimal_model.predict_proba(X_test)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    from sklearn.metrics import roc_auc_score
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC-AUC Score: {auc_score:.4f}")
    
    # Create SHAP analysis
    feature_names = X_test.columns.tolist()
    shap_values = create_binary_shap_plots(optimal_model, X_test, y_test, feature_names)
    
    # Create interpretation report
    create_feature_interpretation_report(shap_values, X_test, feature_names, optimal_params)
    
    print("\n" + "="*80)
    print("✅ OPTIMIZED BINARY SHAP ANALYSIS COMPLETE!")
    print("Generated files:")
    print("  - shap_summary_binary_engagement.png")
    print("  - shap_feature_importance_binary.png")
    print("  - shap_waterfall_binary_opener.png")
    print("\n🎯 Key Results:")
    print(f"  - Optimal ROC-AUC: {auc_score:.4f}")
    print(f"  - Binary Classification: No Opens (0) vs Opens (1)")
    print(f"  - Hyperparameters optimized via Grid Search CV")
    print("="*80)
    
    return optimal_model, optimal_params

if __name__ == "__main__":
    model, params = main() 