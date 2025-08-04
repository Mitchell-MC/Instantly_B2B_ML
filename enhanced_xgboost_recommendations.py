"""
Enhanced XGBoost Model Recommendations
Target: Improve from 69.86% accuracy / 78.0% AUC to 75-80% accuracy / 82-85% AUC

Key Improvements:
1. Advanced Feature Engineering (Domain-Specific)
2. XGBoost-Specific Hyperparameter Optimization
3. Advanced Class Imbalance Handling
4. Early Stopping & Cross-Validation
5. Feature Selection Optimization
6. Ensemble Methods
"""

import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Configuration
CSV_FILE_PATH = Path("merged_contacts.csv")
TARGET_VARIABLE = "opened"

# RECOMMENDATION 1: KEEP MORE PREDICTIVE FEATURES
# Your current model drops too many potentially valuable features
COLS_TO_DROP = [
    # Personal identifiers only
    'id', 'email', 'first_name', 'last_name', 'company_name', 'linkedin_url',
    'website', 'headline', 'company_domain', 'phone', 'apollo_id',
    'apollo_name', 'organization', 'photo_url', 'organization_name',
    'organization_website', 'organization_phone',
    
    # DIRECT LEAKAGE COLUMNS ONLY
    'email_open_count', 'email_reply_count', 'email_click_count',
    'email_opened_variant', 'email_opened_step', 'email_clicked_variant', 
    'email_clicked_step', 'email_replied_variant', 'email_replied_step',
    'timestamp_last_open', 'timestamp_last_reply', 'timestamp_last_click', 
    'timestamp_last_touch', 'timestamp_last_interest_change',
    
    # Keep campaign and technical features - they're predictive!
    # 'daily_limit', 'esp_code', 'campaign_id', 'email_list', etc. - KEEP THESE
]

def create_xgboost_specific_features(df):
    """
    RECOMMENDATION 2: ADVANCED FEATURE ENGINEERING FOR XGBOOST
    XGBoost handles these types of features very well
    """
    print("🔧 Creating XGBoost-optimized features...")
    
    # 1. DAILY LIMIT ENGINEERING (Your top performer according to analysis)
    if 'daily_limit' in df.columns:
        # Based on your daily limit analysis - these buckets perform differently
        def daily_limit_performance_encoding(limit):
            if pd.isna(limit):
                return 0
            elif limit <= 100:
                return 5  # Best performance
            elif limit <= 250:
                return 2  # Lower performance
            elif limit <= 350:
                return 3  # Medium 
            elif limit <= 450:
                return 2  # Medium-low
            elif limit <= 650:
                return 4  # Good
            else:
                return 4  # High performance
        
        df['daily_limit_performance'] = df['daily_limit'].apply(daily_limit_performance_encoding)
        df['daily_limit_log'] = np.log1p(df['daily_limit'].fillna(0))
        df['daily_limit_squared'] = df['daily_limit'].fillna(0) ** 2
        df['daily_limit_is_optimal'] = ((df['daily_limit'] >= 100) & (df['daily_limit'] <= 150)).astype(int)
        df['daily_limit_is_high'] = (df['daily_limit'] >= 500).astype(int)
        
        # Daily limit quantile features (XGBoost loves these)
        df['daily_limit_quantile'] = pd.qcut(df['daily_limit'].fillna(-1), 
                                           q=10, labels=False, duplicates='drop')
    
    # 2. COMPANY SIZE OPTIMIZATION (Your #1 feature: organization_employees)
    if 'organization_employees' in df.columns:
        df['employees_log'] = np.log1p(df['organization_employees'].fillna(0))
        df['employees_sqrt'] = np.sqrt(df['organization_employees'].fillna(0))
        df['employees_quantile'] = pd.qcut(df['organization_employees'].fillna(-1), 
                                         q=20, labels=False, duplicates='drop')
        
        # Size category interactions
        df['is_enterprise'] = (df['organization_employees'] >= 1000).astype(int)
        df['is_mid_market'] = ((df['organization_employees'] >= 200) & 
                              (df['organization_employees'] < 1000)).astype(int)
        df['is_smb'] = (df['organization_employees'] < 200).astype(int)
    
    # 3. GEOGRAPHIC FEATURES (country, state, city are top features)
    location_cols = ['country', 'state', 'city']
    for col in location_cols:
        if col in df.columns:
            # Frequency encoding (very effective for XGBoost)
            freq_map = df[col].value_counts().to_dict()
            df[f'{col}_frequency'] = df[col].map(freq_map).fillna(0)
            
            # Target encoding approximation (use frequency as proxy)
            df[f'{col}_encoded'] = df[f'{col}_frequency'] / df[f'{col}_frequency'].max()
    
    # 4. INDUSTRY-TITLE INTERACTIONS (Your top features)
    if 'organization_industry' in df.columns and 'title' in df.columns:
        # Create high-value combinations
        df['industry_title_combo'] = (df['organization_industry'].astype(str) + '_' + 
                                     df['title'].astype(str))
        
        # Frequency encode the combinations
        combo_freq = df['industry_title_combo'].value_counts().to_dict()
        df['industry_title_frequency'] = df['industry_title_combo'].map(combo_freq).fillna(0)
        
        # High-performing combinations
        tech_titles = ['CEO', 'CTO', 'VP', 'Director', 'Manager']
        df['is_tech_leadership'] = ((df['organization_industry'].str.contains('Technology|Software|Tech', na=False)) &
                                   (df['title'].str.contains('|'.join(tech_titles), na=False))).astype(int)
    
    # 5. ESP CODE OPTIMIZATION (Based on ESP analysis)
    if 'esp_code' in df.columns:
        # From your ESP analysis: ESP 8.0, 11.0, 2.0 perform best
        high_performance_esp = [8.0, 11.0, 2.0]
        df['esp_is_high_performance'] = df['esp_code'].isin(high_performance_esp).astype(int)
        
        # ESP performance scoring
        esp_performance_map = {
            8.0: 5, 11.0: 4, 2.0: 3, 3.0: 2, 1.0: 2, 
            999.0: 1, 4.0: 1, 6.0: 1, 7.0: 1
        }
        df['esp_performance_score'] = df['esp_code'].map(esp_performance_map).fillna(0)
    
    # 6. TEMPORAL FEATURES (Enhanced)
    timestamp_cols = ['timestamp_created_x', 'timestamp_created_y', 'timestamp_last_contact']
    for col in timestamp_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[f'{col}_hour'] = df[col].dt.hour
            df[f'{col}_day_of_week'] = df[col].dt.dayofweek
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_quarter'] = df[col].dt.quarter
            df[f'{col}_is_weekend'] = (df[col].dt.dayofweek >= 5).astype(int)
            df[f'{col}_is_business_hours'] = ((df[col].dt.hour >= 9) & 
                                            (df[col].dt.hour <= 17)).astype(int)
    
    # 7. CAMPAIGN FEATURES (Keep these - they're predictive!)
    campaign_cols = ['campaign_id', 'email_list', 'upload_method']
    for col in campaign_cols:
        if col in df.columns:
            freq_map = df[col].value_counts().to_dict()
            df[f'{col}_frequency'] = df[col].map(freq_map).fillna(0)
    
    # 8. TECHNICAL FEATURES
    technical_cols = ['link_tracking', 'open_tracking', 'email_gap']
    for col in technical_cols:
        if col in df.columns:
            df[f'has_{col}'] = df[col].notna().astype(int)
            if df[col].dtype in ['int64', 'float64']:
                df[f'{col}_log'] = np.log1p(df[col].fillna(0))
    
    print(f"✅ XGBoost feature engineering complete. Shape: {df.shape}")
    return df

def perform_enhanced_xgboost_grid_search(X_train, y_train):
    """
    RECOMMENDATION 3: XGBOOST-SPECIFIC HYPERPARAMETER OPTIMIZATION
    More aggressive parameters for better performance
    """
    print("\n🎯 Enhanced XGBoost Grid Search (3-Phase Optimization)")
    print("="*80)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Phase 1: Broad search with XGBoost-specific parameters
    print("\n📊 Phase 1: Broad XGBoost Search")
    print("-" * 50)
    
    broad_params = {
        'n_estimators': [200, 400, 600],  # More trees
        'max_depth': [6, 8, 10, 12],      # Deeper trees for complex patterns
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2],           # Minimum split loss
        'min_child_weight': [1, 3, 5]     # Minimum sum of weights
    }
    
    xgb_broad = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=1.2  # Handle slight class imbalance
    )
    
    grid_broad = GridSearchCV(
        estimator=xgb_broad,
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
    
    detailed_params = {
        'n_estimators': [max(100, best_broad['n_estimators'] - 100), 
                        best_broad['n_estimators'], 
                        best_broad['n_estimators'] + 100],
        'max_depth': [max(3, best_broad['max_depth'] - 1), 
                     best_broad['max_depth'], 
                     min(15, best_broad['max_depth'] + 1)],
        'learning_rate': [max(0.01, best_broad['learning_rate'] - 0.02), 
                         best_broad['learning_rate'], 
                         min(0.3, best_broad['learning_rate'] + 0.02)],
        'subsample': [max(0.7, best_broad['subsample'] - 0.05), 
                     best_broad['subsample'], 
                     min(1.0, best_broad['subsample'] + 0.05)],
        'colsample_bytree': [max(0.7, best_broad['colsample_bytree'] - 0.05), 
                            best_broad['colsample_bytree'], 
                            min(1.0, best_broad['colsample_bytree'] + 0.05)],
        'reg_alpha': [0, 0.1, 0.3],       # L1 regularization
        'reg_lambda': [1, 1.5, 2.0],      # L2 regularization
        'gamma': [max(0, best_broad['gamma'] - 0.05),
                 best_broad['gamma'],
                 best_broad['gamma'] + 0.1],
        'min_child_weight': [max(1, best_broad['min_child_weight'] - 1),
                           best_broad['min_child_weight'],
                           best_broad['min_child_weight'] + 1]
    }
    
    xgb_detailed = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=1.2
    )
    
    grid_detailed = GridSearchCV(
        estimator=xgb_detailed,
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
    
    # Phase 3: Advanced regularization fine-tuning
    print("\n🔬 Phase 3: Advanced Regularization")
    print("-" * 50)
    
    best_detailed = grid_detailed.best_params_
    
    advanced_params = {
        'reg_alpha': [0, 0.1, 0.3, 0.5, 1.0],
        'reg_lambda': [0.5, 1.0, 1.5, 2.0, 3.0],
        'scale_pos_weight': [1.0, 1.1, 1.2, 1.3]  # Class balance optimization
    }
    
    # Use best detailed params as base
    base_params = {k: v for k, v in best_detailed.items() if k not in advanced_params}
    
    xgb_advanced = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        **base_params
    )
    
    grid_advanced = GridSearchCV(
        estimator=xgb_advanced,
        param_grid=advanced_params,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    
    print("🔄 Fitting advanced grid search...")
    grid_advanced.fit(X_train, y_train)
    
    print(f"✅ Best advanced score: {grid_advanced.best_score_:.4f}")
    print(f"📋 Best advanced params: {grid_advanced.best_params_}")
    
    # Combine all best parameters
    final_params = {**base_params, **grid_advanced.best_params_}
    
    print(f"\n🏆 OPTIMAL XGBOOST PARAMETERS:")
    print("="*50)
    for param, value in final_params.items():
        print(f"{param:<20}: {value}")
    print(f"⭐ Optimal CV Score: {grid_advanced.best_score_:.4f}")
    
    return final_params, grid_advanced.best_score_

def create_enhanced_feature_selection(X, y, n_features=60):
    """
    RECOMMENDATION 4: ADVANCED FEATURE SELECTION
    Use multiple methods to select the best features
    """
    print(f"\n🎯 Enhanced Feature Selection (Target: {n_features} features)")
    print("-" * 50)
    
    # Method 1: Statistical selection
    selector_stats = SelectKBest(f_classif, k=min(80, X.shape[1]))
    X_stats = selector_stats.fit_transform(X, y)
    stats_features = selector_stats.get_support()
    
    # Method 2: XGBoost-based RFE
    xgb_selector = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rfe = RFE(estimator=xgb_selector, n_features_to_select=min(60, X.shape[1]))
    X_rfe = rfe.fit_transform(X, y)
    rfe_features = rfe.support_
    
    # Combine both methods
    combined_features = stats_features | rfe_features
    selected_indices = np.where(combined_features)[0]
    
    print(f"✅ Statistical method selected: {stats_features.sum()} features")
    print(f"✅ RFE method selected: {rfe_features.sum()} features") 
    print(f"✅ Combined selection: {len(selected_indices)} features")
    
    return selected_indices

def create_xgboost_ensemble(best_params, X_train, y_train):
    """
    RECOMMENDATION 5: ENSEMBLE METHOD
    Combine XGBoost with other algorithms for better performance
    """
    print("\n🎯 Creating Enhanced Ensemble")
    print("-" * 40)
    
    # Primary XGBoost model with optimal parameters
    xgb_model = xgb.XGBClassifier(
        **best_params,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    
    # Secondary XGBoost with different parameters for diversity
    xgb_diverse = xgb.XGBClassifier(
        n_estimators=best_params.get('n_estimators', 400) + 200,
        max_depth=min(15, best_params.get('max_depth', 8) + 2),
        learning_rate=max(0.01, best_params.get('learning_rate', 0.1) - 0.02),
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=123,  # Different seed for diversity
        n_jobs=-1
    )
    
    # Logistic Regression for ensemble diversity
    log_reg = LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=1000,
        n_jobs=-1
    )
    
    # Create voting ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('xgb_optimal', xgb_model),
            ('xgb_diverse', xgb_diverse),
            ('logreg', log_reg)
        ],
        voting='soft',  # Use probabilities
        n_jobs=-1
    )
    
    print("✅ Ensemble created with 3 models")
    return ensemble

def main():
    """
    ENHANCED XGBOOST PIPELINE
    Implements all recommendations for maximum performance improvement
    """
    print("=== ENHANCED XGBOOST: EMAIL OPENING PREDICTION ===")
    print("Target: 75-80% accuracy, 82-85% AUC")
    print("="*60)
    
    # Load data
    if not CSV_FILE_PATH.exists():
        print(f"❌ Error: '{CSV_FILE_PATH}' not found.")
        return
    
    df = pd.read_csv(CSV_FILE_PATH, low_memory=False)
    print(f"✅ Data loaded. Shape: {df.shape}")
    
    # Create target
    df[TARGET_VARIABLE] = (df['email_open_count'] > 0).astype(int)
    target_dist = df[TARGET_VARIABLE].value_counts().sort_index()
    print(f"\n📊 Target distribution:")
    print(f"  Not opened: {target_dist[0]:,} ({target_dist[0]/len(df)*100:.1f}%)")
    print(f"  Opened: {target_dist[1]:,} ({target_dist[1]/len(df)*100:.1f}%)")
    
    # Advanced feature engineering
    df = create_xgboost_specific_features(df)
    
    # Drop only essential leakage columns
    df = df.drop(columns=[col for col in COLS_TO_DROP if col in df.columns], errors='ignore')
    
    # Handle categorical encoding
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != TARGET_VARIABLE:
            if df[col].nunique() <= 100:  # Keep more categories
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
            else:
                df = df.drop(columns=[col])
    
    # Drop timestamp columns
    timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower()]
    df = df.drop(columns=timestamp_cols, errors='ignore')
    
    # Prepare features
    y = df[TARGET_VARIABLE]
    X = df.drop(columns=[TARGET_VARIABLE])
    X = X.select_dtypes(include=[np.number])
    
    print(f"\n✅ Feature preparation complete. Shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Preprocessing
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Enhanced feature selection
    selected_indices = create_enhanced_feature_selection(X_train_imputed, y_train, n_features=60)
    X_train_selected = X_train_imputed[:, selected_indices]
    X_test_selected = X_test_imputed[:, selected_indices]
    selected_features = [X.columns[i] for i in selected_indices]
    
    print(f"\n📊 Final preprocessing:")
    print(f"  Training: {X_train_selected.shape}")
    print(f"  Test: {X_test_selected.shape}")
    print(f"  Selected features: {len(selected_features)}")
    
    # Enhanced grid search
    best_params, best_cv_score = perform_enhanced_xgboost_grid_search(X_train_selected, y_train)
    
    # Create and train ensemble
    ensemble = create_xgboost_ensemble(best_params, X_train_selected, y_train)
    print("\n🔄 Training enhanced ensemble...")
    ensemble.fit(X_train_selected, y_train)
    
    # Evaluate
    print("\n=== ENHANCED MODEL EVALUATION ===")
    y_pred = ensemble.predict(X_test_selected)
    y_pred_proba = ensemble.predict_proba(X_test_selected)[:, 1]
    
    print(f"\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Opened', 'Opened']))
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"🎯 Test Accuracy: {accuracy:.4f}")
    print(f"📈 Test ROC AUC: {auc:.4f}")
    
    # Performance improvement
    baseline_accuracy = 0.6986
    baseline_auc = 0.7800
    
    print(f"\n" + "="*60)
    print("🚀 PERFORMANCE IMPROVEMENT SUMMARY:")
    print("="*60)
    print(f"📊 ACCURACY:")
    print(f"  Baseline: {baseline_accuracy:.4f} (69.86%)")
    print(f"  Enhanced: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Improvement: +{accuracy - baseline_accuracy:+.4f} ({(accuracy - baseline_accuracy)*100:+.2f}%)")
    print(f"")
    print(f"📈 ROC AUC:")
    print(f"  Baseline: {baseline_auc:.4f} (78.00%)")
    print(f"  Enhanced: {auc:.4f} ({auc*100:.2f}%)")
    print(f"  Improvement: +{auc - baseline_auc:+.4f} ({(auc - baseline_auc)*100:+.2f}%)")
    print(f"")
    
    if accuracy >= 0.75 and auc >= 0.82:
        print("🎉 TARGET ACHIEVED!")
        print("✅ Accuracy ≥ 75% AND AUC ≥ 82%")
    elif accuracy >= 0.73 and auc >= 0.80:
        print("🎯 STRONG IMPROVEMENT!")
        print("📈 Close to target - consider additional ensemble methods")
    else:
        print("📈 GOOD PROGRESS!")
        print("💡 Next steps: Add SMOTE, more diverse ensemble members")
    
    print("="*60)
    
    return ensemble

if __name__ == "__main__":
    enhanced_model = main()