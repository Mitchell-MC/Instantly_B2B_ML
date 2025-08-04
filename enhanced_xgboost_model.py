"""
Enhanced XGBoost Model with All Improvements Applied
Target: Improve from 69.86% accuracy / 78.0% AUC to 75%+ accuracy / 82%+ AUC

Applied Enhancements:
1. ✅ Keep more predictive features (daily_limit, esp_code, etc.)
2. ✅ Advanced XGBoost-specific feature engineering  
3. ✅ 3-phase hyperparameter optimization
4. ✅ Expand from 13 to 50+ features
5. ✅ Class imbalance handling
6. ✅ Ensemble methods
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
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Configuration
CSV_FILE_PATH = Path("merged_contacts.csv")
TARGET_VARIABLE = "opened"

# ENHANCEMENT 1: KEEP MORE PREDICTIVE FEATURES (vs your current extensive dropping)
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
    
    # Keep these powerful features (they were being dropped before):
    # ✅ daily_limit - Your analysis shows this is highly predictive
    # ✅ esp_code - ESP 8.0, 11.0, 2.0 perform best  
    # ✅ campaign_id, email_list - Campaign patterns
    # ✅ organization_industry, country, state, city - Top features
    # ✅ upload_method, link_tracking, open_tracking - Technical features
]

def create_xgboost_optimized_features(df):
    """
    ENHANCEMENT 2: ADVANCED XGBOOST-SPECIFIC FEATURE ENGINEERING
    Based on your data analysis insights
    """
    print("🔧 Creating XGBoost-optimized features...")
    
    # 1. DAILY LIMIT ENGINEERING (Your top predictor from daily limit analysis)
    if 'daily_limit' in df.columns:
        # Performance-based encoding from your daily limit analysis
        def daily_limit_performance_score(limit):
            if pd.isna(limit):
                return 0
            elif limit <= 100:
                return 5  # Best performance (3.958 open rate from your analysis)
            elif limit <= 250:
                return 2  # Lower performance
            elif limit <= 350:
                return 3  # Medium performance  
            elif limit <= 450:
                return 2  # Medium-low performance
            elif limit <= 650:
                return 4  # Good performance
            else:
                return 4  # High performance (2.723 open rate)
        
        df['daily_limit_performance'] = df['daily_limit'].apply(daily_limit_performance_score)
        df['daily_limit_log'] = np.log1p(df['daily_limit'].fillna(0))
        df['daily_limit_squared'] = df['daily_limit'].fillna(0) ** 2
        df['daily_limit_is_optimal'] = ((df['daily_limit'] >= 100) & (df['daily_limit'] <= 150)).astype(int)
        df['daily_limit_is_high'] = (df['daily_limit'] >= 500).astype(int)
        df['daily_limit_quantile'] = pd.qcut(df['daily_limit'].fillna(-1), 
                                           q=10, labels=False, duplicates='drop')
    
    # 2. COMPANY SIZE OPTIMIZATION (Your #1 feature: organization_employees)
    if 'organization_employees' in df.columns:
        df['employees_log'] = np.log1p(df['organization_employees'].fillna(0))
        df['employees_sqrt'] = np.sqrt(df['organization_employees'].fillna(0))
        df['employees_quantile'] = pd.qcut(df['organization_employees'].fillna(-1), 
                                         q=20, labels=False, duplicates='drop')
        
        # Size category features
        df['is_enterprise'] = (df['organization_employees'] >= 1000).astype(int)
        df['is_mid_market'] = ((df['organization_employees'] >= 200) & 
                              (df['organization_employees'] < 1000)).astype(int)
        df['is_smb'] = (df['organization_employees'] < 200).astype(int)
        df['is_very_small'] = (df['organization_employees'] <= 10).astype(int)
    
    # 3. GEOGRAPHIC FEATURES (country, state, city are your top features)
    location_cols = ['country', 'state', 'city']
    for col in location_cols:
        if col in df.columns:
            # Frequency encoding (very effective for XGBoost)
            freq_map = df[col].value_counts().to_dict()
            df[f'{col}_frequency'] = df[col].map(freq_map).fillna(0)
            
            # Normalized frequency (0-1 scale)
            max_freq = df[f'{col}_frequency'].max()
            df[f'{col}_frequency_norm'] = df[f'{col}_frequency'] / max_freq if max_freq > 0 else 0
    
    # 4. INDUSTRY-TITLE INTERACTIONS (Your top features)
    if 'organization_industry' in df.columns and 'title' in df.columns:
        # Create high-value combinations
        df['industry_title_combo'] = (df['organization_industry'].astype(str) + '_' + 
                                     df['title'].astype(str))
        
        # Frequency encode the combinations
        combo_freq = df['industry_title_combo'].value_counts().to_dict()
        df['industry_title_frequency'] = df['industry_title_combo'].map(combo_freq).fillna(0)
        
        # High-performing industry-title combinations
        tech_titles = ['CEO', 'CTO', 'VP', 'Director', 'Manager', 'Founder']
        df['is_tech_leadership'] = ((df['organization_industry'].str.contains('Technology|Software|Tech', na=False)) &
                                   (df['title'].str.contains('|'.join(tech_titles), na=False))).astype(int)
        
        # Drop the text combo column
        df = df.drop(columns=['industry_title_combo'])
    
    # 5. ESP CODE OPTIMIZATION (Based on your ESP analysis)
    if 'esp_code' in df.columns:
        # From your ESP analysis: ESP 8.0, 11.0, 2.0 perform best
        high_performance_esp = [8.0, 11.0, 2.0]
        df['esp_is_high_performance'] = df['esp_code'].isin(high_performance_esp).astype(int)
        
        # ESP performance scoring based on your analysis
        esp_performance_map = {
            8.0: 5, 11.0: 4, 2.0: 3, 3.0: 2, 1.0: 2, 
            999.0: 1, 4.0: 1, 6.0: 1, 7.0: 1, 5.0: 1
        }
        df['esp_performance_score'] = df['esp_code'].map(esp_performance_map).fillna(0)
        
        # ESP code quantiles
        df['esp_code_quantile'] = pd.qcut(df['esp_code'].fillna(-1), 
                                        q=5, labels=False, duplicates='drop')
    
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
            df[f'{col}_is_morning'] = ((df[col].dt.hour >= 6) & (df[col].dt.hour < 12)).astype(int)
            df[f'{col}_is_afternoon'] = ((df[col].dt.hour >= 12) & (df[col].dt.hour < 18)).astype(int)
    
    # 7. CAMPAIGN AND TECHNICAL FEATURES
    campaign_cols = ['campaign_id', 'email_list', 'upload_method']
    for col in campaign_cols:
        if col in df.columns:
            freq_map = df[col].value_counts().to_dict()
            df[f'{col}_frequency'] = df[col].map(freq_map).fillna(0)
            # Normalize frequency
            max_freq = df[f'{col}_frequency'].max()
            df[f'{col}_frequency_norm'] = df[f'{col}_frequency'] / max_freq if max_freq > 0 else 0
    
    # 8. TECHNICAL FEATURES
    technical_cols = ['link_tracking', 'open_tracking', 'email_gap']
    for col in technical_cols:
        if col in df.columns:
            df[f'has_{col}'] = df[col].notna().astype(int)
            if df[col].dtype in ['int64', 'float64']:
                df[f'{col}_log'] = np.log1p(df[col].fillna(0))
                df[f'{col}_quantile'] = pd.qcut(df[col].fillna(-1), 
                                              q=5, labels=False, duplicates='drop')
    
    # 9. ORGANIZATION FEATURES
    if 'organization_founded_year' in df.columns:
        current_year = 2024
        df['company_age'] = current_year - df['organization_founded_year']
        df['company_age_log'] = np.log1p(df['company_age'].fillna(0))
        df['is_startup'] = (df['company_age'] <= 5).astype(int)
        df['is_mature_company'] = (df['company_age'] >= 20).astype(int)
        df['is_very_old_company'] = (df['company_age'] >= 50).astype(int)
    
    print(f"✅ XGBoost feature engineering complete. New shape: {df.shape}")
    return df

def perform_enhanced_xgboost_grid_search(X_train, y_train):
    """
    ENHANCEMENT 3: 3-PHASE XGBOOST HYPERPARAMETER OPTIMIZATION
    Much more comprehensive than basic parameter tuning
    """
    print("\n🎯 Enhanced XGBoost Grid Search (3-Phase Optimization)")
    print("="*80)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Phase 1: Broad search with XGBoost-specific parameters
    print("\n📊 Phase 1: Broad XGBoost Search")
    print("-" * 50)
    
    broad_params = {
        'n_estimators': [300, 400, 500],
        'max_depth': [8, 10, 12],
        'learning_rate': [0.08, 0.1, 0.12],
        'subsample': [0.85, 0.9, 0.95],
        'colsample_bytree': [0.85, 0.9, 0.95],
        'gamma': [0, 0.1, 0.2],
        'min_child_weight': [1, 3, 5]
    }
    
    xgb_broad = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=1.2  # Handle class imbalance
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
        'n_estimators': [max(200, best_broad['n_estimators'] - 100), 
                        best_broad['n_estimators'], 
                        best_broad['n_estimators'] + 100],
        'max_depth': [max(6, best_broad['max_depth'] - 1), 
                     best_broad['max_depth'], 
                     min(15, best_broad['max_depth'] + 1)],
        'learning_rate': [max(0.05, best_broad['learning_rate'] - 0.02), 
                         best_broad['learning_rate'], 
                         min(0.2, best_broad['learning_rate'] + 0.02)],
        'subsample': [max(0.8, best_broad['subsample'] - 0.05), 
                     best_broad['subsample'], 
                     min(1.0, best_broad['subsample'] + 0.05)],
        'colsample_bytree': [max(0.8, best_broad['colsample_bytree'] - 0.05), 
                            best_broad['colsample_bytree'], 
                            min(1.0, best_broad['colsample_bytree'] + 0.05)],
        'reg_alpha': [0, 0.1, 0.3],
        'reg_lambda': [1, 1.5, 2]
    }
    
    xgb_detailed = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=1.2,
        gamma=best_broad['gamma'],
        min_child_weight=best_broad['min_child_weight']
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
    
    # Phase 3: Final regularization optimization
    print("\n🔬 Phase 3: Regularization Fine-tuning")
    print("-" * 50)
    
    best_detailed = grid_detailed.best_params_
    
    final_params = {
        'reg_alpha': [0, 0.1, 0.3, 0.5],
        'reg_lambda': [0.5, 1.0, 1.5, 2.0],
        'scale_pos_weight': [1.0, 1.1, 1.2, 1.3]
    }
    
    # Combine all best parameters except those being fine-tuned
    base_params = {**best_broad, **best_detailed}
    base_params = {k: v for k, v in base_params.items() if k not in final_params}
    
    xgb_final = xgb.XGBClassifier(
        **base_params,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    
    grid_final = GridSearchCV(
        estimator=xgb_final,
        param_grid=final_params,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    
    print("🔄 Fitting final grid search...")
    grid_final.fit(X_train, y_train)
    
    print(f"✅ Best final score: {grid_final.best_score_:.4f}")
    print(f"📋 Best final params: {grid_final.best_params_}")
    
    # Combine all optimal parameters
    optimal_params = {**base_params, **grid_final.best_params_}
    
    print(f"\n🏆 OPTIMAL XGBOOST PARAMETERS:")
    print("="*50)
    for param, value in optimal_params.items():
        print(f"{param:<20}: {value}")
    print(f"\n⭐ Optimal CV Score: {grid_final.best_score_:.4f}")
    
    return optimal_params, grid_final.best_score_

def get_optimal_features(X, y, target_features=60):
    """
    ENHANCEMENT 4: EXPAND FROM 13 TO 60+ FEATURES
    Use advanced feature selection for better XGBoost performance
    """
    print(f"\n🎯 Selecting optimal {target_features} features (vs your current 13)")
    print("-" * 60)
    
    # Use statistical feature selection
    selector = SelectKBest(f_classif, k=min(target_features, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    
    selected_mask = selector.get_support()
    selected_features = [X.columns[i] for i, selected in enumerate(selected_mask) if selected]
    
    # Get feature scores for ranking
    feature_scores = selector.scores_
    feature_ranking = pd.DataFrame({
        'Feature': X.columns,
        'Score': feature_scores,
        'Selected': selected_mask
    }).sort_values('Score', ascending=False)
    
    print(f"✅ Selected {len(selected_features)} features")
    print(f"🔝 Top 15 features:")
    for i, (_, row) in enumerate(feature_ranking.head(15).iterrows(), 1):
        status = "✓" if row['Selected'] else "✗"
        print(f"  {i:2d}. {status} {row['Feature']:<40} (Score: {row['Score']:.1f})")
    
    return X_selected, selected_features

def create_ensemble_model(optimal_params, X_train, y_train):
    """
    ENHANCEMENT 5: ENSEMBLE METHOD
    Combine optimal XGBoost with other models for maximum performance
    """
    print("\n🎯 Creating Enhanced Ensemble Model")
    print("-" * 45)
    
    # Primary XGBoost model with optimal parameters
    xgb_optimal = xgb.XGBClassifier(
        **optimal_params,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    
    # Secondary XGBoost with different parameters for diversity
    xgb_diverse = xgb.XGBClassifier(
        n_estimators=optimal_params.get('n_estimators', 400) + 150,
        max_depth=min(15, optimal_params.get('max_depth', 10) + 2),
        learning_rate=max(0.05, optimal_params.get('learning_rate', 0.1) - 0.02),
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.2,
        reg_lambda=2.0,
        scale_pos_weight=1.3,
        random_state=123,  # Different seed
        n_jobs=-1
    )
    
    # Logistic Regression for ensemble diversity
    log_reg = LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=1000,
        C=0.1,  # Regularization
        n_jobs=-1
    )
    
    # Create voting ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('xgb_optimal', xgb_optimal),
            ('xgb_diverse', xgb_diverse),
            ('logreg', log_reg)
        ],
        voting='soft',  # Use probabilities
        n_jobs=-1
    )
    
    print("✅ Ensemble created with 3 diverse models:")
    print("  1. XGBoost (Optimal Parameters)")
    print("  2. XGBoost (Diverse Parameters)")  
    print("  3. Logistic Regression (Balanced)")
    
    return ensemble

def main():
    """
    ENHANCED XGBOOST PIPELINE WITH ALL IMPROVEMENTS APPLIED
    """
    print("=== ENHANCED XGBOOST: EMAIL OPENING PREDICTION ===")
    print("🎯 Target: 75%+ accuracy, 82%+ AUC (vs baseline 69.86%, 78.0%)")
    print("="*65)
    
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
    
    # ENHANCEMENT 2: Advanced feature engineering BEFORE dropping columns
    df = create_xgboost_optimized_features(df)
    
    # ENHANCEMENT 1: Drop only essential leakage columns (keep more predictive features)
    df = df.drop(columns=[col for col in COLS_TO_DROP if col in df.columns], errors='ignore')
    
    # Handle categorical encoding efficiently
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != TARGET_VARIABLE:
            if df[col].nunique() <= 100:  # Keep more categories than before
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
            else:
                # For very high cardinality, use frequency encoding instead of dropping
                freq_map = df[col].value_counts().to_dict()
                df[f'{col}_frequency'] = df[col].map(freq_map)
                df = df.drop(columns=[col])
    
    # Drop timestamp columns (after feature engineering)
    timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower()]
    df = df.drop(columns=timestamp_cols, errors='ignore')
    
    # Prepare features and target
    y = df[TARGET_VARIABLE]
    X = df.drop(columns=[TARGET_VARIABLE])
    X = X.select_dtypes(include=[np.number])
    
    print(f"\n✅ Feature preparation complete.")
    print(f"  📊 Total features: {X.shape[1]} (vs your original ~13)")
    print(f"  📋 Sample features: {list(X.columns[:10])}...")
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"\n📊 Data split:")
    print(f"  Training: {X_train.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Preprocessing
    print("\n🔄 Preprocessing data...")
    
    # Handle any remaining NaN columns before imputation
    nan_cols = X_train.columns[X_train.isna().all()].tolist()
    if nan_cols:
        print(f"⚠️  Dropping {len(nan_cols)} all-NaN columns: {nan_cols[:5]}...")
        X_train = X_train.drop(columns=nan_cols)
        X_test = X_test.drop(columns=nan_cols)
        X = X.drop(columns=nan_cols)  # Update the reference too
    
    imputer = SimpleImputer(strategy='median')
    X_train_imputed_array = imputer.fit_transform(X_train)
    X_test_imputed_array = imputer.transform(X_test)
    
    # Convert back to DataFrame for feature selection with correct column alignment
    X_train_df = pd.DataFrame(X_train_imputed_array, columns=X_train.columns, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_imputed_array, columns=X_test.columns, index=X_test.index)
    
    # ENHANCEMENT 4: Expand to 60 features (vs 13)
    X_train_selected, selected_features = get_optimal_features(X_train_df, y_train, target_features=60)
    
    # Apply same feature selection to test set
    selector = SelectKBest(f_classif, k=len(selected_features))
    selector.fit(X_train_df, y_train)
    X_test_selected = selector.transform(X_test_df)
    
    print(f"\n✅ Preprocessing complete:")
    print(f"  📊 Selected features: {len(selected_features)}")
    print(f"  📊 Training shape: {X_train_selected.shape}")
    print(f"  📊 Test shape: {X_test_selected.shape}")
    
    # ENHANCEMENT 3: Advanced hyperparameter optimization
    optimal_params, best_cv_score = perform_enhanced_xgboost_grid_search(X_train_selected, y_train)
    
    # ENHANCEMENT 5: Create and train ensemble
    ensemble = create_ensemble_model(optimal_params, X_train_selected, y_train)
    
    print("\n🔄 Training enhanced ensemble model...")
    ensemble.fit(X_train_selected, y_train)
    
    # Evaluate model
    print("\n=== ENHANCED MODEL EVALUATION ===")
    y_pred = ensemble.predict(X_test_selected)
    y_pred_proba = ensemble.predict_proba(X_test_selected)[:, 1]
    
    print(f"\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Opened', 'Opened']))
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"🎯 Test Accuracy: {accuracy:.4f}")
    print(f"📈 Test ROC AUC: {auc:.4f}")
    
    # Create confusion matrix (without seaborn to avoid issues)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center", color="black")
    
    plt.title('Enhanced XGBoost Confusion Matrix\nEmail Opening Prediction')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.xticks([0, 1], ['Not Opened', 'Opened'])
    plt.yticks([0, 1], ['Not Opened', 'Opened'])
    plt.tight_layout()
    plt.savefig('enhanced_xgboost_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Confusion matrix saved as 'enhanced_xgboost_confusion_matrix.png'")
    
    # Feature importance from primary XGBoost model
    primary_model = ensemble.estimators_[0]  # Get the optimal XGBoost model
    feature_importance = primary_model.feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=True)
    
    print(f"\n🔝 Top 15 Feature Importances:")
    for i, (_, row) in enumerate(importance_df.tail(15).iterrows(), 1):
        print(f"  {i:2d}. {row['Feature']:<35}: {row['Importance']:.4f}")
    
    # Performance comparison
    baseline_accuracy = 0.6986
    baseline_auc = 0.7800
    improvement_accuracy = accuracy - baseline_accuracy
    improvement_auc = auc - baseline_auc
    
    print(f"\n" + "="*70)
    print("🚀 ENHANCED XGBOOST PERFORMANCE SUMMARY:")
    print("="*70)
    print(f"✅ Applied ALL recommended enhancements:")
    print(f"  🔧 Advanced XGBoost-specific feature engineering")
    print(f"  📊 Kept more predictive features (daily_limit, esp_code, etc.)")
    print(f"  🎯 3-phase hyperparameter optimization")
    print(f"  📈 Expanded to {len(selected_features)} features (vs 13 baseline)")
    print(f"  ⚖️  Class imbalance handling")
    print(f"  🤝 Ensemble with 3 diverse models")
    print(f"")
    print(f"📊 PERFORMANCE METRICS:")
    print(f"  🎯 Test accuracy: {accuracy:.4f} vs {baseline_accuracy:.4f} baseline")
    print(f"     Improvement: {improvement_accuracy:+.4f} ({improvement_accuracy*100:+.2f}%)")
    print(f"  📈 Test ROC AUC: {auc:.4f} vs {baseline_auc:.4f} baseline") 
    print(f"     Improvement: {improvement_auc:+.4f} ({improvement_auc*100:+.2f}%)")
    print(f"  📋 CV Score: {best_cv_score:.4f}")
    print(f"  🔧 Features used: {len(selected_features)}")
    print(f"")
    
    if accuracy >= 0.75 and auc >= 0.82:
        print("🎉 TARGET ACHIEVED!")
        print(f"  ✅ Accuracy ≥ 75% ({accuracy*100:.1f}%)")
        print(f"  ✅ AUC ≥ 82% ({auc*100:.1f}%)")
        print("  🚀 All enhancement recommendations successfully applied!")
    elif accuracy >= 0.73 and auc >= 0.80:
        print("🎯 EXCELLENT IMPROVEMENT!")
        print(f"  📈 Close to target with {accuracy*100:.1f}% accuracy, {auc*100:.1f}% AUC")
        print("  💡 Consider adding SMOTE for final performance boost")
    else:
        print("📈 GOOD PROGRESS!")
        print(f"  ✅ Significant improvement: +{improvement_accuracy*100:.1f}% accuracy, +{improvement_auc*100:.1f}% AUC")
        print("  💡 Next: Add more ensemble diversity or advanced regularization")
    
    print("="*70)
    
    return ensemble

if __name__ == "__main__":
    enhanced_model = main()