"""
Optimized Training Pipeline for Real Data Testing
Fixes cross-validation issues and adds comprehensive hyperparameter optimization.
"""

import pandas as pd
import numpy as np
import yaml
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
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

def validate_data(df):
    """Comprehensive data validation."""
    print("🔍 Validating data quality...")
    
    # Basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Missing values analysis
    missing_stats = df.isnull().sum().sort_values(ascending=False)
    missing_percent = (missing_stats / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing_Count': missing_stats,
        'Missing_Percent': missing_percent
    })
    
    print(f"\nTop 10 columns with missing values:")
    print(missing_df.head(10))
    
    # Duplicate analysis
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")
    
    # Data types
    print(f"\nData types:")
    print(df.dtypes.value_counts())
    
    return df

def load_real_data(config):
    """Load and validate real data."""
    print("📊 Loading real data from merged_contacts.csv...")
    
    data_path = Path("merged_contacts.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Real data file not found: {data_path}")
    
    # Load data with proper error handling
    try:
        df = pd.read_csv(data_path, on_bad_lines='warn', low_memory=False)
        print(f"✅ Real data loaded. Shape: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise
    
    # Validate data quality
    df = validate_data(df)
    
    # Standardize timestamp columns
    timestamp_cols = ['timestamp_created', 'timestamp_last_contact', 'retrieval_timestamp', 
                     'enriched_at', 'inserted_at', 'last_contacted_from']
    for col in timestamp_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return df

def create_target_variable(df, config):
    """Create target variable with proper validation."""
    print("🎯 Creating target variable...")
    
    target_type = config.get('data', {}).get('target_type', 'binary')
    target_variable = config['data']['target_variable']
    
    if target_type == 'binary':
        # Binary target: opened (0/1)
        if 'opened' in df.columns:
            df[target_variable] = df['opened'].astype(int)
        elif 'email_open_count' in df.columns:
            # Create binary target from engagement metrics
            df[target_variable] = (df['email_open_count'] > 0).astype(int)
        else:
            raise ValueError("No target variable found. Need 'opened' or 'email_open_count' column.")
    
    elif target_type == 'multiclass':
        # Multi-class target: engagement_level (0, 1, 2)
        if all(col in df.columns for col in ['email_open_count', 'email_click_count', 'email_reply_count']):
            conditions = [
                ((df['email_click_count'] > 0) | (df['email_reply_count'] > 0)),  # Tier 2: Click OR Reply
                (df['email_open_count'] > 0)                                       # Tier 1: Open
            ]
            choices = [2, 1]
            df[target_variable] = np.select(conditions, choices, default=0)
        else:
            raise ValueError("Missing required columns for multi-class target: email_open_count, email_click_count, email_reply_count")
    
    # Validate target distribution
    target_dist = df[target_variable].value_counts().sort_index()
    print(f"Target distribution:\n{target_dist}")
    print(f"Class proportions: {target_dist / len(df)}")
    
    # Check for class imbalance
    min_class_ratio = target_dist.min() / target_dist.max()
    if min_class_ratio < 0.1:
        print(f"⚠️ Warning: Severe class imbalance detected ({min_class_ratio:.3f})")
    elif min_class_ratio < 0.3:
        print(f"⚠️ Warning: Moderate class imbalance detected ({min_class_ratio:.3f})")
    
    return df

def apply_enhanced_feature_engineering(df):
    """Apply comprehensive feature engineering with validation."""
    print("🔧 Applying enhanced feature engineering...")
    
    original_shape = df.shape
    
    # 1. Enhanced text preprocessing
    df = enhanced_text_preprocessing(df)
    
    # 2. Advanced timestamp features
    df = advanced_timestamp_features(df)
    
    # 3. Create interaction features
    df = create_interaction_features(df)
    
    # 4. JSONB features
    df = create_jsonb_features(df)
    
    # 5. Domain-specific XGBoost features
    df = create_xgboost_optimized_features(df)
    
    # 6. Handle outliers
    df = handle_outliers(df)
    
    print(f"✅ Feature engineering complete. Shape: {df.shape}")
    print(f"📈 Feature increase: {df.shape[1] - original_shape[1]} new features")
    
    return df

def prepare_features_for_model_safe(df, target_variable, cols_to_drop=None):
    """Prepare features for model with data leakage prevention."""
    print("🔧 Preparing features for model (with leakage prevention)...")
    
    # Default columns to drop
    default_cols_to_drop = [
        'id', 'email', 'first_name', 'last_name', 'company_name', 'linkedin_url',
        'website', 'headline', 'company_domain', 'phone', 'apollo_id',
        'apollo_name', 'organization', 'photo_url', 'organization_name',
        'organization_website', 'organization_phone', 'combined_text'  # Drop raw text
    ]
    
    if cols_to_drop:
        default_cols_to_drop.extend(cols_to_drop)
    
    # CRITICAL FIX: Remove target variable from features to prevent data leakage
    if target_variable in df.columns:
        print(f"⚠️ Removing target variable '{target_variable}' from features to prevent data leakage")
        default_cols_to_drop.append(target_variable)
    
    # Also remove any columns that might contain target information
    target_related_cols = ['email_open_count', 'email_click_count', 'email_reply_count', 
                          'email_opened_variant', 'email_clicked_variant', 'email_replied_variant']
    for col in target_related_cols:
        if col in df.columns and col not in default_cols_to_drop:
            print(f"⚠️ Removing target-related column '{col}' from features")
            default_cols_to_drop.append(col)
    
    # Create feature set
    X = df.drop(columns=[col for col in default_cols_to_drop if col in df.columns], errors='ignore')
    
    # Select only numeric features
    X = X.select_dtypes(include=[np.number])
    
    # Get target variable
    y = df[target_variable] if target_variable in df.columns else None
    
    print(f"Final feature set shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    
    return X, y, list(X.columns)

def perform_feature_selection(X, y, config):
    """Perform comprehensive feature selection."""
    print("🔍 Performing feature selection...")
    
    # Handle NaN values first
    print(f"Handling NaN values...")
    print(f"NaN count before: {X.isnull().sum().sum()}")
    
    # Check for columns with all NaN values
    all_nan_cols = X.columns[X.isnull().all()].tolist()
    if all_nan_cols:
        print(f"Removing {len(all_nan_cols)} columns with all NaN values: {all_nan_cols}")
        X = X.drop(columns=all_nan_cols)
    
    # Simple imputation for feature selection
    imputer = SimpleImputer(strategy='median')
    X_imputed_array = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(
        X_imputed_array, 
        columns=X.columns, 
        index=X.index
    )
    print(f"NaN count after imputation: {X_imputed.isnull().sum().sum()}")
    
    # 1. Remove low variance features
    variance_selector = VarianceThreshold(threshold=0.01)
    X_var_selected = variance_selector.fit_transform(X_imputed)
    var_selected_features = X_imputed.columns[variance_selector.get_support()].tolist()
    print(f"After variance selection: {len(var_selected_features)} features")
    
    # 2. Remove highly correlated features
    X_df = pd.DataFrame(X_var_selected, columns=var_selected_features)
    corr_matrix = X_df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    X_uncorr = X_df.drop(columns=high_corr_features)
    print(f"After correlation removal: {X_uncorr.shape[1]} features")
    
    # 3. Select top features using mutual information
    mi_scores = mutual_info_classif(X_uncorr, y, random_state=42)
    mi_df = pd.DataFrame({'feature': X_uncorr.columns, 'mi_score': mi_scores})
    mi_df = mi_df.sort_values('mi_score', ascending=False)
    
    # Select top features
    n_features = min(config['features']['target_features'], len(mi_df))
    top_features = mi_df.head(n_features)['feature'].tolist()
    X_final = X_uncorr[top_features]
    
    print(f"Final selected features: {len(top_features)}")
    print(f"Top 10 features by mutual information:")
    print(mi_df.head(10))
    
    return X_final, top_features, mi_df

def compute_balanced_class_weights(y):
    """Compute balanced class weights to handle class imbalance."""
    print("⚖️ Computing balanced class weights...")
    
    # Get unique classes
    classes = np.unique(y)
    
    # Compute balanced class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=classes,
        y=y
    )
    
    # Create weight dictionary
    weight_dict = dict(zip(classes, class_weights))
    print(f"Class weights: {weight_dict}")
    
    return weight_dict

def create_optimized_models(config, class_weights):
    """Create optimized models with class weights."""
    print("🤖 Creating optimized models...")
    
    # XGBoost with class weights (removed early_stopping_rounds)
    xgb_model = xgb.XGBClassifier(
        random_state=config['training']['random_state'],
        scale_pos_weight=class_weights.get(1, 1.0) if len(class_weights) == 2 else None,
        class_weight=class_weights if len(class_weights) > 2 else None,
        eval_metric='logloss',
        verbosity=0
    )
    
    # Logistic Regression with improved convergence settings
    lr_model = LogisticRegression(
        random_state=config['training']['random_state'],
        max_iter=5000,  # Increased significantly for convergence
        solver='liblinear',  # Changed from 'lbfgs' to 'liblinear' for better convergence
        class_weight='balanced',
        C=1.0,
        tol=1e-4  # Slightly relaxed tolerance
    )
    
    return xgb_model, lr_model

def perform_hyperparameter_optimization(X_train, y_train, config, class_weights):
    """Perform comprehensive hyperparameter optimization."""
    print("🔧 Performing hyperparameter optimization...")
    
    # Create base models
    xgb_model, lr_model = create_optimized_models(config, class_weights)
    
    # Simplified XGBoost hyperparameter grid (reduced to avoid too many combinations)
    xgb_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.1, 0.2],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'min_child_weight': [1, 3]
    }
    
    # Simplified Logistic Regression hyperparameter grid with better convergence
    lr_param_grid = {
        'C': [0.1, 1.0, 10.0],
        'max_iter': [3000, 5000],
        'solver': ['liblinear']  # Only use liblinear for better convergence
    }
    
    # Create scoring function for multi-class
    if len(np.unique(y_train)) > 2:
        scoring = 'roc_auc_ovr_weighted'
    else:
        scoring = 'roc_auc'
    
    print(f"Using scoring metric: {scoring}")
    
    # Optimize XGBoost
    print("🔍 Optimizing XGBoost...")
    xgb_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    xgb_grid = GridSearchCV(
        xgb_model,
        xgb_param_grid,
        cv=xgb_cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        error_score=0  # Return 0 score for failed fits
    )
    xgb_grid.fit(X_train, y_train)
    
    print(f"Best XGBoost parameters: {xgb_grid.best_params_}")
    print(f"Best XGBoost score: {xgb_grid.best_score_:.4f}")
    
    # Optimize Logistic Regression
    print("🔍 Optimizing Logistic Regression...")
    lr_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    lr_grid = GridSearchCV(
        lr_model,
        lr_param_grid,
        cv=lr_cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        error_score=0  # Return 0 score for failed fits
    )
    lr_grid.fit(X_train, y_train)
    
    print(f"Best Logistic Regression parameters: {lr_grid.best_params_}")
    print(f"Best Logistic Regression score: {lr_grid.best_score_:.4f}")
    
    return xgb_grid.best_estimator_, lr_grid.best_estimator_

def create_optimized_ensemble(xgb_optimized, lr_optimized, config):
    """Create optimized ensemble model."""
    print("🤖 Creating optimized ensemble...")
    
    # Create ensemble with optimized models
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb_optimized),
            ('lr', lr_optimized)
        ],
        voting=config['model']['ensemble']['voting'],
        n_jobs=config['model']['ensemble']['n_jobs']
    )
    
    return ensemble

def perform_robust_cross_validation(model, X, y, config):
    """Perform robust cross-validation with proper error handling."""
    print("🔄 Performing robust cross-validation...")
    
    # Create custom scoring function for multi-class
    if len(np.unique(y)) > 2:
        scoring = 'roc_auc_ovr_weighted'
    else:
        scoring = 'roc_auc'
    
    # Use stratified k-fold with proper error handling
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    try:
        # Cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        print(f"Cross-validation {scoring} scores: {cv_scores}")
        print(f"Mean CV {scoring}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Check for NaN values
        if np.any(np.isnan(cv_scores)):
            print("⚠️ Warning: Some CV scores are NaN. This may indicate convergence issues.")
            # Filter out NaN values
            cv_scores = cv_scores[~np.isnan(cv_scores)]
            if len(cv_scores) > 0:
                print(f"Valid CV scores: {cv_scores}")
                print(f"Mean valid CV {scoring}: {cv_scores.mean():.4f}")
            else:
                print("❌ No valid CV scores found")
                cv_scores = np.array([np.nan])
        
        return cv_scores
        
    except Exception as e:
        print(f"⚠️ Cross-validation failed: {e}")
        print("Falling back to single train-test evaluation")
        return np.array([np.nan])

def evaluate_model_comprehensive(model, X_test, y_test, config):
    """Comprehensive model evaluation."""
    print("📊 Comprehensive model evaluation...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*60)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")
    
    # Calculate additional metrics
    accuracy = (y_pred == y_test).mean()
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # AUC score (handle both binary and multi-class)
    if len(model.classes_) == 2:
        # Binary classification
        auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
        print(f"ROC AUC Score: {auc_score:.4f}")
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Optimized Model')
        plt.legend()
        plt.savefig('roc_curve_optimized.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ ROC curve saved as 'roc_curve_optimized.png'")
    else:
        # Multi-class classification
        try:
            # Use one-vs-rest AUC for multi-class
            auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            print(f"Multi-class ROC AUC Score (weighted): {auc_score:.4f}")
        except Exception as e:
            print(f"⚠️ Could not calculate AUC for multi-class: {e}")
            auc_score = None
    
    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Optimized Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_optimized.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Confusion matrix saved as 'confusion_matrix_optimized.png'")
    
    return {
        'accuracy': accuracy,
        'auc': auc_score,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

def main():
    """Main training function with optimization."""
    print("🚀 Starting Optimized Real Data Training Pipeline")
    print("="*60)
    print("Testing on merged_contacts.csv with hyperparameter optimization")
    print("="*60)
    
    try:
        # 1. Load configuration
        config = load_config()
        
        # 2. Load real data
        df = load_real_data(config)
        
        # 3. Create target variable
        df = create_target_variable(df, config)
        
        # 4. Apply enhanced feature engineering
        df = apply_enhanced_feature_engineering(df)
        
        # 5. Prepare features for model (with leakage prevention)
        X, y, selected_features = prepare_features_for_model_safe(
            df, 
            target_variable=config['data']['target_variable'],
            cols_to_drop=config['features']['cols_to_drop']
        )
        
        # 6. Perform feature selection
        X_selected, top_features, mi_scores = perform_feature_selection(X, y, config)
        
        # 7. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, 
            test_size=config['data']['test_size'],
            random_state=config['data']['random_state'],
            stratify=y
        )
        
        print(f"\n📊 Data split:")
        print(f"  Training: {X_train.shape}")
        print(f"  Test: {X_test.shape}")
        
        # 8. Compute class weights for handling imbalance
        class_weights = compute_balanced_class_weights(y_train)
        
        # 9. Perform hyperparameter optimization
        xgb_optimized, lr_optimized = perform_hyperparameter_optimization(
            X_train, y_train, config, class_weights
        )
        
        # 10. Create optimized ensemble
        model = create_optimized_ensemble(xgb_optimized, lr_optimized, config)
        
        # 11. Perform robust cross-validation
        cv_scores = perform_robust_cross_validation(model, X_train, y_train, config)
        
        # 12. Train final model
        print("🏋️ Training optimized model...")
        model.fit(X_train, y_train)
        
        # 13. Evaluate model
        performance_metrics = evaluate_model_comprehensive(model, X_test, y_test, config)
        
        # 14. Save model artifacts
        print("💾 Saving optimized model artifacts...")
        artifacts = {
            'model': model,
            'feature_names': top_features,
            'config': config,
            'performance_metrics': performance_metrics,
            'cv_scores': cv_scores,
            'class_weights': class_weights,
            'model_version': config['model']['version'],
            'training_shape': X_train.shape,
            'optimization_info': {
                'xgb_best_params': xgb_optimized.get_params(),
                'lr_best_params': lr_optimized.get_params()
            }
        }
        
        model_path = Path("models/email_open_predictor_optimized_v1.0.joblib")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifacts, model_path)
        print(f"✅ Optimized model artifacts saved to: {model_path}")
        
        # 15. Print summary
        print("\n" + "="*60)
        print("OPTIMIZED REAL DATA TRAINING SUMMARY")
        print("="*60)
        print(f"📊 Dataset: merged_contacts.csv ({df.shape[0]} rows, {df.shape[1]} columns)")
        print(f"🎯 Target: {config['data']['target_variable']}")
        print(f"🔧 Features: {len(top_features)} selected from {X.shape[1]} engineered")
        print(f"⚖️ Class weights applied: {class_weights}")
        print(f"📈 Performance:")
        print(f"  - Accuracy: {performance_metrics['accuracy']:.4f}")
        if performance_metrics['auc'] is not None:
            print(f"  - AUC: {performance_metrics['auc']:.4f}")
        else:
            print(f"  - AUC: Not available (multi-class)")
        
        # Handle NaN CV scores
        if not np.any(np.isnan(cv_scores)):
            print(f"  - CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        else:
            print(f"  - CV AUC: Not available (convergence issues)")
        
        print(f"✅ Optimized model saved to: {model_path}")
        
        # 16. Performance assessment
        if performance_metrics['auc'] is not None:
            if performance_metrics['auc'] >= 0.75:
                print("🎉 EXCELLENT: Optimized model performs well on real data!")
            elif performance_metrics['auc'] >= 0.65:
                print("✅ GOOD: Optimized model performs reasonably on real data")
            else:
                print("⚠️ NEEDS IMPROVEMENT: Optimized model performance below expectations")
        else:
            # For multi-class, use accuracy
            if performance_metrics['accuracy'] >= 0.75:
                print("🎉 EXCELLENT: Optimized model performs well on real data!")
            elif performance_metrics['accuracy'] >= 0.65:
                print("✅ GOOD: Optimized model performs reasonably on real data")
            else:
                print("⚠️ NEEDS IMPROVEMENT: Optimized model performance below expectations")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 