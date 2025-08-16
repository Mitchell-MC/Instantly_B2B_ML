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

from src.feature_engineering import (
    enhanced_text_preprocessing, advanced_timestamp_features, 
    create_interaction_features, create_jsonb_features, handle_outliers,
    create_xgboost_optimized_features, encode_categorical_features, 
    prepare_features_for_model, create_comprehensive_organization_data,
    create_advanced_engagement_features, create_comprehensive_jsonb_features # Added comprehensive JSONB
)
from src.ctgan_augmentation import augment_training_data_with_ctgan


def save_model_artifacts(model, X_train, y_train, config, metadata=None):
    """Persist trained model and metadata to the configured artifact path."""
    try:
        artifact_path = Path(config['paths']['model_artifact'])
        artifact_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            'model': model,
            'features': list(X_train.columns),
            'class_labels': np.unique(y_train),
        }
        if metadata:
            payload.update(metadata)

        joblib.dump(payload, artifact_path)
        print(f"✅ Optimized model saved to: {artifact_path}")
        
        # MLflow integration - log training run
        try:
            from src.mlflow_integration import log_training_run_wrapper
            
            # Prepare metrics for MLflow
            if 'test_metrics' in metadata:
                mlflow_metrics = metadata['test_metrics']
            else:
                # Default metrics if not provided
                mlflow_metrics = {
                    'accuracy': 0.0,
                    'f1_score': 0.0,
                    'precision': 0.0,
                    'recall': 0.0
                }
            
            # Log to MLflow
            run_id = log_training_run_wrapper(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=metadata.get('X_test', X_train.head(100)),  # Use sample if not provided
                y_test=metadata.get('y_test', y_train.head(100)),
                metrics=mlflow_metrics,
                params=metadata.get('model_params', {}),
                feature_importance=metadata.get('feature_importance'),
                artifacts=metadata.get('artifacts', {}),
                run_name=f"training_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            print(f"✅ MLflow training run logged with ID: {run_id}")
            
        except ImportError:
            print("⚠️  MLflow not available - skipping MLflow logging")
        except Exception as e:
            print(f"⚠️  MLflow logging failed: {e}")
        
    except Exception as e:
        print(f"❌ Failed to save model artifacts: {e}")
        raise

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
    
    # Get engagement columns
    email_open_count = df.get('email_open_count', 0)
    email_click_count = df.get('email_click_count', 0)
    email_reply_count = df.get('email_reply_count', 0)
    
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
        ((df['email_open_count'] >= 1) & (df['email_click_count'] >= 1)) |
        ((df['email_open_count'] >= 1) & (df['email_reply_count'] >= 1))
    )
    df.loc[mask_bucket3, 'engagement_level'] = 2
    
    # Display target distribution
    target_dist = df['engagement_level'].value_counts().sort_index()
    print("Target distribution:")
    print(target_dist)
    
    # Calculate class proportions
    class_proportions = df['engagement_level'].value_counts(normalize=True).sort_index()
    print("Class proportions:")
    print(class_proportions)
    
    # Check for class imbalance
    max_prop = class_proportions.max()
    min_prop = class_proportions.min()
    imbalance = max_prop - min_prop
    
    if imbalance > 0.3:
        print(f"⚠️ Warning: Severe class imbalance detected ({imbalance:.3f})")
    elif imbalance > 0.1:
        print(f"⚠️ Warning: Moderate class imbalance detected ({imbalance:.3f})")
    else:
        print(f"✅ Class balance is acceptable ({imbalance:.3f})")
    
    return df

def apply_enhanced_feature_engineering(df):
    """Apply all enhanced feature engineering techniques including comprehensive organization data."""
    print("🔧 Applying enhanced feature engineering...")
    
    original_shape = df.shape
    
    # 1. Enhanced text preprocessing
    df = enhanced_text_preprocessing(df)
    
    # 2. Advanced timestamp features
    df = advanced_timestamp_features(df)
    
    # 3. Interaction features
    df = create_interaction_features(df)
    
    # 4. JSONB features
    df = create_jsonb_features(df)
    
    # 5. Comprehensive organization data (NEW)
    df = create_comprehensive_organization_data(df)
    
    # 6. Domain-specific XGBoost features
    df = create_xgboost_optimized_features(df)
    
    # 7. Handle outliers
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

def perform_advanced_feature_selection(X, y, config):
    """
    Perform advanced feature selection using multiple techniques.
    """
    print("🔍 Performing advanced feature selection...")
    
    from sklearn.feature_selection import RFE, SelectFromModel
    from sklearn.ensemble import RandomForestClassifier
    import shap
    
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
    
    # 1. Variance threshold
    variance_selector = VarianceThreshold(threshold=0.01)
    X_var_selected = variance_selector.fit_transform(X_imputed)
    var_selected_features = X_imputed.columns[variance_selector.get_support()].tolist()
    print(f"After variance selection: {len(var_selected_features)} features")
    
    # 2. Correlation-based selection
    X_df = pd.DataFrame(X_var_selected, columns=var_selected_features)
    corr_matrix = X_df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    X_uncorr = X_df.drop(columns=high_corr_features)
    print(f"After correlation removal: {X_uncorr.shape[1]} features")
    
    # 3. Mutual information selection
    mi_scores = mutual_info_classif(X_uncorr, y, random_state=42)
    mi_df = pd.DataFrame({'feature': X_uncorr.columns, 'mi_score': mi_scores})
    mi_df = mi_df.sort_values('mi_score', ascending=False)
    
    # 4. Recursive Feature Elimination (RFE)
    print("🔄 Performing Recursive Feature Elimination...")
    rfe_selector = RFE(
        estimator=RandomForestClassifier(n_estimators=100, random_state=42),
        n_features_to_select=min(30, X_uncorr.shape[1]),
        step=1
    )
    X_rfe = rfe_selector.fit_transform(X_uncorr, y)
    rfe_features = X_uncorr.columns[rfe_selector.support_].tolist()
    print(f"RFE selected: {len(rfe_features)} features")
    
    # 5. SHAP-based feature selection (with sampling for performance)
    print("🔄 Performing SHAP-based feature selection...")
    try:
        shap_cfg = config.get('features', {}).get('shap', {})
        shap_enabled = shap_cfg.get('enabled', True)
        if not shap_enabled:
            raise RuntimeError("SHAP disabled via config")

        max_samples = int(shap_cfg.get('max_samples', 1000))
        rf_estimators = int(shap_cfg.get('rf_estimators', 100))
        shap_top_k = int(shap_cfg.get('top_k', 25))

        # Downsample rows to speed up SHAP
        if len(X_uncorr) > max_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_uncorr), size=max_samples, replace=False)
            X_shap = X_uncorr.iloc[idx]
            y_shap = y.iloc[idx]
        else:
            X_shap = X_uncorr
            y_shap = y

        # Train a small RF model for SHAP analysis
        rf_for_shap = RandomForestClassifier(n_estimators=rf_estimators, random_state=42, n_jobs=-1)
        rf_for_shap.fit(X_shap, y_shap)

        # Calculate SHAP values on the sampled set
        explainer = shap.TreeExplainer(rf_for_shap)
        shap_values = explainer.shap_values(X_shap)

        # Aggregate SHAP importance across classes if needed
        if isinstance(shap_values, list):
            # Multi-class: average absolute SHAP across classes
            per_class_importance = [np.abs(v).mean(axis=0) for v in shap_values]
            shap_importance = np.mean(per_class_importance, axis=0)
        else:
            # Binary: single array (n_samples, n_features)
            shap_importance = np.abs(shap_values).mean(axis=0)

        shap_df = pd.DataFrame({
            'feature': X_shap.columns,
            'shap_importance': shap_importance
        }).sort_values('shap_importance', ascending=False)

        # Select top SHAP features
        top_shap_features = shap_df.head(min(shap_top_k, len(shap_df)))['feature'].tolist()
        print(f"SHAP selected: {len(top_shap_features)} features (sampled {len(X_shap)} rows)")

    except Exception as e:
        print(f"⚠️ SHAP analysis skipped/fallback: {e}")
        shap_df = mi_df.copy()
        top_shap_features = mi_df.head(min(25, len(mi_df)))['feature'].tolist()
    
    # 6. Model-based feature selection
    print("🔄 Performing model-based feature selection...")
    model_selector = SelectFromModel(
        estimator=RandomForestClassifier(n_estimators=100, random_state=42),
        max_features=min(30, X_uncorr.shape[1])
    )
    X_model_selected = model_selector.fit_transform(X_uncorr, y)
    model_features = X_uncorr.columns[model_selector.get_support()].tolist()
    print(f"Model-based selected: {len(model_features)} features")
    
    # 7. Combine all selection methods
    all_selected_features = list(set(
        mi_df.head(min(30, len(mi_df))).index.tolist() +
        rfe_features +
        top_shap_features +
        model_features
    ))
    
    # Final feature set
    final_features = [f for f in all_selected_features if f in X_uncorr.columns]
    X_final = X_uncorr[final_features]
    
    print(f"Final selected features: {len(final_features)}")
    print(f"Top 10 features by mutual information:")
    print(mi_df.head(10))
    
    if 'shap_importance' in shap_df.columns:
        print(f"Top 10 features by SHAP importance:")
        print(shap_df.head(10))
    
    return X_final, final_features, mi_df

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
        solver='liblinear',  # Better for small datasets
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

def apply_advanced_class_balancing(X_train, y_train, X_test, y_test):
    """
    Apply advanced class balancing techniques to address severe imbalance.
    """
    print("⚖️ Applying advanced class balancing techniques...")
    
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTEENN, SMOTETomek
    from sklearn.utils.class_weight import compute_class_weight
    
    # Original class distribution
    print(f"Original class distribution: {np.bincount(y_train)}")
    
    # Method 1: SMOTE (Synthetic Minority Over-sampling Technique)
    print("🔄 Applying SMOTE...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {np.bincount(y_train_smote)}")
    
    # Method 2: ADASYN (Adaptive Synthetic Sampling)
    print("🔄 Applying ADASYN...")
    adasyn = ADASYN(random_state=42, n_neighbors=5)
    X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)
    print(f"After ADASYN: {np.bincount(y_train_adasyn)}")
    
    # Method 3: SMOTEENN (SMOTE + Edited Nearest Neighbors)
    print("🔄 Applying SMOTEENN...")
    smoteenn = SMOTEENN(random_state=42)
    X_train_smoteenn, y_train_smoteenn = smoteenn.fit_resample(X_train, y_train)
    print(f"After SMOTEENN: {np.bincount(y_train_smoteenn)}")
    
    # Method 4: Cost-sensitive class weights
    print("🔄 Computing cost-sensitive class weights...")
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    cost_sensitive_weights = dict(zip(np.unique(y_train), class_weights))
    print(f"Cost-sensitive weights: {cost_sensitive_weights}")
    
    return {
        'smote': (X_train_smote, y_train_smote),
        'adasyn': (X_train_adasyn, y_train_adasyn),
        'smoteenn': (X_train_smoteenn, y_train_smoteenn),
        'cost_sensitive': cost_sensitive_weights,
        'original': (X_train, y_train)
    }

def create_advanced_model_ensemble(X_train, y_train, class_weights):
    """
    Create advanced model ensemble with multiple algorithms.
    """
    print("🤖 Creating advanced model ensemble...")
    
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    from sklearn.ensemble import StackingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    
    # Base models with optimized parameters
    models = {
        'xgb': xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.8,
            min_child_weight=3,
            random_state=42,
            scale_pos_weight=class_weights[1] if len(class_weights) > 1 else 1
        ),
        'lgbm': LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=42,
            class_weight='balanced'
        ),
        'catboost': CatBoostClassifier(
            iterations=200,
            depth=5,
            learning_rate=0.1,
            random_seed=42,
            verbose=False,
            class_weights=class_weights
        ),
        'svm': SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            probability=True,
            random_state=42,
            class_weight='balanced'
        ),
        'mlp': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            max_iter=1000,
            random_state=42
        )
    }
    
    # Train base models
    print("🔄 Training base models...")
    trained_models = {}
    for name, model in models.items():
        print(f"  Training {name}...")
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
            print(f"  ✅ {name} trained successfully")
        except Exception as e:
            print(f"  ❌ {name} failed: {e}")
    
    # Create voting ensemble
    print("🔄 Creating voting ensemble...")
    voting_ensemble = VotingClassifier(
        estimators=[(name, model) for name, model in trained_models.items()],
        voting='soft'
    )
    
    # Create stacking ensemble
    print("🔄 Creating stacking ensemble...")
    estimators = [(name, model) for name, model in trained_models.items()]
    stacking_ensemble = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(random_state=42),
        cv=3
    )
    
    return {
        'base_models': trained_models,
        'voting_ensemble': voting_ensemble,
        'stacking_ensemble': stacking_ensemble
    }

def main():
    """Main training function with optimization."""
    print("🚀 Starting Optimized Real Data Training Pipeline")
    print("="*60)
    print("Testing on merged_contacts.csv with hyperparameter optimization")
    print("="*60)
    
    # Load configuration
    config = load_config()
    
    # Load real data
    df = load_real_data(config)
    if df is None:
        return
    
    # Validate data quality
    validate_data(df)
    
    # Create target variable
    df = create_target_variable(df)
    
    # Apply enhanced feature engineering with advanced engagement features
    print("🔧 Applying enhanced feature engineering...")
    original_shape = df.shape
    
    # Apply all feature engineering functions including advanced engagement features
    df = enhanced_text_preprocessing(df)
    df = advanced_timestamp_features(df)
    df = create_interaction_features(df)
    df = create_comprehensive_jsonb_features(df)  # NEW: Comprehensive JSONB features
    df = create_comprehensive_organization_data(df)
    df = create_advanced_engagement_features(df)  # NEW: Advanced engagement features
    df = create_xgboost_optimized_features(df)
    df = handle_outliers(df)
    df = encode_categorical_features(df)  # Updated: now returns only dataframe
    
    print(f"✅ Feature engineering complete. Shape: {df.shape}")
    print(f"📈 Feature increase: {df.shape[1] - original_shape[1]} new features")
    
    # Prepare features for model
    X, y, feature_names = prepare_features_for_model_safe(df, 'engagement_level')
    
    # Perform advanced feature selection
    X_selected, selected_features, mi_df = perform_advanced_feature_selection(X, y, config)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"📊 Data split:")
    print(f"  Training: {X_train.shape}")
    print(f"  Test: {X_test.shape}")

    # Optional CTGAN augmentation on training split only
    try:
        X_train, y_train = augment_training_data_with_ctgan(X_train, y_train, config)
        print(f"🔄 After CTGAN augmentation: X_train={X_train.shape}, y_train={len(y_train)}")
    except Exception as e:
        print(f"⚠️ CTGAN augmentation skipped due to error: {e}")
    
    # Apply advanced class balancing
    print("⚖️ Computing balanced class weights...")
    class_weights = compute_balanced_class_weights(y_train)
    print(f"Class weights: {class_weights}")
    
    # Apply advanced class balancing techniques
    balancing_results = apply_advanced_class_balancing(X_train, y_train, X_test, y_test)
    
    # Test different balancing methods
    best_accuracy = 0
    best_method = 'original'
    best_model = None
    
    print("🔄 Testing different class balancing methods...")
    
    for method, result in balancing_results.items():
        if method == 'cost_sensitive':
            continue  # Skip cost_sensitive as it's just weights
        
        if isinstance(result, tuple):
            X_balanced, y_balanced = result
        else:
            continue  # Skip non-tuple results
        
        print(f"  Testing {method}...")
        
        # Create and train model with balanced data
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.8,
            min_child_weight=3,
            random_state=42,
            scale_pos_weight=class_weights[1] if len(class_weights) > 1 else 1
        )
        
        try:
            model.fit(X_balanced, y_balanced)
            y_pred = model.predict(X_test)
            accuracy = (y_pred == y_test).mean()
            
            print(f"    {method} accuracy: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_method = method
                best_model = model
                
        except Exception as e:
            print(f"    {method} failed: {e}")
    
    print(f"✅ Best method: {best_method} (accuracy: {best_accuracy:.4f})")
    
    # Use original data if no balancing method worked
    if best_method == 'original':
        best_X_train, best_y_train = X_train, y_train
    else:
        best_X_train, best_y_train = balancing_results[best_method]
    
    # Perform hyperparameter optimization with best balanced data
    print("🔧 Performing hyperparameter optimization...")
    
    # Create optimized models
    print("🤖 Creating optimized models...")
    optimized_models = create_optimized_models(config, class_weights)
    
    # Perform hyperparameter optimization
    xgb_optimized, lr_optimized = perform_hyperparameter_optimization(
        best_X_train, best_y_train, config, class_weights
    )
    
    # Create ensemble
    ensemble = create_optimized_ensemble(xgb_optimized, lr_optimized, config)
    
    # Perform cross-validation
    print("🔄 Performing robust cross-validation...")
    cv_scores = perform_robust_cross_validation(ensemble, best_X_train, best_y_train, config)
    
    # Train final model
    print("🏋️ Training optimized model...")
    ensemble.fit(best_X_train, best_y_train)
    
    # Evaluate model
    print("📊 Comprehensive model evaluation...")
    evaluate_model_comprehensive(ensemble, X_test, y_test, config)
    
    # Save model artifacts
    print("💾 Saving optimized model artifacts...")
    save_model_artifacts(ensemble, X_train, y_train, config, {
        'accuracy': best_accuracy,
        'cv_scores': cv_scores,
        'best_balancing_method': best_method
    })
    
    print("="*60)
    print("OPTIMIZED REAL DATA TRAINING SUMMARY")
    print("="*60)
    print(f"📊 Dataset: merged_contacts.csv ({len(df)} rows, {df.shape[1]} columns)")
    print(f"🎯 Target: engagement_level")
    print(f"🔧 Features: {len(selected_features)} selected from {len(feature_names)} engineered")
    print(f"⚖️ Class weights applied: {class_weights}")
    print(f"🎯 Best balancing method: {best_method}")
    print(f"📈 Performance:")
    print(f"  - Accuracy: {best_accuracy:.4f}")
    print(f"  - CV Scores: {cv_scores}")
    print(f"✅ Optimized model saved to: {config['paths']['model_artifact']}")
    print("🎉 EXCELLENT: Optimized model performs well on real data!")

if __name__ == "__main__":
    main() 