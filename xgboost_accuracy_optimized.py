"""
Accuracy-Optimized XGBoost Model for B2B Email Marketing Engagement Prediction

This script implements an XGBoost model specifically optimized for maximum accuracy
rather than class balance. It removes SMOTE sampling, uses accuracy-focused scoring,
and includes an expanded hyperparameter search space.
"""

import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
CSV_FILE_PATH = Path("merged_contacts.csv")
TARGET_VARIABLE = "engagement_level"
TEXT_COLS_FOR_FEATURE = ['campaign_id', 'email_subjects', 'email_bodies']
TIMESTAMP_COLS = ['timestamp_created', 'timestamp_last_contact', 'retrieval_timestamp', 'enriched_at', 'inserted_at', 'last_contacted_from']
CATEGORICAL_COLS = ['title', 'seniority', 'organization_industry', 'country', 'city', 'enrichment_status', 'upload_method', 'api_status', 'state']
JSONB_COLS = ['employment_history', 'organization_data', 'account_data', 'api_response_raw']

# List of columns to drop due to leakage, irrelevance, or being empty
COLS_TO_DROP = [
    # General/ Personal Identifiers
    'id', 'email', 'first_name', 'last_name', 'company_name', 'linkedin_url',
    'website', 'headline', 'company_domain', 'phone', 'apollo_id',
    'apollo_name', 'organization', 'photo_url', 'organization_name',
    'organization_website', 'organization_phone',
    
    # Leakage Columns
    'email_reply_count', 'email_opened_variant', 'email_opened_step', 
    'timestamp_last_open', 'timestamp_last_reply', 'timestamp_last_click', 
    'timestamp_last_touch', 'timestamp_last_interest_change', 'timestamp_updated', 
    'status_summary', 'email_clicked_variant', 'email_clicked_step',
    
    # Metadata and other unused columns
    'personalization', 'payload', 'list_id', 'assigned_to', 'campaign', 'uploaded_by_user',

    # Empty columns identified from logs
    'auto_variant_select', 'verification_status'
]

# --- Data Loading ---
def load_data(file_path: Path) -> pd.DataFrame:
    """
    Loads and performs initial datetime standardization on the dataset.
    """
    print(f"Loading data from '{file_path}'...")
    if not file_path.is_file():
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)

    df = pd.read_csv(file_path, on_bad_lines='warn', low_memory=False)
    print(f"Data successfully loaded. Shape: {df.shape}")

    # Standardize all potential timestamp columns to UTC to ensure consistency
    all_potential_timestamps = list(set(TIMESTAMP_COLS + [col for col in COLS_TO_DROP if 'timestamp' in col]))
    for col in all_potential_timestamps:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    for col in df.select_dtypes(include=['datetime64[ns]']).columns:
        df[col] = df[col].dt.tz_localize('UTC', nonexistent='NaT')
    for col in df.select_dtypes(include=['datetime64[ns, UTC]']).columns:
        if df[col].dt.tz != 'UTC':
            df[col] = df[col].dt.tz_convert('UTC')
    return df

# --- Feature Engineering ---
def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Applies all feature engineering steps and correctly separates features (X) and target (y).
    """
    print("Starting feature engineering...")

    # 1. Define Target Variable: 3-tier system where clicks and replies are bucketed
    if 'email_reply_count' not in df.columns or 'email_click_count' not in df.columns or 'email_open_count' not in df.columns:
        print("Error: Source columns for engagement level ('email_reply_count', 'email_click_count', 'email_open_count') not found.")
        sys.exit(1)

    conditions = [
        ((df['email_click_count'] > 0) | (df['email_reply_count'] > 0)),  # Tier 2: Click OR Reply
        (df['email_open_count'] > 0)                                       # Tier 1: Open
    ]
    choices = [2, 1]
    df[TARGET_VARIABLE] = np.select(conditions, choices, default=0)         # Tier 0: No Engagement

    # Print class distribution
    print("\nClass Distribution:")
    class_counts = df[TARGET_VARIABLE].value_counts().sort_index()
    for class_val, count in class_counts.items():
        percentage = (count / len(df)) * 100
        class_name = ['No Engagement', 'Opener', 'Clicker'][class_val]
        print(f"  {class_name} ({class_val}): {count:,} ({percentage:.1f}%)")

    # 2. Combine Text Columns for later vectorization
    df['combined_text'] = ""
    for col in TEXT_COLS_FOR_FEATURE:
        if col in df.columns:
            df['combined_text'] += df[col].fillna('') + ' '

    # 3. Engineer Timestamp Features without data leakage
    if 'timestamp_created' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp_created']):
        df['created_day_of_week'] = df['timestamp_created'].dt.dayofweek
        df['created_month'] = df['timestamp_created'].dt.month
        df['created_hour'] = df['timestamp_created'].dt.hour
        ref_date = df['timestamp_created']
        for col in TIMESTAMP_COLS:
            if col in df.columns and col != 'timestamp_created' and pd.api.types.is_datetime64_any_dtype(df[col]):
                feature_name = f"days_between_{col.replace('timestamp_', '')}_and_created"
                df[feature_name] = (df[col] - ref_date).dt.days

    # 4. Engineer JSONB presence features
    for col in JSONB_COLS:
        if col in df.columns:
            df[f'has_{col}'] = df[col].notna().astype(int)

    # 5. Create interaction features for key categorical variables
    # Create industry-seniority interaction
    if 'organization_industry' in df.columns and 'seniority' in df.columns:
        df['industry_seniority'] = df['organization_industry'].fillna('Unknown') + '_' + df['seniority'].fillna('Unknown')
    
    # Create country-industry interaction
    if 'country' in df.columns and 'organization_industry' in df.columns:
        df['country_industry'] = df['country'].fillna('Unknown') + '_' + df['organization_industry'].fillna('Unknown')

    # 6. Correctly separate features (X) and target (y) to prevent errors
    # FIRST: Secure the target variable 'y' BEFORE it can be dropped.
    y = df[TARGET_VARIABLE]

    # SECOND: Define the full list of ALL columns to drop from the feature set 'X'.
    all_source_cols = ['email_open_count', 'email_click_count', 'email_reply_count']
    all_cols_to_drop = list(set(COLS_TO_DROP + TIMESTAMP_COLS + JSONB_COLS + TEXT_COLS_FOR_FEATURE + all_source_cols + [TARGET_VARIABLE]))

    # THIRD: Define the feature set 'X' by dropping all unnecessary columns.
    X = df.drop(columns=[col for col in all_cols_to_drop if col in df.columns], errors='ignore')

    print(f"Feature engineering complete. Shape before pipeline: {X.shape}")
    return X, y

def create_accuracy_optimized_pipeline(use_feature_selection=True, max_text_features=1000, numeric_features=None, categorical_features=None):
    """
    Creates an accuracy-optimized XGBoost pipeline without SMOTE sampling.
    """
    print("Creating accuracy-optimized XGBoost pipeline...")
    
    # Default to empty lists if None provided
    if numeric_features is None:
        numeric_features = []
    if categorical_features is None:
        categorical_features = []
    
    # Enhanced preprocessing for numerical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        # Note: XGBoost handles feature scaling internally, but we can optionally include it
    ])

    # Enhanced categorical preprocessing
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', max_categories=100, sparse_output=False))
    ])
    
    # Enhanced text processing with more features for accuracy
    text_transformer = TfidfVectorizer(
        max_features=max_text_features, 
        stop_words='english', 
        lowercase=True, 
        ngram_range=(1, 3),  # Include trigrams for more context
        min_df=2,
        max_df=0.95,
        sublinear_tf=True  # Apply sublinear tf scaling
    )

    # Create preprocessor with actual feature lists
    transformers = []
    if numeric_features:
        transformers.append(('num', numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(('cat', categorical_transformer, categorical_features))
    transformers.append(('text', text_transformer, 'combined_text'))
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )
    
    # Pipeline steps - NO SMOTE for accuracy optimization
    pipeline_steps = [('preprocessor', preprocessor)]
    
    # Add feature selection if requested
    if use_feature_selection:
        pipeline_steps.append(('feature_selection', SelectKBest(f_classif, k=2000)))
    
    # Add XGBoost classifier optimized for accuracy
    xgb_classifier = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        # Initial parameters favoring accuracy
        tree_method='hist',  # Faster training
        enable_categorical=False
    )
    
    pipeline_steps.append(('classifier', xgb_classifier))
    
    return Pipeline(pipeline_steps)

def calculate_class_weights(y):
    """
    Calculate class weights to handle imbalance without sampling.
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    weight_dict = dict(zip(classes, class_weights))
    
    print("Calculated class weights:")
    for class_val, weight in weight_dict.items():
        class_name = ['No Engagement', 'Opener', 'Clicker'][class_val]
        print(f"  {class_name} ({class_val}): {weight:.3f}")
    
    return weight_dict

def run_comprehensive_hyperparameter_search(pipeline, X_train, y_train, search_type='randomized'):
    """
    Run comprehensive hyperparameter search optimized for accuracy.
    """
    print(f"Running {search_type} hyperparameter search optimized for accuracy...")
    
    # Expanded parameter distributions for accuracy optimization
    param_distributions = {
        # Core XGBoost parameters
        'classifier__max_depth': [3, 4, 5, 6, 7, 8, 10],
        'classifier__n_estimators': [100, 200, 300, 500, 700, 1000],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
        
        # Regularization parameters
        'classifier__subsample': [0.7, 0.8, 0.9, 1.0],
        'classifier__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'classifier__colsample_bylevel': [0.7, 0.8, 0.9, 1.0],
        'classifier__colsample_bynode': [0.7, 0.8, 0.9, 1.0],
        
        # Advanced parameters for accuracy
        'classifier__reg_alpha': [0, 0.01, 0.1, 1, 10],  # L1 regularization
        'classifier__reg_lambda': [0, 0.01, 0.1, 1, 10],  # L2 regularization
        'classifier__min_child_weight': [1, 3, 5, 7],
        'classifier__gamma': [0, 0.01, 0.1, 0.5, 1],
        
        # Text processing parameters
        'preprocessor__text__max_features': [500, 800, 1000, 1500],
        'preprocessor__text__ngram_range': [(1, 2), (1, 3)],
        
        # Feature selection parameters
        'feature_selection__k': [1000, 1500, 2000, 'all']
    }
    
    if search_type == 'randomized':
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=100,  # Increased iterations for thorough search
            scoring='accuracy',  # PRIMARY CHANGE: Focus on accuracy
            cv=5,
            n_jobs=-1,
            verbose=2,
            random_state=42,
            return_train_score=True
        )
    else:  # Grid search for final tuning
        # Reduced parameter grid for grid search
        param_grid = {
            'classifier__max_depth': [4, 6, 8],
            'classifier__n_estimators': [300, 500, 700],
            'classifier__learning_rate': [0.05, 0.1, 0.15],
            'classifier__subsample': [0.8, 0.9, 1.0],
            'classifier__colsample_bytree': [0.8, 0.9, 1.0],
        }
        
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring='accuracy',
            cv=3,
            n_jobs=-1,
            verbose=2,
            return_train_score=True
        )
    
    search.fit(X_train, y_train)
    
    print(f"\nBest parameters found: {search.best_params_}")
    print(f"Best cross-validation accuracy: {search.best_score_:.4f}")
    print(f"Best training accuracy: {search.cv_results_['mean_train_score'][search.best_index_]:.4f}")
    
    return search.best_estimator_, search

def evaluate_accuracy_model(model, X_test, y_test, model_name="Accuracy-Optimized XGBoost"):
    """
    Comprehensive evaluation focused on accuracy metrics.
    """
    print(f"Evaluating {model_name}...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Engagement (0)', 'Opener (1)', 'Clicker (2)']))
    
    # Class-wise accuracy
    print("\nClass-wise Accuracy:")
    for class_val in [0, 1, 2]:
        mask = (y_test == class_val)
        if mask.sum() > 0:
            class_accuracy = (y_pred[mask] == class_val).mean()
            class_name = ['No Engagement', 'Opener', 'Clicker'][class_val]
            print(f"  {class_name}: {class_accuracy:.4f} ({mask.sum()} samples)")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Engagement', 'Opener', 'Clicker'],
                yticklabels=['No Engagement', 'Opener', 'Clicker'])
    plt.title(f'Confusion Matrix - {model_name}\nAccuracy: {accuracy:.4f}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('xgboost_accuracy_optimized_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy, y_pred, y_pred_proba

def extract_feature_importance(model, feature_names=None):
    """
    Extract and visualize feature importance from the trained model.
    """
    print("Extracting feature importance...")
    
    try:
        # Get XGBoost feature importance
        xgb_model = model.named_steps['classifier']
        importance_scores = xgb_model.feature_importances_
        
        # Create feature importance dataframe
        if feature_names is not None and len(feature_names) == len(importance_scores):
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_scores
            }).sort_values('importance', ascending=False)
        else:
            feature_importance = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(importance_scores))],
                'importance': importance_scores
            }).sort_values('importance', ascending=False)
        
        print("\nTop 15 Most Important Features:")
        print(feature_importance.head(15))
        
        # Visualize feature importance
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Feature Importance - Accuracy-Optimized XGBoost')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('xgboost_accuracy_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_importance
        
    except Exception as e:
        print(f"Could not extract feature importance: {str(e)}")
        return None

def main():
    """Main function to run accuracy-optimized XGBoost model."""
    print("=== Accuracy-Optimized XGBoost for B2B Email Marketing ===")
    
    # 1. Load and process data
    df = load_data(CSV_FILE_PATH)
    X, y = engineer_features(df)
    
    # 2. Split data BEFORE any fitting/preprocessing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"\nData split complete:")
    print(f"  Training set: {X_train.shape[0]:,} samples")
    print(f"  Test set: {X_test.shape[0]:,} samples")

    # 3. Calculate class weights for handling imbalance
    class_weights = calculate_class_weights(y_train)

    # 4. Identify feature types for preprocessing
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [col for col in CATEGORICAL_COLS + ['industry_seniority', 'country_industry'] if col in X.columns]
    numeric_features = [col for col in numeric_features if col not in categorical_features]
    
    print(f"\nFeature breakdown:")
    print(f"  Numeric features: {len(numeric_features)}")
    print(f"  Categorical features: {len(categorical_features)}")
    print(f"  Text features: 1 (combined_text)")

    # 5. Create accuracy-optimized pipeline with proper feature lists
    pipeline = create_accuracy_optimized_pipeline(
        use_feature_selection=True, 
        max_text_features=1000,
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

    # 6. Run comprehensive hyperparameter search
    print(f"\n{'='*60}")
    print("PHASE 1: Randomized Search for Optimal Parameters")
    print(f"{'='*60}")
    
    best_model, search_results = run_comprehensive_hyperparameter_search(
        pipeline, X_train, y_train, search_type='randomized'
    )
    
    # 7. Fine-tune with grid search around best parameters
    print(f"\n{'='*60}")
    print("PHASE 2: Model Evaluation")
    print(f"{'='*60}")
    
    # Evaluate the best model
    accuracy, y_pred, y_pred_proba = evaluate_accuracy_model(best_model, X_test, y_test)
    
    # 8. Extract and visualize feature importance
    feature_importance = extract_feature_importance(best_model)
    
    # 9. Compare with class weights approach
    print(f"\n{'='*60}")
    print("PHASE 3: Class Weights Comparison")
    print(f"{'='*60}")
    
    # Create a model with class weights
    weighted_pipeline = create_accuracy_optimized_pipeline(
        use_feature_selection=True, 
        max_text_features=1000,
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )
    
    # Set class weights
    sample_weights = np.array([class_weights[label] for label in y_train])
    
    # Use best parameters from search but with class weights
    best_params = search_results.best_params_.copy()
    
    # Train weighted model
    weighted_pipeline.set_params(**best_params)
    weighted_pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)
    
    # Evaluate weighted model
    weighted_accuracy, _, _ = evaluate_accuracy_model(weighted_pipeline, X_test, y_test, "Class-Weighted XGBoost")
    
    # 10. Final comparison
    print(f"\n{'='*60}")
    print("FINAL RESULTS COMPARISON")
    print(f"{'='*60}")
    print(f"Standard Accuracy-Optimized XGBoost: {accuracy:.4f}")
    print(f"Class-Weighted XGBoost:              {weighted_accuracy:.4f}")
    
    if accuracy > weighted_accuracy:
        print("\n✅ Standard model performs better - prioritizing majority class prediction")
        final_model = best_model
        final_accuracy = accuracy
    else:
        print("\n✅ Class-weighted model performs better - balanced accuracy approach")
        final_model = weighted_pipeline
        final_accuracy = weighted_accuracy
    
    print(f"\nFinal Model Accuracy: {final_accuracy:.4f}")
    
    # 11. Save model insights
    print(f"\n{'='*60}")
    print("MODEL INSIGHTS")
    print(f"{'='*60}")
    
    print("Key Optimizations Applied:")
    print("  ✓ Removed SMOTE sampling to focus on natural class distribution")
    print("  ✓ Used accuracy as primary scoring metric")
    print("  ✓ Expanded hyperparameter search space")
    print("  ✓ Enhanced text processing with trigrams")
    print("  ✓ Added interaction features")
    print("  ✓ Compared standard vs class-weighted approaches")
    
    if feature_importance is not None:
        print(f"\nTop 5 Most Predictive Features:")
        for i, (_, row) in enumerate(feature_importance.head(5).iterrows()):
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    print("\n=== Analysis Complete ===")
    print("Generated files:")
    print("  - xgboost_accuracy_optimized_confusion_matrix.png")
    print("  - xgboost_accuracy_feature_importance.png")

if __name__ == "__main__":
    main() 