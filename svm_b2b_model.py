"""
SVM Model for B2B Email Marketing Engagement Prediction

This script implements Support Vector Machine models specifically optimized for 
the B2B email marketing dataset. It includes multiple kernel options, proper
scaling, and handles the large dataset efficiently.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
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

    # 2. Combine Text Columns for later vectorization
    df['combined_text'] = ""
    for col in TEXT_COLS_FOR_FEATURE:
        if col in df.columns:
            df['combined_text'] += df[col].fillna('') + ' '

    # 3. Engineer Timestamp Features without data leakage
    if 'timestamp_created' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp_created']):
        df['created_day_of_week'] = df['timestamp_created'].dt.dayofweek
        df['created_month'] = df['timestamp_created'].dt.month
        ref_date = df['timestamp_created']
        for col in TIMESTAMP_COLS:
            if col in df.columns and col != 'timestamp_created' and pd.api.types.is_datetime64_any_dtype(df[col]):
                feature_name = f"days_between_{col.replace('timestamp_', '')}_and_created"
                df[feature_name] = (df[col] - ref_date).dt.days

    # 4. Engineer JSONB presence features
    for col in JSONB_COLS:
        if col in df.columns:
            df[f'has_{col}'] = df[col].notna().astype(int)

    # 5. Correctly separate features (X) and target (y) to prevent errors
    # FIRST: Secure the target variable 'y' BEFORE it can be dropped.
    y = df[TARGET_VARIABLE]

    # SECOND: Define the full list of ALL columns to drop from the feature set 'X'.
    all_source_cols = ['email_open_count', 'email_click_count', 'email_reply_count']
    all_cols_to_drop = list(set(COLS_TO_DROP + TIMESTAMP_COLS + JSONB_COLS + TEXT_COLS_FOR_FEATURE + all_source_cols + [TARGET_VARIABLE]))

    # THIRD: Define the feature set 'X' by dropping all unnecessary columns.
    X = df.drop(columns=[col for col in all_cols_to_drop if col in df.columns], errors='ignore')

    print(f"Feature engineering complete. Shape before pipeline: {X.shape}")
    return X, y

def create_svm_pipeline(kernel='rbf', feature_selection=True, n_features=1000, numeric_features=None, categorical_features=None):
    """
    Creates an optimized SVM pipeline with proper preprocessing.
    
    Args:
        kernel: SVM kernel ('rbf', 'linear', 'poly', 'sigmoid')
        feature_selection: Whether to apply feature selection
        n_features: Number of features to select if feature_selection=True
        numeric_features: List of numeric feature column names
        categorical_features: List of categorical feature column names
    """
    print(f"Creating SVM pipeline with {kernel} kernel...")
    
    # Default to empty lists if None provided
    if numeric_features is None:
        numeric_features = []
    if categorical_features is None:
        categorical_features = []
    
    # Define feature types
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())  # RobustScaler is less sensitive to outliers than StandardScaler
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', max_categories=20, sparse_output=False))  # Limit categories for SVM
    ])
    
    # Reduced max_features for SVM efficiency
    text_transformer = TfidfVectorizer(
        max_features=300, 
        stop_words='english', 
        lowercase=True, 
        ngram_range=(1, 2),
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.95  # Ignore terms that appear in more than 95% of documents
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
    
    # Pipeline steps
    pipeline_steps = [('preprocessor', preprocessor)]
    
    # Add feature selection if requested
    if feature_selection:
        pipeline_steps.append(('feature_selection', SelectKBest(f_classif, k=min(n_features, 1000))))
    
    # Add SVM classifier
    svm_params = {
        'kernel': kernel,
        'probability': True,  # Enable probability estimates
        'random_state': 42,
        'max_iter': 1000  # Limit iterations for faster training
    }
    
    # Kernel-specific parameters
    if kernel == 'rbf':
        svm_params.update({'gamma': 'scale'})
    elif kernel == 'poly':
        svm_params.update({'degree': 3, 'gamma': 'scale'})
    
    pipeline_steps.append(('svm', SVC(**svm_params)))
    
    return Pipeline(pipeline_steps)

def optimize_svm_hyperparameters(pipeline, X_train, y_train, kernel='rbf'):
    """
    Optimize SVM hyperparameters using RandomizedSearchCV for efficiency.
    """
    print(f"Optimizing hyperparameters for {kernel} SVM...")
    
    # Define parameter grids for different kernels
    if kernel == 'rbf':
        param_grid = {
            'svm__C': [0.1, 1, 10, 100],
            'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        }
    elif kernel == 'linear':
        param_grid = {
            'svm__C': [0.1, 1, 10, 100, 1000]
        }
    elif kernel == 'poly':
        param_grid = {
            'svm__C': [0.1, 1, 10, 100],
            'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'svm__degree': [2, 3, 4]
        }
    else:  # sigmoid
        param_grid = {
            'svm__C': [0.1, 1, 10, 100],
            'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        }
    
    # Use RandomizedSearchCV for efficiency with large dataset
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=20,  # Limited iterations for faster execution
        scoring='accuracy',  # Focus on accuracy as requested
        cv=3,  # Reduced CV folds for speed
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    search.fit(X_train, y_train)
    
    print(f"Best parameters for {kernel} SVM: {search.best_params_}")
    print(f"Best cross-validation accuracy: {search.best_score_:.4f}")
    
    return search.best_estimator_

def evaluate_svm_model(model, X_test, y_test, kernel_name):
    """
    Evaluate SVM model performance.
    """
    print(f"Evaluating {kernel_name} SVM model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Engagement (0)', 'Opener (1)', 'Clicker (2)']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Engagement', 'Opener', 'Clicker'],
                yticklabels=['No Engagement', 'Opener', 'Clicker'])
    plt.title(f'Confusion Matrix - {kernel_name} SVM')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'svm_{kernel_name.lower()}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy, y_pred, y_pred_proba

def main():
    """Main function to run SVM models with different kernels."""
    print("=== SVM Model for B2B Email Marketing ===")
    
    # 1. Load and process data
    df = load_data(CSV_FILE_PATH)
    X, y = engineer_features(df)
    
    # 2. Sample data for SVM efficiency (SVMs don't scale well with very large datasets)
    print("Sampling data for SVM efficiency...")
    if len(X) > 50000:
        sample_size = 50000
        X_sample, _, y_sample, _ = train_test_split(X, y, train_size=sample_size, random_state=42, stratify=y)
        print(f"Using sample of {sample_size} records for SVM training")
        X, y = X_sample, y_sample
    
    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # 4. Identify feature types for preprocessing
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [col for col in CATEGORICAL_COLS if col in X.columns]
    numeric_features = [col for col in numeric_features if col not in categorical_features]
    
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    
    # 5. Test different SVM kernels
    kernels = ['rbf', 'linear', 'poly']  # Excluding sigmoid as it often performs poorly
    best_model = None
    best_accuracy = 0
    best_kernel = None
    
    results = {}
    
    for kernel in kernels:
        print(f"\n{'='*50}")
        print(f"Training {kernel.upper()} SVM")
        print(f"{'='*50}")
        
        try:
            # Create pipeline with proper feature lists
            pipeline = create_svm_pipeline(
                kernel=kernel, 
                feature_selection=True, 
                n_features=800,
                numeric_features=numeric_features,
                categorical_features=categorical_features
            )
            
            # Optimize hyperparameters
            optimized_model = optimize_svm_hyperparameters(pipeline, X_train, y_train, kernel)
            
            # Evaluate model
            accuracy, y_pred, y_pred_proba = evaluate_svm_model(optimized_model, X_test, y_test, kernel.upper())
            
            results[kernel] = {
                'model': optimized_model,
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = optimized_model
                best_kernel = kernel
                
        except Exception as e:
            print(f"Error training {kernel} SVM: {str(e)}")
            continue
    
    # 6. Report best model
    print(f"\n{'='*50}")
    print("SUMMARY RESULTS")
    print(f"{'='*50}")
    
    for kernel, result in results.items():
        print(f"{kernel.upper()} SVM Accuracy: {result['accuracy']:.4f}")
    
    if best_model:
        print(f"\nBest performing model: {best_kernel.upper()} SVM")
        print(f"Best accuracy: {best_accuracy:.4f}")
        
        # Feature importance for linear kernel
        if best_kernel == 'linear':
            try:
                # Get feature importance (coefficients for linear SVM)
                if hasattr(best_model.named_steps['svm'], 'coef_'):
                    coef = best_model.named_steps['svm'].coef_[0]  # For binary classification or first class
                    
                    # Create generic feature names since extracting actual names is complex
                    n_features = len(coef)
                    feature_names = [f'feature_{i}' for i in range(n_features)]
                    
                    # If we have a reasonable number of features, create importance dataframe
                    if n_features > 0:
                        feature_importance = pd.DataFrame({
                            'feature': feature_names,
                            'importance': np.abs(coef)
                        }).sort_values('importance', ascending=False)
                        
                        print(f"\nTop 10 Most Important Features ({best_kernel.upper()} SVM):")
                        print(feature_importance.head(10))
                        
                        # Save feature importance plot
                        plt.figure(figsize=(10, 6))
                        top_features = feature_importance.head(15)
                        plt.barh(range(len(top_features)), top_features['importance'])
                        plt.yticks(range(len(top_features)), top_features['feature'])
                        plt.xlabel('Feature Importance (|Coefficient|)')
                        plt.title(f'Top 15 Feature Importance - {best_kernel.upper()} SVM')
                        plt.gca().invert_yaxis()
                        plt.tight_layout()
                        plt.savefig(f'svm_{best_kernel}_feature_importance.png', dpi=300, bbox_inches='tight')
                        plt.show()
                    else:
                        print("No coefficients found for feature importance analysis")
                else:
                    print("Linear SVM coefficients not available for feature importance")
            except Exception as e:
                print(f"Could not extract feature importance: {str(e)}")
    
    print("\n=== SVM Analysis Complete ===")
    print("Check the generated confusion matrix plots for detailed analysis.")

if __name__ == "__main__":
    main() 