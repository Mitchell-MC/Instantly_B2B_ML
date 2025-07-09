"""
This script trains a multiclass XGBoost model to predict user engagement.
It uses RandomizedSearchCV for efficient hyperparameter tuning and a SMOTE 
pipeline to handle class imbalance and prevent data leakage during preprocessing.
"""

import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from typing import Tuple

# Import pipeline and SMOTE from imbalanced-learn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

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
    print("Data successfully loaded.")

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

# --- Main Execution ---
def main():
    """Main function to run the data pipeline with hyperparameter tuning."""
    # 1. Load and process data
    df = load_data(CSV_FILE_PATH)
    X, y = engineer_features(df)
    
    # 2. Split data BEFORE any fitting/preprocessing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # 3. Define preprocessing pipelines for different column types
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [col for col in CATEGORICAL_COLS if col in X.columns]
    text_feature = 'combined_text'
    numeric_features = [col for col in numeric_features if col not in categorical_features]

    numeric_transformer = ImbPipeline(steps=[('imputer', SimpleImputer(strategy='mean'))])
    categorical_transformer = ImbPipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', max_categories=50))
    ])
    text_transformer = TfidfVectorizer(max_features=500, stop_words='english', lowercase=True, ngram_range=(1, 2))

    # Create a preprocessor object using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('text', text_transformer, text_feature)
        ],
        remainder='drop'
    )

    # 4. Create the full pipeline with preprocessor, SMOTE, and model
    model_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    # Replace SMOTE with SMOTETomek
    ('sampler', SMOTETomek(random_state=42)),
    ('classifier', xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        eval_metric='mlogloss',
        use_label_encoder=False
    ))
])
    
    # 5. Define hyperparameter grid for RandomizedSearchCV
    param_distributions = {
        'classifier__max_depth': [4, 6, 8, 10],
        'classifier__n_estimators': [150, 250, 350, 500],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.15],
        'classifier__subsample': [0.7, 0.8, 0.9, 1.0],
        'classifier__colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    }

    # 6. Set up and run RandomizedSearchCV for efficient tuning
    search = RandomizedSearchCV(
        estimator=model_pipeline,
        param_distributions=param_distributions,
        n_iter=25, # Test 25 random combinations
        scoring='f1_macro', # Focus on performance across all classes
        cv=5, # Use 5-fold cross-validation
        n_jobs=5,
        verbose=2,
        random_state=42
    )

    print("\n--- Starting Hyperparameter Tuning with RandomizedSearchCV ---")
    search.fit(X_train, y_train)

    # 7. Report results and evaluate the best model
    print("\n--- Tuning Results ---")
    print(f"Best parameters found: {search.best_params_}")
    print(f"Best cross-validation F1-score (macro): {search.best_score_:.4f}")

    print("\n--- Final Model Evaluation on Test Set ---")
    y_pred = search.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Engagement (0)', 'Opener (1)', 'Clicker (2)']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Engagement', 'Opener', 'Clicker'],
                yticklabels=['No Engagement', 'Opener', 'Clicker'])
    plt.title('Confusion Matrix for Final Tuned Model')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('ultimate_confusion_matrix.png')
    print("Confusion matrix plot saved as 'ultimate_confusion_matrix.png'")
    plt.show()

if __name__ == "__main__":
    main()