"""
This script trains a multiclass XGBoost model to predict user engagement,
using GridSearchCV to find the optimal hyperparameters before evaluating
the final model.
"""

import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from typing import Tuple

# --- Configuration (Unchanged) ---
CSV_FILE_PATH = Path("merged_contacts.csv")
TARGET_VARIABLE = "engagement_level"
TEXT_COLS = ['campaign_id', 'email_subjects', 'email_bodies']
TIMESTAMP_COLS = ['timestamp_created', 'timestamp_last_contact', 'retrieval_timestamp', 'enriched_at', 'inserted_at', 'last_contacted_from']
CATEGORICAL_COLS = ['title', 'seniority', 'organization_industry', 'country', 'city', 'verification_status', 'enrichment_status', 'upload_method', 'api_status', 'state']
JSONB_COLS = ['employment_history', 'organization_data', 'account_data', 'api_response_raw']
COLS_TO_DROP = [
    'id', 'email', 'first_name', 'last_name', 'company_name', 'linkedin_url',
    'website', 'headline', 'company_domain', 'phone', 'apollo_id',
    'apollo_name', 'organization', 'photo_url', 'organization_name',
    'organization_website', 'organization_phone', 'email_reply_count',
    'email_opened_variant', 'email_opened_step', 'timestamp_last_open',
    'timestamp_last_reply', 'timestamp_last_click', 'timestamp_last_touch',
    'timestamp_last_interest_change', 'timestamp_updated', 'personalization',
    'status_summary', 'payload', 'list_id', 'assigned_to', 'campaign',
    'uploaded_by_user',
]

# --- Data Processing Functions (Unchanged) ---
def load_data(file_path: Path) -> pd.DataFrame:
    """Loads and performs initial datetime standardization on the dataset."""
    print(f"Loading data from '{file_path}'...")
    if not file_path.is_file():
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)
    try:
        df = pd.read_csv(file_path, on_bad_lines='warn', low_memory=False)
        print("Data successfully loaded.")
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
    except Exception as e:
        print(f"An error occurred while loading or parsing the CSV: {e}")
        sys.exit(1)

def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Applies all feature engineering and preprocessing steps."""
    print("Starting feature engineering...")
    if 'email_open_count' not in df.columns or 'email_click_count' not in df.columns:
        print("Error: Source columns 'email_open_count' or 'email_click_count' not found.")
        sys.exit(1)

    conditions = [ (df['email_click_count'] > 0), (df['email_open_count'] > 0) ]
    choices = [2, 1]
    df[TARGET_VARIABLE] = np.select(conditions, choices, default=0)
    df = df.drop(columns=['email_open_count', 'email_click_count'])

    text_features_df = pd.DataFrame(index=df.index)
    combined_text = pd.Series("", index=df.index)
    for col in TEXT_COLS:
        if col in df.columns:
            combined_text += df[col].fillna('') + ' '

    if pd.api.types.is_string_dtype(combined_text) and combined_text.str.strip().any():
        vectorizer = TfidfVectorizer(max_features=500, stop_words='english', lowercase=True)
        text_features_matrix = vectorizer.fit_transform(combined_text)
        text_features_df = pd.DataFrame(text_features_matrix.toarray(), index=df.index, columns=[f"tfidf_{name}" for name in vectorizer.get_feature_names_out()])
    
    current_time_utc = pd.Timestamp.now(tz='UTC')
    for col in TIMESTAMP_COLS:
        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
            feature_name = f"days_since_{col.replace('timestamp_', '')}"
            df[feature_name] = (current_time_utc - df[col]).dt.days
    df = df.drop(columns=[col for col in TIMESTAMP_COLS if col in df.columns])

    for col in JSONB_COLS:
        if col in df.columns:
            df[f'has_{col}'] = df[col].notna().astype(int)
    df = df.drop(columns=[col for col in JSONB_COLS if col in df.columns])

    cols_to_onehot = [col for col in CATEGORICAL_COLS if col in df.columns]
    for col in cols_to_onehot:
        df[col] = df[col].fillna('Missing').astype(str)
        if df[col].nunique() > 50:
            top_n = df[col].value_counts().nlargest(49).index
            df[col] = df[col].where(df[col].isin(top_n), 'Other')
    df = pd.get_dummies(df, columns=cols_to_onehot, dtype=int)

    df = df.drop(columns=[col for col in COLS_TO_DROP if col in df.columns], errors='ignore')
    
    X = df.drop(columns=[TARGET_VARIABLE])
    y = df[TARGET_VARIABLE]
    X = pd.concat([X, text_features_df], axis=1)
    X = X.select_dtypes(include=np.number)
    cols_to_drop_all_nan = X.columns[X.isna().all()].tolist()
    if cols_to_drop_all_nan:
        print(f"INFO: Dropping completely empty columns: {cols_to_drop_all_nan}")
        X = X.drop(columns=cols_to_drop_all_nan)

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, index=X.index, columns=X.columns)
    
    print(f"Feature engineering complete. Final feature shape: {X.shape}")
    return X, y

def main():
    """Main function to run the data pipeline with hyperparameter tuning."""
    # 1. Load and process data
    df = load_data(CSV_FILE_PATH)
    X, y = engineer_features(df)

    # 2. Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # 3. Define the model and hyperparameter grid for GridSearchCV
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )

    # Define a smaller grid for faster demonstration. Expand this for a thorough search.
    param_grid = {
        'max_depth': [4, 6],
        'n_estimators': [100, 250],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # 4. Set up and run GridSearchCV
    # cv=3 means 3-fold cross-validation.
    # n_jobs=-1 uses all available CPU cores to speed up the process.
    # verbose=2 provides progress updates.
    # scoring='f1_weighted' is a good metric for imbalanced multiclass problems.
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='f1_weighted',
        cv=3,
        n_jobs=-1,
        verbose=2
    )

    print("\n--- Starting Hyperparameter Tuning with GridSearchCV ---")
    print(f"WARNING: This process can be very time-consuming.")
    grid_search.fit(X_train, y_train)

    # 5. Report the results of the grid search
    print("\n--- GridSearchCV Results ---")
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation F1-score (weighted): {grid_search.best_score_:.4f}")

    # 6. Evaluate the BEST model on the held-out test set
    print("\n--- Final Model Evaluation on Test Set ---")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Engagement (0)', 'Opener (1)', 'Clicker (2)']))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Engagement', 'Opener', 'Clicker'],
                yticklabels=['No Engagement', 'Opener', 'Clicker'])
    plt.title('Confusion Matrix for Tuned Model')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('tuned_model_confusion_matrix.png')
    print("Tuned model confusion matrix saved as 'tuned_model_confusion_matrix.png'")


if __name__ == "__main__":
    main()