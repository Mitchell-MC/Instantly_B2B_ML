"""
This script loads contact data, preprocesses it to create features, trains a
multiclass XGBoost model to predict user engagement level (No Engagement,
Opener, Clicker), and then uses SHAP to interpret the model's predictions.
"""

import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple

# --- Configuration ---
CSV_FILE_PATH = Path("merged_contacts.csv")
# MODIFIED: Changed target variable to reflect multiclass goal
TARGET_VARIABLE = "engagement_level"

# Define column types for processing
TEXT_COLS = [
    'campaign_id',
    'email_subjects',
    'email_bodies'
]
TIMESTAMP_COLS = [
    'timestamp_created',
    'timestamp_last_contact',
    'retrieval_timestamp',
    'enriched_at',
    'inserted_at',
    'last_contacted_from'
]
CATEGORICAL_COLS = [
    'title', 'seniority', 'organization_industry', 'country', 'city',
    'verification_status', 'enrichment_status', 'upload_method', 'api_status',
    'state'
]
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
]

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
    
    # --- KEY MODIFICATION: Create a multiclass target variable ---
    print("Creating multiclass engagement target...")
    if 'email_open_count' not in df.columns or 'email_click_count' not in df.columns:
        print("Error: Source columns 'email_open_count' or 'email_click_count' not found.")
        sys.exit(1)

    # Define conditions for our engagement tiers:
    # Class 2: Clicker (highest engagement, clicks on link and/or replies)
    # Class 1: Opener (medium engagement)
    # Class 0: No Engagement (default)
    conditions = [
        (df['email_click_count'] > 0),
        (df['email_open_count'] > 0)
    ]
    choices = [2, 1]
    df[TARGET_VARIABLE] = np.select(conditions, choices, default=0)
    
    # Drop original leaky columns
    df = df.drop(columns=['email_open_count', 'email_click_count'])
    print("Target variable 'engagement_level' created.")
    # --- END MODIFICATION ---

    print("Processing text features with TF-IDF...")
    text_features_df = pd.DataFrame(index=df.index)
    combined_text = pd.Series("", index=df.index)
    for col in TEXT_COLS:
        if col in df.columns:
            combined_text += df[col].fillna('') + ' '
        else:
            print(f"Warning: Text column '{col}' not found. Skipping.")

    if pd.api.types.is_string_dtype(combined_text) and combined_text.str.strip().any():
        vectorizer = TfidfVectorizer(max_features=500, stop_words='english', lowercase=True)
        text_features_matrix = vectorizer.fit_transform(combined_text)
        text_features_df = pd.DataFrame(
            text_features_matrix.toarray(),
            index=df.index,
            columns=[f"tfidf_{name}" for name in vectorizer.get_feature_names_out()]
        )
        print(f"Created {text_features_df.shape[1]} features from text columns.")
    else:
        print("No text data to process.")

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
    
    if TARGET_VARIABLE not in df.columns:
        print(f"Error: Target variable '{TARGET_VARIABLE}' not found after processing.")
        sys.exit(1)
    
    X = df.drop(columns=[TARGET_VARIABLE])
    y = df[TARGET_VARIABLE]

    X = pd.concat([X, text_features_df], axis=1)
    print("Combined original features with new TF-IDF text features.")

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
    """Main function to run the full data science pipeline."""
    # 1. Load and process data
    df = load_data(CSV_FILE_PATH)
    X, y = engineer_features(df)

    # 2. Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data split into training and testing sets. Test set shape: {X_test.shape}")

    # 3. Train the XGBoost Multiclass Model
    print("Training XGBoost multiclass model...")
    # --- KEY MODIFICATION: Configure XGBoost for multiclass classification ---
    model = xgb.XGBClassifier(
        objective='multi:softmax',  # Specifies multiclass prediction
        num_class=3,                # Number of classes: 0, 1, 2
        use_label_encoder=False,    # Recommended setting
        eval_metric='mlogloss'      # Logloss for multiclass models
    )
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 4. Explain the model's predictions with SHAP
    print("Calculating SHAP values to explain model predictions...")
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    # For multiclass, SHAP returns a list of arrays (one for each class).
    # We can visualize the summary plot for each class or combine them.
    # The default summary plot shows the mean absolute SHAP value across all classes.
    print("Generating and saving SHAP summary plot...")
    plt.figure()
    # Class 2 is the "Clickers", our most important group. Let's plot the features for it.
    shap.summary_plot(shap_values[:,:,2], X_test, show=False)
    plt.title('SHAP Summary for High-Engagement "Clickers" (Class 2)')
    plt.tight_layout()
    plt.savefig("shap_summary_plot_clickers.png")
    print("SHAP plot saved as 'shap_summary_plot_clickers.png'")

    # You can now use the trained `model` to predict on new, unseen data
    # example_prediction = model.predict(new_data) -> will return 0, 1, or 2

if __name__ == "__main__":
    main()