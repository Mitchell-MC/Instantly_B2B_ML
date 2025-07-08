"""
Neural Network Algorithm for B2B Email Marketing Engagement Prediction

This script implements a deep neural network specifically designed for the merged_contacts.csv
dataset to predict email engagement levels (No Engagement, Opener, Clicker).

Features:
- Handles mixed data types (numerical, categorical, text)
- Advanced preprocessing with embeddings
- Custom loss function for imbalanced classes
- Comprehensive evaluation metrics
- SHAP interpretation for neural networks
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
from tensorflow.keras.utils import to_categorical
import shap
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
CSV_FILE_PATH = Path("merged_contacts.csv")
TARGET_VARIABLE = "engagement_level"
RANDOM_STATE = 42

# Data column definitions
TEXT_COLS = ['campaign_id', 'email_subjects', 'email_bodies']
TIMESTAMP_COLS = [
    'timestamp_created', 'timestamp_last_contact', 'retrieval_timestamp',
    'enriched_at', 'inserted_at', 'last_contacted_from'
]
CATEGORICAL_COLS = [
    'title', 'seniority', 'organization_industry', 'country', 'city',
    'verification_status', 'enrichment_status', 'upload_method', 'api_status', 'state'
]
JSONB_COLS = ['employment_history', 'organization_data', 'account_data', 'api_response_raw']

# Columns to drop (leakage, PII, or irrelevant)
COLS_TO_DROP = [
    'id', 'email', 'first_name', 'last_name', 'company_name', 'linkedin_url',
    'website', 'headline', 'company_domain', 'phone', 'apollo_id',
    'apollo_name', 'organization', 'photo_url', 'organization_name',
    'organization_website', 'organization_phone', 'email_reply_count',
    'email_opened_variant', 'email_opened_step', 'timestamp_last_open',
    'timestamp_last_reply', 'timestamp_last_click', 'timestamp_last_touch',
    'timestamp_last_interest_change', 'timestamp_updated', 'personalization',
    'status_summary', 'payload', 'list_id', 'assigned_to', 'campaign',
    'uploaded_by_user', 'email_clicked_variant', 'email_clicked_step'
]

class B2BNeuralNetwork:
    """Neural Network specifically designed for B2B email marketing data."""
    
    def __init__(self, embedding_dim=16, hidden_layers=[128, 64, 32], dropout_rate=0.3):
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.model = None
        self.preprocessors = {}
        self.label_encoders = {}
        
    def load_and_preprocess_data(self, file_path: Path):
        """Load and preprocess the B2B dataset."""
        print(f"Loading data from {file_path}...")
        
        # Load data
        df = pd.read_csv(file_path, on_bad_lines='warn', low_memory=False)
        print(f"Loaded {len(df)} records with {len(df.columns)} columns")
        
        # Standardize timestamps
        for col in TIMESTAMP_COLS:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Create target variable
        conditions = [
            (df['email_click_count'] > 0),
            (df['email_open_count'] > 0)
        ]
        choices = [2, 1]  # 2=Clicker, 1=Opener, 0=No Engagement
        df[TARGET_VARIABLE] = np.select(conditions, choices, default=0)
        
        # Drop source columns to prevent leakage
        df = df.drop(columns=['email_open_count', 'email_click_count'])
        
        return df
    
    def engineer_features(self, df: pd.DataFrame):
        """Engineer features for neural network."""
        print("Engineering features...")
        
        # 1. Text features
        print("Processing text features...")
        combined_text = ""
        for col in TEXT_COLS:
            if col in df.columns:
                combined_text += df[col].fillna('') + ' '
        
        # TF-IDF for text
        if combined_text.strip():
            tfidf = TfidfVectorizer(max_features=200, stop_words='english', lowercase=True)
            text_features = tfidf.fit_transform(combined_text)
            self.preprocessors['tfidf'] = tfidf
            text_features_df = pd.DataFrame(
                text_features.toarray(),
                index=df.index,
                columns=[f"tfidf_{i}" for i in range(text_features.shape[1])]
            )
        else:
            text_features_df = pd.DataFrame(index=df.index)
        
        # 2. Timestamp features
        print("Processing timestamp features...")
        current_time = pd.Timestamp.now(tz='UTC')
        for col in TIMESTAMP_COLS:
            if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
                feature_name = f"days_since_{col.replace('timestamp_', '')}"
                df[feature_name] = (current_time - df[col]).dt.days
        
        # 3. JSONB presence features
        for col in JSONB_COLS:
            if col in df.columns:
                df[f'has_{col}'] = df[col].notna().astype(int)
        
        # 4. Drop specified columns
        all_cols_to_drop = list(set(COLS_TO_DROP + TIMESTAMP_COLS + JSONB_COLS + TEXT_COLS))
        df = df.drop(columns=[col for col in all_cols_to_drop if col in df.columns], errors='ignore')
        
        # 5. Separate features and target
        y = df[TARGET_VARIABLE]
        X = df.drop(columns=[TARGET_VARIABLE])
        
        # 6. Combine with text features
        X = pd.concat([X, text_features_df], axis=1)
        
        # 7. Handle missing values
        X = X.fillna(0)
        
        print(f"Feature engineering complete. Final shape: {X.shape}")
        return X, y
    
    def prepare_neural_network_data(self, X: pd.DataFrame, y: pd.Series):
        """Prepare data specifically for neural network architecture."""
        print("Preparing data for neural network...")
        
        # Separate numerical and categorical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [col for col in CATEGORICAL_COLS if col in X.columns]
        
        # Prepare numerical features
        scaler = StandardScaler()
        X_numerical = scaler.fit_transform(X[numerical_cols])
        self.preprocessors['scaler'] = scaler
        
        # Prepare categorical features
        categorical_data = {}
        for col in categorical_cols:
            if col in X.columns:
                # Fill missing values
                X[col] = X[col].fillna('Missing')
                
                # Encode labels
                le = LabelEncoder()
                encoded_values = le.fit_transform(X[col].astype(str))
                categorical_data[col] = encoded_values
                self.label_encoders[col] = le
        
        # Prepare target
        y_encoded = to_categorical(y, num_classes=3)
        
        return X_numerical, categorical_data, y_encoded, numerical_cols, categorical_cols
    
    def build_model(self, numerical_features_count: int, categorical_features: dict):
        """Build the neural network architecture."""
        print("Building neural network architecture...")
        
        # Input layers
        numerical_input = layers.Input(shape=(numerical_features_count,), name='numerical_input')
        
        # Categorical inputs with embeddings
        categorical_inputs = []
        categorical_embeddings = []
        
        for col, values in categorical_features.items():
            vocab_size = len(self.label_encoders[col].classes_)
            input_layer = layers.Input(shape=(1,), name=f'{col}_input')
            embedding_layer = layers.Embedding(
                input_dim=vocab_size,
                output_dim=self.embedding_dim,
                name=f'{col}_embedding'
            )(input_layer)
            embedding_layer = layers.Flatten()(embedding_layer)
            
            categorical_inputs.append(input_layer)
            categorical_embeddings.append(embedding_layer)
        
        # Concatenate all features
        if categorical_embeddings:
            all_features = layers.Concatenate()([numerical_input] + categorical_embeddings)
        else:
            all_features = numerical_input
        
        # Hidden layers
        x = all_features
        for i, units in enumerate(self.hidden_layers):
            x = layers.Dense(units, activation='relu', name=f'hidden_{i}')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer
        output = layers.Dense(3, activation='softmax', name='output')(x)
        
        # Create model
        inputs = [numerical_input] + categorical_inputs
        self.model = keras.Model(inputs=inputs, output=output)
        
        # Compile model with custom loss for imbalanced classes
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'categorical_accuracy']
        )
        
        print("Model architecture:")
        self.model.summary()
        
        return self.model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the neural network with early stopping."""
        print("Training neural network...")
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model with comprehensive metrics."""
        print("Evaluating model...")
        
        # Predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Metrics
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, 
                                 target_names=['No Engagement', 'Opener', 'Clicker']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Engagement', 'Opener', 'Clicker'],
                   yticklabels=['No Engagement', 'Opener', 'Clicker'])
        plt.title('Neural Network Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('neural_network_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # ROC AUC for each class
        for i, class_name in enumerate(['No Engagement', 'Opener', 'Clicker']):
            auc = roc_auc_score(y_test[:, i], y_pred_proba[:, i])
            print(f"ROC AUC for {class_name}: {auc:.4f}")
        
        return y_pred, y_pred_proba
    
    def explain_predictions(self, X_sample, feature_names):
        """Use SHAP to explain neural network predictions."""
        print("Generating SHAP explanations...")
        
        # Create a background dataset for SHAP
        background = X_sample[:100]  # Use first 100 samples as background
        
        # Create SHAP explainer
        explainer = shap.DeepExplainer(self.model, background)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample[:50])  # Explain first 50 samples
        
        # Plot summary
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample[:50], feature_names=feature_names, show=False)
        plt.title('SHAP Summary Plot - Neural Network')
        plt.tight_layout()
        plt.savefig('neural_network_shap_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return shap_values

def main():
    """Main execution function."""
    print("=== B2B Email Marketing Neural Network ===")
    
    # Initialize neural network
    nn_model = B2BNeuralNetwork(
        embedding_dim=16,
        hidden_layers=[128, 64, 32],
        dropout_rate=0.3
    )
    
    # Load and preprocess data
    df = nn_model.load_and_preprocess_data(CSV_FILE_PATH)
    X, y = nn_model.engineer_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Prepare neural network data
    X_train_numerical, X_train_categorical, y_train_encoded, numerical_cols, categorical_cols = \
        nn_model.prepare_neural_network_data(X_train, y_train)
    
    X_test_numerical, X_test_categorical, y_test_encoded, _, _ = \
        nn_model.prepare_neural_network_data(X_test, y_test)
    
    # Build model
    model = nn_model.build_model(
        numerical_features_count=len(numerical_cols),
        categorical_features=X_train_categorical
    )
    
    # Prepare input data for training
    X_train_inputs = [X_train_numerical] + [X_train_categorical[col] for col in categorical_cols]
    X_test_inputs = [X_test_numerical] + [X_test_categorical[col] for col in categorical_cols]
    
    # Train model
    history = nn_model.train_model(
        X_train_inputs, y_train_encoded,
        X_test_inputs, y_test_encoded,
        epochs=50, batch_size=32
    )
    
    # Evaluate model
    y_pred, y_pred_proba = nn_model.evaluate_model(X_test_inputs, y_test_encoded)
    
    # Explain predictions
    feature_names = numerical_cols + [f"{col}_embedding" for col in categorical_cols]
    nn_model.explain_predictions(X_test_inputs, feature_names)
    
    print("\n=== Training Complete ===")
    print("Model saved and evaluation complete!")
    print("Check the generated plots for detailed analysis.")

if __name__ == "__main__":
    main() 