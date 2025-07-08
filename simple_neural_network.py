"""
Simplified Neural Network for B2B Email Marketing Engagement Prediction

This script implements a straightforward neural network for predicting email engagement
levels using the merged_contacts.csv dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
from tensorflow.keras.utils import to_categorical

# Configuration
CSV_FILE_PATH = Path("merged_contacts.csv")
TARGET_VARIABLE = "engagement_level"
RANDOM_STATE = 42

# Column definitions
TEXT_COLS = ['campaign_id', 'email_subjects', 'email_bodies']
CATEGORICAL_COLS = ['title', 'seniority', 'organization_industry', 'country', 'city']
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

def load_and_preprocess_data():
    """Load and preprocess the dataset."""
    print("Loading data...")
    df = pd.read_csv(CSV_FILE_PATH, on_bad_lines='warn', low_memory=False)
    print(f"Loaded {len(df)} records")
    
    # Create target variable
    conditions = [
        (df['email_click_count'] > 0),
        (df['email_open_count'] > 0)
    ]
    choices = [2, 1]  # 2=Clicker, 1=Opener, 0=No Engagement
    df[TARGET_VARIABLE] = np.select(conditions, choices, default=0)
    
    # Drop source columns
    df = df.drop(columns=['email_open_count', 'email_click_count'])
    
    return df

def engineer_features(df):
    """Engineer features for the neural network."""
    print("Engineering features...")
    
    # Text features
    combined_text = ""
    for col in TEXT_COLS:
        if col in df.columns:
            combined_text += df[col].fillna('') + ' '
    
    # TF-IDF for text
    tfidf = TfidfVectorizer(max_features=100, stop_words='english', lowercase=True)
    text_features = tfidf.fit_transform(combined_text)
    text_features_df = pd.DataFrame(
        text_features.toarray(),
        index=df.index,
        columns=[f"tfidf_{i}" for i in range(text_features.shape[1])]
    )
    
    # Drop specified columns
    df = df.drop(columns=[col for col in COLS_TO_DROP if col in df.columns], errors='ignore')
    
    # Separate features and target
    y = df[TARGET_VARIABLE]
    X = df.drop(columns=[TARGET_VARIABLE])
    
    # Combine with text features
    X = pd.concat([X, text_features_df], axis=1)
    
    # Handle missing values
    X = X.fillna(0)
    
    print(f"Feature engineering complete. Shape: {X.shape}")
    return X, y

def prepare_neural_data(X, y):
    """Prepare data for neural network."""
    print("Preparing data for neural network...")
    
    # Separate numerical and categorical features
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in CATEGORICAL_COLS if col in X.columns]
    
    # Scale numerical features
    scaler = StandardScaler()
    X_numerical = scaler.fit_transform(X[numerical_cols])
    
    # Encode categorical features
    categorical_data = {}
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].fillna('Missing')
            le = LabelEncoder()
            categorical_data[col] = le.fit_transform(X[col].astype(str))
    
    # Encode target
    y_encoded = to_categorical(y, num_classes=3)
    
    return X_numerical, categorical_data, y_encoded, numerical_cols, categorical_cols

def build_neural_network(numerical_features_count, categorical_features):
    """Build the neural network architecture."""
    print("Building neural network...")
    
    # Numerical input
    numerical_input = layers.Input(shape=(numerical_features_count,), name='numerical_input')
    
    # Categorical inputs with embeddings
    categorical_inputs = []
    categorical_embeddings = []
    
    for col, values in categorical_features.items():
        vocab_size = len(np.unique(values)) + 1  # +1 for unknown values
        input_layer = layers.Input(shape=(1,), name=f'{col}_input')
        embedding_layer = layers.Embedding(
            input_dim=vocab_size,
            output_dim=8,  # Smaller embedding dimension
            name=f'{col}_embedding'
        )(input_layer)
        embedding_layer = layers.Flatten()(embedding_layer)
        
        categorical_inputs.append(input_layer)
        categorical_embeddings.append(embedding_layer)
    
    # Concatenate features
    if categorical_embeddings:
        all_features = layers.Concatenate()([numerical_input] + categorical_embeddings)
    else:
        all_features = numerical_input
    
    # Hidden layers
    x = layers.Dense(64, activation='relu')(all_features)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(32, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    output = layers.Dense(3, activation='softmax')(x)
    
    # Create model
    inputs = [numerical_input] + categorical_inputs
    model = keras.Model(inputs=inputs, output=output)
    
    # Compile
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model architecture:")
    model.summary()
    
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    """Train the neural network."""
    print("Training neural network...")
    
    # Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    print("Evaluating model...")
    
    # Predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, 
                             target_names=['No Engagement', 'Opener', 'Clicker']))
    
    # Confusion matrix
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
    
    return y_pred, y_pred_proba

def main():
    """Main execution function."""
    print("=== B2B Email Marketing Neural Network ===")
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    X, y = engineer_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Prepare neural network data
    X_train_numerical, X_train_categorical, y_train_encoded, numerical_cols, categorical_cols = \
        prepare_neural_data(X_train, y_train)
    
    X_test_numerical, X_test_categorical, y_test_encoded, _, _ = \
        prepare_neural_data(X_test, y_test)
    
    # Build model
    model = build_neural_network(
        numerical_features_count=len(numerical_cols),
        categorical_features=X_train_categorical
    )
    
    # Prepare input data
    X_train_inputs = [X_train_numerical] + [X_train_categorical[col] for col in categorical_cols]
    X_test_inputs = [X_test_numerical] + [X_test_categorical[col] for col in categorical_cols]
    
    # Train model
    history = train_model(model, X_train_inputs, y_train_encoded, X_test_inputs, y_test_encoded)
    
    # Evaluate model
    y_pred, y_pred_proba = evaluate_model(model, X_test_inputs, y_test_encoded)
    
    print("\n=== Training Complete ===")
    print("Model evaluation complete! Check the generated confusion matrix plot.")

if __name__ == "__main__":
    main() 