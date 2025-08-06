"""
Feature Name Mapper
Maps generic feature indices back to meaningful business feature names
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

def extract_feature_names_from_pipeline(pipeline, X_sample):
    """
    Extract meaningful feature names from a fitted sklearn pipeline.
    
    Args:
        pipeline: Fitted sklearn pipeline
        X_sample: Sample of original X data (before preprocessing)
        
    Returns:
        List of meaningful feature names
    """
    feature_names = []
    
    # Get the preprocessor (first step in pipeline)
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Get feature names from each transformer
    for name, transformer, columns in preprocessor.transformers_:
        
        if name == 'num':
            # Numeric features keep their original names
            if isinstance(columns, list):
                feature_names.extend([f"num__{col}" for col in columns])
            else:
                feature_names.extend([f"num__{col}" for col in [columns]])
                
        elif name == 'cat':
            # Categorical features get expanded by one-hot encoding
            if hasattr(transformer.named_steps['onehot'], 'get_feature_names_out'):
                # For newer sklearn versions
                cat_features = transformer.named_steps['onehot'].get_feature_names_out(columns)
                feature_names.extend([f"cat__{feat}" for feat in cat_features])
            else:
                # For older sklearn versions
                cat_features = transformer.named_steps['onehot'].get_feature_names(columns)
                feature_names.extend([f"cat__{feat}" for feat in cat_features])
                
        elif name == 'text':
            # Text features from TF-IDF
            if hasattr(transformer, 'get_feature_names_out'):
                text_features = transformer.get_feature_names_out()
                feature_names.extend([f"text__{feat}" for feat in text_features])
            else:
                text_features = transformer.get_feature_names()
                feature_names.extend([f"text__{feat}" for feat in text_features])
    
    return feature_names

def map_feature_importance_to_names(importance_scores, feature_names):
    """
    Create a DataFrame mapping feature importance to meaningful names.
    
    Args:
        importance_scores: Array of feature importance scores
        feature_names: List of meaningful feature names
        
    Returns:
        DataFrame with feature names and importance scores
    """
    # Handle case where we have more features than names (due to feature selection)
    min_length = min(len(importance_scores), len(feature_names))
    
    df = pd.DataFrame({
        'feature_name': feature_names[:min_length],
        'importance': importance_scores[:min_length],
        'feature_type': [name.split('__')[0] for name in feature_names[:min_length]]
    })
    
    return df.sort_values('importance', ascending=False)

def interpret_top_features(feature_df, top_n=10):
    """
    Interpret the top N most important features for business insights.
    
    Args:
        feature_df: DataFrame with feature names and importance
        top_n: Number of top features to interpret
        
    Returns:
        Dictionary with interpretations
    """
    top_features = feature_df.head(top_n)
    
    interpretations = {
        'text_insights': [],
        'categorical_insights': [],
        'numeric_insights': [],
        'business_actions': []
    }
    
    for _, row in top_features.iterrows():
        feature_name = row['feature_name']
        importance = row['importance']
        feature_type = row['feature_type']
        
        if feature_type == 'text':
            # Extract the actual word/phrase
            word = feature_name.replace('text__', '')
            interpretations['text_insights'].append({
                'word': word,
                'importance': importance,
                'insight': f"The word '{word}' is highly predictive of engagement"
            })
            
        elif feature_type == 'cat':
            # Extract categorical value
            cat_value = feature_name.replace('cat__', '')
            interpretations['categorical_insights'].append({
                'category': cat_value,
                'importance': importance,
                'insight': f"The category '{cat_value}' strongly influences engagement"
            })
            
        elif feature_type == 'num':
            # Numeric feature
            num_feature = feature_name.replace('num__', '')
            interpretations['numeric_insights'].append({
                'feature': num_feature,
                'importance': importance,
                'insight': f"The numeric feature '{num_feature}' is crucial for prediction"
            })
    
    # Generate business actions
    interpretations['business_actions'] = generate_business_actions(interpretations)
    
    return interpretations

def generate_business_actions(interpretations):
    """Generate actionable business insights from feature interpretations."""
    actions = []
    
    # Text-based actions
    if interpretations['text_insights']:
        top_word = interpretations['text_insights'][0]['word']
        actions.append(f"💬 Focus email content around '{top_word}' - it's highly predictive")
        
    # Categorical actions
    if interpretations['categorical_insights']:
        top_category = interpretations['categorical_insights'][0]['category']
        actions.append(f"🎯 Target '{top_category}' segment - they show different engagement patterns")
    
    # Numeric actions
    if interpretations['numeric_insights']:
        top_numeric = interpretations['numeric_insights'][0]['feature']
        actions.append(f"📊 Optimize '{top_numeric}' - it's a key engagement driver")
    
    return actions

def create_detailed_feature_importance_plot(feature_df, top_n=20, save_path='detailed_feature_importance.png'):
    """
    Create a detailed feature importance plot with meaningful names.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    top_features = feature_df.head(top_n)
    
    # Create color mapping by feature type
    color_map = {
        'text': '#2E8B57',      # Sea green for text features
        'cat': '#4169E1',       # Royal blue for categorical
        'num': '#FF6347'        # Tomato red for numeric
    }
    
    colors = [color_map.get(ftype, '#808080') for ftype in top_features['feature_type']]
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)
    
    # Customize labels to show feature type
    labels = []
    for _, row in top_features.iterrows():
        name = row['feature_name']
        ftype = row['feature_type']
        
        if ftype == 'text':
            # Show just the word for text features
            clean_name = name.replace('text__', '')[:30]  # Limit length
        elif ftype == 'cat':
            # Show categorical value
            clean_name = name.replace('cat__', '')[:30]
        else:
            # Show numeric feature name
            clean_name = name.replace('num__', '')[:30]
            
        labels.append(f"{clean_name} ({ftype})")
    
    plt.yticks(range(len(top_features)), labels)
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Most Important Features (Business Interpretable)')
    plt.gca().invert_yaxis()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E8B57', label='Text Features'),
        Patch(facecolor='#4169E1', label='Categorical Features'),
        Patch(facecolor='#FF6347', label='Numeric Features')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return save_path

# Example usage function
def analyze_model_features(pipeline, X_sample, importance_scores, top_n=10):
    """
    Complete analysis of model features with business interpretations.
    """
    print("🔍 FEATURE ANALYSIS REPORT")
    print("=" * 50)
    
    # Extract feature names
    feature_names = extract_feature_names_from_pipeline(pipeline, X_sample)
    
    # Map to meaningful names
    feature_df = map_feature_importance_to_names(importance_scores, feature_names)
    
    # Get interpretations
    interpretations = interpret_top_features(feature_df, top_n)
    
    # Print insights
    print(f"\n📊 TOP {top_n} MOST IMPORTANT FEATURES:")
    for i, (_, row) in enumerate(feature_df.head(top_n).iterrows(), 1):
        print(f"{i:2d}. {row['feature_name']:<40} | {row['importance']:.4f} | {row['feature_type']}")
    
    print("\n💡 BUSINESS INSIGHTS:")
    for insight in interpretations['text_insights'][:3]:
        print(f"   📝 {insight['insight']}")
    
    for insight in interpretations['categorical_insights'][:3]:
        print(f"   📋 {insight['insight']}")
    
    for insight in interpretations['numeric_insights'][:3]:
        print(f"   📊 {insight['insight']}")
    
    print("\n🎯 RECOMMENDED ACTIONS:")
    for action in interpretations['business_actions']:
        print(f"   {action}")
    
    # Create detailed plot
    plot_path = create_detailed_feature_importance_plot(feature_df, top_n)
    print(f"\n📈 Detailed plot saved as: {plot_path}")
    
    return feature_df, interpretations 