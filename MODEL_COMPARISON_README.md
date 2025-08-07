# Model Comparison Scripts for B2B Email Marketing

This repository contains two comprehensive model comparison scripts that test multiple machine learning algorithms against your existing XGBoost baseline model.

## 📁 Files Overview

### Core Scripts
- **`model_comparison.py`** - Full comprehensive comparison with 9+ models
- **`quick_model_comparison.py`** - Fast comparison with top 5 models  
- **`xgboost_accuracy_optimized.py`** - Your existing XGBoost baseline

### Generated Outputs
- **Visualizations**: PNG files with performance comparisons and confusion matrices
- **Console Reports**: Detailed metrics and recommendations

## 🚀 Quick Start

### Option 1: Quick Comparison (Recommended for testing)
```bash
python quick_model_comparison.py
```
**Runtime:** ~5-10 minutes  
**Models tested:** XGBoost, Random Forest, Gradient Boosting, Logistic Regression, SVM

### Option 2: Comprehensive Comparison 
```bash
python model_comparison.py
```
**Runtime:** ~15-30 minutes  
**Models tested:** XGBoost, Random Forest, Gradient Boosting, Extra Trees, SVM, Logistic Regression, Neural Network, AdaBoost, K-Nearest Neighbors

## 📊 What the Scripts Do

### Data Processing
Both scripts use **identical data processing** to your existing XGBoost script:
- ✅ Same feature engineering pipeline
- ✅ Same target variable creation (3-tier engagement)
- ✅ Same text processing and interaction features  
- ✅ Same train/test split for fair comparison

### Models Compared

| Model | Type | Strengths | Use Case |
|-------|------|-----------|----------|
| **XGBoost** | Gradient Boosting | High accuracy, handles mixed data types | Your current baseline |
| **Random Forest** | Ensemble | Robust, handles missing data well | Good general-purpose alternative |
| **Gradient Boosting** | Ensemble | Strong sequential learning | Similar to XGBoost but different implementation |
| **Logistic Regression** | Linear | Fast, interpretable | Good baseline for comparison |
| **SVM** | Kernel-based | Effective in high dimensions | Good for text-heavy features |
| **Neural Network** | Deep Learning | Captures complex patterns | Advanced pattern recognition |
| **Extra Trees** | Ensemble | Very fast training | Quick alternative to Random Forest |
| **AdaBoost** | Boosting | Good with weak learners | Classic boosting approach |
| **K-Nearest Neighbors** | Instance-based | Simple, no assumptions | Local pattern matching |

## 📈 Output Interpretations

### Metrics Explained
- **Accuracy**: Overall correctness (main metric)
- **F1-Score**: Balance of precision and recall  
- **Cross-Validation**: Stability across different data splits
- **Precision/Recall**: Class-specific performance

### Key Visualizations Generated

#### 1. Performance Overview (`model_comparison_overview.png`)
- Bar charts comparing accuracy and F1-scores
- Cross-validation scores with error bars
- Precision vs Recall scatter plot

#### 2. Performance Heatmap (`model_comparison_heatmap.png`)
- Color-coded comparison across all metrics
- Easy identification of best/worst performers

#### 3. Confusion Matrices (`top_models_confusion_matrices.png`)
- Class-wise performance for top 3 models
- Shows which engagement types each model predicts best

#### 4. Quick Comparison (`quick_model_comparison.png`)
- Fast overview with top 5 models only

## 🎯 How to Interpret Results

### What to Look For

1. **Best Overall Model**: Highest accuracy with reasonable stability (low CV standard deviation)

2. **Model Consistency**: 
   - Small gap between training and validation scores = good generalization
   - Low CV standard deviation = stable across different data splits

3. **Class Performance**: 
   - Check confusion matrices to see if models struggle with specific engagement types
   - Look for models that balance performance across all 3 classes

4. **Practical Considerations**:
   - **Speed**: Logistic Regression and Random Forest are fastest
   - **Interpretability**: Tree-based models (XGBoost, Random Forest) are most interpretable
   - **Scalability**: Logistic Regression and Neural Networks scale best

### Example Results Interpretation

```
🏆 BEST PERFORMING MODELS:
   Best Accuracy: XGBoost (0.8234)
   Best F1-Score: Random Forest (0.8156)  
   Most Stable: Logistic Regression (CV-Std: 0.0045)
```

**Recommendation**: If XGBoost maintains its lead with good stability, stick with it. If another model significantly outperforms (>1% accuracy gain), consider switching.

## ⚙️ Configuration Options

### Quick Model Parameters
You can modify the quick comparison by editing `get_quick_models()` in `quick_model_comparison.py`:

```python
# Example: Make Random Forest more conservative
'Random Forest': RandomForestClassifier(
    n_estimators=300,  # Increase for better performance
    max_depth=8,       # Decrease for less overfitting
    class_weight='balanced',
    random_state=42
)
```

### Adding New Models
To test additional models, add them to the model dictionary:

```python
'Your Model Name': YourModelClass(
    parameter1=value1,
    parameter2=value2,
    random_state=42
)
```

## 🔧 Troubleshooting

### Common Issues

1. **Memory Errors**: 
   - Use `quick_model_comparison.py` instead
   - Reduce `max_features` in TfidfVectorizer
   - Reduce `k` in SelectKBest feature selection

2. **Slow Performance**:
   - Reduce `n_estimators` for tree-based models
   - Use smaller validation sets
   - Set `n_jobs=-1` for parallel processing

3. **TensorFlow/Neural Network Issues**:
   - If TensorFlow fails to load, the Neural Network will be automatically skipped
   - Windows DLL errors: Try `pip install tensorflow-cpu` instead of `tensorflow`
   - Alternative: Use the quick comparison script which doesn't include Neural Networks

4. **Poor Neural Network Performance** (when TensorFlow works):
   - Increase training epochs
   - Adjust learning rate
   - Check feature scaling

### Dependencies
Make sure you have all required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

**Note on TensorFlow**: If you encounter TensorFlow DLL errors on Windows:
- The Neural Network model will be automatically skipped
- All other models will still run normally
- To fix TensorFlow: Try `pip uninstall tensorflow` then `pip install tensorflow-cpu`

## 📝 Next Steps

### If XGBoost Remains Best:
1. ✅ Your current model is optimal
2. Consider ensemble methods combining top 2-3 models
3. Focus on feature engineering improvements

### If Another Model Performs Better:
1. 🔄 Switch to the better-performing model
2. Optimize hyperparameters specifically for your data
3. Retrain and validate on fresh data

### For Production Deployment:
1. **Speed**: Consider Logistic Regression or optimized XGBoost
2. **Interpretability**: Stick with tree-based models
3. **Accuracy**: Use the best-performing model from comparison

## 🤝 Support

If you encounter issues or want to modify the comparison:
1. Check the console output for detailed error messages
2. Verify your data file path matches `CSV_FILE_PATH` in the scripts
3. Ensure all dependencies are installed correctly

The scripts are designed to be robust and provide clear feedback about any issues encountered during model training or evaluation. 