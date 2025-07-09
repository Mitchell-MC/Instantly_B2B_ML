"""
Model Comparison Script for B2B Email Marketing Engagement Prediction

This script runs and compares three different approaches:
1. Current XGBoost with SMOTE (class balance focused)
2. New Accuracy-Optimized XGBoost (accuracy focused)
3. SVM with multiple kernels (alternative approach)

It provides side-by-side comparison of accuracy, performance metrics, and insights.
"""

import sys
import pandas as pd
import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Import the models
try:
    from new_grid_cv_xgboost import main as current_xgboost_main, load_data, engineer_features
    from xgboost_accuracy_optimized import main as accuracy_xgboost_main
    from svm_b2b_model import main as svm_main
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all model files are in the same directory.")
    sys.exit(1)

def run_model_with_timing(model_func, model_name):
    """
    Run a model function and measure execution time.
    """
    print(f"\n{'='*60}")
    print(f"RUNNING {model_name.upper()}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Capture stdout to extract results
        import io
        import contextlib
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            model_func()
        
        output = f.getvalue()
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"✅ {model_name} completed successfully")
        print(f"⏱️  Execution time: {execution_time/60:.1f} minutes")
        
        return {
            'status': 'success',
            'execution_time': execution_time,
            'output': output
        }
        
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"❌ {model_name} failed: {str(e)}")
        print(f"⏱️  Execution time before failure: {execution_time/60:.1f} minutes")
        
        return {
            'status': 'failed',
            'execution_time': execution_time,
            'error': str(e)
        }

def extract_accuracy_from_output(output):
    """
    Extract accuracy scores from model output.
    """
    lines = output.split('\n')
    accuracies = {}
    
    for line in lines:
        if 'accuracy' in line.lower() and ':' in line:
            try:
                # Try to extract numeric value
                parts = line.split(':')
                if len(parts) >= 2:
                    value_part = parts[1].strip()
                    # Extract first number found
                    import re
                    numbers = re.findall(r'\d+\.?\d*', value_part)
                    if numbers:
                        accuracy = float(numbers[0])
                        if accuracy > 1:  # Convert percentage to decimal
                            accuracy = accuracy / 100
                        accuracies[line.strip()] = accuracy
            except:
                continue
    
    return accuracies

def create_comparison_report(results):
    """
    Create a comprehensive comparison report.
    """
    print(f"\n{'='*80}")
    print("COMPREHENSIVE MODEL COMPARISON REPORT")
    print(f"{'='*80}")
    
    # Execution time comparison
    print("\n📊 EXECUTION TIME COMPARISON:")
    print("-" * 40)
    for model_name, result in results.items():
        status = "✅" if result['status'] == 'success' else "❌"
        time_str = f"{result['execution_time']/60:.1f} min"
        print(f"{status} {model_name:<25} {time_str:>10}")
    
    # Success/Failure summary
    successful_models = [name for name, result in results.items() if result['status'] == 'success']
    failed_models = [name for name, result in results.items() if result['status'] == 'failed']
    
    print(f"\n✅ Successful Models: {len(successful_models)}/{len(results)}")
    for model in successful_models:
        print(f"   - {model}")
    
    if failed_models:
        print(f"\n❌ Failed Models: {len(failed_models)}/{len(results)}")
        for model in failed_models:
            print(f"   - {model}: {results[model].get('error', 'Unknown error')}")
    
    # Accuracy comparison for successful models
    if successful_models:
        print(f"\n🎯 ACCURACY ANALYSIS:")
        print("-" * 40)
        
        for model_name in successful_models:
            output = results[model_name]['output']
            accuracies = extract_accuracy_from_output(output)
            
            if accuracies:
                print(f"\n{model_name}:")
                for metric, value in accuracies.items():
                    print(f"   {metric}: {value:.4f}")
            else:
                print(f"\n{model_name}: Could not extract accuracy metrics")
    
    return successful_models, failed_models

def generate_recommendations(successful_models, results):
    """
    Generate recommendations based on model performance.
    """
    print(f"\n{'='*80}")
    print("🔍 RECOMMENDATIONS")
    print(f"{'='*80}")
    
    if len(successful_models) == 0:
        print("❌ No models completed successfully. Please check:")
        print("   - Data file exists and is accessible")
        print("   - All dependencies are installed")
        print("   - Sufficient memory and computational resources")
        return
    
    if len(successful_models) == 1:
        print(f"✅ Only {successful_models[0]} completed successfully.")
        print("   Recommendations:")
        print("   - Use this model for your predictions")
        print("   - Investigate why other models failed")
        print("   - Consider running models individually for debugging")
        return
    
    # Multiple successful models
    print("🎯 NEXT STEPS:")
    print("\n1. **Accuracy Comparison:**")
    print("   - Review the accuracy metrics above")
    print("   - Choose the model with highest overall accuracy")
    print("   - Consider business impact, not just raw accuracy")
    
    print("\n2. **Model Selection Guidelines:**")
    print("   - **For Production**: Choose accuracy-optimized XGBoost")
    print("   - **For Interpretability**: Choose SVM (if linear kernel won)")
    print("   - **For Class Balance**: Consider current XGBoost with SMOTE")
    
    print("\n3. **Performance Considerations:**")
    fastest_model = min(successful_models, key=lambda x: results[x]['execution_time'])
    slowest_model = max(successful_models, key=lambda x: results[x]['execution_time'])
    
    print(f"   - Fastest model: {fastest_model} ({results[fastest_model]['execution_time']/60:.1f} min)")
    print(f"   - Slowest model: {slowest_model} ({results[slowest_model]['execution_time']/60:.1f} min)")
    
    print("\n4. **Business Impact Analysis:**")
    print("   - Test selected model on historical data")
    print("   - A/B test with small subset of campaigns")
    print("   - Monitor email campaign performance metrics")
    print("   - Measure ROI improvement")
    
    print("\n5. **Further Optimization:**")
    print("   - Fine-tune hyperparameters of best performing model")
    print("   - Collect more data if accuracy is below expectations")
    print("   - Consider ensemble methods combining multiple models")

def create_summary_visualization(successful_models, results):
    """
    Create a summary visualization of model performance.
    """
    if len(successful_models) < 2:
        return
    
    print(f"\n📈 Generating performance comparison chart...")
    
    # Extract execution times
    model_names = []
    execution_times = []
    
    for model_name in successful_models:
        model_names.append(model_name.replace('_', ' ').title())
        execution_times.append(results[model_name]['execution_time'] / 60)  # Convert to minutes
    
    # Create comparison chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(model_names, execution_times, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(model_names)])
    
    ax.set_ylabel('Execution Time (minutes)')
    ax.set_title('Model Training Time Comparison')
    ax.set_xlabel('Models')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}m', ha='center', va='bottom')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('model_comparison_execution_times.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Comparison chart saved as 'model_comparison_execution_times.png'")

def main():
    """
    Main function to run all models and compare performance.
    """
    print("🚀 B2B Email Marketing Model Comparison")
    print("=" * 60)
    print("This script will run three different models and compare their performance:")
    print("1. Current XGBoost with SMOTE (class balance focused)")
    print("2. Accuracy-Optimized XGBoost (accuracy focused)")  
    print("3. SVM with multiple kernels (alternative approach)")
    print("\n⚠️  Warning: This may take 2-4 hours to complete all models!")
    
    # Ask for user confirmation
    response = input("\nDo you want to proceed? (y/n): ").lower().strip()
    if response not in ['y', 'yes']:
        print("Comparison cancelled.")
        return
    
    # Define models to run
    models_to_run = {
        'Current_XGBoost_SMOTE': current_xgboost_main,
        'Accuracy_Optimized_XGBoost': accuracy_xgboost_main,
        'SVM_Multi_Kernel': svm_main
    }
    
    # Track overall execution time
    total_start_time = time.time()
    
    # Run each model
    results = {}
    for model_name, model_func in models_to_run.items():
        result = run_model_with_timing(model_func, model_name)
        results[model_name] = result
        
        # Brief pause between models
        time.sleep(2)
    
    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time
    
    # Generate comparison report
    successful_models, failed_models = create_comparison_report(results)
    
    # Generate recommendations
    generate_recommendations(successful_models, results)
    
    # Create summary visualization
    create_summary_visualization(successful_models, results)
    
    # Final summary
    print(f"\n{'='*80}")
    print("🏁 COMPARISON COMPLETE")
    print(f"{'='*80}")
    print(f"⏱️  Total execution time: {total_execution_time/3600:.1f} hours")
    print(f"✅ Successful models: {len(successful_models)}")
    print(f"❌ Failed models: {len(failed_models)}")
    
    print(f"\n📁 Generated files:")
    print("   - Confusion matrices for each model")
    print("   - Feature importance plots")
    print("   - Model comparison execution time chart")
    print("   - Detailed console output logs")
    
    print(f"\n💡 Next steps:")
    print("   1. Review the accuracy metrics above")
    print("   2. Select the best performing model")
    print("   3. Implement chosen model in production")
    print("   4. Monitor real-world performance")

if __name__ == "__main__":
    main() 