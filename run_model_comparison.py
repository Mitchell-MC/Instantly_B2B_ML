"""
Interactive Model Comparison Launcher

This script provides an easy-to-use menu for running different model comparisons
and understanding the results.
"""

import os
import sys
import subprocess
from pathlib import Path

def clear_screen():
    """Clear the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def check_requirements():
    """Check if the required data file exists."""
    data_file = Path("merged_contacts.csv")
    if not data_file.exists():
        print("❌ Error: 'merged_contacts.csv' not found in current directory!")
        print("\nPlease ensure the data file is in the same folder as these scripts.")
        return False
    
    print(f"✅ Data file found: {data_file} ({data_file.stat().st_size / (1024*1024):.1f} MB)")
    return True

def run_script(script_name):
    """Run a Python script and handle errors."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {script_name}")
    print("="*60)
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, 
                              text=True)
        
        if result.returncode == 0:
            print(f"\n✅ {script_name} completed successfully!")
        else:
            print(f"\n❌ {script_name} failed with return code {result.returncode}")
            
    except Exception as e:
        print(f"\n❌ Error running {script_name}: {str(e)}")
    
    input("\nPress Enter to continue...")

def show_menu():
    """Display the main menu options."""
    clear_screen()
    print("="*70)
    print("🤖 B2B EMAIL MARKETING - MODEL COMPARISON SUITE")
    print("="*70)
    
    print("\nAvailable Options:")
    print("\n1. 🚀 Quick Comparison (5 models, ~5-10 minutes)")
    print("   - XGBoost, Random Forest, Gradient Boosting, Logistic Regression, SVM")
    print("   - Recommended for first-time testing")
    
    print("\n2. 🔬 Comprehensive Comparison (8-9 models, ~15-30 minutes)")
    print("   - All models including Extra Trees, AdaBoost, KNN, Neural Network*")
    print("   - Full analysis with detailed visualizations")
    print("   - *Neural Network requires working TensorFlow installation")
    
    print("\n3. 📊 View Existing Results")
    print("   - Display previously generated visualization files")
    
    print("\n4. 📖 Open Documentation")
    print("   - View the detailed README guide")
    
    print("\n5. ❌ Exit")
    
    print("\n" + "="*70)

def view_results():
    """Show available result files."""
    print("\n📊 LOOKING FOR RESULT FILES...")
    
    result_files = [
        ("quick_model_comparison.png", "Quick Comparison Results"),
        ("model_comparison_overview.png", "Comprehensive Overview"),
        ("model_comparison_heatmap.png", "Performance Heatmap"),
        ("top_models_confusion_matrices.png", "Confusion Matrices")
    ]
    
    found_files = []
    for filename, description in result_files:
        if Path(filename).exists():
            found_files.append((filename, description))
    
    if not found_files:
        print("❌ No result files found. Run a comparison first!")
        return
    
    print(f"\n✅ Found {len(found_files)} result file(s):")
    for i, (filename, description) in enumerate(found_files, 1):
        file_size = Path(filename).stat().st_size / 1024
        print(f"   {i}. {description}")
        print(f"      File: {filename} ({file_size:.1f} KB)")
    
    print(f"\n💡 Open these PNG files to view your model comparison results!")
    print(f"   They are saved in the current directory: {Path.cwd()}")

def open_documentation():
    """Show documentation information."""
    readme_file = Path("MODEL_COMPARISON_README.md")
    
    if readme_file.exists():
        print(f"\n📖 Documentation available: {readme_file}")
        print("\n📝 Quick Guide:")
        print("="*50)
        print("• Use Option 1 for quick testing (5 models)")
        print("• Use Option 2 for comprehensive analysis (9+ models)")
        print("• Results are saved as PNG files for easy viewing")
        print("• Look for the model with highest accuracy and stability")
        print("• If XGBoost remains best, your current model is optimal")
        print("• If another model wins by >1%, consider switching")
        print("• TensorFlow issues? Neural Network will be auto-skipped")
        print("="*50)
    else:
        print("❌ Documentation file not found!")

def get_user_choice():
    """Get and validate user input."""
    while True:
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            if choice in ['1', '2', '3', '4', '5']:
                return int(choice)
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, 4, or 5.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)
        except Exception:
            print("❌ Invalid input. Please enter a number between 1 and 5.")

def main():
    """Main launcher function."""
    
    while True:
        show_menu()
        
        # Check requirements first
        if not check_requirements():
            input("\nPress Enter to exit...")
            break
        
        choice = get_user_choice()
        
        if choice == 1:
            run_script("quick_model_comparison.py")
            
        elif choice == 2:
            run_script("model_comparison.py")
            
        elif choice == 3:
            view_results()
            input("\nPress Enter to continue...")
            
        elif choice == 4:
            open_documentation()
            input("\nPress Enter to continue...")
            
        elif choice == 5:
            print("\n👋 Thank you for using the Model Comparison Suite!")
            print("Your results are saved as PNG files in the current directory.")
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        input("Press Enter to exit...") 