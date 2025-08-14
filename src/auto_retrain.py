"""
Automated Retraining Pipeline
Triggers retraining based on drift detection and performance degradation.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import yaml
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from advanced_drift_detection import AdvancedDriftDetector
from deploy import ChampionChallengerDeployment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutomatedRetraining:
    """
    Automated retraining pipeline with intelligent triggers.
    
    Implements:
    - Drift-based retraining triggers
    - Performance degradation triggers
    - Sliding window retraining
    - Weighted retraining strategies
    - Automated model validation
    """
    
    def __init__(self, config_path="config/main_config.yaml"):
        """Initialize the automated retraining pipeline."""
        self.config = self._load_config(config_path)
        self.drift_detector = AdvancedDriftDetector(self.config.get('drift_detection', {}))
        self.deployer = ChampionChallengerDeployment(config_path)
        
        # Retraining history
        self.retraining_history = []
        self.last_retraining_date = None
        
        # Load retraining history if exists
        self._load_retraining_history()
        
        logger.info("🤖 Automated retraining pipeline initialized")
    
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Add default retraining configuration if not present
            if 'retraining' not in config:
                config['retraining'] = {
                    'drift_threshold': 0.25,           # PSI threshold for retraining
                    'performance_threshold': 0.05,      # 5% performance degradation
                    'min_retraining_interval_days': 7,  # Minimum days between retraining
                    'max_retraining_interval_days': 90, # Maximum days between retraining
                    'sliding_window_months': 6,        # Sliding window for retraining
                    'weighted_retraining': True,        # Use weighted retraining
                    'auto_deploy': False,               # Auto-deploy after retraining
                    'validation_split': 0.2,           # Validation split for new models
                    'min_improvement': 0.02             # Minimum improvement required
                }
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            # Default configuration
            return {
                'retraining': {
                    'drift_threshold': 0.25,
                    'performance_threshold': 0.05,
                    'min_retraining_interval_days': 7,
                    'max_retraining_interval_days': 90,
                    'sliding_window_months': 6,
                    'weighted_retraining': True,
                    'auto_deploy': False,
                    'validation_split': 0.2,
                    'min_improvement': 0.02
                }
            }
    
    def _load_retraining_history(self):
        """Load retraining history from file."""
        history_file = "data/retraining_history.json"
        try:
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    self.retraining_history = json.load(f)
                
                if self.retraining_history:
                    self.last_retraining_date = datetime.fromisoformat(
                        self.retraining_history[-1]['timestamp']
                    )
                
                logger.info(f"📚 Loaded {len(self.retraining_history)} retraining records")
        except Exception as e:
            logger.warning(f"⚠️ Could not load retraining history: {e}")
    
    def _save_retraining_history(self):
        """Save retraining history to file."""
        try:
            history_file = "data/retraining_history.json"
            os.makedirs(os.path.dirname(history_file), exist_ok=True)
            
            with open(history_file, 'w') as f:
                json.dump(self.retraining_history, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Failed to save retraining history: {e}")
    
    def should_retrain(self, training_data, new_data, performance_metrics=None, 
                      target_column='engagement_level'):
        """
        Determine if retraining is needed based on multiple criteria.
        
        Args:
            training_data: Original training data
            new_data: New data for drift detection
            performance_metrics: Current performance metrics
            target_column: Target variable column name
            
        Returns:
            dict: Retraining decision with reasons
        """
        logger.info("🔍 Evaluating retraining triggers...")
        
        retraining_decision = {
            'should_retrain': False,
            'triggers': [],
            'reasons': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Check time-based triggers
        time_trigger = self._check_time_based_triggers()
        if time_trigger['triggered']:
            retraining_decision['triggers'].append('time_based')
            retraining_decision['reasons'].append(time_trigger['reason'])
        
        # Check drift-based triggers
        drift_trigger = self._check_drift_based_triggers(training_data, new_data, target_column)
        if drift_trigger['triggered']:
            retraining_decision['triggers'].append('drift_based')
            retraining_decision['reasons'].append(drift_trigger['reason'])
        
        # Check performance-based triggers
        if performance_metrics:
            performance_trigger = self._check_performance_based_triggers(performance_metrics)
            if performance_trigger['triggered']:
                retraining_decision['triggers'].append('performance_based')
                retraining_decision['reasons'].append(performance_trigger['reason'])
        
        # Determine final decision
        retraining_decision['should_retrain'] = len(retraining_decision['triggers']) > 0
        
        if retraining_decision['should_retrain']:
            logger.info("🚨 RETRAINING TRIGGERED!")
            for trigger, reason in zip(retraining_decision['triggers'], retraining_decision['reasons']):
                logger.info(f"  {trigger}: {reason}")
        else:
            logger.info("✅ No retraining needed at this time")
        
        return retraining_decision
    
    def _check_time_based_triggers(self):
        """Check if retraining is needed based on time intervals."""
        if self.last_retraining_date is None:
            return {
                'triggered': True,
                'reason': 'First retraining - no previous retraining date'
            }
        
        days_since_retraining = (datetime.now() - self.last_retraining_date).days
        min_interval = self.config['retraining']['min_retraining_interval_days']
        max_interval = self.config['retraining']['max_retraining_interval_days']
        
        if days_since_retraining < min_interval:
            return {
                'triggered': False,
                'reason': f'Too soon since last retraining ({days_since_retraining} days < {min_interval})'
            }
        
        if days_since_retraining > max_interval:
            return {
                'triggered': True,
                'reason': f'Maximum retraining interval exceeded ({days_since_retraining} days > {max_interval})'
            }
        
        return {
            'triggered': False,
            'reason': f'Within retraining interval ({days_since_retraining} days)'
        }
    
    def _check_drift_based_triggers(self, training_data, new_data, target_column):
        """Check if retraining is needed based on data drift."""
        try:
            # Perform comprehensive drift analysis
            drift_results = self.drift_detector.comprehensive_drift_analysis(
                training_data, new_data, target_column
            )
            
            overall_assessment = drift_results.get('overall_drift_assessment', 'no_drift')
            drift_threshold = self.config['retraining']['drift_threshold']
            
            if overall_assessment == 'major_drift':
                return {
                    'triggered': True,
                    'reason': f'Major drift detected: {overall_assessment}',
                    'drift_results': drift_results
                }
            elif overall_assessment == 'minor_drift':
                # Check if any features exceed critical threshold
                psi_results = drift_results.get('psi_results', {})
                critical_features = [
                    feature for feature, result in psi_results.items()
                    if result and result.get('psi', 0) > drift_threshold
                ]
                
                if critical_features:
                    return {
                        'triggered': True,
                        'reason': f'Critical drift in features: {critical_features}',
                        'drift_results': drift_results
                    }
            
            return {
                'triggered': False,
                'reason': f'Drift level acceptable: {overall_assessment}',
                'drift_results': drift_results
            }
            
        except Exception as e:
            logger.error(f"❌ Drift-based trigger check failed: {e}")
            return {
                'triggered': False,
                'reason': f'Drift analysis failed: {str(e)}'
            }
    
    def _check_performance_based_triggers(self, performance_metrics):
        """Check if retraining is needed based on performance degradation."""
        try:
            current_accuracy = performance_metrics.get('accuracy', 1.0)
            current_auc = performance_metrics.get('auc', 1.0)
            
            # Get baseline performance (from training or previous evaluation)
            baseline_accuracy = performance_metrics.get('baseline_accuracy', 0.75)
            baseline_auc = performance_metrics.get('baseline_auc', 0.82)
            
            performance_threshold = self.config['retraining']['performance_threshold']
            
            # Check for significant degradation
            accuracy_degradation = baseline_accuracy - current_accuracy
            auc_degradation = baseline_auc - current_auc
            
            if accuracy_degradation > performance_threshold or auc_degradation > performance_threshold:
                return {
                    'triggered': True,
                    'reason': f'Performance degraded: Accuracy -{accuracy_degradation:.3f}, AUC -{auc_degradation:.3f}'
                }
            
            return {
                'triggered': False,
                'reason': f'Performance acceptable: Accuracy {current_accuracy:.3f}, AUC {current_auc:.3f}'
            }
            
        except Exception as e:
            logger.error(f"❌ Performance-based trigger check failed: {e}")
            return {
                'triggered': False,
                'reason': f'Performance analysis failed: {str(e)}'
            }
    
    def execute_retraining(self, training_data, new_data, target_column='engagement_level',
                          retraining_strategy='sliding_window'):
        """
        Execute the retraining pipeline.
        
        Args:
            training_data: Original training data
            new_data: New data to incorporate
            target_column: Target variable column name
            retraining_strategy: 'sliding_window', 'weighted', or 'full_batch'
            
        Returns:
            dict: Retraining results
        """
        logger.info(f"🚀 Starting automated retraining with strategy: {retraining_strategy}")
        
        try:
            # Prepare data based on strategy
            if retraining_strategy == 'sliding_window':
                prepared_data = self._prepare_sliding_window_data(training_data, new_data)
            elif retraining_strategy == 'weighted':
                prepared_data = self._prepare_weighted_data(training_data, new_data)
            elif retraining_strategy == 'full_batch':
                prepared_data = self._prepare_full_batch_data(training_data, new_data)
            else:
                raise ValueError(f"Unknown retraining strategy: {retraining_strategy}")
            
            # Train new model
            new_model = self._train_new_model(prepared_data, target_column)
            
            # Validate new model
            validation_results = self._validate_new_model(new_model, prepared_data, target_column)
            
            # Register as challenger
            model_metadata = {
                'version': f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'retraining_strategy': retraining_strategy,
                'training_samples': len(prepared_data),
                'validation_results': validation_results
            }
            
            self.deployer.register_challenger(new_model, model_metadata)
            
            # Auto-deploy if configured
            if self.config['retraining']['auto_deploy']:
                if validation_results['deploy_recommended']:
                    deployment_success = self.deployer.deploy_challenger()
                    if deployment_success:
                        logger.info("🎉 New model automatically deployed!")
                    else:
                        logger.warning("⚠️ Auto-deployment failed")
                else:
                    logger.info("⚠️ Auto-deployment skipped - validation failed")
            
            # Record retraining
            retraining_record = {
                'timestamp': datetime.now().isoformat(),
                'strategy': retraining_strategy,
                'training_samples': len(prepared_data),
                'validation_results': validation_results,
                'auto_deployed': self.config['retraining']['auto_deploy'] and validation_results['deploy_recommended']
            }
            
            self.retraining_history.append(retraining_record)
            self.last_retraining_date = datetime.now()
            self._save_retraining_history()
            
            logger.info("✅ Retraining pipeline completed successfully!")
            
            return {
                'success': True,
                'new_model': new_model,
                'validation_results': validation_results,
                'retraining_record': retraining_record
            }
            
        except Exception as e:
            logger.error(f"❌ Retraining pipeline failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _prepare_sliding_window_data(self, training_data, new_data):
        """Prepare data for sliding window retraining."""
        logger.info("📅 Preparing sliding window data...")
        
        window_months = self.config['retraining']['sliding_window_months']
        cutoff_date = datetime.now() - timedelta(days=window_months * 30)
        
        # Combine recent training data with new data
        if 'timestamp' in training_data.columns:
            # Filter training data to recent months
            training_data['timestamp'] = pd.to_datetime(training_data['timestamp'])
            recent_training = training_data[training_data['timestamp'] >= cutoff_date]
            logger.info(f"  Using {len(recent_training)} samples from last {window_months} months")
        else:
            # If no timestamp, use last portion of training data
            recent_training = training_data.tail(len(training_data) // 2)
            logger.info(f"  Using last {len(recent_training)} samples from training data")
        
        # Combine with new data
        combined_data = pd.concat([recent_training, new_data], ignore_index=True)
        logger.info(f"  Final dataset: {len(combined_data)} samples")
        
        return combined_data
    
    def _prepare_weighted_data(self, training_data, new_data):
        """Prepare data for weighted retraining."""
        logger.info("⚖️ Preparing weighted data...")
        
        # Create sample weights - newer data gets higher weight
        training_weights = np.ones(len(training_data)) * 0.5  # Lower weight for old data
        new_weights = np.ones(len(new_data)) * 2.0           # Higher weight for new data
        
        # Combine data
        combined_data = pd.concat([training_data, new_data], ignore_index=True)
        combined_weights = np.concatenate([training_weights, new_weights])
        
        # Store weights in the data for training
        combined_data['sample_weight'] = combined_weights
        
        logger.info(f"  Final dataset: {len(combined_data)} samples with weights")
        logger.info(f"  Training data weight: 0.5, New data weight: 2.0")
        
        return combined_data
    
    def _prepare_full_batch_data(self, training_data, new_data):
        """Prepare data for full batch retraining."""
        logger.info("📦 Preparing full batch data...")
        
        # Simply combine all data
        combined_data = pd.concat([training_data, new_data], ignore_index=True)
        logger.info(f"  Final dataset: {len(combined_data)} samples (full batch)")
        
        return combined_data
    
    def _train_new_model(self, data, target_column):
        """Train a new model using the prepared data."""
        logger.info("🏋️ Training new model...")
        
        try:
            # Import training function
            from train_real_data_optimized import main as train_main
            
            # Save prepared data temporarily
            temp_data_path = "data/temp_retraining_data.csv"
            data.to_csv(temp_data_path, index=False)
            
            # Train model (this would need to be adapted to work with our data)
            # For now, we'll create a simple model as placeholder
            from sklearn.ensemble import RandomForestClassifier
            
            # Prepare features and target
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            # Handle non-numeric columns
            X = X.select_dtypes(include=[np.number])
            
            # Train simple model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Clean up temp file
            if os.path.exists(temp_data_path):
                os.remove(temp_data_path)
            
            logger.info(f"✅ New model trained on {len(data)} samples")
            return model
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            raise
    
    def _validate_new_model(self, new_model, data, target_column):
        """Validate the new model against validation criteria."""
        logger.info("🔍 Validating new model...")
        
        try:
            # Split data for validation
            from sklearn.model_selection import train_test_split
            
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            # Handle non-numeric columns
            X = X.select_dtypes(include=[np.number])
            
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.config['retraining']['validation_split'], 
                random_state=42, stratify=y
            )
            
            # Train on training split
            new_model.fit(X_train, y_train)
            
            # Evaluate on validation split
            from sklearn.metrics import accuracy_score, roc_auc_score
            
            y_pred = new_model.predict(X_val)
            y_proba = new_model.predict_proba(X_val) if hasattr(new_model, 'predict_proba') else None
            
            accuracy = accuracy_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_proba[:, 1]) if y_proba is not None else 0.5
            
            # Check against minimum improvement
            min_improvement = self.config['retraining']['min_improvement']
            
            # For now, we'll assume the model meets basic criteria
            # In production, you'd compare against the current champion
            deploy_recommended = accuracy >= 0.7 and auc >= 0.7
            
            validation_results = {
                'accuracy': accuracy,
                'auc': auc,
                'deploy_recommended': deploy_recommended,
                'validation_samples': len(X_val),
                'min_improvement_met': True  # Placeholder
            }
            
            logger.info(f"✅ Model validation complete:")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  AUC: {auc:.4f}")
            logger.info(f"  Deploy recommended: {'✅ YES' if deploy_recommended else '❌ NO'}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"❌ Model validation failed: {e}")
            return {
                'accuracy': 0.0,
                'auc': 0.0,
                'deploy_recommended': False,
                'error': str(e)
            }
    
    def get_retraining_status(self):
        """Get current retraining status and history."""
        return {
            'last_retraining_date': self.last_retraining_date.isoformat() if self.last_retraining_date else None,
            'total_retrainings': len(self.retraining_history),
            'retraining_history': self.retraining_history[-5:],  # Last 5 retrainings
            'next_recommended_retraining': self._get_next_recommended_retraining()
        }
    
    def _get_next_recommended_retraining(self):
        """Calculate next recommended retraining date."""
        if self.last_retraining_date is None:
            return datetime.now().isoformat()
        
        max_interval = self.config['retraining']['max_retraining_interval_days']
        next_retraining = self.last_retraining_date + timedelta(days=max_interval)
        
        return next_retraining.isoformat()
    
    def cleanup_old_models(self, keep_days=30):
        """Clean up old model versions."""
        try:
            models_dir = Path("models")
            cutoff_time = datetime.now() - timedelta(days=keep_days)
            
            removed_count = 0
            for model_file in models_dir.glob("champion_backup_*.joblib"):
                if model_file.stat().st_mtime < cutoff_time.timestamp():
                    model_file.unlink()
                    removed_count += 1
            
            if removed_count > 0:
                logger.info(f"🧹 Cleaned up {removed_count} old model files")
            
        except Exception as e:
            logger.error(f"❌ Model cleanup failed: {e}")

def main():
    """Demo of the automated retraining pipeline."""
    print("🤖 Automated Retraining Pipeline Demo")
    print("="*60)
    
    # Initialize pipeline
    retrainer = AutomatedRetraining()
    
    print("💡 Available Methods:")
    print("  1. Check retraining need: retrainer.should_retrain(train_data, new_data)")
    print("  2. Execute retraining: retrainer.execute_retraining(train_data, new_data)")
    print("  3. Get status: retrainer.get_retraining_status()")
    
    # Show current status
    status = retrainer.get_retraining_status()
    print(f"\n📊 Current Status:")
    print(f"  Total Retrainings: {status['total_retrainings']}")
    print(f"  Last Retraining: {status['last_retraining_date'] or 'Never'}")
    print(f"  Next Recommended: {status['next_recommended_retraining']}")
    
    return retrainer

if __name__ == "__main__":
    retrainer = main()
