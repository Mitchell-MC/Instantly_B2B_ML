"""
Champion/Challenger Deployment Framework
Safe model deployment with statistical significance testing and rollback capabilities.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from scipy import stats
import logging
import yaml
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChampionChallengerDeployment:
    """
    Champion/Challenger deployment framework for safe model updates.
    
    This class implements:
    - Statistical significance testing between models
    - Performance comparison on validation data
    - Automated deployment decisions
    - Rollback mechanisms
    - Model versioning and tracking
    """
    
    def __init__(self, config_path="config/main_config.yaml"):
        """Initialize the deployment framework."""
        self.config = self._load_config(config_path)
        self.champion_model = None
        self.challenger_model = None
        self.deployment_history = []
        self.rollback_history = []
        
        # Load current champion if exists
        self._load_champion_model()
        
        logger.info("🚀 Champion/Challenger deployment framework initialized")
    
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            # Default configuration
            return {
                'deployment': {
                    'significance_level': 0.05,
                    'min_improvement': 0.02,
                    'validation_window_days': 7,
                    'max_rollback_attempts': 3
                }
            }
    
    def _load_champion_model(self):
        """Load the current champion model."""
        try:
            model_path = self.config.get('paths', {}).get('model_artifact', 'models/email_open_predictor_v1.0.joblib')
            if os.path.exists(model_path):
                self.champion_model = joblib.load(model_path)
                logger.info(f"✅ Champion model loaded: {model_path}")
            else:
                logger.warning(f"⚠️ No champion model found at {model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load champion model: {e}")
    
    def register_challenger(self, challenger_model, model_metadata):
        """
        Register a new challenger model for evaluation.
        
        Args:
            challenger_model: The new model to evaluate
            model_metadata: Dictionary containing model information
        """
        self.challenger_model = {
            'model': challenger_model,
            'metadata': model_metadata,
            'registration_time': datetime.now(),
            'status': 'registered'
        }
        
        logger.info(f"🎯 Challenger model registered: {model_metadata.get('version', 'unknown')}")
        return True
    
    def evaluate_challenger(self, validation_data, target_column='engagement_level'):
        """
        Evaluate challenger against champion on validation data.
        
        Args:
            validation_data: DataFrame with features and target
            target_column: Name of the target column
            
        Returns:
            dict: Evaluation results with deployment recommendation
        """
        if self.champion_model is None:
            logger.error("❌ No champion model available for comparison")
            return {'deploy': False, 'reason': 'No champion model'}
        
        if self.challenger_model is None:
            logger.error("❌ No challenger model registered")
            return {'deploy': False, 'reason': 'No challenger model'}
        
        logger.info("🔍 Evaluating challenger vs champion...")
        
        try:
            # Prepare validation data
            X_val = validation_data.drop(columns=[target_column])
            y_val = validation_data[target_column]
            
            # Get predictions from both models
            champion_pred = self.champion_model.predict(X_val)
            challenger_pred = self.challenger_model['model'].predict(X_val)
            
            # Get probabilities if available
            try:
                champion_proba = self.champion_model.predict_proba(X_val)
                challenger_proba = self.challenger_model['model'].predict_proba(X_val)
                
                # Use AUC for comparison
                champion_auc = roc_auc_score(y_val, champion_proba[:, 1]) if champion_proba.shape[1] > 1 else 0.5
                challenger_auc = roc_auc_score(y_val, challenger_proba[:, 1]) if challenger_proba.shape[1] > 1 else 0.5
            except:
                champion_auc = 0.5
                challenger_auc = 0.5
            
            # Calculate metrics
            champion_accuracy = accuracy_score(y_val, champion_pred)
            challenger_accuracy = accuracy_score(y_val, challenger_pred)
            
            champion_f1 = f1_score(y_val, champion_pred, average='weighted')
            challenger_f1 = f1_score(y_val, challenger_pred, average='weighted')
            
            # Statistical significance testing
            significance_result = self._test_statistical_significance(
                champion_pred, challenger_pred, y_val
            )
            
            # Performance improvement analysis
            accuracy_improvement = challenger_accuracy - champion_accuracy
            f1_improvement = challenger_f1 - champion_f1
            auc_improvement = challenger_auc - champion_auc
            
            # Determine if challenger should be deployed
            should_deploy = self._should_deploy_challenger(
                accuracy_improvement, f1_improvement, auc_improvement,
                significance_result, validation_data.shape[0]
            )
            
            # Compile results
            evaluation_results = {
                'deploy': should_deploy,
                'champion_metrics': {
                    'accuracy': champion_accuracy,
                    'f1_score': champion_f1,
                    'auc': champion_auc
                },
                'challenger_metrics': {
                    'accuracy': challenger_accuracy,
                    'f1_score': challenger_f1,
                    'auc': challenger_auc
                },
                'improvements': {
                    'accuracy': accuracy_improvement,
                    'f1_score': f1_improvement,
                    'auc': auc_improvement
                },
                'statistical_significance': significance_result,
                'validation_samples': validation_data.shape[0],
                'evaluation_time': datetime.now().isoformat()
            }
            
            # Update challenger status
            self.challenger_model['evaluation_results'] = evaluation_results
            self.challenger_model['status'] = 'evaluated'
            
            logger.info(f"📊 Challenger evaluation complete:")
            logger.info(f"  Accuracy: {champion_accuracy:.4f} → {challenger_accuracy:.4f} ({accuracy_improvement:+.4f})")
            logger.info(f"  F1-Score: {champion_f1:.4f} → {challenger_f1:.4f} ({f1_improvement:+.4f})")
            logger.info(f"  AUC: {champion_auc:.4f} → {challenger_auc:.4f} ({auc_improvement:+.4f})")
            logger.info(f"  Deploy: {'✅ YES' if should_deploy else '❌ NO'}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"❌ Challenger evaluation failed: {e}")
            return {'deploy': False, 'reason': f'Evaluation error: {str(e)}'}
    
    def _test_statistical_significance(self, champion_pred, challenger_pred, y_true):
        """
        Test statistical significance between champion and challenger predictions.
        
        Args:
            champion_pred: Champion model predictions
            challenger_pred: Challenger model predictions
            y_true: True labels
            
        Returns:
            dict: Statistical test results
        """
        try:
            # McNemar's test for paired predictions
            # Create contingency table
            both_correct = np.sum((champion_pred == y_true) & (challenger_pred == y_true))
            both_wrong = np.sum((champion_pred != y_true) & (challenger_pred != y_true))
            champion_correct = np.sum((champion_pred == y_true) & (challenger_pred != y_true))
            challenger_correct = np.sum((champion_pred != y_true) & (challenger_pred == y_true))
            
            contingency_table = np.array([[both_correct, champion_correct],
                                        [challenger_correct, both_wrong]])
            
            # Perform McNemar's test
            mcnemar_stat, mcnemar_pvalue = stats.mcnemar(contingency_table, exact=True)
            
            # Chi-square test for independence
            chi2_stat, chi2_pvalue = stats.chi2_contingency(contingency_table)[:2]
            
            return {
                'mcnemar_statistic': mcnemar_stat,
                'mcnemar_pvalue': mcnemar_pvalue,
                'chi2_statistic': chi2_stat,
                'chi2_pvalue': chi2_pvalue,
                'significant': mcnemar_pvalue < self.config['deployment']['significance_level'],
                'contingency_table': contingency_table.tolist()
            }
            
        except Exception as e:
            logger.warning(f"Statistical testing failed: {e}")
            return {
                'significant': False,
                'error': str(e)
            }
    
    def _should_deploy_challenger(self, accuracy_improvement, f1_improvement, 
                                 auc_improvement, significance_result, sample_size):
        """
        Determine if challenger should be deployed based on performance and significance.
        
        Args:
            accuracy_improvement: Improvement in accuracy
            f1_improvement: Improvement in F1-score
            auc_improvement: Improvement in AUC
            significance_result: Statistical significance test results
            sample_size: Number of validation samples
            
        Returns:
            bool: Whether to deploy the challenger
        """
        # Check minimum sample size
        if sample_size < 100:
            logger.warning(f"⚠️ Small validation set ({sample_size} samples) - requiring higher improvement")
            min_improvement = self.config['deployment']['min_improvement'] * 2
        else:
            min_improvement = self.config['deployment']['min_improvement']
        
        # Check if improvements meet minimum threshold
        meets_threshold = (
            accuracy_improvement >= min_improvement or
            f1_improvement >= min_improvement or
            auc_improvement >= min_improvement
        )
        
        # Check statistical significance
        is_significant = significance_result.get('significant', False)
        
        # Check if any metric degraded significantly
        no_significant_degradation = (
            accuracy_improvement >= -min_improvement and
            f1_improvement >= -min_improvement and
            auc_improvement >= -min_improvement
        )
        
        should_deploy = meets_threshold and is_significant and no_significant_degradation
        
        logger.info(f"🔍 Deployment criteria:")
        logger.info(f"  Meets improvement threshold: {'✅' if meets_threshold else '❌'}")
        logger.info(f"  Statistically significant: {'✅' if is_significant else '❌'}")
        logger.info(f"  No significant degradation: {'✅' if no_significant_degradation else '❌'}")
        
        return should_deploy
    
    def deploy_challenger(self, backup_champion=True):
        """
        Deploy the challenger as the new champion.
        
        Args:
            backup_champion: Whether to backup the current champion
            
        Returns:
            bool: Success status of deployment
        """
        if self.challenger_model is None:
            logger.error("❌ No challenger model to deploy")
            return False
        
        if self.challenger_model.get('status') != 'evaluated':
            logger.error("❌ Challenger must be evaluated before deployment")
            return False
        
        evaluation = self.challenger_model.get('evaluation_results', {})
        if not evaluation.get('deploy', False):
            logger.error("❌ Challenger evaluation does not recommend deployment")
            return False
        
        try:
            # Backup current champion if requested
            if backup_champion and self.champion_model is not None:
                self._backup_champion()
            
            # Deploy challenger
            self.champion_model = self.challenger_model['model']
            
            # Save new champion
            self._save_champion()
            
            # Update deployment history
            deployment_record = {
                'timestamp': datetime.now().isoformat(),
                'challenger_version': self.challenger_model['metadata'].get('version', 'unknown'),
                'evaluation_results': evaluation,
                'backup_created': backup_champion
            }
            self.deployment_history.append(deployment_record)
            
            # Clear challenger
            self.challenger_model = None
            
            logger.info("🎉 Challenger successfully deployed as new champion!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            return False
    
    def _backup_champion(self):
        """Create a backup of the current champion model."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"models/champion_backup_{timestamp}.joblib"
            
            # Ensure models directory exists
            os.makedirs("models", exist_ok=True)
            
            # Save backup
            joblib.dump(self.champion_model, backup_path)
            
            logger.info(f"💾 Champion backup created: {backup_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to backup champion: {e}")
    
    def _save_champion(self):
        """Save the current champion model."""
        try:
            model_path = self.config.get('paths', {}).get('model_artifact', 'models/email_open_predictor_v1.0.joblib')
            
            # Ensure models directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save champion
            joblib.dump(self.champion_model, model_path)
            
            logger.info(f"💾 Champion saved: {model_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save champion: {e}")
    
    def rollback_to_backup(self, backup_path=None):
        """
        Rollback to a previous champion backup.
        
        Args:
            backup_path: Path to specific backup file (optional)
            
        Returns:
            bool: Success status of rollback
        """
        try:
            if backup_path is None:
                # Find most recent backup
                backup_files = list(Path("models").glob("champion_backup_*.joblib"))
                if not backup_files:
                    logger.error("❌ No backup files found")
                    return False
                
                backup_path = max(backup_files, key=lambda x: x.stat().st_mtime)
            
            # Load backup
            backup_model = joblib.load(backup_path)
            
            # Update champion
            self.champion_model = backup_model
            
            # Save as current champion
            self._save_champion()
            
            # Record rollback
            rollback_record = {
                'timestamp': datetime.now().isoformat(),
                'backup_path': str(backup_path),
                'reason': 'Manual rollback'
            }
            self.rollback_history.append(rollback_record)
            
            logger.info(f"🔄 Rollback successful: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Rollback failed: {e}")
            return False
    
    def get_deployment_status(self):
        """Get current deployment status and history."""
        return {
            'champion_loaded': self.champion_model is not None,
            'challenger_status': self.challenger_model.get('status') if self.challenger_model else None,
            'deployment_history': self.deployment_history,
            'rollback_history': self.rollback_history,
            'total_deployments': len(self.deployment_history),
            'total_rollbacks': len(self.rollback_history)
        }
    
    def cleanup_old_backups(self, keep_days=30):
        """Clean up old backup files."""
        try:
            backup_dir = Path("models")
            cutoff_time = datetime.now() - timedelta(days=keep_days)
            
            removed_count = 0
            for backup_file in backup_dir.glob("champion_backup_*.joblib"):
                if backup_file.stat().st_mtime < cutoff_time.timestamp():
                    backup_file.unlink()
                    removed_count += 1
            
            if removed_count > 0:
                logger.info(f"🧹 Cleaned up {removed_count} old backup files")
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")

def main():
    """Demo of the Champion/Challenger deployment framework."""
    print("🚀 Champion/Challenger Deployment Framework Demo")
    print("="*60)
    
    # Initialize deployment framework
    deployer = ChampionChallengerDeployment()
    
    # Show current status
    status = deployer.get_deployment_status()
    print(f"📊 Current Status:")
    print(f"  Champion Loaded: {'✅' if status['champion_loaded'] else '❌'}")
    print(f"  Total Deployments: {status['total_deployments']}")
    print(f"  Total Rollbacks: {status['total_rollbacks']}")
    
    print("\n💡 Usage Examples:")
    print("  1. Register challenger: deployer.register_challenger(model, metadata)")
    print("  2. Evaluate challenger: deployer.evaluate_challenger(validation_data)")
    print("  3. Deploy challenger: deployer.deploy_challenger()")
    print("  4. Rollback if needed: deployer.rollback_to_backup()")
    
    return deployer

if __name__ == "__main__":
    deployer = main()
