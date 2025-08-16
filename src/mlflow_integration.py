"""
MLflow Integration Module
Centralized MLflow functionality for experiment tracking, model registry, and deployment.
"""

import os
import sys
import yaml
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLflowManager:
    """
    Centralized MLflow management for the email engagement prediction project.
    
    Features:
    - Experiment tracking and management
    - Model registry operations
    - Automated logging of metrics, parameters, and artifacts
    - Model deployment and serving
    - Performance monitoring and comparison
    """
    
    def __init__(self, config_path: str = "config/main_config.yaml"):
        """Initialize MLflow manager with configuration."""
        self.config = self._load_config(config_path)
        self._setup_mlflow()
        self.experiment_name = self.config['mlflow']['experiment_name']
        self.model_name = self.config['mlflow']['model_name']
        
        # Set up experiment
        mlflow.set_experiment(self.experiment_name)
        logger.info(f"✅ MLflow initialized for experiment: {self.experiment_name}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load MLflow configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            if 'mlflow' not in config:
                raise ValueError("MLflow configuration not found in config file")
            
            return config
        except Exception as e:
            logger.error(f"Failed to load MLflow config: {e}")
            raise
    
    def _setup_mlflow(self):
        """Set up MLflow tracking and registry URIs."""
        mlflow_config = self.config['mlflow']
        
        # Set tracking URI
        if mlflow_config.get('tracking_uri'):
            mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
            logger.info(f"MLflow tracking URI set to: {mlflow_config['tracking_uri']}")
        
        # Set registry URI
        if mlflow_config.get('registry_uri'):
            mlflow.set_registry_uri(mlflow_config['registry_uri'])
            logger.info(f"MLflow registry URI set to: {mlflow_config['registry_uri']}")
    
    def start_run(self, run_name: str = None, tags: Dict[str, str] = None) -> mlflow.ActiveRun:
        """Start a new MLflow run with optional tags."""
        run_tags = tags or {}
        run_tags.update({
            'project': 'email_engagement_prediction',
            'timestamp': datetime.now().isoformat()
        })
        
        return mlflow.start_run(run_name=run_name, tags=run_tags)
    
    def log_training_run(self, 
                        model, 
                        X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_test: pd.DataFrame,
                        y_test: pd.Series,
                        metrics: Dict[str, float],
                        params: Dict[str, Any],
                        feature_importance: Optional[pd.DataFrame] = None,
                        artifacts: Optional[Dict[str, str]] = None,
                        run_name: str = "training_run") -> str:
        """
        Log a complete training run with MLflow.
        
        Args:
            model: Trained model object
            X_train, y_train: Training data
            X_test, y_test: Test data
            metrics: Performance metrics
            params: Model parameters
            feature_importance: Feature importance DataFrame
            artifacts: Additional artifacts to log
            run_name: Name for the MLflow run
            
        Returns:
            str: Run ID
        """
        with self.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            
            try:
                # Log parameters
                mlflow.log_params(params)
                logger.info(f"✅ Logged {len(params)} parameters")
                
                # Log metrics
                mlflow.log_metrics(metrics)
                logger.info(f"✅ Logged {len(metrics)} metrics")
                
                # Log model
                if hasattr(model, 'feature_importances_'):
                    # XGBoost model
                    mlflow.xgboost.log_model(model, "model")
                else:
                    # Scikit-learn model
                    mlflow.sklearn.log_model(model, "model")
                
                logger.info("✅ Model logged to MLflow")
                
                # Log feature importance
                if feature_importance is not None:
                    feature_importance_path = "feature_importance.csv"
                    feature_importance.to_csv(feature_importance_path, index=False)
                    mlflow.log_artifact(feature_importance_path)
                    os.remove(feature_importance_path)  # Clean up
                    logger.info("✅ Feature importance logged")
                
                # Log training data sample
                train_sample_path = "training_sample.csv"
                X_train.head(1000).to_csv(train_sample_path, index=False)
                mlflow.log_artifact(train_sample_path, "training_data")
                os.remove(train_sample_path)
                
                # Log test data sample
                test_sample_path = "test_sample.csv"
                X_test.head(1000).to_csv(test_sample_path, index=False)
                mlflow.log_artifact(test_sample_path, "test_data")
                os.remove(test_sample_path)
                
                # Log additional artifacts
                if artifacts:
                    for name, path in artifacts.items():
                        if os.path.exists(path):
                            mlflow.log_artifact(path, name)
                            logger.info(f"✅ Artifact logged: {name}")
                
                # Log dataset info
                dataset_info = {
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'features': len(X_train.columns),
                    'target_distribution': y_train.value_counts().to_dict()
                }
                mlflow.log_dict(dataset_info, "dataset_info.json")
                
                logger.info(f"✅ Training run logged successfully. Run ID: {run_id}")
                return run_id
                
            except Exception as e:
                logger.error(f"❌ Failed to log training run: {e}")
                raise
    
    def log_prediction_batch(self, 
                           predictions: np.ndarray,
                           probabilities: np.ndarray,
                           actuals: Optional[np.ndarray] = None,
                           features: Optional[pd.DataFrame] = None,
                           metadata: Optional[Dict[str, Any]] = None,
                           run_name: str = "prediction_batch") -> str:
        """
        Log a batch of predictions for monitoring.
        
        Args:
            predictions: Model predictions
            probabilities: Prediction probabilities
            actuals: Actual outcomes (if available)
            features: Input features
            metadata: Additional metadata
            run_name: Name for the MLflow run
            
        Returns:
            str: Run ID
        """
        with self.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            
            try:
                # Log prediction statistics
                pred_stats = {
                    'total_predictions': len(predictions),
                    'positive_predictions': np.sum(predictions == 1),
                    'negative_predictions': np.sum(predictions == 0),
                    'avg_confidence': np.mean(np.max(probabilities, axis=1)) if probabilities.ndim > 1 else np.mean(probabilities)
                }
                mlflow.log_metrics(pred_stats)
                
                # Log actuals if available
                if actuals is not None:
                    mlflow.log_artifact(
                        pd.DataFrame({'actuals': actuals, 'predictions': predictions}).to_csv(index=False),
                        "predictions_vs_actuals.csv"
                    )
                
                # Log feature statistics if available
                if features is not None:
                    feature_stats = features.describe()
                    mlflow.log_artifact(feature_stats.to_csv(), "feature_statistics.csv")
                
                # Log metadata
                if metadata:
                    mlflow.log_dict(metadata, "prediction_metadata.json")
                
                logger.info(f"✅ Prediction batch logged successfully. Run ID: {run_id}")
                return run_id
                
            except Exception as e:
                logger.error(f"❌ Failed to log prediction batch: {e}")
                raise
    
    def register_model(self, 
                      model_path: str, 
                      model_name: str = None,
                      description: str = None,
                      tags: Dict[str, str] = None) -> str:
        """
        Register a model in the MLflow Model Registry.
        
        Args:
            model_path: Path to the logged model
            model_name: Name for the registered model
            description: Model description
            tags: Model tags
            
        Returns:
            str: Model version
        """
        model_name = model_name or self.model_name
        description = description or self.config['mlflow']['model_registry']['description']
        
        try:
            # Register model
            model_details = mlflow.register_model(
                model_uri=model_path,
                name=model_name,
                tags=tags or {}
            )
            
            # Add description
            client = mlflow.tracking.MlflowClient()
            client.update_registered_model(
                name=model_name,
                description=description
            )
            
            logger.info(f"✅ Model registered successfully: {model_name} v{model_details.version}")
            return str(model_details.version)
            
        except Exception as e:
            logger.error(f"❌ Failed to register model: {e}")
            raise
    
    def transition_model_stage(self, 
                             model_name: str, 
                             version: str, 
                             stage: str) -> bool:
        """
        Transition a model to a specific stage in the registry.
        
        Args:
            model_name: Name of the registered model
            version: Model version
            stage: Target stage (Production, Staging, Archived)
            
        Returns:
            bool: Success status
        """
        try:
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            
            logger.info(f"✅ Model {model_name} v{version} transitioned to {stage}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to transition model stage: {e}")
            return False
    
    def load_model(self, 
                  model_name: str = None, 
                  version: str = None, 
                  stage: str = None):
        """
        Load a model from the MLflow Model Registry.
        
        Args:
            model_name: Name of the registered model
            version: Model version (if None, loads latest)
            stage: Model stage (if None, loads from Production)
            
        Returns:
            Loaded model object
        """
        model_name = model_name or self.model_name
        
        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
            elif stage:
                model_uri = f"models:/{model_name}/{stage}"
            else:
                model_uri = f"models:/{model_name}/Production"
            
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"✅ Model loaded successfully: {model_uri}")
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def compare_models(self, 
                      model_names: list, 
                      metric: str = "accuracy") -> pd.DataFrame:
        """
        Compare multiple models based on a specific metric.
        
        Args:
            model_names: List of model names to compare
            metric: Metric to compare on
            
        Returns:
            DataFrame with comparison results
        """
        try:
            client = mlflow.tracking.MlflowClient()
            comparison_data = []
            
            for model_name in model_names:
                # Get latest version
                latest_version = client.get_latest_versions(model_name, stages=["Production"])
                if latest_version:
                    version = latest_version[0]
                    # Get metrics for this version
                    metrics = client.get_run(version.run_id).data.metrics
                    comparison_data.append({
                        'model_name': model_name,
                        'version': version.version,
                        metric: metrics.get(metric, None),
                        'run_id': version.run_id
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            logger.info(f"✅ Model comparison completed for {len(model_names)} models")
            return comparison_df
            
        except Exception as e:
            logger.error(f"❌ Failed to compare models: {e}")
            raise
    
    def get_experiment_runs(self, 
                           experiment_name: str = None,
                           max_results: int = 100) -> pd.DataFrame:
        """
        Get all runs for an experiment.
        
        Args:
            experiment_name: Name of the experiment
            max_results: Maximum number of results to return
            
        Returns:
            DataFrame with run information
        """
        experiment_name = experiment_name or self.experiment_name
        
        try:
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(experiment_name)
            
            if not experiment:
                logger.warning(f"Experiment {experiment_name} not found")
                return pd.DataFrame()
            
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=max_results
            )
            
            run_data = []
            for run in runs:
                run_data.append({
                    'run_id': run.info.run_id,
                    'run_name': run.data.tags.get('mlflow.runName', ''),
                    'start_time': run.info.start_time,
                    'end_time': run.info.end_time,
                    'status': run.info.status,
                    'metrics': run.data.metrics,
                    'params': run.data.params
                })
            
            runs_df = pd.DataFrame(run_data)
            logger.info(f"✅ Retrieved {len(runs_df)} runs for experiment {experiment_name}")
            return runs_df
            
        except Exception as e:
            logger.error(f"❌ Failed to get experiment runs: {e}")
            raise
    
    def cleanup_old_runs(self, 
                         experiment_name: str = None,
                         days_to_keep: int = 30) -> int:
        """
        Clean up old runs to save storage space.
        
        Args:
            experiment_name: Name of the experiment
            days_to_keep: Number of days to keep runs
            
        Returns:
            int: Number of runs deleted
        """
        experiment_name = experiment_name or self.experiment_name
        
        try:
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(experiment_name)
            
            if not experiment:
                return 0
            
            cutoff_time = datetime.now().timestamp() * 1000 - (days_to_keep * 24 * 60 * 60 * 1000)
            
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"start_time < {cutoff_time}"
            )
            
            deleted_count = 0
            for run in runs:
                try:
                    client.delete_run(run.info.run_id)
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete run {run.info.run_id}: {e}")
            
            logger.info(f"✅ Cleaned up {deleted_count} old runs")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old runs: {e}")
            return 0

# Global MLflow manager instance
mlflow_manager = None

def get_mlflow_manager(config_path: str = "config/main_config.yaml") -> MLflowManager:
    """Get or create global MLflow manager instance."""
    global mlflow_manager
    if mlflow_manager is None:
        mlflow_manager = MLflowManager(config_path)
    return mlflow_manager

def log_training_run_wrapper(model, 
                            X_train: pd.DataFrame,
                            y_train: pd.Series,
                            X_test: pd.DataFrame,
                            y_test: pd.Series,
                            metrics: Dict[str, float],
                            params: Dict[str, Any],
                            **kwargs) -> str:
    """Wrapper function for logging training runs."""
    manager = get_mlflow_manager()
    return manager.log_training_run(
        model, X_train, y_train, X_test, y_test, metrics, params, **kwargs
    )

def log_prediction_batch_wrapper(predictions: np.ndarray,
                                probabilities: np.ndarray,
                                **kwargs) -> str:
    """Wrapper function for logging prediction batches."""
    manager = get_mlflow_manager()
    return manager.log_prediction_batch(predictions, probabilities, **kwargs)
