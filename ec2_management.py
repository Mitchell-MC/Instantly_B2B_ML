#!/usr/bin/env python3
"""
EC2 Instance Management for ML Processing
Based on transcript discussion about turning EC2 instances on/off for ML tasks
"""

import boto3
import logging
import time
import os
from typing import Optional, Dict, Any
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EC2Manager:
    """Manage EC2 instances for ML processing"""
    
    def __init__(self, instance_id: str, region: str = 'us-east-1'):
        """Initialize EC2 manager"""
        self.instance_id = instance_id
        self.region = region
        
        # Initialize AWS clients
        self.ec2 = boto3.client('ec2', region_name=region)
        self.ec2_resource = boto3.resource('ec2', region_name=region)
        self.instance = self.ec2_resource.Instance(instance_id)
        
    def get_instance_status(self) -> Dict[str, Any]:
        """Get current instance status"""
        try:
            self.instance.reload()
            return {
                'instance_id': self.instance_id,
                'state': self.instance.state['Name'],
                'instance_type': self.instance.instance_type,
                'public_ip': self.instance.public_ip_address,
                'private_ip': self.instance.private_ip_address,
                'availability_zone': self.instance.placement['AvailabilityZone']
            }
        except Exception as e:
            logger.error(f"Error getting instance status: {str(e)}")
            return {'error': str(e)}
    
    def start_instance(self, wait_for_running: bool = True) -> Dict[str, Any]:
        """Start the EC2 instance"""
        try:
            logger.info(f"Starting EC2 instance: {self.instance_id}")
            
            # Check current state
            status = self.get_instance_status()
            if status.get('state') == 'running':
                logger.info("Instance is already running")
                return status
            
            # Start the instance
            response = self.ec2.start_instances(InstanceIds=[self.instance_id])
            logger.info(f"Start request sent: {response}")
            
            if wait_for_running:
                logger.info("Waiting for instance to be running...")
                self.instance.wait_until_running(
                    WaiterConfig={
                        'Delay': 15,  # Check every 15 seconds
                        'MaxAttempts': 40  # Wait up to 10 minutes
                    }
                )
                logger.info("Instance is now running")
                
                # Wait additional time for services to start
                time.sleep(30)
                
            return self.get_instance_status()
            
        except Exception as e:
            logger.error(f"Error starting instance: {str(e)}")
            return {'error': str(e)}
    
    def stop_instance(self, wait_for_stopped: bool = True) -> Dict[str, Any]:
        """Stop the EC2 instance"""
        try:
            logger.info(f"Stopping EC2 instance: {self.instance_id}")
            
            # Check current state
            status = self.get_instance_status()
            if status.get('state') in ['stopped', 'stopping']:
                logger.info("Instance is already stopped or stopping")
                return status
            
            # Stop the instance
            response = self.ec2.stop_instances(InstanceIds=[self.instance_id])
            logger.info(f"Stop request sent: {response}")
            
            if wait_for_stopped:
                logger.info("Waiting for instance to be stopped...")
                self.instance.wait_until_stopped(
                    WaiterConfig={
                        'Delay': 15,  # Check every 15 seconds
                        'MaxAttempts': 40  # Wait up to 10 minutes
                    }
                )
                logger.info("Instance is now stopped")
                
            return self.get_instance_status()
            
        except Exception as e:
            logger.error(f"Error stopping instance: {str(e)}")
            return {'error': str(e)}
    
    def execute_ml_task(self, task_type: str, parameters: Dict = None) -> Dict[str, Any]:
        """Execute ML task on the instance"""
        try:
            # Start instance if not running
            start_result = self.start_instance(wait_for_running=True)
            if 'error' in start_result:
                return start_result
            
            # Get instance IP for SSH/API calls
            status = self.get_instance_status()
            public_ip = status.get('public_ip')
            
            if not public_ip:
                return {'error': 'Instance has no public IP address'}
            
            # Execute the task based on type
            if task_type == 'train_model':
                result = self._execute_training(public_ip, parameters)
            elif task_type == 'predict_batch':
                result = self._execute_prediction(public_ip, parameters)
            elif task_type == 'model_evaluation':
                result = self._execute_evaluation(public_ip, parameters)
            else:
                result = {'error': f'Unknown task type: {task_type}'}
            
            # Optionally stop instance after task completion
            if os.getenv('AUTO_STOP_AFTER_TASK', 'true').lower() == 'true':
                logger.info("Auto-stopping instance after task completion")
                self.stop_instance(wait_for_stopped=False)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing ML task: {str(e)}")
            return {'error': str(e)}
    
    def _execute_training(self, public_ip: str, parameters: Dict) -> Dict[str, Any]:
        """Execute model training on the instance"""
        import requests
        
        try:
            # Wait for ML service to be ready
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    health_url = f"http://{public_ip}:5001/health"
                    response = requests.get(health_url, timeout=10)
                    if response.status_code == 200:
                        break
                except:
                    logger.info(f"ML service not ready, attempt {attempt + 1}/{max_retries}")
                    time.sleep(30)
            else:
                return {'error': 'ML service did not become ready'}
            
            # Start training
            train_url = f"http://{public_ip}:5001/api/train-model"
            response = requests.post(
                train_url, 
                json=parameters or {'model_type': 'random_forest'},
                timeout=300  # 5 minutes timeout
            )
            
            if response.status_code == 200:
                return {
                    'status': 'success',
                    'task': 'training',
                    'result': response.json()
                }
            else:
                return {
                    'status': 'error',
                    'task': 'training',
                    'error': f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {'error': f'Training execution failed: {str(e)}'}
    
    def _execute_prediction(self, public_ip: str, parameters: Dict) -> Dict[str, Any]:
        """Execute batch prediction on the instance"""
        import requests
        
        try:
            # Wait for ML service to be ready
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    health_url = f"http://{public_ip}:5001/health"
                    response = requests.get(health_url, timeout=10)
                    if response.status_code == 200:
                        break
                except:
                    logger.info(f"ML service not ready, attempt {attempt + 1}/{max_retries}")
                    time.sleep(15)
            else:
                return {'error': 'ML service did not become ready'}
            
            # Execute prediction
            predict_url = f"http://{public_ip}:5001/api/predict"
            response = requests.post(
                predict_url,
                json=parameters or {},
                timeout=120  # 2 minutes timeout
            )
            
            if response.status_code == 200:
                return {
                    'status': 'success',
                    'task': 'prediction',
                    'result': response.json()
                }
            else:
                return {
                    'status': 'error',
                    'task': 'prediction',
                    'error': f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {'error': f'Prediction execution failed: {str(e)}'}
    
    def _execute_evaluation(self, public_ip: str, parameters: Dict) -> Dict[str, Any]:
        """Execute model evaluation on the instance"""
        import requests
        
        try:
            # Get model performance metrics
            perf_url = f"http://{public_ip}:5001/api/model-performance"
            response = requests.get(perf_url, timeout=60)
            
            if response.status_code == 200:
                return {
                    'status': 'success',
                    'task': 'evaluation',
                    'result': response.json()
                }
            else:
                return {
                    'status': 'error',
                    'task': 'evaluation',
                    'error': f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {'error': f'Evaluation execution failed: {str(e)}'}

class EC2CostOptimizer:
    """Optimize EC2 costs by managing instance lifecycle"""
    
    def __init__(self, instance_id: str, region: str = 'us-east-1'):
        self.manager = EC2Manager(instance_id, region)
        self.instance_id = instance_id
        
    def estimate_task_duration(self, task_type: str, data_size: int) -> int:
        """Estimate task duration in minutes"""
        # Simple estimation based on task type and data size
        base_times = {
            'train_model': 30,  # 30 minutes base
            'predict_batch': 5,  # 5 minutes base
            'model_evaluation': 10  # 10 minutes base
        }
        
        base_time = base_times.get(task_type, 15)
        
        # Scale based on data size (rough estimation)
        if data_size > 100000:  # Large dataset
            scale_factor = 3
        elif data_size > 10000:  # Medium dataset
            scale_factor = 2
        else:  # Small dataset
            scale_factor = 1
            
        return base_time * scale_factor
    
    def schedule_ml_task(self, task_type: str, parameters: Dict = None, 
                        scheduled_time: Optional[str] = None) -> Dict[str, Any]:
        """Schedule ML task with cost optimization"""
        try:
            if scheduled_time:
                # TODO: Implement scheduling logic
                logger.info(f"Task scheduled for {scheduled_time}")
                
            # For now, execute immediately
            result = self.manager.execute_ml_task(task_type, parameters)
            
            # Log cost information
            duration_estimate = self.estimate_task_duration(
                task_type, 
                parameters.get('data_size', 1000) if parameters else 1000
            )
            
            logger.info(f"Task duration estimate: {duration_estimate} minutes")
            
            return {
                **result,
                'cost_optimization': {
                    'estimated_duration_minutes': duration_estimate,
                    'auto_stop_enabled': os.getenv('AUTO_STOP_AFTER_TASK', 'true') == 'true'
                }
            }
            
        except Exception as e:
            logger.error(f"Error scheduling ML task: {str(e)}")
            return {'error': str(e)}

# CLI interface for EC2 management
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='EC2 ML Instance Management')
    parser.add_argument('--instance-id', required=True, help='EC2 Instance ID')
    parser.add_argument('--region', default='us-east-1', help='AWS Region')
    parser.add_argument('--action', choices=['start', 'stop', 'status', 'train', 'predict'], 
                       required=True, help='Action to perform')
    parser.add_argument('--parameters', type=str, help='JSON parameters for ML tasks')
    
    args = parser.parse_args()
    
    manager = EC2Manager(args.instance_id, args.region)
    
    if args.action == 'start':
        result = manager.start_instance()
    elif args.action == 'stop':
        result = manager.stop_instance()
    elif args.action == 'status':
        result = manager.get_instance_status()
    elif args.action == 'train':
        params = json.loads(args.parameters) if args.parameters else {}
        result = manager.execute_ml_task('train_model', params)
    elif args.action == 'predict':
        params = json.loads(args.parameters) if args.parameters else {}
        result = manager.execute_ml_task('predict_batch', params)
    
    print(json.dumps(result, indent=2))
