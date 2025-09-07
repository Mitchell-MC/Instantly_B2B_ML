#!/usr/bin/env python3
"""
Automated Cleanup Script for ML Lead Scoring Project
Safely removes redundant files while preserving important data
"""

import os
import shutil
import tarfile
import json
from datetime import datetime
from pathlib import Path
import argparse

class ProjectCleanup:
    def __init__(self, dry_run=True, backup=True):
        self.dry_run = dry_run
        self.backup = backup
        self.base_dir = Path(".")
        self.backup_dir = Path(f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Files and directories to remove
        self.removal_targets = {
            'old_ml_implementation': {
                'path': 'Instantly B2B Main/Instantly B2B ML/src',
                'reason': 'Old ML implementation superseded by new system',
                'risk': 'LOW'
            },
            'old_models': {
                'path': 'Instantly B2B Main/Instantly B2B ML/models',
                'reason': 'Old trained models incompatible with new schema',
                'risk': 'LOW'
            },
            'old_n8n_workflows': {
                'files': [
                    'Instantly B2B Main/Instantly B2B ML/N8N automation/full_ml_pipeline.json',
                    'Instantly B2B Main/Instantly B2B ML/N8N automation/lead_search_predict_pipeline.json',
                    'Instantly B2B Main/Instantly B2B ML/N8N automation/lead_enrichment_train_pipeline.json'
                ],
                'reason': 'Replaced by new N8N workflows in project root',
                'risk': 'LOW'
            },
            'analysis_images': {
                'pattern': 'Instantly B2B Main/Instantly B2B ML/*.png',
                'reason': 'Old analysis plots and confusion matrices',
                'risk': 'LOW'
            },
            'log_files': {
                'pattern': 'Instantly B2B Main/Instantly B2B ML/logs/*.log',
                'reason': 'Old log files',
                'risk': 'NONE'
            },
            'temp_data': {
                'path': 'Instantly B2B Main/Instantly B2B ML/data',
                'reason': 'Old monitoring data and temporary files',
                'risk': 'LOW'
            },
            'old_config': {
                'path': 'Instantly B2B Main/Instantly B2B ML/config',
                'reason': 'Old configuration files',
                'risk': 'MEDIUM'  # May contain useful settings
            }
        }
        
        # Files to preserve (critical files)
        self.preserve_files = [
            'ml_model_service.py',
            'data_processing_service.py', 
            'monitoring_dashboard.py',
            'ec2_management.py',
            'lead_maturity_config.py',
            'n8n_instantly_ingestion_workflow.json',
            'n8n_apollo_enrichment_workflow.json', 
            'n8n_prediction_workflow.json',
            'docker-compose.yml',
            'ml_lead_scoring_schema.sql',
            'IMPLEMENTATION_GUIDE.md',
            'requirements.txt'
        ]
        
        # Files to review before removal (may have valuable logic)
        self.review_files = [
            'Instantly B2B Main/Instantly B2B ML/instantly_api_query.py',
            'Instantly B2B Main/Instantly B2B ML/apollo_feature_analysis.py',
            'Instantly B2B Main/Instantly B2B ML/campaign_analysis.py',
            'Instantly B2B Main/Instantly B2B ML/enhanced_*.py',
            'Instantly B2B Main/Instantly B2B ML/*.csv'  # Data files
        ]
    
    def create_backup(self):
        """Create backup of files before removal"""
        if not self.backup:
            return
            
        print(f"Creating backup in {self.backup_dir}")
        
        if not self.dry_run:
            self.backup_dir.mkdir(exist_ok=True)
            
            # Create tarball of old implementation
            old_ml_dir = Path('Instantly B2B Main/Instantly B2B ML')
            if old_ml_dir.exists():
                backup_file = self.backup_dir / 'instantly_b2b_ml_backup.tar.gz'
                with tarfile.open(backup_file, 'w:gz') as tar:
                    tar.add(old_ml_dir, arcname='instantly_b2b_ml_old')
                print(f"✅ Backup created: {backup_file}")
    
    def analyze_removal_impact(self):
        """Analyze what will be removed and calculate impact"""
        impact = {
            'total_files': 0,
            'total_size': 0,
            'categories': {}
        }
        
        for category, config in self.removal_targets.items():
            files_to_remove = []
            total_size = 0
            
            if 'path' in config:
                path = Path(config['path'])
                if path.exists():
                    if path.is_dir():
                        files_to_remove.extend(list(path.rglob('*')))
                    else:
                        files_to_remove.append(path)
            
            elif 'files' in config:
                for file_path in config['files']:
                    path = Path(file_path)
                    if path.exists():
                        files_to_remove.append(path)
            
            elif 'pattern' in config:
                # Handle glob patterns
                pattern_parts = config['pattern'].split('/')
                base_path = Path('/'.join(pattern_parts[:-1]))
                pattern = pattern_parts[-1]
                if base_path.exists():
                    files_to_remove.extend(list(base_path.glob(pattern)))
            
            # Calculate size
            for file_path in files_to_remove:
                if file_path.is_file():
                    try:
                        total_size += file_path.stat().st_size
                    except (OSError, FileNotFoundError):
                        pass
            
            impact['categories'][category] = {
                'files': len([f for f in files_to_remove if f.is_file()]),
                'size_mb': round(total_size / (1024 * 1024), 2),
                'risk': config['risk'],
                'reason': config['reason']
            }
            
            impact['total_files'] += len([f for f in files_to_remove if f.is_file()])
            impact['total_size'] += total_size
        
        impact['total_size_mb'] = round(impact['total_size'] / (1024 * 1024), 2)
        return impact
    
    def check_review_files(self):
        """Check files that should be reviewed before removal"""
        review_items = []
        
        for pattern in self.review_files:
            if '*' in pattern:
                # Handle glob patterns
                path_parts = pattern.split('/')
                base_path = Path('/'.join(path_parts[:-1]))
                glob_pattern = path_parts[-1]
                if base_path.exists():
                    matches = list(base_path.glob(glob_pattern))
                    for match in matches:
                        if match.is_file():
                            review_items.append(str(match))
            else:
                path = Path(pattern)
                if path.exists():
                    review_items.append(str(path))
        
        return review_items
    
    def perform_cleanup(self):
        """Perform the actual cleanup"""
        removed_count = 0
        
        for category, config in self.removal_targets.items():
            print(f"\n🗑️  Processing {category}...")
            
            files_to_remove = []
            
            if 'path' in config:
                path = Path(config['path'])
                if path.exists():
                    files_to_remove.append(path)
            
            elif 'files' in config:
                for file_path in config['files']:
                    path = Path(file_path)
                    if path.exists():
                        files_to_remove.append(path)
            
            elif 'pattern' in config:
                pattern_parts = config['pattern'].split('/')
                base_path = Path('/'.join(pattern_parts[:-1]))
                pattern = pattern_parts[-1]
                if base_path.exists():
                    files_to_remove.extend(list(base_path.glob(pattern)))
            
            for item in files_to_remove:
                if not item.exists():
                    continue
                    
                if self.dry_run:
                    print(f"  [DRY RUN] Would remove: {item}")
                else:
                    try:
                        if item.is_dir():
                            shutil.rmtree(item)
                            print(f"  ✅ Removed directory: {item}")
                        else:
                            item.unlink()
                            print(f"  ✅ Removed file: {item}")
                        removed_count += 1
                    except Exception as e:
                        print(f"  ❌ Failed to remove {item}: {e}")
        
        return removed_count
    
    def run_cleanup(self):
        """Main cleanup execution"""
        print("🔍 ML Lead Scoring Project Cleanup")
        print("=" * 50)
        
        # Analyze impact
        print("\n📊 Analyzing removal impact...")
        impact = self.analyze_removal_impact()
        
        print(f"\n📈 Cleanup Impact Summary:")
        print(f"Total files to remove: {impact['total_files']}")
        print(f"Total storage to free: {impact['total_size_mb']:.2f} MB")
        
        print(f"\n📋 Breakdown by category:")
        for category, data in impact['categories'].items():
            print(f"  {category}: {data['files']} files, {data['size_mb']:.2f} MB, Risk: {data['risk']}")
            print(f"    Reason: {data['reason']}")
        
        # Check files that need review
        print(f"\n⚠️  Files requiring manual review:")
        review_files = self.check_review_files()
        for file_path in review_files[:10]:  # Show first 10
            print(f"  📄 {file_path}")
        if len(review_files) > 10:
            print(f"  ... and {len(review_files) - 10} more files")
        
        # Confirm action
        if not self.dry_run:
            confirm = input(f"\n❓ Proceed with cleanup? (y/N): ")
            if confirm.lower() != 'y':
                print("❌ Cleanup cancelled")
                return
        
        # Create backup
        if self.backup and not self.dry_run:
            self.create_backup()
        
        # Perform cleanup
        print(f"\n🧹 {'[DRY RUN] ' if self.dry_run else ''}Starting cleanup...")
        removed_count = self.perform_cleanup()
        
        print(f"\n✅ Cleanup completed!")
        print(f"{'Would remove' if self.dry_run else 'Removed'} {removed_count} items")
        
        if not self.dry_run and self.backup:
            print(f"💾 Backup saved to: {self.backup_dir}")
        
        # Next steps
        print(f"\n🎯 Next Steps:")
        print("1. Review the files marked for manual review")
        print("2. Test the new ML lead scoring system")
        print("3. Update documentation if needed")
        print("4. Consider consolidating requirements.txt files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cleanup ML Lead Scoring Project')
    parser.add_argument('--dry-run', action='store_true', default=True,
                       help='Show what would be removed without actually removing (default)')
    parser.add_argument('--execute', action='store_true', 
                       help='Actually perform the cleanup')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip creating backup (not recommended)')
    
    args = parser.parse_args()
    
    # Safety check
    dry_run = not args.execute
    backup = not args.no_backup
    
    cleanup = ProjectCleanup(dry_run=dry_run, backup=backup)
    cleanup.run_cleanup()
