#!/usr/bin/env python3
"""
Lead Maturity Configuration
Based on transcript discussion about lead lifecycle timing
"""

import os
from datetime import timedelta

class LeadMaturityConfig:
    """Configuration for lead maturity and lifecycle timing"""
    
    def __init__(self):
        # Timeframe for considering leads "mature" for training
        # Default: 1 month (Mitchell's suggestion)
        # Alternative: 2 weeks (Ricardo's observation from sales team)
        self.maturity_days = int(os.getenv('LEAD_MATURITY_DAYS', '30'))  # 30 days = 1 month
        
        # Alternative configurations for different use cases
        self.quick_maturity_days = 14  # 2 weeks (sales team practice)
        self.extended_maturity_days = 45  # 6 weeks (conservative)
        
        # Follow-up timeframes
        self.instant_follow_up_hours = 24  # Immediate follow-up window
        self.short_follow_up_days = 7     # Short-term follow-up
        self.long_follow_up_days = 30     # Long-term follow-up
        
        # Engagement scoring windows
        self.engagement_window_days = 14   # Primary engagement window
        self.extended_engagement_days = 30 # Extended engagement tracking
        
    def get_maturity_cutoff(self):
        """Get the datetime cutoff for mature leads"""
        from datetime import datetime
        return datetime.now() - timedelta(days=self.maturity_days)
    
    def get_engagement_cutoff(self):
        """Get the datetime cutoff for engagement scoring"""
        from datetime import datetime
        return datetime.now() - timedelta(days=self.engagement_window_days)
    
    def should_consider_mature(self, lead_created_date):
        """Check if a lead should be considered mature for training"""
        from datetime import datetime
        if not lead_created_date:
            return False
        
        age_days = (datetime.now() - lead_created_date).days
        return age_days >= self.maturity_days
    
    def get_lead_stage(self, lead_created_date, last_activity_date=None):
        """Determine the current stage of a lead"""
        from datetime import datetime
        
        if not lead_created_date:
            return 'unknown'
        
        age_days = (datetime.now() - lead_created_date).days
        
        if last_activity_date:
            activity_age_days = (datetime.now() - last_activity_date).days
        else:
            activity_age_days = age_days
        
        # Immediate follow-up period
        if age_days <= 1:
            return 'immediate'
        
        # Active engagement window
        elif age_days <= self.engagement_window_days:
            if activity_age_days <= 3:
                return 'hot'
            elif activity_age_days <= 7:
                return 'warm'
            else:
                return 'cooling'
        
        # Extended engagement window
        elif age_days <= self.extended_engagement_days:
            if activity_age_days <= 7:
                return 'lukewarm'
            else:
                return 'cold'
        
        # Mature leads (ready for training)
        elif age_days >= self.maturity_days:
            return 'mature'
        
        else:
            return 'aging'

# SQL queries for different maturity configurations
class LeadMaturityQueries:
    """SQL queries based on maturity configuration"""
    
    @staticmethod
    def get_mature_leads_query(maturity_days=30):
        """Get SQL query for mature leads"""
        return f"""
        SELECT * FROM ml_lead_scoring.silver_ml_features
        WHERE created_timestamp <= CURRENT_TIMESTAMP - INTERVAL '{maturity_days} days'
          AND lead_id NOT IN (
              SELECT DISTINCT lead_id 
              FROM ml_lead_scoring.gold_training_data 
              WHERE lead_id IS NOT NULL
          )
        """
    
    @staticmethod
    def get_engagement_window_query(engagement_days=14):
        """Get SQL query for leads in engagement window"""
        return f"""
        SELECT * FROM ml_lead_scoring.bronze_instantly_leads
        WHERE created_date >= CURRENT_TIMESTAMP - INTERVAL '{engagement_days} days'
          AND (
              last_activity_date >= CURRENT_TIMESTAMP - INTERVAL '{engagement_days} days'
              OR last_activity_date IS NULL
          )
        """
    
    @staticmethod
    def get_stale_leads_query(stale_days=60):
        """Get SQL query for stale leads that need re-evaluation"""
        return f"""
        SELECT * FROM ml_lead_scoring.silver_ml_features
        WHERE updated_timestamp <= CURRENT_TIMESTAMP - INTERVAL '{stale_days} days'
          AND is_qualified_lead IS NOT NULL
        """

# Environment-based configuration
def get_maturity_config():
    """Get maturity configuration based on environment"""
    config = LeadMaturityConfig()
    
    # Override based on environment variables
    environment = os.getenv('ENVIRONMENT', 'production')
    
    if environment == 'development':
        # Faster iteration for development
        config.maturity_days = 7  # 1 week for testing
        config.engagement_window_days = 3  # 3 days
    elif environment == 'staging':
        # Medium timeframe for staging
        config.maturity_days = 14  # 2 weeks
        config.engagement_window_days = 7  # 1 week
    elif environment == 'sales_team_preference':
        # Based on Ricardo's observation
        config.maturity_days = 14  # 2 weeks (sales team practice)
        config.engagement_window_days = 14  # 2 weeks
    else:
        # Production default (Mitchell's recommendation)
        config.maturity_days = 30  # 1 month
        config.engagement_window_days = 14  # 2 weeks
    
    return config

if __name__ == "__main__":
    # Test the configuration
    config = get_maturity_config()
    print(f"Maturity days: {config.maturity_days}")
    print(f"Engagement window: {config.engagement_window_days}")
    print(f"Maturity cutoff: {config.get_maturity_cutoff()}")
    print(f"Engagement cutoff: {config.get_engagement_cutoff()}")
