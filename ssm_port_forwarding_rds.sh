#!/bin/bash
set -ex

# Add Session Manager Plugin to PATH for Git Bash
export PATH="/c/Program Files/Amazon/SessionManagerPlugin/bin:$PATH"

REGION="us-east-1"
INSTANCE_ID="i-074ff6482997c99a1" # b2b_leads_ssh_tunnel
REMOTE_PORT="5432" # Port on EC2 where Postgres runs
LOCAL_PORT="5431" # Local port to forward to
PROFILE="beamdata-dev" # Replace with your AWS CLI profile
RDS_ENDPOINT="b2b-leads-database.cybm6jthqldn.us-east-1.rds.amazonaws.com"

echo "Starting SSM port forwarding session..."
echo "Region: $REGION"
echo "Instance ID: $INSTANCE_ID"
echo "RDS Endpoint: $RDS_ENDPOINT"
echo "Local Port: $LOCAL_PORT"
echo "Remote Port: $REMOTE_PORT"
echo "Profile: $PROFILE"
echo ""

aws ssm start-session \
    --region $REGION \
    --target $INSTANCE_ID \
    --document-name AWS-StartPortForwardingSessionToRemoteHost \
    --parameters host=$RDS_ENDPOINT,portNumber=$REMOTE_PORT,localPortNumber=$LOCAL_PORT \
    --profile "$PROFILE"
