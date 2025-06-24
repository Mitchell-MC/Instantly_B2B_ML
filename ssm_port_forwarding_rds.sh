#!/bin/bash
set -ex

REGION="us-east-1"
INSTANCE_ID="i-06a1b05741e4930ce" # b2b_leads_ssh_tunnel
REMOTE_PORT="5432" # Port on EC2 where Postgres runs
LOCAL_PORT="5431" # Local port to forward to
PROFILE="beamdata-dev" # Replace with your AWS CLI profile
RDS_ENDPOINT="b2b-leads-database.cybm6jthqldn.us-east-1.rds.amazonaws.com"

aws ssm start-session \
    --region $REGION \
    --target $INSTANCE_ID \
    --document-name AWS-StartPortForwardingSessionToRemoteHost \
    --parameters host=$RDS_ENDPOINT,portNumber=$REMOTE_PORT,localPortNumber=$LOCAL_PORT \
    --profile "$PROFILE"
