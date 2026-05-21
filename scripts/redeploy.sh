#!/bin/bash

# --- ORACLE: Unified Redeployment Script ---
# This script handles the backend Cloud Run deployment and the frontend Firebase Hosting deployment.

set -e # Exit on any error

# Always operate from the repo root regardless of where the caller invokes us.
cd "$(dirname "$0")/.."

PROJECT_ID="oracle-neuro-sym"
REGION="us-central1"
IMAGE_TAG="us-central1-docker.pkg.dev/$PROJECT_ID/cloud-run-source-lib/oracle-backend"

echo "----------------------------------------------------"
echo "🚀 STAGE 1: Backend Deployment (Google Cloud Run)"
echo "----------------------------------------------------"

# Ensure gcloud is configured to the correct project
echo "[*] Setting gcloud project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Build the container using Cloud Builds
echo "[*] Submitting build to Artifact Registry..."
gcloud builds submit --tag $IMAGE_TAG .

# Deploy to Cloud Run
echo "[*] Deploying to Cloud Run..."
gcloud run deploy oracle-backend \
    --image $IMAGE_TAG \
    --region $REGION \
    --allow-unauthenticated \
    --memory 4Gi

echo "✅ Backend deployed successfully."

echo ""
echo "----------------------------------------------------"
echo "🎨 STAGE 2: Frontend Deployment (Firebase Hosting)"
echo "----------------------------------------------------"

# Build the React dashboard
echo "[*] Building dashboard..."
cd dashboard
npm install
npm run build
cd ..

# Deploy to Firebase
echo "[*] Deploying to Firebase Hosting..."
firebase deploy --only hosting --project $PROJECT_ID

echo "✅ Frontend deployed successfully."

echo ""
echo "----------------------------------------------------"
echo "🏁 DEPLOYMENT COMPLETE"
echo "----------------------------------------------------"
echo "Frontend: https://$PROJECT_ID.web.app"
echo "----------------------------------------------------"
