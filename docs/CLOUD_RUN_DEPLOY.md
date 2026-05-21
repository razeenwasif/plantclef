# Cloud Run Deployment Guide: Oracle

## 1. Config
gcloud config set project oracle-neuro-sym
gcloud services enable run.googleapis.com artifactregistry.googleapis.com builds.googleapis.com

## 2. Repository
gcloud artifacts repositories create cloud-run-source-lib --repository-format=docker --location=us-central1

## 3. Build
gcloud builds submit --tag us-central1-docker.pkg.dev/oracle-neuro-sym/cloud-run-source-lib/oracle-backend .

## 4. Deploy
gcloud run deploy oracle-backend --image us-central1-docker.pkg.dev/oracle-neuro-sym/cloud-run-source-lib/oracle-backend --region us-central1 --allow-unauthenticated --memory 4Gi
