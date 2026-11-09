# Customer Segmentation — Complete How to Run Guide
**Author: Ashutosh Behera | Data Scientist | Bandhan Bank Ltd**

---

## What Is This Project?

Customer Segmentation assigns 70,000 Big Basket customers into 7 groups
using K-Means++ clustering trained in Google Colab (Notebook 7).

The deployment is a **simple web app + REST API** — NO drift detection, NO Lambda,
NO EventBridge, NO SageMaker pipeline. Just the model serving.

```
Marketing manager → opens Streamlit at http://EXTERNAL-IP:8501
                  → uploads CSV → gets segments + AI messages → downloads CSV
                  → sends WhatsApp/Email per segment

CRM system        → calls POST /predict/s3 via API at port 8000
                  → results saved back to S3 automatically
```

---

## What Is Already Done (from Google Colab Notebooks)

```
✅ NB1-NB7 ran in Google Colab / SageMaker
✅ K-Means++ model trained on 70,000 customers
✅ All model files saved to S3

S3 bucket: customer-segmentation-2026
  models/kmeans_centroids.csv     ← 7 centroids × 34 features (CORE FILE)
  models/cluster_profiles.csv     ← segment profiles for dashboard
  models/kmeans_plus_model.pkl    ← full model
  models/cluster_features.csv     ← 34 feature names
  data/07_clustered.csv           ← all 70k customers with segments
```

---

## Project Structure

```
customer-segmentation-deploy/
├── src/
│   ├── segment.py          ← Core centroid mapping — instant segment assignment
│   ├── llm_messages.py     ← OpenAI message generation per segment
│   ├── api.py              ← FastAPI REST endpoints for CRM integration
│   └── app.py              ← Streamlit dashboard — 3 tabs
├── k8s/
│   └── deployment.yml      ← Kubernetes — 2 replicas, LoadBalancer
├── .github/workflows/
│   └── deploy.yml          ← CI/CD — push code → ECR → EKS auto deploy
├── Dockerfile              ← Port 8501 (Streamlit) + Port 8000 (FastAPI)
├── requirements.txt
├── .env                    ← Your real AWS keys (gitignored)
└── HOW_TO_RUN.md           ← This file
```

---

## How the LLM Works Here

```
Customer data comes in (CSV or API call)
        ↓
segment.py finds nearest centroid → assigns segment
        ↓
llm_messages.py builds prompt:
  "You are Big Basket customer manager.
   Write WhatsApp message for Priya Sharma.
   She is At-Risk — bought 24x/year spending Rs 22,000.
   Silent 45 days. Usual items: Fruits & Vegetables.
   Offer: Rs 150 off — 48 hours. Tone: warm, personal."
        ↓
OpenAI GPT-4o-mini generates unique personalised message
        ↓
Message returned in CSV column "llm_message"
        ↓
Marketing team sends via WhatsApp Business API
```

---

# PHASE 1 — Local Setup and Test

## Step 1 — Download and Open in VS Code

```bash
# Download and unzip customer-segmentation-deploy.zip
# Right click → Extract All → Desktop
# Open VS Code → File → Open Folder → customer-segmentation-deploy
```

## Step 2 — Create Virtual Environment (Git Bash)

```bash
# Open Git Bash terminal in VS Code:
# Ctrl+Shift+P → Terminal: Select Default Profile → Git Bash
# Then Ctrl+` to open terminal

python -m venv venv
source venv/Scripts/activate        # Windows Git Bash
pip install -r requirements.txt
```

## Step 3 — Verify .env File

```bash
cat .env

# Should show:
# AWS_ACCESS_KEY_ID=<your-access-key>
# AWS_SECRET_ACCESS_KEY=<your-secret-key>
# AWS_REGION=ap-southeast-2
# S3_BUCKET=customer-segmentation-2026-1
# OPENAI_API_KEY=your-openai-key-here
```

## Step 4 — Verify S3 Has Model Files

```bash
aws configure
# AWS_ACCESS_KEY_ID=<your-access-key>
# AWS_SECRET_ACCESS_KEY=<your-secret-key>
# Default region:        ap-southeast-2
# Default output format: json

aws s3 ls s3://customer-segmentation-2026/models/

# Must see all 4 files:
# kmeans_centroids.csv     ✅
# cluster_profiles.csv     ✅
# kmeans_plus_model.pkl    ✅
# cluster_features.csv     ✅

# If any missing → re-run NB6 and NB7 in Google Colab first
```

## Step 5 — Upload Sample Data to S3

```bash
# Upload the 10k sample customers for testing
aws s3 cp sample_10k_customers.csv \
    s3://customer-segmentation-2026/uploads/sample_10k_customers.csv

echo "Sample data uploaded ✅"
```

## Step 6 — Test Segmentation Locally

```bash
python src/segment.py

# Expected output:
# Loading model artifacts from S3...
# Loaded 7 segment centroids with 34 features
# Segment: At Risk | Confidence: 72.45%
```

## Step 7 — Run Streamlit Dashboard Locally

```bash
streamlit run src/app.py
# Open http://localhost:8501 in browser

# Test Tab 1:
#   Fill in customer details → click Predict Segment
#   See: segment card + centroid distance chart + LLM chat bubble

# Test Tab 2:
#   Upload sample_10k_customers.csv
#   Click Segment All Customers
#   See: charts + results table + download buttons
```

## Step 8 — Run FastAPI Locally

```bash
uvicorn src.api:app --reload --port 8000
# Open http://localhost:8000/docs
# Test the /predict and /predict/s3 endpoints
```

---

# PHASE 2 — ECR (Docker Image Storage)

## Step 9 — Create ECR Repository

```bash
aws ecr create-repository \
    --repository-name customer-segmentation \
    --region ap-southeast-2

# Output shows your ECR URL:
# ACCOUNT_ID.dkr.ecr.ap-southeast-2.amazonaws.com/customer-segmentation
```

## Step 10 — Build and Push Docker Image

⚠️ **Switch to mobile hotspot for Docker push**

```bash
# Build image
docker build -t customer-segmentation:v1.0 .

# Login to ECR
aws ecr get-login-password --region ap-southeast-2 | \
    docker login --username AWS --password-stdin \
    ACCOUNT_ID.dkr.ecr.ap-southeast-2.amazonaws.com

# Tag and push
docker tag customer-segmentation:v1.0 \
    ACCOUNT_ID.dkr.ecr.ap-southeast-2.amazonaws.com/customer-segmentation:latest

docker push \
    ACCOUNT_ID.dkr.ecr.ap-southeast-2.amazonaws.com/customer-segmentation:latest

# Verify pushed
aws ecr list-images \
    --repository-name customer-segmentation \
    --region ap-southeast-2
# Should show imageTag: "latest" ✅
```

---

# PHASE 3 — EKS Cluster

## Step 11 — Create EKS Cluster

⚠️ **Costs Rs 600-800/day. Delete after practice.**

```bash
eksctl create cluster \
    --name customer-segmentation-cluster-1 \
    --region ap-southeast-2 \
    --nodegroup-name workers \
    --node-type t3.medium \
    --nodes 2 \
    --nodes-min 1 \
    --nodes-max 3 \
    --managed

# Wait 15-20 minutes...
kubectl get nodes
# Should show 2 nodes Ready ✅
```

## Step 12 — Deploy to EKS

```bash
# Step 1 — Open k8s/deployment.yml
# Find ACCOUNT_ID → replace with ACCOUNT_ID → save

# Step 2 — Push updated file
git add k8s/deployment.yml
git commit -m "Update ECR account ID"
git push

# Step 3 — Connect kubectl to EKS
aws eks update-kubeconfig \
    --region ap-southeast-2 \
    --name customer-segmentation-cluster-1

# Step 4 — Create AWS secrets in Kubernetes
kubectl create secret generic aws-secrets \
    --from-literal=aws-access-key-id=<your-access-key> \
    --from-literal=aws-secret-access-key=<your-secret-key>

# Step 5 — Deploy
kubectl apply -f k8s/deployment.yml

# Step 6 — Check status
kubectl get pods
kubectl get service customer-segmentation-service

# Copy EXTERNAL-IP from the output ↑
```

## Step 13 — Open Security Group Ports

```bash
# Get security group ID
aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=*customer-segmentation*" \
    --region ap-southeast-2 \
    --query "SecurityGroups[*].[GroupId,GroupName]"

# Open port 8501 (Streamlit) — replace sg-XXXXXXXX
aws ec2 authorize-security-group-ingress \
    --group-id sg-XXXXXXXX \
    --protocol tcp --port 8501 --cidr 0.0.0.0/0 --region ap-south-1

# Open port 8000 (FastAPI)
aws ec2 authorize-security-group-ingress \
    --group-id sg-XXXXXXXX \
    --protocol tcp --port 8000 --cidr 0.0.0.0/0 --region ap-south-1
```

## Step 14 — Verify App is Live

```bash
kubectl get service customer-segmentation-service
# Copy EXTERNAL-IP

# Open Streamlit dashboard:
http://EXTERNAL-IP:8501

# Test API health:
curl http://EXTERNAL-IP:8000/health
# Expected: {"status":"healthy","segments":7,...}
```

---

# PHASE 4 — GitHub CI/CD

## Step 15 — Push Code to GitHub

```bash
git init
git add .
git commit -m "Initial commit — Customer Segmentation deployment"
git branch -M main
git remote add origin https://github.com/drsuresh8453/customer-segmentation.git
git push -u origin main
```

## Step 16 — Add GitHub Secrets

```
Go to: https://github.com/drsuresh8453/customer-segmentation/settings/secrets/actions
Click "New repository secret" — add all 6:

Name                      Value
──────────────────────────────────────────────────────────────────
AWS_ACCESS_KEY_ID         <your-access-key>
AWS_SECRET_ACCESS_KEY     <your-secret-key>
AWS_REGION                ap-southeast-2
S3_BUCKET                 customer-segmentation-2026-1
ECR_REGISTRY              ACCOUNT_ID.dkr.ecr.ap-southeast-2.amazonaws.com
EKS_CLUSTER_NAME          customer-segmentation-cluster
```

## Step 17 — Verify CI/CD

```bash
# Make a small change to trigger CI/CD
echo "# test" >> src/segment.py
git add . && git commit -m "Test CI/CD" && git push

# Go to: https://github.com/drsuresh8453/customer-segmentation/actions
# Should see pipeline running:
# ✅ Build Docker image
# ✅ Push to ECR
# ✅ Deploy to EKS (rolling update — zero downtime)
```

---

# How to Use the Live Dashboard

## Tab 1 — Single Customer Lookup

```
1. Fill in customer details:
   Name, Customer ID, Email, Phone
   Recency days, Frequency, Annual Spend
   City tier, Income segment, Favourite category
   Engagement signals (app user, loyalty enrolled etc.)

2. Click "Predict Segment & Generate Message"

3. You will see:
   ┌─────────────────────────────────────────────┐
   │ ⚠️ At Risk                                  │
   │ Confidence: 72.4%  |  Urgency: HIGH          │
   │ Channel: WhatsApp                            │
   │ Offer: 25% discount, personalised reminder   │
   └─────────────────────────────────────────────┘

   Bar chart → distance to all 7 centroids
   (shortest bar = most similar segment)

   Chat bubble:
   ┌─────────────────────────────────────────────┐
   │ 📱 WhatsApp message for Priya Sharma        │
   │                                             │
   │ "Hi Priya, we miss you! It has been 45     │
   │  days since your last order of organic     │
   │  vegetables. Here is Rs 150 off — valid    │
   │  for 48 hours only. Your favourites        │
   │  are waiting! 🛒"                          │
   └─────────────────────────────────────────────┘

4. Download message as .txt or full result as .csv
```

## Tab 2 — Batch CSV Upload

```
1. Upload sample_10k_customers.csv (or your own CSV)

2. Set options:
   Max LLM messages: 100 (for demo — saves API cost)
   Generate LLM Messages: ✅

3. Click "Segment All Customers + Generate Messages"

4. Progress bar runs...

5. You see:
   Chart 1: Customers per segment (bar chart)
   Chart 2: Revenue share per segment (pie chart)
   Chart 3: Confidence distribution (histogram)

   Results table (first 20 rows):
   customer_id | segment_name | confidence | urgency | llm_message

6. Download buttons:
   📥 Full Results CSV         → all 10,000 customers
   🚨 At-Risk + Cannot Lose    → urgent customers only
   ⭐ Champions Only            → VIP customers only

7. Feed to WhatsApp Business API:
   Use "llm_message" column as message body
   Use "recommended_channel" to decide SMS/Email/WhatsApp
```

## Tab 3 — Segment Overview

```
Shows:
  Cluster profiles from trained K-Means++ model
  Strategy cards for each of the 7 segments:
  - Champions      → Exclusivity + early access
  - Loyal          → Loyalty rewards + bundle deals
  - Potential      → Onboarding + 15% off new category
  - At Risk        → Win-back + 25% off + urgency
  - Cannot Lose    → Personal call + premium offer
  - Hibernating    → 20% off + free delivery
  - Lost           → Rs 300 off — last attempt
```

---

# API Quick Reference

```bash
# Health check
curl http://EXTERNAL-IP:8000/health

# Single customer → JSON response
curl -X POST http://EXTERNAL-IP:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST001",
    "name": "Priya Sharma",
    "recency_days": 45,
    "frequency": 24,
    "monetary_value": 22000,
    "income_segment": "High",
    "city_tier": "Metro",
    "favourite_category": "Fruits & Vegetables"
  }'

# Response:
# {
#   "segment": "At Risk",
#   "confidence": 0.7245,
#   "urgency": "HIGH",
#   "channel": "WhatsApp",
#   "llm_message": "Hi Priya, we miss you!..."
# }

# S3 batch job — reads CSV from S3 → segments → saves results to S3
curl -X POST http://EXTERNAL-IP:8000/predict/s3 \
  -H "Content-Type: application/json" \
  -d '{
    "input_s3_key": "uploads/sample_10k_customers.csv",
    "output_s3_key": "results/segmented_10k.csv",
    "generate_messages": true
  }'

# Response: { "job_id": "abc12345", "status": "queued" }

# Check job status
curl http://EXTERNAL-IP:8000/jobs/abc12345

# List all 7 segments and their strategies
curl http://EXTERNAL-IP:8000/segments
```

---

# All Commands Quick Reference

```bash
# ── Local ──────────────────────────────────────────────────────
python -m venv venv && source venv/Scripts/activate
pip install -r requirements.txt
python src/segment.py                    # test segmentation
streamlit run src/app.py                 # dashboard at localhost:8501
uvicorn src.api:app --reload --port 8000 # API at localhost:8000

# ── S3 ─────────────────────────────────────────────────────────
aws s3 ls s3://customer-segmentation-2026-1/models/
aws s3 cp sample_10k_customers.csv \
    s3://customer-segmentation-2026/uploads/sample_10k_customers.csv

# ── ECR ────────────────────────────────────────────────────────
aws ecr create-repository \
    --repository-name customer-segmentation --region ap-southeast-2
aws ecr get-login-password --region ap-southeast-2 | \
    docker login --username AWS --password-stdin \
    ACCOUNT_ID.dkr.ecr.ap-southeast-2.amazonaws.com
docker build -t customer-segmentation:v1.0 .
docker tag customer-segmentation:v1.0 \
    ACCOUNT_ID.dkr.ecr.ap-southeast-2.amazonaws.com/customer-segmentation:latest
docker push \
    ACCOUNT_ID.dkr.ecr.ap-southeast-2.amazonaws.com/customer-segmentation:latest

# ── EKS ────────────────────────────────────────────────────────
eksctl create cluster --name customer-segmentation-cluster-1 \
    --region ap-southeast-2 --nodegroup-name workers \
    --node-type t3.medium --nodes 2 --managed
aws eks update-kubeconfig \
    --region ap-southeast-2 --name customer-segmentation-cluster-1
kubectl apply -f k8s/deployment.yml
kubectl get pods
kubectl get service customer-segmentation-service
kubectl rollout status deployment/customer-segmentation-app

# ── Git ─────────────────────────────────────────────────────────
git add . && git commit -m "message" && git push

# ── Cleanup ─────────────────────────────────────────────────────
eksctl delete cluster \
    --name customer-segmentation-cluster-1 --region ap-southeast-2
aws ecr delete-repository \
    --repository-name customer-segmentation --region ap-southeast-2 --force
```

---

# Complete Automated Flow After Setup

```
ONE TIME SETUP:
  ✅ NB1-NB7 run in Google Colab → model saved to S3
  ✅ ECR repository created
  ✅ Docker image built and pushed
  ✅ EKS cluster running
  ✅ App deployed — live at EXTERNAL-IP:8501
  ✅ GitHub CI/CD set up

DAILY USAGE — MARKETING TEAM:
  Open http://EXTERNAL-IP:8501
  → Upload customer CSV
  → Download segmented CSV with messages
  → Send via WhatsApp Business API

DAILY USAGE — CRM SYSTEM (automated):
  Upload CSV to S3
  → POST /predict/s3
  → Poll /jobs/{id}
  → Download results from S3
  → Send campaigns

NO DRIFT DETECTION:
  Segments are re-run quarterly by re-running NB7 in Colab
  New centroids saved to S3 → app picks them up automatically
  No Lambda, no EventBridge, no SageMaker pipeline needed
```

---

**Author: Ashutosh Behera | Data Scientist**
**Bandhan Bank Ltd — ML Project**
