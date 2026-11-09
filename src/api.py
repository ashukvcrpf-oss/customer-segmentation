"""
api.py — Customer Segmentation FastAPI
Author: Ashutosh Behera | Data Scientist | Bandhan Bank Ltd

Endpoints:
  GET  /health                    → model status
  GET  /segments                  → list all 7 segments with strategies
  POST /predict                   → single customer → segment + LLM message
  POST /predict/batch             → upload CSV → all customers segmented + messages
  POST /predict/s3                → read CSV from S3 → segment → save results to S3
  GET  /results/{job_id}          → download results of an S3 batch job
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import numpy as np
import boto3
import io
import os
import uuid
import json
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from .segment import (
    assign_segment,
    assign_batch,
    load_model_artifacts,
    SEGMENT_STRATEGIES
)
from .llm_messages import (
    generate_message,
    generate_batch_messages
)

app = FastAPI(
    title="Customer Segmentation API",
    description="Assigns customers to segments using K-Means centroid mapping. Generates personalised LLM messages.",
    version="1.0.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# AWS Configuration — secure credential management

BUCKET = os.getenv(
    "S3_BUCKET",
    "customer-segmentation-2026-1"
)

REGION = os.getenv(
    "AWS_REGION",
    "ap-southeast-2"
)

# Boto3 automatically retrieves credentials from AWS CLI,
# environment variables, or an attached IAM role.
s3 = boto3.client(
    "s3",
    region_name=REGION
)


# ── Pydantic Models ────────────────────────────────────────────────────────
class CustomerRecord(BaseModel):
    customer_id:          Optional[str]   = "UNKNOWN"
    name:                 Optional[str]   = "Valued Customer"
    recency_days:         Optional[float] = 30
    frequency:            Optional[float] = 12
    monetary_value:       Optional[float] = 15000
    avg_order_value:      Optional[float] = 1250
    rfm_score:            Optional[float] = 9
    recency_score:        Optional[float] = 3
    frequency_score:      Optional[float] = 3
    monetary_score:       Optional[float] = 3
    tenure_months:        Optional[float] = 12
    discount_usage_rate:  Optional[float] = 0.3
    return_rate:          Optional[float] = 0.02
    basket_size_avg:      Optional[float] = 8
    num_categories:       Optional[float] = 4
    category_concentration: Optional[float] = 0.4
    weekend_purchase_ratio: Optional[float] = 0.35
    night_purchase_ratio:   Optional[float] = 0.15
    premium_brand_ratio:    Optional[float] = 0.2
    organic_product_ratio:  Optional[float] = 0.1
    subscription_items:     Optional[float] = 2
    new_product_trial_rate: Optional[float] = 0.1
    purchase_gap_avg_days:  Optional[float] = 30
    purchase_gap_trend:     Optional[float] = 0.0
    last_3m_vs_prev_3m:     Optional[float] = 0.05
    days_active_per_month:  Optional[float] = 8
    support_tickets:        Optional[float] = 1
    income_segment:         Optional[str]   = "Mid"
    city_tier:              Optional[str]   = "Metro"
    gender:                 Optional[str]   = "Male"
    has_children:           Optional[float] = 0
    loyalty_enrolled:       Optional[float] = 0
    app_user:               Optional[float] = 1
    response_to_campaign:   Optional[float] = 0
    favourite_category:     Optional[str]   = "Staples & Grains"
    city:                   Optional[str]   = "Mumbai"

class S3BatchRequest(BaseModel):
    input_s3_key:    str = Field(..., description="S3 key of input CSV e.g. uploads/customers.csv")
    output_s3_key:   str = Field(..., description="S3 key for results CSV e.g. results/segmented.csv")
    generate_messages: bool = Field(True, description="Generate LLM messages for each customer")
    openai_api_key:  Optional[str] = Field(None, description="OpenAI key — uses env var if not provided")

# ── Job tracking ───────────────────────────────────────────────────────────
jobs = {}

# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Health check — confirms model is loaded and ready."""
    try:
        artifacts = load_model_artifacts()
        return {
            "status":    "healthy",
            "segments":  len(artifacts["segment_names"]),
            "features":  len(artifacts["features"]),
            "model":     "KMeans++ centroids",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not ready: {e}")

@app.get("/segments")
def list_segments():
    """List all 7 segments with their strategies — for CRM integration."""
    return {
        "total_segments": len(SEGMENT_STRATEGIES),
        "segments": SEGMENT_STRATEGIES
    }

@app.post("/predict")
def predict_single(customer: CustomerRecord, generate_llm: bool = True):
    """
    Assign a single customer to a segment.
    Optionally generate a personalised LLM message.

    Used by CRM for real-time scoring when customer opens the app.
    Response time: < 500ms
    """
    try:
        record = customer.dict()
        result = assign_segment(record)

        message = None
        if generate_llm:
            message = generate_message(result["segment_name"], record)

        return {
            "customer_id":     record.get("customer_id"),
            "name":            record.get("name"),
            "segment":         result["segment_name"],
            "confidence":      result["confidence"],
            "urgency":         result["strategy"]["urgency"],
            "channel":         result["strategy"]["channel"],
            "offer":           result["strategy"]["offer"],
            "tone":            result["strategy"]["tone"],
            "llm_message":     message,
            "timestamp":       datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch_upload(
    file: UploadFile = File(..., description="CSV file of customers"),
    generate_messages: bool = True
):
    """
    Upload a CSV of customers. Get back a CSV with segments and LLM messages.

    CSV must have columns: customer_id, recency_days, frequency, monetary_value etc.
    Returns: original columns + segment_name + confidence + urgency + llm_message
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        print(f"Batch received: {len(df):,} customers")

        # Assign segments
        df_result = assign_batch(df)

        # Generate messages
        if generate_messages:
            print("Generating LLM messages...")
            df_result["llm_message"] = generate_batch_messages(df_result)

        df_result["processed_at"] = datetime.now().isoformat()

        # Return as CSV download
        output = io.StringIO()
        df_result.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=segmented_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/s3")
def predict_from_s3(request: S3BatchRequest, background_tasks: BackgroundTasks):
    """
    Read a CSV from S3, segment all customers, generate messages, save results back to S3.

    This is the main PRODUCTION endpoint:
    1. CRM uploads customer CSV to S3
    2. Calls this endpoint with input and output S3 keys
    3. Segmentation runs in background
    4. Results saved to S3 — CRM downloads and sends campaigns

    Returns job_id for tracking.
    """
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "queued", "started_at": datetime.now().isoformat()}

    def run_s3_job():
        try:
            jobs[job_id]["status"] = "running"
            print(f"Job {job_id}: Reading {request.input_s3_key} from S3...")

            # Read input CSV from S3
            obj = s3.get_object(Bucket=BUCKET, Key=request.input_s3_key)
            df  = pd.read_csv(io.BytesIO(obj["Body"].read()))
            print(f"Job {job_id}: {len(df):,} customers loaded")

            # Assign segments
            df_result = assign_batch(df)

            # Generate LLM messages
            if request.generate_messages:
                if request.openai_api_key:
                    os.environ["OPENAI_API_KEY"] = request.openai_api_key
                print(f"Job {job_id}: Generating {len(df):,} LLM messages...")
                df_result["llm_message"] = generate_batch_messages(df_result)

            df_result["processed_at"] = datetime.now().isoformat()

            # Save results to S3
            buf = io.StringIO()
            df_result.to_csv(buf, index=False)
            s3.put_object(Bucket=BUCKET, Key=request.output_s3_key, Body=buf.getvalue())

            # Summary stats
            seg_counts = df_result["segment_name"].value_counts().to_dict()
            jobs[job_id].update({
                "status":         "completed",
                "total_customers": len(df_result),
                "output_s3_key":  request.output_s3_key,
                "segment_counts": seg_counts,
                "completed_at":   datetime.now().isoformat(),
            })
            print(f"Job {job_id}: Complete! Results at s3://{BUCKET}/{request.output_s3_key}")

        except Exception as e:
            jobs[job_id].update({"status": "failed", "error": str(e)})
            print(f"Job {job_id} failed: {e}")

    background_tasks.add_task(run_s3_job)

    return {
        "job_id":       job_id,
        "status":       "queued",
        "input":        request.input_s3_key,
        "output":       request.output_s3_key,
        "message":      "Job started. Poll /jobs/{job_id} for status.",
    }

@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    """Check status of an S3 batch segmentation job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@app.get("/jobs")
def list_jobs():
    """List all batch jobs and their status."""
    return {"total": len(jobs), "jobs": jobs}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
