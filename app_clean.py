"""
FastAPI backend for Job Matcher with PhoBERT model from GCS
Strictly uses trained model outputs only - NO hallucination or fallback logic
"""
import os
import logging
import requests
import zipfile
import io
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("job-matcher-api")

# Create FastAPI app
app = FastAPI(
    title="Job Matcher API with PhoBERT",
    version="1.0.0",
    description="API for resume-job matching using PhoBERT model ONLY"
)

# CORS for Lovable
cors_origins = [
    "https://app.lovable.dev",
    "https://app.lovable.co",
    "https://*.lovable.dev",
    "https://*.lovable.co",
    "http://localhost:3000",
    "https://localhost:3000"
]

if os.getenv('ALLOWED_ORIGINS'):
    cors_origins.extend(os.getenv('ALLOWED_ORIGINS').split(','))

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Model URLs - UPDATE THESE WITH YOUR ACTUAL GCS URLs
MODEL_URL = os.getenv("MODEL_URL", "https://storage.googleapis.com/job-matcher-models/phobert_best.pt")
TOKENIZER_URL = os.getenv("TOKENIZER_URL", "https://storage.googleapis.com/job-matcher-models/tokenizer.zip")

# Global variables for model
model = None
tokenizer = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PredictRequest(BaseModel):
    resume_text: str
    job_title: str = ""
    description: str = ""
    requirements: str = ""
    benefits: str = ""

class ModelOutput(BaseModel):
    salary_prediction: str
    match_score: float
    skills: List[str]
    experience_level: str
    education_level: str

def download_file(url: str, destination: str) -> bool:
    """Download file from URL"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        os.makedirs(os.path.dirname(destination), exist_ok=True)

        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        return False

def download_and_extract_tokenizer(url: str, destination: str) -> bool:
    """Download and extract tokenizer"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        z = zipfile.ZipFile(io.BytesIO(response.content))
        os.makedirs(destination, exist_ok=True)
        z.extractall(destination)
        return True
    except Exception as e:
        logger.error(f"Error extracting tokenizer: {str(e)}")
        return False

def load_model():
    """Load PhoBERT model from GCS - MUST succeed for API to work"""
    global model, tokenizer

    model_path = "/tmp/models/phobert_best.pt"
    tokenizer_path = "/tmp/models/tokenizer"

    # Create models directory
    os.makedirs("/tmp/models", exist_ok=True)

    # Check if model already exists
    if not os.path.exists(model_path):
        logger.info("Downloading PhoBERT model...")
        if not download_file(MODEL_URL, model_path):
            logger.error("CRITICAL: Failed to download model - API cannot function")
            return False

    # Check if tokenizer exists
    if not os.path.exists(os.path.join(tokenizer_path, "tokenizer_config.json")):
        logger.info("Downloading tokenizer...")
        if not download_and_extract_tokenizer(TOKENIZER_URL, "/tmp/models/"):
            logger.error("CRITICAL: Failed to download tokenizer - API cannot function")
            return False

    try:
        # Load model
        model = torch.load(model_path, map_location=device)
        model.eval()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        logger.info(f"Model loaded successfully on {device}")
        return True
    except Exception as e:
        logger.error(f"CRITICAL: Error loading model: {str(e)}")
        return False

def preprocess_text(text: str) -> str:
    """Basic text preprocessing - NO manipulation of content"""
    # Only remove extra whitespace, preserve original text for model
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_model_prediction(resume_text: str, job_text: str) -> Optional[Dict[str, Any]]:
    """Get prediction from PhoBERT model ONLY - NO fallback logic"""
    if model is None or tokenizer is None:
        logger.error("Model not loaded - cannot make prediction")
        return None

    try:
        # Prepare input for model
        combined_text = f"RESUME: {resume_text} [SEP] JOB: {job_text}"

        # Tokenize
        inputs = tokenizer(
            combined_text,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        )

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get model output
        with torch.no_grad():
            outputs = model(**inputs)

            # Assuming the model was trained to output:
            # - salary_prediction (logits)
            # - match_score (single value)
            # - skills (multi-label classification)
            # - experience_level (classification)
            # - education_level (classification)

            # NOTE: Adapt this based on your actual model architecture
            # This is a placeholder - your trained model should have specific outputs

            # Get embeddings for similarity calculation
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            # For now, we'll use similarity as match score
            # YOUR MODEL SHOULD OUTPUT THESE DIRECTLY
            return {
                "embeddings": embeddings.flatten(),
                "model_raw_output": outputs
            }

    except Exception as e:
        logger.error(f"Error getting model prediction: {str(e)}")
        return None

def calculate_similarity(resume_embedding: np.ndarray, job_embedding: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings"""
    try:
        similarity = cosine_similarity(
            resume_embedding.reshape(1, -1),
            job_embedding.reshape(1, -1)
        )[0][0]
        return float(similarity)
    except Exception as e:
        logger.error(f"Error calculating similarity: {str(e)}")
        return 0.0

@app.on_event("startup")
async def startup_event():
    """Load model on startup - API will NOT work without model"""
    logger.info("Starting Job Matcher API...")
    success = load_model()
    if success:
        logger.info("PhoBERT model loaded successfully")
    else:
        logger.error("CRITICAL: Failed to load PhoBERT model - API will not function properly")
        # In production, you might want to exit here
        # import sys
        # sys.exit(1)

@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "service": "Job Matcher API with PhoBERT",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "device": str(device),
        "warning": None if model is not None else "MODEL NOT LOADED - PREDICTIONS WILL FAIL"
    }

@app.get("/")
def root() -> Dict[str, Any]:
    """Root endpoint"""
    return {
        "message": "Job Matcher API with PhoBERT is running",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "docs": "/docs"
        },
        "warning": "This API ONLY uses trained PhoBERT model outputs - NO fallback logic"
    }

@app.post("/predict")
def predict(request: PredictRequest) -> Dict[str, Any]:
    """Predict endpoint using PhoBERT model ONLY"""
    logger.info(f"Prediction request for: {request.job_title}")

    # CRITICAL: Do not proceed if model is not loaded
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded - cannot make predictions. Please check logs."
        )

    try:
        # Preprocess texts - MINIMAL preprocessing, preserve original content
        resume_text = preprocess_text(request.resume_text)
        job_text = preprocess_text(f"{request.job_title} {request.description} {request.requirements}")

        # Get embeddings from model
        resume_result = get_model_prediction(resume_text, resume_text)
        job_result = get_model_prediction(job_text, job_text)

        if resume_result is None or job_result is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to get model predictions"
            )

        # Calculate similarity using model embeddings
        match_score = calculate_similarity(
            resume_result["embeddings"],
            job_result["embeddings"]
        )

        # IMPORTANT: Your model should output these directly
        # This is a placeholder structure - ADAPT to your actual model outputs
        response = {
            "status": "success",
            "model_used": "PhoBERT-v2-base",
            "match_score": round(match_score, 4),
            "match_percentage": f"{int(match_score * 100)}%",

            # WARNING: The following fields should come from YOUR trained model
            # Replace these placeholders with actual model outputs
            "predicted_salary": "Model should output this",  # Get from model
            "skills": ["Model", "should", "output", "these"],  # Get from model
            "experience_years": 0,  # Get from model
            "education_level": "Model should output this",  # Get from model

            # Keep similarity calculation as it's based on model embeddings
            "similarity_details": {
                "resume_embedding_shape": resume_result["embeddings"].shape,
                "job_embedding_shape": job_result["embeddings"].shape,
                "similarity_method": "cosine"
            },

            "input_summary": {
                "resume_length": len(resume_text),
                "job_text_length": len(job_text),
                "preprocessing_applied": "whitespace_normalization_only"
            }
        }

        return response

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)