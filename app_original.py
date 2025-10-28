"""
FastAPI backend for Job Matcher with PhoBERT model from GCS
Optimized for Lovable integration
"""
import os
import logging
import requests
import zipfile
import io
from typing import Dict, Any, List
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
    description="API for resume-job matching using PhoBERT model"
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

# Model URLs
MODEL_URL = "https://storage.googleapis.com/job-matcher-models/phobert_best.pt"
TOKENIZER_URL = "https://storage.googleapis.com/job-matcher-models/tokenizer.zip"

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

def download_file(url: str, destination: str) -> bool:
    """Download file from URL"""
    try:
        response = requests.get(url, stream=True)
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
        response = requests.get(url)
        response.raise_for_status()

        z = zipfile.ZipFile(io.BytesIO(response.content))
        os.makedirs(destination, exist_ok=True)
        z.extractall(destination)
        return True
    except Exception as e:
        logger.error(f"Error extracting tokenizer: {str(e)}")
        return False

def load_model():
    """Load PhoBERT model from GCS"""
    global model, tokenizer

    model_path = "/tmp/models/phobert_best.pt"
    tokenizer_path = "/tmp/models/tokenizer"

    # Create models directory
    os.makedirs("/tmp/models", exist_ok=True)

    # Check if model already exists
    if not os.path.exists(model_path):
        logger.info("Downloading PhoBERT model...")
        if not download_file(MODEL_URL, model_path):
            logger.error("Failed to download model")
            return False

    # Check if tokenizer exists
    if not os.path.exists(os.path.join(tokenizer_path, "tokenizer_config.json")):
        logger.info("Downloading tokenizer...")
        if not download_and_extract_tokenizer(TOKENIZER_URL, "/tmp/models/"):
            logger.error("Failed to download tokenizer")
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
        logger.error(f"Error loading model: {str(e)}")
        return False

def preprocess_text(text: str) -> str:
    """Preprocess Vietnamese text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Convert to lowercase
    text = text.lower()
    return text

def get_embedding(text: str) -> np.ndarray:
    """Get embedding from PhoBERT"""
    if model is None or tokenizer is None:
        return np.zeros(768)  # Return zero embedding if model not loaded

    try:
        # Tokenize
        inputs = tokenizer(text,
                         return_tensors='pt',
                         max_length=512,
                         truncation=True,
                         padding=True)

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get embedding
        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embedding.flatten()
    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        return np.zeros(768)

def extract_skills(text: str) -> List[str]:
    """Extract skills from text"""
    tech_skills = [
        'python', 'java', 'javascript', 'react', 'nodejs', 'node.js', 'docker', 'kubernetes',
        'aws', 'azure', 'gcp', 'mongodb', 'postgresql', 'mysql', 'tensorflow',
        'pytorch', 'machine learning', 'ml', 'ai', 'devops', 'git', 'ci/cd', 'typescript',
        'vue', 'angular', 'flask', 'django', 'fastapi', 'rest api', 'graphql',
        'sql', 'nosql', 'redis', 'elasticsearch', 'kafka', 'microservices'
    ]

    text_lower = text.lower()
    found_skills = []

    for skill in tech_skills:
        if skill in text_lower:
            found_skills.append(skill)

    return found_skills[:20]  # Return top 20 skills

def predict_salary(job_title: str, experience_years: int, skills: List[str]) -> str:
    """Predict salary based on Vietnamese market"""
    job_title_lower = job_title.lower()

    # Base salary by level
    if 'intern' in job_title_lower or 'fresher' in job_title_lower or experience_years < 1:
        return "5-8 million VND"
    elif 'junior' in job_title_lower or 'entry' in job_title_lower or experience_years < 2:
        return "10-15 million VND"
    elif 'mid' in job_title_lower or 2 <= experience_years <= 3:
        return "15-25 million VND"
    elif 'senior' in job_title_lower or experience_years >= 5:
        base = "25-35 million VND"
    elif 'lead' in job_title_lower or 'principal' in job_title_lower:
        base = "30-45 million VND"
    elif 'manager' in job_title_lower or 'head' in job_title_lower:
        base = "40-60 million VND"
    else:
        base = "15-25 million VND"

    # Adjust for in-demand skills
    if any(skill in skills for skill in ['python', 'machine learning', 'ai', 'tensorflow', 'pytorch']):
        if 'senior' in job_title_lower:
            return "30-45 million VND"
        else:
            return "20-30 million VND"

    return base

def extract_experience(text: str) -> int:
    """Extract years of experience"""
    patterns = [
        r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
        r'experience\s*:?\s*(\d+)',
        r'(\d+)\s*-\s*\d+\s*(?:years?|yrs?)'
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            return int(matches[0])

    # Estimate from job titles
    if 'senior' in text.lower() or 'sr' in text.lower():
        return 5
    elif 'mid' in text.lower() or 2 <= any(num in text.lower() for num in ['2-3', '3-5']):
        return 3
    elif 'junior' in text.lower() or 'jr' in text.lower():
        return 1

    return 0

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting Job Matcher API...")
    success = load_model()
    if success:
        logger.info("PhoBERT model loaded successfully")
    else:
        logger.warning("Failed to load PhoBERT model, using mock responses")

@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Job Matcher API with PhoBERT",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "device": str(device)
    }

@app.get("/")
def root() -> Dict[str, Any]:
    """Root endpoint"""
    return {
        "message": "Job Matcher API with PhoBERT is running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "docs": "/docs"
        }
    }

@app.post("/predict")
def predict(request: PredictRequest) -> Dict[str, Any]:
    """Predict endpoint using PhoBERT model"""
    logger.info(f"Prediction request for: {request.job_title}")

    try:
        # Preprocess texts
        resume_text = preprocess_text(request.resume_text)
        job_text = preprocess_text(f"{request.job_title} {request.description} {request.requirements}")

        # Get embeddings
        resume_embedding = get_embedding(resume_text)
        job_embedding = get_embedding(job_text)

        # Calculate similarity score
        if model is not None:
            similarity = cosine_similarity(
                resume_embedding.reshape(1, -1),
                job_embedding.reshape(1, -1)
            )[0][0]
            match_score = float(similarity)
        else:
            # Fallback to keyword matching
            resume_skills = extract_skills(resume_text)
            job_skills = extract_skills(job_text)
            if job_skills:
                matches = len([s for s in resume_skills if s in job_skills])
                match_score = min(0.95, 0.3 + (matches / len(job_skills)))
            else:
                match_score = 0.6

        # Extract information
        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_text)
        missing_skills = list(set(job_skills) - set(resume_skills))[:5]
        experience_years = extract_experience(resume_text)

        # Predict salary
        salary = predict_salary(request.job_title, experience_years, resume_skills)

        return {
            "status": "success",
            "predicted_salary": salary,
            "match_score": round(match_score, 3),
            "match_percentage": f"{int(match_score * 100)}%",
            "missing_skills": missing_skills,
            "found_skills": resume_skills[:15],
            "parsed_resume": {
                "experience_years": experience_years,
                "education": "Bachelor's Degree" if any(d in resume_text.lower() for d in ['bachelor', 'bs', 'beng']) else "Not specified",
                "skills": resume_skills[:15],
                "skill_count": len(resume_skills)
            },
            "analysis": {
                "total_required_skills": len(job_skills),
                "matched_skills": len([s for s in resume_skills if s in job_skills]),
                "relevance": "High" if match_score > 0.7 else "Medium" if match_score > 0.5 else "Low",
                "experience_level": "Senior" if experience_years > 5 else "Mid-level" if experience_years > 2 else "Junior"
            },
            "model_used": "PhoBERT" if model is not None else "Keyword Matching (Fallback)"
        }

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)