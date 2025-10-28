# Job Matcher API with PhoBERT

FastAPI backend with PhoBERT model for Vietnamese resume-job matching. The model is downloaded from Google Cloud Storage on startup.

## Features

- **PhoBERT Integration**: Uses pre-trained PhoBERT model for Vietnamese text understanding
- **Dynamic Model Loading**: Downloads model from GCS on startup
- **Smart Matching**: Calculates semantic similarity between resumes and job descriptions
- **Salary Prediction**: Estimates salary based on Vietnamese market rates
- **Skill Extraction**: Identifies technical skills from text
- **CORS Enabled**: Configured for Lovable domains

## Model Source

The PhoBERT model is hosted on Google Cloud Storage:
- Model: https://storage.googleapis.com/job-matcher-models/phobert_best.pt
- Tokenizer: https://storage.googleapis.com/job-matcher-models/tokenizer.zip

## API Endpoints

### Health Check
```
GET /health
```

### Prediction
```
POST /predict
Content-Type: application/json

{
  "resume_text": "CV của bạn...",
  "job_title": "Senior Software Engineer",
  "description": "Mô tả công việc...",
  "requirements": "Yêu cầu công việc...",
  "benefits": "Phúc lợi..."
}
```

### Documentation
```
GET /docs
```

## Deployment

### DigitalOcean App Platform

1. Push to GitHub
2. Create DigitalOcean App with:
   - **Build Command**: `pip install -r requirements.txt`
   - **Run Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **HTTP Port**: `8080`
   - **Instance Size**: Basic XS or higher (for model loading)

### Environment Variables

- `ALLOWED_ORIGINS`: `https://app.lovable.dev,https://app.lovable.co`
- `PORT`: `8080`

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py

# Test API
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "Your resume...", "job_title": "Software Engineer", "description": "Job desc..."}'
```

## Model Details

The app automatically downloads:
- PhoBERT fine-tuned model (1.1GB)
- Vietnamese tokenizer
- Caches to `/tmp/models` for subsequent runs

## Response Example

```json
{
  "status": "success",
  "predicted_salary": "25-35 million VND",
  "match_score": 0.856,
  "match_percentage": "86%",
  "missing_skills": ["kubernetes"],
  "found_skills": ["python", "react", "aws", "docker"],
  "parsed_resume": {
    "experience_years": 5,
    "education": "Bachelor's Degree",
    "skills": ["python", "react", "aws", "docker"],
    "skill_count": 4
  },
  "model_used": "PhoBERT"
}
```