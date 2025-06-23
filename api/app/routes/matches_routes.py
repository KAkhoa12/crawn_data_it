from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel
from typing import Dict, List
import joblib
import json
import numpy as np
from pathlib import Path
import PyPDF2
import io
from datetime import datetime
from utils.read_file import read_skills
from utils.extract import extract_primary_skills, extract_secondary_skills, extract_adjectives, extract_adverbs
from utils.connection_db import get_db, CVModel, JobModel, MatchesModel
from sqlalchemy.orm import Session
import spacy

router = APIRouter(prefix="/match", tags=["Job-CV Matching"])

# Load spaCy model
nlp_en = spacy.load('en_core_web_md')

# Load skills
primary_skills = read_skills('app/primary_skills.txt')
secondary_skills = read_skills('app/secondary_skills.txt')

# Global model variables
model = None
preprocessing = None
metadata = None

class PredictionResponse(BaseModel):
    suitability_label: str

class MatchResult(BaseModel):
    job_id: int
    job_title: str
    suitability_label: str
    jaccard_scores: Dict[str, float]
    matched_primary_skills: List[str]

class CVMatchResponse(BaseModel):
    cv_id: int
    total_matches: int
    matches: List[MatchResult]

def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except:
        return ""

def extract_all_features(text: str) -> Dict[str, list]:
    primary_skills_list = extract_primary_skills(text, primary_skills)
    secondary_skills_list = extract_secondary_skills(text, secondary_skills)
    adverbs = extract_adverbs(text, nlp_en)
    adjectives = extract_adjectives(text, nlp_en)

    return {
        'primary_skills': primary_skills_list,
        'secondary_skills': secondary_skills_list,
        'adverbs': adverbs,
        'adjectives': adjectives
    }

def load_model():
    global model, preprocessing, metadata
    try:
        models_dir = Path("models")
        print(f"Loading model from: {models_dir.absolute()}")

        # Check if files exist
        model_file = models_dir / "best_model.joblib"
        preprocessing_file = models_dir / "preprocessing.joblib"
        metadata_file = models_dir / "model_metadata.json"

        if not model_file.exists():
            print(f"❌ Model file not found: {model_file}")
            return False
        if not preprocessing_file.exists():
            print(f"❌ Preprocessing file not found: {preprocessing_file}")
            return False
        if not metadata_file.exists():
            print(f"❌ Metadata file not found: {metadata_file}")
            return False

        model = joblib.load(model_file)
        preprocessing = joblib.load(preprocessing_file)
        with open(metadata_file) as f:
            metadata = json.load(f)

        print("✅ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def calculate_similarity(features1: Dict, features2: Dict) -> Dict[str, float]:
    def jaccard_similarity(set1: set, set2: set) -> float:
        if not set1 and not set2:
            return 1.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    return {
        'primary_skills_sim': jaccard_similarity(set(features1['primary_skills']), set(features2['primary_skills'])),
        'secondary_skills_sim': jaccard_similarity(set(features1['secondary_skills']), set(features2['secondary_skills'])),
        'adjectives_sim': jaccard_similarity(set(features1['adjectives']), set(features2['adjectives']))
    }

def get_matched_primary_skills(cv_skills: List[str], job_skills: List[str]) -> List[str]:
    """Get the intersection of primary skills between CV and Job"""
    cv_skills_set = set(skill.lower().strip() for skill in cv_skills)
    job_skills_set = set(skill.lower().strip() for skill in job_skills)
    matched_skills = cv_skills_set.intersection(job_skills_set)
    return list(matched_skills)

def predict_suitability(similarities: Dict[str, float]) -> Dict:
    # Check if model is loaded
    if model is None or preprocessing is None or metadata is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please check model files.")

    feature_vector = []
    for feature_name in metadata['feature_names']:
        feature_vector.append(similarities.get(feature_name, 0.0))

    X = np.array(feature_vector).reshape(1, -1)
    X_scaled = preprocessing['scaler'].transform(X)
    X_selected = preprocessing['feature_selector'].transform(X_scaled)

    prediction = model.predict(X_selected)[0]
    probabilities = model.predict_proba(X_selected)[0]

    confidence_scores = {}
    for i, prob in enumerate(probabilities):
        confidence_scores[metadata['suitability_mapping'][str(i)]] = float(prob)

    return {
        'suitability_label': metadata['suitability_mapping'][str(prediction)],
    }

@router.post("/cv/{cv_id}/match-all-jobs", response_model=CVMatchResponse)
async def match_cv_with_all_jobs(cv_id: int, db: Session = Depends(get_db)):
    """
    Match a CV with all jobs in the database and save results to matches table
    """
    # Get CV from database
    cv = db.query(CVModel).filter(CVModel.id == cv_id).first()
    if not cv:
        raise HTTPException(status_code=404, detail="CV not found")

    # Get CV features
    cv_features = {
        'primary_skills': cv.primary_skills or [],
        'secondary_skills': cv.secondary_skills or [],
        'adverbs': cv.adverbs or [],
        'adjectives': cv.adjectives or []
    }

    # Get all active jobs
    jobs = db.query(JobModel).filter(JobModel.status == 1).all()
    if not jobs:
        raise HTTPException(status_code=404, detail="No active jobs found")

    matches = []

    for job in jobs:
        try:
            # Get job features
            job_features = {
                'primary_skills': job.primary_skills or [],
                'secondary_skills': job.secondary_skills or [],
                'adverbs': job.adverbs or [],
                'adjectives': job.adjectives or []
            }

            # Calculate Jaccard similarities
            similarities = calculate_similarity(cv_features, job_features)

            # Get matched primary skills
            matched_primary_skills = get_matched_primary_skills(
                cv_features['primary_skills'],
                job_features['primary_skills']
            )

            # Predict suitability using the model
            try:
                prediction = predict_suitability(similarities)
                suitability_label = prediction['suitability_label']
            except Exception as e:
                print(f"⚠️ Warning: Could not predict suitability for job {job.id}: {e}")
                suitability_label = "Unknown"

            # Create match result
            match_result = MatchResult(
                job_id=job.id,
                job_title=job.title,
                suitability_label=suitability_label,
                jaccard_scores=similarities,
                matched_primary_skills=matched_primary_skills
            )
            matches.append(match_result)

            # Save to matches table if there are matched skills
            if matched_primary_skills:
                try:
                    # Check if match already exists
                    existing_match = db.query(MatchesModel).filter(
                        MatchesModel.cv_id == cv_id,
                        MatchesModel.job_id == job.id
                    ).first()

                    matched_skills_str = ", ".join(matched_primary_skills)

                    if existing_match:
                        # Update existing match
                        existing_match.matched_skill = matched_skills_str
                        existing_match.time_matches = datetime.now()
                        existing_match.status = 1
                    else:
                        # Create new match
                        new_match = MatchesModel(
                            cv_id=cv_id,
                            job_id=job.id,
                            matched_skill=matched_skills_str,
                            time_matches=datetime.now(),
                            status=1
                        )
                        db.add(new_match)

                except Exception as e:
                    print(f"⚠️ Warning: Could not save match for job {job.id}: {e}")
                    continue

        except Exception as e:
            print(f"⚠️ Warning: Error processing job {job.id}: {e}")
            continue

    # Commit all matches to database
    try:
        db.commit()
        print(f"✅ Successfully processed {len(matches)} job matches for CV {cv_id}")
    except Exception as e:
        db.rollback()
        print(f"❌ Error saving matches to database: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    return CVMatchResponse(
        cv_id=cv_id,
        total_matches=len(matches),
        matches=matches
    )

# Initialize model on startup
load_model()