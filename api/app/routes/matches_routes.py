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
from utils.extract import extract_skills, extract_adjectives, extract_adverbs, clean_text_for_matching
from utils.connection_db import get_db, CVModel, JobModel, MatchesModel
from sqlalchemy.orm import Session
import spacy

router = APIRouter(prefix="/match", tags=["Job-CV Matching"])

# Load spaCy model
nlp_en = spacy.load('en_core_web_md')

# Load skills
skills = read_skills('app/skills.txt')

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

def extract_all_features(text: str) -> Dict[str, list]:
    doc = nlp_en(text)
    tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
    
    return {
        'skills_required': extract_skills(text, skills),
        'secondary_skills': extract_skills(text, skills),  # Có thể thay bằng logic khác
        'adverbs': [token for token in tokens if nlp_en.vocab[token].tag_ == 'RB'],
        'adjectives': [token for token in tokens if nlp_en.vocab[token].tag_ == 'JJ']
    }

def load_model():
    global model, scaler
    try:
        models_dir = Path("app/output/models")
        print(f"Loading model and scaler from: {models_dir.absolute()}")

        # Tìm model và scaler mới nhất
        model_files = list(models_dir.glob("best_model_xgboost.pkl"))
        scaler_file = models_dir / "scaler.pkl"
        
        if not model_files:
            print("❌ Model file not found: best_model_xgboost.pkl")
            return False
            
        if not scaler_file.exists():
            print(f"❌ Scaler file not found: {scaler_file}")
            return False

        # Lấy file mới nhất
        model_file = sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]

        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)

        print(f"✅ Model loaded successfully from {model_file}")
        print(f"✅ Scaler loaded from {scaler_file}")
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

    # Tính similarity cho trọng số tính từ (adj_weight)
    adj_weight_sim = 1 - abs(len(features1['adjectives']) - len(features2['adjectives'])) / max(
        len(features1['adjectives']) + 1, len(features2['adjectives']) + 1)

    return {
        'skills_required_sim': jaccard_similarity(set(features1['skills_required']), set(features2['primary_skills'])),
        'secondary_skills_sim': jaccard_similarity(set(features1['secondary_skills']), set(features2['secondary_skills'])),
        'adjectives_sim': jaccard_similarity(set(features1['adjectives']), set(features2['adjectives'])),
        'adj_weight_sim': adj_weight_sim
    }

def get_matched_primary_skills(cv_skills: List[str], job_skills: List[str]) -> List[str]:
    """Get the intersection of skills_required in CV and primary_skills in Job"""
    cv_skills_set = set(skill.lower().strip() for skill in cv_skills)
    job_skills_set = set(skill.lower().strip() for skill in job_skills)
    matched_skills = cv_skills_set.intersection(job_skills_set)
    return list(matched_skills)

def predict_suitability(similarities: Dict[str, float]) -> Dict:
    # Check if model and scaler are loaded
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded. Please check model files.")

    # Tạo feature vector từ các similarity scores
    features = {
        'primary_sim': similarities.get('skills_required_sim', 0.0),
        'secondary_sim': similarities.get('secondary_skills_sim', 0.0),
        'adj_sim': similarities.get('adjectives_sim', 0.0),
        'adj_weight': 1.0  # Giá trị mặc định
    }
    
    # Tính toán các derived features (giống như trong training)
    features['primary_secondary_ratio'] = features['primary_sim'] / (features['secondary_sim'] + 1e-8)
    features['primary_adj_ratio'] = features['primary_sim'] / (features['adj_sim'] + 1e-8)
    features['primary_secondary_diff'] = features['primary_sim'] - features['secondary_sim']
    features['weighted_primary'] = features['primary_sim'] * features['adj_weight']
    features['composite_score'] = (features['primary_sim'] * 0.5 + 
                                  features['secondary_sim'] * 0.3 + 
                                  features['adj_sim'] * 0.2)
    
    # Tạo input vector theo đúng thứ tự feature
    input_features = np.array([
        features['primary_sim'],
        features['secondary_sim'],
        features['adj_sim'],
        features['adj_weight'],
        features['primary_secondary_ratio'],
        features['primary_adj_ratio'],
        features['primary_secondary_diff'],
        features['weighted_primary'],
        features['composite_score']
    ]).reshape(1, -1)
    
    # Chuẩn hóa dữ liệu
    scaled_input = scaler.transform(input_features)
    
    # Dự đoán
    prediction = model.predict(scaled_input)[0]
    
    # Ánh xạ kết quả
    suitability_mapping = {0: 'Not Suitable', 1: 'Suitable'}
    return {
        'suitability_label': suitability_mapping.get(prediction, 'Unknown')
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

    # Sử dụng skills_required từ CV
    cv_features = {
        'skills_required': cv.skills or [],  # Sử dụng trường skills_required
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

            # Get matched primary skills (skills_required in CV vs primary_skills in Job)
            matched_primary_skills = get_matched_primary_skills(
                cv_features['skills_required'],
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