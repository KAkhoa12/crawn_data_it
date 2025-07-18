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
import os
from utils.read_file import read_skills
from utils.extract import extract_skills, extract_adjectives, extract_adverbs, clean_text_for_matching
from utils.connection_db import get_db, CVModel, JobModel, MatchesModel
from sqlalchemy.orm import Session
import spacy
def write_log(cv_id: int, message: str):
    """Write log message to file"""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create log file name with timestamp and cv_id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"cv_{cv_id}_{timestamp}.txt"
    
    # Write log message with timestamp
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

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
    accuracy: float
    jaccard_scores: Dict[str, float]
    matched_primary_skills: List[str]
    matched_skills: List[str] = []  # Add new field for matched skills
    secondary_skills: List[str] = []  # Add new field for secondary skills

class CVMatchResponse(BaseModel):
    cv_id: int
    total_matches: int
    matches: List[MatchResult]


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
        try:
            # Convert all elements to strings and lowercase
            set1 = {str(item).lower().strip() for item in set1 if item}
            set2 = {str(item).lower().strip() for item in set2 if item}
            
            # If both sets are empty, return 0.0 instead of 1.0
            if not set1 and not set2:
                return 0.0
                
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return intersection / union if union > 0 else 0.0
        except Exception as e:
            print(f"⚠️ Warning: Error in jaccard_similarity: {e}")
            return 0.0

    try:
        print("\n📊 Calculating similarities:")
        print(f"Features1 (CV): {features1}")
        print(f"Features2 (JD): {features2}")

        # Ensure all values are lists before converting to sets
        for key in ['skills_required', 'primary_skills', 'secondary_skills', 'adverbs', 'adjectives']:
            if key in features1 and not isinstance(features1[key], list):
                features1[key] = []
            if key in features2 and not isinstance(features2[key], list):
                features2[key] = []

        # Convert skills to sets for set operations
        cv_skills = {str(skill).lower().strip() for skill in features1.get('skills_required', []) if skill}
        jd_skills = {str(skill).lower().strip() for skill in features2.get('primary_skills', []) if skill}

        # Calculate matched skills (intersection)
        matched_skills = cv_skills.intersection(jd_skills)
        print(f"Matched skills: {matched_skills}")

        # Calculate secondary skills (CV skills that didn't match with JD primary skills)
        secondary_skills = cv_skills - matched_skills
        print(f"Secondary skills (unmatched CV skills): {secondary_skills}")

        # Calculate similarities
        skills_required_sim = len(matched_skills) / len(cv_skills.union(jd_skills)) if cv_skills or jd_skills else 0.0
        
        # For secondary skills, we compare with JD's secondary skills
        jd_secondary_skills = {str(skill).lower().strip() for skill in features2.get('secondary_skills', []) if skill}
        secondary_skills_sim = jaccard_similarity(secondary_skills, jd_secondary_skills)
        
        adjectives_sim = jaccard_similarity(
            set(features1.get('adjectives', [])), 
            set(features2.get('adjectives', []))
        )

        # Tính similarity cho trọng số tính từ (adj_weight)
        adj_weight = 1 - abs(len(features1.get('adjectives', [])) - len(features2.get('adjectives', []))) / max(
            len(features1.get('adjectives', [])) + 1, len(features2.get('adjectives', [])) + 1)

        # Return only float values in jaccard_scores
        jaccard_scores = {
            'skills_required_sim': skills_required_sim,
            'secondary_skills_sim': secondary_skills_sim,
            'adjectives_sim': adjectives_sim,
            'adj_weight_sim': adj_weight
        }

        print(f"Skills required similarity: {skills_required_sim:.4f}")
        print(f"Secondary skills similarity: {secondary_skills_sim:.4f}")
        print(f"Adjectives similarity: {adjectives_sim:.4f}")
        print(f"Adjective weight: {adj_weight:.4f}")
        
        return {
            'jaccard_scores': jaccard_scores,
            'matched_skills': list(matched_skills),
            'secondary_skills': list(secondary_skills)
        }

    except Exception as e:
        print(f"⚠️ Warning: Error in calculate_similarity: {e}")
        # Return default values if calculation fails
        return {
            'jaccard_scores': {
                'skills_required_sim': 0.0,
                'secondary_skills_sim': 0.0,
                'adjectives_sim': 0.0,
                'adj_weight_sim': 1.0
            },
            'matched_skills': [],
            'secondary_skills': []
        }

def get_matched_primary_skills(cv_skills: List[str], job_skills: List[str]) -> List[str]:
    """Get the intersection of skills_required in CV and primary_skills in Job"""
    try:
        print("\n🔍 Matching Process:")
        print(f"CV skills: {cv_skills}")
        print(f"Job skills: {job_skills}")
        
        # Ensure both inputs are lists
        if not isinstance(cv_skills, list) or not isinstance(job_skills, list):
            print("⚠️ Warning: Input is not a list")
            cv_skills = list(cv_skills) if cv_skills else []
            job_skills = list(job_skills) if job_skills else []
        
        # Convert to lowercase and strip whitespace
        cv_skills_set = {str(skill).lower().strip() for skill in cv_skills if skill}
        job_skills_set = {str(skill).lower().strip() for skill in job_skills if skill}
        
        print(f"CV skills set: {cv_skills_set}")
        print(f"Job skills set: {job_skills_set}")
        
        # Get intersection
        matched_skills = cv_skills_set.intersection(job_skills_set)
        print(f"Matched skills: {matched_skills}")
        
        return list(matched_skills)
    except Exception as e:
        print(f"⚠️ Warning: Error in get_matched_primary_skills: {str(e)}")
        return []

def predict_suitability(similarities: Dict[str, float]) -> Dict:
    """Predict suitability and calculate accuracy based on similarities"""
    try:
        print("\n🔍 Predicting suitability:")
        print(f"Input similarities: {similarities}")
        
        # Check if model and scaler are loaded
        if model is None or scaler is None:
            raise HTTPException(status_code=500, detail="Model or scaler not loaded. Please check model files.")

        # Tạo feature vector từ các similarity scores
        features = {
            'primary_sim': similarities.get('skills_required_sim', 0.0),
            'secondary_sim': similarities.get('secondary_skills_sim', 0.0),
            'adj_sim': similarities.get('adjectives_sim', 0.0),
            'adj_weight': similarities.get('adj_weight_sim', 1.0)
        }
        
        print(f"Features for prediction: {features}")
        
        # Tính toán các derived features
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
        
        print(f"Input features shape: {input_features.shape}")
        
        # Chuẩn hóa dữ liệu
        scaled_input = scaler.transform(input_features)
        print(f"Scaled input: {scaled_input}")
        
        # Dự đoán và lấy xác suất
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[0]
        accuracy = float(prediction_proba[prediction])  # Lấy xác suất của class được dự đoán
        
        print(f"Prediction: {prediction}")
        print(f"Prediction probabilities: {prediction_proba}")
        print(f"Accuracy: {accuracy}")
        
        # Ánh xạ kết quả
        suitability_mapping = {
            0: 'Not Suitable',
            1: 'Moderately Suitable',
            2: 'Most Suitable'
        }
        
        result = {
            'suitability_label': suitability_mapping.get(prediction, 'Unknown'),
            'accuracy': accuracy
        }
        print(f"Final result: {result}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in predict_suitability: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details'}")
        return {
            'suitability_label': 'Unknown',
            'accuracy': 0.0
        }

def parse_job_skills(skills_data: str) -> List[str]:
    """Parse job skills from JSON string or comma-separated string"""
    if not skills_data:
        return []
    try:
        # Try to parse as JSON first
        skills_list = json.loads(skills_data)
        if isinstance(skills_list, list):
            return [skill.strip().lower() for skill in skills_list if skill]
    except json.JSONDecodeError:
        # If not JSON, try comma-separated
        return [skill.strip().lower() for skill in skills_data.split(',') if skill.strip()]
    return []

def parse_cv_skills(skills_data: str) -> List[str]:
    """Parse CV skills from JSON string or comma-separated string"""
    if not skills_data:
        return []
    try:
        # Try to parse as JSON first
        skills_list = json.loads(skills_data)
        if isinstance(skills_list, list):
            return [skill.strip().lower() for skill in skills_list if skill]
    except json.JSONDecodeError:
        # If not JSON, try comma-separated
        return [skill.strip().lower() for skill in skills_data.split(',') if skill.strip()]
    return []

@router.post("/cv/{cv_id}/match-all-jobs", response_model=CVMatchResponse)
async def match_cv_with_all_jobs(cv_id: int, db: Session = Depends(get_db)):
    """
    Match a CV with all jobs in the database and save results to matches table
    """
    # Create log file
    write_log(cv_id, f"Starting matching process for CV {cv_id}")
    
    # Get CV from database
    cv = db.query(CVModel).filter(CVModel.id == cv_id).first()
    if not cv:
        write_log(cv_id, f"CV {cv_id} not found")
        raise HTTPException(status_code=404, detail="CV not found")

    # Delete all existing matches for this CV first
    try:
        deleted_count = db.query(MatchesModel).filter(MatchesModel.cv_id == cv_id).delete()
        db.commit()
        write_log(cv_id, f"Deleted {deleted_count} existing matches for CV {cv_id}")
    except Exception as e:
        db.rollback()
        error_msg = f"Error deleting existing matches: {str(e)}"
        write_log(cv_id, f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

    # Parse CV skills
    try:
        cv_skills = parse_cv_skills(cv.primary_skills) if cv.primary_skills else []
        write_log(cv_id, f"\nCV Data:")
        write_log(cv_id, f"CV ID: {cv.id}")
        write_log(cv_id, f"Raw CV skills: {cv.primary_skills}")
        write_log(cv_id, f"Parsed CV skills: {cv_skills}")

        # Sử dụng skills_required từ CV
        cv_features = {
            'skills_required': cv_skills,
            'secondary_skills': parse_cv_skills(cv.secondary_skills) if cv.secondary_skills else [],
            'adverbs': cv.adverbs.split(',') if cv.adverbs else [],
            'adjectives': cv.adjectives.split(',') if cv.adjectives else []
        }
        write_log(cv_id, f"CV features: {cv_features}")
    except Exception as e:
        error_msg = f"Error processing CV features: {str(e)}"
        write_log(cv_id, f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

    # Get all active jobs
    jobs = db.query(JobModel).filter(JobModel.status == 1).all()
    if not jobs:
        write_log(cv_id, "No active jobs found")
        raise HTTPException(status_code=404, detail="No active jobs found")

    write_log(cv_id, f"\nProcessing {len(jobs)} active jobs")
    suitable_matches = []  # List to store only suitable matches
    new_matches = []  # List to store new matches for batch insert

    for job in jobs:
        try:
            write_log(cv_id, f"\nProcessing Job {job.id} - {job.title}")
            
            # Parse job skills from JSON
            try:
                job_skills = parse_job_skills(job.primary_skills) if job.primary_skills else []
                write_log(cv_id, f"Raw job skills: {job.primary_skills}")
                write_log(cv_id, f"Parsed job skills: {job_skills}")

                # Get job features
                job_features = {
                    'primary_skills': job_skills,
                    'secondary_skills': parse_job_skills(job.secondary_skills) if job.secondary_skills else [],
                    'adverbs': job.adverbs.split(',') if job.adverbs else [],
                    'adjectives': job.adjectives.split(',') if job.adjectives else []
                }
                write_log(cv_id, f"Job features: {job_features}")

                # Calculate Jaccard similarities
                try:
                    similarities = calculate_similarity(cv_features, job_features)
                    write_log(cv_id, f"Calculated similarities: {similarities}")

                    # Get matched primary skills
                    matched_primary_skills = get_matched_primary_skills(
                        cv_features['skills_required'],
                        job_features['primary_skills']
                    )
                    write_log(cv_id, f"Matched skills: {matched_primary_skills}")

                    # Predict suitability using the model
                    try:
                        prediction = predict_suitability(similarities['jaccard_scores'])
                        suitability_label = prediction['suitability_label']
                        accuracy = prediction['accuracy']
                        write_log(cv_id, f"Prediction: {suitability_label}, Accuracy: {accuracy:.4f}")
                    except Exception as e:
                        write_log(cv_id, f"⚠️ Warning: Could not predict suitability for job {job.id}: {e}")
                        suitability_label = "Unknown"
                        accuracy = 0.0

                    # Only process suitable matches
                    if suitability_label in ['Moderately Suitable', 'Most Suitable']:
                        # Create match result for response
                        match_result = MatchResult(
                            job_id=job.id,
                            job_title=job.title,
                            suitability_label=suitability_label,
                            accuracy=accuracy,
                            jaccard_scores=similarities['jaccard_scores'],
                            matched_primary_skills=matched_primary_skills,
                            matched_skills=similarities['matched_skills'],
                            secondary_skills=similarities['secondary_skills']
                        )
                        suitable_matches.append(match_result)

                        # Create new match for database
                        if matched_primary_skills:
                            matched_skills_str = ", ".join(matched_primary_skills)
                            new_match = MatchesModel(
                                cv_id=cv_id,
                                job_id=job.id,
                                matched_skill=matched_skills_str,
                                time_matches=datetime.now(),
                                status=1,
                                accuracy=accuracy
                            )
                            new_matches.append(new_match)
                            write_log(cv_id, f"✅ Added suitable match for job {job.id}")

                except Exception as e:
                    write_log(cv_id, f"⚠️ Warning: Error calculating similarities for job {job.id}: {e}")
                    continue
            except Exception as e:
                write_log(cv_id, f"⚠️ Warning: Error processing job features for job {job.id}: {e}")
                continue
        except Exception as e:
            write_log(cv_id, f"⚠️ Warning: Error processing job {job.id}: {e}")
            continue

    # Batch insert all new matches
    try:
        if new_matches:
            db.bulk_save_objects(new_matches)
            db.commit()
            write_log(cv_id, f"\n✅ Successfully saved {len(new_matches)} new matches to database")
        else:
            write_log(cv_id, "\n⚠️ No suitable matches found to save")
    except Exception as e:
        db.rollback()
        error_msg = f"Error saving matches to database: {str(e)}"
        write_log(cv_id, f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

    # Sắp xếp matches theo accuracy giảm dần
    suitable_matches.sort(key=lambda x: x.accuracy, reverse=True)

    write_log(cv_id, f"\nMatching process completed. Found {len(suitable_matches)} suitable matches.")
    
    return CVMatchResponse(
        cv_id=cv_id,
        total_matches=len(suitable_matches),
        matches=suitable_matches
    )

# Initialize model on startup
load_model()