from fastapi import APIRouter, Body, Depends, HTTPException, Request
from pydantic import BaseModel, validator
from typing import List, Dict, Optional
from datetime import datetime
from schemas.job_schemas import JobResponse4Cluster
from utils.read_file import read_skills
from utils.extract import extract_skills, extract_adjectives, extract_adverbs, clean_text_for_matching
from utils.connection_db import get_db, JobModel
from sqlalchemy.orm import Session
import spacy
import re
import json
import unicodedata

router = APIRouter(prefix="/job", tags=["Job Description Processing"])

# Load spaCy model
nlp_en = spacy.load('en_core_web_md')

# Load skills
skills = read_skills('app/skills.txt')

def extract_all_features(text_required: str, text_description: str) -> Dict[str, List[str]]:
    primary_skills_list = extract_skills(text_required, skills)
    secondary_skills_list = extract_skills(text_description, skills)
    adverbs = extract_adverbs(text_description, nlp_en)
    adjectives = extract_adjectives(text_description, nlp_en)

    return {
        'skills_required': primary_skills_list,
        'secondary_skills': secondary_skills_list,
        'adverbs': adverbs,
        'adjectives': adjectives
    }

class JobCreateRequest(BaseModel):
    employer_id: int
    title: str
    description: str
    required: str
    address: str
    location_id: int
    salary: str
    experience_id: int
    member: str
    work_type_id: int
    category_id: int
    posted_expired: Optional[str] = None  # Changed to string to handle JSON better

@router.post("/extract/all-features", response_model=JobResponse4Cluster)
async def extract_all_features_jd_api(
    job_data: JobCreateRequest,
    db: Session = Depends(get_db)
):
    # Clean and normalize all text inputs
    try:
        # Clean the main description text
        cleaned_description = clean_text_for_matching(job_data.description)

        # Also clean other text fields that might have similar issues
        cleaned_title = clean_text_for_matching(job_data.title)
        cleaned_required = clean_text_for_matching(job_data.required)
        cleaned_address = clean_text_for_matching(job_data.address)
        cleaned_salary = clean_text_for_matching(job_data.salary)
        cleaned_member = clean_text_for_matching(job_data.member)

        # Extract features from cleaned job description
        features = extract_all_features(cleaned_required,cleaned_description)

    except Exception as e:
        print(f"❌ Error processing job description: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing job description: {str(e)}")

    # Save job with features to database
    try:
        # Parse posted_expired if provided, otherwise use default
        if job_data.posted_expired:
            try:
                posted_expired = datetime.fromisoformat(job_data.posted_expired.replace('Z', '+00:00'))
            except ValueError:
                # If parsing fails, use current time + 30 days as default
                from datetime import timedelta
                posted_expired = datetime.now() + timedelta(days=30)
        else:
            from datetime import timedelta
            posted_expired = datetime.now() + timedelta(days=30)

        job_record = JobModel(
            employer_id=job_data.employer_id,
            title=cleaned_title,              # Use cleaned title
            description=cleaned_description,  # Use cleaned description
            required=cleaned_required,        # Use cleaned required
            address=cleaned_address,          # Use cleaned address
            location_id=job_data.location_id,
            salary=cleaned_salary,            # Use cleaned salary
            status=1,
            posted_at=datetime.now(),
            posted_expired=posted_expired,
            experience_id=job_data.experience_id,
            required_skills=", ".join(features['skills_required']) if features['skills_required'] else "",  # Sử dụng skills_required thay vì primary_skills
            member=cleaned_member,            # Use cleaned member
            work_type_id=job_data.work_type_id,
            category_id=job_data.category_id,
            primary_skills= ', '.join(features['skills_required']),  # Sử dụng skills_required thay vì primary_skills
            secondary_skills=', '.join(features['secondary_skills']),
            adverbs=', '.join(features['adverbs']),
            adjectives=', '.join(features['adjectives'])
        )

        db.add(job_record)
        db.commit()
        db.refresh(job_record)

        print(f"✅ Job features saved to database with ID: {job_record.id}")

    except Exception as e:
        db.rollback()
        print(f"❌ Error saving Job features to database: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    return JobResponse4Cluster(**features)

