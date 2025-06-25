from fastapi import APIRouter, Body, Depends, HTTPException, Request
from pydantic import BaseModel, validator
from typing import List, Dict, Optional
from datetime import datetime
from schemas.job_schemas import JobResponse4Cluster
from utils.read_file import read_skills
from utils.extract import extract_primary_skills, extract_secondary_skills, extract_adjectives, extract_adverbs
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
primary_skills = read_skills('app/primary_skills.txt')
secondary_skills = read_skills('app/secondary_skills.txt')

def clean_input_text(text: str) -> str:
    """
    Clean and sanitize input text to handle problematic characters
    that can cause parsing errors or JSON issues.
    """
    if not text:
        return ""

    try:
        # Step 1: Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)

        # Step 2: Replace different types of line breaks with standard newlines
        text = text.replace('\r\n', '\n')  # Windows
        text = text.replace('\r', '\n')    # Mac

        # Step 3: Replace tabs with spaces
        text = text.replace('\t', ' ')

        # Step 4: Remove control characters except newlines (keep \n for readability)
        # Remove characters in ranges: 0x00-0x08, 0x0B-0x0C, 0x0E-0x1F, 0x7F-0x9F
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)

        # Step 5: Remove or replace problematic characters for JSON
        # Replace smart quotes with regular quotes
        text = text.replace('"', '"').replace('"', '"')  # Smart double quotes
        text = text.replace(''', "'").replace(''', "'")  # Smart single quotes

        # Step 6: Clean up excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r'[ \t]+', ' ', text)            # Multiple spaces/tabs to single space

        # Step 7: Trim leading/trailing whitespace
        text = text.strip()

        return text

    except Exception as e:
        print(f"⚠️ Warning: Error cleaning text: {e}")
        # Fallback: basic cleaning
        return re.sub(r'[\x00-\x1F\x7F-\x9F]', '', str(text)).strip()

def extract_all_features(text: str) -> Dict[str, List[str]]:
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
        cleaned_description = clean_input_text(job_data.description)

        # Also clean other text fields that might have similar issues
        cleaned_title = clean_input_text(job_data.title)
        cleaned_required = clean_input_text(job_data.required)
        cleaned_address = clean_input_text(job_data.address)
        cleaned_salary = clean_input_text(job_data.salary)
        cleaned_member = clean_input_text(job_data.member)

        # Extract features from cleaned job description
        features = extract_all_features(cleaned_description)

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
            required_skills=", ".join(features['primary_skills']) if features['primary_skills'] else "",
            member=cleaned_member,            # Use cleaned member
            work_type_id=job_data.work_type_id,
            category_id=job_data.category_id,
            primary_skills= ', '.join(features['primary_skills']),
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

