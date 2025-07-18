from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Form
from pydantic import BaseModel
from typing import List, Dict, Optional
import PyPDF2
import io
from datetime import datetime
from utils.read_file import read_skills
from schemas.cv_schemas import CVResponse4Cluster
from utils.extract import extract_skills, extract_adjectives, extract_adverbs, clean_text_for_matching
from utils.connection_db import get_db, CVModel, MatchesModel
from sqlalchemy.orm import Session
import spacy
from docx import Document

router = APIRouter(prefix="/cv", tags=["CV Processing"])

# Load spaCy model
nlp_en = spacy.load('en_core_web_md')

# Load skills
skills = read_skills('app/skills.txt')

class PrimarySkillsResponse(BaseModel):
    skills: List[str]

def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return clean_text_for_matching(text)
    except:
        return ""

def extract_text_from_docx(file_content: bytes) -> str:
    try:
        doc = Document(io.BytesIO(file_content))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return clean_text_for_matching(text)
    except:
        return ""

def extract_text_from_file(file_content: bytes, content_type: str) -> str:
    if content_type == "application/pdf":
        return extract_text_from_pdf(file_content)
    elif content_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
        return extract_text_from_docx(file_content)
    else:
        return ""

def extract_all_features(text: str) -> Dict[str, List[str]]:
    try:
        print("📝 Extracting features from text:", text[:100] + "...")  # Debug log
        
        # Extract features
        skills_list = extract_skills(text, skills)
        print("📝 Extracted skills:", skills_list)  # Debug log
        
        adverbs = extract_adverbs(text, nlp_en)
        print("📝 Extracted adverbs:", adverbs)  # Debug log
        
        adjectives = extract_adjectives(text, nlp_en)
        print("📝 Extracted adjectives:", adjectives)  # Debug log

        # Ensure all values are lists
        features = {
            'skills_required': list(skills_list) if skills_list else [],
            'adverbs': list(adverbs) if adverbs else [],
            'adjectives': list(adjectives) if adjectives else []
        }
        
        print("📝 Final features:", features)  # Debug log
        return features
        
    except Exception as e:
        print(f"❌ Error in extract_all_features: {str(e)}")
        print(f"❌ Error type: {type(e)}")
        print(f"❌ Error details: {e.__dict__}")
        # Return empty lists if extraction fails
        return {
            'skills_required': [],
            'adverbs': [],
            'adjectives': []
        }

@router.post("/extract/skills", response_model=PrimarySkillsResponse)
async def extract_primary_skills_api(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    content = await file.read()
    text = extract_text_from_pdf(content)

    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")

    skills_list = extract_skills(text, skills)
    return PrimarySkillsResponse(skills=skills_list)

class CVExtractRequest(BaseModel):
    seeker_id: int
    name: str

@router.post("/extract/all-features", response_model=CVResponse4Cluster)
async def extract_all_features_api(
    file: UploadFile = File(...),
    seeker_id: int = Form(...),
    db: Session = Depends(get_db)
):
    # Validate file type
    allowed_types = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword"
    ]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and Word documents are supported"
        )

    # Extract text from file
    content = await file.read()
    text = extract_text_from_file(content, file.content_type)

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Could not extract text from the uploaded file"
        )

    # Extract features
    features = extract_all_features(text)
    print("📝 Extracted features:", features)  # Debug log

    # Convert lists to strings for text fields
    skills_str = ', '.join(features['skills_required']) if features['skills_required'] else ""
    print("📝 Skills string:", skills_str)  # Debug log

    # Save to database - Check if CV exists for this seeker_id
    try:
        # Check if CV already exists for this seeker_id
        existing_cv = db.query(CVModel).filter(CVModel.seeker_id == seeker_id).first()
        print("📝 Existing CV:", existing_cv)  # Debug log

        if existing_cv:
            # Delete all existing matches for this CV before updating
            print(f"🗑️ Deleting existing matches for CV ID: {existing_cv.id}")
            db.query(MatchesModel).filter(MatchesModel.cv_id == existing_cv.id).delete()

            # Update existing CV
            existing_cv.name = file.filename
            existing_cv.upload_at = datetime.now()
            existing_cv.skills = skills_str
            existing_cv.primary_skills_list = features['skills_required']  # Use property
            existing_cv.secondary_skills_list = []  # Use property
            existing_cv.adverbs_list = features['adverbs']  # Use property
            existing_cv.adjectives_list = features['adjectives']  # Use property
            existing_cv.experience = None  # Reset experience

            print("📝 Updated CV fields:", {  # Debug log
                'name': existing_cv.name,
                'skills': existing_cv.skills,
                'primary_skills': existing_cv.primary_skills,
                'adverbs': existing_cv.adverbs,
                'adjectives': existing_cv.adjectives
            })

            db.commit()
            db.refresh(existing_cv)

            print(f"✅ CV features updated in database with ID: {existing_cv.id}")
            cv_id = existing_cv.id
        else:
            # Create new CV
            cv_record = CVModel(
                seeker_id=seeker_id,
                name=file.filename,
                skills=skills_str,
                primary_skills=features['skills_required'],  # Will be converted in __init__
                secondary_skills=[],  # Will be converted in __init__
                experience=None,
                status=1,
                upload_at=datetime.now(),
                adverbs=features['adverbs'],  # Will be converted in __init__
                adjectives=features['adjectives']  # Will be converted in __init__
            )

            print("📝 New CV record:", {  # Debug log
                'name': cv_record.name,
                'skills': cv_record.skills,
                'primary_skills': cv_record.primary_skills,
                'adverbs': cv_record.adverbs,
                'adjectives': cv_record.adjectives
            })

            db.add(cv_record)
            db.commit()
            db.refresh(cv_record)

            print(f"✅ CV features saved to database with ID: {cv_record.id}")
            cv_id = cv_record.id

    except Exception as e:  
        db.rollback()
        print(f"❌ Error saving CV features to database: {str(e)}")
        print(f"❌ Error type: {type(e)}")  # Debug log
        print(f"❌ Error details: {e.__dict__}")  # Debug log
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    # Return the features directly since they're already in the correct format
    response = CVResponse4Cluster(**features)
    print("📝 Response:", response)  # Debug log
    return response

