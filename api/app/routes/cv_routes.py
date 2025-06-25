from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Form
from pydantic import BaseModel
from typing import List, Dict, Optional
import PyPDF2
import io
from datetime import datetime
from utils.read_file import read_skills
from schemas.cv_schemas import CVResponse4Cluster
from utils.extract import extract_primary_skills, extract_secondary_skills, extract_adjectives, extract_adverbs
from utils.connection_db import get_db, CVModel, MatchesModel
from sqlalchemy.orm import Session
import spacy
from docx import Document

router = APIRouter(prefix="/cv", tags=["CV Processing"])

# Load spaCy model
nlp_en = spacy.load('en_core_web_md')

# Load skills
primary_skills = read_skills('app/primary_skills.txt')
secondary_skills = read_skills('app/secondary_skills.txt')

class PrimarySkillsResponse(BaseModel):
    primary_skills: List[str]

def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except:
        return ""

def extract_text_from_docx(file_content: bytes) -> str:
    try:
        doc = Document(io.BytesIO(file_content))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
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
    primarys = extract_primary_skills(text, primary_skills)
    secondarys = extract_secondary_skills(text, secondary_skills)
    adverbs = extract_adverbs(text, nlp_en)
    adjectives = extract_adjectives(text, nlp_en)

    return {
        'primary_skills': primarys,
        'secondary_skills': secondarys,
        'adverbs': adverbs,
        'adjectives': adjectives
    }

@router.post("/extract/primary-skills", response_model=PrimarySkillsResponse)
async def extract_primary_skills_api(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    content = await file.read()
    text = extract_text_from_pdf(content)

    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")

    skills = extract_primary_skills(text, primary_skills)
    return PrimarySkillsResponse(primary_skills=skills)

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

    # Save to database - Check if CV exists for this seeker_id
    try:
        # Check if CV already exists for this seeker_id
        existing_cv = db.query(CVModel).filter(CVModel.seeker_id == seeker_id).first()

        if existing_cv:
            # Delete all existing matches for this CV before updating
            print(f"🗑️ Deleting existing matches for CV ID: {existing_cv.id}")
            db.query(MatchesModel).filter(MatchesModel.cv_id == existing_cv.id).delete()

            # Update existing CV
            existing_cv.name = file.filename
            existing_cv.upload_at = datetime.now()
            existing_cv.primary_skills = ', '.join(features['primary_skills'])
            existing_cv.secondary_skills = ', '.join(features['secondary_skills'])
            existing_cv.adverbs = ', '.join(features['adverbs'])
            existing_cv.adjectives = ', '.join(features['adjectives'])

            db.commit()
            db.refresh(existing_cv)

            print(f"✅ CV features updated in database with ID: {existing_cv.id}")
            cv_id = existing_cv.id
        else:
            # Create new CV
            cv_record = CVModel(
                seeker_id=seeker_id,
                name=file.filename,
                skills="skills",
                experience="experiments",
                status=1,
                upload_at=datetime.now(),
                primary_skills=features['primary_skills'],
                secondary_skills=features['secondary_skills'],
                adverbs=features['adverbs'],
                adjectives=features['adjectives']
            )

            db.add(cv_record)
            db.commit()
            db.refresh(cv_record)

            print(f"✅ CV features saved to database with ID: {cv_record.id}")
            cv_id = cv_record.id

    except Exception as e:
        db.rollback()
        print(f"❌ Error saving CV features to database: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    return CVResponse4Cluster(**features)

