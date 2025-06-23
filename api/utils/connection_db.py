from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from datetime import datetime
import os

DATABASE_URL = f"mysql+pymysql://{os.getenv('DB_USER', 'root')}:{os.getenv('DB_PASSWORD', '')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '3306')}/{os.getenv('DB_NAME', 'jobs')}"

# Create engine
engine = create_engine(DATABASE_URL, echo=True)

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# CV Model
class CVModel(Base):
    __tablename__ = "cv"

    id = Column(Integer, primary_key=True, index=True)
    seeker_id = Column(Integer, nullable=False)
    name = Column(Text, nullable=False)
    skills = Column(Text)
    experience = Column(Text)
    type = Column(Text)
    education = Column(Text)
    certification = Column(Text)
    status = Column(Integer, nullable=False)
    offer_salary = Column(Text)
    job_deadline = Column(Text)
    linked_in = Column(Text)
    link_git = Column(Text)
    upload_at = Column(DateTime, nullable=False)
    # New columns for features
    primary_skills = Column(JSON)
    secondary_skills = Column(JSON)
    adverbs = Column(JSON)
    adjectives = Column(JSON)

# Job Model
class JobModel(Base):
    __tablename__ = "job"

    id = Column(Integer, primary_key=True, index=True)
    employer_id = Column(Integer, nullable=False)
    title = Column(Text, nullable=False)
    description = Column(Text)
    description_json = Column(JSON)
    required = Column(Text, nullable=False)
    address = Column(Text, nullable=False)
    location_id = Column(Integer, nullable=False)
    salary = Column(Text, nullable=False)
    status = Column(Integer, nullable=False)
    posted_at = Column(DateTime, nullable=False)
    posted_expired = Column(DateTime, nullable=False)
    experience_id = Column(Integer, nullable=False)
    required_skills = Column(Text, nullable=False)
    member = Column(Text, nullable=False)
    work_type_id = Column(Integer, nullable=False)
    category_id = Column(Integer, nullable=False)
    # New columns for features
    primary_skills = Column(JSON)
    secondary_skills = Column(JSON)
    adverbs = Column(JSON)
    adjectives = Column(JSON)

# Matches Model
class MatchesModel(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True, index=True)
    cv_id = Column(Integer, nullable=False)
    job_id = Column(Integer, nullable=False)
    matched_skill = Column(Text, nullable=False)
    time_matches = Column(DateTime, nullable=False)
    status = Column(Integer, nullable=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()