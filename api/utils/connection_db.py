from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from datetime import datetime
import os
import json
from typing import List, Optional

DATABASE_URL = f"mysql+pymysql://{os.getenv('DB_USER', 'root')}:{os.getenv('DB_PASSWORD', '')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '3306')}/{os.getenv('DB_NAME', 'jobs')}"

# Create engine
engine = create_engine(DATABASE_URL, echo=True)

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

def serialize_list(value: Optional[List[str]]) -> Optional[str]:
    """Convert list to JSON string, handling None values"""
    if value is None:
        return None
    return json.dumps(list(value))

def deserialize_list(value: Optional[str]) -> List[str]:
    """Convert JSON string to list, handling None and invalid JSON"""
    if not value:
        return []
    try:
        result = json.loads(value)
        return list(result) if isinstance(result, list) else []
    except (json.JSONDecodeError, TypeError):
        return []

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
    primary_skills = Column(Text)  # Changed from JSON to Text
    secondary_skills = Column(Text)  # Changed from JSON to Text
    adverbs = Column(Text)  # Changed from JSON to Text
    adjectives = Column(Text)  # Changed from JSON to Text

    def __init__(self, **kwargs):
        # Convert lists to JSON strings before saving
        if 'primary_skills' in kwargs:
            kwargs['primary_skills'] = serialize_list(kwargs['primary_skills'])
        if 'secondary_skills' in kwargs:
            kwargs['secondary_skills'] = serialize_list(kwargs['secondary_skills'])
        if 'adverbs' in kwargs:
            kwargs['adverbs'] = serialize_list(kwargs['adverbs'])
        if 'adjectives' in kwargs:
            kwargs['adjectives'] = serialize_list(kwargs['adjectives'])
        super().__init__(**kwargs)

    @property
    def primary_skills_list(self) -> List[str]:
        return deserialize_list(self.primary_skills)

    @primary_skills_list.setter
    def primary_skills_list(self, value: List[str]):
        self.primary_skills = serialize_list(value)

    @property
    def secondary_skills_list(self) -> List[str]:
        return deserialize_list(self.secondary_skills)

    @secondary_skills_list.setter
    def secondary_skills_list(self, value: List[str]):
        self.secondary_skills = serialize_list(value)

    @property
    def adverbs_list(self) -> List[str]:
        return deserialize_list(self.adverbs)

    @adverbs_list.setter
    def adverbs_list(self, value: List[str]):
        self.adverbs = serialize_list(value)

    @property
    def adjectives_list(self) -> List[str]:
        return deserialize_list(self.adjectives)

    @adjectives_list.setter
    def adjectives_list(self, value: List[str]):
        self.adjectives = serialize_list(value)

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
    primary_skills = Column(Text)  # Changed from JSON to Text
    secondary_skills = Column(Text)  # Changed from JSON to Text
    adverbs = Column(Text)  # Changed from JSON to Text
    adjectives = Column(Text)  # Changed from JSON to Text

# Matches Model
class MatchesModel(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True, index=True)
    cv_id = Column(Integer, nullable=False)
    job_id = Column(Integer, nullable=False)
    accuracy = Column(Float, nullable=False)  # Change from Integer to Float
    label = Column(String(50), nullable=True)  # Add label column
    matched_skill = Column(Text, nullable=False)
    time_matches = Column(DateTime, nullable=False)
    status = Column(Integer, nullable=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()