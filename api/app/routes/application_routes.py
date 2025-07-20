from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime
import os
from pathlib import Path
from typing import List
import re
import logging
from logging.handlers import RotatingFileHandler
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import asyncio

from utils.connection_db import get_db

router = APIRouter(
    prefix="/application",
    tags=["application"]
)

# Initialize scheduler
scheduler = AsyncIOScheduler()

# Set up logging
def setup_logger():
    log_dir = Path(__file__).parent.parent.parent / 'logs'
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    
    log_file = log_dir / 'application_matches.log'
    
    # Check if logger already exists
    logger = logging.getLogger('application_matches')
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    handler.setFormatter(formatter)
    
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    return logger

def read_skills_file():
    skills_file = Path(__file__).parent.parent / 'skills_job.txt'
    with open(skills_file, 'r', encoding='utf-8') as f:
        skills = [line.strip() for line in f.readlines() if line.strip()]
    return skills

def extract_matching_skills(title: str, skills: List[str]) -> List[str]:
    matching_skills = []
    for skill in skills:
        # Create a word boundary regex pattern
        pattern = r'\b' + re.escape(skill.lower().strip()) + r'\b'
        if re.search(pattern, title, re.IGNORECASE):
            matching_skills.append(skill.lower().strip())
    return matching_skills

def process_matches_sync(db: Session):
    logger = setup_logger()
    logger.info("Starting scheduled job matches processing")
    
    try:
        # Read skills from file
        skills = read_skills_file()
        logger.info(f"Loaded {len(skills)} skills from skills_job.txt")
        
        # Get all applications grouped by seeker_id with cv_id
        application_query = text("""
            SELECT DISTINCT 
                a.seeker_id, 
                a.job_id, 
                j.title,
                c.id as cv_id
            FROM application a
            JOIN job j ON a.job_id = j.id
            JOIN cv c ON c.seeker_id = a.seeker_id
            ORDER BY a.seeker_id
        """)
        applications = db.execute(application_query).fetchall()
        logger.info(f"Found {len(applications)} applications to process")
        
        matches_count = 0
        current_seeker = None
        seeker_jobs = []
        
        for app in applications:
            seeker_id = app.seeker_id
            job_id = app.job_id
            title = app.title
            cv_id = app.cv_id
            
            # If we encounter a new seeker_id, process the previous seeker's jobs
            if current_seeker is not None and current_seeker != seeker_id:
                logger.info(f"Processing jobs for seeker_id: {current_seeker}")
                # Process all jobs for the current seeker
                for job in seeker_jobs:
                    logger.info(f"Processing job_id: {job['job_id']}, title: {job['title']}")
                    matching_skills = extract_matching_skills(job['title'], skills)
                    
                    if matching_skills:
                        logger.info(f"Found matching skills for job {job['job_id']}: {', '.join(matching_skills)}")
                        
                        # Check if match already exists
                        check_query = text("""
                            SELECT 1 FROM matches 
                            WHERE cv_id = :cv_id AND job_id = :job_id
                        """)
                        existing_match = db.execute(
                            check_query,
                            {
                                "cv_id": job['cv_id'],
                                "job_id": job['job_id']
                            }
                        ).fetchone()
                        
                        if not existing_match:
                            # Insert match with matched skills
                            insert_query = text("""
                                INSERT INTO matches 
                                (cv_id, job_id, matched_skill, time_matches, status)
                                VALUES 
                                (:cv_id, :job_id, :matched_skill, :time_matches, :status)
                            """)
                            
                            db.execute(
                                insert_query,
                                {
                                    "cv_id": job['cv_id'],
                                    "job_id": job['job_id'],
                                    "matched_skill": ", ".join(matching_skills),
                                    "time_matches": datetime.now(),
                                    "status": 1
                                }
                            )
                            matches_count += 1
                            logger.info(f"Added new match for cv_id: {job['cv_id']}, job_id: {job['job_id']}")
                        else:
                            logger.info(f"Match already exists for cv_id: {job['cv_id']}, job_id: {job['job_id']}")
                    else:
                        logger.info(f"No matching skills found for job {job['job_id']}")
                
                # Clear the jobs list for the next seeker
                seeker_jobs = []
            
            # Update current seeker and add job to the list
            current_seeker = seeker_id
            seeker_jobs.append({
                'job_id': job_id,
                'title': title,
                'cv_id': cv_id
            })
        
        # Process the last seeker's jobs
        if seeker_jobs:
            logger.info(f"Processing jobs for last seeker_id: {current_seeker}")
            for job in seeker_jobs:
                logger.info(f"Processing job_id: {job['job_id']}, title: {job['title']}")
                matching_skills = extract_matching_skills(job['title'], skills)
                
                if matching_skills:
                    logger.info(f"Found matching skills for job {job['job_id']}: {', '.join(matching_skills)}")
                    
                    # Check if match already exists
                    check_query = text("""
                        SELECT 1 FROM matches 
                        WHERE cv_id = :cv_id AND job_id = :job_id
                    """)
                    existing_match = db.execute(
                        check_query,
                        {
                            "cv_id": job['cv_id'],
                            "job_id": job['job_id']
                        }
                    ).fetchone()
                    
                    if not existing_match:
                        # Insert match with matched skills
                        insert_query = text("""
                            INSERT INTO matches 
                            (cv_id, job_id, matched_skill, time_matches, status)
                            VALUES 
                            (:cv_id, :job_id, :matched_skill, :time_matches, :status)
                        """)
                        
                        db.execute(
                            insert_query,
                            {
                                "cv_id": job['cv_id'],
                                "job_id": job['job_id'],
                                "matched_skill": ", ".join(matching_skills),
                                "time_matches": datetime.now(),
                                "status": 1
                            }
                        )
                        matches_count += 1
                        logger.info(f"Added new match for cv_id: {job['cv_id']}, job_id: {job['job_id']}")
                    else:
                        logger.info(f"Match already exists for cv_id: {job['cv_id']}, job_id: {job['job_id']}")
                else:
                    logger.info(f"No matching skills found for job {job['job_id']}")
        
        db.commit()
        logger.info(f"Scheduled job matches processing completed. Added {matches_count} new matches.")
        
    except Exception as e:
        logger.error(f"Error during scheduled job matches processing: {str(e)}")
        db.rollback()

async def process_matches():
    # Get a new database session
    db = next(get_db())
    try:
        process_matches_sync(db)
    finally:
        db.close()

# Add an endpoint to check scheduler status
@router.get("/scheduler-status")
async def get_scheduler_status():
    return {
        "is_running": scheduler.running,
        "job_count": len(scheduler.get_jobs()),
        "next_run_time": scheduler.get_job('process_matches_job').next_run_time if scheduler.running else None
    }
