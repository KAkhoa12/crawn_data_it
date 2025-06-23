from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def root():
    return {"message": "Job-Resume Matching API", "status": "healthy"}

@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Job-Resume Matching API"}