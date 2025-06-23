from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import nltk
from app.routes.root_routes import router as root_router
from app.routes.cv_routes import router as cv_router
from  app.routes.job_routes import router as job_router
from  app.routes.matches_routes import router as matches_router

# Download required NLTK data
nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')

app = FastAPI(title="Job-Resume Matching API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Include routers
app.include_router(root_router)
app.include_router(cv_router)
app.include_router(job_router)
app.include_router(matches_router)

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000)