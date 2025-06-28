from typing import List, Dict
from pydantic import BaseModel

class CVBase(BaseModel):
    id: int
    seeker_id: int
    name: str
    skills: str
    experience: str
    type: str
    education: str
    certification: str
    status: int
    offer_salary: str
    job_deadline: str
    linked_in: str
    link_git: str
    primary_skills: List[str]
    secondary_skills: List[str]
    adverbs: List[str]
    adjectives: List[str]
    upload_at: str

class CVResponse4Cluster(BaseModel):
    skills_required: List[str]
    adverbs: List[str]
    adjectives: List[str]


class CVCreate(CVBase):
    pass

class CVUpdate(CVBase):
    pass

class CV(CVBase):
    id: int

    class Config:
        from_attributes = True