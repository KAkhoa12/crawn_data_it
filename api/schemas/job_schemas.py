from typing import List, Dict
from pydantic import BaseModel

class JobBase(BaseModel):
    id: int
    employer_id: int
    title: str
    description: str
    description_json: Dict
    required: str
    address: str
    location_id: int
    salary: str
    status: int
    posted_at: str
    posted_expired: str
    experience_id: int
    required_skills: List[str]
    member: str
    work_type_id: int
    category_id: int
    primary_skills: List[str]
    secondary_skills: List[str]
    adverbs: List[str]
    adjectives: List[str]


class JobResponse4Cluster(BaseModel):
    primary_skills: List[str]
    secondary_skills: List[str]
    adverbs: List[str]
    adjectives: List[str]

class JobCreate(JobBase):
    pass

class JobUpdate(JobBase):
    pass

class Job(JobBase):
    id: int

    class Config:
        from_attributes = True