from typing import List, Dict
from pydantic import BaseModel

class MatchesBase(BaseModel):
    id: int
    cv_id: int
    job_id: int
    matched_skill: str
    time_matches: str
    status: int

class MatchesCreate(MatchesBase):
    pass

class MatchesUpdate(MatchesBase):
    pass

class Matches(MatchesBase):
    id: int

    class Config:
        from_attributes = True