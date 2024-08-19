from pydantic import BaseModel
from datetime import datetime

class SmileageCreate(BaseModel):
    mileage: int
    emotion: str
    probability: float
    created_at: datetime

    class Config:
        orm_mode = True
