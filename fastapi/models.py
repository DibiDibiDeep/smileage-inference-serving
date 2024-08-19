from sqlalchemy import Column, Integer, String, Float, TIMESTAMP
from .db import Base
from datetime import datetime

class Smileage(Base):
    __tablename__ = "smileage"

    id = Column(Integer, primary_key=True, index=True)
    mileage = Column(Integer, nullable=False)
    emotion = Column(String(50), nullable=False)
    probability = Column(Float, nullable=False)
    created_at = Column(TIMESTAMP, default=datetime.now)
