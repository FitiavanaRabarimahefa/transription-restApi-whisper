from sqlalchemy import Column, Integer, String, DateTime, func
from db.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String,unique=True,index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Audio(Base):
    __tablename__ = "audios"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())