from sqlalchemy import Column, Integer, String, DateTime, func, Date,Time
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


class Meeting(Base):
    __tablename__="meeting"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    date_meeting= Column(Date,nullable=False)
    hour = Column(Time, nullable=False)
    platform = Column(String,nullable=False)
    email=Column(String,nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())



class Resume(Base):
    __tablename__="resume"
    id = Column(Integer, primary_key=True, index=True)
    email_candidate = Column(String, nullable=False)
    resume_link = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
