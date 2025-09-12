from pydantic import BaseModel,EmailStr
from datetime import date, time

class MeetingCreate(BaseModel):
    title: str
    date_meeting: date
    hour: time
    platform: str
    email: str


class ResumeCreate(BaseModel):
    email_candidate: str
    resume_link: str
