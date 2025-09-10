from pydantic import BaseModel
from datetime import date, time

class MeetingCreate(BaseModel):
    title: str
    date_meeting: date
    hour: time
    platform: str
    email: str
