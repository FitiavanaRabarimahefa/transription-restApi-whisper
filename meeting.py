from sqlalchemy.orm import Session
from datetime import date, time
from db.models import Meeting

def create_meeting(db: Session, title: str, date_meeting: date, hour: time, platform: str, email: str):
    new_meeting = Meeting(
        title=title,
        date_meeting=date_meeting,
        hour=hour,
        platform=platform,
        email=email
    )
    db.add(new_meeting)
    db.commit()
    db.refresh(new_meeting)
    return new_meeting
