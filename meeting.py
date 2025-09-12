from sqlalchemy.orm import Session
from datetime import date, time
from db.models import Meeting
from fastapi import HTTPException

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


def get_meetings(db: Session):
    return db.query(Meeting).all()


def delete_meeting(db: Session, meeting_id: int):
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    db.delete(meeting)
    db.commit()
    return {"message": f"Meeting {meeting_id} supprimé avec succès"}
