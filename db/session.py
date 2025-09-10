from sqlalchemy.orm import Session
from db.database import SessionLocal

def get_db():
    """
    Fournit une session SQLAlchemy pour FastAPI et la ferme automatiquement
    après usage grâce à la dépendance Depends().
    """
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
