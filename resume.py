import os
from fastapi import UploadFile
from sqlalchemy.orm import Session
from db.models  import Resume
from dotenv import load_dotenv
from db.schema import ResumeCreate

load_dotenv()

UPLOAD_DIR = "resume"
BASE_URL =  os.getenv("BASE_URL")  # adapte si ton serveur change

def create_resume(db: Session, email: str, file: UploadFile):
    # Vérifier que le dossier existe
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Construire le chemin complet du fichier
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Sauvegarder le fichier sur disque
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    # Construire l’URL publique
    resume_link = f"{BASE_URL}/resume/{file.filename}"

    # Sauvegarde en base
    db_resume = Resume(
        email_candidate=email,
        resume_link=resume_link
    )
    db.add(db_resume)
    db.commit()
    db.refresh(db_resume)

    return db_resume
