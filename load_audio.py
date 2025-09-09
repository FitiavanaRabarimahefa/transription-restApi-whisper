import os
import shutil
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session
from db.models import Audio
from dotenv import load_dotenv
from typing import List

load_dotenv()

UPLOAD_DIR = "audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)
BASE_URL = os.getenv("BASE_URL")


def save_audio_file(file: UploadFile, db: Session) -> Audio:
    # Vérifie le type du fichier
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers audio sont acceptés.")

    # Chemin local pour sauvegarde
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Sauvegarde du fichier sur disque
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # URL publique (utilise toujours des /)
    file_url = f"{BASE_URL}/audio/{file.filename}"

    # Création de l'enregistrement dans la DB
    audio_record = Audio(
        filename=file.filename,
        filepath=file_url,
    )

    db.add(audio_record)
    db.commit()
    db.refresh(audio_record)

    return audio_record


def get_audio_files(db: Session) -> List[Audio]:
    """
    Récupère tous les fichiers audio enregistrés dans la base.
    """
    return db.query(Audio).all()


def get_audio_file_by_id(audio_id: int, db: Session) -> Audio:
    """
    Récupère un fichier audio par son ID.
    """
    audio = db.query(Audio).filter(Audio.id == audio_id).first()
    if not audio:
        raise HTTPException(status_code=404, detail="Fichier audio introuvable.")
    return audio

