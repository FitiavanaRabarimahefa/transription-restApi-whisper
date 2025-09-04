import os
import shutil
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session
from db.models import Audio

UPLOAD_DIR = "audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)
BASE_URL = "http://127.0.0.1:8001"


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
