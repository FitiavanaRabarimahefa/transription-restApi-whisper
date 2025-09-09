import librosa
import tempfile
import os
from transformers import pipeline, AutoModelForAudioClassification, AutoFeatureExtractor

LOCAL_DIR = "./speech-emotion-recognition-whisper-v3"
model = AutoModelForAudioClassification.from_pretrained(LOCAL_DIR)
feature_extractor = AutoFeatureExtractor.from_pretrained(LOCAL_DIR)

classifier = pipeline(
    task="audio-classification",
    model=model,
    feature_extractor=feature_extractor,
)

def analyze_audio(file_content: bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(file_content)
        tmp.flush()
        tmp_name = tmp.name   # garder le chemin

    try:
        # Charger l'audio après avoir fermé le fichier
        y, sr = librosa.load(tmp_name, sr=None, mono=True)
        inputs = {"array": y, "sampling_rate": sr}
        results = classifier(inputs)
        return results
    finally:
        # Nettoyer le fichier
        os.remove(tmp_name)
