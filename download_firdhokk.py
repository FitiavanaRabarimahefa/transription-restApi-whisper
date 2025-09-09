from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
local_dir = "./speech-emotion-recognition-whisper-v3"

# 1. Téléchargement et sauvegarde locale du modèle
model = AutoModelForAudioClassification.from_pretrained(model_id)
model.save_pretrained(local_dir)

# 2. Téléchargement et sauvegarde locale du feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
feature_extractor.save_pretrained(local_dir)

print(f"Modèle et feature extractor sauvegardés dans : {local_dir}")
