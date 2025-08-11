from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

model_id = "openai/whisper-large-v3-turbo"
local_dir = "./whisper-large-v3-turbo"

# Téléchargement et sauvegarde locale
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
model.save_pretrained(local_dir)

processor = AutoProcessor.from_pretrained(model_id)
processor.save_pretrained(local_dir)
