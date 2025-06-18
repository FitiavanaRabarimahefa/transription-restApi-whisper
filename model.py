import torch
import torchaudio
import warnings
import os
from pathlib import Path
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
    GenerationConfig
)
from transformers.utils import logging

# Supprimer les warnings inutiles
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.set_verbosity_error()

# Désactiver le warning HF
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Choix du device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
print(f" Device utilisé : {device} | dtype : {torch_dtype}")

# Chemin vers un fichier audio unique
audio_path = Path("./audio/533826.wav")
if not audio_path.exists():
    raise FileNotFoundError(f" Fichier introuvable : {audio_path.resolve()}")

# Chargement du modèle Whisper local
local_model_dir = "./whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    local_model_dir,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
).to(device)

# Chargement de la configuration de génération
generation_config = GenerationConfig.from_pretrained(local_model_dir)
generation_config.forced_decoder_ids = None
model.generation_config = generation_config

# Chargement du processeur
processor = AutoProcessor.from_pretrained(local_model_dir)

# Création du pipeline ASR
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    generate_kwargs={"language": "fr"}
)

# Chargement de l'audio
print(f"\n▶ Traitement du fichier : {audio_path.name}")
waveform, sample_rate = torchaudio.load(audio_path)

# Vérification et conversion en mono si multicanal
if waveform.shape[0] > 1:
    print(f" Conversion en mono depuis {waveform.shape[0]} canaux")
    waveform = torch.mean(waveform, dim=0, keepdim=True)

# Resampling si nécessaire
if sample_rate != 16000:
    print(f"🎚 Resampling {sample_rate} Hz → 16000 Hz")
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

# Transcription
print(" Transcription en cours...")
result = pipe(waveform.squeeze().numpy(), return_timestamps=True)

# Affichage du texte transcrit
print("\n Transcription :")
print(result["text"])
