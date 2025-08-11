from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse,FileResponse
import shutil
from pathlib import Path
import torch
import torchaudio
import warnings
import os
import tempfile
import librosa
#from sharedFunction.refact_text import nettoyer_transcription


from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
    GenerationConfig
)
from transformers.utils import logging

app = FastAPI()


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.set_verbosity_error()
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Setup du device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32



def setup_audio_backend():
    backends = ["ffmpeg", "sox_io", "soundfile"]
    for backend in backends:
        try:
            if backend == "ffmpeg":
                torchaudio.set_audio_backend("ffmpeg")
            elif backend == "sox_io":
                torchaudio.set_audio_backend("sox_io")
            elif backend == "soundfile":
                torchaudio.set_audio_backend("soundfile")
            print(f"Backend audio configuré : {backend}")
            return backend
        except Exception as e:
            print(f"Backend {backend} non disponible : {e}")
            continue
    print("Aucun backend torchaudio disponible, utilisation de librosa en fallback")
    return "librosa"


audio_backend = setup_audio_backend()


model_path = "./whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_path,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
).to(device)

generation_config = GenerationConfig.from_pretrained(model_path)
generation_config.forced_decoder_ids = None
model.generation_config = generation_config

processor = AutoProcessor.from_pretrained(model_path)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    generate_kwargs={"language": "fr"}
)


def load_audio_with_fallback(file_path):


    if audio_backend != "librosa":
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            return waveform, sample_rate, "torchaudio"
        except Exception as e:
            print(f"Échec torchaudio : {e}")


    try:

        audio, sr = librosa.load(file_path, sr=None, mono=False)


        if audio.ndim == 1:

            waveform = torch.from_numpy(audio).unsqueeze(0)
        else:
            # Stéréo: transposer pour avoir la forme (channels, samples)
            waveform = torch.from_numpy(audio.T if audio.shape[0] > audio.shape[1] else audio)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

        return waveform, sr, "librosa"
    except Exception as e:
        raise Exception(f"Impossible de charger l'audio avec librosa : {e}")


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    temp_audio_path = None
    try:

        allowed_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}
        file_extension = Path(file.filename).suffix.lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Format de fichier non supporté : {file_extension}. "
                       f"Formats acceptés : {', '.join(allowed_extensions)}"
            )


        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            temp_audio_path = Path(tmp_file.name)


        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)


        try:
            waveform, sample_rate, method_used = load_audio_with_fallback(temp_audio_path)
            print(f"Audio chargé avec {method_used} - SR: {sample_rate}, Shape: {waveform.shape}")
        except Exception as audio_error:
            raise HTTPException(
                status_code=400,
                detail=f"Impossible de lire l'audio : {audio_error}"
            )


        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)


        if sample_rate != 16000:
            if audio_backend != "librosa":
                try:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sample_rate,
                        new_freq=16000
                    )
                    waveform = resampler(waveform)
                except:

                    audio_np = waveform.squeeze().numpy()
                    audio_resampled = librosa.resample(audio_np, orig_sr=sample_rate, target_sr=16000)
                    waveform = torch.from_numpy(audio_resampled).unsqueeze(0)
            else:

                audio_np = waveform.squeeze().numpy()
                audio_resampled = librosa.resample(audio_np, orig_sr=sample_rate, target_sr=16000)
                waveform = torch.from_numpy(audio_resampled).unsqueeze(0)


        audio_input = waveform.squeeze().numpy()
        result = pipe(audio_input, return_timestamps=True)

        # Nettoyage
        #clean_text = nettoyer_transcription(result["text"])

        return JSONResponse({
            "text": result["text"],
            "audio_method": method_used,
            "original_sample_rate": sample_rate
        })

    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Erreur serveur : {str(e)}"}
        )
    finally:

        if temp_audio_path and temp_audio_path.exists():
            temp_audio_path.unlink(missing_ok=True)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "audio_backend": audio_backend,
        "device": device,
        "torch_dtype": str(torch_dtype)
    }



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)