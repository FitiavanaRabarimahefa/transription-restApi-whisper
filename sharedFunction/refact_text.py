import re


def nettoyer_transcription(transcription: str) -> str:
    # Remplacer les ponctuations verbalisées par de vraies ponctuations
    transcription = re.sub(r"\bpoint\s*,?", ".", transcription, flags=re.IGNORECASE)
    transcription = re.sub(r"\bvirgule\s*,?", ",", transcription, flags=re.IGNORECASE)
    transcription = re.sub(r"\bpoint d'interrogation\s*,?", "?", transcription, flags=re.IGNORECASE)
    transcription = re.sub(r"\bpoint-virgule\s*,?", ";", transcription, flags=re.IGNORECASE)
    transcription = re.sub(r"\bdouble point\s*,?", ":", transcription, flags=re.IGNORECASE)

    # Supprimer les majuscules en début de phrase après ponctuation, sauf après point réel
    transcription = re.sub(r"(?<=[,;:])\s*([A-Z])", lambda m: m.group(1).lower(), transcription)

    # Nettoyage des espaces inutiles autour de la ponctuation
    transcription = re.sub(r"\s*([.,;:?!])\s*", r"\1 ", transcription)

    # Supprimer les espaces en trop
    transcription = re.sub(r"\s+", " ", transcription).strip()

    # Mettre la première lettre en majuscule (optionnel)
    transcription = transcription[0].upper() + transcription[1:] if transcription else transcription

    return transcription
