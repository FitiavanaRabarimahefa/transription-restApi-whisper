# auth.py
from pydantic import BaseModel
from fastapi import HTTPException
from google.oauth2 import id_token
from google.auth.transport import requests
import jwt
import os

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
JWT_SECRET = os.getenv("JWT_SECRET")

class TokenPayload(BaseModel):
    token: str

def google_auth(payload: TokenPayload):
    try:
        # Vérifie le token Google
        idinfo = id_token.verify_oauth2_token(payload.token, requests.Request(), GOOGLE_CLIENT_ID)
        print("Token valide :", idinfo)

        # idinfo contient email
        user_email = idinfo["email"]

        #JWT
        custom_jwt = jwt.encode(
            {"email": user_email, "name": idinfo.get("name")},
            JWT_SECRET,
            algorithm="HS256"
        )

        return {"token": custom_jwt}

    except ValueError:
        raise HTTPException(status_code=400, detail="Token Google invalide")
