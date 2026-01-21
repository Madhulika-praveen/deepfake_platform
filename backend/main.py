from dotenv import load_dotenv
from pathlib import Path
import os
from typing import Optional
import random
import datetime
import traceback

# ---- Load .env explicitly ----
ROOT = Path(__file__).resolve().parents[1]
BN_ENV = ROOT / "backend" / ".env"
PR_ENV = ROOT / ".env"
if BN_ENV.exists():
    load_dotenv(dotenv_path=str(BN_ENV))
else:
    load_dotenv(dotenv_path=str(PR_ENV))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from .db import init_db, SessionLocal, User
from .auth_utils import create_access_token, verify_token, generate_otp
from .mailer import send_email_otp
from .detector_wrapper import (
    predict_image_bytes,
    predict_video_bytes,
    predict_image_by_category
)

# ---------- INIT ----------
init_db()
app = FastAPI(title="Deepfake Detector Backend (with 2FA)")

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- MODELS ----------
class SendOtpRequest(BaseModel):
    email: EmailStr

class VerifyOtpRequest(BaseModel):
    email: EmailStr
    otp: str

# ---------- AUTH ----------
def get_current_user(token: str):
    payload = verify_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    email = payload.get("sub")
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    finally:
        db.close()

# ---------- OTP ----------
@app.post("/send-otp")
def send_otp(req: SendOtpRequest):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == req.email).first()
        if user is None:
            user = User(email=req.email, is_verified=False)
            db.add(user)
            db.commit()
            db.refresh(user)

        otp = generate_otp(6)
        user.otp_code = otp
        user.otp_expiry = datetime.datetime.utcnow() + datetime.timedelta(minutes=10)
        db.commit()

        send_email_otp(user.email, otp)
        return {"ok": True, "message": "OTP sent"}
    finally:
        db.close()

@app.post("/verify-otp")
def verify_otp(req: VerifyOtpRequest):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == req.email).first()
        if not user or user.otp_code != req.otp:
            raise HTTPException(status_code=400, detail="Invalid OTP")

        if datetime.datetime.utcnow() > user.otp_expiry:
            raise HTTPException(status_code=400, detail="OTP expired")

        user.is_verified = True
        user.otp_code = None
        user.otp_expiry = None
        db.commit()

        token = create_access_token(subject=user.email)
        return {"access_token": token, "token_type": "bearer"}
    finally:
        db.close()

# ---------- IMAGE ----------
@app.post("/predict-image")
async def predict_image(
    file: UploadFile = File(...),
    authorization: str = Form(...),
    category: Optional[str] = Form(None),
):
    token = authorization.replace("Bearer ", "")
    get_current_user(token)
    data = await file.read()

    if category:
        final, meta = predict_image_by_category(data, category)
    else:
        final, meta = predict_image_bytes(data)

    return {"final_prob": final, "meta": meta}

# ---------- VIDEO ----------
@app.post("/predict-video")
async def predict_video(
    file: UploadFile = File(...),
    authorization: str = Form(...),
):
    token = authorization.replace("Bearer ", "")
    get_current_user(token)
    data = await file.read()

    final, meta = predict_video_bytes(data)
    return {"final_prob": final, "meta": meta}

# ---------- CATEGORIES ----------
@app.get("/categories")
def categories():
    return {
        "General Image Forgery Detection": "IQM + CNN + Student + Gating",
        "Face Deepfake Detection": "CNN-based face artifact detection",
        "GAN / AI-Generated Images": "CNN + IQM",
        "Video Deepfake Detection": "Frame-level + temporal aggregation",
        "Document Forgery Detection": "OCR + Layout analysis",
        "Social Media Edit Detection": "LPIPS + NIMA",
        "Signature Verification": "Random similarity (stub)"
    }

# ---------- SIGNATURE (FINAL, NO MODELS) ----------
@app.post("/predict-signature")
async def predict_signature(
    sig1: UploadFile = File(...),
    sig2: UploadFile = File(...),
    authorization: str = Form(...),
):
    token = authorization.replace("Bearer ", "")
    get_current_user(token)

    similarity = random.uniform(0.6, 0.95)

    return {
        "final_prob": similarity,
        "meta": {
            "method": "Signature Stub",
            "similarity": similarity
        }
    }

# ---------- RUN ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
