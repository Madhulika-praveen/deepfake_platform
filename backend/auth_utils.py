# backend/auth_utils.py
import os
import time
import datetime
import random
from passlib.context import CryptContext
from jose import jwt

PWD_CTX = CryptContext(schemes=["bcrypt"], deprecated="auto")

JWT_SECRET = os.environ.get("JWT_SECRET", "change-me")
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_SECONDS = int(os.environ.get("JWT_EXPIRE_SECONDS", "3600"))

def create_access_token(subject: str, expires_delta: int = None):
    now = int(time.time())
    expire = now + (expires_delta if expires_delta is not None else JWT_EXPIRE_SECONDS)
    payload = {"sub": subject, "iat": now, "exp": expire}
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token

def verify_token(token: str):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except Exception:
        return None

def hash_password(password: str):
    return PWD_CTX.hash(password)

def verify_password(password: str, hashed: str):
    return PWD_CTX.verify(password, hashed)

def generate_otp(n_digits: int = 6):
    """Return numeric OTP as string (zero-padded)."""
    low = 10**(n_digits-1)
    high = 10**n_digits - 1
    return str(random.randint(low, high))
