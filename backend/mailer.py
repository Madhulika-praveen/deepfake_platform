# backend/mailer.py
import os
from email.message import EmailMessage
import smtplib
from dotenv import load_dotenv
from pathlib import Path

# load .env explicitly (prefer backend/.env then project root .env)
ROOT = Path(__file__).resolve().parents[1]
BN_ENV = ROOT / "backend" / ".env"
PR_ENV = ROOT / ".env"
if BN_ENV.exists():
    load_dotenv(dotenv_path=str(BN_ENV))
else:
    load_dotenv(dotenv_path=str(PR_ENV))

def _read_smtp_config():
    host = os.environ.get("SMTP_HOST")
    port = os.environ.get("SMTP_PORT")
    user = os.environ.get("SMTP_USER")
    pwd = os.environ.get("SMTP_PASS")
    # Normalize port
    try:
        port = int(port) if port is not None else None
    except Exception:
        port = None
    return host, port, user, pwd

def send_email_otp(to_email: str, otp_code: str):
    """
    Sends a simple OTP email using SMTP. Raises RuntimeError with a helpful message on failure.
    """
    host, port, user, pwd = _read_smtp_config()
    if not (host and port and user and pwd):
        # do not leak actual values; provide clear actionable message
        raise RuntimeError("SMTP not configured: check SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS environment variables")

    msg = EmailMessage()
    msg["Subject"] = "Your verification code"
    msg["From"] = user
    msg["To"] = to_email
    msg.set_content(f"Your verification code is: {otp_code}\nThis code expires in 10 minutes.\nIf you didn't request this, ignore this message.")

    try:
        # Use a context manager for SMTP
        # If port == 465, use SMTP_SSL; otherwise use STARTTLS
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=20) as s:
                s.login(user, pwd)
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=20) as s:
                s.ehlo()
                try:
                    s.starttls()
                    s.ehlo()
                except Exception:
                    # ignore starttls failure (some servers don't need it)
                    pass
                s.login(user, pwd)
                s.send_message(msg)
    except smtplib.SMTPAuthenticationError as e:
        raise RuntimeError(f"Failed to send OTP: SMTP authentication error: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to send OTP: {e}")
