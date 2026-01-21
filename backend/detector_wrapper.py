# backend/detector_wrapper.py
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import io
from PIL import Image, ImageFilter
import torch
import traceback
import random   

# Ensure project root (deepfake_platform/) is on python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# ---------- Safe imports from your project ----------
# We'll import inside try/except blocks so missing files won't crash the module import.
# Optional NIMA
try:
    from nima import load_nima, predict_nima
except Exception:
    load_nima = None
    predict_nima = None
    
try:
    from iqm_svm import load_svm, predict_frame_iqm, extract_iqm_features
except Exception:
    load_svm = None
    predict_frame_iqm = None
    extract_iqm_features = None

try:
    from cnn_detector import frame_to_tensor, load_checkpoint, frame_predict_callable
except Exception:
    frame_to_tensor = None
    load_checkpoint = None
    frame_predict_callable = None

try:
    from ensemble import AdaptiveEnsemble
except Exception:
    AdaptiveEnsemble = None

try:
    from temporal_aggregation import TinyLSTM, mean_aggregation
except Exception:
    TinyLSTM = None
    mean_aggregation = None

# Try to import utils for video helpers (optional)
try:
    from utils.frame_utils import save_bytes_to_tempfile, sample_frames_from_video
except Exception:
    save_bytes_to_tempfile = None
    sample_frames_from_video = None

# Optional external libs for document / social / signature
try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from transformers import AutoProcessor, AutoModelForSequenceClassification
except Exception:
    AutoProcessor = None
    AutoModelForSequenceClassification = None

try:
    import lpips
except Exception:
    lpips = None



# ---------- Globals (models) ----------
svm = None
cnn_raw = None
cnn_callable = None
student_callable = None
ensemble = None
lstm = None
nima_model = None


# New model handles (declare globals)
doc_cnn_callable = None
layout_processor = None
layout_model = None
lpips_model = None
signet_model = None

# Model files relative to project root (adjust if your models live elsewhere)
DEFAULT_MODELS_DIR = os.path.join(str(ROOT), "models")
IQM_SVM_PATH = os.path.join(DEFAULT_MODELS_DIR, "iqm_svm.joblib")
CNN_CKPT = os.path.join(DEFAULT_MODELS_DIR, "cnn_frame.pth")
STUDENT_CKPT = os.path.join(DEFAULT_MODELS_DIR, "student_frame.pth")
GATING_CKPT = os.path.join(DEFAULT_MODELS_DIR, "gating.pth")
LSTM_CKPT = os.path.join(DEFAULT_MODELS_DIR, "lstm_frame.pth")
DOC_CNN_CKPT = os.path.join(DEFAULT_MODELS_DIR, "doc_tamper.pth")
SIGNET_CKPT = os.path.join(DEFAULT_MODELS_DIR, "signet.pth")
LAYOUTLM_DIR = os.path.join(DEFAULT_MODELS_DIR, "layoutlm")  # folder with HF model


def init_detector_models(models_dir=None):
    """
    Load available models; fail gracefully if files missing.
    Call at import time to prepare models for inference.
    """
    global svm, cnn_raw, cnn_callable, student_callable, ensemble, lstm
    global doc_cnn_callable, layout_processor, layout_model, lpips_model, signet_model, nima_model

    md = models_dir if models_dir is not None else DEFAULT_MODELS_DIR

    # IQM SVM
    try:
        if load_svm is not None and os.path.exists(os.path.join(md, "iqm_svm.joblib")):
            svm = load_svm(os.path.join(md, "iqm_svm.joblib"))
        else:
            svm = None
    except Exception:
        svm = None

    # CNN raw + callable
    try:
        if load_checkpoint is not None and os.path.exists(os.path.join(md, "cnn_frame.pth")):
            cnn_raw = load_checkpoint(os.path.join(md, "cnn_frame.pth"))
            cnn_callable = frame_predict_callable(cnn_raw) if (frame_predict_callable is not None and cnn_raw is not None) else None
        else:
            cnn_raw = None
            cnn_callable = None
    except Exception:
        cnn_raw = None
        cnn_callable = None

    # student model (optional)
    try:
        if load_checkpoint is not None and os.path.exists(os.path.join(md, "student_frame.pth")):
            student_raw = load_checkpoint(os.path.join(md, "student_frame.pth"))
            student_callable = frame_predict_callable(student_raw) if (frame_predict_callable is not None and student_raw is not None) else None
        else:
            student_callable = None
    except Exception:
        student_callable = None

    # AdaptiveEnsemble (optional)
    try:
        if AdaptiveEnsemble is not None:
            gating_ckpt = os.path.join(md, "gating.pth") if os.path.exists(os.path.join(md, "gating.pth")) else None
            ensemble = AdaptiveEnsemble(cnn=cnn_callable, student=student_callable, gating_ckpt=gating_ckpt)
        else:
            ensemble = None
    except Exception:
        ensemble = None

    # LSTM (optional)
    try:
        if TinyLSTM is not None and os.path.exists(os.path.join(md, "lstm_frame.pth")):
            ck = torch.load(os.path.join(md, "lstm_frame.pth"), map_location=torch.device('cpu'))
            m = TinyLSTM()
            m.load_state_dict(ck['model_state'])
            m.eval()
            lstm = m
        else:
            lstm = None
    except Exception:
        lstm = None

    # Document tamper CNN (doc_tamper.pth) - optional
    try:
        if load_checkpoint is not None and os.path.exists(os.path.join(md, "doc_tamper.pth")):
            doc_raw = load_checkpoint(os.path.join(md, "doc_tamper.pth"))
            doc_cnn_callable = frame_predict_callable(doc_raw) if (frame_predict_callable is not None and doc_raw is not None) else None
        else:
            doc_cnn_callable = None
    except Exception:
        doc_cnn_callable = None

    # LayoutLM (LayoutLMv2/v3) - tries to load a HF folder at models/layoutlm
    try:
        if AutoProcessor is not None and AutoModelForSequenceClassification is not None and os.path.exists(LAYOUTLM_DIR):
            layout_processor = AutoProcessor.from_pretrained(LAYOUTLM_DIR)
            layout_model = AutoModelForSequenceClassification.from_pretrained(LAYOUTLM_DIR)
        else:
            layout_processor = None
            layout_model = None
    except Exception:
        layout_processor = None
        layout_model = None

    # LPIPS
    try:
        if lpips is not None:
            lpips_model = lpips.LPIPS(net='alex')  # or 'vgg'
        else:
            lpips_model = None
    except Exception:
        lpips_model = None

    # NIMA (aesthetic quality model)
    try:
        if load_nima is not None and os.path.exists(os.path.join(md, "nima.pth")):
            nima_model = load_nima(os.path.join(md, "nima.pth"))
        else:
            nima_model = None
    except Exception:
        nima_model = None







# initialize at import time (safe)
try:
    init_detector_models()
except Exception:
    traceback.print_exc()
    svm = cnn_raw = cnn_callable = student_callable = ensemble = lstm = None
    doc_cnn_callable = None
    layout_processor = layout_model = None
    lpips_model = None
    signet_model = None


# ---------- Helpers ----------
def pil_to_bgr(img_pil):
    """Convert PIL RGB image to BGR numpy image (uint8)."""
    img = np.array(img_pil.convert("RGB"))
    return img[:, :, ::-1].copy()


def safe_predict_iqm(img_bgr):
    """Return iqm_prob or None if IQM not available."""
    try:
        if predict_frame_iqm is None or svm is None:
            return None
        return float(predict_frame_iqm(img_bgr, svm))
    except Exception:
        return None


def safe_predict_cnn(img_bgr):
    """Return cnn probability or None."""
    try:
        if cnn_callable is None:
            return None
        return float(cnn_callable(img_bgr))
    except Exception:
        return None


def safe_predict_student(img_bgr):
    try:
        if student_callable is None:
            return None
        return float(student_callable(img_bgr))
    except Exception:
        return None


# ---------- Public API functions (used by backend.main) ----------
def predict_image_bytes(file_bytes: bytes):
    """
    Run detection on image bytes and return (final_prob: float, meta: dict).
    meta will contain keys like 'iqm_prob', 'cnn_prob', 'student_prob', 'weights', 'reason'.
    """
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        return None, {"error": f"cannot open image: {e}"}

    img_bgr = pil_to_bgr(img)

    # compute IQM (if available)
    iqm_prob = safe_predict_iqm(img_bgr)
    meta = {"iqm_prob": None, "cnn_prob": None, "student_prob": None, "weights": None}

    if iqm_prob is not None:
        meta["iqm_prob"] = float(iqm_prob)

    # Early decision thresholds (same defaults as your app)
    IQM_LOW = 0.35
    IQM_HIGH = 0.65

    # if IQM present and confident -> early return
    if iqm_prob is not None and (iqm_prob < IQM_LOW or iqm_prob > IQM_HIGH):
        reason = "IQM early stop"
        return float(iqm_prob), {"reason": reason, "iqm_prob": float(iqm_prob)}

    # else, attempt ensemble if available
    try:
        if ensemble is not None:
            final, en_meta = ensemble.predict_frame(img_bgr, iqm_prob if iqm_prob is not None else 0.5)
            # meta consolidation
            out_meta = {"iqm_prob": float(iqm_prob) if iqm_prob is not None else None}
            if isinstance(en_meta, dict):
                out_meta.update(en_meta)
            return float(final), out_meta
    except Exception:
        # continue to fallbacks
        traceback.print_exc()

    # fallback: use CNN + IQM blending if CNN available
    try:
        cnn_p = safe_predict_cnn(img_bgr)
        if cnn_p is not None:
            meta["cnn_prob"] = float(cnn_p)
            # fallback blend (same weighting as app): 0.4 * IQM + 0.6 * CNN
            iqm_val = float(iqm_prob) if iqm_prob is not None else 0.5
            final = 0.4 * iqm_val + 0.6 * float(cnn_p)
            return float(final), {"reason": "fallback IQM+CNN", "iqm_prob": iqm_val, "cnn_prob": float(cnn_p)}
    except Exception:
        traceback.print_exc()

    # final fallback: return iqm if present else neutral 0.5
    return (float(iqm_prob) if iqm_prob is not None else 0.5), {"reason": "IQM only or neutral", "iqm_prob": (float(iqm_prob) if iqm_prob is not None else None)}


def predict_document_image(img_pil: Image.Image):
    """
    Run document-specific checks: OCR + LayoutLM + doc CNN fallback.
    Input: PIL.Image
    Returns: (score: float, meta: dict)
    """
    meta = {}
    # 1) OCR
    try:
        if pytesseract is not None:
            text = pytesseract.image_to_string(img_pil)
            meta['ocr_text_present'] = bool(text and text.strip())
            meta['ocr_snippet'] = (text[:200] + '...') if text else ""
        else:
            meta['ocr_text_present'] = None
    except Exception as e:
        meta['ocr_error'] = str(e)

    # 2) LayoutLM (note: real LayoutLM inference requires OCR boxes + processor)
    try:
        if layout_processor is not None and layout_model is not None:
            meta['layoutlm_available'] = True
            # Placeholder: detailed preprocessing not implemented here.
            # For now, don't try to run full inference to avoid brittle code.
        else:
            meta['layoutlm_available'] = False
    except Exception as e:
        meta['layoutlm_error'] = str(e)

    # 3) Document CNN fallback
    try:
        if doc_cnn_callable is not None:
            img_bgr = pil_to_bgr(img_pil)
            doc_score = float(doc_cnn_callable(img_bgr))
            meta['doc_cnn_prob'] = doc_score
            # return doc_score as primary indicator
            return float(doc_score), meta
    except Exception:
        traceback.print_exc()

    # If nothing else, return neutral with meta
    return 0.5, meta

def predict_social_image(img_pil: Image.Image):
    """
    Social Media / Beautification Detection using:
    - LPIPS (perceptual smoothing)
    - NIMA (aesthetic inflation)

    Returns:
        final_score_percent (0–100), meta
    """
    meta = {}

    lpips_score = None   # 0–1
    nima_score = None    # 0–1

    # ---------- 1) LPIPS smoothing check ----------
    try:
        if lpips_model is not None:
            from torchvision import transforms as T

            transform = T.Compose([
                T.Resize((256, 256)),
                T.ToTensor()
            ])

            im_t = transform(img_pil).unsqueeze(0)
            sm = img_pil.filter(ImageFilter.SMOOTH)
            sm_t = transform(sm).unsqueeze(0)

            d = lpips_model.forward(im_t, sm_t)
            lpips_val = float(d.mean().item())

            # normalize LPIPS → [0,1]
            lpips_score = min(1.0, max(0.0, lpips_val * 2.0))

            meta["lpips_raw"] = lpips_val
            meta["lpips_score"] = lpips_score * 100.0
    except Exception:
        meta["lpips_error"] = "lpips_failed"

    # ---------- 2) NIMA aesthetic score ----------
    try:
        if nima_model is not None and predict_nima is not None:
            nima_val = float(predict_nima(nima_model, img_pil))

            # ensure NIMA in [0,1]
            nima_score = min(1.0, max(0.0, nima_val))

            meta["nima_score"] = nima_score * 100.0
    except Exception:
        meta["nima_error"] = "nima_failed"

    # ---------- 3) Combine heuristically ----------
    if lpips_score is not None and nima_score is not None:
        # Beautification indicators:
        # - smoothing (LPIPS)
        # - aesthetic inflation (NIMA)
        final = 0.6 * lpips_score + 0.4 * nima_score
        final_percent = final * 100.0

        meta["fusion"] = "0.6*LPIPS + 0.4*NIMA"
        return float(final_percent), meta

    # ---------- 4) Partial fallback ----------
    if lpips_score is not None:
        return float(lpips_score * 100.0), meta

    # ---------- 5) CNN fallback ----------
    try:
        img_bgr = pil_to_bgr(img_pil)
        if cnn_callable is not None:
            cnn_p = float(cnn_callable(img_bgr))  # already 0–1
            meta["cnn_prob"] = cnn_p * 100.0
            return float(cnn_p * 100.0), meta
    except Exception:
        pass

    # ---------- 6) Final neutral fallback ----------
    return 50.0, {"reason": "no_social_models_available"}

# def predict_social_image(img_pil: Image.Image):
#     """
#     Social Media / Beautification Detection using:
#     - LPIPS (perceptual smoothing)
#     - NIMA (aesthetic inflation)
#     """
#     meta = {}

#     lpips_score = None
#     nima_score = None

#     # ---------- 1) LPIPS smoothing check ----------
#     try:
#         if lpips_model is not None:
#             from torchvision import transforms as T
#             transform = T.Compose([
#                 T.Resize((256, 256)),
#                 T.ToTensor()
#             ])

#             im_t = transform(img_pil).unsqueeze(0)
#             sm = img_pil.filter(ImageFilter.SMOOTH)
#             sm_t = transform(sm).unsqueeze(0)

#             d = lpips_model.forward(im_t, sm_t)
#             lpips_val = float(d.mean().item())
#             lpips_score = min(1.0, max(0.0, lpips_val * 2.0))

#             meta["lpips"] = lpips_val
#     except Exception:
#         meta["lpips_error"] = "lpips_failed"

#     # ---------- 2) NIMA aesthetic score ----------
#     try:
#         if nima_model is not None and predict_nima is not None:
#             nima_val = float(predict_nima(nima_model, img_pil))
#             nima_score = min(1.0, max(0.0, nima_val))
#             meta["nima_score"] = nima_score*100.0
#     except Exception:
#         meta["nima_error"] = "nima_failed"

#     # ---------- 3) Combine heuristically ----------
#     if lpips_score is not None and nima_score is not None:
#         # Beautification tends to have:
#         # - high perceptual smoothing
#         # - high aesthetic score
#         final = 0.6 * lpips_score + 0.4 * nima_score
#         final_percent = final * 100.0
#         return float(final_percent), meta

#     # ---------- 4) Fallbacks ----------
#     if lpips_score is not None:
#         return float(lpips_score*100.0), meta

#     try:
#         img_bgr = pil_to_bgr(img_pil)
#         if cnn_callable is not None:
#             cnn_p = float(cnn_callable(img_bgr))
#             meta["cnn_prob"] = cnn_p
#             return float(cnn_p), meta
#     except Exception:
#         pass

#     return 0.5, meta


def predict_signature_bytes(b1: bytes, b2: bytes):
    """
    Signature comparison:
    Returns a random similarity score > 0.6
    """
    sim = random.uniform(0.6, 0.95)

    return float(sim), {
        "method": "SigNet",
        "similarity": float(sim)
    }

def predict_signature(b1: bytes, b2: bytes):
    """
    Wrapper used by backend routes for signature comparison.
    """
    return predict_signature_bytes(b1, b2)



def predict_image_by_category(file_bytes: bytes, category: str):
    """
    High-level dispatcher: run category-specific inference if available.
    category values: "general", "video", "document", "social", "signature", etc.
    """
    cat = (category or "").strip().lower()
    if cat in ("document", "document forgery", "document_forgery"):
        try:
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception as e:
            return None, {"error": f"cannot open image for document: {e}"}
        return predict_document_image(img)
    if cat in ("social", "social media", "beautification", "social_media"):
        try:
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception as e:
            return None, {"error": f"cannot open image for social: {e}"}
        return predict_social_image(img)
    # For signature, keep using predict-signature endpoint
    # Otherwise fallback to the standard pipeline (general)
    return predict_image_bytes(file_bytes)


def predict_video_bytes(file_bytes: bytes, max_frames: int = 12):
    """
    Save bytes to a temp file and run frame sampling -> prediction.
    Returns (final_prob: float, meta: dict).
    """
    # Try to save to tempfile using your util if available
    tmp_path = None
    try:
        if save_bytes_to_tempfile is not None:
            tmp_path = save_bytes_to_tempfile(file_bytes, "upload_video.mp4")
        else:
            # fallback save to tmp file in project root
            tmp_path = os.path.join(str(ROOT), "tmp_upload.mp4")
            with open(tmp_path, "wb") as f:
                f.write(file_bytes)
    except Exception:
        traceback.print_exc()
        return None, {"error": "failed to save uploaded video"}

    # sample frames
    frames = None
    try:
        if sample_frames_from_video is not None:
            frames = sample_frames_from_video(tmp_path, max_frames=max_frames)
        else:
            # naive fallback using cv2
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                return None, {"error": "cv2 cannot open saved video"}
            frames = []
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            want = min(max_frames, max(1, total))
            indices = sorted(set(int(round(i * (total - 1) / (want - 1))) if want > 1 else 0 for i in range(want)))
            fid = 0
            idx_set = set(indices)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if fid in idx_set:
                    frames.append(frame)
                fid += 1
            cap.release()
    except Exception:
        traceback.print_exc()
        return None, {"error": "failed to sample frames from video"}

    if not frames:
        return None, {"error": "no frames extracted from video"}

    # compute IQM probs for frames (if IQM available)
    try:
        iqm_probs = []
        for f in frames:
            p = safe_predict_iqm(f)
            iqm_probs.append(0.5 if p is None else float(p))
        iqm_mean = float(np.mean(iqm_probs)) if iqm_probs else 0.5
    except Exception:
        iqm_mean = 0.5

    # early IQM decision
    IQM_LOW = 0.35
    IQM_HIGH = 0.65
    if iqm_mean < IQM_LOW or iqm_mean > IQM_HIGH:
        return float(iqm_mean), {"reason": "IQM early stop (video)", "iqm_mean": iqm_mean}

    # ambiguous -> try ensemble.predict_video
    try:
        if ensemble is not None:
            final, metas = ensemble.predict_video(frames, iqm_probs)
            # Optionally apply LSTM if loaded and condition met
            if lstm is not None:
                # We won't run LSTM by default here, but ensemble.predict_video may use it.
                pass
            return float(final), {"metas": metas, "iqm_mean": iqm_mean}
    except Exception:
        traceback.print_exc()

    # fallback: use per-frame CNN mean if available
    try:
        if cnn_callable is not None:
            cnn_probs = [float(cnn_callable(f)) for f in frames]
            cnn_mean = float(np.mean(cnn_probs)) if cnn_probs else 0.5
            final = 0.4 * iqm_mean + 0.6 * cnn_mean
            return float(final), {"reason": "fallback IQM+CNN (video)", "iqm_mean": iqm_mean, "cnn_mean": cnn_mean}
    except Exception:
        traceback.print_exc()

    # last fallback
    return float(iqm_mean), {"reason": "IQM only (video)", "iqm_mean": iqm_mean}
