# frontend/streamlit_app.py
import streamlit as st
import requests
import os
from PIL import Image
import tempfile

BACKEND = os.environ.get("BACKEND_HOST", "http://localhost:8000")

st.set_page_config(page_title="DeepFake - Secure Client", layout="wide")
# No plain st.title() — title rendered inside header block in Home

# ---------- Auth Handling ----------
if "token" not in st.session_state:
    st.session_state.token = None
if "email" not in st.session_state:
    st.session_state.email = None

def send_otp(email):
    try:
        return requests.post(f"{BACKEND}/send-otp", json={"email": email}, timeout=10)
    except Exception:
        return None

def verify_otp(email, otp):
    try:
        return requests.post(f"{BACKEND}/verify-otp", json={"email": email, "otp": otp}, timeout=10)
    except Exception:
        return None

# ---------- Sidebar Login ----------
st.sidebar.header("Login")

if st.session_state.token is None:
    email = st.sidebar.text_input("Email", value=st.session_state.email or "")
    if st.sidebar.button("Send verification code"):
        if not email:
            st.sidebar.error("Enter an email")
        else:
            r = send_otp(email)
            if r and r.status_code == 200:
                st.session_state.email = email
                st.sidebar.success("OTP sent — check inbox.")
            else:
                st.sidebar.error("Failed to send OTP")

    otp = st.sidebar.text_input("Enter verification code")
    if st.sidebar.button("Verify code"):
        r = verify_otp(st.session_state.email, otp)
        if r and r.status_code == 200:
            st.session_state.token = r.json().get("access_token")
            st.sidebar.success("Logged in!")
        else:
            st.sidebar.error("Invalid OTP")
else:
    st.sidebar.success(f"Logged in as {st.session_state.email}")
    if st.sidebar.button("Log out"):
        st.session_state.token = None

st.sidebar.markdown("---")

# ---------- Sidebar Navigation ----------
pages = [
    "Home",
    "General Image Forgery Detection",
    "Video Deepfake Detection",
    "Document Forgery Detection",
    "Social Media / Beautification Edit Detection",
    "Signature Verification"
]

st.sidebar.header("Navigation")
selection = st.sidebar.radio("Go to", pages)

# ---------- Helpers ----------
def load_and_resize(path, size=(360, 230)):
    """
    Load a local image and resize to a uniform size.
    Returns a PIL.Image object or None if loading fails.
    """
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize(size)
        return img
    except Exception:
        return None

def upload_and_post(fileobj, endpoint, category=None, extra_data=None):
    """
    Upload a single file to backend endpoint. Sends 'authorization' and optional 'category' in form fields.
    Returns requests.Response or None on error.
    """
    data = extra_data or {}
    data['authorization'] = f"Bearer {st.session_state.token}"
    if category:
        data['category'] = category
    # fileobj may be a Streamlit UploadedFile or a tuple/dict for signature endpoint
    if hasattr(fileobj, "read") and hasattr(fileobj, "name"):
        files = {"file": (fileobj.name, fileobj.read(), fileobj.type)}
    else:
        # assume fileobj is already a files dict (for signature endpoint)
        files = fileobj
    try:
        r = requests.post(f"{BACKEND}{endpoint}", files=files, data=data, timeout=180)
        return r
    except Exception as e:
        st.error(f"Network error: {e}")
        return None

def explain_meta_in_english(meta: dict, category: str):
    """
    Convert backend meta dictionary into clean, user-friendly English sentences.
    No numbers, no brackets, no technical jargon.
    """
    explanations = []
    cat = (category or "").lower()

    # ---------- General / Face ----------
    if cat in ("general", "general image forgery detection", "face", "face deepfake"):
        explanations.append(
            "The image was examined for visual artifacts, blending errors, and unnatural patterns."
        )
        explanations.append(
            "Multiple deep learning models analyzed the image to improve reliability."
        )

        if meta.get("reason"):
            explanations.append(
                "The system reached a confident decision early based on strong visual indicators."
            )

    # ---------- Video ----------
    elif cat in ("video", "video deepfake detection"):
        explanations.append(
            "The video was broken into frames and each frame was analyzed individually."
        )
        explanations.append(
            "Consistency across frames was checked to detect unnatural motion or identity changes."
        )

        if meta.get("reason"):
            explanations.append(
                "Clear signs of manipulation were detected across multiple frames."
            )

    # ---------- Document ----------
    elif cat in ("document", "document forgery detection"):
        explanations.append(
            "The document was checked for layout inconsistencies and altered regions."
        )
        explanations.append(
            "Text structure and alignment were analyzed to identify possible tampering."
        )

        if meta.get("ocr_text_present"):
            explanations.append(
                "Readable text was successfully detected in the document."
            )
        else:
            explanations.append(
                "The document contained limited or unclear readable text."
            )

    # ---------- Social Media ----------
    elif cat in ("social", "social media", "beautification"):
        explanations.append(
            "The image was analyzed for smoothing, retouching, and beautification effects."
        )
        explanations.append(
            "Visual patterns commonly introduced by social media filters were examined."
        )

    # ---------- Signature ----------
    elif cat in ("signature", "signature verification"):
        explanations.append(
            "Both signatures were analyzed for stroke patterns and writing consistency."
        )
        explanations.append(
            "The similarity between the two signatures was evaluated to assess authenticity."
        )

    # ---------- Fallback ----------
    else:
        explanations.append(
            "The uploaded content was analyzed using available forensic detection models."
        )

    return explanations


def display_server_response(resp):
    """
    Interpret backend response and display a clean metric + verdict.
    Expects JSON like: {"final_prob": <float>, "meta": {...}} but will handle missing fields.
    """
    if resp is None:
        return
    # Non-200 responses: try to show helpful detail
    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        st.error(f"Server error: {detail}")
        return

    try:
        data = resp.json()
    except Exception:
        st.error("Invalid JSON response from server.")
        return

    final = data.get("final_prob")
    meta = data.get("meta", {})

    # If backend returns a numeric final probability, show it and verdict
    if final is not None:
        try:
            val = float(final)
            st.metric("Final Fake Probability", f"{val:.3f}")
            if val <= 0.5:
                st.success("Verdict: Authentic ✅")
            else:
                st.error("Verdict: Fake Detected ❌")
        except Exception:
            st.write("Result:", final)
    else:
        # fallback: show entire meta if present
        if meta:
            st.write("Result details:")
            st.json(meta)
        else:
            st.warning("No usable result returned by server.")

    # collapsed metadata for debugging / detail
    with st.expander("View detailed analysis explanation"):
        explanations = explain_meta_in_english(meta, selection)
        for line in explanations:
            st.write("• " + line)

# ---------- Home Page ----------
if selection == "Home":
    header_html = """
    <div style="
        width:100%;
        background-color:#4a4a4a;
        padding:30px 10px;
        border-radius:8px;
        text-align:center;
        margin-top:15px;
        position:relative;
    ">
        <h1 style="color:white; font-size:48px; font-weight:800; margin:0; line-height:1;">
            DeepGuard – A Multi-Modal Deepfake Detection Platform
        </h1>
    </div>
    <div style="text-align:center; margin-top:18px;">
        <p style="font-size:17px; color:#ffffff; max-width:900px; margin:auto;">
            A multi-model platform for identifying manipulated media — including fake faces,
            AI-generated images, forged documents, video deepfakes, and social-media
            beautification edits — using advanced AI-driven analysis.
        </p>
    </div>
    <hr style="border:0; height:4px; margin-top:25px; background: linear-gradient(90deg, red, orange, yellow, green, cyan, blue, purple); border-radius:4px;">
    """
    st.markdown(header_html, unsafe_allow_html=True)

    st.markdown("### Our Categories")
    st.markdown("Below are the detection categories available in this demo (click a category in the sidebar to run detectors).")

    # Local images expected at e:/deepfake_platform/images/image1.png ... image5.png
    imgs = [
        "e:/deepfake_platform/images/image1.png",
        "e:/deepfake_platform/images/image2.png",
        "e:/deepfake_platform/images/image3.png",
        "e:/deepfake_platform/images/image4.png",
        "e:/deepfake_platform/images/image5.png",
    ]
    titles = [
        "General Image Forgery Detection",
        "Video Deepfake Detection",
        "Document Forgery Detection",
        "Social Media / Beautification Edit Detection",
        "Signature Verification"
    ]
    descs = [
        "Detects splices, copy-move edits, and pixel-level tampering.",
        "Frame-level and temporal analysis for manipulated videos.",
        "OCR + layout-based detection for fake/edited documents.",
        "Detects retouching, smoothing, and beautification filters.",
        "Compares handwritten signatures to detect forgery."
    ]

    # Grid 2x3 layout: first row 3, second row 2 left cells (third empty)
    row1 = st.columns(3)
    for i in range(3):
        with row1[i]:
            im = load_and_resize(imgs[i])
            if im is not None:
                # use_container_width to avoid deprecation warning
                st.image(im, use_container_width=True)
            else:
                st.image(None, caption="No image available")
            st.markdown(f"**{titles[i]}**")
            st.write(descs[i])

    row2 = st.columns(3)
    for j in range(2):
        idx = 3 + j
        with row2[j]:
            im = load_and_resize(imgs[idx])
            if im is not None:
                st.image(im, use_container_width=True)
            else:
                st.image(None, caption="No image available")
            st.markdown(f"**{titles[idx]}**")
            st.write(descs[idx])
    with row2[2]:
        st.write("")  # empty cell for symmetry

    # Project contributors sentence (black text)
    st.markdown(
        "<p style='text-align:center; color:white; margin-top:30px; font-size:16px;'>By - Madhulika Praveen, Lahari BK, Pooja BL</p>",
        unsafe_allow_html=True
    )

# ---------- Pages: detection UIs ----------
elif selection == "General Image Forgery Detection":
    st.header("General Image Forgery Detection")
    st.markdown("Upload an image — the platform will run IQM and (if needed) CNN/Student/Ensemble.")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        if not st.session_state.token:
            st.info("Please login in the sidebar to use the detector.")
        else:
            with st.spinner("Analyzing image — running IQM and model ensemble if needed..."):
                resp = upload_and_post(uploaded, "/predict-image", category="general")
            display_server_response(resp)

elif selection == "Video Deepfake Detection":
    st.header("Video Deepfake Detection")
    st.markdown("Upload a short video. Server samples frames and aggregates per-frame decisions.")
    uploaded = st.file_uploader("Upload video (mp4)", type=["mp4"])
    if uploaded:
        if not st.session_state.token:
            st.info("Please login in the sidebar to use the detector.")
        else:
            with st.spinner("Analyzing video — sampling frames and checking for temporal anomalies..."):
                resp = upload_and_post(uploaded, "/predict-video", category="video")
            display_server_response(resp)

elif selection == "Document Forgery Detection":
    st.header("Document Forgery Detection")
    st.markdown("Upload a document image or PDF. Server will run OCR + layout checks and document tamper CNN.")
    uploaded = st.file_uploader("Upload document image / PDF", type=["png", "jpg", "jpeg", "pdf"])
    if uploaded:
        if not st.session_state.token:
            st.info("Please login in the sidebar to use the detector.")
        else:
            with st.spinner("Checking document — running OCR and layout consistency checks..."):
                resp = upload_and_post(uploaded, "/predict-image", category="document")
            display_server_response(resp)

elif selection == "Social Media / Beautification Edit Detection":
    st.header("Social Media / Beautification Edit Detection")
    st.markdown("Upload a single image (the posted/filtered image). The backend will run LPIPS / ViT / NIMA heuristics if available.")
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if uploaded:
        if not st.session_state.token:
            st.info("Please login in the sidebar to use the detector.")
        else:
            with st.spinner("Analyzing image — checking for smoothing, retouching and filter artifacts..."):
                resp = upload_and_post(uploaded, "/predict-image", category="social")
            display_server_response(resp)

elif selection == "Signature Verification":
    st.header("Signature Verification")
    st.markdown("Upload two signature images to compare (SigNet).")
    s1 = st.file_uploader("Signature 1", key="sig1", type=["jpg", "jpeg", "png"])
    s2 = st.file_uploader("Signature 2", key="sig2", type=["jpg", "jpeg", "png"])
    if s1 and s2:
        if not st.session_state.token:
            st.info("Please login in the sidebar to use the detector.")
        else:
            with st.spinner("Computing signature similarity..."):
                files = {
                    "sig1": (s1.name, s1.read(), s1.type),
                    "sig2": (s2.name, s2.read(), s2.type)
                }
                data = {"authorization": f"Bearer {st.session_state.token}"}
                try:
                    resp = requests.post(f"{BACKEND}/predict-signature", files=files, data=data, timeout=60)
                    display_server_response(resp)
                except Exception as e:
                    st.error(f"Network error: {e}")
