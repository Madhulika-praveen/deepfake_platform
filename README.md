# Deepfake Detection Platform

An end-to-end deepfake detection system built using a multi-stage
ensemble architecture with a FastAPI backend and a Streamlit frontend.

This repository contains inference-ready code and pretrained models.
Training scripts, datasets, and experimental artifacts are intentionally
excluded to keep the repository clean and reproducible.

-----------------------------------------------------------------------

SYSTEM OVERVIEW

The detection pipeline follows a confidence-based strategy:

1. IQM + SVM performs a fast early decision.
2. If confidence is low, frames are passed to:
   - CNN detector
   - Student CNN (distilled model)
3. Outputs are combined using a gating network.
4. Temporal aggregation is applied for video-level prediction.

-----------------------------------------------------------------------

PROJECT STRUCTURE

deepfake_platform/
├── backend/              FastAPI backend (API + inference logic)
├── frontend/             Streamlit UI
│   └── streamlit_app.py
├── models/               Pretrained model weights
├── images/               Frontend UI assets
├── utils/                Helper utilities
├── tools/                Utility scripts
├── README.md
├── requirements.txt
└── .gitignore

-----------------------------------------------------------------------

ENVIRONMENT SETUP

STEP 1: Clone the repository from GitHub

git clone https://github.com/<your-username>/deepfake_platform.git
cd deepfake_platform

-----------------------------------------------------------------------

STEP 2: Create and activate a Python environment (recommended)

Using conda:

conda create -n deepfake_env python=3.10
conda activate deepfake_env

Alternatively, you may use any Python 3.10+ environment.

-----------------------------------------------------------------------

STEP 3: Install dependencies

pip install -r requirements.txt

-----------------------------------------------------------------------

BACKEND ENVIRONMENT CONFIGURATION (.env)

The backend requires environment variables for authentication,
email configuration, and frontend communication.

STEP 1: Create a .env file

Navigate to the backend directory:

cd backend

Create a file named:

.env

-----------------------------------------------------------------------

STEP 2: Add the following content to backend/.env

# JWT
JWT_SECRET=change-this-to-a-long-secret
JWT_ALGORITHM=
JWT_EXPIRE_SECONDS=

# SMTP (example using Gmail SMTP or any SMTP provider)
SMTP_HOST=
SMTP_PORT=
SMTP_USER=
SMTP_PASS=

# App settings
BACKEND_HOST=http://localhost:8000

NOTES:
- Fill in values according to your setup
- Do NOT commit the .env file to GitHub
- .env is ignored using .gitignore

-----------------------------------------------------------------------

RUNNING THE APPLICATION

STEP 1: Start the Backend (FastAPI)

From the project root directory:

python -m uvicorn backend.main:app --reload

The backend will run at:

http://127.0.0.1:8000

-----------------------------------------------------------------------

STEP 2: Start the Frontend (Streamlit)

Open a new terminal window, activate the same environment, then:

cd frontend
streamlit run streamlit_app.py

The Streamlit web interface will open automatically in your browser.

-----------------------------------------------------------------------

MODELS USED

- IQM–SVM (early confidence-based filtering)
- CNN frame-level deepfake detector
- Student CNN (knowledge distillation)
- Gating network ensemble
- Temporal aggregation for video-level decision

All pretrained models are loaded from the models/ directory.

-----------------------------------------------------------------------

NOTES

- Training code and datasets are not included in this repository
- Temporary experiment artifacts are excluded via .gitignore
- backend.db is created automatically on first backend startup
- Ensure the backend is running before starting the frontend

-----------------------------------------------------------------------

REPRODUCIBILITY

To reproduce inference on a new machine:

1. Clone the repository
2. Create a Python environment
3. Install dependencies from requirements.txt
4. Configure backend/.env
5. Start backend and frontend

No training is required.

-----------------------------------------------------------------------

LICENSE

For academic and educational use only.
