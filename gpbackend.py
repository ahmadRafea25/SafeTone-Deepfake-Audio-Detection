from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
import tempfile
import os
from pydantic import BaseModel
from tensorflow.keras.models import load_model


class ReportSchema(BaseModel):
    authenticityScore: float
    authenticityLabel: str

# ===============================
# 1️⃣ Load model ONCE
# ===============================
MODEL_PATH = "fake_audio_detector_cnn.keras"
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# ===============================
# 2️⃣ FastAPI setup
# ===============================
app = FastAPI()

# ✅ CORS (FIXED – allow frontend to access backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ✅ allow React (localhost:5173)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Backend is running with ML model"}

# ===============================
# 3️⃣ MFCC preprocessing (EXACT match with training)
# ===============================
def extract_mfcc_fixed(file_path, n_mfcc=40, fixed_length=216):
    # Load audio at 16 kHz
    y, sr = librosa.load(file_path, sr=16000)

    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # (n_mfcc, time) -> (time, n_mfcc)
    mfcc = mfcc.T

    # Pad or cut to fixed length
    if mfcc.shape[0] < fixed_length:
        mfcc = np.pad(
            mfcc,
            ((0, fixed_length - mfcc.shape[0]), (0, 0)),
            mode="constant",
        )
    else:
        mfcc = mfcc[:fixed_length, :]

    # Add channel dimension -> (216, 40, 1)
    mfcc = mfcc[..., np.newaxis]

    return mfcc
@app.post("/save-report")
async def save_report(report: ReportSchema):
    # Example: save to a text file (simple & valid for graduation project)
    with open("reports.txt", "a") as f:
        f.write(
            f"Label: {report.authenticityLabel}, "
            f"Score: {report.authenticityScore}\n"
        )

    return {"message": "Report saved successfully"}

# ===============================
# 4️⃣ Analyze audio endpoint
# ===============================
@app.post("/analyze-audio")
async def analyze_audio(audio: UploadFile = File(...)):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        temp.write(await audio.read())
        temp_path = temp.name

    try:
        # Preprocess
        mfcc = extract_mfcc_fixed(temp_path)

        # Add batch dimension -> (1, 216, 40, 1)
        mfcc = np.expand_dims(mfcc, axis=0)

        # Predict
        prediction = model.predict(mfcc)[0][0]

        # Convert to label
        authenticity_label = "Fake" if prediction > 0.5 else "Real"

        return {
            "result": {
                "authenticityScore": float(prediction),
                "authenticityLabel": authenticity_label,
                "speakerScore": 0.0,
                "speakerLabel": "N/A"
            }
        }

    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

