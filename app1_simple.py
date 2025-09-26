# app1_simple.py - Optimized for Render deployment
import pickle
import numpy as np
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
from io import BytesIO
from PIL import Image
import os
import logging

# ---------------- Settings ----------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

app = FastAPI(title="Face Attendance API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_FILE = "face_fast_db.pkl"

# Load face database
try:
    with open(DB_FILE, "rb") as f:
        face_db = pickle.load(f)
        print(f"[+] Loaded {len(face_db)} faces from database")
except FileNotFoundError:
    face_db = {}
    print("[!] Face database not found")

# ---------------- Face Recognition ----------------
def recognize_face_bytes(image_bytes):
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(img)

        # Use DeepFace without preloading model (lighter for Render)
        result = DeepFace.represent(
            img_path=img_array,
            model_name="Facenet",
            detector_backend="opencv",
            enforce_detection=False
        )
        
        if not result:
            return "Unknown", 0.0
            
        embedding = result[0]["embedding"]
    except Exception as e:
        print(f"Face detection error: {e}")
        return "Unknown", 0.0

    min_dist = float("inf")
    identity = "Unknown"
    
    for reg, db_emb in face_db.items():
        try:
            dist = np.linalg.norm(np.array(embedding) - np.array(db_emb))
            if dist < min_dist:
                min_dist = dist
                identity = reg
        except:
            continue

    # Lower threshold for better matching
    threshold = 15  # Increased from 10
    similarity = 1 / (1 + min_dist)
    
    if min_dist > threshold:
        identity = "Unknown"

    return identity, similarity

# ---------------- API Endpoints ----------------
@app.get("/")
async def root():
    return {"message": "Face Attendance API is running!", "status": "healthy"}

@app.get("/health")
async def health():
    return {"status": "healthy", "registered_faces": len(face_db)}

@app.post("/verify-attendance")
async def verify_attendance(registerNumber: str = Form(...), file: UploadFile = None):
    try:
        if not file:
            return JSONResponse({"message": "❌ No image uploaded"}, status_code=400)

        if not registerNumber:
            return JSONResponse({"message": "❌ Register number required"}, status_code=400)

        # Check if register number exists
        if registerNumber not in face_db:
            return JSONResponse({
                "message": f"❌ Register number {registerNumber} not found",
                "status": "ABSENT"
            })

        image_bytes = await file.read()
        identity, similarity = recognize_face_bytes(image_bytes)
        similarity_percent = round(similarity * 100, 2)

        if identity == registerNumber and similarity > 0.2:  # Very low threshold
            return JSONResponse({
                "message": f"✅ Attendance marked for {registerNumber} (Similarity: {similarity_percent}%)",
                "status": "PRESENT",
                "confidence": similarity
            })
        else:
            return JSONResponse({
                "message": f"❌ Face does not match {registerNumber} (Similarity: {similarity_percent}%)",
                "status": "ABSENT",
                "confidence": similarity
            })
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse({"message": f"❌ Server error: {str(e)}"}, status_code=500)

# ---------------- Run Server ----------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app1_simple:app", host="0.0.0.0", port=port, workers=1)
