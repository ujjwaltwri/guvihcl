import os
import uuid
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from .auth import validate_api_key
from .models.processor import AudioProcessor
# FIX: Import the correct class name 'VoiceClassifier'
from .models.classifier import VoiceClassifier 
from .models.lid import LanguageDetector 

app = FastAPI()

# Initialize all models
processor = AudioProcessor()
# FIX: Initialize the class with the correct name
classifier = VoiceClassifier() 
lid_detector = LanguageDetector()

class DetectionRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

@app.post("/api/voice-detection", dependencies=[Depends(validate_api_key)])
async def detect_voice(payload: DetectionRequest):
    temp_filename = f"temp_{uuid.uuid4()}.mp3"
    try:
        # 1. Decode & Save Temp File
        audio_io = processor.decode_base64(payload.audioBase64)
        
        # Write to disk for Librosa/LID to read
        with open(temp_filename, "wb") as f:
            f.write(audio_io.getbuffer())
        
        # 2. DETECT LANGUAGE (Facebook MMS)
        detected_lang = lid_detector.detect(temp_filename)
        
        # 3. FORENSIC ANALYSIS (AI vs Human)
        # We pass the decoded audio object or array to the predictor
        # Since our new classifier uses Librosa, passing the filename path 
        # is actually safer/easier if we modify classifier to accept paths,
        # but right now it expects an array. Let's load the array here.
        import librosa
        audio_array, _ = librosa.load(temp_filename, sr=16000)
        
        classification, confidence, explanation = classifier.predict(audio_array)
        
        # 4. Construct "Smart" Explanation
        final_explanation = explanation
        if payload.language.lower() != detected_lang.lower():
            final_explanation += f" (Note: Input labeled as {payload.language}, but AI detected {detected_lang})."

        return {
            "status": "success",
            "language": detected_lang,
            "classification": classification,
            "confidenceScore": confidence,
            "explanation": final_explanation
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
        
    finally:
        # Cleanup temp file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)