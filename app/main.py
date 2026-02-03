import os
import uuid
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from .auth import validate_api_key
from .models.processor import AudioProcessor
from .models.classifier import VoiceClassifier
from .models.lid import LanguageDetector # <--- Import new module

app = FastAPI()

# Initialize all models
processor = AudioProcessor()
classifier = VoiceClassifier()
lid_detector = LanguageDetector() # <--- Initialize LID

class DetectionRequest(BaseModel):
    language: str # We receive this, but we will double-check it!
    audioFormat: str
    audioBase64: str

@app.post("/api/voice-detection", dependencies=[Depends(validate_api_key)])
async def detect_voice(payload: DetectionRequest):
    temp_filename = f"temp_{uuid.uuid4()}.mp3"
    try:
        # 1. Decode & Save Temp File (Needed for efficient LID processing)
        audio_io = processor.decode_base64(payload.audioBase64)
        
        with open(temp_filename, "wb") as f:
            f.write(audio_io.getbuffer())
        
        # 2. DETECT LANGUAGE (The AI decides!)
        detected_lang = lid_detector.detect(temp_filename)
        
        # 3. Process & Classify Voice (Human vs AI)
        # Re-load for the classifier (or pass the path if you optimized classifier)
        # Using our existing flow:
        features = processor.extract_features(temp_filename) # Update processor to accept path
        classification, confidence, explanation = classifier.predict(features)
        
        # 4. Construct "Smart" Explanation
        # If the user said "Tamil" but we heard "Hindi", mention it!
        final_explanation = explanation
        if payload.language.lower() != detected_lang.lower():
            final_explanation += f" (Note: Input labeled as {payload.language}, but AI detected {detected_lang})."

        return {
            "status": "success",
            "language": detected_lang,  # We return the REAL language
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