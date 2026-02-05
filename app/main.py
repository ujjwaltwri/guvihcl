import os
import uuid
import librosa
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from .auth import validate_api_key
from .models.processor import AudioProcessor
from .models.classifier import VoiceClassifier  # This should now use voice_classifier_clean.py
from .models.lid import LanguageDetector

app = FastAPI()

# --- Initialize Models (Global Load) ---
print("🚀 Initializing System Models...")
processor = AudioProcessor()
classifier = VoiceClassifier()  # Using the improved classifier
lid_detector = LanguageDetector()
print("✅ System Ready.")

# --- Request Schema ---
class DetectionRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str
    return_details: bool = False  # NEW: Optional detailed output

# --- API Endpoint ---
@app.post("/api/voice-detection", dependencies=[Depends(validate_api_key)])
async def detect_voice(payload: DetectionRequest):
    """
    Detect if a voice is AI-generated or human.
    
    Returns detailed analysis including:
    - classification: AI_GENERATED or HUMAN
    - confidenceScore: 0.0 to 1.0
    - explanation: Human-readable explanation
    - language: Detected language
    - method: Detection method used
    """
    # Create a unique temp file for this request
    temp_filename = f"temp_{uuid.uuid4()}.{payload.audioFormat}"
    
    try:
        # 1. Decode Base64 & Save to Disk
        audio_io = processor.decode_base64(payload.audioBase64)
        with open(temp_filename, "wb") as f:
            f.write(audio_io.getbuffer())
        
        # 2. Load Audio Data (Optimized: Load once for everyone)
        # Load as 16kHz mono, which is the standard for most AI models
        audio_array, _ = librosa.load(temp_filename, sr=16000, mono=True)
        
        # Validate audio is not silent
        import numpy as np
        rms = np.sqrt(np.mean(audio_array**2))
        if rms < 0.001:
            return {
                "status": "error",
                "message": "Audio appears to be silent or nearly silent"
            }
        
        # 3. Detect Language
        detected_lang = lid_detector.detect(temp_filename)
        
        # 4. Forensic Analysis (AI vs Human) - UPDATED
        # The new classifier returns a dictionary with detailed info
        result = classifier.predict(audio_array, return_details=payload.return_details)
        
        # Extract values from the new classifier format
        classification = result["verdict"]  # "AI_GENERATED" or "HUMAN"
        confidence = result["confidence"]  # 0.0 to 1.0
        explanation = result["explanation"]  # Detailed explanation
        method = result["method"]  # Detection method used
        
        # 5. Construct "Smart" Explanation
        final_explanation = explanation
        
        # Add language mismatch note if detected
        if payload.language.lower() != detected_lang.lower():
            final_explanation += f" (Note: Input labeled as {payload.language}, but detected {detected_lang})."
        
        # 6. Build Response
        response = {
            "status": "success",
            "language": detected_lang,
            "classification": classification,
            "confidenceScore": confidence,
            "explanation": final_explanation,
            "method": method
        }
        
        # 7. Add Optional Detailed Fields
        if payload.return_details:
            # Include all the detailed analysis from the classifier
            if "heuristic_score" in result:
                response["heuristic_score"] = result["heuristic_score"]
            if "heuristic_reason" in result:
                response["heuristic_reason"] = result["heuristic_reason"]
            if "model_confidence" in result:
                response["model_confidence"] = result["model_confidence"]
            if "model_verdict" in result:
                response["model_verdict"] = result["model_verdict"]
            if "segments_analyzed" in result:
                response["segments_analyzed"] = result["segments_analyzed"]
        
        return response
    
    except Exception as e:
        print(f"❌ API Error: {str(e)}")
        import traceback
        traceback.print_exc()  # Print full traceback for debugging
        
        return {
            "status": "error", 
            "message": f"Processing failed: {str(e)}"
        }
    
    finally:
        # 8. Cleanup: Always remove the temp file
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except:
                pass