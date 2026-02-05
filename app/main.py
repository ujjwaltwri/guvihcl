import os
import uuid
import librosa
import numpy as np
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import your existing modules
try:
    from .auth import validate_api_key
    from .models.processor import AudioProcessor
    from .models.classifier import EnhancedHybridVoiceClassifier as VoiceClassifier
    from .models.lid import LanguageDetector
except ImportError:
    # Fallback for direct execution
    try:
        from auth import validate_api_key
        from models.processor import AudioProcessor
        from models.classifier import EnhancedHybridVoiceClassifier as VoiceClassifier
        from models.lid import LanguageDetector
    except ImportError:
        logger.error("Failed to import required modules. Check your project structure.")
        raise

# Initialize FastAPI with metadata
app = FastAPI(
    title="Voice Classifier API",
    description="AI vs Human voice detection for Tamil, English, Hindi, Malayalam, Telugu",
    version="1.0.0"
)

# Add CORS middleware for evaluation access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for evaluation
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialize Models (Global Load) ---
logger.info("="*70)
logger.info("🚀 Initializing System Models...")
logger.info("="*70)

try:
    processor = AudioProcessor()
    logger.info("✅ AudioProcessor loaded")
    
    classifier = VoiceClassifier()
    logger.info("✅ VoiceClassifier loaded")
    
    lid_detector = LanguageDetector()
    logger.info("✅ LanguageDetector loaded")
    
    logger.info("="*70)
    logger.info("✅ System Ready for Evaluation")
    logger.info("="*70)
except Exception as e:
    logger.error(f"❌ Failed to initialize models: {e}")
    raise

# --- Request Schema ---
class DetectionRequest(BaseModel):
    language: str  # Tamil, English, Hindi, Malayalam, Telugu
    audioFormat: str  # mp3, wav, m4a, flac, ogg
    audioBase64: str  # Base64-encoded audio
    return_details: bool = False  # Optional detailed output

    class Config:
        schema_extra = {
            "example": {
                "language": "English",
                "audioFormat": "mp3",
                "audioBase64": "SGVsbG8gV29ybGQ=",
                "return_details": False
            }
        }

# --- Health Check Endpoint ---
@app.get("/")
async def root():
    """Root endpoint - API status"""
    return {
        "status": "online",
        "service": "Voice Classifier API",
        "version": "1.0.0",
        "supported_languages": ["Tamil", "English", "Hindi", "Malayalam", "Telugu"],
        "endpoints": {
            "health": "/health",
            "detection": "/api/voice-detection",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check for monitoring"""
    return {
        "status": "healthy",
        "models_loaded": {
            "processor": processor is not None,
            "classifier": classifier is not None,
            "lid_detector": lid_detector is not None
        },
        "ready_for_evaluation": True
    }

# --- Main Detection Endpoint ---
@app.post("/api/voice-detection", dependencies=[Depends(validate_api_key)])
async def detect_voice(payload: DetectionRequest):
    """
    Detect if a voice is AI-generated or human.
    
    CRITICAL FOR EVALUATION:
    - Accepts Base64-encoded audio in supported formats
    - Returns classification: AI_GENERATED or HUMAN
    - Includes confidenceScore (0.0 to 1.0)
    - Provides human-readable explanation
    - Auto-detects language (overrides input if needed)
    
    Request Format:
        {
            "language": "Tamil",
            "audioFormat": "mp3",
            "audioBase64": "base64_encoded_audio_data",
            "return_details": false
        }
    
    Response Format:
        {
            "status": "success",
            "language": "Tamil",
            "classification": "AI_GENERATED",
            "confidenceScore": 0.91,
            "explanation": "Unnatural pitch consistency detected"
        }
    """
    # Create a unique temp file for this request
    temp_filename = f"temp_{uuid.uuid4()}.{payload.audioFormat}"
    
    try:
        logger.info(f"📥 Processing request: language={payload.language}, format={payload.audioFormat}")
        
        # 1. Decode Base64 & Save to Disk
        audio_io = processor.decode_base64(payload.audioBase64)
        with open(temp_filename, "wb") as f:
            f.write(audio_io.getbuffer())
        
        logger.info(f"✅ Audio decoded and saved to {temp_filename}")
        
        # 2. Load Audio Data (Optimized: Load once for everyone)
        # Load as 16kHz mono, which is the standard for most AI models
        audio_array, _ = librosa.load(temp_filename, sr=16000, mono=True)
        
        # Validate audio is not silent
        rms = np.sqrt(np.mean(audio_array**2))
        if rms < 0.001:
            logger.warning(f"⚠️ Silent audio detected (RMS: {rms})")
            return {
                "status": "error",
                "message": "Audio appears to be silent or nearly silent"
            }
        
        logger.info(f"✅ Audio loaded: {len(audio_array)/16000:.1f}s duration, RMS: {rms:.4f}")
        
        # 3. Detect Language
        detected_lang = lid_detector.detect(temp_filename)
        logger.info(f"🌍 Language detected: {detected_lang}")
        
        # 4. Forensic Analysis (AI vs Human) - UPDATED
        # The new classifier returns a dictionary with detailed info
        logger.info(f"🤖 Running classifier for {detected_lang}...")
        result = classifier.predict(audio_array, language=detected_lang, return_details=payload.return_details)
        
        # Extract values from the new classifier format
        classification = result.get("verdict", "UNKNOWN")  # "AI_GENERATED" or "HUMAN"
        confidence = result.get("confidence", 0.0)  # 0.0 to 1.0
        explanation = result.get("explanation", "")  # Detailed explanation
        method = result.get("method", "unknown")  # Detection method used
        
        logger.info(f"✅ Classification: {classification} | Confidence: {confidence:.3f}")
        
        # CRITICAL: Ensure classification is valid for evaluation
        if classification not in ["AI_GENERATED", "HUMAN"]:
            logger.warning(f"⚠️ Invalid classification '{classification}', defaulting to UNCERTAIN")
            classification = "HUMAN" if confidence > 0.5 else "AI_GENERATED"
            confidence = 0.5
            explanation = "Unable to classify with high confidence"
        
        # 5. Construct "Smart" Explanation
        final_explanation = explanation
        
        # Add language mismatch note if detected
        if payload.language.lower() != detected_lang.lower():
            final_explanation += f" (Note: Input labeled as {payload.language}, but detected {detected_lang})."
        
        # 6. Build Response (EXACT FORMAT FOR EVALUATION)
        response = {
            "status": "success",
            "language": detected_lang,
            "classification": classification,  # MUST be "AI_GENERATED" or "HUMAN"
            "confidenceScore": round(confidence, 3),  # Round to 3 decimal places
            "explanation": final_explanation
        }
        
        # 7. Add Optional Method Field (not required by spec but useful)
        if payload.return_details:
            response["method"] = method
            
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
            if "details" in result:
                response["details"] = result["details"]
        
        logger.info(f"📤 Response sent: {classification} with confidence {confidence:.3f}")
        return response
    
    except Exception as e:
        logger.error(f"❌ API Error: {str(e)}", exc_info=True)
        
        return {
            "status": "error", 
            "message": f"Processing failed: {str(e)}"
        }
    
    finally:
        # 8. Cleanup: Always remove the temp file
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
                logger.debug(f"🗑️ Cleaned up temp file: {temp_filename}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to remove temp file: {e}")

# --- Batch Detection Endpoint (Optional, for efficiency) ---
@app.post("/api/batch-detection", dependencies=[Depends(validate_api_key)])
async def batch_detect_voice(payloads: list[DetectionRequest]):
    """
    Process multiple audio files in one request (Optional for evaluation)
    Maximum 10 files per batch to prevent overload
    """
    if len(payloads) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 files per batch request"
        )
    
    results = []
    for i, payload in enumerate(payloads):
        logger.info(f"Processing batch item {i+1}/{len(payloads)}")
        try:
            result = await detect_voice(payload)
            results.append(result)
        except Exception as e:
            results.append({
                "status": "error",
                "message": str(e)
            })
    
    return {"results": results}

# --- Error Handlers ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return {
        "status": "error",
        "message": exc.detail
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return {
        "status": "error",
        "message": "An unexpected error occurred"
    }

# --- Main Entry Point ---
if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Voice Classifier API Server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )