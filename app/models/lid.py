import torch
import numpy as np
import librosa
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

class LanguageDetector:
    def __init__(self):
        print("🌍 Loading Language ID Model (Facebook MMS)...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Using Facebook's MMS-LID-126 (Supports 126 languages including yours)
        self.model_name = "facebook/mms-lid-126"
        
        try:
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            print("✅ Language ID Model Loaded.")
        except Exception as e:
            print(f"❌ Failed to load LID Model: {e}")
            self.model = None

        # Map MMS ISO codes (3 letters) to Competition Names
        self.lang_map = {
            "tam": "Tamil",
            "tel": "Telugu",
            "mal": "Malayalam",
            "hin": "Hindi",
            "eng": "English"
        }

    def detect(self, audio_path):
        """
        Input: Path to audio file.
        Output: 'Tamil', 'English', etc.
        """
        if not self.model:
            return "English" # Fallback

        try:
            # 1. Load Audio (Resample to 16kHz)
            # MMS expects a specific length, usually cropped to ~2-5 seconds for LID is enough
            audio_array, sr = librosa.load(audio_path, sr=16000)
            
            # 2. Preprocess
            inputs = self.processor(
                audio_array, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            ).input_values.to(self.device)

            # 3. Predict
            with torch.no_grad():
                outputs = self.model(inputs)
                logits = outputs.logits
            
            # 4. Decode
            predicted_id = torch.argmax(logits, dim=-1).item()
            detected_iso = self.model.config.id2label[predicted_id] # Returns e.g., 'tam', 'eng'
            
            # 5. Map to full name
            return self.lang_map.get(detected_iso, "English") # Default to English if it's French/German/etc.
            
        except Exception as e:
            print(f"⚠️ LID Error: {e}")
            return "English"