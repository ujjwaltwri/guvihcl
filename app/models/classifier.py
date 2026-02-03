import torch
import numpy as np
import librosa
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

class VoiceClassifier:
    def __init__(self):
        self.model_name = "facebook/wav2vec2-xls-r-300m"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🔄 Loading Classifier: {self.model_name}...")
        try:
            self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
            self.model = Wav2Vec2Model.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            print("✅ Classifier Model loaded.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e

    def predict(self, audio_array):
        try:
            # Ensure proper shape
            if isinstance(audio_array, list):
                audio_array = np.array(audio_array)

            # 1. THE "PERFECT SILENCE" CHECK (Strongest Indicator)
            # AI often produces exact Zeros (0.0000) at the start or end.
            # Real microphones rarely produce exact zeros due to thermal noise.
            
            # Count how many exact zeros are in the array
            zero_count = np.sum(audio_array == 0.0)
            total_samples = len(audio_array)
            zero_ratio = zero_count / total_samples

            # 2. NOISE FLOOR ANALYSIS
            # Calculate the energy of the quietest frames (Room Tone)
            # Frame size: 2048 samples
            S = np.abs(librosa.stft(audio_array, n_fft=2048, hop_length=512))
            rms = librosa.feature.rms(S=S)[0]
            
            # Find the minimum energy (the "silence")
            # We filter out 0s to avoid log(-inf)
            rms_filtered = rms[rms > 0]
            if len(rms_filtered) > 0:
                min_energy_db = 20 * np.log10(np.min(rms_filtered)) # Decibels
            else:
                min_energy_db = -100.0 # Effective silence

            # 3. HIGH FREQUENCY ROLLOFF (Bandwidth Check)
            # Cheap AI/TTS often cuts frequencies above 12kHz or 16kHz sharply.
            # Real mics usually have noise up to Nyquist.
            spec_rolloff = librosa.feature.spectral_rolloff(y=audio_array, sr=16000, roll_percent=0.99)[0]
            avg_rolloff = np.mean(spec_rolloff)

            # --- SCORING LOGIC ---
            
            reasons = []
            ai_score = 0
            
            # Check A: Digital Zeros (The "Smoking Gun")
            # If > 1% of the file is exact zeros, it's highly suspicious
            if zero_ratio > 0.01: 
                ai_score += 5
                reasons.append("Contains perfect digital silence (0.0)")

            # Check B: Noise Floor
            # Real mic noise floor is usually -40dB to -60dB.
            # Generated audio noise floor is usually < -80dB (too clean).
            if min_energy_db < -75:
                ai_score += 4
                reasons.append(f"Unnaturally clean background ({min_energy_db:.0f}dB)")
            elif min_energy_db < -60:
                ai_score += 2  # Suspicious

            # Check C: Rolloff (Low bandwidth = AI)
            if avg_rolloff < 7000: # 7kHz cutoff (common in 16kHz TTS)
                ai_score += 2
                reasons.append("Low spectral bandwidth")

            # --- VERDICT ---
            # AI Score Threshold: 4+ means likely AI
            
            if ai_score >= 4:
                classification = "AI_GENERATED"
                # Map score 4->0.85, 10->0.99
                confidence = 0.85 + min(0.14, (ai_score - 4) / 6.0)
                explanation = f"Digital artifacts detected: {', '.join(reasons)}."
            else:
                classification = "HUMAN"
                confidence = 0.90
                explanation = f"Natural acoustics detected. Noise floor: {min_energy_db:.0f}dB (Realistic). No digital silence."

            return classification, float(confidence), explanation

        except Exception as e:
            return "HUMAN", 0.0, f"Error: {str(e)}"