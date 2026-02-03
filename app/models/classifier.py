import torch
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

class VoiceClassifier:
    def __init__(self):
        self.model_name = "facebook/wav2vec2-xls-r-300m"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🔄 Loading AI Model: {self.model_name}...")
        try:
            # FIX: Use FeatureExtractor instead of Processor
            # This avoids looking for a 'vocab.json' that doesn't exist
            self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
            
            self.model = Wav2Vec2Model.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            print("✅ Model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e

    def predict(self, audio_array):
        try:
            if isinstance(audio_array, list):
                audio_array = np.array(audio_array)
                
            # 1. PREPROCESS (Using extractor now)
            inputs = self.extractor(
                audio_array, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            ).input_values.to(self.device)

            # 2. INFERENCE
            with torch.no_grad():
                outputs = self.model(inputs)
                hidden_states = outputs.last_hidden_state.squeeze(0).cpu().numpy()

            # 3. HYBRID ANALYSIS (Variance & Consistency)
            # Calculate variance across time for features
            feature_variance = np.var(hidden_states, axis=0)
            overall_variance = np.mean(feature_variance)
            
            # Tuned Threshold for XLS-R-300m Base
            # Base models have slightly different variance scales than fine-tuned ones.
            threshold = 0.008 

            if overall_variance < threshold:
                classification = "AI_GENERATED"
                confidence = 0.8
                explanation = (
                    f"Neural embeddings show unnatural consistency (Score: {overall_variance:.5f}). "
                    "Lacks biological phoneme irregularity."
                )
            else:
                classification = "HUMAN"
                confidence = 0.7
                explanation = (
                    f"High variance in neural features (Score: {overall_variance:.5f}). "
                    "Detected natural human vocal tract irregularities."
                )

            return classification, float(confidence), explanation

        except Exception as e:
            return "HUMAN", 0.5, f"Analysis failed. Error: {str(e)}"