import os
import glob
import numpy as np
import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# SETUP MODEL
model_name = "facebook/wav2vec2-xls-r-300m"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("⏳ Loading Analysis Tools...")
extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name).to(device)
model.eval()

def analyze_file(filepath):
    try:
        y, sr = librosa.load(filepath, sr=16000)
        y, _ = librosa.effects.trim(y, top_db=60)
        if len(y) < 2000: return None

        # 1. Jitter
        try:
            f0 = librosa.yin(y, fmin=50, fmax=300, sr=sr)
            f0 = f0[~np.isnan(f0)]
            jitter = np.mean(np.abs(np.diff(f0))) if len(f0) > 1 else 0
        except: jitter = 0

        # 2. Texture (MFCC Variance)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        texture = np.mean(np.var(mfccs, axis=1))

        # 3. Smoothness
        inputs = extractor(y, sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(device)
        with torch.no_grad():
            hidden = model(inputs).last_hidden_state.squeeze(0).cpu().numpy()
        smoothness = np.mean(np.abs(hidden[1:] - hidden[:-1]))

        return jitter, texture, smoothness
    except Exception as e:
        print(f"Error on {filepath}: {e}")
        return None

# RUN ANALYSIS
files = sorted(glob.glob("samples/*.mp3"))
print(f"\n🚀 RAW DATA ANALYSIS ({len(files)} files)")
print(f"{'FILENAME':<20} | {'JITTER':<10} | {'TEXTURE':<10} | {'SMOOTHNESS'}")
print("-" * 65)

for f in files:
    name = os.path.basename(f)
    res = analyze_file(f)
    if res:
        j, t, s = res
        # Multiply Smoothness by 10,000 to make it readable (e.g. 0.005 -> 50.0)
        print(f"{name[:20]:<20} | {j:<10.2f} | {t:<10.2f} | {s*10000:<10.2f}")

print("-" * 65)