import requests
import base64
import os
import glob

# CONFIG
API_URL = "http://127.0.0.1:8000/api/voice-detection"
API_KEY = "sk_test_123456789"
SAMPLES_DIR = "samples"

# 🔑 THE ANSWER KEY (Ground Truth Manifest)
# Map your specific filenames to what they actually are.
# This avoids the need to rename files.
GROUND_TRUTH = {
    "sample_voice_1.mp3": "AI_GENERATED",  # We explicitly tell the test this is AI
    "my_voice_recording.mp3": "HUMAN",
    "elevenlabs_sample.mp3": "AI_GENERATED",
}

def get_true_label(filename):
    """
    Looks up the filename in the Answer Key. 
    If not found, falls back to guessing from the name.
    """
    if filename in GROUND_TRUTH:
        return GROUND_TRUTH[filename]
    
    # Fallback for other files
    if "ai" in filename.lower(): return "AI_GENERATED"
    if "human" in filename.lower(): return "HUMAN"
    return "UNKNOWN"

def run_suite():
    files = glob.glob(os.path.join(SAMPLES_DIR, "*.mp3"))
    if not files:
        print(f"❌ No MP3 files found in '{SAMPLES_DIR}' folder.")
        return

    print(f"🚀 Starting Professional Test Suite on {len(files)} files...\n")
    
    correct_count = 0
    total_files = 0

    for filepath in files:
        filename = os.path.basename(filepath)
        true_label = get_true_label(filename)
        
        # Skip files we don't know the answer to (optional)
        if true_label == "UNKNOWN":
            print(f"⚠️ Skipping {filename} (No Ground Truth defined)")
            continue

        total_files += 1

        # Encode
        with open(filepath, "rb") as f:
            b64_str = base64.b64encode(f.read()).decode('utf-8')

        # Request
        # Note: We send "English" as default, relying on the API's LID to correct it
        payload = {
            "language": "English", 
            "audioFormat": "mp3",
            "audioBase64": b64_str
        }
        headers = {"x-api-key": API_KEY}
        
        try:
            response = requests.post(API_URL, json=payload, headers=headers)
            data = response.json()
            
            predicted = data.get("classification", "ERROR")
            confidence = data.get("confidenceScore", 0.0)
            detected_lang = data.get("language", "Unknown")
            
            is_correct = (predicted == true_label)
            if is_correct: correct_count += 1
            
            icon = "✅" if is_correct else "❌"
            print(f"{icon} {filename}")
            print(f"   ├─ Truth: {true_label}")
            print(f"   ├─ Pred:  {predicted} (Confidence: {confidence:.2f})")
            print(f"   └─ Lang:  {detected_lang}")
            
        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")

    if total_files > 0:
        accuracy = (correct_count / total_files) * 100
        print(f"\n🎯 Final Accuracy: {accuracy:.2f}% ({correct_count}/{total_files})")
    else:
        print("\n⚠️ No files with known labels were tested.")

if __name__ == "__main__":
    run_suite()