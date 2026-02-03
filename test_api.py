import requests
import base64
import os
import glob
import json

# CONFIG
API_URL = "http://127.0.0.1:8000/api/voice-detection"
API_KEY = "sk_test_123456789"
SAMPLES_DIR = "samples"

def run_scanner():
    files = sorted(glob.glob(os.path.join(SAMPLES_DIR, "*.mp3")))
    if not files:
        print(f"❌ No MP3 files found in '{SAMPLES_DIR}' folder.")
        return

    print(f"\n🚀 Forensic scan started — {len(files)} files\n")
    # ADDED 'LANGUAGE' COLUMN HEADER
    print(f"{'FILE':<20} | {'LANG':<8} | {'VERDICT':<14} | {'CONFIDENCE'} | {'NOTES'}")
    print("-" * 105)

    for filepath in files:
        filename = os.path.basename(filepath)
        
        try:
            with open(filepath, "rb") as f:
                b64_str = base64.b64encode(f.read()).decode('utf-8')
        except:
            print(f"{filename[:20]:<20} | --       | ERROR          | --       | File read error")
            continue

        payload = {
            "language": "English", 
            "audioFormat": "mp3",
            "audioBase64": b64_str
        }
        headers = {"x-api-key": API_KEY}
        
        try:
            response = requests.post(API_URL, json=payload, headers=headers)
            
            if response.status_code != 200:
                print(f"{filename[:20]:<20} | --       | HTTP {response.status_code}     | --       | Server Error")
                continue
                
            data = response.json()
            
            verdict = data.get("classification", "UNKNOWN")
            # GET LANGUAGE FROM RESPONSE
            lang = data.get("language", "Unk")[:8] 
            conf = data.get("confidenceScore", 0.0)
            expl = data.get("explanation", "")[:45]
            
            # Color coding
            color = "\033[92m" if verdict == "HUMAN" else "\033[91m"
            reset = "\033[0m"
            
            # ADDED 'lang' VARIABLE TO PRINT
            print(f"{filename[:20]:<20} | {lang:<8} | {color}{verdict:<14}{reset} | {conf:.1%}    | {expl}...")
            
        except json.JSONDecodeError:
            print(f"{filename[:20]:<20} | --       | JSON ERROR     | --       | Invalid JSON")
        except Exception as e:
            print(f"{filename[:20]:<20} | --       | CONN ERROR     | --       | {str(e)[:20]}")

    print("-" * 105)

if __name__ == "__main__":
    run_scanner()