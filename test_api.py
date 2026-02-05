import requests
import base64
import os
import glob
import json
from pathlib import Path
import sys

# CONFIG
API_URL = "http://127.0.0.1:8000/api/voice-detection"
API_KEY = "sk_test_123456789"
SAMPLES_DIR = "samples"

# Color codes for terminal output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

def run_scanner():
    """Scan all audio files in the samples directory"""
    
    # Support multiple audio formats
    audio_extensions = ["*.mp3", "*.wav", "*.m4a", "*.flac", "*.ogg"]
    files = []
    for ext in audio_extensions:
        files.extend(glob.glob(os.path.join(SAMPLES_DIR, ext)))
    
    files = sorted(files)
    
    if not files:
        print(f"{Colors.RED}❌ No audio files found in '{SAMPLES_DIR}' folder.{Colors.RESET}")
        return
    
    print(f"\n{Colors.BOLD}{'='*110}{Colors.RESET}")
    print(f"{Colors.BOLD}🔍 VOICE CLASSIFIER TEST SCANNER{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*110}{Colors.RESET}\n")
    print(f"📁 Scanning: {SAMPLES_DIR}/")
    print(f"📊 Files found: {len(files)}\n")
    
    # Header - Added LANG column
    print(f"{Colors.BOLD}{'FILE':<20} | {'LANG':<8} | {'VERDICT':<13} | {'CONF':<6} | {'METHOD':<15} | DETAILS{Colors.RESET}")
    print("-" * 130)
    
    # Statistics
    stats = {
        "total": 0,
        "ai_detected": 0,
        "human_detected": 0,
        "errors": 0
    }
    
    for filepath in files:
        filename = os.path.basename(filepath)
        file_ext = Path(filepath).suffix[1:].lower()
        
        try:
            with open(filepath, "rb") as f:
                b64_str = base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            print(f"{Colors.RED}{filename[:20]:<20} | --       | ERROR        | --    | --              | Cannot read file{Colors.RESET}")
            stats["errors"] += 1
            continue
        
        # Prepare payload
        payload = {
            "language": "English", # This is just a hint, the API detects the real one
            "audioFormat": file_ext.replace(".", ""),
            "audioBase64": b64_str
        }
        
        headers = {
            "x-api-key": API_KEY,
            "Content-Type": "application/json"
        }
        
        try:
            # Timeout set to 60s to handle processing time safely
            response = requests.post(API_URL, json=payload, headers=headers, timeout=60)
            
            if response.status_code != 200:
                print(f"{Colors.RED}{filename[:20]:<20} | --       | HTTP {response.status_code:<4} | --    | --              | {response.text[:40]}{Colors.RESET}")
                stats["errors"] += 1
                continue
            
            data = response.json()
            
            # Extract data
            verdict = data.get("classification", "UNKNOWN")
            confidence = data.get("confidenceScore", 0.0)
            explanation = data.get("explanation", "")
            method = data.get("method", "fast_inference")
            # EXTRACT LANGUAGE (Truncate to 8 chars to fit table)
            language = data.get("language", "Unknown")[:8]
            
            # Update stats
            stats["total"] += 1
            if verdict == "AI_GENERATED":
                stats["ai_detected"] += 1
            elif verdict == "HUMAN":
                stats["human_detected"] += 1
            
            # Color coding
            if verdict == "HUMAN":
                verdict_color = Colors.GREEN
            elif verdict == "AI_GENERATED":
                verdict_color = Colors.RED
            else:
                verdict_color = Colors.YELLOW
            
            # Confidence color
            conf_color = Colors.GREEN if confidence >= 0.8 else Colors.YELLOW
            if confidence < 0.6: conf_color = Colors.RED
            
            # Smart Detail Truncation
            detail = explanation
            if len(detail) > 45:
                detail = detail[:42] + "..."

            # Print result with LANGUAGE column
            print(f"{filename[:20]:<20} | {language:<8} | {verdict_color}{verdict:<13}{Colors.RESET} | "
                  f"{conf_color}{confidence:.1%}<6{Colors.RESET} | {method:<15} | {detail}")
        
        except requests.exceptions.Timeout:
            print(f"{Colors.RED}{filename[:20]:<20} | --       | TIMEOUT      | --    | --              | Server took >60s{Colors.RESET}")
            stats["errors"] += 1
        except Exception as e:
            print(f"{Colors.RED}{filename[:20]:<20} | --       | ERROR        | --    | --              | {str(e)[:40]}{Colors.RESET}")
            stats["errors"] += 1
    
    # Print summary
    print("-" * 130)
    print(f"\n{Colors.BOLD}📊 SUMMARY{Colors.RESET}")
    print(f"Total processed: {stats['total']}")
    print(f"{Colors.RED}AI Generated:   {stats['ai_detected']}{Colors.RESET}")
    print(f"{Colors.GREEN}Human voices:   {stats['human_detected']}{Colors.RESET}")
    print(f"{Colors.YELLOW}Errors:         {stats['errors']}{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*110}{Colors.RESET}\n")

if __name__ == "__main__":
    run_scanner()