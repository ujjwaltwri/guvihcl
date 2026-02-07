import requests
import base64
import os
import glob
import json
import time
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Dict, Optional

# CONFIG
API_URL = "https://zenoharsh01-voice-detection-api.hf.space/api/voice-detection"
API_KEY = "sk_test_123456789"
SAMPLES_DIR = "samples"
REQUEST_TIMEOUT = 300  # Increased timeout for complex processing
MAX_WORKERS = 1  # Process one at a time to avoid overwhelming server

# Color codes for terminal output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        return f"{seconds/60:.1f}m"

def process_single_file(filepath: str, show_details: bool = False) -> Dict:
    """Process a single audio file and return results"""
    filename = os.path.basename(filepath)
    file_ext = Path(filepath).suffix[1:].lower()
    
    result = {
        "filename": filename,
        "success": False,
        "error": None,
        "duration": 0.0,
        "classification": None,
        "confidenceScore": None,
        "language": None,
        "explanation": None
    }
    
    start_time = time.time()
    
    try:
        # Read and encode file
        try:
            with open(filepath, "rb") as f:
                file_size = os.path.getsize(filepath)
                if file_size > 100 * 1024 * 1024:  # 100MB limit
                    result["error"] = f"File too large: {file_size/(1024*1024):.1f}MB"
                    return result
                    
                b64_str = base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            result["error"] = f"File read error: {str(e)}"
            return result
        
        # Prepare payload
        payload = {
            "language": "English",  # Hint, API will detect actual language
            "audioFormat": file_ext,
            "audioBase64": b64_str,
            "return_details": show_details
        }
        
        headers = {
            "x-api-key": API_KEY,
            "Content-Type": "application/json"
        }
        
        # Make API request with timeout
        try:
            response = requests.post(
                API_URL, 
                json=payload, 
                headers=headers, 
                timeout=REQUEST_TIMEOUT
            )
            
            result["duration"] = time.time() - start_time
            
            if response.status_code != 200:
                result["error"] = f"HTTP {response.status_code}: {response.text[:100]}"
                return result
            
            data = response.json()
            
            # Check for API-level errors
            if data.get("status") == "error":
                result["error"] = data.get("message", "Unknown API error")
                return result
            
            # Verify success status
            if data.get("status") != "success":
                result["error"] = f"Unexpected status: {data.get('status', 'unknown')}"
                return result
            
            # Extract data - matching the API response format exactly
            result["success"] = True
            result["classification"] = data.get("classification", "UNKNOWN")
            result["confidenceScore"] = data.get("confidenceScore", 0.0)
            result["explanation"] = data.get("explanation", "")
            result["language"] = data.get("language", "Unknown")
            
            return result
        
        except requests.exceptions.Timeout:
            result["error"] = f"Timeout after {REQUEST_TIMEOUT}s"
            result["duration"] = time.time() - start_time
            return result
        except requests.exceptions.ConnectionError:
            result["error"] = "Connection failed - is server running?"
            return result
        except Exception as e:
            result["error"] = f"Request error: {str(e)}"
            return result
    
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"
        return result

def print_result_row(result: Dict, show_details: bool = False):
    """Print a single result row"""
    filename = result["filename"][:25]
    duration = format_duration(result["duration"])
    
    if not result["success"]:
        # Error row
        error_msg = result["error"][:50] if result["error"] else "Unknown error"
        print(f"{Colors.RED}{filename:<25} | {'--':<8} | {'ERROR':<13} | {'--':<6} | {duration:<8} | {error_msg}{Colors.RESET}")
        return
    
    # Success row
    classification = result["classification"]
    confidenceScore = result["confidenceScore"]
    language = result["language"][:8]
    explanation = result["explanation"][:60] if not show_details else result["explanation"][:100]
    
    # Color coding based on classification
    if classification == "HUMAN":
        classification_color = Colors.GREEN
    elif classification == "AI_GENERATED":
        classification_color = Colors.RED
    else:
        classification_color = Colors.YELLOW
    
    # Confidence color
    if confidenceScore >= 0.75:
        conf_color = Colors.GREEN
    elif confidenceScore >= 0.55:
        conf_color = Colors.YELLOW
    else:
        conf_color = Colors.RED
    
    print(f"{filename:<25} | {language:<8} | {classification_color}{classification:<13}{Colors.RESET} | "
          f"{conf_color}{confidenceScore:.3f}{Colors.RESET} | {duration:<8} | {explanation}")

def run_scanner(show_details: bool = False, filter_pattern: Optional[str] = None):
    """Scan all audio files in the samples directory"""
    
    # Support multiple audio formats
    audio_extensions = ["*.mp3", "*.wav", "*.m4a", "*.flac", "*.ogg"]
    files = []
    for ext in audio_extensions:
        files.extend(glob.glob(os.path.join(SAMPLES_DIR, ext)))
    
    # Apply filter if specified
    if filter_pattern:
        files = [f for f in files if filter_pattern.lower() in os.path.basename(f).lower()]
    
    files = sorted(files)
    
    if not files:
        print(f"{Colors.RED}❌ No audio files found in '{SAMPLES_DIR}/' folder.{Colors.RESET}")
        if filter_pattern:
            print(f"   (Filter: '{filter_pattern}')")
        return
    
    print(f"\n{Colors.BOLD}{'='*140}{Colors.RESET}")
    print(f"{Colors.BOLD}🔍 VOICE CLASSIFIER TEST SCANNER{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*140}{Colors.RESET}\n")
    print(f"📁 Scanning: {SAMPLES_DIR}/")
    print(f"📊 Files found: {len(files)}")
    if filter_pattern:
        print(f"🔍 Filter: '{filter_pattern}'")
    print(f"⏱️  Timeout: {REQUEST_TIMEOUT}s per file")
    print(f"🔧 Details mode: {'ON' if show_details else 'OFF'}")
    print()
    
    # Header
    print(f"{Colors.BOLD}{'FILE':<25} | {'LANG':<8} | {'CLASSIFICATION':<13} | {'CONF':<6} | {'TIME':<8} | EXPLANATION{Colors.RESET}")
    print("-" * 140)
    
    # Statistics
    stats = {
        "total": 0,
        "ai_detected": 0,
        "human_detected": 0,
        "errors": 0,
        "total_time": 0.0,
        "avg_confidence_human": [],
        "avg_confidence_ai": []
    }
    
    # Process files
    for filepath in files:
        result = process_single_file(filepath, show_details)
        
        # Update stats
        stats["total"] += 1
        stats["total_time"] += result["duration"]
        
        if result["success"]:
            if result["classification"] == "AI_GENERATED":
                stats["ai_detected"] += 1
                stats["avg_confidence_ai"].append(result["confidenceScore"])
            elif result["classification"] == "HUMAN":
                stats["human_detected"] += 1
                stats["avg_confidence_human"].append(result["confidenceScore"])
        else:
            stats["errors"] += 1
        
        # Print result
        print_result_row(result, show_details)
    
    # Print summary
    print("-" * 140)
    print(f"\n{Colors.BOLD}📊 SUMMARY{Colors.RESET}")
    print(f"{'='*140}")
    print(f"Total processed:     {stats['total']}")
    print(f"{Colors.RED}AI Generated:        {stats['ai_detected']}{Colors.RESET}", end="")
    if stats["avg_confidence_ai"]:
        avg_ai_conf = sum(stats["avg_confidence_ai"]) / len(stats["avg_confidence_ai"])
        print(f" (avg confidence: {avg_ai_conf:.3f})")
    else:
        print()
    
    print(f"{Colors.GREEN}Human voices:        {stats['human_detected']}{Colors.RESET}", end="")
    if stats["avg_confidence_human"]:
        avg_human_conf = sum(stats["avg_confidence_human"]) / len(stats["avg_confidence_human"])
        print(f" (avg confidence: {avg_human_conf:.3f})")
    else:
        print()
    
    print(f"{Colors.YELLOW}Errors:              {stats['errors']}{Colors.RESET}")
    print(f"\nTotal time:          {format_duration(stats['total_time'])}")
    if stats['total'] > 0:
        print(f"Average per file:    {format_duration(stats['total_time'] / stats['total'])}")
    print(f"{Colors.BOLD}{'='*140}{Colors.RESET}\n")
    
    # Performance warnings
    if stats['total_time'] / max(stats['total'], 1) > 30:
        print(f"{Colors.YELLOW}⚠️  Warning: Average processing time is high (>{format_duration(30)}). "
              f"Consider optimizing or reducing audio length.{Colors.RESET}\n")
    
    if stats['errors'] > 0:
        print(f"{Colors.YELLOW}⚠️  Some files failed to process. Check errors above for details.{Colors.RESET}\n")

def test_single_file(filepath: str):
    """Test a single file with detailed output"""
    if not os.path.exists(filepath):
        print(f"{Colors.RED}❌ File not found: {filepath}{Colors.RESET}")
        return
    
    print(f"\n{Colors.BOLD}{'='*140}{Colors.RESET}")
    print(f"{Colors.BOLD}🔍 TESTING SINGLE FILE: {os.path.basename(filepath)}{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*140}{Colors.RESET}\n")
    
    result = process_single_file(filepath, show_details=True)
    
    if not result["success"]:
        print(f"{Colors.RED}❌ Processing failed: {result['error']}{Colors.RESET}")
        return
    
    # Print detailed results
    print(f"{Colors.BOLD}Results:{Colors.RESET}")
    print(f"  Classification: {Colors.GREEN if result['classification']=='HUMAN' else Colors.RED}"
          f"{result['classification']}{Colors.RESET}")
    print(f"  Confidence:     {result['confidenceScore']:.3f}")
    print(f"  Language:       {result['language']}")
    print(f"  Duration:       {format_duration(result['duration'])}")
    print(f"\n{Colors.BOLD}Explanation:{Colors.RESET}")
    print(f"  {result['explanation']}")
    
    print(f"\n{Colors.BOLD}{'='*140}{Colors.RESET}\n")

def print_usage():
    """Print usage instructions"""
    print(f"\n{Colors.BOLD}Usage:{Colors.RESET}")
    print(f"  python test_api.py                    # Scan all files in samples/")
    print(f"  python test_api.py --details          # Show detailed layer analysis")
    print(f"  python test_api.py --filter english   # Only test files matching 'english'")
    print(f"  python test_api.py --test file.wav    # Test single file with details")
    print()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test voice classifier API")
    parser.add_argument("--details", action="store_true", help="Show detailed layer analysis")
    parser.add_argument("--filter", type=str, help="Filter files by name pattern")
    parser.add_argument("--test", type=str, help="Test a single file with detailed output")
    
    args = parser.parse_args()
    
    if args.test:
        # Test single file mode
        test_single_file(args.test)
    else:
        # Scan mode
        run_scanner(show_details=args.details, filter_pattern=args.filter)