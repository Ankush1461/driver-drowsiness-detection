"""
DriveSafe AI - HuggingFace Spaces Keep-Alive Script
===================================================
Prevents the Space from sleeping by sending periodic HTTP pings.

Usage:
    python keep_alive.py                    # Run continuously (default: ping every 10 min)
    python keep_alive.py --interval 300     # Custom interval in seconds (5 min)
    python keep_alive.py --once             # Single ping (for cron/scheduled tasks)

Environment Variables:
    HF_SPACE_URL  - Override the Space URL (default: https://ankushkarmakar-drivesafe-ai-v2.hf.space)
"""

import requests
import time
import sys
import argparse
from datetime import datetime

# Default Space URL - update this if your Space URL changes
DEFAULT_SPACE_URL = "https://ankushkarmakar-drivesafe-ai-v2.hf.space"

def ping_space(url):
    """Send a GET request to keep the Space alive."""
    try:
        start = time.time()
        response = requests.get(url, timeout=30)
        elapsed = time.time() - start
        status = response.status_code
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if status == 200:
            print(f"[{timestamp}] OK  | Status: {status} | Latency: {elapsed:.2f}s | Space is alive")
            return True
        else:
            print(f"[{timestamp}] WARN | Status: {status} | Latency: {elapsed:.2f}s | Space returned non-200")
            return True  # Space is still responding, just not 200
            
    except requests.exceptions.ConnectionError:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] FAIL | Connection error - Space may be sleeping or starting up")
        return False
    except requests.exceptions.Timeout:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] FAIL | Timeout - Space took too long to respond")
        return False
    except Exception as e:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] FAIL | Error: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="HuggingFace Spaces Keep-Alive")
    parser.add_argument("--interval", type=int, default=600,
                        help="Ping interval in seconds (default: 600 = 10 minutes)")
    parser.add_argument("--once", action="store_true",
                        help="Single ping then exit (for cron/scheduled tasks)")
    parser.add_argument("--url", type=str, default=None,
                        help="Space URL (default: from HF_SPACE_URL env or built-in)")
    args = parser.parse_args()
    
    import os
    space_url = args.url or os.environ.get("HF_SPACE_URL", DEFAULT_SPACE_URL)
    
    print("=" * 60)
    print("DriveSafe AI - HuggingFace Spaces Keep-Alive")
    print("=" * 60)
    print(f"Target:  {space_url}")
    print(f"Mode:    {'Single ping' if args.once else f'Continuous (every {args.interval}s)'}")
    print("=" * 60)
    
    if args.once:
        success = ping_space(space_url)
        sys.exit(0 if success else 1)
    
    # Continuous mode
    consecutive_failures = 0
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            success = ping_space(space_url)
            
            if success:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    print(f"  >> {consecutive_failures} consecutive failures. Space may need manual restart.")
            
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\nKeep-alive stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
