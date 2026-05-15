"""
Test Script — Flight Delay Predictor
=====================================
Sends a sample prediction request to the live Gradio application's
internal API endpoint and verifies a valid response is returned.

Requirements:
    pip install requests

Usage:
    python test_project.py

    # Optional: override the app URL
    APP_URL=https://your-space.hf.space python test_project.py

Environment:
    APP_URL  — Base URL of the deployed Hugging Face Space (default below).
               Update this to your actual Space URL before submission.
"""

import sys
import json
import os
import requests

# ── Config ────────────────────────────────────────────────────────────────────
APP_URL = os.environ.get(
    "APP_URL",
    "https://YOUR-HF-USERNAME-flight-delay-predictor.hf.space",  # <-- update this
)

# Gradio exposes a /run/predict REST API automatically
PREDICT_ENDPOINT = f"{APP_URL}/run/predict"

# Sample input: ATL → LAX, Delta, Friday in June, 8 AM, ~2475 miles, 330 min
PAYLOAD = {
    "data": [
        "Delta Air Lines (DL)",   # airline
        "ATL",                     # origin
        "LAX",                     # destination
        "June",                    # month
        "Friday",                  # day of week
        8,                         # departure hour
        2475,                      # distance (miles)
        330,                       # scheduled duration (minutes)
    ]
}

TIMEOUT = 60  # seconds (Hugging Face free tier may cold-start)


# ── Test ──────────────────────────────────────────────────────────────────────
def run_test():
    print("=" * 60)
    print("  Flight Delay Predictor — End-to-End Test")
    print("=" * 60)
    print(f"\nEndpoint : {PREDICT_ENDPOINT}")
    print(f"Payload  : ATL → LAX | Delta | Friday June | 8 AM\n")

    try:
        resp = requests.post(
            PREDICT_ENDPOINT,
            json=PAYLOAD,
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT,
        )
    except requests.exceptions.ConnectionError:
        print(f"FAIL: Could not connect to {APP_URL}")
        print("      Ensure the Hugging Face Space is running.")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print(f"FAIL: Request timed out after {TIMEOUT}s")
        print("      The Space may be cold-starting — try again in ~30s.")
        sys.exit(1)

    if resp.status_code != 200:
        print(f"FAIL: HTTP {resp.status_code}")
        print(f"      Body: {resp.text[:500]}")
        sys.exit(1)

    try:
        result = resp.json()
    except json.JSONDecodeError:
        print(f"FAIL: Response is not valid JSON\n{resp.text[:300]}")
        sys.exit(1)

    # Gradio returns {"data": [verdict, gauge_json, feature_json, tips]}
    data = result.get("data", [])
    if not data:
        print(f"FAIL: Response 'data' field is empty.\nFull response: {result}")
        sys.exit(1)

    verdict = data[0] if len(data) > 0 else "N/A"
    tips    = data[3] if len(data) > 3 else "N/A"

    print("─" * 60)
    print(f"  Prediction : {verdict}")
    print(f"  Tips       : {tips}")
    print("─" * 60)
    print("\nPASS ✅")
    sys.exit(0)


if __name__ == "__main__":
    run_test()
