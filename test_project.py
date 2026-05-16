"""
Test Script — Flight Delay Predictor
=====================================
Sends a sample prediction request to the live Gradio application.

Requirements: pip install gradio_client

Usage: python test_project.py
"""

import sys
from gradio_client import Client

APP_URL = "HugoNCF/flight-delay-predictor"

def run_test():
    print("=" * 60)
    print("  Flight Delay Predictor — End-to-End Test")
    print("=" * 60)
    print(f"\nConnecting to: {APP_URL}")
    print(f"Payload: ATL → LAX | Delta | Friday June | 8 AM\n")

    try:
        client = Client(APP_URL)
        result = client.predict(
            airline_name="Delta Air Lines (DL)",
            origin="ATL",
            dest="LAX",
            month_name="June",
            day_name="Friday",
            dep_hour=8,
            distance=2475,
            sched_duration=330,
            api_name="/run_prediction",
        )

        verdict = result[0]
        tips    = result[3]

        print("─" * 60)
        print(f"  Prediction : {verdict}")
        print(f"  Tips       : {tips}")
        print("─" * 60)
        print("\nPASS ✅")
        sys.exit(0)

    except Exception as e:
        print(f"FAIL: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()