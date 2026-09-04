"""
Measure live investigation latency (Fix 3) and update latency_results.json & evaluation_matrix.json.
"""
import time
import json
import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "src" / "models" / "saved"

def run_latency_measurement():
    print("=" * 60)
    print("STEP 1: Triggering Live Investigation on INC-FF60C217 via API")
    print("=" * 60)

    t0 = time.perf_counter()
    resp = requests.post("http://127.0.0.1:8000/api/incidents/INC-FF60C217/investigate", timeout=120)
    t1 = time.perf_counter()
    duration_ms = (t1 - t0) * 1000

    print(f"HTTP Status: {resp.status_code}")
    data = resp.json()
    print(f"Agent mode: {data.get('agent_mode')}")
    print(f"Mock reason: {data.get('mock_reason')}")
    print(f"Evidence tool calls count: {len(data.get('evidence', []))}")
    print(f"Measured Wall-Clock Latency: {duration_ms:.1f}ms ({duration_ms/1000:.2f}s)")
    
    summary = data.get("summary", "")
    print(f"Summary preview:\n{summary[:300]}...")

    if data.get("agent_mode") == "live":
        print("\nBackfilling measured latency into latency_results.json and evaluation_matrix.json...")
        
        # 1. Update latency_results.json
        latency_path = MODELS_DIR / "latency_results.json"
        with open(latency_path) as f:
            lat_data = json.load(f)
        
        lat_data["agent_investigation_ms"] = round(duration_ms, 1)
        lat_data["agent_measured"] = True
        lat_data["end_to_end_ms"] = round(lat_data.get("ml_inference_ms", 2.19) + duration_ms, 1)
        
        with open(latency_path, "w") as f:
            json.dump(lat_data, f, indent=2)
        print(f"  ✓ Updated {latency_path}: agent_investigation_ms={lat_data['agent_investigation_ms']}ms, agent_measured=True")

        # 2. Update evaluation_matrix.json
        eval_path = MODELS_DIR / "evaluation_matrix.json"
        with open(eval_path) as f:
            eval_data = json.load(f)
        
        if "operational" not in eval_data:
            eval_data["operational"] = {}
        eval_data["operational"]["investigation_time_ms"] = round(duration_ms, 1)
        
        with open(eval_path, "w") as f:
            json.dump(eval_data, f, indent=2)
        print(f"  ✓ Updated {eval_path}: investigation_time_ms={eval_data['operational']['investigation_time_ms']}ms")
    else:
        print(f"Warning: agent_mode was '{data.get('agent_mode')}', not 'live'. Mock reason: {data.get('mock_reason')}")

if __name__ == "__main__":
    run_latency_measurement()
