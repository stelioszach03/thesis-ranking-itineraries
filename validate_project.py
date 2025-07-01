print("=== NYC Itinerary Ranking - Final Validation ===\n")

# 1. Check all imports work
try:
    from src import *
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")

# 2. Test data generation
import os
data_files = ["nyc_pois.json", "distance_matrix.npy", "subway_stations.json"]
missing = [f for f in data_files if not os.path.exists(f"data/nyc_data/{f}")]
if missing:
    print(f"✗ Missing data files: {missing}")
else:
    print("✓ All data files present")

# 3. Run simple end-to-end test
try:
    from src.greedy_algorithms import GreedyPOISelection, Constraints
    import json
    import numpy as np
    
    with open('data/nyc_data/nyc_pois.json', 'r') as f:
        pois = json.load(f)
    dist = np.load('data/nyc_data/distance_matrix.npy')
    
    greedy = GreedyPOISelection(pois[:100], dist[:100, :100])
    result = greedy.select_pois(
        {"museum": 0.8}, 
        Constraints(budget=100, max_time_hours=5)
    )
    print(f"✓ End-to-end test passed - generated {len(result)} POIs")
except Exception as e:
    print(f"✗ End-to-end test failed: {e}")

# 4. Check test status
import subprocess
result = subprocess.run(
    ["venv/bin/python", "-m", "pytest", "tests/", "--tb=no"],
    capture_output=True, text=True
)
if "passed" in result.stdout:
    passed = int(result.stdout.split("passed")[0].strip().split()[-1])
    failed = 0
    if "failed" in result.stdout:
        failed = int(result.stdout.split("failed")[0].strip().split()[-1])
    print(f"\nTest Results: {passed} passed, {failed} failed")
    
print("\n✅ Project validation complete!")