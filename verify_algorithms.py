# save as verify_algorithms.py
print("=== Verifying Algorithm Implementations ===\n")

# Algorithms mentioned in thesis
required_algorithms = {
    "GreedyPOISelection": "Basic greedy from Basu Roy et al.",
    "HeapPrunGreedyPOI": "Heap-optimized greedy", 
    "AStarItineraryPlanner": "A* for optimal pathfinding",
    "LPAStarPlanner": "Dynamic replanning",
    "HybridPlanner": "Two-phase approach"
}

missing = []
for algo, desc in required_algorithms.items():
    try:
        exec(f"from src import {algo}")
        print(f"✓ {algo}: {desc}")
    except:
        print(f"✗ {algo}: MISSING")
        missing.append(algo)

# Check specific features
print("\n=== Feature Verification ===")

# 1. Interactive feedback
try:
    from src.greedy_algorithms import InteractiveFeedback
    print("✓ Interactive feedback mechanism")
except:
    print("✗ Interactive feedback MISSING")

# 2. CSS with correct weights
try:
    with open('src/metrics_definitions.py', 'r') as f:
        content = f.read()
        if "0.35" in content and "0.25" in content and "0.15" in content:
            print("✓ CSS with research-based weights")
        else:
            print("✗ CSS weights don't match research")
except:
    print("✗ Cannot verify CSS implementation")

# 3. LPA* dynamic updates
try:
    from src.lpa_star import DynamicUpdate, UpdateType
    print("✓ Dynamic update capability")
except:
    print("✗ Dynamic updates MISSING")

# 4. NYC data
import os
if os.path.exists('data/nyc_data/nyc_pois.json'):
    import json
    with open('data/nyc_data/nyc_pois.json', 'r') as f:
        pois = json.load(f)
        if len(pois) == 10847:
            print(f"✓ NYC data with exactly 10,847 POIs")
        else:
            print(f"✗ NYC data has {len(pois)} POIs, expected 10,847")
else:
    print("✗ NYC data file missing")

print(f"\n{'='*50}")
print(f"COMPLIANCE: {'PASS' if not missing else 'FAIL - Missing: ' + str(missing)}")