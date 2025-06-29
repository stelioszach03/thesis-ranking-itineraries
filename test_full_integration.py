import sys
sys.path.insert(0, 'src')

from metrics_definitions import POI
from greedy_algorithms import GreedyPOISelection, Constraints
from hybrid_planner import HybridPlanner
import numpy as np

print("Testing full integration...")

# Load real data
import json
try:
    with open('data/nyc_data/nyc_pois.json', 'r') as f:
        pois_data = json.load(f)
    distance_matrix = np.load('data/nyc_data/distance_matrix.npy')
    print(f"✓ Loaded {len(pois_data)} POIs")
    
    # Test hybrid planner
    planner = HybridPlanner(pois_data, distance_matrix)
    constraints = Constraints(budget=100, max_time_hours=6, min_pois=3, max_pois=5)
    preferences = {'museum': 0.8, 'park': 0.7, 'restaurant': 0.6}
    
    result = planner.plan(preferences, constraints)
    print(f"✓ Generated itinerary with {len(result.primary_itinerary)} POIs")
    print(f"✓ CSS Score: {result.metrics.get('css_score', 0):.3f}")
    print("✅ Full integration test PASSED!")
    
except Exception as e:
    print(f"❌ Integration test failed: {e}")
    import traceback
    traceback.print_exc()