import sys
sys.path.insert(0, 'src')

from greedy_algorithms import GreedyPOISelection, Constraints
import json
import numpy as np

print("Testing simple greedy algorithm...")

try:
    # Load data
    with open('data/nyc_data/nyc_pois.json', 'r') as f:
        pois_data = json.load(f)
    distance_matrix = np.load('data/nyc_data/distance_matrix.npy')
    print(f"✓ Loaded {len(pois_data)} POIs")
    
    # Test greedy algorithm with small subset
    subset_pois = pois_data[:100]  # Use only first 100 POIs
    subset_matrix = distance_matrix[:100, :100]
    
    algorithm = GreedyPOISelection(subset_pois, subset_matrix)
    constraints = Constraints(budget=50, max_time_hours=4, min_pois=2, max_pois=4)
    
    itinerary = algorithm.select_pois(
        user_preferences={'museum': 0.8, 'park': 0.7},
        constraints=constraints
    )
    
    print(f"✓ Generated itinerary with {len(itinerary)} POIs")
    print(f"  Total cost: ${sum(poi.entrance_fee for poi in itinerary)}")
    print("✅ Simple integration test PASSED!")
    
except Exception as e:
    print(f"❌ Integration test failed: {e}")
    import traceback
    traceback.print_exc()