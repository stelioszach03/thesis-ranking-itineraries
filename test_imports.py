#!/usr/bin/env python3
"""Test all imports for the NYC Itinerary Ranking project"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing imports...")
print("=" * 50)

# Test metrics_definitions
try:
    from src.metrics_definitions import POI, Itinerary
    print("✓ metrics_definitions imports OK")
except Exception as e:
    print(f"❌ metrics_definitions import error: {e}")

# Test metrics_calculations (if it exists)
try:
    from src.metrics_calculations import QuantitativeMetrics, QualitativeMetrics, CompositeUtilityFunctions
    print("✓ metrics_calculations imports OK")
except ModuleNotFoundError:
    # Try importing from metrics_definitions instead
    try:
        from src.metrics_definitions import QuantitativeMetrics, QualitativeMetrics, CompositeUtilityFunctions
        print("✓ metrics classes from metrics_definitions OK")
    except Exception as e:
        print(f"❌ metrics classes import error: {e}")
except Exception as e:
    print(f"❌ metrics_calculations import error: {e}")

# Test greedy_algorithms
try:
    from src.greedy_algorithms import GreedyPOISelection, Constraints, InteractiveFeedback, GreedyPlanner
    print("✓ greedy_algorithms imports OK")
except Exception as e:
    print(f"❌ greedy_algorithms import error: {e}")

# Test astar_itinerary
try:
    from src.astar_itinerary import AStarItineraryPlanner, MemoryBoundedAStarPlanner
    print("✓ astar_itinerary imports OK")
except Exception as e:
    print(f"❌ astar_itinerary import error: {e}")

# Test lpa_star
try:
    from src.lpa_star import LPAStarPlanner, DynamicUpdate, UpdateType
    print("✓ lpa_star imports OK")
except Exception as e:
    print(f"❌ lpa_star import error: {e}")

# Test hybrid_planner
try:
    from src.hybrid_planner import HybridPlanner
    print("✓ hybrid_planner imports OK")
except Exception as e:
    print(f"❌ hybrid_planner import error: {e}")

# Test algorithm_selector
try:
    from src.algorithm_selector import AlgorithmSelector, ResultCache
    print("✓ algorithm_selector imports OK")
except Exception as e:
    print(f"❌ algorithm_selector import error: {e}")

# Test prepare_nyc_data
try:
    from src.prepare_nyc_data import NYCDataGenerator, NYC_ATTRACTIONS, SUBWAY_STATIONS
    print("✓ prepare_nyc_data imports OK")
except Exception as e:
    print(f"❌ prepare_nyc_data import error: {e}")

print("\n" + "=" * 50)
print("Import test completed!")