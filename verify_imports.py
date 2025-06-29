import sys
print(f"Python: {sys.version}")

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy: {e}")

try:
    import pandas as pd
    print(f"✓ Pandas {pd.__version__}")
except ImportError as e:
    print(f"✗ Pandas: {e}")

try:
    import networkx as nx
    print(f"✓ NetworkX {nx.__version__}")
except ImportError as e:
    print(f"✗ NetworkX: {e}")

try:
    import pytest
    print(f"✓ Pytest {pytest.__version__}")
except ImportError as e:
    print(f"✗ Pytest: {e}")

# Test project imports
sys.path.insert(0, 'src')
try:
    from metrics_definitions import POI, Itinerary
    from greedy_algorithms import GreedyPOISelection
    from astar_itinerary import AStarItineraryPlanner
    from lpa_star import LPAStarPlanner
    from hybrid_planner import HybridItineraryPlanner
    print("✓ All project imports successful!")
except ImportError as e:
    print(f"✗ Project import error: {e}")