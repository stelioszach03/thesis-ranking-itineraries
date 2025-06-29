# NYC Itinerary Ranking - Core Algorithms Package
"""
Core algorithms for quality-based itinerary planning in NYC.

Modules:
- metrics_definitions: CSS and component metrics
- greedy_algorithms: O(n²) heuristic selection
- astar_itinerary: Optimal pathfinding with admissible heuristics
- lpa_star: Dynamic replanning with incremental updates
- hybrid_planner: Two-phase approach combining greedy and A*
- prepare_nyc_data: Data pipeline for NYC POIs
"""

__version__ = "1.0.0"
__author__ = "Stelios Zacharioudakis"
__email__ = "your.email@example.com"

from src.metrics_definitions import (
    POI, Itinerary, QuantitativeMetrics, 
    QualitativeMetrics, CompositeUtilityFunctions
)
from src.greedy_algorithms import (
    GreedyPOISelection, HeapPrunGreedyPOI, 
    Constraints, InteractiveFeedback
)
from src.astar_itinerary import AStarItineraryPlanner
from src.lpa_star import LPAStarPlanner, DynamicUpdate, UpdateType
from src.hybrid_planner import HybridPlanner

__all__ = [
    'POI', 'Itinerary', 'QuantitativeMetrics', 
    'QualitativeMetrics', 'CompositeUtilityFunctions',
    'GreedyPOISelection', 'HeapPrunGreedyPOI',
    'Constraints', 'InteractiveFeedback',
    'AStarItineraryPlanner', 'LPAStarPlanner',
    'DynamicUpdate', 'UpdateType', 'HybridPlanner'
]