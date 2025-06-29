"""
Hybrid Planner: Unified Framework for Dynamic Itinerary Planning

Synthesizes all algorithms from research_context.md:
- Two-phase approach: Greedy selection (O(n²)) → A* routing
- LPA* overlay for dynamic updates
- Automatic algorithm selection based on problem characteristics
- R-trees for spatial indexing
- Caching for efficiency

Based on the "most promising approach" from research_context.md:
"Combines graph-based algorithms with machine learning enhancements"
"""

import json
import numpy as np
from typing import List, Dict, Set, Tuple, Optional, NamedTuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from enum import Enum
import pickle
from collections import defaultdict, deque
from functools import lru_cache
import threading
from rtree import index
import heapq

# Use absolute imports for sibling modules
from src.metrics_definitions import (
    POI, Itinerary, QuantitativeMetrics, QualitativeMetrics,
    CompositeUtilityFunctions
)
from src.greedy_algorithms import (
    GreedyPOISelection, HeapPrunGreedyPOI, GreedyPlanner,
    Constraints, InteractiveFeedback
)
from src.astar_itinerary import (
    AStarItineraryPlanner, MemoryBoundedAStarPlanner,
    ItineraryState, SearchNode, manhattan_distance_numba,
    compute_distance_matrix_numba, get_borough_id, NYC_BOROUGHS
)
from src.lpa_star import (
    LPAStarPlanner, DynamicUpdate, UpdateType
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlgorithmType(Enum):
    """Available algorithms in the hybrid framework"""
    GREEDY = "greedy"
    HEAP_GREEDY = "heap_greedy"
    ASTAR = "astar"
    SMA_STAR = "sma_star"
    LPA_STAR = "lpa_star"
    TWO_PHASE = "two_phase"
    AUTO = "auto"


class PlannerPhase(Enum):
    """Planning phases for two-phase approach"""
    SELECTION = "selection"  # Select POIs to visit
    ROUTING = "routing"      # Determine optimal order


@dataclass
class PlannerConfig:
    """Configuration for hybrid planner"""
    enable_caching: bool = True
    cache_ttl_seconds: int = 900  # 15 minutes as in research
    enable_parallel: bool = True
    max_workers: int = 4
    enable_rtree: bool = True
    enable_explanations: bool = True
    fallback_enabled: bool = True
    pareto_limit: int = 5  # Max Pareto-optimal alternatives


@dataclass
class PlanningResult:
    """Result from hybrid planning"""
    primary_itinerary: List[POI]
    alternatives: List[List[POI]] = field(default_factory=list)
    algorithm_used: str = ""
    phase_times: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    cache_hit: bool = False


class SpatialCache:
    """
    Spatial caching with R-tree indexing
    
    Based on research_context.md: "R-trees for spatial indexing"
    """
    
    def __init__(self, ttl_seconds: int = 900):
        self.ttl = ttl_seconds
        self.cache = {}
        self.spatial_index = index.Index()
        self.timestamps = {}
        self.lock = threading.RLock()
        self._next_id = 0
    
    def _get_bounds(self, constraints: Constraints, preferences: Dict) -> Tuple:
        """Create spatial bounds for cache key"""
        # Simplified bounds based on key parameters
        return (
            constraints.budget,
            constraints.max_pois,
            hash(frozenset(preferences.items())),
            constraints.transportation_mode
        )
    
    def get(self, key: str, bounds: Tuple) -> Optional[PlanningResult]:
        """Retrieve from cache if valid"""
        with self.lock:
            # Check temporal validity
            if key in self.timestamps:
                age = time.time() - self.timestamps[key]
                if age > self.ttl:
                    self._remove(key)
                    return None
            
            # Check spatial index
            candidates = list(self.spatial_index.intersection(bounds))
            for idx in candidates:
                if self.cache.get(idx, {}).get('key') == key:
                    logger.debug(f"Cache hit for key: {key}")
                    return self.cache[idx]['result']
            
            return None
    
    def put(self, key: str, bounds: Tuple, result: PlanningResult):
        """Store in cache with spatial indexing"""
        with self.lock:
            idx = self._next_id
            self._next_id += 1
            
            self.cache[idx] = {
                'key': key,
                'result': result
            }
            self.timestamps[key] = time.time()
            self.spatial_index.insert(idx, bounds)
    
    def _remove(self, key: str):
        """Remove expired entries"""
        with self.lock:
            # Find and remove from spatial index
            for idx, data in list(self.cache.items()):
                if data['key'] == key:
                    # Get bounds to remove from index
                    del self.cache[idx]
                    # Note: rtree doesn't support direct removal by id
            
            if key in self.timestamps:
                del self.timestamps[key]


class ExplanationGenerator:
    """
    Generate explanations for planning decisions
    
    Based on research findings for interpretability
    """
    
    @staticmethod
    def explain_algorithm_choice(algorithm: AlgorithmType, 
                               n_pois: int, constraints: Constraints) -> str:
        """Explain why an algorithm was chosen"""
        explanations = {
            AlgorithmType.GREEDY: (
                f"Using Greedy algorithm (O(n²)) for {n_pois} POIs. "
                "Fast and effective for small-medium problems."
            ),
            AlgorithmType.HEAP_GREEDY: (
                f"Using Heap-optimized Greedy for {n_pois} POIs. "
                "Efficient pruning for better performance."
            ),
            AlgorithmType.ASTAR: (
                f"Using A* search for optimal solution with {n_pois} POIs. "
                "Guarantees optimal solution with admissible heuristic."
            ),
            AlgorithmType.SMA_STAR: (
                f"Using memory-bounded SMA* for {n_pois} POIs. "
                "Large problem requires memory management."
            ),
            AlgorithmType.TWO_PHASE: (
                f"Using two-phase approach for {n_pois} POIs. "
                "Greedy selection followed by A* routing for best efficiency."
            )
        }
        
        base = explanations.get(algorithm, "Using hybrid approach.")
        
        # Add constraint-specific notes
        if constraints.budget < 50:
            base += " Note: Limited budget may restrict options."
        if constraints.transportation_mode == "walking":
            base += " Walking mode limits coverage area."
        
        return base
    
    @staticmethod
    def explain_poi_selection(selected: List[POI], rejected: List[POI],
                            preferences: Dict[str, float]) -> str:
        """Explain POI selection decisions"""
        explanation = "POI Selection:\n"
        
        # Top preferences
        top_prefs = sorted(preferences.items(), key=lambda x: x[1], reverse=True)[:3]
        explanation += f"Prioritizing: {', '.join(p[0] for p in top_prefs)}\n"
        
        # Selected POIs
        if selected:
            explanation += f"\nSelected {len(selected)} POIs:\n"
            for poi in selected[:5]:  # Show top 5
                explanation += f"- {poi.name} ({poi.category}): "
                explanation += f"Rating {poi.rating}, "
                explanation += f"{'Free' if poi.entrance_fee == 0 else f'${poi.entrance_fee}'}\n"
        
        # Rejected POIs (if any high-rated ones)
        if rejected:
            high_rated_rejected = [p for p in rejected if p.rating > 4.5][:3]
            if high_rated_rejected:
                explanation += f"\nNot selected (constraints):\n"
                for poi in high_rated_rejected:
                    explanation += f"- {poi.name}: "
                    if poi.entrance_fee > 50:
                        explanation += "Too expensive"
                    elif poi.avg_visit_duration > 3:
                        explanation += "Too time-consuming"
                    else:
                        explanation += "Distance/timing conflict"
                    explanation += "\n"
        
        return explanation
    
    @staticmethod
    def explain_route_optimization(original_order: List[POI], 
                                 optimized_order: List[POI],
                                 improvement: float) -> str:
        """Explain routing improvements"""
        if improvement < 0.01:
            return "Route already optimal, no reordering needed."
        
        explanation = f"Route Optimization (saved {improvement:.1f} km):\n"
        explanation += "Original: " + " → ".join(p.name for p in original_order[:5])
        if len(original_order) > 5:
            explanation += " → ..."
        explanation += "\n"
        
        explanation += "Optimized: " + " → ".join(p.name for p in optimized_order[:5])
        if len(optimized_order) > 5:
            explanation += " → ..."
        
        return explanation


class HybridPlanner:
    """
    Unified planning framework combining all algorithms
    
    Implements the "most promising approach" from research_context.md
    """
    
    def __init__(self, pois_data: List[Dict], 
                 distance_matrix: Optional[np.ndarray] = None,
                 config: Optional[PlannerConfig] = None):
        """Initialize hybrid planner with all algorithms"""
        self.config = config or PlannerConfig()
        
        # Store POI data
        self.pois_data = pois_data
        self.pois = [self._dict_to_poi(p) for p in pois_data]
        self.poi_map = {poi.id: poi for poi in self.pois}
        
        # Distance matrix
        if distance_matrix is not None:
            self.distance_matrix = distance_matrix.astype(np.float32)
        else:
            lats = np.array([poi.lat for poi in self.pois])
            lons = np.array([poi.lon for poi in self.pois])
            self.distance_matrix = compute_distance_matrix_numba(lats, lons)
        
        # Initialize component algorithms
        self.greedy_planner = GreedyPOISelection(pois_data, self.distance_matrix)
        self.heap_greedy_planner = HeapPrunGreedyPOI(pois_data, self.distance_matrix)
        self.astar_planner = AStarItineraryPlanner(pois_data, self.distance_matrix)
        self.sma_planner = MemoryBoundedAStarPlanner(pois_data, self.distance_matrix)
        self.lpa_planner = LPAStarPlanner(pois_data, self.distance_matrix)
        
        # R-tree spatial index
        if self.config.enable_rtree:
            self._build_spatial_index()
        
        # Caching
        if self.config.enable_caching:
            self.cache = SpatialCache(self.config.cache_ttl_seconds)
        else:
            self.cache = None
        
        # Explanation generator
        self.explainer = ExplanationGenerator()
        
        # Performance monitoring
        self.performance_stats = defaultdict(list)
        
        logger.info(f"Initialized HybridPlanner with {len(self.pois)} POIs")
    
    def _dict_to_poi(self, poi_dict: Dict) -> POI:
        """Convert dictionary to POI object"""
        return POI(
            id=poi_dict['id'],
            name=poi_dict['name'],
            lat=poi_dict['lat'],
            lon=poi_dict['lon'],
            category=poi_dict['category'],
            popularity=poi_dict.get('popularity', 0.5),
            entrance_fee=poi_dict.get('entrance_fee', 0.0),
            avg_visit_duration=poi_dict.get('avg_visit_duration', 1.5),
            opening_hours=self._parse_opening_hours(poi_dict.get('opening_hours', {})),
            rating=poi_dict.get('rating', 4.0)
        )
    
    def _parse_opening_hours(self, hours_dict: Dict) -> Tuple[float, float]:
        """Parse opening hours"""
        if 'weekday' in hours_dict:
            return tuple(hours_dict['weekday'])
        return (9.0, 20.0)
    
    def _build_spatial_index(self):
        """Build R-tree spatial index for POIs"""
        logger.info("Building R-tree spatial index")
        self.spatial_index = index.Index()
        
        for i, poi in enumerate(self.pois):
            # Insert as point (minx, miny, maxx, maxy)
            self.spatial_index.insert(
                i,
                (poi.lon, poi.lat, poi.lon, poi.lat),
                obj={'poi': poi, 'idx': i}
            )
    
    def plan(self, user_preferences: Dict[str, float],
            constraints: Constraints,
            feedback: Optional[InteractiveFeedback] = None,
            algorithm: AlgorithmType = AlgorithmType.AUTO,
            generate_alternatives: bool = True) -> PlanningResult:
        """
        Main planning interface
        
        Args:
            user_preferences: Category preferences
            constraints: Planning constraints
            feedback: User feedback from interactive planning
            algorithm: Algorithm to use (AUTO for automatic selection)
            generate_alternatives: Generate Pareto-optimal alternatives
            
        Returns:
            PlanningResult with primary itinerary and alternatives
        """
        start_time = time.time()
        
        # Check cache
        cache_key = self._generate_cache_key(user_preferences, constraints, feedback)
        if self.cache and algorithm != AlgorithmType.LPA_STAR:
            cache_bounds = self.cache._get_bounds(constraints, user_preferences)
            cached_result = self.cache.get(cache_key, cache_bounds)
            if cached_result:
                cached_result.cache_hit = True
                return cached_result
        
        # Select algorithm
        if algorithm == AlgorithmType.AUTO:
            algorithm = self._select_algorithm(constraints, feedback)
        
        # Generate explanation for algorithm choice
        explanation = self.explainer.explain_algorithm_choice(
            algorithm, len(self.pois), constraints
        )
        
        # Execute planning
        if algorithm == AlgorithmType.TWO_PHASE:
            result = self._two_phase_planning(
                user_preferences, constraints, feedback
            )
        else:
            result = self._single_algorithm_planning(
                algorithm, user_preferences, constraints, feedback
            )
        
        # Generate alternatives if requested
        if generate_alternatives:
            result.alternatives = self._generate_alternatives(
                user_preferences, constraints, feedback, result.primary_itinerary
            )
        
        # Add explanation
        result.explanation = explanation
        
        # Calculate metrics
        if result.primary_itinerary:
            itinerary = Itinerary(
                result.primary_itinerary,
                constraints.start_time,
                constraints.transportation_mode,
                user_preferences
            )
            result.metrics = self._calculate_metrics(itinerary, constraints)
        
        # Record performance
        total_time = time.time() - start_time
        result.phase_times['total'] = total_time
        self.performance_stats[algorithm.value].append(total_time)
        
        # Cache result
        if self.cache and not result.cache_hit:
            self.cache.put(cache_key, cache_bounds, result)
        
        logger.info(f"Planning completed in {total_time:.3f}s using {algorithm.value}")
        
        return result
    
    def _generate_cache_key(self, preferences: Dict, constraints: Constraints,
                          feedback: Optional[InteractiveFeedback]) -> str:
        """Generate cache key for planning request"""
        key_parts = [
            str(sorted(preferences.items())),
            f"{constraints.budget}_{constraints.max_pois}_{constraints.transportation_mode}",
            str(feedback.rejected_pois) if feedback else "none"
        ]
        return "|".join(key_parts)
    
    def _select_algorithm(self, constraints: Constraints,
                        feedback: Optional[InteractiveFeedback]) -> AlgorithmType:
        """
        Automatically select best algorithm based on problem characteristics
        
        Based on complexity analysis from research_context.md
        """
        n_pois = len(self.pois)
        
        # Filter by feedback
        if feedback and feedback.rejected_pois:
            n_pois -= len(feedback.rejected_pois)
        
        # Decision tree based on problem size and constraints
        if n_pois < 20:
            # Small problem: use optimal A*
            return AlgorithmType.ASTAR
        
        elif n_pois < 50:
            # Medium problem: use two-phase for efficiency
            return AlgorithmType.TWO_PHASE
        
        elif n_pois < 200:
            # Large problem: use heap-optimized greedy
            return AlgorithmType.HEAP_GREEDY
        
        elif n_pois < 1000:
            # Very large: memory-bounded A*
            return AlgorithmType.SMA_STAR
        
        else:
            # Huge problem: simple greedy
            return AlgorithmType.GREEDY
    
    def _two_phase_planning(self, user_preferences: Dict[str, float],
                          constraints: Constraints,
                          feedback: Optional[InteractiveFeedback]) -> PlanningResult:
        """
        Two-phase approach: Greedy selection → A* routing
        
        The "most promising approach" from research_context.md
        """
        logger.info("Starting two-phase planning")
        result = PlanningResult(primary_itinerary=[])
        
        # Phase 1: Greedy POI selection (O(n²))
        phase1_start = time.time()
        
        selected_pois = self.heap_greedy_planner.select_pois(
            user_preferences, constraints, feedback
        )
        
        result.phase_times['selection'] = time.time() - phase1_start
        logger.info(f"Phase 1: Selected {len(selected_pois)} POIs in "
                   f"{result.phase_times['selection']:.3f}s")
        
        if not selected_pois:
            result.algorithm_used = "two_phase_greedy_only"
            return result
        
        # Phase 2: A* routing optimization
        phase2_start = time.time()
        
        if len(selected_pois) <= 8:
            # Use exact A* for small selections
            routing_result = self._optimize_route_astar(
                selected_pois, constraints
            )
        else:
            # Use heuristic TSP solver for larger selections
            routing_result = self._optimize_route_heuristic(
                selected_pois, constraints
            )
        
        result.phase_times['routing'] = time.time() - phase2_start
        logger.info(f"Phase 2: Optimized route in {result.phase_times['routing']:.3f}s")
        
        result.primary_itinerary = routing_result
        result.algorithm_used = "two_phase"
        
        # Add routing explanation
        if len(selected_pois) != len(routing_result):
            distance_saved = self._calculate_route_improvement(
                selected_pois, routing_result
            )
            result.explanation += "\n\n" + self.explainer.explain_route_optimization(
                selected_pois, routing_result, distance_saved
            )
        
        return result
    
    def _single_algorithm_planning(self, algorithm: AlgorithmType,
                                 user_preferences: Dict[str, float],
                                 constraints: Constraints,
                                 feedback: Optional[InteractiveFeedback]) -> PlanningResult:
        """Execute planning with a single algorithm"""
        result = PlanningResult(primary_itinerary=[])
        
        if algorithm == AlgorithmType.GREEDY:
            result.primary_itinerary = self.greedy_planner.select_pois(
                user_preferences, constraints, feedback
            )
            
        elif algorithm == AlgorithmType.HEAP_GREEDY:
            result.primary_itinerary = self.heap_greedy_planner.select_pois(
                user_preferences, constraints, feedback
            )
            
        elif algorithm == AlgorithmType.ASTAR:
            result.primary_itinerary = self.astar_planner.plan_itinerary(
                user_preferences, constraints, feedback
            )
            
        elif algorithm == AlgorithmType.SMA_STAR:
            result.primary_itinerary = self.sma_planner.plan_itinerary(
                user_preferences, constraints, feedback
            )
            
        elif algorithm == AlgorithmType.LPA_STAR:
            result.primary_itinerary = self.lpa_planner.plan_with_updates(
                user_preferences, constraints
            )
        
        result.algorithm_used = algorithm.value
        return result
    
    def _optimize_route_astar(self, pois: List[POI], 
                            constraints: Constraints) -> List[POI]:
        """
        Use A* to find optimal visiting order
        
        For small POI sets where optimal solution is feasible
        """
        if len(pois) <= 1:
            return pois
        
        # Create restricted problem with selected POIs
        restricted_data = [p for p in self.pois_data if p['id'] in {poi.id for poi in pois}]
        poi_indices = {poi.id: i for i, poi in enumerate(pois)}
        
        # Extract submatrix
        n = len(pois)
        sub_distance = np.zeros((n, n))
        for i, poi1 in enumerate(pois):
            for j, poi2 in enumerate(pois):
                if i != j:
                    idx1 = self.greedy_planner.poi_index_map[poi1.id]
                    idx2 = self.greedy_planner.poi_index_map[poi2.id]
                    sub_distance[i, j] = self.distance_matrix[idx1, idx2]
        
        # Use A* to find optimal order
        temp_planner = AStarItineraryPlanner(restricted_data, sub_distance)
        
        # Create preferences that ensure all POIs are selected
        uniform_prefs = {cat: 1.0 for cat in set(poi.category for poi in pois)}
        
        # Adjust constraints to force selection of all POIs
        routing_constraints = Constraints(
            budget=float('inf'),  # Ignore budget for routing
            max_pois=len(pois),
            min_pois=len(pois),
            max_time_hours=constraints.max_time_hours,
            transportation_mode=constraints.transportation_mode
        )
        
        optimized = temp_planner.plan_itinerary(uniform_prefs, routing_constraints)
        
        # If A* found a route, use it; otherwise keep original order
        return optimized if optimized else pois
    
    def _optimize_route_heuristic(self, pois: List[POI],
                                constraints: Constraints) -> List[POI]:
        """
        Heuristic routing optimization for larger POI sets
        
        Uses nearest neighbor with 2-opt improvements
        """
        if len(pois) <= 1:
            return pois
        
        # Nearest neighbor construction
        unvisited = set(range(len(pois)))
        route = [0]  # Start with first POI
        unvisited.remove(0)
        
        while unvisited:
            current = route[-1]
            nearest = min(unvisited, key=lambda j: self._get_poi_distance(pois[current], pois[j]))
            route.append(nearest)
            unvisited.remove(nearest)
        
        # 2-opt improvement
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route)):
                    if j - i == 1:
                        continue
                    
                    # Calculate improvement
                    current_dist = (
                        self._get_poi_distance(pois[route[i-1]], pois[route[i]]) +
                        self._get_poi_distance(pois[route[j-1]], pois[route[j % len(route)]])
                    )
                    new_dist = (
                        self._get_poi_distance(pois[route[i-1]], pois[route[j-1]]) +
                        self._get_poi_distance(pois[route[i]], pois[route[j % len(route)]])
                    )
                    
                    if new_dist < current_dist:
                        # Reverse segment
                        route[i:j] = route[i:j][::-1]
                        improved = True
        
        return [pois[i] for i in route]
    
    def _get_poi_distance(self, poi1: POI, poi2: POI) -> float:
        """Get distance between two POIs"""
        idx1 = self.greedy_planner.poi_index_map[poi1.id]
        idx2 = self.greedy_planner.poi_index_map[poi2.id]
        return self.distance_matrix[idx1, idx2]
    
    def _calculate_route_improvement(self, original: List[POI], 
                                   optimized: List[POI]) -> float:
        """Calculate distance saved by route optimization"""
        def route_distance(route):
            total = 0.0
            for i in range(len(route) - 1):
                total += self._get_poi_distance(route[i], route[i + 1])
            return total
        
        original_dist = route_distance(original)
        optimized_dist = route_distance(optimized)
        
        return original_dist - optimized_dist
    
    def _generate_alternatives(self, user_preferences: Dict[str, float],
                             constraints: Constraints,
                             feedback: Optional[InteractiveFeedback],
                             primary: List[POI]) -> List[List[POI]]:
        """
        Generate Pareto-optimal alternative itineraries
        
        Based on research focus on multiple objective optimization
        """
        alternatives = []
        
        if self.config.enable_parallel:
            # Parallel generation of alternatives
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = []
                
                # Vary preferences
                for weight_factor in [0.8, 1.2]:
                    modified_prefs = {
                        k: v * weight_factor for k, v in user_preferences.items()
                    }
                    futures.append(
                        executor.submit(
                            self._generate_single_alternative,
                            modified_prefs, constraints, feedback, primary
                        )
                    )
                
                # Vary constraints
                if constraints.budget > 50:
                    budget_constraint = Constraints(**vars(constraints))
                    budget_constraint.budget *= 0.7
                    futures.append(
                        executor.submit(
                            self._generate_single_alternative,
                            user_preferences, budget_constraint, feedback, primary
                        )
                    )
                
                # Collect results
                for future in futures:
                    alt = future.result()
                    if alt and self._is_pareto_optimal(alt, primary, alternatives):
                        alternatives.append(alt)
        
        else:
            # Sequential generation
            for weight_factor in [0.8, 1.2]:
                modified_prefs = {
                    k: v * weight_factor for k, v in user_preferences.items()
                }
                alt = self._generate_single_alternative(
                    modified_prefs, constraints, feedback, primary
                )
                if alt and self._is_pareto_optimal(alt, primary, alternatives):
                    alternatives.append(alt)
        
        # Limit number of alternatives
        return alternatives[:self.config.pareto_limit]
    
    def _generate_single_alternative(self, preferences: Dict[str, float],
                                   constraints: Constraints,
                                   feedback: Optional[InteractiveFeedback],
                                   exclude: List[POI]) -> Optional[List[POI]]:
        """Generate a single alternative itinerary"""
        # Use different algorithm or parameters
        algorithm = AlgorithmType.HEAP_GREEDY  # Fast alternative
        
        result = self._single_algorithm_planning(
            algorithm, preferences, constraints, feedback
        )
        
        # Check if sufficiently different from primary
        if result.primary_itinerary:
            overlap = len(set(p.id for p in result.primary_itinerary) & 
                         set(p.id for p in exclude))
            if overlap < len(exclude) * 0.5:  # Less than 50% overlap
                return result.primary_itinerary
        
        return None
    
    def _is_pareto_optimal(self, candidate: List[POI], primary: List[POI],
                         others: List[List[POI]]) -> bool:
        """Check if candidate is Pareto-optimal"""
        # Simplified check: different enough from existing
        for existing in [primary] + others:
            overlap = len(set(p.id for p in candidate) & set(p.id for p in existing))
            if overlap > len(candidate) * 0.7:  # Too similar
                return False
        
        return True
    
    def _calculate_metrics(self, itinerary: Itinerary, 
                         constraints: Constraints) -> Dict[str, float]:
        """Calculate comprehensive metrics for itinerary"""
        metrics = {}
        
        # Quantitative metrics
        metrics['total_distance'] = QuantitativeMetrics.total_distance(itinerary)
        metrics['total_time'] = QuantitativeMetrics.total_time(itinerary)
        metrics['total_cost'] = QuantitativeMetrics.total_cost(itinerary)
        metrics['utility_per_time'] = QuantitativeMetrics.utility_per_time(itinerary)
        
        # Qualitative metrics
        metrics['satisfaction'] = QualitativeMetrics.user_satisfaction(
            itinerary, itinerary.user_preferences
        )
        metrics['diversity'] = QualitativeMetrics.diversity_score(itinerary)
        metrics['novelty'] = QualitativeMetrics.novelty_score(itinerary)
        
        # Composite score
        metrics['css_score'] = CompositeUtilityFunctions.composite_satisfaction_score(
            itinerary, itinerary.user_preferences,
            budget=constraints.budget, max_time=constraints.max_time_hours
        )
        
        return metrics
    
    def apply_dynamic_update(self, update: DynamicUpdate) -> PlanningResult:
        """
        Apply dynamic update using LPA*
        
        Demonstrates efficiency of incremental replanning
        """
        logger.info(f"Applying dynamic update: {update.update_type.value}")
        
        # Use LPA* for efficient replanning
        self.lpa_planner.apply_update(update)
        new_path = self.lpa_planner.compute_shortest_path()
        
        result = PlanningResult(
            primary_itinerary=new_path or [],
            algorithm_used="lpa_star",
            explanation=f"Replanned after {update.update_type.value}"
        )
        
        # Add performance metrics
        stats = self.lpa_planner.get_statistics()
        result.metrics['computation_reuse'] = stats['computation_reuse']
        result.metrics['nodes_updated'] = stats['nodes_updated']
        
        return result
    
    def explain_decision(self, itinerary: List[POI], 
                       preferences: Dict[str, float],
                       constraints: Constraints) -> str:
        """
        Generate detailed explanation for planning decisions
        """
        if not itinerary:
            return "No feasible itinerary found with given constraints."
        
        explanation = "Planning Decision Explanation:\n\n"
        
        # Explain selections
        selected = itinerary
        all_pois = self.pois
        rejected = [p for p in all_pois if p not in selected]
        
        explanation += self.explainer.explain_poi_selection(
            selected, rejected, preferences
        )
        
        # Explain constraints impact
        explanation += "\n\nConstraint Analysis:\n"
        
        total_cost = sum(p.entrance_fee for p in itinerary)
        explanation += f"- Budget usage: ${total_cost:.0f} / ${constraints.budget:.0f} "
        explanation += f"({100 * total_cost / constraints.budget:.0f}%)\n"
        
        total_time = sum(p.avg_visit_duration for p in itinerary) + \
                    len(itinerary) * 0.5  # Approximate travel
        explanation += f"- Time usage: {total_time:.1f}h / {constraints.max_time_hours:.0f}h "
        explanation += f"({100 * total_time / constraints.max_time_hours:.0f}%)\n"
        
        explanation += f"- POIs selected: {len(itinerary)} "
        explanation += f"(min: {constraints.min_pois}, max: {constraints.max_pois})\n"
        
        # Borough distribution
        boroughs = defaultdict(int)
        for poi in itinerary:
            borough_id = get_borough_id(poi.lat, poi.lon)
            borough_names = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
            if 0 <= borough_id <= 4:
                boroughs[borough_names[borough_id]] += 1
        
        if boroughs:
            explanation += f"- Borough coverage: {', '.join(f'{b}: {c}' for b, c in boroughs.items())}\n"
        
        return explanation
    
    def get_performance_summary(self) -> Dict:
        """Get performance statistics summary"""
        summary = {}
        
        for algorithm, times in self.performance_stats.items():
            if times:
                summary[algorithm] = {
                    'count': len(times),
                    'avg_time': np.mean(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'std_time': np.std(times)
                }
        
        return summary


# Fallback coordinator for robustness
class FallbackCoordinator:
    """
    Coordinate fallback mechanisms when primary algorithms fail
    
    Ensures robustness in edge cases
    """
    
    def __init__(self, planner: HybridPlanner):
        self.planner = planner
        self.fallback_chain = [
            AlgorithmType.TWO_PHASE,
            AlgorithmType.HEAP_GREEDY,
            AlgorithmType.GREEDY
        ]
    
    def plan_with_fallback(self, user_preferences: Dict[str, float],
                         constraints: Constraints,
                         feedback: Optional[InteractiveFeedback] = None) -> PlanningResult:
        """Plan with automatic fallback on failure"""
        result = None
        
        for algorithm in self.fallback_chain:
            try:
                result = self.planner.plan(
                    user_preferences, constraints, feedback,
                    algorithm=algorithm, generate_alternatives=False
                )
                
                if result.primary_itinerary:
                    logger.info(f"Successful planning with {algorithm.value}")
                    return result
                    
            except Exception as e:
                logger.warning(f"Algorithm {algorithm.value} failed: {e}")
                continue
        
        # Last resort: return empty itinerary with explanation
        return PlanningResult(
            primary_itinerary=[],
            algorithm_used="none",
            explanation="Unable to find feasible itinerary with any algorithm. "
                       "Consider relaxing constraints."
        )


# Unit tests for integration challenges
import unittest


class TestHybridPlanner(unittest.TestCase):
    """Test cases for hybrid planning integration"""
    
    def setUp(self):
        """Set up test data"""
        self.test_pois = [
            {
                'id': f'poi_{i}',
                'name': f'POI {i}',
                'lat': 40.7 + (i % 10) * 0.01,
                'lon': -74.0 + (i // 10) * 0.01,
                'category': ['museum', 'park', 'restaurant', 'landmark'][i % 4],
                'popularity': 0.5 + (i % 5) * 0.1,
                'entrance_fee': [0, 15, 25, 30][i % 4],
                'avg_visit_duration': 1.0 + (i % 3) * 0.5,
                'opening_hours': {'weekday': [9, 20]},
                'rating': 4.0 + (i % 5) * 0.2
            }
            for i in range(20)
        ]
        
        self.planner = HybridPlanner(self.test_pois)
        self.user_prefs = {
            'museum': 0.8,
            'park': 0.7,
            'restaurant': 0.6,
            'landmark': 0.5
        }
        self.constraints = Constraints(budget=100, max_pois=5)
    
    def test_algorithm_selection(self):
        """Test automatic algorithm selection"""
        # Small problem
        small_planner = HybridPlanner(self.test_pois[:10])
        algorithm = small_planner._select_algorithm(self.constraints, None)
        self.assertEqual(algorithm, AlgorithmType.ASTAR)
        
        # Large problem
        large_pois = self.test_pois * 50  # 1000 POIs
        large_planner = HybridPlanner(large_pois)
        algorithm = large_planner._select_algorithm(self.constraints, None)
        self.assertEqual(algorithm, AlgorithmType.GREEDY)
    
    def test_two_phase_approach(self):
        """Test two-phase planning"""
        result = self.planner.plan(
            self.user_prefs, self.constraints,
            algorithm=AlgorithmType.TWO_PHASE
        )
        
        self.assertGreater(len(result.primary_itinerary), 0)
        self.assertEqual(result.algorithm_used, "two_phase")
        self.assertIn('selection', result.phase_times)
        self.assertIn('routing', result.phase_times)
    
    def test_algorithm_disagreement(self):
        """Test handling when algorithms produce different results"""
        # Get results from different algorithms
        greedy_result = self.planner.plan(
            self.user_prefs, self.constraints,
            algorithm=AlgorithmType.GREEDY,
            generate_alternatives=False
        )
        
        astar_result = self.planner.plan(
            self.user_prefs, self.constraints,
            algorithm=AlgorithmType.ASTAR,
            generate_alternatives=False
        )
        
        # Should both produce valid results
        self.assertGreater(len(greedy_result.primary_itinerary), 0)
        self.assertGreater(len(astar_result.primary_itinerary), 0)
        
        # A* should generally produce better or equal CSS score
        greedy_css = greedy_result.metrics.get('css_score', 0)
        astar_css = astar_result.metrics.get('css_score', 0)
        self.assertGreaterEqual(astar_css, greedy_css * 0.9)  # Allow 10% tolerance
    
    def test_resource_exhaustion(self):
        """Test coordination when resources are exhausted"""
        # Very tight constraints
        tight_constraints = Constraints(
            budget=10,  # Very low budget
            max_pois=2,
            max_time_hours=1.0  # Very short time
        )
        
        result = self.planner.plan(self.user_prefs, tight_constraints)
        
        # Should still return something (possibly empty)
        self.assertIsNotNone(result)
        self.assertLessEqual(len(result.primary_itinerary), 2)
    
    def test_borough_based_selection(self):
        """Test NYC-specific borough-based algorithm selection"""
        # Add Staten Island POIs (far from Manhattan)
        staten_pois = [
            {
                'id': f'staten_{i}',
                'name': f'Staten Island POI {i}',
                'lat': 40.5 + i * 0.01,
                'lon': -74.15 + i * 0.01,
                'category': 'park',
                'popularity': 0.6,
                'entrance_fee': 0.0,
                'avg_visit_duration': 1.5,
                'opening_hours': {'weekday': [9, 20]},
                'rating': 4.2
            }
            for i in range(5)
        ]
        
        mixed_planner = HybridPlanner(self.test_pois + staten_pois)
        
        # Plan with walking constraint (should avoid Staten Island)
        walking_constraints = Constraints(
            budget=100,
            max_pois=5,
            transportation_mode='walking',
            max_walking_distance_km=2.0
        )
        
        result = mixed_planner.plan(self.user_prefs, walking_constraints)
        
        # Check no Staten Island POIs selected
        for poi in result.primary_itinerary:
            self.assertNotIn('staten_', poi.id)
    
    def test_fallback_mechanism(self):
        """Test fallback coordinator"""
        coordinator = FallbackCoordinator(self.planner)
        
        # Normal case
        result = coordinator.plan_with_fallback(
            self.user_prefs, self.constraints
        )
        self.assertGreater(len(result.primary_itinerary), 0)
        
        # Impossible constraints
        impossible = Constraints(
            budget=1,  # $1 budget
            max_pois=10,  # Want 10 POIs
            max_time_hours=0.1  # 6 minutes
        )
        
        result = coordinator.plan_with_fallback(
            self.user_prefs, impossible
        )
        # Should return empty with explanation
        self.assertEqual(len(result.primary_itinerary), 0)
        self.assertIn("Unable to find", result.explanation)
    
    def test_caching_efficiency(self):
        """Test caching improves performance"""
        # First call (cache miss)
        start1 = time.time()
        result1 = self.planner.plan(self.user_prefs, self.constraints)
        time1 = time.time() - start1
        self.assertFalse(result1.cache_hit)
        
        # Second call (cache hit)
        start2 = time.time()
        result2 = self.planner.plan(self.user_prefs, self.constraints)
        time2 = time.time() - start2
        self.assertTrue(result2.cache_hit)
        
        # Cache should be faster
        self.assertLess(time2, time1 * 0.5)  # At least 2x faster
    
    def test_pareto_alternatives(self):
        """Test Pareto-optimal alternative generation"""
        result = self.planner.plan(
            self.user_prefs, self.constraints,
            generate_alternatives=True
        )
        
        # Should have primary and alternatives
        self.assertGreater(len(result.primary_itinerary), 0)
        self.assertGreater(len(result.alternatives), 0)
        
        # Alternatives should be different
        primary_ids = set(p.id for p in result.primary_itinerary)
        for alt in result.alternatives:
            alt_ids = set(p.id for p in alt)
            overlap = len(primary_ids & alt_ids) / len(primary_ids)
            self.assertLess(overlap, 0.7)  # Less than 70% overlap
    
    def test_explanation_generation(self):
        """Test explanation generation"""
        result = self.planner.plan(self.user_prefs, self.constraints)
        
        # Should have explanations
        self.assertIn("Using", result.explanation)
        
        # Get detailed explanation
        detailed = self.planner.explain_decision(
            result.primary_itinerary,
            self.user_prefs,
            self.constraints
        )
        
        self.assertIn("Budget usage", detailed)
        self.assertIn("Time usage", detailed)
        self.assertIn("Selected", detailed)
    
    def test_performance_monitoring(self):
        """Test performance statistics collection"""
        # Run multiple plans
        for _ in range(5):
            self.planner.plan(
                self.user_prefs, self.constraints,
                algorithm=AlgorithmType.GREEDY
            )
        
        stats = self.planner.get_performance_summary()
        
        self.assertIn('greedy', stats)
        self.assertEqual(stats['greedy']['count'], 5)
        self.assertGreater(stats['greedy']['avg_time'], 0)


def demonstrate_hybrid_planner():
    """Demonstrate hybrid planner capabilities"""
    print("=== Hybrid Planner Demonstration ===\n")
    
    # Load or create example data
    import os
    if os.path.exists('data/nyc_pois.json'):
        with open('data/nyc_pois.json', 'r') as f:
            pois_data = json.load(f)[:30]  # Use subset
    else:
        # Create example data
        pois_data = [
            {
                'id': 'met',
                'name': 'Metropolitan Museum',
                'lat': 40.7794,
                'lon': -73.9632,
                'category': 'museum',
                'popularity': 0.9,
                'entrance_fee': 25.0,
                'avg_visit_duration': 3.0,
                'opening_hours': {'weekday': [10, 17]},
                'rating': 4.8
            },
            {
                'id': 'central_park',
                'name': 'Central Park',
                'lat': 40.7829,
                'lon': -73.9654,
                'category': 'park',
                'popularity': 0.95,
                'entrance_fee': 0.0,
                'avg_visit_duration': 2.0,
                'opening_hours': {'weekday': [6, 22]},
                'rating': 4.9
            },
            {
                'id': 'times_square',
                'name': 'Times Square',
                'lat': 40.7580,
                'lon': -73.9855,
                'category': 'landmark',
                'popularity': 0.98,
                'entrance_fee': 0.0,
                'avg_visit_duration': 0.5,
                'opening_hours': {'weekday': [0, 24]},
                'rating': 4.3
            }
        ]
    
    # Initialize planner
    planner = HybridPlanner(pois_data)
    
    # User preferences
    user_prefs = {
        'museum': 0.9,
        'park': 0.8,
        'landmark': 0.7,
        'restaurant': 0.6
    }
    
    # Test 1: Automatic algorithm selection
    print("1. Automatic Algorithm Selection:")
    constraints = Constraints(budget=150, max_pois=5)
    
    result = planner.plan(user_prefs, constraints)
    print(f"Algorithm used: {result.algorithm_used}")
    print(f"POIs selected: {len(result.primary_itinerary)}")
    for i, poi in enumerate(result.primary_itinerary):
        print(f"  {i+1}. {poi.name} ({poi.category})")
    print(f"CSS Score: {result.metrics.get('css_score', 0):.3f}")
    print(f"Total time: {result.phase_times.get('total', 0):.3f}s")
    
    # Test 2: Two-phase approach
    print("\n2. Two-Phase Planning:")
    result2 = planner.plan(
        user_prefs, constraints,
        algorithm=AlgorithmType.TWO_PHASE
    )
    print(f"Selection phase: {result2.phase_times.get('selection', 0):.3f}s")
    print(f"Routing phase: {result2.phase_times.get('routing', 0):.3f}s")
    print(f"Total POIs: {len(result2.primary_itinerary)}")
    
    # Test 3: Dynamic update
    print("\n3. Dynamic Update with LPA*:")
    if result.primary_itinerary:
        # Close first POI
        update = DynamicUpdate(
            update_type=UpdateType.POI_CLOSED,
            poi_ids=[result.primary_itinerary[0].id],
            timestamp=datetime.now()
        )
        
        update_result = planner.apply_dynamic_update(update)
        print(f"Replanned after closing {result.primary_itinerary[0].name}")
        print(f"Computation reuse: {update_result.metrics.get('computation_reuse', 0):.1%}")
        print(f"New itinerary: {len(update_result.primary_itinerary)} POIs")
    
    # Test 4: Alternative generation
    print("\n4. Pareto-Optimal Alternatives:")
    result3 = planner.plan(
        user_prefs, constraints,
        generate_alternatives=True
    )
    print(f"Primary itinerary: {len(result3.primary_itinerary)} POIs")
    print(f"Alternatives generated: {len(result3.alternatives)}")
    for i, alt in enumerate(result3.alternatives[:3]):
        print(f"  Alternative {i+1}: {len(alt)} POIs")
    
    # Test 5: Performance summary
    print("\n5. Performance Summary:")
    summary = planner.get_performance_summary()
    for algo, stats in summary.items():
        print(f"{algo}: {stats['count']} runs, avg {stats['avg_time']:.3f}s")


def main():
    """Command-line interface for hybrid planner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NYC Itinerary Planner')
    parser.add_argument('--demo', action='store_true', help='Run demonstration')
    parser.add_argument('--test', action='store_true', help='Run unit tests')
    parser.add_argument('--algorithm', choices=['greedy', 'heap_greedy', 'astar', 'lpa_star', 'auto'],
                        default='auto', help='Algorithm to use')
    parser.add_argument('--pois', type=int, default=5, help='Number of POIs to include')
    parser.add_argument('--budget', type=float, default=200, help='Budget in USD')
    parser.add_argument('--duration', type=float, default=8, help='Duration in hours')
    
    args = parser.parse_args()
    
    if args.demo:
        demonstrate_hybrid_planner()
    elif args.test:
        print("\n=== Running Unit Tests ===")
        unittest.main(argv=[''], exit=False, verbosity=2)
    else:
        # Run planning with specified parameters
        print(f"Planning itinerary with {args.algorithm} algorithm...")
        print(f"Target: {args.pois} POIs, ${args.budget} budget, {args.duration} hours")
        # Implementation would go here
        print("Interactive planning not yet implemented. Use --demo for demonstration.")


if __name__ == "__main__":
    main()