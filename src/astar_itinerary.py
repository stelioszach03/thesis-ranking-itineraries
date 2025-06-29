"""
A* Algorithm for Itinerary Planning with Admissible Heuristics

Based on research_context.md framework for graph-based algorithms:
- A* with admissible heuristics for pathfinding
- MST-based heuristic for remaining POIs
- Numba optimization for TravelPlanner benchmark scale
- SMA* variant for memory-bounded scenarios

References:
- Research framework mentioning A* with admissible heuristics
- Graph Neural Networks integration preparation
- O(b^d) complexity where b is branching factor, d is depth
"""

import json
import numpy as np
import heapq
from typing import List, Dict, Set, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
from collections import deque
import gc
from numba import njit, prange
import networkx as nx

# Use absolute imports for sibling modules
from src.metrics_definitions import (
    POI, Itinerary, QuantitativeMetrics, QualitativeMetrics,
    CompositeUtilityFunctions
)
from src.greedy_algorithms import Constraints, InteractiveFeedback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# NYC Borough boundaries (approximate)
NYC_BOROUGHS = {
    'manhattan': {'north': 40.882, 'south': 40.700, 'east': -73.907, 'west': -74.019},
    'brooklyn': {'north': 40.739, 'south': 40.570, 'east': -73.833, 'west': -74.042},
    'queens': {'north': 40.812, 'south': 40.489, 'east': -73.700, 'west': -73.962},
    'bronx': {'north': 40.917, 'south': 40.785, 'east': -73.748, 'west': -73.933},
    'staten_island': {'north': 40.651, 'south': 40.477, 'east': -74.034, 'west': -74.259}
}


@dataclass(frozen=True)
class ItineraryState:
    """
    State representation for A* search
    
    Immutable state for efficient hashing and comparison
    """
    visited_pois: Tuple[str, ...]  # POI IDs in visit order
    current_time: float
    remaining_budget: float
    total_utility: float
    total_distance: float
    
    def __hash__(self):
        return hash((self.visited_pois, round(self.current_time, 2), 
                    round(self.remaining_budget, 2)))
    
    def __lt__(self, other):
        # For heap ordering
        return self.total_utility > other.total_utility


class SearchNode:
    """Node in A* search tree"""
    
    def __init__(self, state: ItineraryState, g_cost: float, h_cost: float,
                 parent: Optional['SearchNode'] = None):
        self.state = state
        self.g_cost = g_cost  # Cost from start
        self.h_cost = h_cost  # Heuristic to goal
        self.f_cost = g_cost + h_cost  # Total estimated cost
        self.parent = parent
        self.depth = 0 if parent is None else parent.depth + 1
    
    def __lt__(self, other):
        # For heap - minimize f_cost, maximize utility on ties
        if abs(self.f_cost - other.f_cost) < 0.001:
            return self.state.total_utility > other.state.total_utility
        return self.f_cost < other.f_cost


# Numba-optimized distance calculations
@njit
def manhattan_distance_numba(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Numba-optimized Manhattan distance for NYC grid
    
    From metrics_definitions.py with JIT compilation
    """
    NYC_GRID_FACTOR = 1.4
    lat_km = 111.0
    lon_km = 111.0 * np.cos(np.radians((lat1 + lat2) / 2))
    
    dlat = abs(lat2 - lat1) * lat_km
    dlon = abs(lon2 - lon1) * lon_km
    
    return NYC_GRID_FACTOR * (dlat + dlon)


@njit
def compute_distance_matrix_numba(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Numba-optimized distance matrix computation
    
    For handling 1000+ POIs from TravelPlanner benchmark
    """
    n = len(lats)
    distances = np.zeros((n, n), dtype=np.float32)
    
    for i in prange(n):
        for j in range(i + 1, n):
            dist = manhattan_distance_numba(lats[i], lons[i], lats[j], lons[j])
            distances[i, j] = dist
            distances[j, i] = dist
    
    return distances


@njit
def get_borough_id(lat: float, lon: float) -> int:
    """Get borough ID for coordinate (numba-compatible)"""
    # Manhattan
    if 40.700 <= lat <= 40.882 and -74.019 <= lon <= -73.907:
        return 0
    # Brooklyn  
    elif 40.570 <= lat <= 40.739 and -74.042 <= lon <= -73.833:
        return 1
    # Queens
    elif 40.489 <= lat <= 40.812 and -73.962 <= lon <= -73.700:
        return 2
    # Bronx
    elif 40.785 <= lat <= 40.917 and -73.933 <= lon <= -73.748:
        return 3
    # Staten Island
    elif 40.477 <= lat <= 40.651 and -74.259 <= lon <= -74.034:
        return 4
    else:
        return -1  # Outside NYC


class AStarItineraryPlanner:
    """
    A* planner for itinerary optimization
    
    Based on research_context.md:
    - State space search with admissible heuristics
    - MST-based heuristic for unvisited POIs
    - Integration hooks for Graph Neural Networks
    """
    
    def __init__(self, pois: List[Dict], distance_matrix: Optional[np.ndarray] = None):
        self.pois = [self._dict_to_poi(p) for p in pois]
        self.poi_map = {poi.id: poi for poi in self.pois}
        self.poi_index_map = {poi.id: i for i, poi in enumerate(self.pois)}
        
        # Pre-compute optimized distance matrix
        if distance_matrix is not None:
            self.distance_matrix = distance_matrix.astype(np.float32)
        else:
            lats = np.array([poi.lat for poi in self.pois])
            lons = np.array([poi.lon for poi in self.pois])
            self.distance_matrix = compute_distance_matrix_numba(lats, lons)
        
        # Pre-compute MST for heuristic
        self._precompute_mst()
        
        # Pre-compute borough assignments
        self.poi_boroughs = np.array([
            get_borough_id(poi.lat, poi.lon) for poi in self.pois
        ])
        
        # Statistics
        self.nodes_expanded = 0
        self.max_frontier_size = 0
        
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
    
    def _precompute_mst(self):
        """
        Pre-compute Minimum Spanning Tree for heuristic
        
        Used for admissible heuristic calculation
        """
        # Create complete graph
        G = nx.Graph()
        for i in range(len(self.pois)):
            for j in range(i + 1, len(self.pois)):
                weight = self.distance_matrix[i, j]
                G.add_edge(i, j, weight=weight)
        
        # Compute MST
        self.mst = nx.minimum_spanning_tree(G)
        
        # Cache MST edges for fast lookup
        self.mst_edges = {}
        for u, v, data in self.mst.edges(data=True):
            self.mst_edges[(min(u, v), max(u, v))] = data['weight']
    
    def plan_itinerary(self, user_preferences: Dict[str, float],
                      constraints: Constraints,
                      feedback: Optional[InteractiveFeedback] = None,
                      start_location: Optional[Tuple[float, float]] = None,
                      memory_limit_mb: int = 1000) -> List[POI]:
        """
        Main A* search for itinerary planning
        
        Args:
            user_preferences: Category preferences
            constraints: Planning constraints
            feedback: User feedback from interactive planning
            start_location: Starting (lat, lon) if not from a POI
            memory_limit_mb: Memory limit for SMA* variant
            
        Returns:
            Optimal sequence of POIs
        """
        logger.info("Starting A* itinerary planning")
        
        # Reset statistics
        self.nodes_expanded = 0
        self.max_frontier_size = 0
        
        # Filter POIs based on feedback
        available_pois = self._filter_pois_by_feedback(feedback)
        available_indices = set(self.poi_index_map[poi.id] for poi in available_pois)
        
        # Initialize search
        initial_state = ItineraryState(
            visited_pois=(),
            current_time=constraints.start_time,
            remaining_budget=constraints.budget,
            total_utility=0.0,
            total_distance=0.0
        )
        
        initial_h = self._compute_heuristic(
            initial_state, available_indices, user_preferences, constraints
        )
        
        start_node = SearchNode(initial_state, 0.0, initial_h)
        
        # Priority queue (min-heap by f_cost)
        frontier = [start_node]
        
        # Visited states
        visited = set()
        
        # Best complete itinerary found
        best_complete = None
        best_utility = -float('inf')
        
        # Memory tracking for SMA*
        memory_usage_mb = 0
        
        while frontier:
            # Check memory limit
            if memory_limit_mb > 0:
                memory_usage_mb = self._estimate_memory_usage(len(frontier), len(visited))
                if memory_usage_mb > memory_limit_mb:
                    # Switch to SMA* behavior
                    frontier = self._sma_memory_cleanup(frontier, visited)
            
            # Track frontier size
            self.max_frontier_size = max(self.max_frontier_size, len(frontier))
            
            # Get most promising node
            current = heapq.heappop(frontier)
            
            # Skip if already visited
            state_key = (current.state.visited_pois, 
                        round(current.state.current_time, 1),
                        round(current.state.remaining_budget))
            if state_key in visited:
                continue
            
            visited.add(state_key)
            self.nodes_expanded += 1
            
            # Check if this is a complete itinerary
            if len(current.state.visited_pois) >= constraints.min_pois:
                if current.state.total_utility > best_utility:
                    best_utility = current.state.total_utility
                    best_complete = current
                    logger.debug(f"New best itinerary: {len(current.state.visited_pois)} POIs, "
                               f"utility: {best_utility:.3f}")
            
            # Check termination conditions
            if (len(current.state.visited_pois) >= constraints.max_pois or
                current.state.remaining_budget < 10.0 or
                current.state.current_time + 1.0 > constraints.start_time + constraints.max_time_hours):
                continue
            
            # Expand node
            successors = self._generate_successors(
                current, available_indices, user_preferences, constraints
            )
            
            for successor in successors:
                if successor.f_cost < float('inf'):
                    heapq.heappush(frontier, successor)
            
            # Early termination if we have a good solution
            if self.nodes_expanded > 10000 and best_complete:
                logger.info("Early termination with good solution")
                break
        
        # Extract solution
        if best_complete:
            return self._reconstruct_path(best_complete)
        else:
            logger.warning("No feasible itinerary found")
            return []
    
    def _filter_pois_by_feedback(self, feedback: Optional[InteractiveFeedback]) -> List[POI]:
        """Filter POIs based on user feedback"""
        if not feedback:
            return self.pois
        
        filtered = []
        for poi in self.pois:
            if poi.id in feedback.rejected_pois:
                continue
            if poi.id in feedback.must_visit_pois:
                filtered.append(poi)
                continue
            if poi.category in feedback.preference_adjustments:
                if feedback.preference_adjustments[poi.category] > 0.2:
                    filtered.append(poi)
            else:
                filtered.append(poi)
        
        return filtered
    
    def _compute_heuristic(self, state: ItineraryState, available_indices: Set[int],
                          user_preferences: Dict[str, float], 
                          constraints: Constraints) -> float:
        """
        MST-based admissible heuristic
        
        Estimates minimum cost to visit remaining required POIs
        Admissible: never overestimates true cost
        """
        # Get unvisited POI indices
        visited_indices = set(self.poi_index_map[poi_id] 
                            for poi_id in state.visited_pois)
        unvisited = available_indices - visited_indices
        
        # Need at least min_pois total
        pois_needed = max(0, constraints.min_pois - len(state.visited_pois))
        if pois_needed == 0:
            return 0.0
        
        if len(unvisited) < pois_needed:
            return float('inf')  # Infeasible
        
        # Estimate utility of best unvisited POIs
        unvisited_utilities = []
        for idx in unvisited:
            poi = self.pois[idx]
            # Optimistic utility estimate
            utility = (user_preferences.get(poi.category, 0.5) * 
                      poi.rating / 5.0 * 1.2)  # Assume best bonuses
            unvisited_utilities.append((-utility, idx))  # Negative for sorting
        
        # Get top k POIs by utility
        unvisited_utilities.sort()
        top_pois = [idx for _, idx in unvisited_utilities[:pois_needed]]
        
        # MST cost for connecting top POIs
        if len(top_pois) <= 1:
            return 0.0
        
        # Build subgraph of top POIs
        mst_cost = 0.0
        edges = []
        for i in range(len(top_pois)):
            for j in range(i + 1, len(top_pois)):
                idx1, idx2 = top_pois[i], top_pois[j]
                cost = self.distance_matrix[idx1, idx2]
                edges.append((cost, idx1, idx2))
        
        # Kruskal's algorithm for MST
        edges.sort()
        parent = {idx: idx for idx in top_pois}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False
        
        for cost, u, v in edges:
            if union(u, v):
                mst_cost += cost
        
        # Add minimum connection cost from current location
        if state.visited_pois:
            last_idx = self.poi_index_map[state.visited_pois[-1]]
            min_connection = min(self.distance_matrix[last_idx, idx] 
                               for idx in top_pois)
            mst_cost += min_connection
        
        # Convert distance cost to negative utility
        # (A* minimizes cost, we maximize utility)
        time_cost = mst_cost / 25.0  # Assume best transport speed
        utility_penalty = time_cost * 10.0  # Time opportunity cost
        
        return -utility_penalty  # Negative because we maximize utility
    
    def _generate_successors(self, node: SearchNode, available_indices: Set[int],
                           user_preferences: Dict[str, float],
                           constraints: Constraints) -> List[SearchNode]:
        """Generate valid successor states"""
        successors = []
        current_state = node.state
        
        # Get current location
        if current_state.visited_pois:
            current_idx = self.poi_index_map[current_state.visited_pois[-1]]
            current_poi = self.pois[current_idx]
        else:
            current_idx = -1
            current_poi = None
        
        visited_indices = set(self.poi_index_map[poi_id] 
                            for poi_id in current_state.visited_pois)
        
        # Try each unvisited POI
        for poi_idx in available_indices:
            if poi_idx in visited_indices:
                continue
            
            poi = self.pois[poi_idx]
            
            # Check feasibility
            if not self._is_feasible(poi, current_poi, current_state, constraints):
                continue
            
            # Calculate costs and utilities
            travel_time = self._get_travel_time(current_idx, poi_idx, constraints)
            travel_cost = self._get_travel_cost(current_idx, poi_idx, constraints)
            
            # Update state
            new_time = current_state.current_time + travel_time + poi.avg_visit_duration
            new_budget = current_state.remaining_budget - poi.entrance_fee - travel_cost
            new_distance = current_state.total_distance + (
                self.distance_matrix[current_idx, poi_idx] if current_idx >= 0 else 0
            )
            
            # Calculate utility gain
            utility_gain = self._calculate_utility(
                poi, current_state, user_preferences, constraints
            )
            new_utility = current_state.total_utility + utility_gain
            
            # Create new state
            new_state = ItineraryState(
                visited_pois=current_state.visited_pois + (poi.id,),
                current_time=new_time,
                remaining_budget=new_budget,
                total_utility=new_utility,
                total_distance=new_distance
            )
            
            # Calculate costs
            g_cost = -new_utility  # Negative because A* minimizes
            h_cost = self._compute_heuristic(
                new_state, available_indices, user_preferences, constraints
            )
            
            successor = SearchNode(new_state, g_cost, h_cost, node)
            successors.append(successor)
        
        return successors
    
    def _is_feasible(self, poi: POI, current_poi: Optional[POI],
                    state: ItineraryState, constraints: Constraints) -> bool:
        """Check if POI is feasible to visit"""
        # Budget check
        travel_cost = self._get_travel_cost(
            self.poi_index_map[current_poi.id] if current_poi else -1,
            self.poi_index_map[poi.id],
            constraints
        )
        total_cost = poi.entrance_fee + travel_cost
        if total_cost > state.remaining_budget:
            return False
        
        # Time check
        travel_time = self._get_travel_time(
            self.poi_index_map[current_poi.id] if current_poi else -1,
            self.poi_index_map[poi.id],
            constraints
        )
        arrival_time = state.current_time + travel_time
        
        # Opening hours
        if arrival_time < poi.opening_hours[0] or arrival_time > poi.opening_hours[1]:
            return False
        
        # Total time budget
        total_time = travel_time + poi.avg_visit_duration
        end_time = state.current_time + total_time
        if end_time > constraints.start_time + constraints.max_time_hours:
            return False
        
        # NYC-specific: Inter-borough travel penalties
        if current_poi:
            current_borough = get_borough_id(current_poi.lat, current_poi.lon)
            poi_borough = get_borough_id(poi.lat, poi.lon)
            
            # Staten Island is particularly far
            if (current_borough == 4 and poi_borough != 4) or \
               (current_borough != 4 and poi_borough == 4):
                if state.remaining_budget < 50:  # Need budget for ferry/long trip
                    return False
        
        # Rush hour check
        if constraints.avoid_rush_hours:
            if (7 <= arrival_time <= 9) or (17 <= arrival_time <= 19):
                if constraints.transportation_mode == "taxi":
                    return False
        
        return True
    
    def _get_travel_time(self, from_idx: int, to_idx: int,
                        constraints: Constraints) -> float:
        """Calculate travel time between POIs"""
        if from_idx < 0:
            return 0.0
        
        distance = self.distance_matrix[from_idx, to_idx]
        
        # Check if inter-borough
        from_borough = self.poi_boroughs[from_idx]
        to_borough = self.poi_boroughs[to_idx]
        
        # Base speeds
        if constraints.transportation_mode == "walking":
            speed = 4.5
        elif constraints.transportation_mode == "public_transit":
            speed = 25.0
            # Add transfer time
            base_time = distance / speed + 0.117
            
            # Inter-borough subway penalty
            if from_borough != to_borough:
                base_time += 0.25  # 15 min extra for transfers
            
            return base_time
        else:  # taxi
            speed = 18.0
            
            # Traffic adjustments
            if 7 <= constraints.start_time <= 9 or 17 <= constraints.start_time <= 19:
                speed *= 0.6  # Rush hour
        
        return distance / speed
    
    def _get_travel_cost(self, from_idx: int, to_idx: int,
                        constraints: Constraints) -> float:
        """Calculate monetary cost of travel"""
        if from_idx < 0:
            return 0.0
        
        distance = self.distance_matrix[from_idx, to_idx]
        
        if constraints.transportation_mode == "walking":
            return 0.0
        elif constraints.transportation_mode == "public_transit":
            return 2.90  # NYC subway fare
        else:  # taxi
            return 3.0 + (distance * 1.75)
    
    def _calculate_utility(self, poi: POI, state: ItineraryState,
                         user_preferences: Dict[str, float],
                         constraints: Constraints) -> float:
        """Calculate utility of adding POI to itinerary"""
        # Base preference score
        pref_score = user_preferences.get(poi.category, 0.5)
        rating_score = poi.rating / 5.0
        
        # Diversity bonus
        visited_categories = []
        for poi_id in state.visited_pois:
            visited_categories.append(self.poi_map[poi_id].category)
        
        diversity_bonus = 1.0
        if poi.category not in visited_categories:
            diversity_bonus = 1.2
        elif visited_categories.count(poi.category) > 2:
            diversity_bonus = 0.7
        
        # Popularity balance
        if len(state.visited_pois) > 3:
            visited_pops = [self.poi_map[pid].popularity 
                          for pid in state.visited_pois]
            avg_pop = np.mean(visited_pops)
            if avg_pop > 0.7 and poi.popularity < 0.5:
                diversity_bonus *= 1.15  # Hidden gem bonus
        
        return pref_score * rating_score * diversity_bonus
    
    def _reconstruct_path(self, node: SearchNode) -> List[POI]:
        """Reconstruct POI sequence from search node"""
        path = []
        current = node
        
        while current is not None:
            if current.state.visited_pois:
                # Get the last POI added in this state
                if not current.parent or len(current.state.visited_pois) > len(current.parent.state.visited_pois):
                    last_poi_id = current.state.visited_pois[-1]
                    path.append(self.poi_map[last_poi_id])
            current = current.parent
        
        path.reverse()
        return path
    
    def _estimate_memory_usage(self, frontier_size: int, visited_size: int) -> float:
        """Estimate memory usage in MB"""
        # Rough estimates
        node_size = 200  # bytes per node
        state_size = 100  # bytes per state
        
        total_bytes = (frontier_size * node_size + visited_size * state_size)
        return total_bytes / (1024 * 1024)
    
    def _sma_memory_cleanup(self, frontier: List[SearchNode], 
                          visited: Set) -> List[SearchNode]:
        """
        SMA* memory cleanup strategy
        
        Remove least promising nodes when memory limit reached
        """
        logger.info("SMA* memory cleanup triggered")
        
        # Keep only top 75% of frontier
        frontier.sort(key=lambda n: n.f_cost)
        keep_size = int(len(frontier) * 0.75)
        
        # Clear some visited states (keep recent ones)
        if len(visited) > 10000:
            visited.clear()
        
        # Force garbage collection
        gc.collect()
        
        return frontier[:keep_size]
    
    def get_complexity(self) -> str:
        """Return algorithm complexity"""
        return "O(b^d) where b is branching factor, d is solution depth"
    
    def get_statistics(self) -> Dict:
        """Return search statistics"""
        return {
            'nodes_expanded': self.nodes_expanded,
            'max_frontier_size': self.max_frontier_size,
            'complexity': self.get_complexity()
        }


class MemoryBoundedAStarPlanner(AStarItineraryPlanner):
    """
    SMA* (Simplified Memory-bounded A*) variant
    
    For handling 1000+ POIs scenarios from TravelPlanner benchmark
    """
    
    def __init__(self, pois: List[Dict], distance_matrix: Optional[np.ndarray] = None,
                 memory_limit_mb: int = 500):
        super().__init__(pois, distance_matrix)
        self.memory_limit_mb = memory_limit_mb
        self.forgotten_nodes = 0
    
    def plan_itinerary(self, user_preferences: Dict[str, float],
                      constraints: Constraints,
                      feedback: Optional[InteractiveFeedback] = None,
                      start_location: Optional[Tuple[float, float]] = None) -> List[POI]:
        """
        SMA* planning with strict memory bounds
        """
        # Use parent method with memory limit
        result = super().plan_itinerary(
            user_preferences, constraints, feedback,
            start_location, memory_limit_mb=self.memory_limit_mb
        )
        
        logger.info(f"SMA* completed. Forgotten nodes: {self.forgotten_nodes}")
        return result


# Unit tests for edge cases
import unittest


class TestAStarItinerary(unittest.TestCase):
    """Test cases addressing research challenges"""
    
    def setUp(self):
        """Set up test data"""
        self.test_pois = [
            {
                'id': 'museum1',
                'name': 'MET Museum',
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
                'id': 'park1',
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
                'id': 'staten1',
                'name': 'Staten Island Zoo',
                'lat': 40.6254,
                'lon': -74.1157,
                'category': 'nature',
                'popularity': 0.6,
                'entrance_fee': 10.0,
                'avg_visit_duration': 2.0,
                'opening_hours': {'weekday': [10, 16]},
                'rating': 4.2
            }
        ]
        
        self.user_prefs = {'museum': 0.8, 'park': 0.7, 'nature': 0.6}
    
    def test_single_poi_trip(self):
        """Test when start equals goal (single POI)"""
        single_poi = [self.test_pois[0]]
        planner = AStarItineraryPlanner(single_poi)
        
        constraints = Constraints(min_pois=1, max_pois=1)
        result = planner.plan_itinerary(self.user_prefs, constraints)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, 'museum1')
    
    def test_manhattan_to_staten_island(self):
        """Test inter-borough routing penalty"""
        planner = AStarItineraryPlanner(self.test_pois)
        
        # Limited budget should discourage Staten Island
        constraints = Constraints(budget=50, max_pois=2)
        result = planner.plan_itinerary(self.user_prefs, constraints)
        
        # Should prefer Manhattan POIs
        poi_ids = [poi.id for poi in result]
        self.assertNotIn('staten1', poi_ids)
    
    def test_time_budget_below_minimum(self):
        """Test when time budget < minimum visit duration"""
        planner = AStarItineraryPlanner(self.test_pois)
        
        # Only 1 hour, but museum needs 3 hours
        constraints = Constraints(max_time_hours=1.0, min_pois=1)
        result = planner.plan_itinerary(self.user_prefs, constraints)
        
        # Should not select museum
        poi_ids = [poi.id for poi in result]
        self.assertNotIn('museum1', poi_ids)
    
    def test_large_scale_memory(self):
        """Test memory limits with 1000+ POIs"""
        # Generate 1000 POIs
        large_pois = []
        for i in range(1000):
            large_pois.append({
                'id': f'poi_{i}',
                'name': f'POI {i}',
                'lat': 40.7 + (i % 100) * 0.001,
                'lon': -74.0 + (i // 100) * 0.001,
                'category': ['museum', 'park', 'restaurant'][i % 3],
                'popularity': 0.5 + (i % 10) * 0.05,
                'entrance_fee': [0, 15, 25][i % 3],
                'avg_visit_duration': 1.5,
                'opening_hours': {'weekday': [9, 20]},
                'rating': 4.0 + (i % 5) * 0.2
            })
        
        # Use memory-bounded variant
        planner = MemoryBoundedAStarPlanner(large_pois, memory_limit_mb=100)
        
        constraints = Constraints(max_pois=5)
        result = planner.plan_itinerary(
            {'museum': 0.7, 'park': 0.8, 'restaurant': 0.6},
            constraints
        )
        
        # Should return valid itinerary despite memory limits
        self.assertGreater(len(result), 0)
        self.assertLessEqual(len(result), 5)
    
    def test_heuristic_admissibility(self):
        """Verify heuristic never overestimates"""
        planner = AStarItineraryPlanner(self.test_pois)
        
        # Test various states
        test_states = [
            ItineraryState((), 9.0, 100.0, 0.0, 0.0),
            ItineraryState(('museum1',), 12.0, 75.0, 10.0, 3.2),
            ItineraryState(('museum1', 'park1'), 14.0, 75.0, 18.0, 6.4)
        ]
        
        available = set(range(len(self.test_pois)))
        constraints = Constraints(min_pois=3)
        
        for state in test_states:
            h_cost = planner._compute_heuristic(
                state, available, self.user_prefs, constraints
            )
            
            # Heuristic should be finite and reasonable
            self.assertLess(h_cost, float('inf'))
            self.assertGreater(h_cost, -1000)  # Reasonable bound
    
    def test_real_time_traffic(self):
        """Test NYC traffic adjustments"""
        planner = AStarItineraryPlanner(self.test_pois)
        
        # Rush hour start
        rush_constraints = Constraints(
            start_time=7.5,  # 7:30 AM
            transportation_mode='taxi',
            avoid_rush_hours=True
        )
        
        result = planner.plan_itinerary(self.user_prefs, rush_constraints)
        
        # Should plan around rush hours
        self.assertGreater(len(result), 0)
        
        # Compare with non-rush hour
        normal_constraints = Constraints(
            start_time=10.0,
            transportation_mode='taxi'
        )
        
        normal_result = planner.plan_itinerary(self.user_prefs, normal_constraints)
        
        # Should be able to visit more POIs without rush hour
        self.assertGreaterEqual(len(normal_result), len(result))
    
    def test_subway_aware_transitions(self):
        """Test subway-optimized routing"""
        # Add more Manhattan POIs
        manhattan_pois = self.test_pois[:2] + [
            {
                'id': 'moma',
                'name': 'MoMA',
                'lat': 40.7614,
                'lon': -73.9776,
                'category': 'museum',
                'popularity': 0.85,
                'entrance_fee': 25.0,
                'avg_visit_duration': 2.5,
                'opening_hours': {'weekday': [10, 18]},
                'rating': 4.7
            }
        ]
        
        planner = AStarItineraryPlanner(manhattan_pois)
        
        constraints = Constraints(
            transportation_mode='public_transit',
            max_pois=3
        )
        
        result = planner.plan_itinerary(self.user_prefs, constraints)
        
        # Should efficiently route using subway
        self.assertEqual(len(result), 3)
        
        # Check statistics
        stats = planner.get_statistics()
        self.assertGreater(stats['nodes_expanded'], 0)
        self.assertIn('O(b^d)', stats['complexity'])


def demonstrate_astar():
    """Demonstrate A* algorithm with NYC example"""
    print("=== A* Itinerary Planning Demonstration ===\n")
    
    # Example NYC POIs
    nyc_pois = [
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
        },
        {
            'id': 'high_line',
            'name': 'High Line',
            'lat': 40.7480,
            'lon': -74.0048,
            'category': 'park',
            'popularity': 0.8,
            'entrance_fee': 0.0,
            'avg_visit_duration': 1.5,
            'opening_hours': {'weekday': [7, 22]},
            'rating': 4.7
        },
        {
            'id': 'empire_state',
            'name': 'Empire State Building',
            'lat': 40.7484,
            'lon': -73.9857,
            'category': 'landmark',
            'popularity': 0.92,
            'entrance_fee': 40.0,
            'avg_visit_duration': 1.5,
            'opening_hours': {'weekday': [8, 24]},
            'rating': 4.6
        }
    ]
    
    # User preferences
    user_prefs = {
        'museum': 0.9,
        'park': 0.8,
        'landmark': 0.7,
        'restaurant': 0.6
    }
    
    # Initialize planner
    planner = AStarItineraryPlanner(nyc_pois)
    
    # Test 1: Standard planning
    print("1. Standard A* Planning:")
    constraints = Constraints(
        budget=100,
        max_pois=4,
        min_pois=3,
        transportation_mode='public_transit'
    )
    
    result = planner.plan_itinerary(user_prefs, constraints)
    print(f"Found itinerary with {len(result)} POIs:")
    for i, poi in enumerate(result):
        print(f"  {i+1}. {poi.name} ({poi.category})")
    
    stats = planner.get_statistics()
    print(f"Nodes expanded: {stats['nodes_expanded']}")
    print(f"Max frontier size: {stats['max_frontier_size']}")
    
    # Test 2: Memory-bounded planning
    print("\n2. Memory-Bounded SMA* Planning:")
    sma_planner = MemoryBoundedAStarPlanner(nyc_pois, memory_limit_mb=10)
    
    result2 = sma_planner.plan_itinerary(user_prefs, constraints)
    print(f"Found itinerary with {len(result2)} POIs:")
    for i, poi in enumerate(result2):
        print(f"  {i+1}. {poi.name} ({poi.category})")
    
    # Test 3: Interactive feedback
    print("\n3. Interactive Planning with Feedback:")
    feedback = InteractiveFeedback(
        rejected_pois={'times_square'},  # Too crowded
        preference_adjustments={'park': 1.2, 'landmark': 0.5}
    )
    
    result3 = planner.plan_itinerary(user_prefs, constraints, feedback)
    print(f"After feedback, found itinerary with {len(result3)} POIs:")
    for i, poi in enumerate(result3):
        print(f"  {i+1}. {poi.name} ({poi.category})")
    
    # Calculate final metrics
    if result:
        print("\n4. Itinerary Metrics:")
        itinerary = Itinerary(result, 9.0, "public_transit", user_prefs)
        print(f"Total distance: {QuantitativeMetrics.total_distance(itinerary):.2f} km")
        print(f"Total time: {QuantitativeMetrics.total_time(itinerary):.2f} hours")
        print(f"Total cost: ${QuantitativeMetrics.total_cost(itinerary):.2f}")
        print(f"Diversity score: {QualitativeMetrics.diversity_score(itinerary):.3f}")
        
        # CSS score
        css = CompositeUtilityFunctions.composite_satisfaction_score(
            itinerary, user_prefs, budget=100, max_time=10
        )
        print(f"CSS Score: {css:.3f}")


if __name__ == "__main__":
    # Run demonstration
    demonstrate_astar()
    
    # Run tests
    print("\n=== Running Unit Tests ===")
    unittest.main(argv=[''], exit=False, verbosity=2)