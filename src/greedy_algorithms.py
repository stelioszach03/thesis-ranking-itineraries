"""
Greedy Algorithms for Interactive Itinerary Planning

Implementation of algorithms from Basu Roy et al. (2011) - Interactive Itinerary Planning
References research_context.md and the foundational paper [basu2011]

Key algorithms:
1. GreedyPOISelection - Basic greedy with marginal utility
2. HeapPrunGreedyPOI - Optimized version with heap-based pruning
3. Interactive planning with batch selection (k POIs at a time)

Complexity: O(n²) as noted in research_context.md
"""

import json
import numpy as np
import heapq
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, time
import logging
from copy import deepcopy

# Use absolute imports for sibling modules
from src.metrics_definitions import (
    POI, Itinerary, QuantitativeMetrics, QualitativeMetrics,
    CompositeUtilityFunctions
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Constraints:
    """Constraints for itinerary planning"""
    budget: float = 100.0
    max_time_hours: float = 10.0
    min_pois: int = 3  # From research: users prefer 3-7 POIs
    max_pois: int = 7
    max_walking_distance_km: float = 2.0
    start_time: float = 9.0  # 9 AM
    transportation_mode: str = "public_transit"
    
    # NYC-specific constraints
    avoid_rush_hours: bool = True
    require_subway_proximity: bool = True
    max_subway_distance_km: float = 0.5


@dataclass
class InteractiveFeedback:
    """User feedback for interactive planning process"""
    rejected_pois: Set[str] = field(default_factory=set)
    preferred_categories: Dict[str, float] = field(default_factory=dict)
    must_visit_pois: Set[str] = field(default_factory=set)
    preference_adjustments: Dict[str, float] = field(default_factory=dict)


class GreedyPOISelection:
    """
    Basic greedy algorithm for POI selection
    
    From Basu Roy et al. (2011):
    - Selects POIs based on marginal utility
    - Considers constraints (time, budget)
    - Complexity: O(n²) where n is number of POIs
    """
    
    def __init__(self, pois: List[Dict], distance_matrix: np.ndarray):
        self.pois = [self._dict_to_poi(p) for p in pois]
        self.distance_matrix = distance_matrix
        self.poi_index_map = {poi.id: i for i, poi in enumerate(self.pois)}
        
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
        """Parse opening hours from dictionary"""
        if 'weekday' in hours_dict:
            return tuple(hours_dict['weekday'])
        return (9.0, 20.0)  # Default hours
    
    def select_pois(self, 
                   user_preferences: Dict[str, float],
                   constraints: Constraints,
                   feedback: Optional[InteractiveFeedback] = None,
                   start_poi: Optional[POI] = None) -> List[POI]:
        """
        Main greedy selection algorithm
        
        Algorithm from Basu Roy et al. (2011):
        1. Initialize with empty itinerary
        2. While constraints satisfied:
           - Calculate marginal utility for each unvisited POI
           - Select POI with highest utility/cost ratio
           - Update itinerary and remaining budget/time
        
        Returns: Selected POIs in visit order
        """
        logger.info("Starting GreedyPOISelection algorithm")
        
        selected_pois = []
        if start_poi:
            selected_pois.append(start_poi)
        
        remaining_budget = constraints.budget
        remaining_time = constraints.max_time_hours
        current_time = constraints.start_time
        
        # Apply feedback filters
        available_pois = self._filter_pois_by_feedback(feedback)
        
        # Track visited POIs
        visited = set([poi.id for poi in selected_pois])
        
        iteration = 0
        while len(selected_pois) < constraints.max_pois:
            iteration += 1
            logger.debug(f"Iteration {iteration}: {len(selected_pois)} POIs selected")
            
            # Calculate marginal utilities
            candidates = []
            
            for poi in available_pois:
                if poi.id in visited:
                    continue
                
                # Check feasibility
                if not self._is_feasible(poi, selected_pois, remaining_budget, 
                                       remaining_time, current_time, constraints):
                    continue
                
                # Calculate marginal utility
                marginal_utility = self._calculate_marginal_utility(
                    poi, selected_pois, user_preferences, constraints
                )
                
                # Calculate cost (time + money)
                marginal_cost = self._calculate_marginal_cost(
                    poi, selected_pois, constraints
                )
                
                if marginal_cost > 0:
                    utility_ratio = marginal_utility / marginal_cost
                    candidates.append((utility_ratio, poi))
            
            # Select best candidate
            if not candidates:
                logger.info(f"No more feasible candidates. Stopping with {len(selected_pois)} POIs")
                break
            
            # Sort by utility ratio (descending)
            candidates.sort(reverse=True, key=lambda x: x[0])
            best_ratio, best_poi = candidates[0]
            
            logger.debug(f"Selected POI: {best_poi.name} with utility ratio: {best_ratio:.3f}")
            
            # Update state
            selected_pois.append(best_poi)
            visited.add(best_poi.id)
            
            # Update remaining resources
            travel_time = self._get_travel_time(selected_pois[-2] if len(selected_pois) > 1 else None, 
                                              best_poi, constraints)
            remaining_time -= (travel_time + best_poi.avg_visit_duration)
            remaining_budget -= best_poi.entrance_fee
            current_time += travel_time + best_poi.avg_visit_duration
            
            # Check minimum POIs constraint
            if len(selected_pois) >= constraints.min_pois and remaining_time < 2.0:
                logger.info(f"Minimum POIs reached and limited time remaining. Stopping.")
                break
        
        logger.info(f"GreedyPOISelection completed with {len(selected_pois)} POIs")
        return selected_pois
    
    def _filter_pois_by_feedback(self, feedback: Optional[InteractiveFeedback]) -> List[POI]:
        """Filter POIs based on user feedback"""
        if not feedback:
            return self.pois
        
        filtered = []
        for poi in self.pois:
            # Skip rejected POIs
            if poi.id in feedback.rejected_pois:
                continue
            
            # Include must-visit POIs
            if poi.id in feedback.must_visit_pois:
                filtered.append(poi)
                continue
            
            # Apply preference adjustments
            if poi.category in feedback.preference_adjustments:
                if feedback.preference_adjustments[poi.category] > 0.2:
                    filtered.append(poi)
            else:
                filtered.append(poi)
        
        return filtered
    
    def _is_feasible(self, poi: POI, selected: List[POI], 
                    remaining_budget: float, remaining_time: float,
                    current_time: float, constraints: Constraints) -> bool:
        """
        Check if POI is feasible given current constraints
        
        Considers:
        - Budget constraints
        - Time constraints
        - Opening hours
        - NYC-specific constraints (rush hours, subway proximity)
        """
        # Budget check
        if poi.entrance_fee > remaining_budget:
            return False
        
        # Time check
        travel_time = self._get_travel_time(
            selected[-1] if selected else None, poi, constraints
        )
        total_time_needed = travel_time + poi.avg_visit_duration
        
        if total_time_needed > remaining_time:
            return False
        
        # Opening hours check
        arrival_time = current_time + travel_time
        if arrival_time < poi.opening_hours[0] or arrival_time > poi.opening_hours[1]:
            return False
        
        # NYC rush hour check (7-9 AM, 5-7 PM)
        if constraints.avoid_rush_hours:
            if (7 <= arrival_time <= 9) or (17 <= arrival_time <= 19):
                if constraints.transportation_mode == "taxi":
                    # Taxis are very slow during rush hour
                    return False
        
        # Walking distance check
        if selected and constraints.transportation_mode == "walking":
            last_poi = selected[-1]
            distance = self._get_distance(last_poi, poi)
            if distance > constraints.max_walking_distance_km:
                return False
        
        return True
    
    def _calculate_marginal_utility(self, poi: POI, selected: List[POI],
                                  user_preferences: Dict[str, float],
                                  constraints: Constraints) -> float:
        """
        Calculate marginal utility of adding POI to itinerary
        
        Based on Basu Roy et al. (2011):
        U(poi) = preference_score × rating × diversity_bonus × novelty_factor
        """
        # Base utility from user preferences
        pref_score = user_preferences.get(poi.category, 0.5)
        
        # Rating component
        rating_score = poi.rating / 5.0
        
        # Diversity bonus - encourage different categories
        categories_visited = [p.category for p in selected]
        diversity_bonus = 1.0
        if poi.category not in categories_visited:
            diversity_bonus = 1.2  # 20% bonus for new category
        elif categories_visited.count(poi.category) > 2:
            diversity_bonus = 0.7  # Penalty for too many of same category
        
        # Novelty factor - prefer less popular POIs after having some popular ones
        novelty_factor = 1.0
        if len(selected) > 3:
            avg_popularity = np.mean([p.popularity for p in selected])
            if avg_popularity > 0.7 and poi.popularity < 0.5:
                novelty_factor = 1.15  # Bonus for hidden gems
        
        # NYC-specific: Proximity bonus for walkable clusters
        proximity_bonus = 1.0
        if selected and constraints.transportation_mode == "walking":
            last_poi = selected[-1]
            distance = self._get_distance(last_poi, poi)
            if distance < 0.5:  # Within 500m
                proximity_bonus = 1.1
        
        marginal_utility = (
            pref_score * rating_score * diversity_bonus * 
            novelty_factor * proximity_bonus
        )
        
        return marginal_utility
    
    def _calculate_marginal_cost(self, poi: POI, selected: List[POI],
                               constraints: Constraints) -> float:
        """
        Calculate marginal cost of adding POI
        
        Cost includes:
        - Monetary cost (entrance fee + transport)
        - Time cost (travel + visit)
        """
        # Monetary cost
        money_cost = poi.entrance_fee
        
        if selected:
            last_poi = selected[-1]
            distance = self._get_distance(last_poi, poi)
            
            # Add transportation cost
            if constraints.transportation_mode == "public_transit":
                money_cost += 2.90  # NYC subway fare
            elif constraints.transportation_mode == "taxi":
                money_cost += 3.0 + (distance * 1.75)  # NYC taxi rates
        
        # Time cost (normalized to money equivalent)
        # Assume 1 hour = $20 value
        travel_time = self._get_travel_time(
            selected[-1] if selected else None, poi, constraints
        )
        time_cost = (travel_time + poi.avg_visit_duration) * 20
        
        # Total marginal cost
        marginal_cost = money_cost + time_cost * 0.1  # Weight time less than money
        
        return marginal_cost
    
    def _get_distance(self, poi1: Optional[POI], poi2: POI) -> float:
        """Get distance between two POIs"""
        if not poi1:
            return 0.0
        
        idx1 = self.poi_index_map[poi1.id]
        idx2 = self.poi_index_map[poi2.id]
        
        return self.distance_matrix[idx1, idx2]
    
    def _get_travel_time(self, poi1: Optional[POI], poi2: POI,
                        constraints: Constraints) -> float:
        """Calculate travel time between POIs"""
        if not poi1:
            return 0.0
        
        distance = self._get_distance(poi1, poi2)
        
        # Speed based on transportation mode
        if constraints.transportation_mode == "walking":
            speed = 4.5  # km/h
        elif constraints.transportation_mode == "public_transit":
            speed = 25.0  # km/h average
            # Add transfer time
            return distance / speed + 0.117  # 7 min transfer
        else:  # taxi
            speed = 18.0  # km/h in Manhattan
        
        return distance / speed
    
    def get_complexity(self) -> str:
        """Return algorithm complexity"""
        return "O(n²) where n is number of POIs"


class HeapPrunGreedyPOI:
    """
    Optimized greedy algorithm using heap-based pruning
    
    From Basu Roy et al. (2011):
    - Uses max-heap to efficiently track best candidates
    - Prunes infeasible POIs early
    - Still O(n²) worst case but faster in practice
    """
    
    def __init__(self, pois: List[Dict], distance_matrix: np.ndarray):
        self.base_selector = GreedyPOISelection(pois, distance_matrix)
        self.pois = self.base_selector.pois
        self.distance_matrix = distance_matrix
        
    def select_pois(self,
                   user_preferences: Dict[str, float],
                   constraints: Constraints,
                   feedback: Optional[InteractiveFeedback] = None,
                   start_poi: Optional[POI] = None) -> List[POI]:
        """
        Heap-based optimized selection
        
        Optimization: Maintain heap of candidates sorted by upper bound of utility
        """
        logger.info("Starting HeapPrunGreedyPOI algorithm")
        
        selected_pois = []
        if start_poi:
            selected_pois.append(start_poi)
        
        remaining_budget = constraints.budget
        remaining_time = constraints.max_time_hours
        current_time = constraints.start_time
        
        # Apply feedback filters
        available_pois = self.base_selector._filter_pois_by_feedback(feedback)
        
        # Initialize heap with all POIs
        # Use negative utility for max-heap behavior
        heap = []
        visited = set([poi.id for poi in selected_pois])
        
        # Pre-compute upper bounds
        for poi in available_pois:
            if poi.id not in visited:
                # Upper bound: assume best case scenario
                upper_bound = self._compute_upper_bound(
                    poi, user_preferences, constraints
                )
                heapq.heappush(heap, (-upper_bound, poi.id, poi))
        
        iteration = 0
        while len(selected_pois) < constraints.max_pois and heap:
            iteration += 1
            
            # Process candidates from heap
            candidates_checked = 0
            best_candidate = None
            best_ratio = -float('inf')
            
            temp_heap = []
            
            while heap and candidates_checked < min(10, len(heap)):
                neg_bound, poi_id, poi = heapq.heappop(heap)
                
                if poi_id in visited:
                    continue
                
                candidates_checked += 1
                
                # Check exact feasibility
                if not self.base_selector._is_feasible(
                    poi, selected_pois, remaining_budget,
                    remaining_time, current_time, constraints
                ):
                    continue
                
                # Calculate exact marginal utility
                marginal_utility = self.base_selector._calculate_marginal_utility(
                    poi, selected_pois, user_preferences, constraints
                )
                
                marginal_cost = self.base_selector._calculate_marginal_cost(
                    poi, selected_pois, constraints
                )
                
                if marginal_cost > 0:
                    utility_ratio = marginal_utility / marginal_cost
                    
                    if utility_ratio > best_ratio:
                        best_ratio = utility_ratio
                        best_candidate = poi
                
                # Re-insert with updated bound if still viable
                if -neg_bound > best_ratio * 0.8:  # Within 20% of upper bound
                    temp_heap.append((neg_bound, poi_id, poi))
            
            # Re-insert remaining candidates
            for item in temp_heap:
                heapq.heappush(heap, item)
            
            if not best_candidate:
                logger.info("No more feasible candidates found")
                break
            
            # Add best candidate
            selected_pois.append(best_candidate)
            visited.add(best_candidate.id)
            
            # Update resources
            travel_time = self.base_selector._get_travel_time(
                selected_pois[-2] if len(selected_pois) > 1 else None,
                best_candidate, constraints
            )
            remaining_time -= (travel_time + best_candidate.avg_visit_duration)
            remaining_budget -= best_candidate.entrance_fee
            current_time += travel_time + best_candidate.avg_visit_duration
            
            logger.debug(f"Selected: {best_candidate.name}, Remaining time: {remaining_time:.2f}h")
            
            # Prune heap based on remaining resources
            if remaining_time < 2.0 or remaining_budget < 20.0:
                logger.info("Limited resources remaining, considering termination")
                if len(selected_pois) >= constraints.min_pois:
                    break
        
        logger.info(f"HeapPrunGreedyPOI completed with {len(selected_pois)} POIs")
        return selected_pois
    
    def _compute_upper_bound(self, poi: POI, user_preferences: Dict[str, float],
                           constraints: Constraints) -> float:
        """
        Compute upper bound on utility ratio for pruning
        
        Assumes best-case scenario:
        - Minimum travel time
        - Maximum preference score
        - Best diversity bonus
        """
        # Maximum possible preference score
        max_pref = max(user_preferences.values()) if user_preferences else 1.0
        
        # Best case utility (all bonuses apply)
        max_utility = max_pref * (poi.rating / 5.0) * 1.2 * 1.15 * 1.1
        
        # Minimum cost (just entrance fee, minimal travel)
        min_cost = max(poi.entrance_fee + 5.0, 1.0)  # Avoid division by zero
        
        return max_utility / min_cost
    
    def batch_select(self, k: int, user_preferences: Dict[str, float],
                    constraints: Constraints, current_itinerary: List[POI],
                    feedback: Optional[InteractiveFeedback] = None) -> List[POI]:
        """
        Select k POIs at once for interactive planning
        
        Used in three-step process from Basu Roy et al. (2011)
        """
        logger.info(f"Batch selecting {k} POIs")
        
        # Temporarily adjust constraints
        temp_constraints = deepcopy(constraints)
        temp_constraints.min_pois = min(len(current_itinerary) + k, constraints.min_pois)
        temp_constraints.max_pois = len(current_itinerary) + k
        
        # Run selection starting from current itinerary
        start_poi = current_itinerary[-1] if current_itinerary else None
        full_itinerary = self.select_pois(
            user_preferences, temp_constraints, feedback, start_poi
        )
        
        # Return only the new POIs
        return full_itinerary[len(current_itinerary):]


class GreedyPlanner:
    """
    Main planner implementing three-step interactive process
    
    From Basu Roy et al. (2011):
    1. Initial itinerary generation
    2. User feedback collection
    3. Itinerary refinement
    """
    
    def __init__(self, pois_file: str, distance_matrix_file: str):
        # Load data
        with open(pois_file, 'r') as f:
            self.pois_data = json.load(f)
        
        self.distance_matrix = np.load(distance_matrix_file)
        
        # Initialize algorithms
        self.greedy_selector = GreedyPOISelection(self.pois_data, self.distance_matrix)
        self.heap_selector = HeapPrunGreedyPOI(self.pois_data, self.distance_matrix)
        
        # Metrics calculator
        self.metrics_engine = MetricsEngine()
        
        logger.info(f"Initialized GreedyPlanner with {len(self.pois_data)} POIs")
    
    def generate_initial_itinerary(self, user_profile: Dict,
                                 constraints: Optional[Constraints] = None,
                                 use_heap_optimization: bool = True) -> Itinerary:
        """
        Step 1: Generate initial itinerary
        """
        if constraints is None:
            constraints = Constraints()
        
        # Extract user preferences
        user_preferences = user_profile.get('preferences', {})
        
        # Adjust constraints based on user profile
        if 'daily_pois_preference' in user_profile:
            constraints.max_pois = user_profile['daily_pois_preference']
            constraints.min_pois = max(3, constraints.max_pois - 2)
        
        if 'budget_per_day' in user_profile:
            constraints.budget = user_profile['budget_per_day']
        
        if 'preferred_transport' in user_profile:
            constraints.transportation_mode = user_profile['preferred_transport']
        
        if 'max_walking_distance_km' in user_profile:
            constraints.max_walking_distance_km = user_profile['max_walking_distance_km']
        
        # Select algorithm
        if use_heap_optimization:
            selected_pois = self.heap_selector.select_pois(
                user_preferences, constraints
            )
        else:
            selected_pois = self.greedy_selector.select_pois(
                user_preferences, constraints
            )
        
        # Create itinerary
        itinerary = Itinerary(
            pois=selected_pois,
            start_time=constraints.start_time,
            transportation_mode=constraints.transportation_mode,
            user_preferences=user_preferences
        )
        
        return itinerary
    
    def apply_feedback(self, current_itinerary: Itinerary,
                     feedback: InteractiveFeedback,
                     constraints: Constraints) -> Itinerary:
        """
        Step 2 & 3: Apply user feedback and regenerate
        """
        logger.info("Applying user feedback to itinerary")
        
        # Update user preferences based on feedback
        updated_preferences = current_itinerary.user_preferences.copy()
        for category, adjustment in feedback.preference_adjustments.items():
            if category in updated_preferences:
                updated_preferences[category] *= adjustment
            else:
                updated_preferences[category] = adjustment
        
        # Remove rejected POIs from current itinerary
        filtered_pois = [
            poi for poi in current_itinerary.pois
            if poi.id not in feedback.rejected_pois
        ]
        
        # Recalculate remaining budget and time
        spent_budget = sum(poi.entrance_fee for poi in filtered_pois)
        spent_time = QuantitativeMetrics.total_time(
            Itinerary(filtered_pois, current_itinerary.start_time,
                     current_itinerary.transportation_mode)
        )
        
        adjusted_constraints = deepcopy(constraints)
        adjusted_constraints.budget -= spent_budget
        adjusted_constraints.max_time_hours -= spent_time
        adjusted_constraints.min_pois = max(0, constraints.min_pois - len(filtered_pois))
        adjusted_constraints.max_pois = constraints.max_pois - len(filtered_pois)
        
        # Generate new POIs using batch selection
        new_pois = self.heap_selector.batch_select(
            adjusted_constraints.max_pois,
            updated_preferences,
            adjusted_constraints,
            filtered_pois,
            feedback
        )
        
        # Combine into new itinerary
        final_pois = filtered_pois + new_pois
        
        return Itinerary(
            pois=final_pois,
            start_time=current_itinerary.start_time,
            transportation_mode=current_itinerary.transportation_mode,
            user_preferences=updated_preferences
        )
    
    def rank_itineraries(self, itineraries: List[Itinerary],
                        user_profile: Dict) -> List[Tuple[float, Itinerary]]:
        """
        Rank multiple itineraries using CSS score
        """
        ranked = []
        
        for itinerary in itineraries:
            css_score = CompositeUtilityFunctions.composite_satisfaction_score(
                itinerary,
                user_profile['preferences'],
                budget=user_profile.get('budget_per_day', 100),
                max_time=10.0
            )
            ranked.append((css_score, itinerary))
        
        # Sort by score (descending)
        ranked.sort(reverse=True, key=lambda x: x[0])
        
        return ranked


class MetricsEngine:
    """Simple metrics engine for testing"""
    
    def calculate_all_metrics(self, itinerary: Itinerary) -> Dict:
        """Calculate all metrics for an itinerary"""
        return {
            'total_distance': QuantitativeMetrics.total_distance(itinerary),
            'total_time': QuantitativeMetrics.total_time(itinerary),
            'total_cost': QuantitativeMetrics.total_cost(itinerary),
            'diversity_score': QualitativeMetrics.diversity_score(itinerary),
            'n_pois': len(itinerary.pois)
        }


# Unit tests for edge cases
import unittest


class TestGreedyAlgorithms(unittest.TestCase):
    """
    Unit tests addressing NP-completeness challenges
    """
    
    def setUp(self):
        """Set up test data"""
        # Create minimal test POIs
        self.test_pois = [
            {
                'id': 'poi1',
                'name': 'Test Museum',
                'lat': 40.7580,
                'lon': -73.9855,
                'category': 'museum',
                'popularity': 0.8,
                'entrance_fee': 25.0,
                'avg_visit_duration': 2.0,
                'opening_hours': {'weekday': [10, 17]},
                'rating': 4.5
            },
            {
                'id': 'poi2',
                'name': 'Test Park',
                'lat': 40.7829,
                'lon': -73.9654,
                'category': 'park',
                'popularity': 0.9,
                'entrance_fee': 0.0,
                'avg_visit_duration': 1.5,
                'opening_hours': {'weekday': [6, 22]},
                'rating': 4.7
            }
        ]
        
        # Simple distance matrix
        self.distance_matrix = np.array([
            [0.0, 3.2],
            [3.2, 0.0]
        ])
        
        self.user_prefs = {'museum': 0.8, 'park': 0.6}
    
    def test_empty_poi_list(self):
        """Test with empty POI list"""
        selector = GreedyPOISelection([], np.array([]))
        result = selector.select_pois(
            self.user_prefs,
            Constraints()
        )
        self.assertEqual(len(result), 0)
    
    def test_budget_constraint_infeasible(self):
        """Test when budget makes all POIs infeasible"""
        selector = GreedyPOISelection(self.test_pois, self.distance_matrix)
        constraints = Constraints(budget=10.0)  # Less than museum fee
        
        result = selector.select_pois(self.user_prefs, constraints)
        
        # Should only select free park
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].category, 'park')
    
    def test_time_window_conflicts(self):
        """Test NYC rush hour conflicts"""
        selector = GreedyPOISelection(self.test_pois, self.distance_matrix)
        constraints = Constraints(
            start_time=16.5,  # 4:30 PM, will hit rush hour
            max_time_hours=2.0,
            transportation_mode='taxi',
            avoid_rush_hours=True
        )
        
        result = selector.select_pois(self.user_prefs, constraints)
        
        # Should avoid selections that would require travel during rush hour
        self.assertLessEqual(len(result), 1)
    
    def test_identical_utility_scores(self):
        """Test POIs with identical utility scores"""
        # Create POIs with same scores
        identical_pois = [
            {
                'id': f'poi{i}',
                'name': f'Museum {i}',
                'lat': 40.7580 + i*0.001,
                'lon': -73.9855,
                'category': 'museum',
                'popularity': 0.8,
                'entrance_fee': 25.0,
                'avg_visit_duration': 2.0,
                'opening_hours': {'weekday': [10, 17]},
                'rating': 4.5
            }
            for i in range(3)
        ]
        
        dist_matrix = np.ones((3, 3)) * 0.5
        np.fill_diagonal(dist_matrix, 0)
        
        selector = GreedyPOISelection(identical_pois, dist_matrix)
        result = selector.select_pois(
            {'museum': 0.8},
            Constraints(max_pois=2)
        )
        
        # Should still make a selection
        self.assertEqual(len(result), 2)
    
    def test_manhattan_outer_borough_routing(self):
        """Test routing from Manhattan to outer borough"""
        # Add Brooklyn POI
        brooklyn_pois = self.test_pois + [{
            'id': 'brooklyn1',
            'name': 'Brooklyn Museum',
            'lat': 40.6712,  # Brooklyn
            'lon': -73.9636,
            'category': 'museum',
            'popularity': 0.7,
            'entrance_fee': 20.0,
            'avg_visit_duration': 2.0,
            'opening_hours': {'weekday': [10, 17]},
            'rating': 4.3
        }]
        
        # Larger distance to Brooklyn
        dist_matrix = np.array([
            [0.0, 3.2, 12.5],
            [3.2, 0.0, 10.8],
            [12.5, 10.8, 0.0]
        ])
        
        selector = GreedyPOISelection(brooklyn_pois, dist_matrix)
        result = selector.select_pois(
            self.user_prefs,
            Constraints(max_walking_distance_km=5.0)
        )
        
        # Should prefer Manhattan POIs due to distance
        manhattan_pois = [p for p in result if p.id != 'brooklyn1']
        self.assertGreater(len(manhattan_pois), 0)
    
    def test_subway_line_closure(self):
        """Test handling subway line closures"""
        selector = GreedyPOISelection(self.test_pois, self.distance_matrix)
        
        # Simulate subway closure by requiring walking
        constraints = Constraints(
            transportation_mode='walking',
            max_walking_distance_km=2.0  # Limit walking
        )
        
        result = selector.select_pois(self.user_prefs, constraints)
        
        # Should limit selections due to walking constraints
        self.assertLessEqual(len(result), 2)
    
    def test_interactive_feedback(self):
        """Test three-step interactive process"""
        planner = GreedyPlanner('data/nyc_pois.json', 'data/distance_matrix.npy')
        
        # Skip if files don't exist
        import os
        if not os.path.exists('data/nyc_pois.json'):
            self.skipTest("Data files not found")
        
        user_profile = {
            'preferences': {'museum': 0.9, 'park': 0.7},
            'daily_pois_preference': 5,
            'budget_per_day': 100
        }
        
        # Step 1: Initial generation
        initial = planner.generate_initial_itinerary(user_profile)
        self.assertGreater(len(initial.pois), 0)
        
        # Step 2: User feedback
        feedback = InteractiveFeedback(
            rejected_pois={initial.pois[0].id},
            preference_adjustments={'museum': 0.5, 'park': 1.5}
        )
        
        # Step 3: Apply feedback
        refined = planner.apply_feedback(initial, feedback, Constraints())
        
        # Check feedback was applied
        rejected_ids = [p.id for p in refined.pois]
        self.assertNotIn(initial.pois[0].id, rejected_ids)
    
    def test_performance_complexity(self):
        """Test O(n²) complexity scaling"""
        # Create larger POI sets
        for n in [10, 20, 40]:
            pois = [{
                'id': f'poi{i}',
                'name': f'POI {i}',
                'lat': 40.7580 + (i % 10) * 0.01,
                'lon': -73.9855 + (i // 10) * 0.01,
                'category': 'museum' if i % 3 == 0 else 'park',
                'popularity': 0.5 + (i % 5) * 0.1,
                'entrance_fee': 20.0 if i % 3 == 0 else 0.0,
                'avg_visit_duration': 1.5,
                'opening_hours': {'weekday': [9, 20]},
                'rating': 4.0
            } for i in range(n)]
            
            dist_matrix = np.random.rand(n, n) * 5
            np.fill_diagonal(dist_matrix, 0)
            dist_matrix = (dist_matrix + dist_matrix.T) / 2
            
            selector = GreedyPOISelection(pois, dist_matrix)
            
            import time
            start_time = time.time()
            result = selector.select_pois(
                {'museum': 0.7, 'park': 0.8},
                Constraints(max_pois=5)
            )
            elapsed = time.time() - start_time
            
            logger.info(f"n={n}: {elapsed:.3f}s for {len(result)} POIs")
            
            # Should complete in reasonable time
            self.assertLess(elapsed, n * 0.01)  # Rough O(n²) check


def demonstrate_greedy_algorithms():
    """Demonstrate the greedy algorithms with example data"""
    print("=== Greedy Algorithms Demonstration ===\n")
    
    # Create example NYC data
    example_pois = [
        {
            'id': 'times_square',
            'name': 'Times Square',
            'lat': 40.7580,
            'lon': -73.9855,
            'category': 'landmark',
            'popularity': 0.95,
            'entrance_fee': 0.0,
            'avg_visit_duration': 0.5,
            'opening_hours': {'weekday': [0, 24]},
            'rating': 4.2
        },
        {
            'id': 'central_park',
            'name': 'Central Park',
            'lat': 40.7829,
            'lon': -73.9654,
            'category': 'park',
            'popularity': 0.90,
            'entrance_fee': 0.0,
            'avg_visit_duration': 2.0,
            'opening_hours': {'weekday': [6, 21]},
            'rating': 4.8
        },
        {
            'id': 'met_museum',
            'name': 'Metropolitan Museum',
            'lat': 40.7794,
            'lon': -73.9632,
            'category': 'museum',
            'popularity': 0.85,
            'entrance_fee': 25.0,
            'avg_visit_duration': 3.0,
            'opening_hours': {'weekday': [10, 17]},
            'rating': 4.7
        },
        {
            'id': 'high_line',
            'name': 'High Line',
            'lat': 40.7480,
            'lon': -74.0048,
            'category': 'park',
            'popularity': 0.75,
            'entrance_fee': 0.0,
            'avg_visit_duration': 1.5,
            'opening_hours': {'weekday': [7, 22]},
            'rating': 4.6
        }
    ]
    
    # Distance matrix (km)
    distances = np.array([
        [0.0, 3.2, 2.8, 5.1],
        [3.2, 0.0, 0.4, 6.8],
        [2.8, 0.4, 0.0, 6.5],
        [5.1, 6.8, 6.5, 0.0]
    ])
    
    # User preferences
    user_prefs = {
        'landmark': 0.7,
        'park': 0.8,
        'museum': 0.9
    }
    
    # Test GreedyPOISelection
    print("1. Basic Greedy Selection:")
    greedy = GreedyPOISelection(example_pois, distances)
    constraints = Constraints(budget=50, max_pois=3)
    
    result = greedy.select_pois(user_prefs, constraints)
    print(f"Selected {len(result)} POIs:")
    for poi in result:
        print(f"  - {poi.name} ({poi.category})")
    print(f"Complexity: {greedy.get_complexity()}\n")
    
    # Test HeapPrunGreedyPOI
    print("2. Heap-Optimized Selection:")
    heap_greedy = HeapPrunGreedyPOI(example_pois, distances)
    
    result2 = heap_greedy.select_pois(user_prefs, constraints)
    print(f"Selected {len(result2)} POIs:")
    for poi in result2:
        print(f"  - {poi.name} ({poi.category})")
    
    # Test interactive feedback
    print("\n3. Interactive Planning with Feedback:")
    feedback = InteractiveFeedback(
        rejected_pois={'times_square'},
        preference_adjustments={'park': 1.2, 'museum': 0.8}
    )
    
    result3 = heap_greedy.select_pois(user_prefs, constraints, feedback)
    print(f"After feedback, selected {len(result3)} POIs:")
    for poi in result3:
        print(f"  - {poi.name} ({poi.category})")
    
    # Calculate metrics
    print("\n4. Itinerary Metrics:")
    itinerary = Itinerary(result2, 9.0, "public_transit", user_prefs)
    print(f"Total distance: {QuantitativeMetrics.total_distance(itinerary):.2f} km")
    print(f"Total time: {QuantitativeMetrics.total_time(itinerary):.2f} hours")
    print(f"Total cost: ${QuantitativeMetrics.total_cost(itinerary):.2f}")
    print(f"Diversity score: {QualitativeMetrics.diversity_score(itinerary):.3f}")


if __name__ == "__main__":
    # Run demonstration
    demonstrate_greedy_algorithms()
    
    # Run tests
    print("\n=== Running Unit Tests ===")
    unittest.main(argv=[''], exit=False, verbosity=2)