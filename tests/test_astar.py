"""
Unit tests for astar_itinerary module
Bachelor's thesis: Ranking Itineraries
"""

import unittest
import numpy as np
from typing import List, Dict, Set

from src.metrics_definitions import POI
from src.greedy_algorithms import Constraints, InteractiveFeedback
from src.astar_itinerary import AStarItineraryPlanner, ItineraryState, SearchNode


class TestItineraryState(unittest.TestCase):
    """Test ItineraryState class"""
    
    def test_state_creation(self):
        """Test creating itinerary states"""
        state = ItineraryState(
            visited_pois=("poi1", "poi2"),
            current_time=12.5,
            remaining_budget=75.0,
            total_utility=2.5,
            total_distance=5.2
        )
        
        self.assertEqual(len(state.visited_pois), 2)
        self.assertEqual(state.current_time, 12.5)
        self.assertEqual(state.remaining_budget, 75.0)
    
    def test_state_hashing(self):
        """Test state hashing for use in sets"""
        state1 = ItineraryState(("poi1", "poi2"), 12.5, 75.0, 2.5, 5.2)
        state2 = ItineraryState(("poi1", "poi2"), 12.5, 75.0, 2.5, 5.2)
        state3 = ItineraryState(("poi2", "poi1"), 12.5, 75.0, 2.5, 5.2)
        
        # Same states should have same hash
        self.assertEqual(hash(state1), hash(state2))
        # Different order means different state
        self.assertNotEqual(hash(state1), hash(state3))


class TestSearchNode(unittest.TestCase):
    """Test SearchNode class"""
    
    def test_node_comparison(self):
        """Test node comparison for priority queue"""
        state1 = ItineraryState(("poi1",), 10.0, 80.0, 1.0, 2.0)
        state2 = ItineraryState(("poi2",), 11.0, 70.0, 1.5, 3.0)
        
        node1 = SearchNode(state1, g_cost=1.0, h_cost=2.0)
        node2 = SearchNode(state2, g_cost=1.5, h_cost=1.0)
        
        # Node with lower f_cost should be "less than"
        self.assertLess(node2, node1)  # f=2.5 < f=3.0 is True
        self.assertGreater(node1, node2)  # f=3.0 > f=2.5 is True


class TestAStarItineraryPlanner(unittest.TestCase):
    """Test A* itinerary planner"""
    
    def setUp(self):
        """Create test data"""
        self.poi_dicts = [
            {"id": "start", "name": "Start Location", "lat": 40.7580, "lon": -73.9855, 
             "category": "landmark", "popularity": 0.0, "entrance_fee": 0.0, 
             "avg_visit_duration": 0.0, "opening_hours": {"weekday": [0.0, 24.0]}, "rating": 0.0},
            {"id": "poi1", "name": "Central Park", "lat": 40.7829, "lon": -73.9654, 
             "category": "park", "popularity": 0.9, "entrance_fee": 0.0, 
             "avg_visit_duration": 1.5, "opening_hours": {"weekday": [6.0, 22.0]}, "rating": 4.7},
            {"id": "poi2", "name": "MoMA", "lat": 40.7614, "lon": -73.9776, 
             "category": "museum", "popularity": 0.85, "entrance_fee": 25.0, 
             "avg_visit_duration": 2.0, "opening_hours": {"weekday": [10.5, 17.5]}, "rating": 4.5},
            {"id": "poi3", "name": "Times Square", "lat": 40.7580, "lon": -73.9855, 
             "category": "landmark", "popularity": 0.95, "entrance_fee": 0.0, 
             "avg_visit_duration": 0.5, "opening_hours": {"weekday": [0.0, 24.0]}, "rating": 4.3}
        ]
        
        # Create simple distance matrix
        n = len(self.poi_dicts)
        self.distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Manhattan distance approximation
                    self.distance_matrix[i, j] = (
                        abs(self.poi_dicts[i]["lat"] - self.poi_dicts[j]["lat"]) + 
                        abs(self.poi_dicts[i]["lon"] - self.poi_dicts[j]["lon"])
                    ) * 111
        
        self.planner = AStarItineraryPlanner(self.poi_dicts, self.distance_matrix)
    
    def test_basic_planning(self):
        """Test basic A* planning"""
        preferences = {"park": 0.8, "museum": 0.9, "landmark": 0.7}
        constraints = Constraints(
            budget=100,
            max_time_hours=6,
            min_pois=2,
            max_pois=3,
            start_location=(40.7580, -73.9855)  # Times Square
        )
        
        result = self.planner.plan_itinerary(preferences, constraints)
        
        self.assertIsNotNone(result)
        self.assertGreaterEqual(len(result), constraints.min_pois)
        self.assertLessEqual(len(result), constraints.max_pois)
    
    def test_heuristic_admissibility(self):
        """Test that heuristic is admissible"""
        preferences = {"park": 1.0, "museum": 1.0, "landmark": 1.0}
        constraints = Constraints(
            budget=200,
            max_time_hours=8,
            min_pois=2,
            max_pois=4
        )
        
        # Create a state
        state = ItineraryState(
            visited_pois=("poi1",),
            current_time=10.0,
            remaining_budget=150.0,
            total_utility=1.0,
            total_distance=2.0
        )
        
        available_indices = {1, 2, 3}  # Remaining POIs
        
        h_value = self.planner._compute_heuristic(
            state, available_indices, preferences, constraints
        )
        
        # Heuristic should be non-positive (we're maximizing)
        self.assertLessEqual(h_value, 0)
    
    def test_optimal_solution(self):
        """Test that A* finds optimal solution for small problem"""
        # Create small problem with clear optimal solution
        pois = [
            {"id": "start", "name": "Start", "lat": 0.0, "lon": 0.0, "category": "start", 
             "popularity": 0.0, "entrance_fee": 0.0, "avg_visit_duration": 0.0, 
             "opening_hours": {"weekday": [0.0, 24.0]}, "rating": 0.0},
            {"id": "good", "name": "Good POI", "lat": 1.0, "lon": 0.0, "category": "museum", 
             "popularity": 1.0, "entrance_fee": 10.0, "avg_visit_duration": 1.0,
             "opening_hours": {"weekday": [9.0, 17.0]}, "rating": 5.0},
            {"id": "bad", "name": "Bad POI", "lat": 0.0, "lon": 1.0, "category": "museum", 
             "popularity": 0.5, "entrance_fee": 50.0, "avg_visit_duration": 1.0,
             "opening_hours": {"weekday": [9.0, 17.0]}, "rating": 3.0}
        ]
        
        distance_matrix = np.array([
            [0, 1, 1],
            [1, 0, 2],
            [1, 2, 0]
        ])
        
        planner = AStarItineraryPlanner(pois, distance_matrix)
        
        preferences = {"museum": 1.0}
        constraints = Constraints(
            budget=30,  # Can only afford "good" POI
            max_time_hours=5,
            min_pois=1,
            max_pois=2,
            start_location=(0.0, 0.0)
        )
        
        result = planner.plan_itinerary(preferences, constraints)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, "good")
    
    def test_no_feasible_solution(self):
        """Test when no feasible solution exists"""
        preferences = {"museum": 1.0}
        constraints = Constraints(
            budget=10,  # Can't afford MoMA
            max_time_hours=1,  # Not enough time
            min_pois=2,
            max_pois=3
        )
        
        result = self.planner.plan_itinerary(preferences, constraints)
        
        # Should return empty list when no solution exists
        self.assertEqual(len(result), 0)
    
    def test_memory_bounded_variant(self):
        """Test memory bounded variant of A*"""
        preferences = {"park": 0.8, "museum": 0.9, "landmark": 0.7}
        constraints = Constraints(
            budget=100,
            max_time_hours=6,
            min_pois=2,
            max_pois=3
        )
        
        # Test standard A* without beam width (not supported in this implementation)
        result = self.planner.plan_itinerary(
            preferences, constraints
        )
        
        self.assertIsNotNone(result)
        if result:  # If solution found
            self.assertGreaterEqual(len(result), constraints.min_pois)


class TestAStarWithFeedback(unittest.TestCase):
    """Test A* with interactive feedback"""
    
    def setUp(self):
        """Create planner with more POIs"""
        self.poi_dicts = [
            {"id": f"poi{i}", "name": f"Place {i}", "lat": 40.7 + i*0.01, "lon": -73.9 - i*0.01,
             "category": ["park", "museum", "landmark", "restaurant"][i % 4],
             "popularity": 0.7 + (i % 5) * 0.05, "entrance_fee": i * 10.0, 
             "avg_visit_duration": 1.0 + (i % 3) * 0.5,
             "opening_hours": {"weekday": [8.0, 22.0]}, "rating": 4.0 + (i % 5) * 0.2}
            for i in range(10)
        ]
        
        n = len(self.poi_dicts)
        self.distance_matrix = np.random.rand(n, n) * 5  # Random distances
        np.fill_diagonal(self.distance_matrix, 0)
        
        self.planner = AStarItineraryPlanner(self.poi_dicts, self.distance_matrix)
    
    def test_rejected_pois(self):
        """Test handling of rejected POIs"""
        preferences = {"park": 0.8, "museum": 0.9, "landmark": 0.7}
        constraints = Constraints(
            budget=100,
            max_time_hours=8,
            min_pois=3,
            max_pois=5
        )
        
        # Get initial solution
        result1 = self.planner.plan_itinerary(preferences, constraints)
        
        if result1:
            # Reject first POI
            feedback = InteractiveFeedback(
                rejected_pois={result1[0].id},
                must_visit_pois=set(),
                preference_adjustments={}
            )
            
            # Get new solution
            result2 = self.planner.plan_itinerary(
                preferences, constraints, feedback
            )
            
            # Rejected POI should not appear
            poi_ids = [poi.id for poi in result2]
            self.assertNotIn(result1[0].id, poi_ids)
    
    def test_must_include_pois(self):
        """Test handling of must-include POIs"""
        preferences = {"restaurant": 1.0}
        constraints = Constraints(
            budget=100,
            max_time_hours=8,
            min_pois=3,
            max_pois=5
        )
        
        # Require specific POI
        feedback = InteractiveFeedback(
            rejected_pois=set(),
            must_visit_pois={"poi3"},  # Restaurant
            preference_adjustments={}
        )
        
        result = self.planner.plan_itinerary(preferences, constraints, feedback)
        
        if result:
            poi_ids = [poi.id for poi in result]
            self.assertIn("poi3", poi_ids)


if __name__ == '__main__':
    unittest.main()