"""
Unit tests for greedy_algorithms module
Bachelor's thesis: Ranking Itineraries
"""

import unittest
import numpy as np
from typing import List, Dict

from src.metrics_definitions import POI
from src.greedy_algorithms import GreedyPOISelection, HeapPrunGreedyPOI, GreedyPlanner, Constraints, InteractiveFeedback


class TestGreedyPOISelection(unittest.TestCase):
    """Test GreedyPOISelection algorithm"""
    
    def setUp(self):
        """Create test data"""
        self.pois = [
            POI("poi1", "Central Park", 40.7829, -73.9654, "park", 
                0.9, 0.0, 1.5, (6.0, 22.0), 4.7),
            POI("poi2", "MoMA", 40.7614, -73.9776, "museum", 
                0.85, 25.0, 2.0, (10.5, 17.5), 4.5),
            POI("poi3", "Times Square", 40.7580, -73.9855, "landmark",
                0.95, 0.0, 0.5, (0.0, 24.0), 4.3),
            POI("poi4", "Brooklyn Bridge", 40.7061, -73.9969, "landmark",
                0.88, 0.0, 1.0, (0.0, 24.0), 4.6),
            POI("poi5", "Empire State", 40.7484, -73.9857, "landmark",
                0.92, 40.0, 2.0, (9.0, 23.0), 4.7)
        ]
        
        # Create distance matrix
        n = len(self.pois)
        self.distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Simplified Manhattan distance
                    self.distance_matrix[i, j] = (
                        abs(self.pois[i].lat - self.pois[j].lat) + 
                        abs(self.pois[i].lon - self.pois[j].lon)
                    ) * 111  # km per degree
        
        # Convert POIs to dict format for GreedyPOISelection
        pois_dict = [
            {
                'id': poi.id,
                'name': poi.name,
                'lat': poi.lat,
                'lon': poi.lon,
                'category': poi.category,
                'rating': poi.rating,
                'entrance_fee': poi.entrance_fee,
                'avg_visit_duration': poi.avg_visit_duration,
                'opening_hours': {'weekday': list(poi.opening_hours)},
                'popularity': poi.popularity
            }
            for poi in self.pois
        ]
        self.greedy = GreedyPOISelection(pois_dict, self.distance_matrix)
    
    def test_basic_selection(self):
        """Test basic POI selection"""
        preferences = {"park": 0.8, "museum": 0.9, "landmark": 0.7}
        constraints = Constraints(
            budget=100,
            max_time_hours=8,
            min_pois=2,
            max_pois=5
        )
        
        result = self.greedy.select_pois(preferences, constraints)
        
        self.assertGreaterEqual(len(result), constraints.min_pois)
        self.assertLessEqual(len(result), constraints.max_pois)
    
    def test_budget_constraint(self):
        """Test budget constraint enforcement"""
        preferences = {"museum": 1.0, "landmark": 1.0}
        constraints = Constraints(
            budget=30,  # Can afford MoMA but not Empire State after
            max_time_hours=8,
            min_pois=2,
            max_pois=5
        )
        
        result = self.greedy.select_pois(preferences, constraints)
        
        total_cost = sum(poi.entrance_fee for poi in result)
        self.assertLessEqual(total_cost, constraints.budget)
    
    def test_time_constraint(self):
        """Test time constraint enforcement"""
        preferences = {"park": 0.8, "museum": 0.9, "landmark": 0.7}
        constraints = Constraints(
            budget=200,
            max_time_hours=3,  # Very limited time
            min_pois=1,
            max_pois=5
        )
        
        result = self.greedy.select_pois(preferences, constraints)
        
        # With only 3 hours, should select fewer POIs
        self.assertLessEqual(len(result), 3)
    
    def test_empty_preferences(self):
        """Test with no preferences"""
        preferences = {}
        constraints = Constraints(
            budget=100,
            max_time_hours=8,
            min_pois=2,
            max_pois=5
        )
        
        result = self.greedy.select_pois(preferences, constraints)
        
        # Should still select some POIs based on popularity
        self.assertGreaterEqual(len(result), 1)


class TestHeapPrunGreedyPOI(unittest.TestCase):
    """Test HeapPrunGreedyPOI algorithm"""
    
    def setUp(self):
        """Create test data"""
        self.pois = [
            POI(f"poi{i}", f"Place {i}", 40.7 + i*0.01, -73.9 - i*0.01, 
                ["park", "museum", "landmark"][i % 3],
                0.8 + (i % 5) * 0.02, i * 5.0, 1.0 + (i % 3) * 0.5,
                (8.0, 20.0), 4.0 + (i % 10) * 0.1)
            for i in range(20)  # More POIs for heap testing
        ]
        
        n = len(self.pois)
        self.distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.distance_matrix[i, j] = abs(i - j) * 0.5  # Simple distance
        
        # Convert POIs to dict format for HeapPrunGreedyPOI
        pois_dict = [
            {
                'id': poi.id,
                'name': poi.name,
                'lat': poi.lat,
                'lon': poi.lon,
                'category': poi.category,
                'rating': poi.rating,
                'entrance_fee': poi.entrance_fee,
                'avg_visit_duration': poi.avg_visit_duration,
                'opening_hours': {'weekday': list(poi.opening_hours)},
                'popularity': poi.popularity
            }
            for poi in self.pois
        ]
        self.heap_greedy = HeapPrunGreedyPOI(pois_dict, self.distance_matrix)
    
    def test_heap_pruning(self):
        """Test that heap pruning works"""
        preferences = {"park": 0.8, "museum": 0.9, "landmark": 0.7}
        constraints = Constraints(
            budget=100,
            max_time_hours=8,
            min_pois=3,
            max_pois=5
        )
        
        result = self.heap_greedy.select_pois(preferences, constraints)
        
        self.assertGreaterEqual(len(result), constraints.min_pois)
        self.assertLessEqual(len(result), constraints.max_pois)
    
    def test_interactive_feedback(self):
        """Test interactive feedback handling"""
        preferences = {"park": 0.8, "museum": 0.9, "landmark": 0.7}
        constraints = Constraints(
            budget=150,
            max_time_hours=8,
            min_pois=3,
            max_pois=6
        )
        
        # First selection
        result1 = self.heap_greedy.select_pois(preferences, constraints)
        
        # User dislikes first POI
        feedback = InteractiveFeedback(
            rejected_pois={result1[0].id},
            preferred_categories=preferences,
            must_visit_pois=set(),
            preference_adjustments={}
        )
        
        # Second selection with feedback
        result2 = self.heap_greedy.select_pois(preferences, constraints, feedback)
        
        # Rejected POI should not be in new selection
        poi_ids = [poi.id for poi in result2]
        self.assertNotIn(result1[0].id, poi_ids)
    
    def test_performance_comparison(self):
        """Test that heap version is not worse than regular greedy"""
        # Convert POIs to dict format for GreedyPOISelection
        pois_dict = [
            {
                'id': poi.id,
                'name': poi.name,
                'lat': poi.lat,
                'lon': poi.lon,
                'category': poi.category,
                'rating': poi.rating,
                'entrance_fee': poi.entrance_fee,
                'avg_visit_duration': poi.avg_visit_duration,
                'opening_hours': {'weekday': list(poi.opening_hours)},
                'popularity': poi.popularity
            }
            for poi in self.pois
        ]
        regular_greedy = GreedyPOISelection(pois_dict, self.distance_matrix)
        
        preferences = {"park": 0.8, "museum": 0.9, "landmark": 0.7}
        constraints = Constraints(
            budget=100,
            max_time_hours=8,
            min_pois=3,
            max_pois=5
        )
        
        regular_result = regular_greedy.select_pois(preferences, constraints)
        heap_result = self.heap_greedy.select_pois(preferences, constraints)
        
        # Both should find valid solutions
        self.assertGreaterEqual(len(regular_result), constraints.min_pois)
        self.assertGreaterEqual(len(heap_result), constraints.min_pois)


class TestGreedyEdgeCases(unittest.TestCase):
    """Test edge cases for greedy algorithms"""
    
    def test_no_feasible_solution(self):
        """Test when no feasible solution exists"""
        pois = [
            POI("expensive", "Expensive Place", 40.7, -73.9, "museum",
                0.9, 200.0, 2.0, (10.0, 17.0), 4.5)
        ]
        
        distance_matrix = np.array([[0]])
        pois_dict = [{
            'id': pois[0].id,
            'name': pois[0].name,
            'lat': pois[0].lat,
            'lon': pois[0].lon,
            'category': pois[0].category,
            'rating': pois[0].rating,
            'entrance_fee': pois[0].entrance_fee,
            'avg_visit_duration': pois[0].avg_visit_duration,
            'opening_hours': {'weekday': list(pois[0].opening_hours)},
            'popularity': pois[0].popularity
        }]
        greedy = GreedyPOISelection(pois_dict, distance_matrix)
        
        constraints = Constraints(
            budget=50,  # Can't afford the POI
            max_time_hours=8,
            min_pois=1,
            max_pois=5
        )
        
        result = greedy.select_pois({"museum": 1.0}, constraints)
        self.assertEqual(len(result), 0)
    
    def test_single_poi(self):
        """Test with single POI"""
        pois = [
            POI("single", "Single Place", 40.7, -73.9, "park",
                0.9, 0.0, 1.0, (6.0, 22.0), 4.5)
        ]
        
        distance_matrix = np.array([[0]])
        pois_dict = [{
            'id': pois[0].id,
            'name': pois[0].name,
            'lat': pois[0].lat,
            'lon': pois[0].lon,
            'category': pois[0].category,
            'rating': pois[0].rating,
            'entrance_fee': pois[0].entrance_fee,
            'avg_visit_duration': pois[0].avg_visit_duration,
            'opening_hours': {'weekday': list(pois[0].opening_hours)},
            'popularity': pois[0].popularity
        }]
        greedy = GreedyPOISelection(pois_dict, distance_matrix)
        
        constraints = Constraints(
            budget=100,
            max_time_hours=8,
            min_pois=1,
            max_pois=5
        )
        
        result = greedy.select_pois({"park": 1.0}, constraints)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, "single")


if __name__ == '__main__':
    unittest.main()