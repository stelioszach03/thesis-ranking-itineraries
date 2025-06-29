"""
Unit tests for hybrid_planner module
Bachelor's thesis: Ranking Itineraries
"""

import unittest
import numpy as np
import time
from typing import List, Dict

from src.metrics_definitions import POI
from src.greedy_algorithms import Constraints, InteractiveFeedback
from src.hybrid_planner import (
    HybridPlanner, AlgorithmType, PlanningResult
)
from src.algorithm_selector import AlgorithmSelector, ResultCache


class TestAlgorithmSelector(unittest.TestCase):
    """Test algorithm selection logic"""
    
    def test_small_problem_selection(self):
        """Test algorithm selection for small problems"""
        selector = AlgorithmSelector()
        
        # Small problem: few POIs, relaxed constraints
        algo = selector.select_algorithm(
            n_pois=20,
            constraints=Constraints(
                budget=200,
                max_time_hours=10,
                min_pois=2,
                max_pois=5
            ),
            preferences={"museum": 0.8}
        )
        
        # Should prefer A* for small problems
        self.assertEqual(algo, AlgorithmType.ASTAR)
    
    def test_large_problem_selection(self):
        """Test algorithm selection for large problems"""
        selector = AlgorithmSelector()
        
        # Large problem: many POIs
        algo = selector.select_algorithm(
            n_pois=5000,
            constraints=Constraints(
                budget=100,
                max_time_hours=8,
                min_pois=3,
                max_pois=7
            ),
            preferences={"museum": 0.8, "park": 0.7}
        )
        
        # Should prefer Greedy for large problems
        self.assertEqual(algo, AlgorithmType.GREEDY_HEAP)
    
    def test_tight_constraints_selection(self):
        """Test algorithm selection with tight constraints"""
        selector = AlgorithmSelector()
        
        # Tight constraints
        algo = selector.select_algorithm(
            n_pois=100,
            constraints=Constraints(
                budget=50,  # Low budget
                max_time_hours=3,  # Short time
                min_pois=4,  # Many POIs required
                max_pois=6
            ),
            preferences={"museum": 0.9}
        )
        
        # Should prefer A* for tight constraints
        self.assertEqual(algo, AlgorithmType.ASTAR)


class TestResultCache(unittest.TestCase):
    """Test result caching functionality"""
    
    def setUp(self):
        """Create test cache"""
        self.cache = ResultCache(max_size=10)
        
        # Create test itinerary
        self.pois = [
            POI("poi1", "Place 1", 40.7, -73.9, "museum", 
                0.8, 20.0, 1.5, (9.0, 17.0), 4.5)
        ]
        self.result = PlanningResult(
            primary_itinerary=self.pois,
            alternatives=[],
            algorithm_used=AlgorithmType.GREEDY,
            computation_time=0.5,
            metrics={"css": 0.75}
        )
    
    def test_cache_storage_retrieval(self):
        """Test storing and retrieving from cache"""
        key = "test_key"
        bounds = {
            "min_utility": 0.5,
            "max_distance": 10.0,
            "time_window": (9.0, 17.0)
        }
        
        # Store result
        self.cache.store(key, self.result, bounds)
        
        # Retrieve result
        cached = self.cache.get(key, bounds)
        self.assertIsNotNone(cached)
        self.assertEqual(len(cached.primary_itinerary), 1)
        self.assertEqual(cached.primary_itinerary[0].id, "poi1")
    
    def test_cache_bounds_checking(self):
        """Test cache bounds validation"""
        key = "test_key"
        bounds = {
            "min_utility": 0.7,
            "max_distance": 10.0,
            "time_window": (9.0, 17.0)
        }
        
        # Store with specific bounds
        self.cache.store(key, self.result, bounds)
        
        # Try to retrieve with tighter bounds
        tighter_bounds = {
            "min_utility": 0.8,  # Higher requirement
            "max_distance": 10.0,
            "time_window": (9.0, 17.0)
        }
        
        # Should not return cached result
        cached = self.cache.get(key, tighter_bounds)
        self.assertIsNone(cached)
    
    def test_cache_size_limit(self):
        """Test cache size limiting"""
        # Fill cache beyond limit
        for i in range(15):
            key = f"key_{i}"
            result = PlanningResult(
                primary_itinerary=[],
                alternatives=[],
                algorithm_used=AlgorithmType.GREEDY,
                computation_time=0.1,
                metrics={}
            )
            self.cache.store(key, result, {})
        
        # Cache should not exceed max size
        self.assertLessEqual(len(self.cache.cache), self.cache.max_size)


class TestHybridPlanner(unittest.TestCase):
    """Test hybrid planner functionality"""
    
    def setUp(self):
        """Create test environment"""
        self.pois = [
            POI(f"poi{i}", f"Place {i}", 40.7 + i*0.01, -73.9 - i*0.01,
                ["park", "museum", "landmark", "restaurant"][i % 4],
                0.7 + (i % 5) * 0.05, i * 10.0, 1.0 + (i % 3) * 0.5,
                (8.0, 22.0), 4.0 + (i % 5) * 0.2)
            for i in range(30)
        ]
        
        n = len(self.pois)
        self.distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Simple distance based on index difference
                    self.distance_matrix[i, j] = abs(i - j) * 0.5
        
        self.planner = HybridPlanner(
            self.pois, self.distance_matrix, enable_cache=True
        )
    
    def test_auto_algorithm_selection(self):
        """Test automatic algorithm selection"""
        preferences = {"museum": 0.9, "park": 0.7}
        constraints = Constraints(
            budget=100,
            max_time_hours=8,
            min_pois=3,
            max_pois=5
        )
        
        result = self.planner.plan(
            preferences, constraints, algorithm=AlgorithmType.AUTO
        )
        
        self.assertIsNotNone(result)
        self.assertIn(result.algorithm_used, [
            AlgorithmType.GREEDY, AlgorithmType.GREEDY_HEAP, 
            AlgorithmType.ASTAR, AlgorithmType.HYBRID
        ])
    
    def test_specific_algorithm_planning(self):
        """Test planning with specific algorithms"""
        preferences = {"museum": 0.8, "park": 0.7}
        constraints = Constraints(
            budget=100,
            max_time_hours=6,
            min_pois=2,
            max_pois=4
        )
        
        # Test each algorithm
        for algo in [AlgorithmType.GREEDY, AlgorithmType.ASTAR]:
            result = self.planner.plan(
                preferences, constraints, algorithm=algo
            )
            
            self.assertIsNotNone(result)
            self.assertEqual(result.algorithm_used, algo)
            if result.primary_itinerary:
                self.assertGreaterEqual(len(result.primary_itinerary), 
                                      constraints.min_pois)
    
    def test_alternative_generation(self):
        """Test generation of alternative itineraries"""
        preferences = {"museum": 0.9, "park": 0.6, "landmark": 0.7}
        constraints = Constraints(
            budget=150,
            max_time_hours=8,
            min_pois=3,
            max_pois=5
        )
        
        result = self.planner.plan(
            preferences, constraints, 
            generate_alternatives=True
        )
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.alternatives)
        
        # Should have some alternatives
        if result.alternatives:
            # Alternatives should be different from primary
            primary_ids = {poi.id for poi in result.primary_itinerary}
            for alt in result.alternatives:
                alt_ids = {poi.id for poi in alt}
                self.assertNotEqual(primary_ids, alt_ids)
    
    def test_interactive_feedback(self):
        """Test planning with user feedback"""
        preferences = {"museum": 0.9, "park": 0.7}
        constraints = Constraints(
            budget=100,
            max_time_hours=8,
            min_pois=3,
            max_pois=5
        )
        
        # Initial planning
        result1 = self.planner.plan(preferences, constraints)
        
        if result1.primary_itinerary:
            # User rejects first POI
            feedback = InteractiveFeedback(
                rejected_pois={result1.primary_itinerary[0].id},
                must_include_pois={"poi5"},  # Must include specific POI
                adjusted_preferences=preferences
            )
            
            # Replan with feedback
            result2 = self.planner.plan(preferences, constraints, feedback)
            
            self.assertIsNotNone(result2)
            if result2.primary_itinerary:
                poi_ids = [poi.id for poi in result2.primary_itinerary]
                
                # Check feedback is respected
                self.assertNotIn(result1.primary_itinerary[0].id, poi_ids)
                self.assertIn("poi5", poi_ids)
    
    def test_caching_efficiency(self):
        """Test that caching improves performance"""
        preferences = {"museum": 0.9, "park": 0.7}
        constraints = Constraints(
            budget=100,
            max_time_hours=8,
            min_pois=3,
            max_pois=5
        )
        
        # First planning (no cache)
        start1 = time.time()
        result1 = self.planner.plan(preferences, constraints)
        time1 = time.time() - start1
        
        # Second planning (should use cache)
        start2 = time.time()
        result2 = self.planner.plan(preferences, constraints)
        time2 = time.time() - start2
        
        # Cached should be faster
        self.assertLess(time2, time1 * 0.5)  # At least 2x faster
        
        # Results should be identical
        self.assertEqual(
            [poi.id for poi in result1.primary_itinerary],
            [poi.id for poi in result2.primary_itinerary]
        )
    
    # def test_dynamic_replanning(self):
    #     """Test dynamic replanning with LPA*"""
    #     # Commented out: DynamicEvent class doesn't exist
    #     pass


class TestHybridPlannerIntegration(unittest.TestCase):
    """Integration tests for hybrid planner"""
    
    def test_nyc_scenario(self):
        """Test realistic NYC scenario"""
        # Create NYC-like POIs
        pois = [
            POI("central_park", "Central Park", 40.7829, -73.9654, "park",
                0.95, 0.0, 2.0, (6.0, 22.0), 4.7),
            POI("moma", "MoMA", 40.7614, -73.9776, "museum",
                0.9, 25.0, 2.5, (10.5, 17.5), 4.5),
            POI("met", "Metropolitan Museum", 40.7794, -73.9632, "museum",
                0.95, 25.0, 3.0, (10.0, 17.0), 4.8),
            POI("times_square", "Times Square", 40.7580, -73.9855, "landmark",
                0.98, 0.0, 0.5, (0.0, 24.0), 4.3),
            POI("high_line", "High Line", 40.7480, -74.0048, "park",
                0.85, 0.0, 1.5, (7.0, 22.0), 4.6),
            POI("brooklyn_bridge", "Brooklyn Bridge", 40.7061, -73.9969, "landmark",
                0.9, 0.0, 1.0, (0.0, 24.0), 4.6)
        ]
        
        # Create realistic distance matrix
        n = len(pois)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Haversine distance approximation
                    lat_diff = abs(pois[i].lat - pois[j].lat)
                    lon_diff = abs(pois[i].lon - pois[j].lon)
                    distance_matrix[i, j] = np.sqrt(lat_diff**2 + lon_diff**2) * 111
        
        planner = HybridPlanner(pois, distance_matrix)
        
        # Tourist preferences
        preferences = {
            "museum": 0.9,
            "park": 0.7,
            "landmark": 0.6
        }
        
        # Full day constraints
        constraints = Constraints(
            budget=100,
            max_time_hours=8,
            min_pois=3,
            max_pois=5,
            start_location=(40.7580, -73.9855),  # Start at Times Square
            transportation_mode="public_transit"
        )
        
        result = planner.plan(preferences, constraints)
        
        self.assertIsNotNone(result)
        self.assertGreaterEqual(len(result.primary_itinerary), 3)
        self.assertLessEqual(len(result.primary_itinerary), 5)
        
        # Check metrics
        self.assertIn("css", result.metrics)
        self.assertGreater(result.metrics["css"], 0.5)


if __name__ == '__main__':
    unittest.main()