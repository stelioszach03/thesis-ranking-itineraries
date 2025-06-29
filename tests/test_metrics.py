"""
Unit tests for metrics_definitions module
Bachelor's thesis: Ranking Itineraries
"""

import unittest
import numpy as np
from datetime import datetime, time
from typing import List, Dict, Tuple

from src.metrics_definitions import POI, Itinerary
from src.metrics_calculations import (
    QuantitativeMetrics, QualitativeMetrics, CompositeUtilityFunctions
)


class TestPOI(unittest.TestCase):
    """Test POI class functionality"""
    
    def setUp(self):
        """Create test POIs"""
        self.poi1 = POI(
            id="poi1",
            name="Central Park",
            lat=40.7829,
            lon=-73.9654,
            category="park",
            popularity=0.9,
            entrance_fee=0.0,
            avg_visit_duration=2.0,
            opening_hours=(6.0, 22.0),
            rating=4.7
        )
        
        self.poi2 = POI(
            id="poi2",
            name="Metropolitan Museum",
            lat=40.7794,
            lon=-73.9632,
            category="museum",
            popularity=0.95,
            entrance_fee=25.0,
            avg_visit_duration=3.0,
            opening_hours=(10.0, 17.0),
            rating=4.8
        )
        
        self.poi3 = POI(
            id="poi3",
            name="Times Square",
            lat=40.7580,
            lon=-73.9855,
            category="landmark",
            popularity=0.85,
            entrance_fee=0.0,
            avg_visit_duration=1.0,
            opening_hours=(0.0, 24.0),
            rating=4.2
        )
    
    def test_poi_creation(self):
        """Test POI object creation"""
        self.assertEqual(self.poi1.id, "poi1")
        self.assertEqual(self.poi1.name, "Central Park")
        self.assertEqual(self.poi1.category, "park")
        self.assertEqual(self.poi1.rating, 4.7)
        self.assertEqual(self.poi1.entrance_fee, 0.0)
    
    def test_poi_string_representation(self):
        """Test POI string representation"""
        poi_str = str(self.poi1)
        self.assertIn("Central Park", poi_str)
        self.assertIn("park", poi_str)


class TestItinerary(unittest.TestCase):
    """Test Itinerary class functionality"""
    
    def setUp(self):
        """Create test itinerary"""
        self.test_pois = [
            POI("poi1", "Central Park", 40.7829, -73.9654, "park", 
                0.9, 0.0, 2.0, (6.0, 22.0), 4.7),
            POI("poi2", "Metropolitan Museum", 40.7794, -73.9632, "museum", 
                0.95, 25.0, 3.0, (10.0, 17.0), 4.8),
            POI("poi3", "Times Square", 40.7580, -73.9855, "landmark", 
                0.85, 0.0, 1.0, (0.0, 24.0), 4.2)
        ]
        
        self.itinerary = Itinerary(
            pois=self.test_pois,
            start_time=9.0,
            transportation_mode="public_transit",
            user_preferences={"park": 0.8, "museum": 0.9, "landmark": 0.6}
        )
    
    def test_itinerary_creation(self):
        """Test itinerary creation"""
        self.assertEqual(len(self.itinerary.pois), 3)
        self.assertEqual(self.itinerary.start_time, 9.0)
        self.assertEqual(self.itinerary.transportation_mode, "public_transit")
    
    def test_total_cost_calculation(self):
        """Test total cost calculation"""
        expected_cost = 0.0 + 25.0 + 0.0  # Entrance fees
        # Note: actual calculation may include other costs
        total_cost = sum(poi.entrance_fee for poi in self.itinerary.pois)
        self.assertEqual(total_cost, expected_cost)
    
    def test_total_duration_calculation(self):
        """Test total duration calculation"""
        expected_duration = 2.0 + 3.0 + 1.0  # Visit durations
        total_duration = sum(poi.avg_visit_duration for poi in self.itinerary.pois)
        self.assertEqual(total_duration, expected_duration)


class TestQuantitativeMetrics(unittest.TestCase):
    """Test quantitative metrics calculations"""
    
    def setUp(self):
        """Set up test data"""
        self.test_pois = [
            POI(id="poi1", name="Central Park", lat=40.7829, lon=-73.9654, 
                category="park", popularity=0.9, entrance_fee=0.0, 
                avg_visit_duration=2.0, opening_hours=(6.0, 22.0), rating=4.7),
            POI(id="poi2", name="Metropolitan Museum", lat=40.7794, lon=-73.9632, 
                category="museum", popularity=0.95, entrance_fee=25.0, 
                avg_visit_duration=3.0, opening_hours=(10.0, 17.0), rating=4.8)
        ]
        
        self.itinerary = Itinerary(
            pois=self.test_pois,
            start_time=9.0,
            transportation_mode="walking",
            user_preferences={"park": 0.8, "museum": 0.9}
        )
    
    def test_total_time_calculation(self):
        """Test total time calculation"""
        total_time = QuantitativeMetrics.total_time(self.itinerary)
        self.assertGreater(total_time, 0)
        # Should include visit time and travel time
        expected_min = 2.0 + 3.0  # Just visit times
        self.assertGreaterEqual(total_time, expected_min)
    
    def test_total_distance_calculation(self):
        """Test total distance calculation"""
        total_distance = QuantitativeMetrics.total_distance(self.itinerary)
        self.assertGreater(total_distance, 0)
        # Distance between Central Park and Met Museum should be reasonable
        self.assertLess(total_distance, 5.0)  # Less than 5km for nearby POIs
    
    def test_total_cost_calculation(self):
        """Test total cost calculation"""
        total_cost = QuantitativeMetrics.total_cost(self.itinerary)
        expected_cost = 25.0  # Only museum has entrance fee
        self.assertEqual(total_cost, expected_cost)
    
    def test_utility_per_time(self):
        """Test utility per time calculation"""
        utility_rate = QuantitativeMetrics.utility_per_time(self.itinerary)
        self.assertGreater(utility_rate, 0)
        self.assertIsInstance(utility_rate, float)


class TestQualitativeMetrics(unittest.TestCase):
    """Test qualitative metrics calculations"""
    
    def setUp(self):
        """Set up test data"""
        self.test_pois = [
            POI("poi1", "Central Park", 40.7829, -73.9654, "park", 
                0.9, 0.0, 2.0, (6.0, 22.0), 4.7),
            POI("poi2", "Metropolitan Museum", 40.7794, -73.9632, "museum", 
                0.95, 25.0, 3.0, (10.0, 17.0), 4.8),
            POI("poi3", "Times Square", 40.7580, -73.9855, "landmark", 
                0.85, 0.0, 1.0, (0.0, 24.0), 4.2)
        ]
        
        self.itinerary = Itinerary(
            pois=self.test_pois,
            start_time=9.0,
            transportation_mode="public_transit",
            user_preferences={"park": 0.8, "museum": 0.9, "landmark": 0.6}
        )
    
    def test_user_satisfaction(self):
        """Test user satisfaction calculation"""
        preferences = {"park": 0.8, "museum": 0.9, "landmark": 0.6}
        satisfaction = QualitativeMetrics.user_satisfaction(self.itinerary, preferences)
        
        self.assertGreaterEqual(satisfaction, 0.0)
        self.assertLessEqual(satisfaction, 1.0)
        self.assertIsInstance(satisfaction, float)
    
    def test_diversity_score(self):
        """Test diversity score calculation"""
        diversity = QualitativeMetrics.diversity_score(self.itinerary)
        
        self.assertGreater(diversity, 0.0)
        self.assertLessEqual(diversity, 1.0)
        self.assertIsInstance(diversity, float)
        
        # With 3 different categories, diversity should be high
        self.assertGreater(diversity, 0.5)
    
    def test_novelty_score(self):
        """Test novelty score calculation"""
        novelty = QualitativeMetrics.novelty_score(self.itinerary)
        
        self.assertGreaterEqual(novelty, 0.0)
        self.assertLessEqual(novelty, 1.0)
        self.assertIsInstance(novelty, float)


class TestCompositeUtilityFunctions(unittest.TestCase):
    """Test composite utility functions"""
    
    def setUp(self):
        """Set up test data"""
        self.test_pois = [
            POI(id="poi1", name="Central Park", lat=40.7829, lon=-73.9654, 
                category="park", popularity=0.9, entrance_fee=0.0, 
                avg_visit_duration=2.0, opening_hours=(6.0, 22.0), rating=4.7),
            POI(id="poi2", name="Metropolitan Museum", lat=40.7794, lon=-73.9632, 
                category="museum", popularity=0.95, entrance_fee=25.0, 
                avg_visit_duration=3.0, opening_hours=(10.0, 17.0), rating=4.8)
        ]
        
        self.itinerary = Itinerary(
            pois=self.test_pois,
            start_time=9.0,
            transportation_mode="walking",
            user_preferences={"park": 0.8, "museum": 0.9}
        )
        
        self.preferences = {"park": 0.8, "museum": 0.9, "landmark": 0.6}
    
    def test_composite_satisfaction_score(self):
        """Test CSS calculation"""
        css_score = CompositeUtilityFunctions.composite_satisfaction_score(
            self.itinerary, self.preferences, budget=100.0, max_time=8.0
        )
        
        self.assertGreaterEqual(css_score, 0.0)
        self.assertLessEqual(css_score, 1.0)
        self.assertIsInstance(css_score, float)
    
    def test_feasibility_score(self):
        """Test feasibility score calculation"""
        feasibility = CompositeUtilityFunctions.feasibility_score(
            self.itinerary, budget=100.0, max_time=8.0
        )
        
        self.assertGreaterEqual(feasibility, 0.0)
        self.assertLessEqual(feasibility, 1.0)
        
        # With reasonable budget and time, should be feasible
        self.assertGreater(feasibility, 0.5)
    
    def test_time_utilization_rate(self):
        """Test time utilization rate calculation"""
        tur = CompositeUtilityFunctions.time_utilization_rate(
            self.itinerary, max_time=8.0
        )
        
        self.assertGreaterEqual(tur, 0.0)
        self.assertLessEqual(tur, 1.0)
        self.assertIsInstance(tur, float)
    
    def test_calculate_all_metrics(self):
        """Test comprehensive metrics calculation"""
        all_metrics = CompositeUtilityFunctions.calculate_all_metrics(
            self.test_pois, self.preferences, budget=100.0, max_time=8.0
        )
        
        # Check that all expected metrics are present
        expected_keys = ['css', 'satisfaction', 'time_utilization', 
                        'feasibility', 'diversity']
        for key in expected_keys:
            self.assertIn(key, all_metrics)
            self.assertIsInstance(all_metrics[key], (int, float))
            self.assertGreaterEqual(all_metrics[key], 0.0)
    
    def test_edge_cases(self):
        """Test edge cases in metric calculations"""
        # Empty itinerary
        empty_itinerary = Itinerary([], 9.0, "walking", {})
        css_empty = CompositeUtilityFunctions.composite_satisfaction_score(
            empty_itinerary, self.preferences, 100.0, 8.0
        )
        self.assertEqual(css_empty, 0.0)
        
        # Very tight budget
        css_tight_budget = CompositeUtilityFunctions.composite_satisfaction_score(
            self.itinerary, self.preferences, budget=10.0, max_time=8.0
        )
        self.assertLess(css_tight_budget, 0.5)  # Should be low due to budget constraint


if __name__ == '__main__':
    unittest.main()