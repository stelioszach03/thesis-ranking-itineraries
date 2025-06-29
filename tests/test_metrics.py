"""
Unit tests for metrics_definitions module
Bachelor's thesis: Ranking Itineraries
"""

import unittest
import numpy as np
from datetime import datetime, time
from typing import List, Dict, Tuple

from src.metrics_definitions import (
    POI, Itinerary, TimeSlot, Constraints,
    QuantitativeMetrics, QualitativeMetrics, 
    CompositeMetrics, InteractiveFeedback
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
            rating=4.7,
            entrance_fee=0.0,
            avg_visit_duration=1.5,
            opening_hours=(6.0, 22.0),
            popularity_score=0.9,
            accessibility_score=0.95,
            weather_dependency=0.8
        )
        
        self.poi2 = POI(
            id="poi2",
            name="MoMA",
            lat=40.7614,
            lon=-73.9776,
            category="museum",
            rating=4.5,
            entrance_fee=25.0,
            avg_visit_duration=2.0,
            opening_hours=(10.5, 17.5),
            popularity_score=0.85,
            accessibility_score=0.9,
            weather_dependency=0.2
        )
    
    def test_poi_attributes(self):
        """Test POI basic attributes"""
        self.assertEqual(self.poi1.name, "Central Park")
        self.assertEqual(self.poi1.category, "park")
        self.assertEqual(self.poi1.entrance_fee, 0.0)
        self.assertAlmostEqual(self.poi1.lat, 40.7829)
        
    def test_poi_time_constraints(self):
        """Test POI time-related attributes"""
        self.assertEqual(self.poi2.opening_hours, (10.5, 17.5))
        self.assertEqual(self.poi2.avg_visit_duration, 2.0)


class TestQuantitativeMetrics(unittest.TestCase):
    """Test quantitative metrics calculations"""
    
    def setUp(self):
        """Create test itinerary"""
        self.pois = [
            POI("poi1", "Central Park", 40.7829, -73.9654, "park", 
                4.7, 0.0, 1.5, (6.0, 22.0), 0.9, 0.95, 0.8),
            POI("poi2", "MoMA", 40.7614, -73.9776, "museum", 
                4.5, 25.0, 2.0, (10.5, 17.5), 0.85, 0.9, 0.2),
            POI("poi3", "Times Square", 40.7580, -73.9855, "landmark",
                4.3, 0.0, 0.5, (0.0, 24.0), 0.95, 0.8, 0.5)
        ]
        
        self.itinerary = Itinerary(
            pois=self.pois,
            start_time=9.0,
            transportation_mode="public_transit",
            user_preferences={"park": 0.8, "museum": 0.9, "landmark": 0.7}
        )
    
    def test_total_distance(self):
        """Test total distance calculation"""
        distance = QuantitativeMetrics.total_distance(self.itinerary)
        self.assertGreater(distance, 0)
        self.assertLess(distance, 10)  # NYC scale
    
    def test_total_time(self):
        """Test total time calculation"""
        total_time = QuantitativeMetrics.total_time(self.itinerary)
        self.assertGreater(total_time, 4.0)  # At least visit durations
        self.assertLess(total_time, 12.0)  # Reasonable day trip
    
    def test_total_cost(self):
        """Test total cost calculation"""
        cost = QuantitativeMetrics.total_cost(self.itinerary)
        self.assertEqual(cost, 25.0)  # Only MoMA has entrance fee


class TestQualitativeMetrics(unittest.TestCase):
    """Test qualitative metrics calculations"""
    
    def setUp(self):
        """Create diverse test itinerary"""
        self.pois = [
            POI("poi1", "Central Park", 40.7829, -73.9654, "park", 
                4.7, 0.0, 1.5, (6.0, 22.0), 0.9, 0.95, 0.8),
            POI("poi2", "MoMA", 40.7614, -73.9776, "museum", 
                4.5, 25.0, 2.0, (10.5, 17.5), 0.85, 0.9, 0.2),
            POI("poi3", "Statue of Liberty", 40.6892, -74.0445, "landmark",
                4.6, 24.0, 3.0, (8.5, 17.0), 0.95, 0.7, 0.6)
        ]
        
        self.itinerary = Itinerary(
            pois=self.pois,
            start_time=9.0,
            transportation_mode="public_transit",
            user_preferences={"park": 0.8, "museum": 0.9, "landmark": 0.8}
        )
    
    def test_diversity_score(self):
        """Test diversity score calculation"""
        diversity = QualitativeMetrics.diversity_score(self.itinerary)
        self.assertGreater(diversity, 0.5)  # Good diversity
        self.assertLessEqual(diversity, 1.0)
    
    def test_attractiveness_score(self):
        """Test attractiveness score calculation"""
        attractiveness = QualitativeMetrics.attractiveness_score(self.itinerary)
        self.assertGreater(attractiveness, 0.7)  # High-rated POIs
        self.assertLessEqual(attractiveness, 1.0)
    
    def test_preference_alignment(self):
        """Test preference alignment calculation"""
        alignment = QualitativeMetrics.preference_alignment_score(self.itinerary)
        self.assertGreater(alignment, 0.6)
        self.assertLessEqual(alignment, 1.0)


class TestCompositeMetrics(unittest.TestCase):
    """Test composite metrics calculations"""
    
    def setUp(self):
        """Create test data"""
        self.pois = [
            POI("poi1", "Central Park", 40.7829, -73.9654, "park", 
                4.7, 0.0, 1.5, (6.0, 22.0), 0.9, 0.95, 0.8),
            POI("poi2", "MoMA", 40.7614, -73.9776, "museum", 
                4.5, 25.0, 2.0, (10.5, 17.5), 0.85, 0.9, 0.2)
        ]
        
        self.itinerary = Itinerary(
            pois=self.pois,
            start_time=9.0,
            transportation_mode="public_transit",
            user_preferences={"park": 0.8, "museum": 0.9}
        )
        
        self.constraints = Constraints(
            budget=100,
            max_time_hours=8,
            min_pois=2,
            max_pois=5
        )
    
    def test_composite_satisfaction_score(self):
        """Test CSS calculation"""
        css = CompositeMetrics.composite_satisfaction_score(
            self.itinerary, self.constraints
        )
        self.assertGreater(css, 0)
        self.assertLessEqual(css, 1.0)
    
    def test_css_weights(self):
        """Test CSS weight validation"""
        # Default weights should sum to 1
        weights = CompositeMetrics.DEFAULT_WEIGHTS
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=3)
        
        # Test custom weights
        custom_weights = {
            'attractiveness': 0.4,
            'time_efficiency': 0.3,
            'feasibility': 0.2,
            'diversity': 0.1
        }
        css = CompositeMetrics.composite_satisfaction_score(
            self.itinerary, self.constraints, custom_weights
        )
        self.assertGreater(css, 0)


if __name__ == '__main__':
    unittest.main()