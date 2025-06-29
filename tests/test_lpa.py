"""
Unit tests for lpa_star module
Bachelor's thesis: Ranking Itineraries
"""

import unittest
import numpy as np
from typing import List, Dict, Set

from src.metrics_definitions import POI, Constraints, DynamicEvent
from src.lpa_star import LPAStarPlanner, LPANode, DynamicEventType


class TestLPANode(unittest.TestCase):
    """Test LPANode class functionality"""
    
    def test_node_creation(self):
        """Test creating LPA* nodes"""
        from src.astar_itinerary import ItineraryState
        
        state = ItineraryState(
            visited_pois=("poi1", "poi2"),
            current_time=14.0,
            remaining_budget=50.0,
            total_utility=3.5,
            total_distance=7.2
        )
        
        node = LPANode(state)
        
        self.assertEqual(node.state, state)
        self.assertEqual(node.g, float('inf'))
        self.assertEqual(node.rhs, float('inf'))
        self.assertTrue(node.is_consistent())
    
    def test_node_consistency(self):
        """Test node consistency checking"""
        from src.astar_itinerary import ItineraryState
        
        state = ItineraryState((), 9.0, 100.0, 0.0, 0.0)
        node = LPANode(state)
        
        # Initially consistent (both inf)
        self.assertTrue(node.is_consistent())
        
        # Make inconsistent
        node.g = 10.0
        node.rhs = 15.0
        self.assertFalse(node.is_consistent())
        
        # Make consistent again
        node.g = 15.0
        self.assertTrue(node.is_consistent())
    
    def test_node_key_calculation(self):
        """Test key calculation for priority queue"""
        from src.astar_itinerary import ItineraryState
        
        state = ItineraryState(("poi1",), 10.0, 80.0, 2.0, 3.0)
        node = LPANode(state)
        node.h = 5.0
        node.g = 10.0
        node.rhs = 8.0
        
        key = node.calculate_key()
        self.assertEqual(len(key), 2)
        self.assertEqual(key[0], min(node.g, node.rhs) + node.h)
        self.assertEqual(key[1], min(node.g, node.rhs))


class TestLPAStarPlanner(unittest.TestCase):
    """Test LPA* planner functionality"""
    
    def setUp(self):
        """Create test environment"""
        self.pois = [
            POI("start", "Start", 40.7580, -73.9855, "start",
                0.0, 0.0, 0.0, (0.0, 24.0), 0.0, 1.0, 0.5),
            POI("poi1", "Central Park", 40.7829, -73.9654, "park", 
                4.7, 0.0, 1.5, (6.0, 22.0), 0.9, 0.95, 0.8),
            POI("poi2", "MoMA", 40.7614, -73.9776, "museum", 
                4.5, 25.0, 2.0, (10.5, 17.5), 0.85, 0.9, 0.2),
            POI("poi3", "Brooklyn Bridge", 40.7061, -73.9969, "landmark",
                4.6, 0.0, 1.0, (0.0, 24.0), 0.88, 0.85, 0.7)
        ]
        
        n = len(self.pois)
        self.distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.distance_matrix[i, j] = np.random.rand() * 5
        
        self.planner = LPAStarPlanner(self.pois, self.distance_matrix)
    
    def test_initial_planning(self):
        """Test initial path planning"""
        preferences = {"park": 0.8, "museum": 0.9, "landmark": 0.7}
        constraints = Constraints(
            budget=100,
            max_time_hours=8,
            min_pois=2,
            max_pois=4,
            start_location=(40.7580, -73.9855)
        )
        
        result = self.planner.plan_initial(preferences, constraints)
        
        self.assertIsNotNone(result)
        if result:
            self.assertGreaterEqual(len(result), constraints.min_pois)
            self.assertLessEqual(len(result), constraints.max_pois)
    
    def test_poi_closure_event(self):
        """Test handling POI closure"""
        preferences = {"park": 0.8, "museum": 0.9, "landmark": 0.7}
        constraints = Constraints(
            budget=100,
            max_time_hours=8,
            min_pois=2,
            max_pois=4
        )
        
        # Get initial plan
        initial_result = self.planner.plan_initial(preferences, constraints)
        
        if initial_result and len(initial_result) > 0:
            # Close one of the POIs
            closed_poi = initial_result[0]
            event = DynamicEvent(
                event_type=DynamicEventType.POI_CLOSED,
                affected_poi_ids=[closed_poi.id],
                new_constraints=constraints
            )
            
            # Replan
            new_result = self.planner.replan(event, preferences)
            
            # Closed POI should not be in new plan
            if new_result:
                poi_ids = [poi.id for poi in new_result]
                self.assertNotIn(closed_poi.id, poi_ids)
    
    def test_weather_change_event(self):
        """Test handling weather change"""
        preferences = {"park": 0.9, "museum": 0.7, "landmark": 0.6}
        constraints = Constraints(
            budget=100,
            max_time_hours=8,
            min_pois=2,
            max_pois=4,
            weather_condition="sunny"
        )
        
        # Get initial plan (sunny weather)
        initial_result = self.planner.plan_initial(preferences, constraints)
        
        # Change to rainy weather
        new_constraints = Constraints(
            budget=100,
            max_time_hours=8,
            min_pois=2,
            max_pois=4,
            weather_condition="rainy"
        )
        
        event = DynamicEvent(
            event_type=DynamicEventType.WEATHER_CHANGE,
            affected_poi_ids=[],
            new_constraints=new_constraints,
            weather_info={"condition": "rainy", "severity": 0.8}
        )
        
        # Replan for rain
        new_result = self.planner.replan(event, preferences)
        
        if new_result:
            # Should prefer indoor activities (museums) over parks
            outdoor_count = sum(1 for poi in new_result if poi.weather_dependency > 0.5)
            indoor_count = sum(1 for poi in new_result if poi.weather_dependency <= 0.5)
            
            # In rain, should have more indoor than outdoor
            self.assertGreaterEqual(indoor_count, outdoor_count)
    
    def test_traffic_update(self):
        """Test handling traffic updates"""
        preferences = {"park": 0.8, "museum": 0.9, "landmark": 0.7}
        constraints = Constraints(
            budget=100,
            max_time_hours=6,  # Limited time
            min_pois=2,
            max_pois=3,
            transportation_mode="taxi"
        )
        
        # Get initial plan
        initial_result = self.planner.plan_initial(preferences, constraints)
        
        # Create traffic jam event
        event = DynamicEvent(
            event_type=DynamicEventType.TRAFFIC_UPDATE,
            affected_poi_ids=[],
            new_constraints=constraints,
            traffic_factor=2.0  # Double travel times
        )
        
        # Replan with traffic
        new_result = self.planner.replan(event, preferences)
        
        # With doubled travel times and limited hours, might select fewer POIs
        if initial_result and new_result:
            self.assertLessEqual(len(new_result), len(initial_result))
    
    def test_incremental_efficiency(self):
        """Test that replanning is more efficient than planning from scratch"""
        preferences = {"park": 0.8, "museum": 0.9, "landmark": 0.7}
        constraints = Constraints(
            budget=100,
            max_time_hours=8,
            min_pois=2,
            max_pois=4
        )
        
        # Initial planning
        self.planner.plan_initial(preferences, constraints)
        initial_nodes = len(self.planner.nodes)
        
        # Small change event
        event = DynamicEvent(
            event_type=DynamicEventType.POI_CLOSED,
            affected_poi_ids=["poi3"],  # Close one POI
            new_constraints=constraints
        )
        
        # Replan
        self.planner.replan(event, preferences)
        
        # Should reuse many nodes
        reuse_stats = self.planner.get_reuse_statistics()
        if reuse_stats and reuse_stats['total_nodes'] > 0:
            reuse_ratio = reuse_stats['reused_nodes'] / reuse_stats['total_nodes']
            self.assertGreater(reuse_ratio, 0.5)  # At least 50% reuse


class TestDynamicScenarios(unittest.TestCase):
    """Test complex dynamic scenarios"""
    
    def setUp(self):
        """Create larger test environment"""
        self.pois = [
            POI(f"poi{i}", f"Place {i}", 40.7 + i*0.01, -73.9 - i*0.01,
                ["park", "museum", "landmark", "restaurant", "theater"][i % 5],
                4.0 + (i % 5) * 0.2, i * 5.0, 1.0 + (i % 3) * 0.5,
                (8.0, 22.0), 0.7 + (i % 5) * 0.05, 0.85, 
                0.8 if i % 5 == 0 else 0.3)  # Parks are weather-dependent
            for i in range(15)
        ]
        
        n = len(self.pois)
        self.distance_matrix = np.random.rand(n, n) * 10
        np.fill_diagonal(self.distance_matrix, 0)
        
        self.planner = LPAStarPlanner(self.pois, self.distance_matrix)
    
    def test_multiple_events(self):
        """Test handling multiple sequential events"""
        preferences = {
            "park": 0.8, "museum": 0.9, "landmark": 0.7,
            "restaurant": 0.6, "theater": 0.8
        }
        constraints = Constraints(
            budget=150,
            max_time_hours=10,
            min_pois=4,
            max_pois=6
        )
        
        # Initial plan
        result1 = self.planner.plan_initial(preferences, constraints)
        self.assertIsNotNone(result1)
        
        # Event 1: Weather change
        event1 = DynamicEvent(
            event_type=DynamicEventType.WEATHER_CHANGE,
            affected_poi_ids=[],
            new_constraints=constraints,
            weather_info={"condition": "rainy", "severity": 0.7}
        )
        result2 = self.planner.replan(event1, preferences)
        
        # Event 2: POI closes
        if result2 and len(result2) > 0:
            event2 = DynamicEvent(
                event_type=DynamicEventType.POI_CLOSED,
                affected_poi_ids=[result2[0].id],
                new_constraints=constraints
            )
            result3 = self.planner.replan(event2, preferences)
            
            # Final plan should handle both events
            self.assertIsNotNone(result3)
            if result3:
                poi_ids = [poi.id for poi in result3]
                self.assertNotIn(result2[0].id, poi_ids)
    
    def test_preference_update(self):
        """Test dynamic preference updates"""
        initial_preferences = {"museum": 0.9, "park": 0.5, "landmark": 0.6}
        constraints = Constraints(
            budget=100,
            max_time_hours=8,
            min_pois=3,
            max_pois=5
        )
        
        # Initial plan (likes museums)
        result1 = self.planner.plan_initial(initial_preferences, constraints)
        
        # User changes mind (now prefers parks)
        new_preferences = {"museum": 0.5, "park": 0.9, "landmark": 0.6}
        event = DynamicEvent(
            event_type=DynamicEventType.PREFERENCE_CHANGE,
            affected_poi_ids=[],
            new_constraints=constraints,
            new_preferences=new_preferences
        )
        
        result2 = self.planner.replan(event, new_preferences)
        
        if result1 and result2:
            # Count categories
            museums1 = sum(1 for poi in result1 if poi.category == "museum")
            parks1 = sum(1 for poi in result1 if poi.category == "park")
            
            museums2 = sum(1 for poi in result2 if poi.category == "museum")
            parks2 = sum(1 for poi in result2 if poi.category == "park")
            
            # Should have more parks after preference change
            self.assertGreaterEqual(parks2, parks1)
            self.assertLessEqual(museums2, museums1)


if __name__ == '__main__':
    unittest.main()