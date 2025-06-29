"""
NYC Tourist Profile Benchmark Scenarios

Based on TravelPlanner benchmark (1,225 queries) adapted for NYC
Covers diverse tourist profiles, trip durations, and real NYC events

References:
- TravelPlanner benchmark format from research_context.md
- User preference patterns from Lim et al. (2018)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, time
import json


@dataclass
class TouristProfile:
    """Tourist profile with preferences and constraints"""
    profile_id: str
    name: str
    description: str
    preferences: Dict[str, float]
    budget_per_day: float
    preferred_transport: str
    max_walking_distance_km: float
    daily_pois_preference: int  # 3-7 from research
    accessibility_needs: Optional[str] = None
    dietary_restrictions: Optional[List[str]] = None


@dataclass
class TripScenario:
    """Complete trip scenario for benchmarking"""
    scenario_id: str
    profile: TouristProfile
    duration: str  # half-day, full-day, weekend
    season: str  # spring, summer, fall, winter
    start_date: datetime
    special_events: List[str]
    weather_condition: str
    start_location: Dict[str, float]  # lat, lon
    constraints: Dict[str, any]


# Tourist Profiles based on TravelPlanner categories
TOURIST_PROFILES = {
    "first_time_tourist": TouristProfile(
        profile_id="FT001",
        name="First-Time Tourist",
        description="Visiting NYC for the first time, wants to see major attractions",
        preferences={
            'landmark': 0.95,
            'museum': 0.70,
            'park': 0.60,
            'restaurant': 0.75,
            'entertainment': 0.65,
            'shopping': 0.50,
            'nature': 0.40,
            'cultural': 0.60
        },
        budget_per_day=200.0,
        preferred_transport="public_transit",
        max_walking_distance_km=2.0,
        daily_pois_preference=5
    ),
    
    "repeat_visitor": TouristProfile(
        profile_id="RV001",
        name="Repeat Visitor",
        description="Been to NYC before, looking for hidden gems and new experiences",
        preferences={
            'landmark': 0.40,  # Lower - already seen major ones
            'museum': 0.80,
            'park': 0.70,
            'restaurant': 0.90,
            'entertainment': 0.85,
            'shopping': 0.60,
            'nature': 0.65,
            'cultural': 0.85
        },
        budget_per_day=150.0,
        preferred_transport="walking",
        max_walking_distance_km=3.0,
        daily_pois_preference=4
    ),
    
    "budget_traveler": TouristProfile(
        profile_id="BT001",
        name="Budget Traveler",
        description="Cost-conscious traveler, prefers free/cheap attractions",
        preferences={
            'landmark': 0.80,
            'museum': 0.50,  # Museums often have fees
            'park': 0.90,  # Free
            'restaurant': 0.60,
            'entertainment': 0.40,
            'shopping': 0.30,
            'nature': 0.85,
            'cultural': 0.70
        },
        budget_per_day=75.0,
        preferred_transport="walking",
        max_walking_distance_km=4.0,
        daily_pois_preference=6
    ),
    
    "luxury_traveler": TouristProfile(
        profile_id="LX001",
        name="Luxury Traveler",
        description="High-end experiences, comfort and exclusivity",
        preferences={
            'landmark': 0.70,
            'museum': 0.90,
            'park': 0.50,
            'restaurant': 0.95,  # Fine dining
            'entertainment': 0.90,  # Broadway shows
            'shopping': 0.95,  # High-end shopping
            'nature': 0.40,
            'cultural': 0.85
        },
        budget_per_day=500.0,
        preferred_transport="taxi",
        max_walking_distance_km=1.0,
        daily_pois_preference=3  # Quality over quantity
    ),
    
    "family_tourist": TouristProfile(
        profile_id="FM001",
        name="Family with Children",
        description="Traveling with kids aged 6-12, need family-friendly activities",
        preferences={
            'landmark': 0.70,
            'museum': 0.60,  # Kid-friendly museums
            'park': 0.95,  # Playgrounds
            'restaurant': 0.70,
            'entertainment': 0.80,
            'shopping': 0.40,
            'nature': 0.90,  # Zoo, aquarium
            'cultural': 0.50
        },
        budget_per_day=300.0,
        preferred_transport="public_transit",
        max_walking_distance_km=1.5,
        daily_pois_preference=4,
        accessibility_needs="stroller_friendly"
    ),
    
    "senior_tourist": TouristProfile(
        profile_id="SR001",
        name="Senior Tourist",
        description="Older traveler, mobility considerations",
        preferences={
            'landmark': 0.80,
            'museum': 0.85,
            'park': 0.60,
            'restaurant': 0.80,
            'entertainment': 0.70,
            'shopping': 0.50,
            'nature': 0.70,
            'cultural': 0.90
        },
        budget_per_day=250.0,
        preferred_transport="taxi",
        max_walking_distance_km=1.0,
        daily_pois_preference=3,
        accessibility_needs="wheelchair_accessible"
    ),
    
    "foodie": TouristProfile(
        profile_id="FD001",
        name="Food Enthusiast",
        description="Primary focus on culinary experiences",
        preferences={
            'landmark': 0.50,
            'museum': 0.40,
            'park': 0.60,
            'restaurant': 0.98,  # Primary focus
            'entertainment': 0.60,
            'shopping': 0.70,  # Food markets
            'nature': 0.40,
            'cultural': 0.80  # Food culture
        },
        budget_per_day=300.0,
        preferred_transport="public_transit",
        max_walking_distance_km=2.5,
        daily_pois_preference=5,
        dietary_restrictions=["vegetarian_options"]
    ),
    
    "culture_enthusiast": TouristProfile(
        profile_id="CE001",
        name="Culture & Arts Lover",
        description="Interested in museums, galleries, theaters",
        preferences={
            'landmark': 0.60,
            'museum': 0.95,
            'park': 0.50,
            'restaurant': 0.70,
            'entertainment': 0.90,  # Theater
            'shopping': 0.40,
            'nature': 0.40,
            'cultural': 0.98
        },
        budget_per_day=250.0,
        preferred_transport="public_transit",
        max_walking_distance_km=2.0,
        daily_pois_preference=4
    )
}


# Trip Duration Templates
TRIP_DURATIONS = {
    "half_day": {
        "hours": 4,
        "start_times": [9.0, 14.0],  # Morning or afternoon
        "max_pois": 3,
        "constraints": {
            "max_time_hours": 4.0,
            "min_pois": 2,
            "max_pois": 3
        }
    },
    "full_day": {
        "hours": 10,
        "start_times": [9.0],
        "max_pois": 7,
        "constraints": {
            "max_time_hours": 10.0,
            "min_pois": 4,
            "max_pois": 7
        }
    },
    "weekend": {
        "hours": 20,  # 2 days × 10 hours
        "start_times": [9.0],
        "max_pois": 14,
        "constraints": {
            "max_time_hours": 20.0,
            "min_pois": 8,
            "max_pois": 14
        }
    }
}


# NYC Special Events
NYC_EVENTS = {
    "broadway_week": {
        "name": "Broadway Week",
        "months": [1, 9],  # January and September
        "impact": {
            "entertainment": 1.5,  # 50% boost to entertainment preference
            "cost_multiplier": {"entertainment": 0.5}  # 2-for-1 tickets
        }
    },
    "restaurant_week": {
        "name": "Restaurant Week",
        "months": [1, 7],  # January and July
        "impact": {
            "restaurant": 1.3,
            "cost_multiplier": {"restaurant": 0.7}  # Prix fixe menus
        }
    },
    "summer_streets": {
        "name": "Summer Streets",
        "months": [8],  # August Saturdays
        "impact": {
            "park": 1.2,
            "nature": 1.2,
            "transportation": "walking"  # Car-free streets
        }
    },
    "holiday_season": {
        "name": "Holiday Season",
        "months": [11, 12],  # November-December
        "impact": {
            "shopping": 1.4,
            "landmark": 1.2,  # Tree, decorations
            "crowds": 1.5  # More crowded
        }
    },
    "fleet_week": {
        "name": "Fleet Week",
        "months": [5],  # May
        "impact": {
            "landmark": 1.1,
            "cultural": 1.2
        }
    },
    "nyc_marathon": {
        "name": "NYC Marathon",
        "months": [11],  # First Sunday of November
        "impact": {
            "transportation": "limited",  # Road closures
            "park": 0.8  # Some parks affected
        }
    }
}


# Weather Scenarios
WEATHER_CONDITIONS = {
    "sunny": {
        "name": "Sunny",
        "impact": {
            "park": 1.1,
            "nature": 1.1,
            "walking_preference": 1.2
        }
    },
    "rainy": {
        "name": "Rainy",
        "impact": {
            "park": 0.5,
            "nature": 0.5,
            "museum": 1.3,  # Indoor preference
            "shopping": 1.2,
            "walking_preference": 0.7
        }
    },
    "snowy": {
        "name": "Snowy",
        "impact": {
            "park": 0.7,
            "museum": 1.2,
            "transportation": "public_transit",  # Avoid walking/taxi
            "walking_preference": 0.5
        }
    },
    "heatwave": {
        "name": "Heat Wave",
        "impact": {
            "park": 0.8,
            "museum": 1.2,  # Air conditioning
            "shopping": 1.1,
            "visit_duration": 0.8  # Shorter outdoor visits
        }
    }
}


# Starting Locations in NYC
NYC_START_LOCATIONS = {
    "times_square": {"name": "Times Square Hotel", "lat": 40.7580, "lon": -73.9855},
    "grand_central": {"name": "Grand Central Area", "lat": 40.7527, "lon": -73.9772},
    "union_square": {"name": "Union Square", "lat": 40.7359, "lon": -73.9903},
    "columbus_circle": {"name": "Columbus Circle", "lat": 40.7681, "lon": -73.9819},
    "wall_street": {"name": "Financial District", "lat": 40.7074, "lon": -74.0113},
    "williamsburg": {"name": "Williamsburg", "lat": 40.7081, "lon": -73.9571},
    "lic": {"name": "Long Island City", "lat": 40.7447, "lon": -73.9485},
    "harlem": {"name": "Harlem", "lat": 40.8116, "lon": -73.9465}
}


def generate_benchmark_scenarios() -> List[TripScenario]:
    """
    Generate comprehensive benchmark scenarios
    
    Similar to TravelPlanner's 1,225 queries but NYC-specific
    """
    scenarios = []
    scenario_id = 1
    
    # Generate combinations
    for profile_key, profile in TOURIST_PROFILES.items():
        for duration_key, duration_config in TRIP_DURATIONS.items():
            for season in ["spring", "summer", "fall", "winter"]:
                for location_key, location in list(NYC_START_LOCATIONS.items())[:4]:  # Top 4 locations
                    # Determine applicable events
                    month = {"spring": 4, "summer": 7, "fall": 10, "winter": 1}[season]
                    applicable_events = [
                        event_key for event_key, event in NYC_EVENTS.items()
                        if month in event.get("months", [])
                    ]
                    
                    # Weather based on season
                    weather = {
                        "spring": "sunny",
                        "summer": "sunny" if scenario_id % 3 != 0 else "heatwave",
                        "fall": "sunny",
                        "winter": "snowy" if scenario_id % 2 == 0 else "rainy"
                    }[season]
                    
                    # Create scenario
                    scenario = TripScenario(
                        scenario_id=f"NYC_{scenario_id:04d}",
                        profile=profile,
                        duration=duration_key,
                        season=season,
                        start_date=datetime(2024, month, 15, 9, 0),
                        special_events=applicable_events[:1],  # One event max
                        weather_condition=weather,
                        start_location=location,
                        constraints={
                            **duration_config["constraints"],
                            "budget": profile.budget_per_day * (
                                2 if duration_key == "weekend" else 1
                            ),
                            "transportation_mode": profile.preferred_transport,
                            "max_walking_distance_km": profile.max_walking_distance_km,
                            "avoid_rush_hours": profile_key not in ["budget_traveler"],
                            "accessibility": profile.accessibility_needs
                        }
                    )
                    
                    scenarios.append(scenario)
                    scenario_id += 1
    
    return scenarios


def export_scenarios(scenarios: List[TripScenario], output_file: str):
    """Export scenarios to JSON format"""
    scenarios_dict = []
    
    for scenario in scenarios:
        scenario_dict = {
            "scenario_id": scenario.scenario_id,
            "profile": {
                "id": scenario.profile.profile_id,
                "name": scenario.profile.name,
                "preferences": scenario.profile.preferences,
                "budget_per_day": scenario.profile.budget_per_day,
                "transport": scenario.profile.preferred_transport,
                "daily_pois": scenario.profile.daily_pois_preference
            },
            "trip": {
                "duration": scenario.duration,
                "season": scenario.season,
                "start_date": scenario.start_date.isoformat(),
                "events": scenario.special_events,
                "weather": scenario.weather_condition
            },
            "location": scenario.start_location,
            "constraints": scenario.constraints
        }
        scenarios_dict.append(scenario_dict)
    
    with open(output_file, 'w') as f:
        json.dump(scenarios_dict, f, indent=2)
    
    print(f"Exported {len(scenarios)} scenarios to {output_file}")


def create_scenario_statistics(scenarios: List[TripScenario]) -> Dict:
    """Generate statistics about scenario distribution"""
    stats = {
        "total_scenarios": len(scenarios),
        "profiles": {},
        "durations": {},
        "seasons": {},
        "events": {},
        "weather": {},
        "locations": {}
    }
    
    for scenario in scenarios:
        # Count profiles
        profile_name = scenario.profile.name
        stats["profiles"][profile_name] = stats["profiles"].get(profile_name, 0) + 1
        
        # Count durations
        stats["durations"][scenario.duration] = stats["durations"].get(scenario.duration, 0) + 1
        
        # Count seasons
        stats["seasons"][scenario.season] = stats["seasons"].get(scenario.season, 0) + 1
        
        # Count events
        for event in scenario.special_events:
            stats["events"][event] = stats["events"].get(event, 0) + 1
        
        # Count weather
        stats["weather"][scenario.weather_condition] = stats["weather"].get(scenario.weather_condition, 0) + 1
        
        # Count locations
        loc_name = scenario.start_location["name"]
        stats["locations"][loc_name] = stats["locations"].get(loc_name, 0) + 1
    
    return stats


if __name__ == "__main__":
    # Generate benchmark scenarios
    scenarios = generate_benchmark_scenarios()
    
    # Export to JSON
    export_scenarios(scenarios, "benchmarks/scenarios/nyc_benchmark_scenarios.json")
    
    # Generate statistics
    stats = create_scenario_statistics(scenarios)
    
    print("\nScenario Statistics:")
    print(f"Total scenarios: {stats['total_scenarios']}")
    print("\nProfile distribution:")
    for profile, count in stats["profiles"].items():
        print(f"  {profile}: {count}")
    print("\nDuration distribution:")
    for duration, count in stats["durations"].items():
        print(f"  {duration}: {count}")
    print("\nSeason distribution:")
    for season, count in stats["seasons"].items():
        print(f"  {season}: {count}")