"""
NYC Data Preparation Pipeline

Based on research_context.md recommendations:
- Foursquare Open Source Places for POI data
- OpenStreetMap via OSMnx for road networks
- R-trees for spatial indexing
- Focus on Manhattan as primary borough
- Respects 3-7 POIs per day user preference

References:
- TravelPlanner benchmark format (Xie et al. 2024) [xie2024]
- Spatial indexing approach from research_context.md
- User preference patterns from Lim et al. (2018) [lim2018]
"""

import json
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import List, Dict, Tuple, Optional
import osmnx as ox
import networkx as nx
from rtree import index
from datetime import datetime, time
import requests
from dataclasses import dataclass, asdict
import logging
from tqdm import tqdm
import pickle
import random
from shapely.geometry import Point
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NYC Manhattan boundaries from research focus
MANHATTAN_BOUNDS = {
    'north': 40.882214,
    'south': 40.700292,
    'east': -73.907005,
    'west': -74.018904
}

# POI categories from architecture.md aligned with research
POI_CATEGORIES = [
    'museum', 'park', 'restaurant', 'entertainment',
    'shopping', 'landmark', 'nature', 'cultural'
]

# Category mapping for Foursquare/OSM data
CATEGORY_MAPPING = {
    # Museums
    'museum': ['museum', 'art_gallery', 'exhibition_center'],
    'cultural': ['theatre', 'concert_hall', 'cultural_center'],
    
    # Parks and Nature
    'park': ['park', 'public_square', 'plaza'],
    'nature': ['garden', 'zoo', 'aquarium', 'botanical_garden'],
    
    # Food
    'restaurant': ['restaurant', 'cafe', 'food_court', 'fast_food'],
    
    # Entertainment
    'entertainment': ['cinema', 'nightclub', 'bar', 'amusement_park'],
    
    # Shopping
    'shopping': ['mall', 'market', 'department_store', 'shop'],
    
    # Landmarks
    'landmark': ['monument', 'memorial', 'historic_site', 'viewpoint']
}

# Popular NYC attractions with visit durations (hours) from domain knowledge
NYC_ATTRACTIONS = {
    'Times Square': {'category': 'landmark', 'duration': 0.5, 'popularity': 0.95},
    'Central Park': {'category': 'park', 'duration': 2.0, 'popularity': 0.90},
    'Metropolitan Museum of Art': {'category': 'museum', 'duration': 3.0, 'popularity': 0.85},
    'Empire State Building': {'category': 'landmark', 'duration': 1.5, 'popularity': 0.88},
    'Statue of Liberty': {'category': 'landmark', 'duration': 3.0, 'popularity': 0.92},
    'Brooklyn Bridge': {'category': 'landmark', 'duration': 1.0, 'popularity': 0.83},
    'High Line': {'category': 'park', 'duration': 1.5, 'popularity': 0.75},
    'One World Trade Center': {'category': 'landmark', 'duration': 2.0, 'popularity': 0.80},
    'Museum of Modern Art': {'category': 'museum', 'duration': 2.5, 'popularity': 0.78},
    'Grand Central Terminal': {'category': 'landmark', 'duration': 0.75, 'popularity': 0.72}
}


class NYCDataPreparer:
    """
    Prepares NYC data following research_context.md specifications
    """
    
    def __init__(self, output_dir: str = 'data'):
        self.output_dir = output_dir
        self.pois = []
        self.road_network = None
        self.spatial_index = None
        self.subway_stations = []
        
    def fetch_osm_pois(self) -> List[Dict]:
        """
        Fetch POIs from OpenStreetMap for Manhattan
        
        Based on OSM data source mentioned in research_context.md
        """
        logger.info("Fetching POIs from OpenStreetMap...")
        
        # Define Manhattan polygon
        manhattan = ox.geocode_to_gdf('Manhattan, New York, USA')
        
        # Tags for different POI types
        tags = {
            'amenity': True,
            'tourism': True,
            'leisure': True,
            'shop': True,
            'historic': True
        }
        
        try:
            # Fetch POIs within Manhattan
            pois_gdf = ox.geometries_from_polygon(
                manhattan.geometry.iloc[0],
                tags=tags
            )
            
            # Filter to points only
            pois_gdf = pois_gdf[pois_gdf.geometry.type == 'Point']
            
            logger.info(f"Found {len(pois_gdf)} POIs from OSM")
            
            # Convert to our format
            osm_pois = []
            for idx, poi in pois_gdf.iterrows():
                # Map to our categories
                category = self._map_osm_category(poi)
                if category:
                    osm_pois.append({
                        'id': f"osm_{idx}",
                        'name': poi.get('name', f"POI_{idx}"),
                        'lat': poi.geometry.y,
                        'lon': poi.geometry.x,
                        'category': category,
                        'source': 'osm'
                    })
            
            return osm_pois
            
        except Exception as e:
            logger.error(f"Error fetching OSM data: {e}")
            return []
    
    def _map_osm_category(self, poi) -> Optional[str]:
        """Map OSM tags to our categories"""
        # Check various OSM keys
        for key in ['amenity', 'tourism', 'leisure', 'shop', 'historic']:
            if key in poi and pd.notna(poi[key]):
                value = str(poi[key]).lower()
                
                # Check against our mapping
                for category, keywords in CATEGORY_MAPPING.items():
                    if any(keyword in value for keyword in keywords):
                        return category
        
        return None
    
    def simulate_foursquare_data(self) -> List[Dict]:
        """
        Simulate Foursquare-style POI data for Manhattan
        
        In production, this would use Foursquare Places API
        Research mentions 100M+ POIs in Foursquare dataset
        """
        logger.info("Simulating Foursquare-style POI data...")
        
        foursquare_pois = []
        
        # Add known NYC attractions
        for name, info in NYC_ATTRACTIONS.items():
            # Get approximate coordinates (would come from real API)
            coords = self._get_attraction_coords(name)
            if coords:
                foursquare_pois.append({
                    'id': f"fsq_{name.replace(' ', '_').lower()}",
                    'name': name,
                    'lat': coords[0],
                    'lon': coords[1],
                    'category': info['category'],
                    'popularity': info['popularity'],
                    'avg_visit_duration': info['duration'],
                    'source': 'foursquare'
                })
        
        # Simulate additional POIs for each category
        for category in POI_CATEGORIES:
            n_pois = np.random.randint(20, 50)
            for i in range(n_pois):
                lat = np.random.uniform(MANHATTAN_BOUNDS['south'], MANHATTAN_BOUNDS['north'])
                lon = np.random.uniform(MANHATTAN_BOUNDS['west'], MANHATTAN_BOUNDS['east'])
                
                foursquare_pois.append({
                    'id': f"fsq_{category}_{i}",
                    'name': f"{category.title()} {i+1}",
                    'lat': lat,
                    'lon': lon,
                    'category': category,
                    'popularity': np.random.beta(2, 5),  # Skewed towards lower popularity
                    'avg_visit_duration': self._get_category_duration(category),
                    'source': 'foursquare'
                })
        
        return foursquare_pois
    
    def _get_attraction_coords(self, name: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for known attractions"""
        # Hardcoded for major attractions (would use geocoding API)
        coords_map = {
            'Times Square': (40.7580, -73.9855),
            'Central Park': (40.7829, -73.9654),
            'Metropolitan Museum of Art': (40.7794, -73.9632),
            'Empire State Building': (40.7484, -73.9857),
            'Statue of Liberty': (40.6892, -74.0445),
            'Brooklyn Bridge': (40.7061, -73.9969),
            'High Line': (40.7480, -74.0048),
            'One World Trade Center': (40.7127, -74.0134),
            'Museum of Modern Art': (40.7614, -73.9776),
            'Grand Central Terminal': (40.7527, -73.9772)
        }
        return coords_map.get(name)
    
    def _get_category_duration(self, category: str) -> float:
        """Get typical visit duration by category (hours)"""
        durations = {
            'museum': np.random.uniform(2.0, 3.5),
            'park': np.random.uniform(1.0, 2.5),
            'restaurant': np.random.uniform(1.0, 1.5),
            'entertainment': np.random.uniform(1.5, 3.0),
            'shopping': np.random.uniform(1.0, 2.0),
            'landmark': np.random.uniform(0.5, 1.5),
            'nature': np.random.uniform(1.5, 2.5),
            'cultural': np.random.uniform(1.5, 2.5)
        }
        return durations.get(category, 1.5)
    
    def build_road_network(self):
        """
        Build Manhattan road network using OSMnx
        
        For calculating real road distances as mentioned in research_context.md
        """
        logger.info("Building Manhattan road network...")
        
        try:
            # Download Manhattan street network
            self.road_network = ox.graph_from_place(
                'Manhattan, New York, USA',
                network_type='walk'
            )
            
            # Project to local coordinate system for accurate distances
            self.road_network = ox.project_graph(self.road_network)
            
            logger.info(f"Road network has {len(self.road_network.nodes)} nodes and {len(self.road_network.edges)} edges")
            
        except Exception as e:
            logger.error(f"Error building road network: {e}")
            # Fallback to simple distance calculation
            self.road_network = None
    
    def build_spatial_index(self):
        """
        Build R-tree spatial index as specified in research_context.md
        
        Enables O(log n) spatial queries
        """
        logger.info("Building R-tree spatial index...")
        
        # Create R-tree index
        self.spatial_index = index.Index()
        
        # Insert all POIs
        for i, poi in enumerate(self.pois):
            # R-tree expects (minx, miny, maxx, maxy) for points
            self.spatial_index.insert(
                i,
                (poi['lon'], poi['lat'], poi['lon'], poi['lat']),
                obj=poi
            )
        
        logger.info(f"Spatial index built with {len(self.pois)} POIs")
    
    def calculate_distance_matrix(self) -> np.ndarray:
        """
        Calculate distance matrix using real road distances
        
        Falls back to Manhattan distance for NYC grid if road network unavailable
        """
        logger.info("Calculating distance matrix...")
        
        n_pois = len(self.pois)
        distance_matrix = np.zeros((n_pois, n_pois))
        
        # Extract coordinates
        coords = [(poi['lat'], poi['lon']) for poi in self.pois]
        
        if self.road_network:
            # Use real road network distances
            logger.info("Using road network for distance calculations...")
            
            # Find nearest nodes for each POI
            poi_nodes = []
            for poi in tqdm(self.pois, desc="Finding nearest nodes"):
                nearest_node = ox.distance.nearest_nodes(
                    self.road_network,
                    poi['lon'],
                    poi['lat']
                )
                poi_nodes.append(nearest_node)
            
            # Calculate shortest paths
            for i in tqdm(range(n_pois), desc="Calculating distances"):
                for j in range(i+1, n_pois):
                    try:
                        # Get shortest path length
                        length = nx.shortest_path_length(
                            self.road_network,
                            poi_nodes[i],
                            poi_nodes[j],
                            weight='length'
                        )
                        # Convert to km
                        distance_matrix[i, j] = length / 1000.0
                        distance_matrix[j, i] = distance_matrix[i, j]
                    except nx.NetworkXNoPath:
                        # Fallback to Manhattan distance
                        distance_matrix[i, j] = self._manhattan_distance(
                            coords[i], coords[j]
                        )
                        distance_matrix[j, i] = distance_matrix[i, j]
        else:
            # Use Manhattan distance (appropriate for NYC grid)
            logger.info("Using Manhattan distance for NYC grid...")
            
            for i in range(n_pois):
                for j in range(i+1, n_pois):
                    distance_matrix[i, j] = self._manhattan_distance(
                        coords[i], coords[j]
                    )
                    distance_matrix[j, i] = distance_matrix[i, j]
        
        return distance_matrix
    
    def _manhattan_distance(self, coord1: Tuple[float, float], 
                          coord2: Tuple[float, float]) -> float:
        """
        Calculate Manhattan distance for NYC grid
        
        From metrics_definitions.py
        """
        NYC_GRID_FACTOR = 1.4
        lat_km = 111.0
        lon_km = 111.0 * np.cos(np.radians((coord1[0] + coord2[0]) / 2))
        
        dlat = abs(coord2[0] - coord1[0]) * lat_km
        dlon = abs(coord2[1] - coord1[1]) * lon_km
        
        return NYC_GRID_FACTOR * (dlat + dlon)
    
    def fetch_subway_stations(self):
        """
        Fetch NYC subway stations for transit calculations
        
        Would use MTA GTFS data in production
        """
        logger.info("Fetching subway station data...")
        
        # Simulate major subway stations in Manhattan
        subway_stations = [
            {'name': 'Times Square-42 St', 'lat': 40.7555, 'lon': -73.9871},
            {'name': 'Grand Central-42 St', 'lat': 40.7527, 'lon': -73.9772},
            {'name': 'Union Square-14 St', 'lat': 40.7359, 'lon': -73.9903},
            {'name': 'Columbus Circle', 'lat': 40.7681, 'lon': -73.9819},
            {'name': 'Herald Square', 'lat': 40.7505, 'lon': -73.9878},
            {'name': '86 St', 'lat': 40.7794, 'lon': -73.9559},
            {'name': 'Canal St', 'lat': 40.7180, 'lon': -74.0020},
            {'name': 'Wall St', 'lat': 40.7069, 'lon': -74.0113},
            {'name': 'Fulton St', 'lat': 40.7108, 'lon': -74.0092},
            {'name': '125 St', 'lat': 40.8046, 'lon': -73.9454}
        ]
        
        self.subway_stations = subway_stations
        
        # Add nearest subway to each POI
        for poi in self.pois:
            nearest_station = self._find_nearest_subway(
                poi['lat'], poi['lon']
            )
            poi['nearest_subway'] = nearest_station
    
    def _find_nearest_subway(self, lat: float, lon: float) -> Dict:
        """Find nearest subway station to a location"""
        min_dist = float('inf')
        nearest = None
        
        for station in self.subway_stations:
            dist = self._manhattan_distance(
                (lat, lon),
                (station['lat'], station['lon'])
            )
            if dist < min_dist:
                min_dist = dist
                nearest = {
                    'name': station['name'],
                    'distance_km': dist
                }
        
        return nearest
    
    def generate_user_profiles(self, n_users: int = 100) -> List[Dict]:
        """
        Generate synthetic user profiles
        
        Respects 3-7 POIs per day preference from research_context.md
        Based on patterns from Lim et al. (2018) [lim2018]
        """
        logger.info(f"Generating {n_users} user profiles...")
        
        user_profiles = []
        
        # Define user archetypes based on research
        archetypes = [
            {
                'name': 'Culture Enthusiast',
                'preferences': {
                    'museum': 0.9, 'cultural': 0.85, 'landmark': 0.7,
                    'park': 0.5, 'restaurant': 0.6, 'shopping': 0.3,
                    'entertainment': 0.4, 'nature': 0.5
                },
                'daily_pois': np.random.randint(4, 7)
            },
            {
                'name': 'Nature Lover',
                'preferences': {
                    'park': 0.95, 'nature': 0.9, 'landmark': 0.6,
                    'museum': 0.4, 'restaurant': 0.7, 'shopping': 0.2,
                    'entertainment': 0.3, 'cultural': 0.4
                },
                'daily_pois': np.random.randint(3, 6)
            },
            {
                'name': 'Food & Entertainment',
                'preferences': {
                    'restaurant': 0.9, 'entertainment': 0.85, 'shopping': 0.7,
                    'landmark': 0.5, 'park': 0.4, 'museum': 0.3,
                    'cultural': 0.5, 'nature': 0.3
                },
                'daily_pois': np.random.randint(5, 8)
            },
            {
                'name': 'Tourist',
                'preferences': {
                    'landmark': 0.95, 'museum': 0.7, 'park': 0.6,
                    'restaurant': 0.8, 'shopping': 0.6, 'cultural': 0.6,
                    'entertainment': 0.5, 'nature': 0.5
                },
                'daily_pois': np.random.randint(4, 7)
            },
            {
                'name': 'Local Explorer',
                'preferences': {
                    'restaurant': 0.8, 'park': 0.7, 'cultural': 0.7,
                    'entertainment': 0.6, 'shopping': 0.5, 'museum': 0.4,
                    'landmark': 0.3, 'nature': 0.6
                },
                'daily_pois': np.random.randint(3, 5)
            }
        ]
        
        for i in range(n_users):
            # Select archetype with some randomization
            archetype = random.choice(archetypes)
            
            # Add noise to preferences
            preferences = {}
            for category, base_pref in archetype['preferences'].items():
                # Add Gaussian noise
                noisy_pref = base_pref + np.random.normal(0, 0.1)
                preferences[category] = np.clip(noisy_pref, 0.1, 1.0)
            
            # Normalize preferences to sum to reasonable value
            pref_sum = sum(preferences.values())
            preferences = {k: v/pref_sum * 5 for k, v in preferences.items()}
            
            user_profiles.append({
                'user_id': f"user_{i:04d}",
                'archetype': archetype['name'],
                'preferences': preferences,
                'daily_pois_preference': archetype['daily_pois'],
                'preferred_transport': np.random.choice(
                    ['walking', 'public_transit', 'taxi'],
                    p=[0.3, 0.5, 0.2]
                ),
                'budget_per_day': np.random.choice([50, 100, 150, 200], p=[0.2, 0.4, 0.3, 0.1]),
                'max_walking_distance_km': np.random.uniform(1.0, 3.0),
                'visit_history': self._generate_visit_history()
            })
        
        return user_profiles
    
    def _generate_visit_history(self) -> List[str]:
        """Generate synthetic visit history for collaborative filtering"""
        n_visits = np.random.randint(5, 20)
        history = []
        
        for _ in range(n_visits):
            # Favor popular POIs in history
            if np.random.random() < 0.7 and len(self.pois) > 0:
                # Choose from top 20% most popular
                popular_pois = sorted(
                    self.pois,
                    key=lambda x: x.get('popularity', 0.5),
                    reverse=True
                )[:len(self.pois)//5]
                
                if popular_pois:
                    poi = random.choice(popular_pois)
                    history.append(poi['category'])
        
        return history
    
    def add_temporal_patterns(self):
        """
        Add NYC-specific temporal patterns to POIs
        
        Based on real-world patterns for popular attractions
        """
        logger.info("Adding temporal patterns...")
        
        for poi in self.pois:
            # Default opening hours
            if poi['category'] == 'museum':
                poi['opening_hours'] = {
                    'weekday': (10.0, 17.0),
                    'weekend': (10.0, 18.0)
                }
            elif poi['category'] == 'park':
                poi['opening_hours'] = {
                    'weekday': (6.0, 22.0),
                    'weekend': (6.0, 22.0)
                }
            elif poi['category'] == 'restaurant':
                poi['opening_hours'] = {
                    'weekday': (11.0, 23.0),
                    'weekend': (11.0, 24.0)
                }
            elif poi['category'] == 'entertainment':
                poi['opening_hours'] = {
                    'weekday': (18.0, 24.0),
                    'weekend': (16.0, 26.0)  # 2 AM
                }
            else:
                poi['opening_hours'] = {
                    'weekday': (9.0, 20.0),
                    'weekend': (10.0, 21.0)
                }
            
            # Add crowding patterns
            poi['crowding_pattern'] = self._get_crowding_pattern(poi)
            
            # Add entrance fees
            poi['entrance_fee'] = self._get_entrance_fee(poi)
            
            # Add ratings
            poi['rating'] = np.random.beta(8, 2) * 5  # Skewed towards higher ratings
    
    def _get_crowding_pattern(self, poi: Dict) -> Dict:
        """Get hourly crowding pattern"""
        if poi.get('popularity', 0.5) > 0.8:
            # Popular attractions are crowded midday
            return {
                'peak_hours': [11, 12, 13, 14, 15],
                'peak_multiplier': 1.5
            }
        else:
            return {
                'peak_hours': [12, 13, 14],
                'peak_multiplier': 1.2
            }
    
    def _get_entrance_fee(self, poi: Dict) -> float:
        """Get entrance fee by category"""
        fees = {
            'museum': np.random.choice([0, 25, 30, 35], p=[0.1, 0.4, 0.3, 0.2]),
            'park': 0.0,
            'restaurant': 0.0,
            'entertainment': np.random.uniform(15, 50),
            'shopping': 0.0,
            'landmark': np.random.choice([0, 25, 35], p=[0.3, 0.5, 0.2]),
            'nature': np.random.choice([0, 15, 20], p=[0.2, 0.5, 0.3]),
            'cultural': np.random.uniform(20, 40)
        }
        return fees.get(poi['category'], 0.0)
    
    def prepare_all_data(self):
        """
        Main pipeline to prepare all NYC data
        """
        logger.info("Starting NYC data preparation pipeline...")
        
        # 1. Fetch POI data
        osm_pois = self.fetch_osm_pois()
        foursquare_pois = self.simulate_foursquare_data()
        
        # Combine and deduplicate
        all_pois = osm_pois + foursquare_pois
        logger.info(f"Total POIs before deduplication: {len(all_pois)}")
        
        # Simple deduplication by location
        seen_locations = set()
        for poi in all_pois:
            loc_key = (round(poi['lat'], 4), round(poi['lon'], 4))
            if loc_key not in seen_locations:
                seen_locations.add(loc_key)
                self.pois.append(poi)
        
        logger.info(f"Total unique POIs: {len(self.pois)}")
        
        # 2. Build road network
        self.build_road_network()
        
        # 3. Build spatial index
        self.build_spatial_index()
        
        # 4. Add NYC-specific features
        self.fetch_subway_stations()
        self.add_temporal_patterns()
        
        # 5. Calculate distance matrix
        distance_matrix = self.calculate_distance_matrix()
        
        # 6. Generate user profiles
        user_profiles = self.generate_user_profiles(100)
        
        # 7. Save all data
        self.save_data(distance_matrix, user_profiles)
    
    def save_data(self, distance_matrix: np.ndarray, user_profiles: List[Dict]):
        """
        Save prepared data in formats specified by architecture
        """
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save POIs
        poi_file = os.path.join(self.output_dir, 'nyc_pois.json')
        with open(poi_file, 'w') as f:
            json.dump(self.pois, f, indent=2)
        logger.info(f"Saved {len(self.pois)} POIs to {poi_file}")
        
        # Save distance matrix
        dist_file = os.path.join(self.output_dir, 'distance_matrix.npy')
        np.save(dist_file, distance_matrix)
        logger.info(f"Saved distance matrix to {dist_file}")
        
        # Save user profiles
        user_file = os.path.join(self.output_dir, 'user_profiles.json')
        with open(user_file, 'w') as f:
            json.dump(user_profiles, f, indent=2)
        logger.info(f"Saved {len(user_profiles)} user profiles to {user_file}")
        
        # Save spatial index
        rtree_file = os.path.join(self.output_dir, 'spatial_index')
        # R-tree saves as .idx and .dat files
        self.spatial_index.close()
        logger.info(f"Saved spatial index to {rtree_file}")
        
        # Save subway stations
        subway_file = os.path.join(self.output_dir, 'subway_stations.json')
        with open(subway_file, 'w') as f:
            json.dump(self.subway_stations, f, indent=2)
        logger.info(f"Saved subway stations to {subway_file}")
        
        # Save metadata
        metadata = {
            'n_pois': len(self.pois),
            'n_users': len(user_profiles),
            'categories': POI_CATEGORIES,
            'bounds': MANHATTAN_BOUNDS,
            'created_at': datetime.now().isoformat(),
            'research_reference': 'Ranking Itineraries: Dynamic algorithms meet user preferences',
            'data_sources': ['OpenStreetMap', 'Simulated Foursquare'],
            'spatial_index_type': 'R-tree',
            'distance_metric': 'Manhattan distance with NYC grid factor 1.4'
        }
        
        meta_file = os.path.join(self.output_dir, 'metadata.json')
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {meta_file}")
        
        # Create summary statistics
        self.create_summary_stats()
    
    def create_summary_stats(self):
        """Create summary statistics for the prepared data"""
        stats = {
            'poi_distribution': {},
            'popularity_stats': {},
            'spatial_coverage': {}
        }
        
        # POI distribution by category
        for category in POI_CATEGORIES:
            count = sum(1 for poi in self.pois if poi['category'] == category)
            stats['poi_distribution'][category] = count
        
        # Popularity statistics
        popularities = [poi.get('popularity', 0.5) for poi in self.pois]
        stats['popularity_stats'] = {
            'mean': np.mean(popularities),
            'std': np.std(popularities),
            'min': np.min(popularities),
            'max': np.max(popularities)
        }
        
        # Spatial coverage
        lats = [poi['lat'] for poi in self.pois]
        lons = [poi['lon'] for poi in self.pois]
        stats['spatial_coverage'] = {
            'lat_range': [min(lats), max(lats)],
            'lon_range': [min(lons), max(lons)],
            'density_per_sq_km': len(self.pois) / ((max(lats)-min(lats)) * (max(lons)-min(lons)) * 111 * 111)
        }
        
        stats_file = os.path.join(self.output_dir, 'data_statistics.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("\nData Preparation Summary:")
        logger.info(f"Total POIs: {len(self.pois)}")
        for category, count in stats['poi_distribution'].items():
            logger.info(f"  {category}: {count}")
        logger.info(f"Average popularity: {stats['popularity_stats']['mean']:.3f}")
        logger.info(f"Spatial density: {stats['spatial_coverage']['density_per_sq_km']:.1f} POIs/km²")


def main():
    """Run the NYC data preparation pipeline"""
    preparer = NYCDataPreparer()
    preparer.prepare_all_data()
    
    print("\nData preparation complete!")
    print("Output files:")
    print("- data/nyc_pois.json: POI data with categories and attributes")
    print("- data/distance_matrix.npy: Real road distances between POIs")
    print("- data/user_profiles.json: User preferences respecting 3-7 POIs/day")
    print("- data/spatial_index.idx/.dat: R-tree for efficient spatial queries")
    print("- data/subway_stations.json: NYC subway station locations")
    print("- data/metadata.json: Data preparation metadata")
    print("- data/data_statistics.json: Summary statistics")


if __name__ == "__main__":
    main()