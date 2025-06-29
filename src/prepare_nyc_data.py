"""
NYC Data Preparation Pipeline

Generates 10,847 NYC POIs as specified in research_context.md
Uses simulated data to ensure reproducibility for thesis evaluation
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from rtree import index
from datetime import datetime
import logging
from tqdm import tqdm
import pickle
import random
import os
import sys

# No sys.path manipulation needed - file has no local imports

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# NYC Manhattan boundaries from research focus
MANHATTAN_BOUNDS = {
    'north': 40.882214,
    'south': 40.700292,
    'east': -73.907005,
    'west': -74.018904
}

# POI categories from research
POI_CATEGORIES = {
    'museum': 1200,      # ~11% of POIs
    'park': 800,         # ~7% of POIs  
    'restaurant': 3500,  # ~32% of POIs
    'landmark': 1500,    # ~14% of POIs
    'shopping': 1800,    # ~17% of POIs
    'entertainment': 900, # ~8% of POIs
    'cultural': 700,     # ~6% of POIs
    'market': 447        # ~4% of POIs (to reach exactly 10,847)
}

# Major NYC attractions with real coordinates and data
NYC_ATTRACTIONS = [
    {'name': 'Central Park', 'lat': 40.7829, 'lon': -73.9654, 'category': 'park', 'rating': 4.7, 'popularity': 0.95, 'fee': 0, 'duration': 2.0},
    {'name': 'Times Square', 'lat': 40.7580, 'lon': -73.9855, 'category': 'landmark', 'rating': 4.3, 'popularity': 0.98, 'fee': 0, 'duration': 0.5},
    {'name': 'Metropolitan Museum of Art', 'lat': 40.7794, 'lon': -73.9632, 'category': 'museum', 'rating': 4.7, 'popularity': 0.92, 'fee': 25, 'duration': 3.0},
    {'name': 'Museum of Modern Art', 'lat': 40.7614, 'lon': -73.9776, 'category': 'museum', 'rating': 4.5, 'popularity': 0.88, 'fee': 25, 'duration': 2.5},
    {'name': 'Empire State Building', 'lat': 40.7484, 'lon': -73.9857, 'category': 'landmark', 'rating': 4.7, 'popularity': 0.94, 'fee': 42, 'duration': 1.5},
    {'name': 'Statue of Liberty', 'lat': 40.6892, 'lon': -74.0445, 'category': 'landmark', 'rating': 4.6, 'popularity': 0.96, 'fee': 24, 'duration': 3.0},
    {'name': 'Brooklyn Bridge', 'lat': 40.7061, 'lon': -73.9969, 'category': 'landmark', 'rating': 4.6, 'popularity': 0.89, 'fee': 0, 'duration': 1.0},
    {'name': 'High Line', 'lat': 40.7480, 'lon': -74.0048, 'category': 'park', 'rating': 4.6, 'popularity': 0.85, 'fee': 0, 'duration': 1.5},
    {'name': 'Grand Central Terminal', 'lat': 40.7527, 'lon': -73.9772, 'category': 'landmark', 'rating': 4.6, 'popularity': 0.82, 'fee': 0, 'duration': 0.75},
    {'name': 'One World Trade Center', 'lat': 40.7127, 'lon': -74.0134, 'category': 'landmark', 'rating': 4.7, 'popularity': 0.87, 'fee': 35, 'duration': 2.0},
    {'name': 'American Museum of Natural History', 'lat': 40.7813, 'lon': -73.9740, 'category': 'museum', 'rating': 4.6, 'popularity': 0.90, 'fee': 23, 'duration': 3.0},
    {'name': 'Rockefeller Center', 'lat': 40.7587, 'lon': -73.9787, 'category': 'landmark', 'rating': 4.6, 'popularity': 0.86, 'fee': 0, 'duration': 1.0},
    {'name': 'Bryant Park', 'lat': 40.7536, 'lon': -73.9832, 'category': 'park', 'rating': 4.5, 'popularity': 0.78, 'fee': 0, 'duration': 1.0},
    {'name': 'Madison Square Garden', 'lat': 40.7505, 'lon': -73.9934, 'category': 'entertainment', 'rating': 4.5, 'popularity': 0.83, 'fee': 0, 'duration': 2.5},
    {'name': 'Lincoln Center', 'lat': 40.7725, 'lon': -73.9835, 'category': 'cultural', 'rating': 4.6, 'popularity': 0.81, 'fee': 0, 'duration': 2.0}
]

# NYC Subway stations (major hubs)
SUBWAY_STATIONS = [
    {'name': 'Times Square-42 St', 'lat': 40.7555, 'lon': -73.9871, 'lines': ['N', 'Q', 'R', 'W', '1', '2', '3', '7', 'S']},
    {'name': 'Grand Central-42 St', 'lat': 40.7527, 'lon': -73.9772, 'lines': ['4', '5', '6', '7', 'S']},
    {'name': 'Union Square-14 St', 'lat': 40.7359, 'lon': -73.9903, 'lines': ['4', '5', '6', 'L', 'N', 'Q', 'R', 'W']},
    {'name': 'Columbus Circle', 'lat': 40.7681, 'lon': -73.9819, 'lines': ['A', 'B', 'C', 'D', '1']},
    {'name': 'Herald Square', 'lat': 40.7505, 'lon': -73.9878, 'lines': ['B', 'D', 'F', 'M', 'N', 'Q', 'R', 'W']},
    {'name': '86 St', 'lat': 40.7794, 'lon': -73.9559, 'lines': ['4', '5', '6']},
    {'name': 'Canal St', 'lat': 40.7180, 'lon': -74.0020, 'lines': ['6', 'J', 'Z', 'N', 'Q', 'R', 'W']},
    {'name': 'Wall St', 'lat': 40.7069, 'lon': -74.0113, 'lines': ['4', '5']},
    {'name': 'Fulton St', 'lat': 40.7108, 'lon': -74.0092, 'lines': ['2', '3', '4', '5', 'A', 'C', 'J', 'Z']},
    {'name': '125 St', 'lat': 40.8046, 'lon': -73.9454, 'lines': ['4', '5', '6']},
    {'name': 'Penn Station', 'lat': 40.7506, 'lon': -73.9935, 'lines': ['1', '2', '3', 'A', 'C', 'E']},
    {'name': '59 St-Columbus Circle', 'lat': 40.7681, 'lon': -73.9819, 'lines': ['A', 'B', 'C', 'D', '1']},
    {'name': 'Lexington Ave/53 St', 'lat': 40.7575, 'lon': -73.9709, 'lines': ['E', 'M', '6']},
    {'name': 'Astor Pl', 'lat': 40.7301, 'lon': -73.9914, 'lines': ['6']},
    {'name': 'West 4 St', 'lat': 40.7322, 'lon': -74.0001, 'lines': ['A', 'B', 'C', 'D', 'E', 'F', 'M']}
]


class NYCDataGenerator:
    """Generates realistic NYC POI data for thesis evaluation"""
    
    def __init__(self, output_dir: str = 'data/nyc_data'):
        self.output_dir = output_dir
        self.pois = []
        self.distance_matrix = None
        self.spatial_index = None
        
    def generate_pois(self) -> List[Dict]:
        """Generate exactly 10,847 POIs as specified in research"""
        logger.info("Generating 10,847 NYC POIs...")
        
        poi_id = 0
        
        # First add real attractions
        for attraction in NYC_ATTRACTIONS:
            self.pois.append({
                'id': f'poi_{poi_id:05d}',
                'name': attraction['name'],
                'lat': attraction['lat'],
                'lon': attraction['lon'],
                'category': attraction['category'],
                'rating': attraction['rating'],
                'popularity_score': attraction['popularity'],
                'entrance_fee': attraction['fee'],
                'avg_visit_duration': attraction['duration'],
                'opening_hours': self._get_opening_hours(attraction['category']),
                'accessibility_score': np.random.uniform(0.7, 0.95),
                'weather_dependency': self._get_weather_dependency(attraction['category'])
            })
            poi_id += 1
        
        # Generate remaining POIs for each category
        for category, total_count in POI_CATEGORIES.items():
            # Count already added POIs of this category
            existing_count = sum(1 for poi in self.pois if poi['category'] == category)
            remaining_count = total_count - existing_count
            
            for i in range(remaining_count):
                # Generate location within Manhattan bounds
                lat = np.random.uniform(MANHATTAN_BOUNDS['south'], MANHATTAN_BOUNDS['north'])
                lon = np.random.uniform(MANHATTAN_BOUNDS['west'], MANHATTAN_BOUNDS['east'])
                
                # Generate realistic attributes
                self.pois.append({
                    'id': f'poi_{poi_id:05d}',
                    'name': f'{category.title()} {i+1}',
                    'lat': lat,
                    'lon': lon,
                    'category': category,
                    'rating': self._generate_rating(),
                    'popularity_score': self._generate_popularity(),
                    'entrance_fee': self._get_entrance_fee(category),
                    'avg_visit_duration': self._get_visit_duration(category),
                    'opening_hours': self._get_opening_hours(category),
                    'accessibility_score': np.random.uniform(0.6, 0.95),
                    'weather_dependency': self._get_weather_dependency(category)
                })
                poi_id += 1
        
        logger.info(f"Generated {len(self.pois)} POIs")
        return self.pois
    
    def _generate_rating(self) -> float:
        """Generate realistic rating (skewed towards higher ratings)"""
        return round(np.random.beta(8, 2) * 5, 1)
    
    def _generate_popularity(self) -> float:
        """Generate popularity score (most POIs have lower popularity)"""
        return round(np.random.beta(2, 5), 3)
    
    def _get_entrance_fee(self, category: str) -> float:
        """Get entrance fee by category"""
        fees = {
            'museum': lambda: np.random.choice([0, 20, 25, 30], p=[0.1, 0.3, 0.4, 0.2]),
            'park': lambda: 0.0,
            'restaurant': lambda: 0.0,
            'landmark': lambda: np.random.choice([0, 25, 35, 42], p=[0.4, 0.3, 0.2, 0.1]),
            'shopping': lambda: 0.0,
            'entertainment': lambda: np.random.uniform(15, 50),
            'cultural': lambda: np.random.choice([0, 20, 30], p=[0.3, 0.4, 0.3]),
            'market': lambda: 0.0
        }
        return fees.get(category, lambda: 0.0)()
    
    def _get_visit_duration(self, category: str) -> float:
        """Get typical visit duration by category (hours)"""
        durations = {
            'museum': lambda: round(np.random.uniform(2.0, 3.5), 1),
            'park': lambda: round(np.random.uniform(1.0, 2.5), 1),
            'restaurant': lambda: round(np.random.uniform(1.0, 1.5), 1),
            'landmark': lambda: round(np.random.uniform(0.5, 1.5), 1),
            'shopping': lambda: round(np.random.uniform(1.0, 2.0), 1),
            'entertainment': lambda: round(np.random.uniform(1.5, 3.0), 1),
            'cultural': lambda: round(np.random.uniform(1.5, 2.5), 1),
            'market': lambda: round(np.random.uniform(0.5, 1.5), 1)
        }
        return durations.get(category, lambda: 1.5)()
    
    def _get_opening_hours(self, category: str) -> Tuple[float, float]:
        """Get opening hours by category"""
        hours = {
            'museum': (10.0, 17.0),
            'park': (6.0, 22.0),
            'restaurant': (11.0, 23.0),
            'landmark': (9.0, 18.0),
            'shopping': (10.0, 21.0),
            'entertainment': (18.0, 24.0),
            'cultural': (19.0, 23.0),
            'market': (8.0, 18.0)
        }
        return hours.get(category, (9.0, 21.0))
    
    def _get_weather_dependency(self, category: str) -> float:
        """Get weather dependency score (0=indoor, 1=outdoor)"""
        dependencies = {
            'museum': 0.1,
            'park': 0.9,
            'restaurant': 0.2,
            'landmark': 0.6,
            'shopping': 0.1,
            'entertainment': 0.3,
            'cultural': 0.2,
            'market': 0.7
        }
        return dependencies.get(category, 0.5)
    
    def build_spatial_index(self):
        """Build R-tree spatial index for efficient queries"""
        logger.info("Building R-tree spatial index...")
        
        self.spatial_index = index.Index()
        
        for i, poi in enumerate(self.pois):
            # R-tree expects (minx, miny, maxx, maxy)
            self.spatial_index.insert(
                i,
                (poi['lon'], poi['lat'], poi['lon'], poi['lat']),
                obj={'poi': poi, 'idx': i}
            )
        
        logger.info(f"Spatial index built with {len(self.pois)} POIs")
    
    def calculate_distance_matrix(self) -> np.ndarray:
        """Calculate Manhattan distance matrix for all POIs"""
        logger.info("Calculating distance matrix...")
        
        n_pois = len(self.pois)
        self.distance_matrix = np.zeros((n_pois, n_pois))
        
        # NYC grid factor from metrics_definitions.py
        NYC_GRID_FACTOR = 1.4
        
        for i in tqdm(range(n_pois), desc="Computing distances"):
            for j in range(i+1, n_pois):
                # Manhattan distance calculation
                lat1, lon1 = self.pois[i]['lat'], self.pois[i]['lon']
                lat2, lon2 = self.pois[j]['lat'], self.pois[j]['lon']
                
                # Convert to km
                lat_km = 111.0
                lon_km = 111.0 * np.cos(np.radians((lat1 + lat2) / 2))
                
                dlat = abs(lat2 - lat1) * lat_km
                dlon = abs(lon2 - lon1) * lon_km
                
                # Apply NYC grid factor
                distance = NYC_GRID_FACTOR * (dlat + dlon)
                
                self.distance_matrix[i, j] = distance
                self.distance_matrix[j, i] = distance
        
        return self.distance_matrix
    
    def add_subway_proximity(self):
        """Add nearest subway station info to each POI"""
        logger.info("Adding subway proximity data...")
        
        for poi in self.pois:
            min_dist = float('inf')
            nearest_station = None
            
            for station in SUBWAY_STATIONS:
                # Calculate Manhattan distance
                dlat = abs(poi['lat'] - station['lat']) * 111.0
                dlon = abs(poi['lon'] - station['lon']) * 111.0 * np.cos(np.radians(poi['lat']))
                dist = 1.4 * (dlat + dlon)  # NYC grid factor
                
                if dist < min_dist:
                    min_dist = dist
                    nearest_station = station
            
            poi['nearest_subway'] = {
                'name': nearest_station['name'],
                'distance_km': round(min_dist, 3),
                'lines': nearest_station['lines']
            }
    
    def save_all_data(self):
        """Save all generated data"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save POIs
        poi_file = os.path.join(self.output_dir, 'nyc_pois.json')
        with open(poi_file, 'w') as f:
            json.dump(self.pois, f, indent=2)
        logger.info(f"Saved {len(self.pois)} POIs to {poi_file}")
        
        # Save distance matrix
        dist_file = os.path.join(self.output_dir, 'distance_matrix.npy')
        np.save(dist_file, self.distance_matrix)
        logger.info(f"Saved distance matrix to {dist_file}")
        
        # Save subway stations
        subway_file = os.path.join(self.output_dir, 'subway_stations.json')
        with open(subway_file, 'w') as f:
            json.dump(SUBWAY_STATIONS, f, indent=2)
        logger.info(f"Saved subway stations to {subway_file}")
        
        # Save metadata
        metadata = {
            'total_pois': len(self.pois),
            'categories': POI_CATEGORIES,
            'bounds': MANHATTAN_BOUNDS,
            'distance_metric': 'Manhattan distance with NYC grid factor 1.4',
            'created_at': datetime.now().isoformat(),
            'research_alignment': {
                'target_pois': 10847,
                'actual_pois': len(self.pois),
                'user_preference': '3-7 POIs per day',
                'performance_target': 'sub-second response time',
                'success_rate_target': '87.5% (vs 0.6% baseline)'
            }
        }
        
        meta_file = os.path.join(self.output_dir, 'metadata.json')
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create category statistics
        self.create_statistics()
    
    def create_statistics(self):
        """Create and save data statistics"""
        stats = {
            'category_distribution': {},
            'rating_stats': {},
            'popularity_stats': {},
            'fee_stats': {},
            'spatial_density': {}
        }
        
        # Category distribution
        for category in POI_CATEGORIES:
            count = sum(1 for poi in self.pois if poi['category'] == category)
            stats['category_distribution'][category] = {
                'count': count,
                'percentage': round(count / len(self.pois) * 100, 1)
            }
        
        # Rating statistics
        ratings = [poi['rating'] for poi in self.pois]
        stats['rating_stats'] = {
            'mean': round(np.mean(ratings), 2),
            'std': round(np.std(ratings), 2),
            'min': min(ratings),
            'max': max(ratings)
        }
        
        # Popularity statistics
        popularities = [poi['popularity_score'] for poi in self.pois]
        stats['popularity_stats'] = {
            'mean': round(np.mean(popularities), 3),
            'std': round(np.std(popularities), 3),
            'top_10_percent': round(np.percentile(popularities, 90), 3)
        }
        
        # Fee statistics
        fees = [poi['entrance_fee'] for poi in self.pois]
        stats['fee_stats'] = {
            'free_pois': sum(1 for fee in fees if fee == 0),
            'paid_pois': sum(1 for fee in fees if fee > 0),
            'avg_fee': round(np.mean([fee for fee in fees if fee > 0]), 2) if any(fees) else 0
        }
        
        stats_file = os.path.join(self.output_dir, 'data_statistics.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Print summary
        logger.info("\nData Generation Summary:")
        logger.info(f"Total POIs: {len(self.pois)}")
        for category, data in stats['category_distribution'].items():
            logger.info(f"  {category}: {data['count']} ({data['percentage']}%)")
        logger.info(f"Average rating: {stats['rating_stats']['mean']}")
        logger.info(f"Free POIs: {stats['fee_stats']['free_pois']}")
        logger.info(f"Paid POIs: {stats['fee_stats']['paid_pois']} (avg ${stats['fee_stats']['avg_fee']})")
    
    def generate_all(self):
        """Run complete data generation pipeline"""
        logger.info("Starting NYC data generation pipeline...")
        
        # Generate POIs
        self.generate_pois()
        
        # Build spatial index
        self.build_spatial_index()
        
        # Calculate distances
        self.calculate_distance_matrix()
        
        # Add subway data
        self.add_subway_proximity()
        
        # Save everything
        self.save_all_data()
        
        logger.info("\nData generation complete!")
        logger.info(f"All files saved to: {self.output_dir}")


def main():
    """Run the NYC data generation pipeline"""
    generator = NYCDataGenerator()
    generator.generate_all()
    
    print("\n✓ Successfully generated 10,847 NYC POIs")
    print("✓ Created distance matrix with Manhattan distances")
    print("✓ Built R-tree spatial index")
    print("✓ Added subway station proximity data")
    print("\nOutput files in data/nyc_data/:")
    print("  - nyc_pois.json")
    print("  - distance_matrix.npy")
    print("  - subway_stations.json")
    print("  - metadata.json")
    print("  - data_statistics.json")


if __name__ == "__main__":
    main()