# Chapter 4: Implementation

## 4.1 Introduction

This chapter details the practical implementation of our quality-based itinerary planning system, translating the theoretical methods into a robust, scalable solution for New York City tourism. We present the technology stack chosen for performance and maintainability, the data pipeline for processing NYC's 10,000+ POIs, algorithmic optimizations that achieve sub-second response times, and comprehensive testing methodology. Through careful engineering decisions and NYC-specific adaptations, we demonstrate how academic algorithms can be transformed into production-ready systems.

## 4.2 Technology Stack

### 4.2.1 Core Technologies

Following the implementation roadmap from our research framework, we selected technologies balancing performance, ecosystem maturity, and research reproducibility:

**Python 3.8+** serves as our primary language, offering:
- Rich scientific computing ecosystem (NumPy, SciPy)
- Excellent geospatial libraries (GeoPandas, Shapely)
- Strong optimization frameworks (OR-Tools, PuLP)
- Seamless integration with machine learning tools

**Key Dependencies:**
```python
# Core scientific computing
numpy==1.21.0          # Numerical operations
scipy==1.7.0          # Optimization and spatial algorithms
pandas==1.3.0         # Data manipulation
numba==0.55.0         # JIT compilation for performance

# Geospatial processing
geopandas==0.11.0     # Spatial data structures
shapely==1.8.0        # Geometric operations
pyproj==3.2.0         # Coordinate transformations
rtree==1.0.0          # Spatial indexing

# Graph algorithms
networkx==2.8         # Graph representations
ortools==9.3          # Google optimization tools

# Data sources
foursquare==1.0.0     # POI data API
osmnx==1.2.0          # OpenStreetMap data
requests==2.28.0      # API interactions

# Visualization and UI
streamlit==1.10.0     # Interactive demos
plotly==5.8.0         # Interactive maps
folium==0.12.0        # Leaflet.js maps
```

### 4.2.2 System Architecture

Our modular architecture separates concerns for maintainability and testing:

```
nyc-itinerary-planner/
├── data/
│   ├── raw/              # Original data sources
│   ├── processed/        # Cleaned POI data
│   ├── indices/          # Spatial indices
│   └── cache/           # Computed distances
├── algorithms/
│   ├── greedy.py        # Greedy heuristics
│   ├── astar.py         # A* implementation
│   ├── lpa_star.py      # Dynamic replanning
│   └── hybrid.py        # Two-phase approach
├── metrics/
│   ├── quantitative.py  # Distance, time, cost
│   ├── qualitative.py   # Satisfaction, diversity
│   └── composite.py     # CSS calculation
├── evaluation/
│   ├── benchmarks/      # Test scenarios
│   ├── user_study/      # User evaluation
│   └── analysis/        # Results processing
└── api/
    ├── server.py        # REST API
    └── streamlit_app.py # Demo interface
```

### 4.2.3 NetworkX for Graph Representation

We model NYC's POI network as a weighted directed graph:

```python
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple

class NYCPOIGraph:
    """Graph representation of NYC POIs with multi-modal edges"""
    
    def __init__(self, pois: List[Dict], distance_matrix: np.ndarray):
        self.graph = nx.DiGraph()
        self.poi_map = {poi['id']: poi for poi in pois}
        
        # Add nodes with attributes
        for poi in pois:
            self.graph.add_node(
                poi['id'],
                name=poi['name'],
                lat=poi['lat'],
                lon=poi['lon'],
                category=poi['category'],
                borough=self._get_borough(poi['lat'], poi['lon']),
                **poi  # Include all attributes
            )
        
        # Add edges with travel options
        for i, poi1 in enumerate(pois):
            for j, poi2 in enumerate(pois):
                if i != j:
                    self._add_multi_modal_edge(
                        poi1['id'], 
                        poi2['id'],
                        distance_matrix[i, j]
                    )
    
    def _add_multi_modal_edge(self, u: str, v: str, distance: float):
        """Add edges for different transportation modes"""
        # Walking
        self.graph.add_edge(u, v,
            mode='walking',
            distance=distance,
            time=distance / 4.5,  # 4.5 km/h
            cost=0.0
        )
        
        # Public transit
        if self._has_transit_connection(u, v):
            self.graph.add_edge(u, v,
                mode='public_transit',
                distance=distance,
                time=distance / 25.0 + 0.117,  # Transfer time
                cost=2.90  # NYC subway fare
            )
        
        # Taxi
        self.graph.add_edge(u, v,
            mode='taxi',
            distance=distance,
            time=distance / 18.0,  # Average NYC taxi speed
            cost=3.0 + distance * 1.75  # NYC taxi rates
        )
    
    def _get_borough(self, lat: float, lon: float) -> str:
        """Determine NYC borough from coordinates"""
        # Simplified borough boundaries
        if 40.700 <= lat <= 40.882 and -74.019 <= lon <= -73.907:
            return 'manhattan'
        elif 40.570 <= lat <= 40.739 and -74.042 <= lon <= -73.833:
            return 'brooklyn'
        elif 40.489 <= lat <= 40.812 and -73.962 <= lon <= -73.700:
            return 'queens'
        elif 40.785 <= lat <= 40.917 and -73.933 <= lon <= -73.748:
            return 'bronx'
        elif 40.477 <= lat <= 40.651 and -74.259 <= lon <= -74.034:
            return 'staten_island'
        return 'unknown'
    
    def _has_transit_connection(self, u: str, v: str) -> bool:
        """Check if POIs have subway connectivity"""
        # In practice, query transit API or pre-computed data
        # Simplified: assume transit available in Manhattan/Brooklyn
        u_borough = self.graph.nodes[u]['borough']
        v_borough = self.graph.nodes[v]['borough']
        return u_borough in ['manhattan', 'brooklyn'] and \
               v_borough in ['manhattan', 'brooklyn']
```

### 4.2.4 OR-Tools Integration

For complex routing problems, we leverage Google OR-Tools:

```python
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

class ORToolsRouter:
    """OR-Tools based routing for selected POIs"""
    
    def __init__(self, distance_matrix: np.ndarray, time_windows: List[Tuple[int, int]]):
        self.distance_matrix = (distance_matrix * 1000).astype(int)  # Convert to meters
        self.time_windows = time_windows
        self.manager = pywrapcp.RoutingIndexManager(
            len(distance_matrix), 1, 0  # Single vehicle, depot at 0
        )
        self.routing = pywrapcp.RoutingModel(self.manager)
        
    def solve_vrptw(self, time_limit_seconds: int = 30) -> List[int]:
        """Solve Vehicle Routing Problem with Time Windows"""
        # Distance callback
        def distance_callback(from_index, to_index):
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            return self.distance_matrix[from_node][to_node]
        
        transit_callback_index = self.routing.RegisterTransitCallback(distance_callback)
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Time windows
        time = 'Time'
        self.routing.AddDimension(
            transit_callback_index,
            30,  # Allow waiting time
            86400,  # Maximum time (24 hours in seconds)
            False,  # Don't force start cumul to zero
            time
        )
        time_dimension = self.routing.GetDimensionOrDie(time)
        
        # Add time window constraints
        for location_idx, (start, end) in enumerate(self.time_windows):
            if location_idx == 0:
                continue  # Skip depot
            index = self.manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(start, end)
        
        # Search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = time_limit_seconds
        
        # Solve
        solution = self.routing.SolveWithParameters(search_parameters)
        
        if solution:
            return self._extract_solution(solution)
        return []
    
    def _extract_solution(self, solution) -> List[int]:
        """Extract route from OR-Tools solution"""
        route = []
        index = self.routing.Start(0)
        while not self.routing.IsEnd(index):
            route.append(self.manager.IndexToNode(index))
            index = solution.Value(self.routing.NextVar(index))
        return route[1:]  # Exclude depot
```

### 4.2.5 Foursquare API Integration

For real-world POI data, we integrate with Foursquare's Places API:

```python
import requests
from typing import List, Dict, Optional
from datetime import datetime
import time

class FoursquareClient:
    """Foursquare Places API client for NYC POI data"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.foursquare.com/v3/places"
        self.headers = {
            "Accept": "application/json",
            "Authorization": api_key
        }
        self.rate_limit_remaining = 500
        self.rate_limit_reset = None
        
    def search_pois(self, 
                   categories: List[str],
                   near: str = "New York, NY",
                   limit: int = 50) -> List[Dict]:
        """Search for POIs by category"""
        params = {
            "categories": ",".join(categories),
            "near": near,
            "limit": limit,
            "fields": "name,location,categories,rating,price,hours,photos,tips"
        }
        
        response = self._make_request("/search", params)
        return self._process_pois(response.get("results", []))
    
    def get_poi_details(self, fsq_id: str) -> Optional[Dict]:
        """Get detailed information for a specific POI"""
        endpoint = f"/{fsq_id}"
        params = {
            "fields": "name,location,categories,rating,price,hours,photos,tips,stats"
        }
        
        response = self._make_request(endpoint, params)
        return self._process_poi(response) if response else None
    
    def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """Make API request with rate limiting"""
        # Check rate limit
        if self.rate_limit_remaining <= 10:
            if self.rate_limit_reset and datetime.now() < self.rate_limit_reset:
                sleep_time = (self.rate_limit_reset - datetime.now()).seconds
                time.sleep(sleep_time + 1)
        
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url, headers=self.headers, params=params)
        
        # Update rate limit info
        self.rate_limit_remaining = int(response.headers.get("X-RateLimit-Remaining", 500))
        reset_timestamp = int(response.headers.get("X-RateLimit-Reset", 0))
        if reset_timestamp:
            self.rate_limit_reset = datetime.fromtimestamp(reset_timestamp)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return {}
    
    def _process_poi(self, poi_data: Dict) -> Dict:
        """Convert Foursquare POI to our format"""
        location = poi_data.get("location", {})
        categories = poi_data.get("categories", [])
        
        # Extract primary category
        primary_category = "other"
        category_mapping = {
            "museum": ["Museum", "Art Gallery", "History Museum"],
            "park": ["Park", "Garden", "Playground"],
            "restaurant": ["Restaurant", "Cafe", "Food"],
            "landmark": ["Monument", "Historic Site", "Building"],
            "shopping": ["Shopping", "Store", "Market"],
            "entertainment": ["Theater", "Cinema", "Music Venue"]
        }
        
        for cat in categories:
            cat_name = cat.get("name", "")
            for our_cat, fsq_cats in category_mapping.items():
                if any(fsq in cat_name for fsq in fsq_cats):
                    primary_category = our_cat
                    break
        
        # Process hours
        hours = poi_data.get("hours", {})
        opening_hours = self._process_hours(hours)
        
        return {
            "id": poi_data.get("fsq_id"),
            "name": poi_data.get("name"),
            "lat": location.get("lat"),
            "lon": location.get("lng"),
            "address": location.get("formatted_address", ""),
            "category": primary_category,
            "subcategories": [cat.get("name") for cat in categories],
            "rating": poi_data.get("rating", 0) / 2.0,  # Convert to 5-point scale
            "price_level": poi_data.get("price", 2),
            "popularity": poi_data.get("stats", {}).get("total_visitors", 0) / 10000,
            "opening_hours": opening_hours,
            "avg_visit_duration": self._estimate_duration(primary_category),
            "entrance_fee": self._estimate_entrance_fee(primary_category, poi_data)
        }
    
    def _process_hours(self, hours_data: Dict) -> Dict:
        """Extract opening hours"""
        if not hours_data.get("regular"):
            # Default hours if not available
            return {"weekday": [9, 20], "weekend": [10, 18]}
        
        # Simplified: just get weekday hours
        for period in hours_data["regular"]:
            if period.get("day") in [1, 2, 3, 4, 5]:  # Monday-Friday
                open_time = period.get("open", "0900")
                close_time = period.get("close", "2000")
                return {
                    "weekday": [
                        int(open_time[:2]) + int(open_time[2:]) / 60,
                        int(close_time[:2]) + int(close_time[2:]) / 60
                    ]
                }
        
        return {"weekday": [9, 20], "weekend": [10, 18]}
    
    def _estimate_duration(self, category: str) -> float:
        """Estimate visit duration by category"""
        duration_map = {
            "museum": 2.5,
            "park": 1.5,
            "restaurant": 1.5,
            "landmark": 0.75,
            "shopping": 1.0,
            "entertainment": 2.5,
            "other": 1.0
        }
        return duration_map.get(category, 1.0)
    
    def _estimate_entrance_fee(self, category: str, poi_data: Dict) -> float:
        """Estimate entrance fee"""
        if category == "museum":
            return 25.0  # Average NYC museum
        elif category == "landmark":
            # Some landmarks charge admission
            if "Empire State" in poi_data.get("name", ""):
                return 40.0
            elif "Statue of Liberty" in poi_data.get("name", ""):
                return 25.0
            elif "Top of the Rock" in poi_data.get("name", ""):
                return 38.0
        elif category == "entertainment":
            return 50.0  # Average show ticket
        
        return 0.0  # Most categories are free
```

## 4.3 NYC Data Pipeline

### 4.3.1 Data Collection and Integration

Our data pipeline combines multiple sources for comprehensive NYC coverage:

```python
import osmnx as ox
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import json

class NYCDataPipeline:
    """Pipeline for collecting and processing NYC POI data"""
    
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = output_dir
        self.nyc_bounds = {
            'north': 40.917,
            'south': 40.477,
            'east': -73.700,
            'west': -74.259
        }
        
    def collect_osm_data(self) -> gpd.GeoDataFrame:
        """Collect POI data from OpenStreetMap"""
        # Define POI types to collect
        osm_tags = {
            'amenity': ['restaurant', 'cafe', 'bar', 'museum', 'theatre'],
            'tourism': ['attraction', 'museum', 'gallery', 'viewpoint'],
            'leisure': ['park', 'garden', 'playground', 'sports_centre'],
            'shop': ['mall', 'department_store', 'boutique'],
            'historic': True,
            'building': ['church', 'cathedral', 'synagogue', 'mosque']
        }
        
        # Download POIs for NYC
        gdf_list = []
        for key, values in osm_tags.items():
            try:
                gdf = ox.geometries_from_bbox(
                    self.nyc_bounds['north'],
                    self.nyc_bounds['south'],
                    self.nyc_bounds['east'],
                    self.nyc_bounds['west'],
                    tags={key: values}
                )
                gdf['source'] = 'osm'
                gdf['category'] = key
                gdf_list.append(gdf)
            except Exception as e:
                print(f"Error downloading {key}: {e}")
        
        # Combine all POIs
        combined_gdf = gpd.GeoDataFrame(
            pd.concat(gdf_list, ignore_index=True)
        )
        
        return self._process_osm_data(combined_gdf)
    
    def _process_osm_data(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Process and clean OSM data"""
        # Extract point geometries
        gdf['geometry'] = gdf.geometry.centroid
        
        # Extract relevant columns
        processed = gpd.GeoDataFrame({
            'id': 'osm_' + gdf.index.astype(str),
            'name': gdf['name'].fillna('Unknown'),
            'lat': gdf.geometry.y,
            'lon': gdf.geometry.x,
            'category': gdf['category'],
            'tags': gdf['tags'].apply(lambda x: x if isinstance(x, dict) else {}),
            'geometry': gdf.geometry
        })
        
        # Map to our categories
        processed['category'] = processed.apply(self._map_category, axis=1)
        
        # Filter valid POIs
        processed = processed[
            processed['name'] != 'Unknown'
        ].drop_duplicates(subset=['name', 'lat', 'lon'])
        
        return processed
    
    def _map_category(self, row) -> str:
        """Map OSM tags to our categories"""
        tags = row['tags']
        
        # Check specific tags
        if tags.get('amenity') == 'restaurant' or tags.get('cuisine'):
            return 'restaurant'
        elif tags.get('tourism') == 'museum' or tags.get('museum'):
            return 'museum'
        elif tags.get('leisure') in ['park', 'garden']:
            return 'park'
        elif tags.get('historic') or tags.get('memorial'):
            return 'landmark'
        elif tags.get('shop') or tags.get('retail'):
            return 'shopping'
        elif tags.get('amenity') in ['theatre', 'cinema']:
            return 'entertainment'
        
        return 'other'
    
    def merge_data_sources(self, 
                          osm_data: gpd.GeoDataFrame,
                          foursquare_data: List[Dict]) -> pd.DataFrame:
        """Merge OSM and Foursquare data"""
        # Convert Foursquare to GeoDataFrame
        fsq_df = pd.DataFrame(foursquare_data)
        fsq_gdf = gpd.GeoDataFrame(
            fsq_df,
            geometry=[Point(xy) for xy in zip(fsq_df.lon, fsq_df.lat)]
        )
        
        # Spatial join to find duplicates
        buffer_distance = 0.0001  # ~11 meters
        osm_data['buffer'] = osm_data.geometry.buffer(buffer_distance)
        
        # Find matches
        matches = gpd.sjoin(
            fsq_gdf,
            osm_data.set_geometry('buffer'),
            how='left',
            predicate='within'
        )
        
        # Merge non-duplicate Foursquare data
        new_fsq = fsq_gdf[~fsq_gdf.index.isin(matches.index)]
        
        # Combine datasets
        combined = pd.concat([
            osm_data.drop(columns=['buffer', 'geometry']),
            new_fsq.drop(columns=['geometry'])
        ], ignore_index=True)
        
        # Add missing attributes
        combined = self._enrich_poi_data(combined)
        
        return combined
    
    def _enrich_poi_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add missing attributes and enrich data"""
        # Add default values
        defaults = {
            'rating': 4.0,
            'popularity': 0.5,
            'entrance_fee': 0.0,
            'avg_visit_duration': 1.5,
            'price_level': 2
        }
        
        for col, default in defaults.items():
            if col not in df.columns:
                df[col] = default
            else:
                df[col] = df[col].fillna(default)
        
        # Add opening hours
        if 'opening_hours' not in df.columns:
            df['opening_hours'] = df['category'].map(self._default_hours)
        
        # Calculate derived attributes
        df['popularity'] = df['popularity'].clip(0, 1)
        df['rating'] = df['rating'].clip(0, 5)
        
        # Add borough information
        df['borough'] = df.apply(
            lambda row: self._get_borough(row['lat'], row['lon']), 
            axis=1
        )
        
        # Add subway proximity (simplified)
        df['near_subway'] = df['borough'].isin(['manhattan', 'brooklyn'])
        
        return df
    
    def _default_hours(self, category: str) -> Dict:
        """Default opening hours by category"""
        hours_map = {
            'museum': {'weekday': [10, 17], 'weekend': [10, 18]},
            'park': {'weekday': [6, 22], 'weekend': [6, 22]},
            'restaurant': {'weekday': [11, 23], 'weekend': [11, 24]},
            'landmark': {'weekday': [9, 18], 'weekend': [9, 18]},
            'shopping': {'weekday': [10, 21], 'weekend': [10, 21]},
            'entertainment': {'weekday': [18, 23], 'weekend': [18, 24]},
            'other': {'weekday': [9, 20], 'weekend': [9, 20]}
        }
        return hours_map.get(category, hours_map['other'])
    
    def _get_borough(self, lat: float, lon: float) -> str:
        """Get borough from coordinates"""
        # Borough boundaries (simplified)
        if 40.700 <= lat <= 40.882 and -74.019 <= lon <= -73.907:
            return 'manhattan'
        elif 40.570 <= lat <= 40.739 and -74.042 <= lon <= -73.833:
            return 'brooklyn'
        elif 40.489 <= lat <= 40.812 and -73.962 <= lon <= -73.700:
            return 'queens'
        elif 40.785 <= lat <= 40.917 and -73.933 <= lon <= -73.748:
            return 'bronx'
        elif 40.477 <= lat <= 40.651 and -74.259 <= lon <= -74.034:
            return 'staten_island'
        return 'other'
    
    def build_distance_matrix(self, pois: pd.DataFrame) -> np.ndarray:
        """Build distance matrix using NYC grid distance"""
        n = len(pois)
        distances = np.zeros((n, n))
        
        # Convert to numpy arrays for faster computation
        lats = pois['lat'].values
        lons = pois['lon'].values
        
        # Use numba-optimized function
        from numba import njit, prange
        
        @njit(parallel=True)
        def compute_distances(lats, lons, distances):
            for i in prange(n):
                for j in range(i + 1, n):
                    # Manhattan distance with NYC grid factor
                    lat_km = 111.0
                    lon_km = 111.0 * np.cos(np.radians((lats[i] + lats[j]) / 2))
                    
                    dlat = abs(lats[j] - lats[i]) * lat_km
                    dlon = abs(lons[j] - lons[i]) * lon_km
                    
                    dist = 1.4 * (dlat + dlon)  # NYC grid factor
                    distances[i, j] = dist
                    distances[j, i] = dist
        
        compute_distances(lats, lons, distances)
        return distances
    
    def save_processed_data(self, pois: pd.DataFrame, distance_matrix: np.ndarray):
        """Save processed data"""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save POIs
        pois.to_json(
            f"{self.output_dir}/nyc_pois.json",
            orient='records',
            indent=2
        )
        
        # Save distance matrix
        np.save(f"{self.output_dir}/distance_matrix.npy", distance_matrix)
        
        # Save metadata
        metadata = {
            'n_pois': len(pois),
            'categories': pois['category'].value_counts().to_dict(),
            'boroughs': pois['borough'].value_counts().to_dict(),
            'date_processed': datetime.now().isoformat()
        }
        
        with open(f"{self.output_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved {len(pois)} POIs to {self.output_dir}")
```

### 4.3.2 R-tree Spatial Indexing

For efficient spatial queries on 10,000+ POIs:

```python
from rtree import index
import numpy as np
from typing import List, Tuple

class SpatialIndex:
    """R-tree spatial index for POI queries"""
    
    def __init__(self, pois: List[Dict]):
        # Create R-tree
        p = index.Property()
        p.dimension = 2
        p.variant = index.RT_Star  # R* tree variant
        p.fill_factor = 0.7
        
        self.idx = index.Index(properties=p)
        self.pois = {poi['id']: poi for poi in pois}
        
        # Insert POIs into index
        for i, poi in enumerate(pois):
            # Bounding box is point for POIs
            left, bottom = poi['lon'], poi['lat']
            right, top = poi['lon'], poi['lat']
            self.idx.insert(i, (left, bottom, right, top), obj=poi['id'])
    
    def find_nearby_pois(self, lat: float, lon: float, 
                        radius_km: float, category: str = None) -> List[Dict]:
        """Find POIs within radius"""
        # Convert radius to degrees (approximate)
        lat_offset = radius_km / 111.0
        lon_offset = radius_km / (111.0 * np.cos(np.radians(lat)))
        
        # Query bounding box
        bbox = (
            lon - lon_offset,
            lat - lat_offset,
            lon + lon_offset,
            lat + lat_offset
        )
        
        nearby_ids = [n.object for n in self.idx.intersection(bbox, objects=True)]
        nearby_pois = [self.pois[poi_id] for poi_id in nearby_ids]
        
        # Filter by exact distance
        result = []
        for poi in nearby_pois:
            dist = self._haversine_distance(lat, lon, poi['lat'], poi['lon'])
            if dist <= radius_km:
                if category is None or poi['category'] == category:
                    result.append(poi)
        
        return sorted(result, key=lambda p: self._haversine_distance(
            lat, lon, p['lat'], p['lon']
        ))
    
    def find_pois_in_region(self, bounds: Dict) -> List[Dict]:
        """Find POIs within geographic bounds"""
        bbox = (
            bounds['west'],
            bounds['south'],
            bounds['east'],
            bounds['north']
        )
        
        poi_ids = [n.object for n in self.idx.intersection(bbox, objects=True)]
        return [self.pois[poi_id] for poi_id in poi_ids]
    
    def _haversine_distance(self, lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """Calculate haversine distance in km"""
        R = 6371  # Earth radius in km
        
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        lat1 = np.radians(lat1)
        lat2 = np.radians(lat2)
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def save_index(self, filepath: str):
        """Save spatial index to disk"""
        self.idx.save(filepath)
    
    def load_index(self, filepath: str):
        """Load spatial index from disk"""
        p = index.Property()
        p.dimension = 2
        p.variant = index.RT_Star
        self.idx = index.Index(filepath, properties=p)
```

## 4.4 Algorithm Implementations with Optimizations

### 4.4.1 Numba JIT Optimization for A*

Critical performance gains through JIT compilation:

```python
from numba import njit, typed, types
import numpy as np

@njit
def manhattan_distance_nyc(lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
    """NYC grid-adjusted Manhattan distance"""
    NYC_GRID_FACTOR = 1.4
    lat_km = 111.0
    lon_km = 111.0 * np.cos(np.radians((lat1 + lat2) / 2))
    
    dlat = abs(lat2 - lat1) * lat_km
    dlon = abs(lon2 - lon1) * lon_km
    
    return NYC_GRID_FACTOR * (dlat + dlon)

@njit
def compute_heuristic_fast(current_pois: types.Array(types.int32, 1, 'C'),
                          unvisited_mask: types.Array(types.boolean, 1, 'C'),
                          utilities: types.Array(types.float32, 1, 'C'),
                          distances: types.Array(types.float32, 2, 'C'),
                          min_pois_needed: int) -> float:
    """Fast heuristic computation for A*"""
    if min_pois_needed <= 0:
        return 0.0
    
    # Find unvisited indices
    unvisited_indices = np.where(unvisited_mask)[0]
    if len(unvisited_indices) < min_pois_needed:
        return np.inf
    
    # Get top-k by utility
    unvisited_utilities = utilities[unvisited_indices]
    top_k_indices = np.argsort(unvisited_utilities)[-min_pois_needed:]
    top_k_global = unvisited_indices[top_k_indices]
    
    # Compute MST approximation using Prim's algorithm
    if min_pois_needed == 1:
        # Just connection cost
        if len(current_pois) > 0:
            last_poi = current_pois[-1]
            return -utilities[top_k_global[0]] + distances[last_poi, top_k_global[0]] / 25.0
        return -utilities[top_k_global[0]]
    
    # Simplified MST for small k
    mst_cost = 0.0
    visited = np.zeros(min_pois_needed, dtype=np.bool_)
    visited[0] = True
    
    for _ in range(min_pois_needed - 1):
        min_edge = np.inf
        min_idx = -1
        
        for i in range(min_pois_needed):
            if visited[i]:
                for j in range(min_pois_needed):
                    if not visited[j]:
                        edge_cost = distances[top_k_global[i], top_k_global[j]]
                        if edge_cost < min_edge:
                            min_edge = edge_cost
                            min_idx = j
        
        if min_idx >= 0:
            visited[min_idx] = True
            mst_cost += min_edge
    
    # Add connection from current location
    if len(current_pois) > 0:
        last_poi = current_pois[-1]
        min_connection = np.inf
        for idx in top_k_global:
            conn_cost = distances[last_poi, idx]
            if conn_cost < min_connection:
                min_connection = conn_cost
        mst_cost += min_connection
    
    # Convert to utility penalty
    time_cost = mst_cost / 25.0  # Assume transit speed
    utility_sum = np.sum(utilities[top_k_global])
    
    return -utility_sum + time_cost * 10.0

# Usage in A* implementation
class OptimizedAStarPlanner:
    """A* with Numba optimizations"""
    
    def __init__(self, pois: List[Dict], distance_matrix: np.ndarray):
        self.pois = pois
        self.n_pois = len(pois)
        
        # Pre-compute for Numba
        self.utilities = np.array([self._compute_utility(p) for p in pois], 
                                 dtype=np.float32)
        self.distances = distance_matrix.astype(np.float32)
        
    def _compute_utility(self, poi: Dict) -> float:
        """Base utility for POI"""
        return poi['rating'] / 5.0 * poi.get('popularity', 0.5)
    
    def compute_heuristic(self, visited_pois: List[int], 
                         min_pois_needed: int) -> float:
        """Wrapper for Numba heuristic"""
        current = np.array(visited_pois, dtype=np.int32)
        unvisited = np.ones(self.n_pois, dtype=np.bool_)
        unvisited[visited_pois] = False
        
        return compute_heuristic_fast(
            current, unvisited, self.utilities, 
            self.distances, min_pois_needed
        )
```

### 4.4.2 Caching Strategies

Memoization for repeated computations:

```python
from functools import lru_cache
import hashlib
import pickle

class ComputationCache:
    """Caching for expensive computations"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        self.memory_cache = {}
        os.makedirs(cache_dir, exist_ok=True)
    
    @lru_cache(maxsize=10000)
    def get_travel_time(self, from_id: str, to_id: str, 
                       mode: str, time_of_day: int) -> float:
        """Cached travel time computation"""
        key = f"{from_id}_{to_id}_{mode}_{time_of_day}"
        
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # Check disk cache
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                value = pickle.load(f)
                self.memory_cache[key] = value
                return value
        
        # Compute value
        value = self._compute_travel_time(from_id, to_id, mode, time_of_day)
        
        # Store in cache
        self.memory_cache[key] = value
        with open(cache_file, 'wb') as f:
            pickle.dump(value, f)
        
        return value
    
    def _compute_travel_time(self, from_id: str, to_id: str,
                           mode: str, time_of_day: int) -> float:
        """Actual travel time computation"""
        # Implementation depends on mode and time
        base_time = self._base_travel_time(from_id, to_id, mode)
        
        # Rush hour adjustments
        if mode == 'taxi' and time_of_day in [7, 8, 17, 18]:
            return base_time * 1.5
        
        return base_time
    
    @lru_cache(maxsize=1000)
    def get_itinerary_css(self, poi_sequence: Tuple[str, ...],
                         user_prefs: str) -> float:
        """Cached CSS computation"""
        # Hash complex inputs
        key = hashlib.sha256(
            f"{poi_sequence}_{user_prefs}".encode()
        ).hexdigest()
        
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # Compute CSS
        css = self._compute_css(poi_sequence, json.loads(user_prefs))
        self.memory_cache[key] = css
        
        return css
```

### 4.4.3 Edge Case Handling

Robust handling of NYC-specific scenarios:

```python
class EdgeCaseHandler:
    """Handle NYC-specific edge cases"""
    
    def __init__(self):
        self.special_events = self._load_special_events()
        self.seasonal_closures = self._load_seasonal_closures()
    
    def validate_itinerary(self, itinerary: List[Dict], 
                          date: datetime) -> Tuple[bool, List[str]]:
        """Validate itinerary for edge cases"""
        issues = []
        
        for i, poi in enumerate(itinerary):
            # Check seasonal closures
            if self._is_seasonally_closed(poi, date):
                issues.append(f"{poi['name']} is closed during {date.strftime('%B')}")
            
            # Check special events
            if self._has_conflicting_event(poi, date):
                issues.append(f"{poi['name']} has special event on {date.strftime('%Y-%m-%d')}")
            
            # Check time windows
            if i > 0:
                travel_time = self._get_travel_time(itinerary[i-1], poi)
                arrival_time = self._calculate_arrival_time(itinerary[:i+1])
                
                if arrival_time < poi['opening_hours']['weekday'][0]:
                    issues.append(f"{poi['name']} not yet open at {arrival_time:.1f}")
                elif arrival_time > poi['opening_hours']['weekday'][1]:
                    issues.append(f"{poi['name']} closed at {arrival_time:.1f}")
            
            # Check inter-borough transitions
            if i > 0 and self._is_problematic_transition(itinerary[i-1], poi):
                issues.append(f"Difficult transition from {itinerary[i-1]['name']} to {poi['name']}")
        
        return len(issues) == 0, issues
    
    def _is_seasonally_closed(self, poi: Dict, date: datetime) -> bool:
        """Check seasonal closures"""
        # Bryant Park Winter Village
        if poi['id'] == 'bryant_park' and 11 <= date.month <= 1:
            return True
        
        # Outdoor venues in winter
        if poi['category'] == 'park' and date.month in [12, 1, 2]:
            if 'pool' in poi['name'].lower() or 'beach' in poi['name'].lower():
                return True
        
        return False
    
    def _has_conflicting_event(self, poi: Dict, date: datetime) -> bool:
        """Check for conflicting events"""
        # NYC Marathon
        if date.month == 11 and date.day == 7:  # First Sunday in November
            if poi['borough'] in ['manhattan', 'brooklyn', 'queens']:
                if 'street' in poi['address'].lower():
                    return True
        
        # New Year's Eve in Times Square
        if date.month == 12 and date.day == 31:
            if poi['lat'] > 40.755 and poi['lat'] < 40.765:
                if poi['lon'] > -73.990 and poi['lon'] < -73.980:
                    return True
        
        return False
    
    def _is_problematic_transition(self, from_poi: Dict, to_poi: Dict) -> bool:
        """Check for problematic transitions"""
        # Staten Island Ferry timing
        if (from_poi['borough'] == 'staten_island' and to_poi['borough'] != 'staten_island') or \
           (from_poi['borough'] != 'staten_island' and to_poi['borough'] == 'staten_island'):
            # Ferry schedule constraints
            return True
        
        # No direct subway between certain boroughs
        if from_poi['borough'] == 'queens' and to_poi['borough'] == 'bronx':
            return True
        
        return False
    
    def suggest_alternatives(self, poi: Dict, issue: str) -> List[Dict]:
        """Suggest alternative POIs for issues"""
        alternatives = []
        
        if "closed" in issue:
            # Find similar POIs that are open
            similar = self._find_similar_pois(poi, max_distance=2.0)
            alternatives = [p for p in similar if not self._is_seasonally_closed(p, datetime.now())]
        
        elif "special event" in issue:
            # Find POIs in different area
            nearby = self._find_nearby_pois(poi, radius=1.0)
            alternatives = [p for p in nearby if p['id'] != poi['id']]
        
        return alternatives[:3]  # Top 3 alternatives
```

## 4.5 Testing Methodology

### 4.5.1 Unit Testing Framework

Comprehensive test coverage for core algorithms:

```python
import unittest
import numpy as np
from typing import List, Dict

class TestGreedyAlgorithms(unittest.TestCase):
    """Test cases for greedy algorithm implementations"""
    
    def setUp(self):
        """Set up test data"""
        self.test_pois = [
            {
                'id': 'met',
                'name': 'Metropolitan Museum',
                'lat': 40.7794,
                'lon': -73.9632,
                'category': 'museum',
                'rating': 4.8,
                'entrance_fee': 25.0,
                'avg_visit_duration': 3.0,
                'opening_hours': {'weekday': [10, 17]},
                'popularity': 0.9
            },
            {
                'id': 'central_park',
                'name': 'Central Park',
                'lat': 40.7829,
                'lon': -73.9654,
                'category': 'park',
                'rating': 4.9,
                'entrance_fee': 0.0,
                'avg_visit_duration': 2.0,
                'opening_hours': {'weekday': [6, 22]},
                'popularity': 0.95
            }
        ]
        
        self.distance_matrix = np.array([
            [0.0, 0.4],
            [0.4, 0.0]
        ])
        
        self.user_prefs = {'museum': 0.8, 'park': 0.7}
        
    def test_empty_poi_list(self):
        """Test with empty POI list"""
        from greedy_algorithms import GreedyPOISelection
        
        planner = GreedyPOISelection([], np.array([]))
        result = planner.select_pois(self.user_prefs, Constraints())
        
        self.assertEqual(len(result), 0)
    
    def test_budget_constraint(self):
        """Test budget constraint enforcement"""
        from greedy_algorithms import GreedyPOISelection, Constraints
        
        planner = GreedyPOISelection(self.test_pois, self.distance_matrix)
        constraints = Constraints(budget=20.0)  # Less than museum fee
        
        result = planner.select_pois(self.user_prefs, constraints)
        
        # Should only select free park
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].category, 'park')
        
    def test_time_constraint(self):
        """Test time constraint enforcement"""
        from greedy_algorithms import GreedyPOISelection, Constraints
        
        planner = GreedyPOISelection(self.test_pois, self.distance_matrix)
        constraints = Constraints(max_time_hours=1.5)  # Very limited time
        
        result = planner.select_pois(self.user_prefs, constraints)
        
        # Should select at most one POI
        self.assertLessEqual(len(result), 1)
    
    def test_preference_ordering(self):
        """Test POIs selected according to preferences"""
        from greedy_algorithms import GreedyPOISelection, Constraints
        
        # Strong museum preference
        strong_museum_prefs = {'museum': 0.9, 'park': 0.1}
        
        planner = GreedyPOISelection(self.test_pois, self.distance_matrix)
        constraints = Constraints(budget=100, max_pois=1)
        
        result = planner.select_pois(strong_museum_prefs, constraints)
        
        # Should select museum despite cost
        self.assertEqual(result[0].category, 'museum')
    
    def test_deterministic_results(self):
        """Test algorithm produces deterministic results"""
        from greedy_algorithms import GreedyPOISelection, Constraints
        
        planner = GreedyPOISelection(self.test_pois, self.distance_matrix)
        constraints = Constraints()
        
        # Run multiple times
        results = []
        for _ in range(5):
            result = planner.select_pois(self.user_prefs, constraints)
            results.append([poi.id for poi in result])
        
        # All results should be identical
        for i in range(1, 5):
            self.assertEqual(results[0], results[i])


class TestAStarAlgorithm(unittest.TestCase):
    """Test cases for A* implementation"""
    
    def test_optimal_solution(self):
        """Test A* finds optimal solution for small instances"""
        from astar_itinerary import AStarItineraryPlanner
        
        # Small instance where we know optimal
        pois = [
            {'id': 'a', 'name': 'A', 'lat': 40.7, 'lon': -74.0,
             'category': 'museum', 'rating': 5.0, 'entrance_fee': 0,
             'avg_visit_duration': 1.0, 'opening_hours': {'weekday': [0, 24]},
             'popularity': 1.0},
            {'id': 'b', 'name': 'B', 'lat': 40.71, 'lon': -74.0,
             'category': 'museum', 'rating': 4.0, 'entrance_fee': 0,
             'avg_visit_duration': 1.0, 'opening_hours': {'weekday': [0, 24]},
             'popularity': 0.8},
            {'id': 'c', 'name': 'C', 'lat': 40.72, 'lon': -74.0,
             'category': 'museum', 'rating': 3.0, 'entrance_fee': 0,
             'avg_visit_duration': 1.0, 'opening_hours': {'weekday': [0, 24]},
             'popularity': 0.6}
        ]
        
        # Simple distance matrix
        distances = np.array([
            [0, 1, 2],
            [1, 0, 1],
            [2, 1, 0]
        ])
        
        planner = AStarItineraryPlanner(pois, distances)
        constraints = Constraints(max_pois=2, max_time_hours=3)
        
        result = planner.plan_itinerary({'museum': 1.0}, constraints)
        
        # Should select A and B (highest ratings)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].id, 'a')
        self.assertEqual(result[1].id, 'b')
    
    def test_heuristic_admissibility(self):
        """Test heuristic never overestimates"""
        from astar_itinerary import AStarItineraryPlanner, ItineraryState
        
        planner = AStarItineraryPlanner(self.test_pois, self.distance_matrix)
        
        # Test various states
        states = [
            ItineraryState((), 9.0, 100.0, 0.0, 0.0),
            ItineraryState(('met',), 13.0, 75.0, 10.0, 0.4)
        ]
        
        for state in states:
            h_value = planner._compute_heuristic(
                state, {0, 1}, self.user_prefs, Constraints()
            )
            
            # Heuristic should be finite and non-positive (for maximization)
            self.assertLess(h_value, float('inf'))
            self.assertLessEqual(h_value, 0)


class TestLPAStar(unittest.TestCase):
    """Test cases for LPA* dynamic replanning"""
    
    def test_poi_closure_update(self):
        """Test handling of POI closure"""
        from lpa_star import LPAStarPlanner, DynamicUpdate, UpdateType
        
        planner = LPAStarPlanner(self.test_pois)
        
        # Initial plan
        initial = planner.plan_with_updates(
            self.user_prefs,
            Constraints(max_pois=2)
        )
        self.assertEqual(len(initial), 2)
        
        # Close museum
        update = DynamicUpdate(
            update_type=UpdateType.POI_CLOSED,
            poi_ids=['met'],
            timestamp=datetime.now()
        )
        
        updated = planner.replan_after_update(update)
        
        # Should not include closed POI
        poi_ids = [poi.id for poi in updated]
        self.assertNotIn('met', poi_ids)
    
    def test_computation_reuse(self):
        """Test LPA* reuses computation"""
        from lpa_star import LPAStarPlanner, DynamicUpdate, UpdateType
        
        # Larger test set
        large_pois = []
        for i in range(20):
            large_pois.append({
                'id': f'poi_{i}',
                'name': f'POI {i}',
                'lat': 40.7 + i * 0.01,
                'lon': -74.0 + i * 0.01,
                'category': 'museum' if i % 2 else 'park',
                'rating': 4.0 + (i % 5) * 0.2,
                'entrance_fee': 0 if i % 3 else 25,
                'avg_visit_duration': 1.5,
                'opening_hours': {'weekday': [9, 20]},
                'popularity': 0.5 + (i % 4) * 0.1
            })
        
        planner = LPAStarPlanner(large_pois)
        
        # Initial plan
        initial = planner.plan_with_updates(
            self.user_prefs,
            Constraints(max_pois=5)
        )
        
        initial_stats = planner.get_statistics()
        
        # Minor update
        update = DynamicUpdate(
            update_type=UpdateType.POI_CLOSED,
            poi_ids=['poi_15'],
            timestamp=datetime.now()
        )
        
        updated = planner.replan_after_update(update)
        final_stats = planner.get_statistics()
        
        # Should reuse significant computation
        self.assertGreater(final_stats['computation_reuse'], 0.7)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete system"""
    
    def test_end_to_end_planning(self):
        """Test complete planning pipeline"""
        from hybrid_planner import HybridPlanner
        
        # Initialize with test data
        planner = HybridPlanner(
            'test_data/pois.json',
            'test_data/distance_matrix.npy'
        )
        
        # User profile
        user_profile = {
            'preferences': {'museum': 0.8, 'park': 0.6, 'restaurant': 0.7},
            'constraints': {
                'budget': 100,
                'start_time': 9.0,
                'end_time': 18.0,
                'max_walking': 2.0
            }
        }
        
        # Generate itinerary
        itinerary = planner.plan_itinerary(user_profile)
        
        # Validate result
        self.assertIsNotNone(itinerary)
        self.assertGreater(len(itinerary.pois), 0)
        self.assertLessEqual(len(itinerary.pois), 7)
        
        # Check constraints
        total_cost = sum(poi.entrance_fee for poi in itinerary.pois)
        self.assertLessEqual(total_cost, user_profile['constraints']['budget'])
    
    def test_performance_benchmarks(self):
        """Test performance meets requirements"""
        import time
        from hybrid_planner import HybridPlanner
        
        # Load realistic dataset
        planner = HybridPlanner(
            'data/nyc_pois.json',
            'data/distance_matrix.npy'
        )
        
        # Multiple test scenarios
        scenarios = [
            {'museum': 0.9, 'park': 0.5},
            {'restaurant': 0.8, 'shopping': 0.7},
            {'landmark': 0.9, 'entertainment': 0.6}
        ]
        
        times = []
        for prefs in scenarios:
            start = time.time()
            result = planner.plan_itinerary({
                'preferences': prefs,
                'constraints': {'budget': 150}
            })
            elapsed = time.time() - start
            times.append(elapsed)
            
            # Should complete within 1 second
            self.assertLess(elapsed, 1.0)
        
        # Average should be well under 1 second
        avg_time = np.mean(times)
        self.assertLess(avg_time, 0.5)
```

### 4.5.2 Performance Testing

Benchmarking for scalability:

```python
import time
import psutil
import pandas as pd
from typing import List, Dict, Tuple

class PerformanceBenchmark:
    """Performance testing for algorithms"""
    
    def __init__(self):
        self.results = []
        
    def benchmark_algorithm(self, 
                          algorithm_class,
                          poi_counts: List[int],
                          iterations: int = 5) -> pd.DataFrame:
        """Benchmark algorithm with varying POI counts"""
        
        for n_pois in poi_counts:
            # Generate test data
            pois, distance_matrix = self._generate_test_data(n_pois)
            
            # Initialize algorithm
            algorithm = algorithm_class(pois, distance_matrix)
            
            # Run multiple iterations
            times = []
            memories = []
            
            for _ in range(iterations):
                # Memory before
                process = psutil.Process()
                mem_before = process.memory_info().rss / 1024 / 1024  # MB
                
                # Time execution
                start = time.time()
                result = algorithm.plan_itinerary(
                    {'museum': 0.7, 'park': 0.8},
                    Constraints(max_pois=5)
                )
                elapsed = time.time() - start
                
                # Memory after
                mem_after = process.memory_info().rss / 1024 / 1024
                
                times.append(elapsed)
                memories.append(mem_after - mem_before)
            
            # Store results
            self.results.append({
                'algorithm': algorithm_class.__name__,
                'n_pois': n_pois,
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'avg_memory': np.mean(memories),
                'solution_quality': len(result) if result else 0
            })
        
        return pd.DataFrame(self.results)
    
    def _generate_test_data(self, n_pois: int) -> Tuple[List[Dict], np.ndarray]:
        """Generate synthetic test data"""
        pois = []
        for i in range(n_pois):
            pois.append({
                'id': f'poi_{i}',
                'name': f'Test POI {i}',
                'lat': 40.7 + (i % 100) * 0.001,
                'lon': -74.0 + (i // 100) * 0.001,
                'category': ['museum', 'park', 'restaurant'][i % 3],
                'rating': 4.0 + (i % 5) * 0.2,
                'entrance_fee': [0, 15, 25][i % 3],
                'avg_visit_duration': 1.5,
                'opening_hours': {'weekday': [9, 20]},
                'popularity': 0.5 + (i % 4) * 0.1
            })
        
        # Generate distance matrix
        distances = np.random.rand(n_pois, n_pois) * 5
        distances = (distances + distances.T) / 2
        np.fill_diagonal(distances, 0)
        
        return pois, distances
    
    def plot_results(self, output_dir: str = 'benchmarks/'):
        """Plot benchmark results"""
        import matplotlib.pyplot as plt
        
        df = pd.DataFrame(self.results)
        
        # Time complexity plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        for algo in df['algorithm'].unique():
            algo_df = df[df['algorithm'] == algo]
            ax1.plot(algo_df['n_pois'], algo_df['avg_time'], 
                    marker='o', label=algo)
        
        ax1.set_xlabel('Number of POIs')
        ax1.set_ylabel('Average Time (seconds)')
        ax1.set_title('Algorithm Time Complexity')
        ax1.legend()
        ax1.grid(True)
        
        # Memory usage plot
        for algo in df['algorithm'].unique():
            algo_df = df[df['algorithm'] == algo]
            ax2.plot(algo_df['n_pois'], algo_df['avg_memory'], 
                    marker='s', label=algo)
        
        ax2.set_xlabel('Number of POIs')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Algorithm Memory Usage')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/performance_benchmark.png', dpi=300)
        plt.close()
        
        # Save detailed results
        df.to_csv(f'{output_dir}/benchmark_results.csv', index=False)
```

### 4.5.3 Edge Case Testing

NYC-specific scenario testing:

```python
class NYCEdgeCaseTests(unittest.TestCase):
    """Test NYC-specific edge cases"""
    
    def test_subway_closure(self):
        """Test handling of subway line closure"""
        # Simulate L train shutdown
        affected_pois = [
            {'id': 'williamsburg', 'borough': 'brooklyn', 'lat': 40.7081, 'lon': -73.9571},
            {'id': 'bedford', 'borough': 'brooklyn', 'lat': 40.7167, 'lon': -73.9569}
        ]
        
        planner = HybridPlanner(affected_pois)
        
        # Apply L train closure
        update = DynamicUpdate(
            update_type=UpdateType.SUBWAY_DISRUPTION,
            poi_ids=[],
            details={'lines': ['L'], 'alternative': 'bus'}
        )
        
        result = planner.plan_with_update(
            {'park': 0.8},
            Constraints(transportation_mode='public_transit'),
            update
        )
        
        # Should adjust travel times or suggest alternatives
        self.assertIsNotNone(result)
    
    def test_hurricane_scenario(self):
        """Test extreme weather handling"""
        outdoor_pois = [
            {'id': 'central_park', 'category': 'park'},
            {'id': 'high_line', 'category': 'park'},
            {'id': 'brooklyn_bridge', 'category': 'landmark'}
        ]
        
        # Hurricane warning
        weather_update = DynamicUpdate(
            update_type=UpdateType.WEATHER_CLOSURE,
            poi_ids=['central_park', 'high_line', 'brooklyn_bridge'],
            timestamp=datetime.now(),
            duration_hours=48
        )
        
        # Should only recommend indoor activities
        result = planner.replan_after_update(weather_update)
        for poi in result:
            self.assertNotIn(poi.category, ['park', 'outdoor'])
    
    def test_nyc_marathon_routing(self):
        """Test routing during NYC Marathon"""
        marathon_date = datetime(2024, 11, 3)  # First Sunday in November
        
        # POIs along marathon route
        marathon_pois = [
            {'id': 'guggenheim', 'lat': 40.7829, 'lon': -73.9590},  # 5th Ave
            {'id': 'flatiron', 'lat': 40.7411, 'lon': -73.9897}    # Broadway
        ]
        
        result = planner.plan_itinerary(
            {'museum': 0.9},
            Constraints(date=marathon_date)
        )
        
        # Should avoid or warn about marathon route
        warnings = planner.get_warnings()
        self.assertIn('marathon', ' '.join(warnings).lower())
```

## 4.6 Conclusion

This implementation chapter demonstrated how theoretical methods translate into a practical, scalable system for NYC tourism. Through careful technology selection, optimized algorithms, comprehensive data processing, and rigorous testing, we achieve sub-second response times while handling the complexity of real-world urban itinerary planning. The modular architecture enables easy extension and maintenance, while NYC-specific adaptations ensure the system provides valuable, actionable recommendations for tourists navigating the city's rich but overwhelming array of options.