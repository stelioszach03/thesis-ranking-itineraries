"""
NYC Itinerary Ranking Demo
Showcases research contributions with interactive Manhattan map

Features:
- Algorithm selection (Greedy/A*/Hybrid)
- Real-time subway updates (LPA* demonstration)
- CSS ranking metrics display
- Preference learning interface
- Export to Google Maps
"""

from flask import Flask, render_template, request, jsonify, session
import json
import numpy as np
from datetime import datetime, timedelta
import folium
from folium import plugins
import os
import sys
from typing import List, Dict, Tuple
import random
import time
import logging

# Add parent directory to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.greedy_algorithms import GreedyPOISelection, HeapPrunGreedyPOI, Constraints
from src.astar_itinerary import AStarItineraryPlanner
from src.lpa_star import LPAStarPlanner, DynamicUpdate, UpdateType
from src.hybrid_planner import HybridItineraryPlanner
from src.metrics_definitions import CompositeUtilityFunctions

app = Flask(__name__)
app.secret_key = 'nyc-itinerary-demo-2025'

# Load NYC POI data
NYC_POI_DATA = None
DISTANCE_MATRIX = None
ALGORITHMS = {}

def load_nyc_data():
    """Load preprocessed NYC POI data"""
    global NYC_POI_DATA, DISTANCE_MATRIX, ALGORITHMS
    
    # In production, load from prepared files
    # For demo, create sample data
    NYC_POI_DATA = create_sample_nyc_pois()
    DISTANCE_MATRIX = create_sample_distance_matrix(len(NYC_POI_DATA))
    
    # Initialize algorithms
    ALGORITHMS['greedy'] = GreedyPOISelection(NYC_POI_DATA, DISTANCE_MATRIX)
    ALGORITHMS['heap_greedy'] = HeapPrunGreedyPOI(NYC_POI_DATA, DISTANCE_MATRIX)
    ALGORITHMS['astar'] = AStarItineraryPlanner(NYC_POI_DATA, DISTANCE_MATRIX)
    ALGORITHMS['lpa_star'] = LPAStarPlanner(NYC_POI_DATA, DISTANCE_MATRIX)
    ALGORITHMS['hybrid'] = HybridItineraryPlanner(NYC_POI_DATA, DISTANCE_MATRIX)

def create_sample_nyc_pois():
    """Create sample NYC POIs for demo"""
    pois = [
        # Manhattan landmarks
        {'id': 'moma', 'name': 'Museum of Modern Art', 'lat': 40.7614, 'lon': -73.9776,
         'category': 'museum', 'rating': 4.7, 'popularity': 0.95, 'entrance_fee': 25,
         'avg_visit_duration': 2.5, 'borough': 'Manhattan'},
        
        {'id': 'central_park', 'name': 'Central Park', 'lat': 40.7829, 'lon': -73.9654,
         'category': 'park', 'rating': 4.8, 'popularity': 0.98, 'entrance_fee': 0,
         'avg_visit_duration': 2.0, 'borough': 'Manhattan'},
        
        {'id': 'met', 'name': 'Metropolitan Museum', 'lat': 40.7794, 'lon': -73.9632,
         'category': 'museum', 'rating': 4.8, 'popularity': 0.97, 'entrance_fee': 25,
         'avg_visit_duration': 3.0, 'borough': 'Manhattan'},
        
        {'id': 'high_line', 'name': 'High Line', 'lat': 40.7480, 'lon': -74.0048,
         'category': 'park', 'rating': 4.6, 'popularity': 0.92, 'entrance_fee': 0,
         'avg_visit_duration': 1.5, 'borough': 'Manhattan'},
        
        {'id': 'chelsea_market', 'name': 'Chelsea Market', 'lat': 40.7424, 'lon': -74.0061,
         'category': 'market', 'rating': 4.5, 'popularity': 0.88, 'entrance_fee': 0,
         'avg_visit_duration': 1.5, 'borough': 'Manhattan'},
        
        {'id': 'brooklyn_bridge', 'name': 'Brooklyn Bridge', 'lat': 40.7061, 'lon': -73.9969,
         'category': 'landmark', 'rating': 4.7, 'popularity': 0.94, 'entrance_fee': 0,
         'avg_visit_duration': 1.0, 'borough': 'Manhattan/Brooklyn'},
        
        {'id': 'times_square', 'name': 'Times Square', 'lat': 40.7580, 'lon': -73.9855,
         'category': 'landmark', 'rating': 4.3, 'popularity': 0.99, 'entrance_fee': 0,
         'avg_visit_duration': 0.5, 'borough': 'Manhattan'},
        
        {'id': 'natural_history', 'name': 'Natural History Museum', 'lat': 40.7813, 'lon': -73.9740,
         'category': 'museum', 'rating': 4.7, 'popularity': 0.93, 'entrance_fee': 23,
         'avg_visit_duration': 2.5, 'borough': 'Manhattan'},
        
        # Add more POIs for realistic demo...
    ]
    
    # Add opening hours
    for poi in pois:
        if poi['category'] == 'museum':
            poi['opening_hours'] = {'weekday': [10.0, 18.0], 'weekend': [10.0, 19.0]}
        elif poi['category'] == 'park':
            poi['opening_hours'] = {'weekday': [6.0, 22.0], 'weekend': [6.0, 22.0]}
        else:
            poi['opening_hours'] = {'weekday': [9.0, 21.0], 'weekend': [9.0, 22.0]}
    
    return pois

def create_sample_distance_matrix(n_pois):
    """Create sample distance matrix using Manhattan distance"""
    matrix = np.zeros((n_pois, n_pois))
    for i in range(n_pois):
        for j in range(i+1, n_pois):
            # Simplified Manhattan distance
            dist = abs(NYC_POI_DATA[i]['lat'] - NYC_POI_DATA[j]['lat']) * 111.0
            dist += abs(NYC_POI_DATA[i]['lon'] - NYC_POI_DATA[j]['lon']) * 111.0 * 0.75
            matrix[i][j] = matrix[j][i] = dist * 1.4  # NYC grid factor
    return matrix

@app.route('/')
def index():
    """Main demo interface"""
    return render_template('index.html')

@app.route('/api/pois')
def get_pois():
    """Get all POIs for map display"""
    return jsonify({
        'pois': NYC_POI_DATA,
        'categories': list(set(poi['category'] for poi in NYC_POI_DATA))
    })

@app.route('/api/plan', methods=['POST'])
def plan_itinerary():
    """Generate itinerary based on preferences"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
    
        # Extract and validate parameters
        algorithm = data.get('algorithm', 'hybrid')
        if algorithm not in ALGORITHMS:
            return jsonify({'success': False, 'error': f'Invalid algorithm: {algorithm}'}), 400
            
        preferences = data.get('preferences', {})
        
        # Validate constraints
        try:
            budget = float(data.get('budget', 200))
            if budget <= 0:
                return jsonify({'success': False, 'error': 'Budget must be positive'}), 400
                
            duration = float(data.get('duration', 8))
            if not 1 <= duration <= 24:
                return jsonify({'success': False, 'error': 'Duration must be between 1-24 hours'}), 400
                
            start_time = float(data.get('start_time', 9.0))
            if not 0 <= start_time <= 23:
                return jsonify({'success': False, 'error': 'Start time must be between 0-23'}), 400
                
            constraints = Constraints(
                budget=budget,
                max_time_hours=duration,
                min_pois=3,
                max_pois=7,  # Research-validated range
                start_time=start_time
            )
        except (ValueError, TypeError) as e:
            return jsonify({'success': False, 'error': f'Invalid constraint value: {str(e)}'}), 400
    
        # Run selected algorithm with error handling
        start_time = time.time()
        
        try:
            planner = ALGORITHMS[algorithm]
            itinerary = planner.select_pois(preferences, constraints)
            
            if not itinerary:
                return jsonify({
                    'success': False, 
                    'error': 'No feasible itinerary found with given constraints'
                }), 404
            
            # Calculate CSS metrics
            css_scores = CompositeUtilityFunctions.calculate_all_metrics(
                itinerary, preferences, constraints.budget, constraints.max_time_hours
            )
            
            runtime = (time.time() - start_time) * 1000  # milliseconds
            
            # Store in session for dynamic updates
            session['current_itinerary'] = [poi.id for poi in itinerary]
            session['algorithm'] = algorithm
            session['preferences'] = preferences
            session['constraints'] = constraints.__dict__
            
            return jsonify({
                'success': True,
                'itinerary': format_itinerary(itinerary),
                'metrics': {
                    'css': css_scores['css'],
                    'satisfaction': css_scores['satisfaction'],
                    'time_utilization': css_scores['time_utilization'],
                    'feasibility': css_scores['feasibility'],
                    'diversity': css_scores['diversity'],
                    'vendi_score': css_scores.get('vendi_score', 0)
                },
                'runtime': runtime,
                'algorithm': algorithm,
                'session_id': session.sid if hasattr(session, 'sid') else None
            })
            
        except Exception as e:
            app.logger.error(f"Algorithm error: {str(e)}", exc_info=True)
            return jsonify({
                'success': False,
                'error': f'Algorithm failed: {str(e)}'
            }), 500
            
    except Exception as e:
        app.logger.error(f"Request error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@app.route('/api/update', methods=['POST'])
def handle_dynamic_update():
    """Demonstrate LPA* dynamic replanning"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
            
        update_type = data.get('type')
        if not update_type:
            return jsonify({'success': False, 'error': 'Update type required'}), 400
        
        # Get current itinerary from session
        current_itinerary = session.get('current_itinerary')
        if not current_itinerary:
            return jsonify({'success': False, 'error': 'No active itinerary'}), 404
    
        # Create dynamic update
        try:
            if update_type == 'subway_disruption':
                update = DynamicUpdate(
                    update_type=UpdateType.SUBWAY_DISRUPTION,
                    poi_ids=[],
                    timestamp=datetime.now(),
                    details={'lines': ['N', 'Q', 'R', 'W']}
                )
            elif update_type == 'weather_rain':
                # Close outdoor attractions
                outdoor_pois = [p['id'] for p in NYC_POI_DATA if p['category'] in ['park', 'walking_tour']]
                update = DynamicUpdate(
                    update_type=UpdateType.WEATHER_CLOSURE,
                    poi_ids=outdoor_pois,
                    timestamp=datetime.now(),
                    duration_hours=4
                )
            elif update_type == 'poi_closed':
                poi_id = data.get('poi_id')
                if not poi_id:
                    return jsonify({'success': False, 'error': 'POI ID required for closure'}), 400
                update = DynamicUpdate(
                    update_type=UpdateType.POI_CLOSURE,
                    poi_ids=[poi_id],
                    timestamp=datetime.now()
                )
            else:
                return jsonify({'success': False, 'error': f'Unknown update type: {update_type}'}), 400
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to create update: {str(e)}'}), 500
    
        # Use LPA* for replanning
        start_time = time.time()
        
        try:
            lpa_planner = ALGORITHMS['lpa_star']
            
            # Initialize with current itinerary
            lpa_planner.initialize_with_path(current_itinerary)
            
            # Apply update and replan
            new_itinerary = lpa_planner.handle_dynamic_update(update)
            
            if not new_itinerary:
                return jsonify({
                    'success': False,
                    'error': 'No feasible itinerary after update'
                }), 404
                
            runtime = (time.time() - start_time) * 1000
            
            # Calculate metrics
            css_scores = CompositeUtilityFunctions.calculate_all_metrics(
                new_itinerary, session.get('preferences', {}), 
                session.get('constraints', {}).get('budget', 200), 
                session.get('constraints', {}).get('max_time_hours', 8)
            )
            
            # Update session
            session['current_itinerary'] = [poi.id for poi in new_itinerary]
            
            return jsonify({
                'success': True,
                'original_itinerary': current_itinerary,
                'new_itinerary': format_itinerary(new_itinerary),
                'metrics': css_scores,
                'runtime': runtime,
                'computation_reuse': lpa_planner.get_reuse_percentage(),
                'update_details': {
                    'type': update_type,
                    'affected_pois': len(update.poi_ids)
                }
            })
            
        except Exception as e:
            app.logger.error(f"Replanning error: {str(e)}", exc_info=True)
            return jsonify({
                'success': False,
                'error': f'Replanning failed: {str(e)}'
            }), 500
            
    except Exception as e:
        app.logger.error(f"Update handler error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@app.route('/api/learn_preferences', methods=['POST'])
def learn_preferences():
    """Update preferences based on user feedback"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
            
        liked_pois = data.get('liked', [])
        disliked_pois = data.get('disliked', [])
        
        if not liked_pois and not disliked_pois:
            return jsonify({'success': False, 'error': 'No feedback provided'}), 400
        
        # Simple preference learning
        preferences = session.get('preferences', {})
        
        # Increase weight for liked categories
        for poi_id in liked_pois:
            poi = next((p for p in NYC_POI_DATA if p['id'] == poi_id), None)
            if not poi:
                continue  # Skip invalid POI IDs
            category = poi['category']
            preferences[category] = min(1.0, preferences.get(category, 0.5) + 0.1)
        
        # Decrease weight for disliked categories
        for poi_id in disliked_pois:
            poi = next((p for p in NYC_POI_DATA if p['id'] == poi_id), None)
            if not poi:
                continue  # Skip invalid POI IDs
            category = poi['category']
            preferences[category] = max(0.0, preferences.get(category, 0.5) - 0.1)
        
        session['preferences'] = preferences
        
        return jsonify({
            'success': True,
            'updated_preferences': preferences,
            'feedback_processed': {
                'liked': len(liked_pois),
                'disliked': len(disliked_pois)
            }
        })
        
    except Exception as e:
        app.logger.error(f"Preference learning error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Failed to update preferences'
        }), 500

@app.route('/api/export_google_maps', methods=['POST'])
def export_to_google_maps():
    """Generate Google Maps URL for itinerary"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
            
        itinerary = data.get('itinerary', [])
        
        if not itinerary:
            return jsonify({'success': False, 'error': 'No itinerary provided'}), 400
        
        # Build Google Maps directions URL
        base_url = "https://www.google.com/maps/dir/"
        waypoints = []
        
        for poi_id in itinerary:
            poi = next((p for p in NYC_POI_DATA if p['id'] == poi_id), None)
            if not poi:
                return jsonify({
                    'success': False,
                    'error': f'Invalid POI ID: {poi_id}'
                }), 400
            waypoints.append(f"{poi['lat']},{poi['lon']}")
        
        if len(waypoints) > 10:
            return jsonify({
                'success': False,
                'error': 'Google Maps supports maximum 10 waypoints'
            }), 400
            
        maps_url = base_url + "/".join(waypoints)
        
        return jsonify({
            'success': True,
            'maps_url': maps_url,
            'waypoint_count': len(waypoints)
        })
        
    except Exception as e:
        app.logger.error(f"Export error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Failed to generate export'
        }), 500

def format_itinerary(itinerary):
    """Format itinerary for JSON response"""
    try:
        if not itinerary:
            return None
        
        # Handle both list and Itinerary object
        if isinstance(itinerary, list):
            # Convert list of POIs to proper format
            total_cost = sum(poi.get('entrance_fee', 0) for poi in itinerary)
            total_time = sum(poi.get('avg_visit_duration', 1) for poi in itinerary)
            formatted = {
                'pois': [],
                'total_time': total_time,
                'total_distance': 0,  # Will calculate below
                'total_cost': total_cost
            }
            pois_list = itinerary
        else:
            # Handle Itinerary object
            formatted = {
                'pois': [],
                'total_time': getattr(itinerary, 'total_time', 0),
                'total_distance': getattr(itinerary, 'total_distance', 0),
                'total_cost': getattr(itinerary, 'total_cost', 0)
            }
            pois_list = itinerary.pois
        
    formatted = {
        'pois': [],
        'total_time': itinerary.total_time,
        'total_distance': itinerary.total_distance,
        'total_cost': itinerary.total_cost
    }
    
    current_time = 9.0  # Start time
    for i, poi in enumerate(itinerary.pois):
        if i > 0:
            travel_time = DISTANCE_MATRIX[
                NYC_POI_DATA.index(itinerary.pois[i-1]),
                NYC_POI_DATA.index(poi)
            ] / 4.0  # Assume 4 km/h walking
            current_time += travel_time
        
        formatted['pois'].append({
            'id': poi['id'],
            'name': poi['name'],
            'arrival_time': f"{int(current_time)}:{int((current_time % 1) * 60):02d}",
            'duration': poi['avg_visit_duration'],
            'category': poi['category']
        })
        
        current_time += poi.get('avg_visit_duration', 1)
        
        # Update total distance
        if i > 0:
            formatted['total_distance'] += travel_time * 4.0  # km
    
        return formatted
        
    except Exception as e:
        app.logger.error(f"Format error: {str(e)}", exc_info=True)
        return None

def create_demo_map():
    """Create interactive Folium map of Manhattan"""
    # Center on Manhattan
    manhattan_map = folium.Map(
        location=[40.7614, -73.9776],
        zoom_start=13,
        tiles='OpenStreetMap'
    )
    
    # Add POI markers
    for poi in NYC_POI_DATA:
        icon_color = {
            'museum': 'red',
            'park': 'green',
            'restaurant': 'blue',
            'landmark': 'purple',
            'market': 'orange'
        }.get(poi['category'], 'gray')
        
        folium.Marker(
            location=[poi['lat'], poi['lon']],
            popup=f"{poi['name']}<br>{poi['category']}<br>Rating: {poi['rating']}",
            tooltip=poi['name'],
            icon=folium.Icon(color=icon_color, icon='info-sign')
        ).add_to(manhattan_map)
    
    return manhattan_map._repr_html_()

# Global error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(405)
def method_not_allowed_error(error):
    return jsonify({
        'success': False,
        'error': 'Method not allowed'
    }), 405

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Internal error: {str(error)}", exc_info=True)
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

@app.errorhandler(Exception)
def handle_exception(error):
    app.logger.error(f"Unhandled exception: {str(error)}", exc_info=True)
    return jsonify({
        'success': False,
        'error': 'An unexpected error occurred'
    }), 500

# Request validation
@app.before_request
def validate_json():
    """Validate JSON requests"""
    if request.method in ['POST', 'PUT'] and request.endpoint:
        if request.content_type != 'application/json':
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json'
            }), 400

# CORS headers for demo
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        load_nyc_data()
        app.logger.info("NYC data loaded successfully")
        app.run(debug=True, port=5000)
    except Exception as e:
        app.logger.error(f"Failed to start server: {str(e)}", exc_info=True)
        sys.exit(1)