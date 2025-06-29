"""
Tests for NYC Itinerary Demo
"""

import pytest
import json
from demo_nyc import app


@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_index_page(client):
    """Test main page loads"""
    response = client.get('/')
    assert response.status_code == 200


def test_get_pois(client):
    """Test POI endpoint"""
    response = client.get('/api/pois')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'pois' in data
    assert 'categories' in data
    assert len(data['pois']) > 0


def test_plan_itinerary_valid(client):
    """Test valid itinerary planning"""
    request_data = {
        'algorithm': 'greedy',
        'preferences': {
            'museum': 0.8,
            'park': 0.6,
            'restaurant': 0.7
        },
        'budget': 200,
        'duration': 8,
        'start_time': 9.0
    }
    
    response = client.post('/api/plan',
                          json=request_data,
                          content_type='application/json')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['success'] is True
    assert 'itinerary' in data
    assert 'metrics' in data
    assert 'runtime' in data


def test_plan_itinerary_invalid_algorithm(client):
    """Test invalid algorithm handling"""
    request_data = {
        'algorithm': 'invalid_algo',
        'preferences': {},
        'budget': 200
    }
    
    response = client.post('/api/plan',
                          json=request_data,
                          content_type='application/json')
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['success'] is False
    assert 'error' in data


def test_plan_itinerary_invalid_constraints(client):
    """Test invalid constraints handling"""
    request_data = {
        'algorithm': 'greedy',
        'preferences': {},
        'budget': -100  # Invalid negative budget
    }
    
    response = client.post('/api/plan',
                          json=request_data,
                          content_type='application/json')
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['success'] is False
    assert 'Budget must be positive' in data['error']


def test_dynamic_update_no_session(client):
    """Test dynamic update without active itinerary"""
    request_data = {
        'type': 'subway_disruption'
    }
    
    response = client.post('/api/update',
                          json=request_data,
                          content_type='application/json')
    
    assert response.status_code == 404
    data = json.loads(response.data)
    assert data['success'] is False
    assert 'No active itinerary' in data['error']


def test_learn_preferences(client):
    """Test preference learning endpoint"""
    request_data = {
        'liked': ['poi_1', 'poi_2'],
        'disliked': ['poi_3']
    }
    
    response = client.post('/api/learn_preferences',
                          json=request_data,
                          content_type='application/json')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['success'] is True
    assert 'updated_preferences' in data


def test_export_google_maps(client):
    """Test Google Maps export"""
    request_data = {
        'itinerary': ['poi_1', 'poi_2', 'poi_3']
    }
    
    response = client.post('/api/export_google_maps',
                          json=request_data,
                          content_type='application/json')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['success'] is True
    assert 'maps_url' in data
    assert 'google.com/maps/dir' in data['maps_url']


def test_error_handlers(client):
    """Test error handling"""
    # Test 404
    response = client.get('/nonexistent')
    assert response.status_code == 404
    data = json.loads(response.data)
    assert data['success'] is False
    
    # Test 405
    response = client.delete('/api/plan')
    assert response.status_code == 405
    data = json.loads(response.data)
    assert data['success'] is False


def test_content_type_validation(client):
    """Test content type validation"""
    response = client.post('/api/plan',
                          data='invalid',
                          content_type='text/plain')
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'Content-Type must be application/json' in data['error']