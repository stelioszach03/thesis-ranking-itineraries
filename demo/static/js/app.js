/**
 * NYC Itinerary Ranking Demo - Main Application JavaScript
 * Bachelor's Thesis - NKUA
 */

// Global variables
let currentItinerary = null;
let currentSession = null;
let map = null;
let markers = [];
let routeLayer = null;

// Initialize on document ready
$(document).ready(function() {
    initializeApp();
});

function initializeApp() {
    // Initialize map if on main page
    if ($('#map').length) {
        initializeMap();
    }
    
    // Initialize preference sliders
    initializeSliders();
    
    // Initialize form handlers
    initializeFormHandlers();
    
    // Load initial POIs
    loadPOIs();
}

function initializeMap() {
    // Create map centered on Manhattan
    map = L.map('map').setView([40.7614, -73.9776], 13);
    
    // Add OpenStreetMap tiles
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);
    
    // Add scale control
    L.control.scale().addTo(map);
}

function initializeSliders() {
    // Update value displays when sliders change
    $('.form-range').on('input', function() {
        const sliderId = $(this).attr('id');
        const value = $(this).val();
        const displayId = sliderId.replace('Slider', 'Value');
        $('#' + displayId).text(value);
    });
}

function initializeFormHandlers() {
    // Algorithm selection handler
    $('#algorithmSelect').on('change', function() {
        const algorithm = $(this).val();
        updateAlgorithmInfo(algorithm);
    });
    
    // Time input handler
    $('#startTimeInput').on('change', function() {
        validateTimeInput();
    });
    
    // Budget input handler
    $('#budgetInput').on('change', function() {
        validateBudgetInput();
    });
}

function updateAlgorithmInfo(algorithm) {
    const infoText = {
        'hybrid': 'Achieves 96% quality at 1.5% computational cost',
        'greedy': 'Fast O(n²) heuristic selection',
        'heap_greedy': 'Optimized greedy with heap pruning',
        'astar': 'Optimal solution with A* search',
        'lpa_star': 'Dynamic replanning with 70-90% reuse'
    };
    
    $('.algorithm-info').text(infoText[algorithm] || '');
}

function validateTimeInput() {
    const startTime = $('#startTimeInput').val();
    const duration = parseInt($('#durationInput').val());
    
    if (startTime) {
        const [hours, minutes] = startTime.split(':').map(Number);
        const endHour = hours + duration;
        
        if (endHour > 22) {
            showAlert('warning', 'Late finish time. Some attractions may be closed.');
        }
    }
}

function validateBudgetInput() {
    const budget = parseInt($('#budgetInput').val());
    
    if (budget < 50) {
        showAlert('info', 'Low budget may limit paid attractions.');
    }
}

function loadPOIs() {
    $.get('/api/pois', function(data) {
        if (data.success) {
            displayPOIsOnMap(data.pois);
        }
    }).fail(function() {
        showAlert('danger', 'Failed to load POI data');
    });
}

function displayPOIsOnMap(pois) {
    // Clear existing markers
    markers.forEach(marker => map.removeLayer(marker));
    markers = [];
    
    // Add POI markers with clustering for performance
    const markerClusterGroup = L.markerClusterGroup();
    
    pois.forEach(poi => {
        const icon = getCategoryIcon(poi.category);
        const marker = L.marker([poi.lat, poi.lon], { icon: icon })
            .bindPopup(createPOIPopup(poi));
        
        markerClusterGroup.addLayer(marker);
        markers.push(marker);
    });
    
    map.addLayer(markerClusterGroup);
}

function getCategoryIcon(category) {
    const iconColors = {
        'museum': 'purple',
        'park': 'green',
        'restaurant': 'orange',
        'landmark': 'blue',
        'shopping': 'red',
        'entertainment': 'darkred',
        'cultural': 'cadetblue',
        'market': 'darkgreen'
    };
    
    return L.AwesomeMarkers.icon({
        icon: getCategoryIconName(category),
        markerColor: iconColors[category] || 'gray',
        prefix: 'fa'
    });
}

function getCategoryIconName(category) {
    const icons = {
        'museum': 'university',
        'park': 'tree',
        'restaurant': 'utensils',
        'landmark': 'monument',
        'shopping': 'shopping-bag',
        'entertainment': 'theater-masks',
        'cultural': 'palette',
        'market': 'store'
    };
    
    return icons[category] || 'map-marker';
}

function createPOIPopup(poi) {
    return `
        <div class="poi-popup">
            <h6>${poi.name}</h6>
            <p class="mb-1"><strong>Category:</strong> ${poi.category}</p>
            <p class="mb-1"><strong>Rating:</strong> ${poi.rating}/5</p>
            <p class="mb-1"><strong>Duration:</strong> ${poi.avg_visit_duration}h</p>
            ${poi.entrance_fee > 0 ? `<p class="mb-1"><strong>Fee:</strong> $${poi.entrance_fee}</p>` : ''}
            <button class="btn btn-sm btn-primary mt-2" onclick="addPOIToItinerary('${poi.id}')">
                Add to Itinerary
            </button>
        </div>
    `;
}

// Main planning function
function planItinerary() {
    // Show loading state
    showLoading(true);
    
    // Collect preferences
    const preferences = {
        museum: parseFloat($('#museumSlider').val()),
        park: parseFloat($('#parkSlider').val()),
        market: parseFloat($('#foodSlider').val()),
        landmark: parseFloat($('#landmarkSlider').val())
    };
    
    // Collect constraints
    const constraints = {
        budget: parseInt($('#budgetInput').val()),
        max_time_hours: parseInt($('#durationInput').val()),
        start_time: parseFloat($('#startTimeInput').val().split(':')[0]),
        min_pois: 3,
        max_pois: 7,
        transportation_mode: 'public_transit'
    };
    
    // Get selected algorithm
    const algorithm = $('#algorithmSelect').val();
    
    // Prepare request data
    const requestData = {
        algorithm: algorithm,
        preferences: preferences,
        constraints: constraints
    };
    
    // Make API request
    $.ajax({
        url: '/api/plan',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(requestData),
        success: function(response) {
            showLoading(false);
            
            if (response.success) {
                currentItinerary = response.itinerary;
                currentSession = response.session_id;
                displayResults(response);
            } else {
                showAlert('danger', response.error || 'Failed to generate itinerary');
            }
        },
        error: function(xhr) {
            showLoading(false);
            showAlert('danger', 'Error communicating with server');
        }
    });
}

function displayResults(response) {
    // Update metrics
    $('#cssScore').text(response.metrics.css.toFixed(3));
    $('#runtime').text(response.runtime.toFixed(0) + 'ms');
    $('#poiCount').text(response.itinerary.pois.length);
    $('#totalCost').text('$' + response.itinerary.total_cost.toFixed(0));
    
    // Update component scores
    $('#satScore').text(response.metrics.satisfaction.toFixed(3));
    $('#turScore').text(response.metrics.time_utilization.toFixed(3));
    $('#feaScore').text(response.metrics.feasibility.toFixed(3));
    $('#divScore').text(response.metrics.diversity.toFixed(3));
    $('#vendiScore').text(response.metrics.vendi_score.toFixed(2));
    
    // Display itinerary timeline
    displayItinerary(response.itinerary);
    
    // Draw route on map
    drawRoute(response.itinerary.pois);
    
    // Update metrics chart
    updateMetricsChart(response.metrics);
}

function displayItinerary(itinerary) {
    const timeline = $('#itineraryTimeline');
    timeline.empty();
    
    itinerary.pois.forEach((poi, index) => {
        const item = $(`
            <div class="poi-item" data-poi-id="${poi.id}">
                <h6>${poi.arrival_time} - ${poi.name}</h6>
                <p class="text-muted mb-1">${poi.category} • ${poi.duration} hours</p>
                <button class="btn btn-sm btn-outline-success" onclick="likePOI('${poi.id}')">
                    <i class="fas fa-thumbs-up"></i> Like
                </button>
                <button class="btn btn-sm btn-outline-danger" onclick="dislikePOI('${poi.id}')">
                    <i class="fas fa-thumbs-down"></i> Dislike
                </button>
            </div>
        `);
        timeline.append(item);
    });
}

function drawRoute(pois) {
    // Clear existing route
    if (routeLayer) {
        map.removeLayer(routeLayer);
    }
    
    // Create route coordinates
    const coordinates = pois.map(poi => [poi.lat, poi.lon]);
    
    // Draw polyline
    routeLayer = L.polyline(coordinates, {
        color: 'blue',
        weight: 4,
        opacity: 0.7
    }).addTo(map);
    
    // Add numbered markers
    pois.forEach((poi, index) => {
        const marker = L.marker([poi.lat, poi.lon], {
            icon: L.divIcon({
                html: `<div class="route-number">${index + 1}</div>`,
                className: 'route-marker',
                iconSize: [30, 30]
            })
        }).bindPopup(`<b>${poi.name}</b><br>${poi.category}`).addTo(map);
    });
    
    // Fit map to route
    map.fitBounds(routeLayer.getBounds());
}

function updateMetricsChart(metrics) {
    const ctx = document.getElementById('metricsChart');
    if (!ctx) return;
    
    new Chart(ctx.getContext('2d'), {
        type: 'radar',
        data: {
            labels: ['Satisfaction', 'Time Util.', 'Feasibility', 'Diversity'],
            datasets: [{
                label: 'Quality Metrics',
                data: [
                    metrics.satisfaction,
                    metrics.time_utilization,
                    metrics.feasibility,
                    metrics.diversity
                ],
                fill: true,
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgb(54, 162, 235)',
                pointBackgroundColor: 'rgb(54, 162, 235)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgb(54, 162, 235)'
            }]
        },
        options: {
            elements: {
                line: {
                    borderWidth: 3
                }
            },
            scales: {
                r: {
                    angleLines: {
                        display: false
                    },
                    suggestedMin: 0,
                    suggestedMax: 1
                }
            }
        }
    });
}

// Dynamic update functions
function simulateSubwayDisruption() {
    if (!currentItinerary) {
        showAlert('warning', 'Please generate an itinerary first!');
        return;
    }
    
    showUpdateNotification('N/Q/R/W lines suspended - replanning with LPA*...');
    
    $.post('/api/update', {
        type: 'subway_disruption',
        session_id: currentSession
    }).done(function(response) {
        if (response.success) {
            currentItinerary = response.new_itinerary;
            displayResults({
                itinerary: response.new_itinerary,
                metrics: response.metrics,
                runtime: response.runtime
            });
            
            hideUpdateNotification();
            showAlert('success', `Replanning complete! ${response.computation_reuse}% computation reused.`);
        }
    });
}

function simulateWeatherChange() {
    if (!currentItinerary) {
        showAlert('warning', 'Please generate an itinerary first!');
        return;
    }
    
    showUpdateNotification('Rain detected - switching to indoor attractions...');
    
    $.post('/api/update', {
        type: 'weather_rain',
        session_id: currentSession
    }).done(function(response) {
        if (response.success) {
            currentItinerary = response.new_itinerary;
            displayResults({
                itinerary: response.new_itinerary,
                metrics: response.metrics,
                runtime: response.runtime
            });
            
            hideUpdateNotification();
        }
    });
}

// User preference learning
function likePOI(poiId) {
    updatePreferences([poiId], []);
}

function dislikePOI(poiId) {
    updatePreferences([], [poiId]);
}

function updatePreferences(liked, disliked) {
    $.post('/api/learn_preferences', {
        liked: liked,
        disliked: disliked,
        session_id: currentSession
    }).done(function(response) {
        if (response.success) {
            console.log('Preferences updated:', response.updated_preferences);
            showAlert('info', 'Preferences learned for future recommendations');
        }
    });
}

// Export functionality
function exportToGoogleMaps() {
    if (!currentItinerary) {
        showAlert('warning', 'Please generate an itinerary first!');
        return;
    }
    
    const waypoints = currentItinerary.pois.map(poi => `${poi.lat},${poi.lon}`).join('/');
    const googleMapsUrl = `https://www.google.com/maps/dir/${waypoints}`;
    window.open(googleMapsUrl, '_blank');
}

// Utility functions
function showLoading(show) {
    if (show) {
        $('#loadingOverlay').show();
    } else {
        $('#loadingOverlay').hide();
    }
}

function showAlert(type, message) {
    const alert = $(`
        <div class="alert alert-${type} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `);
    
    $('#alertContainer').append(alert);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => alert.alert('close'), 5000);
}

function showUpdateNotification(message) {
    $('#updateMessage').text(message);
    $('#updateNotification').show();
    animateProgress();
}

function hideUpdateNotification() {
    setTimeout(() => $('#updateNotification').fadeOut(), 2000);
}

function animateProgress() {
    let width = 0;
    const interval = setInterval(() => {
        width += 10;
        $('#updateProgress').css('width', width + '%');
        if (width >= 100) {
            clearInterval(interval);
        }
    }, 100);
}

// Add POI to itinerary (for interactive map)
function addPOIToItinerary(poiId) {
    $.post('/api/add_poi', {
        poi_id: poiId,
        session_id: currentSession
    }).done(function(response) {
        if (response.success) {
            showAlert('success', 'POI added to itinerary');
            planItinerary(); // Replan with new POI
        }
    });
}