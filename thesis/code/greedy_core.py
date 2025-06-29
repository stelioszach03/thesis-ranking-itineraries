def quality_aware_greedy_selection(pois, preferences, constraints):
    """
    Quality-aware greedy POI selection algorithm
    Implements CSS-based marginal utility calculation
    """
    itinerary = []
    current_location = constraints.start_location
    time_remaining = constraints.max_time_hours * 60  # Convert to minutes
    budget_remaining = constraints.budget
    
    while time_remaining > 0 and len(itinerary) < constraints.max_pois:
        # Filter feasible POIs
        candidates = []
        for poi in pois:
            if poi in itinerary:
                continue
                
            # Check time feasibility
            travel_time = calculate_travel_time(current_location, poi.location)
            total_time = travel_time + poi.avg_visit_duration * 60
            if total_time > time_remaining:
                continue
                
            # Check budget feasibility
            if poi.entrance_fee > budget_remaining:
                continue
                
            candidates.append(poi)
        
        if not candidates:
            break
            
        # Select POI with maximum marginal CSS
        best_poi = None
        best_css = -1
        
        for poi in candidates:
            # Calculate marginal CSS if this POI is added
            temp_itinerary = itinerary + [poi]
            marginal_css = calculate_css(temp_itinerary, preferences)
            
            if marginal_css > best_css:
                best_css = marginal_css
                best_poi = poi
        
        # Add best POI to itinerary
        if best_poi:
            itinerary.append(best_poi)
            travel_time = calculate_travel_time(current_location, best_poi.location)
            time_remaining -= travel_time + best_poi.avg_visit_duration * 60
            budget_remaining -= best_poi.entrance_fee
            current_location = best_poi.location
    
    return itinerary

def calculate_css(itinerary, preferences):
    """Calculate Composite Satisfaction Score"""
    if not itinerary:
        return 0
        
    # Component weights from research
    w_attractiveness = 0.35
    w_time_efficiency = 0.25
    w_feasibility = 0.25
    w_diversity = 0.15
    
    # Calculate components
    attractiveness = calculate_attractiveness(itinerary, preferences)
    time_efficiency = calculate_time_efficiency(itinerary)
    feasibility = calculate_feasibility(itinerary)
    diversity = calculate_diversity(itinerary)
    
    # Weighted sum
    css = (w_attractiveness * attractiveness +
           w_time_efficiency * time_efficiency +
           w_feasibility * feasibility +
           w_diversity * diversity)
    
    return css