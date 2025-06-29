# Quality-aware Greedy POI Selection
# Core implementation from greedy_algorithms.py

def select_pois(self, user_preferences, constraints, feedback=None):
    """
    Select POIs using quality-aware greedy algorithm
    Based on Basu Roy et al. with quality enhancements
    """
    selected_pois = []
    current_time = constraints.start_time
    current_budget = constraints.budget
    
    # Apply feedback if provided
    candidate_pois = self._filter_by_feedback(self.pois, feedback)
    
    while len(selected_pois) < constraints.max_pois:
        best_poi = None
        best_ratio = -float('inf')
        
        for poi in candidate_pois:
            if poi.id in [p.id for p in selected_pois]:
                continue
                
            # Check feasibility
            if not self._is_feasible(poi, selected_pois, current_time, 
                                   current_budget, constraints):
                continue
            
            # Calculate quality-aware marginal utility
            utility = self._calculate_marginal_utility(
                poi, selected_pois, user_preferences, constraints
            )
            cost = self._calculate_marginal_cost(
                poi, selected_pois, current_time
            )
            
            ratio = utility / cost if cost > 0 else float('inf')
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_poi = poi
        
        # Termination conditions
        if best_poi is None:
            break
        if len(selected_pois) >= constraints.min_pois and best_ratio < 0.1:
            break
            
        # Add POI and update state
        selected_pois.append(best_poi)
        travel_time = self._get_travel_time(selected_pois[-2] if len(selected_pois) > 1 else None, best_poi)
        current_time += travel_time + best_poi.avg_visit_duration
        current_budget -= best_poi.entrance_fee
    
    return Itinerary(
        pois=selected_pois,
        total_distance=self._calculate_total_distance(selected_pois),
        total_time=current_time - constraints.start_time,
        total_cost=constraints.budget - current_budget,
        user_preferences=user_preferences
    )

def _calculate_marginal_utility(self, poi, selected, preferences, constraints):
    """Calculate quality-aware marginal utility with CSS components"""
    # Base preference score (SAT component)
    base_utility = preferences.get(poi.category, 0.5) * poi.rating
    
    # Diversity bonus (DIV component)
    categories_covered = set(p.category for p in selected)
    diversity_bonus = 1.2 if poi.category not in categories_covered else 0.7
    
    # Proximity bonus for walkable POIs
    if selected:
        last_poi = selected[-1]
        dist = self._get_distance(last_poi, poi)
        proximity_bonus = 1.1 if dist < 0.5 else 1.0  # <500m walking
    else:
        proximity_bonus = 1.0
    
    # Popularity factor with diminishing returns
    popularity_factor = poi.popularity ** 0.3
    
    # NYC-specific adjustments
    if self._is_rush_hour(constraints.start_time) and poi.category == 'restaurant':
        time_penalty = 0.8  # Dining during rush hour
    else:
        time_penalty = 1.0
    
    return base_utility * diversity_bonus * proximity_bonus * popularity_factor * time_penalty