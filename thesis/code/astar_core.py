class AStarItineraryPlanner:
    """A* implementation for optimal itinerary planning"""
    
    def plan_itinerary(self, pois, preferences, constraints):
        """Find optimal itinerary using A* search"""
        start_state = ItineraryState(
            current_poi=constraints.start_location,
            visited_pois=set(),
            time_used=0,
            budget_used=0
        )
        
        # Priority queue: (f_cost, state)
        open_set = [(0, start_state)]
        g_costs = {start_state: 0}
        came_from = {}
        
        while open_set:
            current_f, current_state = heapq.heappop(open_set)
            
            # Goal test: reached time/budget limit
            if self.is_goal_state(current_state, constraints):
                return self.reconstruct_path(came_from, current_state)
            
            # Generate successors
            for next_poi in pois:
                if next_poi in current_state.visited_pois:
                    continue
                    
                # Create successor state
                travel_time = self.calculate_travel_time(
                    current_state.current_poi, next_poi
                )
                next_state = ItineraryState(
                    current_poi=next_poi,
                    visited_pois=current_state.visited_pois | {next_poi},
                    time_used=current_state.time_used + travel_time + next_poi.duration,
                    budget_used=current_state.budget_used + next_poi.entrance_fee
                )
                
                # Check constraints
                if not self.is_feasible(next_state, constraints):
                    continue
                
                # Calculate costs
                g_cost = g_costs[current_state] + self.transition_cost(
                    current_state, next_state, preferences
                )
                
                if next_state not in g_costs or g_cost < g_costs[next_state]:
                    g_costs[next_state] = g_cost
                    h_cost = self.heuristic(next_state, pois, preferences)
                    f_cost = g_cost + h_cost
                    
                    heapq.heappush(open_set, (f_cost, next_state))
                    came_from[next_state] = current_state
        
        # No solution found
        return []
    
    def heuristic(self, state, all_pois, preferences):
        """Admissible heuristic: MST of unvisited POIs"""
        unvisited = [p for p in all_pois if p not in state.visited_pois]
        
        if not unvisited:
            return 0
            
        # Minimum spanning tree cost of unvisited POIs
        mst_cost = self.compute_mst_cost(unvisited)
        
        # Minimum connection cost from current location
        min_connection = min(
            self.calculate_travel_time(state.current_poi, poi)
            for poi in unvisited
        )
        
        # Admissible estimate
        return (mst_cost + min_connection) * self.MIN_UTILITY_PER_TIME
    
    def compute_mst_cost(self, pois):
        """Compute minimum spanning tree cost using Prim's algorithm"""
        if len(pois) < 2:
            return 0
            
        # Initialize with first POI
        in_tree = {pois[0]}
        edges = []
        
        # Add all edges from first POI
        for other in pois[1:]:
            cost = self.calculate_travel_time(pois[0], other)
            heapq.heappush(edges, (cost, pois[0], other))
        
        total_cost = 0
        
        while len(in_tree) < len(pois) and edges:
            cost, _, poi = heapq.heappop(edges)
            
            if poi in in_tree:
                continue
                
            in_tree.add(poi)
            total_cost += cost
            
            # Add new edges
            for other in pois:
                if other not in in_tree:
                    edge_cost = self.calculate_travel_time(poi, other)
                    heapq.heappush(edges, (edge_cost, poi, other))
        
        return total_cost