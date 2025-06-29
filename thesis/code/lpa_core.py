class LPAStarPlanner:
    """Lifelong Planning A* for dynamic itinerary updates"""
    
    def __init__(self, pois, distance_matrix):
        self.pois = pois
        self.distance_matrix = distance_matrix
        self.g_values = {}  # Current cost-to-come
        self.rhs_values = {}  # One-step lookahead values
        self.queue = PriorityQueue()
        self.km = 0  # Key modifier for consistent ordering
        
    def calculate_key(self, state):
        """Calculate priority key for state"""
        min_g_rhs = min(self.g_values.get(state, float('inf')),
                       self.rhs_values.get(state, float('inf')))
        return (min_g_rhs + self.heuristic(state) + self.km, min_g_rhs)
    
    def update_vertex(self, state):
        """Update vertex and recompute rhs value"""
        if state != self.start_state:
            # Recompute rhs as minimum over predecessors
            self.rhs_values[state] = float('inf')
            
            for pred in self.get_predecessors(state):
                cost = self.g_values.get(pred, float('inf')) + \
                       self.transition_cost(pred, state)
                if cost < self.rhs_values[state]:
                    self.rhs_values[state] = cost
        
        # Remove from queue if present
        self.queue.remove(state)
        
        # Re-insert if inconsistent
        if self.g_values.get(state, float('inf')) != \
           self.rhs_values.get(state, float('inf')):
            self.queue.insert(state, self.calculate_key(state))
    
    def compute_shortest_path(self):
        """Main LPA* planning loop"""
        while (self.queue.top_key() < self.calculate_key(self.goal_state) or
               self.rhs_values.get(self.goal_state, float('inf')) != 
               self.g_values.get(self.goal_state, float('inf'))):
            
            k_old = self.queue.top_key()
            state = self.queue.pop()
            
            if k_old < self.calculate_key(state):
                # Key increased, re-insert
                self.queue.insert(state, self.calculate_key(state))
                
            elif self.g_values.get(state, float('inf')) > \
                 self.rhs_values.get(state, float('inf')):
                # Overconsistent: update g value
                self.g_values[state] = self.rhs_values[state]
                
                # Update all successors
                for succ in self.get_successors(state):
                    self.update_vertex(succ)
                    
            else:
                # Underconsistent: set g to infinity
                self.g_values[state] = float('inf')
                
                # Update state and all successors
                self.update_vertex(state)
                for succ in self.get_successors(state):
                    self.update_vertex(succ)
    
    def handle_dynamic_update(self, update):
        """Handle dynamic changes efficiently"""
        affected_states = []
        
        if update.type == UpdateType.POI_CLOSED:
            # Find all states containing closed POI
            for state in self.all_states:
                if update.closed_poi in state.visited_pois:
                    affected_states.append(state)
                    
        elif update.type == UpdateType.TRAFFIC:
            # Find states affected by traffic changes
            for state in self.all_states:
                if self.uses_affected_route(state, update.affected_routes):
                    affected_states.append(state)
        
        # Update affected states
        for state in affected_states:
            self.update_vertex(state)
        
        # Recompute shortest path
        self.compute_shortest_path()
        
        # Calculate computation reuse
        total_states = len(self.all_states)
        recomputed_states = len(affected_states)
        reuse_percentage = (1 - recomputed_states / total_states) * 100
        
        return self.extract_path(), reuse_percentage
    
    def extract_path(self):
        """Extract optimal itinerary from goal state"""
        if self.g_values.get(self.goal_state, float('inf')) == float('inf'):
            return []  # No path exists
            
        # Reconstruct path by following optimal predecessors
        path = []
        current = self.goal_state
        
        while current != self.start_state:
            path.append(current.current_poi)
            
            # Find best predecessor
            best_pred = None
            best_cost = float('inf')
            
            for pred in self.get_predecessors(current):
                cost = self.g_values.get(pred, float('inf')) + \
                       self.transition_cost(pred, current)
                if cost < best_cost:
                    best_cost = cost
                    best_pred = pred
            
            current = best_pred
        
        path.reverse()
        return path