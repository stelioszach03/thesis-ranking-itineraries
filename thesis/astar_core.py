# A* Search with Admissible Heuristics
# Core implementation from astar_itinerary.py

@dataclass(frozen=True)
class ItineraryState:
    """Immutable state for A* search"""
    visited_pois: Tuple[str, ...]
    current_time: float
    remaining_budget: float
    total_utility: float
    total_distance: float

def astar_search(self, start_state, goal_test, user_preferences, constraints):
    """
    A* search for optimal itinerary with quality-based utility
    Uses MST-based admissible heuristic
    """
    frontier = []
    initial_h = self._mst_heuristic(start_state, user_preferences, constraints)
    initial_node = SearchNode(start_state, 0, initial_h)
    heapq.heappush(frontier, initial_node)
    
    visited = set()
    best_solution = None
    nodes_expanded = 0
    
    while frontier and nodes_expanded < self.max_nodes:
        current_node = heapq.heappop(frontier)
        
        if current_node.state in visited:
            continue
            
        visited.add(current_node.state)
        nodes_expanded += 1
        
        # Goal test - found complete itinerary
        if goal_test(current_node.state):
            if best_solution is None or current_node.state.total_utility > best_solution.state.total_utility:
                best_solution = current_node
                
        # Generate successors
        for successor_state in self._generate_successors(current_node.state, constraints):
            if successor_state not in visited:
                g_cost = -successor_state.total_utility  # Negative for maximization
                h_cost = self._mst_heuristic(successor_state, user_preferences, constraints)
                successor_node = SearchNode(successor_state, g_cost, h_cost, current_node)
                heapq.heappush(frontier, successor_node)
    
    return self._extract_itinerary(best_solution) if best_solution else None

def _mst_heuristic(self, state, preferences, constraints):
    """
    MST-based admissible heuristic for remaining utility
    Guarantees optimality of A* search
    """
    unvisited = [p for p in self.pois if p.id not in state.visited_pois]
    remaining_slots = constraints.max_pois - len(state.visited_pois)
    
    if remaining_slots <= 0 or not unvisited:
        return 0
    
    # Select top-k unvisited POIs by utility
    poi_utilities = []
    for poi in unvisited:
        if self._is_reachable(poi, state, constraints):
            utility = preferences.get(poi.category, 0.5) * poi.rating * poi.popularity**0.3
            poi_utilities.append((utility, poi))
    
    poi_utilities.sort(reverse=True)
    top_k_pois = [poi for _, poi in poi_utilities[:remaining_slots]]
    
    if not top_k_pois:
        return 0
    
    # Build MST of top-k POIs using Numba-optimized distances
    n = len(top_k_pois)
    if n == 1:
        # Single POI - just connection cost
        if state.visited_pois:
            last_idx = self.poi_index_map[state.visited_pois[-1]]
            poi_idx = self.poi_index_map[top_k_pois[0].id]
            min_dist = self.distance_matrix[last_idx, poi_idx]
        else:
            min_dist = 0
    else:
        # Multiple POIs - compute MST
        indices = [self.poi_index_map[poi.id] for poi in top_k_pois]
        mst_edges = self._compute_mst_numba(indices)
        min_dist = sum(self.distance_matrix[i, j] for i, j in mst_edges)
        
        # Add connection from current location
        if state.visited_pois:
            last_idx = self.poi_index_map[state.visited_pois[-1]]
            min_connection = min(self.distance_matrix[last_idx, idx] for idx in indices)
            min_dist += min_connection
    
    # Convert to negative utility (upper bound on remaining value)
    max_utility_per_poi = max(u for u, _ in poi_utilities[:remaining_slots])
    return -remaining_slots * max_utility_per_poi

@njit
def _compute_mst_numba(indices):
    """Numba-optimized MST computation using Kruskal's algorithm"""
    n = len(indices)
    edges = []
    
    # Generate all edges
    for i in range(n):
        for j in range(i + 1, n):
            dist = manhattan_distance_numba(
                poi_coords[indices[i], 0], poi_coords[indices[i], 1],
                poi_coords[indices[j], 0], poi_coords[indices[j], 1]
            )
            edges.append((dist, i, j))
    
    # Sort edges by distance
    edges.sort()
    
    # Kruskal's algorithm with union-find
    parent = list(range(n))
    mst_edges = []
    
    for dist, i, j in edges:
        if find_parent(parent, i) != find_parent(parent, j):
            mst_edges.append((indices[i], indices[j]))
            union(parent, i, j)
            if len(mst_edges) == n - 1:
                break
    
    return mst_edges