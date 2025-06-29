"""
Algorithm selection and result caching utilities
Bachelor's thesis: Ranking Itineraries
"""

from typing import Dict, Any, Optional
import time


class AlgorithmSelector:
    """Selects best algorithm based on problem characteristics"""
    
    def __init__(self):
        self.selection_history = []
        self.performance_stats = {}
    
    def select_algorithm(self, n_pois: int, constraints: Optional[Dict] = None) -> str:
        """Select optimal algorithm based on problem size and constraints"""
        if constraints is None:
            constraints = {}
            
        # Algorithm selection logic based on research
        if n_pois < 100:
            algorithm = "astar"  # Optimal for small problems
        elif n_pois < 1000:
            algorithm = "hybrid"  # Good balance for medium problems
        else:
            algorithm = "greedy"  # Fast for large problems
        
        # Consider time constraints
        max_time = constraints.get('max_time_hours', 8)
        if max_time < 2 and n_pois > 500:
            algorithm = "greedy"  # Force fast algorithm for tight time
        
        # Consider budget constraints
        budget = constraints.get('budget', 200)
        if budget < 50:
            algorithm = "greedy"  # Simple algorithm for tight budget
        
        # Record selection
        self.selection_history.append({
            'n_pois': n_pois,
            'algorithm': algorithm,
            'constraints': constraints,
            'timestamp': time.time()
        })
        
        return algorithm
    
    def get_algorithm_info(self, algorithm: str) -> Dict[str, Any]:
        """Get information about algorithm characteristics"""
        info = {
            "astar": {
                "complexity": "O(b^d)", 
                "optimal": True, 
                "max_pois": 100,
                "description": "Optimal solution with A* search"
            },
            "greedy": {
                "complexity": "O(n²)", 
                "optimal": False, 
                "max_pois": 10000,
                "description": "Fast heuristic selection"
            },
            "heap_greedy": {
                "complexity": "O(n log k)", 
                "optimal": False, 
                "max_pois": 50000,
                "description": "Optimized greedy with heap pruning"
            },
            "hybrid": {
                "complexity": "Adaptive", 
                "optimal": False, 
                "max_pois": 5000,
                "description": "Combines multiple approaches"
            },
            "lpa_star": {
                "complexity": "O(k log k)", 
                "optimal": True, 
                "max_pois": 1000,
                "description": "Dynamic replanning with computation reuse"
            }
        }
        return info.get(algorithm, {})
    
    def update_performance_stats(self, algorithm: str, runtime: float, 
                                success: bool, quality_score: float = 0.0):
        """Update performance statistics for algorithm"""
        if algorithm not in self.performance_stats:
            self.performance_stats[algorithm] = {
                'runs': 0,
                'total_runtime': 0.0,
                'successes': 0,
                'total_quality': 0.0
            }
        
        stats = self.performance_stats[algorithm]
        stats['runs'] += 1
        stats['total_runtime'] += runtime
        if success:
            stats['successes'] += 1
            stats['total_quality'] += quality_score
    
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary for all algorithms"""
        summary = {}
        for algo, stats in self.performance_stats.items():
            if stats['runs'] > 0:
                summary[algo] = {
                    'avg_runtime': stats['total_runtime'] / stats['runs'],
                    'success_rate': stats['successes'] / stats['runs'],
                    'avg_quality': stats['total_quality'] / max(stats['successes'], 1),
                    'total_runs': stats['runs']
                }
        return summary


class ResultCache:
    """Caches planning results for performance optimization"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = {}
        self.access_times = {}
        self.creation_times = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def _generate_key(self, preferences: Dict, constraints: Dict, 
                     algorithm: str) -> str:
        """Generate cache key from planning parameters"""
        # Create deterministic key from parameters
        pref_str = "_".join(f"{k}:{v}" for k, v in sorted(preferences.items()))
        const_str = "_".join(f"{k}:{v}" for k, v in sorted(constraints.items()))
        return f"{algorithm}_{pref_str}_{const_str}"
    
    def get(self, key: str):
        """Get cached result if available and not expired"""
        if key not in self.cache:
            return None
        
        # Check TTL
        current_time = time.time()
        if current_time - self.creation_times[key] > self.ttl_seconds:
            self._remove_key(key)
            return None
        
        # Update access time
        self.access_times[key] = current_time
        return self.cache[key]
    
    def set(self, key: str, value: Any):
        """Cache a result with a string key"""
        current_time = time.time()
        
        # Remove expired entries
        self._cleanup_expired()
        
        # Remove LRU if at capacity
        if len(self.cache) >= self.max_size:
            self._remove_lru()
        
        # Store result
        self.cache[key] = value
        self.access_times[key] = current_time
        self.creation_times[key] = current_time
    
    def get_with_params(self, preferences: Dict, constraints: Dict, algorithm: str):
        """Get cached result if available and not expired (using parameters)"""
        key = self._generate_key(preferences, constraints, algorithm)
        return self.get(key)
    
    def set_with_params(self, preferences: Dict, constraints: Dict, algorithm: str, 
           result: Any):
        """Cache a planning result using parameters"""
        key = self._generate_key(preferences, constraints, algorithm)
        self.set(key, result)
    
    def _cleanup_expired(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, creation_time in self.creation_times.items():
            if current_time - creation_time > self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_key(key)
    
    def _remove_lru(self):
        """Remove least recently used entry"""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), 
                     key=lambda k: self.access_times[k])
        self._remove_key(lru_key)
    
    def _remove_key(self, key: str):
        """Remove key from all cache structures"""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.creation_times.pop(key, None)
    
    def clear(self):
        """Clear all cached results"""
        self.cache.clear()
        self.access_times.clear()
        self.creation_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': getattr(self, '_hits', 0) / max(getattr(self, '_requests', 1), 1),
            'ttl_seconds': self.ttl_seconds
        }