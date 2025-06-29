"""
Comprehensive Evaluation Metrics for Itinerary Ranking Systems

This module implements evaluation metrics based on the research framework described in:
- Basu Roy et al. (2011) - Interactive Itinerary Planning [basu2011]
- Lim et al. (2018) - Personalized Trip Recommendation [lim2018]
- Research findings from "Ranking Itineraries: Dynamic algorithms meet user preferences"

The metrics are structured according to the evaluation framework in research_context.md,
balancing quantitative measures (distance, time, cost) with qualitative assessments
(satisfaction, diversity, novelty, personalization).
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from scipy.spatial.distance import euclidean, cityblock
from scipy.stats import entropy
import warnings


@dataclass
class POI:
    """Point of Interest representation"""
    id: str
    name: str
    lat: float
    lon: float
    category: str
    popularity: float  # 0-1 normalized
    entrance_fee: float
    avg_visit_duration: float  # in hours
    opening_hours: Tuple[float, float]  # (open, close) in 24h format
    rating: float  # 1-5 scale


@dataclass
class Itinerary:
    """Itinerary representation"""
    pois: List[POI]
    start_time: float
    transportation_mode: str = "walking"  # walking, public_transit, taxi
    user_preferences: Optional[Dict[str, float]] = None


class QuantitativeMetrics:
    """
    Quantitative metrics for itinerary evaluation.
    
    Based on research_context.md findings:
    - Distance metrics for NYC urban environment
    - Time metrics respecting 3-7 POIs per day preference
    - Cost metrics for budget-aware planning
    - Efficiency ratios for optimization
    
    References:
    - Verbeeck et al. (2014) - Time-Dependent Orienteering [verbeeck2014]
    - Gunawan et al. (2016) - Orienteering Problem Survey [gunawan2016]
    """
    
    # NYC-specific constants
    NYC_AVG_WALKING_SPEED = 4.5  # km/h
    NYC_AVG_TRANSIT_SPEED = 25.0  # km/h
    NYC_AVG_TAXI_SPEED = 18.0  # km/h
    NYC_TAXI_BASE_FARE = 3.0  # USD
    NYC_TAXI_PER_KM = 1.75  # USD
    NYC_TRANSIT_FARE = 2.90  # USD per trip
    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate great-circle distance between two points.
        
        Formula:
        $$d = 2r \arcsin\left(\sqrt{\sin^2\left(\frac{\phi_2-\phi_1}{2}\right) + 
        \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\lambda_2-\lambda_1}{2}\right)}\right)$$
        
        where $\phi$ is latitude, $\lambda$ is longitude, $r$ is earth's radius
        """
        R = 6371  # Earth's radius in kilometers
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    @staticmethod
    def manhattan_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Manhattan distance approximation for NYC grid system.
        
        Formula:
        $$d_{manhattan} = k \cdot (|\Delta lat| + |\Delta lon|)$$
        
        where k is a scaling factor for NYC's grid (≈1.4 for realistic routing)
        """
        # NYC grid adjustment factor
        NYC_GRID_FACTOR = 1.4
        lat_km = 111.0  # km per degree latitude
        lon_km = 111.0 * np.cos(np.radians((lat1 + lat2) / 2))  # adjusted for latitude
        
        dlat = abs(lat2 - lat1) * lat_km
        dlon = abs(lon2 - lon1) * lon_km
        
        return NYC_GRID_FACTOR * (dlat + dlon)
    
    @classmethod
    def total_distance(cls, itinerary: Itinerary, metric: str = "manhattan") -> float:
        """
        Calculate total distance of itinerary.
        
        Args:
            itinerary: Itinerary object
            metric: Distance metric - "euclidean", "manhattan", or "haversine"
        
        Returns:
            Total distance in kilometers
        """
        if len(itinerary.pois) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(itinerary.pois) - 1):
            poi1, poi2 = itinerary.pois[i], itinerary.pois[i + 1]
            
            if metric == "euclidean":
                # Simple Euclidean approximation
                lat_km = 111.0
                lon_km = 111.0 * np.cos(np.radians(poi1.lat))
                total += euclidean(
                    [poi1.lat * lat_km, poi1.lon * lon_km],
                    [poi2.lat * lat_km, poi2.lon * lon_km]
                )
            elif metric == "manhattan":
                total += cls.manhattan_distance(poi1.lat, poi1.lon, poi2.lat, poi2.lon)
            elif metric == "haversine":
                total += cls.haversine_distance(poi1.lat, poi1.lon, poi2.lat, poi2.lon)
            else:
                raise ValueError(f"Unknown metric: {metric}")
        
        return total
    
    @classmethod
    def travel_time(cls, itinerary: Itinerary) -> float:
        """
        Calculate total travel time based on transportation mode.
        
        Respects research finding: users prefer 3-7 POIs per day
        
        Returns:
            Travel time in hours
        """
        distance = cls.total_distance(itinerary, metric="manhattan")
        
        speed_map = {
            "walking": cls.NYC_AVG_WALKING_SPEED,
            "public_transit": cls.NYC_AVG_TRANSIT_SPEED,
            "taxi": cls.NYC_AVG_TAXI_SPEED
        }
        
        speed = speed_map.get(itinerary.transportation_mode, cls.NYC_AVG_WALKING_SPEED)
        travel_time = distance / speed
        
        # Add transfer time for public transit
        if itinerary.transportation_mode == "public_transit":
            # Assume average 5 min wait + 2 min walk per transfer
            travel_time += (len(itinerary.pois) - 1) * 0.117  # 7 min in hours
        
        return travel_time
    
    @classmethod
    def visit_duration(cls, itinerary: Itinerary) -> float:
        """
        Calculate total visit duration at POIs.
        
        Based on Lim et al. (2018) findings on POI visit patterns
        """
        return sum(poi.avg_visit_duration for poi in itinerary.pois)
    
    @classmethod
    def total_time(cls, itinerary: Itinerary) -> float:
        """
        Calculate total itinerary time (travel + visits).
        
        Formula:
        $$T_{total} = T_{travel} + \sum_{i=1}^{n} T_{visit}(poi_i)$$
        """
        return cls.travel_time(itinerary) + cls.visit_duration(itinerary)
    
    @classmethod
    def transportation_cost(cls, itinerary: Itinerary) -> float:
        """
        Calculate transportation cost based on mode.
        
        NYC-specific pricing models
        """
        if itinerary.transportation_mode == "walking":
            return 0.0
        elif itinerary.transportation_mode == "public_transit":
            # Assume one fare per 2 POIs (transfers included)
            return cls.NYC_TRANSIT_FARE * np.ceil(len(itinerary.pois) / 2)
        elif itinerary.transportation_mode == "taxi":
            distance = cls.total_distance(itinerary, metric="manhattan")
            return cls.NYC_TAXI_BASE_FARE + (distance * cls.NYC_TAXI_PER_KM)
        else:
            return 0.0
    
    @classmethod
    def entrance_fees(cls, itinerary: Itinerary) -> float:
        """Calculate total entrance fees for all POIs."""
        return sum(poi.entrance_fee for poi in itinerary.pois)
    
    @classmethod
    def total_cost(cls, itinerary: Itinerary) -> float:
        """
        Calculate total itinerary cost.
        
        Formula:
        $$C_{total} = C_{transport} + \sum_{i=1}^{n} C_{entrance}(poi_i)$$
        """
        return cls.transportation_cost(itinerary) + cls.entrance_fees(itinerary)
    
    @classmethod
    def utility_per_time(cls, itinerary: Itinerary, utility_func=None) -> float:
        """
        Calculate utility per unit time (efficiency metric).
        
        Formula:
        $$E_{time} = \frac{U(itinerary)}{T_{total}}$$
        
        Based on efficiency concepts from Verbeeck et al. (2014)
        """
        total_time = cls.total_time(itinerary)
        if total_time == 0:
            return 0.0
        
        if utility_func is None:
            # Simple utility: sum of POI ratings
            utility = sum(poi.rating for poi in itinerary.pois)
        else:
            utility = utility_func(itinerary)
        
        return utility / total_time
    
    @classmethod
    def utility_per_cost(cls, itinerary: Itinerary, utility_func=None) -> float:
        """
        Calculate utility per unit cost (value metric).
        
        Formula:
        $$E_{cost} = \frac{U(itinerary)}{C_{total}}$$
        """
        total_cost = cls.total_cost(itinerary)
        if total_cost == 0:
            return float('inf')
        
        if utility_func is None:
            utility = sum(poi.rating for poi in itinerary.pois)
        else:
            utility = utility_func(itinerary)
        
        return utility / total_cost


class QualitativeMetrics:
    """
    Qualitative metrics for itinerary evaluation.
    
    Based on research_context.md framework:
    - User satisfaction (weight 0.35 for attractiveness)
    - Diversity using Vendi Score with Shannon entropy
    - Novelty based on popularity inverse
    - Personalization for preference matching
    
    References:
    - Rahmani et al. (2022) - Context fusion and fairness [rahmani2022]
    - Werneck et al. (2021) - POI recommendation evaluation [werneck2021]
    """
    
    @staticmethod
    def user_satisfaction(itinerary: Itinerary, user_preferences: Dict[str, float]) -> float:
        """
        Calculate user satisfaction based on preference matching.
        
        Formula (from research_context.md):
        $$SAT = 0.35 \times attractiveness + other\_factors$$
        
        Based on Lim et al. (2018) preference framework
        """
        if not itinerary.pois:
            return 0.0
        
        # Category preference matching
        category_scores = []
        for poi in itinerary.pois:
            pref_score = user_preferences.get(poi.category, 0.5)
            attractiveness = poi.rating / 5.0  # Normalize to 0-1
            
            # Weight attractiveness at 0.35 as per research
            poi_satisfaction = 0.35 * attractiveness + 0.65 * pref_score
            category_scores.append(poi_satisfaction)
        
        # Average satisfaction across POIs
        return np.mean(category_scores)
    
    @staticmethod
    def diversity_score(itinerary: Itinerary) -> float:
        """
        Calculate diversity using Vendi Score with Shannon entropy.
        
        Formula:
        $$D_{vendi} = exp(H(p))$$
        where $H(p) = -\sum_{i=1}^{k} p_i \log(p_i)$ is Shannon entropy
        
        Based on research_context.md diversity measures
        """
        if not itinerary.pois:
            return 0.0
        
        # Count category frequencies
        categories = [poi.category for poi in itinerary.pois]
        unique_categories, counts = np.unique(categories, return_counts=True)
        
        # Calculate probabilities
        probabilities = counts / len(categories)
        
        # Shannon entropy
        h_score = entropy(probabilities, base=np.e)
        
        # Vendi Score (exponential of entropy)
        vendi_score = np.exp(h_score)
        
        # Normalize to 0-1 range (max diversity when all categories different)
        max_vendi = len(itinerary.pois)
        normalized_score = min(vendi_score / max_vendi, 1.0)
        
        return normalized_score
    
    @staticmethod
    def novelty_score(itinerary: Itinerary) -> float:
        """
        Calculate novelty based on inverse popularity.
        
        Formula:
        $$N = \frac{1}{n} \sum_{i=1}^{n} (1 - popularity_i)$$
        
        Less popular POIs contribute to higher novelty
        """
        if not itinerary.pois:
            return 0.0
        
        novelties = [1.0 - poi.popularity for poi in itinerary.pois]
        return np.mean(novelties)
    
    @staticmethod
    def personalization_score(itinerary: Itinerary, user_history: List[str], 
                            category_preferences: Dict[str, float]) -> float:
        """
        Calculate personalization based on user history and preferences.
        
        Combines:
        - Preference matching (from user profile)
        - History-based collaborative filtering concepts
        
        Based on Huang et al. (2019) attention-based preference modeling
        """
        if not itinerary.pois:
            return 0.0
        
        scores = []
        
        for poi in itinerary.pois:
            # Base preference score
            pref_score = category_preferences.get(poi.category, 0.5)
            
            # Boost if category appears frequently in history
            history_boost = user_history.count(poi.category) / max(len(user_history), 1)
            history_boost = min(history_boost * 0.2, 0.2)  # Cap at 0.2 boost
            
            # Combine scores
            poi_score = min(pref_score + history_boost, 1.0)
            scores.append(poi_score)
        
        return np.mean(scores)
    
    @staticmethod
    def temporal_appropriateness(itinerary: Itinerary) -> float:
        """
        Evaluate if POIs are visited at appropriate times.
        
        Based on Khodadadian et al. (2023) time-dependent profits
        """
        if not itinerary.pois:
            return 0.0
        
        current_time = itinerary.start_time
        appropriate_visits = 0
        
        for poi in itinerary.pois:
            # Check if arrival is within opening hours
            if poi.opening_hours[0] <= current_time <= poi.opening_hours[1]:
                # Also check if visit can be completed before closing
                end_time = current_time + poi.avg_visit_duration
                if end_time <= poi.opening_hours[1]:
                    appropriate_visits += 1
            
            # Update current time
            current_time += poi.avg_visit_duration
            # Add approximate travel time (simplified)
            current_time += 0.5  # 30 min average between POIs
        
        return appropriate_visits / len(itinerary.pois)


class CompositeUtilityFunctions:
    """
    Composite utility functions combining multiple metrics.
    
    Based on research_context.md weighted utility framework:
    CSS = 0.25×TUR + 0.35×SAT + 0.25×FEA + 0.15×DIV
    
    where:
    - TUR: Time Utilization Ratio
    - SAT: User Satisfaction
    - FEA: Feasibility
    - DIV: Diversity
    
    References:
    - Gu et al. (2020) - Attractive routes with gravity model [gu2020]
    - Ruiz-Meza & Montoya-Torres (2022) - Systematic review [ruizmeza2022]
    """
    
    @staticmethod
    def time_utilization_ratio(itinerary: Itinerary, max_time: float = 10.0) -> float:
        """
        Calculate time utilization ratio (TUR).
        
        Formula:
        $$TUR = \frac{T_{visits}}{T_{total}} \times \min\left(\frac{T_{total}}{T_{max}}, 1\right)$$
        
        Balances visit time vs travel time, penalizes over-long itineraries
        """
        visit_time = QuantitativeMetrics.visit_duration(itinerary)
        total_time = QuantitativeMetrics.total_time(itinerary)
        
        if total_time == 0:
            return 0.0
        
        utilization = visit_time / total_time
        time_penalty = min(total_time / max_time, 1.0)
        
        return utilization * time_penalty
    
    @staticmethod
    def feasibility_score(itinerary: Itinerary, budget: float = 100.0, 
                         max_time: float = 10.0) -> float:
        """
        Calculate feasibility score (FEA).
        
        Combines:
        - Budget feasibility
        - Time feasibility  
        - Temporal appropriateness
        """
        # Budget feasibility
        total_cost = QuantitativeMetrics.total_cost(itinerary)
        budget_feas = min(budget / max(total_cost, 1.0), 1.0) if total_cost <= budget else 0.0
        
        # Time feasibility (3-7 POIs per day preference)
        n_pois = len(itinerary.pois)
        if 3 <= n_pois <= 7:
            poi_feas = 1.0
        elif n_pois < 3:
            poi_feas = n_pois / 3.0
        else:
            poi_feas = max(0.0, 1.0 - (n_pois - 7) * 0.1)
        
        # Temporal appropriateness
        temp_feas = QualitativeMetrics.temporal_appropriateness(itinerary)
        
        # Weighted combination
        return 0.4 * budget_feas + 0.3 * poi_feas + 0.3 * temp_feas
    
    @classmethod
    def composite_satisfaction_score(cls, itinerary: Itinerary, 
                                   user_preferences: Dict[str, float],
                                   budget: float = 100.0,
                                   max_time: float = 10.0) -> float:
        """
        Calculate Composite Satisfaction Score (CSS).
        
        Formula from research_context.md:
        $$CSS = 0.25 \times TUR + 0.35 \times SAT + 0.25 \times FEA + 0.15 \times DIV$$
        
        This is the primary ranking metric for itineraries
        """
        tur = cls.time_utilization_ratio(itinerary, max_time)
        sat = QualitativeMetrics.user_satisfaction(itinerary, user_preferences)
        fea = cls.feasibility_score(itinerary, budget, max_time)
        div = QualitativeMetrics.diversity_score(itinerary)
        
        css = 0.25 * tur + 0.35 * sat + 0.25 * fea + 0.15 * div
        
        return css
    
    @staticmethod
    def multiplicative_utility(itinerary: Itinerary, 
                             user_preferences: Dict[str, float],
                             alpha: float = 0.5) -> float:
        """
        Multiplicative utility function (alternative to linear).
        
        Formula:
        $$U_{mult} = \prod_{i} metric_i^{w_i}$$
        
        More sensitive to low scores in any dimension
        """
        sat = QualitativeMetrics.user_satisfaction(itinerary, user_preferences)
        div = QualitativeMetrics.diversity_score(itinerary)
        nov = QualitativeMetrics.novelty_score(itinerary)
        
        # Avoid zero values
        sat = max(sat, 0.01)
        div = max(div, 0.01) 
        nov = max(nov, 0.01)
        
        # Weights sum to 1
        w_sat, w_div, w_nov = 0.5, 0.3, 0.2
        
        return (sat ** w_sat) * (div ** w_div) * (nov ** w_nov)
    
    @staticmethod
    def gravity_model_utility(itinerary: Itinerary, 
                            user_preferences: Dict[str, float]) -> float:
        """
        Gravity model-based utility from Gu et al. (2020).
        
        Formula:
        $$U_{gravity} = \sum_{i,j} \frac{attr_i \times attr_j}{d_{ij}^2}$$
        
        Models attraction between consecutive POIs
        """
        if len(itinerary.pois) < 2:
            return 0.0
        
        total_utility = 0.0
        
        for i in range(len(itinerary.pois) - 1):
            poi1, poi2 = itinerary.pois[i], itinerary.pois[i + 1]
            
            # Attractiveness based on ratings and preferences
            attr1 = poi1.rating * user_preferences.get(poi1.category, 0.5)
            attr2 = poi2.rating * user_preferences.get(poi2.category, 0.5)
            
            # Distance between POIs
            dist = QuantitativeMetrics.haversine_distance(
                poi1.lat, poi1.lon, poi2.lat, poi2.lon
            )
            dist = max(dist, 0.1)  # Avoid division by zero
            
            # Gravity contribution
            total_utility += (attr1 * attr2) / (dist ** 2)
        
        # Normalize by number of transitions
        return total_utility / (len(itinerary.pois) - 1)


# NYC-specific example usage
def create_nyc_example():
    """
    Create example NYC itinerary for testing metrics.
    
    Popular NYC tourist route: Times Square → Central Park → MET → High Line
    """
    pois = [
        POI("ts1", "Times Square", 40.7580, -73.9855, "entertainment", 
            0.95, 0.0, 0.5, (0.0, 24.0), 4.2),
        POI("cp1", "Central Park", 40.7829, -73.9654, "nature", 
            0.90, 0.0, 2.0, (6.0, 21.0), 4.7),
        POI("met1", "Metropolitan Museum", 40.7794, -73.9632, "museum", 
            0.85, 25.0, 3.0, (10.0, 17.0), 4.8),
        POI("hl1", "High Line", 40.7480, -74.0048, "park", 
            0.75, 0.0, 1.5, (7.0, 22.0), 4.6)
    ]
    
    user_prefs = {
        "entertainment": 0.7,
        "nature": 0.8,
        "museum": 0.9,
        "park": 0.6
    }
    
    itinerary = Itinerary(
        pois=pois,
        start_time=10.0,  # 10 AM
        transportation_mode="public_transit",
        user_preferences=user_prefs
    )
    
    return itinerary, user_prefs


def demonstrate_metrics():
    """Demonstrate all metrics on NYC example."""
    itinerary, user_prefs = create_nyc_example()
    
    print("=== NYC Itinerary Evaluation ===\n")
    
    # Quantitative metrics
    print("Quantitative Metrics:")
    print(f"Total Distance (Manhattan): {QuantitativeMetrics.total_distance(itinerary):.2f} km")
    print(f"Travel Time: {QuantitativeMetrics.travel_time(itinerary):.2f} hours")
    print(f"Visit Duration: {QuantitativeMetrics.visit_duration(itinerary):.2f} hours")
    print(f"Total Time: {QuantitativeMetrics.total_time(itinerary):.2f} hours")
    print(f"Transportation Cost: ${QuantitativeMetrics.transportation_cost(itinerary):.2f}")
    print(f"Total Cost: ${QuantitativeMetrics.total_cost(itinerary):.2f}")
    print(f"Utility per Time: {QuantitativeMetrics.utility_per_time(itinerary):.2f}")
    print()
    
    # Qualitative metrics
    print("Qualitative Metrics:")
    print(f"User Satisfaction: {QualitativeMetrics.user_satisfaction(itinerary, user_prefs):.3f}")
    print(f"Diversity Score: {QualitativeMetrics.diversity_score(itinerary):.3f}")
    print(f"Novelty Score: {QualitativeMetrics.novelty_score(itinerary):.3f}")
    print(f"Temporal Appropriateness: {QualitativeMetrics.temporal_appropriateness(itinerary):.3f}")
    print()
    
    # Composite scores
    print("Composite Scores:")
    css = CompositeUtilityFunctions.composite_satisfaction_score(
        itinerary, user_prefs, budget=100.0, max_time=10.0
    )
    print(f"CSS (Main Ranking Score): {css:.3f}")
    print(f"Multiplicative Utility: {CompositeUtilityFunctions.multiplicative_utility(itinerary, user_prefs):.3f}")
    print(f"Gravity Model Utility: {CompositeUtilityFunctions.gravity_model_utility(itinerary, user_prefs):.3f}")


if __name__ == "__main__":
    demonstrate_metrics()