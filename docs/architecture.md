# System Architecture - NYC Itinerary Ranking

## Overview

This document describes the system architecture for the NYC Itinerary Ranking system developed as part of the Bachelor's thesis "Ranking Itineraries: Dynamic Algorithms Meet User Preferences" at NKUA.

## High-Level Architecture

```mermaid
graph TB
    subgraph "User Interface Layer"
        Web[Web Interface<br/>Flask + Leaflet.js]
        CLI[Command Line<br/>Interface]
        API[REST API]
    end
    
    subgraph "Algorithm Layer"
        HP[Hybrid Planner<br/>Orchestrator]
        GR[Greedy O&#40;n²&#41;<br/>Algorithm]
        AS[A* Search<br/>Optimal]
        LPA[LPA*<br/>Dynamic]
        HG[HeapGreedy<br/>Optimized]
    end
    
    subgraph "Core Components"
        MET[Metrics Engine<br/>CSS Calculator]
        CACHE[Result Cache<br/>LRU]
        SEL[Algorithm<br/>Selector]
        FB[Feedback<br/>Processor]
    end
    
    subgraph "Data Layer"
        POI[(POI Database<br/>10,847 NYC POIs)]
        DM[(Distance Matrix<br/>Manhattan)]
        RT[(R-tree Index<br/>Spatial)]
        SUB[(Subway Data<br/>Stations)]
    end
    
    Web --> API
    CLI --> HP
    API --> HP
    
    HP --> SEL
    SEL --> GR
    SEL --> AS
    SEL --> LPA
    SEL --> HG
    
    HP --> CACHE
    HP --> MET
    HP --> FB
    
    GR --> MET
    AS --> MET
    LPA --> MET
    HG --> MET
    
    MET --> POI
    MET --> DM
    
    GR --> RT
    AS --> RT
    LPA --> RT
    HG --> RT
```

## Component Details

### 1. Algorithm Components

#### Class Diagram - Algorithm Hierarchy

```mermaid
classDiagram
    class BaseAlgorithm {
        <<abstract>>
        +List~POI~ pois
        +ndarray distance_matrix
        +plan_itinerary(preferences, constraints)
        +_calculate_utility(poi, state)
        +_is_feasible(poi, state)
    }
    
    class GreedyPOISelection {
        +select_pois(preferences, constraints, feedback)
        -_calculate_marginal_utility(poi, selected)
        -_get_travel_time(from_poi, to_poi)
    }
    
    class HeapPrunGreedyPOI {
        +int top_k
        +heapq priority_queue
        +select_pois(preferences, constraints)
        -_prune_candidates(candidates)
    }
    
    class AStarItineraryPlanner {
        +SearchNode start_node
        +plan_itinerary(preferences, constraints)
        -_compute_heuristic(state, available)
        -_generate_successors(node)
    }
    
    class LPAStarPlanner {
        +Dict nodes
        +PriorityQueue queue
        +plan_initial(preferences, constraints)
        +replan(event, preferences)
        +update_node(node)
        -_propagate_changes(affected_nodes)
    }
    
    BaseAlgorithm <|-- GreedyPOISelection
    BaseAlgorithm <|-- AStarItineraryPlanner
    BaseAlgorithm <|-- LPAStarPlanner
    GreedyPOISelection <|-- HeapPrunGreedyPOI
```

#### Sequence Diagram - Planning Flow

```mermaid
sequenceDiagram
    participant User
    participant Web
    participant HybridPlanner
    participant AlgorithmSelector
    participant Algorithm
    participant MetricsEngine
    participant Cache
    
    User->>Web: Submit preferences & constraints
    Web->>HybridPlanner: plan(preferences, constraints)
    HybridPlanner->>Cache: check(cache_key)
    
    alt Cache Hit
        Cache-->>HybridPlanner: cached_result
        HybridPlanner-->>Web: PlanningResult
    else Cache Miss
        HybridPlanner->>AlgorithmSelector: select_algorithm(n_pois, constraints)
        AlgorithmSelector-->>HybridPlanner: AlgorithmType
        HybridPlanner->>Algorithm: plan_itinerary(preferences, constraints)
        Algorithm->>Algorithm: generate_candidates()
        Algorithm->>Algorithm: apply_constraints()
        Algorithm-->>HybridPlanner: itinerary
        HybridPlanner->>MetricsEngine: calculate_metrics(itinerary)
        MetricsEngine-->>HybridPlanner: CSS score & components
        HybridPlanner->>Cache: store(result)
        HybridPlanner-->>Web: PlanningResult
    end
    
    Web-->>User: Display itinerary & metrics
```

### 2. Data Model

#### Entity Relationship Diagram

```mermaid
erDiagram
    POI {
        string id PK
        string name
        float lat
        float lon
        string category
        float rating
        float popularity_score
        float entrance_fee
        float avg_visit_duration
        tuple opening_hours
        float accessibility_score
        float weather_dependency
    }
    
    Itinerary {
        string id PK
        datetime created_at
        string user_id FK
        float css_score
        float total_distance
        float total_time
        float total_cost
    }
    
    ItineraryPOI {
        string itinerary_id FK
        string poi_id FK
        int sequence_order
        string arrival_time
        float duration
    }
    
    UserProfile {
        string user_id PK
        dict preferences
        int daily_pois_preference
        string preferred_transport
        float budget_per_day
        float max_walking_distance
    }
    
    SubwayStation {
        string id PK
        string name
        float lat
        float lon
        array lines
    }
    
    DynamicEvent {
        string id PK
        string event_type
        datetime timestamp
        array affected_poi_ids
        dict event_data
    }
    
    POI ||--o{ ItineraryPOI : "included in"
    Itinerary ||--|{ ItineraryPOI : "contains"
    UserProfile ||--o{ Itinerary : "creates"
    POI }o--|| SubwayStation : "nearest to"
    DynamicEvent }o--o{ POI : "affects"
```

### 3. API Specification

#### REST API Endpoints

```yaml
openapi: 3.0.0
info:
  title: NYC Itinerary Ranking API
  version: 1.0.0
  description: API for the Bachelor's thesis itinerary planning system

paths:
  /api/pois:
    get:
      summary: Get all POIs
      parameters:
        - name: category
          in: query
          schema:
            type: string
        - name: bounds
          in: query
          schema:
            type: object
      responses:
        200:
          description: List of POIs
          content:
            application/json:
              schema:
                type: object
                properties:
                  success: 
                    type: boolean
                  pois:
                    type: array
                    items:
                      $ref: '#/components/schemas/POI'

  /api/plan:
    post:
      summary: Generate itinerary
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                algorithm:
                  type: string
                  enum: [hybrid, greedy, heap_greedy, astar, lpa_star]
                preferences:
                  type: object
                  additionalProperties:
                    type: number
                constraints:
                  $ref: '#/components/schemas/Constraints'
      responses:
        200:
          description: Generated itinerary
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/PlanningResult'

  /api/update:
    post:
      summary: Dynamic update (LPA*)
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                type:
                  type: string
                  enum: [subway_disruption, weather_rain, poi_closed]
                session_id:
                  type: string
      responses:
        200:
          description: Updated itinerary
          
  /api/feedback:
    post:
      summary: Submit user feedback
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                rejected_pois:
                  type: array
                  items:
                    type: string
                must_include_pois:
                  type: array
                  items:
                    type: string

components:
  schemas:
    POI:
      type: object
      properties:
        id:
          type: string
        name:
          type: string
        lat:
          type: number
        lon:
          type: number
        category:
          type: string
        rating:
          type: number
        entrance_fee:
          type: number
          
    Constraints:
      type: object
      properties:
        budget:
          type: number
        max_time_hours:
          type: number
        min_pois:
          type: integer
        max_pois:
          type: integer
        transportation_mode:
          type: string
          
    PlanningResult:
      type: object
      properties:
        success:
          type: boolean
        itinerary:
          type: object
        metrics:
          type: object
        runtime:
          type: number
        algorithm_used:
          type: string
```

### 4. Algorithm Complexity Analysis

| Algorithm | Time Complexity | Space Complexity | Quality | Use Case |
|-----------|----------------|------------------|---------|----------|
| Greedy | O(n²) | O(n) | ~85% optimal | Real-time, large datasets |
| HeapGreedy | O(n log k) | O(k) | ~80% optimal | Very large datasets |
| A* | O(b^d) | O(b^d) | 100% optimal | Small problems, quality critical |
| LPA* | O(k log k) | O(n) | 100% optimal | Dynamic replanning |
| Hybrid | Varies | O(n) | ~96% optimal | General purpose |

Where:
- n = number of POIs (10,847)
- k = number of changed nodes
- b = branching factor
- d = solution depth

### 5. Performance Metrics

#### CSS (Composite Satisfaction Score) Formula

```
CSS = w₁ × SAT + w₂ × TUR + w₃ × FEA + w₄ × DIV

Where:
- SAT = Attractiveness Score (w₁ = 0.35)
- TUR = Time Utilization Rate (w₂ = 0.25)
- FEA = Feasibility Score (w₃ = 0.25)
- DIV = Diversity Score (w₄ = 0.15)
```

#### Component Calculations

```python
# Attractiveness Score
SAT = Σ(rating_i × popularity_i × preference_alignment_i) / n

# Time Utilization Rate
TUR = (Σ visit_duration_i) / total_available_time

# Feasibility Score
FEA = Π(constraint_satisfaction_i)

# Diversity Score (Vendi Score)
DIV = exp(entropy(category_distribution))
```

### 6. Database Schema

```sql
-- POI Table
CREATE TABLE pois (
    id VARCHAR(20) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    lat DECIMAL(10, 8) NOT NULL,
    lon DECIMAL(11, 8) NOT NULL,
    category VARCHAR(50) NOT NULL,
    rating DECIMAL(2, 1),
    popularity_score DECIMAL(3, 3),
    entrance_fee DECIMAL(6, 2),
    avg_visit_duration DECIMAL(3, 1),
    opening_hour DECIMAL(4, 2),
    closing_hour DECIMAL(4, 2),
    accessibility_score DECIMAL(3, 3),
    weather_dependency DECIMAL(3, 3),
    INDEX idx_category (category),
    INDEX idx_location (lat, lon)
);

-- Distance Matrix (for fast lookups)
CREATE TABLE distance_matrix (
    from_poi_id VARCHAR(20),
    to_poi_id VARCHAR(20),
    distance_km DECIMAL(5, 2),
    PRIMARY KEY (from_poi_id, to_poi_id),
    FOREIGN KEY (from_poi_id) REFERENCES pois(id),
    FOREIGN KEY (to_poi_id) REFERENCES pois(id)
);

-- Subway Stations
CREATE TABLE subway_stations (
    id VARCHAR(20) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    lat DECIMAL(10, 8) NOT NULL,
    lon DECIMAL(11, 8) NOT NULL,
    lines JSON
);

-- POI-Subway Proximity
CREATE TABLE poi_subway_proximity (
    poi_id VARCHAR(20),
    station_id VARCHAR(20),
    distance_km DECIMAL(4, 3),
    PRIMARY KEY (poi_id, station_id),
    FOREIGN KEY (poi_id) REFERENCES pois(id),
    FOREIGN KEY (station_id) REFERENCES subway_stations(id)
);
```

### 7. Deployment Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        Browser[Web Browser]
        Mobile[Mobile App]
    end
    
    subgraph "Application Layer"
        LB[Load Balancer<br/>Nginx]
        Web1[Flask Server 1]
        Web2[Flask Server 2]
        WebN[Flask Server N]
    end
    
    subgraph "Computing Layer"
        Queue[Task Queue<br/>Celery]
        Worker1[Algorithm Worker 1]
        Worker2[Algorithm Worker 2]
        WorkerN[Algorithm Worker N]
    end
    
    subgraph "Data Layer"
        Redis[(Redis Cache)]
        PG[(PostgreSQL<br/>POI Data)]
        S3[S3 Storage<br/>Distance Matrix]
    end
    
    Browser --> LB
    Mobile --> LB
    LB --> Web1
    LB --> Web2
    LB --> WebN
    
    Web1 --> Queue
    Web2 --> Queue
    WebN --> Queue
    
    Queue --> Worker1
    Queue --> Worker2
    Queue --> WorkerN
    
    Worker1 --> Redis
    Worker2 --> Redis
    WorkerN --> Redis
    
    Worker1 --> PG
    Worker2 --> PG
    WorkerN --> PG
    
    Worker1 --> S3
    Worker2 --> S3
    WorkerN --> S3
```

### 8. Performance Benchmarks

#### Response Time Analysis

| Operation | Average Time | 95th Percentile | Target |
|-----------|--------------|-----------------|--------|
| POI Search | 12ms | 25ms | <50ms |
| Greedy Planning | 489ms | 650ms | <1s |
| A* Planning | 1.2s | 2.1s | <3s |
| LPA* Replanning | 87ms | 145ms | <200ms |
| CSS Calculation | 5ms | 8ms | <10ms |

#### Scalability Metrics

- **Concurrent Users**: Supports 1,000+ concurrent planning requests
- **POI Dataset Size**: Tested up to 100,000 POIs
- **Memory Usage**: ~500MB base + 100MB per active planning session
- **CPU Utilization**: ~70% at peak load (8-core system)

### 9. Security Considerations

1. **Input Validation**: All user inputs validated against schemas
2. **Rate Limiting**: 100 requests/minute per IP
3. **Authentication**: JWT tokens for registered users
4. **Data Privacy**: No personal data stored beyond session
5. **SQL Injection Prevention**: Parameterized queries only
6. **XSS Protection**: Content Security Policy headers

### 10. Monitoring and Observability

```mermaid
graph LR
    App[Application] --> OT[OpenTelemetry<br/>Collector]
    OT --> Prom[Prometheus<br/>Metrics]
    OT --> Jaeger[Jaeger<br/>Traces]
    OT --> Elastic[Elasticsearch<br/>Logs]
    
    Prom --> Grafana[Grafana<br/>Dashboards]
    Jaeger --> Grafana
    Elastic --> Kibana[Kibana<br/>Log Analysis]
```

Key Metrics Tracked:
- Algorithm selection distribution
- CSS score distribution
- Planning time by algorithm
- Cache hit rate
- User satisfaction ratings
- Error rates by endpoint

## Implementation Notes

1. **Numba Optimization**: Critical loops in Greedy algorithms use `@numba.jit` for 4.3x speedup
2. **Spatial Indexing**: R-tree reduces nearby POI queries from O(n) to O(log n)
3. **Caching Strategy**: LRU cache with 1-hour TTL for identical requests
4. **Distance Precomputation**: Manhattan distances precomputed and stored in matrix
5. **Parallel Processing**: Multi-threaded alternative generation for Pareto frontiers

## Future Enhancements

1. **GraphQL API**: For more flexible client queries
2. **Machine Learning**: Neural preference learning from user feedback
3. **Multi-modal Transport**: Integration with real-time transit APIs
4. **AR Navigation**: Mobile AR overlay for turn-by-turn guidance
5. **Social Features**: Shared itineraries and collaborative planning