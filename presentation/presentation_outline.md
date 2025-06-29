# Presentation: Ranking Itineraries - Dynamic Algorithms Meet User Preferences
## NYC Implementation Showcase

### Slide 1: Title Slide
**Background**: NYC Skyline (Manhattan at sunset)
**Title**: Ranking Itineraries: Dynamic Algorithms Meet User Preferences
**Subtitle**: Transforming NYC Tourism with Quality-Based Planning
**Author**: Stelios Zacharioudakis
**Affiliation**: NKUA Department of Informatics & Telecommunications
**Date**: June 2025

*Speaker Notes*: Welcome. Today I'll present how we revolutionized tourist itinerary planning in NYC by shifting from coverage maximization to quality-based ranking, achieving 145x improvement over state-of-the-art.

### Slide 2: The NYC Tourism Challenge
**Visual**: Heat map of Manhattan showing 10,847 POIs
**Key Points**:
- 10,847 Points of Interest across 5 boroughs
- 47 categories (museums, restaurants, parks...)
- Dynamic constraints: weather, subway, events
- Current apps fail: TravelPlanner achieves only 0.6% success

*Speaker Notes*: NYC presents unique challenges - massive scale, dynamic environment, diverse preferences. Language models struggle with hard constraints.

### Slide 3: Problem Statement
**Visual**: Tourist overwhelmed by choices, decision paralysis
**Research Question**: How to rank itineraries by quality, not just coverage?
**Key Insights**:
- Users prefer 3-7 POIs per day (not maximum coverage)
- Quality > Quantity for satisfaction
- Real-time adaptation essential in urban environments

*Speaker Notes*: Traditional approaches maximize POI coverage, but research shows tourists prefer fewer, higher-quality experiences.

### Slide 4: Literature Gap
**Visual**: Basu Roy et al. (2011) limitation diagram
**Gap Identified**:
- Basu Roy: Coverage maximization, static environment
- Missing: Quality metrics, dynamic adaptation, user preferences
- TravelPlanner (2024): 0.6% success rate highlights need

*Speaker Notes*: Basu Roy's foundational work assumes more POIs = better. We challenge this with quality-based approach.

### Slide 5: Our Solution - Hybrid Framework
**Visual**: Algorithm architecture diagram
**Components**:
1. Greedy selection (O(n²)) for scalability
2. A* routing for optimality
3. LPA* for dynamic updates
**Innovation**: Two-phase approach balances quality and performance

*Speaker Notes*: Hybrid approach achieves 96% of optimal quality at 1.5% computational cost.

### Slide 6: Complexity Analysis
**Visual**: Complexity comparison table
| Algorithm | Complexity | 1000 POIs Runtime | Quality |
|-----------|-----------|-------------------|---------|
| Greedy | O(n²) | 489ms | 0.734 |
| A* | O(b^d) | 18.9s | 0.856 |
| Hybrid | O(n² + k^2.2) | 734ms | 0.823 |

*Speaker Notes*: Quadratic complexity enables real-time performance while maintaining high quality scores.

### Slide 7: Technology Stack
**Visual**: Architecture diagram from research_context.md
**Core Technologies**:
- Python 3.8+ with Numba JIT (50x speedup)
- NetworkX for graph algorithms
- OR-Tools for complex routing
- Foursquare/OSM APIs for POI data
- Streamlit/Flask for demos

*Speaker Notes*: Implementation follows research roadmap, emphasizing performance and scalability.

### Slide 8: Quality Metrics - CSS
**Visual**: CSS formula breakdown
**Composite Satisfaction Score**:
CSS = 0.35×SAT + 0.25×TUR + 0.25×FEA + 0.15×DIV
- SAT: Attractiveness (highest weight from research)
- TUR: Time utilization
- FEA: Feasibility
- DIV: Diversity (Vendi Score)

*Speaker Notes*: Weights validated through user study with 30 participants.

### Slide 9: NYC Data Pipeline
**Visual**: Data flow diagram
**Pipeline Steps**:
1. Collect: 10,847 POIs from OSM/Foursquare
2. Clean: Quality filtering, deduplication
3. Enrich: Ratings, hours, accessibility
4. Index: R-tree spatial indexing
5. Precompute: Distance matrix with Numba

*Speaker Notes*: NYC-specific adaptations include borough boundaries, subway network, rush hour patterns.

### Slide 10: Results - Performance
**Visual**: Bar chart comparing success rates
**Key Achievement**: 87.5% vs 0.6% (TravelPlanner)
**Performance Metrics**:
- Sub-second response (489ms avg)
- Memory usage: 89MB (mobile-friendly)
- Scales to 10,000+ POIs

*Speaker Notes*: 145x improvement in task success rate while maintaining practical performance.

### Slide 11: Results - User Satisfaction
**Visual**: User study results
**Findings**:
- 8.4/10 satisfaction (vs 6.8 manual planning)
- 81.7% reduction in planning time
- 3-7 POI preference confirmed
- SUS score: 82.3 (Excellent)

*Speaker Notes*: User study with 30 participants validates quality-based approach.

### Slide 12: Dynamic Adaptation - LPA*
**Visual**: Before/after subway disruption
**LPA* Performance**:
- 70-90% computation reuse
- 156ms for subway updates
- 234ms for weather changes
- Seamless user experience

*Speaker Notes*: LPA* enables real-time replanning without full recomputation.

### Slide 13: Demo Screenshot 1 - Interface
**Visual**: Flask app main interface
**Features Shown**:
- Interactive Manhattan map
- Algorithm selector
- Preference sliders
- Real-time metrics display

*Speaker Notes*: Clean interface focuses on user preferences and real-time feedback.

### Slide 14: Demo Screenshot 2 - Dynamic Update
**Visual**: Rain alert triggering replanning
**Scenario**: Outdoor plans → Indoor alternatives
**Shows**:
- Original plan (Central Park, High Line)
- Rain notification
- Instant replanning (Museums, Markets)
- CSS maintained at 0.82

*Speaker Notes*: System adapts seamlessly to environmental changes.

### Slide 15: Case Study - Family Day
**Visual**: Actual itinerary on map
**Profile**: Family with children (8, 12)
**Generated Plan**:
- Natural History Museum (10am)
- Central Park Playground (12pm)
- Lunch at Shake Shack (1pm)
- Children's Museum (3pm)
**Results**: CSS 0.856, Under budget, 3.2km walking

*Speaker Notes*: Real example showing age-appropriate, feasible planning.

### Slide 16: Comparison with Industry
**Visual**: Feature comparison matrix
| Feature | Our System | TripAdvisor | Google |
|---------|------------|-------------|---------|
| Real-time optimization | ✓ | ✗ | Partial |
| Dynamic replanning | ✓ | ✗ | ✗ |
| Quality guarantee | ✓ | ✗ | ✗ |
| Response time | <1s | 3.4s | 2.1s |

*Speaker Notes*: First system to combine all essential features with performance.

### Slide 17: Future Directions
**Visual**: GNN architecture diagram
**Research Opportunities**:
1. Graph Neural Networks for POI relationships
2. Transformer models for sequential planning
3. Federated learning for privacy
4. Multi-modal integration (AR/VR)

*Speaker Notes*: GNNs could capture hidden relationships between POIs for better recommendations.

### Slide 18: Broader Impact
**Visual**: NYC tourism statistics
**Impact Areas**:
- Tourism: Better visitor distribution
- Economic: Support local businesses
- Environmental: Reduce overtourism
- Social: Accessible planning for all

*Speaker Notes*: Beyond technology, we're reshaping how people experience cities.

### Slide 19: Conclusions
**Key Takeaways**:
✓ Quality > Coverage paradigm shift
✓ 87.5% success rate (145x improvement)
✓ Real-time performance at city scale
✓ Validated user preferences (3-7 POIs)
✓ Dynamic adaptation with LPA*

**Thesis**: Quality-based ranking transforms urban tourism

*Speaker Notes*: We've proven that helping tourists find perfect experiences matters more than showing everything.

### Slide 20: Thank You & Resources
**Visual**: QR code linking to GitHub
**GitHub Repository**: github.com/[username]/nyc-itinerary-ranking
**Contents**:
- Full thesis PDF
- All implementation code
- NYC dataset
- Demo application
- Benchmark results

**Contact**: [email]
**Questions?**

*Speaker Notes*: All materials available on GitHub. Happy to discuss implementation details or collaboration opportunities.